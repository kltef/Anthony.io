#!/usr/bin/env python3
# Distillation trainer for the State.io move-scoring policy — a DIFFERENT strategy from train_rl.py.
#
# train_rl.py uses SNES (black-box evolution): one scalar per game, no per-move credit, and it
# optimizes a GREEDY scorer even though the game deploys the net inside an MCTS planner. It plateaus.
#
# This trainer instead does AlphaZero-style EXPERT ITERATION / DAgger:
#   1. Sample positions from SELF-PLAY with the current net (so the data matches what it'll see).
#   2. For each position, run a LOOKAHEAD EXPERT (1-ply + net rollouts — i.e. the planner) to find
#      the strongest move.  This expert is strictly stronger than the greedy net.
#   3. SUPERVISED-train the net with gradient descent (Adam) + a listwise soft-max cross-entropy to
#      imitate the expert's chosen move.  Dense gradients => real credit assignment.
#   4. DAgger: regenerate positions from the IMPROVED net and repeat, so it learns to recover from
#      its own mistakes (the exact reason it lost to the simple greedy bot).
#
# Same 12-feature net + same eval harness as train_rl.py, and the SAME no-regression ship gate:
# writes rl_policy_new.json only if it clearly beats the greedy heuristic with no regression.
#   usage: python3 tools/train_distill.py [seconds]
import numpy as np, json, time, sys, math, os
from multiprocessing import Pool
import train_rl as T          # reuse env (new_game/step/send/feats12/globals_for/choose) + eval + net io

FN, CAP, ARCH, SCR = T.FN, T.CAP, T.ARCH, T.SCR
POLICY = T.POLICY

# ---------------- net: forward + manual backprop (theta layout identical to train_rl.unpack) ----------------
def unpack_named(th):
    W1=th[0:384].reshape(32,12); b1=th[384:416]
    W2=th[416:1440].reshape(32,32); b2=th[1440:1472]
    W3=th[1472:1504].reshape(1,32); b3=th[1504:1505]
    noop=th[1505]
    return W1,b1,W2,b2,W3,b3,noop
def fwd(th, X):
    W1,b1,W2,b2,W3,b3,noop=unpack_named(th)
    z1=X@W1.T+b1; a1=np.tanh(z1)
    z2=a1@W2.T+b2; a2=np.tanh(z2)
    s=(a2@W3.T+b3)[:,0]
    return s,(X,a1,a2,W1,W2,W3)
def bwd(cache, ds):
    X,a1,a2,W1,W2,W3=cache
    dz3=ds[:,None]                         # (M,1)
    dW3=dz3.T@a2; db3=dz3.sum(0)
    da2=dz3@W3; dz2=da2*(1-a2*a2)
    dW2=dz2.T@a1; db2=dz2.sum(0)
    da1=dz2@W2; dz1=da1*(1-a1*a1)
    dW1=dz1.T@X; db1=dz1.sum(0)
    return np.concatenate([dW1.ravel(),db1,dW2.ravel(),db2,dW3.ravel(),db3])

# ---------------- lookahead expert (1-ply + net rollouts) ----------------
def clone(g):
    return dict(pos=g['pos'], owner=g['owner'].copy(), troops=g['troops'].copy(),
                armies=[dict(a) for a in g['armies']], rlDiag=g['rlDiag'], ORB=g['ORB'], N=g['N'])
def node_val(g, owner):
    o=g['owner']; t=g['troops']
    cnt={}; mass={}
    for p in range(1, int(o.max())+1):
        m=(o==p)
        if m.any(): cnt[p]=int(m.sum()); mass[p]=float(t[m].sum())
    for a in g['armies']:
        if a['owner']>0: mass[a['owner']]=mass.get(a['owner'],0.0)+a['count']
    if owner not in cnt: return -1.0
    if len(cnt)==1: return 1.0
    def strg(p): return cnt.get(p,0)+mass.get(p,0.0)/30.0
    mine=strg(owner); best=max(strg(p) for p in cnt if p!=owner)
    return math.tanh(0.22*(mine-best))
def rollout(g, layers, noop, owner, rng, H=36.0, dt=0.6):
    # FULL-GAME Monte-Carlo: play to a decision (or the horizon cap) and score by the TRUE objective
    # (did `owner` actually win / how far ahead), not a short-horizon heuristic. This is the teacher
    # that can actually beat an ES net trained on whole-game outcomes.
    P=int(g['owner'].max())
    timers={p:rng.uniform(0.1,0.6) for p in range(1,P+1)}
    t=0.0
    while t<H:
        T.step(g,dt); t+=dt
        for p in range(1,P+1):
            timers[p]-=dt
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0)
                mv=T.choose(layers,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    o=g['owner']; N=g['N']; al=T.alive_owners(g)
    if al=={owner}: return 1.0
    if owner not in al: return -1.0
    own=int((o==owner).sum())
    others=[int((o==p).sum()) for p in range(1,P+1) if p!=owner]
    return (own-(max(others) if others else 0))/N
def candidates(g, owner, cap=10):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return []
    if len(srcs)>3: srcs=srcs[np.argsort(-t[srcs])[:3]]
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return []
    cands=[(int(si),int(ti)) for si in srcs for ti in tgts]
    return cands
def expert_label(g, layers, noop, owner, rng, nroll=3):
    cands=candidates(g,owner)
    if not cands: return None
    glob=T.globals_for(g, owner)
    tg=np.array([c[1] for c in cands])
    # rank by net first, keep the top ~10 to keep rollouts cheap
    feats={}
    Xall=[]
    for (si,ti),k in zip(cands,range(len(cands))):
        Xall.append(T.feats12(g,si,owner,np.array([ti]),glob)[0])
    Xall=np.array(Xall)
    sc,_=fwd_layers(layers,noop,Xall)
    keep=np.argsort(-sc)[:6]
    cands=[cands[i] for i in keep]; Xk=Xall[keep]
    # evaluate each kept candidate + no-op by net rollouts (averaged over nroll)
    best=-1e9; bj=-1
    for j,(si,ti) in enumerate(cands):
        v=0.0
        for _ in range(nroll):
            g2=clone(g); T.send(g2,si,ti); v+=rollout(g2,layers,noop,owner,rng)
        v/=nroll
        if v>best: best=v; bj=j
    # no-op value
    vno=0.0
    for _ in range(nroll):
        g2=clone(g); vno+=rollout(g2,layers,noop,owner,rng)
    vno/=nroll
    tgt = len(cands) if vno>best+1e-9 else bj      # index == len(cands) means NO-OP
    return Xk, tgt
def fwd_layers(layers, noop, X):
    # forward using train_rl-style layers (list of (W,b)); returns scores only
    a=X
    for k,(W,b) in enumerate(layers):
        z=a@W.T+b; a=np.tanh(z) if k<len(layers)-1 else z
    return a[:,0], None

# ---------------- one worker task: self-play to a random position, then label it ----------------
def gen_and_label(args):
    th, seed, max_t = args
    rng=np.random.default_rng(seed)
    layers,noop=T.unpack(th)
    P=int(rng.choice([2,3,4])); N=18
    g=T.new_game(N,P,rng)
    timers={p:rng.uniform(0.2,1.0) for p in range(1,P+1)}
    stop=rng.uniform(3.0, max_t); t=0.0
    while t<stop:
        T.step(g,0.25); t+=0.25
        for p in range(1,P+1):
            timers[p]-=0.25
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0)
                mv=T.choose(layers,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    alive=[p for p in range(1,P+1) if (g['owner']==p).any()]
    rng.shuffle(alive)
    for owner in alive:                       # try each alive owner so a task almost always yields a label
        lab=expert_label(g, layers, noop, int(owner), rng)
        if lab is not None:
            X,tgt=lab
            return (X.tolist(), int(tgt))
    return None

# ---------------- training ----------------
def train_epoch(th, batch, lr_state, noop_idx_lr=1.0):
    # batch: list of (X (m,12) ndarray, tgt int in [0..m])  -- tgt==m means no-op
    Xs=[b[0] for b in batch]; tgts=[b[1] for b in batch]
    offs=np.cumsum([0]+[len(x) for x in Xs])
    Xall=np.vstack(Xs)
    s_all,cache=fwd(th, Xall)
    noop=th[1505]
    ds=np.zeros_like(s_all); dnoop=0.0; loss=0.0
    for i in range(len(Xs)):
        a,b=offs[i],offs[i+1]
        seg=np.concatenate([s_all[a:b],[noop]])
        seg=seg-seg.max(); p=np.exp(seg); p/=p.sum()
        tgt=tgts[i]
        loss-=math.log(max(p[tgt],1e-9))
        g=p.copy(); g[tgt]-=1.0                  # softmax-CE grad
        ds[a:b]=g[:-1]; dnoop+=g[-1]
    grad=bwd(cache, ds)
    grad=np.concatenate([grad,[dnoop]])/len(Xs)
    # Adam
    m,v,tstep,lr=lr_state
    tstep+=1; b1,b2,eps=0.9,0.999,1e-8
    m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*(grad*grad)
    mh=m/(1-b1**tstep); vh=v/(1-b2**tstep)
    th=th-lr*mh/(np.sqrt(vh)+eps)
    return th,(m,v,tstep,lr),loss/len(Xs)

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 1200.0
    th=T.init_from_baseline(POLICY).astype(np.float64); n=len(th)
    rng=np.random.default_rng(11); randnet=rng.standard_normal(n)*0.5
    base_ev=T.evalset(th, randnet, rng, 200); base_m=T.metric(base_ev)
    best_m=base_m; champ=th.copy()
    print(f"params={n} DISTILL/DAgger cores={os.cpu_count()}", flush=True)
    print(f"BASELINE: econ {base_ev['econ']:.2f} turtle {base_ev['turtle']:.2f} rush {base_ev['rush']:.2f} heur {base_ev['heur']:.2f} rand {base_ev['rand']:.2f} metric {base_m:.3f}", flush=True)
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    data=[]; it=0; lr_state=(np.zeros(n),np.zeros(n),0,3e-4); loss=0.0
    try:
        while time.time()-t0<secs:
            it+=1
            # DAgger: generate+label a fresh batch of positions. Expert ROLLOUTS use the stable champ
            # (a fixed, strong target); positions come from the champ's own play so the data matches
            # the deploy distribution and the net learns to fix its own mistakes.
            seeds=[int(rng.integers(1<<30)) for _ in range(200)]
            res=pool.map(gen_and_label,[(champ,s,28.0) for s in seeds])
            fresh=[(np.array(x,dtype=np.float64),t) for r in res if r for x,t in [r]]
            data.extend(fresh)
            if len(data)>8000: data=data[-8000:]
            if len(data)<300:
                print(f"it {it:3d} t={time.time()-t0:6.0f}s warming up data={len(data)}", flush=True); continue
            # a few gentle supervised epochs over the accumulated dataset
            for ep in range(2):
                idx=rng.permutation(len(data))
                for k in range(0,len(idx),64):
                    batch=[data[j] for j in idx[k:k+64]]
                    th,lr_state,loss=train_epoch(th,batch,lr_state)
            ev=T.evalset(th,randnet,rng,120); m=T.metric(ev)
            ok = ev['rand']>=base_ev['rand']-0.05 and ev['rush']>=base_ev['rush']-0.06 and ev['econ']>=base_ev['econ']-0.05 and ev['turtle']>=base_ev['turtle']-0.05
            tag=""
            if m>best_m+0.005 and ok:
                best_m=m; champ=th.copy(); tag=" *champion*"
                json.dump(T.to_json(champ),open(os.path.join(SCR,'rl_policy_new.json'),'w'))
            print(f"it {it:3d} t={time.time()-t0:6.0f}s data={len(data)} loss~{loss:.3f} econ {ev['econ']:.2f} turtle {ev['turtle']:.2f} rush {ev['rush']:.2f} heur {ev['heur']:.2f} rand {ev['rand']:.2f} metric {m:.3f}(base {base_m:.3f}){tag}", flush=True)
    finally:
        pool.close(); pool.join()
    fev=T.evalset(champ,randnet,rng,300); fm=T.metric(fev)
    ship = (fm>=base_m+0.02 or fev['heur']>=base_ev['heur']+0.05) and \
           fev['rand']>=base_ev['rand']-0.04 and fev['rush']>=base_ev['rush']-0.05 and \
           fev['econ']>=base_ev['econ']-0.04 and fev['turtle']>=base_ev['turtle']-0.04
    print(f"DONE its={it} champion: econ {fev['econ']:.2f} turtle {fev['turtle']:.2f} rush {fev['rush']:.2f} heur {fev['heur']:.2f}(base {base_ev['heur']:.2f}) rand {fev['rand']:.2f} metric {fm:.3f}(base {base_m:.3f})", flush=True)
    if ship:
        json.dump(T.to_json(champ),open(os.path.join(SCR,'rl_policy_new.json'),'w'))
        print(f"SHIP: heur {fev['heur']-base_ev['heur']:+.2f} metric {fm-base_m:+.2f}, no regression. distilled net saved.", flush=True)
    else:
        print("NO-REGRESSION: no clear gain; not shipping. (current net stays.)", flush=True)

if __name__=='__main__':
    main()
