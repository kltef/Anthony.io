#!/usr/bin/env python3
# BIGGER policy net via Monte-Carlo expert distillation.
#
# The 12-32-32-1 move-scorer hit a capacity trade-off (gaining general play cost turtle-resistance).
# This trains a larger 12-64-64-1 net (~3x capacity) by distilling the SAME full-game Monte-Carlo
# expert (1-ply lookahead + rollouts of the current shipped net) into it.  If the bigger net can
# hold turtle/econ/rush AND improve heur (clearing the no-regression gate), capacity was the limit
# and we ship it (drop-in: the in-game rlScore handles any layer size).  If not, capacity isn't the
# issue and we keep the current net.
#   usage: python3 tools/train_distill_big.py [seconds]
import numpy as np, json, time, sys, os
from multiprocessing import Pool
import train_rl as T

import os as _os
_W=int(_os.environ.get('NET_WIDTH','64'))      # Phase C: configurable hidden width via NET_WIDTH
ARCHB=[12,_W,_W,1]
T.ARCH=ARCHB                      # so T.unpack / T.to_json / T.evalset operate on the BIGGER net
FN=T.FN; CAP=T.CAP; SCR=T.SCR; POLICY=T.POLICY

# ---- bigger net: forward + manual backprop (theta layout == T.unpack: per-layer W,b then noop) ----
def fwdB(th, X):
    layers,_=T.unpack(th); a=X; post=[X]
    for k,(W,b) in enumerate(layers):
        z=a@W.T+b; a=np.tanh(z) if k<len(layers)-1 else z; post.append(a)
    return a[:,0], (post, layers)
def bwdB(cache, ds):
    post,layers=cache; grads=[]
    da=ds[:,None]
    parts=[None]*len(layers)
    for k in range(len(layers)-1,-1,-1):
        a_out=post[k+1]; a_in=post[k]
        dz=da if k==len(layers)-1 else da*(1-a_out*a_out)
        W,b=layers[k]; parts[k]=(dz.T@a_in, dz.sum(0)); da=dz@W
    out=[]
    for dW,db in parts: out.append(dW.ravel()); out.append(db)
    return np.concatenate(out)
def scoreB(th, X):
    s,_=fwdB(th,X); return s

# ---- rollout/expert net = current shipped net, loaded directly (independent of ARCHB) ----
def load_layers(path):
    p=json.load(open(path)); L=[(np.array(l['W']),np.array(l['b']),l['act']) for l in p['layers']]
    return L, p.get('noop_bias',0.0)
def fwd_layers(L, X):
    a=X
    for k,(W,b,act) in enumerate(L):
        z=a@W.T+b; a=np.tanh(z) if act=='tanh' else z
    return a[:,0]
def choose_net(L, noop, g, owner):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    glob=T.globals_for(g,owner); best=noop; mv=None
    for si in srcs:
        sc=fwd_layers(L, T.feats12(g,si,owner,tgts,glob)); j=int(np.argmax(sc))
        if sc[j]>best: best=sc[j]; mv=(int(si),int(tgts[j]))
    return mv

def clone(g):
    return dict(pos=g['pos'],owner=g['owner'].copy(),troops=g['troops'].copy(),
                armies=[dict(a) for a in g['armies']],rlDiag=g['rlDiag'],ORB=g['ORB'],N=g['N'])
def mc_result(g, layers, noop, owner, rng, H=36.0, dt=0.6):
    P=int(g['owner'].max()); timers={p:rng.uniform(0.1,0.6) for p in range(1,P+1)}; t=0.0
    while t<H:
        T.step(g,dt); t+=dt
        for p in range(1,P+1):
            timers[p]-=dt
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0); mv=choose_net(layers,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    o=g['owner']; al=T.alive_owners(g)
    if al=={owner}: return 1.0
    if owner not in al: return -1.0
    own=int((o==owner).sum()); others=[int((o==p).sum()) for p in range(1,P+1) if p!=owner]
    return (own-(max(others) if others else 0))/g['N']
def expert_label(g, layers, noop, owner, rng, nroll=3):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    if len(srcs)>3: srcs=srcs[np.argsort(-t[srcs])[:3]]
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    cands=[(int(si),int(ti)) for si in srcs for ti in tgts]
    glob=T.globals_for(g,owner)
    X=np.array([T.feats12(g,si,owner,np.array([ti]),glob)[0] for (si,ti) in cands])
    sc=fwd_layers(layers,X); keep=np.argsort(-sc)[:6]
    cands=[cands[i] for i in keep]; Xk=X[keep]
    best=-1e9; bj=-1
    for j,(si,ti) in enumerate(cands):
        v=sum(mc_result(((lambda gg: (T.send(gg,si,ti) or gg))(clone(g))),layers,noop,owner,rng) for _ in range(nroll))/nroll
        if v>best: best=v; bj=j
    vno=sum(mc_result(clone(g),layers,noop,owner,rng) for _ in range(nroll))/nroll
    return Xk, (len(cands) if vno>best+1e-9 else bj)
def gen_and_label(args):
    rollL_ser, noop, seed, max_t = args
    rng=np.random.default_rng(seed)
    layers=[(np.array(W),np.array(b),act) for (W,b,act) in rollL_ser]
    P=int(rng.choice([2,3,4])); N=18; g=T.new_game(N,P,rng)
    timers={p:rng.uniform(0.2,1.0) for p in range(1,P+1)}; stop=rng.uniform(3.0,max_t); t=0.0
    while t<stop:
        T.step(g,0.25); t+=0.25
        for p in range(1,P+1):
            timers[p]-=0.25
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0); mv=choose_net(layers,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    alive=[p for p in range(1,P+1) if (g['owner']==p).any()]; rng.shuffle(alive)
    for owner in alive:
        lab=expert_label(g,layers,noop,int(owner),rng)
        if lab is not None: return (lab[0].tolist(), int(lab[1]))
    return None

def train_epoch(th, batch, st):
    Xs=[b[0] for b in batch]; tgts=[b[1] for b in batch]
    offs=np.cumsum([0]+[len(x) for x in Xs]); Xall=np.vstack(Xs)
    s_all,cache=fwdB(th,Xall); noop=th[-1]
    ds=np.zeros_like(s_all); dnoop=0.0; loss=0.0
    for i in range(len(Xs)):
        a,b=offs[i],offs[i+1]; seg=np.concatenate([s_all[a:b],[noop]]); seg=seg-seg.max()
        p=np.exp(seg); p/=p.sum(); tg=tgts[i]; loss-=np.log(max(p[tg],1e-9))
        gg=p.copy(); gg[tg]-=1.0; ds[a:b]=gg[:-1]; dnoop+=gg[-1]
    grad=np.concatenate([bwdB(cache,ds),[dnoop]])/len(Xs)
    m,v,step,lr=st; step+=1; b1,b2,eps=0.9,0.999,1e-8
    m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad*grad
    th=th-lr*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+eps)
    return th,(m,v,step,lr),loss/len(Xs)

# architecture-independent eval: play a net given as explicit (W,b,act) layers vs the league,
# so we can score the CURRENT 32-net even though T.ARCH is monkeypatched to the bigger arch.
def wr_vs_layers(L2, noop, opp, rng, n=120, N=18):
    cand=('net', L2, noop)
    o=(opp,None,None) if isinstance(opp,str) else ('net',)+T.unpack(opp)
    w=0
    for _ in range(n):
        P=int(rng.choice([2,3])); ci=int(rng.integers(P))
        seats=[cand if p==ci else o for p in range(P)]
        res=T.play(seats,rng,N=N)
        if res[ci+1]>=max(res.values())-1e-9: w+=1
    return w/n
def evalset_layers(L2, noop, randnet, rng, n=120):
    return {'econ':wr_vs_layers(L2,noop,'econ',rng,n),'turtle':wr_vs_layers(L2,noop,'turtle',rng,n),
            'rush':wr_vs_layers(L2,noop,'rush',rng,n),'heur':wr_vs_layers(L2,noop,'heuristic',rng,n),
            'rand':wr_vs_layers(L2,noop,randnet,rng,n)}

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 900.0
    rollL,noop=load_layers(POLICY)
    rollL_ser=[(W.tolist(),b.tolist(),act) for (W,b,act) in rollL]
    rollL2=[(W,b) for (W,b,act) in rollL]                      # 2-tuples for T.play/T.choose
    n=T.nparams(); rng=np.random.default_rng(31)
    # small random init for the bigger net; last element is the no-op bias
    th=(rng.standard_normal(n)*0.15).astype(np.float64); th[-1]=noop
    randnet=rng.standard_normal(n)*0.5
    base_ev=evalset_layers(rollL2, noop, randnet, rng, 200)
    base_m=T.metric(base_ev); best_m=base_m; champ=th.copy()
    print(f"BIG net params={n} arch={ARCHB} cores={os.cpu_count()}", flush=True)
    print(f"BASELINE (current 32-net): econ {base_ev['econ']:.2f} turtle {base_ev['turtle']:.2f} rush {base_ev['rush']:.2f} heur {base_ev['heur']:.2f} rand {base_ev['rand']:.2f} metric {base_m:.3f}", flush=True)
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    data=[]; it=0; st=(np.zeros(n),np.zeros(n),0,5e-4); loss=0.0
    try:
        while time.time()-t0<secs:
            it+=1
            seeds=[int(rng.integers(1<<30)) for _ in range(200)]
            for r in pool.map(gen_and_label,[(rollL_ser,noop,s,28.0) for s in seeds]):
                if r: data.append((np.array(r[0],dtype=np.float64),r[1]))
            if len(data)>9000: data=data[-9000:]
            if len(data)<400:
                print(f"it {it:3d} t={time.time()-t0:5.0f}s warmup data={len(data)}", flush=True); continue
            for ep in range(2):
                idx=rng.permutation(len(data))
                for k in range(0,len(idx),64):
                    th,st,loss=train_epoch(th,[data[j] for j in idx[k:k+64]],st)
            ev=T.evalset(th,randnet,rng,120); m=T.metric(ev)
            ok = ev['rand']>=base_ev['rand']-0.05 and ev['rush']>=base_ev['rush']-0.06 and ev['econ']>=base_ev['econ']-0.05 and ev['turtle']>=base_ev['turtle']-0.05
            tag=""
            if m>best_m+0.005 and ok:
                best_m=m; champ=th.copy(); tag=" *champion*"
                json.dump(T.to_json(champ),open(os.path.join(SCR,f'policy_big{_W}_new.json'),'w'))
            print(f"it {it:3d} t={time.time()-t0:5.0f}s data={len(data)} loss~{loss:.3f} econ {ev['econ']:.2f} turtle {ev['turtle']:.2f} rush {ev['rush']:.2f} heur {ev['heur']:.2f} rand {ev['rand']:.2f} metric {m:.3f}(base {base_m:.3f}){tag}", flush=True)
    finally:
        pool.close(); pool.join()
    fev=T.evalset(champ,randnet,rng,300); fm=T.metric(fev)
    ship = (fm>=base_m+0.02 or fev['heur']>=base_ev['heur']+0.05) and \
           fev['rand']>=base_ev['rand']-0.04 and fev['rush']>=base_ev['rush']-0.05 and \
           fev['econ']>=base_ev['econ']-0.04 and fev['turtle']>=base_ev['turtle']-0.04
    print(f"DONE its={it} BIG champion: econ {fev['econ']:.2f} turtle {fev['turtle']:.2f} rush {fev['rush']:.2f} heur {fev['heur']:.2f}(base {base_ev['heur']:.2f}) rand {fev['rand']:.2f} metric {fm:.3f}(base {base_m:.3f})", flush=True)
    if ship:
        json.dump(T.to_json(champ),open(os.path.join(SCR,f'policy_big{_W}_new.json'),'w'))
        print(f"SHIP: bigger net clears the gate (heur {fev['heur']-base_ev['heur']:+.2f}, metric {fm-base_m:+.2f}). saved.", flush=True)
    else:
        print("NO-REGRESSION: bigger capacity did not clear the gate; not shipping. (current net stays.)", flush=True)

if __name__=='__main__':
    main()
