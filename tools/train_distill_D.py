#!/usr/bin/env python3
# Phase D: expert iteration with a STRONGER teacher.
#
# The current 64-net was distilled from a weak/noisy expert (3 full-game rollouts, 1-ply, NO value net).
# This builds a better teacher: each candidate move is scored by short truncated rollouts BOOTSTRAPPED
# with the learned value net (AlphaZero-style n-step+value: lower variance than playing to terminal,
# and value-guided), averaged over several samples. Better labels -> the SAME 64-net architecture,
# re-distilled (warm-started from the current net), should beat the current net. Arena decides ship.
#   usage: python3 tools/train_distill_D.py [seconds]
import numpy as np, json, time, sys, os
from multiprocessing import Pool
import train_rl as T
import train_value as V

ARCHB=[12,64,64,1]
T.ARCH=ARCHB
FN=T.FN; SCR=T.SCR; POLICY=T.POLICY

# ---- load the value net (flat theta in VARCH layout) for the teacher's leaf eval ----
def load_value_theta(path):
    p=json.load(open(path)); parts=[]
    for l in p['layers']: parts.append(np.array(l['W']).ravel()); parts.append(np.array(l['b']))
    return np.concatenate(parts).astype(np.float64)
VPATH=os.path.join(SCR,'value_net.json')
VTH=load_value_theta(VPATH)

# ---- student net (12-64-64-1): forward + manual backprop ----
def fwdB(th, X):
    layers,_=T.unpack(th); a=X; post=[X]
    for k,(W,b) in enumerate(layers):
        z=a@W.T+b; a=np.tanh(z) if k<len(layers)-1 else z; post.append(a)
    return a[:,0], (post, layers)
def bwdB(cache, ds):
    post,layers=cache; da=ds[:,None]; parts=[None]*len(layers)
    for k in range(len(layers)-1,-1,-1):
        a_out=post[k+1]; a_in=post[k]
        dz=da if k==len(layers)-1 else da*(1-a_out*a_out)
        W,b=layers[k]; parts[k]=(dz.T@a_in, dz.sum(0)); da=dz@W
    out=[]
    for dW,db in parts: out.append(dW.ravel()); out.append(db)
    return np.concatenate(out)

# ---- base/rollout net = current 64-net (loaded directly) ----
def load_layers(path):
    p=json.load(open(path)); return [(np.array(l['W']),np.array(l['b']),l['act']) for l in p['layers']], p.get('noop_bias',0.0)
def fwd_layers(L, X):
    a=X
    for (W,b,act) in L: a=np.tanh(a@W.T+b) if act=='tanh' else a@W.T+b
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

# ---- STRONGER TEACHER: truncated rollout (H seconds) bootstrapped with the value net ----
def trunc_value(g, L, noop, owner, rng, H=10.0, dt=0.6):
    P=int(g['owner'].max()); timers={p:rng.uniform(0.1,0.6) for p in range(1,P+1)}; t=0.0
    while t<H:
        T.step(g,dt); t+=dt
        for p in range(1,P+1):
            timers[p]-=dt
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0); mv=choose_net(L,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    al=T.alive_owners(g)
    if al=={owner}: return 1.0
    if owner not in al: return -1.0
    return float(V.vfwd(VTH, np.array([V.vfeats(g,owner)]))[0][0])   # value-net bootstrap at the horizon (vfwd returns (vals,cache))
def expert_label(g, L, noop, owner, rng, nshort=4):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    if len(srcs)>3: srcs=srcs[np.argsort(-t[srcs])[:3]]
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    cands=[(int(si),int(ti)) for si in srcs for ti in tgts]
    glob=T.globals_for(g,owner)
    X=np.array([T.feats12(g,si,owner,np.array([ti]),glob)[0] for (si,ti) in cands])
    sc=fwd_layers(L,X); keep=np.argsort(-sc)[:6]
    cands=[cands[i] for i in keep]; Xk=X[keep]
    best=-1e9; bj=-1
    for j,(si,ti) in enumerate(cands):
        v=sum(trunc_value((lambda gg:(T.send(gg,si,ti) or gg))(clone(g)),L,noop,owner,rng) for _ in range(nshort))/nshort
        if v>best: best=v; bj=j
    vno=sum(trunc_value(clone(g),L,noop,owner,rng) for _ in range(nshort))/nshort
    return Xk, (len(cands) if vno>best+1e-9 else bj)
def gen_and_label(args):
    L_ser, noop, seed, max_t = args
    rng=np.random.default_rng(seed)
    L=[(np.array(W),np.array(b),act) for (W,b,act) in L_ser]
    P=int(rng.choice([2,3,4])); N=18; g=T.new_game(N,P,rng)
    timers={p:rng.uniform(0.2,1.0) for p in range(1,P+1)}; stop=rng.uniform(3.0,max_t); t=0.0
    while t<stop:
        T.step(g,0.25); t+=0.25
        for p in range(1,P+1):
            timers[p]-=0.25
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0); mv=choose_net(L,noop,g,p)
                if mv: T.send(g,mv[0],mv[1])
        if len(T.alive_owners(g))<=1: break
    alive=[p for p in range(1,P+1) if (g['owner']==p).any()]; rng.shuffle(alive)
    for owner in alive:
        lab=expert_label(g,L,noop,int(owner),rng)
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

def warm_start(L, noop):   # current 64-net -> flat theta in T.unpack order
    parts=[]
    for (W,b,_) in L: parts.append(W.ravel()); parts.append(b)
    parts.append([noop]); return np.concatenate([np.asarray(p,dtype=np.float64).ravel() for p in parts])

# arch-independent league eval (score explicit layers vs scripted league)
def wr_vs_layers(L2, noop, opp, rng, n=120, N=18):
    cand=('net', L2, noop); o=(opp,None,None) if isinstance(opp,str) else ('net',)+T.unpack(opp); w=0
    for _ in range(n):
        P=int(rng.choice([2,3])); ci=int(rng.integers(P)); seats=[cand if p==ci else o for p in range(P)]
        res=T.play(seats,rng,N=N)
        if res[ci+1]>=max(res.values())-1e-9: w+=1
    return w/n
def evalset_layers(L2, noop, randnet, rng, n=120):
    return {'econ':wr_vs_layers(L2,noop,'econ',rng,n),'turtle':wr_vs_layers(L2,noop,'turtle',rng,n),
            'rush':wr_vs_layers(L2,noop,'rush',rng,n),'heur':wr_vs_layers(L2,noop,'heuristic',rng,n),
            'rand':wr_vs_layers(L2,noop,randnet,rng,n)}

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 1000.0
    L,noop=load_layers(POLICY); L_ser=[(W.tolist(),b.tolist(),act) for (W,b,act) in L]; L2=[(W,b) for (W,b,_) in L]
    n=T.nparams(); rng=np.random.default_rng(43)
    th=warm_start(L,noop); assert len(th)==n,(len(th),n)
    randnet=rng.standard_normal(n)*0.5
    base_ev=evalset_layers(L2,noop,randnet,rng,200); base_m=T.metric(base_ev); best_m=base_m; champ=th.copy()
    print(f"D net params={n} arch={ARCHB} value-net teacher (trunc-rollout+value bootstrap) cores={os.cpu_count()}", flush=True)
    print(f"BASELINE current 64-net: econ {base_ev['econ']:.2f} turtle {base_ev['turtle']:.2f} rush {base_ev['rush']:.2f} heur {base_ev['heur']:.2f} rand {base_ev['rand']:.2f} metric {base_m:.3f}", flush=True)
    json.dump(T.to_json(champ),open(os.path.join(SCR,'policy_D_new.json'),'w'))   # warm-start = initial (so arena always has a file)
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    data=[]; it=0; st=(np.zeros(n),np.zeros(n),0,1.5e-4); loss=0.0   # gentle LR: preserve warm-start, improve via better labels
    try:
        while time.time()-t0<secs:
            it+=1
            seeds=[int(rng.integers(1<<30)) for _ in range(200)]
            for r in pool.map(gen_and_label,[(L_ser,noop,s,28.0) for s in seeds]):
                if r: data.append((np.array(r[0],dtype=np.float64),r[1]))
            if len(data)>9000: data=data[-9000:]
            if len(data)<400:
                print(f"it {it:3d} t={time.time()-t0:5.0f}s warmup data={len(data)}", flush=True); continue
            for ep in range(2):
                idx=rng.permutation(len(data))
                for k in range(0,len(idx),64):
                    th,st,loss=train_epoch(th,[data[j] for j in idx[k:k+64]],st)
            ev=T.evalset(th,randnet,rng,120); m=T.metric(ev)
            ok = ev['rush']>=base_ev['rush']-0.06 and ev['econ']>=base_ev['econ']-0.05 and ev['turtle']>=base_ev['turtle']-0.05
            tag=""
            if m>best_m and ok:
                best_m=m; champ=th.copy(); tag=" *champion*"
                json.dump(T.to_json(champ),open(os.path.join(SCR,'policy_D_new.json'),'w'))
            print(f"it {it:3d} t={time.time()-t0:5.0f}s data={len(data)} loss~{loss:.3f} econ {ev['econ']:.2f} turtle {ev['turtle']:.2f} rush {ev['rush']:.2f} heur {ev['heur']:.2f} rand {ev['rand']:.2f} metric {m:.3f}(base {base_m:.3f}){tag}", flush=True)
    finally:
        pool.close(); pool.join()
    fev=T.evalset(champ,randnet,rng,300); fm=T.metric(fev)
    print(f"DONE its={it} D champion: econ {fev['econ']:.2f} turtle {fev['turtle']:.2f} rush {fev['rush']:.2f} heur {fev['heur']:.2f}(base {base_ev['heur']:.2f}) rand {fev['rand']:.2f} metric {fm:.3f}(base {base_m:.3f})", flush=True)
    json.dump(T.to_json(champ),open(os.path.join(SCR,'policy_D_new.json'),'w'))

if __name__=='__main__':
    main()
