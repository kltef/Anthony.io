#!/usr/bin/env python3
# VALUE-NETWORK trainer for the State.io planner (AlphaZero-style leaf evaluation).
#
# The planner currently scores a simulated leaf board with a crude heuristic:
#   tanh(0.22 * (myStrength - bestEnemyStrength)),  strength = states + troops/30.
# That's the weakest link in the search. This trains a LEARNED value net V(board)->[-1,1] that
# predicts the actual game OUTCOME from a board state, so every MCTS rollout leaf is evaluated far
# more accurately. Unlike the policy net (a generalist move-scorer that hit a capacity trade-off),
# a value net is a single regression target, so it improves cleanly.
#
#   * 12 owner-relative board features (see vfeats — MUST match the JS extractor exactly).
#   * MLP 12-32-16-1, tanh hidden, tanh output.
#   * Monte-Carlo regression: self-play games with the current policy net; each sampled state is
#     labelled with that game's FINAL result for the owner. Adam + MSE.
#   * Writes tools/value_net.json and prints held-out accuracy (sign-accuracy + correlation).
#   usage: python3 tools/train_value.py [seconds]
import numpy as np, json, time, sys, os
from multiprocessing import Pool
import train_rl as T

CAP=T.CAP; SCR=T.SCR; POLICY=T.POLICY
VARCH=[12,32,16,1]

# ---------------- board-state features (CONTRACT: replicate exactly in index.html valueFeats) ----------------
def vfeats_from(owner, statesOwner, statesTroops, armies, N):
    # statesOwner/statesTroops: arrays over states; armies: list of dict(owner,count)
    allT=0.0; neu=0; own_s=0; own_t=0.0
    cntS={}; cntT={}; bigEnemy=0.0
    for i in range(N):
        o=int(statesOwner[i]); tr=float(statesTroops[i])
        if o==0: neu+=1; continue
        allT+=tr
        cntS[o]=cntS.get(o,0)+1; cntT[o]=cntT.get(o,0.0)+tr
        if o==owner: own_s+=1; own_t+=tr
        elif tr>bigEnemy: bigEnemy=tr
    ifOwn=0.0; ifAll=0.0; aliveset=set(cntS.keys())
    for a in armies:
        ao=int(a['owner'])
        if ao==0: continue
        ifAll+=a['count']; aliveset.add(ao)
        if ao==owner: ifOwn+=a['count']
    maxES=0; maxET=0.0
    for o in cntS:
        if o==owner: continue
        if cntS[o]>maxES: maxES=cntS[o]
        if cntT[o]>maxET: maxET=cntT[o]
    aT=max(1.0,allT)
    own_avg=own_t/max(1,own_s)
    return [
        own_s/N,                       # 0 own territory
        own_t/aT,                      # 1 own troop share
        maxES/N,                       # 2 strongest rival territory
        maxET/aT,                      # 3 strongest rival troop share
        (own_s-maxES)/N,               # 4 territory lead
        (own_t-maxET)/aT,              # 5 troop-share lead
        neu/N,                         # 6 neutral fraction left
        len(aliveset)/5.0,             # 7 players still alive
        own_avg/CAP,                   # 8 own avg stack (over-extension if low)
        bigEnemy/CAP,                  # 9 biggest single enemy stack (threat)
        own_t/(N*CAP),                 # 10 absolute board control
        (2*ifOwn-ifAll)/aT,            # 11 in-flight balance (mine vs theirs)
    ]
def vfeats(g, owner):
    return vfeats_from(owner, g['owner'], g['troops'], g['armies'], g['N'])
def result(g, owner):
    o=g['owner']; al=T.alive_owners(g)
    if al=={owner}: return 1.0
    if owner not in al: return -1.0
    own=int((o==owner).sum())
    others=[int((o==p).sum()) for p in al if p!=owner]
    return max(-1.0, min(1.0, (own-(max(others) if others else 0))/g['N']*1.5))

# ---------------- value MLP: forward + manual backprop ----------------
def vnparams():
    n=0
    for i in range(len(VARCH)-1): n+=VARCH[i]*VARCH[i+1]+VARCH[i+1]
    return n
def vunpack(th):
    i=0; L=[]
    for k in range(len(VARCH)-1):
        ni,no=VARCH[k],VARCH[k+1]
        W=th[i:i+ni*no].reshape(no,ni); i+=ni*no
        b=th[i:i+no]; i+=no
        L.append((W,b))
    return L
def vfwd(th, X):
    L=vunpack(th); a=X; acts=[X]
    for k,(W,b) in enumerate(L):
        z=a@W.T+b; a=np.tanh(z); acts.append(a)
    return a[:,0], (acts, L)
def vbwd(cache, dout):
    acts,L=cache; grads=[None]*len(L)
    da=dout[:,None]                              # (M,1), output is tanh(zlast)
    for k in range(len(L)-1,-1,-1):
        a_out=acts[k+1]; a_in=acts[k]
        dz=da*(1-a_out*a_out)                    # tanh' at every layer (incl. output)
        W,b=L[k]
        grads[k]=(dz.T@a_in, dz.sum(0))
        da=dz@W
    parts=[]
    for (dW,db) in grads: parts.append(dW.ravel()); parts.append(db)
    return np.concatenate(parts)
def vto_json(th):
    L=vunpack(th)
    return {"format":"value-v1","num_features":VARCH[0],
            "layers":[{"W":W.tolist(),"b":b.tolist(),"act":"tanh"} for (W,b) in L]}

# ---------------- self-play -> Monte-Carlo value samples ----------------
def gen_value_data(args):
    theta, seed = args
    rng=np.random.default_rng(seed)
    layers,noop=T.unpack(theta)
    P=int(rng.choice([2,3,4])); N=18
    g=T.new_game(N,P,rng)
    timers={p:rng.uniform(0.2,1.0) for p in range(1,P+1)}
    caps=[]; t=0.0; nextcap=rng.uniform(2.0,5.0)
    while t<75.0:
        T.step(g,0.5); t+=0.5
        for p in range(1,P+1):
            timers[p]-=0.5
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0)
                mv=T.choose(layers,noop,g,p);
                if mv: T.send(g,mv[0],mv[1])
        if t>=nextcap:
            nextcap=t+rng.uniform(3.0,7.0)
            for p in range(1,P+1):
                if (g['owner']==p).any(): caps.append((p, vfeats(g,p)))
        if len(T.alive_owners(g))<=1: break
    out=[]
    for (p,f) in caps: out.append((f, result(g,p)))
    return out

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 600.0
    pol=T.init_from_baseline(POLICY)
    rng=np.random.default_rng(23); n=vnparams()
    th=(rng.standard_normal(n)*(1.0/np.sqrt(32))).astype(np.float64)
    print(f"value params={n} arch={VARCH} cores={os.cpu_count()}", flush=True)
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    data=[]; m=np.zeros(n); v=np.zeros(n); step=0; lr=1e-3
    val_hold=None
    try:
        while time.time()-t0<secs:
            seeds=[int(rng.integers(1<<30)) for _ in range(120)]
            for chunk in pool.map(gen_value_data,[(pol,s) for s in seeds]):
                data.extend(chunk)
            if len(data)>60000: data=data[-60000:]
            if len(data)<2000:
                print(f"t={time.time()-t0:5.0f}s warmup data={len(data)}", flush=True); continue
            # hold out 15% for honest accuracy
            if val_hold is None or rng.random()<0.2:
                idx=rng.permutation(len(data)); cut=int(len(data)*0.85)
                tr_idx, ho_idx = idx[:cut], idx[cut:]
            X=np.array([data[i][0] for i in tr_idx],dtype=np.float64)
            Y=np.array([data[i][1] for i in tr_idx],dtype=np.float64)
            Xh=np.array([data[i][0] for i in ho_idx],dtype=np.float64)
            Yh=np.array([data[i][1] for i in ho_idx],dtype=np.float64)
            for ep in range(3):
                perm=rng.permutation(len(X))
                for k in range(0,len(perm),256):
                    bi=perm[k:k+256]; xb=X[bi]; yb=Y[bi]
                    pred,cache=vfwd(th,xb)
                    dout=(2.0/len(bi))*(pred-yb)            # MSE grad
                    grad=vbwd(cache,dout)
                    step+=1; b1,b2,eps=0.9,0.999,1e-8
                    m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad*grad
                    th=th-lr*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+eps)
            ph,_=vfwd(th,Xh); mse=float(np.mean((ph-Yh)**2))
            mask=np.abs(Yh)>0.1; sign=float(np.mean(np.sign(ph[mask])==np.sign(Yh[mask]))) if mask.any() else 0.0
            corr=float(np.corrcoef(ph,Yh)[0,1]) if len(Yh)>2 else 0.0
            json.dump(vto_json(th),open(os.path.join(SCR,'value_net.json'),'w'))
            print(f"t={time.time()-t0:5.0f}s data={len(data)} mse={mse:.3f} sign-acc={sign:.3f} corr={corr:.3f}", flush=True)
    finally:
        pool.close(); pool.join()
    print("DONE. value_net.json written.", flush=True)

if __name__=='__main__':
    main()
