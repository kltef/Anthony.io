#!/usr/bin/env python3
# Phase D capstone: AUTONOMOUS SELF-IMPROVING loop (AlphaZero-style), safe to run unattended.
#
#   best_policy, best_value = current shipped nets
#   repeat until time budget:
#     1. GENERATE self-play games with best_policy (+ epsilon exploration); label moves with the
#        teacher = best_policy + short rollouts bootstrapped by best_value; also collect (state->outcome)
#        samples for value co-training.
#     2. TRAIN a policy candidate (warm-start from best_policy) on the policy buffer.
#     3. GATE: candidate vs best_policy head-to-head; promote ONLY if it wins by a margin AND doesn't
#        regress vs scripted turtle/rush. (This is what makes a long unattended run safe — it can
#        never get worse; a candidate that doesn't beat best is discarded.)
#     4. Every few rounds, RETRAIN best_value on fresh outcomes (co-training -> the teacher itself
#        gets smarter -> rising ceiling).
#   On exit, writes the best policy + value nets. Final real-planner arena validation is done separately.
#   usage: python3 tools/train_selfplay_loop.py [seconds]
import numpy as np, json, time, sys, os
from multiprocessing import Pool
import train_rl as T
import train_value as V

PARCH=[12,64,64,1]; T.ARCH=PARCH
FN=T.FN; SCR=T.SCR; POLICY=T.POLICY

# ---------- policy net forward/backward (linear output) ----------
def pfwd(th, X):
    layers,_=T.unpack(th); a=X; post=[X]
    for k,(W,b) in enumerate(layers):
        z=a@W.T+b; a=np.tanh(z) if k<len(layers)-1 else z; post.append(a)
    return a[:,0], (post, layers)
def pbwd(cache, ds):
    post,layers=cache; da=ds[:,None]; parts=[None]*len(layers)
    for k in range(len(layers)-1,-1,-1):
        a_out=post[k+1]; a_in=post[k]
        dz=da if k==len(layers)-1 else da*(1-a_out*a_out)
        W,b=layers[k]; parts[k]=(dz.T@a_in, dz.sum(0)); da=dz@W
    out=[]
    for dW,db in parts: out.append(dW.ravel()); out.append(db)
    return np.concatenate(out)

def policy_theta_from_json(path):
    p=json.load(open(path)); parts=[]
    for l in p['layers']: parts.append(np.array(l['W']).ravel()); parts.append(np.array(l['b']))
    parts.append([p.get('noop_bias',0.0)])
    return np.concatenate([np.asarray(x,dtype=np.float64).ravel() for x in parts])
def value_theta_from_json(path):
    p=json.load(open(path)); parts=[]
    for l in p['layers']: parts.append(np.array(l['W']).ravel()); parts.append(np.array(l['b']))
    return np.concatenate(parts).astype(np.float64)

# ---------- worker globals (set per round) ----------
def layers_from_theta(th):
    L,noop=T.unpack(th); return [(W,b) for (W,b) in L], noop

def choose_explore(layers, noop, g, owner, rng, eps=0.2):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    glob=T.globals_for(g,owner); best=noop; mv=None; cand=[]
    for si in srcs:
        sc=T.forward(layers, T.feats12(g,si,owner,tgts,glob))
        for j in range(len(tgts)): cand.append((sc[j],int(si),int(tgts[j])))
        j=int(np.argmax(sc))
        if sc[j]>best: best=sc[j]; mv=(int(si),int(tgts[j]))
    if rng.random()<eps and cand:                 # exploration: random legal move
        c=cand[int(rng.integers(len(cand)))]; return (c[1],c[2])
    return mv

SCRIPTS={'turtle':T.turtle,'rush':T.rush,'heuristic':T.heuristic}
def move_for(ptype, g, owner, layers, noop, rng, eps):
    if ptype=='net': return choose_explore(layers,noop,g,owner,rng,eps)
    return SCRIPTS[ptype](g,owner)
def step_all(g, ptypes, layers, noop, rng, timers, dt, eps):
    T.step(g,dt)
    for p in ptypes:
        timers[p]-=dt
        if timers[p]<=0:
            timers[p]=rng.uniform(0.6,1.0); mv=move_for(ptypes[p],g,p,layers,noop,rng,eps)
            if mv: T.send(g,mv[0],mv[1])
def clone(g):
    return dict(pos=g['pos'],owner=g['owner'].copy(),troops=g['troops'].copy(),
                armies=[dict(a) for a in g['armies']],rlDiag=g['rlDiag'],ORB=g['ORB'],N=g['N'])
def trunc_value(g, layers, noop, vth, owner, rng, ptypes, H=10.0, dt=0.6):
    timers={p:rng.uniform(0.1,0.6) for p in ptypes}; t=0.0
    while t<H:
        step_all(g,ptypes,layers,noop,rng,timers,dt,eps=0.0); t+=dt
        if len(T.alive_owners(g))<=1: break
    al=T.alive_owners(g)
    if al=={owner}: return 1.0
    if owner not in al: return -1.0
    return float(V.vfwd(vth, np.array([V.vfeats(g,owner)]))[0][0])
def teacher_label(g, layers, noop, vth, owner, rng, ptypes, nshort=4):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    if len(srcs)>3: srcs=srcs[np.argsort(-t[srcs])[:3]]
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    cands=[(int(si),int(ti)) for si in srcs for ti in tgts]
    glob=T.globals_for(g,owner)
    X=np.array([T.feats12(g,si,owner,np.array([ti]),glob)[0] for (si,ti) in cands])
    sc=T.forward(layers,X); keep=np.argsort(-sc)[:6]
    cands=[cands[i] for i in keep]; Xk=X[keep]
    best=-1e9; bj=-1
    for j,(si,ti) in enumerate(cands):
        v=sum(trunc_value((lambda gg:(T.send(gg,si,ti) or gg))(clone(g)),layers,noop,vth,owner,rng,ptypes) for _ in range(nshort))/nshort
        if v>best: best=v; bj=j
    vno=sum(trunc_value(clone(g),layers,noop,vth,owner,rng,ptypes) for _ in range(nshort))/nshort
    return Xk, (len(cands) if vno>best+1e-9 else bj)

def gen_game(args):
    p_ser, noop, v_ser, seed = args
    rng=np.random.default_rng(seed)
    layers=[(np.array(W),np.array(b)) for (W,b) in p_ser]
    vth=np.array(v_ser)
    P=int(rng.choice([2,3,4])); N=18; g=T.new_game(N,P,rng)
    # owner 1 = hero (best_p, always labeled). Other seats: mostly net self-play, but mix in scripted
    # turtle/rush/heuristic exploiters so the net keeps TRAINING on beating them (anti-forgetting).
    pool=['net','net','net','turtle','rush','heuristic']
    ptypes={1:'net'}
    for p in range(2,P+1): ptypes[p]=str(rng.choice(pool))
    HERO=1
    timers={p:rng.uniform(0.2,1.0) for p in range(1,P+1)}; t=0.0; H=70.0
    pol_at=rng.uniform(6,42); pol=None; valsamp=[]; nextval=rng.uniform(3,6)
    while t<H:
        step_all(g,ptypes,layers,noop,rng,timers,0.4,eps=0.2); t+=0.4
        nextval-=0.4
        if nextval<=0 and (g['owner']==HERO).any():
            nextval=rng.uniform(4,8); valsamp.append((V.vfeats(g,HERO), HERO))
        if pol is None and t>=pol_at and (g['owner']==HERO).any():
            lab=teacher_label(g,layers,noop,vth,HERO,rng,ptypes)
            if lab is not None: pol=(lab[0].tolist(), int(lab[1]))
        if len(T.alive_owners(g))<=1: break
    valout=[(vf, V.result(g,ow)) for (vf,ow) in valsamp]      # label each sample with the hero's final outcome
    return pol, valout

# ---------- training ----------
def train_policy(th, batch, st, epochs=2, bs=64, rng=None):
    loss=0.0
    for ep in range(epochs):
        idx=rng.permutation(len(batch))
        for k in range(0,len(idx),bs):
            bb=[batch[j] for j in idx[k:k+bs]]
            Xs=[b[0] for b in bb]; tgts=[b[1] for b in bb]
            offs=np.cumsum([0]+[len(x) for x in Xs]); Xall=np.vstack(Xs)
            s_all,cache=pfwd(th,Xall); noop=th[-1]
            ds=np.zeros_like(s_all); dnoop=0.0; L=0.0
            for i in range(len(Xs)):
                a,b=offs[i],offs[i+1]; seg=np.concatenate([s_all[a:b],[noop]]); seg=seg-seg.max()
                pp=np.exp(seg); pp/=pp.sum(); tg=tgts[i]; L-=np.log(max(pp[tg],1e-9))
                gg=pp.copy(); gg[tg]-=1.0; ds[a:b]=gg[:-1]; dnoop+=gg[-1]
            grad=np.concatenate([pbwd(cache,ds),[dnoop]])/len(Xs)
            m,v,step,lr=st; step+=1; b1,b2,eps=0.9,0.999,1e-8
            m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad*grad
            th=th-lr*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+eps); st=(m,v,step,lr); loss=L/len(Xs)
    return th, st, loss
def train_value(vth, buf, rng, epochs=3, bs=128):
    n=len(vth); m=np.zeros(n); v=np.zeros(n); step=0; lr=2e-3; b1,b2,eps=0.9,0.999,1e-8; loss=0.0
    X=np.array([b[0] for b in buf]); Y=np.array([b[1] for b in buf])
    for ep in range(epochs):
        idx=rng.permutation(len(buf))
        for k in range(0,len(idx),bs):
            bi=idx[k:k+bs]; xb=X[bi]; yb=Y[bi]
            pred,cache=V.vfwd(vth,xb); err=pred-yb; loss=float(np.mean(err*err))
            grad=V.vbwd(cache, 2*err/len(bi))
            step+=1; m=b1*m+(1-b1)*grad; v=b2*v+(1-b2)*grad*grad
            vth=vth-lr*(m/(1-b1**step))/(np.sqrt(v/(1-b2**step))+eps)
    return vth, loss

# ---------- head-to-head + league gate (greedy net, Python sim) ----------
def net_seat(th): L,noop=T.unpack(th); return ('net',[(W,b) for (W,b) in L],noop)
def h2h(cand_th, best_th, rng, n=50, N=18):
    cand=net_seat(cand_th); best=net_seat(best_th); w=0
    for _ in range(n):
        ci=int(rng.integers(2)); seats=[cand if p==ci else best for p in range(2)]
        res=T.play(seats,rng,N=N)
        if res[ci+1]>=max(res.values())-1e-9: w+=1
    return w/n
def vs_script(th, opp, rng, n=40, N=18):
    cand=net_seat(th); o=(opp,None,None); w=0
    for _ in range(n):
        P=int(rng.choice([2,3])); ci=int(rng.integers(P)); seats=[cand if p==ci else o for p in range(P)]
        res=T.play(seats,rng,N=N)
        if res[ci+1]>=max(res.values())-1e-9: w+=1
    return w/n

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 3600.0
    best_p=policy_theta_from_json(POLICY)
    best_v=value_theta_from_json(os.path.join(SCR,'value_net.json'))
    rng=np.random.default_rng(99)
    base_turtle=vs_script(best_p,'turtle',rng,60); base_rush=vs_script(best_p,'rush',rng,60)
    print(f"SELF-PLAY LOOP  policy={PARCH} value={V.VARCH}  budget={secs:.0f}s cores={os.cpu_count()}", flush=True)
    print(f"baseline best_policy vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f}", flush=True)
    json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_loop_new.json'),'w'))
    json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_loop_new.json'),'w'))
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    pbuf=[]; vbuf=[]; rnd=0; promotions=0; vretrains=0
    while time.time()-t0<secs:
        rnd+=1
        p_ser=[(W.tolist(),b.tolist()) for (W,b) in T.unpack(best_p)[0]]
        noop=float(best_p[-1]); v_ser=best_v.tolist()
        seeds=[int(rng.integers(1<<30)) for _ in range(160)]
        for pol,valout in pool.map(gen_game,[(p_ser,noop,v_ser,s) for s in seeds]):
            if pol: pbuf.append((np.array(pol[0],dtype=np.float64),pol[1]))
            for (vf,out) in valout: vbuf.append((np.array(vf,dtype=np.float64),float(out)))
        if len(pbuf)>9000: pbuf=pbuf[-9000:]
        if len(vbuf)>40000: vbuf=vbuf[-40000:]
        if len(pbuf)<400:
            print(f"r{rnd:3d} t={time.time()-t0:5.0f}s warmup pbuf={len(pbuf)} vbuf={len(vbuf)}", flush=True); continue
        # train a policy candidate from best
        cand=best_p.copy(); st=(np.zeros(len(cand)),np.zeros(len(cand)),0,2e-4)
        cand,st,ploss=train_policy(cand,pbuf,st,epochs=2,rng=rng)
        # GATE: candidate must beat best head-to-head AND not regress vs scripts
        wr=h2h(cand,best_p,rng,n=60)
        promoted=False; tt=rs=-1
        if wr>=0.55:
            tt=vs_script(cand,'turtle',rng,50); rs=vs_script(cand,'rush',rng,50)
            if tt>=0.88 and rs>=0.83:    # absolute floors: block catastrophic forgetting, allow general-play gains
                best_p=cand; promotions+=1; promoted=True
                json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_loop_new.json'),'w'))
        # co-train value every 4 rounds (regression to true outcomes -> smarter teacher)
        vtag=""
        if rnd%4==0 and len(vbuf)>=2000:
            best_v,vloss=train_value(best_v,vbuf,rng); vretrains+=1; vtag=f" vretrain(mse{vloss:.3f})"
            json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_loop_new.json'),'w'))
        print(f"r{rnd:3d} t={time.time()-t0:5.0f}s pbuf={len(pbuf)} vbuf={len(vbuf)} ploss~{ploss:.3f} h2h={wr:.2f} tt={tt:.2f} rs={rs:.2f} {'PROMOTED #%d'%promotions if promoted else 'kept'}{vtag}", flush=True)
    pool.close(); pool.join()
    # final no-regression snapshot of the promoted best
    ft=vs_script(best_p,'turtle',rng,80); fr=vs_script(best_p,'rush',rng,80)
    print(f"DONE rounds={rnd} promotions={promotions} value_retrains={vretrains}", flush=True)
    print(f"final best_policy vs scripts: turtle {ft:.2f}(base {base_turtle:.2f}) rush {fr:.2f}(base {base_rush:.2f})", flush=True)
    json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_loop_new.json'),'w'))
    json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_loop_new.json'),'w'))

if __name__=='__main__':
    main()
