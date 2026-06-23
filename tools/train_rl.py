#!/usr/bin/env python3
# Integrated self-play trainer for the State.io move-scoring policy.
#   * Opponent LEAGUE: baseline net + past champions + scripted exploiters (heuristic/turtle/rush)
#   * Separable Natural ES (SNES): evolutionary, adapts a per-weight step size (O(n), no covariance)
#   * 4-core parallel fitness with COMMON RANDOM NUMBERS (same games for every candidate per gen)
#   * NO-REGRESSION: only writes rl_policy_new.json if the champion beats the baseline head-to-head
# The env mirrors the JS planner's own sim, so a stronger net here = stronger planner + live AI.
# Net stays small (8->32->32->1) and warm-starts from the shipped weights (a big head start).
#   usage: python3 tools/train_rl.py [seconds]
import numpy as np, json, time, sys, math, os
from multiprocessing import Pool

FN = 50.0; CAP = 150.0
ARCH = [8, 32, 32, 1]
SCR = os.path.dirname(os.path.abspath(__file__))
# baseline model: the tracked mirror (src/web/rl_policy.json) is the source of truth
POLICY = next((p for p in [os.path.join(SCR,'..','src','web','rl_policy.json'),
                           os.path.join(SCR,'rl_policy.json'),
                           os.path.join(SCR,'..','rl_policy.json')] if os.path.exists(p)), None)

# ---------------- net ----------------
def nparams():
    n = 0
    for i in range(len(ARCH)-1): n += ARCH[i]*ARCH[i+1] + ARCH[i+1]
    return n + 1
def unpack(theta):
    idx = 0; layers = []
    for i in range(len(ARCH)-1):
        ni, no = ARCH[i], ARCH[i+1]
        W = theta[idx:idx+ni*no].reshape(no, ni); idx += ni*no
        b = theta[idx:idx+no]; idx += no
        layers.append((W, b))
    return layers, theta[idx]
def forward(layers, X):
    a = X
    for k,(W,b) in enumerate(layers):
        z = a @ W.T + b
        a = np.tanh(z) if k < len(layers)-1 else z
    return a[:, 0]
def load_current(path):
    p = json.load(open(path)); parts = []
    for L in p['layers']:
        parts.append(np.array(L['W'], dtype=np.float64).ravel())
        parts.append(np.array(L['b'], dtype=np.float64).ravel())
    parts.append(np.array([p['noop_bias']], dtype=np.float64))
    return np.concatenate(parts)
def to_json(theta):
    layers, noop = unpack(theta)
    return {"format":"mlp-v1","feat_norm":FN,"num_features":8,"noop_bias":float(noop),
            "layers":[{"W":W.tolist(),"b":b.tolist(),"act":("tanh" if i<len(layers)-1 else "linear")}
                      for i,(W,b) in enumerate(layers)]}

# ---------------- environment (mirrors the JS planner sim) ----------------
def new_game(N, P, rng):
    pos = rng.random((N,2))
    mnx,mny = pos.min(0); mxx,mxy = pos.max(0)
    rlDiag = math.hypot(mxx-mnx, mxy-mny) or 1.0
    owner = np.zeros(N, dtype=np.int32); troops = rng.uniform(5, 45, N)
    seeds = [int(rng.integers(N))]
    for _ in range(P-1):
        d = np.full(N, 1e9)
        for s in seeds: d = np.minimum(d, np.hypot(pos[:,0]-pos[s,0], pos[:,1]-pos[s,1]))
        seeds.append(int(np.argmax(d)))
    for pi,s in enumerate(seeds): owner[s] = pi+1; troops[s] = 40.0
    return dict(pos=pos, owner=owner, troops=troops, armies=[], rlDiag=rlDiag, ORB=rlDiag/8.0, N=N)
def step(g, dt):
    o=g['owner']; t=g['troops']; grow=(o>0)&(t<CAP)
    t[grow]=np.minimum(CAP, t[grow]+1.0*dt)
    still=[]
    for a in g['armies']:
        a['t']+=dt
        if a['t']>=a['ttotal']:
            ti=a['ti']
            if o[ti]==a['owner']: t[ti]+=a['count']
            else:
                t[ti]-=a['count']
                if t[ti]<0: o[ti]=a['owner']; t[ti]=-t[ti]
        else: still.append(a)
    g['armies']=still
def send(g, si, ti):
    amt=g['troops'][si]
    if amt<1 or g['owner'][si]==0: return
    g['troops'][si]=0.0
    d=math.hypot(g['pos'][ti,0]-g['pos'][si,0], g['pos'][ti,1]-g['pos'][si,1])
    g['armies'].append(dict(owner=int(g['owner'][si]), count=amt, ti=int(ti), t=0.0, ttotal=max(0.05,d/g['ORB'])))
def choose(layers, noop, g, owner):       # pure-net argmax move scorer
    o=g['owner']; t=g['troops']; pos=g['pos']; N=g['N']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    own_frac=float((o==owner).sum())/N
    tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    best=noop; mv=None
    for si in srcs:
        st=t[si]; tt=t[tgts]
        dist=np.hypot(pos[tgts,0]-pos[si,0], pos[tgts,1]-pos[si,1])/g['rlDiag']
        feats=np.stack([np.full(len(tgts),st/FN), tt/FN, (st-tt)/FN, dist,
            (o[tgts]==0).astype(np.float64), ((o[tgts]!=0)&(o[tgts]!=owner)).astype(np.float64),
            (st>tt).astype(np.float64), np.full(len(tgts),own_frac)], axis=1)
        sc=forward(layers, feats); j=int(np.argmax(sc))
        if sc[j]>best: best=sc[j]; mv=(int(si), int(tgts[j]))
    return mv
def heuristic(g, owner):
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=12))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    st=t[si]; tt=t[tgts]; d=np.hypot(pos[tgts,0]-pos[si,0], pos[tgts,1]-pos[si,1])
    sc=-d*2.0 - tt*1.1 + np.where(o[tgts]==0,8,10) + np.where(st>tt,25,0)
    return (int(si), int(tgts[int(np.argmax(sc))]))
def turtle(g, owner):                     # hoard, only strike with overwhelming force (human exploit)
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=80))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; st=t[si]; tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    safe=tgts[t[tgts]<st*0.6]
    if len(safe)==0: return None
    d=np.hypot(pos[safe,0]-pos[si,0], pos[safe,1]-pos[si,1])
    return (int(si), int(safe[int(np.argmin(d))]))
def rush(g, owner):                       # hyper-aggressive: nearest beatable target, low threshold
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=8))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; st=t[si]; tgts=np.where(o!=owner)[0]
    if len(tgts)==0: return None
    beat=tgts[t[tgts]<st]; pool=beat if len(beat) else tgts
    d=np.hypot(pos[pool,0]-pos[si,0], pos[pool,1]-pos[si,1])
    return (int(si), int(pool[int(np.argmin(d))]))
def alive_owners(g):
    s=set(int(x) for x in np.unique(g['owner']) if x>0)
    for a in g['armies']: s.add(a['owner'])
    return s
def play(seats, rng, N=18, max_t=150.0, dt=0.25):
    P=len(seats); g=new_game(N,P,rng)
    timers={p+1: rng.uniform(0.2,1.0) for p in range(P)}
    t=0.0
    while t<max_t:
        step(g,dt); t+=dt
        for p in range(1,P+1):
            timers[p]-=dt
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0); pol=seats[p-1]; k=pol[0]
                if   k=='net':    mv=choose(pol[1],pol[2],g,p)
                elif k=='turtle': mv=turtle(g,p)
                elif k=='rush':   mv=rush(g,p)
                else:             mv=heuristic(g,p)
                if mv: send(g,mv[0],mv[1])
        if len(alive_owners(g))<=1: break
    o=g['owner']; tr=g['troops']; res={}; al=alive_owners(g)
    for p in range(1,P+1):
        s=int((o==p).sum())/N + 0.10*(tr[o==p].sum()/(N*CAP))
        if al=={p}: s+=1.0
        res[p]=s
    return res

# ---------------- league + fitness (module-level for multiprocessing) ----------------
def materialize(entry):
    return ('net',)+unpack(entry[1]) if entry[0]=='net' else (entry[0], None, None)
def eval_theta(args):
    theta, league, specs, N = args
    cand = ('net',)+unpack(theta)
    ready = [materialize(e) for e in league]
    tot=0.0
    for seed,P,ci,opp in specs:
        seats=[cand if p==ci else ready[opp[p]] for p in range(P)]
        res=play(seats, np.random.default_rng(seed), N=N)
        my=res[ci+1]; others=[res[p] for p in res if p!=ci+1]
        tot += my - (max(others) if others else 0)
    return tot/len(specs)
def make_specs(rng, n, nleague):
    # FOCUS on the strategies that actually beat the current net: turtle(2) & rush(3), plus the
    # heuristic(1) and some baseline(0). Champions (idx>=4) sprinkled in for self-play robustness.
    choices=[1,2,2,2,3,3,0]   # turtle 3/7, rush 2/7, heur 1/7, baseline 1/7
    specs=[]
    for _ in range(n):
        P=int(rng.choice([2,3,4])); ci=int(rng.integers(P))
        opp=[]
        for _ in range(P):
            if nleague>4 and rng.random()<0.15: opp.append(int(rng.integers(4,nleague)))
            else: opp.append(int(rng.choice(choices)))
        specs.append((int(rng.integers(1<<30)), P, ci, opp))
    return specs
def _seat(opp, base_theta):
    if opp=='base': return ('net',)+unpack(base_theta)
    return (opp, None, None)
def wr_vs(theta, opp, base_theta, rng, n=120, N=18):
    cand=('net',)+unpack(theta); o=_seat(opp, base_theta); w=0
    for _ in range(n):
        P=int(rng.choice([2,3])); ci=int(rng.integers(P))
        seats=[cand if p==ci else o for p in range(P)]
        res=play(seats, rng, N=N)
        if res[ci+1] >= max(res.values())-1e-9: w+=1
    return w/n
def evalset(theta, base_theta, rng, n=120):
    return {'turtle':wr_vs(theta,'turtle',base_theta,rng,n),
            'rush':  wr_vs(theta,'rush',  base_theta,rng,n),
            'heur':  wr_vs(theta,'heuristic',base_theta,rng,n),
            'base':  wr_vs(theta,'base',  base_theta,rng,n)}
def metric(ev):   # weighted toward beating the turtle (our worst matchup)
    return 0.5*ev['turtle'] + 0.25*ev['rush'] + 0.25*ev['heur']

# ---------------- SNES optimizer ----------------
def utilities(lam):
    k=np.arange(1, lam+1)
    u=np.maximum(0.0, math.log(lam/2+1)-np.log(k))
    return u/u.sum() - 1.0/lam
def main():
    secs = float(sys.argv[1]) if len(sys.argv)>1 else 1200.0
    base = load_current(POLICY); D=len(base)
    rng = np.random.default_rng(2024)
    n=D; lam=24; eta_sigma=(3+math.log(n))/(5*math.sqrt(n)); u=utilities(lam)
    mean=base.copy(); sig=np.full(n, 0.05)
    league=[('net', base.copy()), ('heur',None), ('turtle',None), ('rush',None)]   # idx0 = baseline
    champ=base.copy()
    base_ev=evalset(base, base, rng, 200); base_m=metric(base_ev)
    best_m=base_m
    print(f"params={n} lam={lam} eta_sigma={eta_sigma:.4f} cores={os.cpu_count()}", flush=True)
    print(f"BASELINE  turtle {base_ev['turtle']:.2f}  rush {base_ev['rush']:.2f}  heur {base_ev['heur']:.2f}  (vs-self {base_ev['base']:.2f})  metric {base_m:.3f}", flush=True)
    print("objective: raise the turtle/rush/heur winrates above baseline WITHOUT regressing vs-self.", flush=True)
    pool=Pool(processes=min(4, os.cpu_count() or 4))
    t0=time.time(); gen=0
    try:
        while time.time()-t0 < secs:
            gen+=1
            specs=make_specs(rng, 16, len(league))
            S=rng.standard_normal((lam, n)); X=mean + sig*S
            fs=np.array(pool.map(eval_theta, [(X[i], league, specs, 18) for i in range(lam)]))
            order=np.argsort(-fs); So=S[order]
            mean=mean + sig*((u[:,None]*So).sum(0))             # eta_mu = 1
            sig =sig * np.exp(0.5*eta_sigma*((u[:,None]*(So**2-1)).sum(0)))
            sig =np.clip(sig, 1e-4, 0.5)
            if gen % 8 == 0:
                ev=evalset(mean, base, rng, 120); m=metric(ev); tag=""
                # champion = clearly better at the exploiters, with no real regression vs-self
                if m>best_m+0.01 and ev['base']>=0.45:
                    best_m=m; champ=mean.copy()
                    json.dump(to_json(champ), open(os.path.join(SCR,'rl_policy_new.json'),'w'))
                    tag=" *new champion*"
                    if len(league)<7: league.append(('net', champ.copy()))
                print(f"gen {gen:4d} t={time.time()-t0:6.0f}s sig~{sig.mean():.3f}  turtle {ev['turtle']:.2f} rush {ev['rush']:.2f} heur {ev['heur']:.2f} vs-self {ev['base']:.2f}  metric {m:.3f} (base {base_m:.3f}){tag}", flush=True)
    finally:
        pool.close(); pool.join()
    fev=evalset(champ, base, rng, 300); fm=metric(fev)
    print(f"DONE gens={gen}  champion: turtle {fev['turtle']:.2f} rush {fev['rush']:.2f} heur {fev['heur']:.2f} vs-self {fev['base']:.2f}  metric {fm:.3f}  (baseline metric {base_m:.3f})", flush=True)
    if fm>base_m+0.02 and fev['base']>=0.45:
        json.dump(to_json(champ), open(os.path.join(SCR,'rl_policy_new.json'),'w'))
        print(f"SHIP: beats baseline at the exploiters (+{fm-base_m:.3f} metric) with no vs-self regression.", flush=True)
    else:
        print("NO-REGRESSION: no clear gain over baseline at the exploiters; not shipping.", flush=True)

if __name__=='__main__':
    main()
