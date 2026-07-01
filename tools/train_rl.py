#!/usr/bin/env python3
# Self-play trainer for the State.io move-scoring policy — THREAT-AWARE (12-feature) edition.
#
# The original net saw only 8 numbers about a single move, so it couldn't perceive a "turtle"
# massing a big stack and got crushed by it (~14% win rate). This version adds 4 board-awareness
# features so the net can SEE the threat, and trains specifically to beat the turtle/rush exploiters.
#
#   * 12 features = original 8 + [maxEnemyStack, myArmyShare, myAvgStack(=over-extension), tgtIsBiggestThreat]
#   * WARM-START from the shipped 8-feature net with the 4 new input-weights zeroed, so it begins
#     identical to today's net and only improves from there.
#   * Opponent league = scripted exploiters (heuristic / turtle / rush) + past champions (self-play).
#   * Separable Natural ES (SNES), 4-core parallel, common random numbers.
#   * NO-REGRESSION: writes rl_policy_new.json only if turtle-resistance clearly beats the warm-start
#     baseline WITHOUT regressing vs rush/heuristic or collapsing (still crushes a random net).
#   usage: python3 tools/train_rl.py [seconds]
import numpy as np, json, time, sys, math, os
from multiprocessing import Pool

FN = 50.0; CAP = 150.0
ARCH = [12, 32, 32, 1]
SCR = os.path.dirname(os.path.abspath(__file__))
POLICY = next((p for p in [os.path.join(SCR,'..','src','web','rl_policy.json'),
                           os.path.join(SCR,'rl_policy.json')] if os.path.exists(p)), None)

# ---------------- net ----------------
def nparams():
    n=0
    for i in range(len(ARCH)-1): n+=ARCH[i]*ARCH[i+1]+ARCH[i+1]
    return n+1
def unpack(theta):
    idx=0; layers=[]
    for i in range(len(ARCH)-1):
        ni,no=ARCH[i],ARCH[i+1]
        W=theta[idx:idx+ni*no].reshape(no,ni); idx+=ni*no
        b=theta[idx:idx+no]; idx+=no
        layers.append((W,b))
    return layers, theta[idx]
def forward(layers, X):
    a=X
    for k,(W,b) in enumerate(layers):
        z=a@W.T+b
        a=np.tanh(z) if k<len(layers)-1 else z
    return a[:,0]
def to_json(theta):
    layers,noop=unpack(theta)
    return {"format":"mlp-v1","feat_norm":FN,"num_features":ARCH[0],"noop_bias":float(noop),
            "layers":[{"W":W.tolist(),"b":b.tolist(),"act":("tanh" if i<len(layers)-1 else "linear")}
                      for i,(W,b) in enumerate(layers)]}
def init_from_baseline(path):
    # Warm-start the search from the shipped net. Two cases:
    #   * shipped net is already 12-feature  -> load its weights verbatim (identity start).
    #   * shipped net is the old 8-feature   -> copy the 8 input weights, zero the 4 new ones.
    # Either way training begins behaving EXACTLY like today's net and only improves from there.
    p=json.load(open(path)); L=p['layers']
    W0=np.array(L[0]['W'], dtype=np.float64)                       # (32, nfeat)
    nfeat=W0.shape[1]
    if nfeat==ARCH[0]:
        W0e=W0
    elif nfeat==8:
        W0e=np.concatenate([W0, np.zeros((W0.shape[0], ARCH[0]-8))], axis=1)   # (32,12)
    else:
        raise AssertionError(f"baseline has {nfeat} features; expected 8 or {ARCH[0]}")
    parts=[W0e.ravel(), np.array(L[0]['b'],dtype=np.float64).ravel()]
    for k in range(1,len(L)):
        parts.append(np.array(L[k]['W'],dtype=np.float64).ravel())
        parts.append(np.array(L[k]['b'],dtype=np.float64).ravel())
    parts.append(np.array([p['noop_bias']],dtype=np.float64))
    return np.concatenate(parts)

# ---------------- environment ----------------
def delaunay_adjacency(pos):
    # Brute-force Delaunay triangulation (N is tiny here — 18 points, ~800 candidate triangles — so an
    # O(N^4) circumcircle sweep is trivial and avoids a scipy dependency). This is the synthetic
    # analogue of "shares a border" for a random point set, mirroring the real game's TopoJSON-derived
    # adjacency (index.html) and the arena's identical JS implementation (selfplay_arena.js).
    N = len(pos); adj = [set() for _ in range(N)]
    def orient(i,j,k):
        return (pos[j,0]-pos[i,0])*(pos[k,1]-pos[i,1]) - (pos[j,1]-pos[i,1])*(pos[k,0]-pos[i,0])
    def in_circum(i,j,k,d):   # assumes i,j,k CCW; True iff d lies inside their circumcircle
        ax,ay = pos[i,0]-pos[d,0], pos[i,1]-pos[d,1]
        bx,by = pos[j,0]-pos[d,0], pos[j,1]-pos[d,1]
        gx,gy = pos[k,0]-pos[d,0], pos[k,1]-pos[d,1]
        a2,b2,g2 = ax*ax+ay*ay, bx*bx+by*by, gx*gx+gy*gy
        return ax*(by*g2-b2*gy) - ay*(bx*g2-b2*gx) + a2*(bx*gy-by*gx) > 1e-12
    for i in range(N):
        for j in range(i+1,N):
            for k in range(j+1,N):
                a,b,c = i,j,k
                if orient(a,b,c) < 0: b,c = c,b
                if abs(orient(a,b,c)) < 1e-12: continue   # collinear triple, skip
                if any(in_circum(a,b,c,d) for d in range(N) if d not in (a,b,c)): continue
                adj[a]|={b,c}; adj[b]|={a,c}; adj[c]|={a,b}
    for i in range(N):   # defensive: guarantee no isolated node for degenerate point sets
        if not adj[i]:
            j = min((j for j in range(N) if j!=i), key=lambda j: math.hypot(pos[i,0]-pos[j,0],pos[i,1]-pos[j,1]))
            adj[i].add(j); adj[j].add(i)
    return [sorted(s) for s in adj]

def legal_targets(g, si, owner):
    # Risk-style legality: direct neighbors for attacks, BFS-reachable-through-owned for reinforcement
    # — mirrors isLegalMove()/snapLegal() in index.html exactly. Returns a sorted int array.
    adj = g['adj']; o = g['owner']
    direct = [j for j in adj[si] if o[j] != owner]
    reach = set(); frontier = [si]; seen = {si}
    while frontier:
        cur = frontier.pop()
        for nb in adj[cur]:
            if nb not in seen and o[nb] == owner:
                seen.add(nb); frontier.append(nb); reach.add(nb)
    return np.array(sorted(set(direct) | reach), dtype=np.int64)

def new_game(N, P, rng):
    pos=rng.random((N,2)); mnx,mny=pos.min(0); mxx,mxy=pos.max(0)
    rlDiag=math.hypot(mxx-mnx,mxy-mny) or 1.0
    owner=np.zeros(N,dtype=np.int32); troops=rng.uniform(5,45,N)
    seeds=[int(rng.integers(N))]
    for _ in range(P-1):
        d=np.full(N,1e9)
        for s in seeds: d=np.minimum(d,np.hypot(pos[:,0]-pos[s,0],pos[:,1]-pos[s,1]))
        seeds.append(int(np.argmax(d)))
    for pi,s in enumerate(seeds): owner[s]=pi+1; troops[s]=40.0
    adj = delaunay_adjacency(pos)
    return dict(pos=pos,owner=owner,troops=troops,armies=[],rlDiag=rlDiag,ORB=rlDiag/8.0,N=N,adj=adj)
def step(g,dt):
    o=g['owner']; t=g['troops']; grow=(o>0)&(t<CAP)
    t[grow]=np.minimum(CAP,t[grow]+1.0*dt); still=[]
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
def send(g,si,ti):
    amt=g['troops'][si]
    if amt<1 or g['owner'][si]==0: return
    g['troops'][si]=0.0
    d=math.hypot(g['pos'][ti,0]-g['pos'][si,0],g['pos'][ti,1]-g['pos'][si,1])
    g['armies'].append(dict(owner=int(g['owner'][si]),count=amt,ti=int(ti),t=0.0,ttotal=max(0.05,d/g['ORB'])))
def feats12(g, si, owner, tgts, glob):
    o=g['owner']; t=g['troops']; pos=g['pos']
    st=t[si]; tt=t[tgts]; m=len(tgts)
    dist=np.hypot(pos[tgts,0]-pos[si,0],pos[tgts,1]-pos[si,1])/g['rlDiag']
    maxEnemy,myShare,myAvg,bigIdx,own_frac=glob
    return np.stack([
        np.full(m,st/FN), tt/FN, (st-tt)/FN, dist,
        (o[tgts]==0).astype(np.float64),
        ((o[tgts]!=0)&(o[tgts]!=owner)).astype(np.float64),
        (st>tt).astype(np.float64),
        np.full(m,own_frac),
        np.full(m,maxEnemy/FN),         # biggest enemy stack on the board (a massing turtle!)
        np.full(m,myShare),             # my share of all troops (behind => don't over-extend)
        np.full(m,myAvg/FN),            # my avg troops/state (low => spread thin = turtle bait)
        (tgts==bigIdx).astype(np.float64),  # is THIS target the biggest threat? (pre-empt it)
    ],axis=1)
def globals_for(g, owner):
    o=g['owner']; t=g['troops']; N=g['N']
    enemy=np.where((o!=owner)&(o!=0))[0]
    maxEnemy=float(t[enemy].max()) if enemy.size else 0.0
    bigIdx=int(enemy[int(np.argmax(t[enemy]))]) if enemy.size else -1
    myTot=float(t[o==owner].sum()); myStates=int((o==owner).sum())
    myAvg=myTot/max(1,myStates); allTot=float(t[o>0].sum())
    return (maxEnemy, myTot/max(1e-9,allTot), myAvg, bigIdx, myStates/N)
def incoming_for(g, owner):
    # in-flight troops arriving at each state: enemy (not mine) vs mine. Mirrors the JS incomingTo().
    N=g['N']; incE=np.zeros(N); incM=np.zeros(N)
    for a in g['armies']:
        if a['owner']==owner: incM[a['ti']]+=a['count']
        else: incE[a['ti']]+=a['count']
    return incE, incM
def feats16(g, si, owner, tgts, glob, inc):
    # the 12 threat-aware features + 4 in-flight/timing features the net was blind to:
    #   [incEnemy@tgt, incEnemy@src(=source vulnerability), incMine@tgt, canWinAfterInflight]
    base=feats12(g, si, owner, tgts, glob)
    incE,incM=inc; m=len(tgts); st=g['troops'][si]
    incEtg=incE[tgts]; incMtg=incM[tgts]; incEsrc=incE[si]
    extra=np.stack([
        incEtg/FN,                                   # enemy reinforcements inbound to the target
        np.full(m, incEsrc/FN),                      # enemy attack inbound to my source (don't empty it)
        incMtg/FN,                                   # my troops already en route to the target
        (st > g['troops'][tgts]+incEtg-incMtg).astype(np.float64),  # can I still win after in-flight resolves?
    ],axis=1)
    return np.concatenate([base, extra], axis=1)
def direct_targets(g, si, owner):
    # attack-only legal targets (direct border neighbors that aren't mine) — used by the scripted
    # opponents below, which were never designed to reinforce and don't need to learn to now.
    o=g['owner']
    return np.array(sorted(j for j in g['adj'][si] if o[j]!=owner), dtype=np.int64)
def choose(layers, noop, g, owner):
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    glob=globals_for(g, owner)
    best=noop; mv=None
    for si in srcs:
        tgts=legal_targets(g, si, owner)
        if len(tgts)==0: continue
        sc=forward(layers, feats12(g,si,owner,tgts,glob)); j=int(np.argmax(sc))
        if sc[j]>best: best=sc[j]; mv=(int(si),int(tgts[j]))
    return mv
def choose16(layers, noop, g, owner):   # 16-feature variant; monkeypatch onto `choose` to eval a 16-input net
    o=g['owner']; t=g['troops']
    srcs=np.where((o==owner)&(t>=5))[0]
    if len(srcs)==0: return None
    glob=globals_for(g, owner); inc=incoming_for(g, owner)
    best=noop; mv=None
    for si in srcs:
        tgts=legal_targets(g, si, owner)
        if len(tgts)==0: continue
        sc=forward(layers, feats16(g,si,owner,tgts,glob,inc)); j=int(np.argmax(sc))
        if sc[j]>best: best=sc[j]; mv=(int(si),int(tgts[j]))
    return mv
def heuristic(g,owner):
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=12))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; tgts=direct_targets(g,si,owner)
    if len(tgts)==0: return None
    st=t[si]; tt=t[tgts]; d=np.hypot(pos[tgts,0]-pos[si,0],pos[tgts,1]-pos[si,1])
    sc=-d*2.0-tt*1.1+np.where(o[tgts]==0,8,10)+np.where(st>tt,25,0)
    return (int(si),int(tgts[int(np.argmax(sc))]))
def turtle(g,owner):
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=80))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; st=t[si]; tgts=direct_targets(g,si,owner)
    if len(tgts)==0: return None
    safe=tgts[t[tgts]<st*0.6]
    if len(safe)==0: return None
    d=np.hypot(pos[safe,0]-pos[si,0],pos[safe,1]-pos[si,1])
    return (int(si),int(safe[int(np.argmin(d))]))
def rush(g,owner):
    o=g['owner']; t=g['troops']; pos=g['pos']
    srcs=np.where((o==owner)&(t>=8))[0]
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; st=t[si]; tgts=direct_targets(g,si,owner)
    if len(tgts)==0: return None
    beat=tgts[t[tgts]<st]; pool=beat if len(beat) else tgts
    d=np.hypot(pos[pool,0]-pos[si,0],pos[pool,1]-pos[si,1])
    return (int(si),int(pool[int(np.argmin(d))]))
def economist(g,owner):
    # The "shop turtle" the human used: grab a few cheap states early, then HOARD a huge stack and
    # only strike when overwhelmingly strong. Combined with the airdrop top-ups injected in play(),
    # this is the exact hoard-then-crush pattern the net must learn to pre-empt instead of ignore.
    o=g['owner']; t=g['troops']; pos=g['pos']
    myStates=int((o==owner).sum())
    if myStates<=2:                                  # opening: snap up the nearest weak neutral, then sit
        srcs=np.where((o==owner)&(t>=10))[0]
        if len(srcs)==0: return None
        si=srcs[np.argmax(t[srcs])]; st=t[si]
        neu=direct_targets(g,si,owner); neu=neu[o[neu]==0]
        neu=neu[t[neu]<st*0.7] if len(neu) else neu
        if len(neu)==0: return None
        d=np.hypot(pos[neu,0]-pos[si,0],pos[neu,1]-pos[si,1])
        return (int(si),int(neu[int(np.argmin(d))]))
    srcs=np.where((o==owner)&(t>=70))[0]             # hoard to 70+ before committing
    if len(srcs)==0: return None
    si=srcs[np.argmax(t[srcs])]; st=t[si]; tgts=direct_targets(g,si,owner)
    if len(tgts)==0: return None
    safe=tgts[t[tgts]<st*0.5]                         # only attack when ~2x stronger
    if len(safe)==0: return None
    d=np.hypot(pos[safe,0]-pos[si,0],pos[safe,1]-pos[si,1])
    return (int(si),int(safe[int(np.argmin(d))]))
def alive_owners(g):
    s=set(int(x) for x in np.unique(g['owner']) if x>0)
    for a in g['armies']: s.add(a['owner'])
    return s
def play(seats, rng, N=18, max_t=150.0, dt=0.25):
    P=len(seats); g=new_game(N,P,rng)
    timers={p+1: rng.uniform(0.2,1.0) for p in range(P)}; t=0.0
    air={p+1:7.0 for p in range(P)}            # airdrop cooldown per seat (economist tops up its hoard)
    while t<max_t:
        step(g,dt); t+=dt
        for p in range(1,P+1):
            pol=seats[p-1]; k=pol[0]
            if k=='econ':                       # +25 airdrop onto its biggest state every ~7s (shop perk)
                air[p]-=dt
                if air[p]<=0:
                    air[p]=7.0; own=np.where(g['owner']==p)[0]
                    if len(own): g['troops'][own[int(np.argmax(g['troops'][own]))]]+=25.0
            timers[p]-=dt
            if timers[p]<=0:
                timers[p]=rng.uniform(0.6,1.0)
                if   k=='net':    mv=choose(pol[1],pol[2],g,p)
                elif k=='turtle': mv=turtle(g,p)
                elif k=='rush':   mv=rush(g,p)
                elif k=='econ':   mv=economist(g,p)
                else:             mv=heuristic(g,p)
                if mv: send(g,mv[0],mv[1])
        if len(alive_owners(g))<=1: break
    o=g['owner']; tr=g['troops']; res={}; al=alive_owners(g)
    for p in range(1,P+1):
        s=int((o==p).sum())/N + 0.10*(tr[o==p].sum()/(N*CAP))
        if al=={p}: s+=1.0
        res[p]=s
    return res

# ---------------- league + fitness ----------------
def materialize(e): return ('net',)+unpack(e[1]) if e[0]=='net' else (e[0],None,None)
def eval_theta(args):
    theta, league, specs, N = args
    cand=('net',)+unpack(theta); ready=[materialize(e) for e in league]; tot=0.0
    for seed,P,ci,opp in specs:
        seats=[cand if p==ci else ready[opp[p]] for p in range(P)]
        res=play(seats, np.random.default_rng(seed), N=N)
        my=res[ci+1]; others=[res[p] for p in res if p!=ci+1]
        tot+=my-(max(others) if others else 0)
    return tot/len(specs)
def make_specs(rng, n, nleague):
    choices=[0,1,1,2,2,3,3,3]   # league idx: 0=heur, 1=turtle(x2), 2=rush(x2), 3=economist(x3, the friend)
    specs=[]
    for _ in range(n):
        P=int(rng.choice([2,3,4])); ci=int(rng.integers(P))
        opp=[]
        for _ in range(P):
            if nleague>4 and rng.random()<0.15: opp.append(int(rng.integers(4,nleague)))
            else: opp.append(int(rng.choice(choices)))
        specs.append((int(rng.integers(1<<30)),P,ci,opp))
    return specs
def wr_vs(theta, opp, rng, n=120, N=18):
    cand=('net',)+unpack(theta)
    o=('net',)+unpack(opp) if isinstance(opp,np.ndarray) else (opp,None,None)
    w=0
    for _ in range(n):
        P=int(rng.choice([2,3])); ci=int(rng.integers(P))
        seats=[cand if p==ci else o for p in range(P)]
        res=play(seats,rng,N=N)
        if res[ci+1]>=max(res.values())-1e-9: w+=1
    return w/n
def evalset(theta, randnet, rng, n=120):
    return {'econ':wr_vs(theta,'econ',rng,n), 'turtle':wr_vs(theta,'turtle',rng,n), 'rush':wr_vs(theta,'rush',rng,n),
            'heur':wr_vs(theta,'heuristic',rng,n), 'rand':wr_vs(theta,randnet,rng,n)}
# The net already maxes econ/turtle; its real weakness is general play (beating the greedy heuristic
# and out-playing arbitrary nets). Drive training at THOSE while holding econ/turtle/rush as floors.
def metric(ev): return 0.45*ev['heur']+0.25*ev['rand']+0.12*ev['econ']+0.10*ev['turtle']+0.08*ev['rush']

# ---------------- SNES ----------------
def utilities(lam):
    k=np.arange(1,lam+1); u=np.maximum(0.0,math.log(lam/2+1)-np.log(k))
    return u/u.sum()-1.0/lam
def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 1500.0
    init=init_from_baseline(POLICY); n=len(init)
    rng=np.random.default_rng(7); randnet=rng.standard_normal(n)*0.5
    lam=24; eta_sigma=(3+math.log(n))/(5*math.sqrt(n)); u=utilities(lam)
    mean=init.copy(); sig=np.full(n,0.05)
    league=[('heur',None),('turtle',None),('rush',None),('econ',None)]
    champ=init.copy()
    base_ev=evalset(init, randnet, rng, 200); base_m=metric(base_ev)
    best_m=base_m
    print(f"params={n} (12-feature) lam={lam} cores={os.cpu_count()}", flush=True)
    print(f"WARM-START baseline: econ {base_ev['econ']:.2f} turtle {base_ev['turtle']:.2f} rush {base_ev['rush']:.2f} heur {base_ev['heur']:.2f} rand {base_ev['rand']:.2f}  metric {base_m:.3f}", flush=True)
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time(); gen=0
    try:
        while time.time()-t0<secs:
            gen+=1
            specs=make_specs(rng,16,len(league))
            S=rng.standard_normal((lam,n)); X=mean+sig*S
            fs=np.array(pool.map(eval_theta,[(X[i],league,specs,18) for i in range(lam)]))
            order=np.argsort(-fs); So=S[order]
            mean=mean+sig*((u[:,None]*So).sum(0))
            sig=np.clip(sig*np.exp(0.5*eta_sigma*((u[:,None]*(So**2-1)).sum(0))),1e-4,0.5)
            if gen%8==0:
                ev=evalset(mean,randnet,rng,120); m=metric(ev); tag=""
                ok = ev['rand']>=base_ev['rand']-0.05 and ev['rush']>=base_ev['rush']-0.06 and ev['heur']>=base_ev['heur']-0.06 and ev['turtle']>=base_ev['turtle']-0.06
                if m>best_m+0.01 and ok:
                    best_m=m; champ=mean.copy(); tag=" *champion*"
                    json.dump(to_json(champ),open(os.path.join(SCR,'rl_policy_new.json'),'w'))
                    if len(league)<8: league.append(('net',champ.copy()))
                print(f"gen {gen:4d} t={time.time()-t0:6.0f}s sig~{sig.mean():.3f} econ {ev['econ']:.2f} turtle {ev['turtle']:.2f} rush {ev['rush']:.2f} heur {ev['heur']:.2f} rand {ev['rand']:.2f} metric {m:.3f}(base {base_m:.3f}){tag}", flush=True)
    finally:
        pool.close(); pool.join()
    fev=evalset(champ,randnet,rng,300); fm=metric(fev)
    # Ship only on a clear gain against the hoarder (econ) OR overall metric, with NO regression anywhere.
    ship = (fm>=base_m+0.02 or fev['heur']>=base_ev['heur']+0.05) and \
           fev['rand']>=base_ev['rand']-0.04 and fev['rush']>=base_ev['rush']-0.05 and \
           fev['econ']>=base_ev['econ']-0.04 and fev['turtle']>=base_ev['turtle']-0.04
    print(f"DONE gens={gen}  champion: econ {fev['econ']:.2f}(base {base_ev['econ']:.2f}) turtle {fev['turtle']:.2f}(base {base_ev['turtle']:.2f}) rush {fev['rush']:.2f} heur {fev['heur']:.2f} rand {fev['rand']:.2f} metric {fm:.3f}(base {base_m:.3f})", flush=True)
    if ship:
        json.dump(to_json(champ),open(os.path.join(SCR,'rl_policy_new.json'),'w'))
        print(f"SHIP: econ {fev['econ']-base_ev['econ']:+.2f} turtle {fev['turtle']-base_ev['turtle']:+.2f}, no regression. 12-feature model saved.", flush=True)
    else:
        print("NO-REGRESSION: no clear gain vs the hoarder; not shipping. (current net stays.)", flush=True)

if __name__=='__main__':
    main()
