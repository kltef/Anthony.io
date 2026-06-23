#!/usr/bin/env python3
# Self-play Evolution-Strategies trainer for the State.io move-scoring policy.
# The environment mirrors the JS planner's own sim (snapStep growth, snapSend orbs,
# attack resolution) so a stronger net here = a stronger planner + live AI in-game.
# Net: 8 -> 32 -> 32 -> 1 (tanh hidden, linear out), same format as rl_policy.json.
import numpy as np, json, time, sys, math, os

FN = 50.0
CAP = 150.0
ARCH = [8, 32, 32, 1]
SCR = os.path.dirname(os.path.abspath(__file__))

# ---------- net ----------
def nparams():
    n = 0
    for i in range(len(ARCH)-1): n += ARCH[i]*ARCH[i+1] + ARCH[i+1]
    return n + 1   # + noop_bias

def unpack(theta):
    idx = 0; layers = []
    for i in range(len(ARCH)-1):
        ni, no = ARCH[i], ARCH[i+1]
        W = theta[idx:idx+ni*no].reshape(no, ni); idx += ni*no
        b = theta[idx:idx+no]; idx += no
        layers.append((W, b))
    return layers, theta[idx]

def forward(layers, X):                 # X:[M,8] -> [M]
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
    return np.concatenate(parts), p

def to_json(theta):
    layers, noop = unpack(theta)
    return {"format":"mlp-v1","feat_norm":FN,"num_features":8,"noop_bias":float(noop),
            "layers":[{"W":W.tolist(),"b":b.tolist(),"act":("tanh" if i<len(layers)-1 else "linear")}
                      for i,(W,b) in enumerate(layers)]}

# ---------- environment (mirrors the JS sim) ----------
def new_game(N, P, rng):
    pos = rng.random((N,2))
    mnx,mny = pos.min(0); mxx,mxy = pos.max(0)
    rlDiag = math.hypot(mxx-mnx, mxy-mny) or 1.0
    ORB = rlDiag/8.0
    owner = np.zeros(N, dtype=np.int32)
    troops = rng.uniform(5, 45, N)
    # seed P players far apart (farthest-point)
    seeds = [rng.integers(N)]
    for _ in range(P-1):
        d = np.full(N, 1e9)
        for s in seeds:
            d = np.minimum(d, np.hypot(pos[:,0]-pos[s,0], pos[:,1]-pos[s,1]))
        seeds.append(int(np.argmax(d)))
    for pi,s in enumerate(seeds):
        owner[s] = pi+1; troops[s] = 40.0
    return dict(pos=pos, owner=owner, troops=troops, armies=[], rlDiag=rlDiag, ORB=ORB, N=N)

def step(g, dt):
    o = g['owner']; t = g['troops']
    grow = (o > 0) & (t < CAP)
    t[grow] = np.minimum(CAP, t[grow] + 1.0*dt)     # growRate 1.0/s
    still = []
    for a in g['armies']:
        a['t'] += dt
        if a['t'] >= a['ttotal']:
            ti = a['ti']
            if o[ti] == a['owner']:
                t[ti] += a['count']
            else:
                t[ti] -= a['count']
                if t[ti] < 0:
                    o[ti] = a['owner']; t[ti] = -t[ti]
        else:
            still.append(a)
    g['armies'] = still

def send(g, si, ti):
    amt = g['troops'][si]
    if amt < 1 or g['owner'][si] == 0: return
    g['troops'][si] = 0.0
    d = math.hypot(g['pos'][ti,0]-g['pos'][si,0], g['pos'][ti,1]-g['pos'][si,1])
    g['armies'].append(dict(owner=int(g['owner'][si]), count=amt, ti=int(ti),
                            t=0.0, ttotal=max(0.05, d/g['ORB'])))

def choose(layers, noop, g, owner):
    o = g['owner']; t = g['troops']; pos = g['pos']; N = g['N']
    srcs = np.where((o == owner) & (t >= 5))[0]
    if len(srcs) == 0: return None
    own_frac = float((o == owner).sum()) / N
    best = noop; best_mv = None
    tgts = np.where(o != owner)[0]
    if len(tgts) == 0: return None
    for si in srcs:
        st = t[si]
        tt = t[tgts]
        dx = pos[tgts,0]-pos[si,0]; dy = pos[tgts,1]-pos[si,1]
        dist = np.hypot(dx, dy)/g['rlDiag']
        feats = np.stack([
            np.full(len(tgts), st/FN),
            tt/FN,
            (st-tt)/FN,
            dist,
            (o[tgts] == 0).astype(np.float64),
            ((o[tgts] != 0) & (o[tgts] != owner)).astype(np.float64),
            (st > tt).astype(np.float64),
            np.full(len(tgts), own_frac),
        ], axis=1)
        sc = forward(layers, feats)
        j = int(np.argmax(sc))
        if sc[j] > best:
            best = sc[j]; best_mv = (int(si), int(tgts[j]))
    return best_mv

def heuristic(g, owner):               # cheap net-free opponent (mirrors snapHeuristic spirit)
    o = g['owner']; t = g['troops']; pos = g['pos']
    srcs = np.where((o == owner) & (t >= 12))[0]
    if len(srcs) == 0: return None
    si = srcs[np.argmax(t[srcs])]
    tgts = np.where(o != owner)[0]
    if len(tgts) == 0: return None
    st = t[si]; tt = t[tgts]
    d = np.hypot(pos[tgts,0]-pos[si,0], pos[tgts,1]-pos[si,1])
    sc = -d*2.0 - tt*1.1 + np.where(o[tgts]==0, 8, 10) + np.where(st>tt, 25, 0)
    j = int(np.argmax(sc))
    return (int(si), int(tgts[j]))

def turtle(g, owner):                  # the human-style exploiter: hoard, only strike with overwhelming force
    o = g['owner']; t = g['troops']; pos = g['pos']
    srcs = np.where((o == owner) & (t >= 80))[0]   # waits for a big stack
    if len(srcs) == 0: return None
    si = srcs[np.argmax(t[srcs])]
    st = t[si]
    tgts = np.where(o != owner)[0]
    if len(tgts) == 0: return None
    tt = t[tgts]
    safe = tgts[tt < st*0.6]            # only near-certain captures
    if len(safe) == 0: return None
    d = np.hypot(pos[safe,0]-pos[si,0], pos[safe,1]-pos[si,1])
    return (int(si), int(safe[int(np.argmin(d))]))

def alive_owners(g):
    s = set(int(x) for x in np.unique(g['owner']) if x > 0)
    for a in g['armies']: s.add(a['owner'])
    return s

def play(seat_policies, rng, N=18, max_t=140.0, dt=0.25):  # rng drives map + timers (use a fresh seeded rng for CRN)
    """seat_policies[p] = ('net', layers, noop) or ('heur',). Returns standing dict pl->score."""
    P = len(seat_policies)
    g = new_game(N, P, rng)
    timers = {p+1: rng.uniform(0.2, 1.0) for p in range(P)}
    t = 0.0
    while t < max_t:
        step(g, dt); t += dt
        for p in range(1, P+1):
            timers[p] -= dt
            if timers[p] <= 0:
                timers[p] = rng.uniform(0.6, 1.0)
                pol = seat_policies[p-1]
                if pol[0]=='net':     mv = choose(pol[1], pol[2], g, p)
                elif pol[0]=='turtle': mv = turtle(g, p)
                else:                  mv = heuristic(g, p)
                if mv: send(g, mv[0], mv[1])
        al = alive_owners(g)
        if len(al) <= 1: break
    # standing in [~0..1.3]: territory fraction + small army share + modest sole-survivor bonus
    o = g['owner']; tr = g['troops']; res = {}; al = alive_owners(g)
    for p in range(1, P+1):
        cnt = int((o == p).sum()); mass = float(tr[o == p].sum())
        s = cnt/g['N'] + 0.10*(mass/(g['N']*CAP))
        if al == {p}: s += 1.0            # decisive win bonus (kept modest so reward isn't spiky)
        res[p] = s
    return res

# ---------- fitness with COMMON RANDOM NUMBERS ----------
# A "spec" fixes a whole game (map seed, #players, candidate seat, opponent types) so every
# candidate in a generation is judged on the IDENTICAL set of games — that removes map luck
# from the comparison and lets ES see the true effect of each weight perturbation.
def make_specs(rng, n):
    # opponents are MOSTLY the baseline net (what we measure against), with a little heuristic/turtle
    # spice so the policy stays robust and learns to punish hoarding.
    specs = []
    for _ in range(n):
        P = int(rng.choice([2,3,4]))
        ci = int(rng.integers(P))
        opp = []
        for _ in range(P):
            r = rng.random()
            opp.append('base' if r < 0.78 else ('turtle' if r < 0.89 else 'heur'))
        specs.append((int(rng.integers(1<<30)), P, ci, opp))
    return specs

def eval_specs(theta, base_theta, specs, N=18):
    lc = unpack(theta); lb = unpack(base_theta)
    cand=('net',)+lc; base=('net',)+lb; heur=('heur',None,None); turtle=('turtle',None,None)
    kinds={'base':base,'heur':heur,'turtle':turtle}
    tot=0.0
    for seed,P,ci,opp in specs:
        seats=[cand if p==ci else kinds[opp[p]] for p in range(P)]
        res=play(seats, np.random.default_rng(seed), N=N)
        my=res[ci+1]; others=[res[p] for p in res if p!=ci+1]
        tot += my - (max(others) if others else 0)
    return tot/len(specs)

def winrate(theta, base_theta, rng, n_games=80, N=18):
    lc = unpack(theta); lb = unpack(base_theta)
    cand=('net',)+lc; base=('net',)+lb
    w=0
    for _ in range(n_games):
        P=int(rng.choice([2,3,4])); ci=int(rng.integers(P))
        seats=[cand if p==ci else base for p in range(P)]
        res=play(seats, rng, N=N)
        if res[ci+1] >= max(res.values())-1e-9: w+=1
    return w/n_games

# ---------- ES loop ----------
def main():
    secs = float(sys.argv[1]) if len(sys.argv)>1 else 480.0
    base_theta, meta = load_current(os.path.join(SCR,'rl_policy.json'))
    print("params:", nparams(), "| baseline noop:", round(float(base_theta[-1]),3), flush=True)
    rng = np.random.default_rng(12345)
    theta = base_theta.copy()
    D = len(theta)
    sigma = 0.05; alpha = 0.20; pop = 18        # antithetic pairs
    games = 12                                  # CRN games per fitness eval
    # centered-rank fitness shaping in [-0.5, 0.5] (robust to scale/outliers), sums to 0
    def utilities(fvals):
        rank = np.argsort(np.argsort(-fvals))           # 0 = best
        n = len(fvals)
        return 0.5 - rank/(n-1)
    best = theta.copy(); best_wr = winrate(theta, base_theta, rng, 80)
    print(f"gen 0  baseline-vs-self winrate {best_wr:.3f} (sanity ~0.5)", flush=True)
    t0 = time.time(); gen = 0
    while time.time()-t0 < secs:
        gen += 1
        specs = make_specs(rng, games)                  # SAME games for every candidate this gen
        eps = rng.standard_normal((pop, D))
        perts = np.concatenate([eps, -eps], 0)          # antithetic 2*pop
        fs = np.array([eval_specs(theta + sigma*perts[j], base_theta, specs) for j in range(2*pop)])
        u = utilities(fs)
        grad = (perts * u[:,None]).sum(0) / (2*pop*sigma)   # OpenAI-ES estimator (1/(nσ) Σ uᵢ εᵢ)
        theta = theta + alpha * grad
        if gen % 4 == 0:
            wr = winrate(theta, base_theta, rng, 90)
            tag=""
            if wr > best_wr: best_wr = wr; best = theta.copy(); tag=" *checkpoint*"
            json.dump(to_json(best), open(os.path.join(SCR,'rl_policy_new.json'),'w'))
            print(f"gen {gen:3d}  t={time.time()-t0:6.1f}s  fit[min/max]={fs.min():+.2f}/{fs.max():+.2f}  winrate-vs-baseline {wr:.3f}  best {best_wr:.3f}{tag}", flush=True)
    wr = winrate(theta, base_theta, rng, 150)
    if wr > best_wr: best_wr = wr; best = theta.copy()
    json.dump(to_json(best), open(os.path.join(SCR,'rl_policy_new.json'),'w'))
    print(f"DONE gens={gen}  champion winrate-vs-baseline {best_wr:.3f}", flush=True)

if __name__ == '__main__':
    main()
