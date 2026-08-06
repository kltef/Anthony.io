#!/usr/bin/env python3
"""JS<->Python parity gate for the EXTENDED 24-feature policy input.

Nets are trained in Python and run in JS, so the feature builder is a CONTRACT with four
implementations that must agree bit-for-bit-ish (the repo's standing bar is ~1e-7, and features
specifically have historically matched to ~2e-16):

  0. candFeats24()  — tools/selfplay_arena.js (writes every --dump-visits training example)
  1. rlMoveFeats()  — src/web/index.html, main thread (greedy scorer + in-page fallback search)
  2. polMoveFeats() — src/web/index.html, INSIDE planWorkerMain (the Web Worker; this is the copy
                      selfplay_arena.js extracts, so it produces every training example)
  3. feats24()      — tools/train_rl.py (numpy, used by the trainers)
  4. the reference below — plain stdlib Python, the written-down spec

This script is self-contained (no test framework), same posture as test_gnn_batch_equiv.py.
Run it after touching ANY of the four. numpy is optional: without it, legs 1/2/4 still run, which
is the important part — those three are what ship.

  usage: python3 tools/test_feats24_parity.py
"""
import json
import math
import os
import random
import re
import subprocess
import sys

SCR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCR)
HTML = os.path.join(ROOT, 'src', 'web', 'index.html')
FN = 50.0
CAP = 150.0
FRACTIONS = (1.0, 0.75, 0.5, 0.25)
TOL = 1e-9


# ---------------------------------------------------------------- the reference (the spec)
def ref_feats24(board, owner, si, ti, frac):
    """Plain-Python reference. Mirrors the comment block in train_rl.feats24."""
    o, t, adj = board['owner'], board['troops'], board['adj']
    n = len(o)
    st, tt = float(t[si]), float(t[ti])
    sent = st * frac
    dx = board['cx'][ti] - board['cx'][si]
    dy = board['cy'][ti] - board['cy'][si]
    dist = math.hypot(dx, dy) / board['rlDiag']

    enemy = [i for i in range(n) if o[i] != owner and o[i] != 0]
    maxEnemy = max((float(t[i]) for i in enemy), default=0.0)
    bigIdx = max(enemy, key=lambda i: t[i]) if enemy else -1
    myTot = sum(float(t[i]) for i in range(n) if o[i] == owner)
    myStates = sum(1 for i in range(n) if o[i] == owner)
    allTot = sum(float(t[i]) for i in range(n) if o[i] != 0)
    myShare = myTot / allTot if allTot > 0 else 0.0
    myAvg = myTot / max(1, myStates)
    ownFrac = myStates / n

    def max_host(i):
        best = 0.0
        for j in adj[i]:
            if o[j] != owner and o[j] != 0 and t[j] > best:
                best = float(t[j])
        return best

    inflight = [0.0] * n
    for a in board['armies']:
        if 0 <= a['ti'] < n:
            inflight[a['ti']] += (-a['count'] if a['owner'] == owner else a['count'])

    neutralsFrac = sum(1 for i in range(n) if o[i] == 0) / n

    flight = (dist * board['rlDiag']) / board['orbSpeed']
    grow = 1.0 * flight if (o[ti] != 0 and tt < CAP) else 0.0

    gain = sum(1 for j in adj[ti] if o[j] != owner)
    lose = sum(1 for j in adj[ti] if o[j] == owner)
    borderDelta = (gain - lose) / 8.0

    return [
        st / FN, tt / FN, (sent - tt) / FN, dist,
        1.0 if o[ti] == 0 else 0.0,
        1.0 if (o[ti] != 0 and o[ti] != owner) else 0.0,
        1.0 if sent > tt else 0.0,
        ownFrac,
        maxEnemy / FN, myShare, myAvg / FN,
        1.0 if ti == bigIdx else 0.0,
        (st - sent) / FN, frac, max_host(si) / FN, max_host(ti) / FN,
        (sent - (tt + grow)) / FN, inflight[ti] / FN, inflight[si] / FN,
        len(adj[ti]) / 8.0, borderDelta, len(adj[si]) / 8.0,
        neutralsFrac, myAvg / CAP,
    ]


# ---------------------------------------------------------------- boards
def make_board(rng, n=14):
    cx = [rng.random() for _ in range(n)]
    cy = [rng.random() for _ in range(n)]
    adj = [[] for _ in range(n)]
    for i in range(n):                     # ring + random chords: connected, varied degree
        j = (i + 1) % n
        adj[i].append(j)
        adj[j].append(i)
    for _ in range(n):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b and b not in adj[a]:
            adj[a].append(b)
            adj[b].append(a)
    owner = [rng.choice([0, 1, 2, 2]) for _ in range(n)]
    troops = [rng.uniform(1, 200) for _ in range(n)]
    armies = []
    for _ in range(rng.randrange(4)):
        armies.append({'owner': rng.choice([1, 2]), 'count': rng.uniform(5, 80),
                       'ti': rng.randrange(n), 'ttotal': rng.uniform(0.1, 3.0), 't': 0.0})
    rlDiag = math.hypot(max(cx) - min(cx), max(cy) - min(cy)) or 1.0
    return {'n': n, 'cx': cx, 'cy': cy, 'owner': owner, 'troops': troops,
            'adj': adj, 'armies': armies, 'rlDiag': rlDiag, 'orbSpeed': 70.0}


# ---------------------------------------------------------------- JS side
JS_DRIVER = r'''
const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8').replace(/\r\n/g,'\n');
const cases=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));

// ---- copy 2: the WORKER's polMoveFeats/polBoardCtx, lifted from planWorkerMain verbatim ----
const wsrc=html.match(/function planWorkerMain\(\)\s*\{[\s\S]*?\n  \}\n/)[0];
function grab(name){
  const re=new RegExp('\\n    function '+name+'\\([\\s\\S]*?\\n    \\}');
  const m=wsrc.match(re);
  if(!m) throw new Error('could not lift '+name+' from planWorkerMain');
  return m[0];
}
const workerBody=[grab('polBoardCtx'),grab('polBorderDelta'),grab('polMoveFeats')].join('\n');
const mkWorker=new Function('NEUTRAL','MAX_TROOPS','CAPS','GMUL','DIAG','ORB','growRateFor',
  workerBody+'\n; return {ctx:polBoardCtx, feats:polMoveFeats};');

// ---- copy 1: the MAIN-THREAD rlBoardCtx/rlBorderDelta/rlMoveFeats ----
function grabMain(name){
  const re=new RegExp('\\n  function '+name+'\\([\\s\\S]*?\\n  \\}');
  const m=html.replace(/\r\n/g,'\n').match(re);
  if(!m) throw new Error('could not lift '+name+' from the main thread');
  return m[0];
}
const mainBody=[grabMain('rlBoardCtx'),grabMain('rlBorderDelta'),grabMain('rlMoveFeats')].join('\n');
const mkMain=new Function('NEUTRAL','MAX_TROOPS','rlDiag','ORB_SPEED','growRateFor',
  mainBody+'\n; return {ctx:rlBoardCtx, feats:rlMoveFeats};');

// ---- copy 3: the ARENA's candFeats24 (writes every --dump-visits training example) ----
const arenaSrc=fs.readFileSync(process.argv[4],'utf8').replace(/\r\n/g,'\n');
const am=arenaSrc.match(/\nfunction candFeats24\([\s\S]*?\n\}/);
if(!am) throw new Error('could not lift candFeats24 from selfplay_arena.js');
const mkArena=new Function('ORB', am[0]+'\n; return candFeats24;');

const out=[];
for(const c of cases){
  const s={n:c.board.n,cx:c.board.cx,cy:c.board.cy,owner:c.board.owner,troops:c.board.troops,
           adj:c.board.adj,armies:c.board.armies,caps:null,gmul:null,fort:null};
  const grow=o=>(o===0?0:1.0);   // train_rl.py: every owner grows at 1.0/s
  const W=mkWorker(0,150,null,null,c.board.rlDiag,c.board.orbSpeed,grow);
  const M=mkMain(0,150,c.board.rlDiag,c.board.orbSpeed,grow);
  const glob=(()=>{
    let myTot=0,allTot=0,maxEnemy=0,bigIdx=-1,ownCount=0;
    for(let i=0;i<s.n;i++){const oo=s.owner[i],tr=s.troops[i];
      if(oo===c.owner) ownCount++;
      if(oo!==0){allTot+=tr; if(oo===c.owner) myTot+=tr; else if(tr>maxEnemy){maxEnemy=tr;bigIdx=i;}}}
    return {ownFrac:ownCount/s.n, maxEnemy, myShare:allTot>0?myTot/allTot:0,
            myAvg:myTot/Math.max(1,ownCount), bigIdx};
  })();
  const dist=Math.hypot(s.cx[c.ti]-s.cx[c.si], s.cy[c.ti]-s.cy[c.si])/c.board.rlDiag;
  const A=mkArena(c.board.orbSpeed);
  const env={n:s.n,cx:s.cx,cy:s.cy,owner:s.owner,troops:s.troops,adj:s.adj,armies:s.armies,rlDiag:c.board.rlDiag};
  out.push({worker:W.feats(s,c.owner,W.ctx(s,c.owner),c.si,c.ti,c.frac,dist,glob,50.0),
            main:  M.feats(s,c.owner,M.ctx(s,c.owner),c.si,c.ti,c.frac,dist,glob,50.0),
            arena: A(env,c.si,c.ti,c.frac,c.owner)});
}
process.stdout.write(JSON.stringify(out));
'''

NAMES = ['st', 'tt', 'sent-tt', 'dist', 'isNeutral', 'isEnemy', 'sent>tt', 'ownFrac',
         'maxEnemy', 'myShare', 'myAvg', 'isBiggest', 'residual', 'frac', 'maxHostSrc',
         'maxHostTgt', 'arrivalDelta', 'inflightTgt', 'inflightSrc', 'degTgt', 'borderDelta',
         'degSrc', 'neutralsFrac', 'capPressure']


def main():
    rng = random.Random(20260806)
    cases, refs = [], []
    for _ in range(40):
        b = make_board(rng)
        owner = 2
        mine = [i for i in range(b['n']) if b['owner'][i] == owner]
        if not mine:
            continue
        si = rng.choice(mine)
        ti = rng.choice([i for i in range(b['n']) if i != si])
        frac = rng.choice(FRACTIONS)
        cases.append({'board': b, 'owner': owner, 'si': si, 'ti': ti, 'frac': frac})
        refs.append(ref_feats24(b, owner, si, ti, frac))

    drv = os.path.join(SCR, '_feats24_driver.js')
    dat = os.path.join(SCR, '_feats24_cases.json')
    with open(drv, 'w', newline='\n') as f:
        f.write(JS_DRIVER)
    with open(dat, 'w', newline='\n') as f:
        json.dump(cases, f)
    try:
        raw = subprocess.run(['node', drv, HTML, dat, os.path.join(SCR, 'selfplay_arena.js')],
                             capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        print('JS driver failed:\n' + e.stderr)
        return 1
    finally:
        for p in (drv, dat):
            if os.path.exists(p):
                os.remove(p)
    js = json.loads(raw)

    worst = {}
    fails = 0
    for k, (r, j) in enumerate(zip(refs, js)):
        for which in ('main', 'worker', 'arena'):
            v = j[which]
            if len(v) != 24:
                print(f'case {k}: {which} returned {len(v)} features, expected 24')
                fails += 1
                continue
            for i in range(24):
                d = abs(v[i] - r[i])
                key = (which, i)
                if d > worst.get(key, -1):
                    worst[key] = d
                if d > TOL:
                    fails += 1
                    if fails <= 10:
                        print(f'case {k} {which} feature {i} ({NAMES[i]}): '
                              f'js={v[i]!r} ref={r[i]!r} diff={d:g}')

    mx_main = max((d for (w, _), d in worst.items() if w == 'main'), default=0.0)
    mx_work = max((d for (w, _), d in worst.items() if w == 'worker'), default=0.0)
    mx_aren = max((d for (w, _), d in worst.items() if w == 'arena'), default=0.0)
    print(f'cases: {len(refs)}   features/case: 24')
    print(f'  main-thread rlMoveFeats   vs reference: max abs diff {mx_main:g}')
    print(f'  worker      polMoveFeats  vs reference: max abs diff {mx_work:g}')
    print(f'  arena       candFeats24   vs reference: max abs diff {mx_aren:g}')

    try:
        import numpy as np  # noqa: F401
        sys.path.insert(0, SCR)
        import train_rl as T
        mx_np = 0.0
        for c, r in zip(cases, refs):
            b = c['board']
            g = dict(pos=np.array(list(zip(b['cx'], b['cy'])), dtype=np.float64),
                     owner=np.array(b['owner'], dtype=np.int64),
                     troops=np.array(b['troops'], dtype=np.float64),
                     armies=[dict(a) for a in b['armies']], rlDiag=b['rlDiag'],
                     ORB=b['orbSpeed'], N=b['n'], adj=b['adj'])
            glob = T.globals_for(g, c['owner'])
            ctx = T.board_ctx(g, c['owner'])
            tg = np.array([c['ti']], dtype=np.int64)
            v = T.feats24(g, c['si'], c['owner'], tg, glob, ctx, c['frac'])[0]
            for i in range(24):
                mx_np = max(mx_np, abs(float(v[i]) - r[i]))
        print(f'  train_rl.py feats24       vs reference: max abs diff {mx_np:g}')
        if mx_np > TOL:
            fails += 1
    except ImportError:
        print('  train_rl.py feats24       vs reference: SKIPPED (numpy not installed)')

    print('\nPARITY OK' if fails == 0 else f'\nPARITY FAILED ({fails} mismatches)')
    return 0 if fails == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
