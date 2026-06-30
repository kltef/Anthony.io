#!/usr/bin/env python3
# Phase D, lever #1: self-improvement with a REAL-PLANNER promotion gate.
#
# Phase D's loop failed because it gated on cheap greedy head-to-head, which didn't transfer to MCTS
# planner strength. This fixes the objective: the gate shells out to the Node arena (selfplay_arena.js)
# and plays candidate-vs-best with the ACTUAL deployed planner. A candidate is promoted only if it
# beats best as a *planner* (+ a cheap turtle/rush no-forget check). Far costlier per round (~real
# planner games), so far fewer rounds — but every promotion is a genuine, deployment-relevant gain.
#   usage: python3 tools/train_selfplay_planner.py [seconds]
import numpy as np, json, time, sys, os, subprocess, re
from multiprocessing import Pool
import train_rl as T
import train_value as V
import train_selfplay_loop as M     # reuse generation/training/teacher machinery

PARCH=[12,64,64,1]; T.ARCH=PARCH
SCR=T.SCR; ROOT=os.path.dirname(SCR); POLICY=T.POLICY
SPDIR=os.environ.get('SPDIR', SCR)
CAND=os.path.join(SPDIR,'_cand_net.json'); BEST=os.path.join(SPDIR,'_best_net.json')

def write_net(pth, vth, path):
    j=T.to_json(pth); j['value']=V.vto_json(vth); json.dump(j, open(path,'w'))
def planner_gate(cand_p, best_p, vth, games, budget):
    # both sides share the same value net so this isolates the POLICY change, judged by the real planner
    write_net(cand_p, vth, CAND); write_net(best_p, vth, BEST)
    try:
        out=subprocess.run(['node','tools/selfplay_arena.js',CAND,BEST,str(games),str(budget)],
                           capture_output=True, text=True, cwd=ROOT, timeout=600).stdout
    except Exception as e:
        print("  gate error:",e, flush=True); return None
    m=re.search(r'A won (\d+)/(\d+)', out)
    return (int(m.group(1))/int(m.group(2))) if m else None

def main():
    secs=float(sys.argv[1]) if len(sys.argv)>1 else 3600.0
    GATE_GAMES=int(os.environ.get('GATE_GAMES','40')); GATE_BUDGET=int(os.environ.get('GATE_BUDGET','50'))
    best_p=M.policy_theta_from_json(POLICY)
    best_v=M.value_theta_from_json(os.path.join(SCR,'value_net.json'))
    rng=np.random.default_rng(7)
    base_turtle=M.vs_script(best_p,'turtle',rng,60); base_rush=M.vs_script(best_p,'rush',rng,60)
    print(f"PLANNER-GATED LOOP  gate={GATE_GAMES}g@{GATE_BUDGET}ms  budget={secs:.0f}s cores={os.cpu_count()}", flush=True)
    print(f"baseline best vs scripts: turtle {base_turtle:.2f} rush {base_rush:.2f}", flush=True)
    json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_planner_new.json'),'w'))
    json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_planner_new.json'),'w'))
    pool=Pool(processes=min(4,os.cpu_count() or 4)); t0=time.time()
    pbuf=[]; vbuf=[]; rnd=0; promotions=0; gates=0
    while time.time()-t0<secs:
        rnd+=1
        # 1) generate a chunk of self-play (league-mixed) to refresh the buffers
        p_ser=[(W.tolist(),b.tolist()) for (W,b) in T.unpack(best_p)[0]]; noop=float(best_p[-1]); v_ser=best_v.tolist()
        seeds=[int(rng.integers(1<<30)) for _ in range(200)]
        for pol,valout in pool.map(M.gen_game,[(p_ser,noop,v_ser,s) for s in seeds]):
            if pol: pbuf.append((np.array(pol[0],dtype=np.float64),pol[1]))
            for (vf,out) in valout: vbuf.append((np.array(vf,dtype=np.float64),float(out)))
        if len(pbuf)>9000: pbuf=pbuf[-9000:]
        if len(vbuf)>40000: vbuf=vbuf[-40000:]
        if len(pbuf)<800:
            print(f"r{rnd:3d} t={time.time()-t0:5.0f}s warmup pbuf={len(pbuf)}", flush=True); continue
        # 2) train a candidate from best
        cand=best_p.copy(); st=(np.zeros(len(cand)),np.zeros(len(cand)),0,2e-4)
        cand,st,ploss=M.train_policy(cand,pbuf,st,epochs=3,rng=rng)
        # cheap no-forget screen first (skip the expensive gate if it forgot turtles)
        tt=M.vs_script(cand,'turtle',rng,40); rs=M.vs_script(cand,'rush',rng,40)
        if tt<base_turtle-0.06 or rs<base_rush-0.08:
            print(f"r{rnd:3d} t={time.time()-t0:5.0f}s pbuf={len(pbuf)} ploss~{ploss:.3f} tt={tt:.2f} rs={rs:.2f} no-forget-FAIL (skip gate)", flush=True); continue
        # 3) REAL-PLANNER gate
        wr=planner_gate(cand,best_p,best_v,GATE_GAMES,GATE_BUDGET); gates+=1
        promoted=False
        if wr is not None and wr>=0.56:
            best_p=cand; promotions+=1; promoted=True
            json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_planner_new.json'),'w'))
        # 4) co-train value occasionally
        vtag=""
        if rnd%3==0 and len(vbuf)>=3000:
            best_v,vloss=M.train_value(best_v,vbuf,rng); vtag=f" vretrain(mse{vloss:.3f})"
            json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_planner_new.json'),'w'))
        print(f"r{rnd:3d} t={time.time()-t0:5.0f}s pbuf={len(pbuf)} ploss~{ploss:.3f} tt={tt:.2f} rs={rs:.2f} PLANNER-gate={wr if wr is None else round(wr,2)} {'PROMOTED #%d'%promotions if promoted else 'kept'}{vtag}", flush=True)
    pool.close(); pool.join()
    print(f"DONE rounds={rnd} gates={gates} promotions={promotions}", flush=True)
    ft=M.vs_script(best_p,'turtle',rng,80); fr=M.vs_script(best_p,'rush',rng,80)
    print(f"final best vs scripts: turtle {ft:.2f}(base {base_turtle:.2f}) rush {fr:.2f}(base {base_rush:.2f})", flush=True)
    json.dump(T.to_json(best_p),open(os.path.join(SCR,'policy_planner_new.json'),'w'))
    json.dump(V.vto_json(best_v),open(os.path.join(SCR,'value_planner_new.json'),'w'))

if __name__=='__main__':
    main()
