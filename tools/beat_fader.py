#!/usr/bin/env python3
"""Turnkey orchestrator for the beat-Fader program. One command per phase — see BEAT_FADER.md.

  py tools/beat_fader.py phase1              # analyze his games + fit the bot profile   (~seconds)
  py tools/beat_fader.py phase2              # validate the bot + measure the baseline   (~30 min, inline)
  py tools/beat_fader.py phase3 [--hours 8]  # best-response training vs the fader bot   (DETACHED, GPU)
  py tools/beat_fader.py phase4 [--hours 8]  # long-game / honest-value emphasis         (DETACHED, GPU)
  py tools/beat_fader.py phase5 [--hours 12] # grow the champion (net2net) + train big   (DETACHED, GPU)
  py tools/beat_fader.py gate <net.json>     # the triple ship-gate on any candidate     (~45 min, inline)
  py tools/beat_fader.py status              # is a training run alive? tail its log

Long runs are launched as fully DETACHED processes (they survive this console closing —
trainings run as harness/background children have died on session restarts before; see
memory/gnn-retrain-pipeline). Each phase logs to tools/phaseN_train.log; watch with status.

The trainer's own promotion gate stays active inside every phase (anchor floor, no-forget
screen). gate_fader.py is the FINAL check before you copy a candidate into src/web/.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

SCR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCR)
PY = sys.executable or 'py'
CHAMPION = os.path.join(ROOT, 'src', 'web', 'gnn_policy.json')   # v2.2 — the warm-start seed
COUNTER = os.path.join(SCR, 'counter_gnn.json')        # phase 3/4 output candidate
GROWN = os.path.join(SCR, 'counter_grown.json')        # phase 5 grown-net candidate


def seed_from_champion(out):
    """Warm-start: a challenger from fresh random weights loses to v2.2's lineage on the plateau
    (proven repeatedly, see FINDINGS.md). So a NEW run starts FROM the champion and only has to
    improve against Fader. The trainer's build uses 10 node-features (v2); v2.2 has 7, so we bridge
    with a FEATURE-ONLY net2net growth (embed unchanged -> same arena speed, the thing that beat the
    bigger nets), function-preserving so it starts playing exactly like v2.2. An existing candidate
    is left alone so the trainer resumes it."""
    if os.path.exists(out):
        print('  resuming existing candidate %s' % os.path.basename(out))
        return
    r = subprocess.run([PY, 'tools/grow_net.py', CHAMPION, out, '--dim', '24'],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout or '').strip().splitlines()[-1:] + (r.stderr or '').strip().splitlines()[-3:]
    print('\n'.join('    ' + ln for ln in tail))
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit('seed growth failed — cannot warm-start (see above)')
    print('  warm-start: grew v2.2 -> 10-feature net (embed 24, function-preserving) -> %s'
          % os.path.basename(out))
# fader-heavy league for best-response data generation: fader repeated = oversampled,
# the classic exploiters stay in so the counter never overfits into something they beat.
FADER_LEAGUE = 'script:fader,script:fader,script:fader,script:human,script:turtle,script:rush,script:econ'


def run(argv, **kw):
    print('  $ ' + ' '.join(argv))
    return subprocess.run(argv, cwd=ROOT, **kw)


def detach(argv, log, env_extra=None):
    """Launch a run that survives this console closing (Windows DETACHED_PROCESS)."""
    env = os.environ.copy()
    env.update(env_extra or {})
    fh = open(log, 'a')
    fh.write('\n===== launched %s =====\n$ %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(argv)))
    fh.flush()
    if os.name == 'nt':
        # CREATE_NO_WINDOW (not DETACHED_PROCESS): a fully console-less parent makes every node
        # arena child allocate a VISIBLE console — 16 windows popping per round. A hidden console
        # is inherited by all children (no popups) and still survives our terminal closing.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             creationflags=flags, close_fds=False)
    else:
        # POSIX (cloud VM): its own session survives the SSH connection dropping (like nohup)
        p = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    print('  detached pid %d -> %s' % (p.pid, log))
    print("  watch:  py tools/beat_fader.py status")
    return p.pid


def phase1(a):
    print('== PHASE 1: analyze his games, fit the bot profile ==')
    run([PY, 'tools/scout_fader.py', a.data])
    run([PY, 'tools/analyze_fader_wins.py', a.data])
    run([PY, 'tools/fit_fader_profile.py', a.data])
    print('\nphase 1 done — profile at tools/fader_profile.json. Next: phase2')


def phase2(a):
    print('== PHASE 2: validate the fader bot + measure the baseline ==')
    prof = json.load(open(os.path.join(SCR, 'fader_profile.json')))
    print('  bot profile: warRatio %.1f, reinforce %.0f%%, fracAll %.0f%%'
          % (prof['warRatio'], prof['reinforceRate'] * 100, prof['fracAll'] * 100))
    # the baseline IS the validation: incumbent vs bot at long cap. If the bot never wins a
    # game, it is too weak to train against — refit or strengthen before phase 3.
    r = run([PY, 'tools/gate_fader.py', '--baseline',
             '--games', str(a.games), '--budget', str(a.budget)])
    if r.returncode == 0 and os.path.exists(os.path.join(SCR, 'fader_baseline.json')):
        wr = json.load(open(os.path.join(SCR, 'fader_baseline.json')))['anchor_vs_fader']
        botw = (1 - wr) * 100
        print('\nbot wins %.0f%% vs the shipped net (the human wins ~34%%).' % botw)
        if botw < 5:
            print('WARNING: bot too weak to be a useful sparring partner — strengthen the profile')
        print('phase 2 done. Next: phase3 (GPU)')


def _train(a, log, extra, env_extra=None):
    base = [PY, 'tools/train_gnn_torch.py', '--hours', str(a.hours), '--out', a.out,
            '--script-mix', str(a.script_mix)]
    if a.device:
        base += ['--device', a.device]
    env = {'DEPTH_RULES': '1'}   # train under the SHIPPED game's rules (depth v2, build 13+):
    env.update(env_extra or {})  # inherited by every arena subprocess (data-gen AND internal gates)
    return detach(base + extra, log, env)


def phase3(a):
    print('== PHASE 3: best-response training vs the fader bot (detached, GPU) ==')
    a.out = a.out or COUNTER
    seed_from_champion(a.out)
    _train(a, os.path.join(SCR, 'phase3_train.log'),
           ['--anchor-mix', '0.25', '--long-mix', '0.4', '--long-cap', '240'],
           {'ARENA_SCRIPTS': FADER_LEAGUE})
    print('when it finishes: py tools/beat_fader.py gate ' + a.out)


def phase4(a):
    print('== PHASE 4: long-game emphasis + honest value retraining (detached, GPU) ==')
    a.out = a.out or COUNTER
    seed_from_champion(a.out)
    _train(a, os.path.join(SCR, 'phase4_train.log'),
           ['--anchor-mix', '0.25', '--long-mix', '0.6', '--long-cap', '300',
            '--gate-long-mix', '0.7', '--value-retrain-every', '2'],
           {'ARENA_SCRIPTS': FADER_LEAGUE, 'GAMECAP': '300'})
    print('when it finishes: py tools/beat_fader.py gate ' + a.out)


def phase5(a):
    print('== PHASE 5: grow the champion (net2net) then train the big net (detached, GPU) ==')
    src = a.out or (COUNTER if os.path.exists(COUNTER) else
                    os.path.join(ROOT, 'src', 'web', 'gnn_policy.json'))
    r = run([PY, 'tools/grow_net.py', src, GROWN])
    if r.returncode != 0:
        print('grow_net failed — check tools/grow_net.py usage'); return
    a.out = GROWN
    _train(a, os.path.join(SCR, 'phase5_train.log'),
           ['--anchor-mix', '0.3', '--long-mix', '0.5', '--long-cap', '240'],
           {'ARENA_SCRIPTS': FADER_LEAGUE})
    print('when it finishes: py tools/beat_fader.py gate ' + GROWN)


def gate(a):
    print('== TRIPLE GATE on %s ==' % a.candidate)
    r = run([PY, 'tools/gate_fader.py', a.candidate] + (['--quick'] if a.quick else []))
    if r.returncode == 0:
        print('\nPROMOTABLE. Ship it:')
        print('  copy %s src\\web\\gnn_policy.json' % a.candidate)
        print('  py build\\bump_web.py "Smarter Nightmare: trained specifically to deny the farm-and-snowball strategy"')
        print('  git push origin main && git branch -f release main && git push origin release')


def status(a):
    print('== training processes ==')
    if os.name == 'nt':
        r = subprocess.run(['powershell', '-Command',
                            "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" | "
                            "Where-Object { $_.CommandLine -match 'train_gnn_torch' } | "
                            "Select-Object ProcessId, @{n='Started';e={$_.CreationDate}} | Format-Table -AutoSize"],
                           capture_output=True, text=True)
    else:
        r = subprocess.run(['pgrep', '-af', 'train_gnn_torch'], capture_output=True, text=True)
    out = (r.stdout or '').strip()
    print(out if out else '  (no train_gnn_torch.py running)')
    for n in (3, 4, 5):
        log = os.path.join(SCR, 'phase%d_train.log' % n)
        if os.path.exists(log):
            lines = open(log, errors='replace').read().splitlines()
            print('\n-- tail of %s --' % os.path.basename(log))
            for ln in lines[-8:]:
                print('  ' + ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p1 = sub.add_parser('phase1'); p1.add_argument('--data', default='phone_export/playdata_fader.json')
    p2 = sub.add_parser('phase2'); p2.add_argument('--games', type=int, default=30); p2.add_argument('--budget', type=int, default=60)
    for name in ('phase3', 'phase4', 'phase5'):
        p = sub.add_parser(name)
        p.add_argument('--hours', type=float, default=12.0 if name == 'phase5' else 8.0)
        p.add_argument('--device', default='cuda')
        p.add_argument('--script-mix', type=float, default=0.5)
        p.add_argument('--out', default=None)
    pg = sub.add_parser('gate'); pg.add_argument('candidate'); pg.add_argument('--quick', action='store_true')
    sub.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'phase1': a.__dict__.setdefault('data', 'phone_export/playdata_fader.json')
    {'phase1': phase1, 'phase2': phase2, 'phase3': phase3, 'phase4': phase4,
     'phase5': phase5, 'gate': gate, 'status': status}[a.cmd](a)


if __name__ == '__main__':
    main()
