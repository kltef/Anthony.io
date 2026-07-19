# BEAT_FADER.md — the turnkey program to beat the play-tester

Goal: drive the play-tester's win rate vs Nightmare from ~34% toward ~10-15%, with the AI
playing *clean* (no over-extend cheese). Method: model him from his own exported games,
train the net specifically to deny his strategy, and gate every candidate three ways so we
never ship another regression.

**His measured strategy** (135 games, `tools/analyze_fader_wins.py`): farm-and-snowball.
71% of his moves consolidate troops (banking + defense), he holds attacks until his total
troops reach ~3x the enemy's, then converts with full-stack joint salvos at ~7x local
force. He wins when the snowball compounds (troops AND territory rise together); he loses
when he over-hoards and the AI out-territories him. The counter: **early territory
pressure that denies the farm** — applied cleanly.

## The one-command pipeline

```sh
py tools/beat_fader.py phase1               # analyze + fit the bot        (~seconds, no GPU)
py tools/beat_fader.py phase2               # validate bot + baseline      (~30 min, no GPU)
py tools/beat_fader.py phase3 --hours 8     # best-response training       (DETACHED, GPU)
py tools/beat_fader.py gate tools/counter_gnn.json        # the triple gate (~45 min)
py tools/beat_fader.py phase4 --hours 8     # long-game + value emphasis   (DETACHED, GPU)
py tools/beat_fader.py phase5 --hours 12    # grow the net (net2net) + train (DETACHED, GPU)
py tools/beat_fader.py status               # is training alive? tail logs
```

Phases 3-5 launch **detached** (they survive closing this console; watch with `status`).
Run `gate` after each training phase; promote only on a full PASS.

## What each phase is

| Phase | What it does | Output |
|---|---|---|
| 1 | Scouting report + wins-vs-losses analysis + fits the bot parameters from his real games | `tools/fader_profile.json` |
| 2 | Plays the shipped net vs the fitted **fader bot** (long game-cap) — validates the bot is a real sparring partner and records the incumbent's number as the bar | `tools/fader_baseline.json` |
| 3 | **Best-response training**: data generation runs a fader-heavy league (`ARENA_SCRIPTS`, bot oversampled 3x, classic exploiters kept in so nothing is forgotten), anchor-mix keeps v2.2's lineage in the signal | `tools/counter_gnn.json` |
| 4 | Same, but long games (`GAMECAP 300`) + frequent value retraining — teaches the value head what a snowball loss looks like *before* it happens, which is the honest fix for the over-commit bias | updates `counter_gnn.json` |
| 5 | `grow_net.py` (function-preserving net2net: same policy, bigger capacity) then trains the grown net with the same league | `tools/counter_grown.json` |

## The triple gate (`tools/gate_fader.py`)

A candidate is promotable only if ALL pass:

1. **FADER** — beats the fader bot more than the incumbent does (long game-cap; the goal metric).
2. **ANCHOR** — >= 45% head-to-head vs `src/web/gnn_policy.json` (no regression vs v2.2).
3. **LEAGUE** — >= 80% vs each of turtle/rush/econ/farmer/human (no forgetting, no overfit-to-Fader).

`--quick` gives a fast smoke read; full defaults are the real measurement.

## Shipping a winner

```sh
copy tools\counter_gnn.json src\web\gnn_policy.json
py build\bump_web.py "Smarter Nightmare: trained to deny the farm-and-snowball strategy"
git push origin main && git branch -f release main && git push origin release
```

## The adaptation loop (after shipping)

He will adapt. Each round: collect **~30-50 fresh games of him vs the new build** (in-app
⬇ Data export, or `tools/extract_idb_playdata.py` straight from the phone's IndexedDB over
adb), drop the JSON into `phone_export/`, then rerun from phase1 — the bot re-fits to his
new style automatically and the whole pipeline re-runs unchanged.

## Files

- `tools/beat_fader.py` — orchestrator (this whole document as commands)
- `tools/fit_fader_profile.py` — playdata -> bot parameters
- `tools/gate_fader.py` — the triple gate
- `tools/scout_fader.py`, `tools/analyze_fader_wins.py` — the analysis pair
- `tools/extract_idb_playdata.py` — pull games off a phone without the in-app export
- `selfplay_arena.js` `SCRIPTS.fader` — the sparring bot (env `FADER_PROFILE` overrides;
  gets the econ-perk troop drip in the arena loop, mirroring his real coin-shop economy)
- `train_gnn_torch.py` — reads `ARENA_SCRIPTS` to reshape the data-generation league

## Honest expectations

- The trainer's own gates + this triple gate mean a phase can end with **no promotable
  candidate** — that is the system working, not failing. Rerun with more hours or move to
  the next phase; never ship a candidate that failed a leg.
- The bot is a model of him, not him. Ship-gate wins vs the bot are necessary, not
  sufficient — the final judge is a fresh batch of his real games vs the new build.
