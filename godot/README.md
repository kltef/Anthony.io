# Conquest — Godot 4 native port (vertical slice)

A from-scratch native rewrite of the WebView game, in **Godot 4** (targeting 4.7; written with
plain 4.x APIs so it opens in any 4.x). This first slice exists to de-risk the two hard ports —
the **map projection** and the **trained RL AI** — and give a playable single-player core to
build the rest on.

## What works
- **Menu flow** — a `MENU → PLAYING → OVER` state machine with a real Godot UI (CanvasLayer +
  Control nodes): title screen, **difficulty picker** (Easy → Grandmaster, the full `LEVELS`
  table), **map picker** (East Coast / Midwest / West Coast / Continental), a Play button, an
  in-game **‹ Menu** back button and a **speed** toggle (1×/2×/3×), and a Victory/Defeat screen
  with Play Again / Menu.
- **Per-team HUD bar** — one coloured segment per team (+ neutral), same as the web version.
- **Map** — pre-baked from the original TopoJSON into flat Albers-USA polygons (`data/us_map.json`),
  so there's **no D3/TopoJSON at runtime**. `Game.gd` fits the active region to the viewport (like
  d3's `fitExtent`) and resizes live.
- **Simulation** — territories, troop growth (cap 150), travelling orbs, capture/flip combat,
  win/lose — a direct port of the web sim.
- **AI** — the **same trained RL policy** (`data/rl_policy.json`): an 8→32→32→1 MLP whose forward
  pass and move-selection biases mirror the web game exactly (verified numerically against the JS).
  Heuristic fallback if the policy is missing.
- **Input** — tap/click one of your states to select it, then tap a target to send all its troops.

Validated by importing + running in real Godot 4.3 (clean compile, no runtime errors) and
rendering the menu and an in-game frame under software GL.

## What's not here yet (next milestones)
The shop + coins, the scenario builder, special tiles & cap upgrades, the send-fraction control,
multiplayer, and sound. The web version remains the shipping build until those land.

## Run it
1. Open Godot 4.x → Import → select this `godot/` folder.
2. Press Play (`F5`). Main scene is `scenes/Main.tscn`.

## Re-baking the map (only if you change regions/source)
```sh
cd tools
npm install           # d3-geo + topojson-client
node bake_map.mjs      # rewrites ../data/us_map.json
```

## Layout
- `scripts/Game.gd` — everything for the slice (load, fit, sim, AI, input, render).
- `scenes/Main.tscn` — the main scene (a `Node2D` running `Game.gd`).
- `data/us_map.json` — baked map polygons; `data/rl_policy.json` — the trained policy.
- `tools/bake_map.mjs` — one-time TopoJSON → Albers screen-polygon pre-bake.
