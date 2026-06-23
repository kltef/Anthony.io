extends Node2D
# Native (Godot 4) port of the State.io-style conquest game — vertical slice.
# Single-player vs the trained RL AI on the East Coast map. This proves out the two hard
# ports: the map projection (pre-baked to flat polygons, see tools/bake_map.mjs) and the
# RL policy (the same 8->32->32->1 MLP forward pass as the web game).
#
# Tap/click one of your states to select it, then tap a target to send all its troops.
# Tap anywhere when the match is over to play again.

const NEUTRAL := 0
const PLAYER := 1
const MAX_TROOPS := 150.0
const REGION := "East Coast"   # matches the web game's solo default

const COLORS := {0:"#b4bbc2", 1:"#46a6ef", 2:"#ef5b52", 3:"#f1a13a", 4:"#9b6cf0", 5:"#38c46f", 6:"#e36ec9", 7:"#46c7d6", 8:"#d9b53a"}
const DARK   := {0:"#8b939c", 1:"#2c7fc4", 2:"#c43d36", 3:"#c47d24", 4:"#7549c4", 5:"#27975a", 6:"#b94fa1", 7:"#2f9aa8", 8:"#b08f22"}

# difficulty knobs (Master tier: trained RL enemies, no economic handicap beyond the web defaults)
var enemies := 3
var ai_speed := 0.85
var neutral_min := 15
var neutral_max := 50
var grow_rate := 1.0
var enemy_grow := 1.0
var player_start := 30
var enemy_start := 36
var use_rl := true

var states: Array = []          # each: Dictionary (see _load_map)
var orbs: Array = []            # each: { owner, pos:Vector2, tidx:int, count:float, speed:float }
var policy: Dictionary = {}
var rl_diag := 1.0
var running := false
var selected := -1
var ai_timers: Dictionary = {}
var game_elapsed := 0.0
var mark_r := 16.0
var orb_speed := 200.0
var last_size := Vector2.ZERO
var result_text := ""
var font: Font

func _ready() -> void:
	randomize()
	font = ThemeDB.fallback_font
	_load_policy()
	_load_map()
	_new_game()

# ---------- math helper (GDScript has no tanh) ----------
func _tanh(v: float) -> float:
	if v > 20.0: return 1.0
	if v < -20.0: return -1.0
	var e := exp(2.0 * v)
	return (e - 1.0) / (e + 1.0)

# ---------- loading ----------
func _load_policy() -> void:
	var f := FileAccess.open("res://data/rl_policy.json", FileAccess.READ)
	if f != null:
		var v = JSON.parse_string(f.get_as_text())
		if typeof(v) == TYPE_DICTIONARY:
			policy = v

func _load_map() -> void:
	var f := FileAccess.open("res://data/us_map.json", FileAccess.READ)
	if f == null:
		push_error("us_map.json missing — run tools/bake_map.mjs")
		return
	var data = JSON.parse_string(f.get_as_text())
	var region_list = null
	for r in data["regions"]:
		if r["name"] == REGION:
			region_list = r["list"]
	states.clear()
	for s in data["states"]:
		if region_list != null and not (s["name"] in region_list):
			continue
		var raw_polys: Array = []
		for poly in s["polys"]:
			var rings: Array = []
			for ring in poly:
				var pts: Array = []
				for p in ring:
					pts.append(Vector2(float(p[0]), float(p[1])))
				rings.append(pts)
			raw_polys.append(rings)
		states.append({
			"name": s["name"], "raw": raw_polys, "scr": [],
			"rcx": float(s["cx"]), "rcy": float(s["cy"]), "cx": 0.0, "cy": 0.0,
			"owner": NEUTRAL, "troops": 0.0, "grow_acc": 0.0, "pulse": 0.0,
		})

# ---------- layout: fit the baked Albers coords into the viewport (replicates d3 fitExtent) ----------
func _fit() -> void:
	var vs := get_viewport_rect().size
	last_size = vs
	var minx := 1e9; var miny := 1e9; var maxx := -1e9; var maxy := -1e9
	for s in states:
		for poly in s["raw"]:
			for ring in poly:
				for v in ring:
					minx = min(minx, v.x); miny = min(miny, v.y)
					maxx = max(maxx, v.x); maxy = max(maxy, v.y)
	var bw: float = max(1.0, maxx - minx)
	var bh: float = max(1.0, maxy - miny)
	var ml := 14.0; var mr := 14.0; var mt := 70.0; var mb := 18.0
	var sc: float = min((vs.x - ml - mr) / bw, (vs.y - mt - mb) / bh)
	var ox := ml + ((vs.x - ml - mr) - bw * sc) * 0.5 - minx * sc
	var oy := mt + ((vs.y - mt - mb) - bh * sc) * 0.5 - miny * sc
	for s in states:
		var scr: Array = []
		for poly in s["raw"]:
			var rings: Array = []
			for ring in poly:
				var pv := PackedVector2Array()
				for v in ring:
					pv.append(Vector2(v.x * sc + ox, v.y * sc + oy))
				rings.append(pv)
			scr.append(rings)
		s["scr"] = scr
		s["cx"] = s["rcx"] * sc + ox
		s["cy"] = s["rcy"] * sc + oy
	# marker radius from the median nearest-neighbour gap
	var gaps: Array = []
	for a in states:
		var m := 1e9
		for b in states:
			if a == b: continue
			var d: float = Vector2(a["cx"], a["cy"]).distance_to(Vector2(b["cx"], b["cy"]))
			if d < m: m = d
		if m < 1e9: gaps.append(m)
	gaps.sort()
	var med: float = gaps[gaps.size() / 2] if gaps.size() > 0 else 40.0
	mark_r = clamp(med * 0.46, 8.0, 22.0)
	# distance normaliser for the RL features + a resolution-independent orb pace
	var dminx := 1e9; var dminy := 1e9; var dmaxx := -1e9; var dmaxy := -1e9
	for s in states:
		dminx = min(dminx, s["cx"]); dminy = min(dminy, s["cy"])
		dmaxx = max(dmaxx, s["cx"]); dmaxy = max(dmaxy, s["cy"])
	rl_diag = max(1.0, Vector2(dmaxx - dminx, dmaxy - dminy).length())
	orb_speed = clamp(rl_diag / 4.0, 120.0, 420.0)

# ---------- match setup ----------
func _new_game() -> void:
	result_text = ""
	for s in states:
		s["owner"] = NEUTRAL
		s["troops"] = float(randi_range(neutral_min, neutral_max))
		s["grow_acc"] = 0.0
		s["pulse"] = 0.0
	if last_size != get_viewport_rect().size:
		_fit()
	var first := randi() % states.size()
	states[first]["owner"] = PLAYER
	states[first]["troops"] = float(player_start)
	var starters: Array = [first]
	var ecount := clampi(enemies, 1, states.size() - 1)
	for e in range(ecount):
		var b := _farthest_pick(starters)
		states[b]["owner"] = 2 + e
		states[b]["troops"] = float(enemy_start)
		starters.append(b)
	orbs.clear()
	ai_timers.clear()
	selected = -1
	game_elapsed = 0.0
	running = true
	queue_redraw()

func _farthest_pick(starters: Array) -> int:
	var best := -1
	var bestd := -1.0
	for i in range(states.size()):
		if i in starters: continue
		var d := 1e18
		for j in starters:
			var dd: float = Vector2(states[i]["cx"], states[i]["cy"]).distance_squared_to(Vector2(states[j]["cx"], states[j]["cy"]))
			if dd < d: d = dd
		if starters.is_empty(): d = 0.0
		if d > bestd: bestd = d; best = i
	return best

# ---------- loop ----------
func _process(delta: float) -> void:
	if get_viewport_rect().size != last_size:
		_fit()
		queue_redraw()
	if running:
		game_elapsed += delta
		_step(delta)
		queue_redraw()

func _step(dt: float) -> void:
	for s in states:
		if s["owner"] != NEUTRAL and s["troops"] < MAX_TROOPS:
			s["grow_acc"] += _grow_rate_for(s["owner"]) * dt
			while s["grow_acc"] >= 1.0:
				s["grow_acc"] -= 1.0
				if s["troops"] < MAX_TROOPS: s["troops"] += 1.0
		if s["pulse"] > 0.0:
			s["pulse"] = max(0.0, s["pulse"] - dt * 3.0)
	var i := orbs.size() - 1
	while i >= 0:
		var o = orbs[i]
		var tgt = states[o["tidx"]]
		var to := Vector2(tgt["cx"], tgt["cy"])
		var d: float = o["pos"].distance_to(to)
		var stp: float = o["speed"] * dt
		if d <= stp + 2.0:
			_arrive(o)
			orbs.remove_at(i)
		else:
			o["pos"] += (to - o["pos"]) / d * stp
		i -= 1
	_update_ai(dt)
	_check_win()

func _grow_rate_for(owner: int) -> float:
	if owner == PLAYER: return grow_rate
	if owner >= 2: return grow_rate * enemy_grow
	return 0.0

func _arrive(o: Dictionary) -> void:
	var to = states[o["tidx"]]
	if to["owner"] == o["owner"]:
		to["troops"] += o["count"]
	else:
		to["troops"] -= o["count"]
		if to["troops"] < 0.0:
			to["owner"] = o["owner"]
			to["troops"] = -to["troops"]
			to["pulse"] = 1.2

func _send(from_idx: int, to_idx: int) -> void:
	if from_idx < 0 or to_idx < 0 or from_idx == to_idx: return
	var f = states[from_idx]
	if f["owner"] == NEUTRAL: return
	var amount: float = f["troops"]
	if amount < 1.0: return
	f["troops"] = 0.0
	f["pulse"] = 1.0
	orbs.append({"owner": f["owner"], "pos": Vector2(f["cx"], f["cy"]), "tidx": to_idx, "count": amount, "speed": orb_speed})

# ---------- AI ----------
func _update_ai(dt: float) -> void:
	var present: Dictionary = {}
	for s in states:
		if s["owner"] >= 2: present[s["owner"]] = true
	for owner in present.keys():
		if not ai_timers.has(owner):
			ai_timers[owner] = randf_range(0.4, ai_speed)
		ai_timers[owner] -= dt
		if ai_timers[owner] > 0.0: continue
		ai_timers[owner] = randf_range(0.06, 0.14)
		if use_rl and not policy.is_empty():
			_rl_move(owner)
		else:
			_heuristic_move(owner)

# exact mirror of rlScore() in the web game: feats[8] -> scalar logit
func _rl_score(feats: Array) -> float:
	var x: Array = feats
	for layer in policy["layers"]:
		var b = layer["b"]
		var w = layer["W"]
		var is_tanh: bool = layer["act"] == "tanh"
		var out: Array = []
		for o in range(b.size()):
			var acc: float = float(b[o])
			var row = w[o]
			for k in range(row.size()):
				acc += float(row[k]) * x[k]
			out.append(_tanh(acc) if is_tanh else acc)
		x = out
	return x[0]

# mirror of rlEvaluate(): pick the best (src,tgt) move under the policy + the same biases
func _rl_move(owner: int) -> void:
	var fn: float = float(policy.get("feat_norm", 50.0))
	var n := states.size()
	var own_count := 0
	for s in states:
		if s["owner"] == owner: own_count += 1
	var own_frac := float(own_count) / float(n)
	var srcs: Array = []
	for i in range(n):
		if states[i]["owner"] == owner and states[i]["troops"] >= 5.0:
			srcs.append(i)
	if srcs.is_empty(): return
	var neutrals_left := false
	var rivals_left := false
	var player_states: Array = []
	for i in range(n):
		var ow = states[i]["owner"]
		if ow == NEUTRAL: neutrals_left = true
		elif ow >= 2 and ow != owner: rivals_left = true
		elif ow == PLAYER: player_states.append(i)
	var endgame := not neutrals_left and not rivals_left
	var noop: float = float(policy.get("noop_bias", 0.0))
	var noop_eff: float = noop - 1.0 if endgame else noop
	var best_score := noop_eff
	var best_src := -1
	var best_tgt := -1
	for si in srcs:
		var st: float = states[si]["troops"]
		var sc := Vector2(states[si]["cx"], states[si]["cy"])
		for ti in range(n):
			if ti == si: continue
			var tg = states[ti]
			if tg["owner"] == owner: continue
			var tt: float = tg["troops"]
			var tc := Vector2(tg["cx"], tg["cy"])
			var dist: float = sc.distance_to(tc)
			var feats := [
				st / fn, tt / fn, (st - tt) / fn, dist / rl_diag,
				1.0 if tg["owner"] == NEUTRAL else 0.0,
				1.0 if (tg["owner"] != NEUTRAL and tg["owner"] != owner) else 0.0,
				1.0 if st > tt else 0.0, own_frac,
			]
			var raw := _rl_score(feats)
			var bias := 0.0
			if tg["owner"] == NEUTRAL:
				if not player_states.is_empty():
					var dmin := 1e18
					for pi in player_states:
						var dd: float = tc.distance_to(Vector2(states[pi]["cx"], states[pi]["cy"]))
						if dd < dmin: dmin = dd
					bias = max(0.0, 1.2 - dmin / rl_diag)
			elif tg["owner"] >= 2 and tg["owner"] != owner:
				bias = 1.5
			elif tg["owner"] == PLAYER:
				bias = (3.0 if st > tt else -4.0) if endgame else -6.0
			var score := raw + bias
			if score > best_score:
				best_score = score; best_src = si; best_tgt = ti
	if best_src >= 0 and best_tgt >= 0:
		_send(best_src, best_tgt)

# simple fallback used only if the policy fails to load
func _heuristic_move(owner: int) -> void:
	var mine: Array = []
	for i in range(states.size()):
		if states[i]["owner"] == owner: mine.append(i)
	if mine.is_empty(): return
	var src = mine[0]
	for i in mine:
		if states[i]["troops"] > states[src]["troops"]: src = i
	if states[src]["troops"] < 12.0: return
	var best := -1
	var best_score := -1e18
	for ti in range(states.size()):
		if states[ti]["owner"] == owner: continue
		var d: float = Vector2(states[src]["cx"], states[src]["cy"]).distance_to(Vector2(states[ti]["cx"], states[ti]["cy"]))
		var score: float = -d * 0.02 - states[ti]["troops"] * 1.1
		if states[ti]["owner"] == NEUTRAL: score += 8.0
		elif states[ti]["owner"] >= 2: score += 14.0
		if states[src]["troops"] > states[ti]["troops"]: score += 25.0
		if score > best_score: best_score = score; best = ti
	if best >= 0: _send(src, best)

# ---------- win / lose ----------
func _check_win() -> void:
	if not running: return
	var alive: Dictionary = {}
	for s in states:
		if s["owner"] != NEUTRAL: alive[s["owner"]] = true
	for o in orbs:
		if o["owner"] != NEUTRAL: alive[o["owner"]] = true
	if not alive.has(PLAYER):
		running = false
		result_text = "DEFEAT"
	elif alive.size() == 1 and alive.has(PLAYER):
		var all_owned := true
		for s in states:
			if s["owner"] != PLAYER: all_owned = false
		if all_owned:
			running = false
			result_text = "VICTORY!"

# ---------- input ----------
func _input(event: InputEvent) -> void:
	var pt := Vector2.INF
	if event is InputEventScreenTouch and event.pressed:
		pt = event.position
	elif event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		pt = event.position
	if pt == Vector2.INF: return
	if not running:
		_new_game()
		return
	var hit := _state_at(pt)
	if hit >= 0 and states[hit]["owner"] == PLAYER:
		selected = hit
	elif selected >= 0:
		if hit >= 0: _send(selected, hit)
		selected = -1
	queue_redraw()

func _state_at(pt: Vector2) -> int:
	for i in range(states.size()):
		for poly in states[i]["scr"]:
			if poly.size() > 0 and Geometry2D.is_point_in_polygon(pt, poly[0]):
				return i
	return -1

# ---------- rendering ----------
func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, get_viewport_rect().size), Color("#e3e9ee"))
	for s in states:
		var fill := Color(COLORS[s["owner"]])
		for poly in s["scr"]:
			if poly.size() == 0: continue
			draw_colored_polygon(poly[0], fill)
			var closed := PackedVector2Array(poly[0])
			closed.append(poly[0][0])
			draw_polyline(closed, Color(1, 1, 1, 0.7), 1.5, true)
	for s in states:
		_draw_node(s)
	for o in orbs:
		var col := Color(COLORS[o["owner"]])
		var dcol := Color(DARK[o["owner"]])
		var r: float = clamp(7.0 + sqrt(o["count"]), 7.0, 17.0) * (mark_r / 17.0)
		draw_circle(o["pos"], r + 2.0, dcol)
		draw_circle(o["pos"], r, col)
		if o["count"] > 1.0:
			_text(str(int(o["count"])), o["pos"], int(r), Color("#ffffff"))
	if selected >= 0:
		draw_arc(Vector2(states[selected]["cx"], states[selected]["cy"]), mark_r + 6.0, 0.0, TAU, 40, Color("#2b3440"), 3.0)
	if not running and result_text != "":
		var col := Color("#38c46f") if result_text == "VICTORY!" else Color("#ef5b52")
		_text(result_text, get_viewport_rect().size * 0.5, 46, col)
		_text("tap to play again", get_viewport_rect().size * 0.5 + Vector2(0, 46), 18, Color("#3a4654"))

func _draw_node(s: Dictionary) -> void:
	var c := Vector2(s["cx"], s["cy"])
	var r: float = mark_r + s["pulse"] * 2.0
	draw_circle(c, r, Color(1, 1, 1))
	var ring := Color("#cfd4da") if s["owner"] == NEUTRAL else Color(DARK[s["owner"]])
	draw_arc(c, r, 0.0, TAU, 40, ring, 3.0)
	_text(str(int(round(s["troops"]))), c, int(mark_r * 0.85), Color("#2b3440"))

func _text(t: String, center: Vector2, size: int, col: Color) -> void:
	if font == null: return
	var ts := font.get_string_size(t, HORIZONTAL_ALIGNMENT_LEFT, -1, size)
	draw_string(font, Vector2(center.x - ts.x * 0.5, center.y + size * 0.35), t, HORIZONTAL_ALIGNMENT_LEFT, -1, size, col)
