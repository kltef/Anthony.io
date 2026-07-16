#!/usr/bin/env python3
"""
export_godot.py — translate the State.io web game's DATA into the Godot port's format.

Reads:
  - us-states.json      (TopoJSON; from src/web/ if present, else extracted from the newest APK)
  - src/web/index.html  (tuning tables: LEVELS, COLORS, DARK, TILE, PLAYER_CAP, REGIONS, EXCLUDE, …)
  - src/web/rl_policy.json, src/web/gnn_policy.json  (trained nets — copied verbatim; the JSON
    schema is the Python<->runtime contract and must not change)

Writes (all under godot/data/):
  - map.json     every state feature projected through a faithful pure-Python port of
                 d3.geoAlbersUsa (same constants as d3-geo), plus centroids and the
                 shared-arc border adjacency (identical pairs to topojson.neighbors)
  - config.json  the tuning tables extracted from index.html, so retuning the web game
                 and re-running this script keeps the Godot port in sync
  - rl_policy.json / gnn_policy.json  (copies)

Coordinates are projected once into a fixed 1600x1000 "world" box (fitExtent over the
lower 48, same margins as the game). Godot re-fits the active region's bounding box to
the window at runtime — fitExtent is only a uniform scale + translate, so this is
equivalent to the web game re-projecting per region.

Usage: python3 tools/export_godot.py
"""
import json, math, os, re, shutil, sys, zipfile, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_WEB = os.path.join(ROOT, 'src', 'web')
OUT_DIR = os.path.join(ROOT, 'godot', 'data')

WORLD_W, WORLD_H = 1600.0, 1000.0
# same screen margins as buildGeo(): fitExtent([[12,70],[W-12,H-16]], …)
FIT_EXTENT = [[12.0, 70.0], [WORLD_W - 12.0, WORLD_H - 16.0]]

# ---------------------------------------------------------------- TopoJSON ----
def load_topojson():
    p = os.path.join(SRC_WEB, 'us-states.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f), p
    apks = sorted(glob.glob(os.path.join(ROOT, 'State.io_v*.apk')), key=os.path.getmtime)
    if not apks:
        sys.exit('us-states.json not found in src/web/ and no APK to extract it from')
    apk = apks[-1]
    with zipfile.ZipFile(apk) as z:
        return json.loads(z.read('assets/web/us-states.json')), apk + '!assets/web/us-states.json'

def decode_arcs(topo):
    """Delta-decode quantized arcs -> list of [lon,lat] point lists."""
    sx, sy = topo['transform']['scale']
    tx, ty = topo['transform']['translate']
    out = []
    for arc in topo['arcs']:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx; y += dy
            pts.append((x * sx + tx, y * sy + ty))
        out.append(pts)
    return out

def geom_rings(geom):
    """Yield each ring of a (Multi)Polygon geometry as a list of arc indices."""
    if geom['type'] == 'Polygon':
        polys = [geom['arcs']]
    elif geom['type'] == 'MultiPolygon':
        polys = geom['arcs']
    else:
        polys = []
    for poly in polys:
        for ring in poly:
            yield ring

def geom_outer_rings(geom):
    """Yield only each polygon's OUTER ring (mirrors projectedPolys: poly[0])."""
    if geom['type'] == 'Polygon':
        yield geom['arcs'][0]
    elif geom['type'] == 'MultiPolygon':
        for poly in geom['arcs']:
            yield poly[0]

def ring_points(ring, arcs):
    """Stitch a ring's arcs into one closed point list (drop duplicated joins)."""
    pts = []
    for a in ring:
        seg = arcs[a] if a >= 0 else list(reversed(arcs[~a]))
        if pts and pts[-1] == seg[0]:
            seg = seg[1:]
        pts.extend(seg)
    return pts

def neighbors(geoms):
    """topojson.neighbors: two geometries border iff they share an arc (index, sign-folded)."""
    arc_owners = {}
    for gi, g in enumerate(geoms):
        for ring in geom_rings(g):
            for a in ring:
                arc_owners.setdefault(a if a >= 0 else ~a, set()).add(gi)
    adj = [set() for _ in geoms]
    for owners in arc_owners.values():
        for a in owners:
            for b in owners:
                if a != b:
                    adj[a].add(b)
    return [sorted(s) for s in adj]

# ------------------------------------------------- d3.geoAlbersUsa (faithful) ----
RAD = math.pi / 180
EPS = 1e-6

class ConicEqualArea:
    """d3 conicEqualArea with rotate [lambda0, 0], center, parallels, scale, translate."""
    def __init__(self, rotate_lon, center, parallels):
        self.rotate_lon = rotate_lon
        self.center = center
        y0, y1 = parallels[0] * RAD, parallels[1] * RAD
        sy0 = math.sin(y0)
        self.n = (sy0 + math.sin(y1)) / 2
        self.c = 1 + sy0 * (2 * self.n - sy0)
        self.r0 = math.sqrt(self.c) / self.n
        self.k = 150.0
        self.tx, self.ty = 0.0, 0.0
        self.clip = None          # [[x0,y0],[x1,y1]] or None
        self._recenter()

    def _raw(self, lam, phi):
        r = math.sqrt(max(0.0, self.c - 2 * self.n * math.sin(phi))) / self.n
        a = lam * self.n
        return r * math.sin(a), self.r0 - r * math.cos(a)

    def _recenter(self):
        cx, cy = self._raw(self.center[0] * RAD, self.center[1] * RAD)
        self.dx = self.tx - cx * self.k
        self.dy = self.ty + cy * self.k

    def set(self, k=None, translate=None, clip=None):
        if k is not None: self.k = k
        if translate is not None: self.tx, self.ty = translate
        if clip is not None: self.clip = clip
        self._recenter()

    def point(self, lon, lat):
        lam = (lon + self.rotate_lon) % 360
        if lam > 180: lam -= 360
        px, py = self._raw(lam * RAD, lat * RAD)
        x = px * self.k + self.dx
        y = self.dy - py * self.k
        if self.clip:
            (x0, y0), (x1, y1) = self.clip
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                return None
        return x, y

class AlbersUsa:
    """Composite of lower48 + Alaska + Hawaii, exactly as d3-geo wires it."""
    def __init__(self):
        self.lower48 = ConicEqualArea(96, (-0.6, 38.7), (29.5, 45.5))
        self.alaska  = ConicEqualArea(154, (-2, 58.5), (55, 65))
        self.hawaii  = ConicEqualArea(157, (-3, 19.9), (8, 18))
        self.scale_translate(1070, (480, 300))

    def scale_translate(self, k, t):
        x, y = t
        self.lower48.set(k=k, translate=t,
            clip=[[x - 0.455 * k, y - 0.238 * k], [x + 0.455 * k, y + 0.238 * k]])
        self.alaska.set(k=k * 0.35, translate=(x - 0.307 * k, y + 0.201 * k),
            clip=[[x - 0.425 * k + EPS, y + 0.120 * k + EPS], [x - 0.214 * k - EPS, y + 0.234 * k - EPS]])
        self.hawaii.set(k=k, translate=(x - 0.205 * k, y + 0.212 * k),
            clip=[[x - 0.214 * k + EPS, y + 0.166 * k + EPS], [x - 0.115 * k - EPS, y + 0.234 * k - EPS]])
        self.k, self.t = k, t

    def point(self, lon, lat):
        return (self.lower48.point(lon, lat)
                or self.alaska.point(lon, lat)
                or self.hawaii.point(lon, lat))

def fit_extent(proj, extent, all_rings):
    """d3 fitExtent: bounds at scale 150 / translate [0,0], then one similarity transform."""
    proj.scale_translate(150, (0, 0))
    minx = miny = math.inf; maxx = maxy = -math.inf
    for pts in all_rings:
        for lon, lat in pts:
            p = proj.point(lon, lat)
            if p is None: continue
            minx = min(minx, p[0]); maxx = max(maxx, p[0])
            miny = min(miny, p[1]); maxy = max(maxy, p[1])
    (x0, y0), (x1, y1) = extent
    k = min((x1 - x0) / (maxx - minx), (y1 - y0) / (maxy - miny))
    tx = x0 + (x1 - x0 - k * (maxx - minx)) / 2 - k * minx
    ty = y0 + (y1 - y0 - k * (maxy - miny)) / 2 - k * miny
    proj.scale_translate(150 * k, (tx, ty))

# --------------------------------------------- JS object-literal mini-parser ----
class JsParser:
    """Parse the flat-ish JS literals used by index.html's tuning tables:
    objects (identifier/number/string keys), arrays, strings, numbers, true/false/null."""
    def __init__(self, s, pos):
        self.s, self.i = s, pos

    def _ws(self):
        s = self.s
        while self.i < len(s):
            c = s[self.i]
            if c in ' \t\r\n,':
                self.i += 1
            elif s.startswith('//', self.i):
                self.i = s.index('\n', self.i)
            elif s.startswith('/*', self.i):
                self.i = s.index('*/', self.i) + 2
            else:
                break

    def value(self):
        self._ws()
        c = self.s[self.i]
        if c == '{': return self.obj()
        if c == '[': return self.arr()
        if c in '"\'': return self.string()
        m = re.match(r'true|false|null|[-+0-9.eE]+', self.s[self.i:])
        if not m: raise ValueError('bad literal at %d: %r' % (self.i, self.s[self.i:self.i+40]))
        tok = m.group(0); self.i += len(tok)
        if tok == 'true': return True
        if tok == 'false': return False
        if tok == 'null': return None
        f = float(tok)
        return int(f) if f.is_integer() and '.' not in tok and 'e' not in tok.lower() else f

    def string(self):
        q = self.s[self.i]; self.i += 1
        out = []
        while self.s[self.i] != q:
            c = self.s[self.i]
            if c == '\\':
                self.i += 1; c = self.s[self.i]
            out.append(c); self.i += 1
        self.i += 1
        return ''.join(out)

    def obj(self):
        self.i += 1  # {
        d = {}
        while True:
            self._ws()
            if self.s[self.i] == '}':
                self.i += 1; return d
            if self.s[self.i] in '"\'':
                key = self.string()
            else:
                m = re.match(r'[A-Za-z_$][A-Za-z0-9_$]*|\d+', self.s[self.i:])
                key = m.group(0); self.i += len(key)
            self._ws()
            assert self.s[self.i] == ':', 'expected : at %d' % self.i
            self.i += 1
            d[key] = self.value()

    def arr(self):
        self.i += 1  # [
        a = []
        while True:
            self._ws()
            if self.s[self.i] == ']':
                self.i += 1; return a
            a.append(self.value())

def extract_literal(html, decl):
    """Parse the JS literal that follows e.g. 'const LEVELS = '."""
    m = re.search(re.escape(decl), html)
    if not m:
        sys.exit('could not find %r in index.html' % decl)
    return JsParser(html, m.end()).value()

def extract_number(html, pattern, name):
    m = re.search(pattern, html)
    if not m:
        sys.exit('could not extract %s from index.html' % name)
    return float(m.group(1))

# ------------------------------------------------------------------- main ----
def main():
    topo, topo_src = load_topojson()
    with open(os.path.join(SRC_WEB, 'index.html'), 'r', encoding='utf-8') as f:
        html = f.read()

    # -- config tables (single source of truth stays index.html) --
    config = {
        'LEVELS':     extract_literal(html, 'const LEVELS = '),
        'COLORS':     extract_literal(html, 'const COLORS = '),
        'DARK':       extract_literal(html, 'const DARK   = '),
        'TILE':       extract_literal(html, 'const TILE = '),
        'PLAYER_CAP': extract_literal(html, 'const PLAYER_CAP = '),
        'REGIONS':    extract_literal(html, 'const REGIONS = '),
        'EXCLUDE':    extract_literal(html, 'const EXCLUDE = new Set('),
        'FRACTIONS':  extract_literal(html, 'const FRACTIONS = '),
        'MAX_TROOPS':  extract_number(html, r'const MAX_TROOPS = (\d+)', 'MAX_TROOPS'),
        'VALUE_BLEND': extract_number(html, r'let VALUE_BLEND = ([\d.]+)', 'VALUE_BLEND'),
        'REACT_DELAY': extract_number(html, r'const REACT_DELAY = ([\d.]+)', 'REACT_DELAY'),
        'MCTS_CPUCT':  extract_number(html, r'cPuct:([\d.]+)', 'MCTS cPuct'),
        'MAP_CROSS_SECONDS': extract_number(html, r'const MAP_CROSS_SECONDS = (\d+)', 'MAP_CROSS_SECONDS'),
    }
    assert isinstance(config['LEVELS'], list) and len(config['LEVELS']) >= 6, 'LEVELS parse failed'
    assert config['LEVELS'][-1]['name'] == 'Nightmare', 'unexpected LEVELS tail'

    # -- geometry --
    geoms = topo['objects']['states']['geometries']
    arcs = decode_arcs(topo)
    adj = neighbors(geoms)

    proj = AlbersUsa()
    all_outer = []
    for g in geoms:
        for ring in geom_outer_rings(g):
            all_outer.append(ring_points(ring, arcs))
    fit_extent(proj, FIT_EXTENT, all_outer)

    features = []
    for gi, g in enumerate(geoms):
        polys = []
        areas_cx = areas_cy = areas = 0.0
        for ring in geom_outer_rings(g):
            pts = [proj.point(lon, lat) for lon, lat in ring_points(ring, arcs)]
            pts = [p for p in pts if p is not None]
            if len(pts) < 3:
                continue
            flat = []
            for x, y in pts:
                flat.append(round(x, 2)); flat.append(round(y, 2))
            polys.append(flat)
            # signed-area centroid accumulation (planar; mirrors geoPath.centroid)
            a2 = cx = cy = 0.0
            for i in range(len(pts)):
                x0, y0 = pts[i - 1]; x1, y1 = pts[i]
                cross = x0 * y1 - x1 * y0
                a2 += cross; cx += (x0 + x1) * cross; cy += (y0 + y1) * cross
            if abs(a2) > 1e-9:
                areas += a2; areas_cx += cx; areas_cy += cy
        if areas != 0.0:
            ccx, ccy = areas_cx / (3 * areas), areas_cy / (3 * areas)
        elif polys:
            xs = polys[0][0::2]; ys = polys[0][1::2]
            ccx, ccy = sum(xs) / len(xs), sum(ys) / len(ys)
        else:
            ccx = ccy = 0.0
        features.append({
            'name': g['properties']['name'],
            'polys': polys,
            'cx': round(ccx, 2), 'cy': round(ccy, 2),
        })

    map_out = {
        'world_w': WORLD_W, 'world_h': WORLD_H,
        'features': features,
        'adjacency': adj,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'map.json'), 'w', encoding='utf-8') as f:
        json.dump(map_out, f, separators=(',', ':'))
    with open(os.path.join(OUT_DIR, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=1)
    for net in ('rl_policy.json', 'gnn_policy.json'):
        src = os.path.join(SRC_WEB, net)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(OUT_DIR, net))

    n_polys = sum(len(f['polys']) for f in features)
    print('map source   :', topo_src)
    print('features     : %d (%d outer rings)' % (len(features), n_polys))
    print('empty (off-map):', [f['name'] for f in features if not f['polys']])
    print('levels       :', [l['name'] for l in config['LEVELS']])
    print('wrote        :', OUT_DIR)

if __name__ == '__main__':
    main()
