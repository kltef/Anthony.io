#!/usr/bin/env node
// Extract the REAL game maps (adjacency + normalized centroids) from src/web/us-states.json into
// tools/real_maps.json, so the arena can train/gate on the boards the shipped game actually plays
// on — the synthetic 16-node Delaunay boards are both smaller (real regions run 11-48 states) and
// topologically blander than the TopoJSON-derived maps (see the 2026-07-12 "why can't it go
// higher" analysis: the small synthetic arena compresses skill differences the 3-hop net could
// exploit on real maps). Region lists mirror index.html's REGIONS exactly.
//   usage: node tools/extract_maps.js   ->  tools/real_maps.json
const fs = require('fs');
const path = require('path');
const topojson = require(path.join(__dirname, '..', 'src', 'web', 'topojson-client.min.js'));

const topo = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'src', 'web', 'us-states.json'), 'utf8'));
const geoms = topo.objects.states.geometries;
const feats = topojson.feature(topo, topo.objects.states).features;
const nbrs = topojson.neighbors(geoms);

const names = geoms.map(g => g.properties.name);
// rough planar centroid: average of all ring vertices (consistent, and the nets only ever see
// rlDiag-normalized distances, so projection niceties don't matter for training)
function centroid(f){
  let sx = 0, sy = 0, n = 0;
  const polys = f.geometry.type === 'Polygon' ? [f.geometry.coordinates] : f.geometry.coordinates;
  for (const poly of polys) for (const ring of poly) for (const [x, y] of ring){ sx += x; sy += y; n++; }
  return [sx / n, sy / n];
}
const cents = feats.map(centroid);

const REGIONS = [
  { name: 'East Coast', list: [
    'Maine','New Hampshire','Vermont','Massachusetts','Rhode Island','Connecticut',
    'New York','New Jersey','Pennsylvania','Delaware','Maryland','West Virginia',
    'Virginia','North Carolina','South Carolina','Georgia','Florida',
    'Ohio','Kentucky','Tennessee','Alabama','Indiana' ] },
  { name: 'Midwest', list: [
    'Ohio','Michigan','Indiana','Illinois','Wisconsin','Minnesota','Iowa','Missouri',
    'Kansas','Nebraska','South Dakota','North Dakota','Oklahoma','Kentucky' ] },
  { name: 'West Coast', list: [
    'California','Oregon','Washington','Nevada','Idaho','Arizona','Utah',
    'Montana','Wyoming','Colorado','New Mexico' ] },
  // Continental U.S. = lower 48 + DC (exclude Alaska/Hawaii/territories, matching the game view)
  { name: 'Continental U.S.', list: null },
];
const EXCLUDE = new Set(['Alaska', 'Hawaii', 'Puerto Rico', 'United States Virgin Islands',
  'Guam', 'American Samoa', 'Commonwealth of the Northern Mariana Islands']);

const maps = [];
for (const reg of REGIONS){
  const wanted = reg.list ? new Set(reg.list) : null;
  const idxs = [];
  for (let i = 0; i < names.length; i++){
    if (EXCLUDE.has(names[i])) continue;
    if (wanted && !wanted.has(names[i])) continue;
    idxs.push(i);
  }
  const remap = new Map(idxs.map((gi, li) => [gi, li]));
  // adjacency restricted to the region; drop nodes isolated by the cut (e.g. a region member whose
  // only neighbors are outside the region would be unreachable — none exist in these lists, but
  // stay defensive), then normalize centroids into 0..1 preserving aspect via the larger span.
  const adj = idxs.map(gi => nbrs[gi].filter(j => remap.has(j)).map(j => remap.get(j)));
  const xs = idxs.map(gi => cents[gi][0]), ys = idxs.map(gi => cents[gi][1]);
  const mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  const span = Math.max(mxx - mnx, mxy - mny) || 1;
  maps.push({
    name: reg.name,
    n: idxs.length,
    names: idxs.map(gi => names[gi]),
    cx: xs.map(x => (x - mnx) / span),
    cy: ys.map(y => (mxy - y) / span),   // flip so y grows downward like the canvas
    adj,
  });
  const degs = adj.map(a => a.length);
  console.log(`${reg.name}: ${idxs.length} states, degree min ${Math.min(...degs)} max ${Math.max(...degs)}, isolated: ${degs.filter(d => d === 0).length}`);
}
fs.writeFileSync(path.join(__dirname, 'real_maps.json'), JSON.stringify(maps));
console.log('wrote tools/real_maps.json');
