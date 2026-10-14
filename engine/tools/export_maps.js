#!/usr/bin/env node
// Export the game's real maps (projected centroids + border adjacency) to engine/data/maps.txt.
// Uses the same d3.geoAlbersUsa projection and fitExtent call as buildGeo() in src/web/index.html,
// at a reference landscape viewport, so distances have the same proportions as on a phone.
//   usage: node engine/tools/export_maps.js
const fs = require('fs'), path = require('path');
const web = path.join(__dirname, '..', '..', 'src', 'web');
const d3 = require(path.join(web, 'd3.min.js'));
const topojson = require(path.join(web, 'topojson-client.min.js'));
const topo = JSON.parse(fs.readFileSync(path.join(web, 'us-states.json'), 'utf8'));
const allGeoms = topo.objects.states.geometries;
const allFeatures = topojson.feature(topo, topo.objects.states).features;
const W = 900, H = 420;   // reference viewport (CSS px, landscape phone)

const EXCLUDE = new Set(['Alaska','Hawaii','District of Columbia','Puerto Rico','American Samoa','Guam',
  'Commonwealth of the Northern Mariana Islands','United States Virgin Islands']);
const REGIONS = [
  { name:'East Coast', list:['Maine','New Hampshire','Vermont','Massachusetts','Rhode Island','Connecticut',
    'New York','New Jersey','Pennsylvania','Delaware','Maryland','West Virginia','Virginia','North Carolina',
    'South Carolina','Georgia','Florida','Ohio','Kentucky','Tennessee','Alabama','Indiana'] },
  { name:'Midwest', list:['Ohio','Michigan','Indiana','Illinois','Wisconsin','Minnesota','Iowa','Missouri',
    'Kansas','Nebraska','South Dakota','North Dakota','Oklahoma','Kentucky'] },
  { name:'West Coast', list:['California','Oregon','Washington','Nevada','Idaho','Arizona','Utah','Montana',
    'Wyoming','Colorado','New Mexico'] },
  { name:'Continental U.S.', list:null },
];
let out = '';
for (const r of REGIONS){
  const set = r.list ? new Set(r.list) : null;
  const idxs = allFeatures.map((f,i)=>i).filter(i => { const nm = allFeatures[i].properties.name; return set ? set.has(nm) : !EXCLUDE.has(nm); });
  const feats = idxs.map(i => allFeatures[i]);
  const nbr = topojson.neighbors(idxs.map(i => allGeoms[i]));
  const projection = d3.geoAlbersUsa();
  projection.fitExtent([[12,70],[W-12,H-16]], { type:'FeatureCollection', features:feats });
  const gp = d3.geoPath(projection);
  out += 'map|' + r.name + '|' + idxs.length + '\n';
  feats.forEach((f,i) => { const c = gp.centroid(f);
    out += [f.properties.name, c[0].toFixed(3), c[1].toFixed(3), [...new Set(nbr[i])].sort((a,b)=>a-b).join(',')].join('|') + '\n'; });
}
fs.writeFileSync(path.join(__dirname, '..', 'data', 'maps.txt'), out);
console.log('wrote maps.txt', out.split('\n').filter(l=>l.startsWith('map|')).join('  '));
