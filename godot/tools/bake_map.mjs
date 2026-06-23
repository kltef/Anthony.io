// One-time pre-bake: TopoJSON (lon/lat) -> Albers-USA screen polygons, so the Godot
// port needs no D3/TopoJSON at runtime. Output coords are in d3.geoAlbersUsa() default
// space (~960x500); Godot re-fits the active states' bounding box to the viewport.
import { readFileSync, writeFileSync } from 'node:fs';
import { feature } from 'topojson-client';
import { geoAlbersUsa, geoPath } from 'd3-geo';

const topo = JSON.parse(readFileSync('us-states.json','utf8'));
const fc = feature(topo, topo.objects.states);
const proj = geoAlbersUsa();
const path = geoPath(proj);
const R = n => Math.round(n*100)/100;

// same exclusions as the web game (continental US only)
const EXCLUDE = new Set(['Alaska','Hawaii','District of Columbia','Puerto Rico',
  'American Samoa','Guam','Commonwealth of the Northern Mariana Islands','United States Virgin Islands']);
// region lists mirror REGIONS in index.html (null = all of the lower 48)
const REGIONS = [
  { name:'East Coast', list:['Maine','New Hampshire','Vermont','Massachusetts','Rhode Island','Connecticut','New York','New Jersey','Pennsylvania','Delaware','Maryland','West Virginia','Virginia','North Carolina','South Carolina','Georgia','Florida','Ohio','Kentucky','Tennessee','Alabama','Indiana'] },
  { name:'Midwest', list:['Ohio','Michigan','Indiana','Illinois','Wisconsin','Minnesota','Iowa','Missouri','Kansas','Nebraska','South Dakota','North Dakota','Oklahoma','Kentucky'] },
  { name:'West Coast', list:['California','Oregon','Washington','Nevada','Idaho','Arizona','Utah','Montana','Wyoming','Colorado','New Mexico'] },
  { name:'Continental U.S.', list:null },
];

function ringsOf(geom){
  const polys = geom.type==='Polygon' ? [geom.coordinates] : geom.coordinates;
  const out=[];
  for (const poly of polys){
    const rings=[];
    for (const ring of poly){
      const pts=[];
      for (const c of ring){ const p=proj(c); if(p) pts.push([R(p[0]),R(p[1])]); }
      if (pts.length>2) rings.push(pts);
    }
    if (rings.length) out.push(rings);
  }
  return out;
}

const states=[];
for (const f of fc.features){
  const name=f.properties.name;
  if (EXCLUDE.has(name)) continue;
  const c=path.centroid(f);
  if (!c || isNaN(c[0])) continue;
  states.push({ name, cx:R(c[0]), cy:R(c[1]), polys: ringsOf(f.geometry) });
}
states.sort((a,b)=>a.name.localeCompare(b.name));

const data={ projection:'geoAlbersUsa-default', count:states.length,
  regions: REGIONS.map(r=>({name:r.name,list:r.list})), states };
writeFileSync('../data/us_map.json', JSON.stringify(data));

// sanity report
let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9,verts=0;
for(const s of states) for(const p of s.polys) for(const r of p) for(const [x,y] of r){verts++;minx=Math.min(minx,x);miny=Math.min(miny,y);maxx=Math.max(maxx,x);maxy=Math.max(maxy,y);}
console.log('states baked:', states.length);
console.log('bbox: x',R(minx),'..',R(maxx),' y',R(miny),'..',R(maxy));
console.log('total vertices:', verts);
console.log('sample:', states[0].name, 'centroid', states[0].cx, states[0].cy, 'polys', states[0].polys.length);
