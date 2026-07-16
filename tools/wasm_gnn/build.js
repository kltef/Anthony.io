#!/usr/bin/env node
// Build the GNN WASM kernels (SIMD + scalar fallback) from assembly/gnn.ts.
//
//   node tools/wasm_gnn/build.js
//
// Requires assemblyscript (npm install --no-save assemblyscript). Emits:
//   tools/wasm_gnn/gnn_simd.wasm    (f32x4 dot products; needs WASM SIMD — Chrome 91+/Node 16+)
//   tools/wasm_gnn/gnn_scalar.wasm  (no post-MVP features; universal fallback)
// plus .wat disassembly next to each for review.

const { execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const dir = __dirname;
const src = path.join(dir, 'assembly', 'gnn.ts');
const ascBin = require.resolve('assemblyscript/bin/asc.js', { paths: [path.join(dir, '..', '..')] });

function build(out, extra) {
  const args = [
    ascBin, src,
    '-O3', '--noAssert',
    '--runtime', 'stub',    // no GC; memory is exported by default in asc 0.28
    '-o', path.join(dir, out),
    '-t', path.join(dir, out.replace(/\.wasm$/, '.wat')),
    ...extra,
  ];
  execFileSync(process.execPath, args, { stdio: 'inherit' });
  const bytes = fs.statSync(path.join(dir, out)).size;
  console.log(`${out}: ${bytes} bytes (base64 ~${Math.ceil(bytes / 3) * 4} chars)`);
}

build('gnn_simd.wasm', ['--enable', 'simd']);
build('gnn_scalar.wasm', []);
