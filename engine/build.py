#!/usr/bin/env python3
"""Build the engine to WebAssembly and inline it into src/web/index.html.

    python engine/build.py            # cargo build + inline
    python engine/build.py --check    # exit 1 if index.html does not carry the current build

The game stays one self-contained file: the .wasm is embedded as base64 on the line marked
`// ENGINE_WASM`. Needs the Rust toolchain with the wasm32-unknown-unknown target.
"""
import base64, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(ROOT, 'engine')
WASM = os.path.join(ENGINE, 'target', 'wasm32-unknown-unknown', 'release', 'stateio.wasm')
HTML = os.path.join(ROOT, 'src', 'web', 'index.html')
LINE = re.compile(r'const ENGINE_WASM_B64 = "[A-Za-z0-9+/=]*";(\s*// ENGINE_WASM)')


def main():
    env = dict(os.environ)
    env['PATH'] = os.path.join(os.path.expanduser('~'), '.cargo', 'bin') + os.pathsep + env.get('PATH', '')
    subprocess.run(['cargo', 'build', '--release', '--lib', '--target', 'wasm32-unknown-unknown'], cwd=ENGINE, env=env, check=True)
    b64 = base64.b64encode(open(WASM, 'rb').read()).decode('ascii')
    html = open(HTML, encoding='utf-8', newline='').read()
    if len(LINE.findall(html)) != 1:
        sys.exit('ENGINE_WASM marker line not found exactly once in index.html')
    new = LINE.sub(lambda m: 'const ENGINE_WASM_B64 = "' + b64 + '";' + m.group(1), html)
    if '--check' in sys.argv:
        sys.exit(0 if new == html else 'index.html does not carry the current engine build')
    if new != html:
        open(HTML, 'w', encoding='utf-8', newline='').write(new)   # newline='' keeps LF endings
    print('engine: %d KB wasm inlined into src/web/index.html' % (os.path.getsize(WASM) // 1024))


if __name__ == '__main__':
    main()
