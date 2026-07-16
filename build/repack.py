#!/usr/bin/env python3
# Rebuild the shipping APK from the previous signed APK by swapping in the current
# web assets (index.html, rl_policy.json) and the patched binary AndroidManifest.xml
# (which carries android:appCategory="game" so Samsung Game Launcher / Game Booster
# auto-detect it as a game). Run from the repo root, then sign with apksigner.
#   python3 build/repack.py  ->  build/unsigned.apk
import zipfile, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREV = os.path.join(ROOT, "State.io_v2.1-nightmare-testkey.apk")   # previous signed APK (carries all assets/res)
OUT  = os.path.join(ROOT, "build", "unsigned.apk")
html     = open(os.path.join(ROOT, "src/web/index.html"), "rb").read()
policy   = open(os.path.join(ROOT, "src/web/rl_policy.json"), "rb").read()
gnn      = open(os.path.join(ROOT, "src/web/gnn_policy.json"), "rb").read()  # board-aware net (Impossible)
manifest = open(os.path.join(ROOT, "build", "AndroidManifest.xml"), "rb").read()  # patched AXML
# optional patched resource table (build/patch_arsc_cutout.py writes it — adds the display-cutout
# opt-in to the fullscreen theme so cutout phones stop letterboxing the top of the screen)
ARSC = os.path.join(ROOT, "build", "resources.arsc")
arsc = open(ARSC, "rb").read() if os.path.exists(ARSC) else None
zin = zipfile.ZipFile(PREV, "r"); zout = zipfile.ZipFile(OUT, "w")
swapped = {"html": False, "policy": False, "gnn": False, "manifest": False}
for it in zin.infolist():
    name = it.filename
    if name.startswith("META-INF/") and name.split("/")[-1].upper().rsplit(".",1)[-1] in ("SF","RSA","DSA","EC","MF"):
        continue  # drop old signature
    data = zin.read(name)
    if name == "assets/web/index.html":       data = html;     swapped["html"]=True
    elif name == "assets/web/rl_policy.json":  data = policy;   swapped["policy"]=True
    elif name == "assets/web/gnn_policy.json": data = gnn;      swapped["gnn"]=True
    elif name == "AndroidManifest.xml":        data = manifest; swapped["manifest"]=True
    elif name == "resources.arsc" and arsc is not None: data = arsc   # stays STORED via compress_type copy
    # preserve original per-entry compression (resources.arsc must stay STORED)
    zi = zipfile.ZipInfo(name, date_time=it.date_time)
    zi.compress_type=it.compress_type; zi.external_attr=it.external_attr
    zi.internal_attr=it.internal_attr; zi.create_system=it.create_system
    zout.writestr(zi, data)
# gnn_policy.json is a NEW asset (not in older base APKs) — add it if the base didn't already carry it.
if not swapped["gnn"]:
    zi = zipfile.ZipInfo("assets/web/gnn_policy.json", date_time=(1980,1,1,0,0,0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    zout.writestr(zi, gnn); swapped["gnn"]=True
zin.close(); zout.close()
print("swapped:", swapped, "size:", os.path.getsize(OUT))
assert all(swapped.values()), "FAILED to swap an asset!"
