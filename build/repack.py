#!/usr/bin/env python3
# Rebuild the shipping APK from the previous signed APK by swapping in the current
# web assets (index.html, rl_policy.json) and the patched binary AndroidManifest.xml
# (which carries android:appCategory="game" so Samsung Game Launcher / Game Booster
# auto-detect it as a game). Run from the repo root, then sign with apksigner.
#   python3 build/repack.py  ->  build/unsigned.apk
import zipfile, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREV = os.path.join(ROOT, "State.io_v1.6.apk")   # previous signed APK (carries all assets/res)
OUT  = os.path.join(ROOT, "build", "unsigned.apk")
html     = open(os.path.join(ROOT, "src/web/index.html"), "rb").read()
policy   = open(os.path.join(ROOT, "src/web/rl_policy.json"), "rb").read()
manifest = open(os.path.join(ROOT, "build", "AndroidManifest.xml"), "rb").read()  # patched AXML
zin = zipfile.ZipFile(PREV, "r"); zout = zipfile.ZipFile(OUT, "w")
swapped = {"html": False, "policy": False, "manifest": False}
for it in zin.infolist():
    name = it.filename
    if name.startswith("META-INF/") and name.split("/")[-1].upper().rsplit(".",1)[-1] in ("SF","RSA","DSA","EC","MF"):
        continue  # drop old signature
    data = zin.read(name)
    if name == "assets/web/index.html":      data = html;     swapped["html"]=True
    elif name == "assets/web/rl_policy.json": data = policy;   swapped["policy"]=True
    elif name == "AndroidManifest.xml":       data = manifest; swapped["manifest"]=True
    # preserve original per-entry compression (resources.arsc must stay STORED)
    zi = zipfile.ZipInfo(name, date_time=it.date_time)
    zi.compress_type=it.compress_type; zi.external_attr=it.external_attr
    zi.internal_attr=it.internal_attr; zi.create_system=it.create_system
    zout.writestr(zi, data)
zin.close(); zout.close()
print("swapped:", swapped, "size:", os.path.getsize(OUT))
assert all(swapped.values()), "FAILED to swap an asset!"
