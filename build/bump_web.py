#!/usr/bin/env python3
"""Bump the shipped web build in one shot — keeps the two lockstep places in sync.

Every web release must update BOTH of these, or the OTA banner misbehaves:
  * web-version.json    -> "web" +1, "notes" replaced, a new entry prepended to "history"
  * src/web/index.html  -> const WEB_VERSION = N;

This script does both. Each positional argument is one changelog bullet (>=1 required).

Usage:
  python build/bump_web.py "First note" "Second note"        # bump, dated today
  python build/bump_web.py --date 2026-07-20 "A note"        # override the date
  python build/bump_web.py --dry-run "A note"                # preview, write nothing
  python build/bump_web.py --commit "A note"                 # also stage + commit (never pushes)

After it runs, publish with:
  git push origin main && git branch -f release main && git push origin release

LF line endings are forced on both files (.gitattributes requires them).
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "web-version.json")
INDEX = os.path.join(ROOT, "src", "web", "index.html")
WEB_VERSION_RE = re.compile(r"const WEB_VERSION = (\d+);")


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write(path, text):
    # LF endings are mandatory here (see .gitattributes) — force them on every platform.
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description="Bump the shipped web build (manifest + index.html).")
    ap.add_argument("notes", nargs="+", help="changelog bullet lines for this build (at least one)")
    ap.add_argument("--date", default=None, help="release date YYYY-MM-DD (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="show the change, write nothing")
    ap.add_argument("--commit", action="store_true", help="also `git add` + `git commit` (no push)")
    args = ap.parse_args()

    date = args.date or datetime.date.today().isoformat()

    manifest = json.loads(read(MANIFEST))
    cur = int(manifest.get("web", 0))
    new = cur + 1

    idx = read(INDEX)
    m = WEB_VERSION_RE.search(idx)
    if not m:
        sys.exit("ERROR: could not find `const WEB_VERSION = N;` in src/web/index.html")
    idx_ver = int(m.group(1))
    if idx_ver != cur:
        print("WARNING: index.html WEB_VERSION=%d but manifest web=%d were out of lockstep; "
              "setting both to %d." % (idx_ver, cur, new))

    entry = {"web": new, "date": date, "notes": list(args.notes)}
    manifest["web"] = new
    manifest["notes"] = list(args.notes)                 # keep flat notes = current (old shells read this)
    hist = manifest.get("history")
    manifest["history"] = ([] if not isinstance(hist, list) else hist)
    manifest["history"].insert(0, entry)                 # newest-first

    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    idx_new = WEB_VERSION_RE.sub("const WEB_VERSION = %d;" % new, idx, count=1)

    print("web build %d -> %d   (%s)" % (cur, new, date))
    for n in args.notes:
        print("  • " + n)

    if args.dry_run:
        print("\n(dry run - no files written)")
        return

    write(MANIFEST, manifest_text)
    write(INDEX, idx_new)
    print("\nUpdated web-version.json + src/web/index.html to web %d." % new)

    if args.commit:
        msg = "Web build %d\n\n" % new + "\n".join("- " + n for n in args.notes)
        subprocess.run(["git", "add", MANIFEST, INDEX], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        print("Committed. Publish with:")
    else:
        print("Next:")
    print("  git push origin main && git branch -f release main && git push origin release")


if __name__ == "__main__":
    main()
