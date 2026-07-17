# UPDATING.md — Over-the-air vs. shell rebuild

Which changes reach players instantly over-the-air (OTA), and which need a new APK
("shell rebuild") they have to install. Read this **before planning a feature** — it
often decides how you build it.

---

## The one rule that decides everything

> **If it lives inside the WebView (HTML / CSS / JS / JSON / media), it's OTA.**
> **If it needs the phone to do something the web page can't do by itself — a permission,
> the OS, the app's identity, or native code — it needs a shell rebuild.**

The whole game is `src/web/index.html` + its assets, served to a WebView. Anything you
can change by editing those files ships over-the-air. Everything the *native Android
shell* provides (the window, permissions, the icon, the JS↔native bridge, the updater
itself) is frozen into the installed APK until you build and install a new one.

---

## The two channels at a glance

| | **OTA (web update)** | **Shell rebuild (new APK)** |
|---|---|---|
| What it updates | `src/web/*` — the game | The native app around the game |
| Player action | Tap "Update now" banner, ~2s, no reinstall | Install a new APK |
| How you ship it | `python build/bump_web.py "note"` → push `release` | Android Studio → build → install, same signing key |
| Reaches | Anyone already on an OTA-capable shell (build 3+) | Everyone, but it's a real install |
| Version bumped | `web` in `web-version.json` + `WEB_VERSION` | `versionCode` in `build.gradle.kts` |
| Delivered from | GitHub `release` branch via raw URLs | You hand out / host the `.apk` |

---

## ✅ Ships over-the-air (just edit `src/web/…` and bump)

- **All game logic & balance** — rules, troop sim, caps, tiles, timings, tiers.
- **The AI** — the policy nets are JSON (`gnn_policy.json`, `rl_policy.json`); retrain and
  drop in a new file. (Listed in the manifest `files[]`.)
- **Maps & scenarios**, new game modes, difficulty changes.
- **All UI/UX** — layouts, menus, colors, fonts, CSS, animations, screens.
- **Music & sound** — `.mp3`s (see build 5). Bundled *and* OTA-deliverable.
- **Server-backed leaderboards** — fetching `/api/leaderboard` and rendering it is pure web.
- **Multiplayer** — the WebRTC/PeerJS + server-API netcode is all web.
- **In-app notifications** — banners/toasts *while the app is open* (like the update card).
- **Text, tutorials, help, changelog**, the update-banner UI itself.
- **Local data features** — coins, perks, settings, saved state (localStorage).
- **Analytics / telemetry** via `fetch`.
- **Bug fixes** to any of the above.

## 🔧 Needs a shell rebuild (native)

- **System push notifications** — alerts when the app is **closed** (daily reminder, "your
  turn" in async MP). Needs FCM + a notification channel + the `POST_NOTIFICATIONS`
  permission + background handling. Web Push is not reliable in Android WebView.
- **The app icon, name, or splash** — these are Android resources/manifest.
- **Any new Android permission** — camera, location, notifications, storage, vibrate, etc.
  Permissions live in `AndroidManifest.xml`.
- **New native capabilities exposed to JS** — anything added to the `AndroidHost` bridge
  (see the hybrid pattern below).
- **The updater itself** — how OTA files are fetched/swapped (`UpdateManager`), the manifest
  URL/branch, the tap-to-install-APK flow.
- **WebView configuration** — fullscreen, orientation, display-cutout handling, JS/storage
  settings, foreground/background lifecycle (`onPause`/`onResume`).
- **targetSdk / minSdk / dependency bumps** — Play Store requirements, AndroidX versions.
- **The signing key or package name** — must never change (updates only install over the
  same key + `applicationId`).
- **Google Play Games Services** — *native* leaderboards/achievements with the Play overlay,
  Play Billing (in-app purchases), and anything using a Google/Play native SDK.
- **True OS integration** the web sandbox forbids — background execution when closed,
  Bluetooth/NFC/sensors beyond web APIs, NDK performance, deep links / intent filters,
  home-screen widgets, reading arbitrary files.

## 🔁 The hybrid pattern — build the hook once, tune forever over-the-air

The smart move for anything native: **add a small, generic bridge method once (shell
rebuild), then drive all the logic from JS (OTA).** The native side becomes a dumb pipe;
the behavior stays updatable.

Examples:
- Add `AndroidHost.notify(title, body, whenMs)` once → then *what* the notifications say and
  *when* they fire is web logic you can tweak over-the-air forever.
- Add `AndroidHost.vibrate(ms)` once → then every haptic decision is in the game JS.
- The existing `AndroidHost.installApk()` / `updateWebAndReload()` are exactly this — thin
  native hooks the web layer calls.

So when a feature needs native, ask: *"What's the smallest native primitive that unlocks it,
so the rest lives in the web layer?"* Ship that primitive in the next shell, and you rarely
need another rebuild.

---

## Feature cheat-sheet

| Feature | Channel | Notes |
|---|---|---|
| New maps / modes / balance | **OTA** | Pure web. |
| Retrained AI net | **OTA** | Swap the policy JSON; keep it in `files[]`. |
| New / better music, sound FX | **OTA** | `.mp3` files, flat in `src/web/`. |
| UI redesign, new screens | **OTA** | HTML/CSS/JS. |
| Leaderboard (custom server) | **OTA** | `fetch` an API + render. Already wired. |
| Leaderboard (Google Play Games) | **Shell** | Native Play SDK + overlay. |
| In-app notice while playing | **OTA** | Banner/toast in the page. |
| Push notification (app closed) | **Shell**, then OTA | FCM + permission + channel once; content/scheduling then OTA via a bridge. |
| "Come back tomorrow" reminder | **Shell**, then OTA | Local scheduled notification needs native + permission first. |
| Haptics / vibration | **Shell**, then OTA | `VIBRATE` permission + `AndroidHost.vibrate()` once. |
| Share a result / screenshot | **OTA** | The Web Share API works in WebView (already used by `shareFile`). |
| In-app purchases | **Shell** | Play Billing is native-only. |
| App icon / name change | **Shell** | Android resources. |
| New permission (camera, location…) | **Shell** | `AndroidManifest.xml`. |
| Deep links, widgets, background service | **Shell** | Native OS integration. |
| Fix the OTA sub-folder limitation | **Shell** | It's in `UpdateManager` (see gotcha). |

---

## Current concrete constraints (don't get bitten)

1. **OTA files must be listed** in `web-version.json` → `files[]`. If the game references a
   file that isn't listed, OTA won't deliver it (it only exists if bundled in the APK).
2. **OTA files must be flat** (top-level in `src/web/`), *not* in sub-folders. The installed
   shell's `UpdateManager` swap copies only top-level files by name — a `src/web/audio/…`
   path would break the update. Fixing this to recurse is a **one-time shell change**; until
   then, keep new assets flat (that's why the `.mp3`s live directly in `src/web/`).
3. **Binary files are fine** over OTA (the downloader is byte-safe — mp3 works). New file
   types just need a MIME entry in `OtaThenAssetHandler` (native) for correctness, though
   Web-Audio `fetch`+`decode` ignores MIME.
4. **OTA serves from the `release` branch** via GitHub raw. `main` is your work-in-progress;
   `release` is what reaches testers. (Currently kept identical — `git branch -f release main`.)
5. **Never change the signing key or `applicationId`** — updates only install over the same
   one. Sign every APK with the testkey (`build/test.keystore`, out of the repo).
6. **Bump `versionCode`** on every shell rebuild so it installs over the old app.
7. **A new bridge method only exists once the shell shipping it is installed.** OTA JS can
   only call `AndroidHost` methods the *installed* shell already has — so plan the bridge
   ahead of the web features that will use it.

---

## How to actually ship each

**OTA (web) release**
```sh
python build/bump_web.py "What changed, one bullet" "Another bullet"
git push origin main && git branch -f release main && git push origin release
# testers get the tap-to-apply banner on next launch
```

**Shell rebuild (new APK)** — see `BUILD.md` for the authoritative steps. In short:
bump `versionCode` in `android/app/build.gradle.kts`, build the release variant in Android
Studio (or the repack path), **sign with the testkey**, install over the existing app.
Do this only when the change is in the 🔧 list above.
