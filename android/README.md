# State.io — native shell (rebuilt from source)

The original Android Studio project was lost, so the shipping pipeline (`build/repack.py`) could only
ever inherit the compiled `classes.dex` from the previous signed APK — it never rebuilt native code.
This project **recreates that shell from source** so the native side is editable again, and adds
over-the-air updates.

It reproduces the original shell's behaviour exactly:
- A single fullscreen `WebView` that loads the game from the virtual host `https://stateio.local/web/`.
- Manifest parity: package `io.state.app`, `MainActivity`, `INTERNET` + `ACCESS_NETWORK_STATE`,
  `appCategory="game"` (Samsung Game Launcher), `screenOrientation="fullUser"`, display-cutout
  `shortEdges` (retires `patch_arsc_cutout.py`), immersive fullscreen.
- Signed with the same `build/test.keystore` (alias `testkey`), so it **installs over the existing
  testkey builds without an uninstall** and the tester's `localStorage` (coins, perks, learned
  opponent model — same `stateio.local` origin) is preserved.

…and adds what a WebView can't do alone:
- **Web-asset OTA (the frictionless path).** On launch the shell checks `web-version.json` on GitHub,
  downloads any newer game files into internal storage, and reloads into them. No reinstall, no
  install screen, no Play Protect prompt. `OtaThenAssetHandler` serves the overlaid file if present,
  else the bundled asset.
- **APK self-update (Path A).** `window.AndroidHost.installApk(url)` downloads a same-key-signed APK
  and hands it to the system installer — for when the native shell or manifest itself must change.

## Build & sign (the one step this repo can't do headless)

Requires the Android SDK (Android Studio, or `sdkmanager` command line). From `android/`:

```sh
# Android Studio: File > Open > this android/ folder, let it sync, then Build > Build APK(s).
# Command line:
./gradlew assembleRelease
```

Output: `app/build/outputs/apk/release/app-release.apk`, already signed with `testkey` (the signing
config is in `app/build.gradle.kts`). Verify the signer matches the shipping line before installing:

```sh
java -cp ../build/apksig.jar ../build/SignApk.java verify app/build/outputs/apk/release/app-release.apk
# expect: verified=true v2=true v3=true   signer SHA256: 68:06:03:...:5E:91
```

Install over the existing app:

```sh
adb install -r app/build/outputs/apk/release/app-release.apk
```

`versionCode` is `250` in `app/build.gradle.kts` — must stay **greater than the installed app's**
code or Android rejects the update. Bump it every release.

## Publishing web updates (the OTA channel)

1. Create a `release` branch — this is the "published" channel so work-in-progress on `main` never
   reaches testers.
2. Put `web-version.json` (in this repo root) on that branch and keep `src/web/` current there.
3. To ship a new game (net retrain, tier tweak, bug fix): commit the new `src/web/*` to `release`
   and bump `"web"` in `web-version.json` by one. On next launch every installed shell pulls the
   changed files and reloads into them — no new APK, no install screen.

Only rebuild + reinstall the APK when you change the **native shell or the manifest** (new
permission, new bridge method, orientation, etc.). Everything that lives in `src/web/` — the game,
the nets, the tiers — ships via the OTA channel.

## Notes / not-yet-done

- **This project has not been compiled** in the environment that generated it (no Android SDK there).
  Open it in Android Studio, let Gradle sync (it may offer to bump AGP/Kotlin — accept), fix any
  version nits it flags, build, and **device-test on real phones** before it goes near a tester.
- The launcher icon (`res/drawable/ic_launcher.xml`) is a placeholder — regenerate from
  `src/web/logo.svg` via Image Asset Studio.
- Gradle wrapper: Android Studio will generate `gradle/wrapper/gradle-wrapper.jar` on first sync.
- In-game update UI: to drive the explicit "New Update Available!" alert + spinner, add a small
  script to `src/web/index.html` that calls `window.AndroidHost.updateWebAndReload()` (web OTA) or
  `window.AndroidHost.installApk(url)` (native APK). The shell already updates web assets silently on
  launch without it.
