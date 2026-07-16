package io.state.app

import android.app.Activity
import android.content.Intent
import android.webkit.JavascriptInterface
import androidx.core.content.FileProvider
import java.io.File

// JS↔native bridge, exposed to the game as `window.AndroidHost`. The current index.html does not call
// it, so it is inert until the game opts in — add a small in-game "New Update Available" UI that calls
// these to drive the flow described in the design (alert → Update → spinner → for APK, the system
// install screen). All methods are safe no-ops on failure.
class NativeHost(
    private val activity: Activity,
    private val reload: () -> Unit
) {
    @JavascriptInterface
    fun nativeVersion(): String =
        activity.packageManager.getPackageInfo(activity.packageName, 0).versionName ?: "?"

    // WEB OTA on demand (the frictionless path): check + download newer web assets, then reload.
    // The game can show a spinner between the call and the reload.
    @JavascriptInterface
    fun updateWebAndReload() {
        UpdateManager.checkWebUpdateAsync(activity) { updated -> if (updated) reload() }
    }

    // APK self-update (Path A) — for when the native shell/manifest itself changes. Downloads the
    // APK, then hands it to the system installer via a content:// URI (the "install screen slides up"
    // step). First time only, Android sends the user to grant "install unknown apps" for this app;
    // after that it goes straight to the install confirmation. The APK MUST be signed with the same
    // key (testkey) and carry a higher versionCode, or the update is rejected.
    @JavascriptInterface
    fun installApk(url: String) {
        Thread {
            runCatching {
                val apk = File(activity.cacheDir, "update.apk")
                UpdateManager.download(url, apk)
                val uri = FileProvider.getUriForFile(activity, "${activity.packageName}.fileprovider", apk)
                val intent = Intent(Intent.ACTION_VIEW).apply {
                    setDataAndType(uri, "application/vnd.android.package-archive")
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
                }
                activity.startActivity(intent)
            }
        }.start()
    }
}
