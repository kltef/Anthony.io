package io.state.app

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

// Over-the-air update logic. Two channels:
//   * WEB OTA (default, frictionless): download newer game files (index.html / policy JSONs) into
//     internal storage; OtaThenAssetHandler serves them over the shipped ones after a WebView reload.
//     No reinstall, no permission, no install screen — every `git push` to the release branch + a
//     version bump reaches the tester on next launch.
//   * APK install (NativeHost.installApk): for when the NATIVE shell or manifest must change — hands
//     a downloaded, same-key-signed APK to the system installer (Path A).
object UpdateManager {

    // The "published" channel — a dedicated branch so work-in-progress on main never reaches testers.
    // web-version.json (on that branch):
    //   { "web": 1,
    //     "base": "https://raw.githubusercontent.com/kltef/Anthony.io/release/src/web/",
    //     "files": ["index.html", "gnn_policy.json", "rl_policy.json"] }
    // Bump "web" by one whenever you publish new web assets.
    private const val VERSION_URL =
        "https://raw.githubusercontent.com/kltef/Anthony.io/release/web-version.json"
    private const val PREFS = "stateio_ota"
    private const val KEY_WEB_VER = "web_ver"
    private const val TIMEOUT_MS = 10_000

    fun checkWebUpdateAsync(ctx: Context, done: (Boolean) -> Unit) {
        Thread {
            val updated = runCatching { checkWebUpdate(ctx) }.getOrDefault(false)
            done(updated)
        }.start()
    }

    private fun checkWebUpdate(ctx: Context): Boolean {
        val manifest = JSONObject(httpGet(VERSION_URL))
        val remoteVer = manifest.getInt("web")
        val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (remoteVer <= prefs.getInt(KEY_WEB_VER, 0)) return false

        val base = manifest.getString("base")
        val files = manifest.getJSONArray("files")
        val webDir = File(ctx.filesDir, "web").apply { mkdirs() }
        val tmp = File(ctx.filesDir, "web_tmp").apply { deleteRecursively(); mkdirs() }

        // Download everything to a temp dir first; only swap into place if ALL files succeed, so a
        // dropped connection never leaves a half-updated (broken) game.
        for (i in 0 until files.length()) {
            val name = files.getString(i)
            download(base + name, File(tmp, name))
        }
        tmp.listFiles()?.forEach { it.copyTo(File(webDir, it.name), overwrite = true) }
        tmp.deleteRecursively()
        prefs.edit().putInt(KEY_WEB_VER, remoteVer).apply()
        return true
    }

    private fun httpGet(url: String): String {
        val c = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = TIMEOUT_MS; readTimeout = TIMEOUT_MS; instanceFollowRedirects = true
        }
        try {
            return c.inputStream.bufferedReader().use { it.readText() }
        } finally { c.disconnect() }
    }

    // Public: also used by NativeHost.installApk to fetch the APK.
    fun download(url: String, dest: File) {
        dest.parentFile?.mkdirs()
        val c = (URL(url).openConnection() as HttpURLConnection).apply {
            connectTimeout = TIMEOUT_MS; readTimeout = TIMEOUT_MS; instanceFollowRedirects = true
        }
        try {
            c.inputStream.use { input -> dest.outputStream().use { input.copyTo(it) } }
        } finally { c.disconnect() }
    }
}
