package io.state.app

import android.annotation.SuppressLint
import android.app.Activity
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.view.WindowManager
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.webkit.WebViewAssetLoader
import java.io.File

// The whole game is src/web/index.html (+ the policy JSONs and the runtime libs), served here over
// the same virtual host the original shell used: https://stateio.local/web/. This rebuild reproduces
// that shell from source (the original Android project was lost) and adds two things the WebView
// could never do on its own: (1) a writable overlay so newer web assets can be delivered over-the-air
// and take effect with a reload — no reinstall, no Play Protect, no install screen; and (2) a JS
// bridge that can hand a downloaded APK to the system installer when the NATIVE shell itself must
// change. See README.md for the build/sign/OTA-hosting steps.
class MainActivity : Activity() {

    private lateinit var web: WebView

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Draw into the display cutout; the game's CSS env(safe-area-inset-*) rules position the
        // top UI below the notch. This replaces the binary resources.arsc patch (patch_arsc_cutout.py).
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.attributes = window.attributes.apply {
                layoutInDisplayCutoutMode =
                    WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
            }
        }
        WindowCompat.setDecorFitsSystemWindows(window, false)

        val loader = WebViewAssetLoader.Builder()
            .setDomain("stateio.local")
            .addPathHandler("/web/", OtaThenAssetHandler(this, File(filesDir, "web")))
            .build()

        web = WebView(this)
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true          // localStorage: coins, perks, the learned opponent model
            databaseEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            mediaPlaybackRequiresUserGesture = false
            cacheMode = WebSettings.LOAD_NO_CACHE
        }
        web.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView, request: WebResourceRequest
            ): WebResourceResponse? = loader.shouldInterceptRequest(request.url)
        }
        web.addJavascriptInterface(NativeHost(this) { runOnUiThread { web.reload() } }, "AndroidHost")
        setContentView(web)

        // Immersive fullscreen (transient bars on swipe).
        WindowInsetsControllerCompat(window, web).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }

        web.loadUrl("https://stateio.local/web/index.html")

        // Updates are TAP-TO-APPLY, not silent. The in-game banner (index.html initUpdates) checks
        // the version manifest, shows a "New update available" card with the changelog, and only
        // downloads + reloads when the user taps Update (via AndroidHost.updateWebAndReload). So
        // there is deliberately no auto-download/reload on launch here.
    }

    // Foreground/background: pausing the WebView stops it burning CPU off-screen AND makes the JS
    // Page Visibility API accurate, so the game's visibilitychange handler (which suspends the music)
    // fires reliably. The rAF-driven sim already halts when hidden.
    override fun onPause() {
        super.onPause()
        web.onPause()
    }

    override fun onResume() {
        super.onResume()
        web.onResume()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (web.canGoBack()) web.goBack() else moveTaskToBack(true)
    }
}

// Serves /web/<path>: an OTA-delivered file in internal storage wins; otherwise the bundled asset at
// assets/web/<path>. Returning null falls through to the next matching handler / a 404, so a missing
// OTA file cleanly reverts to the shipped version.
class OtaThenAssetHandler(
    private val context: Context,
    private val otaDir: File
) : WebViewAssetLoader.PathHandler {

    override fun handle(path: String): WebResourceResponse? {
        val clean = path.trimStart('/')
        val overlay = File(otaDir, clean)
        if (overlay.exists() && overlay.isFile) {
            return runCatching {
                WebResourceResponse(mime(clean), enc(clean), overlay.inputStream())
            }.getOrNull()
        }
        return runCatching {
            WebResourceResponse(mime(clean), enc(clean), context.assets.open("web/$clean"))
        }.getOrNull()
    }

    private fun mime(n: String) = when {
        n.endsWith(".html") -> "text/html"
        n.endsWith(".js") -> "text/javascript"
        n.endsWith(".json") -> "application/json"
        n.endsWith(".mp3") -> "audio/mpeg"
        n.endsWith(".svg") -> "image/svg+xml"
        n.endsWith(".css") -> "text/css"
        else -> "application/octet-stream"
    }
    private fun enc(n: String) =
        if (n.endsWith(".html") || n.endsWith(".js") || n.endsWith(".json") ||
            n.endsWith(".css") || n.endsWith(".svg")) "utf-8" else null
}
