plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "io.state.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.state.app"      // MUST match the installed app so updates install in place
        minSdk = 24
        targetSdk = 34
        versionCode = 250                    // MUST be > the installed app's versionCode (v2.4.x line)
        versionName = "2.5"
    }

    // Sign with the SAME throwaway test key the repack pipeline uses, so the rebuilt shell installs
    // over the existing testkey builds without an uninstall (see BUILD.md — same-key rule).
    signingConfigs {
        create("release") {
            storeFile = file("../../build/test.keystore")
            storePassword = "android"
            keyAlias = "testkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            isDebuggable = false             // retires the debuggable=true flag that tripped Play Protect
            isCrunchPngs = false             // the launcher PNGs came from a built APK (already optimized);
                                             // skip aapt2's re-crunch, which rejects them ("failed to compile")
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.webkit:webkit:1.10.0")   // WebViewAssetLoader (the stateio.local virtual host)
}

// The game is a single source of truth in ../../src/web. Copy it into the APK's assets at build
// time (excluding the *.bak / *.pre-* working files), mirroring repack.py's philosophy — so a plain
// `gradlew assembleRelease` always bundles the current game. The OTA path can then overlay newer
// web files at runtime without a rebuild.
val syncWebAssets by tasks.registering(Copy::class) {
    from("../../src/web") {
        include("index.html", "*.json", "*.js", "*.svg")
        exclude("*.bak", "*.bak.json", "*.bak.xml", "*.pre-*")
    }
    into("src/main/assets/web")
}
tasks.named("preBuild") { dependsOn(syncWebAssets) }
