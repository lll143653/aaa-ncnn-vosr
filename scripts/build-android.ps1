. "$PSScriptRoot\env.ps1"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
& "$PSScriptRoot\build-native.ps1"
Push-Location "$ProjectRoot\app\android"
try {
    & .\gradlew.bat --no-daemon --console=plain '-Pkotlin.compiler.execution.strategy=in-process' '-Dkotlin.compiler.execution.strategy=in-process' '-Ptarget-platform=android-arm64' '-Ptarget=lib/main.dart' '-Pflutter.androidSdkRoot=E:\work\ai\tools\android-sdk' '-Pflutter.installedNdkVersions=28.2.13676358' assembleDebug
    if ($LASTEXITCODE -ne 0) { throw 'Android APK 构建失败' }
    $Output = Join-Path $ProjectRoot 'deliverables\vosr-restore-debug.apk'
    New-Item -ItemType Directory -Force (Split-Path $Output -Parent) | Out-Null
    Copy-Item -LiteralPath "$ProjectRoot\app\build\app\outputs\apk\debug\app-debug.apk" -Destination $Output -Force
    Write-Output "APK: $Output"
} finally { Pop-Location }
