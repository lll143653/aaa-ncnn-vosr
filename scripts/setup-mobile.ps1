param([switch]$WithConversion, [string]$Proxy = '')
. "$PSScriptRoot\env.ps1"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$QnnSdk = "$WorkspaceRoot\tools\qairt\2.44.0.260225"
$NcnnSdk = "$WorkspaceRoot\tools\ncnn-20260526-android-vulkan"
$Download = "$WorkspaceRoot\downloads\ncnn-20260526-android-vulkan.zip"
New-Item -ItemType Directory -Force "$WorkspaceRoot\downloads","$ProjectRoot\third_party","$ProjectRoot\evidence" | Out-Null
if (!(Test-Path -LiteralPath "$WorkspaceRoot\tools\android-sdk\ndk\30.0.16248370\toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe")) {
    if ($Proxy) { $env:HTTPS_PROXY=$Proxy }
    & "$WorkspaceRoot\tools\android-sdk\cmdline-tools\bootstrap\cmdline-tools\bin\sdkmanager.bat" "--sdk_root=$WorkspaceRoot\tools\android-sdk" 'ndk/30.0.16248370'
    if ($LASTEXITCODE -ne 0) { throw 'NDK r30 安装失败' }
}
function Fetch-File([string]$Url,[string]$Destination) {
    $CurlArguments=@('-L','--fail','--retry','2','--silent','--show-error',$Url,'-o',$Destination)
    if ($Proxy) { $CurlArguments=@('--proxy',$Proxy)+$CurlArguments }
    & curl.exe @CurlArguments
    if ($LASTEXITCODE -ne 0) { throw "下载失败：$Url" }
}
if (!(Test-Path -LiteralPath "$NcnnSdk\arm64-v8a\lib\libncnn.a")) {
    if (!(Test-Path -LiteralPath $Download)) {
        Fetch-File 'https://github.com/Tencent/ncnn/releases/download/20260526/ncnn-20260526-android-vulkan.zip' $Download
    }
    if ((Get-FileHash -LiteralPath $Download -Algorithm SHA256).Hash -ne '26909C92EED35AFED4A966B5E9E503FCB0A529691EA3F910EC2C94A4FFF52804') { throw 'ncnn SDK SHA-256 不匹配' }
    Expand-Archive -LiteralPath $Download -DestinationPath "$WorkspaceRoot\tools" -Force
}
$NativeRoot="$ProjectRoot\app\android\app\src\main\jniLibs\arm64-v8a"
New-Item -ItemType Directory -Force $NativeRoot | Out-Null
foreach ($Library in @('libQnnHtp.so','libQnnHtpV73Stub.so','libQnnSystem.so')) {
    Copy-Item -LiteralPath "$QnnSdk\lib\aarch64-android\$Library" -Destination "$NativeRoot\$Library" -Force
}
Copy-Item -LiteralPath "$QnnSdk\lib\hexagon-v73\unsigned\libQnnHtpV73Skel.so" -Destination "$NativeRoot\libQnnHtpV73Skel.so" -Force
foreach ($Library in @('libQnnHtpV81Skel.so','libQnnHtpV81.so')) {
    $Source=Join-Path $QnnSdk "lib\hexagon-v81\unsigned\$Library"
    if (!(Test-Path -LiteralPath $Source)) { throw "缺少 v81 运行库：$Source；请按阶段 15 文档补齐 SDK" }
    Copy-Item -LiteralPath $Source -Destination (Join-Path $NativeRoot $Library) -Force
}
Copy-Item -LiteralPath "$QnnSdk\lib\aarch64-android\libQnnHtpV81Stub.so" -Destination "$NativeRoot\libQnnHtpV81Stub.so" -Force
$FlutterPath="$WorkspaceRoot\tools\flutter".Replace('\','\\')
$AndroidPath="$WorkspaceRoot\tools\android-sdk".Replace('\','\\')
"flutter.sdk=$FlutterPath`nsdk.dir=$AndroidPath`n" | Set-Content -LiteralPath "$ProjectRoot\app\android\local.properties" -Encoding utf8
Push-Location "$ProjectRoot\app"
try {
    & "$WorkspaceRoot\tools\flutter\bin\flutter.bat" pub get
    if ($LASTEXITCODE -ne 0) { throw 'Flutter 依赖安装失败' }
} finally { Pop-Location }
if ($WithConversion) {
    if (!(Test-Path -LiteralPath "$ProjectRoot\.venv\Scripts\python.exe")) {
        & "$WorkspaceRoot\tools\uv\uv.exe" venv --python 3.11 "$ProjectRoot\.venv"
        if ($LASTEXITCODE -ne 0) { throw 'Python 环境创建失败' }
    }
    & "$WorkspaceRoot\tools\uv\uv.exe" pip install --python "$ProjectRoot\.venv\Scripts\python.exe" --index-strategy unsafe-best-match -r "$ProjectRoot\conversion\requirements-mobile.txt"
    if ($LASTEXITCODE -ne 0) { throw '转换依赖安装失败' }
    if (!(Test-Path -LiteralPath "$ProjectRoot\third_party\VOSR-main\models\lightningdit.py")) {
        $SourceZip="$WorkspaceRoot\downloads\vosr-516f292.zip"
        Fetch-File 'https://github.com/cswry/VOSR/archive/516f292b99cf23c76fdc33351e86dc4f97711fe8.zip' $SourceZip
        Expand-Archive -LiteralPath $SourceZip -DestinationPath "$ProjectRoot\third_party" -Force
        Move-Item -LiteralPath "$ProjectRoot\third_party\VOSR-516f292b99cf23c76fdc33351e86dc4f97711fe8" -Destination "$ProjectRoot\third_party\VOSR-main"
    }
    & "$ProjectRoot\.venv\Scripts\python.exe" "$PSScriptRoot\download-mobile-assets.py" "$ProjectRoot\conversion\mobile-model-downloads.json"
    if ($LASTEXITCODE -ne 0) { throw '官方模型下载或校验失败' }
}
Write-Output '移动端依赖已就绪；运行 scripts/build-android.ps1 构建双后端 APK。'
