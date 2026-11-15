param(
    [string]$Serial = $env:ANDROID_SERIAL,
    [ValidateSet('npu','gpu','both')][string]$Backend = 'both',
    [ValidateSet(512,1024)][int]$TileSize = 1024,
    [switch]$SkipModels,
    [switch]$RunSmokeTest
)
. "$PSScriptRoot\env.ps1"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$AdbExe = 'E:\work\ai\tools\android-sdk\platform-tools\adb.exe'
$DeviceArgs = if ($Serial) { @('-s', $Serial) } else { @() }
$Package = 'ai.vosr.vosr_restore'
$ApkFile = Join-Path $ProjectRoot 'deliverables\vosr-restore-debug.apk'
if (!(Test-Path -LiteralPath $ApkFile)) { throw "APK 不存在：$ApkFile" }
& $AdbExe @DeviceArgs get-state
if ($LASTEXITCODE -ne 0) { throw '请连接并授权目标 Android 手机' }
& $AdbExe @DeviceArgs install --no-streaming -r $ApkFile
if ($LASTEXITCODE -ne 0) { throw 'APK 安装失败，请查看手机是否要求允许 USB 安装' }
& $AdbExe @DeviceArgs shell am start -n "$Package/.MainActivity"
if (!$SkipModels) {
    $SelectedBackends = if ($Backend -eq 'both') { @('npu','gpu') } else { @($Backend) }
    foreach ($SelectedBackend in $SelectedBackends) {
    $ModelFolder = if ($SelectedBackend -eq 'npu') { 'qnn' } else { 'ncnn-0.5b' }
    if ($TileSize -eq 1024) { $ModelFolder += '-1024' }
    $ModelRoot = Join-Path $ProjectRoot "artifacts\$ModelFolder"
    $ManifestPath = Join-Path $ModelRoot 'manifest.json'
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if ($Manifest.tileSize -ne $TileSize) { throw '模型输出尺寸不匹配' }
    if (!$Manifest.deviceValidated) { throw "$SelectedBackend 模型未通过真机验证" }
    $DeviceFolder=if ($SelectedBackend -eq 'npu') { 'models' } else { 'models-gpu' }
    if ($TileSize -eq 1024) { $DeviceFolder += '-1024' }
    $DeviceRoot = "/sdcard/Android/data/$Package/files/$DeviceFolder"
    $DirectoryReady = $false
    for ($Attempt = 0; $Attempt -lt 40; $Attempt++) {
        & $AdbExe @DeviceArgs shell test -d $DeviceRoot
        if ($LASTEXITCODE -eq 0) { $DirectoryReady = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (!$DirectoryReady) { throw '应用未创建模型目录，请检查启动日志；不要用 shell mkdir 创建该目录' }
    foreach ($Stage in $Manifest.stages) {
        $ModelFile = Join-Path $ModelRoot $Stage.file
        $Hash = (Get-FileHash -LiteralPath $ModelFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Hash -ne $Stage.sha256) { throw "模型校验失败：$ModelFile" }
        & $AdbExe @DeviceArgs push $ModelFile "$DeviceRoot/$($Stage.file)"
        if ($LASTEXITCODE -ne 0) { throw "模型复制失败：$ModelFile" }
        if ($SelectedBackend -eq 'gpu') {
            $ParamFile=Join-Path $ModelRoot $Stage.param
            if ((Get-FileHash -LiteralPath $ParamFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Stage.paramSha256) { throw "GPU 参数校验失败：$ParamFile" }
            & $AdbExe @DeviceArgs push $ParamFile "$DeviceRoot/$($Stage.param)"
            if ($LASTEXITCODE -ne 0) { throw "GPU 参数复制失败：$ParamFile" }
        }
    }
    & $AdbExe @DeviceArgs push $ManifestPath "$DeviceRoot/manifest.json"
    if ($LASTEXITCODE -ne 0) { throw '模型清单复制失败' }
    }
}
if ($RunSmokeTest) {
    $InputFile = Join-Path $ProjectRoot 'tests\fixtures\mobile\native-514x129.png'
    $RemoteInput = "/sdcard/Android/data/$Package/files/test-input.png"
    & $AdbExe @DeviceArgs push $InputFile $RemoteInput
    & $AdbExe @DeviceArgs shell am force-stop $Package
    $TestBackend=if ($Backend -eq 'both') { 'npu' } else { $Backend }
    & $AdbExe @DeviceArgs shell am start -n "$Package/.MainActivity" --es vosr_test_input $RemoteInput --ei vosr_test_scale 2 --es vosr_test_backend $TestBackend --ez vosr_test_photo $(if($TileSize -eq 1024){'true'}else{'false'})
    Write-Output '应用内复原测试已启动，结果写入手机应用文件目录的 app-test-result.json。'
}
