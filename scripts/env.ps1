$ErrorActionPreference = 'Stop'
$WorkspaceRoot = 'E:\work\ai'
$env:UV_PYTHON_INSTALL_DIR = "$WorkspaceRoot\tools\python"
$env:UV_PYTHON_BIN_DIR = "$WorkspaceRoot\tools\python-bin"
$env:UV_CACHE_DIR = "$WorkspaceRoot\cache\uv"
$env:PIP_CACHE_DIR = "$WorkspaceRoot\cache\pip"
$env:HF_HOME = "$WorkspaceRoot\cache\huggingface"
$env:TORCH_HOME = "$WorkspaceRoot\models\vosr\torch_cache"
$env:PUB_CACHE = "$WorkspaceRoot\cache\pub"
$env:GRADLE_USER_HOME = "$WorkspaceRoot\cache\gradle"
$env:ANDROID_USER_HOME = "$WorkspaceRoot\cache\android"
$env:APPDATA = "$WorkspaceRoot\cache\appdata"
$env:FLUTTER_SUPPRESS_ANALYTICS = 'true'
$env:DART_SUPPRESS_ANALYTICS = 'true'
$env:ANDROID_HOME = "$WorkspaceRoot\tools\android-sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
$env:JAVA_HOME = "$WorkspaceRoot\tools\jdk-17.0.21+9"
$env:TEMP = "$WorkspaceRoot\cache\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP,$env:APPDATA | Out-Null
$env:Path = "$env:JAVA_HOME\bin;$WorkspaceRoot\tools\flutter\bin;$WorkspaceRoot\tools\uv;$env:ANDROID_HOME\platform-tools;$env:Path"
# Scoped to this toolchain process; do not change the user's global Git settings.
$GitConfigOffset = if ($env:GIT_CONFIG_COUNT) { [int]$env:GIT_CONFIG_COUNT } else { 0 }
$TrustedRepositories = @("$WorkspaceRoot/tools/flutter", (Split-Path $PSScriptRoot -Parent).Replace('\','/'))
foreach ($TrustedRepository in $TrustedRepositories) {
    [Environment]::SetEnvironmentVariable("GIT_CONFIG_KEY_$GitConfigOffset", 'safe.directory', 'Process')
    [Environment]::SetEnvironmentVariable("GIT_CONFIG_VALUE_$GitConfigOffset", $TrustedRepository, 'Process')
    $GitConfigOffset++
}
$env:GIT_CONFIG_COUNT = [string]$GitConfigOffset
