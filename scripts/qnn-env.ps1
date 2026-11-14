. "$PSScriptRoot\env.ps1"
$env:QNN_SDK_ROOT = 'E:\work\ai\tools\qairt\2.44.0.260225'
$env:QAIRT_SDK_ROOT = $env:QNN_SDK_ROOT
$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python"
$env:Path = "$env:QNN_SDK_ROOT\lib\x86_64-windows-msvc;$env:QNN_SDK_ROOT\bin\x86_64-windows-msvc;$env:Path"
