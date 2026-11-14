param([string]$NcnnVersion='20260526')
. "$PSScriptRoot\env.ps1"
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Toolchain="$WorkspaceRoot\tools\android-sdk\ndk\30.0.16248370\toolchains\llvm\prebuilt\windows-x86_64"
$Ncnn="$WorkspaceRoot\tools\ncnn-$NcnnVersion-android-vulkan\arm64-v8a"
New-Item -ItemType Directory -Force "$ProjectRoot\artifacts" | Out-Null
& "$Toolchain\bin\clang++.exe" '--target=aarch64-none-linux-android29' "--sysroot=$Toolchain\sysroot" '-std=c++17' '-O3' '-fPIE' '-pie' '-static-libstdc++' '-fopenmp' '-static-openmp' "-I$Ncnn\include\ncnn" "$ProjectRoot\cpp\tools\probe_gpu.cpp" "-L$Ncnn\lib" '-Wl,--start-group' '-lncnn' '-lglslang' '-lSPIRV' '-lMachineIndependent' '-lOSDependent' '-lGenericCodeGen' '-lglslang-default-resource-limits' '-Wl,--end-group' '-llog' '-ldl' '-landroid' '-o' "$ProjectRoot\artifacts\probe-gpu"
if ($LASTEXITCODE -ne 0) { throw 'GPU probe build failed' }
