. "$PSScriptRoot\env.ps1"
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$NdkRoot = 'E:\work\ai\tools\android-sdk\ndk\28.2.13676358'
$Compiler = "$NdkRoot\toolchains\llvm\prebuilt\windows-x86_64\bin\clang++.exe"
$Sysroot = "$NdkRoot\toolchains\llvm\prebuilt\windows-x86_64\sysroot"
$Output = "$ProjectRoot\app\android\app\src\main\jniLibs\arm64-v8a\libvosr_jni.so"
& $Compiler '--target=aarch64-none-linux-android29' "--sysroot=$Sysroot" '-std=c++17' '-O3' '-fPIC' '-shared' '-fvisibility=hidden' '-Wl,-z,max-page-size=16384' '-Wl,--no-undefined' '-static-libstdc++' '-IE:\work\ai\tools\qairt\2.44.0.260225\include\QNN' "$ProjectRoot\app\android\app\src\main\cpp\vosr_jni.cpp" '-llog' '-ldl' '-landroid' '-o' $Output
if ($LASTEXITCODE -ne 0) { throw 'JNI native build failed' }
Write-Output "JNI: $Output"
$NcnnSdk = "$WorkspaceRoot\tools\ncnn-20260526-android-vulkan\arm64-v8a"
if (!(Test-Path -LiteralPath "$NcnnSdk\lib\libncnn.a")) { throw '请先运行 scripts/setup-mobile.ps1 安装 ncnn Android Vulkan SDK' }
$GpuOutput = "$ProjectRoot\app\android\app\src\main\jniLibs\arm64-v8a\libvosr_gpu.so"
$GpuToolchain = "$WorkspaceRoot\tools\android-sdk\ndk\30.0.16248370\toolchains\llvm\prebuilt\windows-x86_64"
if (!(Test-Path -LiteralPath "$GpuToolchain\bin\clang++.exe")) { throw 'ncnn 官方库需要 NDK r30；请先运行 scripts/setup-mobile.ps1' }
& "$GpuToolchain\bin\clang++.exe" '--target=aarch64-none-linux-android29' "--sysroot=$GpuToolchain\sysroot" '-std=c++17' '-O3' '-fPIC' '-shared' '-fvisibility=hidden' '-Wl,-z,max-page-size=16384' '-Wl,--no-undefined' '-static-libstdc++' '-fopenmp' '-static-openmp' "-I$NcnnSdk\include\ncnn" "$ProjectRoot\app\android\app\src\main\cpp\ncnn_jni.cpp" '-fno-rtti' "-L$NcnnSdk\lib" '-Wl,--start-group' '-lncnn' '-lglslang' '-lSPIRV' '-lMachineIndependent' '-lOSDependent' '-lGenericCodeGen' '-lglslang-default-resource-limits' '-Wl,--end-group' '-llog' '-ldl' '-landroid' '-o' $GpuOutput
if ($LASTEXITCODE -ne 0) { throw 'Vulkan JNI native build failed' }
Write-Output "GPU JNI: $GpuOutput"
