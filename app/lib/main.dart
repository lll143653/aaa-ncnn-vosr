import 'dart:async';
import 'dart:io';

import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

void main() => runApp(const VosrApp());

class VosrApp extends StatelessWidget {
  const VosrApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'VOSR 图像复原',
    theme: ThemeData(
      useMaterial3: true,
      colorScheme: ColorScheme.fromSeed(
        seedColor: const Color(0xff356555),
        surface: const Color(0xfff6f5ef),
      ),
      scaffoldBackgroundColor: const Color(0xfff6f5ef),
    ),
    home: const RestorePage(),
  );
}

class RestorePage extends StatefulWidget {
  const RestorePage({super.key});
  @override
  State<RestorePage> createState() => _RestorePageState();
}

class _RestorePageState extends State<RestorePage> with WidgetsBindingObserver {
  static const api = MethodChannel('ai.vosr/restore');
  static const events = EventChannel('ai.vosr/progress');
  StreamSubscription<dynamic>? subscription;
  String? input, output;
  String message = '选择一张需要复原的照片', device = '正在检查设备', backend = '尚未运行';
  bool busy = false, ready = false, restoring = false;
  double progress = 0, divider = .5;
  int inputWidth = 0, inputHeight = 0;
  String selectedBackend = 'npu';
  bool checking = false, supported = true;
  bool cameraMode = false, recoveringCamera = false;
  double? cameraZoom;
  int patchCount(int length) =>
      length <= 512 ? 1 : ((length - 512 + 415) ~/ 416) + 1;
  int get workingWidth =>
      (inputWidth * 832 / (inputWidth > inputHeight ? inputWidth : inputHeight))
          .round()
          .clamp(1, 832);
  int get workingHeight =>
      (inputHeight *
              832 /
              (inputWidth > inputHeight ? inputWidth : inputHeight))
          .round()
          .clamp(1, 832);

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    subscription = events.receiveBroadcastStream().listen((dynamic value) {
      if (!mounted) return;
      final data = Map<String, dynamic>.from(value as Map);
      if (data['cameraReady'] == true) {
        recoverCamera();
        return;
      }
      setState(() {
        progress = (data['progress'] as num?)?.toDouble() ?? 0;
        message = data['message'] as String? ?? message;
        if (data['input'] is String && data['output'] is String) {
          input = data['input'] as String;
          output = data['output'] as String;
          backend = data['backend'] as String? ?? backend;
          inputWidth = (data['inputWidth'] as num?)?.toInt() ?? inputWidth;
          inputHeight = (data['inputHeight'] as num?)?.toInt() ?? inputHeight;
          busy = false;
        }
      });
    });
    checkDevice().then((_) => recoverCamera());
  }

  Future<void> checkDevice() async {
    setState(() {
      checking = true;
      ready = false;
    });
    try {
      final data = await api.invokeMapMethod<String, dynamic>('diagnostics', {
        'backend': selectedBackend,
      });
      if (!mounted) return;
      setState(() {
        device = data?['device'] as String? ?? 'Android';
        ready = data?['ready'] == true;
        supported = data?['supported'] == true;
        message = data?['message'] as String? ?? '请先导入模型文件';
      });
    } on PlatformException catch (error) {
      showError(error);
    } finally {
      if (mounted) setState(() => checking = false);
    }
  }

  Future<void> pick() async {
    try {
      final image = await api.invokeMapMethod<String, dynamic>('pickImage');
      if (image == null || !mounted) return;
      setState(() {
        input = image['path'] as String;
        inputWidth = (image['width'] as num).toInt();
        inputHeight = (image['height'] as num).toInt();
        output = null;
        backend = '尚未运行';
        message = '照片已就绪';
        cameraMode = false;
        cameraZoom = null;
      });
    } on PlatformException catch (error) {
      showError(error);
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed && !busy && !checking) {
      recoverCamera();
    }
  }

  Future<void> acceptCamera(Map<String, dynamic> photo) async {
    if (!mounted) return;
    setState(() {
      input = photo['path'] as String;
      inputWidth = (photo['width'] as num).toInt();
      inputHeight = (photo['height'] as num).toInt();
      cameraZoom = (photo['zoomRatio'] as num).toDouble();
      cameraMode = true;
      output = null;
      busy = false;
      message = '拍照完成，正在复原';
    });
    if (ready) await restore();
  }

  Future<void> takePhoto() async {
    setState(() {
      busy = true;
      input = null;
      output = null;
      inputWidth = 0;
      inputHeight = 0;
      cameraMode = false;
      cameraZoom = null;
      message = '请在系统相机使用 ≥10× 拍摄';
    });
    try {
      final photo = await api.invokeMapMethod<String, dynamic>('takePhoto');
      if (photo != null && mounted) {
        await acceptCamera(photo);
      } else if (mounted) {
        setState(() => message = '已取消拍照');
      }
    } on PlatformException catch (error) {
      showError(error);
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  Future<void> recoverCamera() async {
    if (!mounted || busy || checking || recoveringCamera) return;
    recoveringCamera = true;
    try {
      final photo = await api.invokeMapMethod<String, dynamic>(
        'recoverCameraPhoto',
      );
      if (photo != null && mounted) await acceptCamera(photo);
    } on PlatformException catch (error) {
      showError(error);
    } finally {
      recoveringCamera = false;
    }
  }

  Future<void> importModels() async {
    setState(() {
      busy = true;
      message = '请选择模型 ZIP 文件';
      progress = 0;
    });
    try {
      final imported = await api.invokeMethod<bool>('importModels', {
        'backend': selectedBackend,
      });
      if (imported == true) {
        await checkDevice();
        if (mounted) setState(() => message = '模型已就绪，可以选择照片');
      }
    } on PlatformException catch (error) {
      showError(error);
    } finally {
      if (mounted) {
        setState(() {
          busy = false;
          restoring = false;
        });
      }
    }
  }

  Future<void> restore() async {
    if (input == null || busy || !ready) return;
    setState(() {
      busy = true;
      output = null;
      restoring = true;
      progress = 0;
      message = '正在准备复原';
    });
    try {
      final result = await api.invokeMapMethod<String, dynamic>(
        cameraMode ? 'restoreCamera' : 'restorePhoto',
        {'path': input, 'scale': 2, 'seed': 42, 'backend': selectedBackend},
      );
      if (!mounted) return;
      setState(() {
        output = result?['path'] as String?;
        backend = result?['backend'] as String? ?? '未确认';
        final seconds = ((result?['elapsedMs'] as num? ?? 0) / 1000)
            .toStringAsFixed(1);
        message = '复原完成 · $seconds 秒';
        progress = 1;
      });
    } on PlatformException catch (error) {
      showError(error);
    } finally {
      if (mounted) {
        setState(() {
          busy = false;
          restoring = false;
        });
      }
    }
  }

  void showError(PlatformException error) {
    if (!mounted) return;
    setState(() => message = error.message ?? '处理失败，请重试');
    ScaffoldMessenger.of(context)
        .showSnackBar(SnackBar(content: Text(message)));
  }

  Future<void> save() async {
    try {
      await api.invokeMethod<String>('saveImage', {'path': output});
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text('已保存到相册')));
      }
    } on PlatformException catch (error) {
      showError(error);
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    subscription?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(
      systemOverlayStyle: SystemUiOverlayStyle.dark,
      title: const Text(
        'VOSR',
        style: TextStyle(fontWeight: FontWeight.w800, letterSpacing: 2),
      ),
      backgroundColor: Colors.transparent,
      actions: [
        IconButton(
          onPressed: busy
              ? null
              : () {
                  showDialog<void>(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('关于图像复原'),
                      content: Text(
                        'VOSR 0.5B · 单步生成式复原\n$device\n执行后端：$backend\n\n照片在手机本地处理。模型可能生成原图中无法确认的细节。\n\n模型许可及组件说明随交付文档提供。',
                      ),
                      actions: [
                        TextButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('知道了'),
                        ),
                      ],
                    ),
                  );
                },
          icon: const Icon(Icons.info_outline),
        ),
      ],
    ),
    body: SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(24, 8, 24, 24),
        children: [
          const Text(
            '让模糊的瞬间\n重新清晰',
            style: TextStyle(
              fontSize: 34,
              height: 1.2,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 12),
          Text(
            '端侧图像复原  /  照片留在你的手机',
            style: TextStyle(color: Colors.grey.shade700),
          ),
          const SizedBox(height: 24),
          ClipRRect(
            borderRadius: BorderRadius.circular(24),
            child: AspectRatio(
              aspectRatio: 1,
              child: Container(
                color: const Color(0xffe7e9df),
                child: input == null
                    ? InkWell(
                        onTap: busy ? null : pick,
                        child: const Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.add_photo_alternate_outlined,
                              size: 52,
                              color: Color(0xff356555),
                            ),
                            SizedBox(height: 18),
                            Text(
                              '选择照片',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            SizedBox(height: 8),
                            Text('支持 JPG、PNG 和 WebP'),
                          ],
                        ),
                      )
                    : LayoutBuilder(
                        builder: (context, bounds) => Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.file(
                              File(input!),
                              fit: BoxFit.contain,
                              cacheWidth: 1024,
                            ),
                            if (output != null)
                              ClipRect(
                                clipper: _ComparisonClipper(divider),
                                child: Image.file(
                                  File(output!),
                                  fit: BoxFit.contain,
                                  cacheWidth: 1024,
                                ),
                              ),
                            if (output != null)
                              Positioned(
                                left: bounds.maxWidth * divider - 1,
                                top: 0,
                                bottom: 0,
                                child: Container(width: 2, color: Colors.white),
                              ),
                            Positioned(
                              left: 12,
                              bottom: 12,
                              child: _tag(output == null ? '原图' : '复原'),
                            ),
                            if (output != null)
                              Positioned(
                                right: 12,
                                bottom: 12,
                                child: _tag('原图'),
                              ),
                          ],
                        ),
                      ),
              ),
            ),
          ),
          if (output != null)
            Slider(
              value: divider,
              onChanged: (value) => setState(() => divider = value),
            ),
          const SizedBox(height: 20),
          Row(
            children: [
              const Text('推理后端', style: TextStyle(fontWeight: FontWeight.w600)),
              const Spacer(),
              SegmentedButton<String>(
                segments: const [
                  ButtonSegment(value: 'npu', label: Text('NPU')),
                  ButtonSegment(value: 'gpu', label: Text('GPU')),
                ],
                selected: {selectedBackend},
                onSelectionChanged: busy || checking
                    ? null
                    : (value) {
                        setState(() {
                          selectedBackend = value.first;
                          ready = false;
                        });
                        checkDevice();
                      },
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            selectedBackend == 'npu'
                ? 'Qualcomm HTP · 当前模型适配骁龙 8 Gen 2'
                : 'ncnn Vulkan 兼容模式 · 处理较慢，小米 13 建议使用 NPU',
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              const Text('处理倍率', style: TextStyle(fontWeight: FontWeight.w600)),
              const Spacer(),
              const Text('固定 2× · 输出原尺寸'),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            inputWidth > 0
                ? '${cameraMode ? "拍照 ${cameraZoom?.toStringAsFixed(1)}× · " : ""}原图 $inputWidth×$inputHeight → 处理 $workingWidth×$workingHeight，共 ${patchCount(workingWidth) * patchCount(workingHeight)} 块。融合后输出 $inputWidth×$inputHeight。'
                : '等比例调整到最长边 832，按 512×512 输入块复原，重叠 96；融合后保持原尺寸。',
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 18),
          Text(message, style: const TextStyle(fontSize: 14)),
          if (busy)
            Padding(
              padding: const EdgeInsets.only(top: 10),
              child: LinearProgressIndicator(value: progress),
            ),
          const SizedBox(height: 18),
          FilledButton.icon(
            onPressed: input != null && ready && !busy && !checking
                ? restore
                : null,
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(54),
            ),
            icon: const Icon(Icons.auto_awesome_outlined),
            label: Text(busy ? '正在复原…' : '开始复原'),
          ),
          if (restoring)
            TextButton(
              onPressed: () async {
                await api.invokeMethod<void>('cancel');
              },
              child: const Text('取消处理'),
            ),
          const SizedBox(height: 10),
          OutlinedButton.icon(
            onPressed: busy || checking || !ready ? null : takePhoto,
            icon: const Icon(Icons.camera_alt_outlined),
            label: const Text('拍照并复原（≥10×）'),
          ),
          const SizedBox(height: 6),
          const Text(
            '在系统相机设置 ≥10× 后拍摄。仅处理 EXIF 可验证的照片；结果保持拍照原尺寸。',
            style: TextStyle(fontSize: 12),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: busy ? null : pick,
                  child: Text(input == null ? '选择照片' : '更换照片'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton(
                  onPressed: output != null && !busy ? save : null,
                  child: const Text('保存结果'),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Center(
            child: Text(
              '$device · $backend',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
            ),
          ),
          Center(
            child: Wrap(
              alignment: WrapAlignment.center,
              children: [
                TextButton(
                  onPressed: busy || checking || !supported
                      ? null
                      : importModels,
                  child: Text(
                    '${ready ? "更换" : "导入"} ${selectedBackend.toUpperCase()} 模型包',
                  ),
                ),
                TextButton(
                  onPressed: busy || checking ? null : checkDevice,
                  child: const Text('重新检查模型'),
                ),
              ],
            ),
          ),
        ],
      ),
    ),
  );

  Widget _tag(String label) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
    decoration: BoxDecoration(
      color: Colors.black54,
      borderRadius: BorderRadius.circular(20),
    ),
    child: Text(
      label,
      style: const TextStyle(color: Colors.white, fontSize: 12),
    ),
  );
}

class _ComparisonClipper extends CustomClipper<Rect> {
  const _ComparisonClipper(this.fraction);
  final double fraction;
  @override
  Rect getClip(Size size) =>
      Rect.fromLTWH(0, 0, size.width * fraction, size.height);
  @override
  bool shouldReclip(_ComparisonClipper oldClipper) =>
      fraction != oldClipper.fraction;
}
