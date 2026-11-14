import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:vosr_restore/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const api = MethodChannel('ai.vosr/restore');
  const progress = MethodChannel('ai.vosr/progress');

  testWidgets('backend selection uses separate readiness and model import', (
    tester,
  ) async {
    final calls = <MethodCall>[];
    final messenger =
        TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    messenger.setMockMethodCallHandler(progress, (_) async => null);
    messenger.setMockMethodCallHandler(api, (call) async {
      calls.add(call);
      final backend = (call.arguments as Map?)?['backend'];
      if (call.method == 'diagnostics') {
        return {
          'device': 'Xiaomi 13',
          'supported': true,
          'ready': backend == 'npu',
          'message': backend == 'npu' ? 'NPU 模型已就绪' : '请导入 GPU 专用模型包',
        };
      }
      if (call.method == 'importModels') return false;
      if (call.method == 'recoverCameraPhoto') return null;
      throw StateError('Unexpected call: ${call.method}');
    });
    addTearDown(() {
      messenger.setMockMethodCallHandler(api, null);
      messenger.setMockMethodCallHandler(progress, null);
    });
    await tester.pumpWidget(const VosrApp());
    await tester.pumpAndSettle();
    expect(calls.first.arguments, {'backend': 'npu'});
    await tester.scrollUntilVisible(find.text('GPU'), 250);
    await tester.tap(find.text('GPU'));
    await tester.pumpAndSettle();
    expect(calls.last.arguments, {'backend': 'gpu'});
    expect(find.text('请导入 GPU 专用模型包'), findsOneWidget);
    final start = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, '开始复原'),
    );
    expect(start.onPressed, isNull);
    await tester.scrollUntilVisible(find.text('导入 GPU 模型包'), 250);
    await tester.tap(find.text('导入 GPU 模型包'));
    await tester.pumpAndSettle();
    expect(calls.last.method, 'importModels');
    expect(calls.last.arguments, {'backend': 'gpu'});
    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
  });
}
