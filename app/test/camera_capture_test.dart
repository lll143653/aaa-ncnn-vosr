import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:material_ui/material_ui.dart';
import 'package:vosr_restore/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const api = MethodChannel('ai.vosr/restore');
  const progress = MethodChannel('ai.vosr/progress');
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;

  testWidgets('gallery uses the bounded photo pipeline with four LR patches', (
    tester,
  ) async {
    final calls = <MethodCall>[];
    final fixture = File('../tests/fixtures/mobile/profile-128.png')
        .absolute
        .path;
    messenger.setMockMethodCallHandler(progress, (_) async => null);
    messenger.setMockMethodCallHandler(api, (call) async {
      calls.add(call);
      switch (call.method) {
        case 'diagnostics':
          return {'device': 'Xiaomi 13', 'supported': true, 'ready': true};
        case 'recoverCameraPhoto':
          return null;
        case 'pickImage':
          return {'path': fixture, 'width': 4000, 'height': 3000};
        case 'restorePhoto':
          return {'path': fixture, 'backend': 'NPU', 'elapsedMs': 1};
        default:
          throw StateError('Unexpected ${call.method}');
      }
    });
    addTearDown(() {
      messenger.setMockMethodCallHandler(api, null);
      messenger.setMockMethodCallHandler(progress, null);
    });
    await tester.pumpWidget(const VosrApp());
    await tester.pumpAndSettle();
    await tester.tap(find.text('选择照片').first);
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(find.text('开始复原'), 250);
    expect(find.textContaining('处理 832×624，共 4 块'), findsOneWidget);
    expect(find.text('4×'), findsNothing);
    await tester.scrollUntilVisible(find.text('开始复原'), 250);
    await tester.tap(find.text('开始复原'));
    await tester.pumpAndSettle();
    final restores = calls
        .where((c) => c.method.startsWith('restore'))
        .toList();
    expect(restores, hasLength(1));
    expect(restores.single.method, 'restorePhoto');
    expect((restores.single.arguments as Map)['scale'], 2);
    await tester.pumpWidget(const SizedBox());
    await tester.pumpAndSettle();
  });

  for (final accepted in [false, true]) {
    testWidgets(
      accepted
          ? 'verified camera photo automatically uses 2x camera pipeline'
          : 'rejected EXIF never starts restoration',
      (tester) async {
        final calls = <MethodCall>[];
        final fixture = File('../tests/fixtures/mobile/profile-128.png')
            .absolute
            .path;
        messenger.setMockMethodCallHandler(progress, (_) async => null);
        messenger.setMockMethodCallHandler(api, (call) async {
          calls.add(call);
          switch (call.method) {
            case 'diagnostics':
              return {'device': 'Xiaomi 13', 'supported': true, 'ready': true};
            case 'recoverCameraPhoto':
              return null;
            case 'takePhoto':
              if (!accepted) {
                throw PlatformException(
                  code: 'CAMERA_ERROR',
                  message: 'EXIF 未达到 10×',
                );
              }
              return {
                'path': fixture,
                'width': 4000,
                'height': 3000,
                'zoomRatio': 10.0,
                'mode': 'camera',
              };
            case 'restoreCamera':
              return {'path': fixture, 'backend': 'NPU', 'elapsedMs': 1};
            default:
              throw StateError('Unexpected ${call.method}');
          }
        });
        addTearDown(() {
          messenger.setMockMethodCallHandler(api, null);
          messenger.setMockMethodCallHandler(progress, null);
        });
        await tester.pumpWidget(const VosrApp());
        await tester.pumpAndSettle();
        await tester.scrollUntilVisible(find.text('拍照并复原（≥10×）'), 250);
        await tester.tap(find.text('拍照并复原（≥10×）'));
        await tester.pumpAndSettle();
        final restores = calls
            .where((c) => c.method.startsWith('restore'))
            .toList();
        if (accepted) {
          expect(restores, hasLength(1));
          expect(restores.single.method, 'restoreCamera');
          expect((restores.single.arguments as Map)['scale'], 2);
          expect((restores.single.arguments as Map)['backend'], 'npu');
          expect(find.text('固定 2× · 输出原尺寸'), findsOneWidget);
          expect(find.textContaining('处理 832×624，共 4 块'), findsOneWidget);
        } else {
          expect(restores, isEmpty);
          expect(find.text('EXIF 未达到 10×'), findsWidgets);
        }
        await tester.pumpWidget(const SizedBox());
        await tester.pumpAndSettle();
      },
    );
  }
}
