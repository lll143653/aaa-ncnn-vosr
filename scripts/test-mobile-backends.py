"""Exercise actual Android NPU/GPU paths, patch stitching, EXIF and saving. No screen reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ADB=ROOT.parent/'tools/android-sdk/platform-tools/adb.exe'
PKG='ai.vosr.vosr_restore'
REMOTE=f'/sdcard/Android/data/{PKG}/files'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',default=os.environ.get('ANDROID_SERIAL'),help='Default: ANDROID_SERIAL or the single connected device')
    parser.add_argument('--fixtures',type=Path,default=ROOT/'tests/fixtures/mobile')
    parser.add_argument('--backends',nargs='+',default=['npu','gpu','gpu','npu'])
    parser.add_argument('--deploy-gpu',action='store_true')
    args=parser.parse_args()
    out=ROOT/'artifacts/mobile-tests'; out.mkdir(parents=True,exist_ok=True)
    def adb(*cmd):
        result=subprocess.run([str(ADB),*(['-s',args.serial] if args.serial else []),*map(str,cmd)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=50)
        if result.returncode: raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout
    if args.deploy_gpu:
        models=ROOT/'artifacts/ncnn-0.5b'
        bundle=json.loads((models/'manifest.json').read_text())
        for stage in bundle['stages']:
            for field in ('file','param'):
                adb('push',models/stage[field],f'{REMOTE}/models-gpu/{stage[field]}')
        adb('push',models/'manifest.json',f'{REMOTE}/models-gpu/manifest.json')
        print('GPU models deployed',flush=True)
    report=[]
    for index,backend in enumerate(args.backends):
        vertical=index>=2
        case='jpeg-exif6-vertical' if vertical else 'png-horizontal'
        filename='native-exif6.jpg' if vertical else 'native-514x129.png'
        expected=(258,1028) if vertical else (1028,258)
        source=args.fixtures/filename; target=out/f'{index:02}-{backend}-{case}'; target.mkdir(exist_ok=True)
        adb('push',source,f'{REMOTE}/{filename}')
        adb('shell','rm','-f',f'{REMOTE}/app-test-result.json')
        adb('shell','am','start','-n',PKG+'/.MainActivity','--es','vosr_test_input',f'{REMOTE}/{filename}',
            '--ei','vosr_test_scale','2','--ez','vosr_test_save','true','--es','vosr_test_backend',backend)
        print(f'RUN {index} {backend} {case}',flush=True)
        launched=time.monotonic(); deadline=launched+600
        while True:
            text=adb('shell',f'if [ -f {REMOTE}/app-test-result.json ]; then cat {REMOTE}/app-test-result.json; else echo "{{}}"; fi')
            record=json.loads(text)
            if 'error' in record: raise RuntimeError(record['error'])
            if 'backendId' in record: break
            if not adb('shell',f'pidof {PKG} || true').strip() and time.monotonic()-launched>15:
                raise RuntimeError('App process exited')
            if time.monotonic()>deadline: raise TimeoutError(f'{backend} {case}')
            time.sleep(1)
        assert record['backendId']==backend
        assert (record['width'],record['height'])==expected
        assert record['tiles']==3 and record['inputDownscaled'] is False and record['streamingOutput'] is True
        assert record['exifOrientation']==(6 if vertical else 1)
        stats=json.loads(record['engineStats'])
        assert stats['engine']==('JNI_QNN_HTP_RESIDENT' if backend=='npu' else 'NCNN_VULKAN_RESIDENT')
        assert stats['contextCreateCount']==12 and stats['subprocessCount']==0 and stats['intermediateTensorFileBytes']==0
        (target/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        adb('pull',f'{REMOTE}/app-test-output.png',target/'output.png')
        actual=np.asarray(Image.open(target/'output.png').convert('RGB')).astype(np.float32)
        baseline=np.asarray(Image.open(args.fixtures/'native-resolution'/case/'output.png').convert('RGB')).astype(np.float32)
        assert actual.shape==baseline.shape==(expected[1],expected[0],3)
        rmse=float(np.sqrt(np.mean((actual-baseline)**2))); psnr=None if rmse==0 else float(20*np.log10(255/rmse))
        if backend=='npu': assert rmse==0,psnr
        else: assert psnr is None or psnr>=40,psnr
        media=adb('shell','content','query','--uri',record['galleryUri'],'--projection','_data:_size:is_pending')
        assert 'is_pending=0' in media
        (target/'mediastore.txt').write_text(media,encoding='utf-8')
        path=re.search(r'_data=([^,\n]+)',media); assert path
        adb('pull',path.group(1),target/'gallery.png')
        assert hashlib.sha256((target/'gallery.png').read_bytes()).digest()==hashlib.sha256((target/'output.png').read_bytes()).digest()
        summary={'backend':backend,'case':case,'passed':True,'totalMs':record['elapsedMs'],'pipelineMs':record['pipelineMs'],
                 'initializationMs':record['engineInitializationMs'],'sessionReused':record['residentEngineReused'],
                 'pixelExactVsPreviousNpu':rmse==0,'psnrVsPreviousNpu':psnr,'maxPixelDifference':float(np.max(np.abs(actual-baseline))),
                 'stats':stats}
        report.append(summary)
        (out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False),flush=True)
    (out/'final-memory.txt').write_text(adb('shell','dumpsys','meminfo',PKG),encoding='utf-8')
    print('PASS: all requested backends, original-resolution patches, EXIF and MediaStore',flush=True)

if __name__=='__main__': main()
