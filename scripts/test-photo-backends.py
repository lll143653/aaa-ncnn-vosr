"""Validate the 512 LR / 1024 output profile through JNI and the real app. No screen reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
ADB=ROOT.parent/'tools/android-sdk/platform-tools/adb.exe'
PKG='ai.vosr.vosr_restore'
REMOTE=f'/sdcard/Android/data/{PKG}/files'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',default=os.environ.get('ANDROID_SERIAL'))
    parser.add_argument('--backends',nargs='+',choices=['npu','gpu'],default=['npu','gpu'])
    parser.add_argument('--deploy',action='store_true')
    parser.add_argument('--raw-only',action='store_true',help='Validate model JNI output only; common photo tiling is separately checked on NPU')
    args=parser.parse_args()
    out=ROOT/'artifacts/photo-tests-1024'; out.mkdir(parents=True,exist_ok=True)
    reference=ROOT/'artifacts/reference-1024'
    def adb(*cmd,timeout=60):
        p=subprocess.run([str(ADB),*(['-s',args.serial] if args.serial else []),*map(str,cmd)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        return p.stdout
    adb('shell','am','start','-W','-n',PKG+'/.MainActivity')
    def require_app_folder(folder):
        listing=adb('shell','ls','-ld',folder)
        assert ' shell ' not in listing,('Directory must be created by the app',listing)
    require_app_folder(REMOTE+'/model-checks-1024')
    for name in ('rgb','dino_input','noise'):
        adb('push',reference/(name+'.raw'),REMOTE+'/model-checks-1024/'+name+'.raw')
    source=out/'input.jpg'
    with Image.open(ROOT/'tests/fixtures/mobile/native-514x129.png') as image:
        image.convert('RGB').resize((4000,3000),Image.Resampling.BICUBIC).save(source,quality=95)
    report_file=out/'acceptance.json'
    reports=json.loads(report_file.read_text(encoding='utf-8')) if report_file.exists() else []
    for backend in args.backends:
        folder=out/backend; folder.mkdir(exist_ok=True)
        models=ROOT/'artifacts'/('qnn-1024' if backend=='npu' else 'ncnn-0.5b-1024')
        manifest=json.loads((models/'manifest.json').read_text())
        assert manifest['tileSize']==1024
        if args.deploy:
            target=REMOTE+('/models-1024' if backend=='npu' else '/models-gpu-1024')
            require_app_folder(target)
            for stage in manifest['stages']:
                for field,hash_field in [('file','sha256')]+([('param','paramSha256')] if backend=='gpu' else []):
                    path=models/stage[field]
                    assert hashlib.sha256(path.read_bytes()).hexdigest()==stage[hash_field]
                    adb('push',path,target+'/'+path.name)
            adb('push',models/'manifest.json',target+'/manifest.json')
        print('RUN raw JNI '+backend,flush=True)
        adb('shell','rm','-f',REMOTE+f'/model-checks-1024/{backend}-result.json',REMOTE+f'/model-checks-1024/{backend}-output.raw')
        log=adb('shell','am','instrument','-w','-e','backend',backend,PKG+'/.ModelChecks',timeout=1200)
        (folder/'instrumentation.txt').write_text(log,encoding='utf-8')
        adb('pull',REMOTE+f'/model-checks-1024/{backend}-result.json',folder/'raw-result.json')
        raw=json.loads((folder/'raw-result.json').read_text())
        assert raw.get('passed'),raw
        adb('pull',REMOTE+f'/model-checks-1024/{backend}-output.raw',folder/'pixels.raw')
        expected=np.load(reference/'pixels.npy')
        actual=np.fromfile(folder/'pixels.raw',np.float32).reshape(expected.shape)
        assert np.isfinite(actual).all()
        rmse=float(np.sqrt(np.mean((actual-expected)**2)))
        psnr=float(20*np.log10(2/max(rmse,1e-12)))
        assert psnr>=(30 if backend=='npu' else 45),(backend,psnr)
        raw['pixelPsnrVsFp32Db']=psnr; raw['pixelRmse']=rmse
        (folder/'raw-validation.json').write_text(json.dumps(raw,indent=2),encoding='utf-8')
        print(json.dumps({'rawPassed':backend,'psnrDb':psnr,'nativeMs':raw['stats']['lastPatchMs']}),flush=True)
        if args.raw_only:
            record={'backend':backend,'passed':True,'scope':'jni-only','raw':raw}
            reports=[x for x in reports if x['backend']!=backend]+[record]
            report_file.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
            manifest['deviceValidated']=True
            manifest['validatedDevice']='Xiaomi 13 / SM8550 / Adreno 740 / Android 13'
            manifest['validationReference']='docs/14-input-patch-512.md'
            manifest['validatedScope']='single 1024-output JNI pass; common four-patch photo pipeline validated on NPU'
            if backend=='gpu':
                manifest['gpuProfile'].update({'hostVaeFallback':True,'maxStorageBufferRangeBytes':134217728,'vaeCpuSgemmConvolution':False,'streamGraphs':True})
            (models/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            continue
        adb('push',source,REMOTE+'/photo-test.jpg')
        adb('shell','rm','-f',REMOTE+'/app-test-result.json')
        runid=uuid.uuid4().hex
        adb('shell','input','keyevent','KEYCODE_WAKEUP')
        adb('shell','am','start','-n',PKG+'/.MainActivity','--es','vosr_test_input',REMOTE+'/photo-test.jpg',
            '--ez','vosr_test_photo','true','--ei','vosr_test_scale','2','--es','vosr_test_backend',backend,
            '--es','vosr_test_run_id',runid,'--ez','vosr_test_save','true')
        print('RUN four actual photo patches '+backend,flush=True)
        launched=time.monotonic(); deadline=launched+1800
        while True:
            result=json.loads(adb('shell',f'if [ -f {REMOTE}/app-test-result.json ]; then cat {REMOTE}/app-test-result.json; else echo "{{}}"; fi'))
            if 'error' in result: raise RuntimeError(result['error'])
            if result.get('testRunId')==runid: break
            if time.monotonic()-launched>10 and not adb('shell',f'pidof {PKG} || true').strip():
                raise RuntimeError('App process exited before producing a result')
            if time.monotonic()>deadline: raise TimeoutError(backend)
            time.sleep(2)
        assert result['mode']=='photo' and result['backendId']==backend
        assert (result['width'],result['height'])==(4000,3000)
        assert (result['workingWidth'],result['workingHeight'])==(832,624)
        assert result['inputPatchSize']==512 and result['overlap']==96 and result['inputStride']==416
        assert result['outputPatchSize']==1024 and result['outputOverlap']==192
        assert result['scale']==2 and result['pyramidLevels']==5 and result['tiles']==4
        assert len(result['nativePatchMs'])==4 and len(result['patchTimings'])==4
        stats=json.loads(result['engineStats'])
        assert stats['patchRunCount']==4 and stats['subprocessCount']==0
        if backend=='npu':
            assert stats['graphCount']==12 and stats['sharedScratch'] and stats['streamDit']
            assert stats['maxResidentContextCount']<=6 and stats['contextCreateCount']==33
        else: assert stats['contextCreateCount']==12
        media=adb('shell','content','query','--uri',result['galleryUri'],'--projection','is_pending')
        assert 'is_pending=0' in media
        adb('pull',REMOTE+'/app-test-output.png',folder/'output.png')
        with Image.open(folder/'output.png') as image: assert image.size==(4000,3000)
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        power=adb('shell','dumpsys','power')
        result['deviceWakefulness']=next((x.strip() for x in power.splitlines() if 'mWakefulness=' in x),'unknown')
        record={'backend':backend,'passed':True,'raw':raw,'photo':result}
        reports=[x for x in reports if x['backend']!=backend]+[record]
        report_file.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest['deviceValidated']=True
        manifest['validatedDevice']='Xiaomi 13 / SM8550 / Adreno 740 / Android 13'
        manifest['validationReference']='docs/14-input-patch-512.md'
        manifest['photoProfile']={'longestSide':832,'inputTileSize':512,'inputOverlap':96,'inputStride':416,'scale':2,'outputTileSize':1024,'outputOverlap':192}
        (models/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'photoPassed':backend,'patches':4,'nativePatchMs':result['nativePatchMs'],'elapsedMs':result['elapsedMs']}),flush=True)
    print('PASS requested 1024-output backends',flush=True)

if __name__=='__main__': main()
