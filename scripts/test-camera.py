"""Device camera-processing regression with synthetic EXIF fixtures; no screen access."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

ROOT=Path(__file__).resolve().parents[1]
ADB=ROOT.parent/'tools/android-sdk/platform-tools/adb.exe'
PKG='ai.vosr.vosr_restore'
REMOTE=f'/sdcard/Android/data/{PKG}/files'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',default=os.environ.get('ANDROID_SERIAL'))
    parser.add_argument('--cases',nargs='+',help='Run selected case names and retain other recorded cases')
    args=parser.parse_args()
    out=ROOT/'artifacts/camera-tests-1024'; out.mkdir(exist_ok=True,parents=True)
    def adb(*cmd):
        p=subprocess.run([str(ADB),*(['-s',args.serial] if args.serial else []),*map(str,cmd)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=50)
        if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        return p.stdout
    image=Image.open(ROOT/'tests/fixtures/mobile/native-514x129.png').convert('RGB')
    reports=[]
    cases=[('missing',(100,50),None,1),('zoom9',(100,50),9,1),('zoom10-small',(100,50),10,1),
           ('zoom12-exif6',(1001,751),12,6),('zoom10-12mp',(4000,3000),10,1)]
    if args.cases: assert set(args.cases)<={c[0] for c in cases},'Unknown camera test case'
    for name,size,zoom,orientation in cases:
        if args.cases and name not in args.cases: continue
        source=out/(name+'.jpg'); exif=Image.Exif(); exif[274]=orientation
        if zoom is not None: exif[34665]={41988:IFDRational(zoom,1)}
        image.resize(size).save(source,quality=95,exif=exif)
        folder=out/name; folder.mkdir(exist_ok=True)
        adb('push',source,f'{REMOTE}/camera-input.jpg')
        adb('shell','rm','-f',f'{REMOTE}/app-test-result.json')
        runid=uuid.uuid4().hex
        adb('shell','am','start','-n',PKG+'/.MainActivity','--es','vosr_test_input',f'{REMOTE}/camera-input.jpg',
            '--ez','vosr_test_camera','true','--ei','vosr_test_scale','2','--es','vosr_test_backend','npu',
            '--es','vosr_test_run_id',runid,'--ez','vosr_test_save','true')
        print('RUN '+name,flush=True)
        deadline=time.monotonic()+600
        while True:
            raw=adb('shell',f'if [ -f {REMOTE}/app-test-result.json ]; then cat {REMOTE}/app-test-result.json; else echo "{{}}"; fi')
            result=json.loads(raw)
            if 'error' in result or result.get('testRunId')==runid: break
            if time.monotonic()>deadline: raise TimeoutError(name)
            time.sleep(.5)
        (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if zoom is None or zoom<10:
            assert 'error' in result,result
            print('PASS strict rejection '+name,flush=True)
            reports.append({'name':name,'rejected':True,'error':result['error']}); continue
        assert 'error' not in result,result
        logical=size[::-1] if orientation>=5 else size
        assert (result['width'],result['height'])==logical
        assert result['scale']==2 and result['overlap']==96 and result['pyramidLevels']==5
        assert max(result['workingWidth'],result['workingHeight'])==832
        assert result['superResolutionWidth']==result['workingWidth']*2 and result['superResolutionHeight']==result['workingHeight']*2
        expected=lambda n:1 if n<=512 else (n-512+415)//416+1
        assert result['tiles']==expected(result['workingWidth'])*expected(result['workingHeight'])
        assert result['inputPatchSize']==512 and result['outputPatchSize']==1024
        assert result['inputStride']==416 and result['outputOverlap']==192
        assert result['overlapCoordinateSpace']=='resized-input'
        assert len(result['nativePatchMs'])==result['tiles']
        assert result['zoomRatio']==zoom and result['exifOrientation']==orientation
        adb('pull',f'{REMOTE}/app-test-output.png',folder/'output.png')
        with Image.open(folder/'output.png') as output: assert output.size==logical
        media=adb('shell','content','query','--uri',result['galleryUri'],'--projection','_data:is_pending')
        assert 'is_pending=0' in media
        print(json.dumps({'passed':name,'size':logical,'working':(result['workingWidth'],result['workingHeight']),
                          'patches':result['tiles'],'elapsedMs':result['elapsedMs'],'fusionMs':result['fusionMs']}),flush=True)
        reports.append({'name':name,'passed':True,'result':result})
    report_file=out/'acceptance.json'
    if args.cases and report_file.exists():
        previous=json.loads(report_file.read_text(encoding='utf-8'))
        refreshed={r['name']:r for r in reports}
        reports=[refreshed.get(r['name'],r) for r in previous]
    report_file.write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: strict EXIF, up/down resize, EXIF orientation, 2x/96 patches, original-size fusion and gallery saving',flush=True)

if __name__=='__main__': main()
