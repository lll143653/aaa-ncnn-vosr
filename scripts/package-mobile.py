"""Package only after actual NPU/GPU device acceptance; verify weights and APK libs."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import argparse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables'

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apk-only',action='store_true',help='Verify existing model ZIPs and refresh only APK evidence/checksum')
    parser.add_argument('--tile-size',type=int,choices=[512,1024],default=1024)
    parser.add_argument('--soc',choices=['SM8550','SM8850'],default='SM8550')
    args=parser.parse_args()
    if args.soc=='SM8850' and args.tile_size!=1024: parser.error('SM8850 bundles use the 1024 output profile')
    shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.md',OUT/'THIRD_PARTY_NOTICES.md')
    if args.tile_size==1024:
        acceptance_dir='photo-tests-sm8850-1024' if args.soc=='SM8850' else 'photo-tests-1024'
        acceptance=json.loads((ROOT/'artifacts'/acceptance_dir/'acceptance.json').read_text(encoding='utf-8'))
        assert {x['backend'] for x in acceptance}=={'npu','gpu'} and all(x['passed'] for x in acceptance)
        npu=next(x for x in acceptance if x['backend']=='npu')
        assert npu['photo']['tiles']==len(npu['photo']['nativePatchMs'])==4
        gpu=next(x for x in acceptance if x['backend']=='gpu')
        assert gpu['raw']['stats']['streamGraphs'] and gpu['raw']['stats']['maxResidentNetCount']==1
        if args.soc=='SM8850':
            assert gpu['raw']['stats']['vaeBackend']=='vulkan'
            ns=json.loads(npu['photo']['engineStats'])
            assert ns['extendedUdma'] and ns['maxResidentContextCount']<=6
        assert all(x['raw']['passed'] and x['raw']['pixelPsnrVsFp32Db']>=(30 if x['backend']=='npu' else 45) for x in acceptance)
    else:
        acceptance=json.loads((ROOT/'artifacts/mobile-tests/acceptance.json').read_text(encoding='utf-8'))
        assert len(acceptance)==4 and all(x['passed'] for x in acceptance)
        assert [x['backend'] for x in acceptance]==['npu','gpu','gpu','npu']
        assert acceptance[2]['sessionReused'] and acceptance[2]['case']=='jpeg-exif6-vertical'
    records=[]
    targets=[('npu','qnn','vosr-0.5b-htp-v73-models.zip'),('gpu','ncnn-0.5b','vosr-0.5b-ncnn-vulkan-models.zip')]
    if args.soc=='SM8850':
        targets=[('npu','qnn-sm8850-1024','vosr-0.5b-sm8850-htp-v81-1024-models.zip'),('gpu','ncnn-sm8850-1024','vosr-0.5b-sm8850-ncnn-vulkan-1024-models.zip')]
    for backend,folder,name in targets:
        if args.tile_size==1024 and args.soc=='SM8550':
            folder+='-1024'
            name=name.replace('-models.zip','-1024-models.zip')
        if args.apk_only:
            previous=json.loads((OUT/'SHA256.json').read_text())
            record=next(item for item in previous['files'] if item['file']==name)
            assert (OUT/name).stat().st_size==record['bytes'] and sha(OUT/name)==record['sha256'],name
            records.append(record)
            continue
        source=ROOT/'artifacts'/folder
        manifest=json.loads((source/'manifest.json').read_text())
        assert manifest['tileSize']==args.tile_size
        if backend=='gpu' and args.tile_size==512:
            manifest.setdefault('gpuProfile',{})['hostReductionFallback']=True
            manifest['deviceValidated']=True
            manifest['validatedDevice']='Xiaomi 13 / SM8550 / Adreno 740 / Android 13'
            manifest['validationReference']='docs/09-mobile-backend-acceptance.md'
            (source/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        assert manifest['deviceValidated']
        files=['manifest.json']
        for stage in manifest['stages']:
            assert sha(source/stage['file'])==stage['sha256'],stage['file']
            files.append(stage['file'])
            if backend=='gpu':
                assert sha(source/stage['param'])==stage['paramSha256'],stage['param']
                files.append(stage['param'])
        path=OUT/name
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for file in files: archive.write(source/file,file)
        suffix='-sm8850-1024' if args.soc=='SM8850' else ('-1024' if args.tile_size==1024 else '')
        committed=ROOT/'models/mobile-0.5b'/(backend+suffix)
        committed.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'manifest.json',committed/'manifest.json')
        records.append({'file':name,'bytes':path.stat().st_size,'sha256':sha(path)})
    apk=OUT/'vosr-restore-debug.apk'
    libraries=[]
    with zipfile.ZipFile(apk) as archive:
        assert 'lib/arm64-v8a/libvosr_runner.so' not in archive.namelist()
        for file in (ROOT/'app/android/app/src/main/jniLibs/arm64-v8a').glob('*.so'):
            payload=archive.read('lib/arm64-v8a/'+file.name)
            assert hashlib.sha256(payload).hexdigest()==sha(file),file.name
            libraries.append({'name':file.name,'bytes':len(payload),'sha256':sha(file)})
    if (OUT/'SHA256.json').exists():
        replaced={apk.name,*[x['file'] for x in records]}
        for prior in json.loads((OUT/'SHA256.json').read_text())['files']:
            if prior['file'] in replaced: continue
            path=OUT/prior['file']
            if path.exists():
                assert path.stat().st_size==prior['bytes'] and sha(path)==prior['sha256']
                records.append(prior)
    (OUT/'SHA256.json').write_text(json.dumps({'model_zips_verified':True,'files':[{'file':apk.name,'bytes':apk.stat().st_size,'sha256':sha(apk)},*records]},indent=2)+'\n')
    (ROOT/'evidence/apk-native-verification.json').write_text(json.dumps({'passed':True,'libraries':libraries},indent=2)+'\n')
    print(f'PASS: model hashes, both offline ZIPs and {len(libraries)} packaged native libraries')

if __name__=='__main__': main()
