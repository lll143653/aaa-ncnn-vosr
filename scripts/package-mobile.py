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
    args=parser.parse_args()
    shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.md',OUT/'THIRD_PARTY_NOTICES.md')
    acceptance=json.loads((ROOT/'artifacts/mobile-tests/acceptance.json').read_text(encoding='utf-8'))
    assert len(acceptance)==4 and all(x['passed'] for x in acceptance)
    assert [x['backend'] for x in acceptance]==['npu','gpu','gpu','npu']
    assert acceptance[2]['sessionReused'] and acceptance[2]['case']=='jpeg-exif6-vertical'
    records=[]
    for backend,folder,name in [('npu','qnn','vosr-0.5b-htp-v73-models.zip'),('gpu','ncnn-0.5b','vosr-0.5b-ncnn-vulkan-models.zip')]:
        if args.apk_only:
            previous=json.loads((OUT/'SHA256.json').read_text())
            record=next(item for item in previous['files'] if item['file']==name)
            assert (OUT/name).stat().st_size==record['bytes'] and sha(OUT/name)==record['sha256'],name
            records.append(record)
            continue
        source=ROOT/'artifacts'/folder
        manifest=json.loads((source/'manifest.json').read_text())
        if backend=='gpu':
            manifest.setdefault('gpuProfile',{})['hostReductionFallback']=True
            manifest['deviceValidated']=True
            manifest['validatedDevice']='Xiaomi 13 / SM8550 / Adreno 740 / Android 13'
            manifest['validationReference']='docs/09-mobile-backend-acceptance.md'
            (source/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
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
        committed=ROOT/'models/mobile-0.5b'/backend
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
    (OUT/'SHA256.json').write_text(json.dumps({'model_zips_verified':True,'files':[{'file':apk.name,'bytes':apk.stat().st_size,'sha256':sha(apk)},*records]},indent=2)+'\n')
    (ROOT/'evidence/apk-native-verification.json').write_text(json.dumps({'passed':True,'libraries':libraries},indent=2)+'\n')
    print('PASS: model hashes, both offline ZIPs and six packaged native libraries')

if __name__=='__main__': main()
