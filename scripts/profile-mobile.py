"""Cold/warm single-patch wall, QNN events, CPU Reduction and Perfetto profiling.
Never captures or reads the screen. Uses the installed development app.
"""
import argparse
import hashlib
import json
import os
import re
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
OUT=ROOT/'artifacts/profiling/stage10'

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial',default=os.environ.get('ANDROID_SERIAL'),help='Default: ANDROID_SERIAL or the single connected device')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--recover-gpu-cold',action='store_true')
    parser.add_argument('--foreground',action='store_true',help='Resume resident GPU, then benchmark NPU; requires unlocked foreground app')
    parser.add_argument('--resident-gpu',action='store_true',help='Profile already resident GPU under the recorded power state')
    parser.add_argument('--all-cpu-gpu',action='store_true',help='Profile every CPU-only layer and a profiling-disabled control')
    parser.add_argument('--detailed-npu',action='store_true',help='Collect detailed NPU events separately from latency samples')
    parser.add_argument('--profile-start-stage',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args(); OUT=args.output_dir.resolve()
    assert OUT.is_relative_to(ROOT.parent),'Keep profiling artifacts under E:\\work\\ai'
    OUT.mkdir(parents=True,exist_ok=True)
    def adb(*cmd,check=True,stdin=None):
        for attempt in range(4):
            result=subprocess.run([str(ADB),*(['-s',args.serial] if args.serial else []),*map(str,cmd)],capture_output=True,
                                  text=True,encoding='utf-8',errors='replace',timeout=55,input=stdin)
            if not result.returncode or not any(s in result.stderr for s in ['not found','device offline','no devices']): break
            if attempt<3: time.sleep(2)
        if check and result.returncode: raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout
    uid_match=re.search(r'\buid:(\d+)',adb('shell','cmd','package','list','packages','-U',PKG))
    assert uid_match, 'Install the VOSR app before profiling'
    app_uid=uid_match.group(1)
    source=ROOT/'tests/fixtures/mobile/profile-128.png'
    adb('push',source,f'{REMOTE}/profile-128.png')
    def snapshot(name):
        (OUT/(name+'-thermal.txt')).write_text(adb('shell','dumpsys','thermalservice'),encoding='utf-8')
        (OUT/(name+'-power.txt')).write_text(adb('shell','dumpsys power | grep -E "mWakefulness=|mInteractive="'),encoding='utf-8')
        (OUT/(name+'-uid-state.txt')).write_text(adb('shell','cmd','activity','get-uid-state',app_uid,check=False),encoding='utf-8')
    def save(name,backend,profile,record,old=False):
        folder=OUT/name; folder.mkdir(exist_ok=True)
        assert record['backendId']==backend and (record['width'],record['height'])==(512,512)
        assert record['tiles']==1 and record['inputDownscaled'] is False
        adb('pull',f'{REMOTE}/app-test-output.png',folder/'output.png')
        if not old:
            (folder/'timing.json').write_text(adb('shell','run-as',PKG,'cat',record['timingFile']),encoding='utf-8')
            a=np.asarray(Image.open(folder/'output.png').convert('RGB')).astype(np.float32)
            b=np.asarray(Image.open(OUT/'baseline/output.png').convert('RGB')).astype(np.float32)
            rmse=float(np.sqrt(np.mean((a-b)**2)))
            psnr=None if rmse==0 else float(20*np.log10(255/rmse))
            assert rmse==0 if backend=='npu' else (psnr is None or psnr>=40), (name,psnr)
            stats=json.loads(record['engineStats'])
            assert stats['contextCreateCount']==12
            if backend=='npu': assert stats['boostVoteAccepted']
            if backend=='gpu' and profile:
                assert sum(m['calls'] for m in stats['cpuReductionTimings'])>0
            record['comparison']={'pixelExact':rmse==0,'psnrDb':psnr}
        (folder/'result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        snapshot(name)
        print(json.dumps({'name':name,'totalMs':record['elapsedMs'],'pipelineMs':record['pipelineMs'],
                          'nativeMs':record['nativePatchMs'][0],'reused':record['residentEngineReused']}),flush=True)
        return {'name':name,'backend':backend,'profile':profile,'record':record}
    def run(name,backend,profile=False,old=False):
        assert not (OUT/name/'result.json').exists(),f'Already recorded {name}; use a fresh --output-dir'
        if args.foreground:
            power=adb('shell','dumpsys power | grep mWakefulness=')
            assert 'mWakefulness=Awake' in power, 'Phone is not Awake; no foreground sample started'
        run_id=uuid.uuid4().hex
        adb('shell','rm','-f',f'{REMOTE}/app-test-result.json')
        adb('shell','am','start','-n',PKG+'/.MainActivity','--es','vosr_test_input',f'{REMOTE}/profile-128.png',
            '--es','vosr_test_backend',backend,'--ei','vosr_test_scale','4',
            '--ez','vosr_test_profile',str(profile).lower(),'--es','vosr_test_run_id',run_id,
            '--ei','vosr_test_profile_start_stage',str(args.profile_start_stage if profile else 0))
        print(f'RUN {name} {backend} detailed={profile}',flush=True)
        deadline=time.monotonic()+300
        while True:
            raw=adb('shell',f'if [ -f {REMOTE}/app-test-result.json ]; then cat {REMOTE}/app-test-result.json; else echo "{{}}"; fi')
            try: record=json.loads(raw)
            except json.JSONDecodeError: record={}
            if 'error' in record: raise RuntimeError(record['error'])
            if 'backendId' in record and (old or record.get('testRunId')==run_id): break
            if time.monotonic()>deadline: raise TimeoutError(name)
            time.sleep(.5)
        return save(name,backend,profile,record,old)
    if args.recover_gpu_cold:
        sample=save('gpu-cold','gpu',False,json.loads(adb('shell','cat',f'{REMOTE}/app-test-result.json')))
        samples=json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
        assert not any(s['name']=='gpu-cold' for s in samples)
        sample['recoveredAfterUsbDisconnect']=True
        # Recovery snapshots are taken later; they cannot establish state during this run.
        sample['snapshotAfterRecovery']=True
        samples.append(sample)
        (OUT/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
        return
    if args.baseline_only:
        run('baseline','npu',old=True)
        print('PASS: original-APK baseline saved',flush=True); return
    assert (OUT/'baseline/output.png').is_file(),'Run --baseline-only before replacing the original APK'
    resume=args.foreground or args.resident_gpu or args.detailed_npu or args.all_cpu_gpu
    for stale in ([] if resume else ['trace-start-error.txt','npu-detailed-error.txt']):
        (OUT/stale).unlink(missing_ok=True)
    if not resume: adb('shell','am','force-stop',PKG)
    config='''buffers { size_kb: 32768 fill_policy: RING_BUFFER }
data_sources { config { name: "linux.ftrace" ftrace_config {
 ftrace_events: "sched/sched_switch" ftrace_events: "sched/sched_waking"
 ftrace_events: "sched/sched_blocked_reason"
 ftrace_events: "power/cpu_frequency" ftrace_events: "power/cpu_idle"
 atrace_apps: "ai.vosr.vosr_restore"
 atrace_categories: "dalvik"
} } }
data_sources { config { name: "linux.process_stats" process_stats_config { scan_all_processes_on_start: true proc_stats_poll_ms: 1000 } } }
data_sources { config { name: "linux.sys_stats" sys_stats_config { stat_period_ms: 1000 cpufreq_period_ms: 500 } } }
duration_ms: 360000
flush_period_ms: 5000
write_into_file: true
file_write_period_ms: 1000
max_file_size_bytes: 268435456
'''
    trace_name='gpu-all-cpu' if args.all_cpu_gpu else 'foreground' if args.foreground else 'gpu-profile' if args.resident_gpu else 'qnn-detailed-'+str(args.profile_start_stage) if args.detailed_npu else 'vosr-stage10'
    (OUT/(trace_name+'-config.pbtxt')).write_text(config)
    remote_trace='/data/misc/perfetto-traces/'+trace_name+'.perfetto-trace'
    trace_pid=None
    try:
        started=adb('shell','perfetto','--background-wait','--txt','-c','-','-o',remote_trace,stdin=config)
        (OUT/(trace_name+'-start.txt')).write_text(started)
        trace_pid=next(int(s) for s in started.splitlines() if s.strip().isdigit())
    except Exception as error:
        (OUT/(trace_name+'-start-error.txt')).write_text(str(error))
        print('Perfetto unavailable; continuing application/QNN timings',flush=True)
    samples=json.loads((OUT/'samples.json').read_text(encoding='utf-8')) if resume else []
    try:
        sequence=[('npu-cold','npu',False)]+[(f'npu-warm-{i}','npu',False) for i in range(1,6)]+[
            ('npu-detailed','npu',True),('gpu-cold','gpu',False),
            ('gpu-warm-1','gpu',True),('gpu-warm-2','gpu',True),('npu-return','npu',False)]
        if args.foreground:
            sequence=[('foreground-gpu-cold','gpu',False),('foreground-gpu-warm-1','gpu',True),('foreground-gpu-warm-2','gpu',True),
                      ('foreground-npu-cold','npu',False)]+[(f'foreground-npu-warm-{i}','npu',False) for i in range(1,6)]
        if args.resident_gpu:
            sequence=[('gpu-profile-1','gpu',True),('gpu-profile-2','gpu',True)]
        if args.detailed_npu:
            sequence=[('npu-detailed-tail' if args.profile_start_stage else 'npu-detailed-v3','npu',True),
                      (f'final-npu-normal-{args.profile_start_stage}','npu',False)]
        if args.all_cpu_gpu:
            sequence=[('gpu-all-cpu-cold','gpu',True),('gpu-all-cpu-warm','gpu',True),
                      ('gpu-all-cpu-control','gpu',False),('final-npu-after-cpu-profile','npu',False)]
        for name,backend,detailed in sequence:
            try:
                sample=run(name,backend,detailed); samples.append(sample)
            except Exception as error:
                if not name.startswith('npu-detailed'): raise
                (OUT/(name+'-error.txt')).write_text(str(error))
                print('Detailed QNN unsupported in this context; continuing remaining measurements',flush=True)
            (OUT/'samples.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'memory-after.txt').write_text(adb('shell','dumpsys','meminfo',PKG),encoding='utf-8')
        print('PASS: all profile samples and output checks',flush=True)
    finally:
        if trace_pid:
            adb('shell','kill','-TERM',str(trace_pid),check=False)
            time.sleep(2)
            adb('pull',remote_trace,OUT/(trace_name+'.perfetto-trace'),check=False)

if __name__=='__main__': main()
