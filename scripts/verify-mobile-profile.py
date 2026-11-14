"""Check collected profiling evidence and numerically unchanged application outputs."""
from pathlib import Path
import csv
import hashlib
import json
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/profiling/stage10'

def main():
    samples=json.loads((OUT/'samples.json').read_text())
    names=[s['name'] for s in samples]
    assert len(names)==len(set(names))
    foreground=[s for s in samples if s['name'].startswith('foreground')]
    assert len(foreground)==9
    hardware={}
    for sample in samples:
        r=sample['record']; stats=json.loads(r['engineStats'])
        assert r['width']==r['height']==512 and r['tiles']==1 and not r['inputDownscaled']
        assert stats['contextCreateCount']==12 and len(stats['stageMs'])==12
        assert r['comparison']['pixelExact'] if sample['backend']=='npu' else r['comparison']['psnrDb']>=40
        if sample in foreground:
            assert 'mWakefulness=Awake' in (OUT/(sample['name']+'-power.txt')).read_text()
            assert (OUT/(sample['name']+'-uid-state.txt')).read_text().strip()=='2 (TOP)'
            if 'warm' in sample['name']: assert r['residentEngineReused'] and r['engineInitializationMs']==0
        if sample['backend']=='gpu' and sample['profile']:
            reductions=stats['cpuReductionTimings']
            assert len(reductions)==85 and sum(x['calls'] for x in reductions)==85
        for graph in stats.get('qnnProfiles',[]):
            if graph.get('executeStatus',0) or graph.get('truncated'): continue
            assert graph['getEventsStatus']==0 and all(e['status']==0 for e in graph['events'])
            if any(e['type']==3004 and e['unit']==1 for e in graph['events']): hardware[graph['stage']]=sample['name']
    assert len(hardware)==12,hardware
    diagnostic=next(s for s in samples if s['name']=='gpu-all-cpu-warm')
    cpu=json.loads(diagnostic['record']['engineStats'])['cpuLayerTimings']
    assert sum(m['calls'] for m in cpu if m['type']=='MatMul')==134
    assert sum(m['calls'] for m in cpu if m['type']=='Reduction')==85
    control=next(s for s in samples if s['name']=='gpu-all-cpu-control')
    assert not control['profile'] and control['record']['residentEngineReused']
    with Image.open(OUT/'gpu-all-cpu-warm/output.png') as instrumented, Image.open(OUT/'gpu-all-cpu-control/output.png') as ordinary:
        assert ImageChops.difference(instrumented.convert('RGB'),ordinary.convert('RGB')).getbbox() is None
    assert any(s['name'].startswith('final-npu-normal') and s['profile'] is False for s in samples)
    trace_rows=list(csv.DictReader((OUT/'foreground-slices.csv').open()))
    patches=[r for r in trace_rows if r['name'] in ('vosr.npu.patch','vosr.gpu.patch')]
    stages=[r for r in trace_rows if r['name'].startswith(('vosr.npu.','vosr.gpu.')) and not r['name'].endswith('.patch')]
    assert len(patches)==9 and len(stages)==108,(len(patches),len(stages))
    artifacts={}
    for file in [OUT/'samples.json',*OUT.glob('*.perfetto-trace')]:
        with file.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        artifacts[file.name]={'bytes':file.stat().st_size,'sha256':digest}
    evidence={'passed':True,'sampleCount':len(samples),'foregroundSamples':len(foreground),
              'foregroundCompleteNativePatches':len(patches),'foregroundCompleteStages':len(stages),
              'hardwareStageSource':hardware,'artifacts':artifacts,
              'limitations':['Small sample counts; no controlled clock/thermal experiment',
                'Detailed QNN stages combined across two separately labelled runs; no latency aggregation',
                'GPU stage times include compute, CPU fallback, transfers and synchronization; no GPU kernel timestamps',
                'Perfetto systrace parse warnings retained; expected application patch/stage markers independently counted']}
    (OUT/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PASS: all outputs, foreground Awake/TOP, reuse, 134 CPU MatMul + 85 Reduction, profile/control pixel parity, 12 hardware stages, 9 patches/108 trace stages')

if __name__=='__main__': main()
