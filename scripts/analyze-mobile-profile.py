"""Export reproducible wall/CPU/scheduling profiles from application JSON and Perfetto.

No GPU hardware timestamp is inferred from thread sleeping or synchronous wall time.
"""
import csv
import io
import json
from pathlib import Path
import statistics
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/profiling/stage10'
TP=ROOT.parent/'tools/perfetto/trace_processor_shell.exe'
STAGES=['encoder','dino','dit_prepare',*[f'dit_blocks_{n:02d}' for n in range(0,28,4)],'dit_finish','decoder']

def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def query(trace,sql,name):
    (OUT/(name+'.sql')).write_text(sql,encoding='utf-8')
    result=subprocess.run([str(TP),'query','--quiet','--format','csv',str(trace),sql],capture_output=True,text=True,encoding='utf-8',check=True)
    (OUT/(name+'.csv')).write_text(result.stdout,encoding='utf-8')
    (OUT/(name+'-import.txt')).write_text(result.stderr,encoding='utf-8')
    return list(csv.DictReader(io.StringIO(result.stdout.strip())))

def main():
    samples=json.loads((OUT/'samples.json').read_text(encoding='utf-8'))
    summary=[]; stage_rows=[]; reduction_rows=[]; cpu_rows=[]; group={}; qnn_events=[]; qnn_graphs=[]
    for sample in samples:
        record=sample['record']; stats=json.loads(record['engineStats'])
        phases=record['pipelineBreakdownMs']
        # pngEncodeWrite is nested inside stitchAndPng: do not add both.
        phase_sum=sum(v for k,v in phases.items() if k!='pngEncodeWrite')
        metrics={'name':sample['name'],'backend':sample['backend'],'profile':sample['profile'],
          'reused':record['residentEngineReused'],'totalMs':record['elapsedMs'],
          'modelVerificationMs':record['modelVerificationMs'],'engineInitializationMs':record['engineInitializationMs'],
          'pipelineMs':record['pipelineMs'],'nativeMs':record['nativePatchMs'][0],
          'inputCopyMs':stats['inputCopyMs'],'outputCopyMs':stats['outputCopyMs'],
          'profileCollectionMs':stats.get('profileCollectionMs',0),
          **phases,'pipelineUnattributedMs':record['pipelineMs']-phase_sum,
          'powerSnapshot':(OUT/(sample['name']+'-power.txt')).read_text().strip(),
          'psnrDb':record['comparison']['psnrDb'],'pixelExact':record['comparison']['pixelExact']}
        summary.append(metrics)
        for name,value in zip(STAGES,stats['stageMs']):
            stage_rows.append({'sample':sample['name'],'backend':sample['backend'],'stage':name,'wallMs':value})
        for item in stats.get('cpuReductionTimings',[]):
            reduction_rows.append({'sample':sample['name'],**item})
        for item in stats.get('cpuLayerTimings',[]):
            cpu_rows.append({'sample':sample['name'],**item})
        for graph in stats.get('qnnProfiles',[]):
            row={'sample':sample['name'],'stage':graph['stage'],'executeStatus':graph.get('executeStatus',0),
                 'retriedWithoutProfile':graph.get('retriedWithoutProfile',False),
                 'eventCount':graph.get('eventCount',0),'truncated':graph.get('truncated',False),
                 'rpcMs':None,'htpRpcMs':None,'acceleratorMs':None,'acceleratorExcludingWaitMs':None,'cycles':None}
            for event in graph['events']:
                qnn_events.append({'sample':sample['name'],'stage':graph['stage'],**event})
                key={3001:'rpcMs',3002:'htpRpcMs',3004:'acceleratorMs',3012:'acceleratorExcludingWaitMs'}.get(event.get('type'))
                if key and event.get('unit')==1 and event.get('status')==0: row[key]=event['value']/1000
                if event.get('type')==3003 and event.get('unit')==3: row['cycles']=event['value']
            qnn_graphs.append(row)
        if 'warm' in sample['name']:
            key=('foreground-' if sample['name'].startswith('foreground') else 'initial-')+sample['backend']
            group.setdefault(key,[]).append(metrics)
    write_csv(OUT/'application-summary.csv',summary)
    write_csv(OUT/'stage-wall.csv',stage_rows)
    write_csv(OUT/'cpu-reductions.csv',reduction_rows)
    write_csv(OUT/'cpu-layers.csv',cpu_rows)
    write_csv(OUT/'qnn-graphs.csv',qnn_graphs)
    write_csv(OUT/'qnn-events.csv',qnn_events)
    grouped={}
    for key,items in group.items():
        grouped[key]={'samples':len(items)}
        for metric in ['totalMs','pipelineMs','nativeMs','regionDecodeAndBicubic','dinoResizeNormalize','stitchAndPng']:
            values=[r[metric] for r in items]
            grouped[key][metric]={'min':min(values),'median':statistics.median(values),'mean':statistics.mean(values),'max':max(values)}
    (OUT/'summary.json').write_text(json.dumps(grouped,indent=2),encoding='utf-8')
    print(json.dumps(grouped,indent=2))
    # Every complete slice is retained as a separate row; do not blend initialization,
    # detailed QNN instrumentation and ordinary warm inference into one aggregate.
    for trace in sorted(OUT.glob('*.perfetto-trace')):
        prefix=trace.stem
        window="""SELECT s.id,s.name,s.ts,s.dur,t.utid,t.upid,t.tid,p.pid
          FROM slice s JOIN thread_track tt ON tt.id=s.track_id
          JOIN thread t USING(utid) JOIN process p USING(upid)
          WHERE s.dur>0 AND s.name!='vosr.pngEncodeWrite'
          AND (s.name GLOB 'vosr.*' OR s.name='ncnn.cpuReduction' OR s.name GLOB 'ncnn.cpu.*')"""
        query(trace,f"""WITH windows AS ({window})
          SELECT w.id,w.name,w.ts,round(w.dur/1e6,4) wall_ms,w.pid,w.tid,
          round(COALESCE((SELECT SUM(MAX(0,MIN(sc.ts+sc.dur,w.ts+w.dur)-MAX(sc.ts,w.ts)))
            FROM sched sc JOIN thread other ON sc.utid=other.utid
            WHERE other.upid=w.upid AND sc.ts<w.ts+w.dur AND sc.ts+sc.dur>w.ts AND sc.dur>0),0)/1e6,4) app_cpu_ms
          FROM windows w ORDER BY w.ts,w.id""",prefix+'-slices')
        query(trace,f"""WITH windows AS ({window})
          SELECT w.id,w.name,w.ts,st.state,st.blocked_function,
          round(SUM(MAX(0,MIN(st.ts+st.dur,w.ts+w.dur)-MAX(st.ts,w.ts)))/1e6,4) state_ms
          FROM windows w JOIN thread_state st ON st.utid=w.utid
          AND st.ts<w.ts+w.dur AND st.ts+st.dur>w.ts AND st.dur>0
          WHERE w.name GLOB 'vosr.gpu.*' OR w.name GLOB 'vosr.npu.*'
          GROUP BY w.id,st.state,st.blocked_function ORDER BY w.ts,w.id,state_ms DESC""",prefix+'-thread-states')
        query(trace,"SELECT name,value,severity FROM stats WHERE value>0 AND (severity='error' OR name GLOB '*lost*')",prefix+'-health')
    print('PASS: application timing tables and trace queries exported')

if __name__=='__main__': main()
