"""Finite preregistered comparison; chooses on validation before accessing test."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from statistics import mean
import hashlib

BASE=Path('/content/pai_ros2'); SRC=BASE/'src'
prefix=json.loads((BASE/'proot_prefix.json').read_text())

def call(args,name):
    print('START',name,flush=True)
    with (BASE/f'{name}.log').open('w') as f:
        p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print((BASE/f'{name}.log').read_text()[-4000:],flush=True)
        raise RuntimeError(name)
    print('DONE',name,flush=True)

def rpc(path,data):
    r=urllib.request.Request('http://127.0.0.1:8765/'+path,data=json.dumps(data).encode(),
                             headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=120) as f: return json.load(f)

def evaluate(name,cfg,start,count):
    out=BASE/'evaluation'/name; out.mkdir(parents=True,exist_ok=True)
    signature={'config':cfg,'start_seed':start,'count':count,
               'runner_sha256':hashlib.sha256((SRC/'ros_trials.py').read_bytes()).hexdigest(),
               'policy_service_sha256':hashlib.sha256((SRC/'policy_service.py').read_bytes()).hexdigest()}
    if not cfg.get('oracle'):
        signature['checkpoint_sha256']=hashlib.sha256((BASE/'results'/cfg['trial']/'checkpoint/model.safetensors').read_bytes()).hexdigest()
    config_file=out/'config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=signature:
        raise RuntimeError('Refusing to mix evaluation results from different code or checkpoints: '+name)
    config_file.write_text(json.dumps(signature,indent=2))
    if not cfg.get('oracle'): rpc('load',cfg)
    mode=' --oracle' if cfg.get('oracle') else ''
    call(prefix+['bash','-c',
       'source /opt/ros/humble/setup.bash && source /work/ws/install/setup.bash && '
       f'python3 /work/src/ros_trials.py --out /work/evaluation/{name} --start {start} --count {count}{mode}'],name)
    episodes=[json.loads((out/f'episode_{seed:05d}.json').read_text()) for seed in range(start,start+count)]
    valid=[r for r in episodes if 'terminal_rmse_rad' in r]
    row={'name':name,**cfg,'n':count,'successes':sum(r['success'] for r in episodes),
         'technical_errors':sum('error' in r for r in episodes),
         'terminal_rmse_mean':mean(r['terminal_rmse_rad'] for r in valid) if valid else None,
         'jerk_rms_mean':mean(r['joint_jerk_rms'] for r in valid) if valid else None,
         'limit_interventions':sum(r.get('command_limit_violations',0) for r in episodes),
         'latency_ms_mean':mean(r['inference_latency_ms_mean'] for r in valid) if valid else None}
    (out/'summary.json').write_text(json.dumps(row,indent=2)); print(json.dumps(row),flush=True)
    return row

def ranking(row):
    # Technical failures always count as failures, and are separately reported.
    return (-row['successes'],row['terminal_rmse_mean'] if row['terminal_rmse_mean'] is not None else float('inf'))

def main():
    call([sys.executable,str(SRC/'act_experiment.py')],'train_all')
    log=(BASE/'policy_service.log').open('w')
    service=subprocess.Popen([sys.executable,str(SRC/'policy_service.py')],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            if 'POLICY_SERVICE_READY' in (BASE/'policy_service.log').read_text(): break
            if service.poll() is not None: raise RuntimeError('Policy service failed')
            time.sleep(1)
        oracle=evaluate('val_task_specific_oracle',{'oracle':True},1000,20)
        rows=[]
        for trial in ['lesson_baseline','order_only_200','time_only_200','aligned_200','aligned_2000','diverse_2000']:
            rows.append(evaluate('val_'+trial,{'trial':trial,'n_action_steps':1,'ensemble':None},1000,20))
        winner=min(rows,key=ranking)
        for n,ensemble in [(4,None),(8,None),(1,.01)]:
            cfg={'trial':winner['trial'],'n_action_steps':n,'ensemble':ensemble}
            rows.append(evaluate(f"val_execution_n{n}_ensemble{ensemble}",cfg,1000,20))
        winner=min(rows,key=ranking)
        if winner['successes']==0:
            (BASE/'validation_needs_revision.json').write_text(json.dumps(rows,indent=2))
            print('NO_VALIDATED_SUCCESS: final holdout remains untouched',flush=True)
            return
        selected={k:winner[k] for k in ['trial','n_action_steps','ensemble']}
        selection={'selected':selected,'validation':rows,'test_accessed':False,
                   'checkpoint_sha256':hashlib.sha256((BASE/'results'/winner['trial']/'checkpoint/model.safetensors').read_bytes()).hexdigest()}
        (BASE/'selection_before_test.json').write_text(json.dumps(selection,indent=2))
        print('SELECTION_LOCKED',json.dumps(selected),flush=True)
        test=[]
        for name,cfg in [('test_baseline',{'trial':'lesson_baseline','n_action_steps':1,'ensemble':None}),('test_selected',selected)]:
            test.append(evaluate(name,cfg,2000,50))
        (BASE/'final_results.json').write_text(json.dumps({'validation':rows,'selection':selected,'test':test,'task_specific_oracle_validation':oracle},indent=2))
        print('COMPARISON_COMPLETE',json.dumps(test),flush=True)
    finally:
        service.terminate(); service.wait(timeout=20); log.close()

if __name__=='__main__': main()
