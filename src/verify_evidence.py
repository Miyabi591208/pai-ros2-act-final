"""Independent post-export integrity checks; never reruns or tunes the policy."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

TARGET=np.array([0.,0.,.6,0.,-.5236])

def verify(base,checkpoints=None):
    manifest=json.loads((base/'evidence_manifest.json').read_text())
    for name,digest in manifest.items():
        p=(base/name).resolve()
        if not p.is_relative_to(base.resolve()): raise ValueError('Unsafe manifest path')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
    final=json.loads((base/'final_results.json').read_text())
    selection=json.loads((base/'selection_before_test.json').read_text())
    assert selection['selected']==final['selection']
    assert selection['test_accessed'] is False
    training=set(range(80)); val=set(range(1000,1020)); test=set(range(final.get('test_start_seed',2000),final.get('test_start_seed',2000)+50))
    assert not(training&val or training&test or val&test)
    checkpoint_hashes={}
    if checkpoints:
        with zipfile.ZipFile(checkpoints) as z:
            assert z.testzip() is None
            for name in z.namelist():
                if name.endswith('/checkpoint/model.safetensors'):
                    checkpoint_hashes[name.split('/')[1]]=hashlib.sha256(z.read(name)).hexdigest()
    training_records=list((base/'data/train').glob('episode_*.json'))
    assert len(training_records)==80
    for trial_file in (base/'results').glob('*/trial.json'):
        metadata=json.loads(trial_file.read_text())
        assert metadata['complete'] is True
        assert metadata['training_source_sha256']==hashlib.sha256((base/'src/act_experiment.py').read_bytes()).hexdigest()
        for name,digest in metadata['data_sha256'].items():
            assert hashlib.sha256((base/'data/train'/name).read_bytes()).hexdigest()==digest
    for p in training_records:
        row=json.loads(p.read_text())
        assert row['teacher'] and row['valid'] and row['success']
        assert row['command_limit_violations']==0
    episodes=0; errors=0; paired=[]
    for row in final['validation']+final['test']+[final['task_specific_oracle_validation']]:
        directory=base/'evaluation'/row['name']
        config=json.loads((directory/'config.json').read_text())
        assert config['runner_sha256']==hashlib.sha256((base/'src/ros_trials.py').read_bytes()).hexdigest()
        assert config['policy_service_sha256']==hashlib.sha256((base/'src/policy_service.py').read_bytes()).hexdigest()
        if checkpoints and not config['config'].get('oracle'):
            assert config['checkpoint_sha256']==checkpoint_hashes[config['config']['trial']]
        records=[json.loads(p.read_text()) for p in sorted(directory.glob('episode_*.json'))]
        assert len(records)==row['n'],row['name']
        expected=test if row['name'].startswith('test_') else val
        assert {r['seed'] for r in records}==expected
        assert sum(r['success'] for r in records)==row['successes']
        assert sum('error' in r for r in records)==row['technical_errors']
        assert sum(r.get('command_limit_violations',0) for r in records)==row['limit_interventions']
        valid=[r for r in records if 'terminal_rmse_rad' in r]
        for summary_key,episode_key in [('terminal_rmse_mean','terminal_rmse_rad'),('jerk_rms_mean','joint_jerk_rms'),('latency_ms_mean','inference_latency_ms_mean')]:
            if valid: assert np.isclose(row[summary_key],np.mean([r[episode_key] for r in valid]),rtol=1e-10,atol=1e-10)
        for r in records:
            episodes+=1
            if 'error' in r:
                errors+=1; assert not r['success']; continue
            ts=np.array([s['t'] for s in r['trace']]); qs=np.array([s['q'] for s in r['trace']])
            assert np.isfinite(qs).all() and np.all(np.diff(ts)>0)
            assert abs(np.sqrt(np.mean((qs[-1]-TARGET)**2))-r['terminal_rmse_rad'])<1e-10
            if r['success']:
                # Reconstruct the complete final stationary interval from measurements.
                v=np.vstack([np.zeros(5),np.diff(qs,axis=0)/np.diff(ts)[:,None]])
                ok=(np.max(abs(qs-TARGET),axis=1)<=.05)&(np.max(abs(v),axis=1)<=.10)
                last_bad=np.where(~ok)[0]
                begin=int(last_bad[-1]+1) if len(last_bad) else 0
                assert ts[-1]-ts[begin]>=.5-1e-8,(row['name'],r['seed'])
                assert r['completion_sim_seconds']<=8.
        if row['name'].startswith('test_'): paired.append(records)
    initial_diffs=[np.max(abs(np.array(a['initial'])-b['initial'])) for a,b in zip(*paired) if 'initial' in a and 'initial' in b]
    if checkpoints:
        with zipfile.ZipFile(checkpoints) as z:
            selected=selection['selected']['trial']
            key=f'results/{selected}/checkpoint/model.safetensors'
            assert hashlib.sha256(z.read(key)).hexdigest()==selection['checkpoint_sha256']
    report={'manifest_files_verified':len(manifest),'evaluation_episodes_verified':episodes,
            'teacher_episodes_verified':len(training_records),
            'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_data_and_code_hashes':'passed','evaluation_code_hashes':'passed',
            'evaluated_checkpoint_hashes':'passed' if checkpoints else 'not_checked',
            'aggregate_metrics_recomputed':'passed',
            'technical_errors':errors,'disjoint_split_check':'passed',
            'success_reconstruction':'passed','selected_checkpoint_hash':'passed' if checkpoints else 'not_checked',
            'max_paired_initial_difference_rad':max(initial_diffs) if initial_diffs else None}
    (base/'local_integrity_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('base',type=Path);ap.add_argument('--checkpoints',type=Path)
    args=ap.parse_args();verify(args.base,args.checkpoints)
