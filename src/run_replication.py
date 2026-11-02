"""Rebuild lost temporary artifacts; freeze the first run's chosen configuration.

Uses new final evaluation seeds 4000..4049. No tuning on these results.
The first run's summaries are preserved separately, not mixed with new episodes.
"""
import json,subprocess,sys,time,hashlib
from run_comparison import BASE,SRC,call,evaluate

def main():
    original=json.loads((BASE/'original_run_final_results.json').read_text())
    selected=original['selection']
    assert selected=={'trial':'diverse_2000','n_action_steps':1,'ensemble':.01}
    frozen={'selected':selected,'source':'Initial completed run, saved notebook output',
            'test_start_seed':4000,'test_count':50,'tuning_allowed':False}
    (BASE/'replication_frozen_before_training.json').write_text(json.dumps(frozen,indent=2))
    for name in ['lesson_baseline','diverse_2000']:
        call([sys.executable,str(SRC/'act_experiment.py'),'--trial',name],'train_'+name)
    selection={**frozen,'test_accessed':False,
        'checkpoint_sha256':hashlib.sha256((BASE/'results/diverse_2000/checkpoint/model.safetensors').read_bytes()).hexdigest()}
    (BASE/'selection_before_test.json').write_text(json.dumps(selection,indent=2))
    log=(BASE/'policy_service.log').open('w')
    service=subprocess.Popen([sys.executable,str(SRC/'policy_service.py')],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            if 'POLICY_SERVICE_READY' in (BASE/'policy_service.log').read_text():break
            if service.poll() is not None:raise RuntimeError('Policy service failed')
            time.sleep(1)
        oracle=evaluate('val_task_specific_oracle',{'oracle':True},1000,20)
        baseline={'trial':'lesson_baseline','n_action_steps':1,'ensemble':None}
        rows=[evaluate('val_lesson_baseline',baseline,1000,20),evaluate('val_diverse_2000',selected,1000,20)]
        # Configuration was already frozen, independently of this validation.
        test=[evaluate('test_baseline',baseline,4000,50),evaluate('test_selected',selected,4000,50)]
        result={'run_kind':'fixed_configuration_replication','validation':rows,'selection':selected,
                'test':test,'task_specific_oracle_validation':oracle,'test_start_seed':4000,
                'original_run_summary_file':'original_run_final_results.json',
                'incident':'Original run completed 25/50 vs 50/50, but temporary raw files and weights were lost during a native download-save-dialog timeout. Notebook outputs were recovered from Google Drive. This run is a new training and independent evaluation, not recovery of identical original weights.'}
        (BASE/'final_results.json').write_text(json.dumps(result,indent=2))
        print('REPLICATION_COMPLETE',json.dumps(result),flush=True)
    finally:
        service.terminate();service.wait(timeout=20);log.close()

if __name__=='__main__':main()
