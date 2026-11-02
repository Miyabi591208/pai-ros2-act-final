"""Readable derived figures; preserve all original measured evidence unchanged."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(base,out):
    out.mkdir(parents=True,exist_ok=True)
    result=json.loads((base/'final_results.json').read_text());stats=json.loads((base/'statistical_summary.json').read_text())
    original=json.loads((base/'original_run_final_results.json').read_text())
    start=result.get('test_start_seed',2000);provenance={}
    inputs=list(base.glob('*.json'))+[base/'robot.urdf']
    inputs += [base/'evaluation'/name/f'episode_{start:05d}.json' for name in ['test_baseline','test_selected']]
    for p in inputs:
        rel=p.relative_to(base);q=out/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
        provenance[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'presentation_input_manifest.json').write_text(json.dumps(provenance,indent=2))
    names={'val_lesson_baseline':'Baseline','val_order_only_200':'Order only','val_time_only_200':'Time only',
           'val_aligned_200':'Aligned / 200','val_aligned_2000':'Aligned / 2000','val_diverse_2000':'Diverse / 2000',
           'val_execution_n4_ensembleNone':'Execute 4','val_execution_n8_ensembleNone':'Execute 8',
           'val_execution_n1_ensemble0.01':'Temporal mean'}
    with plt.rc_context({'font.size':13,'axes.titlesize':14,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12}):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        rows=original['validation']; labels=[names[r['name']] for r in rows]
        scores=[r['successes']/r['n'] for r in rows]
        axes[0].barh(labels,scores,color=['#7d8894']+['#357ebd']*7+['#138a72'])
        axes[0].invert_yaxis();axes[0].set(xlim=(0,1.23),xlabel='Success rate',title='Initial development (saved summary)')
        axes[0].set_xticks([0,.5,1])
        for i,r in enumerate(rows):axes[0].text(scores[i]+.02,i,f"{r['successes']}/20",va='center',fontsize=11)
        ss=[stats[k]['successes']/stats[k]['n'] for k in ['baseline','selected']]
        ci=np.array([stats[k]['wilson95'] for k in ['baseline','selected']])
        axes[1].bar(['Baseline','Selected ACT'],ss,color=['#7d8894','#138a72'])
        axes[1].errorbar([0,1],ss,yerr=np.maximum(0,np.array([ss-ci[:,0],ci[:,1]-ss])),fmt='none',color='black',capsize=5)
        axes[1].set(ylim=(0,1.2),ylabel='Success rate',title='Fixed-setting replication (new 50)')
        for i,k in enumerate(['baseline','selected']):axes[1].text(i,1.07,f"{stats[k]['successes']}/50",ha='center',fontsize=15)
        axes[1].set_yticks([0,.5,1]);fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    with plt.rc_context({'font.size':17,'axes.titlesize':19,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16}):
        fig,axes=plt.subplots(2,1,figsize=(8,5.333),sharex=True,layout='constrained')
        for name,label,color in [('test_baseline','Baseline','#7d8894'),('test_selected','Selected ACT','#138a72')]:
            row=json.loads((base/'evaluation'/name/f'episode_{start:05d}.json').read_text())
            if not row.get('trace'):continue
            ts=np.array([s['t'] for s in row['trace']]);qs=np.array([s['q'] for s in row['trace']])
            axes[0].plot(ts,qs[:,2],label=label,color=color,lw=2)
            axes[1].plot(ts,np.sqrt(np.mean((qs-np.array([0,0,.6,0,-.5236]))**2,axis=1)),color=color,lw=2)
        axes[0].axhline(.6,ls='--',color='black',lw=1);axes[0].set(ylabel='Joint 3 [rad]',title=f'Measured trace: seed {start}')
        axes[1].set(xlabel='Simulation time [s]',ylabel='RMSE [rad]')
        axes[0].legend(fontsize=15,loc='lower right')
        for ax in axes:ax.grid(alpha=.2)
        fig.savefig(out/'measured_trace.png',dpi=180);plt.close(fig)
    print('Presentation inputs copied without modification; readable derived figures saved',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base',type=Path);p.add_argument('out',type=Path);a=p.parse_args();build(a.base,a.out)
