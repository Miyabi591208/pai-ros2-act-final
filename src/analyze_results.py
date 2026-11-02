"""Auditable held-out statistics and plots; no invented or imputed scores."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

def wilson(k,n):
    if not n: raise ValueError('No trials')
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (0. if k==0 else max(0.,center-half)),(1. if k==n else min(1.,center+half))

def analyze(base):
    result=json.loads((base/'final_results.json').read_text())
    test_start=result.get('test_start_seed',2000)
    a=[json.loads((base/'evaluation/test_baseline'/f'episode_{s:05d}.json').read_text()) for s in range(test_start,test_start+50)]
    b=[json.loads((base/'evaluation/test_selected'/f'episode_{s:05d}.json').read_text()) for s in range(test_start,test_start+50)]
    assert [x['seed'] for x in a]==[x['seed'] for x in b]
    gains=sum(not x['success'] and y['success'] for x,y in zip(a,b))
    regressions=sum(x['success'] and not y['success'] for x,y in zip(a,b))
    n=gains+regressions
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(gains,regressions)+1))/2**n) if n else 1.
    stats={'paired_gains':gains,'paired_regressions':regressions,'mcnemar_exact_two_sided_p':p,
           'scope':'50 paired initial postures from U(-0.08,0.08), fixed task and simulator; no real-robot claim'}
    for name,records in [('baseline',a),('selected',b)]:
        successes=sum(x['success'] for x in records)
        completion=[x['completion_sim_seconds'] for x in records if x['success']]
        stats[name]={'successes':successes,'n':len(records),'wilson95':wilson(successes,len(records)),
                     'technical_errors':sum('error' in x for x in records),
                     'completion_sim_seconds_mean_successes_only':float(np.mean(completion)) if completion else None}
    (base/'statistical_summary.json').write_text(json.dumps(stats,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    rows=result['validation']; labels=[r['name'].removeprefix('val_') for r in rows]
    axes[0].barh(labels,[r['successes']/r['n'] for r in rows],color='#357ebd')
    axes[0].set(xlim=(0,1.05),xlabel='Success rate (validation)',title='Validation: 20 initial postures')
    for i,r in enumerate(rows): axes[0].text(r['successes']/r['n']+.01,i,f"{r['successes']}/{r['n']}",va='center',fontsize=8)
    scores=[stats[k]['successes']/stats[k]['n'] for k in ['baseline','selected']]
    intervals=np.array([stats[k]['wilson95'] for k in ['baseline','selected']])
    axes[1].bar(['Lesson configuration','Selected ACT'],scores,color=['#7d8894','#138a72'])
    axes[1].errorbar([0,1],scores,yerr=np.maximum(0,np.array([scores-intervals[:,0],intervals[:,1]-scores])),fmt='none',color='black',capsize=5)
    axes[1].set(ylim=(0,1.15),ylabel='Success rate (test)',title='Untouched test: 50 paired initial postures')
    axes[1].text(.5,1.08,'Bars: observed rate; whiskers: 95% Wilson interval',ha='center',fontsize=8)
    fig.savefig(base/'comparison.png',dpi=180); plt.close(fig)
    # Representative seed is predetermined, never selected for appearance.
    fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
    for records,label in [(a,'Lesson configuration'),(b,'Selected ACT')]:
        row=records[0]
        if not row.get('trace'): continue
        t=np.array([r['t'] for r in row['trace']]); q=np.array([r['q'] for r in row['trace']])
        axes[0].plot(t,q[:,2],label=label)
        err=np.sqrt(np.mean((q-np.array([0,0,.6,0,-.5236]))**2,axis=1))
        axes[1].plot(t,err,label=label)
    axes[0].axhline(.6,color='black',ls='--',lw=1,label='Target')
    axes[0].set(ylabel='Joint 3 (rad)',title=f'Measured Gazebo trace, preselected test seed {test_start}')
    axes[1].set(xlabel='Simulation time (s)',ylabel='Joint RMSE (rad)')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.savefig(base/'measured_trace.png',dpi=180); plt.close(fig)
    print(json.dumps(stats,indent=2))
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('base',type=Path); args=p.parse_args(); analyze(args.base)
