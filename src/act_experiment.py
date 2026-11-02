"""Pinned upstream ACT, measured ROS data, episode-level separation, CUDA.

Data loader is a transparent in-memory adapter (no image/video data). The lesson
baseline reproduces its model, optimizer, steps, batching, joint-order and target
offset conventions, rather than claiming byte-identical ROS bag conversion.
"""
import argparse
import json
import random
import sys
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

BASE=Path('/content/pai_ros2')
sys.path.insert(0,str(BASE/'pai_training/lerobot'))
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType,PolicyFeature
from data_alignment import causal_sample,uniform_times

PERM=[0,1,3,2,4]  # Upstream teleoperation and infer.py arm joint ordering.
TRIALS={
 'lesson_baseline':{'steps':200,'batch':2,'episodes':20,'baseline':True},
 'order_only_200':{'steps':200,'batch':2,'episodes':20,'baseline':False,'legacy_time':True,'legacy_padding':True},
 'time_only_200':{'steps':200,'batch':2,'episodes':20,'baseline':True,'legacy_time':False,'legacy_padding':True},
 'aligned_200':{'steps':200,'batch':2,'episodes':20,'baseline':False},
 'aligned_2000':{'steps':2000,'batch':16,'episodes':20,'baseline':False},
 'diverse_2000':{'steps':2000,'batch':16,'episodes':80,'baseline':False},
}

class RecordedDataset(Dataset):
    def __init__(self, directory, cfg):
        self.baseline=cfg['baseline']; self.episodes=[]; self.index=[]
        legacy_time=cfg.get('legacy_time',self.baseline)
        legacy_padding=cfg.get('legacy_padding',self.baseline)
        for seed in range(cfg['episodes']):
            obj=json.loads((Path(directory)/f'episode_{seed:05d}.json').read_text())
            if not(obj['teacher'] and obj['success'] and obj['valid']):
                raise ValueError(f'Quality-gate rejected episode {seed}')
            trace=obj['trace']
            times=np.array([r['t'] for r in trace])
            if np.max(np.diff(times))>.075:
                raise ValueError(f'Irregular recording cadence: seed {seed}')
            states=np.array([r['raw'] if self.baseline else r['q'] for r in trace],dtype=np.float32)
            actions=np.array([r['action'] for r in trace],dtype=np.float32)
            if states.shape[1]!=5: raise ValueError('Unexpected JointState shape')
            if self.baseline: actions=actions[:,PERM]
            times_ns=np.rint(times*1e9).astype(np.int64)
            grid=(np.linspace(times_ns[0],times_ns[-1],int((times[-1]-times[0])*20)).astype(np.int64)
                  if legacy_time else uniform_times(times_ns[0],times_ns[-1],20))
            states,_=causal_sample(times_ns,states,grid,max_age_ns=75_000_000)
            actions,_=causal_sample(times_ns,actions,grid,max_age_ns=75_000_000)
            self.episodes.append([states,actions])
        if legacy_padding:
            maxlen=max(len(s) for s,a in self.episodes)
            for pair in self.episodes:
                for j in range(2):
                    pair[j]=np.pad(pair[j],((0,maxlen-len(pair[j])),(0,0)),mode='edge')
        for ep,(states,actions) in enumerate(self.episodes):
            self.index.extend((ep,i) for i in range(len(states)))
        self.offsets=np.arange(-2,30,2) if legacy_time else np.arange(16)
        self.stats={}
        for key, col in [('observation.state',0),('observation.environment_state',0),('action',1)]:
            arr=np.concatenate([pair[col] for pair in self.episodes])
            self.stats[key]={'mean':arr.mean(0),'std':arr.std(0),'min':arr.min(0),'max':arr.max(0)}

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        ep,i=self.index[index]; states,actions=self.episodes[ep]
        indices=i+self.offsets
        # LeRobot masks out-of-episode action queries.
        pad=(indices<0)|(indices>=len(actions))
        a=actions[np.clip(indices,0,len(actions)-1)]
        s=torch.from_numpy(states[i].copy())
        return {'observation.state':s,'observation.environment_state':s.clone(),
                'action':torch.from_numpy(a.copy()),'action_is_pad':torch.from_numpy(pad)}

def make_policy(stats=None, n_action_steps=1, ensemble=None):
    cfg=ACTConfig(input_features={
        'observation.state':PolicyFeature(FeatureType.STATE,(5,)),
        'observation.environment_state':PolicyFeature(FeatureType.ENV,(5,))},
        output_features={'action':PolicyFeature(FeatureType.ACTION,(5,))},
        chunk_size=16,n_action_steps=n_action_steps,temporal_ensemble_coeff=ensemble,device='cuda')
    return ACTPolicy(cfg,dataset_stats=stats).cuda()

def train(name,directory,seed=2026):
    cfg=TRIALS[name]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    dataset=RecordedDataset(directory,cfg)
    policy=make_policy(dataset.stats); policy.train()
    optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4)
    loader=DataLoader(dataset,batch_size=cfg['batch'],shuffle=True,num_workers=0,drop_last=True)
    out=BASE/'results'/name; out.mkdir(parents=True,exist_ok=True)
    metadata={**cfg,'seed':seed,'lr':1e-4,'samples':len(dataset),'torch':torch.__version__,
              'raw_joint_names':json.loads((Path(directory)/'episode_00000.json').read_text())['trace'][0]['raw_names'],
              'normalization':'train episodes only, population mean/std, no variance floor',
              'loader_deviation':'in-memory 20Hz ROS records; no rosbag/LeRobot disk conversion'}
    metadata['training_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['data_sha256']={f'episode_{s:05d}.json':hashlib.sha256((Path(directory)/f'episode_{s:05d}.json').read_bytes()).hexdigest() for s in range(cfg['episodes'])}
    (out/'trial.json').write_text(json.dumps(metadata,indent=2))
    tick=time.monotonic(); step=0
    with (out/'loss.jsonl').open('w') as log:
        while step<cfg['steps']:
            for batch in loader:
                batch={k:v.cuda() for k,v in batch.items()}
                loss, info=policy.forward(batch)
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at {step}')
                optimizer.zero_grad(); loss.backward(); optimizer.step(); step+=1
                row={'step':step,'loss':float(loss.detach()),'seconds':time.monotonic()-tick}
                log.write(json.dumps(row)+'\n')
                if step%100==0: print(name,json.dumps(row),flush=True); log.flush()
                if step>=cfg['steps']: break
    policy.eval(); policy.save_pretrained(out/'checkpoint')
    metadata.update({'training_seconds':time.monotonic()-tick,'final_loss':row['loss'],
                     'parameters':sum(p.numel() for p in policy.parameters()),'complete':True})
    (out/'trial.json').write_text(json.dumps(metadata,indent=2))
    print('TRAIN_COMPLETE',name,flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--trial',choices=list(TRIALS)+['all'],default='all')
    p.add_argument('--data',default=str(BASE/'data/train')); args=p.parse_args()
    for name in (TRIALS if args.trial=='all' else [args.trial]):
        finished=BASE/'results'/name/'trial.json'
        if finished.exists() and json.loads(finished.read_text()).get('complete'):
            meta=json.loads(finished.read_text())
            if meta['training_source_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
                raise RuntimeError('Training source changed: use a new trial name')
            for filename,digest in meta['data_sha256'].items():
                if hashlib.sha256((Path(args.data)/filename).read_bytes()).hexdigest()!=digest:
                    raise RuntimeError('Training data changed: use a new trial name')
            continue
        train(name,args.data)

if __name__=='__main__': main()
