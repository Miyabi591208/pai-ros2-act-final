"""Post-training diagnostic, no policy mutation or model selection."""
import json
import numpy as np
from act_experiment import BASE,RecordedDataset,TRIALS

out={}
for name in ['aligned_2000','diverse_2000']:
    dataset=RecordedDataset(BASE/'data/train',TRIALS[name])
    stats=dataset.stats['observation.state']
    out[name]={'state_mean':stats['mean'].tolist(),'state_std':stats['std'].tolist(),
               'uniform_initial_half_width_rad':[.08,.08,.08,.08,0.],
               'normalizer_epsilon':1e-8,
               'normalized_half_width':(np.array([.08,.08,.08,.08,0.])/(stats['std']+1e-8)).tolist()}
(BASE/'normalization_diagnostic.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
