"""Integration smoke test only; synthetic data are never performance evidence."""
import json
import time
import numpy as np
import torch
from act_experiment import make_policy,BASE

torch.manual_seed(2026); torch.set_num_threads(2)
stats={k:{'mean':np.zeros(5,dtype=np.float32),'std':np.ones(5,dtype=np.float32)}
       for k in ['observation.state','observation.environment_state','action']}
start=time.monotonic(); p=make_policy(stats)
batch={'observation.state':torch.zeros((2,5),device='cuda'),
       'observation.environment_state':torch.zeros((2,5),device='cuda'),
       'action':torch.zeros((2,16,5),device='cuda'),
       'action_is_pad':torch.zeros((2,16),dtype=torch.bool,device='cuda')}
p.train(); loss,_=p(batch); assert torch.isfinite(loss); loss.backward()
p.eval(); p.reset()
with torch.inference_mode(): a=p.select_action(batch)
assert a.shape==(2,5) and torch.isfinite(a).all()
result={'status':'passed','purpose':'synthetic integration smoke test, NOT accuracy evaluation',
        'device':torch.cuda.get_device_name(0),'loss':float(loss.detach()),
        'elapsed_seconds':time.monotonic()-start}
(BASE/'gpu_smoke.json').write_text(json.dumps(result,indent=2))
print('GPU_SMOKE_OK',json.dumps(result),flush=True)
