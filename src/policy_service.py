"""Loopback-only bridge between native Colab CUDA and isolated ROS Python."""
import json
import traceback
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from act_experiment import BASE,PERM,make_policy
from safetensors.torch import load_model

torch.set_num_threads(2)
policy=None; baseline=False

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass

    def do_POST(self):
        global policy,baseline
        try:
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/load':
                trial=data['trial']
                path=(BASE/'results'/trial).resolve()
                if path.parent != (BASE/'results').resolve(): raise ValueError('Invalid trial')
                meta=json.loads((path/'trial.json').read_text())
                policy=make_policy(n_action_steps=data.get('n_action_steps',1),ensemble=data.get('ensemble'))
                load_model(policy,str(path/'checkpoint/model.safetensors'),strict=True)
                policy.eval(); baseline=meta['baseline']; policy.reset(); result={'loaded':trial}
            elif self.path=='/reset':
                policy.reset(); result={'reset':True}
            elif self.path=='/predict':
                x=torch.tensor(data['state'],dtype=torch.float32,device='cuda')
                if baseline: x=x[PERM]  # Reproduce upstream infer.py's name-based input order.
                batch={'observation.state':x[None],'observation.environment_state':x[None]}
                with torch.inference_mode(): action=policy.select_action(batch)[0]
                if baseline: action=action[PERM]  # Canonical physical publisher order.
                result={'action':action.cpu().tolist()}
            else: raise ValueError('Unknown endpoint')
            body=json.dumps(result).encode(); self.send_response(200)
        except Exception:
            body=json.dumps({'error':traceback.format_exc()}).encode(); self.send_response(500)
        self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body)))
        self.end_headers(); self.wfile.write(body)

if __name__=='__main__':
    print('POLICY_SERVICE_READY',flush=True)
    HTTPServer(('127.0.0.1',8765),Handler).serve_forever()
