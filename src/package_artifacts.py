"""Export measured evidence separately from large checkpoints for reliable saving."""
import hashlib
import json
import subprocess
import zipfile
import importlib.metadata
from pathlib import Path

BASE=Path('/content/pai_ros2'); EXPORT=BASE/'exports'; EXPORT.mkdir(exist_ok=True)
metadata={}
for name,path in [('pai_training',BASE/'pai_training'),('lerobot',BASE/'pai_training/lerobot'),('crane_plus',BASE/'ws/src/crane_plus'),('proot',BASE/'proot-src')]:
    metadata[name]=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
(BASE/'source_commits.json').write_text(json.dumps(metadata,indent=2))
versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','numpy','datasets','huggingface-hub','draccus','jsonlines','safetensors','einops']}
(BASE/'environment_packages.json').write_text(json.dumps(versions,indent=2))
prefix=json.loads((BASE/'proot_prefix.json').read_text())
ros_versions=subprocess.check_output(prefix+['dpkg-query','-W','-f=${Package} ${Version}\n','ros-humble-*'],text=True)
(BASE/'ros_package_versions.txt').write_text(ros_versions)
selection=json.loads((BASE/'selection_before_test.json').read_text())
(BASE/'selected_policy.json').write_text(json.dumps({
    **selection['selected'],'checkpoint_sha256':selection['checkpoint_sha256'],
    'load_instruction':'Use policy_service.py /load with trial, n_action_steps and ensemble from this file. The training checkpoint config alone does not encode selected execution overrides.'
},indent=2))
files=[]
for pattern in ['*.json','*.png','*.log','*.txt','*.urdf','src/*.py','data/**/*.json','evaluation/**/*.json','results/*/*.json','results/*/*.jsonl']:
    files.extend(BASE.glob(pattern))
files=sorted(set(p for p in files if p.is_file() and p.name!='evidence_manifest.json'))
manifest={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(BASE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
files.append(BASE/'evidence_manifest.json')
with zipfile.ZipFile(EXPORT/'pai_ros2_act_evidence.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files: z.write(p,p.relative_to(BASE))
selected=selection['selected']['trial']
with zipfile.ZipFile(EXPORT/'pai_ros2_act_checkpoints.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for trial in sorted({'lesson_baseline',selected}):
        for p in (BASE/'results'/trial/'checkpoint').iterdir():
            if p.is_file(): z.write(p,p.relative_to(BASE))
        p=BASE/'results'/trial/'trial.json'; z.write(p,p.relative_to(BASE))
    z.write(BASE/'selection_before_test.json','selection_before_test.json')
    z.write(BASE/'selected_policy.json','selected_policy.json')
for p in EXPORT.iterdir(): print(p.name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
