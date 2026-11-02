"""Run after bootstrap_colab.py, in the same Colab kernel."""
from pathlib import Path
import subprocess
import json
import sys

BASE = Path('/content/pai_ros2')
prefix = json.loads((BASE/'proot_prefix.json').read_text())

def run(args, name):
    print(name, flush=True)
    with (BASE/f'{name}.log').open('w') as log:
        p = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        print((BASE/f'{name}.log').read_text()[-8000:], flush=True)
        raise RuntimeError(name)
    print(name+' OK', flush=True)

repos = [('https://github.com/AI-Robot-Book-Humble/crane_plus.git',
          BASE/'ws/src/crane_plus', '19c1d6069126cd27488bf1bbefd410704bfc515d'),
         ('https://github.com/matsuolab/pai_training.git',
          BASE/'pai_training','b4d6b5459066a80433475a83567504988875248a')]
for url, path, sha in repos:
    if not (path/'.git').exists():
        run(['git','clone',url,str(path)],'clone_'+path.name)
    run(['git','-C',str(path),'checkout',sha],'checkout_'+path.name)
run(['git','-C',str(BASE/'pai_training'),'submodule','update','--init','lerobot'],'lerobot_source')
# Keep the course robot, controllers and world intact; only suppress the GUI.
launch = BASE/'ws/src/crane_plus/crane_plus_gazebo/launch/no_moveit_crane_plus_with_table.launch.py'
text = launch.read_text().replace("'ign gazebo -r'", "'ign gazebo -r -s'")
launch.with_name('headless_crane_plus.launch.py').write_text(text)
run(prefix+['bash','-c','source /opt/ros/humble/setup.bash && cd /work/ws && colcon build --packages-select crane_plus_description crane_plus_control crane_plus_moveit_config crane_plus_gazebo --executor sequential --cmake-args -DBUILD_TESTING=OFF'],'build_crane')
with (BASE/'robot.urdf').open('w') as output:
    subprocess.run(prefix+['bash','-c','source /opt/ros/humble/setup.bash && source /work/ws/install/setup.bash && xacro /work/ws/install/crane_plus_description/share/crane_plus_description/urdf/crane_plus.urdf.xacro use_gazebo:=true use_ignition:=true gz_control_config_package:=crane_plus_control gz_control_config_file_path:=config/crane_plus_controllers.yaml'],stdout=output,check=True)
# Training uses Colab's native CUDA stack; ROS remains isolated in Ubuntu 22.04.
run([sys.executable,'-m','pip','install','-q','draccus==0.10.0','jsonlines','datasets<4','einops','safetensors','huggingface-hub<1','deepdiff','termcolor'],'act_dependencies')
sys.path.insert(0,str(BASE/'pai_training/lerobot'))
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
print('ACT_IMPORT_OK',torch.__version__,torch.cuda.get_device_name(0),flush=True)
(BASE/'environment.json').write_text(json.dumps({'torch':torch.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name(0)},indent=2))
print('EXPERIMENT_SETUP_COMPLETE',flush=True)
