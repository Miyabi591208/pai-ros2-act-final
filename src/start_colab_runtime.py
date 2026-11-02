"""Start the course Gazebo world, then collect one non-evaluation smoke episode."""
from pathlib import Path
import subprocess
import json
import os

BASE=Path('/content/pai_ros2')
prefix=json.loads((BASE/'proot_prefix.json').read_text())
setup='source /opt/ros/humble/setup.bash && source /work/ws/install/setup.bash && '
assert (BASE/'ws/install/setup.bash').exists(), 'Build must succeed first'
assert (BASE/'ws/install/crane_plus_gazebo/share/crane_plus_gazebo/launch/headless_crane_plus.launch.py').exists(), 'Gazebo package must build successfully'
log=(BASE/'gazebo.log').open('w')
p=subprocess.Popen(prefix+['bash','-c',setup+'ros2 launch crane_plus_gazebo headless_crane_plus.launch.py'],
                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(BASE/'gazebo_process.json').write_text(json.dumps({'pid':p.pid,'command':'headless_crane_plus.launch.py'}))
print('GAZEBO_PID',p.pid,flush=True)
with (BASE/'probe.log').open('w') as f:
    probe=subprocess.run(prefix+['bash','-c',setup+'python3 /work/src/ros_trials.py --teacher --start 9999 --count 1 --out /work/data/probe'],stdout=f,stderr=subprocess.STDOUT)
print((BASE/'probe.log').read_text()[-5000:],flush=True)
if probe.returncode:
    print((BASE/'gazebo.log').read_text()[-8000:],flush=True)
    raise RuntimeError('Gazebo/ROS probe failed; no training started')
result=json.loads((BASE/'data/probe/episode_09999.json').read_text())
assert result['success'] and result['valid'], 'Teacher must solve the task first'
print('GAZEBO_ROLLOUT_SMOKE_OK',flush=True)
