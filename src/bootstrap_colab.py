"""Prepare isolated Ubuntu 22.04 ROS Humble userspace on a Colab VM.

No local Docker, no account tokens, no external logging or publication.
Run only on the disposable Colab VM; downloads are from Ubuntu/ROS upstream.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import urllib.request

BASE = Path('/content/pai_ros2')
ROOT = BASE/'jammy'
BASE.mkdir(exist_ok=True)


def run(args, name, env=None):
    print(name, flush=True)
    logfile = BASE/f'{name}.log'
    with logfile.open('w') as f:
        result = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        print(logfile.read_text(errors='replace')[-10000:], flush=True)
        raise RuntimeError(f'{name}: exit {result.returncode}')
    print(f'{name}: OK', flush=True)


run(['apt-get', 'update', '-qq'], 'host_apt_update')
run(['apt-get', 'install', '-y', '-qq', 'git', 'xz-utils', 'libtalloc-dev', 'build-essential', 'pkg-config'], 'host_proot_dependencies')
if not (BASE/'proot-src/.git').exists():
    run(['git','clone','--depth','1','--branch','v5.5.0','https://github.com/proot-me/proot.git',str(BASE/'proot-src')], 'proot_source')
run(['make','-C',str(BASE/'proot-src/src'),'-j2','proot'], 'proot_build')
name = 'ubuntu-22.04-minimal-cloudimg-amd64-root.tar.xz'
url = 'https://cloud-images.ubuntu.com/minimal/releases/jammy/release/'
if not (ROOT/'bin/bash').exists():
    print('Downloading official Ubuntu 22.04 rootfs', flush=True)
    sums = urllib.request.urlopen(url+'SHA256SUMS').read().decode()
    expected = next(line.split()[0] for line in sums.splitlines() if line.split()[-1].lstrip('*') == name)
    archive = BASE/name
    urllib.request.urlretrieve(url+name, archive)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert actual == expected, 'Ubuntu rootfs checksum mismatch'
    ROOT.mkdir(exist_ok=True)
    run(['tar', '-xJf', str(archive), '-C', str(ROOT), '--exclude=dev/*'], 'unpack_jammy')
    (BASE/'rootfs_manifest.json').write_text(json.dumps({'url':url+name,'sha256':actual},indent=2))
resolver = ROOT/'etc/resolv.conf'
if resolver.is_symlink():
    resolver.unlink()
resolver.write_text(Path('/etc/resolv.conf').read_text())
(ROOT/'usr/sbin/policy-rc.d').write_text('#!/bin/sh\nexit 101\n')
(ROOT/'usr/sbin/policy-rc.d').chmod(0o755)
(ROOT/'etc/apt/sources.list').write_text('deb https://archive.ubuntu.com/ubuntu jammy main universe restricted multiverse\ndeb https://archive.ubuntu.com/ubuntu jammy-updates main universe restricted multiverse\ndeb https://security.ubuntu.com/ubuntu jammy-security main universe restricted multiverse\n')
prefix = [str(BASE/'proot-src/src/proot'),'-0','-r',str(ROOT),'-b','/dev','-b','/proc','-b','/sys','-b',f'{BASE}:/work','-w','/root','/usr/bin/env','-i','HOME=/root','PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin','DEBIAN_FRONTEND=noninteractive','LANG=C.UTF-8','LC_ALL=C.UTF-8']
(BASE/'proot_prefix.json').write_text(json.dumps(prefix))
run(prefix+['bash','-c','apt-get update -qq && apt-get install -y -qq ca-certificates curl gnupg git python3-pip python3-venv build-essential cmake xvfb'], 'jammy_tools')
# Official apt transport with an HTTPS-fetched key. Signature verification stays enabled.
run(prefix+['bash','-c','curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list && apt-get update --error-on=any -qq'], 'ros_repository')
run(prefix+['apt-get','install','-y','-qq','ros-humble-ros-base','ros-humble-ros-gz','ros-humble-ign-ros2-control','ros-humble-ros2-controllers','ros-humble-xacro','ros-humble-robot-state-publisher','ros-humble-dynamixel-sdk','python3-colcon-common-extensions'], 'ros_gazebo')
run(prefix+['bash','-c','source /opt/ros/humble/setup.bash && python3 -c "import rclpy; print(rclpy.__file__)" && ign gazebo --versions'], 'smoke_ros')
print('ROS_BOOTSTRAP_COMPLETE',flush=True)
