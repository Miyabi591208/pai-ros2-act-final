"""Actual Gazebo/ROS feedback collection and closed-loop evaluation (Python 3.10).

Teacher reproduces keyboard increments; it is not a human demonstration.
Results are simulator measurements, not real-robot performance.
"""
import argparse
import json
import time
import traceback
import urllib.request
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from data_alignment import JOINTS, reorder

TARGET = np.array([0., 0., .6, 0., -.5236])
LOW = np.deg2rad(np.array([0,45.45,3.52,44.57,109.38])-150)
HIGH = np.deg2rad(np.array([300,252.20,290.62,251.32,188.27])-150)

def rpc(path, payload):
    req=urllib.request.Request('http://127.0.0.1:8765/'+path,
        data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=30) as f:
        return json.load(f)

class Robot(Node):
    def __init__(self):
        super().__init__('pai_measured_trials')
        self.set_parameters([Parameter('use_sim_time', value=True)])
        self.q=None; self.raw=None; self.names=None; self.stamp=None
        self.create_subscription(JointState,'/joint_states',self.receive,10)
        self.arm=self.create_publisher(JointTrajectory,'/crane_plus_arm_controller/joint_trajectory',10)
        self.hand=self.create_publisher(JointTrajectory,'/crane_plus_gripper_controller/joint_trajectory',10)

    def receive(self,msg):
        self.q=reorder(msg.name,msg.position).astype(float)
        self.raw=list(msg.position); self.names=list(msg.name)
        self.stamp=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9

    def now(self):
        return self.get_clock().now().nanoseconds/1e9

    def wait(self, until, wall_timeout=45):
        deadline=time.monotonic()+wall_timeout
        while self.now()<until or self.q is None:
            if time.monotonic()>deadline:
                raise RuntimeError('Simulation clock or JointState timeout')
            rclpy.spin_once(self,timeout_sec=.01)
        # Consume queued feedback; reject a stalled feedback stream.
        for _ in range(3): rclpy.spin_once(self,timeout_sec=0.)
        if self.stamp is None or self.now()-self.stamp>.20:
            raise RuntimeError('JointState older than 200 ms')

    def command(self,q):
        if not np.isfinite(q).all(): raise ValueError('Non-finite action')
        for pub,names,values in [(self.arm,JOINTS[:4],q[:4]),(self.hand,JOINTS[4:],q[4:])]:
            msg=JointTrajectory(); msg.joint_names=list(names)
            point=JointTrajectoryPoint(); point.positions=[float(v) for v in values]
            point.time_from_start.nanosec=100_000_000
            msg.points=[point]; pub.publish(msg)

    def reset_to(self,q):
        start=self.now(); prev=self.q.copy(); stable_since=None
        for k in range(100):
            self.command(q); self.wait(start+(k+1)*.05)
            stationary=np.max(np.abs(self.q-prev))/.05<.05
            if np.max(np.abs(self.q-q))<.005 and stationary:
                if stable_since is None: stable_since=self.now()
                if self.now()-stable_since>=.3: return
            else: stable_since=None
            prev=self.q.copy()
        raise RuntimeError('Reset did not stably reach requested initial posture')

def episode(robot, seed, teacher=False, oracle=False):
    rng=np.random.default_rng(seed)
    # The first twenty episodes reproduce the lesson's zero-start data.
    q0=np.r_[np.zeros(4) if teacher and seed<20 else rng.uniform(-.08,.08,4),-.5236]
    robot.reset_to(q0)
    if not teacher and not oracle: rpc('reset',{})
    initial=robot.q.copy(); start=robot.now(); hold_start=None
    trace=[]; limits=0; completion=None; prev_q=robot.q.copy(); prev_t=start
    interval=float(rng.uniform(.10,.17)) if teacher else None
    for k in range(160):
        scheduled=start+k*.05
        robot.wait(scheduled)
        t=robot.now(); q=robot.q.copy(); raw=robot.raw.copy()
        if t-start>8.0:
            break
        velocity=(q-prev_q)/max(t-prev_t,1e-6) if k else np.zeros(5)
        if np.max(np.abs(q-TARGET))<=.05 and np.max(np.abs(velocity))<=.10:
            if hold_start is None: hold_start=t
            if t-hold_start>=.5 and completion is None: completion=t-start
        else: hold_start=None
        latency=0.
        if oracle:
            action=TARGET.copy()
        elif teacher:
            count=min(12,int((t-start)/interval)+1)
            fraction=count/12
            # 0.05-rad target increments, plus recovery from varied initial posture.
            action=q0+(TARGET-q0)*fraction
        else:
            tic=time.perf_counter(); action=np.asarray(rpc('predict',{'state':q.tolist()})['action'])
            latency=(time.perf_counter()-tic)*1000
        violation=bool(np.any(action<LOW)|np.any(action>HIGH))
        limits+=int(violation)
        # Hard joint limits shared by every method; all interventions counted.
        applied=np.clip(action,LOW,HIGH)
        robot.command(applied)
        trace.append({'t':t-start,'q':q.tolist(),'raw':raw,'raw_names':robot.names,
                      'action':action.tolist(),'applied':applied.tolist(),'latency_ms':latency,
                      'feedback_age_s':t-robot.stamp,'schedule_delay_s':t-scheduled})
        prev_q=q; prev_t=t
        # Keep 0.5 s of stable terminal data in demonstrations, already required by success.
        if completion is not None: break
    qs=np.array([r['q'] for r in trace]); ts=np.array([r['t'] for r in trace])
    valid=bool(np.isfinite(qs).all() and np.all(np.diff(ts)>0))
    jerk=np.gradient(np.gradient(np.gradient(qs,ts,axis=0),ts,axis=0),ts,axis=0)
    result={'seed':seed,'teacher':teacher,'oracle':oracle,'success':completion is not None,'valid':valid,
            'initial':initial.tolist(),'terminal_rmse_rad':float(np.sqrt(np.mean((qs[-1]-TARGET)**2))),
            'completion_sim_seconds':completion,'joint_jerk_rms':float(np.sqrt(np.mean(jerk[3:-3]**2))),
            'command_limit_violations':limits,'inference_latency_ms_mean':float(np.mean([r['latency_ms'] for r in trace])),
            'max_schedule_delay_s':max(r['schedule_delay_s'] for r in trace),'trace':trace}
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True)
    p.add_argument('--teacher',action='store_true'); p.add_argument('--start',type=int,default=0)
    p.add_argument('--oracle',action='store_true')
    p.add_argument('--count',type=int,default=20); args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rclpy.init(); robot=Robot(); robot.wait(1.,wall_timeout=120)
    print('JOINT_STATE_ORDER',robot.names,flush=True)
    for seed in range(args.start,args.start+args.count):
        target=out/f'episode_{seed:05d}.json'
        if target.exists(): continue
        tic=time.monotonic()
        try:
            result=episode(robot,seed,args.teacher,args.oracle)
        except Exception:
            result={'seed':seed,'teacher':args.teacher,'success':False,'valid':False,
                    'error':traceback.format_exc(),'trace':[]}
        result['wall_seconds']=time.monotonic()-tic
        temp=target.with_suffix('.tmp'); temp.write_text(json.dumps(result)); temp.replace(target)
        print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
        if args.teacher and not result['valid']:
            raise RuntimeError('Teacher collection failed; no training on rejected data')
    robot.destroy_node(); rclpy.shutdown()

if __name__=='__main__': main()
