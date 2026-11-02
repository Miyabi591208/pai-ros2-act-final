"""Evidence-based explanatory video; schematic replay, not Gazebo screen capture."""
import argparse,json,math,hashlib,subprocess,tempfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg

FONT='/System/Library/Fonts/Supplemental/Arial Unicode.ttf'
BG='#f3f7fa'; NAVY='#172d43'; TEAL='#087d73'; GRAY='#707c89'
W,H=1280,720
F={n:ImageFont.truetype(FONT,n) for n in [18,22,26,30,36,44,52,76]}

def rot(axis,a):
    axis=np.asarray(axis);axis=axis/np.linalg.norm(axis)
    x,y,z=axis; K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+math.sin(a)*K+(1-math.cos(a))*(K@K)

def skeleton(root,q):
    names=['crane_plus_joint1','crane_plus_joint2','crane_plus_joint3','crane_plus_joint4','crane_plus_joint_hand']
    angles=dict(zip(names,q)); transforms={'world':np.eye(4)}
    pending=list(root.findall('joint'))
    for _ in range(len(pending)+1):
        for j in pending[:]:
            par=j.find('parent').get('link'); child=j.find('child').get('link')
            if par not in transforms: continue
            origin=j.find('origin'); xyz=[0,0,0];rpy=[0,0,0]
            if origin is not None:
                xyz=[float(x) for x in origin.get('xyz','0 0 0').split()]
                rpy=[float(x) for x in origin.get('rpy','0 0 0').split()]
            m=np.eye(4);m[:3,3]=xyz
            m[:3,:3]=rot([0,0,1],rpy[2])@rot([0,1,0],rpy[1])@rot([1,0,0],rpy[0])
            if j.get('type')=='revolute':
                ax=[float(x) for x in j.find('axis').get('xyz').split()]
                m[:3,:3]=m[:3,:3]@rot(ax,angles.get(j.get('name'),0))
            transforms[child]=transforms[par]@m;pending.remove(j)
        if not pending: break
    chain=['crane_plus_base','crane_plus_link1','crane_plus_link2','crane_plus_link3','crane_plus_link4','crane_plus_link_hand']
    return np.array([transforms[x][:3,3] for x in chain])

def frame(title,number):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,12),fill=TEAL)
    d.text((60,44),title,font=F[44],fill=NAVY)
    d.line((60,650,1220,650),fill='#c7d6e1',width=2)
    d.text((60,674),'141960  |  ROS 2 / ACT  |  Google Colab T4',font=F[18],fill=GRAY)
    d.text((1120,674),f'{number} / 5',font=F[18],fill=GRAY)
    return im,d

def lines(d,text,x,y,size=30,color=NAVY,gap=48):
    for row in text.split('\n'):
        d.text((x,y),row,font=F[size],fill=color);y+=gap

def fitted(im,path,box):
    pic=Image.open(path).convert('RGB');pic.thumbnail((box[2]-box[0],box[3]-box[1]))
    im.paste(pic,(box[0]+(box[2]-box[0]-pic.width)//2,box[1]+(box[3]-box[1]-pic.height)//2))

def draw_robot(d,root,q,center,color,width=12):
    xyz=skeleton(root,q)
    # Orthographic projection of the published URDF link origins.
    points=[(center[0]+950*(p[0]-.6*p[1]),center[1]-950*(p[2]+.25*p[1])) for p in xyz]
    d.line(points,fill=color,width=width,joint='curve')
    for x,y in points: d.ellipse((x-7,y-7,x+7,y+7),fill=color)
    return points

def build(base,out,opening=None,ending=None):
    stats=json.loads((base/'statistical_summary.json').read_text());res=json.loads((base/'final_results.json').read_text())
    root=ET.parse(base/'robot.urdf').getroot()
    test_seed=res.get('test_start_seed',2000)
    traces=[json.loads((base/'evaluation'/name/f'episode_{test_seed:05d}.json').read_text()) for name in ['test_baseline','test_selected']]
    if not all(r.get('trace') for r in traces): raise ValueError('Preselected seed has no measured trace; do not fabricate replay')
    out.mkdir(parents=True,exist_ok=True);core=out/'explanation_core.mp4'
    fps=15; durations=[12,17,18,15,18];starts=np.cumsum([0]+durations).tolist()
    with imageio.get_writer(str(core),fps=fps,codec='libx264',quality=8,macro_block_size=16,ffmpeg_params=['-movflags','+faststart']) as writer:
        for n in range(sum(durations)*fps):
            t=n/fps;scene=next(i for i in range(5) if t<starts[i+1]);local=t-starts[scene]
            if scene==0:
                im,d=frame('ROS 2とACTで、学習と実行をつなぐ',1)
                lines(d,'CRANE+ V2のお辞儀動作を、\n学習・実行・評価までつなげて改善する。',65,170,36,gap=62)
                a=stats['baseline']['successes'];b=stats['selected']['successes']
                d.rounded_rectangle((65,335,1210,525),24,fill='#e0f1ed')
                d.text((100,355),'未使用の初期姿勢50条件で比較',font=F[30],fill=TEAL)
                d.text((100,415),f'{a}/50  →  {b}/50 成功',font=F[76],fill=NAVY)
                lines(d,'教材設定を基準にした比較。実機の性能ではありません。',65,565,26)
                d.text((65,610),'初回の集計は別保存。この動画は固定設定で再実行した結果です。',font=F[22],fill=GRAY)
            elif scene==1:
                im,d=frame('工夫：データの多様化と、実行評価',2)
                entries=[('01','実行時の姿勢を学習データへ','品質確認済みの20→80記録。初期姿勢を変えた60件を追加'),('02','動作ラベルとデータの整形','現在からの50 ms刻み・欠測の拒否・末尾の扱いを比較'),('03','モデルを実際に動かして選ぶ','初回の選択設定を固定し、新しい50条件で再確認')]
                for k,(num,title,body) in enumerate(entries):
                    y=155+k*148
                    d.text((70,y),num,font=F[44],fill=TEAL)
                    d.text((170,y),title,font=F[36],fill=NAVY)
                    d.text((170,y+57),body,font=F[26],fill=GRAY)
            elif scene==2:
                im,d=frame('比較：一つの指標だけで判断しない',3)
                fitted(im,base/'comparison.png',(40,135,1240,560))
                lines(d,'左：初回9設定の集計   右：再学習後の新しい50条件',70,574,26)
                d.text((70,610),'誤差棒は成功率の95%区間。全失敗・実行エラーも集計。',font=F[22],fill=GRAY)
            elif scene==3:
                im,d=frame('実測した動き：同じ初期姿勢で比較',4)
                d.text((65,118),f'seed {test_seed}の記録を模式再生（Gazeboの画面録画ではありません）',font=F[22],fill=GRAY)
                simt=max(0.,min(8.,local-2.))
                for idx,(row,label,color) in enumerate(zip(traces,['教材設定','選択したACT'],[GRAY,TEAL])):
                    cx=330+idx*620
                    d.text((cx-160,180),label,font=F[36],fill=color)
                    ts=np.array([r['t'] for r in row['trace']]);qs=np.array([r['q'] for r in row['trace']])
                    q=np.array([np.interp(simt,ts,qs[:,j]) for j in range(5)])
                    d.line((cx-170,540,cx+180,540),fill='#bbcbd7',width=3)
                    draw_robot(d,root,[0,0,.6,0,-.5236],(cx,535),'#cbd8df',5)
                    draw_robot(d,root,q,(cx,535),color)
                    d.text((cx-160,565),f'第3関節 {q[2]:.3f} rad / 目標 0.600',font=F[22],fill=NAVY)
                    d.text((cx-160,605),f'時刻 {min(simt,ts[-1]):.2f} s'+('  記録終了' if simt>=ts[-1] else ''),font=F[22],fill=GRAY)
            else:
                im,d=frame('確認できたことと、次に残る課題',5)
                b=stats['selected'];lo,hi=b['wilson95']
                lines(d,f'選択したACT：{b["successes"]}/50 成功\n95%区間：{lo*100:.1f}〜{hi*100:.1f}%\n初回の検証で選び、設定を固定して再学習。',65,155,30,gap=57)
                d.rounded_rectangle((65,365,1210,585),22,fill='#e5edf3')
                lines(d,'固定目標・限定した初期姿勢・単一の学習seedでの結果。\n実機、把持、未知目標、広い姿勢への汎化は未検証。\nコード・全試行・モデル・再現用Colabを保存。',90,398,26,gap=57)
            writer.append_data(np.asarray(im))
    finish_video(out,core,opening,ending,durations,starts,test_seed)

def finish_video(out,core,opening,ending,durations,starts,test_seed):
    """Join already rendered chapters and verify the complete encoded video."""
    target=out/'ROS2_ACT_141960.mp4'; inputs=[core]
    composition=None
    if opening and ending:
        inputs=[opening,core,ending];meta=[]
        for p in inputs:
            reader=imageio.get_reader(str(p));meta.append(reader.get_meta_data());reader.close()
        if any(m.get('audio_codec') for m in meta): raise ValueError('Supplied assets have audio; mixing requires explicit handling')
        ds=[math.ceil(float(m['duration'])*30-1e-8)/30 for m in meta];ds[1]=sum(durations)
        fade=.4;offsets=[ds[0]-fade,ds[0]+ds[1]-2*fade];filters=[]
        for i,dur in enumerate(ds):
            filters.append(f'[{i}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1,tpad=stop_mode=clone:stop_duration=1,trim=duration={dur},setpts=PTS-STARTPTS,fps=30,format=yuv420p[v{i}]')
        filters += [f'[v0][v1]xfade=transition=fade:duration=0.4:offset={offsets[0]},fps=30[first]',f'[first][v2]xfade=transition=fade:duration=0.4:offset={offsets[1]},format=yuv420p[final]']
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y']
        for p in inputs:cmd+=['-i',str(p)]
        cmd+=['-filter_complex',';'.join(filters),'-map','[final]','-an','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
        subprocess.run(cmd,check=True)
        composition={'fade_seconds':fade,'op_seconds':ds[0],'ed_seconds':ds[2],'expected_seconds':sum(ds)-.8}
    else:
        import shutil;shutil.copyfile(core,target)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(target),'-f','null','-'],check=True)
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(target))
    if composition: assert abs(seconds-composition['expected_seconds'])<.15
    manifest={'title':'ROS 2とACTで、学習と実行をつなぐ','id':'141960','frames':frames,'seconds':seconds,
              'audio':'none; supplied OP/ED silent','core_seconds':sum(durations),'core_chapter_start_seconds':starts[:-1],
              'replay':f'Measured test seed {test_seed}, published URDF forward kinematics, schematic replay; no new simulator measurements',
              'composition':composition,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (out/'video_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # Inspection sheet samples every chapter, including the motion panel.
    reader=imageio.get_reader(str(target));sheet=Image.new('RGB',(960,810),'white')
    offset=composition['op_seconds']-.4 if composition else 0
    samples=[offset+4,offset+16,offset+34,offset+52,offset+59,offset+72]
    actualfps=reader.get_meta_data()['fps']
    for i,second in enumerate(samples):
        pic=Image.fromarray(reader.get_data(round(second*actualfps))).resize((480,270))
        sheet.paste(pic,((i%2)*480,(i//2)*270))
    reader.close();sheet.save(out/'video_contact_sheet.png')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('base',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--opening',type=Path);ap.add_argument('--ending',type=Path)
    a=ap.parse_args();build(a.base,a.out,a.opening,a.ending)
