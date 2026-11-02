"""Create a clean reproducibility notebook from the exact reviewed source files."""
from pathlib import Path
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'
cells=[]
def md(text): cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
def code(text): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':text.splitlines(keepends=True)})

md('# ROS 2とACTで、学習と実行をつなぐ\n\n教材設定を基準にした、Colabでの再現実験です。提出済みのMuJoCo搬送作品とは別です。\n\n「ランタイム → ランタイムのタイプを変更」でGPUを選択し、上から順に実行してください。実行環境の初回準備には時間がかかります。既存契約の範囲で利用し、追加購入を自動で行う処理はありません。\n\nこのノートブックはソースから生成した**再現用**で、結果を装った事前出力は含みません。実測結果は別添のevidence ZIPに保存しています。')
code('import sys,subprocess\nfrom pathlib import Path\nimport torch\nassert Path("/content").exists(), "Google Colabで実行してください"\nassert torch.cuda.is_available(), "GPUランタイムへ接続してください"\nprint(sys.version,torch.__version__,torch.cuda.get_device_name(0))')
md('## 1. ROS 2 Humble用の環境をColab内に準備\nColab本体の新しいUbuntuと、教材のUbuntu 22.04を分離します。署名検証は有効のまま使います。')
code((SRC/'bootstrap_colab.py').read_text())
md('## 2. 教材の固定バージョンを取得してロボットを構築\nロボットモデル、コントローラー、Gazebo世界は教材のものです。GUIのみ省いて起動します。')
code((SRC/'setup_experiment_colab.py').read_text())
md('## 3. 独自実装を保存\n各セルは表示されたPythonファイルを保存します。コードの内容を確認できるよう、圧縮せず記載しています。')
code('SRC=Path("/content/pai_ros2/src"); SRC.mkdir(exist_ok=True)')
for name in ['data_alignment.py','test_data_alignment.py','act_experiment.py','policy_service.py','ros_trials.py','start_colab_runtime.py','smoke_gpu.py','run_comparison.py','analyze_results.py','summarize_normalization.py','package_artifacts.py']:
    code(f'%%writefile /content/pai_ros2/src/{name}\n'+(SRC/name).read_text())
protocol=json.loads((ROOT/'experiment_protocol.json').read_text())
code('import json\nprotocol='+repr(protocol)+'\n(BASE/"experiment_protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2))')
md('## 4. 前処理とGPUの動作確認\n人工データによるGPUテストは、精度評価とは区別します。')
code('subprocess.run([sys.executable,"-m","unittest","discover","-s",str(SRC),"-p","test_data_alignment.py"],check=True)\nsubprocess.run([sys.executable,str(SRC/"smoke_gpu.py")],check=True)')
md('## 5. 実シミュレーションの起動確認とデータ収集\nseed 9999は環境確認専用です。学習はseed 0～79を使います。人手の実演ではなく、公開キーボード操作量を参考にしたスクリプトで収集します。')
code('subprocess.run([sys.executable,str(SRC/"start_colab_runtime.py")],check=True)')
code('subprocess.run(prefix+["bash","-c","source /opt/ros/humble/setup.bash && source /work/ws/install/setup.bash && python3 /work/src/ros_trials.py --teacher --start 0 --count 80 --out /work/data/train"],check=True)')
md('## 6. 比較学習とモデル選択\n学習seedは2026。検証は初期姿勢seed 1000～1019、最終評価は2000～2049です。検証成功数、次に終端RMSEで選びます。全方式で成功が得られない場合は、最終評価に進まず停止します。再学習や実行コードを変えた結果を、古い評価結果と混ぜないための検査も入っています。')
code('subprocess.run([sys.executable,str(SRC/"run_comparison.py")],check=True)')
md('## 7. 実測結果の集計と保存\n最終評価の成功率には95%区間を付けます。これは固定課題・限定した初期姿勢での結果で、実機や別の動作の汎化性能を示すものではありません。')
code('assert (BASE/"final_results.json").exists(), "検証段階で改善が必要です。最終評価結果はまだありません"\nsubprocess.run([sys.executable,str(SRC/"analyze_results.py"),str(BASE)],check=True)\nsubprocess.run([sys.executable,str(SRC/"summarize_normalization.py")],check=True)\nsubprocess.run([sys.executable,str(SRC/"package_artifacts.py")],check=True)\nprint((BASE/"final_results.json").read_text())')
code('print("左のファイル欄で pai_ros2 / exports を開き、2つのZIPを順にダウンロードしてください。保存先の確認画面が出たら完了してください。")\nfor p in sorted((BASE/"exports").glob("*.zip")):\n    print(p.name,p.stat().st_size,"bytes")')
md('## 参照\n- [指定教材](https://skinny-chord-c58.notion.site/5-ROS-2-2-2-2049a7a1061380fc8928e500eafc57d0)\n- [教材コード](https://github.com/matsuolab/pai_training)\n- [CRANE+ V2](https://github.com/AI-Robot-Book-Humble/crane_plus)\n- [LeRobot](https://github.com/huggingface/lerobot)\n- [ACT原論文](https://arxiv.org/abs/2304.13705)\n\n上流コードの著作権・ライセンスを維持します。原教材のrosbag変換そのものの完全再現ではなく、実測JSONを使うメモリ上のローダーによる比較です。')
for i,cell in enumerate(cells):
    cell['id']=hashlib.sha256((str(i)+''.join(cell['source'])).encode()).hexdigest()[:12]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'accelerator':'GPU','kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'}},'cells':cells}
out=ROOT/'PAI_ROS2_ACT_reproduce_141960.ipynb'
out.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
print(out)
