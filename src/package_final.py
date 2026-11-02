"""Assemble an auditable local delivery; never publishes to external services."""
from pathlib import Path
import json,hashlib,zipfile
import re

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'deliverables'; E=ROOT/'experiments/evidence'
result=json.loads((E/'final_results.json').read_text())
stats=json.loads((E/'statistical_summary.json').read_text())
verification=json.loads((E/'local_integrity_verification.json').read_text())
a,b=stats['baseline'],stats['selected']
assert verification['selected_checkpoint_hash']=='passed'
readme=(ROOT/'README.md').read_text()
start=readme.index('## 実測の進捗' if '## 実測の進捗' in readme else '## 実測結果');end=readme.index('## 工夫と、比較から分かったこと')
rows=['| 比較した設定 | 検証成功数 | 終端RMSE [rad] |','|---|---:|---:|']
for r in result['validation']:
    rows.append(f"| `{r['name'].removeprefix('val_')}` | {r['successes']}/{r['n']} | {r['terminal_rmse_mean']:.5f} |")
summary=f'''## 実測結果

初回はGoogle ColabのTesla T4で6モデルを学習し、実行方法を含めた9設定を比較しました。初回の重みと生軌跡は保存待機中のタイムアウトで失われましたが、集計はDriveから保全しています。今回は選択設定を固定して基準・選択モデルを再学習し、新しい50条件（seed 4000〜4049）で測定しました。設定は `{result['selection']['trial']}`, `n_action_steps={result['selection']['n_action_steps']}`, `ensemble={result['selection']['ensemble']}` です。

**未使用の50条件で、成功数は基準モデル {a['successes']}/50 → 選択したACT {b['successes']}/50でした。** 同じ初期条件の組で改善{stats['paired_gains']}件、悪化{stats['paired_regressions']}件。McNemarの正確検定（両側）はp={stats['mcnemar_exact_two_sided_p']:.6g}です。

| 最終評価 | 成功数 | 成功率の95% Wilson区間 | 実行エラー |
|---|---:|---:|---:|
| 基準モデル | {a['successes']}/50 | {a['wilson95'][0]*100:.1f}〜{a['wilson95'][1]*100:.1f}% | {a['technical_errors']} |
| 選択ACT | {b['successes']}/50 | {b['wilson95'][0]*100:.1f}〜{b['wilson95'][1]*100:.1f}% | {b['technical_errors']} |

'''+ '\n'.join(rows)+'''

上の表は再実行の確認用20条件です。設定は初回の検証で成功数、次に終端の平均関節RMSEで選びました。再実行では学習前にその設定を固定し、最終評価前に新しいチェックポイントのSHA-256を保存しました。結果に基づく再調整は行っていません。

'''
readme=readme[:start]+summary+readme[end:]
readme=readme.replace('ログと全軌跡を別途保存します。','ログと全軌跡を証拠記録ZIPへ保存しました。')
diagnostic=json.loads((E/'normalization_diagnostic.json').read_text())
readme=re.sub(r'第1関節の標準偏差が約.*?radでした。',
    f"第1関節の標準偏差が約{diagnostic['aligned_2000']['state_std'][0]:.3g} radしかなく、少し異なる初期姿勢が正規化後に極端な値になります。多様化した記録では約{diagnostic['diverse_2000']['state_std'][0]:.3g} radでした。いずれも再収集データでの診断値です。",readme)
readme=readme.replace('今回の実行用Colab](https://colab.research.google.com/drive/1qU34Y1vqgWAbxG4mjbzJ1BztLlX-_rny)',
    '再実行用Colab](https://colab.research.google.com/drive/1jrUKSu9rRZgUfIkObGiJFwns8Sk4Cmvs)')
readme=readme.replace('GPUランタイムを選択します。','新しい空のGPUランタイムを選択します。')
readme=readme.split('\n## 完成したファイル')[0]
readme=readme.replace('PAI_ROS2_ACT_reproduce_141960.ipynb','PAI_ROS2_ACT_replication_141960.ipynb').replace('最終確認2000～2049','初回の最終確認2000～2049、再実行の最終確認4000～4049').replace('6モデルのGPU学習、9設定の比較、モデル選択、未使用50条件の評価','固定した2モデルのGPU学習、20条件の再確認、新しい50条件の評価').replace('統合した再現ノートブックを新しい空のランタイムで最初から一括再実行する検証は未実施です。','再実行用ノートブックの各工程は、新しい空のランタイムから実行して確認しました。')
readme+='''
## 完成したファイル

- `ROS2_ACT_141960_report.pdf`：日本語の検証レポート。
- `ROS2_ACT_141960.mp4`：配布OP・EDを使った説明動画。実測関節角の模式再生であり、Gazeboの画面録画ではありません。
- `PAI_ROS2_ACT_replication_141960.ipynb`：今回の固定設定による再学習・評価を再現するColab。
- `PAI_ROS2_ACT_reproduce_141960.ipynb`：初回の6モデル・9設定比較を再現するColab。
- `pai_ros2_act_evidence.zip`：全学習記録・評価軌跡・統計値・ソースハッシュ・環境情報。
- `pai_ros2_act_checkpoints.zip`：基準モデルと選択モデルの重み、モデル情報、実行設定。
- `local_integrity_verification.json`：ダウンロード後の独立検査。全ファイルのハッシュ、データ分割、学習コードとデータの対応、評価コードの対応、成功の停止条件、選択重みのハッシュを確認。

動画・PDFの生成スクリプトは、MacのArial Unicodeフォントを参照しています。資料生成に使ったライブラリのバージョンは `render_requirements.txt` に記録しています。学習・物理シミュレーションはGoogle Colabで実行しました。再実行ノートブックのJSON・セルID・Python構文と、新しいランタイムでの各工程の実行を確認しました。初回の詳細はRECOVERY_NOTE.mdと保存済みの元ノートブックを参照してください。
'''
(ROOT/'README.md').write_text(readme)
(OUT/'local_integrity_verification.json').write_text(json.dumps(verification,indent=2))
# Include only authored code, explicit evidence and deliverables, not full third-party checkouts.
files=[ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md',ROOT/'experiment_protocol.json',ROOT/'PAI_ROS2_ACT_reproduce_141960.ipynb',ROOT/'PAI_ROS2_ACT_replication_141960.ipynb',ROOT/'RECOVERY_NOTE.md',ROOT/'experiments/original_run_saved_redacted.ipynb',ROOT/'experiments/original_notebook_redaction.json']
files += [ROOT/'experiments/original_run_final_results.json',ROOT/'experiments/original_summary_statistics.json']
files.append(ROOT/'render_requirements.txt')
files+=sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'licenses').iterdir())
files += [OUT/name for name in ['ROS2_ACT_141960_report.pdf','ROS2_ACT_141960.mp4','video_manifest.json','local_integrity_verification.json','colab_result.png']]
files += [ROOT/'experiments'/name for name in ['pai_ros2_act_evidence.zip','pai_ros2_act_checkpoints.zip']]
manifest={}
for p in files:
    assert p.is_file(),p
    rel=str(p.relative_to(ROOT))
    # Promote user-facing documents and archives to the ZIP's top level.
    if rel.startswith(('deliverables/','experiments/')):rel=p.name
    manifest[rel]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
assert len(manifest)==len(files),'Archive path collision'
(OUT/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
target=OUT/'ROS2_ACT_141960_complete.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p,rel in zip(files,manifest):
        z.write(p,'ROS2_ACT_141960/'+rel,compress_type=zipfile.ZIP_STORED if p.suffix in {'.zip','.mp4'} else zipfile.ZIP_DEFLATED)
    z.write(OUT/'delivery_manifest.json','ROS2_ACT_141960/delivery_manifest.json')
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    for rel,m in manifest.items():assert hashlib.sha256(z.read('ROS2_ACT_141960/'+rel)).hexdigest()==m['sha256']
sha=hashlib.sha256(target.read_bytes()).hexdigest()
(OUT/'ROS2_ACT_141960_complete.zip.sha256').write_text(f'{sha}  {target.name}\n')
print(json.dumps({'file':str(target),'bytes':target.stat().st_size,'sha256':sha,'files':len(files)},indent=2))
