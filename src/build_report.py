"""Japanese report populated only from completed, exported measurements."""
import argparse
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

NAMES={'lesson_baseline':'教材設定・200 step','order_only_200':'関節順の統一のみ・200 step',
       'time_only_200':'時刻設定のみ変更・200 step','aligned_200':'前処理の整合・200 step',
       'aligned_2000':'前処理の整合・2,000 step','diverse_2000':'初期姿勢を多様化・2,000 step'}

def build(base,out):
    results=json.loads((base/'final_results.json').read_text())
    original=json.loads((base/'original_run_final_results.json').read_text()) if (base/'original_run_final_results.json').exists() else results
    replication=results.get('run_kind')=='fixed_configuration_replication'
    test_seed=results.get('test_start_seed',2000)
    stats=json.loads((base/'statistical_summary.json').read_text())
    selection=json.loads((base/'selection_before_test.json').read_text())
    env=json.loads((base/'environment.json').read_text())
    commits=json.loads((base/'source_commits.json').read_text())
    pdfmetrics.registerFont(TTFont('JP','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    navy=colors.HexColor('#172d43'); teal=colors.HexColor('#087d73')
    styles={
        'title':ParagraphStyle('title',fontName='JP',fontSize=23,leading=33,textColor=navy,spaceAfter=13,wordWrap='CJK'),
        'sub':ParagraphStyle('sub',fontName='JP',fontSize=11,leading=17,textColor=teal,spaceAfter=14,wordWrap='CJK'),
        'body':ParagraphStyle('body',fontName='JP',fontSize=10,leading=16,spaceAfter=8,wordWrap='CJK'),
        'small':ParagraphStyle('small',fontName='JP',fontSize=8.2,leading=12.5,spaceAfter=6,wordWrap='CJK'),
        'h':ParagraphStyle('h',fontName='JP',fontSize=13,leading=20,textColor=teal,spaceBefore=12,spaceAfter=8,wordWrap='CJK',keepWithNext=True),
        'cell':ParagraphStyle('cell',fontName='JP',fontSize=8,leading=12,wordWrap='CJK'),
    }
    def p(text,style='body'): return Paragraph(text,styles[style])
    flow=[]
    def add(text,style='body'): flow.append(p(text,style))
    def table(rows,widths):
        t=Table([[p(str(x),'cell') for x in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e0f1ed')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
            ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,teal),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7fa')])]))
        flow.append(t)
    a,b=stats['baseline'],stats['selected']; selected=results['selection']
    add('ROS 2とACTで、<br/>学習と実行をつなぐ','title')
    add('CRANE+ V2のお辞儀動作：教材設定と改善候補の比較','sub')
    add(f"未使用50条件での成功数：教材設定 {a['successes']}/50 → 選択したACT {b['successes']}/50。",'h')
    add('この結果は、固定された目標姿勢に向かうシミュレーション課題の実測値です。検証用データで選んだ設定を固定してから、別の初期姿勢50条件で比較しました。全モデル・全課題の最適性を示すものではありません。')
    if replication:
        add('初回は25/50→50/50成功しましたが、保存ダイアログの待機中に一時環境が終了し、重みと生の軌跡を失いました。上の数値は、選択設定を固定して再学習し、新しい50条件で確認した結果です。初回の集計はDriveの保存済みノートブックから保全しました。','small')
    add('何を作ったか','h')
    add('指定教材のROS 2／Gazebo／CRANE+ V2とACTを基準に、データ収集、前処理、GPU学習、閉ループ評価、結果集計までを実装しました。観測は5関節の角度で、画像や言語を入力するVLAではありません。学習計算はGoogle ColabのT4 GPUで実行しました。')
    table([['段階','処理と記録'],['収集','公開テレオペの操作量をスクリプトで再現し、Gazeboの実際の関節状態を記録'],['学習','同じACT構造を使い、時刻設定・学習量・初期姿勢の多様性を比較'],['実行','ROS 2の関節軌道指令でロボットを動かし、状態を再観測'],['評価','エピソード全体でデータを分割。失敗・実行エラー・制限介入も残す']],[60,439])
    add('独自に工夫した部分','h')
    add('① 初期姿勢の多様化：固定姿勢の20記録に、姿勢を変えた60記録を追加し、80件すべてで品質と目標到達を確認しました。② 前処理の比較：過去から始まる動作列と現在から始まる列を比較し、50ミリ秒周期・欠測の拒否・末尾データの扱いを明確にしました。③ 実行と評価：実行手数や時間方向の平均化も試し、同じ初期条件で比較しました。モデル選択用の条件と最終確認用の条件は分離しています。')
    add('関節名を明示して並べ替える処理も実装しました。ただし実測の関節順は教材の推論側と一致していました。この環境で関節順の不一致が原因だった、とは結論づけていません。','small')
    flow.append(PageBreak())
    add('比較の条件と、初回の検証結果','title')
    add('学習損失だけで判断せず、実際に動かして選ぶ','sub')
    add('課題は、腕の第3関節を0.60 radへ動かし、他の腕関節を0 rad、グリッパーを-0.5236 radにする動作です。8秒以内に全関節の誤差0.05 rad以下、最大関節速度0.10 rad/s以下を0.5秒保てたら成功とします。時間はGazeboのシミュレーション時刻です。')
    add('学習データは20エピソードから80エピソードへ拡張。後半60件は腕の初期角度を各軸±0.08 radの範囲で変えました。データは人の実演ではなく、公開操作量を参考にしたスクリプトによる実シミュレーション記録です。有限値・関節名・時刻・目標到達を検査しています。')
    rows=[['検証した設定','成功／20','終端RMSE [rad]','実行エラー']]
    for r in original['validation']:
        label=NAMES.get(r['trial'],r['trial'])
        if r['name'].startswith('val_execution_'):
            label+=f"／実行{r['n_action_steps']}手"+("・平均化" if r['ensemble'] is not None else '')
        rmse='未計測' if r['terminal_rmse_mean'] is None else f"{r['terminal_rmse_mean']:.4f}"
        rows.append([label,r['successes'],rmse,r['technical_errors']])
    table(rows,[265,64,94,76])
    add('200 stepの比較はbatch 2、追加学習はbatch 16、学習率は全方式0.0001、動作列長は16です。初期重みの乱数seedを固定しました。検証成功数、次に終端RMSEで選択し、選んだモデルの実行手数1／4／8と時間方向の平均化も比較しました。','small')
    if replication:
        add('表は初回の保存済み集計です。初回の生軌跡と重みは残っていないため、その再照合はできません。再実行では選択済みの設定を変えず、基準モデルと選択モデルを再学習しました。再実行の全記録と重みを今回の成果物に同梱しています。','small')
    add('再現範囲を明確にする','h')
    add('教材のモデル・最適化設定・関節順・時間設定・パディング規則を再現する比較です。元の人手収集rosbagは提供されていないため、独自に収集したJSONとメモリ上のデータローダーを使いました。元教材の実験そのものやLeRobot形式変換との完全一致は主張しません。全方式に同じ関節制限ガードを適用し、介入回数を記録しています。','small')
    oracle=results['task_specific_oracle_validation']
    add(f"参考：目標角度を直接与える課題専用制御は、検証20条件中{oracle['successes']}条件で成功しました。単一の既知目標では学習なしの制御も有効であり、ACTが常に必要だという結論ではありません。",'small')
    flow.append(PageBreak())
    add('未使用条件での最終確認','title')
    add(f"選択設定：{NAMES.get(selected['trial'],selected['trial'])}／実行{selected['n_action_steps']}手"+(f"／平均化係数{selected['ensemble']}" if selected['ensemble'] is not None else ''),'sub')
    flow.append(Image(str(base/'comparison.png'),width=440,height=180,hAlign='CENTER'))
    add('図：左は保存済みの初回9設定の集計、右は設定を固定して再学習した後の新しい50条件での評価。誤差棒は成功率の95% Wilson区間です。','small')
    for label,r in [('教材設定',a),('選択したACT',b)]:
        lo,hi=r['wilson95']; add(f"{label}：{r['successes']}/{r['n']}成功、95%区間 {lo*100:.1f}～{hi*100:.1f}%、実行エラー{r['technical_errors']}件。",'small')
    add(f"同じ初期条件の組で、改善{stats['paired_gains']}件、悪化{stats['paired_regressions']}件。対応のある成功／失敗の正確検定（両側）p={stats['mcnemar_exact_two_sided_p']:.3g}。条件範囲とモデルの初期化seedは限定されており、この値だけで広い汎化性能は保証されません。",'small')
    trade=[['最終評価','終端RMSE [rad]','ジャークRMS','推論往復 [ms]','制限介入']]
    for label,r in zip(['教材設定','選択ACT'],results['test']):
        def fmt(key): return '未計測' if r[key] is None else (f'{r[key]:.4f}' if key=='terminal_rmse_mean' else f'{r[key]:.3f}')
        trade.append([label,fmt('terminal_rmse_mean'),fmt('jerk_rms_mean'),fmt('latency_ms_mean'),r['limit_interventions']])
    table(trade,[79,116,104,114,86])
    add('ジャークの単位はrad/s³。失敗試行も含む平均です。推論往復時間にはROS側からGPU側へのローカル通信を含みます。成功率と滑らかさ・処理時間は別々に評価します。','small')
    flow.append(Image(str(base/'measured_trace.png'),width=246,height=164,hAlign='CENTER'))
    add(f'図：あらかじめ決めた最終評価seed {test_seed}の実測軌跡。成功しやすい例を後から選んだ図ではありません。','small')
    flow.append(PageBreak())
    add('なぜ、追加学習だけでは不十分か','title')
    add('学習時と実行時の入力の違いを調べる','sub')
    diagnostic=json.loads((base/'normalization_diagnostic.json').read_text())
    short=diagnostic['aligned_2000']; varied=diagnostic['diverse_2000']
    add('固定姿勢からの20記録では、ほとんど動かさない関節の標準偏差が極端に小さくなっていました。ACTの入力正規化は「平均を引き、標準偏差＋10⁻⁸で割る」方式です。実行時に少し違う姿勢から始めるだけでも、学習時にない大きさの入力になり得ます。')
    normrows=[['関節','20記録の標準偏差 [rad]','80記録の標準偏差 [rad]']]
    for index in [0,1,3]:
        normrows.append([f'第{index+1}関節',f"{short['state_std'][index]:.3g}",f"{varied['state_std'][index]:.3g}"])
    table(normrows,[70,214,215])
    scale_a=.08/(short['state_std'][0]+1e-8); scale_b=.08/(varied['state_std'][0]+1e-8)
    add(f'例えば第1関節では、初期姿勢の範囲±0.08 radを正規化した幅が、20記録では約{scale_a:.2g}、80記録では約{scale_b:.2f}に相当します。これは変換式から計算した診断値で、別の成功率測定ではありません。','small')
    add('実測結果と解釈を分ける','h')
    lookup={r['trial']:r for r in original['validation'] if not r['name'].startswith('val_execution_')}
    add(f"初回の同じ2,000 step・batch 16の比較では、20記録のモデルは{lookup['aligned_2000']['successes']}/20成功、初期姿勢を変えた80記録のモデルは{lookup['diverse_2000']['successes']}/20成功でした。上の標準偏差は再収集データから計算した診断です。『実行時に出会う状態を学習データに含める』という設計の妥当性を検討する材料にしています。")
    add('ただし、データ量と初期姿勢の多様性を同時に変えています。正規化だけを変える追加実験も行っていないため、改善を一つの原因に断定しません。複数の学習seedや一定量のデータでの比較が、次の検証になります。')
    add('今回の課題から持ち帰れること','h')
    add('学習損失の低下と、ロボットを動かした成功率は別です。品質を確認したデータを使い、失敗しやすい状態を含め、モデルを選ぶ条件と最終評価条件を分けることが重要です。単一の既知目標を動かすだけなら、課題専用の通常制御でも成功できます。ACTの必要性まで一般化しないようにしました。')
    flow.append(PageBreak())
    add('根拠、再現方法、残る課題','title')
    add('結果を追跡できる形で保存する','sub')
    add('再現と確認','h')
    add('再実行のソース一式、Colabノートブック、学習ログ、各エピソードの関節状態と指令、検証結果、最終評価結果、評価前に固定した設定、チェックポイントを保存しています。evidence_manifest.jsonのSHA-256で記録の同一性を確認できます。初回の生軌跡・重みとは区別しています。')
    verification=json.loads((base/'local_integrity_verification.json').read_text())
    delta=verification['max_paired_initial_difference_rad']
    add(f"保存後の独立検査：{verification['teacher_episodes_verified']}件の教師データ、{verification['evaluation_episodes_verified']}件の評価記録を照合しました。成功時の停止条件を軌跡から再計算し、評価時と保存後のモデルのハッシュも確認しています。同じseedで与えた初期姿勢に対する、実測初期角度の方式間最大差は{delta:.3g} radでした。",'small')
    add(f"学習環境：Google Colab {env['gpu']}、PyTorch {env['torch']}。ROS 2 Humble／Ubuntu 22.04の実行環境をColab内に分離し、教材のGazebo世界と物理モデルを使用しました。GUIを省いた起動用launchファイルを追加しています。",'small')
    add('今回言えること・言えないこと','h')
    add('今回比較した範囲では、検証用の成功率と終端誤差に基づく最良設定を選び、別の条件で結果を確認しました。一方、実機、把持、物体移動、視覚認識、未知の目標角度、広い初期姿勢、複数の学習seedでの再現性は評価していません。長い動作や別の課題への有効性は、追加の独立評価が必要です。')
    add('複数の変更をまとめた方式もあるため、全改善量を一つの変更の効果とは呼びません。関節順だけを変えても、入力と初期重みの対応が変わります。この環境には関節順の不一致がなく、単一の学習seedで生じた差を「順序バグの修正効果」とは解釈できません。追加学習とデータ拡張も分けて示しました。')
    flow.append(PageBreak())
    add('参照元と再現の手掛かり','title')
    add('公開資料と、今回使ったソースを区別して記録する','sub')
    add('主要な参照元','h')
    for title,url in [
        ('指定教材：知能ロボットシステムエンジニアリング第5回 ROS 2入門（2/2）','https://skinny-chord-c58.notion.site/5-ROS-2-2-2-2049a7a1061380fc8928e500eafc57d0'),
        ('教材の学習・変換・推論コード','https://github.com/matsuolab/pai_training'),
        ('教材のキーボード操作','https://github.com/matsuolab/physicalai_teleop'),
        ('CRANE+ V2モデルとGazebo構成','https://github.com/AI-Robot-Book-Humble/crane_plus'),
        ('LeRobot ACT実装','https://github.com/huggingface/lerobot'),
        ('ACT原論文：Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware','https://arxiv.org/abs/2304.13705')]:
        add(f'{title}<br/><link href="{url}" color="#087d73">{url}</link>','small')
    add('固定したソースのコミット','h')
    for name,sha in commits.items(): add(f'{name}: {sha}','small')
    add('上流コードの著作権・ライセンス表記を維持しています。提出済みのMuJoCo搬送ロボットとは別の実験成果物です。','small')
    out.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d6e0e8')); c.line(48,40,547,40)
        c.setFont('JP',8);c.setFillColor(navy)
        c.drawString(48,25,'141960 | ROS 2 / ACT | 2026-11-03')
        c.drawRightString(547,25,str(doc.page))
    doc=SimpleDocTemplate(str(out),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=45,bottomMargin=55,
        title='ROS 2とACTで、学習と実行をつなぐ',author='141960')
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    print(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('base',type=Path); ap.add_argument('out',type=Path)
    args=ap.parse_args(); build(args.base,args.out)
