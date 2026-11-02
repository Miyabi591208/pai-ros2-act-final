# 参照元とライセンスの扱い

この成果物の独自実装は `src/` 内です。教材のコードやモデルは、再現用ノートブックが以下の公式リポジトリから取得します。上流リポジトリのライセンス・著作権表示を維持し、それぞれの利用条件に従ってください。教材全体の再配布許諾を、このファイルで付与するものではありません。

| 参照元 | 固定したコミット | 役割 |
|---|---|---|
| matsuolab/pai_training | b4d6b5459066a80433475a83567504988875248a | 教材の学習・変換・推論設定 |
| huggingface/lerobot | 768e36660d1408c71118a2760f831c037fbfa17d | ACT実装。上流LICENSEはApache 2.0 |
| matsuolab/physicalai_teleop | 0ef2a00fd6d7bb39377969d10f3d26268e447fd9 | キーボード操作量・関節選択順の参照 |
| AI-Robot-Book-Humble/crane_plus | 19c1d6069126cd27488bf1bbefd410704bfc515d | ロボットモデル・コントローラー・Gazebo世界 |

- 指定教材: https://skinny-chord-c58.notion.site/5-ROS-2-2-2-2049a7a1061380fc8928e500eafc57d0
- ACT原論文: https://arxiv.org/abs/2304.13705
- ROS 2: https://docs.ros.org/en/humble/
- PRoot: https://github.com/proot-me/proot （v5.5.0、実際のコミットは保存したsource_commits.json）

動画のOP・EDはユーザー提供の課題用素材です。素材単体はコードの配布ZIPへ含めません。動画内のロボットは、上流URDFの関節位置と実測した関節角から作る模式再生です。Gazeboの画面を録画したものではありません。
