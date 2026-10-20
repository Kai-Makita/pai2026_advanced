# Mobile Manipulation IL/RL Benchmark

講義用にゼロから作成した、架空の差動二輪モバイルマニピュレータによるPick & Place比較実験です。外部のロボットモデル、CAD、研究固有コードは使っていません。MJCFは`configs/default.yaml`の寸法・質量・関節制限から実行時に生成されます。

## 構成と設計

- 物理: MuJoCo 500 Hz、制御20 Hz。差動二輪、受動球キャスタ、Yaw–Pitch–Pitch–Pitchアーム、平行2指グリッパ、2机、自由物体を含みます。
- 把持: `physical`が既定で、左右指との接触と摩擦で把持します。`assisted`はデバッグ専用の明示的なweld切替です。結果にはモード名を保存し、混同を防ぎます。
- タスク: 6つの学習対象フェーズとSUCCESS/FAILUREを有限状態機械で管理します。移動時はアーム、操作時は台車をマスクします。
- 学習: BC、ランダム初期化PPO、BCでActorだけを初期化したPPOを同一の40次元観測・7次元行動で比較します。
- 出力: 実行ごとに`outputs/<command>_<timestamp>/`を作成し、設定、seed、モデル、評価JSON/CSV、図、動画を上書きせず保存します。

## Ubuntuセットアップ

```bash
cd mobile_manipulation_il_rl
bash scripts/setup_venv.sh
source .venv/bin/activate
pytest -q
```

ヘッドレス描画では環境に応じて`MUJOCO_GL=egl`（GPU）または`MUJOCO_GL=osmesa`（CPU）を設定してください。すべてのCLI例はプロジェクト直下で実行します。

## 再現手順

まず解析教師を100エピソード評価します。データ収集前に成功率と失敗理由を確認し、低い場合は`evaluation.json`の失敗分類を基にモデルまたは制御設定を調整します。

```bash
python -m mobile_manipulation_il_rl.cli --seed 42 teacher-test --episodes 100
python -m mobile_manipulation_il_rl.cli --seed 42 collect --episodes 400 --dataset datasets/default
python -m mobile_manipulation_il_rl.cli --seed 42 train-bc --dataset datasets/default
```

パイプラインだけを先に検証する場合は、サブコマンドより前に`--grasp-mode assisted`を指定します（例: `... --grasp-mode assisted teacher-test --episodes 100`）。設定と動画名にモードが残ります。

続いてPPOとBC→PPOを同じ追加ステップ数で学習します。`RUN`とモデルパスは実際の時刻付き出力へ置き換えます。

```bash
python -m mobile_manipulation_il_rl.cli --seed 42 train-ppo --timesteps 100000
python -m mobile_manipulation_il_rl.cli --seed 42 train-bc-ppo --bc-model outputs/RUN/bc_best.pt --timesteps 100000
```

チェックポイントからの再開には`--resume outputs/RUN/checkpoints/ppo_..._steps.zip`を使います。BC→PPO実行時は、移植前後の同一観測に対する行動平均の最大誤差を`transfer_verification.txt`へ記録します。Actorのみを移植し、CriticはPPO初期値のままです。対数標準偏差の初期値はYAMLで指定します。

評価、3 seed比較、動画生成はそれぞれ1コマンドです。

```bash
python -m mobile_manipulation_il_rl.cli evaluate --policy bc --model outputs/BC_RUN/bc_best.pt --episodes 100
python -m mobile_manipulation_il_rl.cli compare --bc-model outputs/BC_RUN/bc_best.pt --ppo-model outputs/PPO_RUN/model_final.zip --bc-ppo-model outputs/BCPPO_RUN/model_final.zip --seeds 11 22 33 --episodes 100
python -m mobile_manipulation_il_rl.cli video --policy teacher
python -m mobile_manipulation_il_rl.cli viewer --policy teacher --speed 1.0
streamlit run mobile_manipulation_il_rl/dashboard/app.py
```

Physical把持の段階診断は次の1コマンドで実行します。各episodeの全状態トレース、フェーズイベント、接触点・力、IK残差、関節余裕、集計CSV/JSON/PNGを固有runディレクトリへ保存します。

```bash
python -m mobile_manipulation_il_rl.cli --seed 42 physical-debug --episodes 5
```

`evaluate`は既定で学習範囲と範囲外（OOD）の両方を実行します。比較CSVには成功率、成功時完了時間、落下率、評価時間、seed、条件を保存し、各評価JSONにはフェーズ到達率、失敗理由、全報酬項を保存します。TensorBoardは各PPO runの`tensorboard/`を参照します。

## 観測、行動、座標

世界座標はx–y平面、z上向きです。移動目標は台車座標へ回転し、目標方位はsin/cosで表します。それ以外の位置・速度は世界座標、関節値は各関節座標です。40次元ベクトルは固定物理スケールで正規化し、有限値検査とクリップを行います。行動は`[v, ω, Δq1, Δq2, Δq3, Δq4, gripper]`の全要素が[-1, 1]です。

## 報酬設計

進捗項は各フェーズのpotential（移動距離、手先距離、持上げ高さ、配置距離）の1 step差です。絶対距離を毎step加算しないため、目標近傍で滞留して報酬を稼ぐ挙動を抑えます。フェーズが変わるstepではpotentialの定義が変わるため進捗差を0にし、完了ボーナスだけを与えます。

| 項 | 初期値 | 目的 |
|---|---:|---|
| progress | 1.0 × potential差 | 密な学習信号。典型的に1 step当たり小数以下 |
| phase completion | +2 | 一方向のFSM遷移を評価 |
| task success | +20 | 最終目的を支配する疎な報酬 |
| time | -0.005/step | 速い完了を選好 |
| smoothness | -0.001 × 行動差二乗 | 急な指令を抑制 |
| object drop | -10 | 把持後の落下を抑制 |

全項は`info["reward_terms"]`と累積`reward_sums`に独立記録されます。Potential-based shapingは設定項として用意しており、重みはすべてYAMLで変更できます。報酬ハッキングは、無操作・往復動作・教師動作の累積内訳、フェーズ遷移が一方向であること、成功判定が物体位置とリリースを要求することを評価時に確認します。

## 設定と実験上の注意

`configs/default.yaml`で寸法、質量、摩擦ランダム化範囲、アクチュエータ制限、報酬、タイムアウト、学習ハイパーパラメータを管理します。既定の机と停止距離は、全リンク長0.44 mに対して0.36 mのstand-offで、保守的到達範囲内か起動時に検査します。

実験結果はリポジトリに同梱していません。成功率や完了時間は上記コマンドで生成されたJSON/CSVのみを根拠にしてください。物理把持は接触条件に敏感なので、まず教師テストを行い、成功率が不足する場合は動画と失敗理由を確認してください。`assisted`の結果は物理把持の結果として報告しないでください。

## テスト範囲

`pytest`はMJCFロード、Gymnasium `check_env`、40/7次元、有限値、seed再現性、重力下安定性、FSM、BC→PPO行動平均一致を検査します。短時間の学習smoke testと保存・再読込は依存環境上で実行できますが、100k step以上の本実験、100教師episode、400デモepisode、3 seed比較は計算時間を要するため明示的なCLI実行にしています。

