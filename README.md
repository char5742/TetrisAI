# 概要
こちらは自作の[テトリスシミュレータ](https://github.com/char5742/Tetris)上で動作する**テトリスAI**です。

## セットアップ
```bash
git clone https://github.com/char5742/TetrisAI
julia --project
]
(TetrisAI) pkg> instantiate
```

## 学習方法
```bash
# サーバーの起動
julia --project src/server/main.jl
# 学習器の起動
julia --project src/learner/main.jl
# 経験収集機の起動
julia --project src/learner/main.jl 0 0.0 true
# ARGS:
# id: 収集機を区別するためのid.
# ε: 探索率
# use_gpu: gpuを使用するかどうか
```
### 収集機のidについて
id = 0 の場合、log.csvに結果を保存します。

id = 1 の場合、通常速度でプレイし`board.txt`に画面を出力します。
Windowsであれば以下のスクリプトで可視化できます。
```bash
while ($true) { Get-Content "board.txt"; Start-Sleep -Milliseconds 166 }
```

## 学習済みモデルの解析
`/TetrisAI/src/visualizer`にて、モデルの解析をすることができます。

```bash
julia --project src/visualizer/play.jl
# GUIでAIにテトリスをプレイさせることができます。
# ESCキーで終了。

julia --project src/visualizer/analyze.jl 0
# NEXT使用数: 0
# NEXT使用数を指定して、モデルの性能変化を調べることができます。
# 実行結果は'output/'に保存されます。
# ESCキーで終了。
```
上記スクリプトを実行するには、`/TetrisAI/src/visualizer/play`内に実行対象の`layers.jl`, `network.jl`, `mainmodel.jld2`を用意する必要があります。

また`/TetrisAI/src/visualizer/sample.ipyenb`にて、実験で使用したNEXTの解析を行うことができます。


## 分散学習について
Actor、Learner、Server間はHTTPを用いて通信しているため、複数台のマシンで分担して学習を行うことができます。その際は各`main.jl`内の上部にある`server`のアドレスを変更する必要があります。
```julia
const ROOT = "http://127.0.0.1:10513"
# ↓
const ROOT = "http://<Serverのアドレス>:<指定のポート番号>"
```
また、`/TetrisAI/src/server/main.jl`内の`PORT`を変更することで、Serverのポート番号を変更することができます。

ローカルネットワークではなく、インターネットを介して学習を行う場合は、パラメータの送受信間隔を適宜調整する必要がありあます。
単一のマシン内に複数のActorを立ち上げる場合は`/TetrisAI/src/actor/updater.jl`及び`/TetrisAI/src/actor/main_online.jl`を使用することで、パラメータの受信を１本化し通信量を下げることができます。
```bash
julia --project src/actor/updater.jl
# 一定間隔でパラメータを取得し、ローカルに保存する。

julia --project src/actor/main_online.jl 0 0.1 true
# 使い方は'/TetrisAI/src/actor/main.jl'と同じ。
# ただし、'/TetrisAI/src/actor/main.jl'とは異なり、パラメータを直接Serverから取得せず、updaterがローカルに保存したパラメータを読み取る。
```