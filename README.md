# 概要
こちらは自作の[テトリスシミュレータ](https://github.com/char5742/Tetris)上で動作する**テトリスAI**です。

## セットアップ
```console
git clone https://github.com/char5742/TetrisAI
julia --project
]
(TetrisAI) pkg> instantiate
```

## 動作方法
```console
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
```console
while ($true) { Get-Content "board.txt"; Start-Sleep -Milliseconds 166 }
```
