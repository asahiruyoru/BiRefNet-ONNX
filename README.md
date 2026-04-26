# birefnet_onnx

[BiRefNet](https://github.com/ZhengPeng7/BiRefNet)（高精度な前景/背景セグメンテーションモデル）の ONNX 版を、Gradio の Web UI から呼び出すだけのシンプルなツールです。

入力画像から **マスク / 前景（背景透過）/ 背景（前景透過）** の 3 種を生成します。

## 必要なもの

- Python 3.10+
- ONNX 実行環境（CPU だけでも動きます。GPU を使う場合は CUDA 対応版 onnxruntime）
- BiRefNet の ONNX モデルファイル `birefnet_1024x1024.onnx`（**リポジトリには含めません**。後述の手順で用意）

### 依存パッケージ

```bash
pip install opencv-python numpy gradio
# CPU のみ
pip install onnxruntime
# GPU を使う場合（CUDA 対応 onnxruntime）
pip install onnxruntime-gpu
```

## ONNX モデルの準備

`app.py` は同じディレクトリにある `birefnet_1024x1024.onnx` を読み込みます。以下のいずれかで用意してください。

### 方法 A: 自分で変換する（推奨）

同梱の [`birefnet_Convert2ONNX.ipynb`](birefnet_Convert2ONNX.ipynb) を Google Colab で開いて上から実行すると `birefnet_1024x1024.onnx` が生成されます。生成されたファイルを本リポジトリのルートに配置してください。

ノートブックの先頭セルにある `IMSIZE` を変えれば任意の解像度（例: `1344`）でも変換できます。その場合は後述のとおり `app.py` 側の `imsize` も合わせてください。

### 方法 B: 既存の ONNX を持っている場合

ファイル名を `birefnet_1024x1024.onnx` にして、`app.py` と同じディレクトリに置くだけで OK です。

## 実行方法

```bash
python app.py
```

起動するとローカルに Gradio サーバが立ち上がり、コンソールに `http://127.0.0.1:7860` のような URL が表示されます。ブラウザでアクセスしてください。

### 使い方

1. **Input image** に画像をアップロード
2. **背景の扱い** を選択
   - `original` … 透過部分は元画像の色のまま
   - `black` … 透過部分を黒で塗る
   - `white` … 透過部分を白で塗る
3. **Generate** をクリック

出力は 3 つ表示されます。

| 出力 | 内容 |
|---|---|
| Output image | 2 値マスク（前景=白、背景=黒） |
| foreground image | 前景のみ抽出（背景は透過 / 黒 / 白） |
| background image | 背景のみ抽出（前景は透過 / 黒 / 白） |

## 解像度を変えたい場合

`app.py` の末尾近くの `imsize = 1024` を、用意した ONNX のサイズに合わせて変更してください。たとえば 1344 の ONNX を使うなら:

```python
imsize = 1344
onnx_session = load_model(imsize)
```

ONNX には入力解像度が焼き込まれているため、ファイル名の数字と `imsize` は必ず一致させる必要があります。

## GPU について

`app.py` は `CUDAExecutionProvider` を優先し、無ければ `CPUExecutionProvider` に自動フォールバックします。GPU で動かしたい場合は CUDA 対応版の `onnxruntime-gpu` をインストールし、対応する CUDA / cuDNN を環境にセットアップしてください。

## 上流リポジトリと本フォークの違い

本リポジトリは [Kazuhito00/BiRefNet-ONNX-Sample](https://github.com/Kazuhito00/BiRefNet-ONNX-Sample) （MIT License, © Kazuhito Takahashi）からフォークし、用途を Gradio Web UI に絞って作り直したものです。

| 項目 | 上流 (Kazuhito00) | 本フォーク |
|---|---|---|
| UI / 入力 | `demo_onnx.py`（OpenCV、Web カメラ・動画ファイルのリアルタイム推論） | `app.py`（Gradio Web UI、画像 1 枚） |
| 出力 | マスクを重ねた 1 枚 | **3 種**：バイナリマスク / 前景抽出 / 背景抽出 |
| 背景の扱い | なし | UI で `original` / `black` / `white` を選択 |
| 引数 | `--device` `--movie` `--model` `--score_th` | なし（UI から操作） |
| ONNX 変換ノートブック | `Convert2ONNX.ipynb` | `birefnet_Convert2ONNX.ipynb`（上流をベースに、パラメータ集約・冪等パッチ・出力削除などで整理） |
| 同梱サンプル画像 | あり（ぱくたそ） | なし |

なお、上流の `demo_onnx.py` / `LICENSE` / `model/` / `sample.jpg` 等は本フォークでは削除しています（git 履歴上は派生関係が残っています）。

## Reference

- [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — モデル本体
- [Kazuhito00/BiRefNet-ONNX-Sample](https://github.com/Kazuhito00/BiRefNet-ONNX-Sample) — 上流（ONNX 変換手順とパッチを参考）
- [masamitsu-murase/deform_conv2d_onnx_exporter](https://github.com/masamitsu-murase/deform_conv2d_onnx_exporter) — DeformConv2d を ONNX に書き出すためのヘルパー
