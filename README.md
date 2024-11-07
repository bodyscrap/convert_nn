# Pytorch モデル → candle モデルへの変換サンプル

## 1. uv による環境構築
Pytorchのcuda版を使用する環境をuvで構築する。  

### 1.1. pytorchのインストールの詳細を観る
[バージョン履歴のページ](https://pytorch.org/get-started/previous-versions/)で確認。  
今回は2.5.0を入れたいので2.5.0のwheelを確認。  

```
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124/torch_stable.html
```

そして、[こちらを](https://zenn.dev/mjun0812/articles/b32f870bb3cdbf)参考に、pyproject.tomlを編集し、uv syncで環境を合わせる…先人偉い…。  

## 2. PythonでのPytorchモデルの作成
`mnist_sample.py`が作成したファイルです。  
シンプルなCNNによるMNIST画像分類モデルです。  

main()の引数で、学習するか、推論するかを選択できるようにしています。  
また、推論時は読み込むデータが`safetensors`か`pth`かを選択できるようにしています。  
ちなみに、学習時は冗長ですが両フォーマットで保存しています。  

細かい説明についてはコード中のコメントを参照してください。

## 3. Rustでのcandleモデルの作成
`lib.rs`がモデルの定義、`main.rs`が推論処理のサンプルです。  
モチベーションが「世にあるPyTorchの学習モデルを使いたい!」なので、`pth`形式のファイルもサポートしています。  
でも、色々あって最近は`safetensors`形式のファイルが推奨されているので、`pth`を一度なんらかの方法で`safetensors`に変換して保存しなおして推論時は`safetensors`を読み込むようにする運用が良いかもしれません。  

こちらも、詳細はコード中のコメントを参照してください。  
1つハマりポイントとして、candleのリファレンス([例えばcandle-nnのConv2d](https://docs.rs/candle-nn/latest/candle_nn/conv/struct.Conv2d.html))を見ると、殆ど何も書いて居ないです。  
各パラメータの説明やデフォルト値などは、PyTorchのリファレンスを参照してください。  
基本的にcandleはPyTorchの置き換えを狙っているのでクラス名やパラメータ名はPyTorchとそろえてあるので、名前で引きましょう。  
