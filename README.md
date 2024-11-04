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
