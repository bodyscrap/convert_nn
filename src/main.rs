use anyhow::Result;
use candle_core::{Device, DType, D};
use convert_nn::CNN;
use candle_nn::var_builder::VarBuilder;

fn main() -> Result<()> {
    // デバイスの設定
    let device = Device::cuda_if_available(0)?; // GPUを使用
    // モデルのロード
    let use_safetenors = true;  // safetensors形式のモデル or pth形式のモデルかを切り替え
    let vb = if use_safetenors 
    {
        // safetensors形式のモデル(推奨)をロード
        let model_path = std::path::PathBuf::from("mnist_cnn_weights.safetensors");
        unsafe { 
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? 
        }
    }
    else 
    {
        // pth形式のモデルをロード(過去資産を変換もせずにそのまま使用したい需要への対応)
        let model_path = std::path::PathBuf::from("mnist_cnn_weights.pth");
        VarBuilder::from_pth(model_path, DType::F32, &device)?
    };
    // CNNモデルの作成(レイヤー定義および初期値の設定。本サンプルは直上で読み込んだパラメータで初期化)
    let model = CNN::new(&vb)?;
    // mnistのロード
    // 画像の正規化メソッドが無かったのでそのまま使用。背景0, 文字1の2値画像
    let mnist_data = candle_datasets::vision::mnist::load()?;
    // reshapeの()は元の次元とその他の次元のサイズしていから計算される値が入る
    // PyTorchでいうところの-1指定と同じ
    let test_images = mnist_data.test_images.reshape(((), 1, 28, 28))?.to_device(&device)?;
    let test_labels = mnist_data.test_labels.to_dtype(DType::U32)?.to_device(&device)?;
    println!("test_images: {:?}", test_images.dims());
    println!("test_labels: {:?}\n", test_labels.dims());

    // 推論(log softmax出力)
    let test_logits = model.forward(&test_images, false)?;
    // 正解率の計算
    let sum_ok = test_logits
            .argmax(D::Minus1)? // 最大値のインデックスを取得
            .eq(&test_labels)? // 正解ラベルと比較(正解なら1, 不正解なら0)
            .to_dtype(DType::F32)?  // 32bit floatに変換
            .sum_all()? // 合計算出
            .to_scalar::<f32>()?;   // Tensorをf32に変換
        // 要素数で割って正解率に変換
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!("test acc: {:5.2}%", 100. * test_accuracy);
    Ok(())
}
