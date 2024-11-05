use anyhow::Result;
use candle_core::{Device, DType, D, Tensor };
use convert_nn::CNN;
use candle_nn::var_builder::VarBuilder;

fn main() -> Result<()> {
    // デバイスの設定
    let device = Device::cuda_if_available(0)?; // GPUを使用
    // モデルのロード
    let use_safetenors = false;
    let vb = if use_safetenors 
    {
        // safetensors形式のモデルをロード
        let model_path = std::path::PathBuf::from("mnist_cnn_weights.safetensors");
        unsafe { 
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? 
        }
    } else 
    {
        // pth形式のモデルをロード
        let model_path = std::path::PathBuf::from("mnist_cnn_weights.pth");
        VarBuilder::from_pth(model_path, DType::F32, &device)?
    };
    // CNNモデルの作成
    let model = CNN::new(&vb)?;
    // mnistのロード
    // 画像の正規化メソッドが無かったのでそのまま使用。背景0, 文字1の2値画像
    let mnist_data = candle_datasets::vision::mnist::load()?;
    let test_images = mnist_data.test_images.reshape(((), 1, 28, 28))?.to_device(&device)?;
    let test_labels = mnist_data.test_labels.to_dtype(DType::U32)?.to_device(&device)?;
    println!("test_images: {:?}", test_images.dims());
    println!("test_labels: {:?}\n", test_labels.dims());

    // 推論
    let test_logits = model.forward(&test_images, false)?;
    // 正解率の計算
    let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!("test acc: {:5.2}%", 100. * test_accuracy);
    Ok(())
}
