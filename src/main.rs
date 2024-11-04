use anyhow::Result;
use candle_core::{Device, DType};
use convert_nn::CNN;
use candle_nn::var_builder::VarBuilder;

fn main() -> Result<()> {
    // デバイスの設定
    let device = Device::cuda_if_available(0)?; // GPUを使用
    let model_path = std::path::PathBuf::from("mnist_cnn_weights.safetensors");
    let mut vb = unsafe 
    { 
        VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? 
    };
    // CNNモデルの作成
    let mut model = CNN::new(&mut vb)?;
    // ここで、modelを使用して推論や評価を実行
    Ok(())
}
