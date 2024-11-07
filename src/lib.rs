use candle_core::{Tensor, Result};
use candle_nn::{Module, Conv2d, Conv2dConfig, Linear, Dropout};
use candle_nn::conv::conv2d;
use candle_nn::linear::linear;
use candle_nn::ops::log_softmax;
use candle_nn::var_builder::VarBuilder;

// CNN2層、全結合層2層の簡素な画像分類モデル
// mnist_sample.pyに実装されているものを移植

// モデル構造の定義
pub struct CNN {
    conv1: Conv2d,
    conv2: Conv2d,
    dropout1: Dropout,
    dropout2: Dropout,
    fc1: Linear,
    fc2: Linear,
}

// メソッドの定義
impl CNN {
    pub fn new(vb: &VarBuilder) -> Result<Self> {
        // 各レイヤーの初期化
        // PyTorchと似たような感じで書けるビルダーメソッドが定義されている
        // Tensorのサイズ決定に関わるものは直接引数にあって、
        // それ以外はConv2dConfigの様な構造体でまとめられている感じ。
        // また、現時点ではRustのリファレンスを見ても何も書いて居ないが
        // 対応するPyTorchのtorch.nnのレイヤーとパラメータ名が揃えてあるので、
        // 各パラメータの意味やデフォルト値はPyTorchのリファレンスを参照すると良い。
        // また、最後のパラメータはVarBuilderで
        // パラメータロード済みのVarBUilderを階層を指定して参照することで対応するレイヤーのパラメータを抽出する。
        // ちなみに、対応するものが無い場合は一様乱数で初期化される実装が多い模様。
        let conv1 = conv2d(
            1,
            32,
            3, 
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups:1,
            },
            vb.pp("conv1"))?;
        let conv2 = conv2d(
            32,
            64, 
            3, 
            Conv2dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups:1,
            },
            vb.pp("conv2"))?;
        // ここではDropoutは学習パラメータでないため、直接設定している。
        let dropout1 = Dropout::new(0.25);
        let dropout2 = Dropout::new(0.5);
        // 全結合層はVarBuilderでパラメータをロードしている
        let fc1 = linear(9216, 128, vb.pp("fc1"))?;
        let fc2 = linear(128, 10, vb.pp("fc2"))?;

        Ok(Self { conv1, conv2, dropout1, dropout2, fc1, fc2 })
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        // forwardネットワークの構築はPyTorchとほぼ同じ
        // 各行のxは中身はPython等で同名変数を上書きするよりは
        // Rustのシャドーイングの方がしっくりくる。

        // CNN層
        let x = self.conv1.forward(xs)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.relu()?;
        let x = x.max_pool2d(2)?;
        // 本サンプルは推論用なのでtrainは直接falseを入れてしまってもよい
        let x = self.dropout1.forward(&x, train)?;
        // 全結合層
        let x = x.flatten(1, 3)?;
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        let x = self.dropout2.forward(&x, train)?;
        let x = self.fc2.forward(&x)?;
        log_softmax(&x, 1)
    }
}
