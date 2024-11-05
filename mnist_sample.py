import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの前処理とデータローダーの設定
# candle側に画像正規化メソッドが見当たらなかったため、
# 読み込んだ値をそのまま使う(背景:0, 文字:1)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# torchvision.datasets.MNIST のデータセットをダウンロードし、データローダーを作成
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# CNN2層、全結合層2層の簡素な画像分類モデル
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # CNN層
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # 全結合層
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# トレーニング関数
def train(model, device, train_loader, epoch):
    # オプティマイザーと損失関数の設定
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 学習
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# テスト関数
def test(model, device, test_loader):
    # オプティマイザーと損失関数の設定
    model.eval()
    print(model)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

# メイン関数
def main(mode='train', num_epochs=10, weight_name =  "mnist_cnn_weights", weight_type = "safetensors"): 
    if mode == 'train':
        # モデル定義
        model = CNN().to(device)
        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, epoch)
            test(model, device, test_loader)
        # モデルの重みを safetensors 形式で保存
        save_file(model.state_dict(), weight_name + ".safetensors")
        print(f"Model weights saved in {weight_name}.safetensors")
        torch.save(model.state_dict(), weight_name + ".pth")
        print(f"Model weights saved in {weight_name}.pth")
    elif mode == 'eval':
        # モデル定義
        model = CNN().to(device)
        # safetensorsから重みを読み込む場合
        if weight_type == "safetensors":
            model.load_state_dict(load_file(weight_name + ".safetensors"))
        else:
            model.load_state_dict(torch.load(weight_name + ".pth"))
        test(model, device, test_loader)
    else:
        print("Invalid mode! Choose 'train' or 'eval'.")

if __name__ == '__main__':
    #main(mode = 'train', num_epochs=10) # 学習時
    #main(mode = 'eval', weight_type="safetensors") # 評価時(safetensorsから読み込み)
    main(mode = 'eval', weight_type="pth") # 評価時(pthから読み込み)