from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import dataloader
import torchvision.models as models


resnet18 = models.resnet18(pretrained=True)    # use resnet18 model
resnet18.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)  # modify input channel to 1 to fit Chars74k dataset
resnet18.fc = nn.Linear(512, 62,bias=True)        # modify output channel to 62 to fit Chars74k with 62 class

def train(args, model, device, train_loader, optimizer, epoch):
    # 这个函数训练一个epoch的数据
    model.train()  # 模型设为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        #迭代一整个数据集
        data, target = data.to(device), target.to(device)  # 得到数据和标签
        optimizer.zero_grad() # 优化器梯度清零
        output = model(data)  # 前向传播得到输出
        loss = nn.CrossEntropyLoss()(output, target)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step() # 优化器更新模型参数
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    # 此函数测试测试集数据
    model.eval() # 模型设置为测试/推理/验证模式
    test_loss = 0 
    correct = 0
    with torch.no_grad(): # 不计算梯度
        for data, target in test_loader:
            #以下同 train()
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print("label: ",target, "pred: ", pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available() # 是否用CUDA

    torch.manual_seed(args.seed) # 设置随机数

    device = torch.device("cuda" if use_cuda else "cpu")  #设置训练所用为CPU或GPU(CUDA)

    datasets1 = dataloader.CharTrainVal()        # 实例化训练集
    datasets2 = dataloader.CharTrainVal(val=True) # 实例化测试集
    train_loader = torch.utils.data.DataLoader(datasets1,batch_size=12,shuffle=True,num_workers=8,drop_last=True) # 实例化训练数据加载器

    test_loader = torch.utils.data.DataLoader(datasets2,batch_size=10,shuffle=False,num_workers=1,drop_last=True) # 实例化测试数据加载器

    model = resnet18.to(device) # 实例化模型并将模型放到CPU/GPU

    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.0001)  # 实例化优化器：Adam

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)  # 实例化学习率调整器StepLR

    # 迭代循环
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch) # 训练一代模型
        test(model, device, test_loader)  # 测试模型
        scheduler.step()   # 调整学习率

    if args.save_model:
        torch.save(model.state_dict(), "Chars74k_resnet18.pth")  #保存模型参数


if __name__ == '__main__':
    main()