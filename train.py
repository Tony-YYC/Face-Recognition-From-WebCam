# -*- encoding: utf8 -*-
import config
import os
import torch
from data_set import get_dataset, get_transform
from model import Net
import torch.nn.functional as F

# 检查是否有GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    train_loader, test_loader = get_dataset(batch_size=config.BATCH_SIZE)
    print(train_loader,test_loader)
    net = Net().to(DEVICE)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(config.EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            # 使用最大似然 / log似然代价函数
            loss = F.nll_loss(output, y)
            # Pytorch会梯度累计所以需要梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 使用Adam进行梯度更新
            optimizer.step()
            #print("step",step)
            #print(len(x))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, step * len(x), len(train_loader.dataset),
                    100. * step / len(train_loader), loss.item()))
    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                output = model(x)
                test_loss += F.nll_loss(output, y, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\ntest loss={:.4f}, accuracy={:.4f}\n'.format(test_loss, float(correct) / len(test_loader.dataset)))
    # 使用验证集查看模型效果
    test(net, test_loader)
    # 保存模型权重到 config.DATA_MODEL目录
    print(config.DATA_MODEL)
    torch.save(net.state_dict(), os.path.join(config.DATA_MODEL,config.DEFAULT_MODEL))
    return net