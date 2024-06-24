import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
        )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU()
        )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(),
                            *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
                            nn.Linear(hidden_dim, num_classes)
                            )
    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_list, total_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    for X, y in dataloader:
        model.train() if opt else model.eval()

        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss_list.append(loss.numpy())
        total_error += ( pred.numpy().argmax(axis=1) != y.numpy() ).sum()
        
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    sample_nums = len(dataloader.dataset)
    return total_error/sample_nums, np.mean(loss_list)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                                        f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                                        f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = ndl.data.DataLoader(train_set, batch_size, shuffle=True)
    test_loader = ndl.data.DataLoader(test_set, batch_size)
    
    model = MLPResNet(28*28, 
                    hidden_dim=hidden_dim, 
                    num_classes=10)
    opt = optimizer(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
        )
    
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt=opt)
        
    test_err, test_loss = epoch(test_loader, model, opt=None)
    
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    # np.random.seed(1)
    train_mnist(data_dir="../data")
