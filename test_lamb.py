"""MNIST example.

Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from optimizers.lamb import Lamb
from optimizers.lars import Lars
from optimizers.adam import Adam

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net(nn.Module):
    def __init__(self, num_outputs=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
def train(args, model, device, train_loader, optimizer, epoch, event_writer):
    model.train()
    tqdm_bar = tqdm.tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(tqdm_bar):
        data, target = data.to(device), target.to(device)
        target -= 1 # emnist
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            step = batch_idx * len(data) + (epoch-1) * len(train_loader.dataset)
            # log_lamb_rs(optimizer, event_writer, step)
            event_writer.add_scalar('loss', loss.item(), step)
            tqdm_bar.set_description(
                f'Train epoch {epoch} Loss: {loss.item():.6f}')

def test(args, model, device, test_loader, event_writer:SummaryWriter, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target -= 1 # emnist
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    event_writer.add_scalar('loss/test_loss', test_loss, epoch - 1)
    event_writer.add_scalar('loss/test_acc', acc, epoch - 1)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * acc))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam','lars'],
                        help='which optimizer to use')
    parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
                        help='learning rate (default: 0.0025)')
    parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
                        help='weight decay (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eta', type=int, default=0.001, metavar='e',
                        help='LARS coefficient (default: 0.001)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('../data', train=True, download=True,
        datasets.EMNIST('../data', train=True, download=True, split='letters',
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        # datasets.MNIST('../data', train=False, transform=transforms.Compose([
        datasets.EMNIST('../data', train=False, split='letters',
                        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    num_features = 26
    model = Net(num_outputs=num_features).to(device)
    # optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999), adam=(args.optimizer == 'adam'))
    writer = SummaryWriter()
    print(len(train_loader),len(test_loader))
    print(model)
    print(f'total params ---> {count_parameters(model)}')
    if args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999), adam=False, writer=writer)
    elif args.optimizer=='lars':
        optimizer = Lars(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd, eta=args.eta, max_epoch=args.epochs+1, writer=writer)
    else:
        # use adam optimizer
        optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(.9, .999), adam=True, writer=writer)
    print(f'Currently using the {args.optimizer}\n\n')
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, writer, epoch)
    return optimizer

 
if __name__ == '__main__':
    opt = main()
