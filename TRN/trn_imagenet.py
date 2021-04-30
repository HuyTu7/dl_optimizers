# %% md

# Tensor Regression Network vs. CNN with Fully Connected Layers on CIFAR10
# %% md

## Install Tensorly and Import Libraries

# %%
import os
import sys
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import pickle
import time
import numpy as np

import tensorly as tl
import tensorly.tenalg as ta
from tensorly.tenalg import inner
from tensorly import check_random_state
from tensorly.decomposition import tucker
from collections import Counter
from imagenet_data import *

from tensorboardX import SummaryWriter

from sklearn.metrics import f1_score, precision_score, recall_score, auc, roc_auc_score

sys.path.append("../optimizers/lamb")
from lamb import Lamb

sys.path.append("../optimizers/lars")
from lars_extra import Lars

sys.path.append("../optimizers/sgd")
from sgd import SGD

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--working_dir', default='train', help='Working directory for checkpoints and logs')
ap.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size')
ap.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs')
ap.add_argument('--random_seed', type=int, default=12345, help='Random seed')
ap.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Learning Rate')
ap.add_argument('--seed', type=int, default=4796, metavar='S',
                help='random seed (default: 4796)')
ap.add_argument('--eta', type=int, default=0.001, metavar='e',
                help='LARS coefficient (default: 0.001)')
ap.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                help='weight decay (default: 0.01)')
ap.add_argument('--log-interval', type=int, default=100, metavar='N',
                help='how many batches to wait before logging training status')
ap.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam', 'lars', 'sgd'],
                help='which optimizer to use')
ap.add_argument('--trn', action='store_true',
                help='use TRN or not')
ap.add_argument('-t', '--temporal', action='store_true',
                help='use layer-wise or not')

args = ap.parse_args()

# %% md
## Specify Parameters
# %%

useTRN = args.trn
batch_size = 16

tl.set_backend('pytorch')
device = 'cuda'

random_state = 1234
rng = check_random_state(random_state)

# %% md

## Load CIFAR10 Dataset

# %%

# CIFAR 10


test_loader = get_dataloader("./data/Imagenet32/val/", False, int(args.batch_size/2), 4, 32, 1000)
train_loader = get_dataloader("./data/Imagenet32/train/",  True, args.batch_size, 4, 32, 1000)

# %% md
## Model Definition
# %% md
### TCL Implementation
# %%

# Tucker decomposition layer
# x: Input Tensor
# ranks: The ouput dimension for the tensor
class TCL(nn.Module):
    # def __init__(self, x, ranks): #AO
    def __init__(self, weight_size, ranks):
        super().__init__()
        self.ranks = list(ranks)
        # weight_size = list(x.shape) #AO
        self.tensor_estimated = 0

        self.factor0 = nn.Parameter(torch.randn((ranks[0], weight_size[0])), requires_grad=True)
        self.factor1 = nn.Parameter(torch.randn((ranks[1], weight_size[1])), requires_grad=True)
        self.factor2 = nn.Parameter(torch.randn((ranks[2], weight_size[2])), requires_grad=True)
        self.factor3 = nn.Parameter(torch.randn((ranks[3], weight_size[3])), requires_grad=True)

    def forward(self, x):
        # Do the tensor contraction one nmode product at a time
        output = ta.mode_dot(x, self.factor1, mode=1)
        output = ta.mode_dot(output, self.factor2, mode=2)
        output = ta.mode_dot(output, self.factor3, mode=3)
        return output

    def penalty(self, order=2):
        penalty = tl.norm(self.factor1, order)
        return penalty


# %% md

### TRL Implementation

# %%

# input_size: shape of input activation tensor
# ranks: size of core
# output_size: network output -> should be batch_size x number of classes
class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)

        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)

        self.n_outputs = int(np.prod(output_size[1:]))

        # Core of the regression tensor weights
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])

        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.register_parameter('factor_{}'.format(index), self.factors[index])

        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        regression_weights = tl.tucker_to_tensor((self.core, self.factors))
        a = inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias
        return inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias

    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + tl.norm(f, order)
        return penalty


# %% md

### ResNet-based Architecture Implementation

# %%

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        print(type(images))
        out = self(images)  # Generate predictions
        print(out)
        # criterion=nn.CrossEntropyLoss()
        # loss = criterion(images,labels)
        loss = F.cross_entropy(images, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        criterion = nn.CrossEntropyLoss()
        loss = criterion(images, labels)
        # loss = F.cross_entropy(images, labels)   # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ELU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Net(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        # super().__init__()
        super(Net, self).__init__()
        self.conv1 = conv_block(in_channels, 32)
        self.conv2 = conv_block(32, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 64), conv_block(64, 64), conv_block(64, 64))

        self.conv3 = conv_block(64, 128, pool=True)
        # self.conv4 = conv_block(512, 512, pool=True)
        # self.conv5 = conv_block(512, 1024, pool=True)
        # self.res3 = nn.Sequential(conv_block(1024, 1024), conv_block(1024, 1024), conv_block(1024, 1024))
        # self.conv5 = conv_block(512, 512, pool=True)
        # self.res3 = nn.Sequential(conv_block(512, 512), conv_block(512, 512), conv_block(512, 512))
        size_n = 128
        self.res2 = nn.Sequential(conv_block(size_n, size_n), conv_block(size_n, size_n), conv_block(size_n, size_n))

        if useTRN:
            self.tcl = TCL(weight_size=(args.batch_size, size_n, 8, 8), ranks=(args.batch_size, int(size_n / 2), 2, 2))
            self.trl = TRL(ranks=(10, 1, 1, 10), input_size=(args.batch_size, int(size_n / 2), 2, 2),
                           output_size=(args.batch_size, num_classes))
        else:
            self.pool = nn.MaxPool2d(2)
            self.flat = nn.Flatten()
            self.lin = nn.Linear(size_n * 4 * 4, size_n)
            self.lin2 = nn.Linear(size_n, num_classes)

        # self.classifier = nn.Sequential(nn.MaxPool2d(2),
        #                                 nn.Flatten(),
        #                                 nn.Linear(batch_size*2*2*1024, 10))

    def forward(self, xb):
        out = self.conv1(xb)
        a = out.numel()
        # print(a)
        out = self.conv2(out)
        b = out.numel()
        # print(b)
        out = self.res1(out) + out
        out = self.conv3(out)
        c = out.numel()
        # print(c)
        # out = self.conv4(out)
        out = self.res2(out) + out
        d = out.numel()
        # print(d)
        # out = self.conv5(out)
        # out = self.res3(out) + out

        if useTRN:
            out = self.tcl(out)
            e = out.numel()
            # print(e)
            out = self.trl(out)
            f = out.numel()
            # print(f)
        else:
            out = self.pool(out)
            # print(out.shape)
            out = self.flat(out)
            e = out.numel()
            # print(out.shape)
            # print(e)
            out = self.lin(out)
            f = out.numel()
            # print(f)
            out = self.lin2(out)
            g = out.numel()
            # print(g)

        # print("IAM DONEEEEEE")
        return F.log_softmax(out)


def count_parameters(model):
    '''Returns the number of trainable parameters in the model.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


weight_decay = 1e-4
model = Net(3, 1000)
model = model.to(device)

temp_fname = "T" if args.temporal else "noT"
trn_fname = "T" if args.trn else "noT"
writer = SummaryWriter(comment="_%s_%s_%s_%s_%s" % (temp_fname, trn_fname,
                                                    args.optimizer, args.batch_size,
                                                    args.learning_rate))
if args.temporal:
    weight_decay = args.learning_rate / args.epochs
    if args.optimizer == 'lamb':
        optimizer = Lamb(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay,
                         betas=(.9, .999), adam=False, writer=writer)
    elif args.optimizer == 'lars':
        base_optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                         weight_decay=weight_decay)
        # optimizer = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001, writer=writer)
        optimizer = Lars(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.learning_rate, weight_decay=args.wd, eta=args.eta,
                         max_epoch=args.epochs + 1, writer=writer)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.6,
                        weight_decay=weight_decay, writer=writer)
    else:
        # use adam optimizer
        optimizer = Lamb(model.parameters(), lr=args.learning_rate, weight_decay=weight_decay,
                         betas=(.9, .999), adam=True, writer=writer)
    print(f'The model has {count_parameters(model):,} trainable parameters')
else:
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                               weight_decay=weight_decay)

criterion = nn.CrossEntropyLoss()
# Count params
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% md
## Train the model
# %%

n_epoch = 30  # Number of epochs
regularizer = 0.001
grad_clip = 1

loss_train = []
acc_train = []
loss_test = []
acc_test = []
topkacc = []  # on test data


def train(n_epoch, event_writer):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # Important: do not forget to reset the gradients
        optimizer.zero_grad()

        output = model(data)
        # print(model.tcl.penalty(2))
        if useTRN:
            loss = criterion(output, target) + model.tcl.penalty(2) + regularizer * model.trl.penalty(2)
        else:
            loss = criterion(output, target)

        loss.backward()
        # if grad_clip:
        #         nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            step = batch_idx * len(data) + (n_epoch - 1) * len(train_loader.dataset)
            event_writer.add_scalar('loss', loss.item(), step)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))

    return loss


def test(model, optimizer, ckpt_dir_name):
    model, optimizer = load_pretrained_model(model, optimizer,
                                             "%s/ckpt/%s" % (ckpt_dir_name, "best_weights.pt"))
    model.eval()
    test_loss = 0
    correct = 0
    correctk = 0
    actuals = []
    predictions = []
    epsilon = 1e-7
    # confusion = Counter({"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if useTRN:
            test_loss = criterion(output, target) + model.tcl.penalty(2) + regularizer * model.trl.penalty(2)
        else:
            test_loss = criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # Calc top N accuracy
        correctk += topkaccuracy(output, target, topk=(3,))
        prediction = output.argmax(dim=1, keepdim=True)
        actuals.extend(target.view_as(prediction))
        predictions.extend(prediction)

    actuals, predictions = [i.item() for i in actuals], [i.item() for i in predictions]
    # precision = confusion["tp"] / (confusion["tp"] + confusion["fp"] + epsilon)
    # recall = confusion["tp"] / (confusion["tp"] + confusion["fn"] + epsilon)
    f1 = f1_score(actuals, predictions, average='macro')
    precision = precision_score(actuals, predictions, average='macro')
    recall = recall_score(actuals, predictions, average='macro')
    test_loss /= len(test_loader.dataset)

    print('mean: {}'.format(test_loss))
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Top-K Accuracy: {:.0f}%, Precision: {:.0f}%, Recall: {:.0f}%, F1: {:.0f}%\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            100 * correctk / len(test_loader.dataset),
            100 * precision, 100 * recall, 100 * f1))
    acc_test.append(correct / len(test_loader.dataset))
    topkacc.append(correctk / len(test_loader.dataset))

    return {"top1acc": acc_test[0], "topkacc": topkacc[0], "f1": f1, "recall": recall, "precision": precision}


def f1_score_torch(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''

    with torch.no_grad():
        y_true = F.one_hot(ground_truth, 10).to(torch.float32)
        y_pred = F.softmax(prediction, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)
        print(tp, tn, fp, fn)
        # f1 = f1.clamp(min=epsilon, max=1 - epsilon)
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def topkaccuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:maxk].float().sum()
        res = float(correct_k.item())

        return res


def load_pretrained_model(curr_model, curr_optimizer, pretrained_path):
    try:
        model_dict = curr_model.state_dict()
        optim_dict = curr_optimizer.state_dict()
        load_pretrained = torch.load(pretrained_path)
        pretrained_model_specs = load_pretrained['model_state_dict']
        pretrained_model_specs = {k: v for k, v in pretrained_model_specs.items() if k in model_dict}
        pretrained_optim_specs = load_pretrained['optimizer_state_dict']
        pretrained_optim_specs = {k: v for k, v in pretrained_optim_specs.items() if k in optim_dict}

        # update & load
        model_dict.update(pretrained_model_specs)
        optim_dict.update(pretrained_optim_specs)
        curr_model.load_state_dict(model_dict)
        curr_optimizer.load_state_dict(optim_dict)
        print(f"the model loaded successfully")
    except:
        print(f"the pretrained model doesn't exist or it failed to load")
    return curr_model, curr_optimizer


if __name__ == '__main__':
    ckpt_dir_name = "%s_%s_%s_%s_%s" % (temp_fname, trn_fname, args.working_dir,
                                        args.optimizer, args.batch_size)
    print(args)
    print("Path: ", ckpt_dir_name)
    ckpt_dir = os.path.join(ckpt_dir_name, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    lives = 10
    valid_loss_prev = 1000000
    results = {}
    losses = []
    start = time.time()
    for epoch in range(1, args.epochs):
        loss = train(epoch, writer)
        losses.append(loss)
        if loss < valid_loss_prev:
            valid_loss_prev = loss
            state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict()}
            fname = os.path.join(ckpt_dir, 'best_weights.pt'.format(epoch))
            torch.save(state, fname)
        else:
            lives -= 1
            if lives == 0:
                break
    results["time"] = time.time() - start
    results["loss"] = losses
    results.update(test(model, optimizer, ckpt_dir_name))
    print("Training Time Taking: ", results["time"])
    pickle.dump(results, open(os.path.join(ckpt_dir, 'results.p'), 'wb'))