import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
import torch.nn.functional as F
from collections import OrderedDict


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


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)

            })) if self.should_apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels) }))


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation(),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(self, in_channels=3, blocks_sizes=[16, 32, 64, 128], deepths=[2, 2, 2, 2],
                 activation=nn.ReLU, block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x



class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, n_classes, useTRN, batch_size, *args, **kwargs):
        super(ResNet, self).__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.useTRN = useTRN
        if not self.useTRN:
            self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        else:
            self.tcl = TCL(weight_size=(batch_size, 128, 8, 8), ranks=(batch_size, int(128 / 2), 2, 2))
            self.trl = TRL(ranks=(10, 1, 1, 10), input_size=(batch_size, int(128 / 2), 2, 2),
                           output_size=(batch_size, n_classes))

    def forward(self, x):
        x = self.encoder(x)
        if not self.useTRN:
            x = self.decoder(x)
        else:
            x = self.tcl(x)
            x = self.trl(x)
        return F.log_softmax(x)


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