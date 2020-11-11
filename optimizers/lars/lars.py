""" Layer-wise adaptive rate scaling for SGD in PyTorch! """
import torch
from torch.optim.optimizer import Optimizer, required
# use the default values here to start with
#    parser.add_argument('--batch-size', type=int, default=4096, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam','lars'],
#                         help='which optimizer to use')
#     parser.add_argument('--test-batch-size', type=int, default=2048, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 10)')
#     parser.add_argument('--lr', type=float, default=0.0025, metavar='LR',
#                         help='learning rate (default: 0.0025)')
#     parser.add_argument('--wd', type=float, default=0.01, metavar='WD',
#                         help='weight decay (default: 0.01)')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--eta', type=int, default=0.001, metavar='e',
#                         help='LARS coefficient (default: 0.001)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')

class Lars(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate (\gamma_0)
        momentum (float, optional): momentum factor (default: 0) ("m")
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            ("\beta")
        eta (float, optional): LARS coefficient
        max_epoch: maximum training epoch to determine polynomial LR decay.
    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    Large Batch Training of Convolutional Networks:


        writer: the SummaryWriter object for tensorboard
        
        
        https://arxiv.org/abs/1708.03888
    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """
    def __init__(self, params, lr=required, momentum=.9,
                 weight_decay=.0005, eta=0.001, max_epoch=200, writer=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}"
                             .format(weight_decay))
        if eta < 0.0:
            raise ValueError("Invalid LARS coefficient value: {}".format(eta))
        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay,
                        eta=eta, max_epoch=max_epoch)
        if writer:
            self.writer = writer
            self.step_num = 0
        super(Lars, self).__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            epoch: current epoch to calculate polynomial LR decay schedule.
                   if None, uses self.epoch and increments it.
        """
        loss = None
        if closure is not None:
            loss = closure()
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                d_p = p.grad.data
                weight_norm = torch.norm(p.data)
                # save to tensorboard
                if self.writer:
                    self.writer.add_histogram('lars/weight_norm', torch.tensor(weight_norm), self.step_num)
                    self.step_num += 1

                grad_norm = torch.norm(d_p)
                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay
                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)
                # Update the momentum term
                actual_lr = local_lr * global_lr
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = \
                            torch.zeros_like(p.data)
                    # print(buf)
                else:
                    buf = param_state['momentum_buffer']
                    # print(buf)
                buf.mul_(momentum).add_(actual_lr, d_p + weight_decay * p.data)
                p.data.add_(-buf)
        return loss
