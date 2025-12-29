import torch
from torch.nn import init
import torch.nn as nn
from torch.autograd import Variable
import config_2d

num_classes = config_2d.NUM_CLASSES  # 2
from torch.nn import functional as F

### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def dice(y_pred, y_true):
    smooth = 1e-7
    m1 = y_true.flatten()  # Flatten
    m2 = y_pred.flatten()  # Flatten
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(dice_loss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = 0
        dice_coef_class = dice(y_pred, y_true)
        loss = 1 - dice_coef_class + loss
        return loss
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def L2_loss(input, pred):
    #l2_loss = F.mse_loss(input, pred, reduction='mean')
    l2_loss = F.kl_div(input, pred,reduction='mean') # KL_loss
    return l2_loss

def correlation(u_k, v_k):
    eps = torch.finfo(torch.float32).eps

    N, C, _, _ = u_k.shape
    u_k = u_k.reshape(N, C, -1)
    v_k = v_k.reshape(N, C, -1)
    u_k = u_k - u_k.mean(dim=-1, keepdim=True)
    v_k = v_k - v_k.mean(dim=-1, keepdim=True)

    cc = torch.sum(u_k * v_k, dim=-1) / (eps + torch.sqrt(torch.sum(u_k ** 2, dim=-1)) * torch.sum(v_k ** 2, dim=-1))
    cc = torch.clamp(cc, -1., 1.)

    return cc.mean()

class dice_l2_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(dice_l2_loss, self).__init__()

    def forward(self, y_pred, y_true, x_input, x_pred):
        dice_loss_1 = 1- dice(y_pred, y_true)
        l2_loss_2 = L2_loss(x_input, x_pred)
        losses = dice_loss_1 + 0.1 * l2_loss_2
        return losses

CE = nn.BCEWithLogitsLoss() #torch.nn.BCELoss()
mse = torch.nn.MSELoss()


def loss_diff1(u_prediction_1, u_prediction_2):
    loss_a = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_a = CE(u_prediction_1[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_2[:, i, ...].float(), requires_grad=False))

    loss_diff_avg = loss_a.mean().item()
    return loss_diff_avg


def loss_diff2(u_prediction_1, u_prediction_2):
    loss_b = 0.0

    for i in range(u_prediction_2.size(1)):
        loss_b = CE(u_prediction_2[:, i, ...].clamp(1e-8, 1 - 1e-7),
                                 Variable(u_prediction_1[:, i, ...], requires_grad=False))

    loss_diff_avg = loss_b.mean().item()
    return loss_diff_avg

def loss_mask(u_prediction_1, u_prediction_2, critic_segs):
    gen_mask = (critic_segs.squeeze(0)).float()
    loss_a = gen_mask * torch.mean((u_prediction_1-u_prediction_2)**2)

    loss_diff_avg = loss_a.mean()

    return loss_diff_avg


def disc_loss(pred, target, target_zeroes, target_ones):
    real_loss1 = CE(target, target_ones.float())
    fake_loss1 = CE(pred, target_zeroes.float())

    loss = (1/2) * (real_loss1 + fake_loss1)

    return loss


def gen_loss(pred, target_ones):
    fake_loss1 = CE(pred, target_ones.float())

    loss = fake_loss1

    return loss