import os
import argparse
import logging
import random
import sys
import time
import config_2d
#from models import Unet_2d
from models.Decompose import Decompose
from models.Decompose_wdcp import Decompose_wdcp
from torch import optim
from torch.utils.data import DataLoader
from dataloader import CN_MyTrainDataset, CN_MyTestDataset, TwoStreamBatchSampler, CN_make_dirset_train_with_folds
from torchvision.transforms import transforms
from metrics_2d import dice_loss, dice
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import ramps
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd95, assd
import seg_metrics.seg_metrics as sg
from metrics_2d import DiceLoss, dice, correlation

test_model = Decompose#.Decompose

n_epochs = config_2d.NUM_EPOCHS
flag_gpu = config_2d.FLAG_GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_size_w = config_2d.PATCH_SIZE_W
patch_size_h = config_2d.PATCH_SIZE_H
batch_size = config_2d.BATCH_SIZE
image_rows = config_2d.VOLUME_ROWS
image_cols = config_2d.VOLUME_COLS
image_depth = config_2d.VOLUME_DEPS
test_imgs_path = config_2d.test_imgs_path
test_extraction_step = config_2d.TEST_EXTRACTION_STEP
num_classes = 2

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='HCP_20_alpha', help='experiment_name')
parser.add_argument('--model', type=str, default='ours_new', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[240, 240], help='patch size of network input')
parser.add_argument('--seed', type=int, default=887, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=8, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4096, help='labeled data') # HCP 100% 20992, 80% 16793, 60% 12595, 40% 8397, 20% 4198, 10% 2099, 5% 1050, 2% 420, 1% 210
#parser.add_argument('--labeled_num', type=int, default=368, help='labeled data') # MDM 37.5%: 552, 25%: 368, 20%: 294, 10%: 147, 5%: 73, 2%: 29, 1%: 15
#parser.add_argument('--labeled_num', type=int, default=512, help='labeled data') # MMD 100% 2048, 50% 1024, 25% 512
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--n_aug', type=int, default=3, help='Number of folds for cross-validation')
parser.add_argument('--n_tresh', type=int, default=0.05, help='Number of folds for cross-validation') # [0.01, 0.05, 0.1, 0.5, 1.0]
parser.add_argument('--w_dcp', type=int, default=1, help='Decomposition loss weight') # [0.1, 1, 10, 50, 100]
args = parser.parse_args()


# Select n_aug by fixing n_tresh to 0.05. Then select n_tresh by fixing n_aug to 3 (best n_aug in previous step)

def jaccard_index(pred, gt):
    pred = (pred > 0.5).float()
    intersection = torch.sum(pred * gt)
    union = torch.sum(pred) + torch.sum(gt) - intersection
    return (intersection / (union + 1e-6)).item()

def compute_hd95(pred, gt, voxelspacing):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    gt = gt.cpu().numpy().astype(np.uint8)
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    try:
        return hd95(pred, gt, voxelspacing=voxelspacing)
    except:
        return np.inf

def compute_asd(pred, gt, voxelspacing):
    #pred = (pred > 0.5).cpu().numpy().astype(np.uint8)
    gt = gt.cpu().numpy().astype(np.uint8)
    if pred.sum() == 0 or gt.sum() == 0:
        return np.inf
    try:
        return assd(pred, gt, voxelspacing=voxelspacing)
    except:
        return np.inf


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def update_ema_bn_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.buffers(), model.buffers()):
        ema_param.data = ema_param.data * alpha + param.data * (1-alpha)

import torch.nn.functional as F



def consistency_filtering(teacher_out, student_out, threshold, rampup_epochs, current_epoch):


    #teacher_predictions = torch.stack(teacher_predictions, dim=0)
    #student_predictions = torch.stack(student_predictions, dim=0)
    std_teacher_prediction = torch.sum(torch.std(teacher_out, 1), 1)
    std_student_prediction = torch.sum(torch.std(student_out, 1), 1)
    #entropy_t = - torch.sum(teacher_out * torch.log2(teacher_out), dim=1)
    #entropy_s = - torch.sum(student_out * torch.log2(student_out), dim=1)
    #
    # Compute consistency scores
    #consistency_scores = F.mse_loss(torch.sum(entropy_t, dim=1), torch.sum(entropy_s, dim=1), reduction='none')
    consistency_scores = F.mse_loss(std_teacher_prediction, std_student_prediction, reduction='none')

    # Apply ramp-up function
    rampup_value = rampup(current_epoch, rampup_epochs)
    consistency_scores = consistency_scores * rampup_value
    #print(consistency_scores)

    # Filter samples based on the threshold
    filtered_indices = torch.nonzero(consistency_scores < threshold, as_tuple=True)[0]

    return filtered_indices

def rampup(current_epoch, rampup_epochs):
    if current_epoch < rampup_epochs:
        p = max(0.0, float(current_epoch)) / float(rampup_epochs)
        p = 1.0 - p
        return torch.tensor(p, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(p, dtype=torch.float32)
    else:
        return torch.tensor(1.0, dtype=torch.float32).cuda() if torch.cuda.is_available() else torch.tensor(1.0, dtype=torch.float32)

def consistency_loss(teacher_out, student_out, teacher_model, student_model, unlabeled_data, unlabeled_data1, threshold, rampup_epochs, current_epoch):
    #teacher_model.eval()
    #student_model.train()

    #labeled_outputs, _, _, _, _ = student_model(labeled_data, labeled_data1)

    filtered_indices = consistency_filtering(teacher_out, student_out, threshold, rampup_epochs, current_epoch)
    filtered_unlabeled_data = unlabeled_data[filtered_indices]
    filtered_unlabeled_data1 = unlabeled_data1[filtered_indices]
    #print(filtered_indices)

    if len(filtered_indices) == 0:
        return torch.tensor(0.0)

    s_unlabeled_outputs, _, _, _, _ = student_model(filtered_unlabeled_data, filtered_unlabeled_data1)
    t_unlabeled_outputs, _, _, _, _ = teacher_model(filtered_unlabeled_data, filtered_unlabeled_data1)

    s_unlabeled_outputs = torch.sigmoid(s_unlabeled_outputs)
    t_unlabeled_outputs = torch.sigmoid(t_unlabeled_outputs)

    #s_unlabeled_outputs = student_model(filtered_unlabeled_data)
    #t_unlabeled_outputs = teacher_model(filtered_unlabeled_data)

    consistency_loss = F.mse_loss(s_unlabeled_outputs, t_unlabeled_outputs)

    return consistency_loss

'''
    Include normal Mean-Teacher method, return  Net outputs(list prediction probability)
    Update model gradients, retain backward graph
'''
def train_one_step(args, model, ema_model, sample, sample1, label, epoch, iterator):
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    dice_loss = DiceLoss(args.num_classes)

    label_student_weak = sample[:args.labeled_bs]
    unlabel_teacher_weak = sample[args.labeled_bs:]

    label_student_weak1 = sample1[:args.labeled_bs]
    unlabel_teacher_weak1 = sample1[args.labeled_bs:]

    unlabel_teacher_weak = unlabel_teacher_weak.repeat([args.n_aug, 1, 1, 1])
    unlabel_teacher_weak = unlabel_teacher_weak + torch.clamp(torch.randn_like(
                unlabel_teacher_weak) * 0.1, -0.2, 0.2)
            #un_t2_re = un_t2.repeat([args.n_aug, 1, 1, 1])
    unlabel_teacher_weak1 = unlabel_teacher_weak1.repeat([args.n_aug, 1, 1, 1])
    unlabel_teacher_weak1 = unlabel_teacher_weak1 + torch.clamp(torch.randn_like(
                unlabel_teacher_weak1) * 0.1, -0.2, 0.2)

    output_label_student_weak, u_k, v_k, p_x, p_y = model(label_student_weak, label_student_weak1)
    softmax_label_student_weak = torch.softmax(output_label_student_weak, dim=1)
    # for unlabel data apply strong aug
    output_unlabel_student_strong, u_k1, v_k1, p_x1, p_y1 = model(unlabel_teacher_weak, unlabel_teacher_weak1)
    softmax_unlabel_student_strong = torch.softmax(output_unlabel_student_strong, dim=1)
    reshaped_softmax_unlabel_student_strong = softmax_unlabel_student_strong.reshape([args.n_aug, -1, 1])
    # origin_output = torch.cat((origin_output_l, origin_output_u), dim=0)
    # origin_output_soft = torch.softmax(origin_output, dim=1)
    # ema_inputs = sample[args.labeled_bs:]
    with torch.no_grad():
        # ema_model.eval()
        output_unlabel_teacher_weak, u_k2, v_k2, p_x2, p_y2 = ema_model(unlabel_teacher_weak, unlabel_teacher_weak1)   #(6,4,h,w)
        softmax_unlabel_teacher_weak = torch.softmax(output_unlabel_teacher_weak, dim=1)

        reshaped_softmax_unlabel_teacher_weak = softmax_unlabel_teacher_weak.reshape([args.n_aug, -1, 1])  #[2, 1, 96, 96, 96]
    #print(output_label_student_weak.shape, softmax_label_student_weak.shape, label.shape)

    label_loss_ce = ce_loss(output_label_student_weak, label[:args.labeled_bs].squeeze().long())
    label_loss_dice = dice_loss(softmax_label_student_weak.squeeze(), label[:args.labeled_bs])
    label_supervised_loss = label_loss_ce + label_loss_dice
    # unlabel_consistency_loss = torch.mean(((softmax_unlabel_student_strong - softmax_unlabel_teacher_weak)**2).sum(1))
    # mseloss = torch.nn.MSELoss(reduction='mean')
    unlabel_consistency_loss = consistency_loss(reshaped_softmax_unlabel_teacher_weak, reshaped_softmax_unlabel_student_strong, ema_model, model, sample[args.labeled_bs:], sample1[args.labeled_bs:],
                                                threshold=args.n_tresh, rampup_epochs=iterator, current_epoch=epoch)
    #unlabel_consistency_loss = torch.mean((softmax_unlabel_student_strong - softmax_unlabel_teacher_weak)**2)

    cc_loss_B = correlation(u_k, v_k) # uncorrelated
    cc_loss_D = correlation(p_x, p_y) # correlated
    cc_loss_B1 = correlation(u_k1, v_k1) # uncorrelated
    cc_loss_D1 = correlation(p_x1, p_y1) # correlated
    cc_loss_B2 = correlation(u_k2, v_k2) # uncorrelated
    cc_loss_D2 = correlation(p_x2, p_y2) # correlated
    #print (cc_loss_B, cc_loss_D)

    loss_decomp =  ((cc_loss_B) ** 2/ (1.01 + cc_loss_D) + (cc_loss_B1) ** 2/ (1.01 + cc_loss_D1) + (cc_loss_B2) ** 2/ (1.01 + cc_loss_D2)) / 3

    return softmax_label_student_weak, label_supervised_loss, unlabel_consistency_loss, loss_decomp

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = test_model(1).to(device)
        #model = Decompose_wdcp(1).to(device)
        #model = ut1safaFuseUNet1(1, 1).to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # folds = CN_make_dirset_train_with_folds(
    #     '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/x_t1_data/', '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/x_fa_data/', '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/y_data/',
    #     n_splits=5, seed=args.seed
    # )
    folds = CN_make_dirset_train_with_folds(
        '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/x_t1_data/', '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/x_fa_data/', '/mnt/disk1/new_code_ssl/ON_mydata3/train_data/y_data/',
        n_splits=5, seed=args.seed
    )
    # folds = CN_make_dirset_train_with_folds(
    #     '/mnt/disk1/next_w_sota/mydata/train_data/x_t1_data/', '/mnt/disk1/next_w_sota/mydata/train_data/x_fa_data/', '/mnt/disk1/next_w_sota/mydata/train_data/y_data/',
    #     n_splits=5, seed=args.seed
    # )
    # folds = CN_make_dirset_train_with_folds(
    #     '/mnt/disk1/tmi_review2025/ON_MMD/train_data/x_t1_data/', '/mnt/disk1/tmi_review2025/ON_MMD/train_data/x_fa_data/', '/mnt/disk1/tmi_review2025/ON_MMD/train_data/y_data/',
    #     n_splits=5, seed=args.seed
    # )
    #print(f"Number of folds: {len(folds)}")

    fold_results = []
    for fold_idx, fold in enumerate(folds):
        print(f"Training Fold {fold_idx + 1}/5")
        model = create_model()
        ema_model = create_model(ema=True)
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
            #model2 = nn.DataParallel(model2)

        train_dataset = CN_MyTrainDataset(
            fold['train'][0], fold['train'][1], fold['train'][2],
            x_transform=x_transforms, y_transform=y_transforms
        )
        val_dataset = CN_MyTrainDataset(
            fold['val'][0], fold['val'][1], fold['val'][2],
            x_transform=x_transforms, y_transform=y_transforms
        )

        total_slices = len(train_dataset)
        #print(f"Total training slices: {total_slices}")
        labeled_idxs = list(range(0, min(args.labeled_num, total_slices)))
        #print(f"Labeled training slices: {len(labeled_idxs)}")
        unlabeled_idxs = list(range(min(args.labeled_num, total_slices), total_slices))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs
        )

        trainloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=1, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        model.train()
        ema_model.eval()
        optimizer1 = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
        #optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.001)
        

        t = 10
        T = 200
        n_t = 0.5
        lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + np.cos(np.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + np.cos(np.pi * (epoch - t) / (T - t)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda1)

        best_val_dice = 0.0
        best_val_jaccard = 0.0
        best_val_hd95 = float('inf')
        best_val_asd = float('inf')
        #print(max_iterations // len(trainloader) + 1)
        for epoch in range(max_iterations // len(trainloader) + 1):
        #for epoch in range(100):
            epoch_loss_t1fa = 0
            epoch_dice_t1fa = 0
            step = 0
            for i_batch, sampled_batch in enumerate(trainloader):
                step += 1
                volume_batch1, volume_batch_1, gt = (
                    sampled_batch['image_t1'], sampled_batch['image_fa'], sampled_batch['label']
                )
                volume_batch1, volume_batch_1, gt = volume_batch1.cuda(), volume_batch_1.cuda(), gt.cuda()
                label_batch = gt#.squeeze()

                softmax_label_student_weak, label_supervised_loss1, unlabel_consistency_loss1, loss_dcp = train_one_step(args, model, ema_model, volume_batch1, volume_batch_1, label_batch, epoch, max_iterations // len(trainloader))
                

                # total loss
                w_int = get_current_consistency_weight(step//len(trainloader))
                # w_ext
                loss = 0.1 * label_supervised_loss1 + w_int * unlabel_consistency_loss1 + args.w_dcp * loss_dcp # 1

                epoch_loss_t1fa += float(loss.item())
                epoch_dice_t1fa += float(dice(softmax_label_student_weak[:, 1, :, :].squeeze(), label_batch[:args.labeled_bs]).item())

                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()
                #update_ema_variables(model, model2, args.ema_decay, step)

            scheduler.step()
            train_epoch_loss_t1fa = epoch_loss_t1fa / step
            train_epoch_dice_t1fa = epoch_dice_t1fa / step
            print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}, Train Loss: {train_epoch_loss_t1fa:.3f}, Train Dice: {train_epoch_dice_t1fa:.3f}")

            if train_epoch_dice_t1fa > best_val_dice:
                best_val_dice = train_epoch_dice_t1fa
                # best_val_jaccard = val_jaccard
                # best_val_hd95 = val_hd95
                # best_val_asd = val_asd
                torch.save(model.state_dict(), f"{snapshot_path}/fold_{fold_idx + 1}_best_model.pth")
            print(f"Fold {fold_idx + 1}, Epoch {epoch + 1}, Train Loss: {train_epoch_loss_t1fa:.3f}, Train Dice: {train_epoch_dice_t1fa:.3f}")

    #return train_epoch_dice_t1fa 


if __name__ == "__main__":
    x_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    y_transforms = transforms.ToTensor()

    if 'outputs_sa_ours3' not in os.listdir(os.curdir):
        os.mkdir('outputs_sa_ours3')
    if 'loss' not in os.listdir(os.curdir):
        os.mkdir('loss')

    seed = 66
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    snapshot_path = f"{args.root_path}exp/hcp/{args.exp}_{args.labeled_num}_{args.model}_aug_{args.n_aug}_tresh_{args.n_tresh}_w_dcp_{args.w_dcp}_alpha_0_1"
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    start_time = time.time()
    train(args, snapshot_path)
    end_time = time.time()
    print(f"Training time: {(end_time - start_time):.3f} s")
