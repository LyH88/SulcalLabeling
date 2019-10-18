import math
import argparse
import sys
sys.path.append("./meshcnn")
import numpy as np
np.set_printoptions(precision=4)
import os
import shutil
import logging
from collections import OrderedDict

from loader_sulc import S2D3DSegLoader
from model import SphericalUNet

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import scipy

classes = range(0, 7) #0, 7
drop = [0]

def save_checkpoint(state, is_best, epoch, output_folder, filename, logger):
    # if epoch > 1:
    #     os.remove(output_folder + filename + '_%03d' % (epoch - 1) + '.pth.tar')
    torch.save(state, output_folder + filename + '_%03d' % epoch + '.pth.tar')
    if is_best:
        #logger.info("Saving new best model")
        shutil.copyfile(output_folder + filename + '_%03d' % epoch + '.pth.tar',
                        output_folder + filename + '_best.pth.tar')


def dice_loss(output, target, weights=None, ignore_index=None):
    """
    output : NxCxV Variable
    target :  NxV LongTensor
    weights : C FloatTensor
    ignore_index : int index to ignore from loss
    reference: https://discuss.pytorch.org/t/one-hot-encoding-with-autograd-dice-loss/9781/5
    """
    eps = 0.0001

    output = F.softmax(output,1)
    encoded_target = output.detach() * 0
    if ignore_index is not None:
        mask = target == ignore_index
        target = target.clone()
        target[mask] = 0
        encoded_target.scatter_(1, target.unsqueeze(1), 1)
        mask = mask.unsqueeze(1).expand_as(encoded_target)
        encoded_target[mask] = 0
    else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

    if weights is None:
        weights = 1

    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1)
    denominator = output + encoded_target

    if ignore_index is not None:
        denominator[mask] = 0
    denominator = denominator.sum(0).sum(1) + eps
    loss_per_channel = weights * (1 - (numerator / denominator))

    return loss_per_channel.sum() / output.size(1)

def iou_score(pred_cls, true_cls, nclass=7, drop=drop):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    """
    intersect_ = []
    union_ = []
    for i in range(nclass):
        if i not in drop:
            intersect = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            union = ((pred_cls == i) + (true_cls == i)).ge(1).sum().item()
            intersect_.append(intersect)
            union_.append(union)
    return np.array(intersect_), np.array(union_)


def accuracy(pred_cls, true_cls, nclass=7, drop=drop): # nclass = 7
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            tpos.append(true_positive)

            per_cls_counts.append(positive[i])
    return np.array(tpos), np.array(per_cls_counts)

def dice(pred_cls, true_cls, nclass=7, drop=drop):
    positive = torch.histc(true_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    predict = torch.histc(pred_cls.cpu().float(), bins=nclass, min=0, max=nclass, out=None)
    per_cls_counts = []
    tpos = []
    for i in range(nclass):
        if i not in drop:
            true_positive = ((pred_cls == i) + (true_cls == i)).eq(2).sum().item()
            tpos.append(true_positive)
            per_cls_counts.append((positive[i] + predict[i]) / 2)
    return np.array(tpos) / np.array(per_cls_counts)

def train(args, model, train_loader, optimizer, epoch, device, logger, keep_id=None):
    model.train()
    tot_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        if keep_id is not None:
            output = output[:, :, keep_id]
            target = target[:, keep_id]

        # loss = F.cross_entropy(output, target, weight=w)
        loss = F.cross_entropy(output, target)
        #loss=dice_loss(output,target)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        count += 1
        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    tot_loss /= count

    logger.info('[Epoch {} {} stats]: Avg loss: {:.4f}'.format(epoch, train_loader.dataset.partition, tot_loss))

    return tot_loss


def test(args, model, test_loader, epoch, device, logger, keep_id=None):
    model.eval()
    test_loss = 0
    ints_ = np.zeros(len(classes)-len(drop))
    unis_ = np.zeros(len(classes)-len(drop))
    per_cls_counts = np.zeros(len(classes)-len(drop))
    accs = np.zeros(len(classes)-len(drop))
    dices = np.zeros(len(classes)-len(drop))
    count = 0
    dices = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            n_data = data.size()[0]

            if keep_id is not None:
                output = output[:, :, keep_id]
                target = target[:, keep_id]

            #  test_loss += F.cross_entropy(output, target, weight=w).item() # sum up batch loss
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            #test_loss += dice_loss(output, target).item()  # sum up batch loss
            pred = output.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
            int_, uni_ = iou_score(pred, target)
            tpos, pcc = accuracy(pred, target)
            dices += dice(pred, target)
            ints_ += int_
            unis_ += uni_
            accs += tpos
            per_cls_counts += pcc
            count += 1
    ious = ints_ / unis_
    accs /= per_cls_counts
    #print(ints_, 'ints')
    #print(unis_, 'unis')
    #print(per_cls_counts, 'per_cls_counts')

    dices /= count
    test_loss /= count
    # print(per_cls_counts)

    logger.info('[Epoch {} {} stats]: MIoU: {:.4f}; Mean Accuracy: {:.4f}; Mean Dice: {:.4f}; Avg loss: {:.4f}'.format(
        epoch, test_loader.dataset.partition, np.mean(ious), np.mean(accs), np.mean(dices), test_loss))

    #return np.mean(np.mean(ious))
    return np.mean(ious), np.mean(accs), np.mean(dices), test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Segmentation')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--mesh_folder', type=str, default="./mesh_files",
                        help='path to mesh folder (default: ./mesh_files)')
    parser.add_argument('--data_folder', type=str, default="processed_data",
                        help='path to data folder (default: processed_data)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--max_level', type=int, default=7, help='max mesh level')
    parser.add_argument('--min_level', type=int, default=0, help='min mesh level')
    parser.add_argument('--feat', type=int, default=4, help='filter dimensions')
    parser.add_argument('--log_dir', type=str, default="log",
                        help='log directory for run')
    parser.add_argument('--decay', action="store_true", help="switch to decay learning rate")
    parser.add_argument('--optim', type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument('--resume', type=str, default=None, help="path to checkpoint if resume is needed")
    parser.add_argument('--fold', type=int, default=1, choices=[1, 2, 3, 4, 5], required=False,
                        help="choice among 3 fold for cross-validation")
    parser.add_argument('--blackout_id', type=str, default="", help="path to file storing blackout_id")
    parser.add_argument('--in_ch', type=int, default=3, help="input channels")
    parser.add_argument('--train_stats_freq', default=0, type=int,
                        help="frequency for printing training set stats. 0 for never.")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # logger and snapshot current code
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    #shutil.copy2(__file__, os.path.join(args.log_dir, "script.py"))
    #shutil.copy2("model.py", os.path.join(args.log_dir, "model.py"))
    #shutil.copy2("run.sh", os.path.join(args.log_dir, "run.sh"))

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "train.log"))
    logger.addHandler(fh)

    logger.info("%s", repr(args))

    torch.manual_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    trainset = S2D3DSegLoader(args.data_folder, "train", fold=args.fold, sp_level=args.max_level, in_ch=args.in_ch)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valset = S2D3DSegLoader(args.data_folder, "val", fold=args.fold, sp_level=args.max_level, in_ch=args.in_ch)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    testset = S2D3DSegLoader(args.data_folder, "test", fold=args.fold, sp_level=args.max_level, in_ch=args.in_ch)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = SphericalUNet(mesh_folder=args.mesh_folder, in_ch=args.in_ch, out_ch=7,
                          max_level=args.max_level, min_level=args.min_level, fdim=args.feat)
    model = nn.DataParallel(model)
    model.to(device)

    if args.blackout_id:
        blackout_id = np.load(args.blackout_id)
        keep_id = np.argwhere(np.isin(np.arange(model.module.nv_max), blackout_id, invert=True))
    else:
        keep_id = None

    start_ep = 0
    best_dice = 0
    if args.resume:
        resume_dict = torch.load(args.resume)
        start_ep = resume_dict['epoch']
        best_dice = resume_dict['dice']

        def load_my_state_dict(self, state_dict, exclude='none'):
            from torch.nn.parameter import Parameter

            own_state = self.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    continue
                if exclude in name:
                    continue
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[name].copy_(param)

        load_my_state_dict(model, resume_dict['state_dict'])



    logger.info("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))

    if args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    checkpoint_path = os.path.join(args.log_dir, 'checkpoint_latest.pth.tar')

    # training loop
    for epoch in range(start_ep + 1, args.epochs + 1):
        if args.decay:
            scheduler.step(epoch)
        _ = train(args, model, train_loader, optimizer, epoch, device, logger, keep_id)
        _, _, dice, _ = test(args, model, val_loader, epoch, device, logger, keep_id)
        _, _, _, _ = test(args, model, test_loader, epoch, device, logger, keep_id)


        if args.train_stats_freq > 0 and (epoch % args.train_stats_freq == 0):
            _ = test(args, model, train_loader, epoch, device, logger, keep_id)

        if dice > best_dice:
            best_dice = dice
            is_best = True
        else:
            is_best = False  # Do not save the best tar file
        # remove sparse matrices since they cannot be stored
        state_dict_no_sparse = [it for it in model.state_dict().items() if
                                it[1].type() != "torch.cuda.sparse.FloatTensor"]
        state_dict_no_sparse = OrderedDict(state_dict_no_sparse)

        save_checkpoint({
            'epoch': epoch,
            'state_dict': state_dict_no_sparse,
            'dice': dice,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch, checkpoint_path, "_SUNet", logger)

    # Write to logger
    #logger.info(train_loss, 'Training loss')
    #logger.info(test_loss, 'Test loss')


if __name__ == "__main__":
    main()
