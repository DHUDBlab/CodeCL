import math
import os
import parser
import re
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torchvision.datasets
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from lp import db_semisuper
from mean_teacher import architectures, data, losses, ramps
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())

    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.ckpt'
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    # LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print("--- checkpoint copied to %s ---" % best_path)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)
    lpdir = os.path.join(datadir, args.lp_subdir)
    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size, args.fully_supervised])
    dataset = db_semisuper.DBSS(traindir, train_transformation)
    if not args.fully_supervised and args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.fully_supervised:
        sampler = SubsetRandomSampler(range(len(dataset)))
        dataset.labeled_idx = range(len(dataset))
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:  # labeled_batch_size
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=2,
                                               pin_memory=True)
    #
    ex_sampler = SubsetRandomSampler(labeled_idxs)
    ex_batch_sampler = BatchSampler(ex_sampler, args.batch_size, drop_last=True)
    ex_train_loader_1 = torch.utils.data.DataLoader(dataset,
                                                    batch_sampler=ex_batch_sampler,
                                                    num_workers=2,
                                                    pin_memory=True)
    train_loader_noshuff = torch.utils.data.DataLoader(dataset,
                                                       batch_size=args.batch_size * 2,
                                                       shuffle=False,
                                                       num_workers=2,  # Needs images twice as fast
                                                       pin_memory=True,
                                                       drop_last=False)

    eval_dataset = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False)

    return train_loader, eval_loader, ex_train_loader_1, train_loader_noshuff, dataset


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(num_classes, args, ema=False):
    model_factory = architectures.__dict__[args.arch]
    model_params = dict(pretrained=args.pretrained, num_classes=num_classes, isL2=args.isL2,
                        double_output=args.double_output)
    model = model_factory(**model_params)
    model = nn.DataParallel(model).cuda()
    if ema:
        for param in model.parameters():
            param.detach_()

    return model


def train(train_loader, model, optimizer, epoch, global_step, args, ema_model=None):
    class_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    if ema_model is not None:
        isMT = True
    else:
        isMT = False

    # switch to train mode
    model.train()
    if isMT:
        ema_model.train()

    end = time.time()
    epoch_loss = 0
    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:
            input = batch_input[0]
            ema_input = batch_input[1]
        else:
            input = batch_input

        # measure data loading time
        meters.update('data_time', time.time() - end)
        cos_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        if optimizer.param_groups[0]['lr'] > cos_lr:
            optimizer.param_groups[0]['lr'] = cos_lr

        meters.update('lr', optimizer.param_groups[0]['lr'])
        with torch.no_grad():
            input_var = Variable(input.cuda(async=True))
            target_var = Variable(target.cuda(async=True))
            weight_var = Variable(weight.cuda(async=True))
            c_weight_var = Variable(c_weight.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if isMT:
            ema_input_var = torch.autograd.Variable(ema_input.cuda(async=True), volatile=True)

            ema_logit, _, z1, ema_feats = ema_model(ema_input_var)
            class_logit, cons_logit, z2, feats = model(input_var)
            class_logit1, _, _, _ = model(ema_input_var)
            # ema_logit, _,ema_feats = ema_model(ema_input_var)
            # class_logit, cons_logit,feats = model(input_var)
            # class_logit1, _, _ = model(ema_input_var)
            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
            z1 = Variable(z1.detach().data, requires_grad=False)
            ema_feats = Variable(ema_feats.detach().data, requires_grad=False)
            z2_ = ema_model(feats, iss=True)
            # # z2_ = Variable(z2_.detach().data, requires_grad=False)
            z1_ = model(ema_feats, iss=True)
            z1_ = Variable(z1_.detach().data, requires_grad=False)

            if args.logit_distance_cost >= 0:
                res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.item())
            else:
                res_loss = 0

            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)
                meters.update('cons_weight', consistency_weight)
                sim = consistency_weight * consistency_criterion(z1, z2) / minibatch_size
                sim_ = consistency_weight * consistency_criterion(z1_, z2_) / minibatch_size
                contrast_loss = 0.5 * sim + 0.5 * sim_
                # sim = torch.cosine_similarity(z1, z2, dim=1)
                # sim_ = torch.cosine_similarity(z1_, z2_, dim=1)
                # contrast_loss1 = (- sim.sum()) / minibatch_size
                # contrast_loss2 = (- sim_.sum()) / minibatch_size
                # contrast_loss = 0.5 * contrast_loss2 + 0.5 * contrast_loss1

                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)


        else:
            class_logit, _ = model(input_var)

        loss = class_criterion(class_logit, target_var)
        loss = loss * weight_var.float()
        loss = loss * c_weight_var
        loss = loss.sum() / minibatch_size

        if isMT:
            loss = loss + consistency_loss + res_loss + contrast_loss  # + loss1  # + mix_loss
        epoch_loss = epoch_loss + loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        if isMT:
            ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
            meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
            meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if isMT:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]'
                'LR {meters[lr]:.4f}\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Loss {meters[loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))

    return meters, global_step, epoch_loss


def train_covid(train_loader, model, optimizer, epoch, global_step, args, ema_model=None):
    class_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    if ema_model is not None:
        isMT = True
    else:
        isMT = False

    # switch to train mode
    model.train()
    if isMT:
        ema_model.train()

    end = time.time()

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:
            input = batch_input[0]
            ema_input = batch_input[1]
        else:
            input = batch_input

        # measure data loading time
        meters.update('data_time', time.time() - end)
        #

        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = torch.autograd.Variable(input.cuda(async=True))
        target_var = torch.autograd.Variable(target.cuda(async=True))
        weight_var = torch.autograd.Variable(weight.cuda(async=True))
        c_weight_var = torch.autograd.Variable(c_weight.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        if isMT:
            ema_input_var = torch.autograd.Variable(ema_input.cuda(async=True), volatile=True)
            ema_logit, _, z1, ema_feats = ema_model(ema_input_var)
            class_logit, cons_logit, z2, feats = model(input_var)
            class_logit1, _, _, _ = model(ema_input_var)
            # ema_logit, _,ema_feats = ema_model(ema_input_var)
            # class_logit, cons_logit,feats = model(input_var)
            # class_logit1, _, _ = model(ema_input_var)

            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
            z1 = Variable(z1.detach().data, requires_grad=False)
            ema_feats = Variable(ema_feats.detach().data, requires_grad=False)
            z2_ = ema_model(feats, iss=True)
            # # z2_ = Variable(z2_.detach().data, requires_grad=False)
            z1_ = model(ema_feats, iss=True)
            z1_ = Variable(z1_.detach().data, requires_grad=False)
            if args.logit_distance_cost >= 0:
                res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
                meters.update('res_loss', res_loss.item())
            else:
                res_loss = 0

            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)
                meters.update('cons_weight', consistency_weight)
                # sim = consistency_weight * consistency_criterion(z1, z2) / minibatch_size
                # sim_ = consistency_weight * consistency_criterion(z1_, z2_) / minibatch_size
                # contrast_loss = 0.5 * sim + 0.5 * sim_
                sim = torch.cosine_similarity(z1, z2, dim=1)
                sim_ = torch.cosine_similarity(z1_, z2_, dim=1)
                contrast_loss1 = (- sim.sum()) / minibatch_size
                contrast_loss2 = (- sim_.sum()) / minibatch_size
                contrast_loss = 0.5 * contrast_loss2 + 0.5 * contrast_loss1
                consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit) / minibatch_size
                meters.update('cons_loss', consistency_loss.item())
            else:
                consistency_loss = 0
                meters.update('cons_loss', 0)


        else:
            class_logit, _ = model(input_var)

        loss = class_criterion(class_logit, target_var)
        loss = loss * weight_var.float()
        loss = loss * c_weight_var
        loss = loss.sum() / minibatch_size
        meters.update('class_loss', loss.item())

        if isMT:
            loss = loss + consistency_loss + res_loss + contrast_loss
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 2))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top2', prec5[0], labeled_minibatch_size)
        meters.update('error2', 100. - prec5[0], labeled_minibatch_size)

        if isMT:
            ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 2))
            meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
            meters.update('ema_top2', ema_prec5[0], labeled_minibatch_size)
            meters.update('ema_error2', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if isMT:
            update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print(
                'Epoch: [{0}][{1}/{2}]'
                'LR {meters[lr]:.4f}\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top2]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))

    return meters, global_step


def validate(eval_loader, model, global_step, epoch, isMT=False):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()  # 去掉未标记数据
    meters = AverageMeterSet()
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        if isMT:
            output1, _, _, _ = model(input_var)
        else:
            output1, _ = model(input_var)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_loss{loss.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5'], loss=meters['class_loss']))

    return meters['top1'].avg, meters['top5'].avg, meters['class_loss'].avg


def validate_covid(eval_loader, model, global_step, epoch, isMT=False, isAUG=False):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        if isMT:
            output1, _, _, _ = model(input_var)

        else:
            output1, _ = model(input_var)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output1.data, target_var.data, topk=(1, 2))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top2', prec2[0], labeled_minibatch_size)
        meters.update('error2', 100.0 - prec2[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top2.avg:.3f}\tClass_loss{loss.avg:.3f}'
          .format(top1=meters['top1'], top2=meters['top2'], loss=meters['class_loss']))

    return meters['top1'].avg, meters['top2'].avg, meters['class_loss'].avg


# adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    # lr_rampdown_epochs
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)
    return lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res


def extract_features(train_loader, model, isMT=False):
    model.eval()
    embeddings_all, class_all, labels_all, all_input = [], [], [], []

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:
            X = batch_input[0]
        else:
            X = batch_input
        all_input.append(X)
        y = batch_input[1]

        X = torch.autograd.Variable(X.cuda())

        y = torch.autograd.Variable(y.cuda(async=True))
        if isMT:
            class_y, _, _, feats = model(X)
        else:
            class_y, feats = model(X)

        embeddings_all.append(feats.data.cpu())
        class_all.append(class_y.data.cpu())
        labels_all.append(y.data.cpu())
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    labels_all = torch.cat(labels_all).numpy()
    class_all = torch.cat(class_all)
    all_input = torch.cat(all_input)
    # print(class_all)
    return embeddings_all, labels_all, class_all, all_input


def extract_features_covid(train_loader, model, isMT=False):
    model.eval()
    embeddings_all, class_all, labels_all, all_input = [], [], [], []

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        print("batch_input.shape", batch_input[0].shape)
        if isMT:
            X = batch_input[0]
        else:
            X = batch_input
        all_input.append(X)
        y = batch_input[1]

        X = torch.autograd.Variable(X.cuda())

        y = torch.autograd.Variable(y.cuda(async=True))
        if isMT:
            class_y, _, _, feats = model(X)
        else:
            class_y, feats = model(X)

        embeddings_all.append(feats.data.cpu())
        class_all.append(class_y.data.cpu())
        labels_all.append(y.data.cpu())
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    labels_all = torch.cat(labels_all).numpy()
    class_all = torch.cat(class_all)
    all_input = torch.cat(all_input)
    # print(class_all)
    return embeddings_all, labels_all, class_all, all_input


def load_args(args, isMT=False):
    label_dir = './data-local'
    if args.dataset == "cifar100":
        args.batch_size = 100
        args.lr = 0.2
        args.test_batch_size = args.batch_size

        args.epochs = 240
        args.lr_rampdown_epochs = 260
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)
        args.arch = 'cifar_cnn'

    elif args.dataset == "cifar10":
        args.batch_size = 100
        args.test_batch_size = args.batch_size
        args.epochs = 540
        args.lr = 0.05

        args.lr_rampdown_epochs = 560
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0

        args.weight_decay = 2e-4
        args.arch = 'cifar_cnn'
        args.num_classes = 10

        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)

    elif args.dataset == "COVID":

        args.train_subdir = 'train+val'
        args.evaluation_epochs = 10

        args.epochs = 240
        args.batch_size = 128
        args.lr = 0.15
        args.test_batch_size = args.batch_size

        args.epochs = 180
        args.lr_rampdown_epochs = 280
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)
        args.arch = 'resnet18'

    elif args.dataset == "miniimagenet":

        args.train_subdir = 'train'
        args.evaluation_epochs = 1

        args.epochs = 20
        args.batch_size = 128
        args.lr = 0.05
        args.test_batch_size = args.batch_size

        args.lr_rampdown_epochs = 30
        args.ema_decay = 0.97

        args.logit_distance_cost = 0.01
        args.pretrained = True
        args.consistency = 100.0
        args.weight_decay = 2e-4
        args.labels = '%s/labels/%s/%d_balanced_labels/%d.txt' % (
            label_dir, args.dataset, args.num_labeled, args.label_split)
        args.arch = 'MyResNet'
    else:
        sys.exit('Undefined dataset!')

    if isMT:
        args.double_output = True
    else:
        args.double_output = False

    return args


class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        buffer=[[None, None, None] for _ in range(10)])
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = group['buffer'][int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                    N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size * group['lr'], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group['weight_decay'] != 0:
                        p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)
                    p_data_fp32.add_(-step_size * group['lr'], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss
