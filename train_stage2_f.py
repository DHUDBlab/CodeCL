from progress.bar import Bar
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from helpers import *
from mean_teacher import datasets, cli

args = None
best_prec1 = 0
global_step = 0


def create_model(args):
    import wideresnet as models
    model = models.build_wideresnet(depth=args.model_depth,
                                    widen_factor=args.model_width,
                                    dropout=0,
                                    num_classes=args.num_classes)
    model = model.cuda()
    return model


def extract_features_m(train_loader, model, isMT=False):
    model.eval()
    embeddings_all, class_all, labels_all, all_input = [], [], [], []

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        if isMT:
            X = batch_input[0]
        else:
            X = batch_input
        all_input.append(X)

        X = torch.autograd.Variable(X.cuda(), volatile=True)

        class_y, feats, _ = model(X)

        embeddings_all.append(feats.data.cpu())
        class_all.append(class_y.data.cpu())
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    plabels_all = torch.cat(class_all).numpy()
    return plabels_all, embeddings_all


def validate_m(eval_loader, model):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()  # 去掉未标记数据
    meters = AverageMeterSet()
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input.cuda(async=True), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        output1, _, _ = model(input_var)
        output1_var = torch.autograd.Variable(output1.cuda(async=True), volatile=True)
        class_loss = class_criterion(output1_var, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1_var.data, target_var.data, topk=(1, 5))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_loss {loss.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5'], loss=meters['class_loss']))

    return meters['top1'].avg, meters['top5'].avg


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def train_m(trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()
    input_x, target_x, input_u, target_u = [], [], [], []

    bar = Bar('Training', max=args.train_iteration)
    train_iter = iter(trainloader)
    # unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        for input, target in train_iter:
            print()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1, 1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return losses.avg, losses_x.avg, losses_u.avg,


def train(train_loader, model, optimizer, epoch, global_step, args, ema_model=None):
    class_criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none').cuda()
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
        input = batch_input[0]
        ema_input = batch_input[1]

        meters.update('data_time', time.time() - end)
        cos_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        if optimizer.param_groups[0]['lr'] > cos_lr:
            optimizer.param_groups[0]['lr'] = cos_lr
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
            ema_logit, feats, z1 = ema_model(ema_input_var)
            class_logit, ema_feats, z2 = model(input_var)
            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
            z1 = Variable(z1.detach().data, requires_grad=False)
            ema_feats = Variable(ema_feats.detach().data, requires_grad=False)
            z2_ = ema_model(feats, iss=True)
            z2_ = Variable(z2_.detach().data, requires_grad=False)
            z1_ = model(ema_feats, iss=True)
            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)
                meters.update('cons_weight', consistency_weight)
                sim = consistency_weight * consistency_criterion(z1, z2) / minibatch_size
                sim_ = consistency_weight * consistency_criterion(z1_, z2_) / minibatch_size
                contrast_loss = 0.5 * sim + 0.5 * sim_
                consistency_loss = consistency_weight * consistency_criterion(class_logit, ema_logit) / minibatch_size
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
        epoch_loss += loss
        if isMT:
            loss = loss + consistency_loss + contrast_loss

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
                'epoch: [{0}][{1}/{2}]'
                'lr{meters[lr]:.6f}\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'loss {meters[class_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
    print("epoch_loss", epoch_loss)
    return meters, global_step


def main():
    global global_step
    global best_prec1
    checkpoint_path = ''
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    # Create the dataset and loaders---zhelishiLPde
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, _, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    # Create the model
    model = create_model(args)
    ema_model = create_model(args)

    # optimizer = torch.optim.Adam(model.parameters(), 0.002)
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    args.lr = 0.0000005
    optimizer = optim.SGD(grouped_parameters, lr=0.002,
                          momentum=0.9, nesterov=args.nesterov)
    cudnn.benchmark = True

    # Name of the model trained in Stage 1
    resume_fn = ''

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn, map_location='cuda:0')
    best_prec1 = checkpoint['best_acc']
    model.load_state_dict(checkpoint['ema_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    # for param_group in optimizer.param_groups:
    #     args.lr = param_group['lr']
    print(model)
    prec1, prec5 = validate_m(eval_loader, model)
    ema_prec1, ema_prec5 = validate_m(eval_loader, ema_model)

    print('Resuming from:%s' % resume_fn)

    for epoch in range(args.start_epoch, args.epochs):
        # Extract features and update the pseudolabels
        print('Extracting features...')
        predicts, feats = extract_features_m(train_loader_noshuff, ema_model, isMT=args.isMT)
        sel_acc = train_data.update_plabels(feats, k=args.dfs_k, max_iter=20)

        p_labeled_idxs = np.where(train_data.p_labels != -1)
        ex_sampler = SubsetRandomSampler(p_labeled_idxs[0])
        ex_batch_sampler = BatchSampler(ex_sampler, args.batch_size, drop_last=True)
        ex_train_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_sampler=ex_batch_sampler,
                                                      num_workers=args.workers,
                                                      pin_memory=True)
        print('selection accuracy: %.2f   ' % (sel_acc))

        #  Train for one epoch with the new pseudolabels

        train_meter, global_step = train(ex_train_loader, model, optimizer, epoch, global_step, args,
                                         ema_model=ema_model)
        train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args,
                                         ema_model=ema_model)
        print("Evaluating the primary model:")
        prec1, prec5 = validate_m(eval_loader, model)
        print("Evaluating the EMA model:")
        ema_prec1, ema_prec5 = validate_m(eval_loader, ema_model)
        is_best = ema_prec1 > best_prec1
        best_prec1 = max(ema_prec1, best_prec1)
        log.write('%d\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                  (epoch,
                   train_meter['lr'].avg,
                   train_meter['top1'].avg,
                   train_meter['top5'].avg,
                   prec1,
                   prec5,
                   ema_prec1,
                   ema_prec5)
                  )


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    print(args)
    args = load_args(args, isMT=args.isMT)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))

    main()
