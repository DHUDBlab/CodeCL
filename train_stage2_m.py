from torch.backends import cudnn
from torch.utils.data import DataLoader

import wideresnet
from helpers import *
from mean_teacher import datasets, cli

args = None
best_prec1 = 0
global_step = 0


def create_model(num_class, ema=False):
    model = wideresnet.WideResNet(num_classes=num_class)
    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()

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

        X = torch.autograd.Variable(X.cuda())

        class_y, feats = model(X)

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

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\tClass_loss {loss.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5'], loss=meters['class_loss']))

    return meters['top1'].avg, meters['top5'].avg


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
            ema_logit, _ = ema_model(ema_input_var)
            class_logit, _ = model(input_var)
            ema_logit = Variable(ema_logit.detach().data, requires_grad=False)

            ema_class_loss = class_criterion(ema_logit, target_var)
            ema_class_loss = ema_class_loss.sum() / minibatch_size
            meters.update('ema_class_loss', ema_class_loss.item())

            if args.consistency:
                consistency_weight = get_current_consistency_weight(epoch, args)
                meters.update('cons_weight', consistency_weight)

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
            loss = loss + consistency_loss

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
                'lr{meters[lr]:.4f}\t'
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

    model_name = '%s_%d_mixmatch_ss' % (args.dataset, args.num_labeled)

    checkpoint_path = './%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, _, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    # Create the model
    model = create_model(num_classes)
    ema_model = create_model(num_classes, ema=True)

    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    cudnn.benchmark = True
    # # Name of the model trained in Stage 1
    resume_fn = ''

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_acc']
    model.load_state_dict(checkpoint['ema_state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.lr = 0.0002
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print(model)
    validate_m(eval_loader, model)

    validate_m(eval_loader, ema_model)

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
    # Get the command line arguments
    args = cli.parse_commandline_args()
    print(args)

    # Set the other settings
    args = load_args(args, isMT=args.isMT)

    # Use only the specified GPU

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))

    main()
