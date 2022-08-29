import datetime

import torch.backends.cudnn as cudnn
import torch.optim
from helpers import *
from mean_teacher import datasets, cli
from torchvision.models import Inception3, resnet18, resnet50
from mean_teacher import architectures

args = None
best_prec1 = 0
global_step = 0


def train_supervised(train_loader, model, optimizer, epoch, global_step, args):
    # global consistency_loss, res_loss, contrast_loss, ema_class_loss
    class_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none').cuda()

    meters = AverageMeterSet()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (batch_input, target, weight, c_weight) in enumerate(train_loader):
        input = batch_input

        # measure data loading time
        meters.update('data_time', time.time() - end)

        cos_lr = adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        if optimizer.param_groups[0]['lr'] > cos_lr:
            optimizer.param_groups[0]['lr'] = cos_lr

        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = Variable(input.cuda(async=True))
        target_var = Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        class_logit = model(input_var)

        loss = class_criterion(class_logit, target_var)
        loss = loss.sum() / minibatch_size

        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

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

    return meters, global_step


def validate_supervised(eval_loader, model, global_step, epoch, isMT=False):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        output1 = model(input_var)
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


def main():
    global global_step
    global best_prec1
    now = datetime.datetime.now()
    model_name = '%s_%d_supervised_%d_%d_%d_%d' % (
        args.dataset, args.num_labeled, now.day, now.hour, now.minute, now.second)
    checkpoint_path = './%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    dataset_config = datasets.__dict__[args.dataset](isTwice=(args.isMT))
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, ex_train_loader, _, _ = create_data_loaders(**dataset_config, args=args)
    if args.model_name == 'inception':
        model = Inception3(num_classes=num_classes)
    else:
        model = resnet18(num_classes=num_classes)
    model = model.cuda()
    optimizer_radam = RAdam(model.parameters(), lr=0.05)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.9,
    #                                                        patience=10, verbose=1, eps=1e-6)
    cudnn.benchmark = True
    prec1 = 0
    prec5 = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_meter, global_step = train_supervised(train_loader, model, optimizer_radam, epoch, global_step, args)

        # Evaluate
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5, loss = validate_supervised(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            # scheduler.step(prec1)
        else:
            is_best = False

        log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                  (epoch,
                   train_meter['loss'].avg,
                   train_meter['lr'].avg,
                   train_meter['top1'].avg,
                   train_meter['top5'].avg,
                   prec1,
                   prec5,
                   )
                  )


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    args = load_args(args, isMT=args.isMT)
    args.model_name = 'resnet18'
    args.epochs = 180
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))
    main()
