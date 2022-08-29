

import torch.backends.cudnn as cudnn
import torch.optim

from helpers import *
from mean_teacher import datasets, cli

args = None
best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1
    if args.isMT:
        model_name = '%s_%d_self_P1_split_%d_isL2' % (
            args.dataset, args.num_labeled, args.label_split)
    else:
        model_name = '%s_%d_nopro_split_%d' % (args.dataset, args.num_labeled, args.label_split)

    checkpoint_path = './%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    dataset_config = datasets.__dict__[args.dataset](isTwice=(args.isMT))
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, ex_train_loader, _, _ = create_data_loaders(**dataset_config, args=args)
    model = create_model(num_classes, args)
    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes, args, ema=True)

    optimizer_radam = RAdam(model.parameters(), lr=0.2)
    prec1, prec5, loss = validate(eval_loader, model, global_step, 1, isMT=args.isMT)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=args.nesterov)
    optimizer = torch.optim.SGD(model.parameters(), 0.2,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.92,
                                                           patience=10, verbose=1, eps=1e-6)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # for param_group in optimizer.param_groups:
    #     # args.lr = param_group['lr']
    #     param_group['lr'] = 0.005
    # validate(eval_loader, model, global_step, 1, isMT=args.isMT)
    cudnn.benchmark = True
    prec1 = 0
    prec5 = 0
    ema_prec1 = 0
    ema_prec5 = 0
    loss = 0
    loss1 = 0


    for epoch in range(args.start_epoch, args.epochs):

        # Train for one epoch
        if args.isMT:
            train_meter, global_step, _ = train(ex_train_loader, model, optimizer, epoch, global_step, args,
                                                ema_model=ema_model)
            train_meter, global_step, loss1 = train(train_loader, model, optimizer, epoch, global_step, args,
                                                    ema_model=ema_model)
        else:
            # train_meter, global_step = train(ex_train_loader, model, optimizer, epoch, global_step, args)
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args)

        # Evaluate
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")

            prec1, prec5, loss = validate(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)
            print(loss, type(loss))
            # scheduler.step(prec1)
            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5, ema_loss = validate(eval_loader, ema_model, global_step, epoch + 1,
                                                          isMT=args.isMT)
                print(ema_loss)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
                scheduler.step(ema_prec1)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                # scheduler.step(prec1)
        else:
            is_best = False

        # Write to the log file and save the checkpoint
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                      (epoch,
                       train_meter['loss'].avg,
                       train_meter['lr'].avg,
                       train_meter['top1'].avg,
                       train_meter['top5'].avg,
                       prec1,
                       prec5,
                       ema_prec1,
                       ema_prec5,
                       loss1,
                       loss)
                      )
            if args.checkpoint_epochs:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)

        else:
            log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n' %
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
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))
    main()
