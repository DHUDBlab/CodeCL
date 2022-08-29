import torch.backends.cudnn as cudnn

from helpers import *
from mean_teacher import datasets, cli

args = None
best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1
    resume_fn = ''

    log_file = '%s/log.txt' % resume_fn
    log = open(log_file, 'w')
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, train_loader_noshuff, _, ex_train_loader = create_data_loaders(**dataset_config,
                                                                                              args=args)
    model = create_model(num_classes, args)
    ema_model = create_model(num_classes, args, ema=True)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.86,
                                                           patience=8, verbose=1, eps=1e-6)
    # Load the model from Stage pre
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['ema_state_dict'])
    # print(model)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # args.lr = 0.05
    for param_group in optimizer.param_groups:
        # param_group['lr'] = args.lr
        args.lr = param_group['lr']
    # validate_covid(eval_loader, model, global_step,1, isMT=args.isMT, isAUG=args.isAUG)
    cudnn.benchmark = True
    prec1 = 0
    prec5 = 0
    ema_prec1 = 0
    ema_prec2 = 0
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # Train for one epoch
        train_meter, global_step = train_covid(train_loader, model, optimizer, epoch, global_step, args,
                                               ema_model=ema_model)

        # Evaluate
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            print("Evaluating the primary model:")

            prec1, prec2 = validate_covid(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)
            print("Evaluating the EMA model:")
            ema_prec1, ema_prec2 = validate_covid(eval_loader, ema_model, global_step, epoch + 1, isMT=args.isMT)
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
            scheduler.step(ema_prec1)
        else:
            is_best = False
        log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                  (epoch,
                   train_meter['class_loss'].avg,
                   train_meter['lr'].avg,
                   train_meter['top1'].avg,
                   train_meter['top2'].avg,
                   prec1,
                   prec5,
                   ema_prec1,
                   ema_prec2)
                  )


if __name__ == '__main__':
    args = cli.parse_commandline_args()
    args = load_args(args, isMT=args.isMT)
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))
    main()
