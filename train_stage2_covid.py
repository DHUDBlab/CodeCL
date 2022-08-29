import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from helpers import *
from mean_teacher import datasets, cli

args = None
best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1
    checkpoint_path = ''
    # Name of the model to be trained
    model_name = '%s_%d_covid_ss_split_%d_isL2_%d' % (
    args.dataset, args.num_labeled, args.label_split, int(args.isL2))
    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, train_loader_noshuff, train_data, _ = create_data_loaders(**dataset_config, args=args)

    # Create the model
    model = create_model(num_classes, args)

    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes, args, ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    cudnn.benchmark = True

    # Name of the model trained in Stage 1
    resume_fn = ''

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.lr = 0.05
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        # args.lr = param_group['lr']

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.78, patience=10)
    # Compute the starting accuracy
    prec1, prec5 = validate_covid(eval_loader, model, global_step, args.start_epoch, isMT=args.isMT)
    ema_prec1, ema_prec5 = validate_covid(eval_loader, ema_model, global_step, args.start_epoch, isMT=args.isMT)

    print('Resuming from:%s' % resume_fn)

    for epoch in range(args.start_epoch, args.epochs):
        # Extract features and update the pseudolabels
        print('Extracting features...')
        feats, labels, predicts, all_input = extract_features_covid(train_loader_noshuff, model, isMT=args.isMT)
        sel_acc = train_data.update_plabels(feats, k=args.dfs_k, max_iter=20)

        p_labeled_idxs = np.where(train_data.p_labels != -1)
        ex_sampler = SubsetRandomSampler(p_labeled_idxs[0])
        ex_batch_sampler = BatchSampler(ex_sampler, args.batch_size, drop_last=True)
        ex_train_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_sampler=ex_batch_sampler,
                                                      num_workers=args.workers,
                                                      pin_memory=True)
        print('selection accuracy: %.2f   ' % sel_acc)
        #  Train for one epoch with the new pseudolabels
        train_meter, global_step = train_covid(ex_train_loader, model, optimizer, epoch, global_step, args,
                                               ema_model=ema_model)
        train_meter, global_step = train_covid(train_loader, model, optimizer, epoch, global_step, args,
                                               ema_model=ema_model)

        # Evaluate the model
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5 = validate_covid(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)
            print("Evaluating the EMA model:")
            ema_prec1, ema_prec5 = validate_covid(eval_loader, ema_model, global_step, epoch + 1, isMT=args.isMT)
            scheduler.step(ema_prec1)
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        # Write to the log file and save the checkpoint

        log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                  (epoch,
                   train_meter['class_loss'].avg,
                   train_meter['lr'].avg,
                   train_meter['top1'].avg,
                   train_meter['top2'].avg,
                   prec1,
                   prec5,
                   ema_prec1,
                   ema_prec5)
                  )


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()

    # Set the other settings
    args = load_args(args, isMT=args.isMT)

    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))

    main()
