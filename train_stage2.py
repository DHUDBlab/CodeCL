import datetime

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
    now = datetime.datetime.now()
    # Name of the model to be trained
    if args.isMT:
        model_name = '%s_%d_stage2_%d' % (
            args.dataset, args.num_labeled, args.label_split)
    else:
        model_name = '%s_%d_ss_split_%d_isL2_%d' % (args.dataset, args.num_labeled, args.label_split, int(args.isL2))

    checkpoint_path = './%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'w')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    train_loader, eval_loader, _, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    test = train_loader

    # Create the model
    model = create_model(num_classes, args)

    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes, args, ema=True)
    optimizer_radam = RAdam(model.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.8,
                                                           patience=3, verbose=1, eps=1e-6)
    cudnn.benchmark = True

    resume_fn = ''

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    print(best_prec1)
    model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.03
        # args.lr = param_group['lr']

    # Compute the starting accuracy
    prec1, prec5, _ = validate(eval_loader, model, global_step, args.start_epoch, isMT=args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5, _ = validate(eval_loader, ema_model, global_step, args.start_epoch, isMT=args.isMT)

    print('Resuming from:%s' % resume_fn)

    for epoch in range(args.start_epoch, args.epochs):
        # Extract features and update the pseudolabels
        print('Extracting features...')
        feats, labels, predicts, all_input = extract_features(train_loader_noshuff, model, isMT=args.isMT)
        sel_acc = train_data.update_plabels(feats, k=args.dfs_k, max_iter=20)

        validate(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)
        p_labeled_idxs = np.where(train_data.p_labels != -1)
        print("@@@@@@@", len(p_labeled_idxs[0]))
        ex_sampler = SubsetRandomSampler(p_labeled_idxs[0])
        ex_batch_sampler = BatchSampler(ex_sampler, args.batch_size, drop_last=True)
        ex_train_loader = torch.utils.data.DataLoader(train_data,
                                                      batch_sampler=ex_batch_sampler,
                                                      num_workers=args.workers,
                                                      pin_memory=True)
        print('selection accuracy: %.2f' % (sel_acc))
        #  Train for one epoch with the new pseudolabels
        if args.isMT:
            train_meter, global_step, _ = train(ex_train_loader, model, optimizer, epoch, global_step, args,
                                                ema_model=ema_model)
            train_meter, global_step, _ = train(train_loader, model, optimizer, epoch, global_step, args,
                                                ema_model=ema_model)

        else:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args)

        # Evaluate the model
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            print("Evaluating the primary model:")
            prec1, prec5, loss = validate(eval_loader, model, global_step, epoch + 1, isMT=args.isMT)

            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5, _ = validate(eval_loader, ema_model, global_step, epoch + 1, isMT=args.isMT)
                is_best = ema_prec1 > best_prec1
                best_prec1 = max(ema_prec1, best_prec1)
                scheduler.step(ema_prec1)
            else:
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
        else:
            is_best = False

        # Write to the log file and save the checkpoint
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.2f\n' %
                      (epoch,
                       loss,
                       train_meter['lr'].avg,
                       train_meter['top1'].avg,
                       train_meter['top5'].avg,
                       prec1,
                       prec5,
                       ema_prec1,
                       ema_prec5,
                       sel_acc)
                      )
            # if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'global_step': global_step,
            #         'arch': args.arch,
            #         'state_dict': model.state_dict(),
            #         'ema_state_dict': ema_model.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best, checkpoint_path, epoch + 1)

        else:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\n' %
                      (epoch,
                       train_meter['class_loss'].avg,
                       train_meter['lr'].avg,
                       train_meter['top1'].avg,
                       train_meter['top5'].avg,
                       prec1,
                       prec5,
                       )
                      )
            # if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            #     save_checkpoint({
            #         'epoch': epoch + 1,
            #         'global_step': global_step,
            #         'arch': args.arch,
            #         'state_dict': model.state_dict(),
            #         'best_prec1': best_prec1,
            #         'optimizer': optimizer.state_dict(),
            #     }, is_best, checkpoint_path, epoch + 1)


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()

    # Set the other settings
    args = load_args(args, isMT=args.isMT)

    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled, args.label_split, args.gpu_id))

    main()
