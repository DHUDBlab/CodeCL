
import os
import torchvision.transforms as transforms
from . import data
from .utils import export


@export
def cifar10(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5), contrast=(0.5), hue=(0.5)),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5,
            #                        saturation=0.4, hue=0.5),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats),

        ]))


    else:
        train_transformation = transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.5), contrast=(0.5), hue=(0.5)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats),
        ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    myhost = os.uname()[1]
    data_dir = './data-local/images/cifar/cifar10/by-image'
    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 10
    }


@export
def cifar100(isTwice=True):
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])  # should we use different stats - do this
    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=(0.5), contrast=(0.5), hue=(0.5)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            # transforms.Resize(299),
            # transforms.CenterCrop(299),
            # data.RandomTranslateWithReflect(4),
            # transforms.RandomHorizontalFlip(),
            # # transforms.ColorJitter(brightness=(0.5), contrast=(0.5), hue=(0.5)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
            data.RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.1), contrast=(0.1), hue=(0.1)),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        # transforms.Resize(299),
        # transforms.CenterCrop(299),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = './data-local/images/cifar/cifar100/by-image'

    print("Using CIFAR-100 from", data_dir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }


@export
def miniimagenet(isTwice=True):
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=(0.5), contrast=(0.5), hue=(0.5)),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                       saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = data.TransformOnce(transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = './data-local/images/miniimagenet'

    print("Using mini-imagenet from", data_dir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 100
    }

@export
def covid(isTwice=True):
    mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
    std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]

    channel_stats = dict(mean=mean_pix,
                         std=std_pix)

    if isTwice:
        train_transformation = data.TransformTwice(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(10),
            # transforms.RandomCrop(84, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
    else:
        train_transformation = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    data_dir = './data-local/images/COVID19'

    print("Using COVID19 from", data_dir)

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': data_dir,
        'num_classes': 3
    }
