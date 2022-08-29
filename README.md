## Environment
```
>> requirements.txt
```

## Data pre-processing:

### CIFAR-10
As in the original Mean Teacher repository, run the following command:

```
>> cd data-local/bin
>> ./prepare_cifar10.sh
```

### CIFAR-100
Run the similar command for CIFAR-100, taken from the [fastswa-semi-sup](https://github.com/benathi/fastswa-semi-sup/tree/master/data-local/bin) repository:
```
>> cd data-local/bin
>> ./prepare_cifar100.sh
```

### Mini-Imagenet
We took the Mini-Imagenet dataset hosted in [this repository](https://github.com/gidariss/FewShotWithoutForgetting) and pre-processed it.

Download [train.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/train.tar.gz) and [test.tar.gz](http://ptak.felk.cvut.cz/personal/toliageo/share/lpdeep/test.tar.gz), and extract them in the following directory:
```
>> ./data-local/images/miniimagenet/
```
### COVID

We took the COVID dataset hosted in [this repository](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database) and pre-processed it by script_covid.py.

```
>> ./data-local/COVID/
```

## Running the experiments:

There are two stages to our method. 

Stage 1 consists of training a network. Following command can be run to reproduce this experiment:
```
>> python train_stage1.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=10 --isMT=True --isL2=False --dataset=$DATASET



# command used to full_supervised baseline for cifar# full_supervised baseline for miniImageNet is same as "Label Propagation for Deep Semi-supervised Learning"
>> python train_stage1.py --exclude-unlabeled=True --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=10 --isMT=True --isL2=False --dataset=$DATASET

```
where ```$NOLABELS``` is the number of labeled points, ```$GPUID``` is the GPU to be used,```$DATASET``` is the name of the dataset (cifar10, cifar100, COVID or miniimagenet), and ```$LABELED_IN_BATCH``` is the number of labeled images in a batch.

After the training for Stage 1 is completed, run the following command for Stage 2, which resumes the training from the model trained in Stage 1:

```
>> python train_stage2.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=10 --isMT=False --isL2=True --dataset=$DATASET
```

## Combining with Inductive learning:


Using this [mixmatch](https://github.com/YU1ut/MixMatch-pytorch) and [fixmatch](https://github.com/kekmodel/FixMatch-pytorch) pre-train a network(we add a projector), and save the best model to use in step 2.

Stage 2:

```
>> python train_stage2_m.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=10 --isMT=True --isL2=False --dataset=$DATASET
>> python train_stage2_f.py --labeled-batch-size=$LABELED_IN_BATCH --num-labeled=$NOLABELS --gpu-id=$GPUID --label-split=10 --isMT=True --isL2=False --dataset=$DATASET
```

## other compete method

We use [this repository](https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch)

## About

- We provide the default hyperparameters in most cases in the paper. In the specific implementation, the reader needs to tune the hyperparameters according to the specific hardware and software environment and different situations, inlcluding optimizer, epochs, lr and so on.
- For CIFAR-10, Radam maybe would be better most time，and for epochs, it can be set to 240-540. Radam may not require annealing decay, ect. 
- we also provide some pre-trained models for reference.






