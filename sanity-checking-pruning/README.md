# Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot
This repository contains the code for reproducing the results in the following paper:

Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot. [[arXiv]](https://arxiv.org/abs/2009.11094)

Jingtong Su*, Yihang Chen*, [Tianle Cai*](https://tianle.website/), Tianhao Wu, Ruiqi Gao, [Liwei Wang](http://www.liweiwang-pku.com/), [Jason D. Lee](https://jasondlee88.github.io/) (* equal contribution, reverse alphabetical order).

NeurIPS 2020.

In the following, we would like to briefly summarize our paper and give fine introduction of our code files.

## Paper Summary

The [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) (LT) suggests that there exists a sub-network inside a full-network which can be trained isolated with its original initializaion. LT finds this sub-network by training a full network first and prune it. Based on this observation, several prune-at-init methods were introduced to prune individual weights of a full-network without the training procedure while don't hurt the performance much. Two state-of-the-art methods are [GraSP](https://arxiv.org/abs/2002.07376) and [SNIP](https://arxiv.org/abs/1810.02340).

In this paper, we conduct sanity checks for the above methods (LT, GraSP and SNIP) and surprisingly find that: (1) These methods which aims to find good subnetworks of the randomly-initialized network (which we call ''initial tickets''), hardly exploits any information from the training data; (2) For the pruned networks obtained by these methods, randomly changing the preserved weights in each layer, while keeping the total number of preserved weights unchanged per layer, does not affect the final performance. These findings inspire us to choose a series of simple *data-independent* prune ratios for each layer, and randomly prune each layer accordingly to get a subnetwork (which we call ''random tickets''). We call these ratios and the corresponding pruning method ''smart ratio''. (Note that these ratios are very easy to obtain and *without careful tuning*, but still gain performance similar to or higher than those tickets generated by carefully designed algorithms!)

Some of the pruning methods didn't have official PyTorch source code (at the time we did this work), so we had to reimplement them or to inherit from some existing work.

The framework of the code is inherited from [Rethinking the value of Network Pruning](https://github.com/Eric-mingjie/rethinking-network-pruning). For the [GraSP](https://arxiv.org/abs/2002.07376) code, we inherit from its official implementation [GitHub_GraSP](https://github.com/alecwangcq/GraSP). We would like to express our heartfelt thanks to the authors of these codes.


## Dependencies

pip install torch==1.5.0 torchvision==0.6.0, tensorboardX


## Introduction of existing Python Files

We have three commonly used python files: baseline.py, LT_prune.py, and train_ticket.py.

baseline.py is used to train a **full-network** on CIFAR-10/CIFAR-100/Tiny-Imagenet datasets. The commonly-used args are --dataset, --arch, --depth and --save_dir. For our shufflePixel sanity-check method, please check the code with the arg ''--shufflePixel''. 
After running baseline.py, it will output several files to the --save_dir path: init.pth.tar (weights at init), checkpoint.pth.tar (weights at the final epoch, should be used to generate LT masks) and a log file recording the training/test acc/loss.

LT_prune.py is used to **prune a given network with LT method** and show the test accuracy before and after pruning. The commonly-used args are --prune_ratio, --arch, --depth, --dataset and --save_dir. NOTE THAT IT DOESN'T SUPPORT TINY-IMAGENET!
LT_prune.py will output 2 files to the --save_dir path: pruned.pth.tar (the pruned model containing the weights and the masks, will be used in train_ticket.py with --resume arg) and a text file recording the acc before and after pruning.

train_ticket.py is used for **training the tickets** generated by the pruning methods. The commonly-used args are --dataset, --arch, --depth, --save_dir and --resume/--model. --resume is used to read the masks for the LT method, and --model is used to read the weights for initialization (can be used with LT, GraSP, SNIP and our Smart Ratio). The init-pruning methods (i.e. GraSP/SNIP/Smart Ratio) are activated with the args --GraSP/--SNIP/--smart_ratio, and under this situation it is not necessary to use the --resume arg any more.
The train_ticket.py file will output several files to the --save_dir path: scratch.pth.tar (recording the weights at the final epoch, generally will NOT be used) and a log file recording the training/test loss/acc. Also it will output an event file to the --writerdir path, which can be visualized by tensorboard(X), containing the train/test acc/loss curve.

For both baseline.py and train_ticket.py, the NUM OF EPOCHS are set to be 160 (CIFAR-10/100) and 300 (Tiny-Imagenet) automatically. Specifically, for training Tiny-Imagenet the code is written **DEAD**, you cannot modify this setting outside by the --epochs arg. Please note that.

**Next we'll give some typical examples to reproduce our experiment results. The others are easy to extend.**


## baseline.py 

xmc's baseline code

    python baseline.py --dataset cifar100 --arch resnet --depth 164 --save_dir model_baseline/ --train-batch 256 --test-batch 256 --gpu-id 1
    python baseline.py --dataset cifar100 --arch vgg19_bn --depth 19 --save_dir model_baseline/vgg19/ --train-batch 256 --test-batch 256 --gpu-id 1

## original baseline.py 


```shell
// This code will use the full VGG19 network to train on CIFAR-10 for 160 epochs.
python baseline.py --dataset cifar10 --arch vgg19_bn --depth 19 \
    --save_dir [PATH TO SAVE THE MODEL]
    
    
// This code will use the full ResNet32 network to train on CIFAR-100 for 160 epochs.   
python baseline.py --dataset cifar100 --arch resnet --depth 32 \ 
    --save_dir [PATH TO SAVE THE MODEL]


// This code will use the full ResNet32 network to train on Tiny-Imagenet for 300 epochs.
python baseline.py --dataset tinyimagenet --arch resnet --depth 32 \ 
    --save_dir [PATH TO SAVE THE MODEL]

// This code will use the full VGG19 network to train on randLabel&shufflePixel CIFAR-10.
python baseline.py --dataset cifar10 --arch vgg19_bn --depth 19 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --shufflePixel 1
    
// This code is part of our Sanity-Check methods: Half Dataset. It will use the full VGG19 network to train on half of the  CIFAR-10 dataset by turning off the shuffle mode of the dataloader and only using a half of the batches. (please see the code for detail)
python baseline.py --dataset cifar10 --arch vgg19_bn --depth 19 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --max_batch_idx 390    
```

## LT_prune.py

```shell
// This code will prune p of the weights of a given VGG19 network.
python LT_prune.py --dataset cifar10 --arch vgg19_bn --depth 19 \
    --prune_ratio [p] --resume [PATH TO THE MODEL TO BE PRUNED, e.g. THE MODEL GENERATED BY baseline.py] \
    --save_dir [PATH TO SAVE THE PRUNED MODEL]
  
// This code will prune p of the weights of a given ResNet32 network.
python LT_prune.py --dataset cifar100 --arch resnet --depth 32 \
    --prune_ratio [p] --resume [PATH TO THE MODEL TO BE PRUNED, e.g. THE MODEL GENERATED BY baseline.py] \
    --save_dir [PATH TO SAVE THE PRUNED MODEL]
```


## train_ticket.py

```shell
// This code will use the initialization of --model and the mask (generated by LT method) of --resume with VGG19 architecture to train on CIFAR-10 for 160 epochs.
python train_ticket.py --dataset cifar10 --arch vgg19_bn --depth 19 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \
    --resume [PATH TO THE STORED PRUNED MODEL WITH MASK] \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]

// This code will use the initialization of --model and the mask(generated by LT method) of --resume with ResNet32 architecture to train on CIFAR-10 for 160 epochs.
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \
    --resume [PATH TO THE STORED PRUNED MODEL WITH MASK] \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]
    
// This code will use the initialization of --model and generate the mask by this initialization and GraSP method with ResNet32 architecture to train on CIFAR-10 for 160 epochs.    
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \ (not necessary)
    --GraSP 1 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]
    
// This code will use the initialization of --model and generate the mask by this initialization and SNIP method with ResNet32 architecture to train on CIFAR-10 for 160 epochs.    
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \ (not necessary)
    --SNIP 1 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]

// This code will use the initialization of --model and generate the mask by our Smart Ratio method with ResNet32 architecture to train on CIFAR-10 for 160 epochs, i.e. Random Tickets.
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \ (not necessary)
    --smart_ratio 1 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]

// This code is part of our Sanity-Check methods: rearrange. This code will use the initialization of --model and REARRANGE the mask(generated by LT method) of --resume with ResNet32 architecture to train on CIFAR-10 for 160 epochs.
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \
    --resume [PATH TO THE STORED PRUNED MODEL WITH MASK] \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE] \
    --rearrange 1

// This code is part of our Sanity-Check methods: shuffleWeights. This code will use the initialization of --model and the mask(generated by LT method) of --resume with ResNet32 architecture, then SHUFFLE the UNMASKED WEIGHTS and train on CIFAR-10 for 160 epochs.
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \
    --resume [PATH TO THE STORED PRUNED MODEL WITH MASK] \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE]
    --shuffle_unmasked_weights 1
    
// This code is part of our Hybrid Tickets. This code will use the initialization of --model and generate the mask by our Smart Ratio method together to retain the largest-magnitude weights with ResNet32 architecture and train on CIFAR-10 for 160 epochs, i.e. Random Tickets.  
python train_ticket.py --dataset cifar10 --arch resnet --depth 32 \
    --lr 0.1 \
    --model [PATH TO THE STORED INITIALIZATION] \
    --smart_ratio 1 \
    --save_dir [PATH TO SAVE THE MODEL] \
    --writerdir [PATH TO SAVE THE TENSORBOARD EVENT FILE] \
    --hybrid 1
    
```



## Contact
Feel free to discuss papers/code with us through issues/emails!

jtsu at pku.edu.cn

yihang.chen at pku.edu.cn

## Citation
If you use our code in your research, please cite:

~~~
@article{su2020sanity,
  title={Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot},
  author={Su, Jingtong and Chen, Yihang and Cai, Tianle and Wu, Tianhao and Gao, Ruiqi and Wang, Liwei and Lee, Jason D},
  journal={arXiv preprint arXiv:2009.11094},
  year={2020}
}
