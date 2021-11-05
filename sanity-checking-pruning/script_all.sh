# fine-tune
python train_ticket.py --dataset cifar100 --arch vgg19_bn --depth 19 --lr 0.1 --model model_baseline/vgg19/init.pth.tar --resume model_prune/vgg0.3/pruned.pth.tar --save_dir finetune/vgg0.3/ --writerdir tensorboard/vgg0.3 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch vgg19_bn --depth 19 --lr 0.1 --model model_baseline/vgg19/init.pth.tar --resume model_prune/vgg0.6/pruned.pth.tar --save_dir finetune/vgg0.6/ --writerdir tensorboard/vgg0.6 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch vgg19_bn --depth 19 --lr 0.1 --model model_baseline/vgg19/init.pth.tar --resume model_prune/vgg0.7/pruned.pth.tar --save_dir finetune/vgg0.7/ --writerdir tensorboard/vgg0.7 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch vgg19_bn --depth 19 --lr 0.1 --model model_baseline/vgg19/init.pth.tar --resume model_prune/vgg0.8/pruned.pth.tar --save_dir finetune/vgg0.8/ --writerdir tensorboard/vgg0.8 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch vgg19_bn --depth 19 --lr 0.1 --model model_baseline/vgg19/init.pth.tar --resume model_prune/vgg0.9/pruned.pth.tar --save_dir finetune/vgg0.9/ --writerdir tensorboard/vgg0.9 --gpu-id 1 --train-batch 256 --test-batch 256

python train_ticket.py --dataset cifar100 --arch resnet --depth 164 --lr 0.1 --model model_baseline/resnet164/init.pth.tar --resume model_prune/resnet164.3/pruned.pth.tar --save_dir finetune/resnet164.3/ --writerdir tensorboard/resnet164.3 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch resnet --depth 164 --lr 0.1 --model model_baseline/resnet164/init.pth.tar --resume model_prune/resnet164.6/pruned.pth.tar --save_dir finetune/resnet164.6/ --writerdir tensorboard/resnet164.6 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch resnet --depth 164 --lr 0.1 --model model_baseline/resnet164/init.pth.tar --resume model_prune/resnet164.7/pruned.pth.tar --save_dir finetune/resnet164.7/ --writerdir tensorboard/resnet164.7 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch resnet --depth 164 --lr 0.1 --model model_baseline/resnet164/init.pth.tar --resume model_prune/resnet164.8/pruned.pth.tar --save_dir finetune/resnet164.8/ --writerdir tensorboard/resnet164.8 --gpu-id 1 --train-batch 256 --test-batch 256
python train_ticket.py --dataset cifar100 --arch resnet --depth 164 --lr 0.1 --model model_baseline/resnet164/init.pth.tar --resume model_prune/resnet164.9/pruned.pth.tar --save_dir finetune/resnet164.9/ --writerdir tensorboard/resnet164.9 --gpu-id 1 --train-batch 256 --test-batch 256


# pruning
python LT_prune.py --dataset cifar100 --arch vgg19_bn --depth 19 --prune_ratio 0.7 --resume model_baseline/vgg19/model_best.pth.tar --save_dir model_prune/vgg0.7/ --gpu-id 1 --train-batch 256 --test-batch 256
python LT_prune.py --dataset cifar100 --arch vgg19_bn --depth 19 --prune_ratio 0.8 --resume model_baseline/vgg19/model_best.pth.tar --save_dir model_prune/vgg0.8/ --gpu-id 1 --train-batch 256 --test-batch 256
python LT_prune.py --dataset cifar100 --arch vgg19_bn --depth 19 --prune_ratio 0.9 --resume model_baseline/vgg19/model_best.pth.tar --save_dir model_prune/vgg0.9/ --gpu-id 1 --train-batch 256 --test-batch 256

python LT_prune.py --dataset cifar100 --arch resnet --depth 164 --prune_ratio 0.7 --resume model_baseline/resnet164/model_best.pth.tar --save_dir model_prune/resnet164.7/ --gpu-id 1 --train-batch 256 --test-batch 256
python LT_prune.py --dataset cifar100 --arch resnet --depth 164 --prune_ratio 0.8 --resume model_baseline/resnet164/model_best.pth.tar --save_dir model_prune/resnet164.8/ --gpu-id 1 --train-batch 256 --test-batch 256
python LT_prune.py --dataset cifar100 --arch resnet --depth 164 --prune_ratio 0.9 --resume model_baseline/resnet164/model_best.pth.tar --save_dir model_prune/resnet164.9/ --gpu-id 1 --train-batch 256 --test-batch 256
