python main.py --refine pruned_models/resnet/0.2/pruned.pth.tar --dataset cifar10 --arch resnet --depth 164 --epochs 100 --save fine-tune/resnet/0.2/
python main.py --refine pruned_models/resnet/0.4/pruned.pth.tar --dataset cifar10 --arch resnet --depth 164 --epochs 100 --save fine-tune/resnet/0.4/
python main.py --refine pruned_models/resnet/0.6/pruned.pth.tar --dataset cifar10 --arch resnet --depth 164 --epochs 100 --save fine-tune/resnet/0.6/