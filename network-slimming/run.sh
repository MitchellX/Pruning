python main.py --dataset cifar10 --arch vgg --depth 19 --save ./logs_vgg_baseline
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 19 --save ./logs_vgg_sparsity

python main.py --dataset cifar10 --arch resnet --depth 164 --save ./logs_resnet_baseline
python main.py -sr --s 0.00001 --dataset cifar10 --arch resnet --depth 164 --save ./logs_resnet_sparsity

python main.py --dataset cifar10 --arch densenet --depth 40 --save ./logs_densenet_baseline
python main.py -sr --s 0.00001 --dataset cifar10 --arch densenet --depth 40 --save ./logs_densenet_sparsity
