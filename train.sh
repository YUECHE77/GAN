#!/bin/bash

python train.py \
    --batch-size 32 \
    --epochs 100 \
    --latent-size 64 \
    --hidden-size 512 \
    --image-size 784 \
    --conditional \
    --label-dim 16 \
    --dataset FashionMNIST \
    --img-folder /root/autodl-tmp/MNIST/ \
    --csv-path /root/autodl-tmp/FashionMNIST/fashion-mnist_test.csv \
    --sample-dir /root/autodl-tmp/GANs/ConditionalGan/FashionMNIST/images \
    --save-dir /root/autodl-tmp/GANs/ConditionalGan/FashionMNIST/model