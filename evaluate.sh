#!/bin/bash

python evaluate.py \
    --ckpt-path /root/autodl-tmp/GANs/ConditionalGan/MNIST/model/G--100.ckpt \
    --batch-size 4 \
    --conditional \
    --label 7 \
    --latent-size 64 \
    --hidden-size 512 \
    --image-size 784 \
    --label-dim 16 \
    --output-dir /root/GAN/outputs