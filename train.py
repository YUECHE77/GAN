import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pylab

import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.gan import Discriminator, Generator
from dataset.mnist import MNIST, FashionMNIST

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=784, help='Default is 784 = 1 * 28 * 28')
    parser.add_argument("--conditional", action="store_true", help='set to True if use Conditional Gan')
    parser.add_argument("--label-dim", type=int, default=16)

    parser.add_argument("--dataset", type=str, default='MNIST', help='MNIST or FashionMNIST')
    parser.add_argument("--img-folder", type=str, default='/root/autodl-tmp/MNIST/', help='Path to MNIST folder')
    parser.add_argument("--csv-path", type=str, default='/root/autodl-tmp/FashionMNIST/fashion-mnist_test.csv', help='Path to FashionMNIST csv')

    parser.add_argument("--sample-dir", type=str, default='/root/autodl-tmp/GANs/RegularGan/MNIST/images')
    parser.add_argument("--save-dir", type=str, default='/root/autodl-tmp/GANs/RegularGan/MNIST/model')
    args = parser.parse_known_args()[0]

    os.makedirs(args.sample_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert args.dataset.lower() in ['mnist', 'fashionmnist']
    # you can also define your own data augmentation. I'm using default here.
    if args.dataset.lower() == 'mnist':
        dataloader = MNIST(img_folder=args.img_folder, batch_size=args.batch_size)
    else:
        dataloader = FashionMNIST(csv_path=args.csv_path, batch_size=args.batch_size)

    label_num = 10  # For both MNIST and FashionMNIST, num_classes = 10
    label_dim = args.label_dim if args.conditional and args.label_dim > 0 else None
    Gan_D = Discriminator(args.image_size, args.hidden_size, label_num=label_num, label_dim=label_dim)
    Gan_G = Generator(args.latent_size, args.image_size, args.hidden_size, label_num=label_num, label_dim=label_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    Gan_D.to(device)
    Gan_G.to(device)

    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()  # if using BCEWithLogitsLoss -> Sigmoid is not needed in the model
    d_optimizer = torch.optim.Adam(Gan_D.parameters(), lr=2e-4)
    g_optimizer = torch.optim.Adam(Gan_G.parameters(), lr=2e-4)

    # Statistics to be saved
    d_losses = np.zeros(args.epochs)
    g_losses = np.zeros(args.epochs)
    real_scores = np.zeros(args.epochs)
    fake_scores = np.zeros(args.epochs)

    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.view(-1, args.image_size).to(device)  # [batch_size, 1 * 28 * 28] = [batch_size, 784]
            labels = labels.to(device)
            len_in_batch = images.shape[0]

            # Create the labels which are later used as input for the BCE loss
            real_labels = torch.ones(len_in_batch, 1).to(device)
            fake_labels = torch.zeros(len_in_batch, 1).to(device)

            # ------------- Train the Discriminator -------------
            # 1. Compute loss using real images
            outputs = Gan_D(images, labels)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs

            # 2. Compute loss using fake images
            z = torch.randn(len_in_batch, args.latent_size).to(device)  # latent varibles: [batch_size, latent_size]
            g_fake_labels = torch.randint(0, label_num, (len_in_batch, ), dtype=torch.long, device=device)
            fake_images = Gan_G(z, g_fake_labels)  # generate the fake images

            outputs = Gan_D(fake_images, g_fake_labels)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # 3. Update
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # ------------- Train the Generator -------------
            z = torch.randn(len_in_batch, args.latent_size).to(device)
            g_fake_labels = torch.randint(0, label_num, (len_in_batch, ), dtype=torch.long, device=device)
            fake_images = Gan_G(z, g_fake_labels)
            outputs = Gan_D(fake_images, g_fake_labels)

            # All the magic happens here: Use the real labels to fool the Discriminator
            g_loss = criterion(outputs, real_labels)

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # ------------- Update Statistics -------------
            d_losses[epoch] = d_losses[epoch]*(i/(i+1.)) + d_loss*(1./(i+1.))
            g_losses[epoch] = g_losses[epoch]*(i/(i+1.)) + g_loss*(1./(i+1.))
            real_scores[epoch] = real_scores[epoch]*(i/(i+1.)) + real_score.mean()*(1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch]*(i/(i+1.)) + fake_score.mean()*(1./(i+1.))

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}' 
                    .format(epoch, args.epochs, i+1, len(dataloader), d_loss, g_loss, 
                            real_score.mean(), fake_score.mean()))
        
        # Save real images
        if (epoch+1) == 1:
            images = images.view(images.size(0), 1, 28, 28)
            save_image(denorm(images.data), os.path.join(args.sample_dir, 'real_images.png'))
        
        # Save sampled images
        fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images.data), os.path.join(args.sample_dir, 'fake_images-{}.png'.format(epoch+1)))

        # Save and plot Statistics
        np.save(os.path.join(args.save_dir, 'd_losses.npy'), d_losses)
        np.save(os.path.join(args.save_dir, 'g_losses.npy'), g_losses)
        np.save(os.path.join(args.save_dir, 'fake_scores.npy'), fake_scores)
        np.save(os.path.join(args.save_dir, 'real_scores.npy'), real_scores)

        plt.figure()
        pylab.xlim(0, args.epochs + 1)
        plt.plot(range(1, args.epochs + 1), d_losses, label='d loss')
        plt.plot(range(1, args.epochs + 1), g_losses, label='g loss')    
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'loss.pdf'))
        plt.close()

        plt.figure()
        pylab.xlim(0, args.epochs + 1)
        pylab.ylim(0, 1)
        plt.plot(range(1, args.epochs + 1), fake_scores, label='fake score')
        plt.plot(range(1, args.epochs + 1), real_scores, label='real score')    
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'accuracy.pdf'))
        plt.close()

        # Save model at checkpoints
        if (epoch+1) % 50 == 0:
            torch.save(Gan_G.state_dict(), os.path.join(args.save_dir, 'G--{}.ckpt'.format(epoch+1)))
            torch.save(Gan_D.state_dict(), os.path.join(args.save_dir, 'D--{}.ckpt'.format(epoch+1)))
