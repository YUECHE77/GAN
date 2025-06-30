import os
import argparse

import torch
from torchvision.utils import save_image

from model.gan import Generator

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GAN")
    parser.add_argument("--ckpt-path", type=str, help="The Generator model path")
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--conditional", action="store_true", help='set to True if use Conditional Gan')
    parser.add_argument("--label", type=int, default=0, help='Which category do you want to generate')

    parser.add_argument("--latent-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=784, help='Default is 784 = 1 * 28 * 28')
    parser.add_argument("--label-dim", type=int, default=16)

    parser.add_argument("--output-dir", type=str, default="/root/GAN/outputs", help="Path to save the image")
    args = parser.parse_known_args()[0]

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load the model
    label_num = 10  # For both MNIST and FashionMNIST, num_classes = 10
    label_dim = args.label_dim if args.conditional and args.label_dim > 0 else None
    Gan_G = Generator(args.latent_size, args.image_size, args.hidden_size, label_num=label_num, label_dim=label_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    Gan_G.to(device)

    Gan_G.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    Gan_G.eval()

    # 2. Generate the image
    z = torch.randn(args.batch_size, args.latent_size).to(device)  # latent varibles: [batch_size, latent_size]
    if args.conditional:
        labels = torch.tensor([args.label] * args.batch_size, dtype=torch.long, device=device)
    else:
        labels = None
    
    with torch.no_grad():
        fake_images = Gan_G(z, labels)  # [batch_size, 784]
    
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), os.path.join(args.output_dir, "result.png"))
    print(f"Generated images saved at: {os.path.join(args.output_dir, 'result.png')}")
