import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, img_size, hidden_size, label_num=None, label_dim=None):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.hidden_size = hidden_size

        if label_num is not None and label_dim is not None:
            self.label_embed = nn.Embedding(label_num, label_dim)
        else:
            self.label_embed = None
            label_dim = 0

        # LeakyReLU for discriminator
        self.model = nn.Sequential(
            nn.Linear(img_size + label_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, labels=None):
        """
        image: [batch_size, 1 * 28 * 28] = [batch_size, 784]
        labels: [batch_size, ]
        """
        if labels is not None and self.label_embed is not None:
            label_embedding = self.label_embed(labels)
            x = torch.cat([image, label_embedding], dim=1)  # [batch_size, img_size + label_dim]
        else:
            x = image

        return self.model(x)


class Generator(nn.Module):
    def __init__(self, latent_size, img_size, hidden_size, label_num=None, label_dim=None):
        super(Generator, self).__init__()

        self.latent_size = latent_size
        self.img_size = img_size
        self.hidden_size = hidden_size

        if label_num is not None and label_dim is not None:
            self.label_embed = nn.Embedding(label_num, label_dim)
        else:
            self.label_embed = None
            label_dim = 0
        
        # ReLU for generator
        self.model = nn.Sequential(
            nn.Linear(latent_size + label_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, img_size),
            nn.Tanh()
        )
    
    def forward(self, latent_variable, labels=None):
        """
        latent_variable: [batch_size, latent_size]
        labels: [batch_size, ]
        """
        if labels is not None and self.label_embed is not None:
            label_embedding = self.label_embed(labels)
            x = torch.cat([latent_variable, label_embedding], dim=1)  # [batch_size, latent_size + label_dim]
        else:
            x = latent_variable

        return self.model(x)