import torch
import random
import numpy as np
import torch.optim as optim

from utils.utils_data import get_mnist_training_dataloader
from models import Generator, Discriminator
from utils.utils_training import Trainer


# Fix the random seed
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Training settings
dataset_path = "./data/"
batch_size = 64
image_size = 64
offset = 0.2
latent_dim = 100
generator_hidden_dim = 64
discriminator_hidden_dim = 64
lr = 1e-4
betas = (0.5, 0.999)
num_epochs = 100

# Obtain the training dataloader
train_dataloader = get_mnist_training_dataloader(dataset_path = dataset_path, batch_size = batch_size, image_size = image_size, offset = offset)

# Obtain the generator and discriminator
generator = Generator(latent_dim = latent_dim, dim = generator_hidden_dim)
discriminator = Discriminator(dim = discriminator_hidden_dim)

# Initialize optimizers
G_optimizer = optim.Adam(generator.parameters(), lr = lr, betas = betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr = lr, betas = betas)

# Train the model
trainer = Trainer(generator, discriminator, G_optimizer, D_optimizer, use_cuda = torch.cuda.is_available())
trainer.train(train_dataloader, num_epochs, save_training_gif = True)

# Save models
name = 'mnist_model'
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
