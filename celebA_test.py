# CelebA image generation using DCGAN
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np

from torch.utils.data import DataLoader



def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Parameters
image_size = 64
data_dir = './celebA'

# CelebA dataset
transform = transforms.Compose([transforms.Resize((image_size, image_size)),
								transforms.RandomHorizontalFlip(),
								transforms.RandomGrayscale(0.9999),
								transforms.ToTensor(),
                                transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])


celebA_data = dsets.ImageFolder(data_dir, transform = transform)


dataloader = DataLoader(celebA_data, 10)

for i_batch, (sample_batched, _) in enumerate(dataloader):
	print(i_batch, sample_batched.size())



# Visualizing
sample = celebA_data.__getitem__(1)[0]
sample_un = denorm(sample)
sample_PIL = transforms.ToPILImage()(sample_un)
sample_PIL.save('test.png')

