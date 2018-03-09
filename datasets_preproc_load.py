# CelebA image generation using DCGAN
import torch
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
from PIL import ImageFilter


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def test_celebA():
	# Parameters
	image_size = 64
	data_dir = './celebA'

	# CelebA dataset
	transform = transforms.Compose([transforms.Resize((image_size, image_size)),
									transforms.RandomHorizontalFlip(),
									transforms.RandomGrayscale(0.9999),
									transforms.ToTensor(),
		                            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])


	celebA_data = datasets.ImageFolder(data_dir, transform = transform)


	dataloader = DataLoader(celebA_data, 10)

	for i_batch, (sample_batched, _) in enumerate(dataloader):
		print(i_batch, sample_batched.size())



	# Visualizing
	sample = celebA_data.__getitem__(1)[0]
	sample_un = denorm(sample)
	sample_PIL = transforms.ToPILImage()(sample_un)
	sample_PIL.save('test.png')


def test_mnist():

	mnist = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([transforms.Resize((64, 64)), lambda x: x.filter(ImageFilter.GaussianBlur(radius=6)), transforms.ToTensor()]))
	dataloader = torch.utils.data.DataLoader(mnist, batch_size=10, shuffle=True)

	for i_batch, (sample_batched, _) in enumerate(dataloader):
		print(i_batch, sample_batched.size())


	# Visualizing
	sample = mnist.__getitem__(1)[0]
	sample_PIL = transforms.ToPILImage()(sample)
	sample_PIL.save('test.png')

if __name__ == '__main__':
	test_mnist()
