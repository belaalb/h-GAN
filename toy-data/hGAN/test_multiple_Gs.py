from __future__ import print_function
import argparse
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torchvision
from PIL import ImageFilter
import matplotlib.pyplot as plt
import model as model_
import numpy as np
import os
from scipy.stats import chi2
import glob
import scipy.linalg as sla


from ToyData import ToyData

def plot_ellipse(semimaj=1, semimin=1, phi=0, x_cent=0, y_cent=0, theta_num=1e3, ax=None, plot_kwargs=None, cov=None, mass_level=0.01):
	# Get Ellipse Properties from cov matrix
	eig_vec, eig_val, u = np.linalg.svd(cov)
	# Make sure 0th eigenvector has positive x-coordinate
	if eig_vec[0][0] < 0:
		eig_vec[0] *= -1
	semimaj = np.sqrt(eig_val[0])
	semimin = np.sqrt(eig_val[1])
	distances = np.linspace(0,20,20001)
	chi2_cdf = chi2.cdf(distances,df=2)
	multiplier = np.sqrt(distances[np.where(np.abs(chi2_cdf-mass_level)==np.abs(chi2_cdf-mass_level).min())[0][0]])
	semimaj *= multiplier
	semimin *= multiplier
	phi = np.arccos(np.dot(eig_vec[0],np.array([1,0])))
	if eig_vec[0][1] < 0 and phi > 0:
		phi *= -1

	# Generate data for ellipse structure
	theta = np.linspace(0, 2*np.pi, theta_num)
	r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
	x = r*np.cos(theta)
	y = r*np.sin(theta)
	data = np.array([x,y])
	S = np.array([[semimaj, 0], [0, semimin]])
	R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
	T = np.dot(R,S)
	data = np.dot(T, data)
	data[0] += x_cent
	data[1] += y_cent

	return data

def plot_(x, centers, toy_dataset):
	
	if (toy_dataset == '8gaussians'):
		cov_all = np.array([(0.02, 0), (0, 0.02)])
		scale = 1.414

	elif (toy_dataset == '25gaussians'):
		cov_all = np.array([(0.05, 0), (0, 0.05)])
		scale = 2.828
		
	samples = scale*x

	plt.scatter(samples[:, 0], samples[:, 1], c = 'red', marker = 'o', alpha = 0.1)
	plt.scatter(centers[:, 0], centers[:, 1], c = 'black', marker = 'x', alpha = 1)

	for k in range(centers.shape[0]):
		ellipse_data = plot_ellipse(x_cent = centers[k, 0], y_cent = centers[k, 1], cov = cov_all, mass_level = 0.9545)
		plt.plot(ellipse_data[0], ellipse_data[1], c = 'black', alpha = 0.2)

	plt.show()


def calculate_dist(x_, y_):

	dist_matrix = np.zeros([x_.shape[0], y_.shape[0]])

	for i in range(x_.shape[0]):
		for j in range(y_.shape[0]):

			dist_matrix[i, j] = np.sqrt((x_[i, 0] - y_[j, 0])**2 + (x_[i, 1] - y_[j, 1])**2)

	return dist_matrix

def metrics(x, centers, cov, toy_dataset, slack = 2.0):

	if (toy_dataset == '8gaussians'):
		distances = calculate_dist(1.414*x, centers)
	
	elif (toy_dataset == '25gaussians'):
		distances = calculate_dist(2.828*x, centers)
		
	closest_center = np.argmin(distances, 1)

	n_gaussians = centers.shape[0]

	fd = 0
	quality_samples = 0
	quality_modes = 0

	for cent in range(n_gaussians):

		center_samples = x[np.where(closest_center == cent)]

		center_distances = distances[np.where(closest_center == cent)]

		sigma = cov[0, 0]

		quality_samples_center = np.sum(center_distances[:, cent] <= slack*np.sqrt(sigma))
		quality_samples += quality_samples_center

		if (quality_samples_center > 0):
			quality_modes += 1

		if (center_samples.shape[0] > 3):

			m = np.mean(center_samples, 0)
			C = np.cov(center_samples, rowvar = False)

			fd += ((centers[cent] - m)**2).sum() + np.matrix.trace(C + cov - 2*sla.sqrtm( np.matmul(C, cov)))


	fd_all = fd / len(np.unique(closest_center))

	return fd_all, quality_samples, quality_modes


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-folder', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--toy-length', type=int, default=2500, metavar='N', help='number of samples to  (default: 10000)')
	parser.add_argument('--toy-dataset', choices=['8gaussians', '25gaussians'], default='8gaussians')
	args = parser.parse_args()

	if args.cp_folder is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	generator = model_.Generator_toy(512)

	files_list = glob.glob(args.cp_folder+'checkpoint_*.pt')
	files_list.sort()

	fd = []
	covered_centers = []
	quality_samples = []

	toy_data = ToyData(args.toy_dataset, args.toy_length)
	centers = toy_data.get_centers()
	cov = toy_data.get_cov()

	for file_id in files_list:
		ckpt = torch.load(file_id, map_location = lambda storage, loc: storage)
		generator.load_state_dict(ckpt['model_state'])

		fixed_noise = Variable(ckpt['fixed_noise'])

		generator.eval()

		x = generator.forward(fixed_noise)

		fd_, q_samples, cov_modes = metrics(x.data.numpy(), centers, cov, args.toy_dataset)

		print('Epoch:', file_id.split('checkpoint_')[-1].split('ep')[0])
		print('FD:', fd_)
		print('High quality samples:', q_samples)
		print('Covered modes:', cov_modes)


		fd.append(fd_)
		quality_samples.append(q_samples)
		covered_centers.append(cov_modes)

	## do something with lists

	plt.plot(fd)
	plt.plot(quality_samples)
	plt.plot(covered_centers)
