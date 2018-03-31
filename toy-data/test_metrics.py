import numpy as np
import random
import scipy.linalg as sla

def calculate_dist(x_, y_):

	dist_matrix = np.zeros([x_.shape[0], y_.shape[0]])

	for i in range(x_.shape[0]):
		for j in range(y_.shape[0]):

			dist_matrix[i, j] = np.sqrt((x_[i, 0] - y_[j, 0])**2 + (x_[i, 1] - y_[j, 1])**2)

	return dist_matrix

slack = 3.0
cov = [0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2), 0.02*np.eye(2)]

scale = 2.
centers = [
(1, 0),
(-1, 0),
(0, 1),
(0, -1),
(1. / np.sqrt(2), 1. / np.sqrt(2)),
(1. / np.sqrt(2), -1. / np.sqrt(2)),
(-1. / np.sqrt(2), 1. / np.sqrt(2)),
(-1. / np.sqrt(2), -1. / np.sqrt(2))
]

centers = [(scale * x, scale * y) for x, y in centers]
centers = np.asarray(centers)

x = []

for i in range(3):

	sample = np.random.randn(2) * .02
	center = random.choice(centers)
	sample[0] += center[0]
	sample[1] += center[1]

	sample /= 1.414
	x.append(sample)

x = np.asarray(x)
	
distances = calculate_dist(1.414*x, centers)

closest_center = np.argmin(distances, 1)

n_gaussians = centers.shape[0]

fd = 0
quality_samples = 0
quality_modes = 0

for cent in range(n_gaussians):
	
	center_samples = x[np.where(closest_center == cent)]

	center_distances = distances[np.where(closest_center == cent)]

	sigma = cov[cent][0, 0]

	quality_samples_center = np.sum(center_distances[:, cent] <= slack*sigma)
	quality_samples += quality_samples_center

	if (quality_samples_center > 0):
		quality_modes += 1

	if (center_samples.shape[0] > 2):

		m = np.mean(center_samples, 0)
		C = np.cov(center_samples, rowvar = False)

		fd += ((centers[cent] - m)**2).sum() + np.matrix.trace(C + cov[cent] - 2*sla.sqrtm( np.matmul(C, cov[cent])))


fd_all = fd / len(np.unique(closest_center))

