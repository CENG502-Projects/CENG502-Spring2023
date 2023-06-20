import torch
import numpy as np



lambda_c = 10**(-3)
lambda_d = 10**(-3)


def halve_weights():
	lambda_c /= 2.0
	lambda_d /= 2.0


def loss_photometric(source, out):

	return torch.mean(torch.sum(torch.square(source - out)))


def loss_mvc(img_i, img_j, intr_i, intr_j, pose_i, pose_j):
	return 0


def loss_depth():
	return 0


def loss_func(source, target, depth):
	return loss_photometric(source, target) + loss_mvc()*lambda_c + loss_depth()*lambda_d