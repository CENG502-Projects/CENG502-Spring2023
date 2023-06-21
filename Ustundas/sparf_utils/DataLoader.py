import torch
import os
import numpy as np
from PIL import Image
import cv2

class DTUDataset(torch.utils.data.Dataset):
	"""docstring for DTUDataset"""
	def __init__(self, scene, mode, transform=None):
		super(DTUDataset, self).__init__()
		self.scene = scene
		self.mode = mode
		self.transform = transform

		self.set_images = {}
		self.intrinsic_arr = {}
		self.pose_arr = {}

		self.image_idx = []

		scene_path = "data/DTU/{}".format(self.scene)

		train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
		excluded = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]

		if self.mode == "train":
			self.image_idx = train_idx

		else:
			self.image_idx = []

			for i in range(49):
				if i not in train_idx or i not in excluded:
					self.image_idx.append(i)




#		all_images = os.listdir("{}/image/".format(scene_path))


		sorted(self.image_idx)

		for idx in self.image_idx:
			image_name = "{}.png".format(idx).rjust(10, "0")
			full_image_name = "{}/image/{}".format(scene_path, image_name)
			self.set_images[idx] = (full_image_name)


		camera_info = np.load("{}/cameras.npz".format(scene_path))

		bottom = np.array([0., 0., 0., 1.]).reshape(1,4)
		right = np.array([0., 0., 0.]).reshape(3, 1)

		# Adapted from PixelNeRF
		for idx in self.image_idx:
			P = camera_info["world_mat_{}".format(idx)]
			P = P[:3]
			K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
			K /= K[2, 2]
			t = (t[:3] / t[3])[:, 0].reshape(3,1)


			pose = np.concatenate([np.concatenate([R.T, t], 1), bottom], 0)

#			intrinsic = np.concatenate([np.concatenate([K, right], 1), bottom], 0)



			self.pose_arr[idx] = pose
			self.intrinsic_arr[idx] = K

	def __len__(self):
		return len(self.set_images)


	def __getitem__(self, idx):


		image_file = self.set_images[idx]



		image = Image.open(image_file)

		image = self.transform(image)

		return (image, torch.tensor(self.pose_arr[idx]),torch.tensor(self.intrinsic_arr[idx])) 