import torch
import numpy as np
from sparf_models.mlp import MLP

class NeRF(torch.nn.Module):
	"""docstring for NeRF"""
	def __init__(self, args):
		super(NeRF, self).__init__()

		self.args = args
		self.mlp = MLP(args)


	def forward(self, x, d):




		t_values = self.sample_t_values()

		large_x = x.repeat_interleave(self.args.sample_count, dim=1)
		large_d = d.repeat_interleave(self.args.sample_count, dim=1)

		large_t = t_values.repeat(x.shape[1])

		large_t = torch.stack([large_t for _ in range(x.shape[0])]).unsqueeze(2)

		large_x = large_x + torch.mul(large_d, large_t)


		rgb, density = self.mlp(large_x, large_d)


		rgb = torch.reshape(rgb, (x.shape[0], x.shape[1], t_values.shape[0], 3))
		density = torch.reshape(density, (x.shape[0], x.shape[1], t_values.shape[0]))



		diff = torch.zeros(self.args.sample_count, device=self.args.device)


		for i in range(self.args.sample_count-1):
			diff[i] = t_values[i+1] - t_values[i]



		I = self.volume_rendering(rgb, density, t_values, diff)
		z = None


		return I, z


	def volume_rendering(self, rgb, density, t_values, diff):
		

		I = torch.zeros(rgb.shape[0], rgb.shape[1], 3, device=self.args.device)

		for m in range(1, self.args.sample_count):

			a_m = self.alpha(density, diff, m)


			I = I + torch.mul(rgb[:, :, m], a_m.unsqueeze(2))


			return I



	def alpha(self, density, diff, m):


		d = torch.mul(density[:, :, :m], diff[:m])

		T_m = torch.exp(-torch.sum(d)).squeeze()


		l = (1-torch.exp(-torch.mul(density[:, :, m], diff[m])))

		a_m = torch.mul(T_m, l)

		return a_m


	def approximate_depth(self):
		pass



	def sample_t_values(self):

		t_values = []

		N = self.args.sample_count

		t_n = 0.0
		t_f = 1.0

		for i in range(1, N+1):


			lower = t_n + (i-1)/N * (t_f - t_n)
			higher = t_n + i/N * (t_f - t_n)


			t = np.random.uniform(lower, higher)

			t_values.append(t)

		return torch.tensor(t_values, device=self.args.device)





	def get_rays_from_pixels(self, image_shape, pose, intrinsic):

		x_choices = np.random.randint(low=0, high=image_shape[1], size=self.args.random_pixel_count)
		y_choices = np.random.randint(low=0, high=image_shape[2], size=self.args.random_pixel_count)			

		coords = [[x,y] for x, y in zip(x_choices, y_choices)]


		pixels = torch.tensor(coords, dtype=torch.float64, device=self.args.device)

		homogen_pixels = torch.cat([pixels, torch.ones_like(pixels[...,:1])], dim=-1).to(self.args.device)
		homogen_camera_coords = homogen_pixels @ intrinsic.inverse()#.transpose(-1, -2)

		center_3d = torch.zeros_like(homogen_camera_coords)

		pose_inv = self.get_inverse_of_pose(pose)

		homogen_camera_coords = torch.cat([homogen_camera_coords,torch.ones_like(homogen_camera_coords[...,:1])],dim=-1)
		center_3d_homogen = torch.cat([center_3d,torch.ones_like(center_3d[...,:1])],dim=-1)

		homogen_camera_coords = homogen_camera_coords @ pose_inv
		center_3d_homogen = center_3d_homogen @ pose_inv
		ray = homogen_camera_coords-center_3d_homogen

		return center_3d_homogen[..., :-1], ray[..., :-1], (coords)


	def get_rays(self, image_shape, pose, intrinsic):
		# Adapted from BARF Paper
		C, H, W = image_shape
		i, j = torch.meshgrid(torch.arange(W, dtype=torch.float64), torch.arange(H, dtype=torch.float64), indexing="xy")
		i = i.t().to(self.args.device)
		j = j.t().to(self.args.device)


		grid = torch.stack([i, j], dim=-1).view(-1, 2)
#		grid = grid.repeat(batch_size, 1, 1)

		grid_homogen = torch.cat([grid,torch.ones_like(grid[...,:1])],dim=-1)

		grid_3d = grid_homogen @ intrinsic.inverse().transpose(-1, -2)
		center_3d = torch.zeros_like(grid_3d)



		pose_inv = self.get_inverse_of_pose(pose)


		grid_3d_homogen = torch.cat([grid_3d,torch.ones_like(grid_3d[...,:1])],dim=-1)
		center_3d_homogen = torch.cat([center_3d,torch.ones_like(center_3d[...,:1])],dim=-1)

		grid_3d_homogen = grid_3d_homogen @ pose_inv
		center_3d_homogen = center_3d_homogen @ pose_inv
		ray = grid_3d_homogen-center_3d_homogen

		return center_3d_homogen, ray


	def get_inverse_of_pose(self, pose):
		R = pose[:3, :-1]
		t = pose[:-1, -1]


		R_inv = R.transpose(-1, -2)
		t_inv = -R_inv @ t
		t_inv = t_inv.reshape(3, 1)

		bottom = torch.tensor([0., 0., 0., 1.], dtype=pose.dtype, device=pose.device).reshape(1, 4)


		pose_inv = torch.cat([torch.cat([R_inv, t_inv], dim=1), bottom], dim=0)


		return pose_inv