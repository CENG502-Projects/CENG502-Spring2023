import torch
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from tqdm import tqdm
import numpy as np

from sparf_models.NeRF import NeRF
from losses import loss_photometric, loss_mvc
from sparf_utils.DataLoader import DTUDataset
import matplotlib.pyplot as plt
#from sparf_utils.PDCNETWrapper import test


def train(args):



	transformList = transforms.Compose([
		transforms.ToTensor(),
	])
	transform_to_image = transforms.ToPILImage()


	dataset = DTUDataset(scene=args.scene, mode="train", transform=transformList)


	arr = {}

	nerf = NeRF(args).to(args.device)


	optimizer = torch.optim.Adam(nerf.mlp.parameters(), lr=5.e-4)

	lambda_c = 10**(-3)
	lambda_d = 10**(-3)



#	lr_scheduler = ExponentialLR(optimizer, )

	losses = []

	for cur_iter in tqdm(range(args.max_iter)):
		optimizer.zero_grad()

		mvc_image_ids = np.random.choice(dataset.image_idx, 2)

		image_idx_i = mvc_image_ids[0]
		image_idx_j = mvc_image_ids[1]

#		print("Chosen image ids for MVC loss: {} {}".format(image_idx_i, image_idx_j))
		image_i, pose_i, intrinsic_i = dataset[image_idx_i]
		image_j, pose_j, intrinsic_j = dataset[image_idx_j]

		image_i = image_i.to(args.device)
		pose_i = pose_i.to(args.device)
		intrinsic_i = intrinsic_i.to(args.device)


		image_j = image_j.to(args.device)
		pose_j = pose_j.to(args.device)
		intrinsic_j = intrinsic_j.to(args.device)


		random_pixels = []

		all_x = []
		all_d = []
		all_pixel_values = []
		for idx in dataset.image_idx:

			image, pose, intrinsic = dataset[idx]

			image = image.to(args.device)
			pose = pose.to(args.device)
			intrinsic = intrinsic.to(args.device)

			x, d, (coords) = nerf.get_rays_from_pixels(image.shape, pose, intrinsic)

			pixel_values = [image[:, x_coor, y_coor] for (x_coor,y_coor) in coords]
	
			pixel_values = torch.stack(pixel_values)

			all_x.append(x)
			all_d.append(d)
			all_pixel_values.append(pixel_values)


		all_x = torch.stack(all_x, dim=0)
		all_d = torch.stack(all_d, dim=0)
		all_pixel_values = torch.stack(all_pixel_values, dim=0)

		I, z = nerf(all_x, all_d)






		loss_p = loss_photometric(source=all_pixel_values, out=I)
#		loss_mvc
#		loss_depth
		
		#loss = lambda_c*loss_mvc + lambda_d*loss_depth + loss_p


		loss_p.backward()
		losses.append(loss_p.item())
		optimizer.step()

		if (cur_iter != 0) and (cur_iter % 5000) == 0:
			lambda_c /= 2.0
			lambda_d /= 2.0
			torch.save(nerf.state_dict(), "{}/iter_{}_model.pt".format("SavedModels", cur_iter))
			np.save("losses/loss_{}.npy".format(cur_iter), losses)
			plt.plot(losses)
			plt.xlabel("Iteration")
			plt.ylabel("Loss")
			plt.title("Loss vs. Iteration graph")
			plt.savefig("figures/loss_{}.png".format(cur_iter))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("scene", type=str, help="Scene name (e.g scan34)")
	parser.add_argument("--device", type=str, help="Which device to run")
	parser.add_argument("--max_iter", type=int, default=200000, help="Number of iterations for training")
	parser.add_argument("--random_pixel_count", type=int, default=1024, help="Number of random pixels to get from each training image")
	parser.add_argument("--sample_count", type=int, default=256, help="Number of samples along a ray")
	args = parser.parse_args()


	if args.device is None:
		args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("No device is selected! Switching to {}".format(args.device))


	train(args)