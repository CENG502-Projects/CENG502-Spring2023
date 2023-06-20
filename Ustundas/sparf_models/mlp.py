import torch
import torch.nn.functional as F
import numpy as np


class MLP(torch.nn.Module):
	"""docstring for MLP"""
	def __init__(self, args):
		super(MLP, self).__init__()

		self.args = args

		self.progress = torch.nn.Parameter(torch.tensor(0.))
		self.L = 10

		self.fc1 = torch.nn.Linear(in_features=63, out_features=128)
		self.fc2 = torch.nn.Linear(in_features=128, out_features=128)
		self.fc3 = torch.nn.Linear(in_features=128, out_features=128)
		self.fc4 = torch.nn.Linear(in_features=128, out_features=128)

		self.fc5 = torch.nn.Linear(in_features=131, out_features=128) # add skip connection here
		self.fc6 = torch.nn.Linear(in_features=128, out_features=128)
		self.fc7 = torch.nn.Linear(in_features=128, out_features=128)
		self.fc8 = torch.nn.Linear(in_features=128, out_features=128)
		self.fc9 = torch.nn.Linear(in_features=63, out_features=128)
		self.fc10 = torch.nn.Linear(in_features=128, out_features=3)


	def forward(self, x, d):


		encoded_pos = self.c2f_positional_encoding(input=x, L=self.L)


		encoded_pos = torch.cat([x ,encoded_pos], dim=-1).to(torch.float32)

		out = F.relu(self.fc1(encoded_pos))
		out = F.relu(self.fc2(out))
		out = F.relu(self.fc3(out))
		out = F.relu(self.fc4(out))

		out = torch.cat([x, out], dim=-1).to(torch.float32)

		out = F.relu(self.fc5(out))
		out = F.relu(self.fc6(out))
		out = F.relu(self.fc7(out))
		out = self.fc8(out)

		density = out[..., 0]
		features = out[..., 1:]

		density = F.relu(density)

		encoded_dir = self.c2f_positional_encoding(input=d, L=self.L)
		encoded_dir = torch.cat([d, encoded_dir], dim=-1).to(torch.float32)

		out = F.relu(self.fc9(encoded_dir))

		rgb = torch.sigmoid(self.fc10(out))

		return rgb, density


	def positional_encoding(self, input, L):
		shape = input.shape

		freq = 2**torch.arange(L, dtype=torch.float32, device=self.args.device)*np.pi

		spectrum = input[...,None]*freq
		sin = spectrum.sin()
		cos = spectrum.cos()

		input_enc = torch.stack([sin, cos], dim=-2)
		input_enc = input_enc.view(*shape[:-1], -1)

		return input_enc


	def c2f_positional_encoding(self, input, L):

		input_enc = self.positional_encoding(input, L)

		start = 0.4
		end = 0.7

		alpha = (self.progress.data-start)/(end-start)*L
		k = torch.arange(L, device=self.args.device)
		weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2


		shape = input_enc.shape
		input_enc = (input_enc.view(-1,L)*weight).view(*shape).to(torch.float32)


		return input_enc