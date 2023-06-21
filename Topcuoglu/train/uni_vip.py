# Author: Utku Mert Topçuoğlu + https://github.com/lucidrains/byol-pytorch/blob/master/byol_pytorch/byol_pytorch.py
import copy
from functools import wraps
import torch
from torch import nn
import torch.nn.functional as F
from math import cos,pi

from UniVIP.train.dataloader_uvip import IMAGE_SIZE
from transforms import common_augmentations, K_COMMON_INSTANCES
from sink_knop_dist import SinkhornDistance

# helper functions

def flatten(t):
    return t.reshape(t.shape[0], -1)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


# exponential moving average

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.base_beta = beta
        self.end_momentum = 1.0
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
    def update_beta(self, tot_iter, cur_iter):
        """Update momentum with cosine, added by UTKU"""
        self.beta = self.end_momentum - (self.end_momentum - self.base_beta) * (cos(pi * cur_iter / float(tot_iter)) + 1) / 2


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# MLP class for projector and predictor

def MLP(dim, projection_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets

class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        assert not torch.stack([torch.isnan(p).any() for p in self.net.parameters()]).any(), "self.net has nan"
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        self.hidden.clear()
        _ = self.net(x)
        hidden = self.hidden[x.device]
        self.hidden.clear()
        
        assert not hidden.isnan().any(), "encoders produce nan values"
        # TODO to debug this load the final 52th epoch weights and debug from there
        # TODO might use torch.nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None)
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    def forward(self, x, return_projection = True):
        representation = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, representation

# main class

class UVIP(nn.Module):
    def __init__(
        self,
        net,
        image_size = IMAGE_SIZE, # default byol
        hidden_layer = -2,
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99,
        K_common_instances=K_COMMON_INSTANCES, # Number of instances in the ovrelap.
    ):
        super().__init__()
        self.net = net
        self.image_size = image_size

        self.online_encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size)

        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)

        # UniVIP specific layers
        self.K_common_instances = K_common_instances
        self.instances_to_scene_linear = nn.Linear(projection_size*K_common_instances, projection_size) # K concatenated
        self.sinkhorn_distance = SinkhornDistance()

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # # send a mock image tensor to instantiate singleton parameters # NOTE batch size 1 gives error.
        # dummy_data = (*torch.rand(size=(2, 2, 3, image_size, image_size), device=device),torch.rand(size=(2, 4, 3, image_size, image_size), device=device))
        # self.forward(dummy_data)
    

    @singleton('target_encoder')
    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self, tot_iter, cur_iter):
        self.target_ema_updater.update_beta(tot_iter=tot_iter, cur_iter=cur_iter)
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
    
    def byol_loss_fn(self, x, y):
        """Cosine similarity is (proportional (/2)) to MSE when x-y are l2 normalized
        https://stats.stackexchange.com/questions/146221/is-cosine-similarity-identical-to-l2-normalized-euclidean-distance"""
        # L2 normalization (Divided L2 norm), hence, resulting l2_norm = 1 --> MSE = cosine_sim 
        x = F.normalize(x, dim=-1, p=2) # 
        y = F.normalize(y, dim=-1, p=2)
        return (2 - 2 * (x * y).sum(dim=-1)).mean()

    def ii_loss_fn(self, online_pred_instance, target_proj_instance, online_pred_avg, target_proj_avg):
        """
        Calculate the instance to insatnce loss based on sinkhorn knopp iterations and optimal transportation described in the paper.
        matmul behavior: https://pytorch.org/docs/stable/generated/torch.matmul.html#:~:text=%3E%3E%3E%20tensor1%20%3D%20torch.randn(10%2C%203%2C%204)%0A%3E%3E%3E%20tensor2%20%3D%20torch.randn(10%2C%204%2C%205)%0A%3E%3E%3E%20torch.matmul(tensor1%2C%20tensor2).size()%0Atorch.Size(%5B10%2C%203%2C%205%5D) 
        """
        # Step 1: Compute dot_product_matrix
        O_matrix = online_pred_instance  # (batch_size, instance numbers K, number of features)
        T_matrix = target_proj_instance  # (batch_size, instance numbers K, number of features)
        dot_product_matrix = torch.matmul(O_matrix, T_matrix.transpose(1, 2))  # (batch_size, instance numbers K, instance numbers K)
        # Step 2: Compute Norm matrices
        norm_vector_O = torch.norm(O_matrix, dim=-1, keepdim=True) # normalize over features
        norm_vector_T = torch.norm(T_matrix, dim=-1, keepdim=True) # normalize over features
        # Step 3: Compute C
        ot_cosine_similarity_matrix = (dot_product_matrix / torch.matmul(norm_vector_O, norm_vector_T.transpose(1, 2)))# (batch_size, instance numbers K, instance numbers K)
        # TODO 0 division might cause NaN
        cost_matrix = 1 - ot_cosine_similarity_matrix# (batch_size, instance numbers K, instance numbers K)
        assert not ot_cosine_similarity_matrix.isnan().any(), "ot_cosine_similarity_matrix has NaN"
        # demander a, supplier b
        a_vector = torch.nn.functional.relu(torch.matmul(T_matrix, online_pred_avg.unsqueeze(dim=-1))) # (batch_size, instance numbers K, 1)
        b_vector = torch.nn.functional.relu(torch.matmul(O_matrix, target_proj_avg.unsqueeze(dim=-1))) # (batch_size, instance numbers K, 1)
        _, optimal_plan_matrix = self.sinkhorn_distance(mu=a_vector,nu=b_vector,C=cost_matrix) # (batch_size, instance numbers K, instance numbers K)
        # torch.mul is element-wise multiplication, then sum all elements of the cost matrix, then results per batch are averaged with mean
        loss_ii = torch.sum(-torch.mul(optimal_plan_matrix,ot_cosine_similarity_matrix), dim=(-2,-1)).mean() # Forces similar instance representations to be close to each other.
        assert not loss_ii.isnan().any(), "loss_ii has NaN"
        return loss_ii # Averaged out of convenience

    def forward(
        self,
        img_data,
        return_embedding = False,
        return_projection = True
    ):
        scene_one, scene_two, concatenated_instances = img_data
        assert not (self.training and scene_one.shape[0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(scene_one, return_projection = return_projection)
        
        # FEED THE SCENES AND INSTANCES
        assert scene_one.ndim==4 # NOTE image must have batch dimension
        batch_size = scene_one.shape[0]
        scene_one = common_augmentations(scene_one,type_two=False)
        scene_two = common_augmentations(scene_two,type_two=True)
        concatenated_instances = concatenated_instances.view(batch_size*self.K_common_instances,*concatenated_instances.shape[2:]) # Squeezed along batch dim to (b*K, c, 96, 96)
        concatenated_instances = common_augmentations(concatenated_instances,type_two=False)
        assert not (scene_one.isnan().any() or scene_two.isnan().any() or concatenated_instances.isnan().any()), "inputs have NaN values."
        
        with torch.no_grad():
            target_encoder = self._get_target_encoder()
            # Pass scene one and two
            target_proj_one, _ = target_encoder(scene_one)
            target_proj_two, _ = target_encoder(scene_two)
            target_proj_one.detach_()
            target_proj_two.detach_()
        online_proj_one, _ = self.online_encoder(scene_one)
        online_proj_two, _ = self.online_encoder(scene_two)
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)


        # FEED THE INSTANCES
        # Get the crops from the image, resize them to 96, feed to online network, and concatenate them
        # Resize and feed instances in overlapping boxes to the online encoder
        
        online_proj_instance, _ = self.online_encoder(concatenated_instances)
        online_pred_instance = self.online_predictor(online_proj_instance) # online_pred_instance is of shape (batch_size*self.K_common_instances, online_pred_instance.shape[-1])
        online_pred_instance = online_pred_instance.reshape(batch_size, self.K_common_instances, online_pred_instance.shape[-1]) # NOTE restore the batch dimension
        online_pred_concatenated_instance = online_pred_instance.reshape(batch_size, self.K_common_instances*online_pred_instance.shape[-1]) # NOTE concatenate instance representations.
        online_concatenated_final_instance_representations = self.instances_to_scene_linear(online_pred_concatenated_instance)
        with torch.no_grad():
            # Pass instances in the overlapping region
            target_proj_instance, _ = target_encoder(concatenated_instances)
            target_proj_instance.detach_()
            target_proj_instance = target_proj_instance.reshape(batch_size, self.K_common_instances, target_proj_instance.shape[-1])
        
        
        # CALCULATE LOSSES
        # Scene to scene loss
        loss_ss_one = self.byol_loss_fn(online_pred_one, target_proj_two.detach())
        loss_ss_two = self.byol_loss_fn(online_pred_two, target_proj_one.detach())
        loss_ss = loss_ss_one + loss_ss_two
        assert not loss_ss.isnan().any(), "loss_ss has NaN"
        # Scene to instance loss
        loss_si_one = self.byol_loss_fn(online_concatenated_final_instance_representations, target_proj_one.detach())
        loss_si_two = self.byol_loss_fn(online_concatenated_final_instance_representations, target_proj_two.detach())
        loss_si = loss_si_one + loss_si_two
        assert not loss_si.isnan().any(), "loss_si has NaN"
        # instance to instance loss (optimal transport and sinkhorn-knopp)
        online_pred_avg = (online_pred_one+online_pred_two)/2
        target_proj_avg = (target_proj_one.detach()+target_proj_two.detach())/2
        loss_ii = self.ii_loss_fn(online_pred_instance, target_proj_instance, online_pred_avg, target_proj_avg)
        assert not loss_ii.isnan().any(), "loss_ii has NaN"

        return loss_ss + loss_si + loss_ii
