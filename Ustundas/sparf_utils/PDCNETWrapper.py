from pathlib import Path
import sys
from argparse import Namespace

env_path = str(Path(__file__).parent /  '../DenseMatching/')
print(env_path)
if env_path not in sys.path:
	sys.path.append(env_path)



from model_selection import select_model

def get_pdcnet():
	args = Namespace(model="PDCNet",
					 pre_trained_model="megadepth",
					 optim_iter=3,
					 flipping_condition=False,
					 local_optim_iter=None,
					 path_to_pre_trained_models="{}/pre_trained_models/".format(env_path),
					 network_type='PDCNet',
					 confidence_map_R=1.0,
					 multi_stage_type="direct",
					 ransac_thresh='1.0',
					 mask_type="proba_interval_1_above_5",
					 homography_visibility_mask="True",
					 scaling_factors=[0.5, 0.6, 0.88, 1, 1.33, 1.66, 2],
					 compute_cyclic_consistency_error=False
					 )



	return select_model(args.model,
						args.pre_trained_model,
						args,
						args.optim_iter,
						args.local_optim_iter,
						path_to_pre_trained_models=args.path_to_pre_trained_models
						)



def get_corr(image1, image2):

	network, _ = get_pdcnet()

	estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
																					  reference_image_,
																					  mode='channel_first')
	confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
	confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]


	return confidence_map