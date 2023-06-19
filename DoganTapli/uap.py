import torch
import numpy as np

from torch import nn
from tqdm import tqdm

# L-infinity norm
# eps 0.039
# PGD attack
# 30 for Pong 72 for MsPacman
# 6 actions for Pong: 6*5 universal perturbations
# model is the victim agent
# data is a collection of states which receives the same action output from the model

def uap(model, data, target_action, action_space, step_size= 0.006, eps = 0.039, success_rate= 1, batch_size = 64):
  perturbation = torch.zeros_like(data[0])

  attack_success = 0
  y_target = nn.functional.one_hot(torch.tensor(target_action), num_classes= action_space)

  cross_ent = nn.CrossEntropyLoss(reduction = 'none')

  batch_per_epoch = int(data.shape[0]/batch_size)

  # Continue until attack success is %100
  while(attack_success < success_rate):

    for i in tqdm(range(batch_per_epoch)):
      states = data[i*batch_size:(i+1)*batch_size]
      b_perturbation = perturbation.unsqueeze(0).repeat([batch_size, 1, 1, 1])
      b_target = y_target.repeat([batch_size,1])

      perturbed = torch.clamp((states + b_perturbation), 0, 1).requires_grad_(True)
      outputs = model(perturbed)

      loss =  torch.mean(cross_ent(outputs, b_target.float()))
      loss.backward()

      with torch.no_grad():
        grad_sign = perturbed.grad.data.mean(dim = 0).sign()
        perturbation = perturbation - grad_sign * step_size

        # L-infinity norm
        perturbation = torch.clamp(perturbation, -eps, eps)

    # evaluation
    # num_of_samples = 0
    attack_success = 0
    for i in tqdm(range(batch_per_epoch)):
      states = data[i*batch_size:(i+1)*batch_size]
      b_perturbation = perturbation.unsqueeze(0).repeat([batch_size, 1, 1, 1])
      perturbed = torch.clamp((states + b_perturbation), 0, 1)
      outputs = model(perturbed).detach()
      pred = torch.argmax(outputs, axis=1).numpy() # check if correct axis

      # num_of_samples += batch_size
      attack_success += np.sum(pred == target_action)

  return perturbation # detach ?


def query_uap(orig_action, lure_action):
  assert orig_action.shape[0] !=lure_action.shape[0]
  perturb = []
  for i in range(orig_action.shape[0]):
    key = '{}_{}'.format(orig_action[i].item(), lure_action[i].item()) # item ?
    perturb.append(uap_dict.get(key))

  return torch.as_tensor(perturb)