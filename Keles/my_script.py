
'''
Draft for pruning procedure.
Notes:

- What does the signifignace metric consist of?
	- Its the F norm of a product of A and U
	- where A is the product of the attention & binaryvec vals from the current to the last layer
	- U is attention of current layer t times "prev features ????" Zt-1
		- We need to figure out what exactly the Z value is and how we can get this value:
			- Zt is the output features from the corresponding layer t
			- What is |Zt|? Is it the absolute value? Is it a norm?

- Finetuning:
    - To fine tune layer l, we need to freeze all layers except layer l. Authors fine tune for x epochs.
    - TODO: Freeze all layers except layer l.

General todo:
- Check if the significance score computation & other code is correct.
- Obtain data for imagenet. We will probably be able to work with only a subset of the dataset.
- Fine tune the model.
- Package the whole project before 11/6.
- Fix errors till 18/6.
'''

from models_v2 import *
import torch
import subprocess
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


if __name__ == '__main__':
    
    cmd_command = 'python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model deit_small_patch16_224 --batch-size 2 --data-path /home/cankeles/patch_slimming_tfms/imagenet-mini --output_dir /home/cankeles/patch_slimming_tfms/models'
    
    # Execute the command and capture the output
    result = subprocess.run(cmd_command, shell=True, capture_output=True, text=True)

    data_path = r"/home/cankeles/patch_slimming_tfms/imagenet-mini"
    
    # Define the transformation to apply to each image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust mean and std if needed
    ])
    
    # Create the ImageFolder dataset
    dataset = ImageFolder(root=data_path, transform=transform)
    
    # Set the batch size and the number of workers for loading data
    batch_size = 1
    
    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(dataloader)

    pruned_model = deit_small_patch16_18x2(pretrained=True)
    orig_model = deit_small_patch16_18x2(pretrained=True)
    
    num_layers = orig_model.depth #totally made up number
    tol = 1e-1 #made up number
    
    r = 10
    rp = 1
    
    #Freeze the whole model.
    for param in pruned_model.parameters():
        param.requires_grad = False
    
    for l in range(num_layers-1, 0, -1):
        #rand_imgs = torch.rand(4, 3, 224, 224)
        images, _ = next(data_iter)
	
        feats_orig, _  = orig_model.apply_block(images, l)
        feats_pruned, sig_scores = pruned_model.apply_block(images, l)

        error = 1000.0 #should be infinity
        
        while error > tol:
            
            top_inds = torch.topk(sig_scores, r)
            
            #print(f"m: {pruned_model.m[l]}")
            pruned_model.m[l][top_inds[-1]] = 1.0
            #print(f"m: {pruned_model.m[l]}")
            #print("------------------------------------------------")
            
            #finetune layer ?????????
            """
            How can we fine tune the layer? We need to freeze all layers except layer l.
            Freezing is done. +++
            We need to train the model
            """
            
            #print(f"Before: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)}")
            #pruned_model.freeze_layers(l)
            #print(f"After: {sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)}")
            
            #Fine tune layer l somehow
            #

            torch.save(pruned_model.state_dict(), "/home/cankeles/patch_slimming_tfms/models/checkpoint.pth")

            cmd_command = 'python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model deit_small_patch16_224 --batch-size 2 --data-path /home/cankeles/patch_slimming_tfms/imagenet-mini --output_dir /home/cankeles/patch_slimming_tfms/models'

            # Execute the command and capture the output
            result = subprocess.run(cmd_command, shell=True, capture_output=True, text=True)

            # Check the return code for errors
            if result.returncode == 0:
                # Command executed successfully
                print(result.stdout)
            else:
                # Command encountered an error
                print(result.stderr)

            print(f"result: {result}")
            
            #pruned_model = deit_small_patch16_18x2().load_state_dict(torch.load("/home/cankeles/patch_slimming_tfms/models/checkpoint.pth"))

            print(pruned_model)

            feats_orig, _  = orig_model.apply_block(images, l)
            feats_pruned, sig_scores = pruned_model.apply_block(images, l)

            error = torch.norm(feats_orig-feats_pruned)
            print(f"Error: {error}")

            #print(f"Orig feats layer {l}: {feats_orig}")
            #print(f"Pruned feats layer {l}: {feats_pruned}")
            #print(f"-----------------------------------------------")
            
            r += rp
            
            break #bc we are not fine tuning yet

