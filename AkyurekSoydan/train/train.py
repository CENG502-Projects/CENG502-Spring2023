#######################################################################################
# 
#    The code given on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html 
#    is used and modified according to the project
#
#######################################################################################

# PyTorch TensorBoard support
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from train.trainoneepoch import train_one_epoch

def train(
    model,
    loss_fn,
    optimizer,
    num_epoch,
    dataloader_trn,
    dataloader_val,
    loss_batch = 10,
    device = "cpu",
    use_dice_loss = True,
    overfit = False,
):

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/AAformer_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000.

    for epoch in range(num_epoch):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model=model, 
                                   optimizer=optimizer, 
                                   epoch_index=epoch_number, 
                                   dataloader_trn=dataloader_trn, 
                                   loss_fn=loss_fn, 
                                   loss_batch=loss_batch, 
                                   tb_writer=writer, 
                                   device=device,
                                   use_dice_loss=use_dice_loss)
                 
        running_vloss = 0.0
        
        if not overfit:
            # Set the model to evaluation mode
            model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(dataloader_val):
                    v_query_img = vdata['query_img'].to(device)
                    v_query_mask = vdata['query_mask'].to(device)
                    v_supp_imgs = vdata['support_imgs'].to(device)
                    v_supp_masks = vdata['support_masks'].to(device)

                    voutputs = model(v_query_img, v_supp_imgs, v_supp_masks, normalize=use_dice_loss)

                    vloss = loss_fn(voutputs, v_query_mask.unsqueeze(1))
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                #torch.save(model.state_dict(), model_path)
                torch.save(model, model_path)
        
        epoch_number += 1