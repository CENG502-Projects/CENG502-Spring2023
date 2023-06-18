from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
mean = np.array(img_mean)
std = np.array(img_std)

transform_img = transforms.Compose(
    [
        transforms.Normalize(-mean/std, 1/std),
        transforms.ToPILImage(),
    ]
 )

transform_mask = transforms.Compose(
  [
      #transforms.Normalize(-mean/std, 1/std),
      transforms.ToPILImage(),
  ]
)


def image_plot(query_img, query_mask, supp_imgs=None, supp_masks=None, preds=None):
    fig = plt.figure()
   
    # plotting support images & masks
    if supp_imgs != None:
        batch_sup_im = supp_imgs.shape[0]
        shot_sup_im = supp_imgs.shape[1]
        
        i = 0
        for s in range(shot_sup_im):
            for supp_img in supp_imgs[:,s,:,:]:
                plt.subplot(shot_sup_im, batch_sup_im, i+1)
                plt.imshow(transform_img(supp_img))
                plt.suptitle("support images")
                plt.axis('off')
                i += 1
        plt.show()
     
    if supp_masks != None:
        batch_sup_ma = supp_masks.shape[0]
        shot_sup_ma = supp_masks.shape[1]
        
        i = 0
        for s in range(shot_sup_ma):
            for supp_mask in supp_masks[:,s,:,:]:
                plt.subplot(shot_sup_ma, batch_sup_ma, i+1)
                plt.imshow(transform_mask(supp_mask))
                plt.suptitle("support masks")
                plt.axis('off')
                i += 1
        plt.show()
        
    
    # plotting query images
    batch_que_im = query_img.shape[0]
    i = 0
    for q in range(batch_que_im):
        plt.subplot(1, batch_que_im, i+1)
        plt.imshow(transform_img(query_img[q,:,:,:]))
        plt.suptitle("query images")
        plt.axis('off')
        i += 1
    plt.show()  
    
    # plotting query masks
    batch_que_ma = query_mask.shape[0]
    i = 0
    for q in range(batch_que_ma):
        plt.subplot(1, batch_que_ma, i+1)
        plt.imshow(transform_mask(query_mask[q,:,:]))
        plt.suptitle("query masks")
        plt.axis('off')
        i += 1
    plt.show()  
    
    batch_pre_ma = preds.shape[0]
    i = 0
    if preds != None:
        i = 0
        for p in range(batch_pre_ma):
            plt.subplot(1, preds.shape[0], i+1)
            plt.imshow(transform_mask(preds[p,:,:]))
            plt.suptitle("predicted masks")
            plt.axis('off')
            i += 1
        plt.show()

#plt.savefig("test.png", bbox_inches='tight')