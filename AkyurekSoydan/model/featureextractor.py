import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self, 
                 device,
                 layers=50,             # number of layers for ResNet
                 reduce_dim = 256,      # reduced dimension stated in supplementary material of the paper
                 c = 256                # output feature map dimension 
                 ):
        super().__init__()
        self.c = c
        self.model_set(layers)          

        self.down = nn.Sequential(
            nn.Conv2d(self.fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)   
        )   
        self.device = device


    def forward(self,
                x         : torch.Tensor,         # query image          # query_img.shape = [batch,rgb,image.h,image.w]
                x_supp    : torch.Tensor,         # support images       # supp_img.shape  = [batch,shot,rgb,image.h,image.w]
                mask      : torch.Tensor,         # suport images masks  # supp_mask.shape = [batch,shot,image.h,image.w]
                ):
        
        
        q_mid_feat, q_high_feat = self.query_feature_extraction(x)
        
        s_mid_feat_list, s_high_feat_list, s_mask_list = self.support_feature_extraction(x_supp, mask)
        
        #print("q_high_feat.shape = ", q_high_feat.shape)
        prior_mask = self.prior_generation(s_high_feat_list, s_mask_list, q_high_feat, (q_mid_feat.shape[-1],q_mid_feat.shape[-2]))
        
        masked_ave_pool_list = self.masked_average_pooling(s_mid_feat_list, s_mask_list)

        temp_supp = torch.cat([torch.cat([torch.cat(s_mid_feat_list, 1), torch.cat(masked_ave_pool_list, 1)],1), prior_mask],1).to(self.device)
        supp_down = nn.Sequential(
            nn.Conv2d(temp_supp.shape[1], self.c, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)   
        ).to(self.device)

        temp_query = torch.cat([q_mid_feat, torch.cat(masked_ave_pool_list, 1)],1).to(self.device)
        query_down = nn.Sequential(
            nn.Conv2d(temp_query.shape[1], self.c, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)   
        ).to(self.device)
         
#        return q_mid_feat, q_high_feat, s_mid_feat_list, s_high_feat_list, s_mask_list, prior_mask, masked_ave_pool_list
        return query_down(temp_query).permute(0, 2, 3, 1), supp_down(temp_supp).permute(0, 2, 3, 1), s_mask_list

    ################################
    #  model_set: sets the pretrained ResNet model layers according to the given label number
    #
    # inputs: layers: int
    # outputs: None
    #
    def model_set(self,
                  layers: int = 50,         # Number of layer for trained model
                  **kwargs
                  ):

        if layers == 18:
            resnet = models.resnet18(weights='DEFAULT', progress=True, **kwargs)
            self.fea_dim = 128 + 256
        elif layers == 34:
            resnet = models.resnet34(weights='DEFAULT', progress=True, **kwargs)
            self.fea_dim = 128 + 256
        elif layers == 50:                                                  # Paper uses Layer=50
            resnet = models.resnet50(weights='DEFAULT', progress=True, **kwargs)
            self.fea_dim = 1024 + 512       # ResNet Layer3 in_ch size + Layer2 out_ch size  
        elif layers == 101:
            resnet = models.resnet101(weights='DEFAULT', progress=True, **kwargs)
            self.fea_dim = 1024 + 512       # ResNet Layer3 in_ch size + Layer2 out_ch size  
        elif layers == 152:
            resnet = models.resnet152(weights='DEFAULT', progress=True, **kwargs)
            self.fea_dim = 1024 + 512       # ResNet Layer3 in_ch size + Layer2 out_ch size  
        else:
            raise Exception("Irrelevant layer number")

        #self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:                      
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)    # changes ResNet layer3 conv2 layers parameters
            elif 'downsample.0' in n:
                m.stride = (1, 1)                                           # changes ResNet layer3 downsample layers parameters  
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:                                                
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)    # changes ResNet layer4 conv2 layers parameters
            elif 'downsample.0' in n:
                m.stride = (1, 1)    
    #
    # End of model_set
    ####################################### 

    ################################
    # query_feature_extraction: extracts mid-level and high-level query image features from Resnet  
    #
    # inputs: input_image_x: torch.Tensor, model: nn.Moduler
    # outputs: mid_features:torch.Tensor (b,layer2.c+layer3.c,layer3.h,layer3.w) -> (b,1024+512,14,14), high_features: torch.Tensor (b,layer4.c,layer4.h,layer4.w) -> (b,2048,7,7)
    #
    def query_feature_extraction(self,
                                 x     : torch.Tensor,     # input query image,shape = [batch,rgb,image.h,image.w]
                                 ):        

        with torch.no_grad():
            query_feat_0 = self.layer0(x)                       # (b,64,112,112)
            query_feat_1 = self.layer1(query_feat_0)            # (b,256,56,56)
            query_feat_2 = self.layer2(query_feat_1)            # (b,512,28,28)
            query_feat_3 = self.layer3(query_feat_2)            # (b,1024,14,14)
            query_feat_4 = self.layer4(query_feat_3)            # (b,2048,7,7)
            
            query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True) # interpolate from (b,_,layer2.h,layer2.w) to (b,_,layer3.h,layer3.w) 

        mid_features = torch.cat([query_feat_3, query_feat_2], 1) # ? query_feat_3 and query_feat_2 sizes match??? concat to form query feature (b,1024+512,h,w)
        mid_features = self.down(mid_features)                # down to query features to (b,256,h,w)
        
        high_features = query_feat_4
        return mid_features, high_features
    # 
    ###################################

    ###################################
    # support_feature_extraction: extracts mid-level and high-level support images features from Resnet    
    #
    # inputs: s_i: torch.Tensor, mask: torch.Tensor             
    # outputs: mid_supp_list: [torch.Tensor], high_supp_list: [torch.Tensor], mask_list[Torch.Tensor]
    #
    def support_feature_extraction(self,
                                   s_i   : torch.Tensor,         # support images        supp_img.shape  = [batch,shot,rgb,image.h,image.w]
                                   s_m   : torch.Tensor,         # suport images masks   supp_mask.shape = [batch,shot,image.h,image.w]
                                   ):

        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        shot = s_i.shape[1]         # shot number
        for i in range(shot):                                    
            mask = (s_m[:,i,:,:] == 1).float().unsqueeze(1)           # defining mask
            mask_list.append(mask)                                    # list masks
            with torch.no_grad():                                     # feature extraction for support images
                supp_feat_0 = self.layer0(s_i[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)  # interpolate mask accoridng to layer3 output sizes
#                supp_feat_4 = self.layer4(supp_feat_3 * mask)           # layer3_output masking output given to layer_4
                supp_feat_4 = self.layer4(supp_feat_3)
                final_supp_list.append(supp_feat_4)                   # list layer_4 outputs
                supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)      # Concat layer3 and layer2 outputs in ch dimension
            supp_feat = self.down(supp_feat)                     # (1024 + 512) -> 256
#            supp_feat = Weighted_GAP(supp_feat, mask)                 # gap between feature and mask
            supp_feat_list.append(supp_feat)                          # list gaps
        
            mid_supp_list = supp_feat_list
            high_supp_list = final_supp_list
        return mid_supp_list, high_supp_list, mask_list
    #
    # Mid-level-features of I_s   : "supp_feat_list" (len=shot) includes "supp_feat" (b,256,14,14) of each I_s     ###
    # High-level-features of I_s  : "final_supp_list" (len=shot) includes "supp_feat_4" (b,2048,7,7) which is the masked supp_feat3 output 
    # Masks                       : "mask_list" (len=shot) includes "mask" (b,1,input_image.h,input_image.w) orignal size
    #
    # End of support_feature_extraction
    #######################################

    #######################################
    # prior_generation: prior mask generation
    #
    # inputs: high_supp_list: [torch.Tensor], mask_list: [Torch.Tensor], high_query_feat: torch.Tensor, mid_query_size: tuple
    # outpts: corr_query_mask: torch.Tensor
    #
    def prior_generation(self,
                         high_supp_list   : list[torch.Tensor],     # list of high support features
                         mask_list        : list[torch.Tensor],     # list of mask features
                         high_query_feat  : torch.Tensor,           # high query features 
                         mid_query_size   : tuple,
                         ):
      
        bsize, ch_sz, sp_sz, _ = high_query_feat.shape #size()[:]                      # ch_sz = 2048, sp_sz = 7     
        high_query_feat = high_query_feat.contiguous().view(bsize, ch_sz, -1) # resize layer4 query output from 4 dims (b,c,h,w) to 3 dims (b,c,h*w)
        high_query_feat_norm = torch.norm(high_query_feat, 2, 1, True)        # norm calculation: second order (2) over dimension=1 channel -> (b,1,h*w)


        
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(high_supp_list):           # final_supp_list: layer4 outputs
            resize_size = tmp_supp_feat.size(2)                       # layer4 size (7,7)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)  # interpolate mask[i]

            tmp_supp = tmp_supp_feat * tmp_mask                # masking layer4 output     
            
       
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)   # resize masked layer_4 output from 4 dims (b,c,h,w) to 3 dims (b,c,h*w)
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)         # change dimension order to (b,h*w,c) for bacth matrix multiplication, bmm
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)          # norm calculation: second order (2) over dimension=2 channel -> (b,h*w,1)
            """
            print(tmp_supp.shape)
            print(high_query_feat.shape)
            print(tmp_supp_norm.shape)
            print(high_query_feat_norm.shape)
            """
            similarity = torch.bmm(tmp_supp, high_query_feat)/(torch.bmm(tmp_supp_norm, high_query_feat_norm) + cosine_eps)   # cosine simmilarity (b,h*w,h*w) for all pixels
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)    # obtain max of cosine simm for dim=1 (all pixel dimension) and reshape them to get prior mask from (b,h*w,h*w) to (b,layer4.h,layer4.h)    
               
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)  # prior mask min-max normalization
               
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)                # resize normalized prior mask to (b,1,layer4.h,layer4.h) 
            corr_query = F.interpolate(corr_query, size=mid_query_size, mode='bilinear', align_corners=True) # resize normalized prior mask to (b,1,layer3.h(14),layer3.w(14))
            corr_query_mask_list.append(corr_query)                                 # list normalized prior masks
        
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)   # concat normalized prior mask list on (previosuly extended) dim=1, shape (b,self.shot,layer3.h,layer3.w) and calculate mean for dim=1 to form (b,1,layer3.h,layer3.w)   
        corr_query_mask = F.interpolate(corr_query_mask, size=mid_query_size, mode='bilinear', align_corners=True)  # interpolate
        
        return corr_query_mask
    #
    # end of prior generation 
    ############################################

    #######################################
    # masked_average_pooling: average pooling of support image features
    #
    # inputs: mid_supp_list: [torch.Tensor], mask_list: [Torch.Tensor]
    # outpts: masked_ave_pool_list: [torch.Tensor]
    #
    def masked_average_pooling(self,
                               mid_supp_list    : list[torch.Tensor],     # list of mid support features
                               mask_list        : list[torch.Tensor],     # list of mask features
                               ):
        mid_supp_size = mid_supp_list[0].shape[-2], mid_supp_list[0].shape[-1]
        masked_ave_pool_list = []
        for i, tmp_supp_feat in enumerate(mid_supp_list):           # final_supp_list: layer4 outputs
            tmp_mask = F.interpolate(mask_list[i], size=mid_supp_size, mode='bilinear', align_corners=True)  # interpolate mask[i]
            tmp_supp = tmp_supp_feat * tmp_mask / torch.sum(tmp_mask) 
            masked_ave_pool_list.append(tmp_supp)

        return masked_ave_pool_list
    
    #
    # end of masked_ave_pool_list
    ############################################
