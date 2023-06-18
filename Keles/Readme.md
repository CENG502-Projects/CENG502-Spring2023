# Patch Slimming for Efficient Vision Transformers

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

**Disclaimer: The code for this project was adapted from the [deit repository by Facebook](https://github.com/facebookresearch/deit). Most of our contributions are in the main.py and models_v2.py file.**

# 1. Introduction

The paper [***Patch Slimming for Efficient Vision Transformers**](https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Patch_Slimming_for_Efficient_Vision_Transformers_CVPR_2022_paper.pdf) was published at CVPR 2022. Authors aim to reduce the computational costs of doing inference with vision transformers. This is accomplished by pruning a number of the patches in each layer of the network to reduce the number of attention scores to be computed.

We aim to reproduce the DeiT-III-Small models results for this project. Authors state that they have reduced the number of FLOPs by %45 while only a %0.2 drop in accuracy on the ImageNet 1K dataset. While choosing the smaller DeiT-Tiny model would allow us to do more experiments, we were unable to find a pre trained model that met our needs for the project.

## 1.1. Paper summary

The paper aims to reduce the computational costs of vision tranformers by reducing the number of patches to be processed in each layer. Authors state that patches in a layer are highly similar and have cosine similarity scores around 0.8 in the last layers of ViTs. For every layer, the impact of each patch on the last feature representation is estimated. Models are pruned starting from the last layer, preserved patches are passed down in a top down manner from the last layer to the first layer.

For the ViT-Ti models, authors have reduced the number of FLOPs by %45 with only %0.2 top-1 accuracy decrease on the ImageNet dataset. Authors show that their method preserves the models performance better while also reducing the number of FLOPs the most compared to other state of the art ViT pruning methods.

# 2. The method and my interpretation

## 2.1. The original method

![image](https://github.com/CanKeles5/CENG502-Spring2023/assets/52157220/f7375765-19f9-4bdd-ba1b-5e4f19410849)

Figure 1. Average cosine similarity of patches for different layers of a ViT-Base model. Source [1].


Authors show that patches within a layer are mostly redundant as we go deeper in the model. In the last layers, the cosine similratiy between some pathces reaches 0.8. This implies that some of the patches are redundant and can be eliminated without much performance decrease. In this paper, authors propose a method to reduce the number of patches that are fed into the attention layers. For each layer a binary vector ml is used for representing if a patch is preserved or discarded.


![image](https://github.com/CanKeles5/CENG502-Spring2023/assets/52157220/283c3a09-986f-4f33-a29c-f9d4057c8001)

Figure 2. Comparison of pruning in CNN's and ViT's. Source [1].

In CNN's pruning channels is common. Pruning channels in ViT's dont work well mainly becouse in ViT's each of the patches correspond to one another in different layers. Authors propose a method where we prune the ViT's in a top down manner. We start from the last layer and selectively eliminate a number of pathces in each layer, while preserving the patches in the previous layer for each layer.

Starting from the last layer of the model, we calculate significance scores for each of the patches in a layer. We select the top r patches with the highest significance scores and preserved them, while discarding the rest. We keep track of the patches to be preserved using a matrix m, with shape [num_layers, num_patches]. Each element along the first dimension represents the patches to be preserved in the corresponding layer. We want to minimize the number of patches in the models while preserving the accuracy. To preserve the models accuracy, we calculate an error between the output features of the pruned and initial model. We want to minimize this error while maximizing the number of pruned patches to increase efficiency.

To obtain the masks ml for each layer, we calculate a significance score. The impact of the $t$-th layer's patch on the final error $E_L$ can be reflected by a significance metric $s_t \in \mathbb{R}^N$. For the $i$-th patch in the $t$-th layer, we have

$s_{t,i} = \sum_{h \in [H]L \sim t+1} (A_{h,t}[:, i] \cdot U_{h,t}[i, :])^2$

where $A_{h,t} = QL^{l=t+1} \text{diag}(m_l)P_{h,l}$ and $U_{h,t} = P_{h,t}|Z_{t-1}|$.
$A_{h,t}[:,i]$ denotes the $i$-th column of $A_{h,t}$, and $U_{h,t}[i,:]$ is the $i$-th row of $U_{h,t}$.
$[H]L \sim t+1$ denotes all the attention heads in the $(t + 1)$-th to $L$-th layer.


We modify the MSA and MLP modules as following:

The original MSA block is formulated as following:
$\ B_l(Z_{l-1}) = \mathcal{O}\left(\sum\limits_{h=1}^{H} P_{h,l} Z_{l-1}, \{W_l\}\right) \$

The pruned MSA block can be formulated as following:
$\ B_{b,l}(Z_{l-1}, m_l) = \mathcal{O}\left(\sum\limits_{h=1}^{H} \text{diag}(m_l) P_{h,l} Z_{l-1}, \{W_l\}\right) \$

The pruned MLP module can be formulated as following: 
$\ MLP[l(Z_{b0}^l)] = \text{diag}(m_l) \cdot \varphi(Z_{b0}^lW_a^l)W_b \$


To prune the models, we apply the following algorithm:

**Input**: Training dataset D, vision transformer T with L layers, patch masks {ml}, tolerant value ɛ, preserved patch's number r, search granularity r₀.

1. Initialize mL,₀ as 1 and other elements as 0.
2. Iterate over layers l from L - 1 to 1:
     - Randomly sample a subset of training data to obtain significance scores sl in the l-th layer.
     - Set ml = ml₊₁, El = +∞, r = 0.
     - Repeat until El₊₁ > ɛ:
         - Set r elements in ml,i to 1 based on the positions of the largest r scores sl,i.
         - Fine-tune l-th layer Bl(Zl₋₁) for a few epochs.
         - Calculate error El₊₁ in the (l + 1)-th layer.
         - Increase r by r₀.
3. Output: The pruned vision transformer.

**The key detail for reducing the computational cost of the ViTs with this method lies in implementation details. We use indexing to extract the necessary patches from the original input to the attention block. Instead of feeding the attention block the full input, we feed the patches in the corresponding indexes extracted by the vectors ml. The inputs to the MLP modules within the blocks are the zero padded variants of the extracted indexes.**

## 2.2. Our interpretation 

The paper was easy to understand and covered the necessary details most of the time. Below we list a number of details that we werent able to find in the paper:


- In the paper authors state that they only calcuate attention scores for the patches that have 1 in the corresponding position in the mask, and layer the attention scores are padded to the original input shape before feeding them into the MLP layers. In the paper it is not stated if the positions of the patches are preserved or the zero padding is done by adding zeros after the calculated attention scores. We thought that preserving the position of the attention scores whould be a better choise and we have implemeted the padding in this way.

- Hyper parameters for fine tuning indivudial layers were not provided. We used the hyperparameters that were used to train the models originally becouse of time and resource constraints. These hyperparameters might be crucial becouse we are only training a single layer.

- When calculating the attention scores with the mask, authors formulate a new attention calculating where they use a matrix consisting of the vecotr m's values in its diagonal. If we implement the method directly, the number of FLOPs will not decrease for the pruned model as the input shape hasnt changed. We instead use the vector m[l] as a boolean vector and with indexing we extract the patches we are interested in calculating. After we calculate the attention scores, we pad the output before feeding it to the MLP as described previously. For pruning the MLP, we multiply the MLP's input by a diagonal matrix consisting of the corresponding vector ml.

# 3. Experiments and results
## 3.1. Experimental setup

Becouse of resource constraints, we made a few differences on the experimental setup.

**Model:** We conducted our experiments with a pre-trained DeiT-III-Small model. The pre-trained model was trained on the ImageNet 1K dataset. (There are results for this exact model in the paper.)

**Dataset:** Instead of using the full ImageNet1K dataset, we used ImageNet-Mini becouse of resource constraints. Authors state that fine tuning a single layer is relatively fast, but in our PyTorch implementation this was not the case. With the pre-trained model, the top-1 accuracy on the ImageNet 1K dataset was %80.5.

**Fine tuning:** Authors fine tune the models until a treshold for the error is passed. We omited this and only fine tuned each layer for 5 epochs becouse of resource constraints.

## 3.2. Running the code

We prune a DeiT-III-Small model pre-trained on ImageNet1K. The pre-trained model is downloaded and loaded when you run the main.py. You can provide the dataset path you wish to fine tune your models on when running the command below.

- Run main.py with the command, this will prune the model and fine tune it on the dataset you provided.
```
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model deit_small_patch16_224 --batch-size 128 --data-path /path_to_your_dataset --output_dir /your_output_directory
```

If you want to apply this method to another model, simply have a look at the models_v2.py file and make the necessary additions to the standard ViT structure.

## 3.3. Results

There are two metrics we are interested in this project, one is number of FLOPs and the other is how much accuracy can we maintain from the original model.
We have reduced the number of FLOPs by %a.aa. We think that we can further reduce the number of FLOPs by optimizing our implementation.

![image](https://github.com/CanKeles5/CENG502-Spring2023/assets/52157220/06fb0338-ad23-4af8-b757-00802f353436)
Table 1. Results from the paper. Source [1].

We have not managed to preserve the accuracy of the model fully. When pruning the models, we need to fine tune the layers for a few epochs on the ImageNet dataset. While authors state that fine tuning a single layer is very computationally cheap, in our PyTorch implementation training a model with only a single layer unfreezed still takes a lot of time. We fine tuned our models on the ImageNet-mini dataset which is a dataset that has much less samples compared to the original ImageNet dataset.

**We have achived to reduce the number of FLOPs by %33 while dropping the accuracy of the model by %15 on the ImageNet-Mini validation set, the accuracy dropped from %80.5 to %65.1.** We could not preserve the accuracy of the model most probably due to not being able to fine tune the layers long enough. We have started an experiment that fine tunes each layer on the whole dataset but due to time and resource constraints we do not currently have results to present right now. The losses seem promising and we are confident that if we fine tuned on the whole dataset, the performance drop would be much smaller. It should be noted that the number of FLOPs can change with software and hardware specifications. Our results were obtained on a CPU.

The reduction of number of FLOPs depends on the number of pruned patches, in our experiments we have seen that a reduction of %17 to %40 percent in the number of FLOPs is achivable.

# 4. Conclusion

We believe that we have been able to confirm the results of the authors partially. We have reduced the number of FLOPs while maintaining meaningfull weights that still achive a respectable accuracy.

**FLOPs:** We have seen that our results for reducing the number of FLOPs are close enough to confirm the authors results. It should be noted that these results may depend on hardware and software specifications.

**Preserving accuracy:** In our experience, fine tuning the layers is crutial for preserving the accuracy of the models. We have preserved meaningfull accuracies when fine tuning layers, but we had drastic decreases in accuracy. We belive that if we fine tuned on the whole dataset, the decreases in performance would be negligable and close to the ones reported in the paper.

**Fine tuning:** Authors state that fine tuning individual layers is much faster compared to training the whole model. In our PyTorch implementation, freezing the whole model except individual layers and training them took comparable time to training the whole model. This prevented us from experimenting on the full ImageNet 1K dataset. We set the requires_grad parameter to false for the whole model except the layer we want to fine tune.

# 5. References
[1] [Yehui Tang, Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chao Xu, Dacheng Tao; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022, pp. 12165-12174](https://openaccess.thecvf.com/content/CVPR2022/html/Tang_Patch_Slimming_for_Efficient_Vision_Transformers_CVPR_2022_paper.html)

# Contact
Muhammed Can Keleş can.keles@metu.edu.tr
