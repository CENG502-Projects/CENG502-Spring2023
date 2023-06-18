# Uncertainty-Aware Learning Against Label Noise on Imbalanced Datasets

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

Authored by Yingsong Huang, Bing Bai, Shengwei Zhao, Kun Bai and Fei Wang, "Uncertainty-Aware Learning Against Label Noise on Imbalanced Datasets" [1] was published in AAAI 2022. The goal of the paper is to come up with a novel method to overcome the drawbacks of noise and class imbalance, especially when they co-exist. 

My goal with this repository is to provide the code for the presented novel technique and reproduce some of the results.

## 1.1. Paper summary

Since a significant portion of the real-life applications involve dealing with noisy labels, learning against label noise is a common challenge in the machine learning literature. Recent literature aims to address this problem through considering the output probabilities of the models and/or the loss values to separate clean and noisy samples with the aim of treating them differently.

However, these techniques fail to fully address some of the most important real-life cases especially in the presence of class imbalance, even when the imbalance rate is in the order of 1:5.

The authors first conjecture that this might be due to the fact that the existing literature fails to consider the predictive uncertainty of the model's while solely focusing on the output probabilities.

Furthermore, the authors claim that these existing class-agnostic approaches are not working well on imbalanced datasets as inter-class loss distribution varies significantly across classes in imbalanced datasets.

Motivated by these shortcomings, the authors propose a novel framework, "Uncertainty-aware Label Correction (ULC)", to address these gaps.

# 2. The method and my interpretation

## 2.1. The original method

<p align="center">

![Screenshot 2023-06-18 at 21 32 54](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/5c838db7-b2fd-44dd-8fa4-1d300350478a)

</p>

Figure 1. The pseudocode for the proposed method, coined as "Uncertainty-aware Label Correction (ULC)"




- Uncertainty-aware Label Correction (ULC) has two major novelties:
- **"Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)"** module and the **"Aleatoric Uncertainty-aware Learning (AUL)"**

### 2.1.1 Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)
- First of these is the **"Epistemic Uncertainty-Aware Class-specific Noise Modeling (EUCS)"** module. With this module, the authors aim to fit the inter-class discrepancy on the loss distribution.
- Initially, the authors obtain the epistemic uncertainty estimations for each of the samples through utilizing MC Dropout [4]. With MC Dropout [4], _T_ stochastic forward passes are performed with dropout enabled for each of the input samples during the test time. Following obtaining the output probabilities from each of these passes, taking their entropy and then normalizing it would be yielding the epistemic uncertainty for that particular sample.
- After the epistemic uncertainty estimation, the authors fit a GMM with to the each class's loss distribution, then compute the probability of having the mean of the component with the lower $\mu$ given each samples' loss, i.e $p(\mu_{j0} | l_i)$ for class j and sample i.
- Based on these two steps, the authors then come up with the following equation to determine the probability of a given sample i being clean or noisy:

<p align="center">
$\omega_i = (1-\epsilon)r p(\mu_{j0} | l_i)^{1-r}$
</p>

where $\epsilon_i$ corresponds to the epistemic uncertainty for that sample and $r$ being a hyperparameter to weight the uncertainty and the probability from GMM.

- The authors then apply thresholding based on hyperparameter $\tau$ to decide which samples are considered clean and which samples are considered noisy. At this stage, the noisy samples are discarded as being unlabeled.
- Finally, label of the cleaned samples are also refined based on the following equation:

<p align="center">
$y_i = \omega_i y_i^{~} + (1-\omega_i) \hat{y_i}$
</p>

where $y_i^{~}$ is the label possibly with noise and  $\hat{y_i} = \frac{1}{T} \sum_{t} softmax(f(x_i, W))$ 

![Screenshot 2023-06-18 at 21 41 25](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/1d7bbae3-c668-4981-b250-34073a4ec724)

Figure 2. The pseudocode for the EULC module of the proposed framework




### 2.1.2 Aleatoric Uncertainty-aware Learning (AUL)

- In this second module, the authors aim to utilize an objective akin to the one proposed by Kendall and Gal [3].
- The authors claim that this module is particularly important as the noise modeling achieved by the EULC module is not sufficient to account for the residual noise that may contribute to overfitting in certain cases.
- Specifically, the authors aim to model aleatoric uncertainty through logit corruption with Gaussian noise, which leads to a learned loss attenuation as described more in detail in Kendall and Gal [3]. This learned loss attenuation is particularly helpful while learning against noise and providing robustness as it attenuates the effects of corrupted labels.
- Two different types of noise are considered with the assumption of independence between them: Instance-dependent noise and class-dependent noise. Formally, the corresponding corruption process can be observed from the following equation:

<p align="center">
  $\hat{v}_i(W) = \delta^{x_i}(W) + \delta^y f_{W}(x_i)$
</p>

where $\hat{v}_i(;)$ stands for the $i^{th}$ logit, $\delta^{x_i}$ stands for the instance-dependent noise factor and $\delta^y$ stands for the class-dependent noise factor.

- Then, these corrupted logits are passed through a softmax layer to obtain output proabilities:

<p align="center">
  $\hat{y_i} = softmax(\delta^{x_i}(W) + \delta^y f_{W}(x_i))$
</p>

- Furthermore, the authors assume that both $\delta^{x_i}$ and $\delta^y$ are drawn from their respective Gaussian distributions (with the assumption of independence between them) and reserve an additional head from the network to effectively predict and learn $\sigma^{x_i}$ of the Gaussian that we sample the $\delta^{x_i}$ from.

- Finally, for the objective part, the authors leverage the aforementioned bulletpoints to define the following objectives for the labeled and unlabeled samples:

<p align="center">
  $l_x = \frac{-1}{X'} \sum_{x_i, y_i} (y_i)^{T} log \frac{1}{T} \sum_t \hat{y_{it}}(x_i ; W, \sigma^{x_i}, \sigma^{Y})$
</p>

<p align="center">
  $l_u = \frac{1}{U'} \sum_{x_i, y_i} || y_i - \frac{1}{T} \sum_t \hat{y_{it}}(x_i ; W, \sigma^{x_i}, \sigma^{Y}) ||^2_2$
</p>

where $X'$ stands for the labeled inputs (following MixMatch [5] augmentation), $l_x$ stands for the labeled inputs' objective, , $U'$ stands for the unlabeled input (following MixMatch [5] augmentation), $l_u$ stands for the unlabeled inputs' objective, and $\hat{y_{it}}$ stands for the softmax output of the $t^{th}$ sampling pass from the network.

- Introducing a balancing hyperparameter $\lambda_u$ between these two objectives and combining them yields the objective that the paper utilizes:

<p align="center">
  $l_c = l_x + \lambda_u l_u$
</p>


![Screenshot 2023-06-19 at 00 21 58](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/dfb4afce-5dd3-4be6-8c0d-9efe0f517d96)

Figure 3. The pseudocode for the AUL module of the proposed framework




## 2.2. Our interpretation 

- First and foremost, this paper has strict prerequisites of understanding other works such as DivideMix [2], Kendall and Gal [3], MixMatch [5], MC Dropout [4].
- This is primarily due to the fact that crucial procedures such as epistemic uncertainty quantification and/or aleatoric uncertainty-based loss attenuation are not explained in a detailed manner and the authors rather refer the reader to the relevant citations.
- Furthermore, the structure of the objective functions for both the labeled and unlabeled samples do not seem trivial from a first sight but makes much more sense after carefully reading DivideMix [2] as their structure and principle is extremely similar to it.



![Screenshot 2023-06-19 at 00 27 16](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/d2215987-08b2-4c96-a012-b88fa593d1a6)

Figure 4. Objectives for labeled and unlabeled samples respectively from DivideMix[REF]




- Other important details, such as the co-teaching strategy with two networks, the semi-supervised learning strategy for the unlabeled samples and many more design choices are explained in a more detailed manner in the works cited by the authors. Since this work is not the one to introduce them, neither the authors of the paper nor this repository's discussion sections will be providing details on those parts.

- Finally, the overall pipeline and the algorithm borrows heavily from the DivideMix [2] work, while also introducing many novelties to it at the same time, especially from the uncertainty quantification literature. Hence, my suggestion to anyone willing to use this repository is to first thoroughly go through the aforementioned citations in the first bulletpoint of this section.

  
![Screenshot 2023-06-19 at 00 31 16](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/ec35bd1e-b3b4-4b2b-9e1c-f5232ce47842)

Figure 5. The pseudocode for the DivideMix [2], provided here for the sake of highlighting the similarities

**A big thanks for the  [DivideMix GitHub repository](https://github.com/LiJunnan1992/DivideMix) for providing the necessarly SSL code, the MixMatch code and the noise-injecting dataloaders!**

# 3. Experiments and results

## 3.1. Experimental setup

In the paper, training settings are desribed as follows:

<p align="center">
  We train the network using SGD with a momentum of 0.9 for 300 epochs; warm-up 30 epochs for CIFAR-10 and 40 epochs for CIFAR-100. In the Cloth- ing1M experiments, we use ResNet-50 with ImageNet pre-trained weights. The warm-up period is 1 epoch for Clothing1M. 𝜏 is set as 0.6 for 90% noise ratio and 0.5 for oth- ers. 𝜆𝑢 is validated from {0, 25, 50, 150}. Generally, the hyperparameters setting for MixMatch is inherited from DivideMix without heavily tuning, because the SSL part is not our focus and can be replaced by other alternatives. We leverage MC-dropout [4] to estimate uncertainty, setting 𝑇 to 10 and the dropout rate to 0.3. The uncertainty ratio 𝑟 is set as 0.1 to obtain the final clean probability.
</p>

I only performed experiments with the synthetic CIFAR-10 and imbalanced CIFAR-10 datasets. For them, I did not change any training settings. One significant point to note is that normally MC Dropout [4] is used with setting 𝑇 $\in [30, 100]$ while the authors choose 10. I did not change this part either but this point may have significant effects on the reproducibility of the proposed approach as it directly effects both of the proposed modules. Finally, the backbone is chosen as the (PreAct) ResNet18 for the entirety of the CIFAR-10 experiments.


## 3.2. Running the code

First, please download the CIFAR-10 dataset and put it under a directory called ``` data/ ```
The directory structure looks like the following:

```
├── main_cifar.py
├── dataloader_cifar.py
├── PreResNet.py
└── requirements.txt
```

One can directly call the main_cifar.py file through passing the --data_path argument to it (same goes for the imbalanced versions of CIFAR-10, the code will be executing whichever directory, whether balanced or imbalanced passed to it):

```
  python3 main_cifar.py --data_path data/cifar-10-batches-py
```



## 3.3. Results

Directly from the paper, synthetic (balanced) CIFAR-10 results look like the following:

![Screenshot 2023-06-19 at 01 02 51](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/52210294-1ade-44e6-bcb8-554f7d8e51d0)

Table 1. Synthetic CIFAR-10 Results of ULC compared to other well-known methods (Table 1 from the paper)





| CIFAR-10 (Sym.)     | 20%  | 50%  | 80%  | 90%  |
|---------------------|------|------|------|------|
| ULC-Reprod. (Best)  |      |      |      |      |
| ULC-Reprod. (Last)  |      |      |      |      |

Table 2. Synthetic CIFAR-10 Results w/ Symmetric Noise of Reproduction ULC




![Screenshot 2023-06-19 at 01 02 51](https://github.com/selimkuzucu/ULC-CENG502/assets/56355561/de5f784e-bdfd-4125-a826-438c07c2a65a)

Table 3. Synthetic Imbalanced CIFAR-10 Results of ULC compared to other well-known methods (Table 2 from the paper)



| CIFAR-10 (Sym.)     | 1:5 & 20%  | 1:5 & 50%  | 1:10 & 20%  | 1:10 & 50%  |
|---------------------|------------|------------|-------------|-------------|
| ULC-Reprod. (Best)  |            |            |             |             |
| ULC-Reprod. (Last)  |            |            |             |             |


Table 4.  Synthetic Imbalanced CIFAR-10 Results w/ Symmetric Noise of Reproduction ULC



# 4. Conclusion

In conclusion, providing the code for the proposed methods was not an easy task considering the significant prerequisite chain required to appreciate the work. Some of the results could not be reproduced with the same level of success as presented in the paper. This may be primarily due to the fact that the method itself is quite stochastic in nature, whether estimating the epistemic uncertainties with _only 10 MC sampling passes_ or obtaining outputs for learned loss atenuation based on aleatoric uncertainty _again with only 10 MC sampling passes_. Furthermore, even though I believe that the paper lies within an interesting intersection of uncertainty quantification, imbalanced learning and learning under noise, the proposed method is quite expensive especially for the larger datasets. Separately running MC Dropout [4] for each and every single sample in the beginning of every iteration coupled with the expensive strategy of simultaneously training two networks produces a significant computational overhead, which disallowed me from reproducing more experiments from the work. 


# 5. References

- [1] Huang Y, Bai B, Zhao S, Bai K, Wang F. Uncertainty-aware learning against label noise on imbalanced datasets. InProceedings of the AAAI Conference on Artificial Intelligence 2022 Jun 28 (Vol. 36, No. 6, pp. 6960-6969).
- [2] Li J, Socher R, Hoi SC. Dividemix: Learning with noisy labels as semi-supervised learning. arXiv preprint arXiv:2002.07394. 2020 Feb 18.
- [3] Kendall A, Gal Y. What uncertainties do we need in bayesian deep learning for computer vision?. Advances in neural information processing systems. 2017;30.
- [4] Gal Y, Ghahramani Z. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. Ininternational conference on machine learning 2016 Jun 11 (pp. 1050-1059). PMLR.
- [5] Berthelot D, Carlini N, Goodfellow I, Papernot N, Oliver A, Raffel CA. Mixmatch: A holistic approach to semi-supervised learning. Advances in neural information processing systems. 2019;32.

# Contact
Please don't hesitate to ask any questions and correct me if I made any mistakes with the implementation!

Selim Kuzucu - selim686kuzucu@gmail.com
