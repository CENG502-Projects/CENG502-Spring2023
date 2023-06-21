# Deep Radial Embedding for Visual Sequence Learning

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

In this paper, authors aim to address the peaky behavior problem of CTC. They offer a CTC loss, called "RadialCTC"  that claims to surpass many state-of-the-art models by solving the peaky behaviour, and it was published at ECCV 2022 conference. My aim is to implement the Radial CTC architecture with the guidance of the original paper and its supplementary material, and compare the results for reproducibility.

## 1.1. Paper summary
A well-liked goal function in sequence recognition called Connectionist Temporal Classification (CTC) supervises unsegmented sequence data by repeatedly matching the sequence and its related labeling. The peaky behavior of CTC is frequently attributed to its blank class, which is also critical to the alignment process. In this paper, they propose an objective function called RadialCTC that maintains the CTC's iterative alignment mechanism while constricting sequence characteristics to a hypersphere. With a clear geometric interpretation and a more expedient alignment procedure, the learnt features of each non-blank class are dispersed on a radial arc from the center of the blank class. Additionally, RadialCTC can alter the blank class's logit to regulate the peaky behavior.

1.1.1. Contributions

-Putting out the RadialCTC, which keeps the CTC's iterative alignment mechanism while constraining sequence characteristics to a hypersphere. The characteristics of non-blank classes are dispersed around the center of the blank class in radial arcs.

-Proposing a straightforward angular perturbation term that can consistently supervise all sequence data while taking the sequence-wise angular distribution into account in order to control the peaky behavior.

-Conducting thoughtful experiments about the interaction between localization and recognition. The usefulness of RadialCTC, which delivers competitive performance on two sequence recognition applications and may also offer configurable event boundaries, is demonstrated by experimental data.

1.1.2. Important consepts

CTC is proposed to provide supervision for unsegmented sequence data, which has shown advantages in many sequence recognition tasks. A controversial characteristic of CTC is its spike phenomenon. Networks trained with CTC will conservatively predict a series of spikes.

Liu et al. [5] propose a entropy-based regularization method to penalize the peaky distribution and encourage exploration. 

Min et al. [6] propose a visual alignment constraint to enhance feature extraction before the powerful temporal module. Adding constraints on the CTC-based framework can alleviate the overfitting problem. However, the peaky behavior still exists, and it is hard to provide clear event boundaries.

Many works  try to understand the peaky behavior of CTC. Earlier speech recognition works interpret CTC as a special kind of Hidden Markov Model [7], which is trained with the Baum-Welch soft alignment algorithm, and the alignment result is updated at each iteration. Some recent works leverage this iterative fitting characteristic and extend the spiky activations to get better recognition performance. 

However, these methods change the pseudo label at each iteration manually and may break the continuity of the sequence feature.
Similar work to RadialCTC is [8], where the authors find that the peaky behavior is a property of local convergence, and the peaky behavior can be suboptimal. Different to [8], authors constrain sequence features on a hypersphere and control the peaky behavior with an angular perturbation term.

The main goal of deep feature learning is to learn discriminative feature space with proper supervision. In some fine-grained image classification tasks  an important technical route is to learn strong discriminative features by improving the conventional softmax loss.

# 2. The method and my interpretation

## 2.1. The original method


<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/frameworkPNG.PNG" width="720" >
Figure 1: The overall Framework(From the paper)


<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/seq_mnist.PNG" width="720" >
Figure 2: Illustration of the preparation of the Seq-MNIST dataset(From the paper)

To better illustrate the proposed method, they build a simulated sequence recognition dataset named Seq-MNIST using the above scheme. They interpolate with alpha=9 and add 5 frames to the beginning and the end. The Seq-MNIST has 15,000 training sequences and 2500 testing sequences, and each sequence contains 41 frames. 


<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/ffr.PNG" width="256" >
Figure 3: Frame-wise feature extractor and softmax(From the paper)

Then, they used a modified version of LeNet which is LeNet++(a deeper and wider network) for the feature extraction. LeNet takes an input x=(x1,x2,..,xT) and returns frame-wise features v=(v1, v2,..., vT). After that it goes through a fully connected layer and softmax. Fully connected layer has number of labels + 1 out put values for the "blank" token which is used in CTC. With the help of an extra ‘blank’ class, CTC defines a many-to-one mapping to align the alignment path π and its corresponding labeling l. This mapping is achieved by successively removing
the repeated labels and blanks in the path. For example, B(-aaa--aabbb-) =
B(-a-ab-) = aab. The posterior probability of the labeling can be calculated by:


<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/prob.PNG" width="256" >
Figure 4: The posterior probability of the labeling(From the paper)

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/dist_freatures.PNG" width="720" >
Figure 5: Visualization of (a) the distribution of frame-wise features and (b) an example of transition trajectory in the test set of Seq-MNIST.(From the paper)

The frame-wise features v visualized above after training the model with CTC. we can observe that: 

(1) after training with CTC, the framewise features are separable among non-blank classes, but the decision boundary between non-blank classes and the blank class is pretty complicated,

(2) over half of the features are classified to the blank class, which is corresponding to the peaky behavior of CTC, and features of the blank class have a large intraclass variance, and 

(3) although some transition frames are pretty similar to the keyframe, they are classified to the blank class.

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/ctc_loss.PNG" width="720" >
Figure 6: The Design of Loss Function for Sequence Recognition(From the paper)

Normalization: the learned features with supervision from CTC have large intra-class variance, especially on the blank class, and the decision boundary between the blank class and non-blank classes is not clear. The vanilla CTC takes the inner distance between features and weights as input and provides little constraint on the alignment process. To learn a more separable feature space, we normalize both the features and weights and constrain the learned features on a hypersphere, which has been proven a practical approach in face recognition. As shown above, After constraining all features on the hypersphere, the search space of the alignment process is reduced considerably, and features are distributed along several disjoint paths from the center of the
blank class.

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/dist_center.PNG" width="720" >
Figure 7: The distribution of frame-wise features with (a) normalization and (b) normalization, angle and center constraints in the test set of Seq-MNIST.(From the paper)

Angle regularization: any frames between two non-blank keyframes can be classified into the blank class. Therefore, any transition trajectory between two non-blank keyframes will go through the decision region of the blank class. To enhance the discriminative and generalization ability of the model, they propose an angle regularization term to minimize the distance between Wb^T.Wnb and a given value cos(β).

Center regularization: They only apply center regularization on keyframes set KF(v), which is implemented by first estimating the alignment path π = arg maxπ p(π|v,l; θ) with the maximal probability and then minimizing the distance between features of keyframes in π and their corresponding classes.

The conservative supervision from CTC only classified a small ratio of frames to non-blank classes, but we can observe from Fig. 4(b) that features of the blank class are also clustered into several groups, which are distributed along the disjoint arcs to the centers of non-blank classes. This observation raises two questions: 

(1) can we obtain accurate localization information from CTC, and

(2) what is the relationship between the recognition and localization abilities of the model trained with CTC?

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/vanilla_fba.PNG" width="400" >
Figure 8: Vanilla Forward-Backward Algorithm of CTC(From the paper)

The role of the angular margin: Adopting an angular/cosine-margin based constraint is popular in deep feature learning, which can make learned features more discriminative by adding a margin term in softmax loss. 


The frame-wise gradient of CTC has the same formulation as the CrossEntropy (CE) loss, and the optimization of CTC is equivalent to iterative fitting [22]. However, the pseudo label ytk is calculated by considering probabilities of all feasible paths. In other words, changes in the logit also influence the probabilities of relevant paths, and adding a margin term on one frame also changes the pseudo labels of its neighboring frames.

RadialCTC:

Adopting an angular perturbation term can change the pseudo label, which provides a valuable tool to control the peaky behavior. 

They try to control the peaky behavior of CTC by perturbing blank logits of all frames with a sequence-dependent term. As the decision boundaries between the blank class and non-blank classes are similar, they look into the decision criteria of softmax in the binary case.
After normalizing both features and weights and ignoring the bias term, the decision boundary between the blank class b and a non-blank class nb is θ1 = θ2.

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/ang_per_grp.PNG" width="720" >
Figure 9: The geometric interpretation of the angluar perturbation process(From the paper)

To control the peaky behavior flexibly, they propose a radial constraint that is implemented by adding an angular perturbation term m(η, θ,l) between v and Wb and adopt the pseudo label of the perturbed logits to provide supervision for the original logits. given visual features v = (v1, · · · , vT ) and its corresponding labeling l = (l1, · · · , lU ), they find the frame vτ which has the kth (k = U + 1 + ⌊(T − U) ∗ η⌋) largest angular difference between the blank class and the class with the highest probability that has appeared in the labeling. 

This process can be formulated as:

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/m.PNG" width="256"  >
Figure 10: the angluar perturbation process(From the paper)

calculate the perturbed prediction z as below:

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/z.PNG" width="720">
Figure 11: Calculatation the perturbed prediction z(From the paper)

and The loss can be calculated as below:

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/radial_fba.PNG" width="720"  >
Figure 12:  Forward-Backward Algorithm of CTC(From The paper)

As the proposed method adjust the blank ratio based on ‘radial’ feature distribution, we named this method RadialCTC. 

The entire process can be formulated as:

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/radial_loss.PNG" width="720"  >
Figure 13: The Design of Loss Function for Sequence Recognition with RadialCTC(From the paper)


## 2.2. Our interpretation

I constructed the Seq-MNIST, interpolate with alpha=9 and add 5 frames to the beginning and the end. The Seq-MNIST has 15,000 training sequences and 2500 testing sequences, and each sequence contains 41 frames. The example can be seen below. The interpolation was not clear so I used range(0.1, 1, 0.1) for the intervals and [0.1, 0.3, 0.5, 0.7, 0.9] for the ends. Also, LeNet++ uses 32x32 input data so I reshape the image.

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/my_mnist_seq.PNG" width="720"  >
Figure 14: An example of Seq-MNIST dataset(From my results)

Then, I constructed LeNets++ from Wen, Y., Zhang, K., Li, Z., Qiao, Y.: A discriminative feature learning approach for deep face recognition. In: Proceedings of the European Conference on Computer Vision. pp. 499–515. Springer (2016) with below parameters. Some of the convolution layers are followed by max pooling. (5, 32)/1,2 × 2 denotes 2 cascaded convolution layers with 32 filters of size 5 × 5, where the stride and padding are 1 and 2 respectively. 2/2,0 denotes the max-pooling layers with grid of 2×2, where the stride and padding are 2 and 0 respectively. In LeNets++, we use the Parametric Rectified Linear Unit (PReLU) as the nonlinear unit. Where to use PReLu was not clear so I used it after every conv and fc layers(in=2048,out=3).

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/lenetpp.PNG" width="720" >
Figure 15: Architecture of LeNets++(From another paper)

For the output layer, I used 3 as our paper for 3D feature map. the output goes through a fc(in=3,out=11) and softmax. I return feature v for plotting. 

For the CTC implementation, follow the explanation from [4] which is based on [2]

I get a vector, which has the length of vocabulary, for each time step of LeNets++ computation. The softmax function is applied to it to get a vector of probabilities. I exclude all rows that do not include tokens from the target sequence and then rearrange the tokens to form the output sequence. This is done during training only. If a token occur multiple times in the label, we repeat the rows for similar tokens in their appropriate location. This becomes our probability matrix,  y(s,t) like the "door" example below.

<img src="https://github.com/CENG502-Projects/CENG502-Spring2023/blob/main/Akpinar/images/door.PNG" width="720"  >
Figure 15: An example(From [4])

Forward Algorithm for computing  α(s,t):

First, let’s create a matrix of zeros of same shape as our probability  matrix, y(s,t) to store our α values. The forward algorithm is given by;

Initialize:

α-mat = zeros_like(y-mat)

α(0,0)=y(0,0),α(1,0)=y(1,0)

α(s,0)=0 for s>1

Iterate forward:

for t = 1 to T-1:

for s = 0 to S:

α(s,t)=(α(s,t−1)+α(s−1,t−1))y(s,t) if seq(s)=“ε” or seq(s) = seq(s-2)

α(s,t)=(α(s,t−1)+α(s−1,t−1)+α(s−2,t−2))y(s,t) otherwise

Note that α(s,t)=0 for all s<S−2(T−t)−1 which corresponds to the unconnected boxes in the top-right. These variables correspond to states for which there are not enough time-steps left to complete the sequence.

Backward algorithm for computing β(s,t):

Let’s also create a matrix of zeros of same shape as our probability matrix, y(s,t) to store our β values.

Initialize:

β(S−1,T−1)=1, β(S−2,T−1)=1,

β(s,T−1)=0 for s<S−2

Iterate backward:

for t = T-2 to 0:
for s = S-1 to 0:
β(s,t)=β(s,t+1)y(s,t)+β(s+1,t+1)y(s+1,t) if seq(s)=“ε” or seq(s)=seq(s+2)

β(s,t)=β(s,t+1)y(s,t)+β(s+1,t+1)y(s+1,t)+β(s+2,t+2))y(s+2,t) otherwise

Similarly, β(s,t)=0 for all s>2t which corresponds to the unconnected boxes in the bottom-left.

CTC Loss calculation for each timestep:

γ(s,t)=α(s,t)β(s,t)

P(seqt,t)=∑s=0,S γ(s,t)/y(s,t)

loss=-∑t=0,T-1  ln(P(seqt,t))


RadialCTC:

Then, loss as -∑t=0,T-1 ∑s=0,S ln(γ(s,t)/y(s,t))*z(s,t).

But for the calculation of z the paper mentions an s value which is different then the state s above. By checking [9], we can see that it is a scaling value whose value is not given in our paper so I take it as 1.
I added angular and center regularization. 

In the CTC loss and radial CTC loss calculation at the sum of ln gives "RuntimeError: Function 'LogBackward0' returned nan values in its 0th output." in backward propagation. I could not find a solution for that unfortunately so I could not make the experiments and get the results () you can see the error on the training part of the code but I implement the training and validation steps.

I also download and prepare the Phonex14 data set for multisinger and independentsinger. I apply the transformations that are specified in the paper. 
For the model, I use ResNet18 with the last layer of BiLSTM with 2x512 hidden layers. I wrote the training and testing codes of this model with RadialCTC but due to the complication in backwardprob, I could not train the model and get the results



# 3. Experiments and results

## 3.1. Experimental setup

I used the LeNet++ setup but I change some part that I mention above to be able to use for our dataset.

Datasets:

Seq-MNIST. Seq-MNIST maintains the distribution balance from MNIST. They also simulate an unbalanced training set by sampling images at the rate of 0.1 for classes 0 to 4 and remaining unchanged for others.

Phoenix14. As a popular CSLR dataset, Phoenix14 [17] contains about 12.5 hours of video data collected from weather forecast broadcast and is divided into three parts: 5,672 sentence for training, 540 for development (Dev), and 629 for testing (Test). It also provides a signer-independent setting where data of 8 signers are chosen for training and leave out data of signer05 for evaluation.

Scene Text Recognition Datasets. Following the standard experimental setting, we use the synthetic Synth90k as training data and test our methods on four real-world benchmarks (ICDAR-2003 (IC03), ICDAR-2013 (IC13), IIIT5k-word(IIIT5k) and Street View Text(SVT)) without fine-tuning.

For Seq-MNIST and scene text recognition datasets, we use sequence accuracy as the evaluation metric. Word error rate (WER) is adopted as the evaluation metric of CSLR as previous work does. We adopt the mean Average Precious (mAP) to evaluate the localization performance.

For Seq-MNIST implementation:
Adadelta optimizer is used for 30-epoch training with an initial learning rate of 1.0. Each iteration processes 64 sequences and no augmentation technique is applied during training. For the hyperparameter choice, they adopt λ1 = 1.0, β = 0.2π and λ2 = 1.0 as the default setting.

For Phoenix14 implementation:

They select ResNet18 as the FFE. The glosswise temporal layer and two BiLSTM layers with 2×512 dimensional hidden states are adopted for temporal modeling. They adopt Synchronized Cross-GPU Batch Normalization (syncBN) to gather statistics from all devices, which can accelerate the training process. Therefore, they shorten the training time and
train all models for 40 epochs with a batch size of 2. Adam optimizer is used with an initial learning rate of 1e-4, which decays by a factor of 5 after epochs 25 and 35. The training set is augmented with random crop (224x224), horizontal flip (50%), and random temporal scaling (±20%). They replace the extra CTC (visual enhancement loss in ) with the proposed RadialCTC to show its effectiveness as intermediate supervision without using the visual alignment loss for simplicity. For the hyperparameter choice, we adopt λ1 = 1.0, β = 0.2π and λ2 = 0.1 as the default setting.

For  Scene Text Recognition implementation:

They adopt CRNN as their baseline model. CRNN is optimized by CTC loss and has three components, including the convolution module, the recurrent module, and the decoder. The convolution
module converts resized image (1 × 32 × 100) to a feature sequence of size 512 × 1 × 26, where 512 is the dimension of output features. The recurrent layer has two BiLSTM layers with 2 × 256 hidden states and two fully-connected layers. It predicts a probability distribution among 37 predefined classes for each frame in the feature sequence. After that, the decoder converts the predictions into a label sequence. They train the model for 30 epochs with a batch size of 512 under the supervision of RadialCTC loss. Adam optimizer is used with β1 set to 0.5 and an initial learning rate of 1e-3 decaying with a rate of 0.2 after epoch 10 and epoch 20. For the hyperparameter choice, we adopt λ1 = 1.0, β = 0.2π and λ2 = 0.1 as the default setting.


## 3.2. Running the code


Our main file is `radialCTC.ipynb` where we declare step-by-step code cells to run our code. 

## 3.3. Results

I could not finish the results part due to the complication backpropagation

# 4. Conclusion

Connectionist Temporal Classification (CTC), a popular goal function in sequence recognition, manages unsegmented sequence data by continually matching the sequence and its associated labeling. The blank class of CTC, which is also essential to the alignment process, is sometimes blamed for the peaky behavior of the latter. In this study, they put forth the RadialCTC objective function, which preserves the iterative alignment process of the CTC while limiting sequence features to a hypersphere. The learned features of each non-blank class are scattered on a radial arc from the center of the blank class with a clear geometric interpretation and a quicker alignment method. RadialCTC may also change the logit of the blank class to control the peaky behavior.
I could not compare the results due to having difficulties as I mention above.

# 5. References

[1] Y. Min et al., “Deep radial embedding for visual sequence learning,” Lecture Notes in Computer Science, pp. 240–256, 2022. doi:10.1007/978-3-031-20068-7_14 

[2] Graves, A., Fern´andez, S., Gomez, F., Schmidhuber, J.: Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In: Proceedings of International Conference on Machine Learning. pp. 369–376 (2006)

[3] Wen, Y., Zhang, K., Li, Z., Qiao, Y.: A discriminative feature learning approach for deep face recognition. In: Proceedings of the European Conference on Computer Vision. pp. 499–515. Springer (2016)

[4] S. Ogun, “Breaking down the CTC loss,” Sitewide ATOM, https://ogunlao.github.io/blog/2020/07/17/breaking-down-ctc-loss.html (accessed Jun. 19, 2023). 

[5] Liu, H., Jin, S., Zhang, C.: Connectionist temporal classification with maximum entropy regularization. Advances in Neural Information Processing Systems 31,
831–841 (2018)

[6] Min, Y., Hao, A., Chai, X., Chen, X.: Visual alignment constraint for continuous sign language recognition. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 11542–11551 (2021)

[7] Rabiner, L.R.: A tutorial on hidden markov models and selected applications in speech recognition. Proceedings of the IEEE 77(2), 257–286 (1989)

[8] Zeyer, A., Schl¨uter, R., Ney, H.: Why does ctc result in peaky behavior? arXiv preprint arXiv:2105.14849 (2021)

[9]Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., Liu, W.: Cosface: Large margin cosine loss for deep face recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. pp. 5265–5274 (2018)


# Contact

Gülfem Akpınar, gulfem.akpinar@metu.edu.tr
