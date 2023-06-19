# Deep Radial Embedding for Visual Sequence Learning

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

In this paper, authors aim to address the peaky behavior problem of CTC. They offer a CTC loss, called "RadialCTC"  that claims to surpass many state-of-the-art models by solving the peaky behaviour, and it was published at ECCV 2022 conference. My aim is to implement the Radial CTC architecture with the guidance of the original paper and its supplementary material, and compare the results for reproducibility.

## 1.1. Paper summary
A well-liked goal function in sequence recognition called Connectionist Temporal Classification (CTC) supervises unsegmented sequence data by repeatedly matching the sequence and its related labeling. The peaky behavior of CTC is frequently attributed to its blank class, which is also critical to the alignment process. In this paper, they propose an objective function called RadialCTC that maintains the CTC's iterative alignment mechanism while constricting sequence characteristics to a hypersphere. With a clear geometric interpretation and a more expedient alignment procedure, the learnt features of each non-blank class are dispersed on a radial arc from the center of the blank class. Additionally, RadialCTC can alter the blank class's logit to regulate the peaky behavior.

##1.1.1. Contributions

-Putting out the RadialCTC, which keeps the CTC's iterative alignment mechanism while constraining sequence characteristics to a hypersphere. The characteristics of non-blank classes are dispersed around the center of the blank class in radial arcs.
-Proposing a straightforward angular perturbation term that can consistently supervise all sequence data while taking the sequence-wise angular distribution into account in order to control the peaky behavior.
-Conducting thoughtful experiments about the interaction between localization and recognition. The usefulness of RadialCTC, which delivers competitive performance on two sequence recognition applications and may also offer configurable event boundaries, is demonstrated by experimental data.

##1.1.2. Important consepts
CTC is proposed to provide supervision for unsegmented sequence data, which has shown advantages in many sequence recognition tasks. A controversial characteristic of CTC is its spike phenomenon. Networks trained with CTC will conservatively predict a series of spikes.
Liu et al. [23] propose a entropy-based regularization method to penalize the peaky distribution and encourage exploration. 
Min et al. [29] propose a visual alignment constraint to enhance feature extraction before the powerful temporal module. Adding constraints on the CTC-based framework can alleviate the overfitting problem. However, the peaky behavior still exists, and it is hard to provide clear event boundaries.
Many works  try to understand the peaky behavior of CTC. Earlier speech recognition works interpret CTC as a special kind of Hidden Markov Model [33], which is trained with the Baum-Welch soft alignment algorithm, and the alignment result is updated at each iteration. Some recent works leverage this iterative fitting characteristic and extend the spiky activations to get better recognition performance. 
However, these methods change the pseudo label at each iteration manually and may break the continuity of the sequence feature.
Similar work to ours is [44], where the authors find that the peaky behavior is a property of local convergence, and the peaky behavior can be suboptimal. Different to [44], we constrain sequence features on a hypersphere and control the peaky behavior with an angular perturbation term.

The main goal of deep feature learning is to learn discriminative feature space with proper supervision. In some fine-grained image classification tasks  an important technical route is to learn strong discriminative features by improving the conventional softmax loss.

# 2. The method and my interpretation

## 2.1. The original method

@TODO: Explain the original method.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

@TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
