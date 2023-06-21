# DECORE: Deep Compression with Reinforcement Learning


# 1. Introduction

<!--- @TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility). ---> 
This is paper is published in CVPR 2022. My goal is to re-implement the proposed model and study its capability and limitations 

## 1.1. Paper summary

<!--- @TODO: Summarize the paper, the method & its contributions in relation with the existing literature.--->
The paper proposes a model 'DECORE' which aims to compress a deep model architecture by dropping channels or linear layers of low importance, the proposed model is based on a multi-agent reinforcement framework. Previous works targeted the problem of compressing AI models, such as pruning weights based on the network statistcs and learning channel's importance subject to compression constraints, reinforcement learning has been also used to find optimized model's
architecture, however, all these methods suffer from high complexity due to iterative search and fine-tuning. This work work proposed a model in which the 
architecture search and fine-tuning are independent which helps in speeding up the compression process. 

# 2. The method and my interpretation

## 2.1. The original method

Given a Deep Model, the 'DECORE' framework assigns an agent to each layer of the model, the agent holds a vector $s_i \in \mathbb{R}^{C_i}$, where ${C_i}$ is the number of channels within layer $l_i$ and $s_i$ is the state representation of the layer. The state representation is mapped to actions according to the
following policy

$$ p_j = \frac{1}{1+e^{-w_j}}$$

$$\begin{equation}
\pi_{j} = 
  \begin{cases}
  1 & \text{with } p_j \\
  0 & \text{with } 1 - p_j
  \end{cases}
\end{equation}$$

$$a_i = \{\pi_0,\pi_1,...,\pi_{C_i}\}$$

where $w_j$ is the importance weight of channel $j$ and $w_j \in s_i$, $\pi_{j}$ is the policy function which samples actions $\{1,0\}$ according to Bernoulli process. $p_j$ is the probability distribution of $\pi_j$. The model computes a compression reward according to: 
$$R_{i,C} = \sum_{j=1}^{C_i} 1 - a_{i,j}$$
the model incurs a penalty $\lambda$ for incorrect predictions as:

$$\begin{equation}
R_{acc} = 
  \begin{cases}
  1 & \text{if } y_{Pred} == y_{True} \\
  -\lambda & \text{otherwise }
  \end{cases}
\end{equation}$$


The cost function is:
$$max_{w} J(w) = \sum_{i=1}^{L} max_{w} E_{\tau \sim \pi_{w}(\tau_i)}[R_i]$$
and REINFORCE policy gradient algorithm is used to learn policy parameters $w$.


## 2.2. Our interpretation 

<!--- @TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.---> 
In the REINFORCE policy gradient algorithm, the infer of $\pi_w(a_i | s_i)$ was not clear, in this implementation is it interpreted as:
$$\pi_w(a_i | s_i) = P(a_i|s_i) = \prod_{j=1}^{C_i} P(a_{i,j}|s_{i,j})$$

# 3. Experiments and results

## 3.1. Experimental setup

The original work tests their framework by compressing three different models, namely, VGG16, DenseNet and ResNet. They used CIFAR-10 and IMAGENET as datasets.
During training, they used ADAM optimizer with learning rate 0.01 and btch size of 256, initial weights $w_j$ are initialized as 9.6. They  trained the model for 300 number of epochs with 260 for learning weights $w$ and 40 for fine-tuning the model after dropping channels with probabilities less than 50% on CIFAR-10.

## 3.2. Running the code

<!--- @TODO: Explain your code & directory structure and how other people can run it.---> 
Follow the structure in main.ipynb as a demo.

## 3.3. Results

<!--- @TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.--->
We could not reproduce the results in the paper because of unclarity of $\pi_w(a_i | s_i)$, during training we noticed that the favoured actions are to activate all channels hence no compression is osbserved, we expect that this is due to initialization of state representation $s_i$ as 9.6 which gives high probability to all actions to be 1 and thus gradient of the cost function will be almost zero.

# 4. Conclusion

<!---@TODO: Discuss the paper in relation to the results in the paper and your results.--->
The paper proposes a promising work, however, due to lack of some details, the model could not be verified.

# 5. References

<!---@TODO: Provide your references here.--->
DECORE: Deep Compression with Reinforcement Learning, 2022, Manoj Alwani, Yang Wang, Vashisht Madhavan

# Contact

<!---@TODO: Provide your names & email addresses and any other info with which people can contact you.--->
Mahmoud Alasmar (alasmar.mahmoud@metu.edu.tr)
