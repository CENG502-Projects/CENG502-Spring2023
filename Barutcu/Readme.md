# Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

This project aims to reproduce the results of the paper titled "Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints," published in ICML, 2022.

The main goal of the paper is to present an efficient streaming algorithm for solving the problem of maximizing a monotone k-submodular function subject to cardinality constraints. The problem involves selecting a subset of elements from a larger set to maximize a specific objective function while adhering to constraints on the size or cardinality of the selected subset.

The authors propose a novel streaming algorithm that processes the elements in a streaming fashion, making efficient use of limited memory resources. The algorithm provides near-optimal solutions and guarantees a constant-factor approximation to the optimal solution.

In this project, I have reproduced the key results and findings of the paper, implemented the proposed algorithm, and validated its performance on various datasets. The code and experimental results can be found in this GitHub repository.

## 1.1. Paper summary

The paper addresses the problem of maximizing a monotone k-submodular function subject to cardinality constraints. The authors propose a novel streaming algorithm that efficiently processes elements in a streaming fashion, providing near-optimal solutions and a constant-factor approximation guarantee. The algorithm is designed to work with limited memory resources, making it suitable for large-scale datasets.

The paper begins by introducing the problem formulation and its significance in various applications, such as influence maximization, summarization, and recommendation systems. It then presents the streaming algorithm that tackles the problem by processing elements one at a time and maintaining a compact summary of the encountered elements. The algorithm's design leverages the submodularity property to ensure efficiency and approximation guarantees.

The proposed streaming algorithm in the paper makes several key contributions to the existing literature:

1. **Efficient Streaming Approach:** The algorithm processes elements in a streaming fashion, making it suitable for scenarios with limited memory resources or large-scale datasets. It efficiently maintains a compact summary of the encountered elements, allowing for near-optimal solution with approximation guarentee as budget constraint approaches to infinity which is the case with real time applications.

2. **Near-Optimal Solutions:** The algorithm provides near-optimal solutions for the problem of maximizing a monotone k-submodular function subject to cardinality constraints. It achieves constant-factor approximation guarantees, ensuring high-quality solutions even with limited computational resources.

3. **Theoretical Analysis:** The paper provides a theoretical analysis of the algorithm's performance, including bounds on approximation guarantees and computational complexity. It establishes the algorithm's effectiveness and its applicability to real-world problems.

4. **Free Disposal of Algorithm Parameters:** Paper offers an algorithm which does not store results of the previous iterations, so offers efficient computation in term of memory usage and asymptotic complexity, unlike the method in the literature that uses greedy approach [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf) with offline computation other than streaming approach. 

The problem addressed in the paper can be mathematically formulated as follows:

![primal_dual](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/32d89eb0-5f48-4b41-a352-2ed5b7c3d871)
*Figure 1: Primal Dual Formulation of LP Programming (Borrowed from the paper)* 

The formulation captures the objective of maximizing a monotone k-submodular function subject to constraints on the size of the selected subset. The paper discusses the details of the formulation and provides insights into its properties and potential algorithmic approaches for solving it. 
- $x_{e,i}$ ∈ {0, 1} that indicates whether element e is assigned label i. 
- $y_A$ ∈ {0, 1} that indicates whether A is the selected k-tuple. 
- The first set of constraints enforce that e receives label i if and only if e ∈ $A_i$ for the selected labeling. 
- The second set of constraints enforce that we select exactly one k-tuple. 
- The third set of constraints enforce that each element receives at most one label. 
- The fourth set of constraints enforce the size constraints for the labels. 
By relaxing the integrality constraints $x_{e,i}$, $Y_A$ ∈ {0, 1} to $x_{e,i}$ , $Y_A$ ∈ (0, 1), we obtain the LP relaxation for the problem

# 2. The method and my interpretation

### 2.1. The original method

This section describes the original method presented in the paper "Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints." It covers the definition of k-submodularity, the algorithm proposed by the authors, the concept of marginal gain, and the approximation of the submodular function.

### 2.1.1 K-Submodularity Definition

In the paper, the concept of submodularity plays a fundamental role. A function $f: (2+1)^V -> R$, where V is a finite ground set, is said to be k-submodular if it satisfies the diminishing returns property. Specifically, for any sets A, B subset of V, where |A| <= |B|, and any element e not in B, the following inequality holds:

$f(A ∪ {e}) - f(A) \geq f(B ∪ {e}) - f(B)$

This property captures the idea that adding an element to a smaller set yields a higher marginal gain compared to adding it to a larger set. It is a key characteristic of many real-world optimization problems. K-submodularity is the generalization of the submodular definition into k different sets. A function is k-submodular if and only if:

Pairwise Monotone

![pairwise_monotone](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/1df9c18b-ab82-48ae-a47a-647925a18e48) and,

 Orthant Submodular

![orthant_submodular](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/40e42f3c-5a2d-4d9b-af35-e0d67fd8ed81)

 for all $X, Y ∈ (k+1)^V$ such that $X \preccurlyeq Y$, $e \notin supp(Y)$ and $i ∈ (k)$

### 2.1.2 Marginal Gain Definition

The concept of marginal gain plays a crucial role in the algorithm. The marginal gain of adding an element v to a current solution set S is defined as the increase in the objective function value:

$\Delta_{e,i} f(v) = f(S_1 ∪ S_2 ... S_i ∪ {e}... ∪ S_k) - f(S_1 ∪ S_2 ... S_i ∪ S_k)$

It quantifies the utility or contribution of an element towards improving the objective function value when added to the current solution set.

### 2.1.3 Therotical Approximation Bound of Optimal Solution

This section provides an overview of the approximation guarantees for maximizing submodular functions subject to cardinality constraints. The table below summarizes some of the notable approximation guarantee algorithms in the literature.

![comp_2](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/09b00706-ebed-40de-8fe0-f551776feb6e)
*Figure 2 Comparison of Algorithms (Borrowed from the paper)* 

Proof of the approximation bound can be found in [paper](https://proceedings.mlr.press/v162/ene22a/ene22a.pdf)

### 2.1.4 Proposed Algorithm

The paper proposes a novel streaming algorithm for maximizing a monotone k-submodular function subject to cardinality constraints. The algorithm processes the elements in a streaming fashion, making efficient use of limited memory resources.

Algorithm details are as follows:

![primal_dual_algorithm](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/2a97554e-5f37-4d36-a44b-feb3b3bbb458)

*Figure 3 Pseudocode of Algorithm (Borrowed from the paper)* 

Method uses primal-dual variables to increase the treshold of accepting an item as the marjinal gain decreases over time.

## 2.2. Our interpretation 

Although the algorithm is well defined in the paper, some of the aspects are problem dependent. Those are:

**1. Definition of Function:** Problem related function should be defined in order to reflect of the benefits of any combination of elements. In the paper, there are two different experiment settings which both have different evalution functions. For influence maximization problem, function evaluation is very costly even on a small subset of elements. To overcome this, benefit function is approximated according to method proposed by [Borgs et al, 2014](https://arxiv.org/pdf/1212.0884.pdf)

**2. Marginal Benefit of Adding an element to Existing Set:** Marginal benefit determines the whether incoming element should be added to existing element set S. It is very critical to design marginal benefit function to reflect the gain of the element. Function is problem dependent and should be designed carefully. For influence maximization problem in paper, there are no clear statements as to how this function is design. I referred another research [Youze et al, 2015](https://dl.acm.org/doi/10.1145/2723372.2723734) to determine the marginal benefit as the difference of the  values sets in approximated function R that are covered by a node
set S.

**3. Dealing with Missing Values in Dataset:** In sensor placement problem, There are missing rows related to humidity, light condition and temperature columns. Since strategy of dealing with missing values are not clearly stated in paper, I chose my own method that I eliminated all missing rows in the dataset which is about 400.000 rows of data in total dataset.

# 3. Experiments and results

## 3.1. Experimental setup

There are two experiments in paper: Influence maximization and sensor placement problem. These are breifly discussed below:

**1. Influence Maximization with *k* topics:** In this experiment, we aim to maximize influence in a social network with multiple topics of interest. The objective is to identify a set of influential nodes that can maximize the spread of information or influence across the network for each topic. The experiment involves the following steps:

1. Constructing directed network graph from undirected edges in Facebook SNAP dataset.
2. Approximating the influence function proposed by [Borgs et al, 2014](https://arxiv.org/pdf/1212.0884.pdf) with approximation guarentee lowerbound number of simulations $O(n k^2 log(n) log(k))$ where k is number of topics and n is number of nodes in grapgh using k-Independent Cascade Process proposed by [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf)
3. Applying the proposed influence maximization algorithm, leveraging the k-submodular framework, to identify the most influential nodes for each topic.
4. Comparing the performance of the algorithm with k-greedy algorithm proposed by [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf).
5. Evaluating the values of the methods ranging between different budget constraints.

Although the therotical number of simulations required stated at [Borgs et al, 2014](https://arxiv.org/pdf/1212.0884.pdf), author of the paper did not state anything about whether it is used as number of simulations to approximate the function. In addition, cost of evaluating the function with these number of simulations is very costly, so I used this number R = 100 in order to get a frame about comparision of the algorithms with different budget constraints. Simulation process explained in following figure:

![build_hypergraph](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/811e5f3e-b03e-4127-9ca1-9b132d982dd2)
*Figure 4 Simulation Process of Approximating Influence Function (Borrowed from the [paper](https://arxiv.org/pdf/1212.0884.pdf))*


Process uses Independent Cascade Process which can be seen in figure below:

![icp](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/5e21eaee-3ea4-4cdb-9254-5e6eb9846a9a)
*Figure 5 Independent Cascade Process (Borrowed from the [notes](http://people.seas.harvard.edu/~yaron/AM221-S16/index.html))*


**2 Sensor Placement with *k* Measurements:**

In this experiment, we focus on the problem of sensor placement in a given environment. The goal is to determine the optimal locations to deploy sensors of different types.There are k types of sensors and a set V of n sensor locations. We want to place at most one sensor in each possible location and at most $B_i$ sensors of type *i* across all locations. Problem can be redefined as maximizing total entropy in all sensor locations so that maximum information is gained from sensors.The experiment involves the following steps:

1. Clean Intel Lab dataset as described above
2. Discritized temperature, humidity and light values from each reading into bins of 2 degrees Celsius each, 5 points each and 100 luxes each, respectively.
3. Applying the proposed sensor placement algorithm, leveraging the k-submodular maximization framework, to select the optimal sensor locations with different types.
4. Comparing the performance of the algorithm with k-greedy algorithm proposed by [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf).
5. Evaluating the values of the methods ranging between different budget constraints.

## 3.2. Running the code
```
Directory structure
├── datasets/
│   ├── Intellab.txt
│   └── facebook_dataset.txt
├── Images/
│   ├── approximation_table_1.jpg
│   ├── build_hypergraph.jpg
│   ├── greedy_algorithm.jpg
│   ├── independent_cascade_process.jpg
│   ├── lp_formulation.jpg
│   ├── node_selection.jpg
│   ├── primal_dual_algorithm.jpg
│   └── sensor_placement.jpg
├── influence_maximization.ipynb
├── sensor_placement.ipynb
├── predict_graph.ipynb
├── q_learning_sensor.ipynb
└── requirements.txt
```

- Files in datasets folder contains facebook and Intel Lab datasets in txt format
- Images folder contains copy right images used in this readme file
- requirements.txt contains the list of required Python packages.
- influence_maximization.ipynb file contains the influence maximization experiment with data preprocessing functions and k-greedy algorithm implementation
- sensor_placement.ipynb file contains the sensor placement experiment with data preprocessing functions k-greedy algorithm implementation
- predict_graph.ipynb file contains gnn based model for link prediction task
- q_learning_sensor.ipynb file contains the deep reinforcement learning experiment on sensor placement problem with data preprocessing functions

### 3.2.1 Create Environment
- conda create -n (env_name)
- conda activate env_name

### 3.2.2 Install Dependencies
pip install -r requirements.txt

### 3.2.3 Running the Experiments
Cells in jupyter notebooks can sequencially be run to reproduce the results.

## 3.3. Results

### 3.3.1 Experiment Results:
Experiment settings are as follows:

**1. Influence Maximization with *k* Different Topics:**

*Table 1: Parameters for Influence Maximization Problem*
Setup Component | The authors | Us
------------ | ------------- | -------------
B | 1-30 | 1-10
K | 3,10 | 3 
D | $B*(2^{1/B} -1)$ | $B*(2^{1/B} -1)$
R | - | 100
C | 0.5D | 0.5D

Where B is the budget for all types, R represents the number of simulations processes to approximate influence function f, K is number of different topics and C, D are algorithm parameters

Experiment results are as follows:

![Picture1](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/9cb41674-267f-45bd-89d3-69f845f93edf)

*Figure 6: Author's Results on Influence Maximization*


![inf_max_result](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/ff7834db-204c-4920-992c-e626c086ba5a)

*Figure 7: My Results on Influence Maximization*

As can be seen on figures, author's results are more smooth and diminishing return of value is more clear. This is due to the choise of number of simulations to approximate the benefit function. Although there is therotical lower bound for R, I could not simulate such times because of computational resource restrictions. Also, author was able to generate the results of different budgets ranging from 1-30 whereas I could reproduce only between 1-10 because as budget constraint increases, computation time also increases linearly. Due to the limited resources, I chose B between 1-10 to see the trend in the value of the function.

**2. Sensor Placement with *k* Different Measurements:**

*Table 2: Parameters for Sensor Placement Problem*
Setup Component | The authors | Us
------------ | ------------- | -------------
B | 1-30 | 1-30
K | 3 | 3 
D |  $B*(2^{1/B} -1)$ |  $B*(2^{1/B} -1)$
C | 0.5D | 0.5D

Since the computational resource is enough the reproduce the problem, I used same parameters with author. In this experiment I get similar results as shown below:


![Picture2](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/c19606b7-b2a6-45cb-a5bb-d55e0e7c08df)

*Figure 8: Author's Results on Sensor Placement Problem*


![Sensor_plc](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/cf93d1b2-b123-44e1-b543-3ecceae38896)

*Figure 9: My Results on Sensor Placement Problem*

As can be seen on the figures, mine results approaches the greedy solution quicker than the author's. This might be due to strategy to dealing with missing columns in data and order of the arrival of the items.

### 3.3.2 Additional Studies:

**1. Link Prediction on Social Network:** 

Simulation process and function approximation are intensive in terms of computational time. When a new person is added to network, influence of that person should be estimated to reflect the effect of gain in objective function. Considering the real time systems where new incomers constantly joins the network and some users leaves the platform, simulation process should be restarted over again to come up with a true form of appoximating function. Also, simulation process has sthocastic nature, effect of the new nodes might not be directly observable in function. Since the process is already costly and there are constant changes in the network, catching the on time approximation function is impractiable. In order to achive real-time applicable system, link prediction of the new incomers in aproximated function can be used as a solution to reflect the changes in the network.

Graph attention networks [Velickovic et al, 2018](https://arxiv.org/pdf/1710.10903.pdf) have shown good performance on various of the tasks, including node prediction, edge attribute prediction, link prediction etc. In this experiment I designed gnn based model to train a network on approximated function so that new nodes in the network can be predicted whether it can be influenced on existing nodes in function without simulating the process from scratch. Experiment parameters are shown in table below:

*Table 3: Parameters of GNN for Link Prediction*
Setup Component | Parameters 
------------ | ------------- 
Embedding Dimension | 64 
Number of Heads | 4 
Number of GNN Layers | 2 
Dropout | 0.6 
Epoch | 100
Activation Function | Elu
Loss Function | BCE 
Learning Rate | 0.01 

I used standard architecture proposed by [Velickovic et al, 2018](https://arxiv.org/pdf/1710.10903.pdf) for transductive tasks. Train-test Ratio setted as 0.9-0.1. Aim of the model is to predict a link between existing nodes to new ones.

I could not finish the training because of memory constraint in GPU. With more computational power, experiment can be finished and results can be examined on my ready-to-use script named predict_grapg.ipynb. Experiment details can be seen in the script as well.

Fine tune might be required on parameters especially on embedding Dimension, number of heads and learning rate. These are the next level task to advance the proposed method to make applicable in realistic scenarious. In addition, if node level or edge level features exist in the model, predictions accuracy might be effected in positive way.

**2. Reinforcement Learning Approach on Sensor Placement Problem:** 

In addition to the original experiment, we conducted an additional experiment to train an agent that makes decisions about whether to add incoming elements to an existing set with different types, while considering a budget constraint. The aim of this experiment was to explore the agent's ability to optimize the objective function by selecting the most suitable items for inclusion in the set and to show that optimal policies can be determined by reinforcement learning unlike the heuristic approach that author present in paper. To complete the setup, we defined RL components with followings:

- State Definition: The state variables $s_t$ in this experiment were defined by the budget at each time step, which represented the capacity of each item type at time t.
- Action: Action of agent at time t $a_t$ defines adding incoming element to set k with and without the replacement of the existing items in the set considering the budget constraint
- Reward Function: The reward function $r_t$ was designed to capture two scenarios; when an element was added to the existing set without replacing any element, the reward was the marginal gain of the objective function. On the other hand, when an element was added to the set with the replacement of an existing element, the reward was calculated as the difference between the objective value of the previous set $S_t$ and the new set $S_{t+1}$.

To train the agent, we employed a value function approximation method using a neural network with two fully connected layers and the ELU activation function. Loss function is defined as below:

![Loss function](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/1e745bbb-2695-4699-879c-da541eed9660)

The gradients of the layers were computed using the Bellman equation as shown below:

![Belman Equation](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/605254be-0841-4270-957c-0da797066d0d)

Where $\gamma$ represents discounted rate of future rewards. 

During the training process, we used an $\epsilon$-greedy strategy to balance exploration and exploitation. The agent would either choose a random action with probability $\epsilon$ or select the action with the highest Q-value according to the learned value function. By iteratively interacting with the environment and updating the neural network parameters, the agent learned to make informed decisions based on the observed rewards.

Hyperparameters for experiment is shown in table below:

*Table 3: Parameters of Deep Q-Network for Sensor Placement Problem*
Setup Component | Parameters 
------------ | ------------- 
Hidden Dimension | 128 
Gamma | 0.9 
Learning Rate | 0.001
Epoch | 1000 
Budget (B) | 5
Number of Sensor Types (k) | 3

Objective values over training is shown figure below:

![Results_Q](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/7647d4b2-b969-4449-8661-bba2831a6fe9)

*Figure 10: Training Progress on Sensor Placement Problem*

As can be seen on figure, objective value has rapid increase in few epochs and after that oscillation on the values are observed at remaining epochs. This is due to the unstable gradients. To adress this issue, several approaches exist in the literature such as actor-critic method, experience replay. These techniques can be used to overcome this problem as a continuation study. Also, fine tuning the hyperparameters can help to increase the performance of the model.

Our findings showed that the deep reinforcement learning approach can be used for solving the k-submodular maximization problem with the given budget constraint, particulary on sensor placement problem. By selecting the items with the minimum marginal gain for replacement, the agent demonstrated an ability to adapt and improve the objective value of the set over time. 

# 4. Conclusion

In this project, I  implemented and partially reproduced the results of the paper "Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints." The paper introduced a novel algorithm for maximizing influence in a social network with cardinality constraints, providing a constant-factor approximation guarantee. We focused on applying this algorithm to the Influence Maximization problem and the Sensor Placement problem.

For the Influence Maximization problem, we conducted a partial reproduction of the results due to limited computational resources. Despite the smaller number of simulations used to approximate the submodular function, our implementation yielded similar results to those presented in the paper. We verified the effectiveness of the proposed algorithm by comparing it with the k-Greedy algorithm, a commonly used baseline. Eventhough I could not generate the whole range of bugdet constraint in the problem, I could be able to generate some of the results to compare with author's.

In the Sensor Placement problem, we also achieved similar results to those presented in the paper. Our implementation of the proposed algorithm demonstrated a faster convergence towards the results of the Greedy algorithm. Notably, it is observed that our implementation of the proposed algorithm approached the results of the Greedy algorithm more quickly than reported by the authors. This discrepancy could be attributed to our specific strategy for dealing with missing values.

Our implementation confirmed the key findings of the paper and highlighted the effectiveness of the proposed algorithm for both the Influence Maximization and Sensor Placement problems. The project underscores the significance of efficient algorithms for submodular maximization in practical applications such as social network analysis and sensor deployment.

Finally, we set up additional experiments to absorve the overhead cost of the approximiting the function on influence maximization problem and to highlight the potential of using deep reinforcement learning techniques for decision-making in resource-constrained scenarios.

To wrap up, I enjoyed reproducing this paper and learned a lot during the process. I would like to thank the authors for writing such a great and mostly clear paper and Sinan Hoca for equipping us with the skills to take on this project.

# 5. References

- Christian Borgs, Michael Brautbar, Jennifer Chayes, and Brendan Lucier, ["Maximizing Social Influence in Nearly Optimal Time," in Proceedings of the Twenty-Fifth Annual ACM-SIAM Symposium on Discrete Algorithms, 2014, pp. 946-957.](https://arxiv.org/pdf/1212.0884.pdf)
- David P. Williamson and David B. Shmoys, [*The Design of Approximation Algorithms*. Cambridge University Press, 2010.](http://www.designofapproxalgs.com/)
- Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf). arXiv preprint arXiv:1312.5602.
- Naoto Ohsaka and Yuichi Yoshida, ["Monotone k-Submodular Function Maximization with Size Constraints," NeurIPS,2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf)
- Rishabh Iyer and Jeff Bilmes, ["Submodular Optimization with Submodular Cover and Submodular Knapsack Constraints," SIAM Journal on Computing (SICOMP), vol. 47, no. 1, pp. 175-203, 2018.](https://proceedings.neurips.cc/paper/2013/file/a1d50185e7426cbb0acad1e6ca74b9aa-Paper.pdf)
- Stephen Boyd and Lieven Vandenberghe, [*Convex Optimization*. Cambridge University Press, 2004.](https://web.stanford.edu/~boyd/cvxbook/)
- Yaron Singer "AM 221: Advanced Optimization," Harward University, Available: [http://people.seas.harvard.edu/~yaron/AM221-S16/index.html](http://people.seas.harvard.edu/~yaron/AM221-S16/index.html)
- Velickovic Peter et al, [Graph Attention Networks, ICLR, 2018](https://arxiv.org/pdf/1710.10903.pdf)
- Youze Tang, Yanchen Shi, and Xiaokui Xiao, ["Influence Maximization in Near-Linear Time: A Martingale Approach," SIGMOD '15: Proceedings of the 2015 ACM SIGMOD International Conference on Management of DataMay 2015Pages 1539–1554](https://dl.acm.org/doi/10.1145/2723372.2723734)

# Contact

Mehmet Barutcu - mehmetbarutcu00@gmail.com

[Github Link](https://github.com/MehmetBarutcu)

# License

All original code I wrote in this repository is licensed under the MIT License. Everything except for all pictures in the images directory are borrowed from the papers referenced above is my original work. These images belong to their copyright holders. They are provided in this repository for educational purposes only, which constitute fair use under the US copyright law.
