# Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

This project aims to reproduce the results of the paper titled "Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints," published in ICML, 2022.

The main goal of the paper is to present an efficient streaming algorithm for solving the problem of maximizing a monotone k-submodular function subject to cardinality constraints. The problem involves selecting a subset of elements from a larger set to maximize a specific objective function while adhering to constraints on the size or cardinality of the selected subset.

The authors propose a novel streaming algorithm that processes the elements in a streaming fashion, making efficient use of limited memory resources. The algorithm provides near-optimal solutions and guarantees a constant-factor approximation to the optimal solution.

In this project, we have reproduced the key results and findings of the paper, implemented the proposed algorithm, and validated its performance on various datasets. The code and experimental results can be found in this GitHub repository.

## 1.1. Paper summary

The paper addresses the problem of maximizing a monotone k-submodular function subject to cardinality constraints. The authors propose a novel streaming algorithm that efficiently processes elements in a streaming fashion, providing near-optimal solutions and a constant-factor approximation guarantee. The algorithm is designed to work with limited memory resources, making it suitable for large-scale datasets.

The paper begins by introducing the problem formulation and its significance in various applications, such as influence maximization, summarization, and recommendation systems. It then presents the streaming algorithm that tackles the problem by processing elements one at a time and maintaining a compact summary of the encountered elements. The algorithm's design leverages the submodularity property to ensure efficiency and approximation guarantees.

The proposed streaming algorithm in the paper makes several key contributions to the existing literature:

1. **Efficient Streaming Approach:** The algorithm processes elements in a streaming fashion, making it suitable for scenarios with limited memory resources or large-scale datasets. It efficiently maintains a compact summary of the encountered elements, allowing for near-optimal solution with approximation guarentee as budget constraint approaches to infinity which is the case with real time applications.

2. **Near-Optimal Solutions:** The algorithm provides near-optimal solutions for the problem of maximizing a monotone k-submodular function subject to cardinality constraints. It achieves constant-factor approximation guarantees, ensuring high-quality solutions even with limited computational resources.

3. **Theoretical Analysis:** The paper provides a theoretical analysis of the algorithm's performance, including bounds on approximation guarantees and computational complexity. It establishes the algorithm's effectiveness and its applicability to real-world problems.

4. **Free Disposal of Algorithm Parameters:** Paper offers an algorithm which does not stores the previous iterations results which makes it efficient in term of memory usage and asymtotic complexity, unlike the other state of the art approaches in the literature that uses greedy approach [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf) and offline computation other than streaming approach. 

The problem addressed in the paper can be mathematically formulated as follows:

![lp_formulation](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/eb88c61d-c14d-40d1-a3b8-5ea27f5d5680)

The formulation captures the objective of maximizing a monotone k-submodular function subject to constraints on the size of the selected subset. The paper discusses the details of the formulation and provides insights into its properties and potential algorithmic approaches for solving it. 
- xe,i ∈ {0, 1} that indicates whether element e is assigned label i. 
- yA ∈ {0, 1} that indicates whether A is the selected k-tuple. 
- The first set of constraints enforce that e receives label i if and only if e ∈ Ai for the selected labeling. 
- The second set of constraints enforce that we select exactly one k-tuple. 
- The third set of constraints enforce that each element receives at most one label. 
- The fourth set of constraints enforce the size constraints for the labels. 
By relaxing the integrality constraints xe,i, yA ∈ {0, 1} to xe,i, yA ∈ (0, 1), we obtain the LP relaxation for the problem

# 2. The method and my interpretation

### 2.1. The original method

This section describes the original method presented in the paper "Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints." It covers the definition of k-submodularity, the algorithm proposed by the authors, the concept of marginal gain, and the approximation of the submodular function.

### 2.1.1 k-Submodularity Definition

In the paper, the concept of submodularity plays a fundamental role. A function f: (2+1)^V -> R, where V is a finite ground set, is said to be k-submodular if it satisfies the diminishing returns property. Specifically, for any sets A, B subset of V, where |A| <= |B|, and any element e not in B, the following inequality holds:

$f(A ∪ {e}) - f(A) >= f(B ∪ {e}) - f(B)$

This property captures the idea that adding an element to a smaller set yields a higher marginal gain compared to adding it to a larger set. It is a key characteristic of many real-world optimization problems. K-submodularity is the generalization of the submodular definition into k different sets. A function is k-submodular if and only if:

Pairwise Monotone

$\delta_{e,i} f(X) + \delta_{e,j} f(X) \geqslant 0$ and,

 Orthant Submodular
 \delta_{e,i} f(X) \geqslant \delta_{e,i} f(Y)
 
 for all $X, Y ∈ (k+1)^V$ such that X \prec Y, $e \notin supp(Y)$ and $i ∈ (k)$

### 2.1.2 Marginal Gain Definition

The concept of marginal gain plays a crucial role in the algorithm. The marginal gain of adding an element v to a current solution set S is defined as the increase in the objective function value:

$\delta_{e,i} f(v) = f(S_1 ∪ S_2 ... S_i ∪ {e}... ∪ S_k) - f(S_1 ∪ S_2 ... S_i ∪ S_k)$

It quantifies the utility or contribution of an element towards improving the objective function value when added to the current solution set.

### 2.1.3 Therotical Approximation Bound of Optimal Solution

This section provides an overview of the approximation guarantees for maximizing submodular functions subject to cardinality constraints. The table below summarizes some of the notable approximation guarantee algorithms in the literature.

![approximation_table_2](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/0405c199-8f79-41e0-8b68-4c8aac7f0e83)

Proof of the approximation bound can be found in [paper] (https://proceedings.mlr.press/v162/ene22a/ene22a.pdf)

### 2.1.4 Proposed Algorithm

The paper proposes a novel streaming algorithm for maximizing a monotone k-submodular function subject to cardinality constraints. The algorithm processes the elements in a streaming fashion, making efficient use of limited memory resources.

Algorithm details are as follows:

![primal_dual_algorithm](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/2a97554e-5f37-4d36-a44b-feb3b3bbb458)

Method uses primal-dual variables to increase the treshold of accepting an item as the marjinal gain decreases over time.

## 2.2. Our interpretation 

Although the algorithm is well defined in the paper, some of the aspects are problem dependent. Those are:

**1. Definition of Function:** Problem related function should be defined in order to reflect of the benefits of any combination of elements. In the paper, there are two different experiment settings which both have different evalution functions. For influence maximization problem, function evaluation is very costly even on a small subset of elements. To overcome this, benefit function is approximated according to method proposed by [Borgs et al, 2014] (https://arxiv.org/pdf/1212.0884.pdf)

**2. Marginal Benefit of Adding an element to Existing Set:** Marginal benefit determines the whether incoming element should be added to existing element set S. It is very critical to design marginal benefit function to reflect the gain of the element. Function is problem dependent and should be designed carefully. For influence maximization problem in paper, there are no clear statements as to how this function is design. I referred another research [Youze et al, 2015] (https://dl.acm.org/doi/10.1145/2723372.2723734) to determine the marginal benefit as the difference of the  values sets in approximated function R that are covered by a node
set S.

**3. Dealing with Missing Values in Dataset:** In sensor placement problem, There are missing rows related to humidity, light condition and temperature columns. Since strategy of dealing with missing values are not clearly stated in paper, I chose my own method that I eliminated all missing rows in the dataset which is about 400.000 rows of data in total dataset.

# 3. Experiments and results

## 3.1. Experimental setup

There are two experiments in paper: Influence maximization and sensor placement problem. These are breifly discussed below:

**1. Influence Maximization with *k* topics:** In this experiment, we aim to maximize influence in a social network with multiple topics of interest. The objective is to identify a set of influential nodes that can maximize the spread of information or influence across the network for each topic. The experiment involves the following steps:

1. Constructing directed network graph from undirected edges in Facebook SNAP dataset.
2. Approximating the influence function proposed by [Borgs et al, 2014] (https://arxiv.org/pdf/1212.0884.pdf) with approximation guarentee lowerbound number of simulations O(nk^2log(n)log(k)) where k is number of topics and n is number of nodes in grapgh using k-Independent Cascade Process proposed by [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf)
3. Applying the proposed influence maximization algorithm, leveraging the k-submodular framework, to identify the most influential nodes for each topic.
4. Comparing the performance of the algorithm with k-greedy algorithm proposed by [Ohsaka & Yoshida, 2015](https://papers.nips.cc/paper_files/paper/2015/file/f770b62bc8f42a0b66751fe636fc6eb0-Paper.pdf).
5. Evaluating the values of the methods ranging between different budget constraints.

Although the therotical number of simulations required stated at [Borgs et al, 2014] (https://arxiv.org/pdf/1212.0884.pdf), author of the paper did not state anything about whether it is used as number of simulations to approximate the function. In addition, cost of evaluating the function with these number of simulations is very costly, so I used this number R = 100 in order to get a frame about comparision of the algorithms with different budget constraints. Simulation process explained in following figure:

![build_hypergraph](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/811e5f3e-b03e-4127-9ca1-9b132d982dd2)
*Figure N Simulation Process of Approximating Influence Function*

Process uses Independent Cascade Process which can be seen in figure below:

![icp](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/84293711/5e21eaee-3ea4-4cdb-9254-5e6eb9846a9a)
*Figure N Independent Cascade Process*


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
└── requirements.txt
```

- Files in datasets folder contains facebook and Intel Lab datasets in txt format
- Images folder contains copy right images used in this readme file
- requirements.txt contains the list of required Python packages.
- influence_maximization.ipynb file contains the influence maximization experiment with data preprocessing functions and k-greedy algorithm implementation
- sensor_placement.ipynb file contains the sensor placement experiment with data preprocessing functions k-greedy algorithm implementation

### 3.2.1 Create Environment
**conda create (env_name)**

### 3.2.2 Install Dependencies
**pip install -r requirements.txt**

### 3.2.3 Running the Experiments
Cells in jupyter notebooks can sequencially be run to reproduce the results.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

Mehmet Barutcu - mehmetbarutcu00@gmail.com
