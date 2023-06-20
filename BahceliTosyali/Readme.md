# MARINA: An MLP-Attention Model for Multivariate Time-Series Analysis

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

The paper is published for CIKM 2022. The goal of the paper is to achieve better accuracies in multivarite time series anomaly detection.  Our goal was to train the model suggested in the paper 
and see the result for ourselves

## 1.1. Paper summary

Anomaly detection in multivariate time series is a hot-topic in deep learning. this paper proposes a unique idea in terms of data normalization. Rather than using static mean and variance which is done in similiar papers, This paper suggests a dynamic method of normalization in test set which increases mdel's performence overall. Combining this with spatial correlation and temporal correlation methods gives top notch performence on time series anomly detection. 

# 2. The method and my interpretation

## 2.1. The original method

The paper is divided into three sections, data normalization, temporal correlation, and spatial correlation 

### 2.1.1 Data Normalization
Traditionally; in normalization procedures test, validation and training datas should all be normalized before sending it into network. This paper also adopts this technique using mean as 0 and variance as 1, but only in training and validation sets. The procedure for test set is entirely different. In test set we update our mean and variance using previous mean and variances, This allows us to avoid the phenomenon known as concept drifting. Concept drifting refers to the phenomenon in data science where the statistical properties of a target variable or the relationships between features and the target variable change over time, which may cause huge distortions in our anomaly prediction platform , we might take anomalies as usual data because of it or vice versa. 

<p align="center">
  <img src="https://github.com/Batucan2601/MARINA/assets/88089192/5e7f1a5d-99cf-4102-ba1b-46e86a68dfab" alt="drifting" width="600">
  <br>
  <em>Figure 1: Illustration of the concept drifting phenomenon <i>(from the paper)</i>.</em>
</p>



In order to avoid this we use the following formulas in acquiring the next mean and next variances over time. 

<p align="center">
$\mu_0 = \mu$ <br>
$\sigma_0 = \sigma$ <br>
$\mu_i = (1-\alpha)\mu_{i-1} + \alpha E(x_i)$ <br>
$\sigma_i^2 = (1-\alpha) \sigma_{i-1}^2 + \alpha (E(x_i^2) - E(x_i)^2)$

</p>

Where $E(x)$ is the expected value and  alpha is the weight coefficient. As you can see the mean and variance relies on their previou iterations.

### 2.1.2 Temporal Correlation Module
This module uses the correlation between historical and future points. Paper suggests that using simple MLP blocks rather than using complex architectures such as Transformers etc. gives both better computation speed and better accuracy in the end, therefore spatial module uses only MLP blocks and no other structure.
This module uses a window of size $n$  where with $n$ data it tries to predict the future $\eta$ points. This module uses a total of $K$ MLP blocks, where each MLP block is also consisting of an Input subblock, cascading subblock and a forecasting subblock, which are also consisting of multi layer perceptron.
<p align="center">
  <img src="https://github.com/Batucan2601/MARINA/assets/88089192/6d9b016a-d1a2-4752-b42e-8771521e0a02" alt="inputTemporal" width="250">
  <br>
  <em>Figure 2: Illustration of the input temporal data <i>(from the paper)</i>.</em>
</p>

The relation between those three blocks and MLP block is as follows

<p align="center">
$X_{1,I}^{temp} = X,$ <br>
$X_{k+1,I}^{temp} = X_{k,I}^{temp} - X_{k,C}^{temp},$ <br>
$X_{O}^{temp} = \sum\limits_{k=1}^{K} X_{k,F}^{temp}$
</p>


where $X_I$ is the input subblock $X_F$ is the forecasting subblock and $X_C$ is the cascading subblock. $X_O$ represents the output that is leaving the MLP block.
Paper suggests using 2 MLP block is enough for anomaly detection, therefore we used 2. 

### 2.1.3 Spatial Correlation Module

Spatial module works on each of the time-series separately therefore does not exploit the features of time based data like temporal correlation module. In order to extract the features between the elements in a time series data, paper suggests using multi head self attention mechanism in order to imitate graph neural netowrks which is a popular solution for this problem.
Each row of the output of Temporal correlation module has been used as a vertex; and the message passing is done via following

<p align="center">
$Q = K = V = X_O^{temp},$ <br>
$X_{Int}^{Spat} = MultiHeadAttention(Q,K,V),$ <br>
$X_{O}^{spat} = FFN(X_{Int}^{spat})$
</p>

where $X_O$ is the output from temporal module; $Q$, $K$, $V$ are query, key and values in this self attention model respectively; and FFN stands for Feed Forward neural network. 

### 2.1.4 Output Reshaping Module
Paper also uses a final outut reshaping module which is a simple MLP used in order to change the dimension of output

### Loss
Paper suggests using the frobenius norm for loss. Frobenius norm is basically the L2 norm which corresponds to the lenght of the vector. Therefore Frobenius Loss is the length of the difference of two vectors. Paper suggests using a threshold for this loss; if the threshold is exceeded model informs that this data is an anomaly.  
## 2.2. Our interpretation 
- The threshold for frobenius norm has not been specified therefore we tried many random thresholds for best performence. Best is around 1.85.
- We followed every constant given by paper for anomaly detection. The window length is 100; number of elements to predict in future is 1. Alpha used in normalization is 0.1 and so on.
- We moved the windows with interval of 5 rather than traditional 1 for testing.
- We could only test the dataset of SMAP.
- We did not used CUDA for training. It is not mentioned in the paper but looking at the time statistic it seems the authors also have not used CUDA support. Therefore in order to get the most similar result we also have not used it.

### 2.2.1 Splitting data into windows
In accordance with the findings presented in section 2.1.2, the authors employed a strategy of dividing the dataset into windows. Notably, Figure 4 in the article illustrates that these windows do not overlap. However, when considering the equation provided for calculating the number of targets:

$$
B = T - \eta + 1
$$

where $B$ represents the number of targets, $T$ denotes the sequence length, and $\eta$ signifies the number of predicted points, it becomes evident that these windows should indeed overlap. To address this requirement, we have implemented a function called ```windowed_Set()```. This function accepts the following input parameters: original_data, window_size, shifting, and horizon. By utilizing the windowed_Set function, one can manipulate the data in the following manner:

```windows_size``` : Determine the desired length of the window.

```shifting```     : Specify the number of jumps for the window, akin to the *stride* value used in convolution.

```horizon```      :  Indicate the number of predicted points within the upcoming windows.

The illustration for this function and handling behavior can be seen in the Figure 3 below.

<p align="center"> 
  <img src="https://github.com/Batucan2601/MARINA/assets/88089192/7fdbb055-b443-40fe-90ee-8a5387f8ee44" alt="windowed_image">
  <br>
  <em>Figure 3: Illustration for <i>windowed_Set</i> function <i>ours</i>.</em>
</p>


# 3. Experiments and results

## 3.1. Experimental setup

With each iteration paper suggests using a window of data ( where window length is 100 in this case ), and iterate the data in training by 1. Because of the performence issues we shifted the 
window by 5 rather than 1. After this we put our data into our temporal correlation module with MLPs. Since we used pytorch this was an easy process; and we completely implemented what paper was 
suggesting. Following temporal correlation, we put our outpur of datas into spatial correlation module; which is also fully implementable on pytorch therefore there is not any changes with the setup given in the paper. After this process we simpply used an output moduel which is basically a MLP module which is also implemented without problems. All of the setup requirements for neural 
networks are given in the following figure. 

<p align="center"> 
  <img src="https://github.com/CENG502-Projects/CENG502-Spring2023/assets/88089192/21ad7e85-3db4-4883-8c26-d717fce4ef85" alt="windowed_image" width="450">
  <br>
  <em>Figure 4.1: The chart for parameters in the neural network for long sequence forecasting model <i>(from the paper)</i>. </em>
</p>

<p align="center"> 
  <img src="https://github.com/CENG502-Projects/CENG502-Spring2023/assets/88089192/1730d020-0221-4c1a-b8bc-15fddd67403f" alt="windowed_image" width="450">
  <br>
  <em>Figure 4.2: The chart for parameters in the neural network for anomaly detection model <i>(from the paper)</i>. </em>
</p>

For tests we used Frobenius norm (L2 norm) as suggested without any changes.
## 3.2. Running the code
Directory structure:

```bash
├── SMAP MSL/
│   ├── data/data
│   │       ├── 2018-05-19_15.00.10/
│   │       │   test/
│   │       │   train/
│   └── labeled_anomalies.csv
├── sample_output.ipynb
├── Marina.py
├── README.md
└── marina.pdf
```
- One should use ```Marina.ipynb``` for training and testing the data. For ones who wants to see a simple output of the code can visit ```sample_output``` file. 


## 3.3. Results
### 3.3.1 Long Sequence Forceasting
In this section, they implement the model for forecasting target. 
#### 3.3.1.1 Results on ETTh1
We created the model and trained the model with 30 epochs as mentioned in the paper. Then we used *optuna* for optimizing the hyperparameters such as layer numbers and batch sizes. Here the loss curves can be seen in the below figure.

<p align="center"> 
  <img src="https://github.com/Batucan2601/MARINA/assets/88089192/d0fcb565-de5c-4329-8a5c-4b368742cf43" alt="windowed_image">
  <br>
  <em>Figure 5: Training and validation loss for Etth1 dataset <i>ours</i>. </em>
</p>

Then we calculated the MSE and MAE loss values for this dataset with the model. The results can be seen in the table below.

<p align="center"> 
  <img src="https://github.com/CENG502-Projects/CENG502-Spring2023/assets/88089192/55555f08-ac09-448b-a652-9dfa62a3b086" alt="windowed_image">
  <br>
  <em>Figure 6: Metric scores for different models and ours for Etth1 dataset <i>ours</i>. </em>
</p>




#### 3.3.1.2 Results on SMAP
  The following chart shows the start and end sequences of anomalies in SMAP's first five entries.
  
<p align="center"> 
  <img src="https://github.com/Batucan2601/MARINA/assets/52931384/0ce06ec2-bad6-4f4d-8e2f-e1ae2c07f910" alt="windowed_image">
  <br>
  <em>Figure 7: The anomaly sequences of SMAP dataset </em>
</p>

And the following chart shows us the anomaly sequences that are found by our model.

<p align="center"> 
  <img src="https://github.com/Batucan2601/MARINA/assets/52931384/d93044e6-8685-480d-8345-c7683fdcac42" alt="windowed_image">
  <br>
  <em>Figure 8: The anomaly sequences of our model.</em>
</p>

As you can see our model's accuracies ranges from 0.2 to 0.6; our mean accuracies 0.4 overall. 
# 4. Conclusion
This study aimed to replicate the detection model proposed in the referenced article. The paper introduces an effective approach for predicting time-based data and identifying anomalies within the predicted data. However, during our replication attempt, we encountered certain ambiguities and omissions in the article. Specifically, the article fails to provide information regarding the shifting (stride) value for the window algorithm, which plays a crucial role in model training. Consequently, we introduced additional code to incorporate this method and assessed its impact on the results. Additionally, to enhance computational efficiency, we opted for smaller shifting values, such as 5, as mentioned earlier.

Furthermore, while the paper suggests the use of its model for both future prediction and anomaly detection in the predicted data, it does not explicitly clarify whether a single model was employed for both tasks or if separate models were trained. To address this uncertainty, we devised a solution by implementing two distinct models for prediction and anomaly detection.

# 5. References

[1] Original paper: Xie, J., Cui, Y., Huang, F., Liu, C., Zheng, K. (2022). MARINA: An MLP-Attention Model for Multivariate Time-Series Analysis. In Proceedings of the CIKM Conference on Information and Knowledge Management.


# Contact

Furkan Bahçeli, furkan.bahceli@metu.edu.tr

Batuhan Tosyalı, batuhan.tosyali@metu.edu.tr
