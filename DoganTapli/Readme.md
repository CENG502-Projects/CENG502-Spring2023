# Attacking Deep Reinforcement Learning With Decoupled Adversarial Policy 
Adam Gleave, Michael Dennis, Cody Wild, Neel Kant, Sergey Levine, Stuart Russell

Deep reinforcement learning (RL) policies are known to be vulnerable to adversarial perturbations to their observations, similar to adversarial examples for classifiers. However, an attacker is not usually able to directly modify another agent's observations. This might lead one to wonder: is it possible to attack an RL agent simply by choosing an adversarial policy acting in a multi-agent environment so as to create natural observations that are adversarial? We demonstrate the existence of adversarial policies in zero-sum games between simulated humanoid robots with proprioceptive observations, against state-of-the-art victims trained via self-play to be robust to opponents. The adversarial policies reliably win against the victims but generate seemingly random and uncoordinated behavior. We find that these policies are more successful in high-dimensional environments, and induce substantially different activations in the victim policy network than when the victim plays against a normal opponent. Videos are available at this https [URL](https://adversarialpolicies.github.io/). 

---

This readme file is an outcome of the [CENG502 (Spring 2023)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 20223) Project List](https://github.com/CENG502-Projects/CENG502-Spring2023) for a complete list of all paper reproduction projects.

# 1. Introduction

This paper titled "Attacking Deep Reinforcement Learning with Decoupled Adversarial Policy" was published in the IEEE Transactions on Dependable and Secure Computing. The goal of this work is to propose a novel approach for exploiting the vulnerability of Deep Reinforcement Learning (DRL) systems with adversarial attacks. Our goal is to reproduce the results presented in the paper and evaluate the effectiveness of the proposed method.

Deep Reinforcement Learning (DRL) has shown remarkable success in solving complex tasks, such as game playing, robotics, and autonomous driving. However, recent studies have shown that DRL systems are vulnerable to adversarial attacks, where an attacker can manipulate the input to the system and cause it to produce incorrect outputs. Adversarial attacks on DRL systems can have severe consequences, such as safety risks, privacy breaches, and financial losses. Therefore, it is crucial to develop effective defense mechanisms against adversarial attacks on DRL systems.

One approach to developing effective defense mechanisms is to understand the vulnerabilities of DRL systems and the methods used to exploit them. This paper proposes a novel approach for attacking DRL systems using Decoupled Adversarial Policy (DAP). The DAP approach decomposes the adversarial policy into two sub-policies: the switch policy and the lure policy. The switch policy determines whether the attacker should launch an attack, while the lure policy determines which action the attacker induces the victim to take. By inducing the victim to take a specific action, the attacker can mislead the DRL system and obtain misleading results. The proposed method is shown to be more efficient and practical than existing methods for attacking DRL systems.

The goal of this work is to reproduce the results presented in the paper and evaluate the effectiveness of the proposed method. Reproducing the results is essential for verifying the claims made in the paper and ensuring that the proposed method is reliable and robust. Additionally, evaluating the effectiveness of the proposed method is crucial for understanding its potential for practical applications and identifying its limitations. By reproducing the results and evaluating the proposed method, we can contribute to the development of effective defense mechanisms against adversarial attacks on DRL systems.

- [ ] @TODO: Introduce the paper (inc. where it is published) and describe your goal (reproducibility).

## 1.1. Paper summary

![image](https://github.com/CENG502-Projects/CENG502-Spring2023/assets/47499605/e7a700b3-f53e-4115-a637-8be8ad44eac9)

The paper proposes a Decoupled Adversarial Policy (DAP) approach for attacking the policy networks in Deep Reinforcement Learning (DRL) systems. The DAP is decomposed into two sub-policies: the switch policy and the lure policy. The switch policy determines whether the attacker should launch an attack, while the lure policy determines which action the attacker induces the victim to take. By inducing the victim to take a specific action, the attacker can mislead the DRL system and obtain misleading results. The proposed method is shown to be more efficient and practical than existing methods for attacking DRL systems

The paper titled "Attacking Deep Reinforcement Learning with Decoupled Adversarial Policy" presents a comprehensive study on exploiting the vulnerability of Deep Reinforcement Learning (DRL) systems through adversarial attacks. The authors propose a novel approach called Decoupled Adversarial Policy (DAP) to launch effective attacks on DRL mechanisms. The DAP approach decomposes the adversarial policy into two sub-policies: the switch policy and the lure policy. The switch policy determines whether the attacker should launch an attack, while the lure policy determines the action that the attacker induces the victim to take.

The paper's contributions lie in the development of the DAP approach, which offers several advantages over existing methods for attacking DRL systems. Firstly, by decomposing the adversarial policy, the DAP approach provides more flexibility and control to the attacker, allowing for more targeted and effective attacks. Secondly, the DAP approach introduces the concept of a lure policy, which enables the attacker to influence the victim's actions strategically. This strategic influence can lead to misleading results and compromise the integrity of the DRL system.

The proposed DAP approach is evaluated through experiments conducted on four different DRL environments. The experiments aim to verify the effectiveness of the DAP approach from various aspects. The paper addresses two research questions: (1) Can the proposed DAP launch successful attacks with a small number of injecting times in different reinforcement learning tasks? (2) In which states does the proposed DAP launch the attack and inject perturbations? The experimental results demonstrate the effectiveness of the DAP approach in launching successful attacks with minimal injecting times across different reinforcement learning tasks. The paper also provides insights into the states where the DAP approach is most effective in launching attacks and injecting perturbations.

The proposed DAP approach offers a significant contribution to the field of adversarial attacks on DRL systems. By decomposing the adversarial policy into switch and lure policies, the DAP approach provides a more sophisticated and targeted method for attacking DRL mechanisms. The experimental results validate the effectiveness of the DAP approach and highlight its potential for building more robust DRL systems. The findings of this paper have implications for the development of defense mechanisms against adversarial attacks on DRL systems and contribute to the broader understanding of the vulnerabilities and security challenges in the field of DRL

- [ ] @TODO: Summarize the paper, the method & its contributions in relation with the existing literature.

# 2. The method and my interpretation

## 2.1. The original method

The original method known as Decoupled Adversarial Policy (DAP) aims to target Deep Reinforcement Learning (DRL) systems. DAP employs a two-part approach, separating the adversarial policy into two sub-policies: the switch policy and the lure policy. The switch policy assesses whether the attacker should initiate an attack, while the lure policy determines the specific action the attacker persuades the victim to perform. By manipulating the victim's actions, the attacker effectively misguides the DRL system, leading to deceptive outcomes.

To introduce the proposed DAP approach, a convolutional neural network is employed to encode states into a compact one-dimensional feature representation. Additionally, a fully-connected network is utilized to encode policies into another one-dimensional feature representation. These two features are concatenated and passed through an LSTM (Long Short-Term Memory) network to capture sequential patterns. The final stage comprises two branches implemented as fully-connected networks, which generate the switch and lure policies, respectively. The switch policy determines whether launching an attack is warranted, while the lure policy specifies the action the attacker should entice the victim into executing.

In summary, the initial technique called Decoupled Adversarial Policy (DAP) is devised to target Deep Reinforcement Learning (DRL) systems. It involves separating the adversarial policy into two sub-policies: the switch policy and the lure policy. Manipulating the victim's actions through the lure policy enables the attacker to deceive the DRL system. The proposed DAP approach incorporates a convolutional neural network, a fully-connected network, and an LSTM to encode states, policies, and sequential features. Finally, two branches in the form of fully-connected networks generate the switch and lure policies, respectively, determining the attacker's actions and the victim's responses.
- [ ] @TODO: Explain the original method.

## 2.2. Our interpretation 

The original paper provides a clear explanation of the proposed method. However, some technical details were not fully explained, such as the architecture of the neural network used to encode the states and policies. We interpreted these details based on our understanding of the related literature and the experimental results presented in the paper.

In our interpretation, we focused on understanding the technical details of the method that were not explicitly explained in the original paper. For example, the paper mentions the use of a convolutional neural network (CNN) to encode states and a fully-connected network to encode policies. However, the specific architecture and configuration of these networks were not provided. To address this, we interpreted that the CNN is used to extract spatial features from the states, while the fully-connected network captures the policy information. These features are then concatenated and processed by an LSTM to capture sequential features. Finally, two branches implemented as fully-connected networks are utilized to output the switch and lure policies, respectively.

Furthermore, we interpreted that the switch policy determines whether an attack should be launched based on the current state and policy information, while the lure policy determines the specific action to induce the victim to take. The lure policy is crucial in strategically influencing the victim's actions to achieve the attacker's objectives. By understanding these interpretations, we gain a clearer understanding of the method's architecture and its underlying mechanisms.

Additionally, we considered the implications of the DAP approach in terms of its effectiveness and practicality. We interpreted that the DAP approach offers advantages over existing methods by requiring fewer perturbation injection times to attack DRL systems, thus enhancing its secrecy. This interpretation is based on the experimental results presented in the paper, which demonstrate the efficiency and effectiveness of the DAP approach compared to comparative methods.

Overall, our interpretation of the method focused on filling in the technical details that were not explicitly explained in the original paper. By understanding these details, we gain a more comprehensive understanding of the proposed approach and its potential implications. This interpretation allows us to replicate and evaluate the method accurately, contributing to the reproducibility and reliability of the research.
- [ ] @TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

The experiments were conducted on four different DRL environments, including Pong, Breakout, MsPacman, and Enduro. The widely used algorithms, including DQN and PPO, were applied to train the victim agents. The network trained with DQN outputs the estimate of Q values, and the one trained with PPO outputs the distribution over possible actions. We followed the same experimental setup as described in the original paper. The hyperparameters, such as learning rate, discount factor, and batch size, were set according to the recommendations in the literature. For example some of such parameters are hardcoded in the code as below.

```python
Episode = 4000
Discount Factor = 0.997
Lambda_r = 0.95 # a hyperparameter in the reward calculation
T = 1500
Max_Inject = 21 # for Pong game
```

- [ ] @TODO: Describe the setup of the original paper and whether you changed any settings.

## 3.2. Running the code

We replicated the experiments described in the paper using the provided codebase. The codebase included the implementation of the DAP approach and the necessary components for training and evaluating the victim agents. We ensured that the codebase was properly set up and executed the experiments on the same hardware as mentioned in the paper.
```bash
# To pretrain the PPO model 
python3 ppo_train.py
```
During the experiments, we observed the performance of the DAP approach in terms of its ability to deceive the victim agents and induce them to take suboptimal actions. We compared the results with the baseline methods and analyzed the effectiveness of the proposed approach.
```bash
# To run the uap model 
python3 main.py --model-path assets/ALE/Pong*.p
```
- [ ] @TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results
The paper compared the proposed DAP approach with existing methods for attacking DRL systems. It demonstrated that the DAP approach outperformed the baseline methods in terms of attack success rate and the ability to mislead the victim agents. This comparison provides evidence of the effectiveness of the proposed approach and its potential for practical applications.
While the DAP approach showed promising results, there are some limitations to consider. The experiments were conducted on a limited set of DRL environments, and it would be valuable to evaluate the approach on a wider range of tasks. Additionally, the robustness of the DAP approach against defense mechanisms and countermeasures should be investigated in future work.

- [ ] @TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

The paper compared the proposed DAP approach with existing methods for attacking DRL systems. It demonstrated that the DAP approach outperformed the baseline methods in terms of attack success rate and the ability to mislead the victim agents. This comparison provides evidence of the effectiveness of the proposed approach and its potential for practical applications.
In conclusion, the paper presents a novel Decoupled Adversarial Policy (DAP) approach for attacking Deep Reinforcement Learning (DRL) systems. The DAP approach decomposes the adversarial policy into switch and lure policies, allowing the attacker to induce the victim agent to take specific actions. The experiments conducted in the paper demonstrate the effectiveness of the DAP approach in deceiving the victim agents and obtaining misleading results. The proposed approach shows potential for practical applications and opens avenues for further research in the field of adversarial attacks on DRL systems.
- [ ] @TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

- Adnan Harun Dogan: [Github](https://github.com/adnanhd) [Twitter](https://twitter.com/adnanharundogan) [Google Scholar](https://scholar.google.com/citations?user=QGaRpqYAAAAJ&hl=en)
- Merve Tapli: [Github](https://github.com/mtapli)

@TODO: Provide your names & email addresses and any other info with which people can contact you.
