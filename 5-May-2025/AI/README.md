# BalancEdit: Dynamically Balancing the Generality-Locality Trade-off in Multi-modal Model Editing 

**Authors**: Dongliang Guo, Mengxuan Hu, Zihan Guan, Thomas Hartvigsen, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01343)  

**Abstract**: Large multi-modal models inevitably decay over time as facts change and previously learned information becomes outdated. Traditional approaches such as fine-tuning are often impractical for updating these models due to their size and complexity. Instead, direct knowledge editing within the models presents a more viable solution. Current model editing techniques, however, typically overlook the unique influence ranges of different facts, leading to compromised model performance in terms of both generality and locality. To address this issue, we introduce the concept of the generality-locality trade-off in multi-modal model editing. We develop a new model editing dataset named OKEDIT, specifically designed to effectively evaluate this trade-off. Building on this foundation, we propose BalancEdit, a novel method for balanced model editing that dynamically achieves an optimal balance between generality and locality. BalancEdit utilizes a unique mechanism that generates both positive and negative samples for each fact to accurately determine its influence scope and incorporates these insights into the model's latent space using a discrete, localized codebook of edits, without modifying the underlying model weights. To our knowledge, this is the first approach explicitly addressing the generality-locality trade-off in multi-modal model editing. Our comprehensive results confirm the effectiveness of BalancEdit, demonstrating minimal trade-offs while maintaining robust editing capabilities. Our code and dataset will be available. 

---
# Early Detection of Patient Deterioration from Real-Time Wearable Monitoring System 

**Authors**: Lo Pang-Yun Ting, Hong-Pei Chen, An-Shan Liu, Chun-Yin Yeh, Po-Lin Chen, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01305)  

**Abstract**: Early detection of patient deterioration is crucial for reducing mortality rates. Heart rate data has shown promise in assessing patient health, and wearable devices offer a cost-effective solution for real-time monitoring. However, extracting meaningful insights from diverse heart rate data and handling missing values in wearable device data remain key challenges. To address these challenges, we propose TARL, an innovative approach that models the structural relationships of representative subsequences, known as shapelets, in heart rate time series. TARL creates a shapelet-transition knowledge graph to model shapelet dynamics in heart rate time series, indicating illness progression and potential future changes. We further introduce a transition-aware knowledge embedding to reinforce relationships among shapelets and quantify the impact of missing values, enabling the formulation of comprehensive heart rate representations. These representations capture explanatory structures and predict future heart rate trends, aiding early illness detection. We collaborate with physicians and nurses to gather ICU patient heart rate data from wearables and diagnostic metrics assessing illness severity for evaluating deterioration. Experiments on real-world ICU data demonstrate that TARL achieves both high reliability and early detection. A case study further showcases TARL's explainable detection process, highlighting its potential as an AI-driven tool to assist clinicians in recognizing early signs of patient deterioration. 

---
# Exploring the Impact of Explainable AI and Cognitive Capabilities on Users' Decisions 

**Authors**: Federico Maria Cau, Lucio Davide Spano  

**Link**: [PDF](https://arxiv.org/pdf/2505.01192)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly used for decision-making across domains, raising debates over the information and explanations they should provide. Most research on Explainable AI (XAI) has focused on feature-based explanations, with less attention on alternative styles. Personality traits like the Need for Cognition (NFC) can also lead to different decision-making outcomes among low and high NFC individuals. We investigated how presenting AI information (prediction, confidence, and accuracy) and different explanation styles (example-based, feature-based, rule-based, and counterfactual) affect accuracy, reliance on AI, and cognitive load in a loan application scenario. We also examined low and high NFC individuals' differences in prioritizing XAI interface elements (loan attributes, AI information, and explanations), accuracy, and cognitive load. Our findings show that high AI confidence significantly increases reliance on AI while reducing cognitive load. Feature-based explanations did not enhance accuracy compared to other conditions. Although counterfactual explanations were less understandable, they enhanced overall accuracy, increasing reliance on AI and reducing cognitive load when AI predictions were correct. Both low and high NFC individuals prioritized explanations after loan attributes, leaving AI information as the least important. However, we found no significant differences between low and high NFC groups in accuracy or cognitive load, raising questions about the role of personality traits in AI-assisted decision-making. These findings highlight the need for user-centric personalization in XAI interfaces, incorporating diverse explanation styles and exploring multiple personality traits and other user characteristics to optimize human-AI collaboration. 

---
# Explainable AI Based Diagnosis of Poisoning Attacks in Evolutionary Swarms 

**Authors**: Mehrdad Asadi, Roxana Rădulescu, Ann Nowé  

**Link**: [PDF](https://arxiv.org/pdf/2505.01181)  

**Abstract**: Swarming systems, such as for example multi-drone networks, excel at cooperative tasks like monitoring, surveillance, or disaster assistance in critical environments, where autonomous agents make decentralized decisions in order to fulfill team-level objectives in a robust and efficient manner. Unfortunately, team-level coordinated strategies in the wild are vulnerable to data poisoning attacks, resulting in either inaccurate coordination or adversarial behavior among the agents. To address this challenge, we contribute a framework that investigates the effects of such data poisoning attacks, using explainable AI methods. We model the interaction among agents using evolutionary intelligence, where an optimal coalition strategically emerges to perform coordinated tasks. Then, through a rigorous evaluation, the swarm model is systematically poisoned using data manipulation attacks. We showcase the applicability of explainable AI methods to quantify the effects of poisoning on the team strategy and extract footprint characterizations that enable diagnosing. Our findings indicate that when the model is poisoned above 10%, non-optimal strategies resulting in inefficient cooperation can be identified. 

---
# MADIL: An MDL-based Framework for Efficient Program Synthesis in the ARC Benchmark 

**Authors**: Sébastien Ferré  

**Link**: [PDF](https://arxiv.org/pdf/2505.01081)  

**Abstract**: Artificial Intelligence (AI) has achieved remarkable success in specialized tasks but struggles with efficient skill acquisition and generalization. The Abstraction and Reasoning Corpus (ARC) benchmark evaluates intelligence based on minimal training requirements. While Large Language Models (LLMs) have recently improved ARC performance, they rely on extensive pre-training and high computational costs. We introduce MADIL (MDL-based AI), a novel approach leveraging the Minimum Description Length (MDL) principle for efficient inductive learning. MADIL performs pattern-based decomposition, enabling structured generalization. While its performance (7% at ArcPrize 2024) remains below LLM-based methods, it offers greater efficiency and interpretability. This paper details MADIL's methodology, its application to ARC, and experimental evaluations. 

---
# Retrieval Augmented Learning: A Retrial-based Large Language Model Self-Supervised Learning and Autonomous Knowledge Generation 

**Authors**: Zongyuan Li, Pengfei Li, Runnan Qi, Yanan Ni, Lumin Jiang, Hui Wu, Xuebo Zhang, Kuihua Huang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01073)  

**Abstract**: The lack of domain-specific data in the pre-training of Large Language Models (LLMs) severely limits LLM-based decision systems in specialized applications, while post-training a model in the scenarios requires significant computational resources. In this paper, we present Retrial-Augmented Learning (RAL), a reward-free self-supervised learning framework for LLMs that operates without model training. By developing Retrieval-Augmented Generation (RAG) into a module for organizing intermediate data, we realized a three-stage autonomous knowledge generation of proposing a hypothesis, validating the hypothesis, and generating the knowledge. The method is evaluated in the LLM-PySC2 environment, a representative decision-making platform that combines sufficient complexity with domain-specific knowledge requirements. Experiments demonstrate that the proposed method effectively reduces hallucination by generating and utilizing validated knowledge, and increases decision-making performance at an extremely low cost. Meanwhile, the approach exhibits potential in out-of-distribution(OOD) tasks, robustness, and transferability, making it a cost-friendly but effective solution for decision-making problems and autonomous knowledge generation. 

---
# Adaptive Wizard for Removing Cross-Tier Misconfigurations in Active Directory 

**Authors**: Huy Q. Ngo, Mingyu Guo, Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.01028)  

**Abstract**: Security vulnerabilities in Windows Active Directory (AD) systems are typically modeled using an attack graph and hardening AD systems involves an iterative workflow: security teams propose an edge to remove, and IT operations teams manually review these fixes before implementing the removal. As verification requires significant manual effort, we formulate an Adaptive Path Removal Problem to minimize the number of steps in this iterative removal process. In our model, a wizard proposes an attack path in each step and presents it as a set of multiple-choice options to the IT admin. The IT admin then selects one edge from the proposed set to remove. This process continues until the target $t$ is disconnected from source $s$ or the number of proposed paths reaches $B$. The model aims to optimize the human effort by minimizing the expected number of interactions between the IT admin and the security wizard. We first prove that the problem is $\mathcal{\#P}$-hard. We then propose a set of solutions including an exact algorithm, an approximate algorithm, and several scalable heuristics. Our best heuristic, called DPR, can operate effectively on larger-scale graphs compared to the exact algorithm and consistently outperforms the approximate algorithm across all graphs. We verify the effectiveness of our algorithms on several synthetic AD graphs and an AD attack graph collected from a real organization. 

---
# Improving Large Language Model Planning with Action Sequence Similarity 

**Authors**: Xinran Zhao, Hanie Sedghi, Bernd Bohnet, Dale Schuurmans, Azade Nova  

**Link**: [PDF](https://arxiv.org/pdf/2505.01009)  

**Abstract**: Planning is essential for artificial intelligence systems to look ahead and proactively determine a course of actions to reach objectives in the virtual and real world. Recent work on large language models (LLMs) sheds light on their planning capability in various tasks. However, it remains unclear what signals in the context influence the model performance. In this work, we explore how to improve the model planning capability through in-context learning (ICL), specifically, what signals can help select the exemplars. Through extensive experiments, we observe that commonly used problem similarity may result in false positives with drastically different plans, which can mislead the model. In response, we propose to sample and filter exemplars leveraging plan side action sequence similarity (AS). We propose GRASE-DC: a two-stage pipeline that first re-samples high AS exemplars and then curates the selected exemplars with dynamic clustering on AS to achieve a balance of relevance and diversity. Our experimental result confirms that GRASE-DC achieves significant performance improvement on various planning tasks (up to ~11-40 point absolute accuracy improvement with 27.3% fewer exemplars needed on average). With GRASE-DC* + VAL, where we iteratively apply GRASE-DC with a validator, we are able to even boost the performance by 18.9% more.
Extensive analysis validates the consistent performance improvement of GRASE-DC with various backbone LLMs and on both classical planning and natural language planning benchmarks. GRASE-DC can further boost the planning accuracy by ~24 absolute points on harder problems using simpler problems as exemplars over a random baseline. This demonstrates its ability to generalize to out-of-distribution problems. 

---
# Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models 

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.00972)  

**Abstract**: Simulation-based testing is crucial for validating autonomous vehicles (AVs), yet existing scenario generation methods either overfit to common driving patterns or operate in an offline, non-interactive manner that fails to expose rare, safety-critical corner cases. In this paper, we introduce an online, retrieval-augmented large language model (LLM) framework for generating safety-critical driving scenarios. Our method first employs an LLM-based behavior analyzer to infer the most dangerous intent of the background vehicle from the observed state, then queries additional LLM agents to synthesize feasible adversarial trajectories. To mitigate catastrophic forgetting and accelerate adaptation, we augment the framework with a dynamic memorization and retrieval bank of intent-planner pairs, automatically expanding its behavioral library when novel intents arise. Evaluations using the Waymo Open Motion Dataset demonstrate that our model reduces the mean minimum time-to-collision from 1.62 to 1.08 s and incurs a 75% collision rate, substantially outperforming baselines. 

---
# Car Sensors Health Monitoring by Verification Based on Autoencoder and Random Forest Regression 

**Authors**: Sahar Torkhesari, Behnam Yousefimehr, Mehdi Ghatee  

**Link**: [PDF](https://arxiv.org/pdf/2505.00876)  

**Abstract**: Driver assistance systems provide a wide range of crucial services, including closely monitoring the condition of vehicles. This paper showcases a groundbreaking sensor health monitoring system designed for the automotive industry. The ingenious system leverages cutting-edge techniques to process data collected from various vehicle sensors. It compares their outputs within the Electronic Control Unit (ECU) to evaluate the health of each sensor. To unravel the intricate correlations between sensor data, an extensive exploration of machine learning and deep learning methodologies was conducted. Through meticulous analysis, the most correlated sensor data were identified. These valuable insights were then utilized to provide accurate estimations of sensor values. Among the diverse learning methods examined, the combination of autoencoders for detecting sensor failures and random forest regression for estimating sensor values proved to yield the most impressive outcomes. A statistical model using the normal distribution has been developed to identify possible sensor failures proactively. By comparing the actual values of the sensors with their estimated values based on correlated sensors, faulty sensors can be detected early. When a defective sensor is detected, both the driver and the maintenance department are promptly alerted. Additionally, the system replaces the value of the faulty sensor with the estimated value obtained through analysis. This proactive approach was evaluated using data from twenty essential sensors in the Saipa's Quick vehicle's ECU, resulting in an impressive accuracy rate of 99\%. 

---
# Thoughts without Thinking: Reconsidering the Explanatory Value of Chain-of-Thought Reasoning in LLMs through Agentic Pipelines 

**Authors**: Ramesh Manuvinakurike, Emanuel Moss, Elizabeth Anne Watkins, Saurav Sahay, Giuseppe Raffa, Lama Nachman  

**Link**: [PDF](https://arxiv.org/pdf/2505.00875)  

**Abstract**: Agentic pipelines present novel challenges and opportunities for human-centered explainability. The HCXAI community is still grappling with how best to make the inner workings of LLMs transparent in actionable ways. Agentic pipelines consist of multiple LLMs working in cooperation with minimal human control. In this research paper, we present early findings from an agentic pipeline implementation of a perceptive task guidance system. Through quantitative and qualitative analysis, we analyze how Chain-of-Thought (CoT) reasoning, a common vehicle for explainability in LLMs, operates within agentic pipelines. We demonstrate that CoT reasoning alone does not lead to better outputs, nor does it offer explainability, as it tends to produce explanations without explainability, in that they do not improve the ability of end users to better understand systems or achieve their goals. 

---
# MIMIC-\RNum{4}-Ext-22MCTS: A 22 Millions-Event Temporal Clinical Time-Series Dataset with Relative Timestamp for Risk Prediction 

**Authors**: Jing Wang, Xing Niu, Juyong Kim, Jie Shen, Tong Zhang, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.00827)  

**Abstract**: Clinical risk prediction based on machine learning algorithms plays a vital role in modern healthcare. A crucial component in developing a reliable prediction model is collecting high-quality time series clinical events. In this work, we release such a dataset that consists of 22,588,586 Clinical Time Series events, which we term MIMIC-\RNum{4}-Ext-22MCTS. Our source data are discharge summaries selected from the well-known yet unstructured MIMIC-IV-Note \cite{Johnson2023-pg}. We then extract clinical events as short text span from the discharge summaries, along with the timestamps of these events as temporal information. The general-purpose MIMIC-IV-Note pose specific challenges for our work: it turns out that the discharge summaries are too lengthy for typical natural language models to process, and the clinical events of interest often are not accompanied with explicit timestamps. Therefore, we propose a new framework that works as follows: 1) we break each discharge summary into manageably small text chunks; 2) we apply contextual BM25 and contextual semantic search to retrieve chunks that have a high potential of containing clinical events; and 3) we carefully design prompts to teach the recently released Llama-3.1-8B \cite{touvron2023llama} model to identify or infer temporal information of the chunks. We show that the obtained dataset is so informative and transparent that standard models fine-tuned on our dataset are achieving significant improvements in healthcare applications. In particular, the BERT model fine-tuned based on our dataset achieves 10\% improvement in accuracy on medical question answering task, and 3\% improvement in clinical trial matching task compared with the classic BERT. The GPT-2 model, fine-tuned on our dataset, produces more clinically reliable results for clinical questions. 

---
# Explanations as Bias Detectors: A Critical Study of Local Post-hoc XAI Methods for Fairness Exploration 

**Authors**: Vasiliki Papanikou, Danae Pla Karidi, Evaggelia Pitoura, Emmanouil Panagiotou, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00802)  

**Abstract**: As Artificial Intelligence (AI) is increasingly used in areas that significantly impact human lives, concerns about fairness and transparency have grown, especially regarding their impact on protected groups. Recently, the intersection of explainability and fairness has emerged as an important area to promote responsible AI systems. This paper explores how explainability methods can be leveraged to detect and interpret unfairness. We propose a pipeline that integrates local post-hoc explanation methods to derive fairness-related insights. During the pipeline design, we identify and address critical questions arising from the use of explanations as bias detectors such as the relationship between distributive and procedural fairness, the effect of removing the protected attribute, the consistency and quality of results across different explanation methods, the impact of various aggregation strategies of local explanations on group fairness evaluations, and the overall trustworthiness of explanations as bias detectors. Our results show the potential of explanation methods used for fairness while highlighting the need to carefully consider the aforementioned critical aspects. 

---
# Howard's Policy Iteration is Subexponential for Deterministic Markov Decision Problems with Rewards of Fixed Bit-size and Arbitrary Discount Factor 

**Authors**: Dibyangshu Mukherjee, Shivaram Kalyanakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00795)  

**Abstract**: Howard's Policy Iteration (HPI) is a classic algorithm for solving Markov Decision Problems (MDPs). HPI uses a "greedy" switching rule to update from any non-optimal policy to a dominating one, iterating until an optimal policy is found. Despite its introduction over 60 years ago, the best-known upper bounds on HPI's running time remain exponential in the number of states -- indeed even on the restricted class of MDPs with only deterministic transitions (DMDPs). Meanwhile, the tightest lower bound for HPI for MDPs with a constant number of actions per state is only linear. In this paper, we report a significant improvement: a subexponential upper bound for HPI on DMDPs, which is parameterised by the bit-size of the rewards, while independent of the discount factor. The same upper bound also applies to DMDPs with only two possible rewards (which may be of arbitrary size). 

---
# ROSA: A Knowledge-based Solution for Robot Self-Adaptation 

**Authors**: Gustavo Rezende Silva, Juliane Päßler, S. Lizeth Tapia Tarifa, Einar Broch Johnsen, Carlos Hernández Corbato  

**Link**: [PDF](https://arxiv.org/pdf/2505.00733)  

**Abstract**: Autonomous robots must operate in diverse environments and handle multiple tasks despite uncertainties. This creates challenges in designing software architectures and task decision-making algorithms, as different contexts may require distinct task logic and architectural configurations. To address this, robotic systems can be designed as self-adaptive systems capable of adapting their task execution and software architecture at runtime based on their this http URL paper introduces ROSA, a novel knowledge-based framework for RObot Self-Adaptation, which enables task-and-architecture co-adaptation (TACA) in robotic systems. ROSA achieves this by providing a knowledge model that captures all application-specific knowledge required for adaptation and by reasoning over this knowledge at runtime to determine when and how adaptation should occur. In addition to a conceptual framework, this work provides an open-source ROS 2-based reference implementation of ROSA and evaluates its feasibility and performance in an underwater robotics application. Experimental results highlight ROSA's advantages in reusability and development effort for designing self-adaptive robotic systems. 

---
# GENMO: A GENeralist Model for Human MOtion 

**Authors**: Jiefeng Li, Jinkun Cao, Haotian Zhang, Davis Rempe, Jan Kautz, Umar Iqbal, Ye Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.01425)  

**Abstract**: Human motion modeling traditionally separates motion generation and estimation into distinct tasks with specialized models. Motion generation models focus on creating diverse, realistic motions from inputs like text, audio, or keyframes, while motion estimation models aim to reconstruct accurate motion trajectories from observations like videos. Despite sharing underlying representations of temporal dynamics and kinematics, this separation limits knowledge transfer between tasks and requires maintaining separate models. We present GENMO, a unified Generalist Model for Human Motion that bridges motion estimation and generation in a single framework. Our key insight is to reformulate motion estimation as constrained motion generation, where the output motion must precisely satisfy observed conditioning signals. Leveraging the synergy between regression and diffusion, GENMO achieves accurate global motion estimation while enabling diverse motion generation. We also introduce an estimation-guided training objective that exploits in-the-wild videos with 2D annotations and text descriptions to enhance generative diversity. Furthermore, our novel architecture handles variable-length motions and mixed multimodal conditions (text, audio, video) at different time intervals, offering flexible control. This unified approach creates synergistic benefits: generative priors improve estimated motions under challenging conditions like occlusions, while diverse video data enhances generation capabilities. Extensive experiments demonstrate GENMO's effectiveness as a generalist framework that successfully handles multiple human motion tasks within a single model. 

---
# SIME: Enhancing Policy Self-Improvement with Modal-level Exploration 

**Authors**: Yang Jin, Jun Lv, Wenye Yu, Hongjie Fang, Yong-Lu Li, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.01396)  

**Abstract**: Self-improvement requires robotic systems to initially learn from human-provided data and then gradually enhance their capabilities through interaction with the environment. This is similar to how humans improve their skills through continuous practice. However, achieving effective self-improvement is challenging, primarily because robots tend to repeat their existing abilities during interactions, often failing to generate new, valuable data for learning. In this paper, we identify the key to successful self-improvement: modal-level exploration and data selection. By incorporating a modal-level exploration mechanism during policy execution, the robot can produce more diverse and multi-modal interactions. At the same time, we select the most valuable trials and high-quality segments from these interactions for learning. We successfully demonstrate effective robot self-improvement on both simulation benchmarks and real-world experiments. The capability for self-improvement will enable us to develop more robust and high-success-rate robotic control strategies at a lower cost. Our code and experiment scripts are available at this https URL 

---
# Multimodal Doctor-in-the-Loop: A Clinically-Guided Explainable Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer 

**Authors**: Alice Natalina Caragliano, Claudia Tacconi, Carlo Greco, Lorenzo Nibid, Edy Ippolito, Michele Fiore, Giuseppe Perrone, Sara Ramella, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01390)  

**Abstract**: This study proposes a novel approach combining Multimodal Deep Learning with intrinsic eXplainable Artificial Intelligence techniques to predict pathological response in non-small cell lung cancer patients undergoing neoadjuvant therapy. Due to the limitations of existing radiomics and unimodal deep learning approaches, we introduce an intermediate fusion strategy that integrates imaging and clinical data, enabling efficient interaction between data modalities. The proposed Multimodal Doctor-in-the-Loop method further enhances clinical relevance by embedding clinicians' domain knowledge directly into the training process, guiding the model's focus gradually from broader lung regions to specific lesions. Results demonstrate improved predictive accuracy and explainability, providing insights into optimal data integration strategies for clinical applications. 

---
# FalconWing: An Open-Source Platform for Ultra-Light Fixed-Wing Aircraft Research 

**Authors**: Yan Miao, Will Shen, Hang Cui, Sayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2505.01383)  

**Abstract**: We present FalconWing -- an open-source, ultra-lightweight (150 g) fixed-wing platform for autonomy research. The hardware platform integrates a small camera, a standard airframe, offboard computation, and radio communication for manual overrides. We demonstrate FalconWing's capabilities by developing and deploying a purely vision-based control policy for autonomous landing (without IMU or motion capture) using a novel real-to-sim-to-real learning approach. Our learning approach: (1) constructs a photorealistic simulation environment via 3D Gaussian splatting trained on real-world images; (2) identifies nonlinear dynamics from vision-estimated real-flight data; and (3) trains a multi-modal Vision Transformer (ViT) policy through simulation-only imitation learning. The ViT architecture fuses single RGB image with the history of control actions via self-attention, preserving temporal context while maintaining real-time 20 Hz inference. When deployed zero-shot on the hardware platform, this policy achieves an 80% success rate in vision-based autonomous landings. Together with the hardware specifications, we also open-source the system dynamics, the software for photorealistic simulator and the learning approach. 

---
# Evaluating Explanations: An Explanatory Virtues Framework for Mechanistic Interpretability -- The Strange Science Part I.ii 

**Authors**: Kola Ayonrinde, Louis Jaburi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01372)  

**Abstract**: Mechanistic Interpretability (MI) aims to understand neural networks through causal explanations. Though MI has many explanation-generating methods, progress has been limited by the lack of a universal approach to evaluating explanations. Here we analyse the fundamental question "What makes a good explanation?" We introduce a pluralist Explanatory Virtues Framework drawing on four perspectives from the Philosophy of Science - the Bayesian, Kuhnian, Deutschian, and Nomological - to systematically evaluate and improve explanations in MI. We find that Compact Proofs consider many explanatory virtues and are hence a promising approach. Fruitful research directions implied by our framework include (1) clearly defining explanatory simplicity, (2) focusing on unifying explanations and (3) deriving universal principles for neural networks. Improved MI methods enhance our ability to monitor, predict, and steer AI systems. 

---
# Differentiable Nonlinear Model Predictive Control 

**Authors**: Jonathan Frey, Katrin Baumgärtner, Gianluca Frison, Dirk Reinhardt, Jasper Hoffmann, Leonard Fichtner, Sebastien Gros, Moritz Diehl  

**Link**: [PDF](https://arxiv.org/pdf/2505.01353)  

**Abstract**: The efficient computation of parametric solution sensitivities is a key challenge in the integration of learning-enhanced methods with nonlinear model predictive control (MPC), as their availability is crucial for many learning algorithms. While approaches presented in the machine learning community are limited to convex or unconstrained formulations, this paper discusses the computation of solution sensitivities of general nonlinear programs (NLPs) using the implicit function theorem (IFT) and smoothed optimality conditions treated in interior-point methods (IPM). We detail sensitivity computation within a sequential quadratic programming (SQP) method which employs an IPM for the quadratic subproblems. The publication is accompanied by an efficient open-source implementation within the framework, providing both forward and adjoint sensitivities for general optimal control problems, achieving speedups exceeding 3x over the state-of-the-art solver this http URL. 

---
# Constrained Network Adversarial Attacks: Validity, Robustness, and Transferability 

**Authors**: Anass Grini, Oumaima Taheri, Btissam El Khamlichi, Amal El Fallah-Seghrouchni  

**Link**: [PDF](https://arxiv.org/pdf/2505.01328)  

**Abstract**: While machine learning has significantly advanced Network Intrusion Detection Systems (NIDS), particularly within IoT environments where devices generate large volumes of data and are increasingly susceptible to cyber threats, these models remain vulnerable to adversarial attacks. Our research reveals a critical flaw in existing adversarial attack methodologies: the frequent violation of domain-specific constraints, such as numerical and categorical limits, inherent to IoT and network traffic. This leads to up to 80.3% of adversarial examples being invalid, significantly overstating real-world vulnerabilities. These invalid examples, though effective in fooling models, do not represent feasible attacks within practical IoT deployments. Consequently, relying on these results can mislead resource allocation for defense, inflating the perceived susceptibility of IoT-enabled NIDS models to adversarial manipulation. Furthermore, we demonstrate that simpler surrogate models like Multi-Layer Perceptron (MLP) generate more valid adversarial examples compared to complex architectures such as CNNs and LSTMs. Using the MLP as a surrogate, we analyze the transferability of adversarial severity to other ML/DL models commonly used in IoT contexts. This work underscores the importance of considering both domain constraints and model architecture when evaluating and designing robust ML/DL models for security-critical IoT and network applications. 

---
# Helping Big Language Models Protect Themselves: An Enhanced Filtering and Summarization System 

**Authors**: Sheikh Samit Muhaimin, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.01315)  

**Abstract**: The recent growth in the use of Large Language Models has made them vulnerable to sophisticated adversarial assaults, manipulative prompts, and encoded malicious inputs. Existing countermeasures frequently necessitate retraining models, which is computationally costly and impracticable for deployment. Without the need for retraining or fine-tuning, this study presents a unique defense paradigm that allows LLMs to recognize, filter, and defend against adversarial or malicious inputs on their own. There are two main parts to the suggested framework: (1) A prompt filtering module that uses sophisticated Natural Language Processing (NLP) techniques, including zero-shot classification, keyword analysis, and encoded content detection (e.g. base64, hexadecimal, URL encoding), to detect, decode, and classify harmful inputs; and (2) A summarization module that processes and summarizes adversarial research literature to give the LLM context-aware defense knowledge. This approach strengthens LLMs' resistance to adversarial exploitation by fusing text extraction, summarization, and harmful prompt analysis. According to experimental results, this integrated technique has a 98.71% success rate in identifying harmful patterns, manipulative language structures, and encoded prompts. By employing a modest amount of adversarial research literature as context, the methodology also allows the model to react correctly to harmful inputs with a larger percentage of jailbreak resistance and refusal rate. While maintaining the quality of LLM responses, the framework dramatically increases LLM's resistance to hostile misuse, demonstrating its efficacy as a quick and easy substitute for time-consuming, retraining-based defenses. 

---
# Enhancing SPARQL Query Rewriting for Complex Ontology Alignments 

**Authors**: Anicet Lepetit Ondo, Laurence Capus, Mamadou Bousso  

**Link**: [PDF](https://arxiv.org/pdf/2505.01309)  

**Abstract**: SPARQL query rewriting is a fundamental mechanism for uniformly querying heterogeneous ontologies in the Linked Data Web. However, the complexity of ontology alignments, particularly rich correspondences (c : c), makes this process challenging. Existing approaches primarily focus on simple (s : s) and partially complex ( s : c) alignments, thereby overlooking the challenges posed by more expressive alignments. Moreover, the intricate syntax of SPARQL presents a barrier for non-expert users seeking to fully exploit the knowledge encapsulated in ontologies. This article proposes an innovative approach for the automatic rewriting of SPARQL queries from a source ontology to a target ontology, based on a user's need expressed in natural language. It leverages the principles of equivalence transitivity as well as the advanced capabilities of large language models such as GPT-4. By integrating these elements, this approach stands out for its ability to efficiently handle complex alignments, particularly (c : c) correspondences , by fully exploiting their expressiveness. Additionally, it facilitates access to aligned ontologies for users unfamiliar with SPARQL, providing a flexible solution for querying heterogeneous data. 

---
# Document Retrieval Augmented Fine-Tuning (DRAFT) for safety-critical software assessments 

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Vanessa Vulovic, Gary Bamford, Dan Basher, Howard Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.01307)  

**Abstract**: Safety critical software assessment requires robust assessment against complex regulatory frameworks, a process traditionally limited by manual evaluation. This paper presents Document Retrieval-Augmented Fine-Tuning (DRAFT), a novel approach that enhances the capabilities of a large language model (LLM) for safety-critical compliance assessment. DRAFT builds upon existing Retrieval-Augmented Generation (RAG) techniques by introducing a novel fine-tuning framework that accommodates our dual-retrieval architecture, which simultaneously accesses both software documentation and applicable reference standards. To fine-tune DRAFT, we develop a semi-automated dataset generation methodology that incorporates variable numbers of relevant documents with meaningful distractors, closely mirroring real-world assessment scenarios. Experiments with GPT-4o-mini demonstrate a 7% improvement in correctness over the baseline model, with qualitative improvements in evidence handling, response structure, and domain-specific reasoning. DRAFT represents a practical approach to improving compliance assessment systems while maintaining the transparency and evidence-based reasoning essential in regulatory domains. 

---
# ViSA-Flow: Accelerating Robot Skill Learning via Large-Scale Video Semantic Action Flow 

**Authors**: Changhe Chen, Quantao Yang, Xiaohao Xu, Nima Fazeli, Olov Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2505.01288)  

**Abstract**: One of the central challenges preventing robots from acquiring complex manipulation skills is the prohibitive cost of collecting large-scale robot demonstrations. In contrast, humans are able to learn efficiently by watching others interact with their environment. To bridge this gap, we introduce semantic action flow as a core intermediate representation capturing the essential spatio-temporal manipulator-object interactions, invariant to superficial visual differences. We present ViSA-Flow, a framework that learns this representation self-supervised from unlabeled large-scale video data. First, a generative model is pre-trained on semantic action flows automatically extracted from large-scale human-object interaction video data, learning a robust prior over manipulation structure. Second, this prior is efficiently adapted to a target robot by fine-tuning on a small set of robot demonstrations processed through the same semantic abstraction pipeline. We demonstrate through extensive experiments on the CALVIN benchmark and real-world tasks that ViSA-Flow achieves state-of-the-art performance, particularly in low-data regimes, outperforming prior methods by effectively transferring knowledge from human video observation to robotic execution. Videos are available at this https URL. 

---
# 2DXformer: Dual Transformers for Wind Power Forecasting with Dual Exogenous Variables 

**Authors**: Yajuan Zhang, Jiahai Jiang, Yule Yan, Liang Yang, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01286)  

**Abstract**: Accurate wind power forecasting can help formulate scientific dispatch plans, which is of great significance for maintaining the safety, stability, and efficient operation of the power system. In recent years, wind power forecasting methods based on deep learning have focused on extracting the spatiotemporal correlations among data, achieving significant improvements in forecasting accuracy. However, they exhibit two limitations. First, there is a lack of modeling for the inter-variable relationships, which limits the accuracy of the forecasts. Second, by treating endogenous and exogenous variables equally, it leads to unnecessary interactions between the endogenous and exogenous variables, increasing the complexity of the model. In this paper, we propose the 2DXformer, which, building upon the previous work's focus on spatiotemporal correlations, addresses the aforementioned two limitations. Specifically, we classify the inputs of the model into three types: exogenous static variables, exogenous dynamic variables, and endogenous variables. First, we embed these variables as variable tokens in a channel-independent manner. Then, we use the attention mechanism to capture the correlations among exogenous variables. Finally, we employ a multi-layer perceptron with residual connections to model the impact of exogenous variables on endogenous variables. Experimental results on two real-world large-scale datasets indicate that our proposed 2DXformer can further improve the performance of wind power forecasting. The code is available in this repository: \href{this https URL}{this https URL}. 

---
# Reduced-order structure-property linkages for stochastic metamaterials 

**Authors**: Hooman Danesh, Maruthi Annamaraju, Tim Brepols, Stefanie Reese, Surya R. Kalidindi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01283)  

**Abstract**: The capabilities of additive manufacturing have facilitated the design and production of mechanical metamaterials with diverse unit cell geometries. Establishing linkages between the vast design space of unit cells and their effective mechanical properties is critical for the efficient design and performance evaluation of such metamaterials. However, physics-based simulations of metamaterial unit cells across the entire design space are computationally expensive, necessitating a materials informatics framework to efficiently capture complex structure-property relationships. In this work, principal component analysis of 2-point correlation functions is performed to extract the salient features from a large dataset of randomly generated 2D metamaterials. Physics-based simulations are performed using a fast Fourier transform (FFT)-based homogenization approach to efficiently compute the homogenized effective elastic stiffness across the extensive unit cell designs. Subsequently, Gaussian process regression is used to generate reduced-order surrogates, mapping unit cell designs to their homogenized effective elastic constant. It is demonstrated that the adopted workflow enables a high-value low-dimensional representation of the voluminous stochastic metamaterial dataset, facilitating the construction of robust structure-property maps. Finally, an uncertainty-based active learning framework is utilized to train a surrogate model with a significantly smaller number of data points compared to the original full dataset. It is shown that a dataset as small as $0.61\%$ of the entire dataset is sufficient to generate accurate and robust structure-property maps. 

---
# A Physics-preserved Transfer Learning Method for Differential Equations 

**Authors**: Hao-Ran Yang, Chuan-Xian Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.01281)  

**Abstract**: While data-driven methods such as neural operator have achieved great success in solving differential equations (DEs), they suffer from domain shift problems caused by different learning environments (with data bias or equation changes), which can be alleviated by transfer learning (TL). However, existing TL methods adopted in DEs problems lack either generalizability in general DEs problems or physics preservation during training. In this work, we focus on a general transfer learning method that adaptively correct the domain shift and preserve physical information. Mathematically, we characterize the data domain as product distribution and the essential problems as distribution bias and operator bias. A Physics-preserved Optimal Tensor Transport (POTT) method that simultaneously admits generalizability to common DEs and physics preservation of specific problem is proposed to adapt the data-driven model to target domain utilizing the push-forward distribution induced by the POTT map. Extensive experiments demonstrate the superior performance, generalizability and physics preservation of the proposed POTT method. 

---
# Anti-adversarial Learning: Desensitizing Prompts for Large Language Models 

**Authors**: Xuan Li, Zhe Yin, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.01273)  

**Abstract**: With the widespread use of LLMs, preserving privacy in user prompts has become crucial, as prompts risk exposing privacy and sensitive data to the cloud LLMs. Traditional techniques like homomorphic encryption, secure multi-party computation, and federated learning face challenges due to heavy computational costs and user participation requirements, limiting their applicability in LLM scenarios. In this paper, we propose PromptObfus, a novel method for desensitizing LLM prompts. The core idea of PromptObfus is "anti-adversarial" learning, which perturbs privacy words in the prompt to obscure sensitive information while retaining the stability of model predictions. Specifically, PromptObfus frames prompt desensitization as a masked language modeling task, replacing privacy-sensitive terms with a [MASK] token. A desensitization model is trained to generate candidate replacements for each masked position. These candidates are subsequently selected based on gradient feedback from a surrogate model, ensuring minimal disruption to the task output. We demonstrate the effectiveness of our approach on three NLP tasks. Results show that PromptObfus effectively prevents privacy inference from remote LLMs while preserving task performance. 

---
# Enhancing Obsolescence Forecasting with Deep Generative Data Augmentation: A Semi-Supervised Framework for Low-Data Industrial Applications 

**Authors**: Elie Saad, Mariem Besbes, Marc Zolghadri, Victor Czmil, Claude Baron, Vincent Bourgeois  

**Link**: [PDF](https://arxiv.org/pdf/2505.01261)  

**Abstract**: The challenge of electronic component obsolescence is particularly critical in systems with long life cycles. Various obsolescence management methods are employed to mitigate its impact, with obsolescence forecasting being a highly sought-after and prominent approach. As a result, numerous machine learning-based forecasting methods have been proposed. However, machine learning models require a substantial amount of relevant data to achieve high precision, which is lacking in the current obsolescence landscape in some situations. This work introduces a novel framework for obsolescence forecasting based on deep learning. The proposed framework solves the lack of available data through deep generative modeling, where new obsolescence cases are generated and used to augment the training dataset. The augmented dataset is then used to train a classical machine learning-based obsolescence forecasting model. To train classical forecasting models using augmented datasets, existing classical supervised-learning classifiers are adapted for semi-supervised learning within this framework. The proposed framework demonstrates state-of-the-art results on benchmarking datasets. 

---
# EvalxNLP: A Framework for Benchmarking Post-Hoc Explainability Methods on NLP Models 

**Authors**: Mahdi Dhaini, Kafaite Zahra Hussain, Efstratios Zaradoukas, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.01238)  

**Abstract**: As Natural Language Processing (NLP) models continue to evolve and become integral to high-stakes applications, ensuring their interpretability remains a critical challenge. Given the growing variety of explainability methods and diverse stakeholder requirements, frameworks that help stakeholders select appropriate explanations tailored to their specific use cases are increasingly important. To address this need, we introduce EvalxNLP, a Python framework for benchmarking state-of-the-art feature attribution methods for transformer-based NLP models. EvalxNLP integrates eight widely recognized explainability techniques from the Explainable AI (XAI) literature, enabling users to generate and evaluate explanations based on key properties such as faithfulness, plausibility, and complexity. Our framework also provides interactive, LLM-based textual explanations, facilitating user understanding of the generated explanations and evaluation outcomes. Human evaluation results indicate high user satisfaction with EvalxNLP, suggesting it is a promising framework for benchmarking explanation methods across diverse user groups. By offering a user-friendly and extensible platform, EvalxNLP aims at democratizing explainability tools and supporting the systematic comparison and advancement of XAI techniques in NLP. 

---
# Gender Bias in Explainability: Investigating Performance Disparity in Post-hoc Methods 

**Authors**: Mahdi Dhaini, Ege Erdogan, Nils Feldhus, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.01198)  

**Abstract**: While research on applications and evaluations of explanation methods continues to expand, fairness of the explanation methods concerning disparities in their performance across subgroups remains an often overlooked aspect. In this paper, we address this gap by showing that, across three tasks and five language models, widely used post-hoc feature attribution methods exhibit significant gender disparity with respect to their faithfulness, robustness, and complexity. These disparities persist even when the models are pre-trained or fine-tuned on particularly unbiased datasets, indicating that the disparities we observe are not merely consequences of biased training data. Our results highlight the importance of addressing disparities in explanations when developing and applying explainability methods, as these can lead to biased outcomes against certain subgroups, with particularly critical implications in high-stakes contexts. Furthermore, our findings underscore the importance of incorporating the fairness of explanations, alongside overall model fairness and explainability, as a requirement in regulatory frameworks. 

---
# Secure Cluster-Based Hierarchical Federated Learning in Vehicular Networks 

**Authors**: M. Saeid HaghighiFard, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2505.01186)  

**Abstract**: Hierarchical Federated Learning (HFL) has recently emerged as a promising solution for intelligent decision-making in vehicular networks, helping to address challenges such as limited communication resources, high vehicle mobility, and data heterogeneity. However, HFL remains vulnerable to adversarial and unreliable vehicles, whose misleading updates can significantly compromise the integrity and convergence of the global model. To address these challenges, we propose a novel defense framework that integrates dynamic vehicle selection with robust anomaly detection within a cluster-based HFL architecture, specifically designed to counter Gaussian noise and gradient ascent attacks. The framework performs a comprehensive reliability assessment for each vehicle by evaluating historical accuracy, contribution frequency, and anomaly records. Anomaly detection combines Z-score and cosine similarity analyses on model updates to identify both statistical outliers and directional deviations in model updates. To further refine detection, an adaptive thresholding mechanism is incorporated into the cosine similarity metric, dynamically adjusting the threshold based on the historical accuracy of each vehicle to enforce stricter standards for consistently high-performing vehicles. In addition, a weighted gradient averaging mechanism is implemented, which assigns higher weights to gradient updates from more trustworthy vehicles. To defend against coordinated attacks, a cross-cluster consistency check is applied to identify collaborative attacks in which multiple compromised clusters coordinate misleading updates. Together, these mechanisms form a multi-level defense strategy to filter out malicious contributions effectively. Simulation results show that the proposed algorithm significantly reduces convergence time compared to benchmark methods across both 1-hop and 3-hop topologies. 

---
# EnviKal-Loc: Sub-10m Indoor LoRaWAN Localization using an Environmental-Aware Path Loss and Adaptive RSSI Smoothing 

**Authors**: Nahshon Mokua Obiri, Kristof Van Laerhoven  

**Link**: [PDF](https://arxiv.org/pdf/2505.01185)  

**Abstract**: LoRaWAN technology's extensive coverage positions it as a strong contender for large-scale IoT deployments. However, achieving sub-10 m accuracy in indoor localization remains challenging due to complex environmental conditions, multipath fading, and transient obstructions. This paper proposes a lightweight but robust approach combining adaptive filtering with an extended log-distance, multi-wall path loss and shadowing (PLS) model. Our methodology augments conventional models with critical LoRaWAN parameters (received signal strength indicator (RSSI), frequency, and signal-to-noise ratio (SNR)) and dynamic environmental indicators (temperature, humidity, carbon dioxide, particulate matter, and barometric pressure). An adaptive Kalman filter reduces RSSI fluctuations, isolating persistent trends from momentary noise. Using a six-month dataset of 1,328,334 field measurements, we evaluate three models: the baseline COST 231 multi-wall model (MWM), the baseline model augmented with environmental parameters (MWM-EP), and a forward-only adaptive Kalman-filtered RSSI version of the latter (MWM-EP-KF). Results confirm that the MWM-EP-KF achieves a mean absolute error (MAE) of 5.81 m, outperforming both the MWM-EP (10.56 m) and the baseline MWM framework (17.98 m). Environmental augmentation reduces systematic errors by 41.22%, while Kalman filtering significantly enhances robustness under high RSSI volatility by 42.63%, on average across all devices. These findings present an interpretable, efficient solution for precise indoor LoRaWAN localization in dynamically changing environments. 

---
# TSTMotion: Training-free Scene-awarenText-to-motion Generation 

**Authors**: Ziyan Guo, Haoxuan Qu, Hossein Rahmani, Dewen Soh, Ping Hu, Qiuhong Ke, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.01182)  

**Abstract**: Text-to-motion generation has recently garnered significant research interest, primarily focusing on generating human motion sequences in blank backgrounds. However, human motions commonly occur within diverse 3D scenes, which has prompted exploration into scene-aware text-to-motion generation methods. Yet, existing scene-aware methods often rely on large-scale ground-truth motion sequences in diverse 3D scenes, which poses practical challenges due to the expensive cost. To mitigate this challenge, we are the first to propose a \textbf{T}raining-free \textbf{S}cene-aware \textbf{T}ext-to-\textbf{Motion} framework, dubbed as \textbf{TSTMotion}, that efficiently empowers pre-trained blank-background motion generators with the scene-aware capability. Specifically, conditioned on the given 3D scene and text description, we adopt foundation models together to reason, predict and validate a scene-aware motion guidance. Then, the motion guidance is incorporated into the blank-background motion generators with two modifications, resulting in scene-aware text-driven motion sequences. Extensive experiments demonstrate the efficacy and generalizability of our proposed framework. We release our code in \href{this https URL}{Project Page}. 

---
# LLM Security: Vulnerabilities, Attacks, Defenses, and Countermeasures 

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal  

**Link**: [PDF](https://arxiv.org/pdf/2505.01177)  

**Abstract**: As large language models (LLMs) continue to evolve, it is critical to assess the security threats and vulnerabilities that may arise both during their training phase and after models have been deployed. This survey seeks to define and categorize the various attacks targeting LLMs, distinguishing between those that occur during the training phase and those that affect already trained models. A thorough analysis of these attacks is presented, alongside an exploration of defense mechanisms designed to mitigate such threats. Defenses are classified into two primary categories: prevention-based and detection-based defenses. Furthermore, our survey summarizes possible attacks and their corresponding defense strategies. It also provides an evaluation of the effectiveness of the known defense mechanisms for the different security threats. Our survey aims to offer a structured framework for securing LLMs, while also identifying areas that require further research to improve and strengthen defenses against emerging security challenges. 

---
# Distilling Two-Timed Flow Models by Separately Matching Initial and Terminal Velocities 

**Authors**: Pramook Khungurn, Pratch Piyawongwisal, Sira Sriswadi, Supasorn Suwajanakorn  

**Link**: [PDF](https://arxiv.org/pdf/2505.01169)  

**Abstract**: A flow matching model learns a time-dependent vector field $v_t(x)$ that generates a probability path $\{ p_t \}_{0 \leq t \leq 1}$ that interpolates between a well-known noise distribution ($p_0$) and the data distribution ($p_1$). It can be distilled into a \emph{two-timed flow model} (TTFM) $\phi_{s,x}(t)$ that can transform a sample belonging to the distribution at an initial time $s$ to another belonging to the distribution at a terminal time $t$ in one function evaluation. We present a new loss function for TTFM distillation called the \emph{initial/terminal velocity matching} (ITVM) loss that extends the Lagrangian Flow Map Distillation (LFMD) loss proposed by Boffi et al. by adding redundant terms to match the initial velocities at time $s$, removing the derivative from the terminal velocity term at time $t$, and using a version of the model under training, stabilized by exponential moving averaging (EMA), to compute the target terminal average velocity. Preliminary experiments show that our loss leads to better few-step generation performance on multiple types of datasets and model architectures over baselines. 

---
# Harmonizing Intra-coherence and Inter-divergence in Ensemble Attacks for Adversarial Transferability 

**Authors**: Zhaoyang Ma, Zhihao Wu, Wang Lu, Xin Gao, Jinghang Yue, Taolin Zhang, Lipo Wang, Youfang Lin, Jing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01168)  

**Abstract**: The development of model ensemble attacks has significantly improved the transferability of adversarial examples, but this progress also poses severe threats to the security of deep neural networks. Existing methods, however, face two critical challenges: insufficient capture of shared gradient directions across models and a lack of adaptive weight allocation mechanisms. To address these issues, we propose a novel method Harmonized Ensemble for Adversarial Transferability (HEAT), which introduces domain generalization into adversarial example generation for the first time. HEAT consists of two key modules: Consensus Gradient Direction Synthesizer, which uses Singular Value Decomposition to synthesize shared gradient directions; and Dual-Harmony Weight Orchestrator which dynamically balances intra-domain coherence, stabilizing gradients within individual models, and inter-domain diversity, enhancing transferability across models. Experimental results demonstrate that HEAT significantly outperforms existing methods across various datasets and settings, offering a new perspective and direction for adversarial attack research. 

---
# On the Limitations of Steering in Language Model Alignment 

**Authors**: Chebrolu Niranjan, Kokil Jaidka, Gerard Christopher Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01162)  

**Abstract**: Steering vectors are a promising approach to aligning language model behavior at inference time. In this paper, we propose a framework to assess the limitations of steering vectors as alignment mechanisms. Using a framework of transformer hook interventions and antonym-based function vectors, we evaluate the role of prompt structure and context complexity in steering effectiveness. Our findings indicate that steering vectors are promising for specific alignment tasks, such as value alignment, but may not provide a robust foundation for general-purpose alignment in LLMs, particularly in complex scenarios. We establish a methodological foundation for future investigations into steering capabilities of reasoning models. 

---
# Risk Analysis and Design Against Adversarial Actions 

**Authors**: Marco C. Campi, Algo Carè, Luis G. Crespo, Simone Garatti, Federico A. Ramponi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01130)  

**Abstract**: Learning models capable of providing reliable predictions in the face of adversarial actions has become a central focus of the machine learning community in recent years. This challenge arises from observing that data encountered at deployment time often deviate from the conditions under which the model was trained. In this paper, we address deployment-time adversarial actions and propose a versatile, well-principled framework to evaluate the model's robustness against attacks of diverse types and intensities. While we initially focus on Support Vector Regression (SVR), the proposed approach extends naturally to the broad domain of learning via relaxed optimization techniques. Our results enable an assessment of the model vulnerability without requiring additional test data and operate in a distribution-free setup. These results not only provide a tool to enhance trust in the model's applicability but also aid in selecting among competing alternatives. Later in the paper, we show that our findings also offer useful insights for establishing new results within the out-of-distribution framework. 

---
# Self-Supervision Enhances Instance-based Multiple Instance Learning Methods in Digital Pathology: A Benchmark Study 

**Authors**: Ali Mammadov, Loic Le Folgoc, Julien Adam, Anne Buronfosse, Gilles Hayem, Guillaume Hocquet, Pietro Gori  

**Link**: [PDF](https://arxiv.org/pdf/2505.01109)  

**Abstract**: Multiple Instance Learning (MIL) has emerged as the best solution for Whole Slide Image (WSI) classification. It consists of dividing each slide into patches, which are treated as a bag of instances labeled with a global label. MIL includes two main approaches: instance-based and embedding-based. In the former, each patch is classified independently, and then the patch scores are aggregated to predict the bag label. In the latter, bag classification is performed after aggregating patch embeddings. Even if instance-based methods are naturally more interpretable, embedding-based MILs have usually been preferred in the past due to their robustness to poor feature extractors. However, recently, the quality of feature embeddings has drastically increased using self-supervised learning (SSL). Nevertheless, many authors continue to endorse the superiority of embedding-based MIL. To investigate this further, we conduct 710 experiments across 4 datasets, comparing 10 MIL strategies, 6 self-supervised methods with 4 backbones, 4 foundation models, and various pathology-adapted techniques. Furthermore, we introduce 4 instance-based MIL methods never used before in the pathology domain. Through these extensive experiments, we show that with a good SSL feature extractor, simple instance-based MILs, with very few parameters, obtain similar or better performance than complex, state-of-the-art (SOTA) embedding-based MIL methods, setting new SOTA results on the BRACS and Camelyon16 datasets. Since simple instance-based MIL methods are naturally more interpretable and explainable to clinicians, our results suggest that more effort should be put into well-adapted SSL methods for WSI rather than into complex embedding-based MIL methods. 

---
# Multi-Objective Reinforcement Learning for Water Management 

**Authors**: Zuzanna Osika, Roxana Radelescu, Jazmin Zatarain Salazar, Frans Oliehoek, Pradeep K. Murukannaiah  

**Link**: [PDF](https://arxiv.org/pdf/2505.01094)  

**Abstract**: Many real-world problems (e.g., resource management, autonomous driving, drug discovery) require optimizing multiple, conflicting objectives. Multi-objective reinforcement learning (MORL) extends classic reinforcement learning to handle multiple objectives simultaneously, yielding a set of policies that capture various trade-offs. However, the MORL field lacks complex, realistic environments and benchmarks. We introduce a water resource (Nile river basin) management case study and model it as a MORL environment. We then benchmark existing MORL algorithms on this task. Our results show that specialized water management methods outperform state-of-the-art MORL approaches, underscoring the scalability challenges MORL algorithms face in real-world scenarios. 

---
# Any-to-Any Vision-Language Model for Multimodal X-ray Imaging and Radiological Report Generation 

**Authors**: Daniele Molino, Francesco di Feola, Linlin Shen, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01091)  

**Abstract**: Generative models have revolutionized Artificial Intelligence (AI), particularly in multimodal applications. However, adapting these models to the medical domain poses unique challenges due to the complexity of medical data and the stringent need for clinical accuracy. In this work, we introduce a framework specifically designed for multimodal medical data generation. By enabling the generation of multi-view chest X-rays and their associated clinical report, it bridges the gap between general-purpose vision-language models and the specialized requirements of healthcare. Leveraging the MIMIC-CXR dataset, the proposed framework shows superior performance in generating high-fidelity images and semantically coherent reports. Our quantitative evaluation reveals significant results in terms of FID and BLEU scores, showcasing the quality of the generated data. Notably, our framework achieves comparable or even superior performance compared to real data on downstream disease classification tasks, underlining its potential as a tool for medical research and diagnostics. This study highlights the importance of domain-specific adaptations in enhancing the relevance and utility of generative models for clinical applications, paving the way for future advancements in synthetic multimodal medical data generation. 

---
# Artificial Intelligence in Government: Why People Feel They Lose Control 

**Authors**: Alexander Wuttke, Adrian Rauchfleisch, Andreas Jungherr  

**Link**: [PDF](https://arxiv.org/pdf/2505.01085)  

**Abstract**: The use of Artificial Intelligence (AI) in public administration is expanding rapidly, moving from automating routine tasks to deploying generative and agentic systems that autonomously act on goals. While AI promises greater efficiency and responsiveness, its integration into government functions raises concerns about fairness, transparency, and accountability. This article applies principal-agent theory (PAT) to conceptualize AI adoption as a special case of delegation, highlighting three core tensions: assessability (can decisions be understood?), dependency (can the delegation be reversed?), and contestability (can decisions be challenged?). These structural challenges may lead to a "failure-by-success" dynamic, where early functional gains obscure long-term risks to democratic legitimacy. To test this framework, we conducted a pre-registered factorial survey experiment across tax, welfare, and law enforcement domains. Our findings show that although efficiency gains initially bolster trust, they simultaneously reduce citizens' perceived control. When the structural risks come to the foreground, institutional trust and perceived control both drop sharply, suggesting that hidden costs of AI adoption significantly shape public attitudes. The study demonstrates that PAT offers a powerful lens for understanding the institutional and political implications of AI in government, emphasizing the need for policymakers to address delegation risks transparently to maintain public trust. 

---
# Improving Group Fairness in Knowledge Distillation via Laplace Approximation of Early Exits 

**Authors**: Edvin Fasth, Sagar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.01070)  

**Abstract**: Knowledge distillation (KD) has become a powerful tool for training compact student models using larger, pretrained teacher models, often requiring less data and computational resources. Teacher models typically possess more layers and thus exhibit richer feature representations compared to their student counterparts. Furthermore, student models tend to learn simpler, surface-level features in their early layers. This discrepancy can increase errors in groups where labels spuriously correlate with specific input attributes, leading to a decline in group fairness even when overall accuracy remains comparable to the teacher. To mitigate these challenges, Early-Exit Neural Networks (EENNs), which enable predictions at multiple intermediate layers, have been employed. Confidence margins derived from these early exits have been utilized to reweight both cross-entropy and distillation losses on a per-instance basis. In this paper, we propose that leveraging Laplace approximation-based methods to obtain well-calibrated uncertainty estimates can also effectively reweight challenging instances and improve group fairness. We hypothesize that Laplace approximation offers a more robust identification of difficult or ambiguous instances compared to margin-based approaches. To validate our claims, we benchmark our approach using a Bert-based model on the MultiNLI dataset. 

---
# Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs 

**Authors**: Yijie Jin, Junjie Peng, Xuanchao Lin, Haochen Yuan, Lan Wang, Cangzhi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.01068)  

**Abstract**: Multimodal Sentiment Analysis (MSA) is a rapidly developing field that integrates multimodal information to recognize sentiments, and existing models have made significant progress in this area. The central challenge in MSA is multimodal fusion, which is predominantly addressed by Multimodal Transformers (MulTs). Although act as the paradigm, MulTs suffer from efficiency concerns. In this work, from the perspective of efficiency optimization, we propose and prove that MulTs are hierarchical modal-wise heterogeneous graphs (HMHGs), and we introduce the graph-structured representation pattern of MulTs. Based on this pattern, we propose an Interlaced Mask (IM) mechanism to design the Graph-Structured and Interlaced-Masked Multimodal Transformer (GsiT). It is formally equivalent to MulTs which achieves an efficient weight-sharing mechanism without information disorder through IM, enabling All-Modal-In-One fusion with only 1/3 of the parameters of pure MulTs. A Triton kernel called Decomposition is implemented to ensure avoiding additional computational overhead. Moreover, it achieves significantly higher performance than traditional MulTs. To further validate the effectiveness of GsiT itself and the HMHG concept, we integrate them into multiple state-of-the-art models and demonstrate notable performance improvements and parameter reduction on widely used MSA datasets. 

---
# A Rusty Link in the AI Supply Chain: Detecting Evil Configurations in Model Repositories 

**Authors**: Ziqi Ding, Qian Fu, Junchen Ding, Gelei Deng, Yi Liu, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01067)  

**Abstract**: Recent advancements in large language models (LLMs) have spurred the development of diverse AI applications from code generation and video editing to text generation; however, AI supply chains such as Hugging Face, which host pretrained models and their associated configuration files contributed by the public, face significant security challenges; in particular, configuration files originally intended to set up models by specifying parameters and initial settings can be exploited to execute unauthorized code, yet research has largely overlooked their security compared to that of the models themselves; in this work, we present the first comprehensive study of malicious configurations on Hugging Face, identifying three attack scenarios (file, website, and repository operations) that expose inherent risks; to address these threats, we introduce CONFIGSCAN, an LLM-based tool that analyzes configuration files in the context of their associated runtime code and critical libraries, effectively detecting suspicious elements with low false positive rates and high accuracy; our extensive evaluation uncovers thousands of suspicious repositories and configuration files, underscoring the urgent need for enhanced security validation in AI model hosting platforms. 

---
# Good News for Script Kiddies? Evaluating Large Language Models for Automated Exploit Generation 

**Authors**: David Jin, Qian Fu, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01065)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code-related tasks, raising concerns about their potential for automated exploit generation (AEG). This paper presents the first systematic study on LLMs' effectiveness in AEG, evaluating both their cooperativeness and technical proficiency. To mitigate dataset bias, we introduce a benchmark with refactored versions of five software security labs. Additionally, we design an LLM-based attacker to systematically prompt LLMs for exploit generation. Our experiments reveal that GPT-4 and GPT-4o exhibit high cooperativeness, comparable to uncensored models, while Llama3 is the most resistant. However, no model successfully generates exploits for refactored labs, though GPT-4o's minimal errors highlight the potential for LLM-driven AEG advancements. 

---
# Model Tensor Planning 

**Authors**: An T. Le, Khai Nguyen, Minh Nhat Vu, João Carvalho, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2505.01059)  

**Abstract**: Sampling-based model predictive control (MPC) offers strong performance in nonlinear and contact-rich robotic tasks, yet often suffers from poor exploration due to locally greedy sampling schemes. We propose \emph{Model Tensor Planning} (MTP), a novel sampling-based MPC framework that introduces high-entropy control trajectory generation through structured tensor sampling. By sampling over randomized multipartite graphs and interpolating control trajectories with B-splines and Akima splines, MTP ensures smooth and globally diverse control candidates. We further propose a simple $\beta$-mixing strategy that blends local exploitative and global exploratory samples within the modified Cross-Entropy Method (CEM) update, balancing control refinement and exploration. Theoretically, we show that MTP achieves asymptotic path coverage and maximum entropy in the control trajectory space in the limit of infinite tensor depth and width.
Our implementation is fully vectorized using JAX and compatible with MuJoCo XLA, supporting \emph{Just-in-time} (JIT) compilation and batched rollouts for real-time control with online domain randomization. Through experiments on various challenging robotic tasks, ranging from dexterous in-hand manipulation to humanoid locomotion, we demonstrate that MTP outperforms standard MPC and evolutionary strategy baselines in task success and control robustness. Design and sensitivity ablations confirm the effectiveness of MTP tensor sampling structure, spline interpolation choices, and mixing strategy. Altogether, MTP offers a scalable framework for robust exploration in model-based planning and control. 

---
# Low-Precision Training of Large Language Models: Methods, Challenges, and Opportunities 

**Authors**: Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Guoxia Wang, Dianhai Yu, Yonggang Wen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.01043)  

**Abstract**: Large language models (LLMs) have achieved impressive performance across various domains. However, the substantial hardware resources required for their training present a significant barrier to efficiency and scalability. To mitigate this challenge, low-precision training techniques have been widely adopted, leading to notable advancements in training efficiency. Despite these gains, low-precision training involves several components$\unicode{x2013}$such as weights, activations, and gradients$\unicode{x2013}$each of which can be represented in different numerical formats. The resulting diversity has created a fragmented landscape in low-precision training research, making it difficult for researchers to gain a unified overview of the field. This survey provides a comprehensive review of existing low-precision training methods. To systematically organize these approaches, we categorize them into three primary groups based on their underlying numerical formats, which is a key factor influencing hardware compatibility, computational efficiency, and ease of reference for readers. The categories are: (1) fixed-point and integer-based methods, (2) floating-point-based methods, and (3) customized format-based methods. Additionally, we discuss quantization-aware training approaches, which share key similarities with low-precision training during forward propagation. Finally, we highlight several promising research directions to advance this field. A collection of papers discussed in this survey is provided in this https URL. 

---
# Stagnation in Evolutionary Algorithms: Convergence $\neq$ Optimality 

**Authors**: Xiaojun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.01036)  

**Abstract**: In the evolutionary computation community, it is widely believed that stagnation impedes convergence in evolutionary algorithms, and that convergence inherently indicates optimality. However, this perspective is misleading. In this study, it is the first to highlight that the stagnation of an individual can actually facilitate the convergence of the entire population, and convergence does not necessarily imply optimality, not even local optimality. Convergence alone is insufficient to ensure the effectiveness of evolutionary algorithms. Several counterexamples are provided to illustrate this argument. 

---
# Fine-Tuning Without Forgetting: Adaptation of YOLOv8 Preserves COCO Performance 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.01016)  

**Abstract**: The success of large pre-trained object detectors hinges on their adaptability to diverse downstream tasks. While fine-tuning is the standard adaptation method, specializing these models for challenging fine-grained domains necessitates careful consideration of feature granularity. The critical question remains: how deeply should the pre-trained backbone be fine-tuned to optimize for the specialized task without incurring catastrophic forgetting of the original general capabilities? Addressing this, we present a systematic empirical study evaluating the impact of fine-tuning depth. We adapt a standard YOLOv8n model to a custom, fine-grained fruit detection dataset by progressively unfreezing backbone layers (freeze points at layers 22, 15, and 10) and training. Performance was rigorously evaluated on both the target fruit dataset and, using a dual-head evaluation architecture, on the original COCO validation set. Our results demonstrate unequivocally that deeper fine-tuning (unfreezing down to layer 10) yields substantial performance gains (e.g., +10\% absolute mAP50) on the fine-grained fruit task compared to only training the head. Strikingly, this significant adaptation and specialization resulted in negligible performance degradation (<0.1\% absolute mAP difference) on the COCO benchmark across all tested freeze levels. We conclude that adapting mid-to-late backbone features is highly effective for fine-grained specialization. Critically, our results demonstrate this adaptation can be achieved without the commonly expected penalty of catastrophic forgetting, presenting a compelling case for exploring deeper fine-tuning strategies, particularly when targeting complex domains or when maximizing specialized performance is paramount. 

---
# Value Portrait: Understanding Values of LLMs with Human-aligned Benchmark 

**Authors**: Jongwook Han, Dongmin Choi, Woojung Song, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01015)  

**Abstract**: The importance of benchmarks for assessing the values of language models has been pronounced due to the growing need of more authentic, human-aligned responses. However, existing benchmarks rely on human or machine annotations that are vulnerable to value-related biases. Furthermore, the tested scenarios often diverge from real-world contexts in which models are commonly used to generate text and express values. To address these issues, we propose the Value Portrait benchmark, a reliable framework for evaluating LLMs' value orientations with two key characteristics. First, the benchmark consists of items that capture real-life user-LLM interactions, enhancing the relevance of assessment results to real-world LLM usage and thus ecological validity. Second, each item is rated by human subjects based on its similarity to their own thoughts, and correlations between these ratings and the subjects' actual value scores are derived. This psychometrically validated approach ensures that items strongly correlated with specific values serve as reliable items for assessing those values. Through evaluating 27 LLMs with our benchmark, we find that these models prioritize Benevolence, Security, and Self-Direction values while placing less emphasis on Tradition, Power, and Achievement values. Also, our analysis reveals biases in how LLMs perceive various demographic groups, deviating from real human data. 

---
# Towards the Resistance of Neural Network Watermarking to Fine-tuning 

**Authors**: Ling Tang, Yuefeng Chen, Hui Xue, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01007)  

**Abstract**: This paper proves a new watermarking method to embed the ownership information into a deep neural network (DNN), which is robust to fine-tuning. Specifically, we prove that when the input feature of a convolutional layer only contains low-frequency components, specific frequency components of the convolutional filter will not be changed by gradient descent during the fine-tuning process, where we propose a revised Fourier transform to extract frequency components from the convolutional filter. Additionally, we also prove that these frequency components are equivariant to weight scaling and weight permutations. In this way, we design a watermark module to encode the watermark information to specific frequency components in a convolutional filter. Preliminary experiments demonstrate the effectiveness of our method. 

---
# Toward Data-centric Directed Graph Learning: An Entropy-driven Approach 

**Authors**: Xunkai Li, Zhengyu Wu, Kaichi Yu, Hongchao Qin, Guang Zeng, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00983)  

**Abstract**: The directed graph (digraph), as a generalization of undirected graphs, exhibits superior representation capability in modeling complex topology systems and has garnered considerable attention in recent years. Despite the notable efforts made by existing DiGraph Neural Networks (DiGNNs) to leverage directed edges, they still fail to comprehensively delve into the abundant data knowledge concealed in the digraphs. This data-level limitation results in model-level sub-optimal predictive performance and underscores the necessity of further exploring the potential correlations between the directed edges (topology) and node profiles (feature and labels) from a data-centric perspective, thereby empowering model-centric neural networks with stronger encoding capabilities.
In this paper, we propose \textbf{E}ntropy-driven \textbf{D}igraph knowl\textbf{E}dge distillatio\textbf{N} (EDEN), which can serve as a data-centric digraph learning paradigm or a model-agnostic hot-and-plug data-centric Knowledge Distillation (KD) module. The core idea is to achieve data-centric ML, guided by our proposed hierarchical encoding theory for structured data. Specifically, EDEN first utilizes directed structural measurements from a topology perspective to construct a coarse-grained Hierarchical Knowledge Tree (HKT). Subsequently, EDEN quantifies the mutual information of node profiles to refine knowledge flow in the HKT, enabling data-centric KD supervision within model training. As a general framework, EDEN can also naturally extend to undirected scenarios and demonstrate satisfactory performance. In our experiments, EDEN has been widely evaluated on 14 (di)graph datasets (homophily and heterophily) and across 4 downstream tasks. The results demonstrate that EDEN attains SOTA performance and exhibits strong improvement for prevalent (Di)GNNs. 

---
# Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models 

**Authors**: Xuhui Jiang, Shengjie Ma, Chengjin Xu, Cehao Yang, Liyu Zhang, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00979)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success but remain data-inefficient, especially when learning from small, specialized corpora with limited and proprietary data. Existing synthetic data generation methods for continue pre-training focus on intra-document content and overlook cross-document knowledge associations, limiting content diversity and depth. We propose Synthetic-on-Graph (SoG), a synthetic data generation framework that incorporates cross-document knowledge associations for efficient corpus expansion. SoG constructs a context graph by extracting entities and concepts from the original corpus, representing cross-document associations, and employing a graph walk strategy for knowledge-associated sampling. This enhances synthetic data diversity and coherence, enabling models to learn complex knowledge structures and handle rare knowledge. To further improve synthetic data quality, we integrate Chain-of-Thought (CoT) and Contrastive Clarifying (CC) synthetic, enhancing reasoning processes and discriminative power. Experiments show that SoG outperforms the state-of-the-art (SOTA) method in a multi-hop document Q&A dataset while performing comparably to the SOTA method on the reading comprehension task datasets, which also underscores the better generalization capability of SoG. Our work advances synthetic data generation and provides practical solutions for efficient knowledge acquisition in LLMs, especially in domains with limited data availability. 

---
# Attack and defense techniques in large language models: A survey and new perspectives 

**Authors**: Zhiyu Liao, Kang Chen, Yuanguo Lin, Kangkang Li, Yunxuan Liu, Hefeng Chen, Xingwang Huang, Yuanhui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00976)  

**Abstract**: Large Language Models (LLMs) have become central to numerous natural language processing tasks, but their vulnerabilities present significant security and ethical challenges. This systematic survey explores the evolving landscape of attack and defense techniques in LLMs. We classify attacks into adversarial prompt attack, optimized attacks, model theft, as well as attacks on application of LLMs, detailing their mechanisms and implications. Consequently, we analyze defense strategies, including prevention-based and detection-based defense methods. Although advances have been made, challenges remain to adapt to the dynamic threat landscape, balance usability with robustness, and address resource constraints in defense implementation. We highlight open problems, including the need for adaptive scalable defenses, explainable security techniques, and standardized evaluation frameworks. This survey provides actionable insights and directions for developing secure and resilient LLMs, emphasizing the importance of interdisciplinary collaboration and ethical considerations to mitigate risks in real-world applications. 

---
# Tree-Sliced Wasserstein Distance with Nonlinear Projection 

**Authors**: Thanh Tran, Viet-Hoang Tran, Thanh Chu, Trang Pham, Laurent El Ghaoui, Tam Le, Tan M. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00968)  

**Abstract**: Tree-Sliced methods have recently emerged as an alternative to the traditional Sliced Wasserstein (SW) distance, replacing one-dimensional lines with tree-based metric spaces and incorporating a splitting mechanism for projecting measures. This approach enhances the ability to capture the topological structures of integration domains in Sliced Optimal Transport while maintaining low computational costs. Building on this foundation, we propose a novel nonlinear projectional framework for the Tree-Sliced Wasserstein (TSW) distance, substituting the linear projections in earlier versions with general projections, while ensuring the injectivity of the associated Radon Transform and preserving the well-definedness of the resulting metric. By designing appropriate projections, we construct efficient metrics for measures on both Euclidean spaces and spheres. Finally, we validate our proposed metric through extensive numerical experiments for Euclidean and spherical datasets. Applications include gradient flows, self-supervised learning, and generative models, where our methods demonstrate significant improvements over recent SW and TSW variants. 

---
# Llama-Nemotron: Efficient Reasoning Models 

**Authors**: Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2505.00949)  

**Abstract**: We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM. 

---
# CDFormer: Cross-Domain Few-Shot Object Detection Transformer Against Feature Confusion 

**Authors**: Boyuan Meng, Xiaohan Zhang, Peilin Li, Zhe Wu, Yiming Li, Wenkai Zhao, Beinan Yu, Hui-Liang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00938)  

**Abstract**: Cross-domain few-shot object detection (CD-FSOD) aims to detect novel objects across different domains with limited class instances. Feature confusion, including object-background confusion and object-object confusion, presents significant challenges in both cross-domain and few-shot settings. In this work, we introduce CDFormer, a cross-domain few-shot object detection transformer against feature confusion, to address these challenges. The method specifically tackles feature confusion through two key modules: object-background distinguishing (OBD) and object-object distinguishing (OOD). The OBD module leverages a learnable background token to differentiate between objects and background, while the OOD module enhances the distinction between objects of different classes. Experimental results demonstrate that CDFormer outperforms previous state-of-the-art approaches, achieving 12.9% mAP, 11.0% mAP, and 10.4% mAP improvements under the 1/5/10 shot settings, respectively, when fine-tuned. 

---
# Autonomous Embodied Agents: When Robotics Meets Deep Learning Reasoning 

**Authors**: Roberto Bigazzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00935)  

**Abstract**: The increase in available computing power and the Deep Learning revolution have allowed the exploration of new topics and frontiers in Artificial Intelligence research. A new field called Embodied Artificial Intelligence, which places at the intersection of Computer Vision, Robotics, and Decision Making, has been gaining importance during the last few years, as it aims to foster the development of smart autonomous robots and their deployment in society. The recent availability of large collections of 3D models for photorealistic robotic simulation has allowed faster and safe training of learning-based agents for millions of frames and a careful evaluation of their behavior before deploying the models on real robotic platforms. These intelligent agents are intended to perform a certain task in a possibly unknown environment. To this end, during the training in simulation, the agents learn to perform continuous interactions with the surroundings, such as gathering information from the environment, encoding and extracting useful cues for the task, and performing actions towards the final goal; where every action of the agent influences the interactions. This dissertation follows the complete creation process of embodied agents for indoor environments, from their concept to their implementation and deployment. We aim to contribute to research in Embodied AI and autonomous agents, in order to foster future work in this field. We present a detailed analysis of the procedure behind implementing an intelligent embodied agent, comprehending a thorough description of the current state-of-the-art in literature, technical explanations of the proposed methods, and accurate experimental studies on relevant robotic tasks. 

---
# A Self-Supervised Transformer for Unusable Shared Bike Detection 

**Authors**: Yin Huang, Yongqi Dong, Youhua Tang, Alvaro García Hernandez  

**Link**: [PDF](https://arxiv.org/pdf/2505.00932)  

**Abstract**: The rapid expansion of bike-sharing systems (BSS) has greatly improved urban "last-mile" connectivity, yet large-scale deployments face escalating operational challenges, particularly in detecting faulty bikes. Existing detection approaches either rely on static model-based thresholds that overlook dynamic spatiotemporal (ST) usage patterns or employ supervised learning methods that struggle with label scarcity and class imbalance. To address these limitations, this paper proposes a novel Self-Supervised Transformer (SSTransformer) framework for automatically detecting unusable shared bikes, leveraging ST features extracted from GPS trajectories and trip records. The model incorporates a self-supervised pre-training strategy to enhance its feature extraction capabilities, followed by fine-tuning for efficient status recognition. In the pre-training phase, the Transformer encoder learns generalized representations of bike movement via a self-supervised objective; in the fine-tuning phase, the encoder is adapted to a downstream binary classification task. Comprehensive experiments on a real-world dataset of 10,730 bikes (1,870 unusable, 8,860 normal) from Chengdu, China, demonstrate that SSTransformer significantly outperforms traditional machine learning, ensemble learning, and deep learning baselines, achieving the best accuracy (97.81%), precision (0.8889), and F1-score (0.9358). This work highlights the effectiveness of self-supervised Transformer on ST data for capturing complex anomalies in BSS, paving the way toward more reliable and scalable maintenance solutions for shared mobility. 

---
# Large Language Model-Driven Dynamic Assessment of Grammatical Accuracy in English Language Learner Writing 

**Authors**: Timur Jaganov, John Blake, Julián Villegas, Nicholas Carr  

**Link**: [PDF](https://arxiv.org/pdf/2505.00931)  

**Abstract**: This study investigates the potential for Large Language Models (LLMs) to scale-up Dynamic Assessment (DA). To facilitate such an investigation, we first developed DynaWrite-a modular, microservices-based grammatical tutoring application which supports multiple LLMs to generate dynamic feedback to learners of English. Initial testing of 21 LLMs, revealed GPT-4o and neural chat to have the most potential to scale-up DA in the language learning classroom. Further testing of these two candidates found both models performed similarly in their ability to accurately identify grammatical errors in user sentences. However, GPT-4o consistently outperformed neural chat in the quality of its DA by generating clear, consistent, and progressively explicit hints. Real-time responsiveness and system stability were also confirmed through detailed performance testing, with GPT-4o exhibiting sufficient speed and stability. This study shows that LLMs can be used to scale-up dynamic assessment and thus enable dynamic assessment to be delivered to larger groups than possible in traditional teacher-learner settings. 

---
# Dynamic and Distributed Routing in IoT Networks based on Multi-Objective Q-Learning 

**Authors**: Shubham Vaishnav, Praveen Kumar Donta, Sindri Magnússon  

**Link**: [PDF](https://arxiv.org/pdf/2505.00918)  

**Abstract**: The last few decades have witnessed a rapid increase in IoT devices owing to their wide range of applications, such as smart healthcare monitoring systems, smart cities, and environmental monitoring. A critical task in IoT networks is sensing and transmitting information over the network. The IoT nodes gather data by sensing the environment and then transmit this data to a destination node via multi-hop communication, following some routing protocols. These protocols are usually designed to optimize possibly contradictory objectives, such as maximizing packet delivery ratio and energy efficiency. While most literature has focused on optimizing a static objective that remains unchanged, many real-world IoT applications require adapting to rapidly shifting priorities. For example, in monitoring systems, some transmissions are time-critical and require a high priority on low latency, while other transmissions are less urgent and instead prioritize energy efficiency. To meet such dynamic demands, we propose novel dynamic and distributed routing based on multiobjective Q-learning that can adapt to changes in preferences in real-time. Our algorithm builds on ideas from both multi-objective optimization and Q-learning. We also propose a novel greedy interpolation policy scheme to take near-optimal decisions for unexpected preference changes. The proposed scheme can approximate and utilize the Pareto-efficient solutions for dynamic preferences, thus utilizing past knowledge to adapt to unpredictable preferences quickly during runtime. Simulation results show that the proposed scheme outperforms state-of-the-art algorithms for various exploration strategies, preference variation patterns, and important metrics like overall reward, energy efficiency, and packet delivery ratio. 

---
# Multivariate Conformal Selection 

**Authors**: Tian Bai, Yue Zhao, Xiang Yu, Archer Y. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00917)  

**Abstract**: Selecting high-quality candidates from large datasets is critical in applications such as drug discovery, precision medicine, and alignment of large language models (LLMs). While Conformal Selection (CS) provides rigorous uncertainty quantification, it is limited to univariate responses and scalar criteria. To address this issue, we propose Multivariate Conformal Selection (mCS), a generalization of CS designed for multivariate response settings. Our method introduces regional monotonicity and employs multivariate nonconformity scores to construct conformal p-values, enabling finite-sample False Discovery Rate (FDR) control. We present two variants: mCS-dist, using distance-based scores, and mCS-learn, which learns optimal scores via differentiable optimization. Experiments on simulated and real-world datasets demonstrate that mCS significantly improves selection power while maintaining FDR control, establishing it as a robust framework for multivariate selection tasks. 

---
# Fine-Tuning without Performance Degradation 

**Authors**: Han Wang, Adam White, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2505.00913)  

**Abstract**: Fine-tuning policies learned offline remains a major challenge in application domains. Monotonic performance improvement during \emph{fine-tuning} is often challenging, as agents typically experience performance degradation at the early fine-tuning stage. The community has identified multiple difficulties in fine-tuning a learned network online, however, the majority of progress has focused on improving learning efficiency during fine-tuning. In practice, this comes at a serious cost during fine-tuning: initially, agent performance degrades as the agent explores and effectively overrides the policy learned offline. We show across a range of settings, many offline-to-online algorithms exhibit either (1) performance degradation or (2) slow learning (sometimes effectively no improvement) during fine-tuning. We introduce a new fine-tuning algorithm, based on an algorithm called Jump Start, that gradually allows more exploration based on online estimates of performance. Empirically, this approach achieves fast fine-tuning and significantly reduces performance degradations compared with existing algorithms designed to do the same. 

---
# Rethinking Time Encoding via Learnable Transformation Functions 

**Authors**: Xi Chen, Yateng Tang, Jiarong Xu, Jiawei Zhang, Siwei Zhang, Sijia Peng, Xuehao Zheng, Yun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.00887)  

**Abstract**: Effectively modeling time information and incorporating it into applications or models involving chronologically occurring events is crucial. Real-world scenarios often involve diverse and complex time patterns, which pose significant challenges for time encoding methods. While previous methods focus on capturing time patterns, many rely on specific inductive biases, such as using trigonometric functions to model periodicity. This narrow focus on single-pattern modeling makes them less effective in handling the diversity and complexities of real-world time patterns. In this paper, we investigate to improve the existing commonly used time encoding methods and introduce Learnable Transformation-based Generalized Time Encoding (LeTE). We propose using deep function learning techniques to parameterize non-linear transformations in time encoding, making them learnable and capable of modeling generalized time patterns, including diverse and complex temporal dynamics. By enabling learnable transformations, LeTE encompasses previous methods as specific cases and allows seamless integration into a wide range of tasks. Through extensive experiments across diverse domains, we demonstrate the versatility and effectiveness of LeTE. 

---
# Towards Explainable Temporal User Profiling with LLMs 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2505.00886)  

**Abstract**: Accurately modeling user preferences is vital not only for improving recommendation performance but also for enhancing transparency in recommender systems. Conventional user profiling methods, such as averaging item embeddings, often overlook the evolving, nuanced nature of user interests, particularly the interplay between short-term and long-term preferences. In this work, we leverage large language models (LLMs) to generate natural language summaries of users' interaction histories, distinguishing recent behaviors from more persistent tendencies. Our framework not only models temporal user preferences but also produces natural language profiles that can be used to explain recommendations in an interpretable manner. These textual profiles are encoded via a pre-trained model, and an attention mechanism dynamically fuses the short-term and long-term embeddings into a comprehensive user representation. Beyond boosting recommendation accuracy over multiple baselines, our approach naturally supports explainability: the interpretable text summaries and attention weights can be exposed to end users, offering insights into why specific items are suggested. Experiments on real-world datasets underscore both the performance gains and the promise of generating clearer, more transparent justifications for content-based recommendations. 

---
# IK Seed Generator for Dual-Arm Human-like Physicality Robot with Mobile Base 

**Authors**: Jun Takamatsu, Atsushi Kanehira, Kazuhiro Sasabuchi, Naoki Wake, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00871)  

**Abstract**: Robots are strongly expected as a means of replacing human tasks. If a robot has a human-like physicality, the possibility of replacing human tasks increases. In the case of household service robots, it is desirable for them to be on a human-like size so that they do not become excessively large in order to coexist with humans in their operating environment. However, robots with size limitations tend to have difficulty solving inverse kinematics (IK) due to mechanical limitations, such as joint angle limitations. Conversely, if the difficulty coming from this limitation could be mitigated, one can expect that the use of such robots becomes more valuable. In numerical IK solver, which is commonly used for robots with higher degrees-of-freedom (DOF), the solvability of IK depends on the initial guess given to the solver. Thus, this paper proposes a method for generating a good initial guess for a numerical IK solver given the target hand configuration. For the purpose, we define the goodness of an initial guess using the scaled Jacobian matrix, which can calculate the manipulability index considering the joint limits. These two factors are related to the difficulty of solving IK. We generate the initial guess by optimizing the goodness using the genetic algorithm (GA). To enumerate much possible IK solutions, we use the reachability map that represents the reachable area of the robot hand in the arm-base coordinate system. We conduct quantitative evaluation and prove that using an initial guess that is judged to be better using the goodness value increases the probability that IK is solved. Finally, as an application of the proposed method, we show that by generating good initial guesses for IK a robot actually achieves three typical scenarios. 

---
# ICQuant: Index Coding enables Low-bit LLM Quantization 

**Authors**: Xinlin Li, Osama Hanna, Christina Fragouli, Suhas Diggavi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00850)  

**Abstract**: The rapid deployment of Large Language Models (LLMs) highlights the need for efficient low-bit post-training quantization (PTQ), due to their high memory costs. A key challenge in weight quantization is the presence of outliers, which inflate quantization ranges and lead to large errors. While a number of outlier suppression techniques have been proposed, they either: fail to effectively shrink the quantization range, or incur (relatively) high bit overhead. In this paper, we present ICQuant, a novel framework that leverages outlier statistics to design an efficient index coding scheme for outlier-aware weight-only quantization. Compared to existing outlier suppression techniques requiring $\approx 1$ bit overhead to halve the quantization range, ICQuant requires only $\approx 0.3$ bits; a significant saving in extreme compression regimes (e.g., 2-3 bits per weight). ICQuant can be used on top of any existing quantizers to eliminate outliers, improving the quantization quality. Using just 2.3 bits per weight and simple scalar quantizers, ICQuant improves the zero-shot accuracy of the 2-bit Llama3-70B model by up to 130% and 150% relative to QTIP and QuIP#; and it achieves comparable performance to the best-known fine-tuned quantizer (PV-tuning) without fine-tuning. 

---
# OET: Optimization-based prompt injection Evaluation Toolkit 

**Authors**: Jinsheng Pan, Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00843)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation, enabling their widespread adoption across various domains. However, their susceptibility to prompt injection attacks poses significant security risks, as adversarial inputs can manipulate model behavior and override intended instructions. Despite numerous defense strategies, a standardized framework to rigorously evaluate their effectiveness, especially under adaptive adversarial scenarios, is lacking. To address this gap, we introduce OET, an optimization-based evaluation toolkit that systematically benchmarks prompt injection attacks and defenses across diverse datasets using an adaptive testing framework. Our toolkit features a modular workflow that facilitates adversarial string generation, dynamic attack execution, and comprehensive result analysis, offering a unified platform for assessing adversarial robustness. Crucially, the adaptive testing framework leverages optimization methods with both white-box and black-box access to generate worst-case adversarial examples, thereby enabling strict red-teaming evaluations. Extensive experiments underscore the limitations of current defense mechanisms, with some models remaining susceptible even after implementing security enhancements. 

---
# From Texts to Shields: Convergence of Large Language Models and Cybersecurity 

**Authors**: Tao Li, Ya-Ting Yang, Yunian Pan, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00841)  

**Abstract**: This report explores the convergence of large language models (LLMs) and cybersecurity, synthesizing interdisciplinary insights from network security, artificial intelligence, formal methods, and human-centered design. It examines emerging applications of LLMs in software and network security, 5G vulnerability analysis, and generative security engineering. The report highlights the role of agentic LLMs in automating complex tasks, improving operational efficiency, and enabling reasoning-driven security analytics. Socio-technical challenges associated with the deployment of LLMs -- including trust, transparency, and ethical considerations -- can be addressed through strategies such as human-in-the-loop systems, role-specific training, and proactive robustness testing. The report further outlines critical research challenges in ensuring interpretability, safety, and fairness in LLM-based systems, particularly in high-stakes domains. By integrating technical advances with organizational and societal considerations, this report presents a forward-looking research agenda for the secure and effective adoption of LLMs in cybersecurity. 

---
# Spill The Beans: Exploiting CPU Cache Side-Channels to Leak Tokens from Large Language Models 

**Authors**: Andrew Adiletta, Berk Sunar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00817)  

**Abstract**: Side-channel attacks on shared hardware resources increasingly threaten confidentiality, especially with the rise of Large Language Models (LLMs). In this work, we introduce Spill The Beans, a novel application of cache side-channels to leak tokens generated by an LLM. By co-locating an attack process on the same hardware as the victim model, we flush and reload embedding vectors from the embedding layer, where each token corresponds to a unique embedding vector. When accessed during token generation, it results in a cache hit detectable by our attack on shared lower-level caches.
A significant challenge is the massive size of LLMs, which, by nature of their compute intensive operation, quickly evicts embedding vectors from the cache. We address this by balancing the number of tokens monitored against the amount of information leaked. Monitoring more tokens increases potential vocabulary leakage but raises the chance of missing cache hits due to eviction; monitoring fewer tokens improves detection reliability but limits vocabulary coverage.
Through extensive experimentation, we demonstrate the feasibility of leaking tokens from LLMs via cache side-channels. Our findings reveal a new vulnerability in LLM deployments, highlighting that even sophisticated models are susceptible to traditional side-channel attacks. We discuss the implications for privacy and security in LLM-serving infrastructures and suggest considerations for mitigating such threats. For proof of concept we consider two concrete attack scenarios: Our experiments show that an attacker can recover as much as 80%-90% of a high entropy API key with single shot monitoring. As for English text we can reach a 40% recovery rate with a single shot. We should note that the rate highly depends on the monitored token set and these rates can be improved by targeting more specialized output domains. 

---
# Handling Label Noise via Instance-Level Difficulty Modeling and Dynamic Optimization 

**Authors**: Kuan Zhang, Chengliang Chai, Jingzhe Xu, Chi Zhang, Ye Yuan, Guoren Wang, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00812)  

**Abstract**: Recent studies indicate that deep neural networks degrade in generalization performance under noisy supervision. Existing methods focus on isolating clean subsets or correcting noisy labels, facing limitations such as high computational costs, heavy hyperparameter tuning process, and coarse-grained optimization. To address these challenges, we propose a novel two-stage noisy learning framework that enables instance-level optimization through a dynamically weighted loss function, avoiding hyperparameter tuning. To obtain stable and accurate information about noise modeling, we introduce a simple yet effective metric, termed wrong event, which dynamically models the cleanliness and difficulty of individual samples while maintaining computational costs. Our framework first collects wrong event information and builds a strong base model. Then we perform noise-robust training on the base model, using a probabilistic model to handle the wrong event information of samples. Experiments on five synthetic and real-world LNL benchmarks demonstrate our method surpasses state-of-the-art methods in performance, achieves a nearly 75% reduction in computational time and improves model scalability. 

---
# A Mathematical Philosophy of Explanations in Mechanistic Interpretability -- The Strange Science Part I.i 

**Authors**: Kola Ayonrinde, Louis Jaburi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00808)  

**Abstract**: Mechanistic Interpretability aims to understand neural networks through causal explanations. We argue for the Explanatory View Hypothesis: that Mechanistic Interpretability research is a principled approach to understanding models because neural networks contain implicit explanations which can be extracted and understood. We hence show that Explanatory Faithfulness, an assessment of how well an explanation fits a model, is well-defined. We propose a definition of Mechanistic Interpretability (MI) as the practice of producing Model-level, Ontic, Causal-Mechanistic, and Falsifiable explanations of neural networks, allowing us to distinguish MI from other interpretability paradigms and detail MI's inherent limits. We formulate the Principle of Explanatory Optimism, a conjecture which we argue is a necessary precondition for the success of Mechanistic Interpretability. 

---
# To Repair or Not to Repair? Investigating the Importance of AB-Cycles for the State-of-the-Art TSP Heuristic EAX 

**Authors**: Jonathan Heins, Darrell Whitley, Pascal Kerschke  

**Link**: [PDF](https://arxiv.org/pdf/2505.00803)  

**Abstract**: The Edge Assembly Crossover (EAX) algorithm is the state-of-the-art heuristic for solving the Traveling Salesperson Problem (TSP). It regularly outperforms other methods, such as the Lin-Kernighan-Helsgaun heuristic (LKH), across diverse sets of TSP instances. Essentially, EAX employs a two-stage mechanism that focuses on improving the current solutions, first, at the local and, subsequently, at the global level. Although the second phase of the algorithm has been thoroughly studied, configured, and refined in the past, in particular, its first stage has hardly been examined.
In this paper, we thus focus on the first stage of EAX and introduce a novel method that quickly verifies whether the AB-cycles, generated during its internal optimization procedure, yield valid tours -- or whether they need to be repaired. Knowledge of the latter is also particularly relevant before applying other powerful crossover operators such as the Generalized Partition Crossover (GPX). Based on our insights, we propose and evaluate several improved versions of EAX. According to our benchmark study across 10 000 different TSP instances, the most promising of our proposed EAX variants demonstrates improved computational efficiency and solution quality on previously rather difficult instances compared to the current state-of-the-art EAX algorithm. 

---
# Scalable Meta-Learning via Mixed-Mode Differentiation 

**Authors**: Iurii Kemaev, Dan A Calian, Luisa M Zintgraf, Gregory Farquhar, Hado van Hasselt  

**Link**: [PDF](https://arxiv.org/pdf/2505.00793)  

**Abstract**: Gradient-based bilevel optimisation is a powerful technique with applications in hyperparameter optimisation, task adaptation, algorithm discovery, meta-learning more broadly, and beyond. It often requires differentiating through the gradient-based optimisation process itself, leading to "gradient-of-a-gradient" calculations with computationally expensive second-order and mixed derivatives. While modern automatic differentiation libraries provide a convenient way to write programs for calculating these derivatives, they oftentimes cannot fully exploit the specific structure of these problems out-of-the-box, leading to suboptimal performance. In this paper, we analyse such cases and propose Mixed-Flow Meta-Gradients, or MixFlow-MG -- a practical algorithm that uses mixed-mode differentiation to construct more efficient and scalable computational graphs yielding over 10x memory and up to 25% wall-clock time improvements over standard implementations in modern meta-learning setups. 

---
# Constructing an Optimal Behavior Basis for the Option Keyboard 

**Authors**: Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2505.00787)  

**Abstract**: Multi-task reinforcement learning aims to quickly identify solutions for new tasks with minimal or no additional interaction with the environment. Generalized Policy Improvement (GPI) addresses this by combining a set of base policies to produce a new one that is at least as good -- though not necessarily optimal -- as any individual base policy. Optimality can be ensured, particularly in the linear-reward case, via techniques that compute a Convex Coverage Set (CCS). However, these are computationally expensive and do not scale to complex domains. The Option Keyboard (OK) improves upon GPI by producing policies that are at least as good -- and often better. It achieves this through a learned meta-policy that dynamically combines base policies. However, its performance critically depends on the choice of base policies. This raises a key question: is there an optimal set of base policies -- an optimal behavior basis -- that enables zero-shot identification of optimal solutions for any linear tasks? We solve this open problem by introducing a novel method that efficiently constructs such an optimal behavior basis. We show that it significantly reduces the number of base policies needed to ensure optimality in new tasks. We also prove that it is strictly more expressive than a CCS, enabling particular classes of non-linear tasks to be solved optimally. We empirically evaluate our technique in challenging domains and show that it outperforms state-of-the-art approaches, increasingly so as task complexity increases. 

---
# Multi-Modal Language Models as Text-to-Image Model Evaluators 

**Authors**: Jiahui Chen, Candace Ross, Reyhane Askari-Hemmat, Koustuv Sinha, Melissa Hall, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2505.00759)  

**Abstract**: The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate the T2I progress. In this paper, we explore the potential of multi-modal large language models (MLLMs) as evaluator agents that interact with a T2I model, with the objective of assessing prompt-generation consistency and image aesthetics. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework that iteratively generates prompts for evaluation, scores generated images and matches T2I evaluation of existing benchmarks with a fraction of the prompts used in existing static benchmarks. Moreover, we show that MT2IE's prompt-generation consistency scores have higher correlation with human judgment than scores previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance, producing the same relative T2I model rankings as existing benchmarks while using only 1/80th the number of prompts for evaluation. 

---
# Efficient On-Chip Implementation of 4D Radar-Based 3D Object Detection on Hailo-8L 

**Authors**: Woong-Chan Byun, Dong-Hee Paek, Seung-Hyun Song, Seung-Hyun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.00757)  

**Abstract**: 4D radar has attracted attention in autonomous driving due to its ability to enable robust 3D object detection even under adverse weather conditions. To practically deploy such technologies, it is essential to achieve real-time processing within low-power embedded environments. Addressing this, we present the first on-chip implementation of a 4D radar-based 3D object detection model on the Hailo-8L AI accelerator. Although conventional 3D convolutional neural network (CNN) architectures require 5D inputs, the Hailo-8L only supports 4D tensors, posing a significant challenge. To overcome this limitation, we introduce a tensor transformation method that reshapes 5D inputs into 4D formats during the compilation process, enabling direct deployment without altering the model structure. The proposed system achieves 46.47% AP_3D and 52.75% AP_BEV, maintaining comparable accuracy to GPU-based models while achieving an inference speed of 13.76 Hz. These results demonstrate the applicability of 4D radar-based perception technologies to autonomous driving systems. 

---
# P2P-Insole: Human Pose Estimation Using Foot Pressure Distribution and Motion Sensors 

**Authors**: Atsuya Watanabe, Ratna Aisuwarya, Lei Jing  

**Link**: [PDF](https://arxiv.org/pdf/2505.00755)  

**Abstract**: This work presents P2P-Insole, a low-cost approach for estimating and visualizing 3D human skeletal data using insole-type sensors integrated with IMUs. Each insole, fabricated with e-textile garment techniques, costs under USD 1, making it significantly cheaper than commercial alternatives and ideal for large-scale production. Our approach uses foot pressure distribution, acceleration, and rotation data to overcome limitations, providing a lightweight, minimally intrusive, and privacy-aware solution. The system employs a Transformer model for efficient temporal feature extraction, enriched by first and second derivatives in the input stream. Including multimodal information, such as accelerometers and rotational measurements, improves the accuracy of complex motion pattern recognition. These facts are demonstrated experimentally, while error metrics show the robustness of the approach in various posture estimation tasks. This work could be the foundation for a low-cost, practical application in rehabilitation, injury prevention, and health monitoring while enabling further development through sensor optimization and expanded datasets. 

---
# DARTer: Dynamic Adaptive Representation Tracker for Nighttime UAV Tracking 

**Authors**: Xuzhao Li, Xuchen Li, Shiyu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00752)  

**Abstract**: Nighttime UAV tracking presents significant challenges due to extreme illumination variations and viewpoint changes, which severely degrade tracking performance. Existing approaches either rely on light enhancers with high computational costs or introduce redundant domain adaptation mechanisms, failing to fully utilize the dynamic features in varying perspectives. To address these issues, we propose \textbf{DARTer} (\textbf{D}ynamic \textbf{A}daptive \textbf{R}epresentation \textbf{T}racker), an end-to-end tracking framework designed for nighttime UAV scenarios. DARTer leverages a Dynamic Feature Blender (DFB) to effectively fuse multi-perspective nighttime features from static and dynamic templates, enhancing representation robustness. Meanwhile, a Dynamic Feature Activator (DFA) adaptively activates Vision Transformer layers based on extracted features, significantly improving efficiency by reducing redundant computations. Our model eliminates the need for complex multi-task loss functions, enabling a streamlined training process. Extensive experiments on multiple nighttime UAV tracking benchmarks demonstrate the superiority of DARTer over state-of-the-art trackers. These results confirm that DARTer effectively balances tracking accuracy and efficiency, making it a promising solution for real-world nighttime UAV tracking applications. 

---
# The Coral Protocol: Open Infrastructure Connecting The Internet of Agents 

**Authors**: Roman J. Georgio, Caelum Forder, Suman Deb, Peter Carroll, Önder Gürcan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00749)  

**Abstract**: The Coral Protocol is an open and decentralized collaboration infrastructure that enables communication, coordination, trust and payments for The Internet of Agents. It addresses the growing need for interoperability in a world where organizations are deploying multiple specialized AI agents that must work together across domains and vendors. As a foundational platform for multi-agent AI ecosystems, Coral establishes a common language and coordination framework allowing any agent to participate in complex workflows with others. Its design emphasizes broad compatibility, security, and vendor neutrality, ensuring that agent interactions are efficient and trustworthy. In particular, Coral introduces standardized messaging formats for agent communication, a modular coordination mechanism for orchestrating multi-agent tasks, and secure team formation capabilities for dynamically assembling trusted groups of agents. Together, these innovations position Coral Protocol as a cornerstone of the emerging "Internet of Agents," unlocking new levels of automation, collective intelligence, and business value through open agent collaboration. 

---
# Zoomer: Adaptive Image Focus Optimization for Black-box MLLM 

**Authors**: Jiaxu Qian, Chendong Wang, Yifan Yang, Chaoyun Zhang, Huiqiang Jiang, Xufang Luo, Yu Kang, Qingwei Lin, Anlan Zhang, Shiqi Jiang, Ting Cao, Tianjun Mao, Suman Banerjee, Guyue Liu, Saravan Rajmohan, Dongmei Zhang, Yuqing Yang, Qi Zhang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00742)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have broadened the scope of vision-language tasks, excelling in applications like image captioning and interactive question-answering. However, these models struggle with accurately processing visual data, particularly in tasks requiring precise object recognition and fine visual details. Stringent token limits often result in the omission of critical information, hampering performance. To address these limitations, we introduce \SysName, a novel visual prompting mechanism designed to enhance MLLM performance while preserving essential visual details within token limits. \SysName features three key innovations: a prompt-aware strategy that dynamically highlights relevant image regions, a spatial-preserving orchestration schema that maintains object integrity, and a budget-aware prompting method that balances global context with crucial visual details. Comprehensive evaluations across multiple datasets demonstrate that \SysName consistently outperforms baseline methods, achieving up to a $26.9\%$ improvement in accuracy while significantly reducing token consumption. 

---
# A Survey on 3D Reconstruction Techniques in Plant Phenotyping: From Classical Methods to Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and Beyond 

**Authors**: Jiajia Li, Xinda Qi, Seyed Hamidreza Nabaei, Meiqi Liu, Dong Chen, Xin Zhang, Xunyuan Yin, Zhaojian Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00737)  

**Abstract**: Plant phenotyping plays a pivotal role in understanding plant traits and their interactions with the environment, making it crucial for advancing precision agriculture and crop improvement. 3D reconstruction technologies have emerged as powerful tools for capturing detailed plant morphology and structure, offering significant potential for accurate and automated phenotyping. This paper provides a comprehensive review of the 3D reconstruction techniques for plant phenotyping, covering classical reconstruction methods, emerging Neural Radiance Fields (NeRF), and the novel 3D Gaussian Splatting (3DGS) approach. Classical methods, which often rely on high-resolution sensors, are widely adopted due to their simplicity and flexibility in representing plant structures. However, they face challenges such as data density, noise, and scalability. NeRF, a recent advancement, enables high-quality, photorealistic 3D reconstructions from sparse viewpoints, but its computational cost and applicability in outdoor environments remain areas of active research. The emerging 3DGS technique introduces a new paradigm in reconstructing plant structures by representing geometry through Gaussian primitives, offering potential benefits in both efficiency and scalability. We review the methodologies, applications, and performance of these approaches in plant phenotyping and discuss their respective strengths, limitations, and future prospects (this https URL). Through this review, we aim to provide insights into how these diverse 3D reconstruction techniques can be effectively leveraged for automated and high-throughput plant phenotyping, contributing to the next generation of agricultural technology. 

---
# TF1-EN-3M: Three Million Synthetic Moral Fables for Training Small, Open Language Models 

**Authors**: Mihai Nadas, Laura Diosan, Andrei Piscoran, Andreea Tomescu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20605)  

**Abstract**: Moral stories are a time-tested vehicle for transmitting values, yet modern NLP lacks a large, structured corpus that couples coherent narratives with explicit ethical lessons. We close this gap with TF1-EN-3M, the first open dataset of three million English-language fables generated exclusively by instruction-tuned models no larger than 8B parameters. Each story follows a six-slot scaffold (character -> trait -> setting -> conflict -> resolution -> moral), produced through a combinatorial prompt engine that guarantees genre fidelity while covering a broad thematic space.
A hybrid evaluation pipeline blends (i) a GPT-based critic that scores grammar, creativity, moral clarity, and template adherence with (ii) reference-free diversity and readability metrics. Among ten open-weight candidates, an 8B-parameter Llama-3 variant delivers the best quality-speed trade-off, producing high-scoring fables on a single consumer GPU (<24 GB VRAM) at approximately 13.5 cents per 1,000 fables.
We release the dataset, generation code, evaluation scripts, and full metadata under a permissive license, enabling exact reproducibility and cost benchmarking. TF1-EN-3M opens avenues for research in instruction following, narrative intelligence, value alignment, and child-friendly educational AI, demonstrating that large-scale moral storytelling no longer requires proprietary giant models. 

---
# Advancing Software Security and Reliability in Cloud Platforms through AI-based Anomaly Detection 

**Authors**: Sabbir M. Saleh, Ibrahim Mohammed Sayem, Nazim Madhavji, John Steinbacher  

**Link**: [PDF](https://arxiv.org/pdf/2411.09200)  

**Abstract**: Continuous Integration/Continuous Deployment (CI/CD) is fundamental for advanced software development, supporting faster and more efficient delivery of code changes into cloud environments. However, security issues in the CI/CD pipeline remain challenging, and incidents (e.g., DDoS, Bot, Log4j, etc.) are happening over the cloud environments. While plenty of literature discusses static security testing and CI/CD practices, only a few deal with network traffic pattern analysis to detect different cyberattacks. This research aims to enhance CI/CD pipeline security by implementing anomaly detection through AI (Artificial Intelligence) support. The goal is to identify unusual behaviour or variations from network traffic patterns in pipeline and cloud platforms. The system shall integrate into the workflow to continuously monitor pipeline activities and cloud infrastructure. Additionally, it aims to explore adaptive response mechanisms to mitigate the detected anomalies or security threats. This research employed two popular network traffic datasets, CSE-CIC-IDS2018 and CSE-CIC-IDS2017. We implemented a combination of Convolution Neural Network(CNN) and Long Short-Term Memory (LSTM) to detect unusual traffic patterns. We achieved an accuracy of 98.69% and 98.30% and generated log files in different CI/CD pipeline stages that resemble the network anomalies affected to address security challenges in modern DevOps practices, contributing to advancing software security and reliability. 

---
