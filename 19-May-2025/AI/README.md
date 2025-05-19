# MOSAAIC: Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation 

**Authors**: Alayt Issak, Jeba Rezwana, Casper Harteveld  

**Link**: [PDF](https://arxiv.org/pdf/2505.11481)  

**Abstract**: Striking the appropriate balance between humans and co-creative AI is an open research question in computational creativity. Co-creativity, a form of hybrid intelligence where both humans and AI take action proactively, is a process that leads to shared creative artifacts and ideas. Achieving a balanced dynamic in co-creativity requires characterizing control and identifying strategies to distribute control between humans and AI. We define control as the power to determine, initiate, and direct the process of co-creation. Informed by a systematic literature review of 172 full-length papers, we introduce MOSAAIC (Managing Optimization towards Shared Autonomy, Authority, and Initiative in Co-creation), a novel framework for characterizing and balancing control in co-creation. MOSAAIC identifies three key dimensions of control: autonomy, initiative, and authority. We supplement our framework with control optimization strategies in co-creation. To demonstrate MOSAAIC's applicability, we analyze the distribution of control in six existing co-creative AI case studies and present the implications of using this framework. 

---
# Automatic Reward Shaping from Confounded Offline Data 

**Authors**: Mingxuan Li, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2505.11478)  

**Abstract**: A key task in Artificial Intelligence is learning effective policies for controlling agents in unknown environments to optimize performance measures. Off-policy learning methods, like Q-learning, allow learners to make optimal decisions based on past experiences. This paper studies off-policy learning from biased data in complex and high-dimensional domains where \emph{unobserved confounding} cannot be ruled out a priori. Building on the well-celebrated Deep Q-Network (DQN), we propose a novel deep reinforcement learning algorithm robust to confounding biases in observed data. Specifically, our algorithm attempts to find a safe policy for the worst-case environment compatible with the observations. We apply our method to twelve confounded Atari games, and find that it consistently dominates the standard DQN in all games where the observed input to the behavioral and target policies mismatch and unobserved confounders exist. 

---
# Extracting Explainable Dates From Medical Images By Reverse-Engineering UNIX Timestamps 

**Authors**: Lee Harris, James Bentham, Philippe De Wilde  

**Link**: [PDF](https://arxiv.org/pdf/2505.11451)  

**Abstract**: Dates often contribute towards highly impactful medical decisions, but it is rarely clear how to extract this data. AI has only just begun to be used transcribe such documents, and common methods are either to trust that the output produced by a complex AI model, or to parse the text using regular expressions. Recent work has established that regular expressions are an explainable form of logic, but it is difficult to decompose these into the component parts that are required to construct precise UNIX timestamps. First, we test publicly-available regular expressions, and we found that these were unable to capture a significant number of our dates. Next, we manually created easily-decomposable regular expressions, and we found that these were able to detect the majority of real dates, but also a lot of sequences of text that look like dates. Finally, we used regular expression synthesis to automatically identify regular expressions from the reverse-engineered UNIX timestamps that we created. We find that regular expressions created by regular expression synthesis detect far fewer sequences of text that look like dates than those that were manually created, at the cost of a slight increase to the number of missed dates. Overall, our results show that regular expressions can be created through regular expression synthesis to identify complex dates and date ranges in text transcriptions. To our knowledge, our proposed way of learning deterministic logic by reverse-engineering several many-one mappings and feeding these into a regular expression synthesiser is a new approach. 

---
# Meta-World+: An Improved, Standardized, RL Benchmark 

**Authors**: Reginald McLean, Evangelos Chatzaroulas, Luc McCutcheon, Frank Röder, Tianhe Yu, Zhanpeng He, K.R. Zentner, Ryan Julian, J K Terry, Isaac Woungang, Nariman Farsad, Pablo Samuel Castro  

**Link**: [PDF](https://arxiv.org/pdf/2505.11289)  

**Abstract**: Meta-World is widely used for evaluating multi-task and meta-reinforcement learning agents, which are challenged to master diverse skills simultaneously. Since its introduction however, there have been numerous undocumented changes which inhibit a fair comparison of algorithms. This work strives to disambiguate these results from the literature, while also leveraging the past versions of Meta-World to provide insights into multi-task and meta-reinforcement learning benchmark design. Through this process we release a new open-source version of Meta-World (this https URL) that has full reproducibility of past results, is more technically ergonomic, and gives users more control over the tasks that are included in a task set. 

---
# SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning 

**Authors**: Zheng Li, Qingxiu Dong, Jingyuan Ma, Di Zhang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.11274)  

**Abstract**: Recently, large reasoning models demonstrate exceptional performance on various tasks. However, reasoning models inefficiently over-process both trivial and complex queries, leading to resource waste and prolonged user latency. To address this challenge, we propose SelfBudgeter - a self-adaptive controllable reasoning strategy for efficient reasoning. Our approach adopts a dual-phase training paradigm: first, the model learns to pre-estimate the reasoning cost based on the difficulty of the query. Then, we introduce budget-guided GPRO for reinforcement learning, which effectively maintains accuracy while reducing output length. SelfBudgeter allows users to anticipate generation time and make informed decisions about continuing or interrupting the process. Furthermore, our method enables direct manipulation of reasoning length via pre-filling token budget. Experimental results demonstrate that SelfBudgeter can rationally allocate budgets according to problem complexity, achieving up to 74.47% response length compression on the MATH benchmark while maintaining nearly undiminished accuracy. 

---
# LD-Scene: LLM-Guided Diffusion for Controllable Generation of Adversarial Safety-Critical Driving Scenarios 

**Authors**: Mingxing Peng, Yuting Xie, Xusen Guo, Ruoyu Yao, Hai Yang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11247)  

**Abstract**: Ensuring the safety and robustness of autonomous driving systems necessitates a comprehensive evaluation in safety-critical scenarios. However, these safety-critical scenarios are rare and difficult to collect from real-world driving data, posing significant challenges to effectively assessing the performance of autonomous vehicles. Typical existing methods often suffer from limited controllability and lack user-friendliness, as extensive expert knowledge is essentially required. To address these challenges, we propose LD-Scene, a novel framework that integrates Large Language Models (LLMs) with Latent Diffusion Models (LDMs) for user-controllable adversarial scenario generation through natural language. Our approach comprises an LDM that captures realistic driving trajectory distributions and an LLM-based guidance module that translates user queries into adversarial loss functions, facilitating the generation of scenarios aligned with user queries. The guidance module integrates an LLM-based Chain-of-Thought (CoT) code generator and an LLM-based code debugger, enhancing the controllability and robustness in generating guidance functions. Extensive experiments conducted on the nuScenes dataset demonstrate that LD-Scene achieves state-of-the-art performance in generating realistic, diverse, and effective adversarial scenarios. Furthermore, our framework provides fine-grained control over adversarial behaviors, thereby facilitating more effective testing tailored to specific driving scenarios. 

---
# Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs 

**Authors**: Zhangying Feng, Qianglong Chen, Ning Lu, Yongqian Li, Siqi Cheng, Shuangmu Peng, Duyu Tang, Shengcai Liu, Zhirui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11227)  

**Abstract**: The development of reasoning capabilities represents a critical frontier in large language models (LLMs) research, where reinforcement learning (RL) and process reward models (PRMs) have emerged as predominant methodological frameworks. Contrary to conventional wisdom, empirical evidence from DeepSeek-R1 demonstrates that pure RL training focused on mathematical problem-solving can progressively enhance reasoning abilities without PRM integration, challenging the perceived necessity of process supervision. In this study, we conduct a systematic investigation of the relationship between RL training and PRM capabilities. Our findings demonstrate that problem-solving proficiency and process supervision capabilities represent complementary dimensions of reasoning that co-evolve synergistically during pure RL training. In particular, current PRMs underperform simple baselines like majority voting when applied to state-of-the-art models such as DeepSeek-R1 and QwQ-32B. To address this limitation, we propose Self-PRM, an introspective framework in which models autonomously evaluate and rerank their generated solutions through self-reward mechanisms. Although Self-PRM consistently improves the accuracy of the benchmark (particularly with larger sample sizes), analysis exposes persistent challenges: The approach exhibits low precision (<10\%) on difficult problems, frequently misclassifying flawed solutions as valid. These analyses underscore the need for continued RL scaling to improve reward alignment and introspective accuracy. Overall, our findings suggest that PRM may not be essential for enhancing complex reasoning, as pure RL not only improves problem-solving skills but also inherently fosters robust PRM capabilities. We hope these findings provide actionable insights for building more reliable and self-aware complex reasoning models. 

---
# GLOVA: Global and Local Variation-Aware Analog Circuit Design with Risk-Sensitive Reinforcement Learning 

**Authors**: Dongjun Kim, Junwoo Park, Chaehyeon Shin, Jaeheon Jung, Kyungho Shin, Seungheon Baek, Sanghyuk Heo, Woongrae Kim, Inchul Jeong, Joohwan Cho, Jongsun Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.11208)  

**Abstract**: Analog/mixed-signal circuit design encounters significant challenges due to performance degradation from process, voltage, and temperature (PVT) variations. To achieve commercial-grade reliability, iterative manual design revisions and extensive statistical simulations are required. While several studies have aimed to automate variation aware analog design to reduce time-to-market, the substantial mismatches in real-world wafers have not been thoroughly addressed. In this paper, we present GLOVA, an analog circuit sizing framework that effectively manages the impact of diverse random mismatches to improve robustness against PVT variations. In the proposed approach, risk-sensitive reinforcement learning is leveraged to account for the reliability bound affected by PVT variations, and ensemble-based critic is introduced to achieve sample-efficient learning. For design verification, we also propose $\mu$-$\sigma$ evaluation and simulation reordering method to reduce simulation costs of identifying failed designs. GLOVA supports verification through industrial-level PVT variation evaluation methods, including corner simulation as well as global and local Monte Carlo (MC) simulations. Compared to previous state-of-the-art variation-aware analog sizing frameworks, GLOVA achieves up to 80.5$\times$ improvement in sample efficiency and 76.0$\times$ reduction in time. 

---
# Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration 

**Authors**: Kasra Borazjani, Payam Abdisarabshali, Fardis Nadimi, Naji Khosravan, Minghui Liwang, Xianbin Wang, Yiguang Hong, Seyyedali Hosseinalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.11191)  

**Abstract**: As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: Foundation Models (FMs) provide a pathway toward generalization across tasks and modalities, whereas Federated Learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied environments. In this vision paper, we introduce Federated Foundation Models (FFMs) for embodied AI, a new paradigm that unifies the strengths of multi-modal multi-task (M3T) FMs with the privacy-preserving distributed nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying FFMs in embodied AI systems, along with the associated trade-offs. 

---
# Can Global XAI Methods Reveal Injected Bias in LLMs? SHAP vs Rule Extraction vs RuleSHAP 

**Authors**: Francesco Sovrano  

**Link**: [PDF](https://arxiv.org/pdf/2505.11189)  

**Abstract**: Generative AI systems can help spread information but also misinformation and biases, potentially undermining the UN Sustainable Development Goals (SDGs). Explainable AI (XAI) aims to reveal the inner workings of AI systems and expose misbehaviours or biases. However, current XAI tools, built for simpler models, struggle to handle the non-numerical nature of large language models (LLMs). This paper examines the effectiveness of global XAI methods, such as rule-extraction algorithms and SHAP, in detecting bias in LLMs. To do so, we first show a text-to-ordinal mapping strategy to convert non-numerical inputs/outputs into numerical features, enabling these tools to identify (some) misinformation-related biases in LLM-generated content. Then, we inject non-linear biases of varying complexity (univariate, conjunctive, and non-convex) into widespread LLMs like ChatGPT and Llama via system instructions, using global XAI methods to detect them. This way, we found that RuleFit struggles with conjunctive and non-convex biases, while SHAP can approximate conjunctive biases but cannot express them as actionable rules. Hence, we introduce RuleSHAP, a global rule extraction algorithm combining SHAP and RuleFit to detect more non-univariate biases, improving injected bias detection over RuleFit by +94% (MRR@1) on average. 

---
# Feasibility with Language Models for Open-World Compositional Zero-Shot Learning 

**Authors**: Jae Myung Kim, Stephan Alaniz, Cordelia Schmid, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11181)  

**Abstract**: Humans can easily tell if an attribute (also called state) is realistic, i.e., feasible, for an object, e.g. fire can be hot, but it cannot be wet. In Open-World Compositional Zero-Shot Learning, when all possible state-object combinations are considered as unseen classes, zero-shot predictors tend to perform poorly. Our work focuses on using external auxiliary knowledge to determine the feasibility of state-object combinations. Our Feasibility with Language Model (FLM) is a simple and effective approach that leverages Large Language Models (LLMs) to better comprehend the semantic relationships between states and objects. FLM involves querying an LLM about the feasibility of a given pair and retrieving the output logit for the positive answer. To mitigate potential misguidance of the LLM given that many of the state-object compositions are rare or completely infeasible, we observe that the in-context learning ability of LLMs is essential. We present an extensive study identifying Vicuna and ChatGPT as best performing, and we demonstrate that our FLM consistently improves OW-CZSL performance across all three benchmarks. 

---
# Reinforcement Learning for AMR Charging Decisions: The Impact of Reward and Action Space Design 

**Authors**: Janik Bischoff, Alexandru Rinciog, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11136)  

**Abstract**: We propose a novel reinforcement learning (RL) design to optimize the charging strategy for autonomous mobile robots in large-scale block stacking warehouses. RL design involves a wide array of choices that can mostly only be evaluated through lengthy experimentation. Our study focuses on how different reward and action space configurations, ranging from flexible setups to more guided, domain-informed design configurations, affect the agent performance. Using heuristic charging strategies as a baseline, we demonstrate the superiority of flexible, RL-based approaches in terms of service times. Furthermore, our findings highlight a trade-off: While more open-ended designs are able to discover well-performing strategies on their own, they may require longer convergence times and are less stable, whereas guided configurations lead to a more stable learning process but display a more limited generalization potential. Our contributions are threefold. First, we extend SLAPStack, an open-source, RL-compatible simulation-framework to accommodate charging strategies. Second, we introduce a novel RL design for tackling the charging strategy problem. Finally, we introduce several novel adaptive baseline heuristics and reproducibly evaluate the design using a Proximal Policy Optimization agent and varying different design configurations, with a focus on reward. 

---
# Scalability of Reinforcement Learning Methods for Dispatching in Semiconductor Frontend Fabs: A Comparison of Open-Source Models with Real Industry Datasets 

**Authors**: Patrick Stöckermann, Henning Südfeld, Alessandro Immordino, Thomas Altenmüller, Marc Wegmann, Martin Gebser, Konstantin Schekotihin, Georg Seidel, Chew Wye Chan, Fei Fei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11135)  

**Abstract**: Benchmark datasets are crucial for evaluating approaches to scheduling or dispatching in the semiconductor industry during the development and deployment phases. However, commonly used benchmark datasets like the Minifab or SMT2020 lack the complex details and constraints found in real-world scenarios. To mitigate this shortcoming, we compare open-source simulation models with a real industry dataset to evaluate how optimization methods scale with different levels of complexity. Specifically, we focus on Reinforcement Learning methods, performing optimization based on policy-gradient and Evolution Strategies. Our research provides insights into the effectiveness of these optimization methods and their applicability to realistic semiconductor frontend fab simulations. We show that our proposed Evolution Strategies-based method scales much better than a comparable policy-gradient-based approach. Moreover, we identify the selection and combination of relevant bottleneck tools to control by the agent as crucial for an efficient optimization. For the generalization across different loading scenarios and stochastic tool failure patterns, we achieve advantages when utilizing a diverse training dataset. While the overall approach is computationally expensive, it manages to scale well with the number of CPU cores used for training. For the real industry dataset, we achieve an improvement of up to 4% regarding tardiness and up to 1% regarding throughput. For the less complex open-source models Minifab and SMT2020, we observe double-digit percentage improvement in tardiness and single digit percentage improvement in throughput by use of Evolution Strategies. 

---
# Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining 

**Authors**: Yu Shi, Yitong Duan, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11122)  

**Abstract**: Alpha factor mining is pivotal in quantitative investment for identifying predictive signals from complex financial data. While traditional formulaic alpha mining relies on human expertise, contemporary automated methods, such as those based on genetic programming or reinforcement learning, often suffer from search inefficiency or yield poorly interpretable alpha factors. This paper introduces a novel framework that integrates Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) to overcome these limitations. Our approach leverages the LLM's instruction-following and reasoning capability to iteratively generate and refine symbolic alpha formulas within an MCTS-driven exploration. A key innovation is the guidance of MCTS exploration by rich, quantitative feedback from financial backtesting of each candidate factor, enabling efficient navigation of the vast search space. Furthermore, a frequent subtree avoidance mechanism is introduced to bolster search efficiency and alpha factor performance. Experimental results on real-world stock market data demonstrate that our LLM-based framework outperforms existing methods by mining alphas with superior predictive accuracy, trading performance, and improved interpretability, while offering a more efficient solution for formulaic alpha mining. 

---
# Predicting Student Dropout Risk With A Dual-Modal Abrupt Behavioral Changes Approach 

**Authors**: Jiabei Cheng, Zhen-Qun Yang, Jiannong Cao, Yu Yang, Xinzhe Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11119)  

**Abstract**: Timely prediction of students at high risk of dropout is critical for early intervention and improving educational outcomes. However, in offline educational settings, poor data quality, limited scale, and high heterogeneity often hinder the application of advanced machine learning models. Furthermore, while educational theories provide valuable insights into dropout phenomena, the lack of quantifiable metrics for key indicators limits their use in data-driven modeling. Through data analysis and a review of educational literature, we identified abrupt changes in student behavior as key early signals of dropout risk. To address this, we propose the Dual-Modal Multiscale Sliding Window (DMSW) Model, which integrates academic performance and behavioral data to dynamically capture behavior patterns using minimal data. The DMSW model improves prediction accuracy by 15% compared to traditional methods, enabling educators to identify high-risk students earlier, provide timely support, and foster a more inclusive learning environment. Our analysis highlights key behavior patterns, offering practical insights for preventive strategies and tailored support. These findings bridge the gap between theory and practice in dropout prediction, giving educators an innovative tool to enhance student retention and outcomes. 

---
# Group Think: Multiple Concurrent Reasoning Agents Collaborating at Token Level Granularity 

**Authors**: Chan-Jan Hsu, Davide Buffelli, Jamie McGowan, Feng-Ting Liao, Yi-Chang Chen, Sattar Vakili, Da-shan Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11107)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated the power of reasoning through self-generated chains of thought. Multiple reasoning agents can collaborate to raise joint reasoning quality above individual outcomes. However, such agents typically interact in a turn-based manner, trading increased latency for improved quality. In this paper, we propose Group Think--a single LLM that acts as multiple concurrent reasoning agents, or thinkers. With shared visibility into each other's partial generation progress, Group Think introduces a new concurrent-reasoning paradigm in which multiple reasoning trajectories adapt dynamically to one another at the token level. For example, a reasoning thread may shift its generation mid-sentence upon detecting that another thread is better positioned to continue. This fine-grained, token-level collaboration enables Group Think to reduce redundant reasoning and improve quality while achieving significantly lower latency. Moreover, its concurrent nature allows for efficient utilization of idle computational resources, making it especially suitable for edge inference, where very small batch size often underutilizes local~GPUs. We give a simple and generalizable modification that enables any existing LLM to perform Group Think on a local GPU. We also present an evaluation strategy to benchmark reasoning latency and empirically demonstrate latency improvements using open-source LLMs that were not explicitly trained for Group Think. We hope this work paves the way for future LLMs to exhibit more sophisticated and more efficient collaborative behavior for higher quality generation. 

---
# Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations for Sequential Data 

**Authors**: Keita Kinjo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11086)  

**Abstract**: Recently, the proliferation of omni-channel platforms has attracted interest in customer journeys, particularly regarding their role in developing marketing strategies. However, few efforts have been taken to quantitatively study or comprehensively analyze them owing to the sequential nature of their data and the complexity involved in analysis. In this study, we propose a novel approach comprising three steps for analyzing customer journeys. First, the distance between sequential data is defined and used to identify and visualize representative sequences. Second, the likelihood of purchase is predicted based on this distance. Third, if a sequence suggests no purchase, counterfactual sequences are recommended to increase the probability of a purchase using a proposed method, which extracts counterfactual explanations for sequential data. A survey was conducted, and the data were analyzed; the results revealed that typical sequences could be extracted, and the parts of those sequences important for purchase could be detected. We believe that the proposed approach can support improvements in various marketing activities. 

---
# A Multi-modal Fusion Network for Terrain Perception Based on Illumination Aware 

**Authors**: Rui Wang, Shichun Yang, Yuyi Chen, Zhuoyang Li, Zexiang Tong, Jianyi Xu, Jiayi Lu, Xinjie Feng, Yaoguang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11066)  

**Abstract**: Road terrains play a crucial role in ensuring the driving safety of autonomous vehicles (AVs). However, existing sensors of AVs, including cameras and Lidars, are susceptible to variations in lighting and weather conditions, making it challenging to achieve real-time perception of road conditions. In this paper, we propose an illumination-aware multi-modal fusion network (IMF), which leverages both exteroceptive and proprioceptive perception and optimizes the fusion process based on illumination features. We introduce an illumination-perception sub-network to accurately estimate illumination features. Moreover, we design a multi-modal fusion network which is able to dynamically adjust weights of different modalities according to illumination features. We enhance the optimization process by pre-training of the illumination-perception sub-network and incorporating illumination loss as one of the training constraints. Extensive experiments demonstrate that the IMF shows a superior performance compared to state-of-the-art methods. The comparison results with single modality perception methods highlight the comprehensive advantages of multi-modal fusion in accurately perceiving road terrains under varying lighting conditions. Our dataset is available at: this https URL. 

---
# Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction 

**Authors**: Changyue Jiang, Xudong Pan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11063)  

**Abstract**: LLM-based autonomous agents possess capabilities such as reasoning, tool invocation, and environment interaction, enabling the execution of complex multi-step tasks. The internal reasoning process, i.e., thought, of behavioral trajectory significantly influences tool usage and subsequent actions but can introduce potential risks. Even minor deviations in the agent's thought may trigger cascading effects leading to irreversible safety incidents. To address the safety alignment challenges in long-horizon behavioral trajectories, we propose Thought-Aligner, a plug-in dynamic thought correction module. Utilizing a lightweight and resource-efficient model, Thought-Aligner corrects each high-risk thought on the fly before each action execution. The corrected thought is then reintroduced to the agent, ensuring safer subsequent decisions and tool interactions. Importantly, Thought-Aligner modifies only the reasoning phase without altering the underlying agent framework, making it easy to deploy and widely applicable to various agent frameworks. To train the Thought-Aligner model, we construct an instruction dataset across ten representative scenarios and simulate ReAct execution trajectories, generating 5,000 diverse instructions and more than 11,400 safe and unsafe thought pairs. The model is fine-tuned using contrastive learning techniques. Experiments across three agent safety benchmarks involving 12 different LLMs demonstrate that Thought-Aligner raises agent behavioral safety from approximately 50% in the unprotected setting to 90% on average. Additionally, Thought-Aligner maintains response latency below 100ms with minimal resource usage, demonstrating its capability for efficient deployment, broad applicability, and timely responsiveness. This method thus provides a practical dynamic safety solution for the LLM-based agents. 

---
# GuardReasoner-VL: Safeguarding VLMs via Reinforced Reasoning 

**Authors**: Yue Liu, Shengfang Zhai, Mingzhe Du, Yulin Chen, Tri Cao, Hongcheng Gao, Cheng Wang, Xinfeng Li, Kun Wang, Junfeng Fang, Jiaheng Zhang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11049)  

**Abstract**: To enhance the safety of VLMs, this paper introduces a novel reasoning-based VLM guard model dubbed GuardReasoner-VL. The core idea is to incentivize the guard model to deliberatively reason before making moderation decisions via online RL. First, we construct GuardReasoner-VLTrain, a reasoning corpus with 123K samples and 631K reasoning steps, spanning text, image, and text-image inputs. Then, based on it, we cold-start our model's reasoning ability via SFT. In addition, we further enhance reasoning regarding moderation through online RL. Concretely, to enhance diversity and difficulty of samples, we conduct rejection sampling followed by data augmentation via the proposed safety-aware data concatenation. Besides, we use a dynamic clipping parameter to encourage exploration in early stages and exploitation in later stages. To balance performance and token efficiency, we design a length-aware safety reward that integrates accuracy, format, and token cost. Extensive experiments demonstrate the superiority of our model. Remarkably, it surpasses the runner-up by 19.27% F1 score on average. We release data, code, and models (3B/7B) of GuardReasoner-VL at this https URL 

---
# Most General Explanations of Tree Ensembles 

**Authors**: Yacine Izza, Alexey Ignatiev, Joao Marques-Silva, Peter J. Stuckey  

**Link**: [PDF](https://arxiv.org/pdf/2505.10991)  

**Abstract**: Explainable Artificial Intelligence (XAI) is critical for attaining trust in the operation of AI systems. A key question of an AI system is ``why was this decision made this way''. Formal approaches to XAI use a formal model of the AI system to identify abductive explanations. While abductive explanations may be applicable to a large number of inputs sharing the same concrete values, more general explanations may be preferred for numeric inputs. So-called inflated abductive explanations give intervals for each feature ensuring that any input whose values fall withing these intervals is still guaranteed to make the same prediction. Inflated explanations cover a larger portion of the input space, and hence are deemed more general explanations. But there can be many (inflated) abductive explanations for an instance. Which is the best? In this paper, we show how to find a most general abductive explanation for an AI decision. This explanation covers as much of the input space as possible, while still being a correct formal explanation of the model's behaviour. Given that we only want to give a human one explanation for a decision, the most general explanation gives us the explanation with the broadest applicability, and hence the one most likely to seem sensible. (The paper has been accepted at IJCAI2025 conference.) 

---
# RAGSynth: Synthetic Data for Robust and Faithful RAG Component Optimization 

**Authors**: Haiyang Shen, Hang Yan, Zhongshi Xing, Mugeng Liu, Yue Li, Zhiyang Chen, Yuxiang Wang, Jiuzheng Wang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.10989)  

**Abstract**: RAG can enhance the performance of LLMs on knowledge-intensive tasks. Various RAG paradigms, including vanilla, planning-based, and iterative RAG, are built upon 2 cores: the retriever, which should robustly select relevant documents across complex queries, and the generator, which should faithfully synthesize responses. However, existing retrievers rely heavily on public knowledge and struggle with queries of varying logical complexity and clue completeness, while generators frequently face fidelity problems. In this work, we introduce RAGSynth, a framework that includes a data construction modeling and a corresponding synthetic data generation implementation, designed to optimize retriever robustness and generator fidelity. Additionally, we present SynthBench, a benchmark encompassing 8 domain-specific documents across 4 domains, featuring diverse query complexities, clue completeness, and fine-grained citation granularity. Leveraging RAGSynth, we generate a large-scale synthetic dataset, including single and multi-hop. Extensive experiments demonstrate that the synthetic data significantly improves the robustness of the retrievers and the fidelity of the generators. Additional evaluations confirm that RAGSynth can also generalize well across different domains. By integrating the optimized retrievers into various RAG paradigms, we consistently observe enhanced RAG system performance. We have open-sourced the implementation on this https URL. 

---
# DRL-Based Injection Molding Process Parameter Optimization for Adaptive and Profitable Production 

**Authors**: Joon-Young Kim, Jecheon Yu, Heekyu Kim, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10988)  

**Abstract**: Plastic injection molding remains essential to modern manufacturing. However, optimizing process parameters to balance product quality and profitability under dynamic environmental and economic conditions remains a persistent challenge. This study presents a novel deep reinforcement learning (DRL)-based framework for real-time process optimization in injection molding, integrating product quality and profitability into the control objective. A profit function was developed to reflect real-world manufacturing costs, incorporating resin, mold wear, and electricity prices, including time-of-use variations. Surrogate models were constructed to predict product quality and cycle time, enabling efficient offline training of DRL agents using soft actor-critic (SAC) and proximal policy optimization (PPO) algorithms. Experimental results demonstrate that the proposed DRL framework can dynamically adapt to seasonal and operational variations, consistently maintaining product quality while maximizing profit. Compared to traditional optimization methods such as genetic algorithms, the DRL models achieved comparable economic performance with up to 135x faster inference speeds, making them well-suited for real-time applications. The framework's scalability and adaptability highlight its potential as a foundation for intelligent, data-driven decision-making in modern manufacturing environments. 

---
# Facets in Argumentation: A Formal Approach to Argument Significance 

**Authors**: Johannes Fichte, Nicolas Fröhlich, Markus Hecher, Victor Lagerkvist, Yasir Mahmood, Arne Meier, Jonathan Persson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10982)  

**Abstract**: Argumentation is a central subarea of Artificial Intelligence (AI) for modeling and reasoning about arguments. The semantics of abstract argumentation frameworks (AFs) is given by sets of arguments (extensions) and conditions on the relationship between them, such as stable or admissible. Today's solvers implement tasks such as finding extensions, deciding credulous or skeptical acceptance, counting, or enumerating extensions. While these tasks are well charted, the area between decision, counting/enumeration and fine-grained reasoning requires expensive reasoning so far. We introduce a novel concept (facets) for reasoning between decision and enumeration. Facets are arguments that belong to some extensions (credulous) but not to all extensions (skeptical). They are most natural when a user aims to navigate, filter, or comprehend the significance of specific arguments, according to their needs. We study the complexity and show that tasks involving facets are much easier than counting extensions. Finally, we provide an implementation, and conduct experiments to demonstrate feasibility. 

---
# Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory 

**Authors**: Yexiang Liu, Zekun Li, Zhi Fang, Nan Xu, Ran He, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10981)  

**Abstract**: Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a method according to probability theory to quickly and accurately predict the scaling performance and select the best strategy under large sampling times without extra resource-intensive inference in practice. It can serve as the test-time scaling law for majority voting. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. 

---
# MPS-Prover: Advancing Stepwise Theorem Proving by Multi-Perspective Search and Data Curation 

**Authors**: Zhenwen Liang, Linfeng Song, Yang Li, Tao Yang, Feng Zhang, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10962)  

**Abstract**: Automated Theorem Proving (ATP) in formal languages remains a formidable challenge in AI, demanding rigorous logical deduction and navigating vast search spaces. While large language models (LLMs) have shown promising performance, existing stepwise provers often suffer from biased search guidance, leading to inefficiencies and suboptimal proof strategies. This paper introduces the Multi-Perspective Search Prover (MPS-Prover), a novel stepwise ATP system designed to overcome these limitations. MPS-Prover incorporates two key innovations: a highly effective post-training data curation strategy that prunes approximately 40% of redundant training data without sacrificing performance, and a multi-perspective tree search mechanism. This search integrates a learned critic model with strategically designed heuristic rules to diversify tactic selection, prevent getting trapped in unproductive states, and enhance search robustness. Extensive evaluations demonstrate that MPS-Prover achieves state-of-the-art performance on multiple challenging benchmarks, including miniF2F and ProofNet, outperforming prior 7B parameter models. Furthermore, our analyses reveal that MPS-Prover generates significantly shorter and more diverse proofs compared to existing stepwise and whole-proof methods, highlighting its efficiency and efficacy. Our work advances the capabilities of LLM-based formal reasoning and offers a robust framework and a comprehensive analysis for developing more powerful theorem provers. 

---
# InfantAgent-Next: A Multimodal Generalist Agent for Automated Computer Interaction 

**Authors**: Bin Lei, Weitai Kang, Zijian Zhang, Winson Chen, Xi Xie, Shan Zuo, Mimi Xie, Ali Payani, Mingyi Hong, Yan Yan, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.10887)  

**Abstract**: This paper introduces \textsc{InfantAgent-Next}, a generalist agent capable of interacting with computers in a multimodal manner, encompassing text, images, audio, and video. Unlike existing approaches that either build intricate workflows around a single large model or only provide workflow modularity, our agent integrates tool-based and pure vision agents within a highly modular architecture, enabling different models to collaboratively solve decoupled tasks in a step-by-step manner. Our generality is demonstrated by our ability to evaluate not only pure vision-based real-world benchmarks (i.e., OSWorld), but also more general or tool-intensive benchmarks (e.g., GAIA and SWE-Bench). Specifically, we achieve $\mathbf{7.27\%}$ accuracy on OSWorld, higher than Claude-Computer-Use. Codes and evaluation scripts are open-sourced at this https URL. 

---
# MCU: Improving Machine Unlearning through Mode Connectivity 

**Authors**: Yingdan Shi, Ren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10859)  

**Abstract**: Machine Unlearning (MU) aims to remove the information of specific training data from a trained model, ensuring compliance with privacy regulations and user requests. While one line of existing MU methods relies on linear parameter updates via task arithmetic, they suffer from weight entanglement. In this work, we propose a novel MU framework called Mode Connectivity Unlearning (MCU) that leverages mode connectivity to find an unlearning pathway in a nonlinear manner. To further enhance performance and efficiency, we introduce a parameter mask strategy that not only improves unlearning effectiveness but also reduces computational overhead. Moreover, we propose an adaptive adjustment strategy for our unlearning penalty coefficient to adaptively balance forgetting quality and predictive performance during training, eliminating the need for empirical hyperparameter tuning. Unlike traditional MU methods that identify only a single unlearning model, MCU uncovers a spectrum of unlearning models along the pathway. Overall, MCU serves as a plug-and-play framework that seamlessly integrates with any existing MU methods, consistently improving unlearning efficacy. Extensive experiments on the image classification task demonstrate that MCU achieves superior performance. 

---
# Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models 

**Authors**: Simeng Han, Stephen Xia, Grant Zhang, Howard Dai, Chen Liu, Lichang Chen, Hoang Huy Nguyen, Hongyuan Mei, Jiayuan Mao, R. Thomas McCoy  

**Link**: [PDF](https://arxiv.org/pdf/2505.10844)  

**Abstract**: Accuracy remains a standard metric for evaluating AI systems, but it offers limited insight into how models arrive at their solutions. In this work, we introduce a benchmark based on brainteasers written in long narrative form to probe more deeply into the types of reasoning strategies that models use. Brainteasers are well-suited for this goal because they can be solved with multiple approaches, such as a few-step solution that uses a creative insight or a longer solution that uses more brute force. We investigate large language models (LLMs) across multiple layers of reasoning, focusing not only on correctness but also on the quality and creativity of their solutions. We investigate many aspects of the reasoning process: (1) semantic parsing of the brainteasers into precise mathematical competition style formats; (2) generating solutions from these mathematical forms; (3) self-correcting solutions based on gold solutions; (4) producing step-by-step sketches of solutions; and (5) making use of hints. We find that LLMs are in many cases able to find creative, insightful solutions to brainteasers, suggesting that they capture some of the capacities needed to solve novel problems in creative ways. Nonetheless, there also remain situations where they rely on brute force despite the availability of more efficient, creative solutions, highlighting a potential direction for improvement in the reasoning abilities of LLMs. 

---
# TACO: Rethinking Semantic Communications with Task Adaptation and Context Embedding 

**Authors**: Achintha Wijesinghe, Weiwei Wang, Suchinthaka Wanninayaka, Songyang Zhang, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.10834)  

**Abstract**: Recent advancements in generative artificial intelligence have introduced groundbreaking approaches to innovating next-generation semantic communication, which prioritizes conveying the meaning of a message rather than merely transmitting raw data. A fundamental challenge in semantic communication lies in accurately identifying and extracting the most critical semantic information while adapting to downstream tasks without degrading performance, particularly when the objective at the receiver may evolve over time. To enable flexible adaptation to multiple tasks at the receiver, this work introduces a novel semantic communication framework, which is capable of jointly capturing task-specific information to enhance downstream task performance and contextual information. Through rigorous experiments on popular image datasets and computer vision tasks, our framework shows promising improvement compared to existing work, including superior performance in downstream tasks, better generalizability, ultra-high bandwidth efficiency, and low reconstruction latency. 

---
# PoE-World: Compositional World Modeling with Products of Programmatic Experts 

**Authors**: Wasu Top Piriyakulkij, Yichao Liang, Hao Tang, Adrian Weller, Marta Kryven, Kevin Ellis  

**Link**: [PDF](https://arxiv.org/pdf/2505.10819)  

**Abstract**: Learning how the world works is central to building AI agents that can adapt to complex environments. Traditional world models based on deep learning demand vast amounts of training data, and do not flexibly update their knowledge from sparse observations. Recent advances in program synthesis using Large Language Models (LLMs) give an alternate approach which learns world models represented as source code, supporting strong generalization from little data. To date, application of program-structured world models remains limited to natural language and grid-world domains. We introduce a novel program synthesis method for effectively modeling complex, non-gridworld domains by representing a world model as an exponentially-weighted product of programmatic experts (PoE-World) synthesized by LLMs. We show that this approach can learn complex, stochastic world models from just a few observations. We evaluate the learned world models by embedding them in a model-based planning agent, demonstrating efficient performance and generalization to unseen levels on Atari's Pong and Montezuma's Revenge. We release our code and display the learned world models and videos of the agent's gameplay at this https URL. 

---
# Developing and Integrating Trust Modeling into Multi-Objective Reinforcement Learning for Intelligent Agricultural Management 

**Authors**: Zhaoan Wang, Wonseok Jang, Bowen Ruan, Jun Wang, Shaoping Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10803)  

**Abstract**: Precision agriculture, enhanced by artificial intelligence (AI), offers promising tools such as remote sensing, intelligent irrigation, fertilization management, and crop simulation to improve agricultural efficiency and sustainability. Reinforcement learning (RL), in particular, has outperformed traditional methods in optimizing yields and resource management. However, widespread AI adoption is limited by gaps between algorithmic recommendations and farmers' practical experience, local knowledge, and traditional practices. To address this, our study emphasizes Human-AI Interaction (HAII), focusing on transparency, usability, and trust in RL-based farm management. We employ a well-established trust framework - comprising ability, benevolence, and integrity - to develop a novel mathematical model quantifying farmers' confidence in AI-based fertilization strategies. Surveys conducted with farmers for this research reveal critical misalignments, which are integrated into our trust model and incorporated into a multi-objective RL framework. Unlike prior methods, our approach embeds trust directly into policy optimization, ensuring AI recommendations are technically robust, economically feasible, context-aware, and socially acceptable. By aligning technical performance with human-centered trust, this research supports broader AI adoption in agriculture. 

---
# SECRET: Semi-supervised Clinical Trial Document Similarity Search 

**Authors**: Trisha Das, Afrah Shafquat, Beigi Mandis, Jacob Aptekar, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.10780)  

**Abstract**: Clinical trials are vital for evaluation of safety and efficacy of new treatments. However, clinical trials are resource-intensive, time-consuming and expensive to conduct, where errors in trial design, reduced efficacy, and safety events can result in significant delays, financial losses, and damage to reputation. These risks underline the importance of informed and strategic decisions in trial design to mitigate these risks and improve the chances of a successful trial. Identifying similar historical trials is critical as these trials can provide an important reference for potential pitfalls and challenges including serious adverse events, dosage inaccuracies, recruitment difficulties, patient adherence issues, etc. Addressing these challenges in trial design can lead to development of more effective study protocols with optimized patient safety and trial efficiency. In this paper, we present a novel method to identify similar historical trials by summarizing clinical trial protocols and searching for similar trials based on a query trial's protocol. Our approach significantly outperforms all baselines, achieving up to a 78% improvement in recall@1 and a 53% improvement in precision@1 over the best baseline. We also show that our method outperforms all other baselines in partial trial similarity search and zero-shot patient-trial matching, highlighting its superior utility in these tasks. 

---
# Qualia Optimization 

**Authors**: Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2505.10779)  

**Abstract**: This report explores the speculative question: what if current or future AI systems have qualia, such as pain or pleasure? It does so by assuming that AI systems might someday possess qualia -- and that the quality of these subjective experiences should be considered alongside performance metrics. Concrete mathematical problem settings, inspired by reinforcement learning formulations and theories from philosophy of mind, are then proposed and initial approaches and properties are presented. These properties enable refinement of the problem setting, culminating with the proposal of methods that promote reinforcement. 

---
# Code-Driven Planning in Grid Worlds with Large Language Models 

**Authors**: Ashwath Vaithinathan Aravindan, Zhisheng Tang, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2505.10749)  

**Abstract**: We propose an iterative programmatic planning (IPP) framework for solving grid-based tasks by synthesizing interpretable agent policies expressed in code using large language models (LLMs). Instead of relying on traditional search or reinforcement learning, our approach uses code generation as policy synthesis, where the LLM outputs executable programs that map environment states to action sequences. Our proposed architecture incorporates several prompting strategies, including direct code generation, pseudocode-conditioned refinement, and curriculum-based prompting, but also includes an iterative refinement mechanism that updates code based on task performance feedback. We evaluate our approach using six leading LLMs and two challenging grid-based benchmarks (GRASP and MiniGrid). Our IPP framework demonstrates improvements over direct code generation ranging from 10\% to as much as 10x across five of the six models and establishes a new state-of-the-art result for GRASP. IPP is found to significantly outperform direct elicitation of a solution from GPT-o3-mini (by 63\% on MiniGrid to 116\% on GRASP), demonstrating the viability of the overall approach. Computational costs of all code generation approaches are similar. While code generation has a higher initial prompting cost compared to direct solution elicitation (\$0.08 per task vs. \$0.002 per instance for GPT-o3-mini), the code can be reused for any number of instances, making the amortized cost significantly lower (by 400x on GPT-o3-mini across the complete GRASP benchmark). 

---
# Evaluations at Work: Measuring the Capabilities of GenAI in Use 

**Authors**: Brandon Lepine, Gawesha Weerantunga, Juho Kim, Pamela Mishkin, Matthew Beane  

**Link**: [PDF](https://arxiv.org/pdf/2505.10742)  

**Abstract**: Current AI benchmarks miss the messy, multi-turn nature of human-AI collaboration. We present an evaluation framework that decomposes real-world tasks into interdependent subtasks, letting us track both LLM performance and users' strategies across a dialogue. Complementing this framework, we develop a suite of metrics, including a composite usage derived from semantic similarity, word overlap, and numerical matches; structural coherence; intra-turn diversity; and a novel measure of the "information frontier" reflecting the alignment between AI outputs and users' working knowledge. We demonstrate our methodology in a financial valuation task that mirrors real-world complexity. Our empirical findings reveal that while greater integration of LLM-generated content generally enhances output quality, its benefits are moderated by factors such as response incoherence, excessive subtask diversity, and the distance of provided information from users' existing knowledge. These results suggest that proactive dialogue strategies designed to inject novelty may inadvertently undermine task performance. Our work thus advances a more holistic evaluation of human-AI collaboration, offering both a robust methodological framework and actionable insights for developing more effective AI-augmented work processes. 

---
# Embodied AI in Machine Learning -- is it Really Embodied? 

**Authors**: Matej Hoffmann, Shubhan Parag Patni  

**Link**: [PDF](https://arxiv.org/pdf/2505.10705)  

**Abstract**: Embodied Artificial Intelligence (Embodied AI) is gaining momentum in the machine learning communities with the goal of leveraging current progress in AI (deep learning, transformers, large language and visual-language models) to empower robots. In this chapter we put this work in the context of "Good Old-Fashioned Artificial Intelligence" (GOFAI) (Haugeland, 1989) and the behavior-based or embodied alternatives (R. A. Brooks 1991; Pfeifer and Scheier 2001). We claim that the AI-powered robots are only weakly embodied and inherit some of the problems of GOFAI. Moreover, we review and critically discuss the possibility of cross-embodiment learning (Padalkar et al. 2024). We identify fundamental roadblocks and propose directions on how to make progress. 

---
# Interpretable Risk Mitigation in LLM Agent Systems 

**Authors**: Jan Chojnacki  

**Link**: [PDF](https://arxiv.org/pdf/2505.10670)  

**Abstract**: Autonomous agents powered by large language models (LLMs) enable novel use cases in domains where responsible action is increasingly important. Yet the inherent unpredictability of LLMs raises safety concerns about agent reliability. In this work, we explore agent behaviour in a toy, game-theoretic environment based on a variation of the Iterated Prisoner's Dilemma. We introduce a strategy-modification method-independent of both the game and the prompt-by steering the residual stream with interpretable features extracted from a sparse autoencoder latent space. Steering with the good-faith negotiation feature lowers the average defection probability by 28 percentage points. We also identify feasible steering ranges for several open-source LLM agents. Finally, we hypothesise that game-theoretic evaluation of LLM agents, combined with representation-steering alignment, can generalise to real-world applications on end-user devices and embodied platforms. 

---
# On the Evaluation of Engineering Artificial General Intelligence 

**Authors**: Sandeep Neema, Susmit Jha, Adam Nagel, Ethan Lew, Chandrasekar Sureshkumar, Aleksa Gordic, Chase Shimmin, Hieu Nguygen, Paul Eremenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.10653)  

**Abstract**: We discuss the challenges and propose a framework for evaluating engineering artificial general intelligence (eAGI) agents. We consider eAGI as a specialization of artificial general intelligence (AGI), deemed capable of addressing a broad range of problems in the engineering of physical systems and associated controllers. We exclude software engineering for a tractable scoping of eAGI and expect dedicated software engineering AI agents to address the software implementation challenges. Similar to human engineers, eAGI agents should possess a unique blend of background knowledge (recall and retrieve) of facts and methods, demonstrate familiarity with tools and processes, exhibit deep understanding of industrial components and well-known design families, and be able to engage in creative problem solving (analyze and synthesize), transferring ideas acquired in one context to another. Given this broad mandate, evaluating and qualifying the performance of eAGI agents is a challenge in itself and, arguably, a critical enabler to developing eAGI agents. In this paper, we address this challenge by proposing an extensible evaluation framework that specializes and grounds Bloom's taxonomy - a framework for evaluating human learning that has also been recently used for evaluating LLMs - in an engineering design context. Our proposed framework advances the state of the art in benchmarking and evaluation of AI agents in terms of the following: (a) developing a rich taxonomy of evaluation questions spanning from methodological knowledge to real-world design problems; (b) motivating a pluggable evaluation framework that can evaluate not only textual responses but also evaluate structured design artifacts such as CAD models and SysML models; and (c) outlining an automatable procedure to customize the evaluation benchmark to different engineering contexts. 

---
# Modeling cognitive processes of natural reading with transformer-based Language Models 

**Authors**: Bruno Bianchi, Fermín Travi, Juan E. Kamienkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11485)  

**Abstract**: Recent advances in Natural Language Processing (NLP) have led to the development of highly sophisticated language models for text generation. In parallel, neuroscience has increasingly employed these models to explore cognitive processes involved in language comprehension. Previous research has shown that models such as N-grams and LSTM networks can partially account for predictability effects in explaining eye movement behaviors, specifically Gaze Duration, during reading. In this study, we extend these findings by evaluating transformer-based models (GPT2, LLaMA-7B, and LLaMA2-7B) to further investigate this relationship. Our results indicate that these architectures outperform earlier models in explaining the variance in Gaze Durations recorded from Rioplantense Spanish readers. However, similar to previous studies, these models still fail to account for the entirety of the variance captured by human predictability. These findings suggest that, despite their advancements, state-of-the-art language models continue to predict language in ways that differ from human readers. 

---
# Improving Assembly Code Performance with Large Language Models via Reinforcement Learning 

**Authors**: Anjiang Wei, Tarun Suresh, Huanmi Tan, Yinglun Xu, Gagandeep Singh, Ke Wang, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2505.11480)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance across a wide range of programming tasks, yet their potential for code optimization remains underexplored. This work investigates whether LLMs can optimize the performance of assembly code, where fine-grained control over execution enables improvements that are difficult to express in high-level languages. We present a reinforcement learning framework that trains LLMs using Proximal Policy Optimization (PPO), guided by a reward function that considers both functional correctness, validated through test cases, and execution performance relative to the industry-standard compiler gcc -O3. To support this study, we introduce a benchmark of 8,072 real-world programs. Our model, Qwen2.5-Coder-7B-PPO, achieves 96.0% test pass rates and an average speedup of 1.47x over the gcc -O3 baseline, outperforming all 20 other models evaluated, including Claude-3.7-sonnet. These results indicate that reinforcement learning can unlock the potential of LLMs to serve as effective optimizers for assembly code performance. 

---
# HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages 

**Authors**: Zhilin Wang, Jiaqi Zeng, Olivier Delalleau, Hoo-Chang Shin, Felipe Soares, Alexander Bukharin, Ellie Evans, Yi Dong, Oleksii Kuchaiev  

**Link**: [PDF](https://arxiv.org/pdf/2505.11475)  

**Abstract**: Preference datasets are essential for training general-domain, instruction-following language models with Reinforcement Learning from Human Feedback (RLHF). Each subsequent data release raises expectations for future data collection, meaning there is a constant need to advance the quality and diversity of openly available preference data. To address this need, we introduce HelpSteer3-Preference, a permissively licensed (CC-BY-4.0), high-quality, human-annotated preference dataset comprising of over 40,000 samples. These samples span diverse real-world applications of large language models (LLMs), including tasks relating to STEM, coding and multilingual scenarios. Using HelpSteer3-Preference, we train Reward Models (RMs) that achieve top performance on RM-Bench (82.4%) and JudgeBench (73.7%). This represents a substantial improvement (~10% absolute) over the previously best-reported results from existing RMs. We demonstrate HelpSteer3-Preference can also be applied to train Generative RMs and how policy models can be aligned with RLHF using our RMs. Dataset (CC-BY-4.0): this https URL 

---
# Disentangling Reasoning and Knowledge in Medical Large Language Models 

**Authors**: Rahul Thapa, Qingyang Wu, Kevin Wu, Harrison Zhang, Angela Zhang, Eric Wu, Haotian Ye, Suhana Bedi, Nevin Aresh, Joseph Boen, Shriya Reddy, Ben Athiwaratkun, Shuaiwen Leon Song, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11462)  

**Abstract**: Medical reasoning in large language models (LLMs) aims to emulate clinicians' diagnostic thinking, but current benchmarks such as MedQA-USMLE, MedMCQA, and PubMedQA often mix reasoning with factual recall. We address this by separating 11 biomedical QA benchmarks into reasoning- and knowledge-focused subsets using a PubMedBERT classifier that reaches 81 percent accuracy, comparable to human performance. Our analysis shows that only 32.8 percent of questions require complex reasoning. We evaluate biomedical models (HuatuoGPT-o1, MedReason, m1) and general-domain models (DeepSeek-R1, o4-mini, Qwen3), finding consistent gaps between knowledge and reasoning performance. For example, m1 scores 60.5 on knowledge but only 47.1 on reasoning. In adversarial tests where models are misled with incorrect initial reasoning, biomedical models degrade sharply, while larger or RL-trained general models show more robustness. To address this, we train BioMed-R1 using fine-tuning and reinforcement learning on reasoning-heavy examples. It achieves the strongest performance among similarly sized models. Further gains may come from incorporating clinical case reports and training with adversarial and backtracking scenarios. 

---
# HumaniBench: A Human-Centric Framework for Large Multimodal Models Evaluation 

**Authors**: Shaina Raza, Aravind Narayanan, Vahid Reza Khazaie, Ashmal Vayani, Mukund S. Chettiar, Amandeep Singh, Mubarak Shah, Deval Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2505.11454)  

**Abstract**: Large multimodal models (LMMs) now excel on many vision language benchmarks, however, they still struggle with human centered criteria such as fairness, ethics, empathy, and inclusivity, key to aligning with human values. We introduce HumaniBench, a holistic benchmark of 32K real-world image question pairs, annotated via a scalable GPT4o assisted pipeline and exhaustively verified by domain experts. HumaniBench evaluates seven Human Centered AI (HCAI) principles: fairness, ethics, understanding, reasoning, language inclusivity, empathy, and robustness, across seven diverse tasks, including open and closed ended visual question answering (VQA), multilingual QA, visual grounding, empathetic captioning, and robustness tests. Benchmarking 15 state of the art LMMs (open and closed source) reveals that proprietary models generally lead, though robustness and visual grounding remain weak points. Some open-source models also struggle to balance accuracy with adherence to human-aligned principles. HumaniBench is the first benchmark purpose built around HCAI principles. It provides a rigorous testbed for diagnosing alignment gaps and guiding LMMs toward behavior that is both accurate and socially responsible. Dataset, annotation prompts, and evaluation code are available at: this https URL 

---
# LLMs unlock new paths to monetizing exploits 

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11449)  

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.
We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches. 

---
# SurgPose: Generalisable Surgical Instrument Pose Estimation using Zero-Shot Learning and Stereo Vision 

**Authors**: Utsav Rai, Haozheng Xu, Stamatia Giannarou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11439)  

**Abstract**: Accurate pose estimation of surgical tools in Robot-assisted Minimally Invasive Surgery (RMIS) is essential for surgical navigation and robot control. While traditional marker-based methods offer accuracy, they face challenges with occlusions, reflections, and tool-specific designs. Similarly, supervised learning methods require extensive training on annotated datasets, limiting their adaptability to new tools. Despite their success in other domains, zero-shot pose estimation models remain unexplored in RMIS for pose estimation of surgical instruments, creating a gap in generalising to unseen surgical tools. This paper presents a novel 6 Degrees of Freedom (DoF) pose estimation pipeline for surgical instruments, leveraging state-of-the-art zero-shot RGB-D models like the FoundationPose and SAM-6D. We advanced these models by incorporating vision-based depth estimation using the RAFT-Stereo method, for robust depth estimation in reflective and textureless environments. Additionally, we enhanced SAM-6D by replacing its instance segmentation module, Segment Anything Model (SAM), with a fine-tuned Mask R-CNN, significantly boosting segmentation accuracy in occluded and complex conditions. Extensive validation reveals that our enhanced SAM-6D surpasses FoundationPose in zero-shot pose estimation of unseen surgical instruments, setting a new benchmark for zero-shot RGB-D pose estimation in RMIS. This work enhances the generalisability of pose estimation for unseen objects and pioneers the application of RGB-D zero-shot methods in RMIS. 

---
# GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art 

**Authors**: Chenkai Zhang, Yiming Lei, Zeming Liu, Haitao Leng, Shaoguo Liu, Tingting Gao, Qingjie Liu, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11436)  

**Abstract**: Video Comment Art enhances user engagement by providing creative content that conveys humor, satire, or emotional resonance, requiring a nuanced and comprehensive grasp of cultural and contextual subtleties. Although Multimodal Large Language Models (MLLMs) and Chain-of-Thought (CoT) have demonstrated strong reasoning abilities in STEM tasks (e.g. mathematics and coding), they still struggle to generate creative expressions such as resonant jokes and insightful satire. Moreover, existing benchmarks are constrained by their limited modalities and insufficient categories, hindering the exploration of comprehensive creativity in video-based Comment Art creation. To address these limitations, we introduce GODBench, a novel benchmark that integrates video and text modalities to systematically evaluate MLLMs' abilities to compose Comment Art. Furthermore, inspired by the propagation patterns of waves in physics, we propose Ripple of Thought (RoT), a multi-step reasoning framework designed to enhance the creativity of MLLMs. Extensive experiments reveal that existing MLLMs and CoT methods still face significant challenges in understanding and generating creative video comments. In contrast, RoT provides an effective approach to improve creative composing, highlighting its potential to drive meaningful advancements in MLLM-based creativity. GODBench is publicly available at this https URL. 

---
# Mergenetic: a Simple Evolutionary Model Merging Library 

**Authors**: Adrian Robert Minut, Tommaso Mencattini, Andrea Santilli, Donato Crisostomi, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2505.11427)  

**Abstract**: Model merging allows combining the capabilities of existing models into a new one - post hoc, without additional training. This has made it increasingly popular thanks to its low cost and the availability of libraries that support merging on consumer GPUs. Recent work shows that pairing merging with evolutionary algorithms can boost performance, but no framework currently supports flexible experimentation with such strategies in language models. We introduce Mergenetic, an open-source library for evolutionary model merging. Mergenetic enables easy composition of merging methods and evolutionary algorithms while incorporating lightweight fitness estimators to reduce evaluation costs. We describe its design and demonstrate that Mergenetic produces competitive results across tasks and languages using modest hardware. 

---
# Improving Object Detection Performance through YOLOv8: A Comprehensive Training and Evaluation Study 

**Authors**: Rana Poureskandar, Shiva Razzagzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11424)  

**Abstract**: This study evaluated the performance of a YOLOv8-based segmentation model for detecting and segmenting wrinkles in facial images. 

---
# EdgeWisePersona: A Dataset for On-Device User Profiling from Natural Language Interactions 

**Authors**: Patryk Bartkowiak, Michal Podstawski  

**Link**: [PDF](https://arxiv.org/pdf/2505.11417)  

**Abstract**: This paper introduces a novel dataset and evaluation benchmark designed to assess and improve small language models deployable on edge devices, with a focus on user profiling from multi-session natural language interactions in smart home environments. At the core of the dataset are structured user profiles, each defined by a set of routines - context-triggered, repeatable patterns of behavior that govern how users interact with their home systems. Using these profiles as input, a large language model (LLM) generates corresponding interaction sessions that simulate realistic, diverse, and context-aware dialogues between users and their devices.
The primary task supported by this dataset is profile reconstruction: inferring user routines and preferences solely from interactions history. To assess how well current models can perform this task under realistic conditions, we benchmarked several state-of-the-art compact language models and compared their performance against large foundation models. Our results show that while small models demonstrate some capability in reconstructing profiles, they still fall significantly short of large models in accurately capturing user behavior. This performance gap poses a major challenge - particularly because on-device processing offers critical advantages, such as preserving user privacy, minimizing latency, and enabling personalized experiences without reliance on the cloud. By providing a realistic, structured testbed for developing and evaluating behavioral modeling under these constraints, our dataset represents a key step toward enabling intelligent, privacy-respecting AI systems that learn and adapt directly on user-owned devices. 

---
# MID-L: Matrix-Interpolated Dropout Layer with Layer-wise Neuron Selection 

**Authors**: Pouya Shaeri, Ariane Middel  

**Link**: [PDF](https://arxiv.org/pdf/2505.11416)  

**Abstract**: Modern neural networks often activate all neurons for every input, leading to unnecessary computation and inefficiency. We introduce Matrix-Interpolated Dropout Layer (MID-L), a novel module that dynamically selects and activates only the most informative neurons by interpolating between two transformation paths via a learned, input-dependent gating vector. Unlike conventional dropout or static sparsity methods, MID-L employs a differentiable Top-k masking strategy, enabling per-input adaptive computation while maintaining end-to-end differentiability. MID-L is model-agnostic and integrates seamlessly into existing architectures. Extensive experiments on six benchmarks, including MNIST, CIFAR-10, CIFAR-100, SVHN, UCI Adult, and IMDB, show that MID-L achieves up to average 55\% reduction in active neurons, 1.7$\times$ FLOPs savings, and maintains or exceeds baseline accuracy. We further validate the informativeness and selectivity of the learned neurons via Sliced Mutual Information (SMI) and observe improved robustness under overfitting and noisy data conditions. Additionally, MID-L demonstrates favorable inference latency and memory usage profiles, making it suitable for both research exploration and deployment on compute-constrained systems. These results position MID-L as a general-purpose, plug-and-play dynamic computation layer, bridging the gap between dropout regularization and efficient inference. 

---
# Visual Planning: Let's Think Only with Images 

**Authors**: Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, Ivan Vulić  

**Link**: [PDF](https://arxiv.org/pdf/2505.11409)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations, independent of text. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising alternative to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference. 

---
# Large Language Model Use Impact Locus of Control 

**Authors**: Jenny Xiyu Fu, Brennan Antone, Kowe Kadoma, Malte Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.11406)  

**Abstract**: As AI tools increasingly shape how we write, they may also quietly reshape how we perceive ourselves. This paper explores the psychological impact of co-writing with AI on people's locus of control. Through an empirical study with 462 participants, we found that employment status plays a critical role in shaping users' reliance on AI and their locus of control. Current results demonstrated that employed participants displayed higher reliance on AI and a shift toward internal control, while unemployed users tended to experience a reduction in personal agency. Through quantitative results and qualitative observations, this study opens a broader conversation about AI's role in shaping personal agency and identity. 

---
# Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner 

**Authors**: Wenchuan Zhang, Penghao Zhang, Jingru Guo, Tao Cheng, Jie Chen, Shuwan Zhang, Zhang Zhang, Yuhao Yi, Hong Bu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11404)  

**Abstract**: Recent advances in vision language models (VLMs) have enabled broad progress in the general medical field. However, pathology still remains a more challenging subdomain, with current pathology specific VLMs exhibiting limitations in both diagnostic accuracy and reasoning plausibility. Such shortcomings are largely attributable to the nature of current pathology datasets, which are primarily composed of image description pairs that lack the depth and structured diagnostic paradigms employed by real world pathologists. In this study, we leverage pathology textbooks and real world pathology experts to construct high-quality, reasoning-oriented datasets. Building on this, we introduce Patho-R1, a multimodal RL-based pathology Reasoner, trained through a three-stage pipeline: (1) continued pretraining on 3.5 million image-text pairs for knowledge infusion; (2) supervised fine-tuning on 500k high-quality Chain-of-Thought samples for reasoning incentivizing; (3) reinforcement learning using Group Relative Policy Optimization and Decoupled Clip and Dynamic sAmpling Policy Optimization strategies for multimodal reasoning quality refinement. To further assess the alignment quality of our dataset, we propose PathoCLIP, trained on the same figure-caption corpus used for continued pretraining. Comprehensive experimental results demonstrate that both PathoCLIP and Patho-R1 achieve robust performance across a wide range of pathology-related tasks, including zero-shot classification, cross-modal retrieval, Visual Question Answering, and Multiple Choice Question. Our project is available at the Patho-R1 repository: this https URL. 

---
# Phare: A Safety Probe for Large Language Models 

**Authors**: Pierre Le Jeune, Benoît Malésieux, Weixuan Xiao, Matteo Dora  

**Link**: [PDF](https://arxiv.org/pdf/2505.11365)  

**Abstract**: Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems. 

---
# DecompileBench: A Comprehensive Benchmark for Evaluating Decompilers in Real-World Scenarios 

**Authors**: Zeyu Gao, Yuxin Cui, Hao Wang, Siliang Qin, Yuanda Wang, Bolun Zhang, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11340)  

**Abstract**: Decompilers are fundamental tools for critical security tasks, from vulnerability discovery to malware analysis, yet their evaluation remains fragmented. Existing approaches primarily focus on syntactic correctness through synthetic micro-benchmarks or subjective human ratings, failing to address real-world requirements for semantic fidelity and analyst usability. We present DecompileBench, the first comprehensive framework that enables effective evaluation of decompilers in reverse engineering workflows through three key components: \textit{real-world function extraction} (comprising 23,400 functions from 130 real-world programs), \textit{runtime-aware validation}, and \textit{automated human-centric assessment} using LLM-as-Judge to quantify the effectiveness of decompilers in reverse engineering workflows. Through a systematic comparison between six industrial-strength decompilers and six recent LLM-powered approaches, we demonstrate that LLM-based methods surpass commercial tools in code understandability despite 52.2% lower functionality correctness. These findings highlight the potential of LLM-based approaches to transform human-centric reverse engineering. We open source \href{this https URL}{DecompileBench} to provide a framework to advance research on decompilers and assist security experts in making informed tool selections based on their specific requirements. 

---
# Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language Models 

**Authors**: Keunwoo Peter Yu, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11326)  

**Abstract**: Vision-language models (VLMs) have shown remarkable progress in offline tasks such as image captioning and video question answering. However, real-time interactive environments impose new demands on VLMs, requiring them to generate utterances that are not only semantically accurate but also precisely timed. We identify two core capabilities necessary for such settings -- $\textit{perceptual updating}$ and $\textit{contingency awareness}$ -- and propose a new benchmark task, $\textbf{Temporally-Grounded Language Generation (TGLG)}$, to evaluate them. TGLG requires models to generate utterances in response to streaming video such that both content and timing align with dynamic visual input. To support this benchmark, we curate evaluation datasets from sports broadcasting and egocentric human interaction domains, and introduce a new metric, $\textbf{TRACE}$, to evaluate TGLG by jointly measuring semantic similarity and temporal alignment. Finally, we present $\textbf{Vision-Language Model with Time-Synchronized Interleaving (VLM-TSI)}$, a model that interleaves visual and linguistic tokens in a time-synchronized manner, enabling real-time language generation without relying on turn-based assumptions. Experimental results show that VLM-TSI significantly outperforms a strong baseline, yet overall performance remains modest -- highlighting the difficulty of TGLG and motivating further research in real-time VLMs. Code and data available $\href{this https URL}{here}$. 

---
# Explaining Strategic Decisions in Multi-Agent Reinforcement Learning for Aerial Combat Tactics 

**Authors**: Ardian Selmonaj, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger, Matthias Sommer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11311)  

**Abstract**: Artificial intelligence (AI) is reshaping strategic planning, with Multi-Agent Reinforcement Learning (MARL) enabling coordination among autonomous agents in complex scenarios. However, its practical deployment in sensitive military contexts is constrained by the lack of explainability, which is an essential factor for trust, safety, and alignment with human strategies. This work reviews and assesses current advances in explainability methods for MARL with a focus on simulated air combat scenarios. We proceed by adapting various explainability techniques to different aerial combat scenarios to gain explanatory insights about the model behavior. By linking AI-generated tactics with human-understandable reasoning, we emphasize the need for transparency to ensure reliable deployment and meaningful human-machine interaction. By illuminating the crucial importance of explainability in advancing MARL for operational defense, our work supports not only strategic planning but also the training of military personnel with insightful and comprehensible analyses. 

---
# Heterogeneity-Aware Client Sampling: A Unified Solution for Consistent Federated Learning 

**Authors**: Shudi Weng, Chao Ren, Ming Xiao, Mikael Skoglund  

**Link**: [PDF](https://arxiv.org/pdf/2505.11304)  

**Abstract**: Federated learning (FL) commonly involves clients with diverse communication and computational capabilities. Such heterogeneity can significantly distort the optimization dynamics and lead to objective inconsistency, where the global model converges to an incorrect stationary point potentially far from the pursued optimum. Despite its critical impact, the joint effect of communication and computation heterogeneity has remained largely unexplored, due to the intrinsic complexity of their interaction. In this paper, we reveal the fundamentally distinct mechanisms through which heterogeneous communication and computation drive inconsistency in FL. To the best of our knowledge, this is the first unified theoretical analysis of general heterogeneous FL, offering a principled understanding of how these two forms of heterogeneity jointly distort the optimization trajectory under arbitrary choices of local solvers. Motivated by these insights, we propose Federated Heterogeneity-Aware Client Sampling, FedACS, a universal method to eliminate all types of objective inconsistency. We theoretically prove that FedACS converges to the correct optimum at a rate of $O(1/\sqrt{R})$, even in dynamic heterogeneous environments. Extensive experiments across multiple datasets show that FedACS outperforms state-of-the-art and category-specific baselines by 4.3%-36%, while reducing communication costs by 22%-89% and computation loads by 14%-105%, respectively. 

---
# Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs 

**Authors**: Yaorui Shi, Shihan Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11277)  

**Abstract**: Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively. 

---
# TCC-Bench: Benchmarking the Traditional Chinese Culture Understanding Capabilities of MLLMs 

**Authors**: Pengju Xu, Yan Wang, Shuyuan Zhang, Xuan Zhou, Xin Li, Yue Yuan, Fengzhao Li, Shunyuan Zhou, Xingyu Wang, Yi Zhang, Haiying Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11275)  

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) have significantly enhanced the ability of artificial intelligence systems to understand and generate multimodal content. However, these models often exhibit limited effectiveness when applied to non-Western cultural contexts, which raises concerns about their wider applicability. To address this limitation, we propose the \textbf{T}raditional \textbf{C}hinese \textbf{C}ulture understanding \textbf{Bench}mark (\textbf{TCC-Bench}), a bilingual (\textit{i.e.}, Chinese and English) Visual Question Answering (VQA) benchmark specifically designed for assessing the understanding of traditional Chinese culture by MLLMs. TCC-Bench comprises culturally rich and visually diverse data, incorporating images from museum artifacts, everyday life scenes, comics, and other culturally significant contexts. We adopt a semi-automated pipeline that utilizes GPT-4o in text-only mode to generate candidate questions, followed by human curation to ensure data quality and avoid potential data leakage. The benchmark also avoids language bias by preventing direct disclosure of cultural concepts within question texts. Experimental evaluations across a wide range of MLLMs demonstrate that current models still face significant challenges when reasoning about culturally grounded visual content. The results highlight the need for further research in developing culturally inclusive and context-aware multimodal systems. The code and data can be found at: this https URL. 

---
# Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models 

**Authors**: Camille Couturier, Spyros Mastorakis, Haiying Shen, Saravan Rajmohan, Victor Rühle  

**Link**: [PDF](https://arxiv.org/pdf/2505.11271)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across edge and cloud platforms for real-time question-answering and retrieval-augmented generation. However, processing lengthy contexts in distributed systems incurs high computational overhead, memory usage, and network bandwidth. This paper introduces a novel semantic caching approach for storing and reusing intermediate contextual summaries, enabling efficient information reuse across similar queries in LLM-based QA workflows. Our method reduces redundant computations by up to 50-60% while maintaining answer accuracy comparable to full document processing, as demonstrated on NaturalQuestions, TriviaQA, and a synthetic ArXiv dataset. This approach balances computational cost and response quality, critical for real-time AI assistants. 

---
# TAIJI: MCP-based Multi-Modal Data Analytics on Data Lakes 

**Authors**: Chao Zhang, Shaolei Zhang, Quehuan Liu, Sibei Chen, Tong Li, Ju Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11270)  

**Abstract**: The variety of data in data lakes presents significant challenges for data analytics, as data scientists must simultaneously analyze multi-modal data, including structured, semi-structured, and unstructured data. While Large Language Models (LLMs) have demonstrated promising capabilities, they still remain inadequate for multi-modal data analytics in terms of accuracy, efficiency, and freshness. First, current natural language (NL) or SQL-like query languages may struggle to precisely and comprehensively capture users' analytical intent. Second, relying on a single unified LLM to process diverse data modalities often leads to substantial inference overhead. Third, data stored in data lakes may be incomplete or outdated, making it essential to integrate external open-domain knowledge to generate timely and relevant analytics results.
In this paper, we envision a new multi-modal data analytics system. Specifically, we propose a novel architecture built upon the Model Context Protocol (MCP), an emerging paradigm that enables LLMs to collaborate with knowledgeable agents. First, we define a semantic operator hierarchy tailored for querying multi-modal data in data lakes and develop an AI-agent-powered NL2Operator translator to bridge user intent and analytical execution. Next, we introduce an MCP-based execution framework, in which each MCP server hosts specialized foundation models optimized for specific data modalities. This design enhances both accuracy and efficiency, while supporting high scalability through modular deployment. Finally, we propose a updating mechanism by harnessing the deep research and machine unlearning techniques to refresh the data lakes and LLM knowledges, with the goal of balancing the data freshness and inference efficiency. 

---
# Equal is Not Always Fair: A New Perspective on Hyperspectral Representation Non-Uniformity 

**Authors**: Wuzhou Quan, Mingqiang Wei, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11267)  

**Abstract**: Hyperspectral image (HSI) representation is fundamentally challenged by pervasive non-uniformity, where spectral dependencies, spatial continuity, and feature efficiency exhibit complex and often conflicting behaviors. Most existing models rely on a unified processing paradigm that assumes homogeneity across dimensions, leading to suboptimal performance and biased representations. To address this, we propose FairHyp, a fairness-directed framework that explicitly disentangles and resolves the threefold non-uniformity through cooperative yet specialized modules. We introduce a Runge-Kutta-inspired spatial variability adapter to restore spatial coherence under resolution discrepancies, a multi-receptive field convolution module with sparse-aware refinement to enhance discriminative features while respecting inherent sparsity, and a spectral-context state space model that captures stable and long-range spectral dependencies via bidirectional Mamba scanning and statistical aggregation. Unlike one-size-fits-all solutions, FairHyp achieves dimension-specific adaptation while preserving global consistency and mutual reinforcement. This design is grounded in the view that non-uniformity arises from the intrinsic structure of HSI representations, rather than any particular task setting. To validate this, we apply FairHyp across four representative tasks including classification, denoising, super-resolution, and inpaintin, demonstrating its effectiveness in modeling a shared structural flaw. Extensive experiments show that FairHyp consistently outperforms state-of-the-art methods under varied imaging conditions. Our findings redefine fairness as a structural necessity in HSI modeling and offer a new paradigm for balancing adaptability, efficiency, and fidelity in high-dimensional vision tasks. 

---
# A Set-Sequence Model for Time Series 

**Authors**: Elliot L. Epstein, Apaar Sadhwani, Kay Giesecke  

**Link**: [PDF](https://arxiv.org/pdf/2505.11243)  

**Abstract**: In many financial prediction problems, the behavior of individual units (such as loans, bonds, or stocks) is influenced by observable unit-level factors and macroeconomic variables, as well as by latent cross-sectional effects. Traditional approaches attempt to capture these latent effects via handcrafted summary features. We propose a Set-Sequence model that eliminates the need for handcrafted features. The Set model first learns a shared cross-sectional summary at each period. The Sequence model then ingests the summary-augmented time series for each unit independently to predict its outcome. Both components are learned jointly over arbitrary sets sampled during training. Our approach harnesses the set nature of the cross-section and is computationally efficient, generating set summaries in linear time relative to the number of units. It is also flexible, allowing the use of existing sequence models and accommodating a variable number of units at inference. Empirical evaluations demonstrate that our Set-Sequence model significantly outperforms benchmarks on stock return prediction and mortgage behavior tasks. Code will be released. 

---
# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization 

**Authors**: Yanhao Jia, Ji Xie, S Jivaganesh, Hao Li, Xu Wu, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11217)  

**Abstract**: Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we finetune a state-of-the-art model using a stereo audio-image dataset generated via 3D simulations. Even with limited training data, the refined model surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy. 

---
# Bayesian Hierarchical Invariant Prediction 

**Authors**: Francisco Madaleno, Pernille Julie Viuff Sand, Francisco C. Pereira, Sergio Hernan Garrido Mejia  

**Link**: [PDF](https://arxiv.org/pdf/2505.11211)  

**Abstract**: We propose Bayesian Hierarchical Invariant Prediction (BHIP) reframing Invariant Causal Prediction (ICP) through the lens of Hierarchical Bayes. We leverage the hierarchical structure to explicitly test invariance of causal mechanisms under heterogeneous data, resulting in improved computational scalability for a larger number of predictors compared to ICP. Moreover, given its Bayesian nature BHIP enables the use of prior information. In this paper, we test two sparsity inducing priors: horseshoe and spike-and-slab, both of which allow us a more reliable identification of causal features. We test BHIP in synthetic and real-world data showing its potential as an alternative inference method to ICP. 

---
# RanDeS: Randomized Delta Superposition for Multi-Model Compression 

**Authors**: Hangyu Zhou, Aaron Gokaslan, Volodymyr Kuleshov, Bharath Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11204)  

**Abstract**: From a multi-model compression perspective, model merging enables memory-efficient serving of multiple models fine-tuned from the same base, but suffers from degraded performance due to interference among their task-specific parameter adjustments (i.e., deltas). In this paper, we reformulate model merging as a compress-and-retrieve scheme, revealing that the task interference arises from the summation of irrelevant deltas during model retrieval. To address this issue, we use random orthogonal transformations to decorrelate these vectors into self-cancellation. We show that this approach drastically reduces interference, improving performance across both vision and language tasks. Since these transformations are fully defined by random seeds, adding new models requires no extra memory. Further, their data- and model-agnostic nature enables easy addition or removal of models with minimal compute overhead, supporting efficient and flexible multi-model serving. 

---
# Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese 

**Authors**: Xihuai Wang, Ziyi Zhao, Siyu Ren, Shao Zhang, Song Li, Xiaoyu Li, Ziwen Wang, Lin Qiu, Guanglu Wan, Xuezhi Cao, Xunliang Cai, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11200)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved text-to-speech (TTS) systems, enhancing control over speech style, naturalness, and emotional expression, which brings TTS Systems closer to human-level performance. Although the Mean Opinion Score (MOS) remains the standard for TTS System evaluation, it suffers from subjectivity, environmental inconsistencies, and limited interpretability. Existing evaluation datasets also lack a multi-dimensional design, often neglecting factors such as speaking styles, context diversity, and trap utterances, which is particularly evident in Chinese TTS evaluation. To address these challenges, we introduce the Audio Turing Test (ATT), a multi-dimensional Chinese corpus dataset ATT-Corpus paired with a simple, Turing-Test-inspired evaluation protocol. Instead of relying on complex MOS scales or direct model comparisons, ATT asks evaluators to judge whether a voice sounds human. This simplification reduces rating bias and improves evaluation robustness. To further support rapid model development, we also finetune Qwen2-Audio-Instruct with human judgment data as Auto-ATT for automatic evaluation. Experimental results show that ATT effectively differentiates models across specific capability dimensions using its multi-dimensional design. Auto-ATT also demonstrates strong alignment with human evaluations, confirming its value as a fast and reliable assessment tool. The white-box ATT-Corpus and Auto-ATT can be found in ATT Hugging Face Collection (this https URL). 

---
# User-centric Music Recommendations 

**Authors**: Jaime Ramirez Castillo, M. Julia Flores, Ann E. Nicholson  

**Link**: [PDF](https://arxiv.org/pdf/2505.11198)  

**Abstract**: This work presents a user-centric recommendation framework, designed as a pipeline with four distinct, connected, and customizable phases. These phases are intended to improve explainability and boost user engagement.
We have collected the historical this http URL track playback records of a single user over approximately 15 years. The collected dataset includes more than 90,000 playbacks and approximately 14,000 unique tracks.
From track playback records, we have created a dataset of user temporal contexts (each row is a specific moment when the user listened to certain music descriptors). As music descriptors, we have used community-contributed this http URL tags and Spotify audio features. They represent the music that, throughout years, the user has been listening to.
Next, given the most relevant this http URL tags of a moment (e.g. the hour of the day), we predict the Spotify audio features that best fit the user preferences in that particular moment. Finally, we use the predicted audio features to find tracks similar to these features. The final aim is to recommend (and discover) tracks that the user may feel like listening to at a particular moment.
For our initial study case, we have chosen to predict only a single audio feature target: danceability. The framework, however, allows to include more target variables.
The ability to learn the musical habits from a single user can be quite powerful, and this framework could be extended to other users. 

---
# FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Pretraining 

**Authors**: Myunsoo Kim, Seong-Woong Shim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.11192)  

**Abstract**: False negatives pose a critical challenge in vision-language pretraining (VLP) due to the many-to-many correspondence between images and texts in large-scale datasets. These false negatives introduce conflicting supervision signals that degrade the learned embedding space and diminish the effectiveness of hard negative sampling. In this paper, we propose FALCON (False-negative Aware Learning of COntrastive Negatives), a learning-based mini-batch construction strategy that adaptively balances the trade-off between hard and false negatives during VLP. Rather than relying on fixed heuristics, FALCON employs a negative mining scheduler that dynamically selects negative samples of appropriate hardness for each anchor instance during mini-batch construction, guided by a proxy for cross-modal alignment improvement. Experimental results demonstrate that FALCON significantly improves performance across two widely adopted VLP frameworks (ALBEF, BLIP-2) and a broad range of downstream tasks and evaluation settings, underscoring its effectiveness and robustness in mitigating the impact of false negatives. 

---
# Imputation-free and Alignment-free: Incomplete Multi-view Clustering Driven by Consensus Semantic Learning 

**Authors**: Yuzhuo Dai, Jiaqi Jin, Zhibin Dong, Siwei Wang, Xinwang Liu, En Zhu, Xihong Yang, Xinbiao Gan, Yu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.11182)  

**Abstract**: In incomplete multi-view clustering (IMVC), missing data induce prototype shifts within views and semantic inconsistencies across views. A feasible solution is to explore cross-view consistency in paired complete observations, further imputing and aligning the similarity relationships inherently shared across views. Nevertheless, existing methods are constrained by two-tiered limitations: (1) Neither instance- nor cluster-level consistency learning construct a semantic space shared across views to learn consensus semantics. The former enforces cross-view instances alignment, and wrongly regards unpaired observations with semantic consistency as negative pairs; the latter focuses on cross-view cluster counterparts while coarsely handling fine-grained intra-cluster relationships within views. (2) Excessive reliance on consistency results in unreliable imputation and alignment without incorporating view-specific cluster information. Thus, we propose an IMVC framework, imputation- and alignment-free for consensus semantics learning (FreeCSL). To bridge semantic gaps across all observations, we learn consensus prototypes from available data to discover a shared space, where semantically similar observations are pulled closer for consensus semantics learning. To capture semantic relationships within specific views, we design a heuristic graph clustering based on modularity to recover cluster structure with intra-cluster compactness and inter-cluster separation for cluster semantics enhancement. Extensive experiments demonstrate, compared to state-of-the-art competitors, FreeCSL achieves more confident and robust assignments on IMVC task. 

---
# CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback 

**Authors**: Yixin Wan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11178)  

**Abstract**: State-of-the-art T2I models are capable of generating high-resolution images given textual prompts. However, they still struggle with accurately depicting compositional scenes that specify multiple objects, attributes, and spatial relations. We present CompAlign, a challenging benchmark with an emphasis on assessing the depiction of 3D-spatial relationships, for evaluating and improving models on compositional image generation. CompAlign consists of 900 complex multi-subject image generation prompts that combine numerical and 3D-spatial relationships with varied attribute bindings. Our benchmark is remarkably challenging, incorporating generation tasks with 3+ generation subjects with complex 3D-spatial relationships. Additionally, we propose CompQuest, an interpretable and accurate evaluation framework that decomposes complex prompts into atomic sub-questions, then utilizes a MLLM to provide fine-grained binary feedback on the correctness of each aspect of generation elements in model-generated images. This enables precise quantification of alignment between generated images and compositional prompts. Furthermore, we propose an alignment framework that uses CompQuest's feedback as preference signals to improve diffusion models' compositional image generation abilities. Using adjustable per-image preferences, our method is easily scalable and flexible for different tasks. Evaluation of 9 T2I models reveals that: (1) models remarkable struggle more with compositional tasks with more complex 3D-spatial configurations, and (2) a noticeable performance gap exists between open-source accessible models and closed-source commercial models. Further empirical study on using CompAlign for model alignment yield promising results: post-alignment diffusion models achieve remarkable improvements in compositional accuracy, especially on complex generation tasks, outperforming previous approaches. 

---
# Low-Resource Language Processing: An OCR-Driven Summarization and Translation Pipeline 

**Authors**: Hrishit Madhavi, Jacob Cherian, Yuvraj Khamkar, Dhananjay Bhagat  

**Link**: [PDF](https://arxiv.org/pdf/2505.11177)  

**Abstract**: This paper presents an end-to-end suite for multilingual information extraction and processing from image-based documents. The system uses Optical Character Recognition (Tesseract) to extract text in languages such as English, Hindi, and Tamil, and then a pipeline involving large language model APIs (Gemini) for cross-lingual translation, abstractive summarization, and re-translation into a target language. Additional modules add sentiment analysis (TensorFlow), topic classification (Transformers), and date extraction (Regex) for better document comprehension. Made available in an accessible Gradio interface, the current research shows a real-world application of libraries, models, and APIs to close the language gap and enhance access to information in image media across different linguistic environments 

---
# From Intent Discovery to Recognition with Topic Modeling and Synthetic Data 

**Authors**: Aaron Rodrigues, Mahmood Hegazy, Azzam Naeem  

**Link**: [PDF](https://arxiv.org/pdf/2505.11176)  

**Abstract**: Understanding and recognizing customer intents in AI systems is crucial, particularly in domains characterized by short utterances and the cold start problem, where recommender systems must include new products or services without sufficient real user data. Customer utterances are characterized by infrequent word co-occurences and high term variability, which poses significant challenges for traditional methods in specifying distinct user needs and preparing synthetic queries. To address this, we propose an agentic LLM framework for topic modeling and synthetic query generation, which accelerates the discovery and recognition of customer intents. We first apply hierarchical topic modeling and intent discovery to expand a human-curated taxonomy from 36 generic user intents to 278 granular intents, demonstrating the potential of LLMs to significantly enhance topic specificity and diversity. Next, to support newly discovered intents and address the cold start problem, we generate synthetic user query data, which augments real utterances and reduces dependency on human annotation, especially in low-resource settings. Topic model experiments show substantial improvements in coherence and relevance after topic expansion, while synthetic data experiments indicate that in-class few-shot prompting significantly improves the quality and utility of synthetic queries without compromising diversity. We also show that LLM-generated intent descriptions and keywords can effectively substitute for human-curated versions when used as context for synthetic query generation. Our research underscores the scalability and utility of LLM agents in topic modeling and highlights the strategic use of synthetic utterances to enhance dataset variability and coverage for intent recognition. We present a comprehensive and robust framework for online discovery and recognition of new customer intents in dynamic domains. 

---
# Real-Time Verification of Embodied Reasoning for Generative Skill Acquisition 

**Authors**: Bo Yue, Shuqi Guo, Kaiyu Hu, Chujiao Wang, Benyou Wang, Kui Jia, Guiliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11175)  

**Abstract**: Generative skill acquisition enables embodied agents to actively learn a scalable and evolving repertoire of control skills, crucial for the advancement of large decision models. While prior approaches often rely on supervision signals from generalist agents (e.g., LLMs), their effectiveness in complex 3D environments remains unclear; exhaustive evaluation incurs substantial computational costs, significantly hindering the efficiency of skill learning. Inspired by recent successes in verification models for mathematical reasoning, we propose VERGSA (Verifying Embodied Reasoning in Generative Skill Acquisition), a framework that systematically integrates real-time verification principles into embodied skill learning. VERGSA establishes 1) a seamless extension from verification of mathematical reasoning into embodied learning by dynamically incorporating contextually relevant tasks into prompts and defining success metrics for both subtasks and overall tasks, and 2) an automated, scalable reward labeling scheme that synthesizes dense reward signals by iteratively finalizing the contribution of scene configuration and subtask learning to overall skill acquisition. To the best of our knowledge, this approach constitutes the first comprehensive training dataset for verification-driven generative skill acquisition, eliminating arduous manual reward engineering. Experiments validate the efficacy of our approach: 1) the exemplar task pool improves the average task success rates by 21%, 2) our verification model boosts success rates by 24% for novel tasks and 36% for encountered tasks, and 3) outperforms LLM-as-a-Judge baselines in verification quality. 

---
# CheX-DS: Improving Chest X-ray Image Classification with Ensemble Learning Based on DenseNet and Swin Transformer 

**Authors**: Xinran Li, Yu Liu, Xiujuan Xu, Xiaowei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11168)  

**Abstract**: The automatic diagnosis of chest diseases is a popular and challenging task. Most current methods are based on convolutional neural networks (CNNs), which focus on local features while neglecting global features. Recently, self-attention mechanisms have been introduced into the field of computer vision, demonstrating superior performance. Therefore, this paper proposes an effective model, CheX-DS, for classifying long-tail multi-label data in the medical field of chest X-rays. The model is based on the excellent CNN model DenseNet for medical imaging and the newly popular Swin Transformer model, utilizing ensemble deep learning techniques to combine the two models and leverage the advantages of both CNNs and Transformers. The loss function of CheX-DS combines weighted binary cross-entropy loss with asymmetric loss, effectively addressing the issue of data imbalance. The NIH ChestX-ray14 dataset is selected to evaluate the model's effectiveness. The model outperforms previous studies with an excellent average AUC score of 83.76\%, demonstrating its superior performance. 

---
# SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization 

**Authors**: Huashan Sun, Shengyi Liao, Yansen Han, Yu Bai, Yang Gao, Cheng Fu, Weizhou Shen, Fanqi Wan, Ming Yan, Ji Zhang, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11166)  

**Abstract**: Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency. 

---
# Maximizing Asynchronicity in Event-based Neural Networks 

**Authors**: Haiqing Hao, Nikola Zubić, Weihua He, Zhipeng Sui, Davide Scaramuzza, Wenhui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11165)  

**Abstract**: Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned representations for ML pipelines, existing A2S approaches often sacrifice representation expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous representation learning), a novel A2S framework to generate highly expressive and generalizable event-by-event representations. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a remarkable 47.7 mAP on the Gen1 dataset. These results underscore EVA's transformative potential for advancing real-time event-based vision applications. 

---
# Attention on the Sphere 

**Authors**: Boris Bonev, Max Rietmann, Andrea Paris, Alberto Carpentieri, Thorsten Kurth  

**Link**: [PDF](https://arxiv.org/pdf/2505.11157)  

**Abstract**: We introduce a generalized attention mechanism for spherical domains, enabling Transformer architectures to natively process data defined on the two-dimensional sphere - a critical need in fields such as atmospheric physics, cosmology, and robotics, where preserving spherical symmetries and topology is essential for physical accuracy. By integrating numerical quadrature weights into the attention mechanism, we obtain a geometrically faithful spherical attention that is approximately rotationally equivariant, providing strong inductive biases and leading to better performance than Cartesian approaches. To further enhance both scalability and model performance, we propose neighborhood attention on the sphere, which confines interactions to geodesic neighborhoods. This approach reduces computational complexity and introduces the additional inductive bias for locality, while retaining the symmetry properties of our method. We provide optimized CUDA kernels and memory-efficient implementations to ensure practical applicability. The method is validated on three diverse tasks: simulating shallow water equations on the rotating sphere, spherical image segmentation, and spherical depth estimation. Across all tasks, our spherical Transformers consistently outperform their planar counterparts, highlighting the advantage of geometric priors for learning on spherical domains. 

---
# X2C: A Dataset Featuring Nuanced Facial Expressions for Realistic Humanoid Imitation 

**Authors**: Peizhen Li, Longbing Cao, Xiao-Ming Wu, Runze Yang, Xiaohan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11146)  

**Abstract**: The ability to imitate realistic facial expressions is essential for humanoid robots engaged in affective human-robot communication. However, the lack of datasets containing diverse humanoid facial expressions with proper annotations hinders progress in realistic humanoid facial expression imitation. To address these challenges, we introduce X2C (Anything to Control), a dataset featuring nuanced facial expressions for realistic humanoid imitation. With X2C, we contribute: 1) a high-quality, high-diversity, large-scale dataset comprising 100,000 (image, control value) pairs. Each image depicts a humanoid robot displaying a diverse range of facial expressions, annotated with 30 control values representing the ground-truth expression configuration; 2) X2CNet, a novel human-to-humanoid facial expression imitation framework that learns the correspondence between nuanced humanoid expressions and their underlying control values from X2C. It enables facial expression imitation in the wild for different human performers, providing a baseline for the imitation task, showcasing the potential value of our dataset; 3) real-world demonstrations on a physical humanoid robot, highlighting its capability to advance realistic humanoid facial expression imitation. Code and Data: this https URL 

---
# Human-Aligned Bench: Fine-Grained Assessment of Reasoning Ability in MLLMs vs. Humans 

**Authors**: Yansheng Qiu, Li Xiao, Zhaopan Xu, Pengfei Zhou, Zheng Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11141)  

**Abstract**: The goal of achieving Artificial General Intelligence (AGI) is to imitate humans and surpass them. Models such as OpenAI's o1, o3, and DeepSeek's R1 have demonstrated that large language models (LLMs) with human-like reasoning capabilities exhibit exceptional performance and are being gradually integrated into multimodal large language models (MLLMs). However, whether these models possess capabilities comparable to humans in handling reasoning tasks remains unclear at present. In this paper, we propose Human-Aligned Bench, a benchmark for fine-grained alignment of multimodal reasoning with human performance. Specifically, we collected 9,794 multimodal questions that solely rely on contextual reasoning, including bilingual (Chinese and English) multimodal questions and pure text-based questions, encompassing four question types: visual reasoning, definition judgment, analogical reasoning, and logical judgment. More importantly, each question is accompanied by human success rates and options that humans are prone to choosing incorrectly. Extensive experiments on the Human-Aligned Bench reveal notable differences between the performance of current MLLMs in multimodal reasoning and human performance. The findings on our benchmark provide insights into the development of the next-generation models. 

---
# Scaling Reasoning can Improve Factuality in Large Language Models 

**Authors**: Mike Zhang, Johannes Bjerva, Russa Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11140)  

**Abstract**: Recent studies on large language model (LLM) reasoning capabilities have demonstrated promising improvements in model performance by leveraging a lengthy thinking process and additional computational resources during inference, primarily in tasks involving mathematical reasoning (Muennighoff et al., 2025). However, it remains uncertain if longer reasoning chains inherently enhance factual accuracy, particularly beyond mathematical contexts. In this work, we thoroughly examine LLM reasoning within complex open-domain question-answering (QA) scenarios. We initially distill reasoning traces from advanced, large-scale reasoning models (QwQ-32B and DeepSeek-R1-671B), then fine-tune a variety of models ranging from smaller, instruction-tuned variants to larger architectures based on Qwen2.5. To enrich reasoning traces, we introduce factual information from knowledge graphs in the form of paths into our reasoning traces. Our experimental setup includes four baseline approaches and six different instruction-tuned models evaluated across a benchmark of six datasets, encompassing over 22.6K questions. Overall, we carry out 168 experimental runs and analyze approximately 1.7 million reasoning traces. Our findings indicate that, within a single run, smaller reasoning models achieve noticeable improvements in factual accuracy compared to their original instruction-tuned counterparts. Moreover, our analysis demonstrates that adding test-time compute and token budgets factual accuracy consistently improves by 2-8%, further confirming the effectiveness of test-time scaling for enhancing performance and consequently improving reasoning accuracy in open-domain QA tasks. We release all the experimental artifacts for further research. 

---
# One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework 

**Authors**: Feiran Li, Qianqian Xu, Shilong Bao, Zhiyong Yang, Xiaochun Cao, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11131)  

**Abstract**: Concept erasing has recently emerged as an effective paradigm to prevent text-to-image diffusion models from generating visually undesirable or even harmful content. However, current removal methods heavily rely on manually crafted text prompts, making it challenging to achieve a high erasure (efficacy) while minimizing the impact on other benign concepts (usability). In this paper, we attribute the limitations to the inherent gap between the text and image modalities, which makes it hard to transfer the intricately entangled concept knowledge from text prompts to the image generation process. To address this, we propose a novel solution by directly integrating visual supervision into the erasure process, introducing the first text-image Collaborative Concept Erasing (Co-Erasing) framework. Specifically, Co-Erasing describes the concept jointly by text prompts and the corresponding undesirable images induced by the prompts, and then reduces the generating probability of the target concept through negative guidance. This approach effectively bypasses the knowledge gap between text and image, significantly enhancing erasure efficacy. Additionally, we design a text-guided image concept refinement strategy that directs the model to focus on visual features most relevant to the specified text concept, minimizing disruption to other benign concepts. Finally, comprehensive experiments suggest that Co-Erasing outperforms state-of-the-art erasure approaches significantly with a better trade-off between efficacy and usability. Codes are available at this https URL. 

---
# PhiNet v2: A Mask-Free Brain-Inspired Vision Foundation Model from Video 

**Authors**: Makoto Yamada, Kian Ming A. Chai, Ayoub Rhim, Satoki Ishikawa, Mohammad Sabokrou, Yao-Hung Hubert Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2505.11129)  

**Abstract**: Recent advances in self-supervised learning (SSL) have revolutionized computer vision through innovative architectures and learning objectives, yet they have not fully leveraged insights from biological visual processing systems. Recently, a brain-inspired SSL model named PhiNet was proposed; it is based on a ResNet backbone and operates on static image inputs with strong augmentation. In this paper, we introduce PhiNet v2, a novel Transformer-based architecture that processes temporal visual input (that is, sequences of images) without relying on strong augmentation. Our model leverages variational inference to learn robust visual representations from continuous input streams, similar to human visual processing. Through extensive experimentation, we demonstrate that PhiNet v2 achieves competitive performance compared to state-of-the-art vision foundation models, while maintaining the ability to learn from sequential input without strong data augmentation. This work represents a significant step toward more biologically plausible computer vision systems that process visual information in a manner more closely aligned with human cognitive processes. 

---
# Conditioning Matters: Training Diffusion Policies is Faster Than You Think 

**Authors**: Zibin Dong, Yicheng Liu, Yinchuan Li, Hang Zhao, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11123)  

**Abstract**: Diffusion policies have emerged as a mainstream paradigm for building vision-language-action (VLA) models. Although they demonstrate strong robot control capabilities, their training efficiency remains suboptimal. In this work, we identify a fundamental challenge in conditional diffusion policy training: when generative conditions are hard to distinguish, the training objective degenerates into modeling the marginal action distribution, a phenomenon we term loss collapse. To overcome this, we propose Cocos, a simple yet general solution that modifies the source distribution in the conditional flow matching to be condition-dependent. By anchoring the source distribution around semantics extracted from condition inputs, Cocos encourages stronger condition integration and prevents the loss collapse. We provide theoretical justification and extensive empirical results across simulation and real-world benchmarks. Our method achieves faster convergence and higher success rates than existing approaches, matching the performance of large-scale pre-trained VLAs using significantly fewer gradient steps and parameters. Cocos is lightweight, easy to implement, and compatible with diverse policy architectures, offering a general-purpose improvement to diffusion policy training. 

---
# FairSHAP: Preprocessing for Fairness Through Attribution-Based Data Augmentation 

**Authors**: Lin Zhu, Yijun Bian, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2505.11111)  

**Abstract**: Ensuring fairness in machine learning models is critical, particularly in high-stakes domains where biased decisions can lead to serious societal consequences. Existing preprocessing approaches generally lack transparent mechanisms for identifying which features or instances are responsible for unfairness. This obscures the rationale behind data modifications. We introduce FairSHAP, a novel pre-processing framework that leverages Shapley value attribution to improve both individual and group fairness. FairSHAP identifies fairness-critical instances in the training data using an interpretable measure of feature importance, and systematically modifies them through instance-level matching across sensitive groups. This process reduces discriminative risk - an individual fairness metric - while preserving data integrity and model accuracy. We demonstrate that FairSHAP significantly improves demographic parity and equality of opportunity across diverse tabular datasets, achieving fairness gains with minimal data perturbation and, in some cases, improved predictive performance. As a model-agnostic and transparent method, FairSHAP integrates seamlessly into existing machine learning pipelines and provides actionable insights into the sources of this http URL code is on this https URL. 

---
# MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark 

**Authors**: Florinel-Alin Croitoru, Vlad Hondru, Marius Popescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.11109)  

**Abstract**: We present the first large-scale open-set benchmark for multilingual audio-video deepfake detection. Our dataset comprises over 250 hours of real and fake videos across eight languages, with 60% of data being generated. For each language, the fake videos are generated with seven distinct deepfake generation models, selected based on the quality of the generated content. We organize the training, validation and test splits such that only a subset of the chosen generative models and languages are available during training, thus creating several challenging open-set evaluation setups. We perform experiments with various pre-trained and fine-tuned deepfake detectors proposed in recent literature. Our results show that state-of-the-art detectors are not currently able to maintain their performance levels when tested in our open-set scenarios. We publicly release our data and code at: this https URL. 

---
# PARSEC: Preference Adaptation for Robotic Object Rearrangement from Scene Context 

**Authors**: Kartik Ramachandruni, Sonia Chernova  

**Link**: [PDF](https://arxiv.org/pdf/2505.11108)  

**Abstract**: Object rearrangement is a key task for household robots requiring personalization without explicit instructions, meaningful object placement in environments occupied with objects, and generalization to unseen objects and new environments. To facilitate research addressing these challenges, we introduce PARSEC, an object rearrangement benchmark for learning user organizational preferences from observed scene context to place objects in a partially arranged environment. PARSEC is built upon a novel dataset of 110K rearrangement examples crowdsourced from 72 users, featuring 93 object categories and 15 environments. We also propose ContextSortLM, an LLM-based rearrangement model that places objects in partially arranged environments by adapting to user preferences from prior and current scene context while accounting for multiple valid placements. We evaluate ContextSortLM and existing personalized rearrangement approaches on the PARSEC benchmark and complement these findings with a crowdsourced evaluation of 108 online raters ranking model predictions based on alignment with user preferences. Our results indicate that personalized rearrangement models leveraging multiple scene context sources perform better than models relying on a single context source. Moreover, ContextSortLM outperforms other models in placing objects to replicate the target user's arrangement and ranks among the top two in all three environment categories, as rated by online evaluators. Importantly, our evaluation highlights challenges associated with modeling environment semantics across different environment categories and provides recommendations for future work. 

---
# Inferring the Most Similar Variable-length Subsequences between Multidimensional Time Series 

**Authors**: Thanadej Rattanakornphan, Piyanon Charoenpoonpanich, Chainarong Amornbunchornvej  

**Link**: [PDF](https://arxiv.org/pdf/2505.11106)  

**Abstract**: Finding the most similar subsequences between two multidimensional time series has many applications: e.g. capturing dependency in stock market or discovering coordinated movement of baboons. Considering one pattern occurring in one time series, we might be wondering whether the same pattern occurs in another time series with some distortion that might have a different length. Nevertheless, to the best of our knowledge, there is no efficient framework that deals with this problem yet. In this work, we propose an algorithm that provides the exact solution of finding the most similar multidimensional subsequences between time series where there is a difference in length both between time series and between subsequences. The algorithm is built based on theoretical guarantee of correctness and efficiency. The result in simulation datasets illustrated that our approach not just only provided correct solution, but it also utilized running time only quarter of time compared against the baseline approaches. In real-world datasets, it extracted the most similar subsequences even faster (up to 20 times faster against baseline methods) and provided insights regarding the situation in stock market and following relations of multidimensional time series of baboon movement. Our approach can be used for any time series. The code and datasets of this work are provided for the public use. 

---
# Bidirectional Distillation: A Mixed-Play Framework for Multi-Agent Generalizable Behaviors 

**Authors**: Lang Feng, Jiahao Lin, Dong Xing, Li Zhang, De Ma, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11100)  

**Abstract**: Population-population generalization is a challenging problem in multi-agent reinforcement learning (MARL), particularly when agents encounter unseen co-players. However, existing self-play-based methods are constrained by the limitation of inside-space generalization. In this study, we propose Bidirectional Distillation (BiDist), a novel mixed-play framework, to overcome this limitation in MARL. BiDist leverages knowledge distillation in two alternating directions: forward distillation, which emulates the historical policies' space and creates an implicit self-play, and reverse distillation, which systematically drives agents towards novel distributions outside the known policy space in a non-self-play manner. In addition, BiDist operates as a concise and efficient solution without the need for the complex and costly storage of past policies. We provide both theoretical analysis and empirical evidence to support BiDist's effectiveness. Our results highlight its remarkable generalization ability across a variety of cooperative, competitive, and social dilemma tasks, and reveal that BiDist significantly diversifies the policy distribution space. We also present comprehensive ablation studies to reinforce BiDist's effectiveness and key success factors. Source codes are available in the supplementary material. 

---
# A Fast Kernel-based Conditional Independence test with Application to Causal Discovery 

**Authors**: Oliver Schacht, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11085)  

**Abstract**: Kernel-based conditional independence (KCI) testing is a powerful nonparametric method commonly employed in causal discovery tasks. Despite its flexibility and statistical reliability, cubic computational complexity limits its application to large datasets. To address this computational bottleneck, we propose \textit{FastKCI}, a scalable and parallelizable kernel-based conditional independence test that utilizes a mixture-of-experts approach inspired by embarrassingly parallel inference techniques for Gaussian processes. By partitioning the dataset based on a Gaussian mixture model over the conditioning variables, FastKCI conducts local KCI tests in parallel, aggregating the results using an importance-weighted sampling scheme. Experiments on synthetic datasets and benchmarks on real-world production data validate that FastKCI maintains the statistical power of the original KCI test while achieving substantial computational speedups. FastKCI thus represents a practical and efficient solution for conditional independence testing in causal inference on large-scale data. 

---
# Fault Diagnosis across Heterogeneous Domains via Self-Adaptive Temporal-Spatial Attention and Sample Generation 

**Authors**: Guangqiang Li, M. Amine Atoui, Xiangshun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11083)  

**Abstract**: Deep learning methods have shown promising performance in fault diagnosis for multimode process. Most existing studies assume that the collected health state categories from different operating modes are identical. However, in real industrial scenarios, these categories typically exhibit only partial overlap. The incompleteness of the available data and the large distributional differences between the operating modes pose a significant challenge to existing fault diagnosis methods. To address this problem, a novel fault diagnosis model named self-adaptive temporal-spatial attention network (TSA-SAN) is proposed. First, inter-mode mappings are constructed using healthy category data to generate multimode samples. To enrich the diversity of the fault data, interpolation is performed between healthy and fault samples. Subsequently, the fault diagnosis model is trained using real and generated data. The self-adaptive instance normalization is established to suppress irrelevant information while retaining essential statistical features for diagnosis. In addition, a temporal-spatial attention mechanism is constructed to focus on the key features, thus enhancing the generalization ability of the model. The extensive experiments demonstrate that the proposed model significantly outperforms the state-of-the-art methods. The code will be available on Github at this https URL. 

---
# BLEUBERI: BLEU is a surprisingly effective reward for instruction following 

**Authors**: Yapei Chang, Yekyung Kim, Michael Krumdick, Amir Zadeh, Chuan Li, Chris Tanner, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11080)  

**Abstract**: Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at this https URL. 

---
# Towards Self-Improvement of Diffusion Models via Group Preference Optimization 

**Authors**: Renjie Chen, Wenfeng Lin, Yichen Zhang, Jiangchuan Wei, Boyuan Liu, Chao Feng, Jiao Ran, Mingyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11070)  

**Abstract**: Aligning text-to-image (T2I) diffusion models with Direct Preference Optimization (DPO) has shown notable improvements in generation quality. However, applying DPO to T2I faces two challenges: the sensitivity of DPO to preference pairs and the labor-intensive process of collecting and annotating high-quality data. In this work, we demonstrate that preference pairs with marginal differences can degrade DPO performance. Since DPO relies exclusively on relative ranking while disregarding the absolute difference of pairs, it may misclassify losing samples as wins, or vice versa. We empirically show that extending the DPO from pairwise to groupwise and incorporating reward standardization for reweighting leads to performance gains without explicit data selection. Furthermore, we propose Group Preference Optimization (GPO), an effective self-improvement method that enhances performance by leveraging the model's own capabilities without requiring external data. Extensive experiments demonstrate that GPO is effective across various diffusion models and tasks. Specifically, combining with widely used computer vision models, such as YOLO and OCR, the GPO improves the accurate counting and text rendering capabilities of the Stable Diffusion 3.5 Medium by 20 percentage points. Notably, as a plug-and-play method, no extra overhead is introduced during inference. 

---
# Assessing the Performance of Analog Training for Transfer Learning 

**Authors**: Omobayode Fagbohungbe, Corey Lammie, Malte J. Rasch, Takashi Ando, Tayfun Gokmen, Vijay Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11067)  

**Abstract**: Analog in-memory computing is a next-generation computing paradigm that promises fast, parallel, and energy-efficient deep learning training and transfer learning (TL). However, achieving this promise has remained elusive due to a lack of suitable training algorithms. Analog memory devices exhibit asymmetric and non-linear switching behavior in addition to device-to-device variation, meaning that most, if not all, of the current off-the-shelf training algorithms cannot achieve good training outcomes. Also, recently introduced algorithms have enjoyed limited attention, as they require bi-directionally switching devices of unrealistically high symmetry and precision and are highly sensitive. A new algorithm chopped TTv2 (c-TTv2), has been introduced, which leverages the chopped technique to address many of the challenges mentioned above. In this paper, we assess the performance of the c-TTv2 algorithm for analog TL using a Swin-ViT model on a subset of the CIFAR100 dataset. We also investigate the robustness of our algorithm to changes in some device specifications, including weight transfer noise, symmetry point skew, and symmetry point variability 

---
# Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking 

**Authors**: Changlun Li, Yao Shi, Chen Wang, Qiqi Duan, Runke Ruan, Weijie Huang, Haonan Long, Lijun Huang, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11065)  

**Abstract**: Large Language Models (LLMs) have demonstrated notable capabilities across financial tasks, including financial report summarization, earnings call transcript analysis, and asset classification. However, their real-world effectiveness in managing complex fund investment remains inadequately assessed. A fundamental limitation of existing benchmarks for evaluating LLM-driven trading strategies is their reliance on historical back-testing, inadvertently enabling LLMs to "time travel"-leveraging future information embedded in their training corpora, thus resulting in possible information leakage and overly optimistic performance estimates. To address this issue, we introduce DeepFund, a live fund benchmark tool designed to rigorously evaluate LLM in real-time market conditions. Utilizing a multi-agent architecture, DeepFund connects directly with real-time stock market data-specifically data published after each model pretraining cutoff-to ensure fair and leakage-free evaluations. Empirical tests on nine flagship LLMs from leading global institutions across multiple investment dimensions-including ticker-level analysis, investment decision-making, portfolio management, and risk control-reveal significant practical challenges. Notably, even cutting-edge models such as DeepSeek-V3 and Claude-3.7-Sonnet incur net trading losses within DeepFund real-time evaluation environment, underscoring the present limitations of LLMs for active fund management. Our code is available at this https URL. 

---
# CUBIC: Concept Embeddings for Unsupervised Bias Identification using VLMs 

**Authors**: David Méndez, Gianpaolo Bontempo, Elisa Ficarra, Roberto Confalonieri, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2505.11060)  

**Abstract**: Deep vision models often rely on biases learned from spurious correlations in datasets. To identify these biases, methods that interpret high-level, human-understandable concepts are more effective than those relying primarily on low-level features like heatmaps. A major challenge for these concept-based methods is the lack of image annotations indicating potentially bias-inducing concepts, since creating such annotations requires detailed labeling for each dataset and concept, which is highly labor-intensive. We present CUBIC (Concept embeddings for Unsupervised Bias IdentifiCation), a novel method that automatically discovers interpretable concepts that may bias classifier behavior. Unlike existing approaches, CUBIC does not rely on predefined bias candidates or examples of model failures tied to specific biases, as such information is not always available. Instead, it leverages image-text latent space and linear classifier probes to examine how the latent representation of a superclass label$\unicode{x2014}$shared by all instances in the dataset$\unicode{x2014}$is influenced by the presence of a given concept. By measuring these shifts against the normal vector to the classifier's decision boundary, CUBIC identifies concepts that significantly influence model predictions. Our experiments demonstrate that CUBIC effectively uncovers previously unknown biases using Vision-Language Models (VLMs) without requiring the samples in the dataset where the classifier underperforms or prior knowledge of potential biases. 

---
# Halting Recurrent GNNs and the Graded $μ$-Calculus 

**Authors**: Jeroen Bollen, Jan Van den Bussche, Stijn Vansummeren, Jonni Virtema  

**Link**: [PDF](https://arxiv.org/pdf/2505.11050)  

**Abstract**: Graph Neural Networks (GNNs) are a class of machine-learning models that operate on graph-structured data. Their expressive power is intimately related to logics that are invariant under graded bisimilarity. Current proposals for recurrent GNNs either assume that the graph size is given to the model, or suffer from a lack of termination guarantees. In this paper, we propose a halting mechanism for recurrent GNNs. We prove that our halting model can express all node classifiers definable in graded modal mu-calculus, even for the standard GNN variant that is oblivious to the graph size. A recent breakthrough in the study of the expressivity of graded modal mu-calculus in the finite suggests that conversely, restricted to node classifiers definable in monadic second-order logic, recurrent GNNs can express only node classifiers definable in graded modal mu-calculus. To prove our main result, we develop a new approximate semantics for graded mu-calculus, which we believe to be of independent interest. We leverage this new semantics into a new model-checking algorithm, called the counting algorithm, which is oblivious to the graph size. In a final step we show that the counting algorithm can be implemented on a halting recurrent GNN. 

---
# CleanPatrick: A Benchmark for Image Data Cleaning 

**Authors**: Fabian Gröger, Simone Lionetti, Philippe Gottfrois, Alvaro Gonzalez-Jimenez, Ludovic Amruthalingam, Elisabeth Victoria Goessinger, Hanna Lindemann, Marie Bargiela, Marie Hofbauer, Omar Badri, Philipp Tschandl, Arash Koochek, Matthew Groh, Alexander A. Navarini, Marc Pouly  

**Link**: [PDF](https://arxiv.org/pdf/2505.11034)  

**Abstract**: Robust machine learning depends on clean data, yet current image data cleaning benchmarks rely on synthetic noise or narrow human studies, limiting comparison and real-world relevance. We introduce CleanPatrick, the first large-scale benchmark for data cleaning in the image domain, built upon the publicly available Fitzpatrick17k dermatology dataset. We collect 496,377 binary annotations from 933 medical crowd workers, identify off-topic samples (4%), near-duplicates (21%), and label errors (22%), and employ an aggregation model inspired by item-response theory followed by expert review to derive high-quality ground truth. CleanPatrick formalizes issue detection as a ranking task and adopts typical ranking metrics mirroring real audit workflows. Benchmarking classical anomaly detectors, perceptual hashing, SSIM, Confident Learning, NoiseRank, and SelfClean, we find that, on CleanPatrick, self-supervised representations excel at near-duplicate detection, classical methods achieve competitive off-topic detection under constrained review budgets, and label-error detection remains an open challenge for fine-grained medical classification. By releasing both the dataset and the evaluation framework, CleanPatrick enables a systematic comparison of image-cleaning strategies and paves the way for more reliable data-centric artificial intelligence. 

---
# DexGarmentLab: Dexterous Garment Manipulation Environment with Generalizable Policy 

**Authors**: Yuran Wang, Ruihai Wu, Yue Chen, Jiarui Wang, Jiaqi Liang, Ziyu Zhu, Haoran Geng, Jitendra Malik, Pieter Abbeel, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.11032)  

**Abstract**: Garment manipulation is a critical challenge due to the diversity in garment categories, geometries, and deformations. Despite this, humans can effortlessly handle garments, thanks to the dexterity of our hands. However, existing research in the field has struggled to replicate this level of dexterity, primarily hindered by the lack of realistic simulations of dexterous garment manipulation. Therefore, we propose DexGarmentLab, the first environment specifically designed for dexterous (especially bimanual) garment manipulation, which features large-scale high-quality 3D assets for 15 task scenarios, and refines simulation techniques tailored for garment modeling to reduce the sim-to-real gap. Previous data collection typically relies on teleoperation or training expert reinforcement learning (RL) policies, which are labor-intensive and inefficient. In this paper, we leverage garment structural correspondence to automatically generate a dataset with diverse trajectories using only a single expert demonstration, significantly reducing manual intervention. However, even extensive demonstrations cannot cover the infinite states of garments, which necessitates the exploration of new algorithms. To improve generalization across diverse garment shapes and deformations, we propose a Hierarchical gArment-manipuLation pOlicy (HALO). It first identifies transferable affordance points to accurately locate the manipulation area, then generates generalizable trajectories to complete the task. Through extensive experiments and detailed analysis of our method and baseline, we demonstrate that HALO consistently outperforms existing methods, successfully generalizing to previously unseen instances even with significant variations in shape and deformation where others fail. Our project page is available at: this https URL. 

---
# The heteronomy of algorithms: Traditional knowledge and computational knowledge 

**Authors**: David M. Berry  

**Link**: [PDF](https://arxiv.org/pdf/2505.11030)  

**Abstract**: If an active citizen should increasingly be a computationally enlightened one, replacing the autonomy of reason with the heteronomy of algorithms, then I argue in this article that we must begin teaching the principles of critiquing the computal through new notions of what we might call digital Bildung. Indeed, if civil society itself is mediated by computational systems and media, the public use of reason must also be complemented by skills for negotiating and using these computal forms to articulate such critique. Not only is there a need to raise the intellectual tone regarding computation and its related softwarization processes, but there is an urgent need to attend to the likely epistemic challenges from computation which, as presently constituted, tends towards justification through a philosophy of utility rather than through a philosophy of care for the territory of the intellect. We therefore need to develop an approach to this field that uses concepts and methods drawn from philosophy, politics, history, anthropology, sociology, media studies, computer science, and the humanities more generally, to try to understand these issues - particularly the way in which software and data increasingly penetrate our everyday life and the pressures and fissures that are created. We must, in other words, move to undertake a critical interdisciplinary research program to understand the way in which these systems are created, instantiated, and normatively engendered in both specific and general contexts. 

---
# StRuCom: A Novel Dataset of Structured Code Comments in Russian 

**Authors**: Maria Dziuba, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11026)  

**Abstract**: Structured code comments in docstring format are essential for code comprehension and maintenance, but existing machine learning models for their generation perform poorly for Russian compared to English. To bridge this gap, we present StRuCom - the first large-scale dataset (153K examples) specifically designed for Russian code documentation. Unlike machine-translated English datasets that distort terminology (e.g., technical loanwords vs. literal translations) and docstring structures, StRuCom combines human-written comments from Russian GitHub repositories with synthetically generated ones, ensuring compliance with Python, Java, JavaScript, C#, and Go standards through automated validation. Fine-tuning Qwen2.5-Coder models (0.5B-7B) on StRuCom shows statistically significant improvements of chrf++ and BERTScore over baseline models. 

---
# Humans expect rationality and cooperation from LLM opponents in strategic games 

**Authors**: Darija Barak, Miguel Costa-Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2505.11011)  

**Abstract**: As Large Language Models (LLMs) integrate into our social and economic interactions, we need to deepen our understanding of how humans respond to LLMs opponents in strategic settings. We present the results of the first controlled monetarily-incentivised laboratory experiment looking at differences in human behaviour in a multi-player p-beauty contest against other humans and LLMs. We use a within-subject design in order to compare behaviour at the individual level. We show that, in this environment, human subjects choose significantly lower numbers when playing against LLMs than humans, which is mainly driven by the increased prevalence of `zero' Nash-equilibrium choices. This shift is mainly driven by subjects with high strategic reasoning ability. Subjects who play the zero Nash-equilibrium choice motivate their strategy by appealing to perceived LLM's reasoning ability and, unexpectedly, propensity towards cooperation. Our findings provide foundational insights into the multi-player human-LLM interaction in simultaneous choice games, uncover heterogeneities in both subjects' behaviour and beliefs about LLM's play when playing against them, and suggest important implications for mechanism design in mixed human-LLM systems. 

---
# Review-Instruct: A Review-Driven Multi-Turn Conversations Generation Method for Large Language Models 

**Authors**: Jiangxu Wu, Cong Wang, TianHuang Su, Jun Yang, Haozhi Lin, Chao Zhang, Ming Peng, Kai Shi, SongPan Yang, BinQing Pan, ZiXian Li, Ni Yang, ZhenYu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11010)  

**Abstract**: The effectiveness of large language models (LLMs) in conversational AI is hindered by their reliance on single-turn supervised fine-tuning (SFT) data, which limits contextual coherence in multi-turn dialogues. Existing methods for generating multi-turn dialogue data struggle to ensure both diversity and quality in instructions. To address this, we propose Review-Instruct, a novel framework that synthesizes multi-turn conversations through an iterative "Ask-Respond-Review" process involving three agent roles: a Candidate, multiple Reviewers, and a Chairman. The framework iteratively refines instructions by incorporating Reviewer feedback, enhancing dialogue diversity and difficulty. We construct a multi-turn dataset using the Alpaca dataset and fine-tune the LLaMA2-13B model. Evaluations on MT-Bench, MMLU-Pro, and Auto-Arena demonstrate significant improvements, achieving absolute gains of 2.9\% on MMLU-Pro and 2\% on MT-Bench compared to prior state-of-the-art models based on LLaMA2-13B. Ablation studies confirm the critical role of the Review stage and the use of multiple Reviewers in boosting instruction diversity and difficulty. Our work highlights the potential of review-driven, multi-agent frameworks for generating high-quality conversational data at scale. 

---
# Illusion or Algorithm? Investigating Memorization, Emergence, and Symbolic Processing in In-Context Learning 

**Authors**: Jingcheng Niu, Subhabrata Dutta, Ahmed Elshabrawy, Harish Tayyar Madabushi, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2505.11004)  

**Abstract**: Large-scale Transformer language models (LMs) trained solely on next-token prediction with web-scale data can solve a wide range of tasks after seeing just a few examples. The mechanism behind this capability, known as in-context learning (ICL), remains both controversial and poorly understood. Some studies argue that it is merely the result of memorizing vast amounts of data, while others contend that it reflects a fundamental, symbolic algorithmic development in LMs. In this work, we introduce a suite of investigative tasks and a novel method to systematically investigate ICL by leveraging the full Pythia scaling suite, including interim checkpoints that capture progressively larger amount of training data. By carefully exploring ICL performance on downstream tasks and simultaneously conducting a mechanistic analysis of the residual stream's subspace, we demonstrate that ICL extends beyond mere "memorization" of the training corpus, yet does not amount to the implementation of an independent symbolic algorithm. Our results also clarify several aspects of ICL, including the influence of training dynamics, model capabilities, and elements of mechanistic interpretability. Overall, our work advances the understanding of ICL and its implications, offering model developers insights into potential improvements and providing AI security practitioners with a basis for more informed guidelines. 

---
# Space Group Equivariant Crystal Diffusion 

**Authors**: Rees Chang, Angela Pak, Alex Guerra, Ni Zhan, Nick Richardson, Elif Ertekin, Ryan P. Adams  

**Link**: [PDF](https://arxiv.org/pdf/2505.10994)  

**Abstract**: Accelerating inverse design of crystalline materials with generative models has significant implications for a range of technologies. Unlike other atomic systems, 3D crystals are invariant to discrete groups of isometries called the space groups. Crucially, these space group symmetries are known to heavily influence materials properties. We propose SGEquiDiff, a crystal generative model which naturally handles space group constraints with space group invariant likelihoods. SGEquiDiff consists of an SE(3)-invariant, telescoping discrete sampler of crystal lattices; permutation-invariant, transformer-based autoregressive sampling of Wyckoff positions, elements, and numbers of symmetrically unique atoms; and space group equivariant diffusion of atomic coordinates. We show that space group equivariant vector fields automatically live in the tangent spaces of the Wyckoff positions. SGEquiDiff achieves state-of-the-art performance on standard benchmark datasets as assessed by quantitative proxy metrics and quantum mechanical calculations. 

---
# GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models 

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Han Liu, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10983)  

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features. 

---
# Group-in-Group Policy Optimization for LLM Agent Training 

**Authors**: Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.10978)  

**Abstract**: Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to long-horizon LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on two challenging agent benchmarks, ALFWorld and WebShop, using Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals and achieves performance gains of > 12\% on ALFWorld and > 9\% on WebShop over the GRPO baseline: all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost. 

---
# Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio 

**Authors**: Xinlu He, Jacob Whitehill  

**Link**: [PDF](https://arxiv.org/pdf/2505.10975)  

**Abstract**: Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR. 

---
# GROQLoco: Generalist and RObot-agnostic Quadruped Locomotion Control using Offline Datasets 

**Authors**: Narayanan PP, Sarvesh Prasanth Venkatesan, Srinivas Kantha Reddy, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.10973)  

**Abstract**: Recent advancements in large-scale offline training have demonstrated the potential of generalist policy learning for complex robotic tasks. However, applying these principles to legged locomotion remains a challenge due to continuous dynamics and the need for real-time adaptation across diverse terrains and robot morphologies. In this work, we propose GROQLoco, a scalable, attention-based framework that learns a single generalist locomotion policy across multiple quadruped robots and terrains, relying solely on offline datasets. Our approach leverages expert demonstrations from two distinct locomotion behaviors - stair traversal (non-periodic gaits) and flat terrain traversal (periodic gaits) - collected across multiple quadruped robots, to train a generalist model that enables behavior fusion for both behaviors. Crucially, our framework operates directly on proprioceptive data from all robots without incorporating any robot-specific encodings. The policy is directly deployable on an Intel i7 nuc, producing low-latency control outputs without any test-time optimization. Our extensive experiments demonstrate strong zero-shot transfer across highly diverse quadruped robots and terrains, including hardware deployment on the Unitree Go1, a commercially available 12kg robot. Notably, we evaluate challenging cross-robot training setups where different locomotion skills are unevenly distributed across robots, yet observe successful transfer of both flat walking and stair traversal behaviors to all robots at test time. We also show preliminary walking on Stoch 5, a 70kg quadruped, on flat and outdoor terrains without requiring any fine tuning. These results highlight the potential for robust generalist locomotion across diverse robots and terrains. 

---
# Let the Trial Begin: A Mock-Court Approach to Vulnerability Detection using LLM-Based Agents 

**Authors**: Ratnadira Widyasari, Martin Weyssow, Ivana Clairine Irsan, Han Wei Ang, Frank Liauw, Eng Lieh Ouh, Lwin Khin Shar, Hong Jin Kang, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10961)  

**Abstract**: Detecting vulnerabilities in source code remains a critical yet challenging task, especially when benign and vulnerable functions share significant similarities. In this work, we introduce VulTrial, a courtroom-inspired multi-agent framework designed to enhance automated vulnerability detection. It employs four role-specific agents, which are security researcher, code author, moderator, and review board. Through extensive experiments using GPT-3.5 and GPT-4o we demonstrate that Vultrial outperforms single-agent and multi-agent baselines. Using GPT-4o, VulTrial improves the performance by 102.39% and 84.17% over its respective baseline. Additionally, we show that role-specific instruction tuning in multi-agent with small data (50 pair samples) improves the performance of VulTrial further by 139.89% and 118.30%. Furthermore, we analyze the impact of increasing the number of agent interactions on VulTrial's overall performance. While multi-agent setups inherently incur higher costs due to increased token usage, our findings reveal that applying VulTrial to a cost-effective model like GPT-3.5 can improve its performance by 69.89% compared to GPT-4o in a single-agent setting, at a lower overall cost. 

---
# Relational Graph Transformer 

**Authors**: Vijay Prakash Dwivedi, Sri Jaladi, Yangyi Shen, Federico López, Charilaos I. Kanatsoulis, Rishi Puri, Matthias Fey, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2505.10960)  

**Abstract**: Relational Deep Learning (RDL) is a promising approach for building state-of-the-art predictive models on multi-table relational data by representing it as a heterogeneous temporal graph. However, commonly used Graph Neural Network models suffer from fundamental limitations in capturing complex structural patterns and long-range dependencies that are inherent in relational data. While Graph Transformers have emerged as powerful alternatives to GNNs on general graphs, applying them to relational entity graphs presents unique challenges: (i) Traditional positional encodings fail to generalize to massive, heterogeneous graphs; (ii) existing architectures cannot model the temporal dynamics and schema constraints of relational data; (iii) existing tokenization schemes lose critical structural information. Here we introduce the Relational Graph Transformer (RelGT), the first graph transformer architecture designed specifically for relational tables. RelGT employs a novel multi-element tokenization strategy that decomposes each node into five components (features, type, hop distance, time, and local structure), enabling efficient encoding of heterogeneity, temporality, and topology without expensive precomputation. Our architecture combines local attention over sampled subgraphs with global attention to learnable centroids, incorporating both local and database-wide representations. Across 21 tasks from the RelBench benchmark, RelGT consistently matches or outperforms GNN baselines by up to 18%, establishing Graph Transformers as a powerful architecture for Relational Deep Learning. 

---
# Constrained Preferential Bayesian Optimization and Its Application in Banner Ad Design 

**Authors**: Koki Iwai, Yusuke Kumagae, Yuki Koyama, Masahiro Hamasaki, Masataka Goto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10954)  

**Abstract**: Preferential Bayesian optimization (PBO) is a variant of Bayesian optimization that observes relative preferences (e.g., pairwise comparisons) instead of direct objective values, making it especially suitable for human-in-the-loop scenarios. However, real-world optimization tasks often involve inequality constraints, which existing PBO methods have not yet addressed. To fill this gap, we propose constrained preferential Bayesian optimization (CPBO), an extension of PBO that incorporates inequality constraints for the first time. Specifically, we present a novel acquisition function for this purpose. Our technical evaluation shows that our CPBO method successfully identifies optimal solutions by focusing on exploring feasible regions. As a practical application, we also present a designer-in-the-loop system for banner ad design using CPBO, where the objective is the designer's subjective preference, and the constraint ensures a target predicted click-through rate. We conducted a user study with professional ad designers, demonstrating the potential benefits of our approach in guiding creative design under real-world constraints. 

---
# Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-lingual Transfer 

**Authors**: Seungyoon Lee, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.10945)  

**Abstract**: Large Language Models (LLMs) increasingly incorporate multilingual capabilities, fueling the demand to transfer them into target language-specific models. However, most approaches, which blend the source model's embedding by replacing the source vocabulary with the target language-specific vocabulary, may constrain expressive capacity in the target language since the source model is predominantly trained on English data. In this paper, we propose Semantic Aware Linear Transfer (SALT), a novel cross-lingual transfer technique that recycles embeddings from target language Pre-trained Language Models (PLMs) to transmit the deep representational strengths of PLM-derived embedding to LLMs. SALT derives unique regression lines based on the similarity in the overlap of the source and target vocabularies, to handle each non-overlapping token's embedding space. Our extensive experiments show that SALT significantly outperforms other transfer methods and achieves lower loss with accelerating faster convergence during language adaptation. Notably, SALT obtains remarkable performance in cross-lingual understanding setups compared to other methods. Furthermore, we highlight the scalable use of PLMs to enhance the functionality of contemporary LLMs by conducting experiments with varying architectures. 

---
# Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation 

**Authors**: Qing Yu, Xiaobei Wang, Shuchang Liu, Yandong Bai, Xiaoyu Yang, Xueliang Wang, Chang Meng, Shanshan Wu, Hailan Yang, Huihui Xiao, Xiang Li, Fan Yang, Xiaoqiang Feng, Lantao Hu, Han Li, Kun Gai, Lixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10940)  

**Abstract**: Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, the exploitation of the LLM's world knowledge and logic inference ability produces a virtual logic graph that reveals dynamic and expressive knowledge of users, augmenting the recommendation performance. On the other hand, the user role aligns the user behavioral logic with the observed user feedback, refining our understanding of user behaviors. Additionally, we also show that the extracted user-item logic graph is empirically a general knowledge that can benefit a wide range of recommendation tasks, and conduct experiments on industrial and several public datasets as verification. 

---
# GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction 

**Authors**: Mohammadtaha Bagherifard, Sahar Rajabi, Ali Edalat, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.10939)  

**Abstract**: Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at this https URL. 

---
# Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations 

**Authors**: Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10937)  

**Abstract**: The emergence of large reasoning models (LRMs) has transformed Natural Language Processing by excelling in complex tasks such as mathematical problem-solving and code generation. These models leverage chain-of-thought (CoT) processes, enabling them to emulate human-like reasoning strategies. However, the advancement of LRMs is hindered by the lack of comprehensive CoT datasets. Current resources often fail to provide extensive reasoning problems with coherent CoT processes distilled from multiple teacher models and do not account for multifaceted properties describing the internal characteristics of CoTs. To address these challenges, we introduce OmniThought, a large-scale dataset featuring 2 million CoT processes generated and validated by two powerful LRMs as teacher models. Each CoT process in OmniThought is annotated with novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which describe the appropriateness of CoT verbosity and cognitive difficulty level for models to comprehend these reasoning processes. We further establish a self-reliant pipeline to curate this dataset. Extensive experiments using Qwen2.5 models of various sizes demonstrate the positive impact of our proposed scores on LRM training effectiveness. Based on the proposed OmniThought dataset, we further train and release a series of high-performing LRMs, specifically equipped with stronger reasoning abilities and optimal CoT output length and difficulty level. Our contributions significantly enhance the development and training of LRMs for solving complex tasks. 

---
# A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron? 

**Authors**: Ada Chen, Yongjiang Wu, Junyuan Zhang, Shu Yang, Jen-tse Huang, Kun Wang, Wenxuan Wang, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10924)  

**Abstract**: Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents. 

---
# Vaiage: A Multi-Agent Solution to Personalized Travel Planning 

**Authors**: Binwen Liu, Jiexi Ge, Jiamin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10922)  

**Abstract**: Planning trips is a cognitively intensive task involving conflicting user preferences, dynamic external information, and multi-step temporal-spatial optimization. Traditional platforms often fall short - they provide static results, lack contextual adaptation, and fail to support real-time interaction or intent refinement.
Our approach, Vaiage, addresses these challenges through a graph-structured multi-agent framework built around large language models (LLMs) that serve as both goal-conditioned recommenders and sequential planners. LLMs infer user intent, suggest personalized destinations and activities, and synthesize itineraries that align with contextual constraints such as budget, timing, group size, and weather. Through natural language interaction, structured tool use, and map-based feedback loops, Vaiage enables adaptive, explainable, and end-to-end travel planning grounded in both symbolic reasoning and conversational understanding.
To evaluate Vaiage, we conducted human-in-the-loop experiments using rubric-based GPT-4 assessments and qualitative feedback. The full system achieved an average score of 8.5 out of 10, outperforming the no-strategy (7.2) and no-external-API (6.8) variants, particularly in feasibility. Qualitative analysis indicated that agent coordination - especially the Strategy and Information Agents - significantly improved itinerary quality by optimizing time use and integrating real-time context. These results demonstrate the effectiveness of combining LLM reasoning with symbolic agent coordination in open-ended, real-world planning tasks. 

---
# Phi: Leveraging Pattern-based Hierarchical Sparsity for High-Efficiency Spiking Neural Networks 

**Authors**: Chiyue Wei, Bowen Duan, Cong Guo, Jingyang Zhang, Qingyue Song, Hai "Helen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.10909)  

**Abstract**: Spiking Neural Networks (SNNs) are gaining attention for their energy efficiency and biological plausibility, utilizing 0-1 activation sparsity through spike-driven computation. While existing SNN accelerators exploit this sparsity to skip zero computations, they often overlook the unique distribution patterns inherent in binary activations. In this work, we observe that particular patterns exist in spike activations, which we can utilize to reduce the substantial computation of SNN models. Based on these findings, we propose a novel \textbf{pattern-based hierarchical sparsity} framework, termed \textbf{\textit{Phi}}, to optimize computation.
\textit{Phi} introduces a two-level sparsity hierarchy: Level 1 exhibits vector-wise sparsity by representing activations with pre-defined patterns, allowing for offline pre-computation with weights and significantly reducing most runtime computation. Level 2 features element-wise sparsity by complementing the Level 1 matrix, using a highly sparse matrix to further reduce computation while maintaining accuracy. We present an algorithm-hardware co-design approach. Algorithmically, we employ a k-means-based pattern selection method to identify representative patterns and introduce a pattern-aware fine-tuning technique to enhance Level 2 sparsity. Architecturally, we design \textbf{\textit{Phi}}, a dedicated hardware architecture that efficiently processes the two levels of \textit{Phi} sparsity on the fly. Extensive experiments demonstrate that \textit{Phi} achieves a $3.45\times$ speedup and a $4.93\times$ improvement in energy efficiency compared to state-of-the-art SNN accelerators, showcasing the effectiveness of our framework in optimizing SNN computation. 

---
# On the Security Risks of ML-based Malware Detection Systems: A Survey 

**Authors**: Ping He, Yuhao Mao, Changjiang Li, Lorenzo Cavallaro, Ting Wang, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.10903)  

**Abstract**: Malware presents a persistent threat to user privacy and data integrity. To combat this, machine learning-based (ML-based) malware detection (MD) systems have been developed. However, these systems have increasingly been attacked in recent years, undermining their effectiveness in practice. While the security risks associated with ML-based MD systems have garnered considerable attention, the majority of prior works is limited to adversarial malware examples, lacking a comprehensive analysis of practical security risks. This paper addresses this gap by utilizing the CIA principles to define the scope of security risks. We then deconstruct ML-based MD systems into distinct operational stages, thus developing a stage-based taxonomy. Utilizing this taxonomy, we summarize the technical progress and discuss the gaps in the attack and defense proposals related to the ML-based MD systems within each stage. Subsequently, we conduct two case studies, using both inter-stage and intra-stage analyses according to the stage-based taxonomy to provide new empirical insights. Based on these analyses and insights, we suggest potential future directions from both inter-stage and intra-stage perspectives. 

---
# Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With LLM 

**Authors**: Wenqing Zheng, Noah Fatsi, Daniel Barcklow, Dmitri Kalaev, Steven Yao, Owen Reinert, C. Bayan Bruss, Daniele Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10900)  

**Abstract**: Interaction sparsity is the primary obstacle for recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It also is found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this sparsity issue shifts the performance bottleneck to other areas in the computational pipeline. Those that focus on enriching sparse representations with connectivity data from other external sources propose methods that are resource demanding and require careful domain expert aided addition of this newly introduced data. Others that turn to Large Language Model (LLM) based recommenders will quickly encounter limitations surrounding data quality and availability. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR learns latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets. 

---
# BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset 

**Authors**: Istiaq Ahmed Fahad, Kamruzzaman Asif, Sifat Sikder  

**Link**: [PDF](https://arxiv.org/pdf/2505.10885)  

**Abstract**: Deepfake audio detection is challenging for low-resource languages like Bengali due to limited datasets and subtle acoustic features. To address this, we introduce BangalFake, a Bengali Deepfake Audio Dataset with 12,260 real and 13,260 deepfake utterances. Synthetic speech is generated using SOTA Text-to-Speech (TTS) models, ensuring high naturalness and quality. We evaluate the dataset through both qualitative and quantitative analyses. Mean Opinion Score (MOS) from 30 native speakers shows Robust-MOS of 3.40 (naturalness) and 4.01 (intelligibility). t-SNE visualization of MFCCs highlights real vs. fake differentiation challenges. This dataset serves as a crucial resource for advancing deepfake detection in Bengali, addressing the limitations of low-resource language research. 

---
# Graph and Simplicial Complex Prediction Gaussian Process via the Hodgelet Representations 

**Authors**: Mathieu Alain, So Takao, Xiaowen Dong, Bastian Rieck, Emmanuel Noutahi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10877)  

**Abstract**: Predicting the labels of graph-structured data is crucial in scientific applications and is often achieved using graph neural networks (GNNs). However, when data is scarce, GNNs suffer from overfitting, leading to poor performance. Recently, Gaussian processes (GPs) with graph-level inputs have been proposed as an alternative. In this work, we extend the Gaussian process framework to simplicial complexes (SCs), enabling the handling of edge-level attributes and attributes supported on higher-order simplices. We further augment the resulting SC representations by considering their Hodge decompositions, allowing us to account for homological information, such as the number of holes, in the SC. We demonstrate that our framework enhances the predictions across various applications, paving the way for GPs to be more widely used for graph and SC-level predictions. 

---
# Preference Isolation Forest for Structure-based Anomaly Detection 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10876)  

**Abstract**: We address the problem of detecting anomalies as samples that do not conform to structured patterns represented by low-dimensional manifolds. To this end, we conceive a general anomaly detection framework called Preference Isolation Forest (PIF), that combines the benefits of adaptive isolation-based methods with the flexibility of preference embedding. The key intuition is to embed the data into a high-dimensional preference space by fitting low-dimensional manifolds, and to identify anomalies as isolated points. We propose three isolation approaches to identify anomalies: $i$) Voronoi-iForest, the most general solution, $ii$) RuzHash-iForest, that avoids explicit computation of distances via Local Sensitive Hashing, and $iii$) Sliding-PIF, that leverages a locality prior to improve efficiency and effectiveness. 

---
# MultiLink: Multi-class Structure Recovery via Agglomerative Clustering and Model Selection 

**Authors**: Luca Magri, Filippo Leveni, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10874)  

**Abstract**: We address the problem of recovering multiple structures of different classes in a dataset contaminated by noise and outliers. In particular, we consider geometric structures defined by a mixture of underlying parametric models (e.g. planes and cylinders, homographies and fundamental matrices), and we tackle the robust fitting problem by preference analysis and clustering. We present a new algorithm, termed MultiLink, that simultaneously deals with multiple classes of models. MultiLink combines on-the-fly model fitting and model selection in a novel linkage scheme that determines whether two clusters are to be merged. The resulting method features many practical advantages with respect to methods based on preference analysis, being faster, less sensitive to the inlier threshold, and able to compensate limitations deriving from hypotheses sampling. Experiments on several public datasets demonstrate that Multi-Link favourably compares with state of the art alternatives, both in multi-class and single-class problems. Code is publicly made available for download. 

---
# Hashing for Structure-based Anomaly Detection 

**Authors**: Filippo Leveni, Luca Magri, Cesare Alippi, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10873)  

**Abstract**: We focus on the problem of identifying samples in a set that do not conform to structured patterns represented by low-dimensional manifolds. An effective way to solve this problem is to embed data in a high dimensional space, called Preference Space, where anomalies can be identified as the most isolated points. In this work, we employ Locality Sensitive Hashing to avoid explicit computation of distances in high dimensions and thus improve Anomaly Detection efficiency. Specifically, we present an isolation-based anomaly detection technique designed to work in the Preference Space which achieves state-of-the-art performance at a lower computational cost. Code is publicly available at this https URL. 

---
# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning? 

**Authors**: Chenxi Jiang, Chuhao Zhou, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10872)  

**Abstract**: Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark with vague REs (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 77.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompt and chains of thought. This work contributes to the research community of human-robot interaction (HRI) by making robot task planning more practical, particularly for non-expert users, e.g., the elderly and children. 

---
# Optimal Allocation of Privacy Budget on Hierarchical Data Release 

**Authors**: Joonhyuk Ko, Juba Ziani, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2505.10871)  

**Abstract**: Releasing useful information from datasets with hierarchical structures while preserving individual privacy presents a significant challenge. Standard privacy-preserving mechanisms, and in particular Differential Privacy, often require careful allocation of a finite privacy budget across different levels and components of the hierarchy. Sub-optimal allocation can lead to either excessive noise, rendering the data useless, or to insufficient protections for sensitive information. This paper addresses the critical problem of optimal privacy budget allocation for hierarchical data release. It formulates this challenge as a constrained optimization problem, aiming to maximize data utility subject to a total privacy budget while considering the inherent trade-offs between data granularity and privacy loss. The proposed approach is supported by theoretical analysis and validated through comprehensive experiments on real hierarchical datasets. These experiments demonstrate that optimal privacy budget allocation significantly enhances the utility of the released data and improves the performance of downstream tasks. 

---
# Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate 

**Authors**: Ziyang Huang, Wangtao Sun, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10870)  

**Abstract**: This paper systematically addresses the challenges of rule retrieval, a crucial yet underexplored area. Vanilla retrieval methods using sparse or dense retrievers to directly search for relevant rules to support downstream reasoning, often suffer from low accuracy. This is primarily due to a significant semantic gap between the instantiated facts in the queries and the abstract representations of the rules. Such misalignment results in suboptimal retrieval quality, which in turn negatively impacts reasoning performance. To overcome these challenges, we propose Self-Induction Augmented Retrieval (SIAR), a novel approach that utilizes Large Language Models (LLMs) to induce potential inferential rules that might offer benefits for reasoning by abstracting the underlying knowledge and logical structure in queries. These induced rules are then used for query augmentation to improve retrieval effectiveness. Additionally, we introduce Rule Relevance ReEstimate (R$^3$), a method that re-estimates the relevance of retrieved rules by assessing whether the abstract knowledge they contain can be instantiated to align with the facts in the queries and the helpfulness for reasoning. Extensive experiments across various settings demonstrate the effectiveness and versatility of our proposed methods. 

---
# ImputeINR: Time Series Imputation via Implicit Neural Representations for Disease Diagnosis with Missing Data 

**Authors**: Mengxuan Li, Ke Liu, Jialong Guo, Jiajun Bu, Hongwei Wang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10856)  

**Abstract**: Healthcare data frequently contain a substantial proportion of missing values, necessitating effective time series imputation to support downstream disease diagnosis tasks. However, existing imputation methods focus on discrete data points and are unable to effectively model sparse data, resulting in particularly poor performance for imputing substantial missing values. In this paper, we propose a novel approach, ImputeINR, for time series imputation by employing implicit neural representations (INR) to learn continuous functions for time series. ImputeINR leverages the merits of INR in that the continuous functions are not coupled to sampling frequency and have infinite sampling frequency, allowing ImputeINR to generate fine-grained imputations even on extremely sparse observed values. Extensive experiments conducted on eight datasets with five ratios of masked values show the superior imputation performance of ImputeINR, especially for high missing ratios in time series data. Furthermore, we validate that applying ImputeINR to impute missing values in healthcare data enhances the performance of downstream disease diagnosis tasks. Codes are available. 

---
# Ready2Unlearn: A Learning-Time Approach for Preparing Models with Future Unlearning Readiness 

**Authors**: Hanyu Duan, Yi Yang, Ahmed Abbasi, Kar Yan Tam  

**Link**: [PDF](https://arxiv.org/pdf/2505.10845)  

**Abstract**: This paper introduces Ready2Unlearn, a learning-time optimization approach designed to facilitate future unlearning processes. Unlike the majority of existing unlearning efforts that focus on designing unlearning algorithms, which are typically implemented reactively when an unlearning request is made during the model deployment phase, Ready2Unlearn shifts the focus to the training phase, adopting a "forward-looking" perspective. Building upon well-established meta-learning principles, Ready2Unlearn proactively trains machine learning models with unlearning readiness, such that they are well prepared and can handle future unlearning requests in a more efficient and principled manner. Ready2Unlearn is model-agnostic and compatible with any gradient ascent-based machine unlearning algorithms. We evaluate the method on both vision and language tasks under various unlearning settings, including class-wise unlearning and random data unlearning. Experimental results show that by incorporating such preparedness at training time, Ready2Unlearn produces an unlearning-ready model state, which offers several key advantages when future unlearning is required, including reduced unlearning time, improved retention of overall model capability, and enhanced resistance to the inadvertent recovery of forgotten data. We hope this work could inspire future efforts to explore more proactive strategies for equipping machine learning models with built-in readiness towards more reliable and principled machine unlearning. 

---
# Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL 

**Authors**: Songjun Tu, Jiahao Lin, Qichao Zhang, Xiangyu Tian, Linjing Li, Xiangyuan Lan, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.10832)  

**Abstract**: Large reasoning models (LRMs) are proficient at generating explicit, step-by-step reasoning sequences before producing final answers. However, such detailed reasoning can introduce substantial computational overhead and latency, particularly for simple problems. To address this over-thinking problem, we explore how to equip LRMs with adaptive thinking capabilities: enabling them to dynamically decide whether or not to engage in explicit reasoning based on problem complexity. Building on R1-style distilled models, we observe that inserting a simple ellipsis ("...") into the prompt can stochastically trigger either a thinking or no-thinking mode, revealing a latent controllability in the reasoning behavior. Leveraging this property, we propose AutoThink, a multi-stage reinforcement learning (RL) framework that progressively optimizes reasoning policies via stage-wise reward shaping. AutoThink learns to invoke explicit reasoning only when necessary, while defaulting to succinct responses for simpler tasks. Experiments on five mainstream mathematical benchmarks demonstrate that AutoThink achieves favorable accuracy-efficiency trade-offs compared to recent prompting and RL-based pruning methods. It can be seamlessly integrated into any R1-style model, including both distilled and further fine-tuned variants. Notably, AutoThink improves relative accuracy by 6.4 percent while reducing token usage by 52 percent on DeepSeek-R1-Distill-Qwen-1.5B, establishing a scalable and adaptive reasoning paradigm for LRMs. 

---
# Creating General User Models from Computer Use 

**Authors**: Omar Shaikh, Shardul Sapkota, Shan Rizvi, Eric Horvitz, Joon Sung Park, Diyi Yang, Michael S. Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.10831)  

**Abstract**: Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture that user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs. 

---
# Enhancing Low-Resource Minority Language Translation with LLMs and Retrieval-Augmented Generation for Cultural Nuances 

**Authors**: Chen-Chi Chang, Chong-Fu Li, Chu-Hsuan Lee, Hung-Shin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10829)  

**Abstract**: This study investigates the challenges of translating low-resource languages by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). Various model configurations were tested on Hakka translations, with BLEU scores ranging from 12% (dictionary-only) to 31% (RAG with Gemini 2.0). The best-performing model (Model 4) combined retrieval and advanced language modeling, improving lexical coverage, particularly for specialized or culturally nuanced terms, and enhancing grammatical coherence. A two-stage method (Model 3) using dictionary outputs refined by Gemini 2.0 achieved a BLEU score of 26%, highlighting iterative correction's value and the challenges of domain-specific expressions. Static dictionary-based approaches struggled with context-sensitive content, demonstrating the limitations of relying solely on predefined resources. These results emphasize the need for curated resources, domain knowledge, and ethical collaboration with local communities, offering a framework that improves translation accuracy and fluency while supporting cultural preservation. 

---
# Attention-Based Reward Shaping for Sparse and Delayed Rewards 

**Authors**: Ian Holmes, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.10802)  

**Abstract**: Sparse and delayed reward functions pose a significant obstacle for real-world Reinforcement Learning (RL) applications. In this work, we propose Attention-based REward Shaping (ARES), a general and robust algorithm which uses a transformer's attention mechanism to generate shaped rewards and create a dense reward function for any environment. ARES requires a set of episodes and their final returns as input. It can be trained entirely offline and is able to generate meaningful shaped rewards even when using small datasets or episodes produced by agents taking random actions. ARES is compatible with any RL algorithm and can handle any level of reward sparsity. In our experiments, we focus on the most challenging case where rewards are fully delayed until the end of each episode. We evaluate ARES across a diverse range of environments, widely used RL algorithms, and baseline methods to assess the effectiveness of the shaped rewards it produces. Our results show that ARES can significantly improve learning in delayed reward settings, enabling RL agents to train in scenarios that would otherwise require impractical amounts of data or even be unlearnable. To our knowledge, ARES is the first approach that works fully offline, remains robust to extreme reward delays and low-quality data, and is not limited to goal-based tasks. 

---
# Analyzing Patterns and Influence of Advertising in Print Newspapers 

**Authors**: N Harsha Vardhan, Ponnurangam Kumaraguru, Kiran Garimella  

**Link**: [PDF](https://arxiv.org/pdf/2505.10791)  

**Abstract**: This paper investigates advertising practices in print newspapers across India using a novel data-driven approach. We develop a pipeline employing image processing and OCR techniques to extract articles and advertisements from digital versions of print newspapers with high accuracy. Applying this methodology to five popular newspapers that span multiple regions and three languages, English, Hindi, and Telugu, we assembled a dataset of more than 12,000 editions containing several hundred thousand advertisements. Collectively, these newspapers reach a readership of over 100 million people. Using this extensive dataset, we conduct a comprehensive analysis to answer key questions about print advertising: who advertises, what they advertise, when they advertise, where they place their ads, and how they advertise. Our findings reveal significant patterns, including the consistent level of print advertising over the past six years despite declining print circulation, the overrepresentation of company ads on prominent pages, and the disproportionate revenue contributed by government ads. Furthermore, we examine whether advertising in a newspaper influences the coverage an advertiser receives. Through regression analyses on coverage volume and sentiment, we find strong evidence supporting this hypothesis for corporate advertisers. The results indicate a clear trend where increased advertising correlates with more favorable and extensive media coverage, a relationship that remains robust over time and across different levels of advertiser popularity. 

---
# Neural-Inspired Advances in Integral Cryptanalysis 

**Authors**: Liu Zhang, Yiran Yao, Danping Shi, Dongchen Chai, Jian Guo, Zilong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10790)  

**Abstract**: The study by Gohr this http URL at CRYPTO 2019 and sunsequent related works have shown that neural networks can uncover previously unused features, offering novel insights into cryptanalysis. Motivated by these findings, we employ neural networks to learn features specifically related to integral properties and integrate the corresponding insights into optimized search frameworks. These findings validate the framework of using neural networks for feature exploration, providing researchers with novel insights that advance established cryptanalysis methods.
Neural networks have inspired the development of more precise integral search models. By comparing the integral distinguishers obtained via neural networks with those identified by classical methods, we observe that existing automated search models often fail to find optimal distinguishers. To address this issue, we develop a meet in the middle search framework that balances model accuracy and computational efficiency. As a result, we reduce the number of active plaintext bits required for an 11 rounds integral distinguisher on SKINNY64/64, and further identify a 12 rounds key dependent integral distinguisher achieving one additional round over the previous best-known result.
The integral distinguishers discovered by neural networks enable key recovery attacks on more rounds. We identify a 7 rounds key independent integral distinguisher from neural networks with even only one active plaintext cell, which is based on linear combinations of bits. This distinguisher enables a 15 rounds key recovery attack on SKINNYn/n, improving upon the previous record by one round. Additionally, we discover an 8 rounds key dependent integral distinguisher using neural network that further reduces the time complexity of key recovery attacks against SKINNY. 

---
# Completely Weakly Supervised Class-Incremental Learning for Semantic Segmentation 

**Authors**: David Minkwan Kim, Soeun Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10781)  

**Abstract**: This work addresses the task of completely weakly supervised class-incremental learning for semantic segmentation to learn segmentation for both base and additional novel classes using only image-level labels. While class-incremental semantic segmentation (CISS) is crucial for handling diverse and newly emerging objects in the real world, traditional CISS methods require expensive pixel-level annotations for training. To overcome this limitation, partially weakly-supervised approaches have recently been proposed. However, to the best of our knowledge, this is the first work to introduce a completely weakly-supervised method for CISS. To achieve this, we propose to generate robust pseudo-labels by combining pseudo-labels from a localizer and a sequence of foundation models based on their uncertainty. Moreover, to mitigate catastrophic forgetting, we introduce an exemplar-guided data augmentation method that generates diverse images containing both previous and novel classes with guidance. Finally, we conduct experiments in three common experimental settings: 15-5 VOC, 10-10 VOC, and COCO-to-VOC, and in two scenarios: disjoint and overlap. The experimental results demonstrate that our completely weakly supervised method outperforms even partially weakly supervised methods in the 15-5 VOC and 10-10 VOC settings while achieving competitive accuracy in the COCO-to-VOC setting. 

---
# A Systematic Analysis of Base Model Choice for Reward Modeling 

**Authors**: Kian Ahrabian, Pegah Jandaghi, Negar Mokhberian, Sai Praneeth Karimireddy, Jay Pujara  

**Link**: [PDF](https://arxiv.org/pdf/2505.10775)  

**Abstract**: Reinforcement learning from human feedback (RLHF) and, at its core, reward modeling have become a crucial part of training powerful large language models (LLMs). One commonly overlooked factor in training high-quality reward models (RMs) is the effect of the base model, which is becoming more challenging to choose given the rapidly growing pool of LLMs. In this work, we present a systematic analysis of the effect of base model selection on reward modeling performance. Our results show that the performance can be improved by up to 14% compared to the most common (i.e., default) choice. Moreover, we showcase the strong statistical relation between some existing benchmarks and downstream performances. We also demonstrate that the results from a small set of benchmarks could be combined to boost the model selection ($+$18% on average in the top 5-10). Lastly, we illustrate the impact of different post-training steps on the final performance and explore using estimated data distributions to reduce performance prediction error. 

---
# Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting 

**Authors**: Yueyang Yao, Jiajun Li, Xingyuan Dai, MengMeng Zhang, Xiaoyan Gong, Fei-Yue Wang, Yisheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.10774)  

**Abstract**: Time series forecasting is important for applications spanning energy markets, climate analysis, and traffic management. However, existing methods struggle to effectively integrate exogenous texts and align them with the probabilistic nature of large language models (LLMs). Current approaches either employ shallow text-time series fusion via basic prompts or rely on deterministic numerical decoding that conflict with LLMs' token-generation paradigm, which limits contextual awareness and distribution modeling. To address these limitations, we propose CAPTime, a context-aware probabilistic multimodal time series forecasting method that leverages text-informed abstraction and autoregressive LLM decoding. Our method first encodes temporal patterns using a pretrained time series encoder, then aligns them with textual contexts via learnable interactions to produce joint multimodal representations. By combining a mixture of distribution experts with frozen LLMs, we enable context-aware probabilistic forecasting while preserving LLMs' inherent distribution modeling capabilities. Experiments on diverse time series forecasting tasks demonstrate the superior accuracy and generalization of CAPTime, particularly in multimodal scenarios. Additional analysis highlights its robustness in data-scarce scenarios through hybrid probabilistic decoding. 

---
# Geofenced Unmanned Aerial Robotic Defender for Deer Detection and Deterrence (GUARD) 

**Authors**: Ebasa Temesgen, Mario Jerez, Greta Brown, Graham Wilson, Sree Ganesh Lalitaditya Divakarla, Sarah Boelter, Oscar Nelson, Robert McPherson, Maria Gini  

**Link**: [PDF](https://arxiv.org/pdf/2505.10770)  

**Abstract**: Wildlife-induced crop damage, particularly from deer, threatens agricultural productivity. Traditional deterrence methods often fall short in scalability, responsiveness, and adaptability to diverse farmland environments. This paper presents an integrated unmanned aerial vehicle (UAV) system designed for autonomous wildlife deterrence, developed as part of the Farm Robotics Challenge. Our system combines a YOLO-based real-time computer vision module for deer detection, an energy-efficient coverage path planning algorithm for efficient field monitoring, and an autonomous charging station for continuous operation of the UAV. In collaboration with a local Minnesota farmer, the system is tailored to address practical constraints such as terrain, infrastructure limitations, and animal behavior. The solution is evaluated through a combination of simulation and field testing, demonstrating robust detection accuracy, efficient coverage, and extended operational time. The results highlight the feasibility and effectiveness of drone-based wildlife deterrence in precision agriculture, offering a scalable framework for future deployment and extension. 

---
# ChestyBot: Detecting and Disrupting Chinese Communist Party Influence Stratagems 

**Authors**: Matthew Stoffolano, Ayush Rout, Justin M. Pelletier  

**Link**: [PDF](https://arxiv.org/pdf/2505.10746)  

**Abstract**: Foreign information operations conducted by Russian and Chinese actors exploit the United States' permissive information environment. These campaigns threaten democratic institutions and the broader Westphalian model. Yet, existing detection and mitigation strategies often fail to identify active information campaigns in real time. This paper introduces ChestyBot, a pragmatics-based language model that detects unlabeled foreign malign influence tweets with up to 98.34% accuracy. The model supports a novel framework to disrupt foreign influence operations in their formative stages. 

---
# Automating Security Audit Using Large Language Model based Agent: An Exploration Experiment 

**Authors**: Jia Hui Chin, Pu Zhang, Yu Xin Cheong, Jonathan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10732)  

**Abstract**: In the current rapidly changing digital environment, businesses are under constant stress to ensure that their systems are secured. Security audits help to maintain a strong security posture by ensuring that policies are in place, controls are implemented, gaps are identified for cybersecurity risks mitigation. However, audits are usually manual, requiring much time and costs. This paper looks at the possibility of developing a framework to leverage Large Language Models (LLMs) as an autonomous agent to execute part of the security audit, namely with the field audit. password policy compliance for Windows operating system. Through the conduct of an exploration experiment of using GPT-4 with Langchain, the agent executed the audit tasks by accurately flagging password policy violations and appeared to be more efficient than traditional manual audits. Despite its potential limitations in operational consistency in complex and dynamic environment, the framework suggests possibilities to extend further to real-time threat monitoring and compliance checks. 

---
# Learning Repetition-Invariant Representations for Polymer Informatics 

**Authors**: Yihan Zhu, Gang Liu, Eric Inae, Tengfei Luo, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10726)  

**Abstract**: Polymers are large macromolecules composed of repeating structural units known as monomers and are widely applied in fields such as energy storage, construction, medicine, and aerospace. However, existing graph neural network methods, though effective for small molecules, only model the single unit of polymers and fail to produce consistent vector representations for the true polymer structure with varying numbers of units. To address this challenge, we introduce Graph Repetition Invariance (GRIN), a novel method to learn polymer representations that are invariant to the number of repeating units in their graph representations. GRIN integrates a graph-based maximum spanning tree alignment with repeat-unit augmentation to ensure structural consistency. We provide theoretical guarantees for repetition-invariance from both model and data perspectives, demonstrating that three repeating units are the minimal augmentation required for optimal invariant representation learning. GRIN outperforms state-of-the-art baselines on both homopolymer and copolymer benchmarks, learning stable, repetition-invariant representations that generalize effectively to polymer chains of unseen sizes. 

---
# AI-enhanced semantic feature norms for 786 concepts 

**Authors**: Siddharth Suresh, Kushin Mukherjee, Tyler Giallanza, Xizheng Yu, Mia Patil, Jonathan D. Cohen, Timothy T. Rogers  

**Link**: [PDF](https://arxiv.org/pdf/2505.10718)  

**Abstract**: Semantic feature norms have been foundational in the study of human conceptual knowledge, yet traditional methods face trade-offs between concept/feature coverage and verifiability of quality due to the labor-intensive nature of norming studies. Here, we introduce a novel approach that augments a dataset of human-generated feature norms with responses from large language models (LLMs) while verifying the quality of norms against reliable human judgments. We find that our AI-enhanced feature norm dataset, NOVA: Norms Optimized Via AI, shows much higher feature density and overlap among concepts while outperforming a comparable human-only norm dataset and word-embedding models in predicting people's semantic similarity judgments. Taken together, we demonstrate that human conceptual knowledge is richer than captured in previous norm datasets and show that, with proper validation, LLMs can serve as powerful tools for cognitive science research. 

---
# A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment 

**Authors**: Jean-Philippe Corbeil, Amin Dada, Jean-Michel Attendu, Asma Ben Abacha, Alessandro Sordoni, Lucas Caccia, François Beaulieu, Thomas Lin, Jens Kleesiek, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2505.10717)  

**Abstract**: High computation costs and latency of large language models such as GPT-4 have limited their deployment in clinical settings. Small language models (SLMs) offer a cost-effective alternative, but their limited capacity requires biomedical domain adaptation, which remains challenging. An additional bottleneck is the unavailability and high sensitivity of clinical data. To address these challenges, we propose a novel framework for adapting SLMs into high-performing clinical models. We introduce the MediPhi collection of 3.8B-parameter SLMs developed with our novel framework: pre-instruction tuning of experts on relevant medical and clinical corpora (PMC, Medical Guideline, MedWiki, etc.), model merging, and clinical-tasks alignment. To cover most clinical tasks, we extended the CLUE benchmark to CLUE+, doubling its size. Our expert models deliver relative improvements on this benchmark over the base model without any task-specific fine-tuning: 64.3% on medical entities, 49.5% on radiology reports, and 44% on ICD-10 coding (outperforming GPT-4-0125 by 14%). We unify the expert models into MediPhi via model merging, preserving gains across benchmarks. Furthermore, we built the MediFlow collection, a synthetic dataset of 2.5 million high-quality instructions on 14 medical NLP tasks, 98 fine-grained document types, and JSON format support. Alignment of MediPhi using supervised fine-tuning and direct preference optimization achieves further gains of 18.9% on average. 

---
# GNN-Suite: a Graph Neural Network Benchmarking Framework for Biomedical Informatics 

**Authors**: Sebestyén Kamp, Giovanni Stracquadanio, T. Ian Simpson  

**Link**: [PDF](https://arxiv.org/pdf/2505.10711)  

**Abstract**: We present GNN-Suite, a robust modular framework for constructing and benchmarking Graph Neural Network (GNN) architectures in computational biology. GNN-Suite standardises experimentation and reproducibility using the Nextflow workflow to evaluate GNN performance. We demonstrate its utility in identifying cancer-driver genes by constructing molecular networks from protein-protein interaction (PPI) data from STRING and BioGRID and annotating nodes with features from the PCAWG, PID, and COSMIC-CGC repositories.
Our design enables fair comparisons among diverse GNN architectures including GAT, GAT3H, GCN, GCN2, GIN, GTN, HGCN, PHGCN, and GraphSAGE and a baseline Logistic Regression (LR) model. All GNNs were configured as standardised two-layer models and trained with uniform hyperparameters (dropout = 0.2; Adam optimiser with learning rate = 0.01; and an adjusted binary cross-entropy loss to address class imbalance) over an 80/20 train-test split for 300 epochs. Each model was evaluated over 10 independent runs with different random seeds to yield statistically robust performance metrics, with balanced accuracy (BACC) as the primary measure. Notably, GCN2 achieved the highest BACC (0.807 +/- 0.035) on a STRING-based network, although all GNN types outperformed the LR baseline, highlighting the advantage of network-based learning over feature-only approaches.
Our results show that a common framework for implementing and evaluating GNN architectures aids in identifying not only the best model but also the most effective means of incorporating complementary data. By making GNN-Suite publicly available, we aim to foster reproducible research and promote improved benchmarking standards in computational biology. Future work will explore additional omics datasets and further refine network architectures to enhance predictive accuracy and interpretability in biomedical applications. 

---
# Predicting Human Behavior in Autonomous Systems: A Collaborative Machine Teaching Approach for Reducing Transfer of Control Events 

**Authors**: Julian Wolter, Amr Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2505.10695)  

**Abstract**: As autonomous systems become integral to various industries, effective strategies for fault handling are essential to ensure reliability and efficiency. Transfer of Control (ToC), a traditional approach for interrupting automated processes during faults, is often triggered unnecessarily in non-critical situations. To address this, we propose a data-driven method that uses human interaction data to train AI models capable of preemptively identifying and addressing issues or assisting users in resolution. Using an interactive tool simulating an industrial vacuum cleaner, we collected data and developed an LSTM-based model to predict user behavior. Our findings reveal that even data from non-experts can effectively train models to reduce unnecessary ToC events, enhancing the system's robustness. This approach highlights the potential of AI to learn directly from human problem-solving behaviors, complementing sensor data to improve industrial automation and human-AI collaboration. 

---
# Predicting Risk of Pulmonary Fibrosis Formation in PASC Patients 

**Authors**: Wanying Dou, Gorkem Durak, Koushik Biswas, Ziliang Hong, Andrea Mia Bejar, Elif Keles, Kaan Akin, Sukru Mehmet Erturk, Alpay Medetalibeyoglu, Marc Sala, Alexander Misharin, Hatice Savas, Mary Salvatore, Sachin Jambawalikar, Drew Torigian, Jayaram K. Udupa, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2505.10691)  

**Abstract**: While the acute phase of the COVID-19 pandemic has subsided, its long-term effects persist through Post-Acute Sequelae of COVID-19 (PASC), commonly known as Long COVID. There remains substantial uncertainty regarding both its duration and optimal management strategies. PASC manifests as a diverse array of persistent or newly emerging symptoms--ranging from fatigue, dyspnea, and neurologic impairments (e.g., brain fog), to cardiovascular, pulmonary, and musculoskeletal abnormalities--that extend beyond the acute infection phase. This heterogeneous presentation poses substantial challenges for clinical assessment, diagnosis, and treatment planning. In this paper, we focus on imaging findings that may suggest fibrotic damage in the lungs, a critical manifestation characterized by scarring of lung tissue, which can potentially affect long-term respiratory function in patients with PASC. This study introduces a novel multi-center chest CT analysis framework that combines deep learning and radiomics for fibrosis prediction. Our approach leverages convolutional neural networks (CNNs) and interpretable feature extraction, achieving 82.2% accuracy and 85.5% AUC in classification tasks. We demonstrate the effectiveness of Grad-CAM visualization and radiomics-based feature analysis in providing clinically relevant insights for PASC-related lung fibrosis prediction. Our findings highlight the potential of deep learning-driven computational methods for early detection and risk assessment of PASC-related lung fibrosis--presented for the first time in the literature. 

---
# Towards an LLM-powered Social Digital Twinning Platform 

**Authors**: Önder Gürcan, Vanja Falck, Markus G. Rousseau, Larissa L. Lima  

**Link**: [PDF](https://arxiv.org/pdf/2505.10681)  

**Abstract**: We present Social Digital Twinner, an innovative social simulation tool for exploring plausible effects of what-if scenarios in complex adaptive social systems. The architecture is composed of three seamlessly integrated parts: a data infrastructure featuring real-world data and a multi-dimensionally representative synthetic population of citizens, an LLM-enabled agent-based simulation engine, and a user interface that enable intuitive, natural language interactions with the simulation engine and the artificial agents (i.e. citizens). Social Digital Twinner facilitates real-time engagement and empowers stakeholders to collaboratively design, test, and refine intervention measures. The approach is promoting a data-driven and evidence-based approach to societal problem-solving. We demonstrate the tool's interactive capabilities by addressing the critical issue of youth school dropouts in Kragero, Norway, showcasing its ability to create and execute a dedicated social digital twin using natural language. 

---
# A Conformal Predictive Measure for Assessing Catastrophic Forgetting 

**Authors**: Ioannis Pitsiorlas, Nour Jamoussi, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2505.10677)  

**Abstract**: This work introduces a novel methodology for assessing catastrophic forgetting (CF) in continual learning. We propose a new conformal prediction (CP)-based metric, termed the Conformal Prediction Confidence Factor (CPCF), to quantify and evaluate CF effectively. Our framework leverages adaptive CP to estimate forgetting by monitoring the model's confidence on previously learned tasks. This approach provides a dynamic and practical solution for monitoring and measuring CF of previous tasks as new ones are introduced, offering greater suitability for real-world applications. Experimental results on four benchmark datasets demonstrate a strong correlation between CPCF and the accuracy of previous tasks, validating the reliability and interpretability of the proposed metric. Our results highlight the potential of CPCF as a robust and effective tool for assessing and understanding CF in dynamic learning environments. 

---
# Seasonal Forecasting of Pan-Arctic Sea Ice with State Space Model 

**Authors**: Wei Wang, Weidong Yang, Lei Wang, Guihua Wang, Ruibo Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.10665)  

**Abstract**: The rapid decline of Arctic sea ice resulting from anthropogenic climate change poses significant risks to indigenous communities, ecosystems, and the global climate system. This situation emphasizes the immediate necessity for precise seasonal sea ice forecasts. While dynamical models perform well for short-term forecasts, they encounter limitations in long-term forecasts and are computationally intensive. Deep learning models, while more computationally efficient, often have difficulty managing seasonal variations and uncertainties when dealing with complex sea ice dynamics. In this research, we introduce IceMamba, a deep learning architecture that integrates sophisticated attention mechanisms within the state space model. Through comparative analysis of 25 renowned forecast models, including dynamical, statistical, and deep learning approaches, our experimental results indicate that IceMamba delivers excellent seasonal forecasting capabilities for Pan-Arctic sea ice concentration. Specifically, IceMamba outperforms all tested models regarding average RMSE and anomaly correlation coefficient (ACC) and ranks second in Integrated Ice Edge Error (IIEE). This innovative approach enhances our ability to foresee and alleviate the effects of sea ice variability, offering essential insights for strategies aimed at climate adaptation. 

---
# CLIP Embeddings for AI-Generated Image Detection: A Few-Shot Study with Lightweight Classifier 

**Authors**: Ziyang Ou  

**Link**: [PDF](https://arxiv.org/pdf/2505.10664)  

**Abstract**: Verifying the authenticity of AI-generated images presents a growing challenge on social media platforms these days. While vision-language models (VLMs) like CLIP outdo in multimodal representation, their capacity for AI-generated image classification is underexplored due to the absence of such labels during the pre-training process. This work investigates whether CLIP embeddings inherently contain information indicative of AI generation. A proposed pipeline extracts visual embeddings using a frozen CLIP model, feeds its embeddings to lightweight networks, and fine-tunes only the final classifier. Experiments on the public CIFAKE benchmark show the performance reaches 95% accuracy without language reasoning. Few-shot adaptation to curated custom with 20% of the data results in performance to 85%. A closed-source baseline (Gemini-2.0) has the best zero-shot accuracy yet fails on specific styles. Notably, some specific image types, such as wide-angle photographs and oil paintings, pose significant challenges to classification. These results indicate previously unexplored difficulties in classifying certain types of AI-generated images, revealing new and more specific questions in this domain that are worth further investigation. 

---
# Artificial Intelligence Bias on English Language Learners in Automatic Scoring 

**Authors**: Shuchen Guo, Yun Wang, Jichao Yu, Xuansheng Wu, Bilgehan Ayik, Field M. Watts, Ehsan Latif, Ninghao Liu, Lei Liu, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2505.10643)  

**Abstract**: This study investigated potential scoring biases and disparities toward English Language Learners (ELLs) when using automatic scoring systems for middle school students' written responses to science assessments. We specifically focus on examining how unbalanced training data with ELLs contributes to scoring bias and disparities. We fine-tuned BERT with four datasets: responses from (1) ELLs, (2) non-ELLs, (3) a mixed dataset reflecting the real-world proportion of ELLs and non-ELLs (unbalanced), and (4) a balanced mixed dataset with equal representation of both groups. The study analyzed 21 assessment items: 10 items with about 30,000 ELL responses, five items with about 1,000 ELL responses, and six items with about 200 ELL responses. Scoring accuracy (Acc) was calculated and compared to identify bias using Friedman tests. We measured the Mean Score Gaps (MSGs) between ELLs and non-ELLs and then calculated the differences in MSGs generated through both the human and AI models to identify the scoring disparities. We found that no AI bias and distorted disparities between ELLs and non-ELLs were found when the training dataset was large enough (ELL = 30,000 and ELL = 1,000), but concerns could exist if the sample size is limited (ELL = 200). 

---
# The Hitchhikers Guide to Production-ready Trustworthy Foundation Model powered Software (FMware) 

**Authors**: Kirill Vasilevski, Benjamin Rombaut, Gopi Krishnan Rajbahadur, Gustavo A. Oliva, Keheliya Gallaba, Filipe R. Cogo, Jiahuei, Dayi Lin, Haoxiang Zhang, Bouyan Chen, Kishanthan Thangarajah, Ahmed E. Hassan, Zhen Ming, Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10640)  

**Abstract**: Foundation Models (FMs) such as Large Language Models (LLMs) are reshaping the software industry by enabling FMware, systems that integrate these FMs as core components. In this KDD 2025 tutorial, we present a comprehensive exploration of FMware that combines a curated catalogue of challenges with real-world production concerns. We first discuss the state of research and practice in building FMware. We further examine the difficulties in selecting suitable models, aligning high-quality domain-specific data, engineering robust prompts, and orchestrating autonomous agents. We then address the complex journey from impressive demos to production-ready systems by outlining issues in system testing, optimization, deployment, and integration with legacy software. Drawing on our industrial experience and recent research in the area, we provide actionable insights and a technology roadmap for overcoming these challenges. Attendees will gain practical strategies to enable the creation of trustworthy FMware in the evolving technology landscape. 

---
# Agent Name Service (ANS): A Universal Directory for Secure AI Agent Discovery and Interoperability 

**Authors**: Ken Huang, Vineeth Sai Narajala, Idan Habler, Akram Sheriff  

**Link**: [PDF](https://arxiv.org/pdf/2505.10609)  

**Abstract**: The proliferation of AI agents requires robust mechanisms for secure discovery. This paper introduces the Agent Name Service (ANS), a novel architecture based on DNS addressing the lack of a public agent discovery framework. ANS provides a protocol-agnostic registry infrastructure that leverages Public Key Infrastructure (PKI) certificates for verifiable agent identity and trust. The architecture features several key innovations: a formalized agent registration and renewal mechanism for lifecycle management; DNS-inspired naming conventions with capability-aware resolution; a modular Protocol Adapter Layer supporting diverse communication standards (A2A, MCP, ACP etc.); and precisely defined algorithms for secure resolution. We implement structured communication using JSON Schema and conduct a comprehensive threat analysis of our proposal. The result is a foundational directory service addressing the core challenges of secured discovery and interaction in multi-agent systems, paving the way for future interoperable, trustworthy, and scalable agent ecosystems. 

---
# MONAQ: Multi-Objective Neural Architecture Querying for Time-Series Analysis on Resource-Constrained Devices 

**Authors**: Patara Trirat, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.10607)  

**Abstract**: The growing use of smartphones and IoT devices necessitates efficient time-series analysis on resource-constrained hardware, which is critical for sensing applications such as human activity recognition and air quality prediction. Recent efforts in hardware-aware neural architecture search (NAS) automate architecture discovery for specific platforms; however, none focus on general time-series analysis with edge deployment. Leveraging the problem-solving and reasoning capabilities of large language models (LLM), we propose MONAQ, a novel framework that reformulates NAS into Multi-Objective Neural Architecture Querying tasks. MONAQ is equipped with multimodal query generation for processing multimodal time-series inputs and hardware constraints, alongside an LLM agent-based multi-objective search to achieve deployment-ready models via code generation. By integrating numerical data, time-series images, and textual descriptions, MONAQ improves an LLM's understanding of time-series data. Experiments on fifteen datasets demonstrate that MONAQ-discovered models outperform both handcrafted models and NAS baselines while being more efficient. 

---
# Continuity and Isolation Lead to Doubts or Dilemmas in Large Language Models 

**Authors**: Hector Pasten, Felipe Urrutia, Hector Jimenez, Cristian B. Calderon, Cristóbal Rojas, Alexander Kozachinskiy  

**Link**: [PDF](https://arxiv.org/pdf/2505.10606)  

**Abstract**: Understanding how Transformers work and how they process information is key to the theoretical and empirical advancement of these machines. In this work, we demonstrate the existence of two phenomena in Transformers, namely isolation and continuity. Both of these phenomena hinder Transformers to learn even simple pattern sequences. Isolation expresses that any learnable sequence must be isolated from another learnable sequence, and hence some sequences cannot be learned by a single Transformer at the same time. Continuity entails that an attractor basin forms around a learned sequence, such that any sequence falling in that basin will collapse towards the learned sequence. Here, we mathematically prove these phenomena emerge in all Transformers that use compact positional encoding, and design rigorous experiments, demonstrating that the theoretical limitations we shed light on occur on the practical scale. 

---
# MIRAGE: A Multi-modal Benchmark for Spatial Perception, Reasoning, and Intelligence 

**Authors**: Chonghan Liu, Haoran Wang, Felix Henry, Pu Miao, Yajie Zhang, Yu Zhao, Peiran Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10604)  

**Abstract**: Spatial perception and reasoning are core components of human cognition, encompassing object recognition, spatial relational understanding, and dynamic reasoning. Despite progress in computer vision, existing benchmarks reveal significant gaps in models' abilities to accurately recognize object attributes and reason about spatial relationships, both essential for dynamic reasoning. To address these limitations, we propose MIRAGE, a multi-modal benchmark designed to evaluate models' capabilities in Counting (object attribute recognition), Relation (spatial relational reasoning), and Counting with Relation. Through diverse and complex scenarios requiring fine-grained recognition and reasoning, MIRAGE highlights critical limitations in state-of-the-art models, underscoring the need for improved representations and reasoning frameworks. By targeting these foundational abilities, MIRAGE provides a pathway toward spatiotemporal reasoning in future research. 

---
# Toward a Public and Secure Generative AI: A Comparative Analysis of Open and Closed LLMs 

**Authors**: Jorge Machado  

**Link**: [PDF](https://arxiv.org/pdf/2505.10603)  

**Abstract**: Generative artificial intelligence (Gen AI) systems represent a critical technology with far-reaching implications across multiple domains of society. However, their deployment entails a range of risks and challenges that require careful evaluation. To date, there has been a lack of comprehensive, interdisciplinary studies offering a systematic comparison between open-source and proprietary (closed) generative AI systems, particularly regarding their respective advantages and drawbacks. This study aims to: i) critically evaluate and compare the characteristics, opportunities, and challenges of open and closed generative AI models; and ii) propose foundational elements for the development of an Open, Public, and Safe Gen AI framework. As a methodology, we adopted a combined approach that integrates three methods: literature review, critical analysis, and comparative analysis. The proposed framework outlines key dimensions, openness, public governance, and security, as essential pillars for shaping the future of trustworthy and inclusive Gen AI. Our findings reveal that open models offer greater transparency, auditability, and flexibility, enabling independent scrutiny and bias mitigation. In contrast, closed systems often provide better technical support and ease of implementation, but at the cost of unequal access, accountability, and ethical oversight. The research also highlights the importance of multi-stakeholder governance, environmental sustainability, and regulatory frameworks in ensuring responsible development. 

---
# Enhancing IoT Cyber Attack Detection in the Presence of Highly Imbalanced Data 

**Authors**: Md. Ehsanul Haque, Md. Saymon Hosen Polash, Md Al-Imran Sanjida Simla, Md Alomgir Hossain, Sarwar Jahan  

**Link**: [PDF](https://arxiv.org/pdf/2505.10600)  

**Abstract**: Due to the rapid growth in the number of Internet of Things (IoT) networks, the cyber risk has increased exponentially, and therefore, we have to develop effective IDS that can work well with highly imbalanced datasets. A high rate of missed threats can be the result, as traditional machine learning models tend to struggle in identifying attacks when normal data volume is much higher than the volume of attacks. For example, the dataset used in this study reveals a strong class imbalance with 94,659 instances of the majority class and only 28 instances of the minority class, making it quite challenging to determine rare attacks accurately. The challenges presented in this research are addressed by hybrid sampling techniques designed to improve data imbalance detection accuracy in IoT domains. After applying these techniques, we evaluate the performance of several machine learning models such as Random Forest, Soft Voting, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Logistic Regression with respect to the classification of cyber-attacks. The obtained results indicate that the Random Forest model achieved the best performance with a Kappa score of 0.9903, test accuracy of 0.9961, and AUC of 0.9994. Strong performance is also shown by the Soft Voting model, with an accuracy of 0.9952 and AUC of 0.9997, indicating the benefits of combining model predictions. Overall, this work demonstrates the value of hybrid sampling combined with robust model and feature selection for significantly improving IoT security against cyber-attacks, especially in highly imbalanced data environments. 

---
# UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech 

**Authors**: Jiaxuan Liu, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.10599)  

**Abstract**: Recent neural codec language models have made great progress in the field of text-to-speech (TTS), but controllable emotional TTS still faces many challenges. Traditional methods rely on predefined discrete emotion labels to control emotion categories and intensities, which can't capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotion annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a neural codec language model unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotion annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along the three dimensions of ADV space, and exhibits superior end-to-end emotional speech synthesis capabilities. 

---
# Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment 

**Authors**: Jiazheng Zhang, Wenqing Jing, Zizhuo Zhang, Zhiheng Xi, Shihan Dou, Rongxiang Weng, Jiahuan Li, Jingang Wang, MingXu Cai, Shibo Hong, Tao Gui, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10597)  

**Abstract**: Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment. 

---
# Inclusivity of AI Speech in Healthcare: A Decade Look Back 

**Authors**: Retno Larasati  

**Link**: [PDF](https://arxiv.org/pdf/2505.10596)  

**Abstract**: The integration of AI speech recognition technologies into healthcare has the potential to revolutionize clinical workflows and patient-provider communication. However, this study reveals significant gaps in inclusivity, with datasets and research disproportionately favouring high-resource languages, standardized accents, and narrow demographic groups. These biases risk perpetuating healthcare disparities, as AI systems may misinterpret speech from marginalized groups. This paper highlights the urgent need for inclusive dataset design, bias mitigation research, and policy frameworks to ensure equitable access to AI speech technologies in healthcare. 

---
# CRPE: Expanding The Reasoning Capability of Large Language Model for Code Generation 

**Authors**: Ningxin Gui, Qianghuai Jia, Feijun Jiang, Yuling Jiao, dechun wang, Jerry Zhijian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.10594)  

**Abstract**: We introduce CRPE (Code Reasoning Process Enhancer), an innovative three-stage framework for data synthesis and model training that advances the development of sophisticated code reasoning capabilities in large language models (LLMs). Building upon existing system-1 models, CRPE addresses the fundamental challenge of enhancing LLMs' analytical and logical processing in code generation tasks. Our framework presents a methodologically rigorous yet implementable approach to cultivating advanced code reasoning abilities in language models. Through the implementation of CRPE, we successfully develop an enhanced COT-Coder that demonstrates marked improvements in code generation tasks. Evaluation results on LiveCodeBench (20240701-20240901) demonstrate that our COT-Coder-7B-StepDPO, derived from Qwen2.5-Coder-7B-Base, with a pass@1 accuracy of 21.88, exceeds all models with similar or even larger sizes. Furthermore, our COT-Coder-32B-StepDPO, based on Qwen2.5-Coder-32B-Base, exhibits superior performance with a pass@1 accuracy of 35.08, outperforming GPT4O on the benchmark. Overall, CRPE represents a comprehensive, open-source method that encompasses the complete pipeline from instruction data acquisition through expert code reasoning data synthesis, culminating in an autonomous reasoning enhancement mechanism. 

---
# LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps 

**Authors**: Shanhui Zhao, Hao Wen, Wenjie Du, Cheng Liang, Yunxin Liu, Xiaozhou Ye, Ye Ouyang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10593)  

**Abstract**: Large language models (LLMs) have opened new opportunities for automated mobile app exploration, an important and challenging problem that used to suffer from the difficulty of generating meaningful UI interactions. However, existing LLM-based exploration approaches rely heavily on LLMs to generate actions in almost every step, leading to a huge cost of token fees and computational resources. We argue that such extensive usage of LLMs is neither necessary nor effective, since many actions during exploration do not require, or may even be biased by the abilities of LLMs. Further, based on the insight that a precise and compact knowledge plays the central role for effective exploration, we introduce LLM-Explorer, a new exploration agent designed for efficiency and affordability. LLM-Explorer uses LLMs primarily for maintaining the knowledge instead of generating actions, and knowledge is used to guide action generation in a LLM-less manner. Based on a comparison with 5 strong baselines on 20 typical apps, LLM-Explorer was able to achieve the fastest and highest coverage among all automated app explorers, with over 148x lower cost than the state-of-the-art LLM-based approach. 

---
# Anchoring AI Capabilities in Market Valuations: The Capability Realization Rate Model and Valuation Misalignment Risk 

**Authors**: Xinmin Fang, Lingfeng Tao, Zhengxiong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10590)  

**Abstract**: Recent breakthroughs in artificial intelligence (AI) have triggered surges in market valuations for AI-related companies, often outpacing the realization of underlying capabilities. We examine the anchoring effect of AI capabilities on equity valuations and propose a Capability Realization Rate (CRR) model to quantify the gap between AI potential and realized performance. Using data from the 2023--2025 generative AI boom, we analyze sector-level sensitivity and conduct case studies (OpenAI, Adobe, NVIDIA, Meta, Microsoft, Goldman Sachs) to illustrate patterns of valuation premium and misalignment. Our findings indicate that AI-native firms commanded outsized valuation premiums anchored to future potential, while traditional companies integrating AI experienced re-ratings subject to proof of tangible returns. We argue that CRR can help identify valuation misalignment risk-where market prices diverge from realized AI-driven value. We conclude with policy recommendations to improve transparency, mitigate speculative bubbles, and align AI innovation with sustainable market value. 

---
# Super-Resolution Generative Adversarial Networks based Video Enhancement 

**Authors**: Kağan ÇETİN  

**Link**: [PDF](https://arxiv.org/pdf/2505.10589)  

**Abstract**: This study introduces an enhanced approach to video super-resolution by extending ordinary Single-Image Super-Resolution (SISR) Super-Resolution Generative Adversarial Network (SRGAN) structure to handle spatio-temporal data. While SRGAN has proven effective for single-image enhancement, its design does not account for the temporal continuity required in video processing. To address this, a modified framework that incorporates 3D Non-Local Blocks is proposed, which is enabling the model to capture relationships across both spatial and temporal dimensions. An experimental training pipeline is developed, based on patch-wise learning and advanced data degradation techniques, to simulate real-world video conditions and learn from both local and global structures and details. This helps the model generalize better and maintain stability across varying video content while maintaining the general structure besides the pixel-wise correctness. Two model variants-one larger and one more lightweight-are presented to explore the trade-offs between performance and efficiency. The results demonstrate improved temporal coherence, sharper textures, and fewer visual artifacts compared to traditional single-image methods. This work contributes to the development of practical, learning-based solutions for video enhancement tasks, with potential applications in streaming, gaming, and digital restoration. 

---
# Understanding Gen Alpha Digital Language: Evaluation of LLM Safety Systems for Content Moderation 

**Authors**: Manisha Mehta, Fausto Giunchiglia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10588)  

**Abstract**: This research offers a unique evaluation of how AI systems interpret the digital language of Generation Alpha (Gen Alpha, born 2010-2024). As the first cohort raised alongside AI, Gen Alpha faces new forms of online risk due to immersive digital engagement and a growing mismatch between their evolving communication and existing safety tools. Their distinct language, shaped by gaming, memes, and AI-driven trends, often conceals harmful interactions from both human moderators and automated systems. We assess four leading AI models (GPT-4, Claude, Gemini, and Llama 3) on their ability to detect masked harassment and manipulation within Gen Alpha discourse. Using a dataset of 100 recent expressions from gaming platforms, social media, and video content, the study reveals critical comprehension failures with direct implications for online safety. This work contributes: (1) a first-of-its-kind dataset capturing Gen Alpha expressions; (2) a framework to improve AI moderation systems for youth protection; (3) a multi-perspective evaluation including AI systems, human moderators, and parents, with direct input from Gen Alpha co-researchers; and (4) an analysis of how linguistic divergence increases youth vulnerability. Findings highlight the urgent need to redesign safety systems attuned to youth communication, especially given Gen Alpha reluctance to seek help when adults fail to understand their digital world. This study combines the insight of a Gen Alpha researcher with systematic academic analysis to address critical digital safety challenges. 

---
# GRNN:Recurrent Neural Network based on Ghost Features for Video Super-Resolution 

**Authors**: Yutong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10577)  

**Abstract**: Modern video super-resolution (VSR) systems based on convolutional neural networks (CNNs) require huge computational costs. The problem of feature redundancy is present in most models in many domains, but is rarely discussed in VSR. We experimentally observe that many features in VSR models are also similar to each other, so we propose to use "Ghost features" to reduce this redundancy. We also analyze the so-called "gradient disappearance" phenomenon generated by the conventional recurrent convolutional network (RNN) model, and combine the Ghost module with RNN to complete the modeling on time series. The current frame is used as input to the model together with the next frame, the output of the previous frame and the hidden state. Extensive experiments on several benchmark models and datasets show that the PSNR and SSIM of our proposed modality are improved to some extent. Some texture details in the video are also better preserved. 

---
# Large Language Models for Cancer Communication: Evaluating Linguistic Quality, Safety, and Accessibility in Generative AI 

**Authors**: Agnik Saha, Victoria Churchill, Anny D. Rodriguez, Ugur Kursuncu, Muhammed Y. Idris  

**Link**: [PDF](https://arxiv.org/pdf/2505.10472)  

**Abstract**: Effective communication about breast and cervical cancers remains a persistent health challenge, with significant gaps in public understanding of cancer prevention, screening, and treatment, potentially leading to delayed diagnoses and inadequate treatments. This study evaluates the capabilities and limitations of Large Language Models (LLMs) in generating accurate, safe, and accessible cancer-related information to support patient understanding. We evaluated five general-purpose and three medical LLMs using a mixed-methods evaluation framework across linguistic quality, safety and trustworthiness, and communication accessibility and affectiveness. Our approach utilized quantitative metrics, qualitative expert ratings, and statistical analysis using Welch's ANOVA, Games-Howell, and Hedges' g. Our results show that general-purpose LLMs produced outputs of higher linguistic quality and affectiveness, while medical LLMs demonstrate greater communication accessibility. However, medical LLMs tend to exhibit higher levels of potential harm, toxicity, and bias, reducing their performance in safety and trustworthiness. Our findings indicate a duality between domain-specific knowledge and safety in health communications. The results highlight the need for intentional model design with targeted improvements, particularly in mitigating harm and bias, and improving safety and affectiveness. This study provides a comprehensive evaluation of LLMs for cancer communication, offering critical insights for improving AI-generated health content and informing future development of accurate, safe, and accessible digital health tools. 

---
# GarmentPile: Point-Level Visual Affordance Guided Retrieval and Adaptation for Cluttered Garments Manipulation 

**Authors**: Ruihai Wu, Ziyu Zhu, Yuran Wang, Yue Chen, Jiarui Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2503.09243)  

**Abstract**: Cluttered garments manipulation poses significant challenges due to the complex, deformable nature of garments and intricate garment relations. Unlike single-garment manipulation, cluttered scenarios require managing complex garment entanglements and interactions, while maintaining garment cleanliness and manipulation stability. To address these demands, we propose to learn point-level affordance, the dense representation modeling the complex space and multi-modal manipulation candidates, while being aware of garment geometry, structure, and inter-object relations. Additionally, as it is difficult to directly retrieve a garment in some extremely entangled clutters, we introduce an adaptation module, guided by learned affordance, to reorganize highly-entangled garments into states plausible for manipulation. Our framework demonstrates effectiveness over environments featuring diverse garment types and pile configurations in both simulation and the real world. Project page: this https URL. 

---
