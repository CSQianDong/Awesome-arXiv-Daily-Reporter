# From Fragments to Facts: A Curriculum-Driven DPO Approach for Generating Hindi News Veracity Explanations 

**Authors**: Pulkit Bansal, Raghvendra Kumar, Shakti Singh, Sriparna Saha, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2507.05179)  

**Abstract**: In an era of rampant misinformation, generating reliable news explanations is vital, especially for under-represented languages like Hindi. Lacking robust automated tools, Hindi faces challenges in scaling misinformation detection. To bridge this gap, we propose a novel framework integrating Direct Preference Optimization (DPO) with curriculum learning to align machine-generated explanations with human reasoning. Fact-checked explanations from credible sources serve as preferred responses, while LLM outputs highlight system limitations and serve as non-preferred responses. To refine task-specific alignment, we introduce two key parameters -- Actuality and Finesse -- into the DPO loss function, enhancing explanation quality and consistency. Experiments with LLMs (Mistral, Llama, Gemma) and PLMs (mBART, mT5) confirm the framework's effectiveness in generating coherent, contextually relevant explanations. This scalable approach combats misinformation and extends automated explanation generation to low-resource languages. 

---
# SMART: Simulated Students Aligned with Item Response Theory for Question Difficulty Prediction 

**Authors**: Alexander Scarlatos, Nigel Fernandez, Christopher Ormerod, Susan Lottridge, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05129)  

**Abstract**: Item (question) difficulties play a crucial role in educational assessments, enabling accurate and efficient assessment of student abilities and personalization to maximize learning outcomes. Traditionally, estimating item difficulties can be costly, requiring real students to respond to items, followed by fitting an item response theory (IRT) model to get item difficulty estimates. This approach cannot be applied to the cold-start setting for previously unseen items either. In this work, we present SMART (Simulated Students Aligned with IRT), a novel method for aligning simulated students with instructed ability, which can then be used in simulations to predict the difficulty of open-ended items. We achieve this alignment using direct preference optimization (DPO), where we form preference pairs based on how likely responses are under a ground-truth IRT model. We perform a simulation by generating thousands of responses, evaluating them with an LLM-based scoring model, and fit the resulting data to an IRT model to obtain item difficulty estimates. Through extensive experiments on a real-world student response dataset, we show that SMART outperforms other item difficulty prediction methods by leveraging its improved ability alignment. 

---
# Pre-Trained Policy Discriminators are General Reward Models 

**Authors**: Shihan Dou, Shichun Liu, Yuming Yang, Yicheng Zou, Yunhua Zhou, Shuhao Xing, Chenhao Huang, Qiming Ge, Demin Song, Haijun Lv, Songyang Gao, Chengqi Lv, Enyu Zhou, Honglin Guo, Zhiheng Xi, Wenwei Zhang, Qipeng Guo, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Tao Gui, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05197)  

**Abstract**: We offer a novel perspective on reward modeling by formulating it as a policy discriminator, which quantifies the difference between two policies to generate a reward signal, guiding the training policy towards a target policy with desired behaviors. Based on this conceptual insight, we propose a scalable pre-training method named Policy Discriminative Learning (POLAR), which trains a reward model (RM) to discern identical policies and discriminate different ones. Unlike traditional reward modeling methods relying on absolute preferences, POLAR captures the relative difference between one policy and an arbitrary target policy, which is a scalable, high-level optimization objective suitable for modeling generic ranking relationships. Leveraging the POLAR pre-training paradigm, we present a series of RMs with parameter scales from 1.8B to 7B. Empirical results show that POLAR substantially outperforms traditional non-pre-trained methods, significantly enhancing RM performance. For instance, POLAR-7B could improve preference accuracy from 54.8% to 81.0% on STEM tasks and from 57.9% to 85.5% on creative writing tasks compared to SOTA baselines. POLAR also shows robust generalization capabilities in RLHF using Reinforcement Fine-tuning (RFT), providing reliable reward signals and markedly enhancing policy performance--improving LLaMa3.1-8B from an average of 47.36% to 56.33% and Qwen2.5-32B from 64.49% to 70.47% on 20 benchmarks. Moreover, scaling experiments reveal a clear power-law relationship between computation and performance, supported by linear correlation coefficients approaching 0.99. The impressive performance, strong generalization, and scaling properties suggest that POLAR is a promising direction for developing general and strong reward models. 

---
# R1-RE: Cross-Domain Relationship Extraction with RLVR 

**Authors**: Runpeng Dai, Tong Zheng, Run Yang, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04642)  

**Abstract**: Relationship extraction (RE) is a core task in natural language processing. Traditional approaches typically frame RE as a supervised learning problem, directly mapping context to labels-an approach that often suffers from poor out-of-domain (OOD) generalization. Inspired by the workflow of human annotators, we reframe RE as a reasoning task guided by annotation guidelines and introduce R1-RE, the first reinforcement learning with verifiable reward (RLVR) framework for RE tasks. Our method elicits the reasoning abilities of small language models for annotation tasks, resulting in significantly improved OOD robustness. We evaluate our approach on the public Sem-2010 dataset and a private MDKG dataset. The R1-RE-7B model attains an average OOD accuracy of approximately 70%, on par with leading proprietary models such as GPT-4o. Additionally, our comprehensive analysis provides novel insights into the training dynamics and emergent reasoning behaviors of the RLVR paradigm for RE. 

---
# Does Learning Mathematical Problem-Solving Generalize to Broader Reasoning? 

**Authors**: Ruochen Zhou, Minrui Xu, Shiqi Chen, Junteng Liu, Yunqi Li, Xinxin Lin, Zhengyu Chen, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04391)  

**Abstract**: There has been a growing interest in enhancing the mathematical problem-solving (MPS) capabilities of large language models. While the majority of research efforts concentrate on creating specialized models to solve mathematical problems, it remains unknown how learning mathematical problem-solving generalizes to help develop other reasoning abilities. In this paper, we present an empirical investigation into the generalization potential of various MPS training approaches, such as continual pretraining, instruction tuning, and rule-based reinforcement learning across various data sources, including both short and long chain-of-thought (CoT) samples. Evaluation on 5 mathematical and 8 general reasoning benchmarks show that continual pretraining on math text is able to generalize to general reasoning tasks to some extent. In constrast, instruction tuning on conventional, short MPS samples provides limited benefits and, in many cases, even impairs generalization performance. Notably, training with long CoT responses for MPS samples and incorporating rule-based reinforcement learning on MPS queries exhibit distinct behavior, significantly enhancing generalization by extending the model's reasoning processes into other domains. These results suggest that traditional approaches to learning MPS with short reasoning chains largely fail to achieve robust generalization. However, the emerging paradigm of longer reasoning chains, coupled with self-reflection, offers a promising direction for improving generalized reasoning abilities through learning from specialized domains. 

---
# Conversation Forests: The Key to Fine Tuning Large Language Models for Multi-Turn Medical Conversations is Branching 

**Authors**: Thomas Savage  

**Link**: [PDF](https://arxiv.org/pdf/2507.04099)  

**Abstract**: Fine-tuning methods such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) have demonstrated success in training large language models (LLMs) for single-turn tasks. However, these methods fall short in multi-turn applications, such as diagnostic patient interviewing, where understanding how early conversational turns influence downstream completions and outcomes is essential. In medicine, a multi-turn perspective is critical for learning diagnostic schemas and better understanding conversation dynamics. To address this gap, I introduce Savage Conversation Forests (SCF), a reinforcement learning framework that leverages a branched conversation architecture to fine-tune LLMs for multi-turn dialogue. SCF generates multiple possible conversation continuations at each turn, enabling the model to learn how different early responses affect downstream interactions and diagnostic outcomes. In experiments simulating doctor-patient conversations, SCF with branching outperforms linear conversation architectures on diagnostic accuracy. I hypothesize that SCF's improvements stem from its ability to provide richer, interdependent training signals across conversation turns. These results suggest that a branched training architecture is an important strategy for fine tuning LLMs in complex multi-turn conversational tasks. 

---
# Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making 

**Authors**: Sang Quang Nguyen, Kiet Van Nguyen, Vinh-Tiep Nguyen, Thanh Duc Ngo, Ngan Luu-Thuy Nguyen, Dinh-Duy Le  

**Link**: [PDF](https://arxiv.org/pdf/2507.03711)  

**Abstract**: In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, Ô Ăn Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the Ô Ăn Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities. 

---
# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents 

**Authors**: Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03112)  

**Abstract**: Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents. 

---
# ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization 

**Authors**: YuXuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03069)  

**Abstract**: With the rapid advancement of Reinforcement Learning from Human Feedback (RLHF) and autoregressive transformers, state-of-the-art models such as GPT-4.0, DeepSeek R1, and Llama 3.3 increasingly emphasize answer depth and personalization. However, most existing RLHF approaches (e.g., PPO, DPO) still rely on a binary-preference (BT) paradigm, which, while reducing annotation costs, still requires substantial human effort and captures only group-level tendencies rather than individual preferences. To overcome these limitations, we propose Adaptive Reward-Following (ARF), a self-assessment framework that leverages a high-precision emotion analyzer achieving over 70% accuracy on GoEmotions, Sentiment140, and DailyDialog to convert free-form user feedback into continuous preference scores. We further enrich and debias these signals through lightweight data augmentations, including synonym replacement, random trace truncation, and score bias annotation algorithm. A Dynamic Adapter Preference Tracker continuously models evolving user tastes in real time, enabling our novel Trace Bias (TB) fine-tuning algorithm to optimize directly on these tracked rewards instead of coarse binary labels. Experiments on Qwen-2/2.5, Gemma-2, and Llama-3.2 across four preference domains demonstrate that ARF achieves an improvement of 3.3% over PPO and 7.6% over DPO. Moreover, TB preserves theoretical alignment with PPO and DPO objectives. Overall, ARF presents a scalable, personalized, and cost-effective approach to RLHF LLMs through autonomous reward modeling. 

---
# OpenTable-R1: A Reinforcement Learning Augmented Tool Agent for Open-Domain Table Question Answering 

**Authors**: Zipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03018)  

**Abstract**: Open-domain table question answering traditionally relies on a two-stage pipeline: static table retrieval followed by a closed-domain answer. In contrast, we propose an end-to-end agentic framework that embeds multi-turn tool calls-using a BM25+-based search API and a SQLite SQL executor-directly into a large language model. To further adapt a compact 4B-parameter model, we introduce a two-stage fine-tuning process: supervised cold-start on easy questions, then Async GRPO reinforcement learning on harder cases with LoRA adapters and a rollout buffer. This unified approach enables the model to jointly retrieve, reason, and execute queries, yielding a dramatic accuracy improvement from single-digit zero-shot performance to over 0.86 exact match on a held-out test set. Our results underscore the effectiveness of integrating structured tool calls with targeted RL fine-tuning for scalable, accurate table QA. The code is available at this https URL. 

---
# RADIANT: Retrieval AugmenteD entIty-context AligNmenT -- Introducing RAG-ability and Entity-Context Divergence 

**Authors**: Vipula Rawte, Rajarshi Roy, Gurpreet Singh, Danush Khanna, Yaswanth Narsupalli, Basab Ghosh, Abhay Gupta, Argha Kamal Samanta, Aditya Shingote, Aadi Krishna Vikram, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.02949)  

**Abstract**: As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation. 

---
# Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning 

**Authors**: Yana Wei, Liang Zhao, Jianjian Sun, Kangheng Lin, Jisheng Yin, Jingcheng Hu, Yinmin Zhang, En Yu, Haoran Lv, Zejia Weng, Jia Wang, Chunrui Han, Yuang Peng, Qi Han, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2507.05255)  

**Abstract**: The remarkable reasoning capability of large language models (LLMs) stems from cognitive behaviors that emerge through reinforcement with verifiable rewards. This work investigates how to transfer this principle to Multimodal LLMs (MLLMs) to unlock advanced visual reasoning. We introduce a two-stage paradigm built on Qwen2.5-VL-7B: a massive linguistic cold-start fine-tuning, followed by multimodal reinforcement learning (RL) spanning nearly 1,000 steps, surpassing all previous open-source efforts in scale. This pioneering work reveals three fundamental insights: 1) Behavior transfer emerges surprisingly early in cold start due to linguistic mental imagery. 2) Cold start broadly memorizes visual behaviors, while RL critically discerns and scales up effective patterns. 3) Transfer strategically favors high-utility behaviors such as visual reflection. Our resulting model, Open-Vision-Reasoner (OVR), achieves state-of-the-art performance on a suite of reasoning benchmarks, including 95.3% on MATH500, 51.8% on MathVision and 54.6% on MathVerse. We release our model, data, and training dynamics to catalyze the development of more capable, behavior-aligned multimodal reasoners. 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

---
# Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Authors**: Riya Naik, Ashwin Srinivasan, Swati Agarwal, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03726)  

**Abstract**: Many of us now treat LLMs as modern-day oracles asking it almost any kind of question. However, consulting an LLM does not have to be a single turn activity. But long multi-turn interactions can get tedious if it is simply to clarify contextual information that can be arrived at through reasoning. In this paper, we examine the use of agent-based architecture to bolster LLM-based Question-Answering systems with additional reasoning capabilities. We examine the automatic resolution of potential incompleteness or ambiguities in questions by transducers implemented using LLM-based agents. We focus on several benchmark datasets that are known to contain questions with these deficiencies to varying degrees. We equip different LLMs (GPT-3.5-Turbo and Llama-4-Scout) with agents that act as specialists in detecting and resolving deficiencies of incompleteness and ambiguity. The agents are implemented as zero-shot ReAct agents. Rather than producing an answer in a single step, the model now decides between 3 actions a) classify b) resolve c) answer. Action a) decides if the question is incomplete, ambiguous, or normal. Action b) determines if any deficiencies identified can be resolved. Action c) answers the resolved form of the question. We compare the use of LLMs with and without the use of agents with these components. Our results show benefits of agents with transducer 1) A shortening of the length of interactions with human 2) An improvement in the answer quality and 3) Explainable resolution of deficiencies in the question. On the negative side we find while it may result in additional LLM invocations and in some cases, increased latency. But on tested datasets, the benefits outweigh the costs except when questions already have sufficient context. Suggesting the agent-based approach could be a useful mechanism to harness the power of LLMs to develop more robust QA systems. 

---
# Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2507.03051)  

**Abstract**: Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection.
We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul.
The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection. 

---
# ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning 

**Authors**: Zhirong Chen, Kaiyan Chang, Zhuolin Li, Xinyang He, Chujie Chen, Cangyuan Li, Mengdi Wang, Haobo Xu, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04736)  

**Abstract**: Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities.
To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github. 

---
# A Technical Survey of Reinforcement Learning Techniques for Large Language Models 

**Authors**: Saksham Sahai Srivastava, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.04136)  

**Abstract**: Reinforcement Learning (RL) has emerged as a transformative approach for aligning and enhancing Large Language Models (LLMs), addressing critical challenges in instruction following, ethical alignment, and reasoning capabilities. This survey offers a comprehensive foundation on the integration of RL with language models, highlighting prominent algorithms such as Proximal Policy Optimization (PPO), Q-Learning, and Actor-Critic methods. Additionally, it provides an extensive technical overview of RL techniques specifically tailored for LLMs, including foundational methods like Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), as well as advanced strategies such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). We systematically analyze their applications across domains, i.e., from code generation to tool-augmented reasoning. We also present a comparative taxonomy based on reward modeling, feedback mechanisms, and optimization strategies. Our evaluation highlights key trends. RLHF remains dominant for alignment, and outcome-based RL such as RLVR significantly improves stepwise reasoning. However, persistent challenges such as reward hacking, computational costs, and scalable feedback collection underscore the need for continued innovation. We further discuss emerging directions, including hybrid RL algorithms, verifier-guided training, and multi-objective alignment frameworks. This survey serves as a roadmap for researchers advancing RL-driven LLM development, balancing capability enhancement with safety and scalability. 

---
# Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03871)  

**Abstract**: The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods. 

---
# CTR-Guided Generative Query Suggestion in Conversational Search 

**Authors**: Erxue Min, Hsiu-Yuan Huang, Xihong Yang, Min Yang, Xin Jia, Yunfang Wu, Hengyi Cai, Junfeng Wang, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04072)  

**Abstract**: Generating effective query suggestions in conversational search requires aligning model outputs with user preferences, which is challenging due to sparse and noisy click signals. We propose GQS, a generative framework that integrates click modeling and preference optimization to enhance real-world user engagement. GQS consists of three key components: (1) a Multi-Source CTR Modeling module that captures diverse contextual signals to estimate fine-grained click-through rates; (2) a Diversity-Aware Preference Alignment strategy using CTR-weighted Direct Preference Optimization (DPO), which balances relevance and semantic diversity; and (3) a CTR-Calibrated Iterative Optimization process that jointly refines the CTR and generation models across training rounds. Experiments on two real-world tasks demonstrate that GQS outperforms strong baselines in CTR, relevance, and diversity. 

---
