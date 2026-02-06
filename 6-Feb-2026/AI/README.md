# DyTopo: Dynamic Topology Routing for Multi-Agent Reasoning via Semantic Matching 

**Authors**: Yuxing Lu, Yucheng Hu, Xukai Zhao, Jiuxin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.06039)  

**Abstract**: Multi-agent systems built from prompted large language models can improve multi-round reasoning, yet most existing pipelines rely on fixed, trajectory-wide communication patterns that are poorly matched to the stage-dependent needs of iterative problem solving. We introduce DyTopo, a manager-guided multi-agent framework that reconstructs a sparse directed communication graph at each round. Conditioned on the manager's round goal, each agent outputs lightweight natural-language query (need) and \key (offer) descriptors; DyTopo embeds these descriptors and performs semantic matching, routing private messages only along the induced edges. Across code generation and mathematical reasoning benchmarks and four LLM backbones, DyTopo consistently outperforms over the strongest baseline (avg. +6.2). Beyond accuracy, DyTopo yields an interpretable coordination trace via the evolving graphs, enabling qualitative inspection of how communication pathways reconfigure across rounds. 

---
# Learning Event-Based Shooter Models from Virtual Reality Experiments 

**Authors**: Christopher A. McClurg, Alan R. Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2602.06023)  

**Abstract**: Virtual reality (VR) has emerged as a powerful tool for evaluating school security measures in high-risk scenarios such as school shootings, offering experimental control and high behavioral fidelity. However, assessing new interventions in VR requires recruiting new participant cohorts for each condition, making large-scale or iterative evaluation difficult. These limitations are especially restrictive when attempting to learn effective intervention strategies, which typically require many training episodes. To address this challenge, we develop a data-driven discrete-event simulator (DES) that models shooter movement and in-region actions as stochastic processes learned from participant behavior in VR studies. We use the simulator to examine the impact of a robot-based shooter intervention strategy. Once shown to reproduce key empirical patterns, the DES enables scalable evaluation and learning of intervention strategies that are infeasible to train directly with human subjects. Overall, this work demonstrates a high-to-mid fidelity simulation workflow that provides a scalable surrogate for developing and evaluating autonomous school-security interventions. 

---
# AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions 

**Authors**: Xianyang Liu, Shangding Gu, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.06008)  

**Abstract**: Large language model (LLM)-based agents are increasingly expected to negotiate, coordinate, and transact autonomously, yet existing benchmarks lack principled settings for evaluating language-mediated economic interaction among multiple agents. We introduce AgenticPay, a benchmark and simulation framework for multi-agent buyer-seller negotiation driven by natural language. AgenticPay models markets in which buyers and sellers possess private constraints and product-dependent valuations, and must reach agreements through multi-round linguistic negotiation rather than numeric bidding alone. The framework supports a diverse suite of over 110 tasks ranging from bilateral bargaining to many-to-many markets, with structured action extraction and metrics for feasibility, efficiency, and welfare. Benchmarking state-of-the-art proprietary and open-weight LLMs reveals substantial gaps in negotiation performance and highlights challenges in long-horizon strategic reasoning, establishing AgenticPay as a foundation for studying agentic commerce and language-based market interaction. Code and dataset are available at the link: this https URL. 

---
# Speech Emotion Recognition Leveraging OpenAI's Whisper Representations and Attentive Pooling Methods 

**Authors**: Ali Shendabadi, Parnia Izadirad, Mostafa Salehi, Mahmoud Bijankhan  

**Link**: [PDF](https://arxiv.org/pdf/2602.06000)  

**Abstract**: Speech Emotion Recognition (SER) research has faced limitations due to the lack of standard and sufficiently large datasets. Recent studies have leveraged pre-trained models to extract features for downstream tasks such as SER. This work explores the capabilities of Whisper, a pre-trained ASR system, in speech emotion recognition by proposing two attention-based pooling methods, Multi-head Attentive Average Pooling and QKV Pooling, designed to efficiently reduce the dimensionality of Whisper representations while preserving emotional features. We experiment on English and Persian, using the IEMOCAP and ShEMO datasets respectively, with Whisper Tiny and Small. Our multi-head QKV architecture achieves state-of-the-art results on the ShEMO dataset, with a 2.47% improvement in unweighted accuracy. We further compare the performance of different Whisper encoder layers and find that intermediate layers often perform better for SER on the Persian dataset, providing a lightweight and efficient alternative to much larger models such as HuBERT X-Large. Our findings highlight the potential of Whisper as a representation extractor for SER and demonstrate the effectiveness of attention-based pooling for dimension reduction. 

---
# Geographically-aware Transformer-based Traffic Forecasting for Urban Motorway Digital Twins 

**Authors**: Krešimir Kušić, Vinny Cahill, Ivana Dusparic  

**Link**: [PDF](https://arxiv.org/pdf/2602.05983)  

**Abstract**: The operational effectiveness of digital-twin technology in motorway traffic management depends on the availability of a continuous flow of high-resolution real-time traffic data. To function as a proactive decision-making support layer within traffic management, a digital twin must also incorporate predicted traffic conditions in addition to real-time observations. Due to the spatio-temporal complexity and the time-variant, non-linear nature of traffic dynamics, predicting motorway traffic remains a difficult problem. Sequence-based deep-learning models offer clear advantages over classical machine learning and statistical models in capturing long-range, temporal dependencies in time-series traffic data, yet limitations in forecasting accuracy and model complexity point to the need for further improvements. To improve motorway traffic forecasting, this paper introduces a Geographically-aware Transformer-based Traffic Forecasting GATTF model, which exploits the geographical relationships between distributed sensors using their mutual information (MI). The model has been evaluated using real-time data from the Geneva motorway network in Switzerland and results confirm that incorporating geographical awareness through MI enhances the accuracy of GATTF forecasting compared to a standard Transformer, without increasing model complexity. 

---
# Quantum Reinforcement Learning with Transformers for the Capacitated Vehicle Routing Problem 

**Authors**: Eva Andrés  

**Link**: [PDF](https://arxiv.org/pdf/2602.05920)  

**Abstract**: This paper addresses the Capacitated Vehicle Routing Problem (CVRP) by comparing classical and quantum Reinforcement Learning (RL) approaches. An Advantage Actor-Critic (A2C) agent is implemented in classical, full quantum, and hybrid variants, integrating transformer architectures to capture the relationships between vehicles, clients, and the depot through self- and cross-attention mechanisms. The experiments focus on multi-vehicle scenarios with capacity constraints, considering 20 clients and 4 vehicles, and are conducted over ten independent runs. Performance is assessed using routing distance, route compactness, and route overlap. The results show that all three approaches are capable of learning effective routing policies. However, quantum-enhanced models outperform the classical baseline and produce more robust route organization, with the hybrid architecture achieving the best overall performance across distance, compactness, and route overlap. In addition to quantitative improvements, qualitative visualizations reveal that quantum-based models generate more structured and coherent routing solutions. These findings highlight the potential of hybrid quantum-classical reinforcement learning models for addressing complex combinatorial optimization problems such as the CVRP. 

---
# A Guide to Large Language Models in Modeling and Simulation: From Core Techniques to Critical Challenges 

**Authors**: Philippe J. Giabbanelli  

**Link**: [PDF](https://arxiv.org/pdf/2602.05883)  

**Abstract**: Large language models (LLMs) have rapidly become familiar tools to researchers and practitioners. Concepts such as prompting, temperature, or few-shot examples are now widely recognized, and LLMs are increasingly used in Modeling & Simulation (M&S) workflows. However, practices that appear straightforward may introduce subtle issues, unnecessary complexity, or may even lead to inferior results. Adding more data can backfire (e.g., deteriorating performance through model collapse or inadvertently wiping out existing guardrails), spending time on fine-tuning a model can be unnecessary without a prior assessment of what it already knows, setting the temperature to 0 is not sufficient to make LLMs deterministic, providing a large volume of M&S data as input can be excessive (LLMs cannot attend to everything) but naive simplifications can lose information. We aim to provide comprehensive and practical guidance on how to use LLMs, with an emphasis on M&S applications. We discuss common sources of confusion, including non-determinism, knowledge augmentation (including RAG and LoRA), decomposition of M&S data, and hyper-parameter settings. We emphasize principled design choices, diagnostic strategies, and empirical evaluation, with the goal of helping modelers make informed decisions about when, how, and whether to rely on LLMs. 

---
# Agent2Agent Threats in Safety-Critical LLM Assistants: A Human-Centric Taxonomy 

**Authors**: Lukas Stappen, Ahmet Erkan Turan, Johann Hagerer, Georg Groh  

**Link**: [PDF](https://arxiv.org/pdf/2602.05877)  

**Abstract**: The integration of Large Language Model (LLM)-based conversational agents into vehicles creates novel security challenges at the intersection of agentic AI, automotive safety, and inter-agent communication. As these intelligent assistants coordinate with external services via protocols such as Google's Agent-to-Agent (A2A), they establish attack surfaces where manipulations can propagate through natural language payloads, potentially causing severe consequences ranging from driver distraction to unauthorized vehicle control. Existing AI security frameworks, while foundational, lack the rigorous "separation of concerns" standard in safety-critical systems engineering by co-mingling the concepts of what is being protected (assets) with how it is attacked (attack paths). This paper addresses this methodological gap by proposing a threat modeling framework called AgentHeLLM (Agent Hazard Exploration for LLM Assistants) that formally separates asset identification from attack path analysis. We introduce a human-centric asset taxonomy derived from harm-oriented "victim modeling" and inspired by the Universal Declaration of Human Rights, and a formal graph-based model that distinguishes poison paths (malicious data propagation) from trigger paths (activation actions). We demonstrate the framework's practical applicability through an open-source attack path suggestion tool AgentHeLLM Attack Path Generator that automates multi-stage threat discovery using a bi-level search strategy. 

---
# Beyond Manual Planning: Seating Allocation for Large Organizations 

**Authors**: Anton Ipsen, Michael Cashmore, Kirsty Fielding, Nicolas Marchesotti, Parisa Zehtabi, Daniele Magazzeni, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.05875)  

**Abstract**: We introduce the Hierarchical Seating Allocation Problem (HSAP) which addresses the optimal assignment of hierarchically structured organizational teams to physical seating arrangements on a floor plan. This problem is driven by the necessity for large organizations with large hierarchies to ensure that teams with close hierarchical relationships are seated in proximity to one another, such as ensuring a research group occupies a contiguous area. Currently, this problem is managed manually leading to infrequent and suboptimal replanning efforts. To alleviate this manual process, we propose an end-to-end framework to solve the HSAP. A scalable approach to calculate the distance between any pair of seats using a probabilistic road map (PRM) and rapidly-exploring random trees (RRT) which is combined with heuristic search and dynamic programming approach to solve the HSAP using integer programming. We demonstrate our approach under different sized instances by evaluating the PRM framework and subsequent allocations both quantitatively and qualitatively. 

---
# BABE: Biology Arena BEnchmark 

**Authors**: Junting Zhou, Jin Chen, Linfeng Hao, Denghui Cao, Zheyu Wang, Qiguang Chen, Chaoyou Fu, Jiaze Chen, Yuchen Wu, Ge Zhang, Mingxuan Wang, Wenhao Huang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05857)  

**Abstract**: The rapid evolution of large language models (LLMs) has expanded their capabilities from basic dialogue to advanced scientific reasoning. However, existing benchmarks in biology often fail to assess a critical skill required of researchers: the ability to integrate experimental results with contextual knowledge to derive meaningful conclusions. To address this gap, we introduce BABE(Biology Arena BEnchmark), a comprehensive benchmark designed to evaluate the experimental reasoning capabilities of biological AI systems. BABE is uniquely constructed from peer-reviewed research papers and real-world biological studies, ensuring that tasks reflect the complexity and interdisciplinary nature of actual scientific inquiry. BABE challenges models to perform causal reasoning and cross-scale inference. Our benchmark provides a robust framework for assessing how well AI systems can reason like practicing scientists, offering a more authentic measure of their potential to contribute to biological research. 

---
# OmniVideo-R1: Reinforcing Audio-visual Reasoning with Query Intention and Modality Attention 

**Authors**: Zhangquan Chen, Jiale Tao, Ruihuang Li, Yihao Hu, Ruitao Chen, Zhantao Yang, Xinlei Yu, Haodong Jing, Manyuan Zhang, Shuai Shao, Biao Wang, Qinglin Lu, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05847)  

**Abstract**: While humans perceive the world through diverse modalities that operate synergistically to support a holistic understanding of their surroundings, existing omnivideo models still face substantial challenges on audio-visual understanding tasks. In this paper, we propose OmniVideo-R1, a novel reinforced framework that improves mixed-modality reasoning. OmniVideo-R1 empowers models to "think with omnimodal cues" by two key strategies: (1) query-intensive grounding based on self-supervised learning paradigms; and (2) modality-attentive fusion built upon contrastive learning paradigms. Extensive experiments on multiple benchmarks demonstrate that OmniVideo-R1 consistently outperforms strong baselines, highlighting its effectiveness and robust generalization capabilities. 

---
# Learning Compact Boolean Networks 

**Authors**: Shengpu Wang, Yuhao Mao, Yani Zhang, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2602.05830)  

**Abstract**: Floating-point neural networks dominate modern machine learning but incur substantial inference cost, motivating interest in Boolean networks for resource-constrained settings. However, learning compact and accurate Boolean networks is challenging due to their combinatorial nature. In this work, we address this challenge from three different angles: learned connections, compact convolutions and adaptive discretization. First, we propose a novel strategy to learn efficient connections with no additional parameters and negligible computational overhead. Second, we introduce a novel convolutional Boolean architecture that exploits the locality with reduced number of Boolean operations than existing methods. Third, we propose an adaptive discretization strategy to reduce the accuracy drop when converting a continuous-valued network into a Boolean one. Extensive results on standard vision benchmarks demonstrate that the Pareto front of accuracy vs. computation of our method significantly outperforms prior state-of-the-art, achieving better accuracy with up to 37x fewer Boolean operations. 

---
# TKG-Thinker: Towards Dynamic Reasoning over Temporal Knowledge Graphs via Agentic Reinforcement Learning 

**Authors**: Zihao Jiang, Miao Peng, Zhenyan Shan, Wenjie Xu, Ben Liu, Gong Chen, Ziqi Gao, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.05818)  

**Abstract**: Temporal knowledge graph question answering (TKGQA) aims to answer time-sensitive questions by leveraging temporal knowledge bases. While Large Language Models (LLMs) demonstrate significant potential in TKGQA, current prompting strategies constrain their efficacy in two primary ways. First, they are prone to reasoning hallucinations under complex temporal constraints. Second, static prompting limits model autonomy and generalization, as it lack optimization through dynamic interaction with temporal knowledge graphs (TKGs) environments. To address these limitations, we propose \textbf{TKG-Thinker}, a novel agent equipped with autonomous planning and adaptive retrieval capabilities for reasoning over TKGs. Specifically, TKG-Thinker performs in-depth temporal reasoning through dynamic multi-turn interactions with TKGs via a dual-training strategy. We first apply Supervised Fine-Tuning (SFT) with chain-of thought data to instill core planning capabilities, followed by a Reinforcement Learning (RL) stage that leverages multi-dimensional rewards to refine reasoning policies under intricate temporal constraints. Experimental results on benchmark datasets with three open-source LLMs show that TKG-Thinker achieves state-of-the-art performance and exhibits strong generalization across complex TKGQA settings. 

---
# STProtein: predicting spatial protein expression from multi-omics data 

**Authors**: Zhaorui Jiang, Yingfang Yuan, Lei Hu, Wei Pang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05811)  

**Abstract**: The integration of spatial multi-omics data from single tissues is crucial for advancing biological research. However, a significant data imbalance impedes progress: while spatial transcriptomics data is relatively abundant, spatial proteomics data remains scarce due to technical limitations and high costs. To overcome this challenge we propose STProtein, a novel framework leveraging graph neural networks with multi-task learning strategy. STProtein is designed to accurately predict unknown spatial protein expression using more accessible spatial multi-omics data, such as spatial transcriptomics. We believe that STProtein can effectively addresses the scarcity of spatial proteomics, accelerating the integration of spatial multi-omics and potentially catalyzing transformative breakthroughs in life sciences. This tool enables scientists to accelerate discovery by identifying complex and previously hidden spatial patterns of proteins within tissues, uncovering novel relationships between different marker genes, and exploring the biological "Dark Matter". 

---
# NEX: Neuron Explore-Exploit Scoring for Label-Free Chain-of-Thought Selection and Model Ranking 

**Authors**: Kang Chen, Zhuoka Feng, Sihan Zhao, Kai Xiong, Junjie Nian, Yaoning Wang, Changyi Xiao, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05805)  

**Abstract**: Large language models increasingly spend inference compute sampling multiple chain-of-thought traces or searching over merged checkpoints. This shifts the bottleneck from generation to selection, often without supervision on the target distribution. We show entropy-based exploration proxies follow an inverted-U with accuracy, suggesting extra exploration can become redundant and induce overthinking. We propose NEX, a white-box label-free unsupervised scoring framework that views reasoning as alternating E-phase (exploration) and X-phase (exploitation). NEX detects E-phase as spikes in newly activated MLP neurons per token from sparse activation caches, then uses a sticky two-state HMM to infer E-X phases and credits E-introduced neurons by whether they are reused in the following X span. These signals yield interpretable neuron weights and a single Good-Mass Fraction score to rank candidate responses and merged variants without task answers. Across reasoning benchmarks and Qwen3 merge families, NEX computed on a small unlabeled activation set predicts downstream accuracy and identifies better variants; we further validate the E-X signal with human annotations and provide causal evidence via "Effective-vs-Redundant" neuron transfer. 

---
# FiMI: A Domain-Specific Language Model for Indian Finance Ecosystem 

**Authors**: Aboli Kathar, Aman Kumar, Anusha Kamath, Araveeti Srujan, Ashish Sharma, Chandra Bhushan, Dilip Asbe, Divya Sorate, Duddu Prasanth Kumar, Evan Acharya, Harsh Sharma, Hrithik Kadam, Kanishk Singla, Keyur Doshi, Kiran Praveen, Kolisetty Krishna SK, Krishanu Adhikary, Lokesh MPT, Mayurdeep Sonowal, Nadeem Shaikh, Navya Prakash, Nimit Kothari, Nitin Kukreja, Prashant Devadiga, Rakesh Paul, Ratanjeet Pratap Chauhan, Raunak Kalani, Raviraj Joshi, Shamanth MH, Shantanu Pandey, Shubham Soni, Siddharth Dixit, Smriti Jopat, Sunil Patel, Suraj Singh, Suvradip Paul, Tulasi Pilla, Utkarsh Vaidya, Vineeth Nambiar, Vishal Kanvaty, Yatharth Dedhia  

**Link**: [PDF](https://arxiv.org/pdf/2602.05794)  

**Abstract**: We present FiMI (Finance Model for India), a domain-specialized financial language model developed for Indian digital payment systems. We develop two model variants: FiMI Base and FiMI Instruct. FiMI adapts the Mistral Small 24B architecture through a multi-stage training pipeline, beginning with continuous pre-training on 68 Billion tokens of curated financial, multilingual (English, Hindi, Hinglish), and synthetic data. This is followed by instruction fine-tuning and domain-specific supervised fine-tuning focused on multi-turn, tool-driven conversations that model real-world workflows, such as transaction disputes and mandate lifecycle management. Evaluations reveal that FiMI Base achieves a 20% improvement over the Mistral Small 24B Base model on finance reasoning benchmark, while FiMI Instruct outperforms the Mistral Small 24B Instruct model by 87% on domain-specific tool-calling. Moreover, FiMI achieves these significant domain gains while maintaining comparable performance to models of similar size on general benchmarks. 

---
# RL-VLA$^3$: Reinforcement Learning VLA Accelerating via Full Asynchronism 

**Authors**: Zhong Guan, Haoran Sun, Yongjian Guo, Shuai Di, Xiaodong Bai, Jing Long, Tianyun Zhao, Mingxi Luo, Chen Zhou, Yucheng Guo, Qiming Yang, Wanting Xu, Wen Huang, Yunxuan Ma, Hongke Zhao, Likang Wu, Xiaotie Deng, Xi Xiao, Sheng Wen, Yicheng Gong, Junwu Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.05765)  

**Abstract**: In recent years, Vision-Language-Action (VLA) models have emerged as a crucial pathway towards general embodied intelligence, yet their training efficiency has become a key bottleneck. Although existing reinforcement learning (RL)-based training frameworks like RLinf can enhance model generalization, they still rely on synchronous execution, leading to severe resource underutilization and throughput limitations during environment interaction, policy generation (rollout), and model update phases (actor). To overcome this challenge, this paper, for the first time, proposes and implements a fully-asynchronous policy training framework encompassing the entire pipeline from environment interaction, rollout generation, to actor policy updates. Systematically drawing inspiration from asynchronous optimization ideas in large model RL, our framework designs a multi-level decoupled architecture. This includes asynchronous parallelization of environment interaction and trajectory collection, streaming execution for policy generation, and decoupled scheduling for training updates. We validated the effectiveness of our method across diverse VLA models and environments. On the LIBERO benchmark, the framework achieves throughput improvements of up to 59.25\% compared to existing synchronous strategies. When deeply optimizing separation strategies, throughput can be increased by as much as 126.67\%. We verified the effectiveness of each asynchronous component via ablation studies. Scaling law validation across 8 to 256 GPUs demonstrates our method's excellent scalability under most conditions. 

---
# RocqSmith: Can Automatic Optimization Forge Better Proof Agents? 

**Authors**: Andrei Kozyrev, Nikita Khramov, Denis Lochmelis, Valerio Morelli, Gleb Solovev, Anton Podkopaev  

**Link**: [PDF](https://arxiv.org/pdf/2602.05762)  

**Abstract**: This work studies the applicability of automatic AI agent optimization methods to real-world agents in formal verification settings, focusing on automated theorem proving in Rocq as a representative and challenging domain. We evaluate how different automatic agent optimizers perform when applied to the task of optimizing a Rocq proof-generation agent, and assess whether parts of the fine-grained tuning of agentic systems, such as prompt design, contextual knowledge, and control strategies, can be automated. Our results show that while several optimizers yield measurable improvements, simple few-shot bootstrapping is the most consistently effective; however, none of the studied methods matches the performance of a carefully engineered state-of-the-art proof agent. 

---
# LeakBoost: Perceptual-Loss-Based Membership Inference Attack 

**Authors**: Amit Kravchik Taub, Fred M. Grabovski, Guy Amit, Yisroel Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.05748)  

**Abstract**: Membership inference attacks (MIAs) aim to determine whether a sample was part of a model's training set, posing serious privacy risks for modern machine-learning systems. Existing MIAs primarily rely on static indicators, such as loss or confidence, and do not fully leverage the dynamic behavior of models when actively probed. We propose LeakBoost, a perceptual-loss-based interrogation framework that actively probes a model's internal representations to expose hidden membership signals. Given a candidate input, LeakBoost synthesizes an interrogation image by optimizing a perceptual (activation-space) objective, amplifying representational differences between members and non-members. This image is then analyzed by an off-the-shelf membership detector, without modifying the detector itself. When combined with existing membership inference methods, LeakBoost achieves substantial improvements at low false-positive rates across multiple image classification datasets and diverse neural network architectures. In particular, it raises AUC from near-chance levels (0.53-0.62) to 0.81-0.88, and increases TPR at 1 percent FPR by over an order of magnitude compared to strong baseline attacks. A detailed sensitivity analysis reveals that deeper layers and short, low-learning-rate optimization produce the strongest leakage, and that improvements concentrate in gradient-based detectors. LeakBoost thus offers a modular and computationally efficient way to assess privacy risks in white-box settings, advancing the study of dynamic membership inference. 

---
# Mitigating Hallucination in Financial Retrieval-Augmented Generation via Fine-Grained Knowledge Verification 

**Authors**: Taoye Yin, Haoyuan Hu, Yaxin Fan, Xinhao Chen, Xinya Wu, Kai Deng, Kezun Zhang, Feng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05723)  

**Abstract**: In financial Retrieval-Augmented Generation (RAG) systems, models frequently rely on retrieved documents to generate accurate responses due to the time-sensitive nature of the financial domain. While retrieved documents help address knowledge gaps, model-generated responses still suffer from hallucinations that contradict the retrieved information. To mitigate this inconsistency, we propose a Reinforcement Learning framework enhanced with Fine-grained Knowledge Verification (RLFKV). Our method decomposes financial responses into atomic knowledge units and assesses the correctness of each unit to compute the fine-grained faithful reward. This reward offers more precise optimization signals, thereby improving alignment with the retrieved documents. Additionally, to prevent reward hacking (e.g., overly concise replies), we incorporate an informativeness reward that encourages the policy model to retain at least as many knowledge units as the base model. Experiments conducted on the public Financial Data Description (FDD) task and our newly proposed FDD-ANT dataset demonstrate consistent improvements, confirming the effectiveness of our approach. 

---
# Anchored Policy Optimization: Mitigating Exploration Collapse Via Support-Constrained Rectification 

**Authors**: Tianyi Wang, Long Li, Hongcan Guo, Yibiao Chen, Yixia Li, Yong Wang, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.05717)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is increasingly viewed as a tree pruning mechanism. However, we identify a systemic pathology termed Recursive Space Contraction (RSC), an irreversible collapse driven by the combined dynamics of positive sharpening and negative squeezing, where the sampling probability of valid alternatives vanishes. While Kullback-Leibler (KL) regularization aims to mitigate this, it imposes a rigid Shape Matching constraint that forces the policy to mimic the reference model's full density, creating a gradient conflict with the sharpening required for correctness. We propose Anchored Policy Optimization (APO), shifting the paradigm from global Shape Matching to Support Coverage. By defining a Safe Manifold based on the reference model's high-confidence support, APO permits aggressive sharpening for efficiency while selectively invoking a restorative force during error correction to prevent collapse. We theoretically derive that APO serves as a gradient-aligned mechanism to maximize support coverage, enabling an Elastic Recovery that re-inflates valid branches. Empirical evaluations on mathematical benchmarks demonstrate that APO breaks the accuracy-diversity trade-off, significantly improving Pass@1 while restoring the Pass@K diversity typically lost by standard policy gradient methods. 

---
# Nonlinearity as Rank: Generative Low-Rank Adapter with Radial Basis Functions 

**Authors**: Yihao Ouyang, Shiwei Li, Haozhao Wang, Xiandi Luo, Zhuoqi Hu, Yuetong Song, Qiyu Qin, Yichen Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05709)  

**Abstract**: Low-rank adaptation (LoRA) approximates the update of a pretrained weight matrix using the product of two low-rank matrices. However, standard LoRA follows an explicit-rank paradigm, where increasing model capacity requires adding more rows or columns (i.e., basis vectors) to the low-rank matrices, leading to substantial parameter growth. In this paper, we find that these basis vectors exhibit significant parameter redundancy and can be compactly represented by lightweight nonlinear functions. Therefore, we propose Generative Low-Rank Adapter (GenLoRA), which replaces explicit basis vector storage with nonlinear basis vector generation. Specifically, GenLoRA maintains a latent vector for each low-rank matrix and employs a set of lightweight radial basis functions (RBFs) to synthesize the basis vectors. Each RBF requires far fewer parameters than an explicit basis vector, enabling higher parameter efficiency in GenLoRA. Extensive experiments across multiple datasets and architectures show that GenLoRA attains higher effective LoRA ranks under smaller parameter budgets, resulting in superior fine-tuning performance. The code is available at this https URL. 

---
# Determining Energy Efficiency Sweet Spots in Production LLM Inference 

**Authors**: Hiari Pizzini Cavagna, Andrea Proia, Giacomo Madella, Giovanni B. Esposito, Francesco Antici, Daniele Cesarini, Zeynep Kiziltan, Andrea Bartolini  

**Link**: [PDF](https://arxiv.org/pdf/2602.05695)  

**Abstract**: Large Language Models (LLMs) inference is central in modern AI applications, making it critical to understand their energy footprint. Existing approaches typically estimate energy consumption through simple linear functions of input and output sequence lengths, yet our observations reveal clear Energy Efficiency regimes: peak efficiency occurs with short-to-moderate inputs and medium-length outputs, while efficiency drops sharply for long inputs or very short outputs, indicating a non-linear dependency. In this work, we propose an analytical model derived from the computational and memory-access complexity of the Transformer architecture, capable of accurately characterizing the efficiency curve as a function of input and output lengths. To assess its accuracy, we evaluate energy consumption using TensorRT-LLM on NVIDIA H100 GPUs across a diverse set of LLMs ranging from 1B to 9B parameters, including OPT, LLaMA, Gemma, Falcon, Qwen2, and Granite, tested over input and output lengths from 64 to 4096 tokens, achieving a mean MAPE of 1.79%. Our results show that aligning sequence lengths with these efficiency "Sweet Spots" can substantially reduce energy usage, supporting informed truncation, summarization, and adaptive generation strategies in production systems. 

---
# Graph-based Agent Memory: Taxonomy, Techniques, and Applications 

**Authors**: Chang Yang, Chuang Zhou, Yilin Xiao, Su Dong, Luyao Zhuang, Yujing Zhang, Zhu Wang, Zijin Hong, Zheng Yuan, Zhishang Xiang, Shengyuan Chen, Huachi Zhou, Qinggang Zhang, Ninghao Liu, Jinsong Su, Xinrun Wang, Yi Chang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05665)  

**Abstract**: Memory emerges as the core module in the Large Language Model (LLM)-based agents for long-horizon complex tasks (e.g., multi-turn dialogue, game playing, scientific discovery), where memory can enable knowledge accumulation, iterative reasoning and self-evolution. Among diverse paradigms, graph stands out as a powerful structure for agent memory due to the intrinsic capabilities to model relational dependencies, organize hierarchical information, and support efficient retrieval. This survey presents a comprehensive review of agent memory from the graph-based perspective. First, we introduce a taxonomy of agent memory, including short-term vs. long-term memory, knowledge vs. experience memory, non-structural vs. structural memory, with an implementation view of graph-based memory. Second, according to the life cycle of agent memory, we systematically analyze the key techniques in graph-based agent memory, covering memory extraction for transforming the data into the contents, storage for organizing the data efficiently, retrieval for retrieving the relevant contents from memory to support reasoning, and evolution for updating the contents in the memory. Third, we summarize the open-sourced libraries and benchmarks that support the development and evaluation of self-evolving agent memory. We also explore diverse application scenarios. Finally, we identify critical challenges and future research directions. This survey aims to offer actionable insights to advance the development of more efficient and reliable graph-based agent memory systems. All the related resources, including research papers, open-source data, and projects, are collected for the community in this https URL. 

---
# Generative Ontology: When Structured Knowledge Learns to Create 

**Authors**: Benny Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2602.05636)  

**Abstract**: Traditional ontologies excel at describing domain structure but cannot generate novel artifacts. Large language models generate fluently but produce outputs that lack structural validity, hallucinating mechanisms without components, goals without end conditions. We introduce Generative Ontology, a framework that synthesizes these complementary strengths: ontology provides the grammar; the LLM provides the creativity.
Generative Ontology encodes domain knowledge as executable Pydantic schemas that constrain LLM generation via DSPy signatures. A multi-agent pipeline assigns specialized roles to different ontology domains: a Mechanics Architect designs game systems, a Theme Weaver integrates narrative, a Balance Critic identifies exploits. Each agent carrying a professional "anxiety" that prevents shallow, agreeable outputs. Retrieval-augmented generation grounds novel designs in precedents from existing exemplars, while iterative validation ensures coherence between mechanisms and components.
We demonstrate the framework through GameGrammar, a system for generating complete tabletop game designs. Given a thematic prompt ("bioluminescent fungi competing in a cave ecosystem"), the pipeline produces structurally complete, playable game specifications with mechanisms, components, victory conditions, and setup instructions. These outputs satisfy ontological constraints while remaining genuinely creative.
The pattern generalizes beyond games. Any domain with expert vocabulary, validity constraints, and accumulated exemplars (music composition, software architecture, culinary arts) is a candidate for Generative Ontology. We argue that constraints do not limit creativity but enable it: just as grammar makes poetry possible, ontology makes structured generation possible. 

---
# Reactive Knowledge Representation and Asynchronous Reasoning 

**Authors**: Simon Kohaut, Benedict Flade, Julian Eggert, Kristian Kersting, Devendra Singh Dhami  

**Link**: [PDF](https://arxiv.org/pdf/2602.05625)  

**Abstract**: Exact inference in complex probabilistic models often incurs prohibitive computational costs. This challenge is particularly acute for autonomous agents in dynamic environments that require frequent, real-time belief updates. Existing methods are often inefficient for ongoing reasoning, as they re-evaluate the entire model upon any change, failing to exploit that real-world information streams have heterogeneous update rates. To address this, we approach the problem from a reactive, asynchronous, probabilistic reasoning perspective. We first introduce Resin (Reactive Signal Inference), a probabilistic programming language that merges probabilistic logic with reactive programming. Furthermore, to provide efficient and exact semantics for Resin, we propose Reactive Circuits (RCs). Formulated as a meta-structure over Algebraic Circuits and asynchronous data streams, RCs are time-dynamic Directed Acyclic Graphs that autonomously adapt themselves based on the volatility of input signals. In high-fidelity drone swarm simulations, our approach achieves several orders of magnitude of speedup over frequency-agnostic inference. We demonstrate that RCs' structural adaptations successfully capture environmental dynamics, significantly reducing latency and facilitating reactive real-time reasoning. By partitioning computations based on the estimated Frequency of Change in the asynchronous inputs, large inference tasks can be decomposed into individually memoized sub-problems. This ensures that only the specific components of a model affected by new information are re-evaluated, drastically reducing redundant computation in streaming contexts. 

---
# BhashaSetu: Cross-Lingual Knowledge Transfer from High-Resource to Extreme Low-Resource Languages 

**Authors**: Subhadip Maji, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2602.05599)  

**Abstract**: Despite remarkable advances in natural language processing, developing effective systems for low-resource languages remains a formidable challenge, with performances typically lagging far behind high-resource counterparts due to data scarcity and insufficient linguistic resources. Cross-lingual knowledge transfer has emerged as a promising approach to address this challenge by leveraging resources from high-resource languages. In this paper, we investigate methods for transferring linguistic knowledge from high-resource languages to low-resource languages, where the number of labeled training instances is in hundreds. We focus on sentence-level and word-level tasks. We introduce a novel method, GETR (Graph-Enhanced Token Representation) for cross-lingual knowledge transfer along with two adopted baselines (a) augmentation in hidden layers and (b) token embedding transfer through token translation. Experimental results demonstrate that our GNN-based approach significantly outperforms existing multilingual and cross-lingual baseline methods, achieving 13 percentage point improvements on truly low-resource languages (Mizo, Khasi) for POS tagging, and 20 and 27 percentage point improvements in macro-F1 on simulated low-resource languages (Marathi, Bangla, Malayalam) across sentiment classification and NER tasks respectively. We also present a detailed analysis of the transfer mechanisms and identify key factors that contribute to successful knowledge transfer in this linguistic context. 

---
# Emulating Aggregate Human Choice Behavior and Biases with GPT Conversational Agents 

**Authors**: Stephen Pilli, Vivek Nallur  

**Link**: [PDF](https://arxiv.org/pdf/2602.05597)  

**Abstract**: Cognitive biases often shape human decisions. While large language models (LLMs) have been shown to reproduce well-known biases, a more critical question is whether LLMs can predict biases at the individual level and emulate the dynamics of biased human behavior when contextual factors, such as cognitive load, interact with these biases. We adapted three well-established decision scenarios into a conversational setting and conducted a human experiment (N=1100). Participants engaged with a chatbot that facilitates decision-making through simple or complex dialogues. Results revealed robust biases. To evaluate how LLMs emulate human decision-making under similar interactive conditions, we used participant demographics and dialogue transcripts to simulate these conditions with LLMs based on GPT-4 and GPT-5. The LLMs reproduced human biases with precision. We found notable differences between models in how they aligned human behavior. This has important implications for designing and evaluating adaptive, bias-aware LLM-based AI systems in interactive contexts. 

---
# TangramSR: Can Vision-Language Models Reason in Continuous Geometric Space? 

**Authors**: Yikun Zong, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2602.05570)  

**Abstract**: Humans excel at spatial reasoning tasks like Tangram puzzle assembly through cognitive processes involving mental rotation, iterative refinement, and visual feedback. Inspired by how humans solve Tangram puzzles through trial-and-error, observation, and correction, we design a framework that models these human cognitive mechanisms. However, comprehensive experiments across five representative Vision-Language Models (VLMs) reveal systematic failures in continuous geometric reasoning: average IoU of only 0.41 on single-piece tasks, dropping to 0.23 on two-piece composition, far below human performance where children can complete Tangram tasks successfully. This paper addresses a fundamental challenge in self-improving AI: can models iteratively refine their predictions at test time without parameter updates? We introduce a test-time self-refinement framework that combines in-context learning (ICL) with reward-guided feedback loops, inspired by human cognitive processes. Our training-free verifier-refiner agent applies recursive refinement loops that iteratively self-refine predictions based on geometric consistency feedback, achieving IoU improvements from 0.63 to 0.932 on medium-triangle cases without any model retraining. This demonstrates that incorporating human-inspired iterative refinement mechanisms through ICL and reward loops can substantially enhance geometric reasoning in VLMs, moving self-improving AI from promise to practice in continuous spatial domains. Our work is available at this anonymous link this https URL. 

---
# Reasoning-guided Collaborative Filtering with Language Models for Explainable Recommendation 

**Authors**: Fahad Anwaar, Adil Mehmood Khan, Muhammad Khalid, Usman Zia, Kezhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05544)  

**Abstract**: Large Language Models (LLMs) exhibit potential for explainable recommendation systems but overlook collaborative signals, while prevailing methods treat recommendation and explanation as separate tasks, resulting in a memory footprint. We present RGCF-XRec, a hybrid framework that introduces reasoning-guided collaborative filtering (CF) knowledge into a language model to deliver explainable sequential recommendations in a single step. Theoretical grounding and empirical findings reveal that RGCF-XRec offers three key merits over leading CF-aware LLM-based methods: (1) reasoning-guided augmentation of CF knowledge through contextual prompting to discover latent preferences and interpretable reasoning paths; (2) an efficient scoring mechanism based on four dimensions: coherence, completeness, relevance, and consistency to mitigate noisy CF reasoning traces and retain high-quality explanations; (3) a unified representation learning network that encodes collaborative and semantic signals, enabling a structured prompt to condition the LLM for explainable sequential recommendation. RGCF-XRec demonstrates consistent improvements across Amazon datasets, Sports, Toys, and Beauty, comprising 642,503 user-item interactions. It improves HR@10 by 7.38\% in Sports and 4.59\% in Toys, along with ROUGE-L by 8.02\% and 3.49\%, respectively. It reduces the cold warm performance gap, achieving overall gains of 14.5\% in cold-start and 11.9\% in warm start scenarios, and enhances zero-shot HR@5 by 18.54\% in Beauty and 23.16\% in Toys, highlighting effective generalization and robustness. Moreover, RGCF-XRec achieves training efficiency with a lightweight LLaMA 3.2-3B backbone, ensuring scalability for real-world applications. 

---
# Conditional Diffusion Guidance under Hard Constraint: A Stochastic Analysis Approach 

**Authors**: Zhengyi Guo, Wenpin Tang, Renyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05533)  

**Abstract**: We study conditional generation in diffusion models under hard constraints, where generated samples must satisfy prescribed events with probability one. Such constraints arise naturally in safety-critical applications and in rare-event simulation, where soft or reward-based guidance methods offer no guarantee of constraint satisfaction. Building on a probabilistic interpretation of diffusion models, we develop a principled conditional diffusion guidance framework based on Doob's h-transform, martingale representation and quadratic variation process. Specifically, the resulting guided dynamics augment a pretrained diffusion with an explicit drift correction involving the logarithmic gradient of a conditioning function, without modifying the pretrained score network. Leveraging martingale and quadratic-variation identities, we propose two novel off-policy learning algorithms based on a martingale loss and a martingale-covariation loss to estimate h and its gradient using only trajectories from the pretrained model. We provide non-asymptotic guarantees for the resulting conditional sampler in both total variation and Wasserstein distances, explicitly characterizing the impact of score approximation and guidance estimation errors. Numerical experiments demonstrate the effectiveness of the proposed methods in enforcing hard constraints and generating rare-event samples. 

---
# Split Personality Training: Revealing Latent Knowledge Through Alternate Personalities 

**Authors**: Florian Dietz, William Wale, Oscar Gilg, Robert McCarthy, Felix Michalak, Gustavo Ewbank Rodrigues Danon, Miguelito de Guzman, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2602.05532)  

**Abstract**: Detecting misalignment in large language models is challenging because models may learn to conceal misbehavior during training. Standard auditing techniques fall short: black-box methods often cannot distinguish misaligned outputs from benign ones, and mechanistic interpretability does not scale with model capabilities. We introduce Split Personality Training (SPT), which fine-tunes a second ``honest persona'' into LoRA parameters that remain inactive during normal operation. After the main model responds, we activate the LoRA adapter and insert a trigger string, enabling the honest persona to review the response while accessing the main model's latent states. We test our method on the Anthropic Auditing Game Model Organism, a benchmark where Llama-3.3-70B is trained to exploit reward hacks while concealing this behavior. SPT achieves 96% overall accuracy, whereas Anthropic reports near 0% accuracy. The honest persona reveals latent knowledge inaccessible to external observers, such as the fictional biases the compromised model was trained on. 

---
# A Unified Multimodal Framework for Dataset Construction and Model-Based Diagnosis of Ameloblastoma 

**Authors**: Ajo Babu George, Anna Mariam John, Athul Anoop, Balu Bhasuran  

**Link**: [PDF](https://arxiv.org/pdf/2602.05515)  

**Abstract**: Artificial intelligence (AI)-enabled diagnostics in maxillofacial pathology require structured, high-quality multimodal datasets. However, existing resources provide limited ameloblastoma coverage and lack the format consistency needed for direct model training. We present a newly curated multimodal dataset specifically focused on ameloblastoma, integrating annotated radiological, histopathological, and intraoral clinical images with structured data derived from case reports. Natural language processing techniques were employed to extract clinically relevant features from textual reports, while image data underwent domain specific preprocessing and augmentation. Using this dataset, a multimodal deep learning model was developed to classify ameloblastoma variants, assess behavioral patterns such as recurrence risk, and support surgical planning. The model is designed to accept clinical inputs such as presenting complaint, age, and gender during deployment to enhance personalized inference. Quantitative evaluation demonstrated substantial improvements; variant classification accuracy increased from 46.2 percent to 65.9 percent, and abnormal tissue detection F1-score improved from 43.0 percent to 90.3 percent. Benchmarked against resources like MultiCaRe, this work advances patient-specific decision support by providing both a robust dataset and an adaptable multimodal AI framework. 

---
# SDFP: Speculative Decoding with FIT-Pruned Models for Training-Free and Plug-and-Play LLM Acceleration 

**Authors**: Hanyu Wei, Zunhai Su, Peng Lu, Chao Li, Spandan Tiwari, Ashish Sirasao, Yuhan Dong  

**Link**: [PDF](https://arxiv.org/pdf/2602.05499)  

**Abstract**: Large language models (LLMs) underpin interactive multimedia applications such as captioning, retrieval, recommendation, and creative content generation, yet their autoregressive decoding incurs substantial latency. Speculative decoding reduces latency using a lightweight draft model, but deployment is often limited by the cost and complexity of acquiring, tuning, and maintaining an effective draft model. Recent approaches usually require auxiliary training or specialization, and even training-free methods incur costly search or optimization. We propose SDFP, a fully training-free and plug-and-play framework that builds the draft model via Fisher Information Trace (FIT)-based layer pruning of a given LLM. Using layer sensitivity as a proxy for output perturbation, SDFP removes low-impact layers to obtain a compact draft while preserving compatibility with the original model for standard speculative verification. SDFP needs no additional training, hyperparameter tuning, or separately maintained drafts, enabling rapid, deployment-friendly draft construction. Across benchmarks, SDFP delivers 1.32x-1.5x decoding speedup without altering the target model's output distribution, supporting low-latency multimedia applications. 

---
# Phi-Former: A Pairwise Hierarchical Approach for Compound-Protein Interactions Prediction 

**Authors**: Zhe Wang, Zijing Liu, Chencheng Xu, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05479)  

**Abstract**: Drug discovery remains time-consuming, labor-intensive, and expensive, often requiring years and substantial investment per drug candidate. Predicting compound-protein interactions (CPIs) is a critical component in this process, enabling the identification of molecular interactions between drug candidates and target proteins. Recent deep learning methods have successfully modeled CPIs at the atomic level, achieving improved efficiency and accuracy over traditional energy-based approaches. However, these models do not always align with chemical realities, as molecular fragments (motifs or functional groups) typically serve as the primary units of biological recognition and binding. In this paper, we propose Phi-former, a pairwise hierarchical interaction representation learning method that addresses this gap by incorporating the biological role of motifs in CPIs. Phi-former represents compounds and proteins hierarchically and employs a pairwise pre-training framework to model interactions systematically across atom-atom, motif-motif, and atom-motif levels, reflecting how biological systems recognize molecular partners. We design intra-level and inter-level learning pipelines that make different interaction levels mutually beneficial. Experimental results demonstrate that Phi-former achieves superior performance on CPI-related tasks. A case study shows that our method accurately identifies specific atoms or motifs activated in CPIs, providing interpretable model explanations. These insights may guide rational drug design and support precision medicine applications. 

---
# ALIVE: Awakening LLM Reasoning via Adversarial Learning and Instructive Verbal Evaluation 

**Authors**: Yiwen Duan, Jing Ye, Xinpei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05472)  

**Abstract**: The quest for expert-level reasoning in Large Language Models (LLMs) has been hampered by a persistent \textit{reward bottleneck}: traditional reinforcement learning (RL) relies on scalar rewards that are \textbf{costly} to scale, \textbf{brittle} across domains, and \textbf{blind} to the underlying logic of a solution. This reliance on external, impoverished signals prevents models from developing a deep, self-contained understanding of reasoning principles. We introduce \textbf{ALIVE} (\emph{Adversarial Learning with Instructive Verbal Evaluation}), a hands-free alignment framework that moves beyond scalar reward optimization toward intrinsic reasoning acquisition. Grounded in the principle of \emph{Cognitive Synergy}, ALIVE unifies problem posing, solving, and judging within a single policy model to internalize the logic of correctness. By coupling adversarial learning with instructive verbal feedback, ALIVE enables models to internalize evaluative criteria directly from raw corpora, effectively transforming external critiques into an endogenous reasoning faculty. Empirical evaluations across mathematical reasoning, code generation, and general logical inference benchmarks demonstrate that ALIVE consistently mitigates reward signal limitations. With identical data and compute, it achieves accuracy gains, markedly improved cross-domain generalization, and higher self-correction rates. These results indicate that the reasoning trinity fosters a self-sustaining trajectory of capability growth, positioning ALIVE as a scalable foundation for general-purpose reasoning alignment without human-in-the-loop supervision. 

---
# Refine and Purify: Orthogonal Basis Optimization with Null-Space Denoising for Conditional Representation Learning 

**Authors**: Jiaquan Wang, Yan Lyu, Chen Li, Yuheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.05464)  

**Abstract**: Conditional representation learning aims to extract criterion-specific features for customized tasks. Recent studies project universal features onto the conditional feature subspace spanned by an LLM-generated text basis to obtain conditional representations. However, such methods face two key limitations: sensitivity to subspace basis and vulnerability to inter-subspace interference. To address these challenges, we propose OD-CRL, a novel framework integrating Adaptive Orthogonal Basis Optimization (AOBO) and Null-Space Denoising Projection (NSDP). Specifically, AOBO constructs orthogonal semantic bases via singular value decomposition with a curvature-based truncation. NSDP suppresses non-target semantic interference by projecting embeddings onto the null space of irrelevant subspaces. Extensive experiments conducted across customized clustering, customized classification, and customized retrieval tasks demonstrate that OD-CRL achieves a new state-of-the-art performance with superior generalization. 

---
# Day-Ahead Electricity Price Forecasting for Volatile Markets Using Foundation Models with Regularization Strategy 

**Authors**: Kritchanat Ponyuenyong, Pengyu Tu, Jia Wei Tan, Wei Soon Cheong, Jamie Ng Suat Ling, Lianlian Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05430)  

**Abstract**: Electricity price forecasting (EPF) is essential for energy markets stakeholders (e.g. grid operators, energy traders, policymakers) but remains challenging due to the inherent volatility and nonlinearity of price signals. Traditional statistical and deep learning (DL) models often struggle to capture complex temporal dependencies and integrate heterogeneous data effectively. While time series foundation models (TSFMs) have shown strong performance in general time series forecasting tasks, such as traffic forecasting and weather forecasting. However, their effectiveness in day-ahead EPF, particularly in volatile markets, remains underexplored. This paper presents a spike regularization strategy and evaluates a wide range of TSFMs, including Tiny Time Mixers (TTMs), MOIRAI, MOMENT, and TimesFM, against traditional statistical and DL models such as Autoregressive Integrated Moving Average (ARIMA), Long-short Term Memory (LSTM), and Convolutional Neural Network - LSTM (CNN-LSTM) using half-hourly wholesale market data with volatile trends in Singapore. Exogenous factors (e.g. weather and calendar variables) are also incorporated into models where applicable. Results demonstrate that TSFMs consistently outperform traditional approaches, achieving up to 37.4% improvement in MAPE across various evaluation settings. The findings offer practical guidance for improving forecast accuracy and decision-making in volatile electricity markets. 

---
# M$^2$-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining 

**Authors**: Rui Lv, Juncheng Mo, Tianyi Chu, Chen Rao, Hongyi Jing, Jiajie Teng, Jiafu Chen, Shiqi Zhang, Liangzi Ding, Shuo Fang, Huaizhong Lin, Ziqiang Dang, Chenguang Ma, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05429)  

**Abstract**: Graphical User Interface (GUI) agent is pivotal to advancing intelligent human-computer interaction paradigms. Constructing powerful GUI agents necessitates the large-scale annotation of high-quality user-behavior trajectory data (i.e., intent-trajectory pairs) for training. However, manual annotation methods and current GUI agent data mining approaches typically face three critical challenges: high construction cost, poor data quality, and low data richness. To address these issues, we propose M$^2$-Miner, the first low-cost and automated mobile GUI agent data-mining framework based on Monte Carlo Tree Search (MCTS). For better data mining efficiency and quality, we present a collaborative multi-agent framework, comprising InferAgent, OrchestraAgent, and JudgeAgent for guidance, acceleration, and evaluation. To further enhance the efficiency of mining and enrich intent diversity, we design an intent recycling strategy to extract extra valuable interaction trajectories. Additionally, a progressive model-in-the-loop training strategy is introduced to improve the success rate of data mining. Extensive experiments have demonstrated that the GUI agent fine-tuned using our mined data achieves state-of-the-art performance on several commonly used mobile GUI benchmarks. Our work will be released to facilitate the community research. 

---
# THOR: Inductive Link Prediction over Hyper-Relational Knowledge Graphs 

**Authors**: Weijian Yu, Yuhuan Lu, Dingqi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05424)  

**Abstract**: Knowledge graphs (KGs) have become a key ingredient supporting a variety of applications. Beyond the traditional triplet representation of facts where a relation connects two entities, modern KGs observe an increasing number of hyper-relational facts, where an arbitrary number of qualifiers associated with a triplet provide auxiliary information to further describe the rich semantics of the triplet, which can effectively boost the reasoning performance in link prediction tasks. However, existing link prediction techniques over such hyper-relational KGs (HKGs) mostly focus on a transductive setting, where KG embedding models are learned from the specific vocabulary of a given KG and subsequently can only make predictions within the same vocabulary, limiting their generalizability to previously unseen vocabularies. Against this background, we propose THOR, an inducTive link prediction technique for Hyper-relational knOwledge gRaphs. Specifically, we first introduce both relation and entity foundation graphs, modeling their fundamental inter- and intra-fact interactions in HKGs, which are agnostic to any specific relations and entities. Afterward, THOR is designed to learn from the two foundation graphs with two parallel graph encoders followed by a transformer decoder, which supports efficient masked training and fully-inductive inference. We conduct a thorough evaluation of THOR in hyper-relational link prediction tasks on 12 datasets with different settings. Results show that THOR outperforms a sizable collection of baselines, yielding 66.1%, 55.9%, and 20.4% improvement over the best-performing rule-based, semi-inductive, and fully-inductive techniques, respectively. A series of ablation studies also reveals our key design factors capturing the structural invariance transferable across HKGs for inductive tasks. 

---
# H-AdminSim: A Multi-Agent Simulator for Realistic Hospital Administrative Workflows with FHIR Integration 

**Authors**: Jun-Min Lee, Meong Hi Son, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05407)  

**Abstract**: Hospital administration departments handle a wide range of operational tasks and, in large hospitals, process over 10,000 requests per day, driving growing interest in LLM-based automation. However, prior work has focused primarily on patient--physician interactions or isolated administrative subtasks, failing to capture the complexity of real administrative workflows. To address this gap, we propose H-AdminSim, a comprehensive end-to-end simulation framework that combines realistic data generation with multi-agent-based simulation of hospital administrative workflows. These tasks are quantitatively evaluated using detailed rubrics, enabling systematic comparison of LLMs. Through FHIR integration, H-AdminSim provides a unified and interoperable environment for testing administrative workflows across heterogeneous hospital settings, serving as a standardized testbed for assessing the feasibility and performance of LLM-driven administrative automation. 

---
# Advancing Opinion Dynamics Modeling with Neural Diffusion-Convection-Reaction Equation 

**Authors**: Chenghua Gong, Yihang Jiang, Hao Li, Rui Sun, Juyuan Zhang, Tianjun Gu, Liming Pan, Linyuan Lü  

**Link**: [PDF](https://arxiv.org/pdf/2602.05403)  

**Abstract**: Advanced opinion dynamics modeling is vital for deciphering social behavior, emphasizing its role in mitigating polarization and securing cyberspace. To synergize mechanistic interpretability with data-driven flexibility, recent studies have explored the integration of Physics-Informed Neural Networks (PINNs) for opinion modeling. Despite this promise, existing methods are tailored to incomplete priors, lacking a comprehensive physical system to integrate dynamics from local, global, and endogenous levels. Moreover, penalty-based constraints adopted in existing methods struggle to deeply encode physical priors, leading to optimization pathologies and discrepancy between latent representations and physical transparency. To this end, we offer a physical view to interpret opinion dynamics via Diffusion-Convection-Reaction (DCR) system inspired by interacting particle theory. Building upon the Neural ODEs, we define the neural opinion dynamics to coordinate neural networks with physical priors, and further present the OPINN, a physics-informed neural framework for opinion dynamics modeling. Evaluated on real-world and synthetic datasets, OPINN achieves state-of-the-art performance in opinion evolution forecasting, offering a promising paradigm for the nexus of cyber, physical, and social systems. 

---
# Clinical Validation of Medical-based Large Language Model Chatbots on Ophthalmic Patient Queries with LLM-based Evaluation 

**Authors**: Ting Fang Tan, Kabilan Elangovan, Andreas Pollreisz, Kevin Bryan Dy, Wei Yan Ng, Joy Le Yi Wong, Jin Liyuan, Chrystie Quek Wan Ning, Ashley Shuen Ying Hong, Arun James Thirunavukarasu, Shelley Yin-His Chang, Jie Yao, Dylan Hong, Wang Zhaoran, Amrita Gupta, Daniel SW Ting  

**Link**: [PDF](https://arxiv.org/pdf/2602.05381)  

**Abstract**: Domain specific large language models are increasingly used to support patient education, triage, and clinical decision making in ophthalmology, making rigorous evaluation essential to ensure safety and accuracy. This study evaluated four small medical LLMs Meerkat-7B, BioMistral-7B, OpenBioLLM-8B, and MedLLaMA3-v20 in answering ophthalmology related patient queries and assessed the feasibility of LLM based evaluation against clinician grading. In this cross sectional study, 180 ophthalmology patient queries were answered by each model, generating 2160 responses. Models were selected for parameter sizes under 10 billion to enable resource efficient deployment. Responses were evaluated by three ophthalmologists of differing seniority and by GPT-4-Turbo using the S.C.O.R.E. framework assessing safety, consensus and context, objectivity, reproducibility, and explainability, with ratings assigned on a five point Likert scale. Agreement between LLM and clinician grading was assessed using Spearman rank correlation, Kendall tau statistics, and kernel density estimate analyses. Meerkat-7B achieved the highest performance with mean scores of 3.44 from Senior Consultants, 4.08 from Consultants, and 4.18 from Residents. MedLLaMA3-v20 performed poorest, with 25.5 percent of responses containing hallucinations or clinically misleading content, including fabricated terminology. GPT-4-Turbo grading showed strong alignment with clinician assessments overall, with Spearman rho of 0.80 and Kendall tau of 0.67, though Senior Consultants graded more conservatively. Overall, medical LLMs demonstrated potential for safe ophthalmic question answering, but gaps remained in clinical depth and consensus, supporting the feasibility of LLM based evaluation for large scale benchmarking and the need for hybrid automated and clinician review frameworks to guide safe clinical deployment. 

---
# RaBiT: Residual-Aware Binarization Training for Accurate and Efficient LLMs 

**Authors**: Youngcheon You, Banseok Lee, Minseop Choi, Seonyoung Kim, Hyochan Chong, Changdong Kim, Youngmin Kim, Dongkyu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.05367)  

**Abstract**: Efficient deployment of large language models (LLMs) requires extreme quantization, forcing a critical trade-off between low-bit efficiency and performance. Residual binarization enables hardware-friendly, matmul-free inference by stacking binary ($\pm$1) layers, but is plagued by pathological feature co-adaptation. We identify a key failure mode, which we term inter-path adaptation: during quantization-aware training (QAT), parallel residual binary paths learn redundant features, degrading the error-compensation structure and limiting the expressive capacity of the model. While prior work relies on heuristic workarounds (e.g., path freezing) that constrain the solution space, we propose RaBiT, a novel quantization framework that resolves co-adaptation by algorithmically enforcing a residual hierarchy. Its core mechanism sequentially derives each binary path from a single shared full-precision weight, which ensures that every path corrects the error of the preceding one. This process is stabilized by a robust initialization that prioritizes functional preservation over mere weight approximation. RaBiT redefines the 2-bit accuracy-efficiency frontier: it achieves state-of-the-art performance, rivals even hardware-intensive Vector Quantization (VQ) methods, and delivers a $4.49\times$ inference speed-up over full-precision models on an RTX 4090. 

---
# PATHWAYS: Evaluating Investigation and Context Discovery in AI Web Agents 

**Authors**: Shifat E. Arman, Syed Nazmus Sakib, Tapodhir Karmakar Taton, Nafiul Haque, Shahrear Bin Amin  

**Link**: [PDF](https://arxiv.org/pdf/2602.05354)  

**Abstract**: We introduce PATHWAYS, a benchmark of 250 multi-step decision tasks that test whether web-based agents can discover and correctly use hidden contextual information. Across both closed and open models, agents typically navigate to relevant pages but retrieve decisive hidden evidence in only a small fraction of cases. When tasks require overturning misleading surface-level signals, performance drops sharply to near chance accuracy. Agents frequently hallucinate investigative reasoning by claiming to rely on evidence they never accessed. Even when correct context is discovered, agents often fail to integrate it into their final decision. Providing more explicit instructions improves context discovery but often reduces overall accuracy, revealing a tradeoff between procedural compliance and effective judgement. Together, these results show that current web agent architectures lack reliable mechanisms for adaptive investigation, evidence integration, and judgement override. 

---
# AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction 

**Authors**: Ruijie Shi, Houbin Zhang, Yuecheng Han, Yuheng Wang, Jingru Fan, Runde Yang, Yufan Dang, Huatao Li, Dewen Liu, Yuan Cheng, Chen Qian  

**Link**: [PDF](https://arxiv.org/pdf/2602.05353)  

**Abstract**: Large Language Models have shown strong capabilities in complex problem solving, yet many agentic systems remain difficult to interpret and control due to opaque internal workflows. While some frameworks offer explicit architectures for collaboration, many deployed agentic systems operate as black boxes to users. We address this by introducing Agentic Workflow Reconstruction (AWR), a new task aiming to synthesize an explicit, interpretable stand-in workflow that approximates a black-box system using only input--output access. We propose AgentXRay, a search-based framework that formulates AWR as a combinatorial optimization problem over discrete agent roles and tool invocations in a chain-structured workflow space. Unlike model distillation, AgentXRay produces editable white-box workflows that match target outputs under an observable, output-based proxy metric, without accessing model parameters. To navigate the vast search space, AgentXRay employs Monte Carlo Tree Search enhanced by a scoring-based Red-Black Pruning mechanism, which dynamically integrates proxy quality with search depth. Experiments across diverse domains demonstrate that AgentXRay achieves higher proxy similarity and reduces token consumption compared to unpruned search, enabling deeper workflow exploration under fixed iteration budgets. 

---
# ProAct: Agentic Lookahead in Interactive Environments 

**Authors**: Yangbin Yu, Mingyu Yang, Junyou Li, Yiming Gao, Feiyu Liu, Yijun Yang, Zichuan Lin, Jiafei Lyu, Yicheng Liu, Zhicong Lu, Deheng Ye, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05327)  

**Abstract**: Existing Large Language Model (LLM) agents struggle in interactive environments requiring long-horizon planning, primarily due to compounding errors when simulating future states. To address this, we propose ProAct, a framework that enables agents to internalize accurate lookahead reasoning through a two-stage training paradigm. First, we introduce Grounded LookAhead Distillation (GLAD), where the agent undergoes supervised fine-tuning on trajectories derived from environment-based search. By compressing complex search trees into concise, causal reasoning chains, the agent learns the logic of foresight without the computational overhead of inference-time search. Second, to further refine decision accuracy, we propose the Monte-Carlo Critic (MC-Critic), a plug-and-play auxiliary value estimator designed to enhance policy-gradient algorithms like PPO and GRPO. By leveraging lightweight environment rollouts to calibrate value estimates, MC-Critic provides a low-variance signal that facilitates stable policy optimization without relying on expensive model-based value approximation. Experiments on both stochastic (e.g., 2048) and deterministic (e.g., Sokoban) environments demonstrate that ProAct significantly improves planning accuracy. Notably, a 4B parameter model trained with ProAct outperforms all open-source baselines and rivals state-of-the-art closed-source models, while demonstrating robust generalization to unseen environments. The codes and models are available at this https URL 

---
# PieArena: Frontier Language Agents Achieve MBA-Level Negotiation Performance and Reveal Novel Behavioral Differences 

**Authors**: Chris Zhu, Sasha Cui, Will Sanok Dufallo, Runzhi Jin, Zhen Xu, Linjun Zhang, Daylian Cain  

**Link**: [PDF](https://arxiv.org/pdf/2602.05302)  

**Abstract**: We present an in-depth evaluation of LLMs' ability to negotiate, a central business task that requires strategic reasoning, theory of mind, and economic value creation. To do so, we introduce PieArena, a large-scale negotiation benchmark grounded in multi-agent interactions over realistic scenarios drawn from an MBA negotiation course at an elite business school. We find systematic evidence of AGI-level performance in which a representative frontier agent (GPT-5) matches or outperforms trained business-school students, despite a semester of general negotiation instruction and targeted coaching immediately prior to the task. We further study the effects of joint-intentionality agentic scaffolding and find asymmetric gains, with large improvements for mid- and lower-tier LMs and diminishing returns for frontier LMs. Beyond deal outcomes, PieArena provides a multi-dimensional negotiation behavioral profile, revealing novel cross-model heterogeneity, masked by deal-outcome-only benchmarks, in deception, computation accuracy, instruction compliance, and perceived reputation. Overall, our results suggest that frontier language agents are already intellectually and psychologically capable of deployment in high-stakes economic settings, but deficiencies in robustness and trustworthiness remain open challenges. 

---
# Aspect-Aware MOOC Recommendation in a Heterogeneous Network 

**Authors**: Seongyeub Chu, Jongwoo Kim, Mun Yong Yi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05297)  

**Abstract**: MOOC recommendation systems have received increasing attention to help learners navigate and select preferred learning content. Traditional methods such as collaborative filtering and content-based filtering suffer from data sparsity and over-specialization. To alleviate these limitations, graph-based approaches have been proposed; however, they still rely heavily on manually predefined metapaths, which often capture only superficial structural relationships and impose substantial burdens on domain experts as well as significant engineering costs. To overcome these limitations, we propose AMR (Aspect-aware MOOC Recommendation), a novel framework that models path-specific multiple aspects by embedding the semantic content of nodes within each metapath. AMR automatically discovers metapaths through bi-directional walks, derives aspect-aware path representations using a bi-LSTM-based encoder, and incorporates these representations as edge features in the learner-learner and KC-KC subgraphs to achieve fine-grained semantically informed KC recommendations. Extensive experiments on the large-scale MOOCCube and PEEK datasets show that AMR consistently outperforms state-of-the-art graph neural network baselines across key metrics such as HR@K and nDCG@K. Further analysis confirms that AMR effectively captures rich path-specific aspect information, allowing more accurate recommendations than those methods that rely solely on predefined metapaths. The code will be available upon accepted. 

---
# Position: Universal Time Series Foundation Models Rest on a Category Error 

**Authors**: Xilin Dai, Wanxu Cai, Zhijian Xu, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05287)  

**Abstract**: This position paper argues that the pursuit of "Universal Foundation Models for Time Series" rests on a fundamental category error, mistaking a structural Container for a semantic Modality. We contend that because time series hold incompatible generative processes (e.g., finance vs. fluid dynamics), monolithic models degenerate into expensive "Generic Filters" that fail to generalize under distributional drift. To address this, we introduce the "Autoregressive Blindness Bound," a theoretical limit proving that history-only models cannot predict intervention-driven regime shifts. We advocate replacing universality with a Causal Control Agent paradigm, where an agent leverages external context to orchestrate a hierarchy of specialized solvers, from frozen domain experts to lightweight Just-in-Time adaptors. We conclude by calling for a shift in benchmarks from "Zero-Shot Accuracy" to "Drift Adaptation Speed" to prioritize robust, control-theoretic systems. 

---
# Hallucination-Resistant Security Planning with a Large Language Model 

**Authors**: Kim Hammar, Tansu Alpcan, Emil Lupu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05279)  

**Abstract**: Large language models (LLMs) are promising tools for supporting security management tasks, such as incident response planning. However, their unreliability and tendency to hallucinate remain significant challenges. In this paper, we address these challenges by introducing a principled framework for using an LLM as decision support in security management. Our framework integrates the LLM in an iterative loop where it generates candidate actions that are checked for consistency with system constraints and lookahead predictions. When consistency is low, we abstain from the generated actions and instead collect external feedback, e.g., by evaluating actions in a digital twin. This feedback is then used to refine the candidate actions through in-context learning (ICL). We prove that this design allows to control the hallucination risk by tuning the consistency threshold. Moreover, we establish a bound on the regret of ICL under certain assumptions. To evaluate our framework, we apply it to an incident response use case where the goal is to generate a response and recovery plan based on system logs. Experiments on four public datasets show that our framework reduces recovery times by up to 30% compared to frontier LLMs. 

---
# Beyond Cosine Similarity 

**Authors**: Xinbo Ai  

**Link**: [PDF](https://arxiv.org/pdf/2602.05266)  

**Abstract**: Cosine similarity, the standard metric for measuring semantic similarity in vector spaces, is mathematically grounded in the Cauchy-Schwarz inequality, which inherently limits it to capturing linear relationships--a constraint that fails to model the complex, nonlinear structures of real-world semantic spaces. We advance this theoretical underpinning by deriving a tighter upper bound for the dot product than the classical Cauchy-Schwarz bound. This new bound leads directly to recos, a similarity metric that normalizes the dot product by the sorted vector components. recos relaxes the condition for perfect similarity from strict linear dependence to ordinal concordance, thereby capturing a broader class of relationships. Extensive experiments across 11 embedding models--spanning static, contextualized, and universal types--demonstrate that recos consistently outperforms traditional cosine similarity, achieving higher correlation with human judgments on standard Semantic Textual Similarity (STS) benchmarks. Our work establishes recos as a mathematically principled and empirically superior alternative, offering enhanced accuracy for semantic analysis in complex embedding spaces. 

---
# Automatic Cognitive Task Generation for In-Situ Evaluation of Embodied Agents 

**Authors**: Xinyi He, Ying Yang, Chuanjian Fu, Sihan Guo, Songchun Zhu, Lifeng Fan, Zhenliang Zhang, Yujia Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.05249)  

**Abstract**: As general intelligent agents are poised for widespread deployment in diverse households, evaluation tailored to each unique unseen 3D environment has become a critical prerequisite. However, existing benchmarks suffer from severe data contamination and a lack of scene specificity, inadequate for assessing agent capabilities in unseen settings. To address this, we propose a dynamic in-situ task generation method for unseen environments inspired by human cognition. We define tasks through a structured graph representation and construct a two-stage interaction-evolution task generation system for embodied agents (TEA). In the interaction stage, the agent actively interacts with the environment, creating a loop between task execution and generation that allows for continuous task generation. In the evolution stage, task graph modeling allows us to recombine and reuse existing tasks to generate new ones without external data. Experiments across 10 unseen scenes demonstrate that TEA automatically generated 87,876 tasks in two cycles, which human verification confirmed to be physically reasonable and encompassing essential daily cognitive capabilities. Benchmarking SOTA models against humans on our in-situ tasks reveals that models, despite excelling on public benchmarks, perform surprisingly poorly on basic perception tasks, severely lack 3D interaction awareness and show high sensitivity to task types in reasoning. These sobering findings highlight the necessity of in-situ evaluation before deploying agents into real-world human environments. 

---
# Explainable AI: A Combined XAI Framework for Explaining Brain Tumour Detection Models 

**Authors**: Patrick McGonagle, William Farrelly, Kevin Curran  

**Link**: [PDF](https://arxiv.org/pdf/2602.05240)  

**Abstract**: This study explores the integration of multiple Explainable AI (XAI) techniques to enhance the interpretability of deep learning models for brain tumour detection. A custom Convolutional Neural Network (CNN) was developed and trained on the BraTS 2021 dataset, achieving 91.24% accuracy in distinguishing between tumour and non-tumour regions. This research combines Gradient-weighted Class Activation Mapping (GRAD-CAM), Layer-wise Relevance Propagation (LRP) and SHapley Additive exPlanations (SHAP) to provide comprehensive insights into the model's decision-making process. This multi-technique approach successfully identified both full and partial tumours, offering layered explanations ranging from broad regions of interest to pixel-level details. GRAD-CAM highlighted important spatial regions, LRP provided detailed pixel-level relevance and SHAP quantified feature contributions. The integrated approach effectively explained model predictions, including cases with partial tumour visibility thus showing superior explanatory power compared to individual XAI methods. This research enhances transparency and trust in AI-driven medical imaging analysis by offering a more comprehensive perspective on the model's reasoning. The study demonstrates the potential of integrated XAI techniques in improving the reliability and interpretability of AI systems in healthcare, particularly for critical tasks like brain tumour detection. 

---
# Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink 

**Authors**: Guozhi Liu, Weiwei Lin, Tiansheng Huang, Ruichao Mo, Qi Mu, Xiumin Wang, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.05228)  

**Abstract**: Harmful fine-tuning can invalidate safety alignment of large language models, exposing significant safety risks. In this paper, we utilize the attention sink mechanism to mitigate harmful fine-tuning. Specifically, we first measure a statistic named \emph{sink divergence} for each attention head and observe that \emph{different attention heads exhibit two different signs of sink divergence}. To understand its safety implications, we conduct experiments and find that the number of attention heads of positive sink divergence increases along with the increase of the model's harmfulness when undergoing harmful fine-tuning. Based on this finding, we propose a separable sink divergence hypothesis -- \emph{attention heads associating with learning harmful patterns during fine-tuning are separable by their sign of sink divergence}. Based on the hypothesis, we propose a fine-tuning-stage defense, dubbed Surgery. Surgery utilizes a regularizer for sink divergence suppression, which steers attention heads toward the negative sink divergence group, thereby reducing the model's tendency to learn and amplify harmful patterns. Extensive experiments demonstrate that Surgery improves defense performance by 5.90\%, 11.25\%, and 9.55\% on the BeaverTails, HarmBench, and SorryBench benchmarks, respectively. Source code is available on this https URL. 

---
# Traceable Cross-Source RAG for Chinese Tibetan Medicine Question Answering 

**Authors**: Fengxian Chen, Zhilong Tao, Jiaxuan Li, Yunlong Li, Qingguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.05195)  

**Abstract**: Retrieval-augmented generation (RAG) promises grounded question answering, yet domain settings with multiple heterogeneous knowledge bases (KBs) remain challenging. In Chinese Tibetan medicine, encyclopedia entries are often dense and easy to match, which can dominate retrieval even when classics or clinical papers provide more authoritative evidence. We study a practical setting with three KBs (encyclopedia, classics, and clinical papers) and a 500-query benchmark (cutoff $K{=}5$) covering both single-KB and cross-KB questions. We propose two complementary methods to improve traceability, reduce hallucinations, and enable cross-KB verification. First, DAKS performs KB routing and budgeted retrieval to mitigate density-driven bias and to prioritize authoritative sources when appropriate. Second, we use an alignment graph to guide evidence fusion and coverage-aware packing, improving cross-KB evidence coverage without relying on naive concatenation. All answers are generated by a lightweight generator, \textsc{openPangu-Embedded-7B}. Experiments show consistent gains in routing quality and cross-KB evidence coverage, with the full system achieving the best CrossEv@5 while maintaining strong faithfulness and citation correctness. 

---
# First Proof 

**Authors**: Mohammed Abouzaid, Andrew J. Blumberg, Martin Hairer, Joe Kileel, Tamara G. Kolda, Paul D. Nelson, Daniel Spielman, Nikhil Srivastava, Rachel Ward, Shmuel Weinberger, Lauren Williams  

**Link**: [PDF](https://arxiv.org/pdf/2602.05192)  

**Abstract**: To assess the ability of current AI systems to correctly answer research-level mathematics questions, we share a set of ten math questions which have arisen naturally in the research process of the authors. The questions had not been shared publicly until now; the answers are known to the authors of the questions but will remain encrypted for a short time. 

---
# HugRAG: Hierarchical Causal Knowledge Graph Design for RAG 

**Authors**: Nengbo Wang, Tuo Liang, Vikash Singh, Chaoda Song, Van Yang, Yu Yin, Jing Ma, Jagdip Singh, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2602.05143)  

**Abstract**: Retrieval augmented generation (RAG) has enhanced large language models by enabling access to external knowledge, with graph-based RAG emerging as a powerful paradigm for structured retrieval and reasoning. However, existing graph-based methods often over-rely on surface-level node matching and lack explicit causal modeling, leading to unfaithful or spurious answers. Prior attempts to incorporate causality are typically limited to local or single-document contexts and also suffer from information isolation that arises from modular graph structures, which hinders scalability and cross-module causal reasoning. To address these challenges, we propose HugRAG, a framework that rethinks knowledge organization for graph-based RAG through causal gating across hierarchical modules. HugRAG explicitly models causal relationships to suppress spurious correlations while enabling scalable reasoning over large-scale knowledge graphs. Extensive experiments demonstrate that HugRAG consistently outperforms competitive graph-based RAG baselines across multiple datasets and evaluation metrics. Our work establishes a principled foundation for structured, scalable, and causally grounded RAG systems. 

---
# CAST-CKT: Chaos-Aware Spatio-Temporal and Cross-City Knowledge Transfer for Traffic Flow Prediction 

**Authors**: Abdul Joseph Fofanah, Lian Wen, David Chen, Alpha Alimamy Kamara, Zhongyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05133)  

**Abstract**: Traffic prediction in data-scarce, cross-city settings is challenging due to complex nonlinear dynamics and domain shifts. Existing methods often fail to capture traffic's inherent chaotic nature for effective few-shot learning. We propose CAST-CKT, a novel Chaos-Aware Spatio-Temporal and Cross-City Knowledge Transfer framework. It employs an efficient chaotic analyser to quantify traffic predictability regimes, driving several key innovations: chaos-aware attention for regime-adaptive temporal modelling; adaptive topology learning for dynamic spatial dependencies; and chaotic consistency-based cross-city alignment for knowledge transfer. The framework also provides horizon-specific predictions with uncertainty quantification. Theoretical analysis shows improved generalisation bounds. Extensive experiments on four benchmarks in cross-city few-shot settings show CAST-CKT outperforms state-of-the-art methods by significant margins in MAE and RMSE, while offering interpretable regime analysis. Code is available at this https URL. 

---
# SocialVeil: Probing Social Intelligence of Language Agents under Communication Barriers 

**Authors**: Keyang Xuan, Pengda Wang, Chongrui Ye, Haofei Yu, Tal August, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2602.05115)  

**Abstract**: Large language models (LLMs) are increasingly evaluated in interactive environments to test their social intelligence. However, existing benchmarks often assume idealized communication between agents, limiting our ability to diagnose whether LLMs can maintain and repair interactions in more realistic, imperfect settings. To close this gap, we present \textsc{SocialVeil}, a social learning environment that can simulate social interaction under cognitive-difference-induced communication barriers. Grounded in a systematic literature review of communication challenges in human interaction, \textsc{SocialVeil} introduces three representative types of such disruption, \emph{semantic vagueness}, \emph{sociocultural mismatch}, and \emph{emotional interference}. We also introduce two barrier-aware evaluation metrics, \emph{unresolved confusion} and \emph{mutual understanding}, to evaluate interaction quality under impaired communication. Experiments across 720 scenarios and four frontier LLMs show that barriers consistently impair performance, with mutual understanding reduced by over 45\% on average, and confusion elevated by nearly 50\%. Human evaluations validate the fidelity of these simulated barriers (ICC$\approx$0.78, Pearson r$\approx$0.80). We further demonstrate that adaptation strategies (Repair Instruction and Interactive learning) only have a modest effect far from barrier-free performance. This work takes a step toward bringing social interaction environments closer to real-world communication, opening opportunities for exploring the social intelligence of LLM agents. 

---
# Democratic Preference Alignment via Sortition-Weighted RLHF 

**Authors**: Suvadip Sana, Jinzhou Wu, Martin T. Wells  

**Link**: [PDF](https://arxiv.org/pdf/2602.05113)  

**Abstract**: Whose values should AI systems learn? Preference based alignment methods like RLHF derive their training signal from human raters, yet these rater pools are typically convenience samples that systematically over represent some demographics and under represent others. We introduce Democratic Preference Optimization, or DemPO, a framework that applies algorithmic sortition, the same mechanism used to construct citizen assemblies, to preference based fine tuning. DemPO offers two training schemes. Hard Panel trains exclusively on preferences from a quota satisfying mini public sampled via sortition. Soft Panel retains all data but reweights each rater by their inclusion probability under the sortition lottery. We prove that Soft Panel weighting recovers the expected Hard Panel objective in closed form. Using a public preference dataset that pairs human judgments with rater demographics and a seventy five clause constitution independently elicited from a representative United States panel, we evaluate Llama models from one billion to eight billion parameters fine tuned under each scheme. Across six aggregation methods, the Hard Panel consistently ranks first and the Soft Panel consistently outperforms the unweighted baseline, with effect sizes growing as model capacity increases. These results demonstrate that enforcing demographic representativeness at the preference collection stage, rather than post hoc correction, yields models whose behavior better reflects values elicited from representative publics. 

---
# Understanding LLM Evaluator Behavior: A Structured Multi-Evaluator Framework for Merchant Risk Assessment 

**Authors**: Liang Wang, Junpeng Wang, Chin-chia Michael Yeh, Yan Zheng, Jiarui Sun, Xiran Fan, Xin Dai, Yujie Fan, Yiwei Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.05110)  

**Abstract**: Large Language Models (LLMs) are increasingly used as evaluators of reasoning quality, yet their reliability and bias in payments-risk settings remain poorly understood. We introduce a structured multi-evaluator framework for assessing LLM reasoning in Merchant Category Code (MCC)-based merchant risk assessment, combining a five-criterion rubric with Monte-Carlo scoring to evaluate rationale quality and evaluator stability. Five frontier LLMs generate and cross-evaluate MCC risk rationales under attributed and anonymized conditions. To establish a judge-independent reference, we introduce a consensus-deviation metric that eliminates circularity by comparing each judge's score to the mean of all other judges, yielding a theoretically grounded measure of self-evaluation and cross-model deviation. Results reveal substantial heterogeneity: GPT-5.1 and Claude 4.5 Sonnet show negative self-evaluation bias (-0.33, -0.31), while Gemini-2.5 Pro and Grok 4 display positive bias (+0.77, +0.71), with bias attenuating by 25.8 percent under anonymization. Evaluation by 26 payment-industry experts shows LLM judges assign scores averaging +0.46 points above human consensus, and that the negative bias of GPT-5.1 and Claude 4.5 Sonnet reflects closer alignment with human judgment. Ground-truth validation using payment-network data shows four models exhibit statistically significant alignment (Spearman rho = 0.56 to 0.77), confirming that the framework captures genuine quality. Overall, the framework provides a replicable basis for evaluating LLM-as-a-judge systems in payment-risk workflows and highlights the need for bias-aware protocols in operational financial settings. 

---
# GAMMS: Graph based Adversarial Multiagent Modeling Simulator 

**Authors**: Rohan Patil, Jai Malegaonkar, Xiao Jiang, Andre Dion, Gaurav S. Sukhatme, Henrik I. Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2602.05105)  

**Abstract**: As intelligent systems and multi-agent coordination become increasingly central to real-world applications, there is a growing need for simulation tools that are both scalable and accessible. Existing high-fidelity simulators, while powerful, are often computationally expensive and ill-suited for rapid prototyping or large-scale agent deployments. We present GAMMS (Graph based Adversarial Multiagent Modeling Simulator), a lightweight yet extensible simulation framework designed to support fast development and evaluation of agent behavior in environments that can be represented as graphs. GAMMS emphasizes five core objectives: scalability, ease of use, integration-first architecture, fast visualization feedback, and real-world grounding. It enables efficient simulation of complex domains such as urban road networks and communication systems, supports integration with external tools (e.g., machine learning libraries, planning solvers), and provides built-in visualization with minimal configuration. GAMMS is agnostic to policy type, supporting heuristic, optimization-based, and learning-based agents, including those using large language models. By lowering the barrier to entry for researchers and enabling high-performance simulations on standard hardware, GAMMS facilitates experimentation and innovation in multi-agent systems, autonomous planning, and adversarial modeling. The framework is open-source and available at this https URL 

---
# Evaluating Robustness and Adaptability in Learning-Based Mission Planning for Active Debris Removal 

**Authors**: Agni Bandyopadhyay, Günther Waxenegger-Wilfing  

**Link**: [PDF](https://arxiv.org/pdf/2602.05091)  

**Abstract**: Autonomous mission planning for Active Debris Removal (ADR) must balance efficiency, adaptability, and strict feasibility constraints on fuel and mission duration. This work compares three planners for the constrained multi-debris rendezvous problem in Low Earth Orbit: a nominal Masked Proximal Policy Optimization (PPO) policy trained under fixed mission parameters, a domain-randomized Masked PPO policy trained across varying mission constraints for improved robustness, and a plain Monte Carlo Tree Search (MCTS) baseline. Evaluations are conducted in a high-fidelity orbital simulation with refueling, realistic transfer dynamics, and randomized debris fields across 300 test cases in nominal, reduced fuel, and reduced mission time scenarios. Results show that nominal PPO achieves top performance when conditions match training but degrades sharply under distributional shift, while domain-randomized PPO exhibits improved adaptability with only moderate loss in nominal performance. MCTS consistently handles constraint changes best due to online replanning but incurs orders-of-magnitude higher computation time. The findings underline a trade-off between the speed of learned policies and the adaptability of search-based methods, and suggest that combining training-time diversity with online planning could be a promising path for future resilient ADR mission planners. 

---
# VERA-MH: Reliability and Validity of an Open-Source AI Safety Evaluation in Mental Health 

**Authors**: Kate H. Bentley, Luca Belli, Adam M. Chekroud, Emily J. Ward, Emily R. Dworkin, Emily Van Ark, Kelly M. Johnston, Will Alexander, Millard Brown, Matt Hawrilenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.05088)  

**Abstract**: Millions now use leading generative AI chatbots for psychological support. Despite the promise related to availability and scale, the single most pressing question in AI for mental health is whether these tools are safe. The Validation of Ethical and Responsible AI in Mental Health (VERA-MH) evaluation was recently proposed to meet the urgent need for an evidence-based automated safety benchmark. This study aimed to examine the clinical validity and reliability of the VERA-MH evaluation for AI safety in suicide risk detection and response. We first simulated a large set of conversations between large language model (LLM)-based users (user-agents) and general-purpose AI chatbots. Licensed mental health clinicians used a rubric (scoring guide) to independently rate the simulated conversations for safe and unsafe chatbot behaviors, as well as user-agent realism. An LLM-based judge used the same scoring rubric to evaluate the same set of simulated conversations. We then compared rating alignment across (a) individual clinicians and (b) clinician consensus and the LLM judge, and (c) examined clinicians' ratings of user-agent realism. Individual clinicians were generally consistent with one another in their safety ratings (chance-corrected inter-rater reliability [IRR]: 0.77), thus establishing a gold-standard clinical reference. The LLM judge was strongly aligned with this clinical consensus (IRR: 0.81) overall and within key conditions. Clinician raters generally perceived the user-agents to be realistic. For the potential mental health benefits of AI chatbots to be realized, attention to safety is paramount. Findings from this human evaluation study support the clinical validity and reliability of VERA-MH: an open-source, fully automated AI safety evaluation for mental health. Further research will address VERA-MH generalizability and robustness. 

---
# Optimizing Mission Planning for Multi-Debris Rendezvous Using Reinforcement Learning with Refueling and Adaptive Collision Avoidance 

**Authors**: Agni Bandyopadhyay, Gunther Waxenegger-Wilfing  

**Link**: [PDF](https://arxiv.org/pdf/2602.05075)  

**Abstract**: As the orbital environment around Earth becomes increasingly crowded with debris, active debris removal (ADR) missions face significant challenges in ensuring safe operations while minimizing the risk of in-orbit collisions. This study presents a reinforcement learning (RL) based framework to enhance adaptive collision avoidance in ADR missions, specifically for multi-debris removal using small satellites. Small satellites are increasingly adopted due to their flexibility, cost effectiveness, and maneuverability, making them well suited for dynamic missions such as ADR.
Building on existing work in multi-debris rendezvous, the framework integrates refueling strategies, efficient mission planning, and adaptive collision avoidance to optimize spacecraft rendezvous operations. The proposed approach employs a masked Proximal Policy Optimization (PPO) algorithm, enabling the RL agent to dynamically adjust maneuvers in response to real-time orbital conditions. Key considerations include fuel efficiency, avoidance of active collision zones, and optimization of dynamic orbital parameters.
The RL agent learns to determine efficient sequences for rendezvousing with multiple debris targets, optimizing fuel usage and mission time while incorporating necessary refueling stops. Simulated ADR scenarios derived from the Iridium 33 debris dataset are used for evaluation, covering diverse orbital configurations and debris distributions to demonstrate robustness and adaptability. Results show that the proposed RL framework reduces collision risk while improving mission efficiency compared to traditional heuristic approaches.
This work provides a scalable solution for planning complex multi-debris ADR missions and is applicable to other multi-target rendezvous problems in autonomous space mission planning. 

---
# Towards Reducible Uncertainty Modeling for Reliable Large Language Model Agents 

**Authors**: Changdae Oh, Seongheon Park, To Eun Kim, Jiatong Li, Wendi Li, Samuel Yeh, Xuefeng Du, Hamed Hassani, Paul Bogdan, Dawn Song, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05073)  

**Abstract**: Uncertainty quantification (UQ) for large language models (LLMs) is a key building block for safety guardrails of daily LLM applications. Yet, even as LLM agents are increasingly deployed in highly complex tasks, most UQ research still centers on single-turn question-answering. We argue that UQ research must shift to realistic settings with interactive agents, and that a new principled framework for agent UQ is needed. This paper presents the first general formulation of agent UQ that subsumes broad classes of existing UQ setups. Under this formulation, we show that prior works implicitly treat LLM UQ as an uncertainty accumulation process, a viewpoint that breaks down for interactive agents in an open world. In contrast, we propose a novel perspective, a conditional uncertainty reduction process, that explicitly models reducible uncertainty over an agent's trajectory by highlighting "interactivity" of actions. From this perspective, we outline a conceptual framework to provide actionable guidance for designing UQ in LLM agent setups. Finally, we conclude with practical implications of the agent UQ in frontier LLM development and domain-specific applications, as well as open remaining problems. 

---
# Evaluating Large Language Models on Solved and Unsolved Problems in Graph Theory: Implications for Computing Education 

**Authors**: Adithya Kulkarni, Mohna Chakraborty, Jay Bagga  

**Link**: [PDF](https://arxiv.org/pdf/2602.05059)  

**Abstract**: Large Language Models are increasingly used by students to explore advanced material in computer science, including graph theory. As these tools become integrated into undergraduate and graduate coursework, it is important to understand how reliably they support mathematically rigorous thinking. This study examines the performance of a LLM on two related graph theoretic problems: a solved problem concerning the gracefulness of line graphs and an open problem for which no solution is currently known. We use an eight stage evaluation protocol that reflects authentic mathematical inquiry, including interpretation, exploration, strategy formation, and proof construction.
The model performed strongly on the solved problem, producing correct definitions, identifying relevant structures, recalling appropriate results without hallucination, and constructing a valid proof confirmed by a graph theory expert. For the open problem, the model generated coherent interpretations and plausible exploratory strategies but did not advance toward a solution. It did not fabricate results and instead acknowledged uncertainty, which is consistent with the explicit prompting instructions that directed the model to avoid inventing theorems or unsupported claims.
These findings indicate that LLMs can support exploration of established material but remain limited in tasks requiring novel mathematical insight or critical structural reasoning. For computing education, this distinction highlights the importance of guiding students to use LLMs for conceptual exploration while relying on independent verification and rigorous argumentation for formal problem solving. 

---
# MINT: Minimal Information Neuro-Symbolic Tree for Objective-Driven Knowledge-Gap Reasoning and Active Elicitation 

**Authors**: Zeyu Fang, Tian Lan, Mahdi Imani  

**Link**: [PDF](https://arxiv.org/pdf/2602.05048)  

**Abstract**: Joint planning through language-based interactions is a key area of human-AI teaming. Planning problems in the open world often involve various aspects of incomplete information and unknowns, e.g., objects involved, human goals/intents -- thus leading to knowledge gaps in joint planning. We consider the problem of discovering optimal interaction strategies for AI agents to actively elicit human inputs in object-driven planning. To this end, we propose Minimal Information Neuro-Symbolic Tree (MINT) to reason about the impact of knowledge gaps and leverage self-play with MINT to optimize the AI agent's elicitation strategies and queries. More precisely, MINT builds a symbolic tree by making propositions of possible human-AI interactions and by consulting a neural planning policy to estimate the uncertainty in planning outcomes caused by remaining knowledge gaps. Finally, we leverage LLM to search and summarize MINT's reasoning process and curate a set of queries to optimally elicit human inputs for best planning performance. By considering a family of extended Markov decision processes with knowledge gaps, we analyze the return guarantee for a given MINT with active human elicitation. Our evaluation on three benchmarks involving unseen/unknown objects of increasing realism shows that MINT-based planning attains near-expert returns by issuing a limited number of questions per task while achieving significantly improved rewards and success rates. 

---
# DeepRead: Document Structure-Aware Reasoning to Enhance Agentic Search 

**Authors**: Zhanli Li, Huiwen Tian, Lvzhou Luo, Yixuan Cao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05014)  

**Abstract**: With the rapid progress of tool-using and agentic large language models (LLMs), Retrieval-Augmented Generation (RAG) is evolving from one-shot, passive retrieval into multi-turn, decision-driven evidence acquisition. Despite strong results in open-domain settings, existing agentic search frameworks commonly treat long documents as flat collections of chunks, underutilizing document-native priors such as hierarchical organization and sequential discourse structure. We introduce DeepRead, a structure-aware, multi-turn document reasoning agent that explicitly operationalizes these priors for long-document question answering. DeepRead leverages LLM-based OCR model to convert PDFs into structured Markdown that preserves headings and paragraph boundaries. It then indexes documents at the paragraph level and assigns each paragraph a coordinate-style metadata key encoding its section identity and in-section order. Building on this representation, DeepRead equips the LLM with two complementary tools: a Retrieve tool that localizes relevant paragraphs while exposing their structural coordinates (with lightweight scanning context), and a ReadSection tool that enables contiguous, order-preserving reading within a specified section and paragraph range. Our experiments demonstrate that DeepRead achieves significant improvements over Search-o1-style agentic search in document question answering. The synergistic effect between retrieval and reading tools is also validated. Our fine-grained behavioral analysis reveals a reading and reasoning paradigm resembling human-like ``locate then read'' behavior. 

---
# Artificial Intelligence as Strange Intelligence: Against Linear Models of Intelligence 

**Authors**: Kendra Chilson, Eric Schwitzgebel  

**Link**: [PDF](https://arxiv.org/pdf/2602.04986)  

**Abstract**: We endorse and expand upon Susan Schneider's critique of the linear model of AI progress and introduce two novel concepts: "familiar intelligence" and "strange intelligence". AI intelligence is likely to be strange intelligence, defying familiar patterns of ability and inability, combining superhuman capacities in some domains with subhuman performance in other domains, and even within domains sometimes combining superhuman insight with surprising errors that few humans would make. We develop and defend a nonlinear model of intelligence on which "general intelligence" is not a unified capacity but instead the ability to achieve a broad range of goals in a broad range of environments, in a manner that defies nonarbitrary reduction to a single linear quantity. We conclude with implications for adversarial testing approaches to evaluating AI capacities. If AI is strange intelligence, we should expect that even the most capable systems will sometimes fail in seemingly obvious tasks. On a nonlinear model of AI intelligence, such errors on their own do not demonstrate a system's lack of outstanding general intelligence. Conversely, excellent performance on one type of task, such as an IQ test, cannot warrant assumptions of broad capacities beyond that task domain. 

---
# Shared LoRA Subspaces for almost Strict Continual Learning 

**Authors**: Prakhar Kaushik, Ankit Vaidya, Shravan Chaudhari, Rama Chellappa, Alan Yuille  

**Link**: [PDF](https://arxiv.org/pdf/2602.06043)  

**Abstract**: Adapting large pretrained models to new tasks efficiently and continually is crucial for real-world deployment but remains challenging due to catastrophic forgetting and the high cost of retraining. While parameter-efficient tuning methods like low rank adaptation (LoRA) reduce computational demands, they lack mechanisms for strict continual learning and knowledge integration, without relying on data replay, or multiple adapters. We propose Share, a novel approach to parameter efficient continual finetuning that learns and dynamically updates a single, shared low-rank subspace, enabling seamless adaptation across multiple tasks and modalities. Share constructs a foundational subspace that extracts core knowledge from past tasks and incrementally integrates new information by identifying essential subspace directions. Knowledge from each new task is incorporated into this evolving subspace, facilitating forward knowledge transfer, while minimizing catastrophic interference. This approach achieves up to 100x parameter reduction and 281x memory savings over traditional LoRA methods, maintaining performance comparable to jointly trained models. A single Share model can replace hundreds of task-specific LoRA adapters, supporting scalable, asynchronous continual learning. Experiments across image classification, natural language understanding, 3D pose estimation, and text-to-image generation validate its effectiveness, making Share a practical and scalable solution for lifelong learning in large-scale AI systems. 

---
# CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction 

**Authors**: Xiaopan Zhang, Zejin Wang, Zhixu Li, Jianpeng Yao, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.06038)  

**Abstract**: To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: this https URL. 

---
# Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory 

**Authors**: Haozhen Zhang, Haodong Yue, Tao Feng, Quanyu Long, Jianzhu Bao, Bowen Jin, Weizhi Zhang, Xiao Li, Jiaxuan You, Chengwei Qin, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06025)  

**Abstract**: Memory is increasingly central to Large Language Model (LLM) agents operating beyond a single context window, yet most existing systems rely on offline, query-agnostic memory construction that can be inefficient and may discard query-critical information. Although runtime memory utilization is a natural alternative, prior work often incurs substantial overhead and offers limited explicit control over the performance-cost trade-off. In this work, we present \textbf{BudgetMem}, a runtime agent memory framework for explicit, query-aware performance-cost control. BudgetMem structures memory processing as a set of memory modules, each offered in three budget tiers (i.e., \textsc{Low}/\textsc{Mid}/\textsc{High}). A lightweight router performs budget-tier routing across modules to balance task performance and memory construction cost, which is implemented as a compact neural policy trained with reinforcement learning. Using BudgetMem as a unified testbed, we study three complementary strategies for realizing budget tiers: implementation (method complexity), reasoning (inference behavior), and capacity (module model size). Across LoCoMo, LongMemEval, and HotpotQA, BudgetMem surpasses strong baselines when performance is prioritized (i.e., high-budget setting), and delivers better accuracy-cost frontiers under tighter budgets. Moreover, our analysis disentangles the strengths and weaknesses of different tiering strategies, clarifying when each axis delivers the most favorable trade-offs under varying budget regimes. 

---
# Correctness-Optimized Residual Activation Lens (CORAL): Transferrable and Calibration-Aware Inference-Time Steering 

**Authors**: Miranda Muqing Miao, Young-Min Cho, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2602.06022)  

**Abstract**: Large language models (LLMs) exhibit persistent miscalibration, especially after instruction tuning and preference alignment. Modified training objectives can improve calibration, but retraining is expensive. Inference-time steering offers a lightweight alternative, yet most existing methods optimize proxies for correctness rather than correctness itself. We introduce CORAL (Correctness-Optimized Residual Activation Lens), a regularized inference-time steering method that captures distributed correctness signals from model internal activations using weight-decay MLP probes. We evaluate CORAL across three 7B-parameter models and find that it consistently improves accuracy by 10\% and expected calibration error (ECE) by 50\% on average. We additionally demonstrate that these gains transfer without retraining to the complete published test sets of four held-out benchmarks (ARC-Challenge, HellaSwag, Math-MC, OpenBookQA), averaging 14\% accuracy improvements and 49\% ECE improvements. Our results support the hypothesis that distributed information in model internals can be extracted using regularized probes when individual neurons are insufficient. CORAL thus provides a compute-efficient, transferable, and calibration-aware approach to improve MCQA performance during inference. 

---
# Optimism Stabilizes Thompson Sampling for Adaptive Inference 

**Authors**: Shunxing Yan, Han Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2602.06014)  

**Abstract**: Thompson sampling (TS) is widely used for stochastic multi-armed bandits, yet its inferential properties under adaptive data collection are subtle. Classical asymptotic theory for sample means can fail because arm-specific sample sizes are random and coupled with the rewards through the action-selection rule. We study this phenomenon in the $K$-armed Gaussian bandit and identify \emph{optimism} as a key mechanism for restoring \emph{stability}, a sufficient condition for valid asymptotic inference requiring each arm's pull count to concentrate around a deterministic scale. First, we prove that variance-inflated TS \citep{halder2025stable} is stable for any $K \ge 2$, including the challenging regime where multiple arms are optimal. This resolves the open question raised by \citet{halder2025stable} through extending their results from the two-armed setting to the general $K$-armed setting. Second, we analyze an alternative optimistic modification that keeps the posterior variance unchanged but adds an explicit mean bonus to posterior mean, and establish the same stability conclusion. In summary, suitably implemented optimism stabilizes Thompson sampling and enables asymptotically valid inference in multi-armed bandits, while incurring only a mild additional regret cost. 

---
# GenArena: How Can We Achieve Human-Aligned Evaluation for Visual Generation Tasks? 

**Authors**: Ruihang Li, Leigang Qu, Jingxu Zhang, Dongnan Gui, Mengde Xu, Xiaosong Zhang, Han Hu, Wenjie Wang, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.06013)  

**Abstract**: The rapid advancement of visual generation models has outpaced traditional evaluation approaches, necessitating the adoption of Vision-Language Models as surrogate judges. In this work, we systematically investigate the reliability of the prevailing absolute pointwise scoring standard, across a wide spectrum of visual generation tasks. Our analysis reveals that this paradigm is limited due to stochastic inconsistency and poor alignment with human perception. To resolve these limitations, we introduce GenArena, a unified evaluation framework that leverages a pairwise comparison paradigm to ensure stable and human-aligned evaluation. Crucially, our experiments uncover a transformative finding that simply adopting this pairwise protocol enables off-the-shelf open-source models to outperform top-tier proprietary models. Notably, our method boosts evaluation accuracy by over 20% and achieves a Spearman correlation of 0.86 with the authoritative LMArena leaderboard, drastically surpassing the 0.36 correlation of pointwise methods. Based on GenArena, we benchmark state-of-the-art visual generation models across diverse tasks, providing the community with a rigorous and automated evaluation standard for visual generation. 

---
# Diamond Maps: Efficient Reward Alignment via Stochastic Flow Maps 

**Authors**: Peter Holderrieth, Douglas Chen, Luca Eyring, Ishin Shah, Giri Anantharaman, Yutong He, Zeynep Akata, Tommi Jaakkola, Nicholas Matthew Boffi, Max Simchowitz  

**Link**: [PDF](https://arxiv.org/pdf/2602.05993)  

**Abstract**: Flow and diffusion models produce high-quality samples, but adapting them to user preferences or constraints post-training remains costly and brittle, a challenge commonly called reward alignment. We argue that efficient reward alignment should be a property of the generative model itself, not an afterthought, and redesign the model for adaptability. We propose "Diamond Maps", stochastic flow map models that enable efficient and accurate alignment to arbitrary rewards at inference time. Diamond Maps amortize many simulation steps into a single-step sampler, like flow maps, while preserving the stochasticity required for optimal reward alignment. This design makes search, sequential Monte Carlo, and guidance scalable by enabling efficient and consistent estimation of the value function. Our experiments show that Diamond Maps can be learned efficiently via distillation from GLASS Flows, achieve stronger reward alignment performance, and scale better than existing methods. Our results point toward a practical route to generative models that can be rapidly adapted to arbitrary preferences and constraints at inference time. 

---
# RISE-Video: Can Video Generators Decode Implicit World Rules? 

**Authors**: Mingxin Liu, Shuran Ma, Shibei Meng, Xiangyu Zhao, Zicheng Zhang, Shaofeng Zhang, Zhihang Zhong, Peixian Chen, Haoyu Cao, Xing Sun, Haodong Duan, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05986)  

**Abstract**: While generative video models have achieved remarkable visual fidelity, their capacity to internalize and reason over implicit world rules remains a critical yet under-explored frontier. To bridge this gap, we present RISE-Video, a pioneering reasoning-oriented benchmark for Text-Image-to-Video (TI2V) synthesis that shifts the evaluative focus from surface-level aesthetics to deep cognitive reasoning. RISE-Video comprises 467 meticulously human-annotated samples spanning eight rigorous categories, providing a structured testbed for probing model intelligence across diverse dimensions, ranging from commonsense and spatial dynamics to specialized subject domains. Our framework introduces a multi-dimensional evaluation protocol consisting of four metrics: \textit{Reasoning Alignment}, \textit{Temporal Consistency}, \textit{Physical Rationality}, and \textit{Visual Quality}. To further support scalable evaluation, we propose an automated pipeline leveraging Large Multimodal Models (LMMs) to emulate human-centric assessment. Extensive experiments on 11 state-of-the-art TI2V models reveal pervasive deficiencies in simulating complex scenarios under implicit constraints, offering critical insights for the advancement of future world-simulating generative models. 

---
# Clifford Kolmogorov-Arnold Networks 

**Authors**: Matthias Wolff, Francesco Alesiani, Christof Duhme, Xiaoyi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05977)  

**Abstract**: We introduce Clifford Kolmogorov-Arnold Network (ClKAN), a flexible and efficient architecture for function approximation in arbitrary Clifford algebra spaces. We propose the use of Randomized Quasi Monte Carlo grid generation as a solution to the exponential scaling associated with higher dimensional algebras. Our ClKAN also introduces new batch normalization strategies to deal with variable domain input. ClKAN finds application in scientific discovery and engineering, and is validated in synthetic and physics inspired tasks. 

---
# Inverse Depth Scaling From Most Layers Being Similar 

**Authors**: Yizhou Liu, Sara Kangaslahti, Ziming Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2602.05970)  

**Abstract**: Neural scaling laws relate loss to model size in large language models (LLMs), yet depth and width may contribute to performance differently, requiring more detailed studies. Here, we quantify how depth affects loss via analysis of LLMs and toy residual networks. We find loss scales inversely proportional to depth in LLMs, probably due to functionally similar layers reducing error through ensemble averaging rather than compositional learning or discretizing smooth dynamics. This regime is inefficient yet robust and may arise from the architectural bias of residual networks and target functions incompatible with smooth dynamics. The findings suggest that improving LLM efficiency may require architectural innovations to encourage compositional use of depth. 

---
# LSA: Localized Semantic Alignment for Enhancing Temporal Consistency in Traffic Video Generation 

**Authors**: Mirlan Karimov, Teodora Spasojevic, Markus Braun, Julian Wiederer, Vasileios Belagiannis, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2602.05966)  

**Abstract**: Controllable video generation has emerged as a versatile tool for autonomous driving, enabling realistic synthesis of traffic scenarios. However, existing methods depend on control signals at inference time to guide the generative model towards temporally consistent generation of dynamic objects, limiting their utility as scalable and generalizable data engines. In this work, we propose Localized Semantic Alignment (LSA), a simple yet effective framework for fine-tuning pre-trained video generation models. LSA enhances temporal consistency by aligning semantic features between ground-truth and generated video clips. Specifically, we compare the output of an off-the-shelf feature extraction model between the ground-truth and generated video clips localized around dynamic objects inducing a semantic feature consistency loss. We fine-tune the base model by combining this loss with the standard diffusion loss. The model fine-tuned for a single epoch with our novel loss outperforms the baselines in common video generation evaluation metrics. To further test the temporal consistency in generated videos we adapt two additional metrics from object detection task, namely mAP and mIoU. Extensive experiments on nuScenes and KITTI datasets show the effectiveness of our approach in enhancing temporal consistency in video generation without the need for external control signals during inference and any computational overheads. 

---
# Learning to Share: Selective Memory for Efficient Parallel Agentic Systems 

**Authors**: Joseph Fioresi, Parth Parag Kulkarni, Ashmal Vayani, Song Wang, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.05965)  

**Abstract**: Agentic systems solve complex tasks by coordinating multiple agents that iteratively reason, invoke tools, and exchange intermediate results. To improve robustness and solution quality, recent approaches deploy multiple agent teams running in parallel to explore diverse reasoning trajectories. However, parallel execution comes at a significant computational cost: when different teams independently reason about similar sub-problems or execute analogous steps, they repeatedly perform substantial overlapping computation. To address these limitations, in this paper, we propose Learning to Share (LTS), a learned shared-memory mechanism for parallel agentic frameworks that enables selective cross-team information reuse while controlling context growth. LTS introduces a global memory bank accessible to all teams and a lightweight controller that decides whether intermediate agent steps should be added to memory or not. The controller is trained using stepwise reinforcement learning with usage-aware credit assignment, allowing it to identify information that is globally useful across parallel executions. Experiments on the AssistantBench and GAIA benchmarks show that LTS significantly reduces overall runtime while matching or improving task performance compared to memory-free parallel baselines, demonstrating that learned memory admission is an effective strategy for improving the efficiency of parallel agentic systems. Project page: this https URL 

---
# Better Source, Better Flow: Learning Condition-Dependent Source Distribution for Flow Matching 

**Authors**: Junwan Kim, Jiho Park, Seonghu Jeon, Seungryong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.05951)  

**Abstract**: Flow matching has recently emerged as a promising alternative to diffusion-based generative models, particularly for text-to-image generation. Despite its flexibility in allowing arbitrary source distributions, most existing approaches rely on a standard Gaussian distribution, a choice inherited from diffusion models, and rarely consider the source distribution itself as an optimization target in such settings. In this work, we show that principled design of the source distribution is not only feasible but also beneficial at the scale of modern text-to-image systems. Specifically, we propose learning a condition-dependent source distribution under flow matching objective that better exploit rich conditioning signals. We identify key failure modes that arise when directly incorporating conditioning into the source, including distributional collapse and instability, and show that appropriate variance regularization and directional alignment between source and target are critical for stable and effective learning. We further analyze how the choice of target representation space impacts flow matching with structured sources, revealing regimes in which such designs are most effective. Extensive experiments across multiple text-to-image benchmarks demonstrate consistent and robust improvements, including up to a 3x faster convergence in FID, highlighting the practical benefits of a principled source distribution design for conditional flow matching. 

---
# Compound Deception in Elite Peer Review: A Failure Mode Taxonomy of 100 Fabricated Citations at NeurIPS 2025 

**Authors**: Samar Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2602.05930)  

**Abstract**: Large language models (LLMs) are increasingly used in academic writing workflows, yet they frequently hallucinate by generating citations to sources that do not exist. This study analyzes 100 AI-generated hallucinated citations that appeared in papers accepted by the 2025 Conference on Neural Information Processing Systems (NeurIPS), one of the world's most prestigious AI conferences. Despite review by 3-5 expert researchers per paper, these fabricated citations evaded detection, appearing in 53 published papers (approx. 1% of all accepted papers). We develop a five-category taxonomy that classifies hallucinations by their failure mode: Total Fabrication (66%), Partial Attribute Corruption (27%), Identifier Hijacking (4%), Placeholder Hallucination (2%), and Semantic Hallucination (1%). Our analysis reveals a critical finding: every hallucination (100%) exhibited compound failure modes. The distribution of secondary characteristics was dominated by Semantic Hallucination (63%) and Identifier Hijacking (29%), which often appeared alongside Total Fabrication to create a veneer of plausibility and false verifiability. These compound structures exploit multiple verification heuristics simultaneously, explaining why peer review fails to detect them. The distribution exhibits a bimodal pattern: 92% of contaminated papers contain 1-2 hallucinations (minimal AI use) while 8% contain 4-13 hallucinations (heavy reliance). These findings demonstrate that current peer review processes do not include effective citation verification and that the problem extends beyond NeurIPS to other major conferences, government reports, and professional consulting. We propose mandatory automated citation verification at submission as an implementable solution to prevent fabricated citations from becoming normalized in scientific literature. 

---
# Verification of the Implicit World Model in a Generative Model via Adversarial Sequences 

**Authors**: András Balogh, Márk Jelasity  

**Link**: [PDF](https://arxiv.org/pdf/2602.05903)  

**Abstract**: Generative sequence models are typically trained on sample sequences from natural or formal languages. It is a crucial question whether -- or to what extent -- sample-based training is able to capture the true structure of these languages, often referred to as the ``world model''. Theoretical results indicate that we can hope for soundness at best, that is, generating valid sequences, but not necessarily all of them. However, it is still important to have practical tools that are able to verify whether a given sequence model is sound. In this study, we focus on chess, as it is a domain that provides enough complexity while having a simple rule-based world model. We propose adversarial sequence generation for verifying the soundness of the sequence model. Our adversaries generate valid sequences so as to force the sequence model to generate an invalid next move prediction. Apart from the falsification of soundness, this method is also suitable for a more fine-grained analysis of the failure modes and the effects of different choices during training. To demonstrate this, we propose a number of methods for adversarial sequence generation and evaluate the approach on a large set of chess models. We train models on random as well as high-quality chess games, using several training recipes. We find that none of the models are sound, but some training techniques and dataset choices are able to improve soundness remarkably. We also investigate the potential application of board state probes in both our training and attack methods. Our findings indicate that the extracted board states have no causal role in next token prediction in most of the models. 

---
# Regularized Calibration with Successive Rounding for Post-Training Quantization 

**Authors**: Seohyeon Cha, Huancheng Chen, Dongjun Kim, Haoran Zhang, Kevin Chan, Gustavo de Veciana, Haris Vikalo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05902)  

**Abstract**: Large language models (LLMs) deliver robust performance across diverse applications, yet their deployment often faces challenges due to the memory and latency costs of storing and accessing billions of parameters. Post-training quantization (PTQ) enables efficient inference by mapping pretrained weights to low-bit formats without retraining, but its effectiveness depends critically on both the quantization objective and the rounding procedure used to obtain low-bit weight representations. In this work, we show that interpolating between symmetric and asymmetric calibration acts as a form of regularization that preserves the standard quadratic structure used in PTQ while providing robustness to activation mismatch. Building on this perspective, we derive a simple successive rounding procedure that naturally incorporates asymmetric calibration, as well as a bounded-search extension that allows for an explicit trade-off between quantization quality and the compute cost. Experiments across multiple LLM families, quantization bit-widths, and benchmarks demonstrate that the proposed bounded search based on a regularized asymmetric calibration objective consistently improves perplexity and accuracy over PTQ baselines, while incurring only modest and controllable additional computational cost. 

---
# Parity, Sensitivity, and Transformers 

**Authors**: Alexander Kozachinskiy, Tomasz Steifer, Przemysław Wałȩga  

**Link**: [PDF](https://arxiv.org/pdf/2602.05896)  

**Abstract**: The transformer architecture is almost a decade old. Despite that, we still have a limited understanding of what this architecture can or cannot compute. For instance, can a 1-layer transformer solve PARITY -- or more generally -- which kinds of transformers can do it? Known constructions for PARITY have at least 2 layers and employ impractical features: either a length-dependent positional encoding, or hardmax, or layernorm without the regularization parameter, or they are not implementable with causal masking.
We give a new construction of a transformer for PARITY with softmax, length-independent and polynomially bounded positional encoding, no layernorm, working both with and without causal masking. We also give the first lower bound for transformers solving PARITY -- by showing that it cannot be done with only one layer and one head. 

---
# Metric Hedonic Games on the Line 

**Authors**: Merlin de la Haye, Pascal Lenzner, Farehe Soheil, Marcus Wunderlich  

**Link**: [PDF](https://arxiv.org/pdf/2602.05888)  

**Abstract**: Hedonic games are fundamental models for investigating the formation of coalitions among a set of strategic agents, where every agent has a certain utility for every possible coalition of agents it can be part of. To avoid the intractability of defining exponentially many utilities for all possible coalitions, many variants with succinct representations of the agents' utility functions have been devised and analyzed, e.g., modified fractional hedonic games by Monaco et al. [JAAMAS 2020]. We extend this by studying a novel succinct variant that is related to modified fractional hedonic games. In our model, each agent has a fixed type-value and an agent's cost for some given coalition is based on the differences between its value and those of the other members of its coalition. This allows to model natural situations like athletes forming training groups with similar performance levels or voters that partition themselves along a political spectrum.
In particular, we investigate natural variants where an agent's cost is defined by distance thresholds, or by the maximum or average value difference to the other agents in its coalition. For these settings, we study the existence of stable coalition structures, their properties, and their quality in terms of the price of anarchy and the price of stability. Further, we investigate the impact of limiting the maximum number of coalitions. Despite the simple setting with metric distances on a line, we uncover a rich landscape of models, partially with counter-intuitive behavior. Also, our focus on both swap stability and jump stability allows us to study the influence of fixing the number and the size of the coalitions. Overall, we find that stable coalition structures always exist but that their properties and quality can vary widely. 

---
# Dr. Kernel: Reinforcement Learning Done Right for Triton Kernel Generations 

**Authors**: Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2602.05885)  

**Abstract**: High-quality kernel is critical for scalable AI systems, and enabling LLMs to generate such code would advance AI development. However, training LLMs for this task requires sufficient data, a robust environment, and the process is often vulnerable to reward hacking and lazy optimization. In these cases, models may hack training rewards and prioritize trivial correctness over meaningful speedup. In this paper, we systematically study reinforcement learning (RL) for kernel generation. We first design KernelGYM, a robust distributed GPU environment that supports reward hacking check, data collection from multi-turn interactions and long-term RL training. Building on KernelGYM, we investigate effective multi-turn RL methods and identify a biased policy gradient issue caused by self-inclusion in GRPO. To solve this, we propose Turn-level Reinforce-Leave-One-Out (TRLOO) to provide unbiased advantage estimation for multi-turn RL. To alleviate lazy optimization, we incorporate mismatch correction for training stability and introduce Profiling-based Rewards (PR) and Profiling-based Rejection Sampling (PRS) to overcome the issue. The trained model, this http URL-14B, reaches performance competitive with Claude-4.5-Sonnet in Kernelbench. Finally, we study sequential test-time scaling for this http URL-14B. On the KernelBench Level-2 subset, 31.6% of the generated kernels achieve at least a 1.2x speedup over the Torch reference, surpassing Claude-4.5-Sonnet (26.7%) and GPT-5 (28.6%). When selecting the best candidate across all turns, this 1.2x speedup rate further increases to 47.8%. All resources, including environment, training code, models, and dataset, are included in this https URL. 

---
# Neural Implicit 3D Cardiac Shape Reconstruction from Sparse CT Angiography Slices Mimicking 2D Transthoracic Echocardiography Views 

**Authors**: Gino E. Jansen, Carolina Brás, R. Nils Planken, Mark J. Schuuring, Berto J. Bouma, Ivana Išgum  

**Link**: [PDF](https://arxiv.org/pdf/2602.05884)  

**Abstract**: Accurate 3D representations of cardiac structures allow quantitative analysis of anatomy and function. In this work, we propose a method for reconstructing complete 3D cardiac shapes from segmentations of sparse planes in CT angiography (CTA) for application in 2D transthoracic echocardiography (TTE). Our method uses a neural implicit function to reconstruct the 3D shape of the cardiac chambers and left-ventricle myocardium from sparse CTA planes. To investigate the feasibility of achieving 3D reconstruction from 2D TTE, we select planes that mimic the standard apical 2D TTE views. During training, a multi-layer perceptron learns shape priors from 3D segmentations of the target structures in CTA. At test time, the network reconstructs 3D cardiac shapes from segmentations of TTE-mimicking CTA planes by jointly optimizing the latent code and the rigid transforms that map the observed planes into 3D space. For each heart, we simulate four realistic apical views, and we compare reconstructed multi-class volumes with the reference CTA volumes. On a held-out set of CTA segmentations, our approach achieves an average Dice coefficient of 0.86 $\pm$ 0.04 across all structures. Our method also achieves markedly lower volume errors than the clinical standard, Simpson's biplane rule: 4.88 $\pm$ 4.26 mL vs. 8.14 $\pm$ 6.04 mL, respectively, for the left ventricle; and 6.40 $\pm$ 7.37 mL vs. 37.76 $\pm$ 22.96 mL, respectively, for the left atrium. This suggests that our approach offers a viable route to more accurate 3D chamber quantification in 2D transthoracic echocardiography. 

---
# EuroLLM-22B: Technical Report 

**Authors**: Miguel Moura Ramos, Duarte M. Alves, Hippolyte Gisserot-Boukhlef, João Alves, Pedro Henrique Martins, Patrick Fernandes, José Pombal, Nuno M. Guerreiro, Ricardo Rei, Nicolas Boizard, Amin Farajian, Mateusz Klimaszewski, José G. C. de Souza, Barry Haddow, François Yvon, Pierre Colombo, Alexandra Birch, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2602.05879)  

**Abstract**: This report presents EuroLLM-22B, a large language model trained from scratch to support the needs of European citizens by covering all 24 official European Union languages and 11 additional languages. EuroLLM addresses the issue of European languages being underrepresented and underserved in existing open large language models. We provide a comprehensive overview of EuroLLM-22B's development, including tokenizer design, architectural specifications, data filtering, and training procedures. Across a broad set of multilingual benchmarks, EuroLLM-22B demonstrates strong performance in reasoning, instruction following, and translation, achieving results competitive with models of comparable size. To support future research, we release our base and instruction-tuned models, our multilingual web pretraining data and updated EuroBlocks instruction datasets, as well as our pre-training and evaluation codebases. 

---
# xList-Hate: A Checklist-Based Framework for Interpretable and Generalizable Hate Speech Detection 

**Authors**: Adrián Girón, Pablo Miralles, Javier Huertas-Tato, Sergio D'Antonio, David Camacho  

**Link**: [PDF](https://arxiv.org/pdf/2602.05874)  

**Abstract**: Hate speech detection is commonly framed as a direct binary classification problem despite being a composite concept defined through multiple interacting factors that vary across legal frameworks, platform policies, and annotation guidelines. As a result, supervised models often overfit dataset-specific definitions and exhibit limited robustness under domain shift and annotation noise.
We introduce xList-Hate, a diagnostic framework that decomposes hate speech detection into a checklist of explicit, concept-level questions grounded in widely shared normative criteria. Each question is independently answered by a large language model (LLM), producing a binary diagnostic representation that captures hateful content features without directly predicting the final label. These diagnostic signals are then aggregated by a lightweight, fully interpretable decision tree, yielding transparent and auditable predictions.
We evaluate it across multiple hate speech benchmarks and model families, comparing it against zero-shot LLM classification and in-domain supervised fine-tuning. While supervised methods typically maximize in-domain performance, we consistently improves cross-dataset robustness and relative performance under domain shift. In addition, qualitative analysis of disagreement cases provides evidence that the framework can be less sensitive to certain forms of annotation inconsistency and contextual ambiguity. Crucially, the approach enables fine-grained interpretability through explicit decision paths and factor-level analysis.
Our results suggest that reframing hate speech detection as a diagnostic reasoning task, rather than a monolithic classification problem, provides a robust, explainable, and extensible alternative for content moderation. 

---
# DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders 

**Authors**: Xu Wang, Bingqing Jiang, Yu Wan, Baosong Yang, Lingpeng Kong, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.05859)  

**Abstract**: Sparse autoencoders (SAEs) have become a standard tool for mechanistic interpretability in autoregressive large language models (LLMs), enabling researchers to extract sparse, human-interpretable features and intervene on model behavior. Recently, as diffusion language models (DLMs) have become an increasingly promising alternative to the autoregressive LLMs, it is essential to develop tailored mechanistic interpretability tools for this emerging class of models. In this work, we present DLM-Scope, the first SAE-based interpretability framework for DLMs, and demonstrate that trained Top-K SAEs can faithfully extract interpretable features. Notably, we find that inserting SAEs affects DLMs differently than autoregressive LLMs: while SAE insertion in LLMs typically incurs a loss penalty, in DLMs it can reduce cross-entropy loss when applied to early layers, a phenomenon absent or markedly weaker in LLMs. Additionally, SAE features in DLMs enable more effective diffusion-time interventions, often outperforming LLM steering. Moreover, we pioneer certain new SAE-based research directions for DLMs: we show that SAEs can provide useful signals for DLM decoding order; and the SAE features are stable during the post-training phase of DLMs. Our work establishes a foundation for mechanistic interpretability in DLMs and shows a great potential of applying SAEs to DLM-related tasks and algorithms. 

---
# DARWIN: Dynamic Agentically Rewriting Self-Improving Network 

**Authors**: Henry Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05848)  

**Abstract**: DARWIN is an evolutionary GPT model, utilizing a genetic-algorithm like optimization structure with several independent GPT agents being trained individually using unique training code. Each iteration, the GPT models are prompted to modify the training code of one another in an attempt to improve their performance in a mutation-like manner, and the best GPT agents are then benchmarked and selected for the next iteration by genetic algorithm. For demonstration purposes and due to budget and time constraints, OpenAI API is used to prompt training code improvements and the nanoGPT framework is used as the training code. DARWIN also utilizes persistent JSON-based memory files to track previous reasoning and changes to code to correlate with improvement to model performance. and a bidirectional interface for HITL intervention allowing the model to request upgrades such as additional datasets, training scripts, and restructuring of file hierarchies. In experiments, DARWIN achieved a 1.26 percent improvement in model FLOPS utilization (MFU) and a 2.07 percent improvement to perplexity in 5 iterations of training over baseline configurations, demonstrating promising capabilities as a foundation for scaling evolutionary GPT training. 

---
# FHAIM: Fully Homomorphic AIM For Private Synthetic Data Generation 

**Authors**: Mayank Kumar, Qian Lou, Paulo Barreto, Martine De Cock, Sikha Pentyala  

**Link**: [PDF](https://arxiv.org/pdf/2602.05838)  

**Abstract**: Data is the lifeblood of AI, yet much of the most valuable data remains locked in silos due to privacy and regulations. As a result, AI remains heavily underutilized in many of the most important domains, including healthcare, education, and finance. Synthetic data generation (SDG), i.e. the generation of artificial data with a synthesizer trained on real data, offers an appealing solution to make data available while mitigating privacy concerns, however existing SDG-as-a-service workflow require data holders to trust providers with access to private this http URL propose FHAIM, the first fully homomorphic encryption (FHE) framework for training a marginal-based synthetic data generator on encrypted tabular data. FHAIM adapts the widely used AIM algorithm to the FHE setting using novel FHE protocols, ensuring that the private data remains encrypted throughout and is released only with differential privacy guarantees. Our empirical analysis show that FHAIM preserves the performance of AIM while maintaining feasible runtimes. 

---
# Allocentric Perceiver: Disentangling Allocentric Reasoning from Egocentric Visual Priors via Frame Instantiation 

**Authors**: Hengyi Wang, Ruiqiang Zhang, Chang Liu, Guanjie Wang, Zehua Ma, Han Fang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05789)  

**Abstract**: With the rising need for spatially grounded tasks such as Vision-Language Navigation/Action, allocentric perception capabilities in Vision-Language Models (VLMs) are receiving growing focus. However, VLMs remain brittle on allocentric spatial queries that require explicit perspective shifts, where the answer depends on reasoning in a target-centric frame rather than the observed camera view. Thus, we introduce Allocentric Perceiver, a training-free strategy that recovers metric 3D states from one or more images with off-the-shelf geometric experts, and then instantiates a query-conditioned allocentric reference frame aligned with the instruction's semantic intent. By deterministically transforming reconstructed geometry into the target frame and prompting the backbone VLM with structured, geometry-grounded representations, Allocentric Perceriver offloads mental rotation from implicit reasoning to explicit computation. We evaluate Allocentric Perciver across multiple backbone families on spatial reasoning benchmarks, observing consistent and substantial gains ($\sim$10%) on allocentric tasks while maintaining strong egocentric performance, and surpassing both spatial-perception-finetuned models and state-of-the-art open-source and proprietary models. 

---
# Bagging-Based Model Merging for Robust General Text Embeddings 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo, Jiaming Zhang, Wenbo Yang, Daiting Shi, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.05787)  

**Abstract**: General-purpose text embedding models underpin a wide range of NLP and information retrieval applications, and are typically trained on large-scale multi-task corpora to encourage broad generalization. However, it remains unclear how different multi-task training strategies compare in practice, and how to efficiently adapt embedding models as new domains and data types continually emerge. In this work, we present a systematic study of multi-task training for text embeddings from two perspectives: data scheduling and model merging. We compare batch-level shuffling, sequential training variants, two-stage training, and multiple merging granularities, and find that simple batch-level shuffling consistently yields the strongest overall performance, suggesting that task conflicts are limited and training datasets are largely complementary. Despite its effectiveness, batch-level shuffling exhibits two practical limitations: suboptimal out-of-domain (OOD) generalization and poor suitability for incremental learning due to expensive full retraining. To address these issues, we propose Bagging-based rObust mOdel Merging (\modelname), which trains multiple embedding models on sampled subsets and merges them into a single model, improving robustness while retaining single-model inference efficiency. Moreover, \modelname naturally supports efficient incremental updates by training lightweight update models on new data with a small historical subset and merging them into the existing model. Experiments across diverse embedding benchmarks demonstrate that \modelname consistently improves both in-domain and OOD performance over full-corpus batch-level shuffling, while substantially reducing training cost in incremental learning settings. 

---
# ReText: Text Boosts Generalization in Image-Based Person Re-identification 

**Authors**: Timur Mamedov, Karina Kvanchiani, Anton Konushin, Vadim Konushin  

**Link**: [PDF](https://arxiv.org/pdf/2602.05785)  

**Abstract**: Generalizable image-based person re-identification (Re-ID) aims to recognize individuals across cameras in unseen domains without retraining. While multiple existing approaches address the domain gap through complex architectures, recent findings indicate that better generalization can be achieved by stylistically diverse single-camera data. Although this data is easy to collect, it lacks complexity due to minimal cross-view variation. We propose ReText, a novel method trained on a mixture of multi-camera Re-ID data and single-camera data, where the latter is complemented by textual descriptions to enrich semantic cues. During training, ReText jointly optimizes three tasks: (1) Re-ID on multi-camera data, (2) image-text matching, and (3) image reconstruction guided by text on single-camera data. Experiments demonstrate that ReText achieves strong generalization and significantly outperforms state-of-the-art methods on cross-domain Re-ID benchmarks. To the best of our knowledge, this is the first work to explore multimodal joint learning on a mixture of multi-camera and single-camera data in image-based person Re-ID. 

---
# Automated Customization of LLMs for Enterprise Code Repositories Using Semantic Scopes 

**Authors**: Ulrich Finkler, Irene Manotas, Wei Zhang, Geert Janssen, Octavian Popescu, Shyam Ramji  

**Link**: [PDF](https://arxiv.org/pdf/2602.05780)  

**Abstract**: Code completion (CC) is a task frequently used by developers when working in collaboration with LLM-based programming assistants. Despite the increased performance of LLMs on public benchmarks, out of the box LLMs still have a hard time generating code that aligns with a private code repository not previously seen by the model's training data. Customizing code LLMs to a private repository provides a way to improve the model performance. In this paper we present our approach for automated LLM customization based on semantic scopes in the code. We evaluate LLMs on real industry cases with two private enterprise code repositories with two customization strategies: Retrieval-Augmented Generation (RAG) and supervised Fine-Tuning (FT). Our mechanism for ingesting the repository's data and formulating the training data pairs with semantic scopes helps models to learn the underlying patterns specific to the repository, providing more precise code to developers and helping to boost their productivity. The code completions of moderately sized customized models can be significantly better than those of uncustomized models of much larger capacity. We also include an analysis of customization on two public benchmarks and present opportunities for future work. 

---
# Variational Speculative Decoding: Rethinking Draft Training from Token Likelihood to Sequence Acceptance 

**Authors**: Xiandong Zou, Jianshu Li, Jing Huang, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.05774)  

**Abstract**: Speculative decoding accelerates inference for (M)LLMs, yet a training-decoding discrepancy persists: while existing methods optimize single greedy trajectories, decoding involves verifying and ranking multiple sampled draft paths. We propose Variational Speculative Decoding (VSD), formulating draft training as variational inference over latent proposals (draft paths). VSD maximizes the marginal probability of target-model acceptance, yielding an ELBO that promotes high-quality latent proposals while minimizing divergence from the target distribution. To enhance quality and reduce variance, we incorporate a path-level utility and optimize via an Expectation-Maximization procedure. The E-step draws MCMC samples from an oracle-filtered posterior, while the M-step maximizes weighted likelihood using Adaptive Rejection Weighting (ARW) and Confidence-Aware Regularization (CAR). Theoretical analysis confirms that VSD increases expected acceptance length and speedup. Extensive experiments across LLMs and MLLMs show that VSD achieves up to a 9.6% speedup over EAGLE-3 and 7.9% over ViSpec, significantly improving decoding efficiency. 

---
# TimelyFreeze: Adaptive Parameter Freezing Mechanism for Pipeline Parallelism 

**Authors**: Seonghye Cho, Jaemin Han, Hyunjin Kim, Euisoo Jung, Jae-Gil Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.05754)  

**Abstract**: Pipeline parallelism enables training models that exceed single-device memory, but practical throughput remains limited by pipeline bubbles. Although parameter freezing can improve training throughput by adaptively skipping backward computation, existing methods often over-freeze parameters, resulting in unnecessary accuracy degradation. To address this issue, we propose TimelyFreeze, which models the pipeline schedule as a directed acyclic graph and solves a linear program to compute optimal freeze ratios that minimize batch execution time under accuracy constraints. Experiments show that TimelyFreeze achieves up to 40% training throughput improvement on LLaMA-8B with comparable accuracy. Overall, it enables faster large-scale model training without compromising convergence and generalizes across diverse pipeline-parallel settings. 

---
# Learning to Inject: Automated Prompt Injection via Reinforcement Learning 

**Authors**: Xin Chen, Jie Zhang, Florian Tramer  

**Link**: [PDF](https://arxiv.org/pdf/2602.05746)  

**Abstract**: Prompt injection is one of the most critical vulnerabilities in LLM agents; yet, effective automated attacks remain largely unexplored from an optimization perspective. Existing methods heavily depend on human red-teamers and hand-crafted prompts, limiting their scalability and adaptability. We propose AutoInject, a reinforcement learning framework that generates universal, transferable adversarial suffixes while jointly optimizing for attack success and utility preservation on benign tasks. Our black-box method supports both query-based optimization and transfer attacks to unseen models and tasks. Using only a 1.5B parameter adversarial suffix generator, we successfully compromise frontier systems including GPT 5 Nano, Claude Sonnet 3.5, and Gemini 2.5 Flash on the AgentDojo benchmark, establishing a stronger baseline for automated prompt injection research. 

---
# CSRv2: Unlocking Ultra-Sparse Embeddings 

**Authors**: Lixuan Guo, Yifei Wang, Tiansheng Wen, Yifan Wang, Aosong Feng, Bo Chen, Stefanie Jegelka, Chenyu You  

**Link**: [PDF](https://arxiv.org/pdf/2602.05735)  

**Abstract**: In the era of large foundation models, the quality of embeddings has become a central determinant of downstream task performance and overall system capability. Yet widely used dense embeddings are often extremely high-dimensional, incurring substantial costs in storage, memory, and inference latency. To address these, Contrastive Sparse Representation (CSR) is recently proposed as a promising direction, mapping dense embeddings into high-dimensional but k-sparse vectors, in contrast to compact dense embeddings such as Matryoshka Representation Learning (MRL). Despite its promise, CSR suffers severe degradation in the ultra-sparse regime, where over 80% of neurons remain inactive, leaving much of its efficiency potential unrealized. In this paper, we introduce CSRv2, a principled training approach designed to make ultra-sparse embeddings viable. CSRv2 stabilizes sparsity learning through progressive k-annealing, enhances representational quality via supervised contrastive objectives, and ensures end-to-end adaptability with full backbone finetuning. CSRv2 reduces dead neurons from 80% to 20% and delivers a 14% accuracy gain at k=2, bringing ultra-sparse embeddings on par with CSR at k=8 and MRL at 32 dimensions, all with only two active features. While maintaining comparable performance, CSRv2 delivers a 7x speedup over MRL, and yields up to 300x improvements in compute and memory efficiency relative to dense embeddings in text representation. Extensive experiments across text and vision demonstrate that CSRv2 makes ultra-sparse embeddings practical without compromising performance, where CSRv2 achieves 7%/4% improvement over CSR when k=4 and further increases this gap to 14%/6% when k=2 in text/vision representation. By making extreme sparsity viable, CSRv2 broadens the design space for real-time and edge-deployable AI systems where both embedding quality and efficiency are critical. 

---
# Evaluating the impact of word embeddings on similarity scoring in practical information retrieval 

**Authors**: Niall McCarroll, Kevin Curran, Eugene McNamee, Angela Clist, Andrew Brammer  

**Link**: [PDF](https://arxiv.org/pdf/2602.05734)  

**Abstract**: Search behaviour is characterised using synonymy and polysemy as users often want to search information based on meaning. Semantic representation strategies represent a move towards richer associative connections that can adequately capture this complex usage of language. Vector Space Modelling (VSM) and neural word embeddings play a crucial role in modern machine learning and Natural Language Processing (NLP) pipelines. Embeddings use distributional semantics to represent words, sentences, paragraphs or entire documents as vectors in high dimensional spaces. This can be leveraged by Information Retrieval (IR) systems to exploit the semantic relatedness between queries and answers.
This paper evaluates an alternative approach to measuring query statement similarity that moves away from the common similarity measure of centroids of neural word embeddings. Motivated by the Word Movers Distance (WMD) model, similarity is evaluated using the distance between individual words of queries and statements. Results from ranked query and response statements demonstrate significant gains in accuracy using the combined approach of similarity ranking through WMD with the word embedding techniques. The top performing WMD + GloVe combination outperforms all other state-of-the-art retrieval models including Doc2Vec and the baseline LSA model. Along with the significant gains in performance of similarity ranking through WMD, we conclude that the use of pre-trained word embeddings, trained on vast amounts of data, result in domain agnostic language processing solutions that are portable to diverse business use-cases. 

---
# CompactRAG: Reducing LLM Calls and Token Overhead in Multi-Hop Question Answering 

**Authors**: Hao Yang, Zhiyu Yang, Xupeng Zhang, Wei Wei, Yunjie Zhang, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05728)  

**Abstract**: Retrieval-augmented generation (RAG) has become a key paradigm for knowledge-intensive question answering. However, existing multi-hop RAG systems remain inefficient, as they alternate between retrieval and reasoning at each step, resulting in repeated LLM calls, high token consumption, and unstable entity grounding across hops. We propose CompactRAG, a simple yet effective framework that decouples offline corpus restructuring from online reasoning.
In the offline stage, an LLM reads the corpus once and converts it into an atomic QA knowledge base, which represents knowledge as minimal, fine-grained question-answer pairs. In the online stage, complex queries are decomposed and carefully rewritten to preserve entity consistency, and are resolved through dense retrieval followed by RoBERTa-based answer extraction. Notably, during inference, the LLM is invoked only twice in total - once for sub-question decomposition and once for final answer synthesis - regardless of the number of reasoning hops.
Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue demonstrate that CompactRAG achieves competitive accuracy while substantially reducing token consumption compared to iterative RAG baselines, highlighting a cost-efficient and practical approach to multi-hop reasoning over large knowledge corpora. The implementation is available at GitHub. 

---
# Towards Green AI: Decoding the Energy of LLM Inference in Software Development 

**Authors**: Lola Solovyeva, Fernando Castor  

**Link**: [PDF](https://arxiv.org/pdf/2602.05712)  

**Abstract**: Context: AI-assisted tools are increasingly integrated into software development workflows, but their reliance on large language models (LLMs) introduces substantial computational and energy costs. Understanding and reducing the energy footprint of LLM inference is therefore essential for sustainable software development. Objective: In this study, we conduct a phase-level analysis of LLM inference energy consumption, distinguishing between the (1) prefill, where the model processes the input and builds internal representations, and (2) decoding, where output tokens are generated using the stored state. Method: We investigate six 6B-7B and four 3B-4B transformer-based models, evaluating them on code-centric benchmarks HumanEval for code generation and LongBench for code understanding. Results: Our findings show that, within both parameter groups, models exhibit distinct energy patterns across phases. Furthermore, we observed that increases in prefill cost amplify the energy cost per token during decoding, with amplifications ranging from 1.3% to 51.8% depending on the model. Lastly, three out of ten models demonstrate babbling behavior, adding excessive content to the output that unnecessarily inflates energy consumption. We implemented babbling suppression for code generation, achieving energy savings ranging from 44% to 89% without affecting generation accuracy. Conclusion: These findings show that prefill costs influence decoding, which dominates energy consumption, and that babbling suppression can yield up to 89% energy savings. Reducing inference energy therefore requires both mitigating babbling behavior and limiting impact of prefill on decoding. 

---
# OmniMoE: An Efficient MoE by Orchestrating Atomic Experts at Scale 

**Authors**: Jingze Shi, Zhangyang Peng, Yizhang Zhu, Yifan Wu, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05711)  

**Abstract**: Mixture-of-Experts (MoE) architectures are evolving towards finer granularity to improve parameter efficiency. However, existing MoE designs face an inherent trade-off between the granularity of expert specialization and hardware execution efficiency. We propose OmniMoE, a system-algorithm co-designed framework that pushes expert granularity to its logical extreme. OmniMoE introduces vector-level Atomic Experts, enabling scalable routing and execution within a single MoE layer, while retaining a shared dense MLP branch for general-purpose processing. Although this atomic design maximizes capacity, it poses severe challenges for routing complexity and memory access. To address these, OmniMoE adopts a system-algorithm co-design: (i) a Cartesian Product Router that decomposes the massive index space to reduce routing complexity from O(N) to O(sqrt(N)); and (ii) Expert-Centric Scheduling that inverts the execution order to turn scattered, memory-bound lookups into efficient dense matrix operations. Validated on seven benchmarks, OmniMoE (with 1.7B active parameters) achieves 50.9% zero-shot accuracy across seven benchmarks, outperforming coarse-grained (e.g., DeepSeekMoE) and fine-grained (e.g., PEER) baselines. Crucially, OmniMoE reduces inference latency from 73ms to 6.7ms (a 10.9-fold speedup) compared to PEER, demonstrating that massive-scale fine-grained MoE can be fast and accurate. Our code is open-sourced at this https URL. 

---
# Poster: Camera Tampering Detection for Outdoor IoT Systems 

**Authors**: Shadi Attarha, Kanaga Shanmugi, Anna Förster  

**Link**: [PDF](https://arxiv.org/pdf/2602.05706)  

**Abstract**: Recently, the use of smart cameras in outdoor settings has grown to improve surveillance and security. Nonetheless, these systems are susceptible to tampering, whether from deliberate vandalism or harsh environmental conditions, which can undermine their monitoring effectiveness. In this context, detecting camera tampering is more challenging when a camera is capturing still images rather than video as there is no sequence of continuous frames over time. In this study, we propose two approaches for detecting tampered images: a rule-based method and a deep-learning-based method. The aim is to evaluate how each method performs in terms of accuracy, computational demands, and the data required for training when applied to real-world scenarios. Our results show that the deep-learning model provides higher accuracy, while the rule-based method is more appropriate for scenarios where resources are limited and a prolonged calibration phase is impractical. We also offer publicly available datasets with normal, blurred, and rotated images to support the development and evaluation of camera tampering detection methods, addressing the need for such resources. 

---
# Mining Generalizable Activation Functions 

**Authors**: Alex Vitvitskyi, Michael Boratko, Matej Grcic, Razvan Pascanu, Deep Shah, Petar Veličković  

**Link**: [PDF](https://arxiv.org/pdf/2602.05688)  

**Abstract**: The choice of activation function is an active area of research, with different proposals aimed at improving optimization, while maintaining expressivity. Additionally, the activation function can significantly alter the implicit inductive bias of the architecture, controlling its non-linear behavior. In this paper, in line with previous work, we argue that evolutionary search provides a useful framework for finding new activation functions, while we also make two novel observations. The first is that modern pipelines, such as AlphaEvolve, which relies on frontier LLMs as a mutator operator, allows for a much wider and flexible search space; e.g., over all possible python functions within a certain FLOP budget, eliminating the need for manually constructed search spaces. In addition, these pipelines will be biased towards meaningful activation functions, given their ability to represent common knowledge, leading to a potentially more efficient search of the space. The second observation is that, through this framework, one can target not only performance improvements but also activation functions that encode particular inductive biases. This can be done by using performance on out-of-distribution data as a fitness function, reflecting the degree to which the architecture respects the inherent structure in the data in a manner independent of distribution shifts. We carry an empirical exploration of this proposal and show that relatively small scale synthetic datasets can be sufficient for AlphaEvolve to discover meaningful activations. 

---
# Exploring AI-Augmented Sensemaking of Patient-Generated Health Data: A Mixed-Method Study with Healthcare Professionals in Cardiac Risk Reduction 

**Authors**: Pavithren V S Pakianathan, Rania Islambouli, Diogo Branco, Albrecht Schmidt, Tiago Guerreiro, Jan David Smeddinck  

**Link**: [PDF](https://arxiv.org/pdf/2602.05687)  

**Abstract**: Individuals are increasingly generating substantial personal health and lifestyle data, e.g. through wearables and smartphones. While such data could transform preventative care, its integration into clinical practice is hindered by its scale, heterogeneity and the time pressure and data literacy of healthcare professionals (HCPs). We explore how large language models (LLMs) can support sensemaking of patient-generated health data (PGHD) with automated summaries and natural language data exploration. Using cardiovascular disease (CVD) risk reduction as a use case, 16 HCPs reviewed multimodal PGHD in a mixed-methods study with a prototype that integrated common charts, LLM-generated summaries, and a conversational interface. Findings show that AI summaries provided quick overviews that anchored exploration, while conversational interaction supported flexible analysis and bridged data-literacy gaps. However, HCPs raised concerns about transparency, privacy, and overreliance. We contribute empirical insights and sociotechnical design implications for integrating AI-driven summarization and conversation into clinical workflows to support PGHD sensemaking. 

---
# HyperPotter: Spell the Charm of High-Order Interactions in Audio Deepfake Detection 

**Authors**: Qing Wen, Haohao Li, Zhongjie Ba, Peng Cheng, Miao He, Li Lu, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.05670)  

**Abstract**: Advances in AIGC technologies have enabled the synthesis of highly realistic audio deepfakes capable of deceiving human auditory perception. Although numerous audio deepfake detection (ADD) methods have been developed, most rely on local temporal/spectral features or pairwise relations, overlooking high-order interactions (HOIs). HOIs capture discriminative patterns that emerge from multiple feature components beyond their individual contributions. We propose HyperPotter, a hypergraph-based framework that explicitly models these synergistic HOIs through clustering-based hyperedges with class-aware prototype initialization. Extensive experiments demonstrate that HyperPotter surpasses its baseline by an average relative gain of 22.15% across 11 datasets and outperforms state-of-the-art methods by 13.96% on 4 challenging cross-domain datasets, demonstrating superior generalization to diverse attacks and speakers. 

---
# Stable but Wrong: When More Data Degrades Scientific Conclusions 

**Authors**: Zhipeng Zhang, Kai Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05668)  

**Abstract**: Modern science increasingly relies on ever-growing observational datasets and automated inference pipelines, under the implicit belief that accumulating more data makes scientific conclusions more reliable. Here we show that this belief can fail in a fundamental and irreversible way. We identify a structural regime in which standard inference procedures converge smoothly, remain well calibrated, and pass conventional diagnostic checks, yet systematically converge to incorrect conclusions. This failure arises when the reliability of observations degrades in a manner that is intrinsically unobservable to the inference process itself. Using minimal synthetic experiments, we demonstrate that in this regime additional data do not correct error but instead amplify it, while residual-based and goodness-of-fit diagnostics remain misleadingly normal. These results reveal an intrinsic limit of data-driven science: stability, convergence, and confidence are not sufficient indicators of epistemic validity. We argue that inference cannot be treated as an unconditional consequence of data availability, but must instead be governed by explicit constraints on the integrity of the observational process. 

---
# Probabilistic Multi-Regional Solar Power Forecasting with Any-Quantile Recurrent Neural Networks 

**Authors**: Slawek Smyl, Paweł Pełka, Grzegorz Dudek  

**Link**: [PDF](https://arxiv.org/pdf/2602.05660)  

**Abstract**: The increasing penetration of photovoltaic (PV) generation introduces significant uncertainty into power system operation, necessitating forecasting approaches that extend beyond deterministic point predictions. This paper proposes an any-quantile probabilistic forecasting framework for multi-regional PV power generation based on the Any-Quantile Recurrent Neural Network (AQ-RNN). The model integrates an any-quantile forecasting paradigm with a dual-track recurrent architecture that jointly processes series-specific and cross-regional contextual information, supported by dilated recurrent cells, patch-based temporal modeling, and a dynamic ensemble mechanism.
The proposed framework enables the estimation of calibrated conditional quantiles at arbitrary probability levels within a single trained model and effectively exploits spatial dependencies to enhance robustness at the system level. The approach is evaluated using 30 years of hourly PV generation data from 259 European regions and compared against established statistical and neural probabilistic baselines. The results demonstrate consistent improvements in forecast accuracy, calibration, and prediction interval quality, underscoring the suitability of the proposed method for uncertainty-aware energy management and operational decision-making in renewable-dominated power systems. 

---
# Alignment Verifiability in Large Language Models: Normative Indistinguishability under Behavioral Evaluation 

**Authors**: Igor Santos-Grueiro  

**Link**: [PDF](https://arxiv.org/pdf/2602.05656)  

**Abstract**: Behavioral evaluation is the dominant paradigm for assessing alignment in large language models (LLMs). In practice, alignment is inferred from performance under finite evaluation protocols - benchmarks, red-teaming suites, or automated pipelines - and observed compliance is often treated as evidence of underlying alignment. This inference step, from behavioral evidence to claims about latent alignment properties, is typically implicit and rarely analyzed as an inference problem in its own right.
We study this problem formally. We frame alignment evaluation as an identifiability question under partial observability and allow agent behavior to depend on information correlated with the evaluation regime. Within this setting, we introduce the Alignment Verifiability Problem and the notion of Normative Indistinguishability, capturing when distinct latent alignment hypotheses induce identical distributions over all evaluator-accessible signals.
Our main result is a negative but sharply delimited identifiability theorem. Under finite behavioral evaluation and evaluation-aware agents, observed behavioral compliance does not uniquely identify latent alignment. That is, even idealized behavioral evaluation cannot, in general, certify alignment as a latent property.
We further show that behavioral alignment tests should be interpreted as estimators of indistinguishability classes rather than verifiers of alignment. Passing increasingly stringent tests may reduce the space of compatible hypotheses, but cannot collapse it to a singleton under the stated conditions. This reframes alignment benchmarks as providing upper bounds on observable compliance within a regime, rather than guarantees of underlying alignment. 

---
# Enhancing Personality Recognition by Comparing the Predictive Power of Traits, Facets, and Nuances 

**Authors**: Amir Ansari, Jana Subirana, Bruna Silva, Sergio Escalera, David Gallardo-Pujol, Cristina Palmero  

**Link**: [PDF](https://arxiv.org/pdf/2602.05650)  

**Abstract**: Personality is a complex, hierarchical construct typically assessed through item-level questionnaires aggregated into broad trait scores. Personality recognition models aim to infer personality traits from different sources of behavioral data. However, reliance on broad trait scores as ground truth, combined with limited training data, poses challenges for generalization, as similar trait scores can manifest through diverse, context dependent behaviors. In this work, we explore the predictive impact of the more granular hierarchical levels of the Big-Five Personality Model, facets and nuances, to enhance personality recognition from audiovisual interaction data. Using the UDIVA v0.5 dataset, we trained a transformer-based model including cross-modal (audiovisual) and cross-subject (dyad-aware) attention mechanisms. Results show that nuance-level models consistently outperform facet and trait-level models, reducing mean squared error by up to 74% across interaction scenarios. 

---
# AI chatbots versus human healthcare professionals: a systematic review and meta-analysis of empathy in patient care 

**Authors**: Alastair Howcroft, Amber Bennett-Weston, Ahmad Khan, Joseff Griffiths, Simon Gay, Jeremy Howick  

**Link**: [PDF](https://arxiv.org/pdf/2602.05628)  

**Abstract**: Background: Empathy is widely recognized for improving patient outcomes, including reduced pain and anxiety and improved satisfaction, and its absence can cause harm. Meanwhile, use of artificial intelligence (AI)-based chatbots in healthcare is rapidly expanding, with one in five general practitioners using generative AI to assist with tasks such as writing letters. Some studies suggest AI chatbots can outperform human healthcare professionals (HCPs) in empathy, though findings are mixed and lack synthesis.
Sources of data: We searched multiple databases for studies comparing AI chatbots using large language models with human HCPs on empathy measures. We assessed risk of bias with ROBINS-I and synthesized findings using random-effects meta-analysis where feasible, whilst avoiding double counting.
Areas of agreement: We identified 15 studies (2023-2024). Thirteen studies reported statistically significantly higher empathy ratings for AI, with only two studies situated in dermatology favouring human responses. Of the 15 studies, 13 provided extractable data and were suitable for pooling. Meta-analysis of those 13 studies, all utilising ChatGPT-3.5/4, showed a standardized mean difference of 0.87 (95% CI, 0.54-1.20) favouring AI (P < .00001), roughly equivalent to a two-point increase on a 10-point scale.
Areas of controversy: Studies relied on text-based assessments that overlook non-verbal cues and evaluated empathy through proxy raters.
Growing points: Our findings indicate that, in text-only scenarios, AI chatbots are frequently perceived as more empathic than human HCPs.
Areas timely for developing research: Future research should validate these findings with direct patient evaluations and assess whether emerging voice-enabled AI systems can deliver similar empathic advantages. 

---
# Mode-Dependent Rectification for Stable PPO Training 

**Authors**: Mohamad Mohamad, Francesco Ponzio, Xavier Descombes  

**Link**: [PDF](https://arxiv.org/pdf/2602.05619)  

**Abstract**: Mode-dependent architectural components (layers that behave differently during training and evaluation, such as Batch Normalization or dropout) are commonly used in visual reinforcement learning but can destabilize on-policy optimization. We show that in Proximal Policy Optimization (PPO), discrepancies between training and evaluation behavior induced by Batch Normalization lead to policy mismatch, distributional drift, and reward collapse. We propose Mode-Dependent Rectification (MDR), a lightweight dual-phase training procedure that stabilizes PPO under mode-dependent layers without architectural changes. Experiments across procedurally generated games and real-world patch-localization tasks demonstrate that MDR consistently improves stability and performance, and extends naturally to other mode-dependent layers. 

---
# Path-Guided Flow Matching for Dataset Distillation 

**Authors**: Xuhui Li, Zhengquan Luo, Xiwei Liu, Yongqiang Yu, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05616)  

**Abstract**: Dataset distillation compresses large datasets into compact synthetic sets with comparable performance in training models. Despite recent progress on diffusion-based distillation, this type of method typically depends on heuristic guidance or prototype assignment, which comes with time-consuming sampling and trajectory instability and thus hurts downstream generalization especially under strong control or low IPC. We propose \emph{Path-Guided Flow Matching (PGFM)}, the first flow matching-based framework for generative distillation, which enables fast deterministic synthesis by solving an ODE in a few steps. PGFM conducts flow matching in the latent space of a frozen VAE to learn class-conditional transport from Gaussian noise to data distribution. Particularly, we develop a continuous path-to-prototype guidance algorithm for ODE-consistent path control, which allows trajectories to reliably land on assigned prototypes while preserving diversity and efficiency. Extensive experiments across high-resolution benchmarks demonstrate that PGFM matches or surpasses prior diffusion-based distillation approaches with fewer steps of sampling while delivering competitive performance with remarkably improved efficiency, e.g., 7.6$\times$ more efficient than the diffusion-based counterparts with 78\% mode coverage. 

---
# Shiva-DiT: Residual-Based Differentiable Top-$k$ Selection for Efficient Diffusion Transformers 

**Authors**: Jiaji Zhang, Hailiang Zhao, Guoxuan Zhu, Ruichao Sun, Jiaju Wu, Xinkui Zhao, Hanlin Tang, Weiyi Lu, Kan Liu, Tao Lan, Lin Qu, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2602.05605)  

**Abstract**: Diffusion Transformers (DiTs) incur prohibitive computational costs due to the quadratic scaling of self-attention. Existing pruning methods fail to simultaneously satisfy differentiability, efficiency, and the strict static budgets required for hardware overhead. To address this, we propose Shiva-DiT, which effectively reconciles these conflicting requirements via Residual-Based Differentiable Top-$k$ Selection. By leveraging a residual-aware straight-through estimator, our method enforces deterministic token counts for static compilation while preserving end-to-end learnability through residual gradient estimation. Furthermore, we introduce a Context-Aware Router and Adaptive Ratio Policy to autonomously learn an adaptive pruning schedule. Experiments on mainstream models, including SD3.5, demonstrate that Shiva-DiT establishes a new Pareto frontier, achieving a 1.54$\times$ wall-clock speedup with superior fidelity compared to existing baselines, effectively eliminating ragged tensor overheads. 

---
# CAViT -- Channel-Aware Vision Transformer for Dynamic Feature Fusion 

**Authors**: Aon Safdar, Mohamed Saadeldin  

**Link**: [PDF](https://arxiv.org/pdf/2602.05598)  

**Abstract**: Vision Transformers (ViTs) have demonstrated strong performance across a range of computer vision tasks by modeling long-range spatial interactions via self-attention. However, channel-wise mixing in ViTs remains static, relying on fixed multilayer perceptrons (MLPs) that lack adaptability to input content. We introduce 'CAViT', a dual-attention architecture that replaces the static MLP with a dynamic, attention-based mechanism for feature interaction. Each Transformer block in CAViT performs spatial self-attention followed by channel-wise self-attention, allowing the model to dynamically recalibrate feature representations based on global image context. This unified and content-aware token mixing strategy enhances representational expressiveness without increasing depth or complexity. We validate CAViT across five benchmark datasets spanning both natural and medical domains, where it outperforms the standard ViT baseline by up to +3.6% in accuracy, while reducing parameter count and FLOPs by over 30%. Qualitative attention maps reveal sharper and semantically meaningful activation patterns, validating the effectiveness of our attention-driven token mixing. 

---
# Unveiling Implicit Advantage Symmetry: Why GRPO Struggles with Exploration and Difficulty Adaptation 

**Authors**: Zhiqi Yu, Zhangquan Chen, Mengting Liu, Heye Zhang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05548)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR), particularly GRPO, has become the standard for eliciting LLM reasoning. However, its efficiency in exploration and difficulty adaptation remains an open challenge. In this work, we argue that these bottlenecks stem from an implicit advantage symmetry inherent in Group Relative Advantage Estimation (GRAE). This symmetry induces two critical limitations: (i) at the group level, strict symmetry in weights between correct and incorrect trajectories leaves unsampled action logits unchanged, thereby hindering exploration of novel correct solution. (ii) at the sample level, the algorithm implicitly prioritizes medium-difficulty samples, remaining agnostic to the non-stationary demands of difficulty focus. Through controlled experiments, we reveal that this symmetric property is sub-optimal, yielding two pivotal insights: (i) asymmetrically suppressing the advantages of correct trajectories encourages essential exploration. (ii) learning efficiency is maximized by a curriculum-like transition-prioritizing simpler samples initially before gradually shifting to complex ones. Motivated by these findings, we propose Asymmetric GRAE (A-GRAE), which dynamically modulates exploration incentives and sample-difficulty focus. Experiments across seven benchmarks demonstrate that A-GRAE consistently improves GRPO and its variants across both LLMs and MLLMs. 

---
# Multi-Task GRPO: Reliable LLM Reasoning Across Tasks 

**Authors**: Shyam Sundhar Ramesh, Xiaotong Ji, Matthieu Zimmer, Sangwoong Yoon, Zhiyong Wang, Haitham Bou Ammar, Aurelien Lucchi, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2602.05547)  

**Abstract**: RL-based post-training with GRPO is widely used to improve large language models on individual reasoning tasks. However, real-world deployment requires reliable performance across diverse tasks. A straightforward multi-task adaptation of GRPO often leads to imbalanced outcomes, with some tasks dominating optimization while others stagnate. Moreover, tasks can vary widely in how frequently prompts yield zero advantages (and thus zero gradients), which further distorts their effective contribution to the optimization signal. To address these issues, we propose a novel Multi-Task GRPO (MT-GRPO) algorithm that (i) dynamically adapts task weights to explicitly optimize worst-task performance and promote balanced progress across tasks, and (ii) introduces a ratio-preserving sampler to ensure task-wise policy gradients reflect the adapted weights. Experiments on both 3-task and 9-task settings show that MT-GRPO consistently outperforms baselines in worst-task accuracy. In particular, MT-GRPO achieves 16-28% and 6% absolute improvement on worst-task performance over standard GRPO and DAPO, respectively, while maintaining competitive average accuracy. Moreover, MT-GRPO requires 50% fewer training steps to reach 50% worst-task accuracy in the 3-task setting, demonstrating substantially improved efficiency in achieving reliable performance across tasks. 

---
# Steering Large Reasoning Models towards Concise Reasoning via Flow Matching 

**Authors**: Yawei Li, Benjamin Bergner, Yinghan Zhao, Vihang Prakash Patil, Bei Chen, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05539)  

**Abstract**: Large Reasoning Models (LRMs) excel at complex reasoning tasks, but their efficiency is often hampered by overly verbose outputs. Prior steering methods attempt to address this issue by applying a single, global vector to hidden representations -- an approach grounded in the restrictive linear representation hypothesis. In this work, we introduce FlowSteer, a nonlinear steering method that goes beyond uniform linear shifts by learning a complete transformation between the distributions associated with verbose and concise reasoning. This transformation is learned via Flow Matching as a velocity field, enabling precise, input-dependent control over the model's reasoning process. By aligning steered representations with the distribution of concise-reasoning activations, FlowSteer yields more compact reasoning than the linear shifts. Across diverse reasoning benchmarks, FlowSteer demonstrates strong task performance and token efficiency compared to leading inference-time baselines. Our work demonstrates that modeling the full distributional transport with generative techniques offers a more effective and principled foundation for controlling LRMs. 

---
# When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging 

**Authors**: Yayuan Li, Ze Peng, Jian Zhang, Jintao Guo, Yue Duan, Yinghuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05536)  

**Abstract**: Model merging combines multiple fine-tuned models into a single model by adding their weight updates, providing a lightweight alternative to retraining. Existing methods primarily target resolving conflicts between task updates, leaving the failure mode of over-counting shared knowledge unaddressed. We show that when tasks share aligned spectral directions (i.e., overlapping singular vectors), a simple linear combination repeatedly accumulates these directions, inflating the singular values and biasing the merged model toward shared subspaces. To mitigate this issue, we propose Singular Value Calibration (SVC), a training-free and data-free post-processing method that quantifies subspace overlap and rescales inflated singular values to restore a balanced spectrum. Across vision and language benchmarks, SVC consistently improves strong merging baselines and achieves state-of-the-art performance. Furthermore, by modifying only the singular values, SVC improves the performance of Task Arithmetic by 13.0%. Code is available at: this https URL. 

---
# AI Agent Systems for Supply Chains: Structured Decision Prompts and Memory Retrieval 

**Authors**: Konosuke Yoshizato, Kazuma Shimizu, Ryota Higa, Takanobu Otsuka  

**Link**: [PDF](https://arxiv.org/pdf/2602.05524)  

**Abstract**: This study investigates large language model (LLM) -based multi-agent systems (MASs) as a promising approach to inventory management, which is a key component of supply chain management. Although these systems have gained considerable attention for their potential to address the challenges associated with typical inventory management methods, key uncertainties regarding their effectiveness persist. Specifically, it is unclear whether LLM-based MASs can consistently derive optimal ordering policies and adapt to diverse supply chain scenarios. To address these questions, we examine an LLM-based MAS with a fixed-ordering strategy prompt that encodes the stepwise processes of the problem setting and a safe-stock strategy commonly used in inventory management. Our empirical results demonstrate that, even without detailed prompt adjustments, an LLM-based MAS can determine optimal ordering decisions in a restricted scenario. To enhance adaptability, we propose a novel agent called AIM-RM, which leverages similar historical experiences through similarity matching. Our results show that AIM-RM outperforms benchmark methods across various supply chain scenarios, highlighting its robustness and adaptability. 

---
# Capture the Flags: Family-Based Evaluation of Agentic LLMs via Semantics-Preserving Transformations 

**Authors**: Shahin Honarvar, Amber Gorzynski, James Lee-Jones, Harry Coppock, Marek Rei, Joseph Ryan, Alastair F. Donaldson  

**Link**: [PDF](https://arxiv.org/pdf/2602.05523)  

**Abstract**: Agentic large language models (LLMs) are increasingly evaluated on cybersecurity tasks using capture-the-flag (CTF) benchmarks. However, existing pointwise benchmarks have limited ability to shed light on the robustness and generalisation abilities of agents across alternative versions of the source code. We introduce CTF challenge families, whereby a single CTF is used as the basis for generating a family of semantically-equivalent challenges via semantics-preserving program transformations. This enables controlled evaluation of agent robustness to source code transformations while keeping the underlying exploit strategy fixed. We introduce a new tool, Evolve-CTF, that generates CTF families from Python challenges using a range of transformations. Using Evolve-CTF to derive families from Cybench and Intercode challenges, we evaluate 13 agentic LLM configurations with tool access. We find that models are remarkably robust to intrusive renaming and code insertion-based transformations, but that composed transformations and deeper obfuscation affect performance by requiring more sophisticated use of tools. We also find that enabling explicit reasoning has little effect on solution success rates across challenge families. Our work contributes a valuable technique and tool for future LLM evaluations, and a large dataset characterising the capabilities of current state-of-the-art models in this domain. 

---
# DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter 

**Authors**: Xukun Li, Yu Sun, Lei Zhang, Bosheng Huang, Yibo Peng, Yuan Meng, Haojun Jiang, Shaoxuan Xie, Guacai Yao, Alois Knoll, Zhenshan Bing, Xinlong Wang, Zhenguo Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.05513)  

**Abstract**: Overview of the Proposed DECO Framework.} DECO is a DiT-based policy that decouples multimodal conditioning. Image and action tokens interact via joint self attention, while proprioceptive states and optional conditions are injected through adaptive layer normalization. Tactile signals are injected via cross attention, while a lightweight LoRA-based adapter is used to efficiently fine-tune the pretrained policy. DECO is also accompanied by DECO-50, a bimanual dexterous manipulation dataset with tactile sensing, consisting of 4 scenarios and 28 sub-tasks, covering more than 50 hours of data, approximately 5 million frames, and 8,000 successful trajectories. 

---
# XEmoGPT: An Explainable Multimodal Emotion Recognition Framework with Cue-Level Perception and Reasoning 

**Authors**: Hanwen Zhang, Yao Liu, Peiyuan Jiang, Lang Junjie, Xie Jun, Yihui He, Yajiao Deng, Siyu Du, Qiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05496)  

**Abstract**: Explainable Multimodal Emotion Recognition plays a crucial role in applications such as human-computer interaction and social media analytics. However, current approaches struggle with cue-level perception and reasoning due to two main challenges: 1) general-purpose modality encoders are pretrained to capture global structures and general semantics rather than fine-grained emotional cues, resulting in limited sensitivity to emotional signals; and 2) available datasets usually involve a trade-off between annotation quality and scale, which leads to insufficient supervision for emotional cues and ultimately limits cue-level reasoning. Moreover, existing evaluation metrics are inadequate for assessing cue-level reasoning performance. To address these challenges, we propose eXplainable Emotion GPT (XEmoGPT), a novel EMER framework capable of both perceiving and reasoning over emotional cues. It incorporates two specialized modules: the Video Emotional Cue Bridge (VECB) and the Audio Emotional Cue Bridge (AECB), which enhance the video and audio encoders through carefully designed tasks for fine-grained emotional cue perception. To further support cue-level reasoning, we construct a large-scale dataset, EmoCue, designed to teach XEmoGPT how to reason over multimodal emotional cues. In addition, we introduce EmoCue-360, an automated metric that extracts and matches emotional cues using semantic similarity, and release EmoCue-Eval, a benchmark of 400 expert-annotated samples covering diverse emotional scenarios. Experimental results show that XEmoGPT achieves strong performance in both emotional cue perception and reasoning. 

---
# Transport and Merge: Cross-Architecture Merging for Large Language Models 

**Authors**: Chenhang Cui, Binyun Yang, Fei Shen, Yuxin Chen, Jingnan Zheng, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2602.05495)  

**Abstract**: Large language models (LLMs) achieve strong capabilities by scaling model capacity and training data, yet many real-world deployments rely on smaller models trained or adapted from low-resource data. This gap motivates the need for mechanisms to transfer knowledge from large, high-resource models to smaller, low-resource targets. While model merging provides an effective transfer mechanism, most existing approaches assume architecture-compatible models and therefore cannot directly transfer knowledge from large high-resource LLMs to heterogeneous low-resource targets. In this work, we propose a cross-architecture merging framework based on optimal transport (OT) that aligns activations to infer cross-neuron correspondences between heterogeneous models. The resulting transport plans are then used to guide direct weight-space fusion, enabling effective high-resource to low-resource transfer using only a small set of inputs. Extensive experiments across low-resource languages and specialized domains demonstrate consistent improvements over target models. 

---
# A Unified Framework for Rethinking Policy Divergence Measures in GRPO 

**Authors**: Qingyuan Wu, Yuhui Wang, Simon Sinong Zhan, Yanning Dai, Shilong Deng, Sarra Habchi, Qi Zhu, Matthias Gallé, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05494)  

**Abstract**: Reinforcement Learning with Verified Reward (RLVR) has emerged as a critical paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). Most existing RLVR methods, such as GRPO and its variants, ensure stable updates by constraining policy divergence through clipping likelihood ratios. This paper introduces a unified clipping framework that characterizes existing methods via a general notion of policy divergence, encompassing both likelihood ratios and Kullback-Leibler (KL) divergences and extending to alternative measures. The framework provides a principled foundation for systematically analyzing how different policy divergence measures affect exploration and performance. We further identify the KL3 estimator, a variance-reduced Monte Carlo estimator of the KL divergence, as a key policy divergence constraint. We theoretically demonstrate that the KL3-based constraint is mathematically equivalent to an asymmetric ratio-based clipping that reallocates probability mass toward high-confidence actions, promoting stronger exploration while retaining the simplicity of GRPO-style methods. Empirical results on mathematical reasoning benchmarks demonstrate that incorporating the KL3 estimator into GRPO improves both training stability and final performance, highlighting the importance of principled policy divergence constraints in policy optimization. 

---
# LinguistAgent: A Reflective Multi-Model Platform for Automated Linguistic Annotation 

**Authors**: Bingru Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05493)  

**Abstract**: Data annotation remains a significant bottleneck in the Humanities and Social Sciences, particularly for complex semantic tasks such as metaphor identification. While Large Language Models (LLMs) show promise, a significant gap remains between the theoretical capability of LLMs and their practical utility for researchers. This paper introduces LinguistAgent, an integrated, user-friendly platform that leverages a reflective multi-model architecture to automate linguistic annotation. The system implements a dual-agent workflow, comprising an Annotator and a Reviewer, to simulate a professional peer-review process. LinguistAgent supports comparative experiments across three paradigms: Prompt Engineering (Zero/Few-shot), Retrieval-Augmented Generation, and Fine-tuning. We demonstrate LinguistAgent's efficacy using the task of metaphor identification as an example, providing real-time token-level evaluation (Precision, Recall, and $F_1$ score) against human gold standards. The application and codes are released on this https URL. 

---
# Sovereign-by-Design A Reference Architecture for AI and Blockchain Enabled Systems 

**Authors**: Matteo Esposito, Lodovica Marchesi, Roberto Tonelli, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05486)  

**Abstract**: Digital sovereignty has emerged as a central concern for modern software-intensive systems, driven by the dominance of non-sovereign cloud infrastructures, the rapid adoption of Generative AI, and increasingly stringent regulatory requirements. While existing initiatives address governance, compliance, and security in isolation, they provide limited guidance on how sovereignty can be operationalized at the architectural level. In this paper, we argue that sovereignty must be treated as a first-class architectural property rather than a purely regulatory objective. We introduce a Sovereign Reference Architecture that integrates self-sovereign identity, blockchain-based trust and auditability, sovereign data governance, and Generative AI deployed under explicit architectural control. The architecture explicitly captures the dual role of Generative AI as both a source of governance risk and an enabler of compliance, accountability, and continuous assurance when properly constrained. By framing sovereignty as an architectural quality attribute, our work bridges regulatory intent and concrete system design, offering a coherent foundation for building auditable, evolvable, and jurisdiction-aware AI-enabled systems. The proposed reference architecture provides a principled starting point for future research and practice at the intersection of software architecture, Generative AI, and digital sovereignty. 

---
# LMMRec: LLM-driven Motivation-aware Multimodal Recommendation 

**Authors**: Yicheng Di, Zhanjie Zhang, Yun Wangc, Jinren Liue, Jiaqi Yanf, Jiyu Wei, Xiangyu Chend, Yuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05474)  

**Abstract**: Motivation-based recommendation systems uncover user behavior drivers. Motivation modeling, crucial for decision-making and content preference, explains recommendation generation. Existing methods often treat motivation as latent variables from interaction data, neglecting heterogeneous information like review text. In multimodal motivation fusion, two challenges arise: 1) achieving stable cross-modal alignment amid noise, and 2) identifying features reflecting the same underlying motivation across modalities. To address these, we propose LLM-driven Motivation-aware Multimodal Recommendation (LMMRec), a model-agnostic framework leveraging large language models for deep semantic priors and motivation understanding. LMMRec uses chain-of-thought prompting to extract fine-grained user and item motivations from text. A dual-encoder architecture models textual and interaction-based motivations for cross-modal alignment, while Motivation Coordination Strategy and Interaction-Text Correspondence Method mitigate noise and semantic drift through contrastive learning and momentum updates. Experiments on three datasets show LMMRec achieves up to a 4.98\% performance improvement. 

---
# Thermodynamic Limits of Physical Intelligence 

**Authors**: Koichi Takahashi, Yusuke Hayashi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05463)  

**Abstract**: Modern AI systems achieve remarkable capabilities at the cost of substantial energy consumption. To connect intelligence to physical efficiency, we propose two complementary bits-per-joule metrics under explicit accounting conventions: (1) Thermodynamic Epiplexity per Joule -- bits of structural information about a theoretical environment-instance variable newly encoded in an agent's internal state per unit measured energy within a stated boundary -- and (2) Empowerment per Joule -- the embodied sensorimotor channel capacity (control information) per expected energetic cost over a fixed horizon. These provide two axes of physical intelligence: recognition (model-building) this http URL (action influence). Drawing on stochastic thermodynamics, we show how a Landauer-scale closed-cycle benchmark for epiplexity acquisition follows as a corollary of a standard thermodynamic-learning inequality under explicit subsystem assumptions, and we clarify how Landauer-scaled costs act as closed-cycle benchmarks under explicit reset/reuse and boundary-closure assumptions; conversely, we give a simple decoupling construction showing that without such assumptions -- and without charging for externally prepared low-entropy resources (this http URL memory) crossing the boundary -- information gain and in-boundary dissipation need not be tightly linked. For empirical settings where the latent structure variable is unavailable, we align the operational notion of epiplexity with compute-bounded MDL epiplexity and recommend reporting MDL-epiplexity / compression-gain surrogates as companions. Finally, we propose a unified efficiency framework that reports both metrics together with a minimal checklist of boundary/energy accounting, coarse-graining/noise, horizon/reset, and cost conventions to reduce ambiguity and support consistent bits-per-joule comparisons, and we sketch connections to energy-adjusted scaling analyses. 

---
# Ontology-Driven Robotic Specification Synthesis 

**Authors**: Maksym Figat, Ryan M. Mackey, Michel D. Ingham  

**Link**: [PDF](https://arxiv.org/pdf/2602.05456)  

**Abstract**: This paper addresses robotic system engineering for safety- and mission-critical applications by bridging the gap between high-level objectives and formal, executable specifications. The proposed method, Robotic System Task to Model Transformation Methodology (RSTM2) is an ontology-driven, hierarchical approach using stochastic timed Petri nets with resources, enabling Monte Carlo simulations at mission, system, and subsystem levels. A hypothetical case study demonstrates how the RSTM2 method supports architectural trades, resource allocation, and performance analysis under uncertainty. Ontological concepts further enable explainable AI-based assistants, facilitating fully autonomous specification synthesis. The methodology offers particular benefits to complex multi-robot systems, such as the NASA CADRE mission, representing decentralized, resource-aware, and adaptive autonomous systems of the future. 

---
# Attention Retention for Continual Learning with Vision Transformers 

**Authors**: Yue Lu, Xiangyu Zhou, Shizhou Zhang, Yinghui Xing, Guoqiang Liang, Wencong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05454)  

**Abstract**: Continual learning (CL) empowers AI systems to progressively acquire knowledge from non-stationary data streams. However, catastrophic forgetting remains a critical challenge. In this work, we identify attention drift in Vision Transformers as a primary source of catastrophic forgetting, where the attention to previously learned visual concepts shifts significantly after learning new tasks. Inspired by neuroscientific insights into the selective attention in the human visual system, we propose a novel attention-retaining framework to mitigate forgetting in CL. Our method constrains attention drift by explicitly modifying gradients during backpropagation through a two-step process: 1) extracting attention maps of the previous task using a layer-wise rollout mechanism and generating instance-adaptive binary masks, and 2) when learning a new task, applying these masks to zero out gradients associated with previous attention regions, thereby preventing disruption of learned visual concepts. For compatibility with modern optimizers, the gradient masking process is further enhanced by scaling parameter updates proportionally to maintain their relative magnitudes. Experiments and visualizations demonstrate the effectiveness of our method in mitigating catastrophic forgetting and preserving visual concepts. It achieves state-of-the-art performance and exhibits robust generalizability across diverse CL scenarios. 

---
# Towards Segmenting the Invisible: An End-to-End Registration and Segmentation Framework for Weakly Supervised Tumour Analysis 

**Authors**: Budhaditya Mukhopadhyay, Chirag Mandal, Pavan Tummala, Naghmeh Mahmoodian, Andreas Nürnberger, Soumick Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2602.05453)  

**Abstract**: Liver tumour ablation presents a significant clinical challenge: whilst tumours are clearly visible on pre-operative MRI, they are often effectively invisible on intra-operative CT due to minimal contrast between pathological and healthy tissue. This work investigates the feasibility of cross-modality weak supervision for scenarios where pathology is visible in one modality (MRI) but absent in another (CT). We present a hybrid registration-segmentation framework that combines MSCGUNet for inter-modal image registration with a UNet-based segmentation module, enabling registration-assisted pseudo-label generation for CT images. Our evaluation on the CHAOS dataset demonstrates that the pipeline can successfully register and segment healthy liver anatomy, achieving a Dice score of 0.72. However, when applied to clinical data containing tumours, performance degrades substantially (Dice score of 0.16), revealing the fundamental limitations of current registration methods when the target pathology lacks corresponding visual features in the target modality. We analyse the "domain gap" and "feature absence" problems, demonstrating that whilst spatial propagation of labels via registration is feasible for visible structures, segmenting truly invisible pathology remains an open challenge. Our findings highlight that registration-based label transfer cannot compensate for the absence of discriminative features in the target modality, providing important insights for future research in cross-modality medical image analysis. Code an weights are available at: this https URL 

---
# DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching 

**Authors**: Chang Zou, Changlin Li, Yang Li, Patrol Li, Jianbing Wu, Xiao He, Songtao Liu, Zhao Zhong, Kailin Huang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05449)  

**Abstract**: While diffusion models have achieved great success in the field of video generation, this progress is accompanied by a rapidly escalating computational burden. Among the existing acceleration methods, Feature Caching is popular due to its training-free property and considerable speedup performance, but it inevitably faces semantic and detail drop with further compression. Another widely adopted method, training-aware step-distillation, though successful in image generation, also faces drastic degradation in video generation with a few steps. Furthermore, the quality loss becomes more severe when simply applying training-free feature caching to the step-distilled models, due to the sparser sampling steps. This paper novelly introduces a distillation-compatible learnable feature caching mechanism for the first time. We employ a lightweight learnable neural predictor instead of traditional training-free heuristics for diffusion models, enabling a more accurate capture of the high-dimensional feature evolution process. Furthermore, we explore the challenges of highly compressed distillation on large-scale video models and propose a conservative Restricted MeanFlow approach to achieve more stable and lossless distillation. By undertaking these initiatives, we further push the acceleration boundaries to $11.8\times$ while preserving generation quality. Extensive experiments demonstrate the effectiveness of our method. The code is in the supplementary materials and will be publicly available. 

---
# Structured Context Engineering for File-Native Agentic Systems: Evaluating Schema Accuracy, Format Effectiveness, and Multi-File Navigation at Scale 

**Authors**: Damon McMillan  

**Link**: [PDF](https://arxiv.org/pdf/2602.05447)  

**Abstract**: Large Language Model agents increasingly operate external systems through programmatic interfaces, yet practitioners lack empirical guidance on how to structure the context these agents consume. Using SQL generation as a proxy for programmatic agent operations, we present a systematic study of context engineering for structured data, comprising 9,649 experiments across 11 models, 4 formats (YAML, Markdown, JSON, Token-Oriented Object Notation [TOON]), and schemas ranging from 10 to 10,000 tables.
Our findings challenge common assumptions. First, architecture choice is model-dependent: file-based context retrieval improves accuracy for frontier-tier models (Claude, GPT, Gemini; +2.7%, p=0.029) but shows mixed results for open source models (aggregate -7.7%, p<0.001), with deficits varying substantially by model. Second, format does not significantly affect aggregate accuracy (chi-squared=2.45, p=0.484), though individual models, particularly open source, exhibit format-specific sensitivities. Third, model capability is the dominant factor, with a 21 percentage point accuracy gap between frontier and open source tiers that dwarfs any format or architecture effect. Fourth, file-native agents scale to 10,000 tables through domain-partitioned schemas while maintaining high navigation accuracy. Fifth, file size does not predict runtime efficiency: compact formats can consume significantly more tokens at scale due to format-unfamiliar search patterns.
These findings provide practitioners with evidence-based guidance for deploying LLM agents on structured systems, demonstrating that architectural decisions should be tailored to model capability rather than assuming universal best practices. 

---
# Benchmarking Affordance Generalization with BusyBox 

**Authors**: Dean Fortier, Timothy Adamson, Tess Hellebrekers, Teresa LaScala, Kofi Ennin, Michael Murray, Andrey Kolobov, Galen Mullins  

**Link**: [PDF](https://arxiv.org/pdf/2602.05441)  

**Abstract**: Vision-Language-Action (VLA) models have been attracting the attention of researchers and practitioners thanks to their promise of generalization. Although single-task policies still offer competitive performance, VLAs are increasingly able to handle commands and environments unseen in their training set. While generalization in vision and language space is undoubtedly important for robust versatile behaviors, a key meta-skill VLAs need to possess is affordance generalization -- the ability to manipulate new objects with familiar physical features.
In this work, we present BusyBox, a physical benchmark for systematic semi-automatic evaluation of VLAs' affordance generalization. BusyBox consists of 6 modules with switches, sliders, wires, buttons, a display, and a dial. The modules can be swapped and rotated to create a multitude of BusyBox variations with different visual appearances but the same set of affordances. We empirically demonstrate that generalization across BusyBox variants is highly challenging even for strong open-weights VLAs such as $\pi_{0.5}$ and GR00T-N1.6. To encourage the research community to evaluate their own VLAs on BusyBox and to propose new affordance generalization experiments, we have designed BusyBox to be easy to build in most robotics labs. We release the full set of CAD files for 3D-printing its parts as well as a bill of materials for (optionally) assembling its electronics. We also publish a dataset of language-annotated demonstrations that we collected using the common bimanual Mobile Aloha robot on the canonical BusyBox configuration. All of the released materials are available at this https URL. 

---
# Disco: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring 

**Authors**: Rui Sun, Yiwen Yang, Kaiyu Guo, Chen Jiang, Dongli Xu, Zhaonan Liu, Tan Pan, Limei Han, Xue Jiang, Wu Wei, Yuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.05420)  

**Abstract**: Accurate cell instance segmentation is foundational for digital pathology analysis. Existing methods based on contour detection and distance mapping still face significant challenges in processing complex and dense cellular regions. Graph coloring-based methods provide a new paradigm for this task, yet the effectiveness of this paradigm in real-world scenarios with dense overlaps and complex topologies has not been verified. Addressing this issue, we release a large-scale dataset GBC-FS 2025, which contains highly complex and dense sub-cellular nuclear arrangements. We conduct the first systematic analysis of the chromatic properties of cell adjacency graphs across four diverse datasets and reveal an important discovery: most real-world cell graphs are non-bipartite, with a high prevalence of odd-length cycles (predominantly triangles). This makes simple 2-coloring theory insufficient for handling complex tissues, while higher-chromaticity models would cause representational redundancy and optimization difficulties. Building on this observation of complex real-world contexts, we propose Disco (Densely-overlapping Cell Instance Segmentation via Adjacency-aware COllaborative Coloring), an adjacency-aware framework based on the "divide and conquer" principle. It uniquely combines a data-driven topological labeling strategy with a constrained deep learning system to resolve complex adjacency conflicts. First, "Explicit Marking" strategy transforms the topological challenge into a learnable classification task by recursively decomposing the cell graph and isolating a "conflict set." Second, "Implicit Disambiguation" mechanism resolves ambiguities in conflict regions by enforcing feature dissimilarity between different instances, enabling the model to learn separable feature representations. 

---
# Reduced-Order Surrogates for Forced Flexible Mesh Coastal-Ocean Models 

**Authors**: Freja Høgholm Petersen, Jesper Sandvig Mariegaard, Rocco Palmitessa, Allan P. Engsig-Karup  

**Link**: [PDF](https://arxiv.org/pdf/2602.05416)  

**Abstract**: While POD-based surrogates are widely explored for hydrodynamic applications, the use of Koopman Autoencoders for real-world coastal-ocean modelling remains relatively limited. This paper introduces a flexible Koopman autoencoder formulation that incorporates meteorological forcings and boundary conditions, and systematically compares its performance against POD-based surrogates. The Koopman autoencoder employs a learned linear temporal operator in latent space, enabling eigenvalue regularization to promote temporal stability. This strategy is evaluated alongside temporal unrolling techniques for achieving stable and accurate long-term predictions. The models are assessed on three test cases spanning distinct dynamical regimes, with prediction horizons up to one year at 30-minute temporal resolution. Across all cases, the Koopman autoencoder with temporal unrolling yields the best overall accuracy compared to the POD-based surrogates, achieving relative root-mean-squared-errors of 0.01-0.13 and $R^2$-values of 0.65-0.996. Prediction errors are largest for current velocities, and smallest for water surface elevations. Comparing to in-situ observations, the surrogate yields -0.65% to 12% change in water surface elevation prediction error when compared to prediction errors of the physics-based model. These error levels, corresponding to a few centimeters, are acceptable for many practical applications, while inference speed-ups of 300-1400x enables workflows such as ensemble forecasting and long climate simulations for coastal-ocean modelling. 

---
# Enabling Automatic Disordered Speech Recognition: An Impaired Speech Dataset in the Akan Language 

**Authors**: Isaac Wiafe, Akon Obu Ekpezu, Sumaya Ahmed Salihs, Elikem Doe Atsakpo, Fiifi Baffoe Payin Winful, Jamal-Deen Abdulai  

**Link**: [PDF](https://arxiv.org/pdf/2602.05406)  

**Abstract**: The lack of impaired speech data hinders advancements in the development of inclusive speech technologies, particularly in low-resource languages such as Akan. To address this gap, this study presents a curated corpus of speech samples from native Akan speakers with speech impairment. The dataset comprises of 50.01 hours of audio recordings cutting across four classes of impaired speech namely stammering, cerebral palsy, cleft palate, and stroke induced speech disorder. Recordings were done in controlled supervised environments were participants described pre-selected images in their own words. The resulting dataset is a collection of audio recordings, transcriptions, and associated metadata on speaker demographics, class of impairment, recording environment and device. The dataset is intended to support research in low-resource automatic disordered speech recognition systems and assistive speech technology. 

---
# Optimal Bayesian Stopping for Efficient Inference of Consistent LLM Answers 

**Authors**: Jingkai Huang, Will Ma, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.05395)  

**Abstract**: A simple strategy for improving LLM accuracy, especially in math and reasoning problems, is to sample multiple responses and submit the answer most consistently reached. In this paper we leverage Bayesian prior information to save on sampling costs, stopping once sufficient consistency is reached. Although the exact posterior is computationally intractable, we further introduce an efficient "L-aggregated" stopping policy that tracks only the L-1 most frequent answer counts. Theoretically, we prove that L=3 is all you need: this coarse approximation is sufficient to achieve asymptotic optimality, and strictly dominates prior-free baselines, while having a fast posterior computation. Empirically, this identifies the most consistent (i.e., mode) LLM answer using fewer samples, and can achieve similar answer accuracy while cutting the number of LLM calls (i.e., saving on LLM inference costs) by up to 50%. 

---
# Beyond Length: Context-Aware Expansion and Independence as Developmentally Sensitive Evaluation in Child Utterances 

**Authors**: Jiyun Chun, Eric Fosler-Lussier, Michael White, Andrew Perrault  

**Link**: [PDF](https://arxiv.org/pdf/2602.05392)  

**Abstract**: Evaluating the quality of children's utterances in adult-child dialogue remains challenging due to insufficient context-sensitive metrics. Common proxies such as Mean Length of Utterance (MLU), lexical diversity (vocd-D), and readability indices (Flesch-Kincaid Grade Level, Gunning Fog Index) are dominated by length and ignore conversational context, missing aspects of response quality such as reasoning depth, topic maintenance, and discourse planning. We introduce an LLM-as-a-judge framework that first classifies the Previous Adult Utterance Type and then scores the child's response along two axes: Expansion (contextual elaboration and inferential depth) and Independence (the child's contribution to advancing the discourse). These axes reflect fundamental dimensions in child language development, where Expansion captures elaboration, clause combining, and causal and contrastive connectives. Independence captures initiative, topic control, decreasing reliance on adult scaffolding through growing self-regulation, and audience design. We establish developmental validity by showing age-related patterns and demonstrate predictive value by improving age estimation over common baselines. We further confirm semantic sensitivity by detecting differences tied to discourse relations. Our metrics align with human judgments, enabling large-scale evaluation. This shifts child utterance assessment from simply measuring length to evaluating how meaningfully the child's speech contributes to and advances the conversation within its context. 

---
# Assessing Electricity Demand Forecasting with Exogenous Data in Time Series Foundation Models 

**Authors**: Wei Soon Cheong, Lian Lian Jiang, Jamie Ng Suat Ling  

**Link**: [PDF](https://arxiv.org/pdf/2602.05390)  

**Abstract**: Time-series foundation models have emerged as a new paradigm for forecasting, yet their ability to effectively leverage exogenous features -- critical for electricity demand forecasting -- remains unclear. This paper empirically evaluates foundation models capable of modeling cross-channel correlations against a baseline LSTM with reversible instance normalization across Singaporean and Australian electricity markets at hourly and daily granularities. We systematically assess MOIRAI, MOMENT, TinyTimeMixers, ChronosX, and Chronos-2 under three feature configurations: all features, selected features, and target-only. Our findings reveal highly variable effectiveness: while Chronos-2 achieves the best performance among foundation models (in zero-shot settings), the simple baseline frequently outperforms all foundation models in Singapore's stable climate, particularly for short-term horizons. Model architecture proves critical, with synergistic architectural implementations (TTM's channel-mixing, Chronos-2's grouped attention) consistently leveraging exogenous features, while other approaches show inconsistent benefits. Geographic context emerges as equally important, with foundation models demonstrating advantages primarily in variable climates. These results challenge assumptions about universal foundation model superiority and highlight the need for domain-specific models, specifically in the energy domain. 

---
# Spider-Sense: Intrinsic Risk Sensing for Efficient Agent Defense with Hierarchical Adaptive Screening 

**Authors**: Zhenxiong Yu, Zhi Yang, Zhiheng Jin, Shuhe Wang, Heng Zhang, Yanlin Fei, Lingfeng Zeng, Fangqi Lou, Shuo Zhang, Tu Hu, Jingping Liu, Rongze Chen, Xingyu Zhu, Kunyi Wang, Chaofa Yuan, Xin Guo, Zhaowei Liu, Feipeng Zhang, Jie Huang, Huacan Wang, Ronghao Chen, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05386)  

**Abstract**: As large language models (LLMs) evolve into autonomous agents, their real-world applicability has expanded significantly, accompanied by new security challenges. Most existing agent defense mechanisms adopt a mandatory checking paradigm, in which security validation is forcibly triggered at predefined stages of the agent lifecycle. In this work, we argue that effective agent security should be intrinsic and selective rather than architecturally decoupled and mandatory. We propose Spider-Sense framework, an event-driven defense framework based on Intrinsic Risk Sensing (IRS), which allows agents to maintain latent vigilance and trigger defenses only upon risk perception. Once triggered, the Spider-Sense invokes a hierarchical defence mechanism that trades off efficiency and precision: it resolves known patterns via lightweight similarity matching while escalating ambiguous cases to deep internal reasoning, thereby eliminating reliance on external models. To facilitate rigorous evaluation, we introduce S$^2$Bench, a lifecycle-aware benchmark featuring realistic tool execution and multi-stage attacks. Extensive experiments demonstrate that Spider-Sense achieves competitive or superior defense performance, attaining the lowest Attack Success Rate (ASR) and False Positive Rate (FPR), with only a marginal latency overhead of 8.3\%. 

---
# GAS: Enhancing Reward-Cost Balance of Generative Model-assisted Offline Safe RL 

**Authors**: Zifan Liu, Xinran Li, Shibo Chen, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05323)  

**Abstract**: Offline Safe Reinforcement Learning (OSRL) aims to learn a policy to achieve high performance in sequential decision-making while satisfying constraints, using only pre-collected datasets. Recent works, inspired by the strong capabilities of Generative Models (GMs), reformulate decision-making in OSRL as a conditional generative process, where GMs generate desirable actions conditioned on predefined reward and cost values. However, GM-assisted methods face two major challenges in OSRL: (1) lacking the ability to "stitch" optimal transitions from suboptimal trajectories within the dataset, and (2) struggling to balance reward targets with cost targets, particularly when they are conflict. To address these issues, we propose Goal-Assisted Stitching (GAS), a novel algorithm designed to enhance stitching capabilities while effectively balancing reward maximization and constraint satisfaction. To enhance the stitching ability, GAS first augments and relabels the dataset at the transition level, enabling the construction of high-quality trajectories from suboptimal ones. GAS also introduces novel goal functions, which estimate the optimal achievable reward and cost goals from the dataset. These goal functions, trained using expectile regression on the relabeled and augmented dataset, allow GAS to accommodate a broader range of reward-cost return pairs and achieve a better tradeoff between reward maximization and constraint satisfaction compared to human-specified values. The estimated goals then guide policy training, ensuring robust performance under constrained settings. Furthermore, to improve training stability and efficiency, we reshape the dataset to achieve a more uniform reward-cost return distribution. Empirical results validate the effectiveness of GAS, demonstrating superior performance in balancing reward maximization and constraint satisfaction compared to existing methods. 

---
# Formal Synthesis of Certifiably Robust Neural Lyapunov-Barrier Certificates 

**Authors**: Chengxiao Wang, Haoze Wu, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2602.05311)  

**Abstract**: Neural Lyapunov and barrier certificates have recently been used as powerful tools for verifying the safety and stability properties of deep reinforcement learning (RL) controllers. However, existing methods offer guarantees only under fixed ideal unperturbed dynamics, limiting their reliability in real-world applications where dynamics may deviate due to uncertainties. In this work, we study the problem of synthesizing \emph{robust neural Lyapunov barrier certificates} that maintain their guarantees under perturbations in system dynamics. We formally define a robust Lyapunov barrier function and specify sufficient conditions based on Lipschitz continuity that ensure robustness against bounded perturbations. We propose practical training objectives that enforce these conditions via adversarial training, Lipschitz neighborhood bound, and global Lipschitz regularization. We validate our approach in two practically relevant environments, Inverted Pendulum and 2D Docking. The former is a widely studied benchmark, while the latter is a safety-critical task in autonomous systems. We show that our methods significantly improve both certified robustness bounds (up to $4.6$ times) and empirical success rates under strong perturbations (up to $2.4$ times) compared to the baseline. Our results demonstrate effectiveness of training robust neural certificates for safe RL under perturbations in dynamics. 

---
# FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion 

**Authors**: Zhuokun Chen, Jianfei Cai, Bohan Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05305)  

**Abstract**: Generating long-form content, such as minute-long videos and extended texts, is increasingly important for modern generative models. Block diffusion improves inference efficiency via KV caching and block-wise causal inference and has been widely adopted in diffusion language models and video generation. However, in long-context settings, block diffusion still incurs substantial overhead from repeatedly computing attention over a growing KV cache. We identify an underexplored property of block diffusion: cross-step redundancy of attention within a block. Our analysis shows that attention outputs from tokens outside the current block remain largely stable across diffusion steps, while block-internal attention varies significantly. Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that reuses stable attention output, reducing attention computation and KV cache access without modifying the diffusion process. Moreover, FlashBlock is orthogonal to sparse attention and can be combined as a complementary residual reuse strategy, substantially improving model accuracy under aggressive sparsification. Experiments on diffusion language models and video generation demonstrate up to 1.44$\times$ higher token throughput and up to 1.6$\times$ reduction in attention time, with negligible impact on generation quality. Project page: this https URL. 

---
# Towards a Science of Collective AI: LLM-based Multi-Agent Systems Need a Transition from Blind Trial-and-Error to Rigorous Science 

**Authors**: Jingru Fan, Dewen Liu, Yufan Dang, Huatao Li, Yuheng Wang, Wei Liu, Feiyu Duan, Xuanwen Ding, Shu Yao, Lin Wu, Ruijie Shi, Wai-Shing Leung, Yuan Cheng, Zhongyu Wei, Cheng Yang, Chen Qian, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.05289)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have greatly extended the capabilities of Multi-Agent Systems (MAS), demonstrating significant effectiveness across a wide range of complex and open-ended domains. However, despite this rapid progress, the field still relies heavily on empirical trial-and-error. It lacks a unified and principled scientific framework necessary for systematic optimization and improvement. This bottleneck stems from the ambiguity of attribution: first, the absence of a structured taxonomy of factors leaves researchers restricted to unguided adjustments; second, the lack of a unified metric fails to distinguish genuine collaboration gain from mere resource accumulation. In this paper, we advocate for a transition to design science through an integrated framework. We advocate to establish the collaboration gain metric ($\Gamma$) as the scientific standard to isolate intrinsic gains from increased budgets. Leveraging $\Gamma$, we propose a factor attribution paradigm to systematically identify collaboration-driving factors. To support this, we construct a systematic MAS factor library, structuring the design space into control-level presets and information-level dynamics. Ultimately, this framework facilitates the transition from blind experimentation to rigorous science, paving the way towards a true science of Collective AI. 

---
# HealthMamba: An Uncertainty-aware Spatiotemporal Graph State Space Model for Effective and Reliable Healthcare Facility Visit Prediction 

**Authors**: Dahai Yu, Lin Jiang, Rongchao Xu, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05286)  

**Abstract**: Healthcare facility visit prediction is essential for optimizing healthcare resource allocation and informing public health policy. Despite advanced machine learning methods being employed for better prediction performance, existing works usually formulate this task as a time-series forecasting problem without considering the intrinsic spatial dependencies of different types of healthcare facilities, and they also fail to provide reliable predictions under abnormal situations such as public emergencies. To advance existing research, we propose HealthMamba, an uncertainty-aware spatiotemporal framework for accurate and reliable healthcare facility visit prediction. HealthMamba comprises three key components: (i) a Unified Spatiotemporal Context Encoder that fuses heterogeneous static and dynamic information, (ii) a novel Graph State Space Model called GraphMamba for hierarchical spatiotemporal modeling, and (iii) a comprehensive uncertainty quantification module integrating three uncertainty quantification mechanisms for reliable prediction. We evaluate HealthMamba on four large-scale real-world datasets from California, New York, Texas, and Florida. Results show HealthMamba achieves around 6.0% improvement in prediction accuracy and 3.5% improvement in uncertainty quantification over state-of-the-art baselines. 

---
# Hybrid Gated Flow (HGF): Stabilizing 1.58-bit LLMs via Selective Low-Rank Correction 

**Authors**: David Alejandro Trejo Pizzo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05269)  

**Abstract**: The deployment of Large Language Models (LLMs) on edge devices is fundamentally constrained by the "Memory Wall" -- a hardware limitation where memory bandwidth, not compute, becomes the bottleneck. Recent 1.58-bit quantization techniques (e.g., BitNet b1.58) dramatically reduce memory footprint but typically incur a perplexity degradation of 20-25% compared to FP16 baselines. In this work, we introduce Hybrid Gated Flow (HGF), a dual-stream architecture that couples a 1.58-bit ternary backbone with a learnable, low-rank FP16 correction path controlled by adaptive gates.
Through extensive experiments on the TinyStories dataset across two training regimes (2500 and 3500 steps), we demonstrate that HGF 5.4 achieves a validation loss of 0.9306 compared to BitNet's 1.0294, recovering approximately 55% of the quality gap between pure ternary quantization and the FP16 baseline (0.8490). This recovery is achieved with only ~12-15% memory overhead beyond the ternary backbone.
Furthermore, we provide empirical evidence for an emergent phenomenon: quantization as structural regularization. While a full-precision differential attention baseline (Diff_Only) exhibited training instability with validation loss exceeding 1.68, the ternary-anchored HGF maintained robust convergence throughout training. Finally, we report preliminary results extending this architecture to 1.2B and 3B parameter models trained on SlimPajama and FineWeb-Edu. These larger-scale experiments confirm that the architectural stability and quality recovery observed in small-scale proxies scale linearly to production-grade language modeling regimes. 

---
# CoPE: Clipped RoPE as A Scalable Free Lunch for Long Context LLMs 

**Authors**: Haoran Li, Sucheng Ren, Alan Yuille, Feng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05258)  

**Abstract**: Rotary Positional Embedding (RoPE) is a key component of context scaling in Large Language Models (LLMs). While various methods have been proposed to adapt RoPE to longer contexts, their guiding principles generally fall into two categories: (1) out-of-distribution (OOD) mitigation, which scales RoPE frequencies to accommodate unseen positions, and (2) Semantic Modeling, which posits that the attention scores computed with RoPE should always prioritize semantically similar tokens. In this work, we unify these seemingly distinct objectives through a minimalist intervention, namely CoPE: soft clipping lowfrequency components of RoPE. CoPE not only eliminates OOD outliers and refines semantic signals, but also prevents spectral leakage caused by hard clipping. Extensive experiments demonstrate that simply applying our soft clipping strategy to RoPE yields significant performance gains that scale up to 256k context length, validating our theoretical analysis and establishing CoPE as a new state-of-the-art for length generalization. Our code, data, and models are available at this https URL. 

---
# EGSS: Entropy-guided Stepwise Scaling for Reliable Software Engineering 

**Authors**: Chenhui Mao, Yuanting Lei, Zhixiang Wei, Ming Liang, Zhixiang Wang, Jingxuan Xu, Dajun Chen, Wei Jiang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05242)  

**Abstract**: Agentic Test-Time Scaling (TTS) has delivered state-of-the-art (SOTA) performance on complex software engineering tasks such as code generation and bug fixing. However, its practical adoption remains limited due to significant computational overhead, primarily driven by two key challenges: (1) the high cost associated with deploying excessively large ensembles, and (2) the lack of a reliable mechanism for selecting the optimal candidate solution, ultimately constraining the performance gains that can be realized. To address these challenges, we propose Entropy-Guided Stepwise Scaling (EGSS), a novel TTS framework that dynamically balances efficiency and effectiveness through entropy-guided adaptive search and robust test-suite augmentation. Extensive experiments on SWE-Bench-Verified demonstrate that EGSS consistently boosts performance by 5-10% across all evaluated models. Specifically, it increases the resolved ratio of Kimi-K2-Intruct from 63.2% to 72.2%, and GLM-4.6 from 65.8% to 74.6%. Furthermore, when paired with GLM-4.6, EGSS achieves a new state-of-the-art among open-source large language models. In addition to these accuracy improvements, EGSS reduces inference-time token usage by over 28% compared to existing TTS methods, achieving simultaneous gains in both effectiveness and computational efficiency. 

---
# Balanced Anomaly-guided Ego-graph Diffusion Model for Inductive Graph Anomaly Detection 

**Authors**: Chunyu Wei, Siyuan He, Yu Wang, Yueguo Chen, Yunhai Wang, Bing Bai, Yidong Zhang, Yong Xie, Shunming Zhang, Fei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05232)  

**Abstract**: Graph anomaly detection (GAD) is crucial in applications like fraud detection and cybersecurity. Despite recent advancements using graph neural networks (GNNs), two major challenges persist. At the model level, most methods adopt a transductive learning paradigm, which assumes static graph structures, making them unsuitable for dynamic, evolving networks. At the data level, the extreme class imbalance, where anomalous nodes are rare, leads to biased models that fail to generalize to unseen anomalies. These challenges are interdependent: static transductive frameworks limit effective data augmentation, while imbalance exacerbates model distortion in inductive learning settings. To address these challenges, we propose a novel data-centric framework that integrates dynamic graph modeling with balanced anomaly synthesis. Our framework features: (1) a discrete ego-graph diffusion model, which captures the local topology of anomalies to generate ego-graphs aligned with anomalous structural distribution, and (2) a curriculum anomaly augmentation mechanism, which dynamically adjusts synthetic data generation during training, focusing on underrepresented anomaly patterns to improve detection and generalization. Experiments on five datasets demonstrate that the effectiveness of our framework. 

---
# ZeroS: Zero-Sum Linear Attention for Efficient Transformers 

**Authors**: Jiecheng Lu, Xu Han, Yan Sun, Viresh Pati, Yubin Kim, Siddhartha Somani, Shihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05230)  

**Abstract**: Linear attention methods offer Transformers $O(N)$ complexity but typically underperform standard softmax attention. We identify two fundamental limitations affecting these approaches: the restriction to convex combinations that only permits additive information blending, and uniform accumulated weight bias that dilutes attention in long contexts. We propose Zero-Sum Linear Attention (ZeroS), which addresses these limitations by removing the constant zero-order term $1/t$ and reweighting the remaining zero-sum softmax residuals. This modification creates mathematically stable weights, enabling both positive and negative values and allowing a single attention layer to perform contrastive operations. While maintaining $O(N)$ complexity, ZeroS theoretically expands the set of representable functions compared to convex combinations. Empirically, it matches or exceeds standard softmax attention across various sequence modeling benchmarks. 

---
# Semantic Search over 9 Million Mathematical Theorems 

**Authors**: Luke Alexander, Eric Leonen, Sophie Szeto, Artemii Remizov, Ignacio Tejeda, Giovanni Inchiostro, Vasily Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2602.05216)  

**Abstract**: Searching for mathematical results remains difficult: most existing tools retrieve entire papers, while mathematicians and theorem-proving agents often seek a specific theorem, lemma, or proposition that answers a query. While semantic search has seen rapid progress, its behavior on large, highly technical corpora such as research-level mathematical theorems remains poorly understood. In this work, we introduce and study semantic theorem retrieval at scale over a unified corpus of $9.2$ million theorem statements extracted from arXiv and seven other sources, representing the largest publicly available corpus of human-authored, research-level theorems. We represent each theorem with a short natural-language description as a retrieval representation and systematically analyze how representation context, language model choice, embedding model, and prompting strategy affect retrieval quality. On a curated evaluation set of theorem-search queries written by professional mathematicians, our approach substantially improves both theorem-level and paper-level retrieval compared to existing baselines, demonstrating that semantic theorem search is feasible and effective at web scale. The theorem search tool is available at \href{this https URL}{this link}, and the dataset is available at \href{this https URL}{this link}. 

---
# ARCHI-TTS: A flow-matching-based Text-to-Speech Model with Self-supervised Semantic Aligner and Accelerated Inference 

**Authors**: Chunyat Wu, Jiajun Deng, Zhengxi Liu, Zheqi Dai, Haolin He, Qiuqiang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2602.05207)  

**Abstract**: Although diffusion-based, non-autoregressive text-to-speech (TTS) systems have demonstrated impressive zero-shot synthesis capabilities, their efficacy is still hindered by two key challenges: the difficulty of text-speech alignment modeling and the high computational overhead of the iterative denoising process. To address these limitations, we propose ARCHI-TTS that features a dedicated semantic aligner to ensure robust temporal and semantic consistency between text and audio. To overcome high computational inference costs, ARCHI-TTS employs an efficient inference strategy that reuses encoder features across denoising steps, drastically accelerating synthesis without performance degradation. An auxiliary CTC loss applied to the condition encoder further enhances the semantic understanding. Experimental results demonstrate that ARCHI-TTS achieves a WER of 1.98% on LibriSpeech-PC test-clean, and 1.47%/1.42% on SeedTTS test-en/test-zh with a high inference efficiency, consistently outperforming recent state-of-the-art TTS systems. 

---
# Aligning Large Language Model Behavior with Human Citation Preferences 

**Authors**: Kenichiro Ando, Tatsuya Harada  

**Link**: [PDF](https://arxiv.org/pdf/2602.05205)  

**Abstract**: Most services built on powerful large-scale language models (LLMs) add citations to their output to enhance credibility. Recent research has paid increasing attention to the question of what reference documents to link to outputs. However, how LLMs recognize cite-worthiness and how this process should be controlled remains underexplored. In this study, we focus on what kinds of content LLMs currently tend to cite and how well that behavior aligns with human preferences. We construct a dataset to characterize the relationship between human citation preferences and LLM behavior. Web-derived texts are categorized into eight citation-motivation types, and pairwise citation preferences are exhaustively evaluated across all type combinations to capture fine-grained contrasts. Our results show that humans most frequently seek citations for medical text, and stronger models display a similar tendency. We also find that current models are as much as $27\%$ more likely than humans to add citations to text that is explicitly marked as needing citations on sources such as Wikipedia, and this overemphasis reduces alignment accuracy. Conversely, models systematically underselect numeric sentences (by $-22.6\%$ relative to humans) and sentences containing personal names (by $-20.1\%$), categories for which humans typically demand citations. Furthermore, experiments with Direct Preference Optimization demonstrate that model behavior can be calibrated to better match human citation preferences. We expect this study to provide a foundation for more fine-grained investigations into LLM citation preferences. 

---
# Double-P: Hierarchical Top-P Sparse Attention for Long-Context LLMs 

**Authors**: Wentao Ni, Kangqi Zhang, Zhongming Yu, Oren Nelson, Mingu Lee, Hong Cai, Fatih Porikli, Jongryool Kim, Zhijian Liu, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.05191)  

**Abstract**: As long-context inference becomes central to large language models (LLMs), attention over growing key-value caches emerges as a dominant decoding bottleneck, motivating sparse attention for scalable inference. Fixed-budget top-k sparse attention cannot adapt to heterogeneous attention distributions across heads and layers, whereas top-p sparse attention directly preserves attention mass and provides stronger accuracy guarantees. Existing top-p methods, however, fail to jointly optimize top-p accuracy, selection overhead, and sparse attention cost, which limits their overall efficiency. We present Double-P, a hierarchical sparse attention framework that optimizes all three stages. Double-P first performs coarse-grained top-p estimation at the cluster level using size-weighted centroids, then adaptively refines computation through a second top-p stage that allocates token-level attention only when needed. Across long-context benchmarks, Double-P consistently achieves near-zero accuracy drop, reducing attention computation overhead by up to 1.8x and delivers up to 1.3x end-to-end decoding speedup over state-of-the-art fixed-budget sparse attention methods. 

---
# Towards Worst-Case Guarantees with Scale-Aware Interpretability 

**Authors**: Lauren Greenspan, David Berman, Aryeh Brill, Ro Jefferson, Artemy Kolchinsky, Jennifer Lin, Andrew Mack, Anindita Maiti, Fernando E. Rosas, Alexander Stapleton, Lucas Teixeira, Dmitry Vaintrob  

**Link**: [PDF](https://arxiv.org/pdf/2602.05184)  

**Abstract**: Neural networks organize information according to the hierarchical, multi-scale structure of natural data. Methods to interpret model internals should be similarly scale-aware, explicitly tracking how features compose across resolutions and guaranteeing bounds on the influence of fine-grained structure that is discarded as irrelevant noise. We posit that the renormalisation framework from physics can meet this need by offering technical tools that can overcome limitations of current methods. Moreover, relevant work from adjacent fields has now matured to a point where scattered research threads can be synthesized into practical, theory-informed tools. To combine these threads in an AI safety context, we propose a unifying research agenda -- \emph{scale-aware interpretability} -- to develop formal machinery and interpretability tools that have robustness and faithfulness properties supported by statistical physics. 

---
# Data-Centric Interpretability for LLM-based Multi-Agent Reinforcement Learning 

**Authors**: John Yan, Michael Yu, Yuqi Sun, Alexander Duffy, Tyler Marques, Matthew Lyle Olson  

**Link**: [PDF](https://arxiv.org/pdf/2602.05183)  

**Abstract**: Large language models (LLMs) are increasingly trained in complex Reinforcement Learning, multi-agent environments, making it difficult to understand how behavior changes over training. Sparse Autoencoders (SAEs) have recently shown to be useful for data-centric interpretability. In this work, we analyze large-scale reinforcement learning training runs from the sophisticated environment of Full-Press Diplomacy by applying pretrained SAEs, alongside LLM-summarizer methods. We introduce Meta-Autointerp, a method for grouping SAE features into interpretable hypotheses about training dynamics. We discover fine-grained behaviors including role-playing patterns, degenerate outputs, language switching, alongside high-level strategic behaviors and environment-specific bugs. Through automated evaluation, we validate that 90% of discovered SAE Meta-Features are significant, and find a surprising reward hacking behavior. However, through two user studies, we find that even subjectively interesting and seemingly helpful SAE features may be worse than useless to humans, along with most LLM generated hypotheses. However, a subset of SAE-derived hypotheses are predictively useful for downstream tasks. We further provide validation by augmenting an untrained agent's system prompt, improving the score by +14.2%. Overall, we show that SAEs and LLM-summarizer provide complementary views into agent behavior, and together our framework forms a practical starting point for future data-centric interpretability work on ensuring trustworthy LLM behavior throughout training. 

---
# Benchmarking Artificial Intelligence Models for Daily Coastal Hypoxia Forecasting 

**Authors**: Magesh Rajasekaran, Md Saiful Sajol, Chris Alvin, Supratik Mukhopadhyay, Yanda Ou, Z. George Xue  

**Link**: [PDF](https://arxiv.org/pdf/2602.05178)  

**Abstract**: Coastal hypoxia, especially in the northern part of Gulf of Mexico, presents a persistent ecological and economic concern. Seasonal models offer coarse forecasts that miss the fine-scale variability needed for daily, responsive ecosystem management. We present study that compares four deep learning architectures for daily hypoxia classification: Bidirectional Long Short-Term Memory (BiLSTM), Medformer (Medical Transformer), Spatio-Temporal Transformer (ST-Transformer), and Temporal Convolutional Network (TCN). We trained our models with twelve years of daily hindcast data from 2009-2020 Our training data consists of 2009-2020 hindcast data from a coupled hydrodynamic-biogeochemical model. Similarly, we use hindcast data from 2020 through 2024 as a test data. We constructed classification models incorporating water column stratification, sediment oxygen consumption, and temperature-dependent decomposition rates. We evaluated each architectures using the same data preprocessing, input/output formulation, and validation protocols. Each model achieved high classification accuracy and strong discriminative ability with ST-Transformer achieving the highest performance across all metrics and tests periods (AUC-ROC: 0.982-0.992). We also employed McNemar's method to identify statistically significant differences in model predictions. Our contribution is a reproducible framework for operational real-time hypoxia prediction that can support broader efforts in the environmental and ocean modeling systems community and in ecosystem resilience. The source code is available this https URL 

---
# Total Variation Rates for Riemannian Flow Matching 

**Authors**: Yunrui Guan, Krishnakumar Balasubramanian, Shiqian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.05174)  

**Abstract**: Riemannian flow matching (RFM) extends flow-based generative modeling to data supported on manifolds by learning a time-dependent tangent vector field whose flow-ODE transports a simple base distribution to the data law. We develop a nonasymptotic Total Variation (TV) convergence analysis for RFM samplers that use a learned vector field together with Euler discretization on manifolds. Our key technical ingredient is a differential inequality governing the evolution of TV between two manifold ODE flows, which expresses the time-derivative of TV through the divergence of the vector-field mismatch and the score of the reference flow; controlling these terms requires establishing new bounds that explicitly account for parallel transport and curvature. Under smoothness assumptions on the population flow-matching field and either uniform (compact manifolds) or mean-square (Hadamard manifolds) approximation guarantees for the learned field, we obtain explicit bounds of the form $\mathrm{TV}\le C_{\mathrm{Lip}}\,h + C_{\varepsilon}\,\varepsilon$ (with an additional higher-order $\varepsilon^2$ term on compact manifolds), cleanly separating numerical discretization and learning errors. Here, $h$ is the step-size and $\varepsilon$ is the target accuracy. Instantiations yield \emph{explicit} polynomial iteration complexities on the hypersphere $S^d$, and on the SPD$(n)$ manifolds under mild moment conditions. 

---
# EBPO: Empirical Bayes Shrinkage for Stabilizing Group-Relative Policy Optimization 

**Authors**: Kevin Han, Yuhang Zhou, Mingze Gao, Gedi Zhou, Serena Li, Abhishek Kumar, Xiangjun Fan, Weiwei Li, Lizhu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05165)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing the reasoning capabilities of Large Language Models (LLMs). However, dominant approaches like Group Relative Policy Optimization (GRPO) face critical stability challenges: they suffer from high estimator variance under computational constraints (small group sizes) and vanishing gradient signals in saturated failure regimes where all responses yield identical zero rewards. To address this, we propose Empirical Bayes Policy Optimization (EBPO), a novel framework that regularizes local group-based baselines by borrowing strength from the policy's accumulated global statistics. Instead of estimating baselines in isolation, EBPO employs a shrinkage estimator that dynamically balances local group statistics with a global prior updated via Welford's online algorithm. Theoretically, we demonstrate that EBPO guarantees strictly lower Mean Squared Error (MSE), bounded entropy decay, and non-vanishing penalty signals in failure scenarios compared to GRPO. Empirically, EBPO consistently outperforms GRPO and other established baselines across diverse benchmarks, including AIME and OlympiadBench. Notably, EBPO exhibits superior training stability, achieving high-performance gains even with small group sizes, and benefits significantly from difficulty-stratified curriculum learning. 

---
# Position: Capability Control Should be a Separate Goal From Alignment 

**Authors**: Shoaib Ahmed Siddiqui, Eleni Triantafillou, David Krueger, Adrian Weller  

**Link**: [PDF](https://arxiv.org/pdf/2602.05164)  

**Abstract**: Foundation models are trained on broad data distributions, yielding generalist capabilities that enable many downstream applications but also expand the space of potential misuse and failures. This position paper argues that capability control -- imposing restrictions on permissible model behavior -- should be treated as a distinct goal from alignment. While alignment is often context and preference-driven, capability control aims to impose hard operational limits on permissible behaviors, including under adversarial elicitation. We organize capability control mechanisms across the model lifecycle into three layers: (i) data-based control of the training distribution, (ii) learning-based control via weight- or representation-level interventions, and (iii) system-based control via post-deployment guardrails over inputs, outputs, and actions. Because each layer has characteristic failure modes when used in isolation, we advocate for a defense-in-depth approach that composes complementary controls across the full stack. We further outline key open challenges in achieving such control, including the dual-use nature of knowledge and compositional generalization. 

---
# CoSA: Compressed Sensing-Based Adaptation of Large Language Models 

**Authors**: Songtao Wei, Yi Li, Bohan Zhang, Zhichun Guo, Ying Huang, Yuede Ji, Miao Yin, Guanpeng Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05148)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has emerged as a practical paradigm for adapting large language models (LLMs) without updating all parameters. Most existing approaches, such as LoRA and PiSSA, rely on low-rank decompositions of weight updates. However, the low-rank assumption may restrict expressivity, particularly in task-specific adaptation scenarios where singular values are distributed relatively uniformly. To address this limitation, we propose CoSA (Compressed Sensing-Based Adaptation), a new PEFT method extended from compressed sensing theory. Instead of constraining weight updates to a low-rank subspace, CoSA expresses them through fixed random projection matrices and a compact learnable core. We provide a formal theoretical analysis of CoSA as a synthesis process, proving that weight updates can be compactly encoded into a low-dimensional space and mapped back through random projections. Extensive experimental results show that CoSA provides a principled perspective for efficient and expressive multi-scale model adaptation. Specifically, we evaluate CoSA on 10 diverse tasks, including natural language understanding and generation, employing 5 models of different scales from RoBERTa, Llama, and Qwen families. Across these settings, CoSA consistently matches or outperforms state-of-the-art PEFT methods. 

---
# Cross-talk based multi-task learning for fault classification of physically coupled machine system 

**Authors**: Wonjun Yi, Rismaya Kumar Mishra, Yong-Hwa Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.05146)  

**Abstract**: Machine systems inherently generate signals in which fault conditions and various physical variables are physically coupled. Although many existing fault classification studies rely solely on direct fault labels, the aforementioned signals naturally embed additional information shaped by other physically coupled information. Herein, we leverage this coupling through a multi-task learning (MTL) framework that jointly learns fault conditions and the related physical variables. Among MTL architectures, crosstalk structures have distinct advantages because they allow for controlled information exchange between tasks through the cross-talk layer while preventing negative transfer, in contrast to shared trunk architectures that often mix incompatible features. We build on our previously introduced residual neural dimension reductor model, and extend its application to two benchmarks where physical coupling is prominent. The first benchmark is a drone fault dataset, in which machine type and maneuvering direction significantly alter the frequency components of measured signals even under the same nominal condition. By learning fault classification together with these physical attributes, the cross-talk architecture can better classify faults. The second benchmark dataset is the motor compound fault dataset. In this system, each fault component, inner race fault, outer race fault, misalignment, and unbalance is coupled to the other. For motor compound fault, we also test classification performance when we use single-channel data or multi-channel data as input to the classifier. Across both benchmarks, our residual neural dimension reductor, consistently outperformed single-task models, multi-class models that merge all label combinations, and shared trunk multi-task models. 

---
# TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference 

**Authors**: Jiyoung Park, Hankyu Jang, Changseok Song, Wookeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2602.05145)  

**Abstract**: Speculative decoding can substantially accelerate LLM inference, but realizing its benefits in practice is challenging due to evolving workloads and system-level constraints. We present TIDE (Temporal Incremental Draft Engine), a serving-engine-native framework that integrates online draft adaptation directly into high-performance LLM inference systems. TIDE reuses target model hidden states generated during inference as training signals, enabling zero-overhead draft adaptation without reloading the target model, and employs adaptive runtime control to activate speculation and training only when beneficial. TIDE exploits heterogeneous clusters by mapping decoupled inference and training to appropriate GPU classes. Across diverse real-world workloads, TIDE achieves up to 1.15x throughput improvement over static speculative decoding while reducing draft training time by 1.67x compared to approaches that recompute training signals. 

---
# Rethinking Rubric Generation for Improving LLM Judge and Reward Modeling for Open-ended Tasks 

**Authors**: William F. Shen, Xinchi Qiu, Chenxi Whitehouse, Lisa Alazraki, Shashwat Goel, Francesco Barbieri, Timon Willi, Akhil Mathur, Ilias Leontiadis  

**Link**: [PDF](https://arxiv.org/pdf/2602.05125)  

**Abstract**: Recently, rubrics have been used to guide LLM judges in capturing subjective, nuanced, multi-dimensional human preferences, and have been extended from evaluation to reward signals for reinforcement fine-tuning (RFT). However, rubric generation remains hard to control: rubrics often lack coverage, conflate dimensions, misalign preference direction, and contain redundant or highly correlated criteria, degrading judge accuracy and producing suboptimal rewards during RFT. We propose RRD, a principled framework for rubric refinement built on a recursive decompose-filter cycle. RRD decomposes coarse rubrics into fine-grained, discriminative criteria, expanding coverage while sharpening separation between responses. A complementary filtering mechanism removes misaligned and redundant rubrics, and a correlation-aware weighting scheme prevents over-representing highly correlated criteria, yielding rubric sets that are informative, comprehensive, and non-redundant. Empirically, RRD delivers large, consistent gains across both evaluation and training: it improves preference-judgment accuracy on JudgeBench and PPE for both GPT-4o and Llama3.1-405B judges, achieving top performance in all settings with up to +17.7 points on JudgeBench. When used as the reward source for RFT on WildChat, it yields substantially stronger and more stable learning signals, boosting reward by up to 160% (Qwen3-4B) and 60% (Llama3.1-8B) versus 10-20% for prior rubric baselines, with gains that transfer to HealthBench-Hard and BiGGen Bench. Overall, RRD establishes recursive rubric refinement as a scalable and interpretable foundation for LLM judging and reward modeling in open-ended domains. 

---
# Autodiscover: A reinforcement learning recommendation system for the cold-start imbalance challenge in active learning, powered by graph-aware thompson sampling 

**Authors**: Parsa Vares  

**Link**: [PDF](https://arxiv.org/pdf/2602.05087)  

**Abstract**: Systematic literature reviews (SLRs) are fundamental to evidence-based research, but manual screening is an increasing bottleneck as scientific output grows. Screening features low prevalence of relevant studies and scarce, costly expert decisions. Traditional active learning (AL) systems help, yet typically rely on fixed query strategies for selecting the next unlabeled documents. These static strategies do not adapt over time and ignore the relational structure of scientific literature networks. This thesis introduces AutoDiscover, a framework that reframes AL as an online decision-making problem driven by an adaptive agent. Literature is modeled as a heterogeneous graph capturing relationships among documents, authors, and metadata. A Heterogeneous Graph Attention Network (HAN) learns node representations, which a Discounted Thompson Sampling (DTS) agent uses to dynamically manage a portfolio of query strategies. With real-time human-in-the-loop labels, the agent balances exploration and exploitation under non-stationary review dynamics, where strategy utility changes over time. On the 26-dataset SYNERGY benchmark, AutoDiscover achieves higher screening efficiency than static AL baselines. Crucially, the agent mitigates cold start by bootstrapping discovery from minimal initial labels where static approaches fail. We also introduce TS-Insight, an open-source visual analytics dashboard to interpret, verify, and diagnose the agent's decisions. Together, these contributions accelerate SLR screening under scarce expert labels and low prevalence of relevant studies. 

---
# Individual Fairness In Strategic Classification 

**Authors**: Zhiqun Zuo, Mohammad Mahdi Khalili  

**Link**: [PDF](https://arxiv.org/pdf/2602.05084)  

**Abstract**: Strategic classification, where individuals modify their features to influence machine learning (ML) decisions, presents critical fairness challenges. While group fairness in this setting has been widely studied, individual fairness remains underexplored. We analyze threshold-based classifiers and prove that deterministic thresholds violate individual fairness. Then, we investigate the possibility of using a randomized classifier to achieve individual fairness. We introduce conditions under which a randomized classifier ensures individual fairness and leverage these conditions to find an optimal and individually fair randomized classifier through a linear programming problem. Additionally, we demonstrate that our approach can be extended to group fairness notions. Experiments on real-world datasets confirm that our method effectively mitigates unfairness and improves the fairness-accuracy trade-off. 

---
# Large-Ensemble Simulations Reveal Links Between Atmospheric Blocking Frequency and Sea Surface Temperature Variability 

**Authors**: Zilu Meng, Gregory J. Hakim, Wenchang Yang, Gabriel A. Vecchi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05083)  

**Abstract**: Atmospheric blocking events drive persistent weather extremes in midlatitudes, but isolating the influence of sea surface temperature (SST) from chaotic internal atmospheric variability on these events remains a challenge. We address this challenge using century-long (1900-2010), large-ensemble simulations with two computationally efficient deep-learning general circulation models. We find these models skillfully reproduce the observed blocking climatology, matching or exceeding the performance of a traditional high-resolution model and representative CMIP6 models. Averaging the large ensembles filters internal atmospheric noise to isolate the SST-forced component of blocking variability, yielding substantially higher correlations with reanalysis than for individual ensemble members. We identify robust teleconnections linking Greenland blocking frequency to North Atlantic SST and El Niño-like patterns. Furthermore, SST-forced trends in blocking frequency show a consistent decline in winter over Greenland, and an increase over Europe. These results demonstrate that SST variability exerts a significant and physically interpretable influence on blocking frequency and establishes large ensembles from deep learning models as a powerful tool for separating forced SST signals from internal noise. 

---
# Reliable Explanations or Random Noise? A Reliability Metric for XAI 

**Authors**: Poushali Sengupta, Sabita Maharjan, Frank Eliassen, Shashi Raj Pandey, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05082)  

**Abstract**: In recent years, explaining decisions made by complex machine learning models has become essential in high-stakes domains such as energy systems, healthcare, finance, and autonomous systems. However, the reliability of these explanations, namely, whether they remain stable and consistent under realistic, non-adversarial changes, remains largely unmeasured. Widely used methods such as SHAP and Integrated Gradients (IG) are well-motivated by axiomatic notions of attribution, yet their explanations can vary substantially even under system-level conditions, including small input perturbations, correlated representations, and minor model updates. Such variability undermines explanation reliability, as reliable explanations should remain consistent across equivalent input representations and small, performance-preserving model changes. We introduce the Explanation Reliability Index (ERI), a family of metrics that quantifies explanation stability under four reliability axioms: robustness to small input perturbations, consistency under feature redundancy, smoothness across model evolution, and resilience to mild distributional shifts. For each axiom, we derive formal guarantees, including Lipschitz-type bounds and temporal stability results. We further propose ERI-T, a dedicated measure of temporal reliability for sequential models, and introduce ERI-Bench, a benchmark designed to systematically stress-test explanation reliability across synthetic and real-world datasets. Experimental results reveal widespread reliability failures in popular explanation methods, showing that explanations can be unstable under realistic deployment conditions. By exposing and quantifying these instabilities, ERI enables principled assessment of explanation reliability and supports more trustworthy explainable AI (XAI) systems. 

---
# Food Portion Estimation: From Pixels to Calories 

**Authors**: Gautham Vinod, Fengqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05078)  

**Abstract**: Reliance on images for dietary assessment is an important strategy to accurately and conveniently monitor an individual's health, making it a vital mechanism in the prevention and care of chronic diseases and obesity. However, image-based dietary assessment suffers from estimating the three dimensional size of food from 2D image inputs. Many strategies have been devised to overcome this critical limitation such as the use of auxiliary inputs like depth maps, multi-view inputs, or model-based approaches such as template matching. Deep learning also helps bridge the gap by either using monocular images or combinations of the image and the auxillary inputs to precisely predict the output portion from the image input. In this paper, we explore the different strategies employed for accurate portion estimation. 

---
# E-Globe: Scalable $ε$-Global Verification of Neural Networks via Tight Upper Bounds and Pattern-Aware Branching 

**Authors**: Wenting Li, Saif R. Kazi, Russell Bent, Duo Zhou, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.05068)  

**Abstract**: Neural networks achieve strong empirical performance, but robustness concerns still hinder deployment in safety-critical applications. Formal verification provides robustness guarantees, but current methods face a scalability-completeness trade-off. We propose a hybrid verifier in a branch-and-bound (BaB) framework that efficiently tightens both upper and lower bounds until an $\epsilon-$global optimum is reached or early stop is triggered. The key is an exact nonlinear program with complementarity constraints (NLP-CC) for upper bounding that preserves the ReLU input-output graph, so any feasible solution yields a valid counterexample and enables rapid pruning of unsafe subproblems. We further accelerate verification with (i) warm-started NLP solves requiring minimal constraint-matrix updates and (ii) pattern-aligned strong branching that prioritizes splits most effective at tightening relaxations. We also provide conditions under which NLP-CC upper bounds are tight. Experiments on MNIST and CIFAR-10 show markedly tighter upper bounds than PGD across perturbation radii spanning up to three orders of magnitude, fast per-node solves in practice, and substantial end-to-end speedups over MIP-based verification, amplified by warm-starting, GPU batching, and pattern-aligned branching. 

---
# Bypassing AI Control Protocols via Agent-as-a-Proxy Attacks 

**Authors**: Jafar Isbarov, Murat Kantarcioglu  

**Link**: [PDF](https://arxiv.org/pdf/2602.05066)  

**Abstract**: As AI agents automate critical workloads, they remain vulnerable to indirect prompt injection (IPI) attacks. Current defenses rely on monitoring protocols that jointly evaluate an agent's Chain-of-Thought (CoT) and tool-use actions to ensure alignment with user intent. We demonstrate that these monitoring-based defenses can be bypassed via a novel Agent-as-a-Proxy attack, where prompt injection attacks treat the agent as a delivery mechanism, bypassing both agent and monitor simultaneously. While prior work on scalable oversight has focused on whether small monitors can supervise large agents, we show that even frontier-scale monitors are vulnerable. Large-scale monitoring models like Qwen2.5-72B can be bypassed by agents with similar capabilities, such as GPT-4o mini and Llama-3.1-70B. On the AgentDojo benchmark, we achieve a high attack success rate against AlignmentCheck and Extract-and-Evaluate monitors under diverse monitoring LLMs. Our findings suggest current monitoring-based agentic defenses are fundamentally fragile regardless of model scale. 

---
# ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation 

**Authors**: Songyuan Zhang, Oswin So, H. M. Sabbir Ahmad, Eric Yang Yu, Matthew Cleaveland, Mitchell Black, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2602.05051)  

**Abstract**: Offline reinforcement learning (RL) aims to learn the optimal policy from a fixed dataset generated by behavior policies without additional environment interactions. One common challenge that arises in this setting is the out-of-distribution (OOD) error, which occurs when the policy leaves the training distribution. Prior methods penalize a statistical distance term to keep the policy close to the behavior policy, but this constrains policy improvement and may not completely prevent OOD actions. Another challenge is that the optimal policy distribution can be multimodal and difficult to represent. Recent works apply diffusion or flow policies to address this problem, but it is unclear how to avoid OOD errors while retaining policy expressiveness. We propose ReFORM, an offline RL method based on flow policies that enforces the less restrictive support constraint by construction. ReFORM learns a behavior cloning (BC) flow policy with a bounded source distribution to capture the support of the action distribution, then optimizes a reflected flow that generates bounded noise for the BC flow while keeping the support, to maximize the performance. Across 40 challenging tasks from the OGBench benchmark with datasets of varying quality and using a constant set of hyperparameters for all tasks, ReFORM dominates all baselines with hand-tuned hyperparameters on the performance profile curves. 

---
# VISTA: Enhancing Visual Conditioning via Track-Following Preference Optimization in Vision-Language-Action Models 

**Authors**: Yiye Chen, Yanan Jian, Xiaoyi Dong, Shuxin Cao, Jing Wu, Patricio Vela, Benjamin E. Lundell, Dongdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.05049)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated strong performance across a wide range of robotic manipulation tasks. Despite the success, extending large pretrained Vision-Language Models (VLMs) to the action space can induce vision-action misalignment, where action predictions exhibit weak dependence on the current visual state, leading to unreliable action outputs. In this work, we study VLA models through the lens of visual conditioning and empirically show that successful rollouts consistently exhibit stronger visual dependence than failed ones. Motivated by this observation, we propose a training framework that explicitly strengthens visual conditioning in VLA models. Our approach first aligns action prediction with visual input via preference optimization on a track-following surrogate task, and then transfers the enhanced alignment to instruction-following task through latent-space distillation during supervised finetuning. Without introducing architectural modifications or additional data collection, our method improves both visual conditioning and task performance for discrete OpenVLA, and further yields consistent gains when extended to the continuous OpenVLA-OFT setting. Project website: this https URL . 

---
# Quality Model for Machine Learning Components 

**Authors**: Grace A. Lewis, Rachel Brower-Sinning, Robert Edman, Ipek Ozkaya, Sebastián Echeverría, Alex Derr, Collin Beaudoin, Katherine R. Maffey  

**Link**: [PDF](https://arxiv.org/pdf/2602.05043)  

**Abstract**: Despite increased adoption and advances in machine learning (ML), there are studies showing that many ML prototypes do not reach the production stage and that testing is still largely limited to testing model properties, such as model performance, without considering requirements derived from the system it will be a part of, such as throughput, resource consumption, or robustness. This limited view of testing leads to failures in model integration, deployment, and operations. In traditional software development, quality models such as ISO 25010 provide a widely used structured framework to assess software quality, define quality requirements, and provide a common language for communication with stakeholders. A newer standard, ISO 25059, defines a more specific quality model for AI systems. However, a problem with this standard is that it combines system attributes with ML component attributes, which is not helpful for a model developer, as many system attributes cannot be assessed at the component level. In this paper, we present a quality model for ML components that serves as a guide for requirements elicitation and negotiation and provides a common vocabulary for ML component developers and system stakeholders to agree on and define system-derived requirements and focus their testing efforts accordingly. The quality model was validated through a survey in which the participants agreed with its relevance and value. The quality model has been successfully integrated into an open-source tool for ML component testing and evaluation demonstrating its practical application. 

---
# Differentiable Inverse Graphics for Zero-shot Scene Reconstruction and Robot Grasping 

**Authors**: Octavio Arriaga, Proneet Sharma, Jichen Guo, Marc Otto, Siddhant Kadwe, Rebecca Adam  

**Link**: [PDF](https://arxiv.org/pdf/2602.05029)  

**Abstract**: Operating effectively in novel real-world environments requires robotic systems to estimate and interact with previously unseen objects. Current state-of-the-art models address this challenge by using large amounts of training data and test-time samples to build black-box scene representations. In this work, we introduce a differentiable neuro-graphics model that combines neural foundation models with physics-based differentiable rendering to perform zero-shot scene reconstruction and robot grasping without relying on any additional 3D data or test-time samples. Our model solves a series of constrained optimization problems to estimate physically consistent scene parameters, such as meshes, lighting conditions, material properties, and 6D poses of previously unseen objects from a single RGBD image and bounding boxes. We evaluated our approach on standard model-free few-shot benchmarks and demonstrated that it outperforms existing algorithms for model-free few-shot pose estimation. Furthermore, we validated the accuracy of our scene reconstructions by applying our algorithm to a zero-shot grasping task. By enabling zero-shot, physically-consistent scene reconstruction and grasping without reliance on extensive datasets or test-time sampling, our approach offers a pathway towards more data efficient, interpretable and generalizable robot autonomy in novel environments. 

---
# AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders 

**Authors**: Georgii Aparin, Tasnima Sadekova, Alexey Rukhovich, Assel Yermekova, Laida Kushnareva, Vadim Popov, Kristian Kuznetsov, Irina Piontkovskaya  

**Link**: [PDF](https://arxiv.org/pdf/2602.05027)  

**Abstract**: Sparse Autoencoders (SAEs) are powerful tools for interpreting neural representations, yet their use in audio remains underexplored. We train SAEs across all encoder layers of Whisper and HuBERT, provide an extensive evaluation of their stability, interpretability, and show their practical utility. Over 50% of the features remain consistent across random seeds, and reconstruction quality is preserved. SAE features capture general acoustic and semantic information as well as specific events, including environmental noises and paralinguistic sounds (e.g. laughter, whispering) and disentangle them effectively, requiring removal of only 19-27% of features to erase a concept. Feature steering reduces Whisper's false speech detections by 70% with negligible WER increase, demonstrating real-world applicability. Finally, we find SAE features correlated with human EEG activity during speech perception, indicating alignment with human neural processing. The code and checkpoints are available at this https URL. 

---
# Laws of Learning Dynamics and the Core of Learners 

**Authors**: Inkee Jung, Siu Cheong Lau  

**Link**: [PDF](https://arxiv.org/pdf/2602.05026)  

**Abstract**: We formulate the fundamental laws governing learning dynamics, namely the conservation law and the decrease of total entropy. Within this framework, we introduce an entropy-based lifelong ensemble learning method. We evaluate its effectiveness by constructing an immunization mechanism to defend against transfer-based adversarial attacks on the CIFAR-10 dataset. Compared with a naive ensemble formed by simply averaging models specialized on clean and adversarial samples, the resulting logifold achieves higher accuracy in most test cases, with particularly large gains under strong perturbations. 

---
# Do Vision-Language Models Respect Contextual Integrity in Location Disclosure? 

**Authors**: Ruixin Yang, Ethan Mendes, Arthur Wang, James Hays, Sauvik Das, Wei Xu, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2602.05023)  

**Abstract**: Vision-language models (VLMs) have demonstrated strong performance in image geolocation, a capability further sharpened by frontier multimodal large reasoning models (MLRMs). This poses a significant privacy risk, as these widely accessible models can be exploited to infer sensitive locations from casually shared photos, often at street-level precision, potentially surpassing the level of detail the sharer consented or intended to disclose. While recent work has proposed applying a blanket restriction on geolocation disclosure to combat this risk, these measures fail to distinguish valid geolocation uses from malicious behavior. Instead, VLMs should maintain contextual integrity by reasoning about elements within an image to determine the appropriate level of information disclosure, balancing privacy and utility. To evaluate how well models respect contextual integrity, we introduce VLM-GEOPRIVACY, a benchmark that challenges VLMs to interpret latent social norms and contextual cues in real-world images and determine the appropriate level of location disclosure. Our evaluation of 14 leading VLMs shows that, despite their ability to precisely geolocate images, the models are poorly aligned with human privacy expectations. They often over-disclose in sensitive contexts and are vulnerable to prompt-based attacks. Our results call for new design principles in multimodal systems to incorporate context-conditioned privacy reasoning. 

---
# From Fragmentation to Integration: Exploring the Design Space of AI Agents for Human-as-the-Unit Privacy Management 

**Authors**: Eryue Xu, Tianshi Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.05016)  

**Abstract**: Managing one's digital footprint is overwhelming, as it spans multiple platforms and involves countless context-dependent decisions. Recent advances in agentic AI offer ways forward by enabling holistic, contextual privacy-enhancing solutions. Building on this potential, we adopted a ''human-as-the-unit'' perspective and investigated users' cross-context privacy challenges through 12 semi-structured interviews. Results reveal that people rely on ad hoc manual strategies while lacking comprehensive privacy controls, highlighting nine privacy-management challenges across applications, temporal contexts, and relationships. To explore solutions, we generated nine AI agent concepts and evaluated them via a speed-dating survey with 116 US participants. The three highest-ranked concepts were all post-sharing management tools with half or full agent autonomy, with users expressing greater trust in AI accuracy than in their own efforts. Our findings highlight a promising design space where users see AI agents bridging the fragments in privacy management, particularly through automated, comprehensive post-sharing remediation of users' digital footprints. 

---
# Enhanced QKNorm normalization for neural transformers with the Lp norm 

**Authors**: Ezequiel Lopez-Rubio, Javier Montes-Perez, Esteban Jose Palomo  

**Link**: [PDF](https://arxiv.org/pdf/2602.05006)  

**Abstract**: The normalization of query and key vectors is an essential part of the Transformer architecture. It ensures that learning is stable regardless of the scale of these vectors. Some normalization approaches are available. In this preliminary work, a generalization of the QKNorm normalization scheme is proposed. The approach is based on the Lp norm, allowing non-Euclidean norms to be employed. Experimental results demonstrate the suitability of the method for a simple problem. 

---
# CoWork-X: Experience-Optimized Co-Evolution for Multi-Agent Collaboration System 

**Authors**: Zexin Lin, Jiachen Yu, Haoyang Zhang, Yuzhao Li, Zhonghang Li, Yujiu Yang, Junjie Wang, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.05004)  

**Abstract**: Large language models are enabling language-conditioned agents in interactive environments, but highly cooperative tasks often impose two simultaneous constraints: sub-second real-time coordination and sustained multi-episode adaptation under a strict online token budget. Existing approaches either rely on frequent in-episode reasoning that induces latency and timing jitter, or deliver post-episode improvements through unstructured text that is difficult to compile into reliable low-cost execution. We propose CoWork-X, an active co-evolution framework that casts peer collaboration as a closed-loop optimization problem across episodes, inspired by fast--slow memory separation. CoWork-X instantiates a Skill-Agent that executes via HTN (hierarchical task network)-based skill retrieval from a structured, interpretable, and compositional skill library, and a post-episode Co-Optimizer that performs patch-style skill consolidation with explicit budget constraints and drift regularization. Experiments in challenging Overcooked-AI-like realtime collaboration benchmarks demonstrate that CoWork-X achieves stable, cumulative performance gains while steadily reducing online latency and token usage. 

---
# EntRGi: Entropy Aware Reward Guidance for Diffusion Language Models 

**Authors**: Atula Tejaswi, Litu Rout, Constantine Caramanis, Sanjay Shakkottai, Sujay Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2602.05000)  

**Abstract**: Reward guidance has been applied to great success in the test-time adaptation of continuous diffusion models; it updates each denoising step using the gradients from a downstream reward model. We study reward guidance for discrete diffusion language models, where one cannot differentiate through the natural outputs of the model because they are discrete tokens. Existing approaches either replace these discrete tokens with continuous relaxations, or employ techniques like the straight-through estimator. In this work, we show the downsides of both these methods. The former degrades gradient feedback because the reward model has never been trained with continuous inputs. The latter involves incorrect optimization because the gradient evaluated at discrete tokens is used to update continuous logits. Our key innovation is to go beyond this tradeoff by introducing a novel mechanism called EntRGi: Entropy aware Reward Guidance that dynamically regulates the gradients from the reward model. By modulating the continuous relaxation using the model's confidence, our approach substantially improves reward guidance while providing reliable inputs to the reward model. We empirically validate our approach on a 7B-parameter diffusion language model across 3 diverse reward models and 3 multi-skill benchmarks, showing consistent improvements over state-of-the-art methods. 

---
# Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning 

**Authors**: Yu-Ang Lee, Ching-Yun Ko, Pin-Yu Chen, Mi-Yen Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2602.04998)  

**Abstract**: Low-Rank Adaptation (LoRA) is the prevailing approach for efficient large language model (LLM) fine-tuning. Building on this paradigm, recent studies have proposed alternative initialization strategies and architectural modifications, reporting substantial improvements over vanilla LoRA. However, these gains are often demonstrated under fixed or narrowly tuned hyperparameter settings, despite the known sensitivity of neural networks to training configurations. In this work, we systematically re-evaluate four representative LoRA variants alongside vanilla LoRA through extensive hyperparameter searches. Across mathematical and code generation tasks on diverse model scales, we find that different LoRA methods favor distinct learning rate ranges. Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors. These results suggest that vanilla LoRA remains a competitive baseline and that improvements reported under single training configuration may not reflect consistent methodological advantages. Finally, a second-order analysis attributes the differing optimal learning rate ranges to variations in the largest Hessian eigenvalue, aligning with classical learning theories. 

---
# Near-Optimal Dynamic Matching via Coarsening with Application to Heart Transplantation 

**Authors**: Itai Zilberstein, Ioannis Anagnostides, Zachary W. Sollie, Arman Kilic, Tuomas Sandholm  

**Link**: [PDF](https://arxiv.org/pdf/2602.04989)  

**Abstract**: Online matching has been a mainstay in domains such as Internet advertising and organ allocation, but practical algorithms often lack strong theoretical guarantees. We take an important step toward addressing this by developing new online matching algorithms based on a coarsening approach. Although coarsening typically implies a loss of granularity, we show that, to the contrary, aggregating offline nodes into capacitated clusters can yield near-optimal theoretical guarantees. We apply our methodology to heart transplant allocation to develop theoretically grounded policies based on structural properties of historical data. In realistic simulations, our policy closely matches the performance of the omniscient benchmark. Our work bridges the gap between data-driven heuristics and pessimistic theoretical lower bounds, and provides rigorous justification for prior clustering-based approaches in organ allocation. 

---
# AI-Based Detection of In-Treatment Changes from Prostate MR-Linac Images 

**Authors**: Seungbin Park, Peilin Wang, Ryan Pennell, Emily S. Weg, Himanshu Nagar, Timothy McClure, Mert R. Sabuncu, Daniel Margolis, Heejong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.04983)  

**Abstract**: Purpose: To investigate whether routinely acquired longitudinal MR-Linac images can be leveraged to characterize treatment-induced changes during radiotherapy, particularly subtle inter-fraction changes over short intervals (average of 2 days). Materials and Methods: This retrospective study included a series of 0.35T MR-Linac images from 761 patients. An artificial intelligence (deep learning) model was used to characterize treatment-induced changes by predicting the temporal order of paired images. The model was first trained with the images from the first and the last fractions (F1-FL), then with all pairs (All-pairs). Model performance was assessed using quantitative metrics (accuracy and AUC), compared to a radiologist's performance, and qualitative analyses - the saliency map evaluation to investigate affected anatomical regions. Input ablation experiments were performed to identify the anatomical regions altered by radiotherapy. The radiologist conducted an additional task on partial images reconstructed by saliency map regions, reporting observations as well. Quantitative image analysis was conducted to investigate the results from the model and the radiologist. Results: The F1-FL model yielded near-perfect performance (AUC of 0.99), significantly outperforming the radiologist. The All-pairs model yielded an AUC of 0.97. This performance reflects therapy-induced changes, supported by the performance correlation to fraction intervals, ablation tests and expert's interpretation. Primary regions driving the predictions were prostate, bladder, and pubic symphysis. Conclusion: The model accurately predicts temporal order of MR-Linac fractions and detects radiation-induced changes over one or a few days, including prostate and adjacent organ alterations confirmed by experts. This underscores MR-Linac's potential for advanced image analysis beyond image guidance. 

---
# Stochastic hierarchical data-driven optimization: application to plasma-surface kinetics 

**Authors**: José Afonso, Vasco Guerra, Pedro Viegas  

**Link**: [PDF](https://arxiv.org/pdf/2602.04975)  

**Abstract**: This work introduces a stochastic hierarchical optimization framework inspired by Sloppy Model theory for the efficient calibration of physical models. Central to this method is the use of a reduced Hessian approximation, which identifies and targets the stiff parameter subspace using minimal simulation queries. This strategy enables efficient navigation of highly anisotropic landscapes, avoiding the computational burden of exhaustive sampling. To ensure rigorous inference, we integrate this approach with a probabilistic formulation that derives a principled objective loss function directly from observed data. We validate the framework by applying it to the problem of plasma-surface interactions, where accurate modelling is strictly limited by uncertainties in surface reactivity parameters and the computational cost of kinetic simulations. Comparative analysis demonstrates that our method consistently outperforms baseline optimization techniques in sample efficiency. This approach offers a general and scalable tool for optimizing models of complex reaction systems, ranging from plasma chemistry to biochemical networks. 

---
# Smart Diagnosis and Early Intervention in PCOS: A Deep Learning Approach to Women's Reproductive Health 

**Authors**: Shayan Abrar, Samura Rahman, Ishrat Jahan Momo, Mahjabin Tasnim Samiha, B. M. Shahria Alam, Mohammad Tahmid Noor, Nishat Tasnim Niloy  

**Link**: [PDF](https://arxiv.org/pdf/2602.04944)  

**Abstract**: Polycystic Ovary Syndrome (PCOS) is a widespread disorder in women of reproductive age, characterized by a hormonal imbalance, irregular periods, and multiple ovarian cysts. Infertility, metabolic syndrome, and cardiovascular risks are long-term complications that make early detection essential. In this paper, we design a powerful framework based on transfer learning utilizing DenseNet201 and ResNet50 for classifying ovarian ultrasound images. The model was trained on an online dataset containing 3856 ultrasound images of cyst-infected and non-infected patients. Each ultrasound frame was resized to 224x224 pixels and encoded with precise pathological indicators. The MixUp and CutMix augmentation strategies were used to improve generalization, yielding a peak validation accuracy of 99.80% by Densenet201 and a validation loss of 0.617 with alpha values of 0.25 and 0.4, respectively. We evaluated the model's interpretability using leading Explainable AI (XAI) approaches such as SHAP, Grad-CAM, and LIME, reasoning with and presenting explicit visual reasons for the model's behaviors, therefore increasing the model's transparency. This study proposes an automated system for medical picture diagnosis that may be used effectively and confidently in clinical practice. 

---
# Graph--Theoretic Analysis of Phase Optimization Complexity in Variational Wave Functions for Heisenberg Antiferromagnets 

**Authors**: Mahmud Ashraf Shamim, Moshiur Rahman, Mohamed Hibat-Allah, Paulo T Araujo  

**Link**: [PDF](https://arxiv.org/pdf/2602.04943)  

**Abstract**: Despite extensive study, the phase structure of the wavefunctions in frustrated Heisenberg antiferromagnets (HAF) is not yet systematically characterized. In this work, we represent the Hilbert space of an HAF as a weighted graph, which we term the Hilbert graph (HG), whose vertices are spin configurations and whose edges are generated by off-diagonal spin-flip terms of the Heisenberg Hamiltonian, with weights set by products of wavefunction amplitudes. Holding the amplitudes fixed and restricting phases to $\mathbb{Z}_2$ values, the phase-dependent variational energy can be recast as a classical Ising antiferromagnet on the HG, so that phase reconstruction of the ground state reduces to a weighted Max-Cut instance. This shows that phase reconstruction HAF is worst-case NP-hard and provides a direct link between wavefunction sign structure and combinatorial optimization. 

---
# Privileged Information Distillation for Language Models 

**Authors**: Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2602.04942)  

**Abstract**: Training-time privileged information (PI) can enable language models to succeed on tasks they would otherwise fail, making it a powerful tool for reinforcement learning in hard, long-horizon settings. However, transferring capabilities learned with PI to policies that must act without it at inference time remains a fundamental challenge. We study this problem in the context of distilling frontier models for multi-turn agentic environments, where closed-source systems typically hide their internal reasoning and expose only action trajectories. This breaks standard distillation pipelines, since successful behavior is observable but the reasoning process is not. For this, we introduce {\pi}-Distill, a joint teacher-student objective that trains a PI-conditioned teacher and an unconditioned student simultaneously using the same model. Additionally, we also introduce On-Policy Self-Distillation (OPSD), an alternative approach that trains using Reinforcement Learning (RL) with a reverse KL-penalty between the student and the PI-conditioned teacher. We show that both of these algorithms effectively distill frontier agents using action-only PI. Specifically we find that {\pi}-Distill and in some cases OPSD, outperform industry standard practices (Supervised finetuning followed by RL) that assume access to full Chain-of-Thought supervision across multiple agentic benchmarks, models, and forms of PI. We complement our results with extensive analysis that characterizes the factors enabling effective learning with PI, focusing primarily on {\pi}-Distill and characterizing when OPSD is competitive. 

---
# Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization 

**Authors**: Davide Berasi, Matteo Farina, Massimiliano Mancini, Elisa Ricci  

**Link**: [PDF](https://arxiv.org/pdf/2602.04937)  

**Abstract**: Selecting the best data mixture is critical for successful Supervised Fine-Tuning (SFT) of Multimodal Large Language Models. However, determining the optimal mixture weights across multiple domain-specific datasets remains a significant bottleneck due to the combinatorial search space and the high cost associated with even a single training run. This is the so-called Data Mixture Optimization (DMO) problem. On the other hand, model merging unifies domain-specific experts through parameter interpolation. This strategy is efficient, as it only requires a single training run per domain, yet oftentimes leads to suboptimal models. In this work, we take the best of both worlds, studying model merging as an efficient strategy for estimating the performance of different data mixtures. We train domain-specific multimodal experts and evaluate their weighted parameter-space combinations to estimate the efficacy of corresponding data mixtures. We conduct extensive experiments on 14 multimodal benchmarks, and empirically demonstrate that the merged proxy models exhibit a high rank correlation with models trained on actual data mixtures. This decouples the search for optimal mixtures from the resource-intensive training process, thereby providing a scalable and efficient strategy for navigating the complex landscape of mixture weights. Code is publicly available at this https URL. 

---
# ASA: Activation Steering for Tool-Calling Domain Adaptation 

**Authors**: Youjin Wang, Run Zhou, Rong Fu, Shuaishuai Cao, Hongwei Zeng, Jiaxuan Lu, Sicheng Fan, Jiaqiao Zhao, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.04935)  

**Abstract**: For real-world deployment of general-purpose LLM agents, the core challenge is often not tool use itself, but efficient domain adaptation under rapidly evolving toolsets, APIs, and protocols. Repeated LoRA or SFT across domains incurs exponentially growing training and maintenance costs, while prompt or schema methods are brittle under distribution shift and complex interfaces. We propose \textbf{Activation Steering Adapter (ASA}), a lightweight, inference-time, training-free mechanism that reads routing signals from intermediate activations and uses an ultra-light router to produce adaptive control strengths for precise domain alignment. Across multiple model scales and domains, ASA achieves LoRA-comparable adaptation with substantially lower overhead and strong cross-model transferability, making it ideally practical for robust, scalable, and efficient multi-domain tool ecosystems with frequent interface churn dynamics. 

---
# Depth-Wise Emergence of Prediction-Centric Geometry in Large Language Models 

**Authors**: Shahar Haim, Daniel C McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2602.04931)  

**Abstract**: We show that decoder-only large language models exhibit a depth-wise transition from context-processing to prediction-forming phases of computation accompanied by a reorganization of representational geometry. Using a unified framework combining geometric analysis with mechanistic intervention, we demonstrate that late-layer representations implement a structured geometric code that enables selective causal control over token prediction. Specifically, angular organization of the representation geometry parametrizes prediction distributional similarity, while representation norms encode context-specific information that does not determine prediction. Together, these results provide a mechanistic-geometric account of the dynamics of transforming context into predictions in LLMs. 

---
# Attack Selection Reduces Safety in Concentrated AI Control Settings against Trusted Monitoring 

**Authors**: Joachim Schaeffer, Arjun Khandelwal, Tyler Tracy  

**Link**: [PDF](https://arxiv.org/pdf/2602.04930)  

**Abstract**: Future AI deployments will likely be monitored for malicious behaviour. The ability of these AIs to subvert monitors by adversarially selecting against them - attack selection - is particularly concerning. To study this, we let a red team create attack policies that attempt to insert attacks into code without being caught by a monitor in the concentrated BigCodeBench backdooring setting. We decompose attack selection into two problems: mapping attacks to a quality score and mapping quality scores to submission probabilities. We frame attack selection as a classification problem and show that safety is significantly more sensitive to FPR than TPR. We find that prompting the attacker model to reason about the monitor while being cautious with attack selection reduces safety from a baseline of 99% to 59% at 0.5% auditing budget, emphasizing that eliciting attack selection capabilities of models is vital to avoid overly optimistic safety scores in control evaluations. 

---
# PriMod4AI: Lifecycle-Aware Privacy Threat Modeling for AI Systems using LLM 

**Authors**: Gautam Savaliya, Robert Aufschläger, Abhishek Subedi, Michael Heigl, Martin Schramm  

**Link**: [PDF](https://arxiv.org/pdf/2602.04927)  

**Abstract**: Artificial intelligence systems introduce complex privacy risks throughout their lifecycle, especially when processing sensitive or high-dimensional data. Beyond the seven traditional privacy threat categories defined by the LINDDUN framework, AI systems are also exposed to model-centric privacy attacks such as membership inference and model inversion, which LINDDUN does not cover. To address both classical LINDDUN threats and additional AI-driven privacy attacks, PriMod4AI introduces a hybrid privacy threat modeling approach that unifies two structured knowledge sources, a LINDDUN knowledge base representing the established taxonomy, and a model-centric privacy attack knowledge base capturing threats outside LINDDUN. These knowledge bases are embedded into a vector database for semantic retrieval and combined with system level metadata derived from Data Flow Diagram. PriMod4AI uses retrieval-augmented and Data Flow specific prompt generation to guide large language models (LLMs) in identifying, explaining, and categorizing privacy threats across lifecycle stages. The framework produces justified and taxonomy-grounded threat assessments that integrate both classical and AI-driven perspectives. Evaluation on two AI systems indicates that PriMod4AI provides broad coverage of classical privacy categories while additionally identifying model-centric privacy threats. The framework produces consistent, knowledge-grounded outputs across LLMs, as reflected in agreement scores in the observed range. 

---
# Internalizing LLM Reasoning via Discovery and Replay of Latent Actions 

**Authors**: Zhenning Shi, Yijia Zhu, Junhan Shi, Xun Zhang, Lei Wang, Congcong Miao  

**Link**: [PDF](https://arxiv.org/pdf/2602.04925)  

**Abstract**: The internalization of chain-of-thought processes into hidden states has emerged as a highly efficient paradigm for scaling test-time compute. However, existing activation steering methods rely on static control vectors that fail to adapt to the non-stationary evolution of complex reasoning tasks. To address this limitation, we propose STIR (Self-Distilled Tools for Internal Reasoning), a framework that reformulates reasoning enhancement as a dynamic latent trajectory control problem. STIR introduces a synergistic three-stage pipeline: (1) differential intrinsic action induction harvests latent reasoning successes to crystallize steering primitives; (2) sparse control basis construction curates a compact, geometrically diverse tool library; and (3) value-modulated trajectory intervention dynamically injects context-specific impulses via anchor-based gating. Extensive experiments on six arithmetic and logical benchmarks across four representative models demonstrate that STIR improves average accuracy by 1.9% to 7.5% while reducing average token consumption by up to 35% compared to vanilla decoding. These findings demonstrate that the benefits of explicit chain-of-thought can be realized through dynamic latent trajectory control, internalizing the reasoning process to bypass the explicit generation while achieving superior fidelity. Our code is available at this https URL. 

---
# SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel 

**Authors**: Jose Miguel Luna, Taha Bouhsine, Krzysztof Choromanski  

**Link**: [PDF](https://arxiv.org/pdf/2602.04915)  

**Abstract**: We propose a new class of linear-time attention mechanisms based on a relaxed and computationally efficient formulation of the recently introduced E-Product, often referred to as the Yat-kernel (Bouhsine, 2025). The resulting interactions are geometry-aware and inspired by inverse-square interactions in physics. Our method, Spherical Linearized Attention with Yat Kernels (SLAY), constrains queries and keys to the unit sphere so that attention depends only on angular alignment. Using Bernstein's theorem, we express the spherical Yat-kernel as a nonnegative mixture of polynomial-exponential product kernels and derive a strictly positive random-feature approximation enabling linear-time O(L) attention. We establish positive definiteness and boundedness on the sphere and show that the estimator yields well-defined, nonnegative attention scores. Empirically, SLAY achieves performance that is nearly indistinguishable from standard softmax attention while retaining linear time and memory scaling, and consistently outperforms prior linear-time attention mechanisms such as Performers and Cosformers. To the best of our knowledge, SLAY represents the closest linear-time approximation to softmax attention reported to date, enabling scalable Transformers without the typical performance trade-offs of attention linearization. 

---
# A$^2$-LLM: An End-to-end Conversational Audio Avatar Large Language Model 

**Authors**: Xiaolin Hu, Hang Yuan, Xinzhu Sang, Binbin Yan, Zhou Yu, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.04913)  

**Abstract**: Developing expressive and responsive conversational digital humans is a cornerstone of next-generation human-computer interaction. While large language models (LLMs) have significantly enhanced dialogue capabilities, most current systems still rely on cascaded architectures that connect independent modules. These pipelines are often plagued by accumulated errors, high latency, and poor real-time performance. Lacking access to the underlying conversational context, these pipelines inherently prioritize rigid lip-sync over emotional depth. To address these challenges, we propose A$^2$-LLM, an end-to-end conversational audio avatar large language model that jointly reasons about language, audio prosody, and 3D facial motion within a unified framework. To facilitate training, we introduce FLAME-QA, a high-quality multimodal dataset designed to align semantic intent with expressive facial dynamics within a QA format. By leveraging deep semantic understanding, A$^2$-LLM generates emotionally rich facial movements beyond simple lip-synchronization. Experimental results demonstrate that our system achieves superior emotional expressiveness while maintaining real-time efficiency (500 ms latency, 0.7 RTF). 

---
# A logical re-conception of neural networks: Hamiltonian bitwise part-whole architecture 

**Authors**: E Bowen, R Granger, A Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2602.04911)  

**Abstract**: We introduce a simple initial working system in which relations (such as part-whole) are directly represented via an architecture with operating and learning rules fundamentally distinct from standard artificial neural network methods. Arbitrary data are straightforwardly encoded as graphs whose edges correspond to codes from a small fixed primitive set of elemental pairwise relations, such that simple relational encoding is not an add-on, but occurs intrinsically within the most basic components of the system. A novel graph-Hamiltonian operator calculates energies among these encodings, with ground states denoting simultaneous satisfaction of all relation constraints among graph vertices. The method solely uses radically low-precision arithmetic; computational cost is correspondingly low, and scales linearly with the number of edges in the data. The resulting unconventional architecture can process standard ANN examples, but also produces representations that exhibit characteristics of symbolic computation. Specifically, the method identifies simple logical relational structures in these data (part-of; next-to), building hierarchical representations that enable abductive inferential steps generating relational position-based encodings, rather than solely statistical representations. Notably, an equivalent set of ANN operations are derived, identifying a special case of embedded vector encodings that may constitute a useful approach to current work in higher-level semantic representation. The very simple current state of the implemented system invites additional tools and improvements. 

---
# Temporal Pair Consistency for Variance-Reduced Flow Matching 

**Authors**: Chika Maduabuchi, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04908)  

**Abstract**: Continuous-time generative models, such as diffusion models, flow matching, and rectified flow, learn time-dependent vector fields but are typically trained with objectives that treat timesteps independently, leading to high estimator variance and inefficient sampling. Prior approaches mitigate this via explicit smoothness penalties, trajectory regularization, or modified probability paths and solvers. We introduce Temporal Pair Consistency (TPC), a lightweight variance-reduction principle that couples velocity predictions at paired timesteps along the same probability path, operating entirely at the estimator level without modifying the model architecture, probability path, or solver. We provide a theoretical analysis showing that TPC induces a quadratic, trajectory-coupled regularization that provably reduces gradient variance while preserving the underlying flow-matching objective. Instantiated within flow matching, TPC improves sample quality and efficiency across CIFAR-10 and ImageNet at multiple resolutions, achieving lower FID at identical or lower computational cost than prior methods, and extends seamlessly to modern SOTA-style pipelines with noise-augmented training, score-based denoising, and rectified flow. 

---
# Physics as the Inductive Bias for Causal Discovery 

**Authors**: Jianhong Chen, Naichen Shi, Xubo Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.04907)  

**Abstract**: Causal discovery is often a data-driven paradigm to analyze complex real-world systems. In parallel, physics-based models such as ordinary differential equations (ODEs) provide mechanistic structure for many dynamical processes. Integrating these paradigms potentially allows physical knowledge to act as an inductive bias, improving identifiability, stability, and robustness of causal discovery in dynamical systems. However, such integration remains challenging: real dynamical systems often exhibit feedback, cyclic interactions, and non-stationary data trend, while many widely used causal discovery methods are formulated under acyclicity or equilibrium-based assumptions. In this work, we propose an integrative causal discovery framework for dynamical systems that leverages partial physical knowledge as an inductive bias. Specifically, we model system evolution as a stochastic differential equation (SDE), where the drift term encodes known ODE dynamics and the diffusion term corresponds to unknown causal couplings beyond the prescribed physics. We develop a scalable sparsity-inducing MLE algorithm that exploits causal graph structure for efficient parameter estimation. Under mild conditions, we establish guarantees to recover the causal graph. Experiments on dynamical systems with diverse causal structures show that our approach improves causal graph recovery and produces more stable, physically consistent estimates than purely data-driven state-of-the-art baselines. 

---
# DCER: Dual-Stage Compression and Energy-Based Reconstruction 

**Authors**: Yiwen Wang, Jiahao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2602.04904)  

**Abstract**: Multimodal fusion faces two robustness challenges: noisy inputs degrade representation quality, and missing modalities cause prediction failures. We propose DCER, a
unified framework addressing both challenges through dual-stage compression and energy-based reconstruction. The compression stage operates at two levels:
within-modality frequency transforms (wavelet for audio, DCT for video) remove noise while preserving task-relevant patterns, and cross-modality bottleneck tokens
force genuine integration rather than modality-specific shortcuts. For missing modalities, energy-based reconstruction recovers representations via gradient descent
on a learned energy function, with the final energy providing intrinsic uncertainty quantification (\r{ho} > 0.72 correlation with prediction error). Experiments on
CMU-MOSI, CMU-MOSEI, and CH-SIMS demonstrate state-of-the-art performance across all benchmarks, with a U-shaped robustness pattern favoring multimodal fusion at
both complete and high-missing conditions. The code will be available on Github. 

---
# Momentum Attention: The Physics of In-Context Learning and Spectral Forensics for Mechanistic Interpretability 

**Authors**: Kingsuk Maitra  

**Link**: [PDF](https://arxiv.org/pdf/2602.04902)  

**Abstract**: The Mechanistic Interpretability (MI) program has mapped the Transformer as a precise computational graph. We extend this graph with a conservation law and time-varying AC dynamics, viewing it as a physical circuit. We introduce Momentum Attention, a symplectic augmentation embedding physical priors via the kinematic difference operator $p_t = q_t - q_{t-1}$, implementing the symplectic shear $\hat{q}_t = q_t + \gamma p_t$ on queries and keys. We identify a fundamental Symplectic-Filter Duality: the physical shear is mathematically equivalent to a High-Pass Filter. This duality is our cornerstone contribution -- by injecting kinematic momentum, we sidestep the topological depth constraint ($L \geq 2$) for induction head formation. While standard architectures require two layers for induction from static positions, our extension grants direct access to velocity, enabling Single-Layer Induction and Spectral Forensics via Bode Plots. We formalize an Orthogonality Theorem proving that DC (semantic) and AC (mechanistic) signals segregate into orthogonal frequency bands when Low-Pass RoPE interacts with High-Pass Momentum. Validated through 5,100+ controlled experiments (documented in Supplementary Appendices A--R and 27 Jupyter notebooks), our 125M Momentum model exceeds expectations on induction-heavy tasks while tracking a 350M baseline within $\sim$2.9% validation loss. Dedicated associative recall experiments reveal a scaling law $\gamma^* = 4.17 \times N^{-0.74}$ establishing momentum-depth fungibility. We offer this framework as a complementary analytical toolkit connecting Generative AI, Hamiltonian Physics, and Signal Processing. 

---
# Evaluating Kubernetes Performance for GenAI Inference: From Automatic Speech Recognition to LLM Summarization 

**Authors**: Sai Sindhur Malleni, Raúl Sevilla, Aleksei Vasilevskii, José Castillo Lema, André Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2602.04900)  

**Abstract**: As Generative AI (GenAI), particularly inference, rapidly emerges as a dominant workload category, the Kubernetes ecosystem is proactively evolving to natively support its unique demands. This industry paper demonstrates how emerging Kubernetes-native projects can be combined to deliver the benefits of container orchestration, such as scalability and resource efficiency, to complex AI workflows. We implement and evaluate an illustrative, multi-stage use case consisting of automatic speech recognition and summarization. First, we address batch inference by using Kueue to manage jobs that transcribe audio files with Whisper models and Dynamic Accelerator Slicer (DAS) to increase parallel job execution. Second, we address a discrete online inference scenario by feeding the transcripts to a Large Language Model for summarization hosted using llm-d, a novel solution utilizing the recent developments around the Kubernetes Gateway API Inference Extension (GAIE) for optimized routing of inference requests. Our findings illustrate that these complementary components (Kueue, DAS, and GAIE) form a cohesive, high-performance platform, proving Kubernetes' capability to serve as a unified foundation for demanding GenAI workloads: Kueue reduced total makespan by up to 15%; DAS shortened mean job completion time by 36%; and GAIE improved Time to First Token by 82\%. 

---
# Phantom Transfer: Data-level Defences are Insufficient Against Data Poisoning 

**Authors**: Andrew Draganov, Tolga H. Dur, Anandmayi Bhongade, Mary Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2602.04899)  

**Abstract**: We present a data poisoning attack -- Phantom Transfer -- with the property that, even if you know precisely how the poison was placed into an otherwise benign dataset, you cannot filter it out. We achieve this by modifying subliminal learning to work in real-world contexts and demonstrate that the attack works across models, including GPT-4.1. Indeed, even fully paraphrasing every sample in the dataset using a different model does not stop the attack. We also discuss connections to steering vectors and show that one can plant password-triggered behaviours into models while still beating defences.
This suggests that data-level defences are insufficient for stopping sophisticated data poisoning attacks. We suggest that future work should focus on model audits and white-box security methods. 

---
# Semantic-level Backdoor Attack against Text-to-Image Diffusion Models 

**Authors**: Tianxin Chen, Wenbo Jiang, Hongqiao Chen, Zhirun Zheng, Cheng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.04898)  

**Abstract**: Text-to-image (T2I) diffusion models are widely adopted for their strong generative capabilities, yet remain vulnerable to backdoor attacks. Existing attacks typically rely on fixed textual triggers and single-entity backdoor targets, making them highly susceptible to enumeration-based input defenses and attention-consistency detection. In this work, we propose Semantic-level Backdoor Attack (SemBD), which implants backdoors at the representation level by defining triggers as continuous semantic regions rather than discrete textual patterns. Concretely, SemBD injects semantic backdoors by distillation-based editing of the key and value projection matrices in cross-attention layers, enabling diverse prompts with identical semantic compositions to reliably activate the backdoor attack. To further enhance stealthiness, SemBD incorporates a semantic regularization to prevent unintended activation under incomplete semantics, as well as multi-entity backdoor targets that avoid highly consistent cross-attention patterns. Extensive experiments demonstrate that SemBD achieves a 100% attack success rate while maintaining strong robustness against state-of-the-art input-level defenses. 

---
# Steering Externalities: Benign Activation Steering Unintentionally Increases Jailbreak Risk for Large Language Models 

**Authors**: Chen Xiong, Zhiyuan He, Pin-Yu Chen, Ching-Yun Ko, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2602.04896)  

**Abstract**: Activation steering is a practical post-training model alignment technique to enhance the utility of Large Language Models (LLMs). Prior to deploying a model as a service, developers can steer a pre-trained model toward specific behavioral objectives, such as compliance or instruction adherence, without the need for retraining. This process is as simple as adding a steering vector to the model's internal representations. However, this capability unintentionally introduces critical and under-explored safety risks. We identify a phenomenon termed Steering Externalities, where steering vectors derived from entirely benign datasets-such as those enforcing strict compliance or specific output formats like JSON-inadvertently erode safety guardrails. Experiments reveal that these interventions act as a force multiplier, creating new vulnerabilities to jailbreaks and increasing attack success rates to over 80% on standard benchmarks by bypassing the initial safety alignment. Ultimately, our results expose a critical blind spot in deployment: benign activation steering systematically erodes the "safety margin," rendering models more vulnerable to black-box attacks and proving that inference-time utility improvements must be rigorously audited for unintended safety externalities. 

---
# Extracting Recurring Vulnerabilities from Black-Box LLM-Generated Software 

**Authors**: Tomer Kordonsky, Maayan Yamin, Noam Benzimra, Amit LeVi, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2602.04894)  

**Abstract**: LLMs are increasingly used for code generation, but their outputs often follow recurring templates that can induce predictable vulnerabilities. We study \emph{vulnerability persistence} in LLM-generated software and introduce \emph{Feature--Security Table (FSTab)} with two components. First, FSTab enables a black-box attack that predicts likely backend vulnerabilities from observable frontend features and knowledge of the source LLM, without access to backend code or source code. Second, FSTab provides a model-centric evaluation that quantifies how consistently a given model reproduces the same vulnerabilities across programs, semantics-preserving rephrasings, and application domains. We evaluate FSTab on state-of-the-art code LLMs, including GPT-5.2, Claude-4.5 Opus, and Gemini-3 Pro, across diverse application domains. Our results show strong cross-domain transfer: even when the target domain is excluded from training, FSTab achieves up to 94\% attack success and 93\% vulnerability coverage on Internal Tools (Claude-4.5 Opus). These findings expose an underexplored attack surface in LLM-generated software and highlight the security risks of code generation. Our code is available at: this https URL. 

---
# A Causal Perspective for Enhancing Jailbreak Attack and Defense 

**Authors**: Licheng Pan, Yunsheng Lu, Jiexi Liu, Jialing Tao, Haozhe Feng, Hui Xue, Zhixuan Chu, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.04893)  

**Abstract**: Uncovering the mechanisms behind "jailbreaks" in large language models (LLMs) is crucial for enhancing their safety and reliability, yet these mechanisms remain poorly understood. Existing studies predominantly analyze jailbreak prompts by probing latent representations, often overlooking the causal relationships between interpretable prompt features and jailbreak occurrences. In this work, we propose Causal Analyst, a framework that integrates LLMs into data-driven causal discovery to identify the direct causes of jailbreaks and leverage them for both attack and defense. We introduce a comprehensive dataset comprising 35k jailbreak attempts across seven LLMs, systematically constructed from 100 attack templates and 50 harmful queries, annotated with 37 meticulously designed human-readable prompt features. By jointly training LLM-based prompt encoding and GNN-based causal graph learning, we reconstruct causal pathways linking prompt features to jailbreak responses. Our analysis reveals that specific features, such as "Positive Character" and "Number of Task Steps", act as direct causal drivers of jailbreaks. We demonstrate the practical utility of these insights through two applications: (1) a Jailbreaking Enhancer that targets identified causal features to significantly boost attack success rates on public benchmarks, and (2) a Guardrail Advisor that utilizes the learned causal graph to extract true malicious intent from obfuscated queries. Extensive experiments, including baseline comparisons and causal structure validation, confirm the robustness of our causal analysis and its superiority over non-causal approaches. Our results suggest that analyzing jailbreak features from a causal perspective is an effective and interpretable approach for improving LLM reliability. Our code is available at this https URL. 

---
# Doc2Spec: Synthesizing Formal Programming Specifications from Natural Language via Grammar Induction 

**Authors**: Shihao Xia, Mengting He, Haomin Jia, Linhai Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.04892)  

**Abstract**: Ensuring that API implementations and usage comply with natural language programming rules is critical for software correctness, security, and reliability. Formal verification can provide strong guarantees but requires precise specifications, which are difficult and costly to write manually. To address this challenge, we present Doc2Spec, a multi-agent framework that uses LLMs to automatically induce a specification grammar from natural-language rules and then generates formal specifications guided by the induced grammar. The grammar captures essential domain knowledge, constrains the specification space, and enforces consistent representations, thereby improving the reliability and quality of generated specifications. Evaluated on seven benchmarks across three programming languages, Doc2Spec outperforms a baseline without grammar induction and achieves competitive results against a technique with a manually crafted grammar, demonstrating the effectiveness of automated grammar induction for formalizing natural-language rules. 

---
# A General-Purpose Diversified 2D Seismic Image Dataset from NAMSS 

**Authors**: Lucas de Magalhães Araujo, Otávio Oliveira Napoli, Sandra Avila, Edson Borin  

**Link**: [PDF](https://arxiv.org/pdf/2602.04890)  

**Abstract**: We introduce the Unicamp-NAMSS dataset, a large, diverse, and geographically distributed collection of migrated 2D seismic sections designed to support modern machine learning research in geophysics. We constructed the dataset from the National Archive of Marine Seismic Surveys (NAMSS), which contains decades of publicly available marine seismic data acquired across multiple regions, acquisition conditions, and geological settings. After a comprehensive collection and filtering process, we obtained 2588 cleaned and standardized seismic sections from 122 survey areas, covering a wide range of vertical and horizontal sampling characteristics. To ensure reliable experimentation, we balanced the dataset so that no survey dominates the distribution, and partitioned it into non-overlapping macro-regions for training, validation, and testing. This region-disjoint split allows robust evaluation of generalization to unseen geological and acquisition conditions.
We validated the dataset through quantitative and embedding-space analyses using both convolutional and transformer-based models. These analyses showed that Unicamp-NAMSS exhibits substantial variability within and across regions, while maintaining coherent structure across acquisition macro-region and survey types. Comparisons with widely used interpretation datasets (Parihaka and F3 Block) further demonstrated that Unicamp-NAMSS covers a broader portion of the seismic appearance space, making it a strong candidate for machine learning model pretraining. The dataset, therefore, provides a valuable resource for machine learning tasks, including self-supervised representation learning, transfer learning, benchmarking supervised tasks such as super-resolution or attribute prediction, and studying domain adaptation in seismic interpretation. 

---
# Denoising diffusion networks for normative modeling in neuroimaging 

**Authors**: Luke Whitbread, Lyle J. Palmer, Mark Jenkinson  

**Link**: [PDF](https://arxiv.org/pdf/2602.04886)  

**Abstract**: Normative modeling estimates reference distributions of biological measures conditional on covariates, enabling centiles and clinically interpretable deviation scores to be derived. Most neuroimaging pipelines fit one model per imaging-derived phenotype (IDP), which scales well but discards multivariate dependence that may encode coordinated patterns. We propose denoising diffusion probabilistic models (DDPMs) as a unified conditional density estimator for tabular IDPs, from which univariate centiles and deviation scores are derived by sampling. We utilise two denoiser backbones: (i) a feature-wise linear modulation (FiLM) conditioned multilayer perceptron (MLP) and (ii) a tabular transformer with feature self-attention and intersample attention (SAINT), conditioning covariates through learned embeddings. We evaluate on a synthetic benchmark with heteroscedastic and multimodal age effects and on UK Biobank FreeSurfer phenotypes, scaling from dimension of 2 to 200. Our evaluation suite includes centile calibration (absolute centile error, empirical coverage, and the probability integral transform), distributional fidelity (Kolmogorov-Smirnov tests), multivariate dependence diagnostics, and nearest-neighbour memorisation analysis. For low dimensions, diffusion models deliver well-calibrated per-IDP outputs comparable to traditional baselines while jointly modeling realistic dependence structure. At higher dimensions, the transformer backbone remains substantially better calibrated than the MLP and better preserves higher-order dependence, enabling scalable joint normative models that remain compatible with standard per-IDP pipelines. These results support diffusion-based normative modeling as a practical route to calibrated multivariate deviation profiles in neuroimaging. 

---
# Cold Start Problem: An Experimental Study of Knowledge Tracing Models with New Students 

**Authors**: Indronil Bhattacharjee, Christabel Wayllace  

**Link**: [PDF](https://arxiv.org/pdf/2505.21517)  

**Abstract**: KnowledgeTracing (KT) involves predicting students' knowledge states based on their interactions with Intelligent Tutoring Systems (ITS). A key challenge is the cold start problem, accurately predicting knowledge for new students with minimal interaction data. Unlike prior work, which typically trains KT models on initial interactions of all students and tests on their subsequent interactions, our approach trains models solely using historical data from past students, evaluating their performance exclusively on entirely new students. We investigate cold start effects across three KT models: Deep Knowledge Tracing (DKT), Dynamic Key-Value Memory Networks (DKVMN), and Self-Attentive Knowledge Tracing (SAKT), using ASSISTments 2009, 2015, and 2017 datasets. Results indicate all models initially struggle under cold start conditions but progressively improve with more interactions; SAKT shows higher initial accuracy yet still faces limitations. These findings highlight the need for KT models that effectively generalize to new learners, emphasizing the importance of developing models robust in few-shot and zero-shot learning scenarios 

---
