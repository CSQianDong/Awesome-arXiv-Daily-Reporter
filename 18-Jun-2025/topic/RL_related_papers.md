# Ring-lite: Scalable Reasoning via C3PO-Stabilized Reinforcement Learning for LLMs 

**Authors**: Ring Team, Bin Hu, Cai Chen, Deng Zhao, Ding Liu, Dingnan Jin, Feng Zhu, Hao Dai, Hongzhi Luan, Jia Guo, Jiaming Liu, Jiewei Wu, Jun Mei, Jun Zhou, Junbo Zhao, Junwu Xiong, Kaihong Zhang, Kuan Xu, Lei Liang, Liang Jiang, Liangcheng Fu, Longfei Zheng, Qiang Gao, Qing Cui, Quan Wan, Shaomian Zheng, Shuaicheng Li, Tongkai Yang, Wang Ren, Xiaodong Yan, Xiaopei Wan, Xiaoyun Feng, Xin Zhao, Xinxing Yang, Xinyu Kong, Xuemin Yang, Yang Li, Yingting Wu, Yongkang Liu, Zhankai Xu, Zhenduo Zhang, Zhenglei Zhou, Zhenyu Huang, Zhiqiang Zhang, Zihao Wang, Zujie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.14731)  

**Abstract**: We present Ring-lite, a Mixture-of-Experts (MoE)-based large language model optimized via reinforcement learning (RL) to achieve efficient and robust reasoning capabilities. Built upon the publicly available Ling-lite model, a 16.8 billion parameter model with 2.75 billion activated parameters, our approach matches the performance of state-of-the-art (SOTA) small-scale reasoning models on challenging benchmarks (e.g., AIME, LiveCodeBench, GPQA-Diamond) while activating only one-third of the parameters required by comparable models. To accomplish this, we introduce a joint training pipeline integrating distillation with RL, revealing undocumented challenges in MoE RL training. First, we identify optimization instability during RL training, and we propose Constrained Contextual Computation Policy Optimization(C3PO), a novel approach that enhances training stability and improves computational throughput via algorithm-system co-design methodology. Second, we empirically demonstrate that selecting distillation checkpoints based on entropy loss for RL training, rather than validation metrics, yields superior performance-efficiency trade-offs in subsequent RL training. Finally, we develop a two-stage training paradigm to harmonize multi-domain data integration, addressing domain conflicts that arise in training with mixed dataset. We will release the model, dataset, and code. 

---
# Expectation Confirmation Preference Optimization for Multi-Turn Conversational Recommendation Agent 

**Authors**: Xueyang Feng, Jingsen Zhang, Jiakai Tang, Wei Li, Guohao Cai, Xu Chen, Quanyu Dai, Yue Zhu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14302)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly propelled the development of Conversational Recommendation Agents (CRAs). However, these agents often generate short-sighted responses that fail to sustain user guidance and meet expectations. Although preference optimization has proven effective in aligning LLMs with user expectations, it remains costly and performs poorly in multi-turn dialogue. To address this challenge, we introduce a novel multi-turn preference optimization (MTPO) paradigm ECPO, which leverages Expectation Confirmation Theory to explicitly model the evolution of user satisfaction throughout multi-turn dialogues, uncovering the underlying causes of dissatisfaction. These causes can be utilized to support targeted optimization of unsatisfactory responses, thereby achieving turn-level preference optimization. ECPO ingeniously eliminates the significant sampling overhead of existing MTPO methods while ensuring the optimization process drives meaningful improvements. To support ECPO, we introduce an LLM-based user simulator, AILO, to simulate user feedback and perform expectation confirmation during conversational recommendations. Experimental results show that ECPO significantly enhances CRA's interaction capabilities, delivering notable improvements in both efficiency and effectiveness over existing MTPO methods. 

---
# GRAM: A Generative Foundation Reward Model for Reward Generalization 

**Authors**: Chenglong Wang, Yang Gan, Yifu Huo, Yongyu Mu, Qiaozhi He, Murun Yang, Bei Li, Tong Xiao, Chunliang Zhang, Tongran Liu, Jingbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14175)  

**Abstract**: In aligning large language models (LLMs), reward models have played an important role, but are standardly trained as discriminative models and rely only on labeled human preference data. In this paper, we explore methods that train reward models using both unlabeled and labeled data. Building on the generative models in LLMs, we develop a generative reward model that is first trained via large-scale unsupervised learning and then fine-tuned via supervised learning. We also show that by using label smoothing, we are in fact optimizing a regularized pairwise ranking loss. This result, in turn, provides a new view of training reward models, which links generative models and discriminative models under the same class of training objectives. The outcome of these techniques is a foundation reward model, which can be applied to a wide range of tasks with little or no further fine-tuning effort. Extensive experiments show that this model generalizes well across several tasks, including response ranking, reinforcement learning from human feedback, and task adaptation with fine-tuning, achieving significant performance improvements over several strong baseline models. 

---
# VL-GenRM: Enhancing Vision-Language Verification via Vision Experts and Iterative Training 

**Authors**: Jipeng Zhang, Kehao Miao, Renjie Pi, Zhaowei Wang, Runtao Liu, Rui Pan, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13888)  

**Abstract**: Reinforcement Fine-Tuning (RFT) with verifiable rewards has advanced large language models but remains underexplored for Vision-Language (VL) models. The Vision-Language Reward Model (VL-RM) is key to aligning VL models by providing structured feedback, yet training effective VL-RMs faces two major challenges. First, the bootstrapping dilemma arises as high-quality training data depends on already strong VL models, creating a cycle where self-generated supervision reinforces existing biases. Second, modality bias and negative example amplification occur when VL models hallucinate incorrect visual attributes, leading to flawed preference data that further misguides training. To address these issues, we propose an iterative training framework leveraging vision experts, Chain-of-Thought (CoT) rationales, and Margin-based Rejection Sampling. Our approach refines preference datasets, enhances structured critiques, and iteratively improves reasoning. Experiments across VL-RM benchmarks demonstrate superior performance in hallucination detection and multimodal reasoning, advancing VL model alignment with reinforcement learning. 

---
# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs 

**Authors**: Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, Jiang Bian, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14245)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for advancing the reasoning capabilities of Large Language Models (LLMs). However, a critical paradox clouds its efficacy: RLVR-tuned models often underperform their base models on the $Pass@K$ metric for solution-finding, leading to the hypothesis that RLVR merely re-weights existing reasoning paths at the cost of reasoning diversity. In this work, we resolve this contradiction by identifying the source of the problem: the $Pass@K$ metric itself is a flawed measure of reasoning, as it credits correct final answers that probably arise from inaccurate or incomplete chains of thought (CoTs). To address this, we introduce a more precise evaluation metric, $CoT$-$Pass@K$, which mandates that both the reasoning path and the final answer be correct. We provide a new theoretical foundation that formalizes how RLVR, unlike traditional RL, is uniquely structured to incentivize logical integrity. Our empirical results are supportive: using $CoT$-$Pass@K$, we observe that RLVR can incentivize the generalization of correct reasoning for all values of $K$. Furthermore, by analyzing the training dynamics, we find that this enhanced reasoning capability emerges early in the training process and smoothly generalizes. Our work provides a clear perspective on the role of RLVR, offers a more reliable method for its evaluation, and confirms its potential to genuinely advance machine reasoning. 

---
# Adaptive Guidance Accelerates Reinforcement Learning of Reasoning Models 

**Authors**: Vaskar Nath, Elaine Lau, Anisha Gunjal, Manasi Sharma, Nikhil Baharte, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.13923)  

**Abstract**: We study the process through which reasoning models trained with reinforcement learning on verifiable rewards (RLVR) can learn to solve new problems. We find that RLVR drives performance through two main means: (1) by compressing pass@$k$ into pass@1 and (2) via "capability gain" in which models learn to solve new problems that they previously could not solve even at high $k$. We find that while capability gain exists across model scales, learning to solve new problems is primarily driven through self-distillation. We demonstrate these findings across model scales ranging from 0.5B to 72B on >500,000 reasoning problems with prompts and verifiable final answers across math, science, and code domains. We further show that we can significantly improve pass@$k$ rates by leveraging natural language guidance for the model to consider within context while still requiring the model to derive a solution chain from scratch. Based of these insights, we derive $\text{Guide}$ - a new class of online training algorithms. $\text{Guide}$ adaptively incorporates hints into the model's context on problems for which all rollouts were initially incorrect and adjusts the importance sampling ratio for the "off-policy" trajectories in order to optimize the policy for contexts in which the hints are no longer present. We describe variants of $\text{Guide}$ for GRPO and PPO and empirically show that Guide-GRPO on 7B and 32B parameter models improves generalization over its vanilla counterpart with up to 4$\%$ macro-average improvement across math benchmarks. We include careful ablations to analyze $\text{Guide}$'s components and theoretically analyze Guide's learning efficiency. 

---
# TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization 

**Authors**: Mingkang Zhu, Xi Chen, Zhongdao Wang, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.14574)  

**Abstract**: Recent advancements in reinforcement learning from human feedback have shown that utilizing fine-grained token-level reward models can substantially enhance the performance of Proximal Policy Optimization (PPO) in aligning large language models. However, it is challenging to leverage such token-level reward as guidance for Direct Preference Optimization (DPO), since DPO is formulated as a sequence-level bandit problem. To address this challenge, this work decomposes the sequence-level PPO into a sequence of token-level proximal policy optimization problems and then frames the problem of token-level PPO with token-level reward guidance, from which closed-form optimal token-level policy and the corresponding token-level reward can be derived. Using the obtained reward and Bradley-Terry model, this work establishes a framework of computable loss functions with token-level reward guidance for DPO, and proposes a practical reward guidance based on the induced DPO reward. This formulation enables different tokens to exhibit varying degrees of deviation from reference policy based on their respective rewards. Experiment results demonstrate that our method achieves substantial performance improvements over DPO, with win rate gains of up to 7.5 points on MT-Bench, 6.2 points on AlpacaEval 2, and 4.3 points on Arena-Hard. Code is available at this https URL. 

---
# AviationLLM: An LLM-based Knowledge System for Aviation Training 

**Authors**: Jia'ang Wan, Feng Shen, Fujuan Li, Yanjin Sun, Yan Li, Shiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14336)  

**Abstract**: Aviation training is a core link in ensuring flight safety, improving industry efficiency and promoting sustainable development. It not only involves flight simulation but also requires the learning of a great deal of professional aviation theory knowledge. In the existing training system, the knowledge is mainly imparted by the the instructors. However, the number of instructors is limited and the professional answers obtained from the Internet are not accurate enough, resulting in low training efficiency. To address this, we introduced LLM, but the basic pre-trained model cannot provide accurate answers to professional fields, so we fine-tuned it. Traditional Supervised Fine-Tuning (SFT) risk generating superficially plausible but factually incorrect responses due to insufficient data coverage. To address this, we employ Direct Preference Optimization(DPO). This paper proposes Retrieval-Augmented LLM Alignment via Direct Preference Optimization(RALA-DPO). We select open source pre-trained LLM Qwen and adapt it to aviation theory training through DPO-based domain alignment. Simultaneously, to mitigate hallucinations caused by training data biases, knowledge obsolescence, or domain knowledge gaps, we implement Retrieval-Augmented Generation(RAG) technology that combines generative and retrieval models. RALA-DPO effectively retrieves relevant information from external knowledge bases and delivers precise and high-quality responses through the generative model. Experimental results demonstrate that RALA-DPO can improve accuracy in response to professional aviation knowledge. With integrated RAG mechanisms, this system can further improve the accuracy of answers and achieve zero-cost knowledge updates simultaneously. 

---
# Med-REFL: Medical Reasoning Enhancement via Self-Corrected Fine-grained Reflection 

**Authors**: Zongxian Yang, Jiayu Qian, Zegao Peng, Haoyu Zhang, Zhi-An Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13793)  

**Abstract**: Large reasoning models have recently made significant strides in mathematical and code reasoning, yet their success has not transferred smoothly to the medical domain. While multiple factors contribute to this disparity, a critical issue is the inadequate focus on the quality of intermediate reflection steps, which is particularly crucial in high-stakes medical scenarios. To address this challenge, we propose Med-REFL, a \underline{\textbf{Med}}ical \underline{\textbf{R}}easoning \underline{\textbf{E}}nhancement via self-corrected \underline{\textbf{F}}ine-grained ref\underline{\textbf{L}}ection. Our method leverages a tree-of-thought approach to decompose medical questions into fine-grained reasoning paths, quantitatively evaluating each step and its subsequent reflections. These assessments enable automatic construction of direct preference optimization data, reducing reliance on expensive expert annotations while guiding models to identify and correct reasoning errors. Experimental results on the MedQA-USMLE benchmark demonstrate Med-REFL achieves consistent improvements, with average gains up to 4.11\%. Notably, it further boosts the state-of-the-art performance of 7B/8B models by an additional 4.13\%. Furthermore, Med-REFL exhibits strong generalization capabilities and robustness across several challenging medical question-answering datasets. Our work illustrates that prioritizing reflection quality leads to more accurate and trustworthy reasoning in medical AI applications. Checkpoints, code, and data can be found \href{this https URL}{here}. 

---
# Personalized Constitutionally-Aligned Agentic Superego: Secure AI Behavior Aligned to Diverse Human Values 

**Authors**: Nell Watson, Ahmed Amer, Evan Harris, Preeti Ravindra, Shujun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13774)  

**Abstract**: Agentic AI systems, possessing capabilities for autonomous planning and action, exhibit immense potential across diverse domains. However, their practical deployment is significantly hampered by challenges in aligning their behavior with varied human values, complex safety requirements, and specific compliance needs. Existing alignment methodologies often falter when faced with the intricate task of providing deep, personalized contextual information without inducing confabulation or operational inefficiencies. This paper introduces a novel solution: a 'superego' agent, designed as a personalized oversight mechanism for agentic AI. This system dynamically steers AI planning by referencing user-selected "Creed Constitutions"-encapsulating diverse rule sets-with adjustable adherence levels to fit non-negotiable values. A real-time compliance enforcer validates plans against these constitutions and a universal ethical floor before execution. We present a functional system, including a demonstration interface (this http URL) with a prototypical constitution-sharing portal, and successful integration with third-party models via the Model Context Protocol (MCP). Comprehensive benchmark evaluations (HarmBench, AgentHarm) demonstrate that our Superego agent dramatically reduces harmful outputs, achieving up to a 98.3% harm score reduction and near-perfect refusal rates (e.g., 100% with Claude Sonnet 4 on AgentHarm's harmful set) for leading LLMs like Gemini 2.5 Flash and GPT-4o. This approach substantially simplifies personalized AI alignment, rendering agentic systems more reliably attuned to individual and cultural contexts, while also enabling substantial safety improvements. An overview on this research with examples is available at this https URL. 

---
# Alignment Quality Index (AQI) : Beyond Refusals: AQI as an Intrinsic Alignment Diagnostic via Latent Geometry, Cluster Divergence, and Layer wise Pooled Representations 

**Authors**: Abhilekh Borah, Chhavi Sharma, Danush Khanna, Utkarsh Bhatt, Gurpreet Singh, Hasnat Md Abdullah, Raghav Kaushik Ravi, Vinija Jain, Jyoti Patel, Shubham Singh, Vasu Sharma, Arpita Vats, Rahul Raja, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.13901)  

**Abstract**: Alignment is no longer a luxury, it is a necessity. As large language models (LLMs) enter high-stakes domains like education, healthcare, governance, and law, their behavior must reliably reflect human-aligned values and safety constraints. Yet current evaluations rely heavily on behavioral proxies such as refusal rates, G-Eval scores, and toxicity classifiers, all of which have critical blind spots. Aligned models are often vulnerable to jailbreaking, stochasticity of generation, and alignment faking.
To address this issue, we introduce the Alignment Quality Index (AQI). This novel geometric and prompt-invariant metric empirically assesses LLM alignment by analyzing the separation of safe and unsafe activations in latent space. By combining measures such as the Davies-Bouldin Score (DBS), Dunn Index (DI), Xie-Beni Index (XBI), and Calinski-Harabasz Index (CHI) across various formulations, AQI captures clustering quality to detect hidden misalignments and jailbreak risks, even when outputs appear compliant. AQI also serves as an early warning signal for alignment faking, offering a robust, decoding invariant tool for behavior agnostic safety auditing.
Additionally, we propose the LITMUS dataset to facilitate robust evaluation under these challenging conditions. Empirical tests on LITMUS across different models trained under DPO, GRPO, and RLHF conditions demonstrate AQI's correlation with external judges and ability to reveal vulnerabilities missed by refusal metrics. We make our implementation publicly available to foster future research in this area. 

---
# Fake it till You Make it: Reward Modeling as Discriminative Prediction 

**Authors**: Runtao Liu, Jiahao Zhan, Yingqing He, Chen Wei, Alan Yuille, Qifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13846)  

**Abstract**: An effective reward model plays a pivotal role in reinforcement learning for post-training enhancement of visual generative models. However, current approaches of reward modeling suffer from implementation complexity due to their reliance on extensive human-annotated preference data or meticulously engineered quality dimensions that are often incomplete and engineering-intensive. Inspired by adversarial training in generative adversarial networks (GANs), this paper proposes GAN-RM, an efficient reward modeling framework that eliminates manual preference annotation and explicit quality dimension engineering. Our method trains the reward model through discrimination between a small set of representative, unpaired target samples(denoted as Preference Proxy Data) and model-generated ordinary outputs, requiring only a few hundred target samples. Comprehensive experiments demonstrate our GAN-RM's effectiveness across multiple key applications including test-time scaling implemented as Best-of-N sample filtering, post-training approaches like Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). 

---
