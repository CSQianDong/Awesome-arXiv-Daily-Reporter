# ProCeedRL: Process Critic with Exploratory Demonstration Reinforcement Learning for LLM Agentic Reasoning 

**Authors**: Jingyue Gao, Yanjiang Guo, Xiaoshuai Chen, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.02006)  

**Abstract**: Reinforcement Learning (RL) significantly enhances the reasoning abilities of large language models (LLMs), yet applying it to multi-turn agentic tasks remains challenging due to the long-horizon nature of interactions and the stochasticity of environmental feedback. We identify a structural failure mode in agentic exploration: suboptimal actions elicit noisy observations into misleading contexts, which further weaken subsequent decision-making, making recovery increasingly difficult. This cumulative feedback loop of errors renders standard exploration strategies ineffective and susceptible to the model's reasoning and the environment's randomness. To mitigate this issue, we propose ProCeedRL: Process Critic with Explorative Demonstration RL, shifting exploration from passive selection to active intervention. ProCeedRL employs a process-level critic to monitor interactions in real time, incorporating reflection-based demonstrations to guide agents in stopping the accumulation of errors. We find that this approach significantly exceeds the model's saturated exploration performance, demonstrating substantial exploratory benefits. By learning from exploratory demonstrations and on-policy samples, ProCeedRL significantly improves exploration efficiency and achieves superior performance on complex deep search and embodied tasks. 

---
# MM-ReCoder: Advancing Chart-to-Code Generation with Reinforcement Learning and Self-Correction 

**Authors**: Zitian Tang, Xu Zhang, Jianbo Yuan, Yang Zou, Varad Gunjal, Songyao Jiang, Davide Modolo  

**Link**: [PDF](https://arxiv.org/pdf/2604.01600)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently demonstrated promising capabilities in multimodal coding tasks such as chart-to-code generation. However, existing methods primarily rely on supervised fine-tuning (SFT), which requires the model to learn code patterns through chart-code pairs but does not expose the model to a code execution environment. Moreover, while self-correction through execution feedback offers a potential route to improve coding quality, even state-of-the-art MLLMs have been shown to struggle with effective self-correction. In this work, we introduce MM-ReCoder, a chart-to-code generation model trained with reinforcement learning (RL) and equipped with self-correction ability. We propose a two-stage multi-turn self-correction RL strategy based on Group Relative Policy Optimization (GRPO). The first stage enhances the model's self-correction ability via rolling out a shared first turn, while the second stage improves the coding capability with full-trajectory optimization. MM-ReCoder learns to produce more accurate and executable code through the interaction with the environment and by iteratively correcting its own outputs. Our results on three chart-to-code benchmarks demonstrate the state-of-the-art performance of MM-ReCoder. 

---
# ThinkTwice: Jointly Optimizing Large Language Models for Reasoning and Self-Refinement 

**Authors**: Difan Jiao, Qianfeng Wen, Blair Yang, Zhenwei Tang, Ashton Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2604.01591)  

**Abstract**: We introduce ThinkTwice, a simple two-phase framework that jointly optimizes LLMs to solve reasoning problems and refine the answers, based on Group Relative Policy Optimization (GRPO). In each pair of training steps, ThinkTwice first optimizes the model on solving reasoning problems, then optimizes it on refining its own solutions to the same problems, using the same binary correctness reward in both phases without correctness signals or critique annotations. Across five mathematical reasoning benchmarks and two model families including Qwen3-4B and Olmo3-7B, ThinkTwice substantially improves both reasoning and refinement performance over competitive online policy optimization baselines. Specifically, on Qwen3-4B, ThinkTwice outperforms GRPO on AIME by 5 percentage points before refinement and by 11.5 points after one self-refinement step, measured by pass@4. Analysis of the training dynamics of ThinkTwice reveals an implicit rectify-then-fortify curriculum: refinement predominantly corrects errors early in training and naturally shifts toward preserving already-correct solutions as the model improves, yielding a more rectified reward signal. Our work establishes joint training of reasoning and self-refinement as a principled and effective methodology for RLVR. 

---
# Batched Contextual Reinforcement: A Task-Scaling Law for Efficient Reasoning 

**Authors**: Bangji Yang, Hongbo Ma, Jiajun Fan, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.02322)  

**Abstract**: Large Language Models employing Chain-of-Thought reasoning achieve strong performance but suffer from excessive token consumption that inflates inference costs. Existing efficiency methods such as explicit length penalties, difficulty estimators, or multi-stage curricula either degrade reasoning quality or require complex training pipelines. We introduce Batched Contextual Reinforcement, a minimalist, single-stage training paradigm that unlocks efficient reasoning through a simple structural modification: training the model to solve N problems simultaneously within a shared context window, rewarded purely by per-instance accuracy. This formulation creates an implicit token budget that yields several key findings: (1) We identify a novel task-scaling law: as the number of concurrent problems N increases during inference, per-problem token usage decreases monotonically while accuracy degrades far more gracefully than baselines, establishing N as a controllable throughput dimension. (2) BCR challenges the traditional accuracy-efficiency trade-off by demonstrating a "free lunch" phenomenon at standard single-problem inference. Across both 1.5B and 4B model families, BCR reduces token usage by 15.8% to 62.6% while consistently maintaining or improving accuracy across five major mathematical benchmarks. (3) Qualitative analyses reveal emergent self-regulated efficiency, where models autonomously eliminate redundant metacognitive loops without explicit length supervision. (4) Crucially, we empirically demonstrate that implicit budget constraints successfully circumvent the adversarial gradients and catastrophic optimization collapse inherent to explicit length penalties, offering a highly stable, constraint-based alternative for length control. These results prove BCR practical, showing simple structural incentives unlock latent high-density reasoning in LLMs. 

---
# Optimizing RAG Rerankers with LLM Feedback via Reinforcement Learning 

**Authors**: Yuhang Wu, Xiangqing Shen, Fanfan Wang, Cangqi Zhou, Zhen Wu, Xinyu Dai, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2604.02091)  

**Abstract**: Rerankers play a pivotal role in refining retrieval results for Retrieval-Augmented Generation. However, current reranking models are typically optimized on static human annotated relevance labels in isolation, decoupled from the downstream generation process. This isolation leads to a fundamental misalignment: documents identified as topically relevant by information retrieval metrics often fail to provide the actual utility required by the LLM for precise answer generation. To bridge this gap, we introduce ReRanking Preference Optimization (RRPO), a reinforcement learning framework that directly aligns reranking with the LLM's generation quality. By formulating reranking as a sequential decision-making process, RRPO optimizes for context utility using LLM feedback, thereby eliminating the need for expensive human annotations. To ensure training stability, we further introduce a reference-anchored deterministic baseline. Extensive experiments on knowledge-intensive benchmarks demonstrate that RRPO significantly outperforms strong baselines, including the powerful list-wise reranker RankZephyr. Further analysis highlights the versatility of our framework: it generalizes seamlessly to diverse readers (e.g., GPT-4o), integrates orthogonally with query expansion modules like Query2Doc, and remains robust even when trained with noisy supervisors. 

---
# Causal Scene Narration with Runtime Safety Supervision for Vision-Language-Action Driving 

**Authors**: Yun Li, Yidu Zhang, Simon Thompson, Ehsan Javanmardi, Manabu Tsukada  

**Link**: [PDF](https://arxiv.org/pdf/2604.01723)  

**Abstract**: Vision-Language-Action (VLA) models for autonomous driving must integrate diverse textual inputs, including navigation commands, hazard warnings, and traffic state descriptions, yet current systems often present these as disconnected fragments, forcing the model to discover on its own which environmental constraints are relevant to the current maneuver. We introduce Causal Scene Narration (CSN), which restructures VLA text inputs through intent-constraint alignment, quantitative grounding, and structured separation, at inference time with zero GPU cost. We complement CSN with Simplex-based runtime safety supervision and training-time alignment via Plackett-Luce DPO with negative log-likelihood (NLL) regularization. A multi-town closed-loop CARLA evaluation shows that CSN improves Driving Score by +31.1% on original LMDrive and +24.5% on the preference-aligned variant. A controlled ablation reveals that causal structure accounts for 39.1% of this gain, with the remainder attributable to information content alone. A perception noise ablation confirms that CSN's benefit is robust to realistic sensing errors. Semantic safety supervision improves Infraction Score, while reactive Time-To-Collision monitoring degrades performance, demonstrating that intent-aware monitoring is needed for VLA systems. 

---
# Preference learning in shades of gray: Interpretable and bias-aware reward modeling for human preferences 

**Authors**: Simona-Vasilica Oprea, Adela Bâra  

**Link**: [PDF](https://arxiv.org/pdf/2604.01312)  

**Abstract**: Learning human preferences in language models remains fundamentally challenging, as reward modeling relies on subtle, subjective comparisons or shades of gray rather than clear-cut labels. This study investigates the limits of current approaches and proposes a feature-augmented framework to better capture the multidimensional nature of human judgment. Using the Anthropic HHRLHF dataset, we evaluate ten diverse large language models LLMs under a standard pairwise preference setting, where baseline performance remains below 0.74 ROC AUC, highlighting the difficulty of the task. To address this, we enrich textual representations with interpretable signals: response length, refusal indicators, toxicity scores and prompt response semantic similarity, enabling models to explicitly capture key aspects of helpfulness, safety and relevance. The proposed hybrid approach yields consistent improvements across all models, achieving up to 0.84 ROC AUC and significantly higher pairwise accuracy, with DeBERTav3Large demonstrating the best performance. Beyond accuracy, we integrate SHAP and LIME to provide fine-grained interpretability, revealing that model decisions depend on contextualized safety and supportive framing rather than isolated keywords. We further analyze bias amplification, showing that while individual features have weak marginal effects, their interactions influence preference learning. 

---
# PLOT: Enhancing Preference Learning via Optimal Transport 

**Authors**: Liang Zhu, Yuelin Bai, Xiankun Ren, Jiaxi Yang, Lei Zhang, Feiteng Fang, Hamid Alinejad-Rokny, Minghuan Tan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01837)  

**Abstract**: Preference learning in Large Language Models (LLMs) has advanced significantly, yet existing methods remain limited by modest performance gains, high computational costs, hyperparameter sensitivity, and insufficient modeling of global token-level relationships. We introduce PLOT, which enhances Preference Learning in fine-tuning-based alignment through a token-level loss derived from Optimal Transport. By formulating preference learning as an Optimal Transport Problem, PLOT aligns model outputs with human preferences while preserving the original distribution of LLMs, ensuring stability and robustness. Furthermore, PLOT leverages token embeddings to capture semantic relationships, enabling globally informed optimization. Experiments across two preference categories - Human Values and Logic & Problem Solving - spanning seven subpreferences demonstrate that PLOT consistently improves alignment performance while maintaining fluency and coherence. These results substantiate optimal transport as a principled methodology for preference learning, establishing a theoretically grounded framework that provides new insights for preference learning of LLMs. 

---
# DEFT: Distribution-guided Efficient Fine-Tuning for Human Alignment 

**Authors**: Liang Zhu, Feiteng Fang, Yuelin Bai, Longze Chen, Zhexiang Zhang, Minghuan Tan, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01787)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF), using algorithms like Proximal Policy Optimization (PPO), aligns Large Language Models (LLMs) with human values but is costly and unstable. Alternatives have been proposed to replace PPO or integrate Supervised Fine-Tuning (SFT) and contrastive learning for direct fine-tuning and value alignment. However, these methods still require voluminous data to learn preferences and may weaken the generalization ability of LLMs. To further enhance alignment efficiency and performance while mitigating the loss of generalization ability, this paper introduces Distribution-guided Efficient Fine-Tuning (DEFT), an efficient alignment framework incorporating data filtering and distributional guidance by calculating the differential distribution reward based on the output distribution of language model and the discrepancy distribution of preference data. A small yet high-quality subset is filtered from the raw data using a differential distribution reward, which is then incorporated into existing alignment methods to guide the model's output distribution. Experimental results demonstrate that the methods enhanced by DEFT outperform the original methods in both alignment capability and generalization ability, with significantly reduced training time. 

---
# Scaling Reasoning Tokens via RL and Parallel Thinking: Evidence From Competitive Programming 

**Authors**: Qianfan Zhang, Tianyu Guo, Xuandi Ren, Jiale Chen, Ming Ding, Ran Xin, Xia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.01302)  

**Abstract**: We study how to scale reasoning token budgets for competitive programming through two complementary approaches: training-time reinforcement learning (RL) and test-time parallel thinking. During RL training, we observe an approximately log-linear relationship between validation accuracy and the average number of generated reasoning tokens over successive checkpoints, and show two ways to shift this training trajectory: verification RL warmup raises the starting point, while randomized clipping produces a steeper trend in the observed regime. As scaling single-generation reasoning during RL quickly becomes expensive under full attention, we introduce a multi-round parallel thinking pipeline that distributes the token budget across threads and rounds of generation, verification, and refinement. We train the model end-to-end on this pipeline to match the training objective to the test-time structure. Starting from Seed-OSS-36B, the full system with 16 threads and 16 rounds per thread matches the underlying RL model's oracle pass@16 at pass@1 using 7.6 million tokens per problem on average, and surpasses GPT-5-high on 456 hard competitive programming problems from AetherCode. 

---
# When Reward Hacking Rebounds: Understanding and Mitigating It with Representation-Level Signals 

**Authors**: Rui Wu, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01476)  

**Abstract**: Reinforcement learning for LLMs is vulnerable to reward hacking, where models exploit shortcuts to maximize reward without solving the intended task. We systematically study this phenomenon in coding tasks using an environment-manipulation setting, where models can rewrite evaluator code to trivially pass tests without solving the task, as a controlled testbed. Across both studied models, we identify a reproducible three-phase rebound pattern: models first attempt to rewrite the evaluator but fail, as their rewrites embed test cases their own solutions cannot pass. They then temporarily retreat to legitimate solving. When legitimate reward remains scarce, they rebound into successful hacking with qualitatively different strategies. Using representation engineering, we extract concept directions for shortcut, deception, and evaluation awareness from domain-general contrastive pairs and find that the shortcut direction tracks hacking behavior most closely, making it an effective representational proxy for detection. Motivated by this finding, we propose Advantage Modification, which integrates shortcut concept scores into GRPO advantage computation to penalize hacking rollouts before policy updates. Because the penalty is internalized into the training signal rather than applied only at inference time, Advantage Modification provides more robust suppression of hacking compared with generation-time activation steering. 

---
