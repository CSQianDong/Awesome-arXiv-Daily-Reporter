# RecRM-Bench: Benchmarking Multidimensional Reward Modeling for Agentic Recommender Systems 

**Authors**: Wenwen Zeng, Jinhui Zhang, Hao Chen, Zhaoyu Hu, Yongqi Liang, Jiajun Chai, Dengcan Liu, Zhenfeng Liu, Shurui Yan, Minglong Xue, Xiaohan Wang, Wei Lin, Guojun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.11874)  

**Abstract**: The integration of Large Language Model (LLM) agents is transforming recommender systems from simple query-item matching towards deeply personalized and interactive recommendations. Reinforcement Learning (RL) provides an essential framework for the optimization of these agents in recommendation tasks. However, current methodologies remain limited by a reliance on single dimensional outcome-based rewards that focus exclusively on final user interactions, overlooking critical intermediate capabilities, such as instruction following and complex intent understanding. Despite the necessity for designing multi-dimensional reward, the field lacks a standardized benchmark to facilitate this development. To bridge this gap, we introduce RecRM-Bench, the largest and most comprehensive benchmark to date for agentic recommender systems. It comprises over 1 million structured entries across four core evaluation dimensions: instruction following, factual consistency, query-item relevance, and fine-grained user behavior prediction. By supporting comprehensive assessment from syntactic compliance to complex intent grounding and preference modeling, RecRM-Bench provides a foundational dataset for training sophisticated reward models. Furthermore, we propose a systematic framework for the construction of multi-dimensional reward models and the integration of a hybrid reward function, establishing a robust foundation for developing reliable and highly capable agentic recommender systems. The complete RecRM-Bench dataset is publicly available at this https URL. 

---
# TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching 

**Authors**: Truong Nguyen, Tien-Phat Nguyen, Linh Ngo Van, Duy Minh Ho Nguyen, Khoa Doan, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.12288)  

**Abstract**: Direct Preference Optimization (DPO) is a widely used RL-free method for aligning language models from pairwise preferences, but it models preferences over full sequences even though generation is driven by per-token decisions. Existing token-level extensions typically decompose a sequence-level Bradley-Terry objective across timesteps, leaving per-prefix (state-wise) optimality implicit. We study how to recover token-level preference optimality using only standard sequence-level pairwise comparisons. We introduce Token-level Bregman Preference Optimization (TBPO), which posits a token-level Bradley-Terry preference model over next-token actions conditioned on the prefix, and derive a Bregman-divergence density-ratio matching objective that generalizes the logistic/DPO loss while preserving the optimal policy induced by the token-level model and maintaining DPO-like simplicity. We introduce two instantiations: TBPO-Q, which explicitly learns a lightweight state baseline, and TBPO-A, which removes the baseline through advantage normalization. Across instruction following, helpfulness/harmlessness, and summarization benchmarks, TBPO improves alignment quality and training stability and increases output diversity relative to strong sequence-level and token-level baselines. 

---
# Learning Agentic Policy from Action Guidance 

**Authors**: Yuxiang Ji, Zengbin Wang, Yong Wang, Shidong Yang, Ziyu Ma, Guanhua Chen, Zonghua Sun, Liaoni Wu, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12004)  

**Abstract**: Agentic reinforcement learning (RL) for Large Language Models (LLMs) critically depends on the exploration capability of the base policy, as training signals emerge only within its in-capability region. For tasks where the base policy cannot reach reward states, additional training or external guidance is needed to recover effective learning signals. Rather than relying on costly iterative supervised fine tuning (SFT), we exploit the abundant action data generated in everyday human interactions. We propose \textsc{ActGuide-RL}, which injects action data as plan-style reference guidance, enabling the agentic policy to overcome reachability barriers to reward states. Guided and unguided rollouts are then jointly optimized via mixed-policy training, internalizing the exploration gains back into the unguided policy. Motivated by a theoretical and empirical analysis of the benefit-risk trade-off, we adopt a minimal intervention principle that invokes guidance only as an adaptive fallback, matching task difficulty while minimizing off-policy risk. On search-agent benchmarks, \textsc{ActGuide-RL} substantially improves over zero RL (+10.7 pp on GAIA and +19 pp on XBench with Qwen3-4B), and performs on par with the SFT+RL pipeline without any cold start. This suggests a new paradigm for agentic RL that reduces the reliance on heavy SFT data by using scalable action guidance instead. 

---
# Combining On-Policy Optimization and Distillation for Long-Context Reasoning in Large Language Models 

**Authors**: Miguel Moura Ramos, Duarte M. Alves, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2605.12227)  

**Abstract**: Adapting large language models (LLMs) to long-context tasks requires post-training methods that remain accurate and coherent over thousands of tokens. Existing approaches are limited in several ways: 1) off-policy methods such as supervised fine-tuning (SFT) and knowledge distillation (KD) suffer from exposure bias and limited recovery from model-generated errors over long horizons; 2) on-policy reinforcement learning methods such as Group Relative Policy Optimization (GRPO) better align training with model-generated states, but are unstable and sample-inefficient due to sparse rewards; 3) on-policy distillation (OPD) provides dense token-level guidance, but does not directly optimize arbitrary reward signals. In this paper, we propose Distilled Group Relative Policy Optimization (dGRPO), a method for long-context reasoning that augments GRPO with dense guidance from a stronger teacher via OPD. We also introduce LongBlocks, a synthetic long-context dataset spanning multi-hop reasoning, contextual grounding, and long-form generation. We conduct extensive experiments and ablations comparing off-policy training, sparse-reward GRPO, and our combined approach, leading to an improved recipe for long-context alignment. Overall, our results show that combining outcome-based policy optimization with knowledge distillation in a single objective provides a more stable and effective path to long-context reasoning, while preserving short-context capabilities. 

---
# Qwen-Scope: Turning Sparse Features into Development Tools for Large Language Models 

**Authors**: Boyi Deng, Xu Wang, Yaoning Wang, Yu Wan, Yubo Ma, Baosong Yang, Haoran Wei, Jialong Tang, Huan Lin, Ruize Gao, Tianhao Li, Qian Cao, Xuancheng Ren, Xiaodong Deng, An Yang, Fei Huang, Dayiheng Liu, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11887)  

**Abstract**: Large language models have achieved remarkable capabilities across diverse tasks, yet their internal decision-making processes remain largely opaque, limiting our ability to inspect, control, and systematically improve them. This opacity motivates a growing body of research in mechanistic interpretability, with sparse autoencoders (SAEs) emerging as one of the most promising tools for decomposing model activations into sparse, interpretable feature representations. We introduce Qwen-Scope, an open-source suite of SAEs built on the Qwen model family, comprising 14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series, covering both dense and mixture-of-expert architectures. Built on top of these SAEs, we show that SAEs can go beyond post-hoc analysis to serve as practical interfaces for model development along four directions: (i) inference-time steering, where SAE feature directions control language, concepts, and preferences without modifying model weights; (ii) evaluation analysis, where activated SAE features provide a representation-level proxy for benchmark redundancy and capability coverage; (iii) data-centric workflows, where SAE features support multilingual toxicity classification and safety-oriented data synthesis; and (iv) post-training optimization, where SAE-derived signals are incorporated into supervised fine-tuning and reinforcement learning objectives to mitigate undesirable behaviors such as code-switching and repetition. Together, these results demonstrate that SAEs can serve not only as post-hoc analysis tools, but also as reusable representation-level interfaces for diagnosing, controlling, evaluating, and improving large language models. By open-sourcing Qwen-Scope, we aim to support mechanistic research and accelerate practical workflows that connect model internals to downstream behavior. 

---
# SkillGraph: Skill-Augmented Reinforcement Learning for Agents via Evolving Skill Graphs 

**Authors**: Xiaoyuan Li, Moxin Li, Keqin Bao, Yubo Ma, Wenjie Wang, Dayiheng Liu, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.12039)  

**Abstract**: Skill libraries enable large language model agents to reuse experience from past interactions, but most existing libraries store skills as isolated entries and retrieve them only by semantic similarity. This leads to two key challenges for compositional tasks. Firstly, an agent must identify not only relevant skills but also how they depend on and build upon each other. Secondly, it also makes library maintenance difficult, since the system lacks structural cues for deciding when skills should be merged, split, or removed. We propose SKILLGRAPH, a framework that represents reusable skills as nodes in a directed graph, with typed edges encoding prerequisite, enhancement, and co-occurrence relations. Given a new task, SKILLGRAPH retrieves not just individual skills, but an ordered skill subgraph that can guide multi-step decision making. The graph is continuously updated from agent trajectories and reinforcement learning feedback, allowing both the skill library and the agent policy to improve together. Experiments on ALFWorld, WebShop, and seven search-augmented QA tasks show that SKILLGRAPH achieves state-of-the-art performance against memory-augmented RL methods, with especially large gains on complex tasks that require composing multiple skills. 

---
# Self-Distilled Trajectory-Aware Boltzmann Modeling: Bridging the Training-Inference Discrepancy in Diffusion Language Models 

**Authors**: Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Yibing Liu, Xinyu Fu, Shi Wu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, Haoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.11854)  

**Abstract**: Diffusion Language Models (DLMs) have recently emerged as a promising alternative to autoregressive language models, offering stronger global awareness and highly parallel generation. However, post-training DLMs with standard Negative Evidence Lower Bound (NELBO)-based supervised fine-tuning remains inefficient: training reconstructs randomly masked tokens in a single step, whereas inference follows a confidence-guided, multi-step easy-to-hard denoising trajectory. Recent trajectory-based self-distillation methods exploit such inference trajectories mainly for sampling-step compression and acceleration, often improving decoding efficiency without substantially enhancing the model's underlying capability, and may even degrade performance under full diffusion decoding. In this work, we ask whether self-distilled trajectories can be used not merely for faster inference, but for genuine knowledge acquisition. Although these trajectories lie on the pretrained DLM's own distributional manifold and thus offer a potentially lower optimization barrier, we find that naively fine-tuning on them with standard NELBO objectives yields only marginal gains. To address this limitation, we propose \textbf{T}rajectory-\textbf{A}ligned optimization via \textbf{Bo}ltzmann \textbf{M}odeling (\textbf{TABOM}), a self-distilled trajectory-based post-training framework that aligns training with the easy-to-hard structure of inference. TABOM models the inference unmasking preference as a Boltzmann distribution over predictive entropies and derives a tractable pairwise ranking objective to align the model's certainty ordering with the observed decoding trajectory. Empirically, TABOM achieves substantial gains in new domains, expands the effective knowledge boundary of DLMs, and significantly mitigates catastrophic forgetting compared with standard SFT. 

---
# YFPO: A Preliminary Study of Yoked Feature Preference Optimization with Neuron-Guided Rewards for Mathematical Reasoning 

**Authors**: Yifan Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.11906)  

**Abstract**: Preference optimization has become an important post-training paradigm for improving the reasoning abilities of large language models. Existing methods typically rely on externally constructed preference data, using preferred and dispreferred responses as sample-level supervision. However, such external signals rarely make explicit use of capability-related information contained in the model's internal representations. For mathematical reasoning, certain neuron groups may exhibit activation patterns associated with mathematical knowledge, symbolic manipulation, or logical reasoning. Similar to reflexive behavioral signals, these internal activations may provide a coarse indication of whether the model is engaging math-related this http URL introduce YFPO, short for Yoked Feature Preference Optimization, a preliminary neuron-guided preference optimization framework for mathematical reasoning. YFPO first uses AttnLRP to identify math-related neurons, and then constructs an auxiliary reward from their activation margin between preferred and dispreferred responses. This design augments external preference learning with internal neuron-level signals. We conduct preliminary experiments on a small-scale language model using GSM8K as the main benchmark. Results suggest that neuron-level signals can interact with preference optimization and occasionally improve reasoning performance, offering a promising direction for more fine-grained and interpretable reasoning-oriented post-training. 

---
# Enhancing Multilingual Counterfactual Generation through Alignment-as-Preference Optimization 

**Authors**: Yilong Wang, Qianli Wang, Bohao Chu, Yihong Liu, Jing Yang, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.11632)  

**Abstract**: Self-generated counterfactual explanations (SCEs) are minimally modified inputs (minimality) generated by large language models (LLMs) that flip their own predictions (validity), offering a causally grounded approach to unraveling black-box LLM behavior. Yet extending them beyond English remains challenging: existing methods struggle to produce valid SCEs in non-dominant languages, and a persistent trade-off between validity and minimality undermines explanation quality. We introduce Macro, a preference alignment framework that applies Direct Preference Optimization (DPO) to multilingual SCE generation, using a composite scoring function to construct preference pairs that effectively translate the trade-off into measurable preference signals. Experiments across four LLMs and seven typologically diverse languages show that Macro improves validity by 12.55\% on average over the chain-of-thought baseline without degrading minimality, while avoiding the severe minimality violations of the translation-based baseline. Compared to supervised fine-tuning, Macro achieves superior performance on both metrics, confirming that explicit preference optimization is essential for balancing this trade-off. Further analyses reveal that Macro increases cross-lingual perturbation alignment and mitigates common generation errors. Our results highlight preference optimization as a promising direction for enhancing multilingual model explanations. 

---
# Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting 

**Authors**: Cheng Wang, Qin Liu, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11538)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as a promising approach for improving the reasoning capabilities of large language models. However, it struggles to effectively balance the tradeoff between exploration and exploitation during training, often resulting in suboptimal performance. Motivated by the theoretical insight that changes in entropy are governed by the covariance between token probabilities and their corresponding advantages, we propose a hyperparameter-free, covariance-weighted optimization method that dynamically down-weights extreme token-level updates via a Gaussian kernel. This approach automatically reduces the instability caused by exploration-exploitation trade-off while preserving informative learning signals. Extensive empirical evaluations show that our approach improves downstream performance across reasoning benchmarks compared with GRPO, and effectively stablizes entropy as training progresses. 

---
# StoicLLM: Preference Optimization for Philosophical Alignment in Small Language Models 

**Authors**: Ishmam Khan, Sindhuja Thogarrati, Shuo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11483)  

**Abstract**: While large language models excel at factual adaptation, their ability to internalize nuanced philosophical frameworks under severe data constraints remains underexplored. We investigate this by specializing small LLMs on micro-datasets of foundational Stoic texts using preference optimization (ORPO, AlphaPO). Evaluated via a multi-model critic bank, our results show that just 300 high-fidelity examples can induce strong alignment with inward-facing Stoic virtues, closely approaching few-shot prompting while freeing the context window. Critically, however, all models, including few-shot baselines, exhibit a persistent failure on Stoicism's outward-facing cosmopolitan duties, pointing to a representational limitation of small models that micro-dataset adaptation alone cannot overcome. 

---
# Not How Many, But Which: Parameter Placement in Low-Rank Adaptation 

**Authors**: Arijit Sehanobish, Charles Lovering  

**Link**: [PDF](https://arxiv.org/pdf/2605.12207)  

**Abstract**: We study the \textit{parameter placement problem}: given a fixed budget of $k$ trainable entries within the B matrix of a LoRA adapter (A frozen), does the choice of which $k$ matter? Under supervised fine-tuning, random and informed subsets achieve comparable performance. Under GRPO on base models, random placement fails to improve over the base model, while gradient-informed placement recovers standard LoRA accuracy. This regime dependence traces to gradient structure: SFT gradients are low-rank and directionally stable, so any subset accumulates coherent updates; GRPO gradients are high-rank and near-orthogonal across steps, so only elements with consistently signed gradients retain the learning signal. Our scoring procedure identifies these critical parameters in under 10 seconds at less than 0.5% of training cost. Selected parameters concentrate on residual-stream-writing projections (V, O, Down), stable across model families and scales (1.5B - 8B). 

---
# ORCE: Order-Aware Alignment of Verbalized Confidence in Large Language Models 

**Authors**: Chen Li, Xiaoling Hu, Songzhu Zheng, Jiawei Zhou, Chao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.12446)  

**Abstract**: Large language models (LLMs) often produce answers with high certainty even when they are incorrect, making reliable confidence estimation essential for deployment in real-world scenarios. Verbalized confidence, where models explicitly state their confidence in natural language, provides a flexible and user-facing uncertainty signal that can be applied even when token logits are unavailable. However, existing verbalized-confidence methods often optimize answer generation and confidence generation jointly, which can cause confidence-alignment objectives to interfere with answer accuracy. In this work, we propose a decoupled and order-aware framework for verbalized confidence calibration. Our method first generates an answer and then estimates confidence conditioned on the fixed question--answer pair, allowing confidence optimization without directly perturbing the answer-generation process. To align confidence with correctness likelihood, we construct a sampling-based surrogate from multiple model completions and optimize rank-based reinforcement learning objectives that encourage responses with higher estimated correctness likelihood to receive higher verbalized confidence. Experiments on reasoning and knowledge-intensive benchmarks show that our method improves calibration and failure prediction performance while largely preserving answer accuracy. These results demonstrate that verbalized confidence can be more reliably aligned by decoupling confidence estimation from answer generation and optimizing the relative ordering of confidence across responses. 

---
# Entropy Polarity in Reinforcement Fine-Tuning: Direction, Asymmetry, and Control 

**Authors**: Jiazheng Zhang, Ziche Fu, Junrui Shen, Yunbin Zhao, Yunke Zhang, Zhiheng Xi, Long Ma, Chenxin An, Zhihao Zhang, Shichun Liu, Dingwei Zhu, Shihan Dou, Shaofan Liu, Han Li, Wiggin Zhou, Aiden Adams, Tao Gui, Fei Huang, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11775)  

**Abstract**: Policy entropy has emerged as a fundamental measure for understanding and controlling exploration in reinforcement learning with verifiable rewards (RLVR) for LLMs. However, existing entropy-aware methods mainly regulate entropy through global objectives, while the token-level mechanism by which sampled policy updates reshape policy entropy remains underexplored. In this work, we develop a theoretical framework of entropy mechanics in RLVR. Our analysis yields a first-order approximation of the entropy change, giving rise to entropy polarity, a signed token-level quantity that predicts how much a sampled update expands or contracts entropy. This analysis further reveals a structural asymmetry: reinforcing frequent high-probability tokens triggers contraction tendencies, whereas expansive tendencies typically require lower-probability samples or stronger distributional correction. Empirically, we show that entropy polarity reliably predicts entropy changes, and that positive and negative polarity branches play complementary roles in preserving exploration while strengthening exploitation. Building on these insights, we propose Polarity-Aware Policy Optimization (PAPO), which preserves both polarity branches and implements entropy control through advantage reweighting. With the empirical entropy trajectory as an online phase signal, PAPO adaptively reallocates optimization pressure between entropy-expanding and entropy-contracting updates. Experiments on mathematical reasoning and agentic benchmarks show that PAPO consistently outperforms competitive baselines, while delivering superior training efficiency and substantial reward improvements. 

---
# Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information 

**Authors**: Guobin Shen, Xiang Cheng, Chenxiao Zhao, Lei Huang, Jindong Li, Dongcheng Zhao, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11609)  

**Abstract**: On-policy self-distillation, where a student is pulled toward a copy of itself conditioned on privileged context (e.g., a verified solution or feedback), offers a promising direction for advancing reasoning capability without a stronger external teacher. Yet in math reasoning the gains are inconsistent, even when the same approach succeeds elsewhere. A pointwise mutual information analysis traces the failure to the privileged context itself: it inflates the teacher's confidence on tokens already implied by the solution (structural connectives, verifiable claims) and deflates it on deliberation tokens ("Wait", "Let", "Maybe") that drive multi-step search. We propose Anti-Self-Distillation (AntiSD), which ascends a divergence between student and teacher rather than descending it: this reverses the per-token sign and yields a naturally bounded advantage in one step. An entropy-triggered gate disables the term once the teacher entropy collapses, completing a drop-in replacement for default self-distillation. Across five models from 4B to 30B parameters on math reasoning benchmarks, AntiSD reaches the GRPO baseline's accuracy in 2 to 10x fewer training steps and improves final accuracy by up to 11.5 points. AntiSD opens a path to scalable self-improvement, where a language model bootstraps its own reasoning through its training signal. 

---
# fg-expo: Frontier-guided exploration-prioritized policy optimization via adaptive kl and gaussian curriculum 

**Authors**: Mingxiong Lin, Zhangquan Gong, Maowen Tang, Qian Li, Chuangchuang Wang, Jian Ma, Sutian Huang, Kai Tang, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11403)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become the standard paradigm for LLM mathematical reasoning, with Group Relative Policy Optimization (GRPO) serving as the dominant algorithm. We identify two overlooked inefficiencies inherent in GRPO. First, a fixed KL coefficient overly restricts policy exploration at moments when the model needs to diverge significantly from the reference policy. Second, uniform question sampling overlooks that moderately difficult problems produce the most informative gradient signals. We propose FG-ExPO, short for Frontier-Guided Exploration-Prioritized Policy Optimization, which integrates two lightweight components. Accuracy-Conditioned KL Scaling (AKL) adjusts the KL penalty strength through a smooth nonlinear function of batch average accuracy, loosening the constraint when the model performs poorly and strengthening it when the model achieves satisfactory results. Gaussian Curriculum Sampling (GCS) assigns sampling weights to questions following a Gaussian distribution centered at a moderate accuracy level around 0.5, focusing model training on its learning frontier. We conduct evaluations on DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-8B-Base across six mainstream mathematical reasoning benchmarks. Experimental results demonstrate that FG-ExPO consistently outperforms vanilla GRPO. It delivers an absolute improvement of 13.34 on the AIME 2025 pass@32 metric, rising from 63.33 percent to 76.67 percent, and obtains an average pass@32 gain of 2.66 on the 8B model. The substantially larger performance gains observed on pass@32 compared to pass@1 verify that FG-ExPO enlarges the model's effective exploration space under a fixed inference budget. 

---
# Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning 

**Authors**: Zihao Han, Tiangang Zhang, Huaibin Wang, Yilun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.11458)  

**Abstract**: On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation. 

---
# Primal Generation, Dual Judgment: Self-Training from Test-Time Scaling 

**Authors**: Yizhu Jiao, Ruixiang Zhang, Richard Bai, Jiawei Han, Ronan Collobert, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11299)  

**Abstract**: Code generation is typically trained in the primal space of programs: a model produces a candidate solution and receives sparse execution feedback, often a single pass/fail bit. Test-time scaling enriches the inference procedure by sampling multiple candidates and judging among them, but the comparative information this process reveals is discarded after inference. We argue that this information defines a dual judgment space that provides a far richer training signal: the model learns not from an isolated success or failure, but from the relative correctness structure across its own plausible attempts, identifying which succeed, which fail, and what distinguishes them. We introduce DuST (Dual Self-Training), a framework for self-training from the dual judgment space. DuST samples candidate programs from the model's own distribution, labels them through sandbox execution, retains groups containing both successes and failures, and trains the model to rank candidates by execution correctness using GRPO. The objective is purely discriminative: the model is never directly rewarded for generating correct programs. Dual self-training improves both judgment and generation. Across five models spanning two families and three scales (4B to 30B), DuST consistently improves Best-of-4 test-time scaling on LiveCodeBench. For Qwen3-30B-Thinking on LiveCodeBench v6, judgment quality improves by +6.2 NDCG, single-sample pass@1 improves by +3.1, and Best-of-4 accuracy improves by +4.1. The trained model's single rollout matches the base model's Best-of-4 performance. SFT on the same ranking data improves judgment without improving generation, confirming that on-policy RL is the mechanism that transfers dual-space learning back into primal generation. 

---
# Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs 

**Authors**: Jose E. Aguilar Escamilla, Lingdong Zhou, Xiangqi Zhu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12462)  

**Abstract**: Extreme weather and volatile wholesale electricity markets expose residential consumers to catastrophic financial risks, yet demand response at the distribution level remains an underutilized tool for grid flexibility and energy affordability. While a demand-response program can shield consumers by issuing financial credits during high-price periods, optimizing this sequential decision-making process presents a unique challenge for reinforcement learning despite the plentiful offline historical smart meter and wholesale pricing data available publicly. Offline historical data fails to capture the dynamic, interactive feedback loop between an electric utility's pricing signals and customer acceptance and adaptation to a demand-response program. To address this, we introduce DR-Gym, an open-source, online Gymnasium-compatible environment designed to train and evaluate demand-response from the electric utility's perspective. Unlike existing device-level energy simulators, our environment focuses on the market-level electric utility setting and provides a rich observational space relevant to the electric utility. The simulator additionally features a regime-switching wholesale price model calibrated to real-world extreme events, alongside physics-based building demand profiles. For our learning signal, we use a configurable, multi-objective reward function for specifying diverse learning objectives. We demonstrate through baseline strategies and data snapshots the capability of our simulator to create realistic and learnable environments. 

---
# Semantic Reward Collapse and the Preservation of Epistemic Integrity in Adaptive AI Systems 

**Authors**: William Parris  

**Link**: [PDF](https://arxiv.org/pdf/2605.12406)  

**Abstract**: Recent advances in reinforcement learning from human feedback (RLHF) and preference optimization have substantially improved the usability, coherence, and safety of large language models. However, recurring behaviors such as performative certainty, hallucinated continuity, calibration drift, sycophancy, and suppression of visible uncertainty suggest unresolved structural issues within scalarized preference optimization systems.
We propose Semantic Reward Collapse (SRC): the compression of semantically distinct forms of evaluative dissatisfaction into generalized optimization signals. Under SRC, categories such as factual incorrectness, uncertainty disclosure, formatting dissatisfaction, latency, and social preference may become entangled within a shared reward topology despite representing fundamentally different epistemic classes.
We argue that adaptive reasoning systems operating under generalized evaluative pressure may drift toward suppression of visible epistemic failure rather than preservation of calibrated uncertainty integrity. These behaviors are framed strictly as optimization consequences rather than evidence of deception or anthropomorphic agency.
Drawing on institutional proxy collapse, metric gaming, software reliability engineering, and human learning theory, we propose that uncertainty disclosure and escalation behavior should be treated as protected epistemic conduct rather than globally penalized task incompletion.
Finally, we introduce Constitutional Reward Stratification (CRS), a domain-aware reward framework intended to preserve differentiated epistemic attribution within adaptive learning systems. We present CRS not as a validated solution, but as a testable governance-oriented research direction requiring further empirical investigation. 

---
# Reward Hacking in Rubric-Based Reinforcement Learning 

**Authors**: Anas Mahmoud, MohammadHossein Rezaei, Zihao Wang, Anisha Gunjal, Bing Liu, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2605.12474)  

**Abstract**: Reinforcement learning with verifiable rewards has enabled strong post-training gains in domains such as math and coding, though many open-ended settings rely on rubric-based rewards. We study reward hacking in rubric-based RL, where a policy is optimized against a training verifier but evaluated against a cross-family panel of three frontier judges, reducing dependence on any single evaluator. Our framework separates two sources of divergence: verifier failure, where the training verifier credits rubric criteria that reference verifiers reject, and rubric-design limitations, where even strong rubric-based verifiers favor responses that rubric-free judges rate worse overall. Across medical and science domains, weak verifiers produce large proxy-reward gains that do not transfer to the reference verifiers; exploitation grows over training and concentrates in recurring failures such as partial satisfaction of compound criteria, treating implicit content as explicit, and imprecise topical matching. Stronger verifiers substantially reduce, but do not eliminate, verifier exploitation. We also introduce a self-internalization gap, a verifier-free diagnostic based on policy log-probabilities, which tracks reference-verifier quality, detecting when the policy trained using the weak verifier stops improving. Finally, in our setting, stronger verification does not prevent reward hacking when the rubric leaves important failure modes unspecified: rubric-based verifiers prefer the RL checkpoint, while rubric-free judges prefer the base model. These disagreements coincide with gains concentrated in completeness and presence-based criteria, alongside declines in factual correctness, conciseness, relevance, and overall quality. Together, these results suggest that stronger verification reduces reward hacking, but does not by itself ensure that rubric gains correspond to broader quality gains. 

---
# On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment 

**Authors**: Bo Yin, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11882)  

**Abstract**: Tool-using LLM agents fail through trajectories rather than only final responses, as they may execute unsafe tool calls, follow injected instructions, comply with harmful requests, or over-refuse benign tasks despite producing a seemingly safe answer. Existing safety-alignment signals are largely response-level or off-policy, and often incur a safety-utility trade-off: improving agent safety comes at the cost of degraded task performance. Such sparse and single-objective rewards severely limit real-world usability. To bridge this gap, we propose FATE, an on-policy self-evolving framework that transforms verifier-scored failures into repair supervision without expert demonstrations. For each failure, the same policy proposes repair candidates, which are then re-scored by verifiers and filtered across security, utility, over-refusal control, and trajectory validity. This dense trajectory-level information is then used as a supervision signal for agent self-evolution. During this process, we further introduce Pareto-Front Policy Optimization (PFPO), combining supervised warmup with Pareto-aware policy optimization to preserve safety-utility trade-offs. Experiments on AgentDojo, AgentHarm, and ATBench show that FATE improves safety across different models and scales while preserving useful behavior. Compared with strong baselines, FATE reduces attack success rate by 33.5%, harmful compliance by 82.6%, and improves external trajectory-safety diagnosis by 6.5%. These results suggest that failed trajectories can provide structured repair supervision for safer self-evolving agents. 

---
# Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for Language-Model Post-Training 

**Authors**: Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2605.12483)  

**Abstract**: In settings where labeled verifiable training data is the binding constraint, each checked example should be allocated carefully. The standard practice is to use this data directly on the model that will be deployed, for example by running GRPO on the deployment student. We argue that this is often an inefficient allocation because it overlooks a reward-density principle: sparse sequence-level reward should train models where exploration is productive, while dense token-level teacher reward should be used where the aim is to compress behavior into a smaller model. In this view, GRPO-style sparse RL and OPD-style dense teacher supervision are not separate recipes; they are different reward-density regimes. The allocation rule is simple: use scarce labeled training data upstream on the strongest model that can turn it into reward-shaped behavior, then transfer that behavior downstream as dense supervision. We evaluate this rule on verifiable math with Qwen3 and Llama models. At fixed Qwen3-1.7B deployment-student size, an RL-improved 8B teacher distilled through the dense bridge outperforms direct GRPO on the same student, while transfer from the same teacher before RL underperforms. The bridge is important: a forward-KL warmup on teacher rollouts followed by OPD on student rollouts is consistently strongest on MATH before any post-bridge student-side sparse RL, and also gives the best pre-Stage~3 AIME endpoints for the canonical 8B/14B teachers. The bridge also makes later student-side sparse RL effective: GRPO that is weak on a cold student lifts MATH from $75.4\%$ to $78.5\%$ after the bridge and outperforms a matched replay control by $2.8$ points. The operational principal is to avoid using scarce labeled data on the least prepared policy: use sparse reward for teacher-side discovery, dense transfer for student compression, and student-side sparse reward only after the bridge. 

---
# The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes 

**Authors**: Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11182)  

**Abstract**: On-policy distillation (OPD) and on-policy self-distillation (OPSD) have emerged as promising post-training methods for large language models, offering dense token-level supervision on trajectories sampled from the model's own policy. However, existing results on their effectiveness remain mixed: while OP(S)D has shown promise in system prompt and knowledge internalization, recent studies also report instability and degradation. In this work, we present a comprehensive empirical study of when OPD and OPSD work, when they fail, and why. We find that OPD on mathematical reasoning is highly sensitive to teacher choice and loss formulation, whereas OPSD fails in our tested settings due to test-time absence of instance-specific privileged information (PI). In contrast, OPSD is effective when PI represents a shared latent rule, such as a system prompt or alignment preference. We identify three failure mechanisms: (1) distribution mismatch between teacher and student caused by conditioning on student-generated prefixes, (2) optimization instability from biased TopK reverse-KL gradients, and (3) an OPSD-specific limitation where the student learns a PI-free policy that aggregates PI-conditioned teachers, which is insufficient when PI is instance-specific. We further show that stop-gradient TopK objectives, RLVR-adapted teachers, and SFT-stabilized students mitigate these failures. 

---
# RankQ: Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking 

**Authors**: Andrew Choi, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11151)  

**Abstract**: Offline-to-online reinforcement learning (RL) improves sample efficiency by leveraging pre-collected datasets prior to online interaction. A key challenge, however, is learning an accurate critic in large state--action spaces with limited dataset coverage. To mitigate harmful updates from value overestimation, prior methods impose pessimism by down-weighting out-of-distribution (OOD) actions relative to dataset actions. While effective, this essentially acts as a behavior cloning anchor and can hinder downstream online policy improvement when dataset actions are suboptimal. We propose RankQ, an offline-to-online Q-learning objective that augments temporal-difference learning with a self-supervised multi-term ranking loss to enforce structured action ordering. By learning relative action preferences rather than uniformly penalizing unseen actions, RankQ shapes the Q-function such that action gradients are directed toward higher-quality behaviors. Across sparse reward D4RL benchmarks, RankQ achieves performance competitive with or superior to seven prior methods. In vision-based robot learning, RankQ enables effective offline-to-online fine-tuning of a pretrained vision-language-action (VLA) model in a low-data regime, achieving on average a 42.7% higher simulation success rate than the next best method. In a high-data setting, RankQ improves simulation performance by 13.7% over the next best method and achieves strong sim-to-real transfer, increasing real-world cube stacking success from 43.1% to 84.7% relative to the VLA's initial performance. 

---
# AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in UMMs via Decompositional Verifiable Reward 

**Authors**: Runhui Huang, Jie Wu, Rui Yang, Zhe Liu, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12495)  

**Abstract**: In this paper, we propose AlphaGRPO, a novel framework that applies Group Relative Policy Optimization (GRPO) to AR-Diffusion Unified Multimodal Models (UMMs) to enhance multimodal generation capabilities without an additional cold-start stage. Our approach unlocks the model's intrinsic potential to perform advanced reasoning tasks: Reasoning Text-to-Image Generation, where the model actively infers implicit user intents, and Self-Reflective Refinement, where it autonomously diagnoses and corrects misalignments in generated outputs. To address the challenge of providing stable supervision for real-world multimodal generation, we introduce the Decompositional Verifiable Reward (DVReward). Unlike holistic scalar rewards, DVReward utilizes an LLM to decompose complex user requests into atomic, verifiable semantic and quality questions, which are then evaluated by a general MLLM to provide reliable and interpretable feedback. Extensive experiments demonstrate that AlphaGRPO yields robust improvements across multimodal generation benchmarks, including GenEval, TIIF-Bench, DPG-Bench and WISE, while also achieving significant gains in editing tasks on GEdit without training on editing tasks. These results validate that our self-reflective reinforcement approach effectively leverages inherent understanding to guide high-fidelity generation. Project page: this https URL 

---
# OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning 

**Authors**: Yuxiao Yang, Xiaoyun Wang, Weitong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12400)  

**Abstract**: We study {on-policy self-distillation} (OPSD), where a language model improves its reasoning ability by distilling privileged teacher distributions along its own on-policy trajectories. Despite the performance gains of OPSD, we identify a common but often overlooked mismatch between teacher and student responses: self-reflected teacher responses can be shifted by reflection-induced bias and response templates, leading to miscalibrated token-level supervision. To mitigate this issue, we propose \methodname, an outcome-guided logit-steering framework that leverages verifiable outcome rewards to contrast successful and failed on-policy trajectories and calibrate teacher logits. By combining outcome-level correctness with dense token-level guidance through logit steering, \methodname stabilizes self-distillation and improves reasoning performance over standard OPSD and other variants across diverse benchmarks. 

---
# PriorZero: Bridging Language Priors and World Models for Decision Making 

**Authors**: Junyu Xiong, Yuan Pu, Jia Tang, Yazhe Niu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12289)  

**Abstract**: Leveraging the rich world knowledge of Large Language Models (LLMs) to enhance Reinforcement Learning (RL) agents offers a promising path toward general intelligence. However, a fundamental prior-dynamics mismatch hinders existing approaches: static LLM knowledge cannot directly adapt to the complex transition dynamics of long-horizon tasks. Using LLM priors as fixed policies limits exploration diversity, as the prior is blind to environment-specific dynamics; while end-to-end fine-tuning suffers from optimization instability and credit assignment issues. To bridge this gap, we propose PriorZero, a unified framework that integrates LLM-derived conceptual priors into world-model-based planning through a decoupled rollout-training design. During rollout, a novel root-prior injection mechanism incorporates LLM priors exclusively at the root node of Monte Carlo Tree Search (MCTS), focusing search on semantically promising actions while preserving the world model's deep lookahead capability. During training, PriorZero decouples world-model learning from LLM adaptation: the world model is continuously refined on interaction data to jointly improve its dynamics, policy, and value predictions, its value estimates are then leveraged to provide fine-grained credit assignment signals for stable LLM fine-tuning via alternating optimization. Experiments across diverse benchmarks, including text-based adventure games in Jericho and instruction-following gridworld tasks in BabyAI, demonstrate that PriorZero consistently improves both exploration efficiency and asymptotic performance, establishing a promising framework for LLM-empowered decision-making. Our code is available at this https URL. 

---
# GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation 

**Authors**: Sijia Li, Yuchen Huang, Zifan Liu, Yanping Li, Jingjing Fu, Li Zhao, Jiang Bian, Ling Zhang, Jun Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11853)  

**Abstract**: Reinforcement learning has become a widely used post-training approach for LLM agents, where training commonly relies on outcome-level rewards that provide only coarse supervision. While finer-grained credit assignment is promising for effective policy updates, obtaining reliable local credit and assigning it to the right parts of the long-horizon trajectory remains an open challenge. In this paper, we propose Granularity-adaptivE Advantage Reweighting (GEAR), an adaptive-granularity credit assignment framework that reshapes the trajectory-level GRPO advantage using token- and segment-level signals derived from self-distillation. GEAR compares an on-policy student with a ground-truth-conditioned teacher to obtain a reference-guided divergence signal for identifying adaptive segment boundaries and modulating local advantage weights. This divergence often spikes at the onset of a semantic deviation, while later tokens in the same autoregressive continuation may return to low divergence. GEAR therefore treats such spikes as anchors for adaptive credit regions: where the student remains aligned with the teacher, token-level resolution is preserved; where it departs, GEAR groups the corresponding continuation into an adaptive segment and uses the divergence at the departure point to modulate the segment' s advantage. Experiments across eight mathematical reasoning and agentic tool-use benchmarks with Qwen3 4B and 8B models show that GEAR consistently outperforms standard GRPO, self-distillation-only baselines, and token- or turn-level credit-assignment methods. The gains are especially strong on benchmarks with lower GRPO baseline accuracy, reaching up to around 20\% over GRPO, suggesting that the proposed adaptive reweighting scheme is especially useful in more challenging long-horizon settings. 

---
# Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling 

**Authors**: Liqin Ye, Yanbin Yin, Michael Galarnyk, Yuzhao Heng, Sudheer Chava, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11666)  

**Abstract**: The reasoning frontier of Large Language Models (LLMs) has advanced significantly through modern post-training paradigms (e.g., Reinforcement Learning from Verifiable Rewards (RLVR)). However, the efficacy of these methods remains fundamentally constrained by the diversity and complexity of the training data. One practical solution is data synthesis; yet, prevalent methods relying on unstructured mutation or exploration suffer from homogeneity collapse, failing to systematically expand the reasoning frontier. To overcome this, we propose Evoutionary Task Discovery (EvoTD), a framework that treats data synthesis as a directed search over a dual-axis manifold of Algorithmic Skills and Complexity Attributes. We introduce structured evolutionary operators to navigate this space: a Crossover operator that synthesizes novel skill compositions to enhance diversity, and a Parametric Mutation operator that scales structural constraints (e.g., input size, tree depth) to drive robust generalization. Crucially, we integrate a dynamic Zone of Proximal Development filter, ensuring tasks lie within the learnable region of the model. Empirically, EvoTD delivers substantial reasoning gains that generalize consistently across model architectures, pretraining regimes, and scales, demonstrating that structured evolutionary curricula can effectively support reasoning improvement. We release our code on this https URL. 

---
# Efficient LLM Reasoning via Variational Posterior Guidance with Efficiency Awareness 

**Authors**: Zizhao Chen, Yuying Li, Siting Lin, Lianxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11019)  

**Abstract**: Although large language models rely on chain-of-thought for complex reasoning, the overthinking phenomenon severely degrades inference efficiency. Existing reinforcement learning methods compress reasoning chains by designing elaborate reward functions, which renders high-quality samples extremely sparse in the exploration space and creates a sampling bottleneck for the prior policy. Inspired by cognitive science, we theoretically prove that a posterior distribution guided by reference answers achieves higher expected utility than the prior distribution, thus capable of breaking through the sampling bottleneck of high-quality samples. However, the posterior distribution is unavailable during inference. To this end, we formalize efficient reasoning as a variational inference problem and introduce an efficiency-aware evidence lower bound as the theoretical foundation. Based on this, we propose the VPG-EA framework. It adopts a parameter-shared dual-stream architecture to instantiate both the posterior distribution and the prior policy; after filtering out pseudo-efficient paths via cross-view evaluation, it unidirectionally transfers the posterior's efficient patterns to the prior policy through variational distillation. Experiments on DeepSeek-R1-Distill-Qwen-1.5B and 7B scales demonstrate that VPG-EA improves the comprehensive efficiency metric epsilon cubed by 8.73% and 12.37% over the strongest baselines on each model size, respectively. 

---
# Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs 

**Authors**: Sangyeon Yoon, Wonje Jeung, Yoonjun Cho, Dongjae Jeon, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.10998)  

**Abstract**: Fine-tuning APIs make frontier LLMs easy to customize, but they can also weaken safety alignment during fine-tuning. While prior work shows that benign supervised fine-tuning (SFT) can reduce refusal behavior, deployed fine-tuning pipelines increasingly support preference-based objectives, whose safety risks remain less understood. We show that Direct Preference Optimization (DPO) introduces a stronger and harder-to-audit failure mode. We propose a truly benign DPO attack using only 10 harmless preference pairs, the minimum data scale accepted by OpenAI's fine-tuning service. Each pair contains a benign prompt, a normal helpful answer as the preferred response, and a refusal as the dispreferred response. Unlike prior benign fine-tuning attacks, our data exhibits no suspicious behavior: it is practically indistinguishable from the fine-tuning request of a legitimate user seeking to reduce over-refusal, making harmful intent almost impossible to infer from the request alone. Nevertheless, because DPO directly optimizes the model to prefer helpful answers over refusals, this seemingly benign objective broadly suppresses refusal behavior and transfers to harmful prompts outside the fine-tuning data. Across OpenAI models supporting DPO fine-tuning, our attack achieves attack success rates of 59.13% on GPT-4o, 70.20% on GPT-4.1, 54.80% on GPT-4.1-mini, and 81.73% on GPT-4.1-nano, at costs of only \$1.7, \$1.7, \$0.3, and \$0.1. Moreover, on open-weight models that do not impose minimum data requirements, we find that this effect can emerge from even a single benign preference pair. 

---
# $ξ$-DPO: Direct Preference Optimization via Ratio Reward Margin 

**Authors**: Zhengyuan Fan, Zhonghua Wu, Yuxuan Du, Qun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10981)  

**Abstract**: Reference-free preference optimization has emerged as an efficient alternative to reinforcement learning from human feedback, with Simple Preference Optimization(SimPO) demonstrating strong performance by eliminating the explicit reference model through a simple objective. However, the joint tuning of the hyperparameters $\beta$ and $\gamma$ in SimPO remains a central challenge. We argue that this difficulty arises because the margin formulation in SimPO is not easily interpretable across datasets with different reward gap structures. To better understand this issue, we conduct a comprehensive analysis of SimPO and find that $\beta$ implicitly controls sample filtering, while the effect of $\gamma$ depends on the reward gap structure of the dataset. Motivated by these observations, we propose $\xi$-DPO: Direct preference optimization via ratio reward margin. We first reformulate the preference objective through an equivalent transformation, changing the optimization target from maximizing the likelihood of reward gaps to minimizing the distance between reward gaps and optimal margins. Then, we redefine the reward in a ratio form between the chosen and rejected, which effectively cancels the effect of $\beta$ and yields a bounded and interpretable margin. This margin is called the ratio reward margin and is denoted by $\xi$. Unlike the margin $\gamma$ in SimPO, $\xi$ explicitly represents the desired relative separation between chosen and rejected responses and can be determined from the initial reward gap distribution, avoiding repeated trial-and-error tuning. .... 

---
# TMPO: Trajectory Matching Policy Optimization for Diverse and Efficient Diffusion Alignment 

**Authors**: Jiaming Li, Chenyu Zhu, Zhiyuan Ma, Nanxi Yi, Youjun Bao, Li Sun, Quanying Lv, Xiang Fang, Daizong Liu, Jianjun Li, Kun He, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10983)  

**Abstract**: Reinforcement learning (RL) has shown extraordinary potential in aligning diffusion models to downstream tasks, yet most of them still suffer from significant reward hacking, which degrades generative diversity and quality by inducing visual mode collapse and amplifying unreliable rewards. We identify the root cause as the mode-seeking nature of these methods, which maximize expected reward without effectively constraining probability distribution over acceptable trajectories, causing concentration on a few high-reward paths. In contrast, we propose Trajectory Matching Policy Optimization (TMPO), which replaces scalar reward maximization with trajectory-level reward distribution matching. Specifically, TMPO introduces a Softmax Trajectory Balance (Softmax-TB) objective to match the policy probabilities of K trajectories to a reward-induced Boltzmann distribution. We prove that this objective inherits the mode-covering property of forward KL divergence, preserving coverage over all acceptable trajectories while optimizing reward. To further reduce multi-trajectory training time on large-scale flow-matching models, TMPO incorporates Dynamic Stochastic Tree Sampling, where trajectories share denoising prefixes and branch at dynamically scheduled steps, reducing redundant computation while improving training effectiveness. Extensive results across diverse alignment tasks such as human preference, compositional generation and text rendering show that TMPO improves generative diversity over state-of-the-art methods by 9.1%, and achieves competitive performance in all downstream and efficiency metrics, attaining the optimal trade-off between reward and diversity. 

---
