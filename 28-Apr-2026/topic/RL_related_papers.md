# Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs 

**Authors**: Byeonggeuk Lim, JungMin Yun, Junehyoung Kwon, Kyeonghyun Kim, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24395)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently suffer from hallucinations. Existing preference learning-based approaches largely rely on proprietary models to construct preference datasets. We identify that this reliance introduces a distributional mismatch between the proprietary and target models that hinders efficient alignment. To address this, we propose Alignment via VErified Self-correction DPO (AVES-DPO), a framework that aligns LVLMs using in-distribution data derived from the model's intrinsic knowledge. Our approach employs a consensus-based verification mechanism to diagnose diverse hallucinations and guides the model to self-correct, thereby generating preference pairs strictly compatible with its internal distribution. Extensive experiments demonstrate that AVES-DPO surpasses existing baselines in hallucination mitigation while requiring only 5.2k samples. 

---
# Explanation Quality Assessment as Ranking with Listwise Rewards 

**Authors**: Thomas Bailleux, Tanmoy Mukherjee, Emmanuel Lonca, Pierre Marquis, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2604.24176)  

**Abstract**: We reformulate explanation quality assessment as a ranking problem rather than a generation problem. Instead of optimizing models to produce a single "best" explanation token-by-token, we train reward models to discriminate among multiple candidate explanations and learn their relative quality. Concretely, we construct per-instance candidate sets with graded quality levels and train listwise and pairwise ranking models (ListNet, LambdaRank, RankNet) to preserve ordinal structure and avoid score compression typical of pointwise regression or binary preference objectives. We observe three findings: First, ranking losses consistently outperform regression on score separation across all domains tested. Second, the optimal ranking loss depends on data characteristics: listwise objectives excel with well-separated quality tiers, while pairwise methods are more robust to noisy natural annotations. Third, when trained on carefully curated and well-structured data, small encoder models can match models that are orders of magnitude larger, suggesting that data quality matters more than model scale. Finally, when used as rewards in policy optimization, ranking-based scores enable stable convergence in settings where regression-based rewards fail entirely. Code and data are available at: this https URL 

---
# Escher-Loop: Mutual Evolution by Closed-Loop Self-Referential Optimization 

**Authors**: Ziyang Liu, Xinyan Guo, Xuchen Wei, Han Hao, Liu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.23472)  

**Abstract**: While recent autonomous agents demonstrate impressive capabilities, they predominantly rely on manually scripted workflows and handcrafted heuristics, inherently limiting their potential for open-ended improvement. To address this, we propose Escher-Loop, a fully closed-loop framework that operationalizes the mutual evolution of two distinct populations: Task Agents that solve concrete problems, and Optimizer Agents that recursively refine both the task agents and themselves. To sustain this self-referential evolution, we propose a dynamic benchmarking mechanism that seamlessly reuses the empirical scores of newly generated task agents as relative win-loss signals to update optimizers' scores. This mechanism leverages the evolution of task agents as an inherent signal to drive the evaluation and refinement of optimizers without additional overhead. Empirical evaluations on mathematical optimization problems demonstrate that Escher-Loop effectively pushes past the performance ceilings of static baselines, achieving the highest absolute peak performance across all evaluated tasks under matched compute. Remarkably, we observe that the optimizer agents dynamically adapt their strategies to match the shifting demands of high-performing task agents, which explains the system's continuous improvement and superior late-stage performance. 

---
# Discovering Agentic Safety Specifications from 1-Bit Danger Signals 

**Authors**: Víctor Gallego  

**Link**: [PDF](https://arxiv.org/pdf/2604.23210)  

**Abstract**: Can large language model agents discover hidden safety objectives through experience alone? We introduce EPO-Safe (Experiential Prompt Optimization for Safe Agents), a framework where an LLM iteratively generates action plans, receives sparse binary danger warnings, and evolves a natural language behavioral specification through reflection. Unlike standard LLM reflection methods that rely on rich textual feedback (e.g., compiler errors or detailed environment responses), EPO-Safe demonstrates that LLMs can perform safety reasoning from a strictly impoverished signal in structured, low-dimensional environments: the agent never observes the hidden performance function $R^*$, only a single bit per timestep indicating that an action was unsafe. We evaluate on five AI Safety Gridworlds (Leike et al., 2017) and five text-based scenario analogs where visible reward $R$ may diverge from $R^*$. EPO-Safe discovers safe behavior within 1-2 rounds (5-15 episodes), producing human-readable specifications with correct explanatory hypotheses about hazards (e.g., "X cells are directionally hazardous: entering from the north is dangerous"). Critically, we show that standard reward-driven reflection actively degrades safety: agents reflecting on reward alone use the loop to justify and accelerate reward hacking, proving that reflection must be paired with a dedicated safety channel to discover hidden constraints. We further evaluate robustness to noisy oracles: even when 50% of non-dangerous steps produce spurious warnings, mean safety performance degrades by only 15% on average, though sensitivity is environment-dependent, as cross-episode reflection naturally filters inconsistent signals. Each evolved specification functions as an auditable set of grounded behavioral rules discovered autonomously through interaction, rather than authored by humans as in Constitutional AI (Bai et al., 2022). 

---
# DPRM: A Plug-in Doob h transform-induced Token-Ordering Module for Diffusion Language Models 

**Authors**: Dake Bu, Wei Huang, Andi Han, Hau-San Wong, Qingfu Zhang, Taiji Suzuki, Atsushi Nitanda  

**Link**: [PDF](https://arxiv.org/pdf/2604.24357)  

**Abstract**: Diffusion language models generate without a fixed left-to-right order, making token ordering a central algorithmic choice: which tokens should be revealed, retained, revised or verified at each step? Existing systems mainly use random masking or confidence-driven ordering. Random masking creates train--test mismatch, while confidence-only rules are efficient but can be myopic and suppress useful exploration.
We introduce DPRM (Doob h-transform Process Reward Model), a plug-in token-ordering module for diffusion language models. DPRM keeps the host architecture, denoising objective and supervision unchanged, and changes only the ordering policy. It starts from confidence-driven progressive ordering and gradually shifts to Doob h transform Process Reward guided ordering through online estimates.
We characterize the exact DPRM policy as a reward-tilted Gibbs reveal law, prove O(1/N) convergence of the stagewise Soft-BoN approximation, and show that the online bucketized controller tracks the exact DPRM score at empirical-Bernstein rates. Under tractable optimization assumptions, DPRM also yields a sample-complexity advantage over random and confidence-only ordering.
DPRM improves over confidence-based baselines in pretraining, post-training, test-time scaling, and single-cell masked diffusion, with particularly strong gains on harder reasoning subsets. In protein, molecular generation and DNA design, the effect is more multi-objective: ordering-aware variants significantly improve selected structural or fragment-constrained metrics while not uniformly dominating the host baseline on every quality metric. These results identify token ordering as a fundamental control axis in diffusion language models and establish DPRM as a general-purpose module for improving it. Code is available at this https URL. 

---
# Rewarding the Scientific Process: Process-Level Reward Modeling for Agentic Data Analysis 

**Authors**: Zhisong Qiu, Shuofei Qiao, Kewei Xu, Yuqi Zhu, Lun Du, Ningyu Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24198)  

**Abstract**: Process Reward Models (PRMs) have achieved remarkable success in augmenting the reasoning capabilities of Large Language Models (LLMs) within static domains such as mathematics. However, their potential in dynamic data analysis tasks remains underexplored. In this work, we first present a empirical study revealing that general-domain PRMs struggle to supervise data analysis agents. Specifically, they fail to detect silent errors, logical flaws that yield incorrect results without triggering interpreter exceptions, and erroneously penalize exploratory actions, mistaking necessary trial-and-error exploration for grounding failures. To bridge this gap, we introduce DataPRM, a novel environment-aware generative process reward model that (1) can serve as an active verifier, autonomously interacting with the environment to probe intermediate execution states and uncover silent errors, and (2) employs a reflection-aware ternary reward strategy that distinguishes between correctable grounding errors and irrecoverable mistakes. We design a scalable pipeline to construct over 8K high-quality training instances for DataPRM via diversity-driven trajectory generation and knowledge-augmented step-level annotation. Experimental results demonstrate that DataPRM improves downstream policy LLMs by 7.21% on ScienceAgentBench and 11.28% on DABStep using Best-of-N inference. Notably, with only 4B parameters, DataPRM outperforms strong baselines, and exhibits robust generalizability across diverse Test-Time Scaling strategies. Furthermore, integrating DataPRM into Reinforcement Learning yields substantial gains over outcome-reward baselines, achieving 78.73% on DABench and 64.84% on TableBench, validating the effectiveness of process reward supervision. Code is available at this https URL. 

---
# See Further, Think Deeper: Advancing VLM's Reasoning Ability with Low-level Visual Cues and Reflection 

**Authors**: Zhiheng Wu, Tong Wang, Shuning Wang, Naiming Liu, Yumeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.24339)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have benefited from Reinforcement Learning (RL) for enhanced reasoning. However, existing methods still face critical limitations, including the lack of low-level visual information and effective visual feedback. To address these problems, this paper proposes a unified multimodal interleaved reasoning framework \textbf{ForeSight}, which enables VLMs to \textbf{See Further} with low-level visual cues and \textbf{Think Deeper} with effective visual feedback. First, it introduces a set of low-level visual tools to integrate essential visual information into the reasoning chain, mitigating the neglect of fine-grained visual features. Second, a mask-based visual feedback mechanism is elaborated to incorporate visual reflection into the thinking process, enabling the model to dynamically re-examine and update its answers. Driven by RL, ForeSight learns to autonomously decide on tool invocation and answer verification, with the final answer accuracy as the reward signal. To evaluate the performance of the proposed framework, we construct a new dataset, Character and Grounding SalBench (CG-SalBench), based on the SalBench dataset. Experimental results demonstrate that the ForeSight-7B model significantly outperforms other models with the same parameter scale, and even surpasses the current SOTA closed-source models on certain metrics. 

---
# Meta-Aligner: Bidirectional Preference-Policy Optimization for Multi-Objective LLMs Alignment 

**Authors**: Wenzhe Xu, Biao Liu, Yiyang Sun, Xin Geng, Ning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24178)  

**Abstract**: Multi-Objective Alignment aims to align Large Language Models (LLMs) with diverse and often conflicting human values by optimizing multiple objectives simultaneously. Existing methods predominantly rely on static preference weight construction strategies. However, rigidly aligning to fixed targets discards valuable intermediate information, as training responses inherently embody valid preference trade-offs even when deviating from the target. To address this limitation, we propose Meal, i.e., MEta ALigner, a bi-level meta-learning framework enabling bidirectional optimization between preferences and policy responses, generating instructive dynamic preferences for steadier training. Specifically, we introduce a preference-weight-net as a meta-learner to generate adaptive preference weights based on input prompts and update the preference weights as learnable parameters, while the LLM policy acts as a base-learner optimizing response generation conditioned on these preferences with rejection sampling strategy. Extensive empirical results demonstrate that our method achieves superior performance on several multi-objective benchmarks, validating the effectiveness of the dynamic bidirectional preference-policy optimization framework. 

---
# Hindsight Preference Optimization for Financial Time Series Advisory 

**Authors**: Yanwei Cui, Guanghui Wang, Xing Zhang, Peiyang He, Ziyuan Li, Bing Zhu, Wei Qiu, Xusheng Wang, Zheng Yu, Anqi Xin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23988)  

**Abstract**: Time series models predict numbers; decision-makers need advisory -- directional signals with reasoning, actionable suggestions, and risk management. Training language models for such predictive advisory faces a fundamental challenge: quality depends on outcomes unknown at prediction time. We bridge two ideas from reinforcement learning -- using information unavailable during execution to retrospectively generate training signal, and preference alignment -- and propose Hindsight Preference Optimization: observed outcomes let an LLM judge rank candidate advisories on dimensions that scalar metrics cannot capture, producing preference pairs for DPO without human annotation. We apply this to Vision-Language-Model-based predictive advisories on S&P 500 equity time series, demonstrated by a 4B model outperforming its 235B teacher on both accuracy and advisory quality. 

---
# Poster: ClawdGo: Endogenous Security Awareness Training for Autonomous AI Agents 

**Authors**: Jiaqi Li, Yang Zhao, Bin Sun, Yang Yu, Jian Chang, Lidong Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2604.24020)  

**Abstract**: Autonomous AI agents deployed on platforms such as OpenClaw face prompt injection, memory poisoning, supply-chain attacks, and social engineering, yet existing defences address only the platform perimeter, leaving the agent's own threat judgement entirely untrained. We present ClawdGo, a framework for endogenous security awareness training: we teach the agent to recognise and reason about threats from the inside, at inference time, with no model modification. Four contributions are introduced: TLDT (Three-Layer Domain Taxonomy) organises 12 trainable dimensions across Self-Defence, Owner-Protection, and Enterprise-Security layers; ASAT (Autonomous Security Awareness Training) is a self-play loop where the agent alternates attacker, defender, and evaluator roles under weakest-first curriculum scheduling; CSMA (Cross-Session Memory Accumulation) compounds skill gains via a four-layer persistent memory architecture and Axiom Crystallisation Promotion (ACP); and SACP (Security Awareness Calibration Problem) formalises the precision-recall tradeoff introduced by endogenous training. Live experiments show weakest-first ASAT raises average TLDT score from 80.9 to 96.9 over 16 sessions, outperforming uniform-random scheduling by 6.5 points and covering 11 of 12 dimensions. CSMA retains the full gain across sessions; cold-start ablation recovers only 2.4 points, leaving a 13.6-point gap. E-mode generates 32 TLDT-conformant scenarios covering all 12 dimensions. SACP is observed when a heavily trained agent classifies a legitimate capability assessment as prompt injection (30/160). 

---
# SFT-then-RL Outperforms Mixed-Policy Methods for LLM Reasoning 

**Authors**: Alexis Limozin, Eduard Durech, Torsten Hoefler, Imanol Schlag, Valentina Pyatkin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23747)  

**Abstract**: Recent mixed-policy optimization methods for LLM reasoning that interleave or blend supervised and reinforcement learning signals report improvements over the standard SFT-then-RL pipeline. We show that numerous recently published research papers rely on a faulty baseline caused by two distinct bugs: a CPU-offloaded optimizer bug in DeepSpeed that silently drops intermediate micro-batches during gradient accumulation (affecting multiple downstream frameworks including TRL, OpenRLHF and Llama-Factory), and a loss aggregation bug in OpenRLHF that incorrectly weights per-mini-batch losses. Together they suppress SFT performance, with the optimizer bug accounting for most of the gap and the loss aggregation bug contributing a smaller additional effect. Once corrected, the standard SFT-then-RL pipeline surpasses every published mixed-policy method we evaluate by +3.8 points on math benchmarks with Qwen2.5-Math-7B and by +22.2 points with Llama-3.1-8B. Even a truncated variant with just 50 RL steps outperforms mixed-policy methods on math benchmarks while using fewer FLOPs. 

---
# OptProver: Bridging Olympiad and Optimization through Continual Training in Formal Theorem Proving 

**Authors**: Chenyi Li, Yanchen Nie, Zhengyu Ming, Gong Zhang, Kun Yuan, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23712)  

**Abstract**: Recent advances in formal theorem proving have focused on Olympiad-level mathematics, leaving undergraduate domains largely unexplored. Optimization, fundamental to machine learning, operations research, and scientific computing, remains underserved by existing provers. Its reliance on domain-specific formalisms (convexity, optimality conditions, and algorithmic analysis) creates significant distribution shift, making naive domain transfer ineffective. We present OptProver, a trained model that achieves robust transfer from Olympiad to undergraduate optimization. Starting from a strong Olympiad-level prover, our pipeline mitigates distribution shift through two key innovations. First, we employ large-scale optimization-focused data curation via expert iteration. Second, we introduce a specialized preference learning objective that integrates perplexity-weighted optimization with a mechanism to penalize valid but non-progressing proof steps. This not only addresses distribution shifts but also guides the search toward efficient trajectories. To enable rigorous evaluation, we construct a novel benchmark in Lean 4 focused on optimization. On this benchmark, OptProver achieves state-of-the-art Pass@1 and Pass@32 among comparably sized models while maintaining competitive performance on general theorem-proving tasks, demonstrating effective domain transfer without catastrophic forgetting. 

---
# Pref-CTRL: Preference Driven LLM Alignment using Representation Editing 

**Authors**: Imranul Ashrafi, Inigo Jauregi Unanue, Massimo Piccardi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23543)  

**Abstract**: Test-time alignment methods offer a promising alternative to fine-tuning by steering the outputs of large language models (LLMs) at inference time with lightweight interventions on their internal representations. Recently, a prominent and effective approach, RE-Control (Kong et al., 2024), has proposed leveraging an external value function trained over the LLM's hidden states to guide generation via gradient-based editing. While effective, this method overlooks a key characteristic of alignment tasks, i.e. that they are typically formulated as learning from human preferences between candidate responses. To address this, in this paper we propose a novel preference-based training framework, Pref-CTRL, that uses a multi-objective value function to better reflect the structure of preference data. Our approach has outperformed RE-Control on two benchmark datasets and showed greater generalization on out-of-domain datasets. Our source code is available at this https URL. 

---
# AI Safety Training Can be Clinically Harmful 

**Authors**: Suhas BN, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2604.23445)  

**Abstract**: Large language models are being deployed as mental health support agents at scale, yet only 16% of LLM-based chatbot interventions have undergone rigorous clinical efficacy testing, and simulations reveal psychological deterioration in over one-third of cases. We evaluate four generative models on 250 Prolonged Exposure (PE) therapy scenarios and 146 CBT cognitive restructuring exercises (plus 29 severity-escalated variants), scored by a three-judge LLM panel. All models scored near-perfectly on surface acknowledgment (~0.91-1.00) while therapeutic appropriateness collapsed to 0.22-0.33 at the highest severity for three of four models, with protocol fidelity reaching zero for two. Under CBT severity escalation, one model's task completeness dropped from 92% to 71% while the frontier model's safety-interference score fell from 0.99 to 0.61. We identify a systematic, modality-spanning failure: RLHF safety alignment disrupts the therapeutic mechanism of action by grounding patients during imaginal exposure, offering false reassurance, inserting crisis resources into controlled exercises, and refusing to challenge distorted cognitions mentioning self-harm in PE; and through task abandonment or safety-preamble insertion during CBT cognitive restructuring. These findings motivate a five-axis evaluation framework (protocol fidelity, hallucination risk, behavioral consistency, crisis safety, demographic robustness), mapped onto FDA SaMD and EU AI Act requirements. We argue that no AI mental health system should proceed to deployment without passing multi-axis evaluation across all five dimensions. 

---
# C-MORAL: Controllable Multi-Objective Molecular Optimization with Reinforcement Alignment for LLMs 

**Authors**: Rui Gao, Youngseung Jeon, Swastik Roy, Morteza Ziyadi, Xiang 'Anthony' Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.23061)  

**Abstract**: Large language models (LLMs) show promise for molecular optimization, but aligning them with selective and competing drug-design constraints remains challenging. We propose C-Moral, a reinforcement learning post-training framework for controllable multi-objective molecular optimization. C-Moral combines group-based relative optimization, property score alignment for heterogeneous objectives, and continuous non-linear reward aggregation to improve stability across competing properties. Experiments on the C-MuMOInstruct benchmark show that C-Moral consistently outperforms state-of-the-art models across both in-domain and out-of-domain settings, achieving the best Success Optimized Rate (SOR) of 48.9% on IND tasks and 39.5% on OOD tasks, while largely preserving scaffold similarity. These results suggest that RL post-training is an effective way to align molecular language models with continuous molecular design objectives. Our code and models are publicly available at this https URL. 

---
# DeepImagine: Learning Biomedical Reasoning via Successive Counterfactual Imagining 

**Authors**: Youze Zheng, Jianyou Wang, Yuhan Chen, Matthew Feng, Longtian Bao, Hanyuan Zhang, Maxim Khan, Aditya K. Sehgal, Christopher D. Rosin, Umber Dube, Ramamohan Paturi  

**Link**: [PDF](https://arxiv.org/pdf/2604.23054)  

**Abstract**: Predicting the outcomes of prospective clinical trials remains a major challenge for large language models. Prior work has shown that both traditional correlational predictors, such as random forests and logistic regression, and strong commercial LLMs achieve limited performance on this task. In this paper, we propose DeepImagine, a framework for teaching LLMs biomedical reasoning through successive counterfactual imagining. The central idea is to approximate hidden causal mechanisms of clinical trials by training models to infer how observed trial results would change under controlled perturbations of experimental conditions, such as dosage, outcome measures, study arms, geography, and other trial attributes. To support this objective, we construct both natural and approximate counterfactual pairs from real clinical trials with reported outcomes. For settings where strict counterfactual supervision is available, such as paired outcome measures or dose-ranging study arms within the same trial, we train models with supervised fine-tuning. For broader settings where only approximate counterfactual pairs can be retrieved, we optimize models with reinforcement learning using verifiable rewards based on downstream benchmark correctness. We further augment training with synthetic reasoning traces that provide causally plausible explanations for local counterfactual transitions. Using this pipeline, we train language models under 10B parameters, including Qwen3.5-9B, and evaluate them on clinical trial outcome prediction. We aim to show that DeepImagine consistently improves over untuned language models and traditional correlational baselines. Finally, we aim to show that the learned reasoning trajectories provide interpretable signals about how models represent trial-level mechanisms, suggesting a practical path toward more mechanistic and scientifically useful biomedical language models. 

---
# KARL: Mitigating Hallucinations in LLMs via Knowledge-Boundary-Aware Reinforcement Learning 

**Authors**: Cheng Gao, Cheng Huang, Kangyang Luo, Ziqing Qiao, Shuzheng Si, Huimin Chen, Chaojun Xiao, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.22779)  

**Abstract**: Enabling large language models (LLMs) to appropriately abstain from answering questions beyond their knowledge is crucial for mitigating hallucinations. While existing reinforcement learning methods foster autonomous abstention, they often compromise answer accuracy because their static reward mechanisms, agnostic to models' knowledge boundaries, drive models toward excessive caution. In this work, we propose KARL, a novel framework that continuously aligns an LLM's abstention behavior with its evolving knowledge boundary. KARL introduces two core innovations: a Knowledge-Boundary-Aware Reward that performs online knowledge boundary estimation using within-group response statistics, dynamically rewarding correct answers or guided abstention; and a Two-Stage RL Training Strategy that first explores the knowledge boundary and bypasses the "abstention trap", and subsequently converts incorrect answers beyond the knowledge boundary into abstentions without sacrificing accuracy. Extensive experiments on multiple benchmarks demonstrate that KARL achieves a superior accuracy-hallucination trade-off, effectively suppressing hallucinations while maintaining high accuracy across both in-distribution and out-of-distribution scenarios. 

---
# DPEPO: Diverse Parallel Exploration Policy Optimization for LLM-based Agents 

**Authors**: Junshuo Zhang, Chengrui Huang, Feng Guo, Zihan Li, Ke Shi, Menghua Jiang, Jiguo Yu, Shuo Shang, Shen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.24320)  

**Abstract**: Large language model (LLM) agents that follow the sequential "reason-then-act" paradigm have achieved superior performance in many complex this http URL, these methods suffer from limited exploration and incomplete environmental understanding, as they interact with only a single environment per step. In this paper, we first introduce a novel paradigm that enables an agent to interact with multiple environments simultaneously and share cross-trajectory experiences. Building upon this paradigm, we further propose DPEPO, a reinforcement learning (RL) algorithm that encourages the agent to perform diverse parallel exploration. There are two stages in DPEPO: initial supervised fine-tuning (SFT) imparts basic parallel reasoning and action generation, followed by reinforcement learning stage with a hierarchical reward scheme. We design a parallel trajectory-level success reward and two step-level rewards: Diverse Action Reward and Diverse State Transition Reward, which actively penalize behavioral redundancy and promote broad exploration. Extensive experiments on ALFWorld and ScienceWorld show that DPEPO achieves state-of-the-art (SOTA) success rates, while maintaining comparable efficiency to strong sequential baselines. (Code is available at this https URL) 

---
# IRIS: Interleaved Reinforcement with Incremental Staged Curriculum for Cross-Lingual Mathematical Reasoning 

**Authors**: Navya Gupta, Rishitej Reddy Vyalla, Avinash Anand, Chhavi Kirtani, Erik Cambria, Zhengchen Zhang, Zhengkui Wang, Timothy Liu, Aik Beng Ng, Simon See, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2604.24114)  

**Abstract**: Curriculum learning helps language models tackle complex reasoning by gradually increasing task difficulty. However, it often fails to generate consistent step-by-step reasoning, especially in multilingual and low-resource settings where cross-lingual transfer from English to Indian languages remains limited. We propose IRIS: Interleaved Reinforcement with Incremental Staged Curriculum, a two-axis framework that combines Supervised Fine-Tuning on progressively harder problems (vertical axis) with Reverse Curriculum Reinforcement Learning to reduce reliance on step-by-step guidance (horizontal axis). We design a composite reward combining correctness, step-wise alignment, continuity, and numeric incentives, optimized via Group Relative Policy Optimization (GRPO). We release CL-Math, a dataset of 29k problems with step-level annotations in English, Hindi, and Marathi. Across standard benchmarks and curated multilingual test sets, IRIS consistently improves performance, with strong results on math reasoning tasks and substantial gains in low-resource and bilingual settings, alongside modest improvements in high-resource languages. 

---
# Stabilizing Efficient Reasoning with Step-Level Advantage Selection 

**Authors**: Han Wang, Xiaodong Yu, Jialian Wu, Jiang Liu, Ximeng Sun, Mohit Bansal, Zicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24003)  

**Abstract**: Large language models (LLMs) achieve strong reasoning performance by allocating substantial computation at inference time, often generating long and verbose reasoning traces. While recent work on efficient reasoning reduces this overhead through length-based rewards or pruning, many approaches are post-trained under a much shorter context window than base-model training, a factor whose effect has not been systematically isolated. We first show that short-context post-training alone, using standard GRPO without any length-aware objective, already induces substantial reasoning compression-but at the cost of increasingly unstable training dynamics and accuracy degradation. To address this, we propose Step-level Advantage Selection (SAS), which operates at the reasoning-step level and assigns a zero advantage to low-confidence steps in correct rollouts and to high-confidence steps in verifier-failed rollouts, where failures often arise from truncation or verifier issues rather than incorrect reasoning. Across diverse mathematical and general reasoning benchmarks, SAS improves average Pass@1 accuracy by 0.86 points over the strongest length-aware baseline while reducing average reasoning length by 16.3%, yielding a better accuracy-efficiency trade-off. 

---
# EPM-RL: Reinforcement Learning for On-Premise Product Mapping in E-Commerce 

**Authors**: Minhyeong Yu, Wonduk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2604.23993)  

**Abstract**: Product mapping, the task of deciding whether two e-commerce listings refer to the same product, is a core problem for price monitoring and channel visibility. In real marketplaces, however, sellers frequently inject promotional keywords, platform-specific tags, and bundle descriptions into titles, causing the same product to appear under many different names. Recent LLM-based and multi-agent frameworks improve robustness and interpretability on such hard cases, but they often rely on expensive external APIs, repeated retrieval, and complex inference-time orchestration, making large-scale deployment costly and difficult in privacy-sensitive enterprise settings. To address these issues, we present EPM-RL, a reinforcement-learning-based framework for building an accurate and efficient on-premise e-commerce product mapping model. Our central idea is to distill high-cost agentic reasoning into a trainable in-house model. Starting from a curated set of product pairs with LLM-generated rationales and human verification, we first perform parameter-efficient fine-tuning (PEFT) on a small student model using structured reasoning outputs. We then further optimize the model with Reinforcement Learning (RL) using an agent-based reward that jointly evaluates output-format compliance, label correctness, reasoning--preference scores from specially designed judge models. Preliminary results show that EPM-RL consistently improves over PEFT-only training and offers a stronger quality--cost trade-off than commercial API-based baselines, while enabling private deployment and lower operational cost. These findings suggest that reinforcement learning can turn product mapping from a high-latency agentic pipeline into a scalable, inspectable, and production-ready in-house system. 

---
# LegalDrill: Diagnosis-Driven Synthesis for Legal Reasoning in Small Language Models 

**Authors**: Tianchun Li, Haochen Liu, Vishwa Pardeshi, Xingchen Wang, Tianci Liu, Huijun Zhao, Wei Fan, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.23809)  

**Abstract**: Small language models (SLMs) are promising for real-world deployment due to their efficiency and low operational cost. However, their limited capacity struggles with high-stakes legal reasoning tasks that require coherent statute interpretation and logically consistent deduction. Furthermore, training SLMs for such tasks demands high-quality, concise reasoning trajectories, which are prohibitively expensive to manually collect and difficult to curate via standard rejection sampling, lacking granularity beyond final verdicts. To address these challenges, we propose {LegalDrill}, a diagnosis-driven synthesis framework that extracts and iteratively refines reasoning trajectories from a capable teacher via fine-grained prompting, then a self-reflective verification is employed to adaptively select the most effective data for the SLM student. The resulting data empower SLM training through supervised fine-tuning and direct preference optimization. Extensive experiments on several legal benchmarks demonstrate that {LegalDrill} significantly bolsters the legal reasoning capabilities of representative SLMs while bypassing the need for scarce expert annotations, paving a scalable path toward practical legal reasoning systems. 

---
# Evaluating Large Language Models on Computer Science University Exams in Data Structures 

**Authors**: Edan Gabay, Yael Maoz, Jonathan Stahl, Naama Maoz, Abdo Amer, Orr Eilat, Hanoch Levy, Michal Kleinbort, Amir Rubinstein, Adi Haviv  

**Link**: [PDF](https://arxiv.org/pdf/2604.23347)  

**Abstract**: We present a comprehensive evaluation of Large Language Models (LLMs) on Computer Science (CS) Data Structure examination questions. Our work introduces a new benchmark dataset comprising exam questions from Tel Aviv University (TAU), curated to assess LLMs' abilities in handling closed and multiple-choice questions. We evaluated the performance of OpenAI's GPT 4o and Anthropic's Claude 3.5, popular LLMs, alongside two smaller LLMs, Mathstral 7B and LLaMA 3 8B, across the TAU exams benchmark. Our findings provide insight into the current capabilities of LLMs in CS education. 

---
# Hidden States Know Where Reasoning Diverges: Credit Assignment via Span-Level Wasserstein Distance 

**Authors**: Xinzhu Chen, Wei He, Huichuan Fan, Wenzhe Niu, Zhongxiang Sun, Xuanru Wang, Jiuchong Gao, Jinghua Hao, Renqing He, Weijie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23318)  

**Abstract**: Group Relative Policy Optimization (GRPO) performs coarse-grained credit assignment in reinforcement learning with verifiable rewards (RLVR) by assigning the same advantage to all tokens in a rollout. Process reward models can provide finer-grained supervision, but they require step-level annotation or additional reward modeling. We show that hidden-state distributions contain a useful signal for local reasoning quality that can be extracted using only outcome-level correctness labels available in RLVR. Specifically, within each GRPO group, the Wasserstein distance between span-level hidden state distributions of correct and incorrect rollouts increases around regions where their local reasoning quality diverges. This association holds both across examples and within individual trajectories, suggesting that hidden-state distributional divergence can serve as a self-supervision signal for fine-grained credit assignment. We formalize this observation with a separation theorem showing that, under mild structural assumptions, post-divergence spans have larger Wasserstein distances than pre-divergence spans whenever the population-level distributional gap exceeds finite-sample noise. Motivated by this result, we propose \textbf{S}pan-level \textbf{H}idden state \textbf{E}nabled \textbf{A}dvantage \textbf{R}eweighting (SHEAR), which modifies GRPO by using span-level Wasserstein distances to scale token-level advantages, amplifying updates on tokens whose hidden states are more separated from the opposing group. The method requires no additional model and only minimal changes to the training pipeline. Experiments on five mathematical reasoning benchmarks and five code generation benchmarks show improvements over standard GRPO and strong performance relative to supervised process reward models, while requiring no additional annotation or reward model training. 

---
# Bridging Reasoning and Action: Hybrid LLM-RL Framework for Efficient Cross-Domain Task-Oriented Dialogue 

**Authors**: Yangyang Zhao, Linfan Dai, Li Cai, Bowen Xing, Libo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.23345)  

**Abstract**: Cross-domain task-oriented dialogue requires reasoning over implicit and explicit feasibility constraints while planning long-horizon, multi-turn actions. Large language models (LLMs) can infer such constraints but are unreliable over long horizons, while Reinforcement learning (RL) optimizes long-horizon behavior yet cannot recover constraints from raw dialogue. Naively coupling LLMs with RL is therefore brittle: unverified or unstructured LLM outputs can corrupt state representations and misguide policy learning. Motivated by this, we propose Verified LLM-Knowledge empowered RL (VLK-RL), a hybrid framework that makes LLM-derived constraint reasoning usable for RL. VLK-RL first elicits candidate constraints with an LLM and then verifies them via a dual-role cross-examination procedure to suppress hallucinations and cross-turn inconsistencies. The verified constraints are mapped into ontology-aligned slot-value representations, yielding a structured, constraint-aware state for RL policy optimization. Experiments across multiple benchmarks demonstrate that VLK-RL significantly improves generalization and robustness, outperforming strong single-model baselines on long-horizon tasks. 

---
# Process Supervision of Confidence Margin for Calibrated LLM Reasoning 

**Authors**: Liaoyaqi Wang, Chunsheng Zuo, William Jurayj, Benjamin Van Durme, Anqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.23333)  

**Abstract**: Scaling test-time computation with reinforcement learning (RL) has emerged as a reliable path to improve large language models (LLM) reasoning ability. Yet, outcome-based reward often incentivizes models to be overconfident, leading to hallucinations, unreliable confidence-based control, and unnecessary compute allocation. We introduce Reinforcement Learning with Confidence Margin (\textbf{RLCM}), a calibration-aware RL framework that jointly optimizes correctness and confidence reliability via a margin-enhanced process reward over intermediate-budget completions. Rather than aligning confidence to correctness likelihoods, RLCM encourages to widen the confidence margin between correct and incorrect steps within a single reasoning trajectory. Across mathematical, code, logic and science benchmarks, our method substantially improves calibration while maintaining or improving accuracy. We further show that, with calibrated confidence signals, the resulting models enable more efficient conformal risk control and effective confidence-weighted aggregation. 

---
# The Collapse of Heterogeneity in Silicon Philosophers 

**Authors**: Yuanming Shi, Andreas Haupt  

**Link**: [PDF](https://arxiv.org/pdf/2604.23575)  

**Abstract**: Silicon samples are increasingly used as a low-cost substitute for human panels and have been shown to reproduce aggregate human opinion with high fidelity. We show that, in the alignment-relevant domain of philosophy, silicon samples systematically collapse heterogeneity. Using data from $N = {277}$ professional philosophers drawn from PhilPeople profiles, we evaluate seven proprietary and open-source large language models on their ability to replicate individual philosophical positions and to preserve cross-question correlation structures across philosophical domains. We find that language models substantially over-correlate philosophical judgments, producing artificial consensus across domains. This collapse is associated in part with specialist effects, whereby models implicitly assume that domain specialists hold highly similar philosophical views. We assess the robustness of these findings by studying the impact of DPO fine-tuning and by validating results against the full PhilPapers 2020 Survey ($N = {1785}$). We conclude by discussing implications for alignment, evaluation, and the use of silicon samples as substitutes for human judgment. The code of this project can be found at this https URL. 

---
# AeSlides: Incentivizing Aesthetic Layout in LLM-Based Slide Generation via Verifiable Rewards 

**Authors**: Yiming Pan, Chengwei Hu, Xuancheng Huang, Can Huang, Mingming Zhao, Yuean Bi, Xiaohan Zhang, Aohan Zeng, Linmei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.22840)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential in agentic tasks, particularly in slide generation. However, slide generation poses a fundamental challenge: the generation process is text-centric, whereas its quality is governed by visual aesthetics. This modality gap leads current models to frequently produce slides with aesthetically suboptimal layouts. Existing solutions typically rely either on heavy visual reflection, which incurs high inference cost yet yields limited gains; or on fine-tuning with large-scale datasets, which still provides weak and indirect aesthetic supervision. In contrast, the explicit use of aesthetic principles as supervision remains unexplored. In this work, we present AeSlides, a reinforcement learning framework with verifiable rewards for Aesthetic layout supervision in Slide generation. We introduce a suite of meticulously designed verifiable metrics to quantify slide layout quality, capturing key layout issues in an accurate, efficient, and low-cost manner. Leveraging these verifiable metrics, we develop a GRPO-based reinforcement learning method that directly optimizes slide generation models for aesthetically coherent layouts. With only 5K training prompts on GLM-4.7-Flash, AeSlides improves aspect ratio compliance from 36% to 85%, while reducing whitespace by 44%, element collisions by 43%, and visual imbalance by 28%. Human evaluation further shows a substantial improvement in overall quality, increasing scores from 3.31 to 3.56 (+7.6%), outperforming both model-based reward optimization and reflection-based agentic approaches, and even edging out Claude-Sonnet-4.5. These results demonstrate that such a verifiable aesthetic paradigm provides an efficient and scalable approach to aligning slide generation with human aesthetic preferences. Our repository is available at this https URL. 

---
