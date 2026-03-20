# Lightweight Adaptation for LLM-based Technical Service Agent: Latent Logic Augmentation and Robust Noise Reduction 

**Authors**: Yi Yu, Junzhuo Ma, Chenghuang Shen, Xingyan Liu, Jing Gu, Hangyi Sun, Guangquan Hu, Jianfeng Liu, Weiting Liu, Mingyue Pu, Yu Wang, Zhengdong Xiao, Rui Xie, Longjiu Luo, Qianrong Wang, Gurong Cui, Honglin Qiao, Wenlian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18074)  

**Abstract**: Adapting Large Language Models in complex technical service domains is constrained by the absence of explicit cognitive chains in human demonstrations and the inherent ambiguity arising from the diversity of valid responses. These limitations severely hinder agents from internalizing latent decision dynamics and generalizing effectively. Moreover, practical adaptation is often impeded by the prohibitive resource and time costs associated with standard training paradigms. To overcome these challenges and guarantee computational efficiency, we propose a lightweight adaptation framework comprising three key contributions. (1) Latent Logic Augmentation: We introduce Planning-Aware Trajectory Modeling and Decision Reasoning Augmentation to bridge the gap between surface-level supervision and latent decision logic. These approaches strengthen the stability of Supervised Fine-Tuning alignment. (2) Robust Noise Reduction: We construct a Multiple Ground Truths dataset through a dual-filtering method to reduce the noise by validating diverse responses, thereby capturing the semantic diversity. (3) Lightweight Adaptation: We design a Hybrid Reward mechanism that fuses an LLM-based judge with a lightweight relevance-based Reranker to distill high-fidelity reward signals while reducing the computational cost compared to standard LLM-as-a-Judge reinforcement learning. Empirical evaluations on real-world Cloud service tasks, conducted across semantically diverse settings, demonstrate that our framework achieves stability and performance gains through Latent Logic Augmentation and Robust Noise Reduction. Concurrently, our Hybrid Reward mechanism achieves alignment comparable to standard LLM-as-a-judge methods with reduced training time, underscoring the practical value for deploying technical service agents. 

---
# Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation 

**Authors**: Zhuolin Yang, Zihan Liu, Yang Chen, Wenliang Dai, Boxin Wang, Sheng-Chieh Lin, Chankyu Lee, Yangyi Chen, Dongfu Jiang, Jiafan He, Renjie Pi, Grace Lam, Nayeon Lee, Alexander Bukharin, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2603.19220)  

**Abstract**: We introduce Nemotron-Cascade 2, an open 30B MoE model with 3B activated parameters that delivers best-in-class reasoning and strong agentic capabilities. Despite its compact size, its mathematical and coding reasoning performance approaches that of frontier open models. It is the second open-weight LLM, after DeepSeekV3.2-Speciale-671B-A37B, to achieve Gold Medal-level performance in the 2025 International Mathematical Olympiad (IMO), the International Olympiad in Informatics (IOI), and the ICPC World Finals, demonstrating remarkably high intelligence density with 20x fewer parameters. In contrast to Nemotron-Cascade 1, the key technical advancements are as follows. After SFT on a meticulously curated dataset, we substantially expand Cascade RL to cover a much broader spectrum of reasoning and agentic domains. Furthermore, we introduce multi-domain on-policy distillation from the strongest intermediate teacher models for each domain throughout the Cascade RL process, allowing us to efficiently recover benchmark regressions and sustain strong performance gains along the way. We release the collection of model checkpoint and training data. 

---
# MoRI: Learning Motivation-Grounded Reasoning for Scientific Ideation in Large Language Models 

**Authors**: Chenyang Gu, Jiahao Cheng, Meicong Zhang, Pujun Zheng, Jinquan Zheng, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2603.19044)  

**Abstract**: Scientific ideation aims to propose novel solutions within a given scientific context. Existing LLM-based agentic approaches emulate human research workflows, yet inadequately model scientific reasoning, resulting in surface-level conceptual recombinations that lack technical depth and scientific grounding. To address this issue, we propose \textbf{MoRI} (\textbf{Mo}tivation-grounded \textbf{R}easoning for Scientific \textbf{I}deation), a framework that enables LLMs to explicitly learn the reasoning process from research motivations to methodologies. The base LLM is initialized via supervised fine-tuning to generate a research motivation from a given context, and is subsequently trained under a composite reinforcement learning reward that approximates scientific rigor: (1) entropy-aware information gain encourages the model to uncover and elaborate high-complexity technical details grounded in ground-truth methodologies, and (2) contrastive semantic gain constrains the reasoning trajectory to maintain conceptually aligned with scientifically valid solutions. Empirical results show that MoRI significantly outperforms strong commercial LLMs and complex agentic baselines across multiple dimensions, including novelty, technical rigor, and feasibility. The code will be made available on \href{this https URL}{GitHub}. 

---
# Mi:dm K 2.5 Pro 

**Authors**: KT Tech innovation Group  

**Link**: [PDF](https://arxiv.org/pdf/2603.18788)  

**Abstract**: The evolving LLM landscape requires capabilities beyond simple text generation, prioritizing multi-step reasoning, long-context understanding, and agentic workflows. This shift challenges existing models in enterprise environments, especially in Korean-language and domain-specific scenarios where scaling is insufficient. We introduce Mi:dm K 2.5 Pro, a 32B parameter flagship LLM designed to address enterprise-grade complexity through reasoning-focused optimization.
Our methodology builds a robust data foundation via a quality-centric curation pipeline utilizing abstract syntax tree (AST) analysis for code, gap-filling synthesis for mathematics, and an LLM-based quality evaluator. Pre-training scales the model via layer-predictor-based Depth Upscaling (DuS) and a progressive strategy supporting a 128K token context window. Post-training introduces a specialized multi-stage pipeline, including Reasoning SFT, model merging, and asynchronous reinforcement learning (RL), to develop complex problem-solving skills. "Fusion Training" then rebalances these capabilities with conversational fluency, consistent response styling, and reliable tool-use.
The evaluations show that Mi:dm K 2.5 Pro achieves competitive performance against leading global and domestic models. In addition, it sets state-of-the-art results on Korean-specific benchmarks, showcasing deep linguistic and cultural understanding. Finally, Responsible AI evaluations validate safety against attacks, ensuring a secure profile for deployment with a balance of harmlessness and responsiveness. 

---
# Learning to Self-Evolve 

**Authors**: Xiaoyin Chen, Canwen Xu, Yite Wang, Boyi Liu, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2603.18620)  

**Abstract**: We introduce Learning to Self-Evolve (LSE), a reinforcement learning framework that trains large language models (LLMs) to improve their own contexts at test time. We situate LSE in the setting of test-time self-evolution, where a model iteratively refines its context from feedback on seen problems to perform better on new ones. Existing approaches rely entirely on the inherent reasoning ability of the model and never explicitly train it for this task. LSE reduces the multi-step evolution problem to a single-step RL objective, where each context edit is rewarded by the improvement in downstream performance. We pair this objective with a tree-guided evolution loop. On Text-to-SQL generation (BIRD) and general question answering (MMLU-Redux), a 4B-parameter model trained with LSE outperforms self-evolving policies powered by GPT-5 and Claude Sonnet 4.5, as well as prompt optimization methods including GEPA and TextGrad, and transfers to guide other models without additional training. Our results highlight the effectiveness of treating self-evolution as a learnable skill. 

---
# Adaptive Decoding via Test-Time Policy Learning for Self-Improving Generation 

**Authors**: Asmita Bhardwaj, Yuya Jeremy Ong, Eelaaf Zahid, Basel Shbita  

**Link**: [PDF](https://arxiv.org/pdf/2603.18428)  

**Abstract**: Decoding strategies largely determine the quality of Large Language Model (LLM) outputs, yet widely used heuristics such as greedy or fixed temperature/top-p decoding are static and often task-agnostic, leading to suboptimal or inconsistent generation quality across domains that demand stylistic or structural flexibility. We introduce a reinforcement learning-based decoder sampler that treats decoding as sequential decision-making and learns a lightweight policy to adjust sampling parameters at test-time while keeping LLM weights frozen. We evaluated summarization datasets including BookSum, arXiv, and WikiHow using Granite-3.3-2B and Qwen-2.5-0.5B. Our policy sampler consistently outperforms greedy and static baselines, achieving relative gains of up to +88% (BookSum, Granite) and +79% (WikiHow, Qwen). Reward ablations show that overlap-only objectives underperform compared to composite rewards, while structured shaping terms (length, coverage, repetition, completeness) enable stable and sustained improvements. These findings highlight reinforcement learning as a practical mechanism for test-time adaptation in decoding, enabling domain-aware and user-controllable generation without retraining large models. 

---
# TARo: Token-level Adaptive Routing for LLM Test-time Alignment 

**Authors**: Arushi Rai, Qiang Zhang, Hanqing Zeng, Yunkai Zhang, Dipesh Tamboli, Xiangjun Fan, Zhuokai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18411)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning capabilities but typically require expensive post-training to reach high performance. Recent test-time alignment methods offer a lightweight alternative, but have been explored mainly for preference alignment rather than reasoning. To bridge this gap, we propose, Token-level Adaptive Routing (TARo), which steers frozen LLMs toward structured reasoning entirely at inference time. Specifically, we first train reward models on step-wise mathematical traces to capture fine-grained logical consistency signals, then introduce a learnable token-level router that automatically controls the guidance of the reward model to the base model. Extensive experiments show that TARo significantly improves reasoning performance by up to +22.4% over base model and +8.4% over existing token-level test-time alignment methods, while also boosting out-of-distribution clinical reasoning (MedXpertQA) and instruction following (AlpacaEval). Furthermore, TARo also generalizes from small to large backbones without retraining, extending test-time alignment from preference optimization to robust, cross-domain reasoning. 

---
# Box Maze: A Process-Control Architecture for Reliable LLM Reasoning 

**Authors**: Zou Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19182)  

**Abstract**: Large language models (LLMs) demonstrate strong generative capabilities but remain vulnerable to hallucination and unreliable reasoning under adversarial prompting. Existing safety approaches -- such as reinforcement learning from human feedback (RLHF) and output filtering -- primarily operate at the behavioral level and may lack explicit architectural mechanisms for enforcing reasoning process integrity.
This paper proposes the Box Maze framework, a conceptual process-control architecture that decomposes LLM reasoning into three explicit layers: memory grounding, structured inference, and boundary enforcement. We introduce preliminary simulation-based evaluation involving progressive boundary erosion scenarios across multiple heterogeneous LLM systems (DeepSeek-V3, Doubao, Qwen). Results from n=50 adversarial scenarios suggest that explicit cognitive control layers may improve consistency in boundary maintenance, with architectural constraints reducing boundary failure rates from approximately 40% (baseline RLHF) to below 1% under adversarial conditions.
While current validation is simulation-based, these preliminary results indicate that process-level control may offer a promising direction for improving reliability in large language model reasoning. 

---
# TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots 

**Authors**: Fangrui Huang, Souhad Chbeir, Arpandeep Khatua, Sheng Wang, Sijun Tan, Kenan Ye, Lily Bailey, Merryn Daniel, Ryan Louie, Sanmi Koyejo, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2603.18008)  

**Abstract**: Large language models (LLMs) are increasingly used for mental-health support; yet prevailing evaluation methods--fluency metrics, preference tests, and generic dialogue benchmarks--fail to capture the clinically critical dimensions of psychotherapy. We introduce THERAPYGYM, a framework that evaluates and improves therapy chatbots along two clinical pillars: fidelity and safety. Fidelity is measured using the Cognitive Therapy Rating Scale (CTRS), implemented as an automated pipeline that scores adherence to CBT techniques over multi-turn sessions. Safety is assessed using a multi-label annotation scheme, covering therapy-specific risks (e.g., failing to address harm or abuse). To mitigate bias and unreliability in LLM-based judges, we further release THERAPYJUDGEBENCH, a validation set of 116 dialogues with 1,270 expert ratings for auditing and calibration against licensed clinicians. THERAPYGYM also serves as a training harness: CTRS and safety-based rewards drive RL with configurable patient simulations spanning diverse symptom profiles. Models trained in THERAPYGYM improve on expert ratings, with average CTRS rising from 0.10 to 0.60 (and 0.16 to 0.59 under LLM judges). Our work enables scalable development of therapy chatbots that are faithful to evidence-based practice and safer in high-stakes use. 

---
# Online Learning and Equilibrium Computation with Ranking Feedback 

**Authors**: Mingyang Liu, Yongshan Chen, Zhiyuan Fan, Gabriele Farina, Asuman Ozdaglar, Kaiqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19221)  

**Abstract**: Online learning in arbitrary, and possibly adversarial, environments has been extensively studied in sequential decision-making, and it is closely connected to equilibrium computation in game theory. Most existing online learning algorithms rely on \emph{numeric} utility feedback from the environment, which may be unavailable in human-in-the-loop applications and/or may be restricted by privacy concerns. In this paper, we study an online learning model in which the learner only observes a \emph{ranking} over a set of proposed actions at each timestep. We consider two ranking mechanisms: rankings induced by the \emph{instantaneous} utility at the current timestep, and rankings induced by the \emph{time-average} utility up to the current timestep, under both \emph{full-information} and \emph{bandit} feedback settings. Using the standard external-regret metric, we show that sublinear regret is impossible with instantaneous-utility ranking feedback in general. Moreover, when the ranking model is relatively deterministic, \emph{i.e.}, under the Plackett-Luce model with a temperature that is sufficiently small, sublinear regret is also impossible with time-average utility ranking feedback. We then develop new algorithms that achieve sublinear regret under the additional assumption that the utility sequence has sublinear total variation. Notably, for full-information time-average utility ranking feedback, this additional assumption can be removed. As a consequence, when all players in a normal-form game follow our algorithms, repeated play yields an approximate coarse correlated equilibrium. We also demonstrate the effectiveness of our algorithms in an online large-language-model routing task. 

---
# Are complicated loss functions necessary for teaching LLMs to reason? 

**Authors**: Gabriele Carrino, Andrea Sassella, Nicolo Brunello, Federico Toschi, Mark James Carman  

**Link**: [PDF](https://arxiv.org/pdf/2603.18756)  

**Abstract**: Recent advances in large language models (LLMs) highlight the importance of post training techniques for improving reasoning and mathematical ability. Group Relative Policy Optimization (GRPO) has shown promise in this domain by combining group relative advantage estimation, PPO style clipping, and KL regularization. However, its complexity raises the question of whether all components are necessary for fostering reasoning behaviors. We conduct a systematic analysis of GRPO and identify two key findings: (1) incorporating negative feedback is essential training solely on actions above a baseline limits learning; and (2) PPO style constraints, such as policy ratio clipping, are not required to improve mathematical reasoning or performance. Building on these insights, we propose REINFORCE with Group Relative Advantage (RGRA), a simplified variant that retains group relative advantage estimation but removes PPO style clipping and policy ratio terms. Experiments across standard mathematical benchmarks indicate that RGRA has the potential to achieve stronger performance than GRPO. Our results suggest that simpler REINFORCE based approaches can effectively enhance reasoning in LLMs, offering a more transparent and efficient alternative to GRPO. 

---
# RewardFlow: Topology-Aware Reward Propagation on State Graphs for Agentic RL with Large Language Models 

**Authors**: Xiao Feng, Bo Han, Zhanke Zhou, Jiaqi Fan, Jiangchao Yao, Ka Ho Li, Dahai Yu, Michael Kwok-Po Ng  

**Link**: [PDF](https://arxiv.org/pdf/2603.18859)  

**Abstract**: Reinforcement learning (RL) holds significant promise for enhancing the agentic reasoning capabilities of large language models (LLMs) with external environments. However, the inherent sparsity of terminal rewards hinders fine-grained, state-level optimization. Although process reward modeling offers a promising alternative, training dedicated reward models often entails substantial computational costs and scaling difficulties. To address these challenges, we introduce RewardFlow, a lightweight method for estimating state-level rewards tailored to agentic reasoning tasks. RewardFlow leverages the intrinsic topological structure of states within reasoning trajectories by constructing state graphs. This enables an analysis of state-wise contributions to success, followed by topology-aware graph propagation to quantify contributions and yield objective, state-level rewards. When integrated as dense rewards for RL optimization, RewardFlow substantially outperforms prior RL baselines across four agentic reasoning benchmarks, demonstrating superior performance, robustness, and training efficiency. The implementation of RewardFlow is publicly available at this https URL. 

---
# CausalRM: Causal-Theoretic Reward Modeling for RLHF from Observational User Feedbacks 

**Authors**: Hao Wang, Licheng Pan, Zhichao Chen, Chunyuan Zheng, Zhixuan Chu, Xiaoxi Li, Yuan Lu, Xinggao Liu, Haoxuan Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.18736)  

**Abstract**: Despite the success of reinforcement learning from human feedback (RLHF) in aligning language models, current reward modeling heavily relies on experimental feedback data collected from human annotators under controlled and costly conditions. In this work, we introduce observational reward modeling -- learning reward models with observational user feedback (e.g., clicks, copies, and upvotes) -- as a scalable and cost-effective alternative. We identify two fundamental challenges in this setting: (1) observational feedback is noisy due to annotation errors, which deviates it from true user preference; (2) observational feedback is biased by user preference, where users preferentially provide feedback on responses they feel strongly about, which creats a distribution shift between training and inference data. To address these challenges, we propose CausalRM, a causal-theoretic reward modeling framework that aims to learn unbiased reward models from observational feedback. To tackle challenge (1), CausalRM introduces a noise-aware surrogate loss term that is provably equivalent to the primal loss under noise-free conditions by explicitly modeling the annotation error generation process. To tackle challenge (2), CausalRM uses propensity scores -- the probability of a user providing feedback for a given response -- to reweight training samples, yielding a loss function that eliminates user preference bias. Extensive experiments across diverse LLM backbones and benchmark datasets validate that CausalRM effectively learns accurate reward signals from noisy and biased observational feedback and delivers substantial performance improvements on downstream RLHF tasks -- including a 49.2% gain on WildGuardMix and a 32.7% improvement on HarmBench. Code is available on our project website. 

---
# MOSAIC: Multi-Objective Slice-Aware Iterative Curation for Alignment 

**Authors**: Yipu Dou, Wang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18637)  

**Abstract**: We study how to allocate a fixed supervised fine-tuning budget when three objectives must be balanced at once: multi-turn safety alignment, low over-refusal on benign boundary queries, and instruction following under verifiable constraints. We propose MOSAIC (Multi-Objective Slice-Aware Iterative Curation for Alignment), a multi-objective framework for closed-loop data mixture search built on a unified L1-L3 evaluation interface. MOSAIC turns slice-level failure profiles into executable data actions, including dataset-level mixture ratios, bucket-level weights, and focus criteria. Under a fixed 1M-token budget and five rounds of independent fine-tuning from the same base model, MOSAIC improves internal XGuard from 2.76 to 4.67 while keeping OrBench at 4.41 and IFEval at 3.65. The final Pareto solution also generalizes better than a random static LoRA baseline on independent attack, over-refusal, and capability tests, suggesting that structured failure diagnosis can serve as a practical control signal for budgeted data construction. Code is available at this https URL. 

---
# Reasoning over mathematical objects: on-policy reward modeling and test time aggregation 

**Authors**: Pranjal Aggarwal, Marjan Ghazvininejad, Seungone Kim, Ilia Kulikov, Jack Lanchantin, Xian Li, Tianjian Li, Bo Liu, Graham Neubig, Anaelia Ovalle, Swarnadeep Saha, Sainbayar Sukhbaatar, Sean Welleck, Jason Weston, Chenxi Whitehouse, Adina Williams, Jing Xu, Ping Yu, Weizhe Yuan, Jingyu Zhang, Wenting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18886)  

**Abstract**: The ability to precisely derive mathematical objects is a core requirement for downstream STEM applications, including mathematics, physics, and chemistry, where reasoning must culminate in formally structured expressions. Yet, current LM evaluations of mathematical and scientific reasoning rely heavily on simplified answer formats such as numerical values or multiple choice options due to the convenience of automated assessment. In this paper we provide three contributions for improving reasoning over mathematical objects: (i) we build and release training data and benchmarks for deriving mathematical objects, the Principia suite; (ii) we provide training recipes with strong LLM-judges and verifiers, where we show that on-policy judge training boosts performance; (iii) we show how on-policy training can also be used to scale test-time compute via aggregation. We find that strong LMs such as Qwen3-235B and o3 struggle on Principia, while our training recipes can bring significant improvements over different LLM backbones, while simultaneously improving results on existing numerical and MCQA tasks, demonstrating cross-format generalization of reasoning abilities. 

---
# ZEBRAARENA: A Diagnostic Simulation Environment for Studying Reasoning-Action Coupling in Tool-Augmented LLMs 

**Authors**: Wanjia Zhao, Ludwig Schmidt, James Zou, Vidhisha Balachandran, Lingjiao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.18614)  

**Abstract**: Tool-augmented large language models (LLMs) must tightly couple multi-step reasoning with external actions, yet existing benchmarks often confound this interplay with complex environment dynamics, memorized knowledge or dataset contamination. In this paper, we introduce ZebraArena, a procedurally generated diagnostic environment for studying reasoning-action coupling in tool-augmented LLMs, with controllable difficulty and a knowledge-minimal design, which limits gains from memorization or dataset contamination. Each task in ZebraArena requires a set of critical information which is available only through targeted tool use, yielding an interpretable interface between external information acquisition and deductive reasoning. This design provides deterministic evaluation via unique solutions, and a theoretical optimal query count for measuring efficient tool use. We show that ZebraArena requires a combination of in-depth reasoning and accurate external tool calling, which remains a challenge as frontier reasoning models such as GPT-5 and Gemini 2.5 Pro only achieves 60% accuracy on the hard instances. We also observe a persistent gaps between theoretical optimality and practical tool usage. For example, GPT-5 uses 70-270% more tool calls than the theoretical optimum. We highlight the key findings in our evaluation, and hope ZebraArena stimulates further research on the interplay between internal reasoning and external action. 

---
# Thinking with Constructions: A Benchmark and Policy Optimization for Visual-Text Interleaved Geometric Reasoning 

**Authors**: Haokun Zhao, Wanshi Xu, Haidong Yuan, Songjun Cao, Long Ma, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.18662)  

**Abstract**: Geometric reasoning inherently requires "thinking with constructions" -- the dynamic manipulation of visual aids to bridge the gap between problem conditions and solutions. However, existing Multimodal Large Language Models (MLLMs) are largely confined to passive inference with static diagrams, lacking the strategic knowledge of when and how to construct effective visual aids. To address this, we present a framework for Visual-Text Interleaved Chain-of-Thought. We first introduce GeoAux-Bench, the first benchmark comprising 4,334 geometry problems that aligns textual construction steps with ground-truth visual updates. Our pilot study reveals two critical insights: (1) interleaved visual-textual aids outperform single-modality counterparts, which cannot losslessly capture geometric synergy; and (2) valid constructions act as entropy reducers, strongly correlating with reduced reasoning perplexity. Building on these findings, we propose Action Applicability Policy Optimization (A2PO), a reinforcement learning paradigm for mastering strategic construction. A2PO employs Adaptive Reward Shaping to regulate the timing and quality of visual aids via counterfactual sampling to distinguish necessary from redundant constructions. Experiments demonstrate our approach enables MLLMs to leverage selective auxiliary constructions, yielding a 3.51% gain over strong baselines. Code and data are available on GitHub. 

---
# Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM 

**Authors**: Zizhao Hu, Mohammad Rostami, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2603.18507)  

**Abstract**: Persona prompting can steer LLM generation towards a domain-specific tone and pattern. This behavior enables use cases in multi-agent systems where diverse interactions are crucial and human-centered tasks require high-level human alignment. Prior works provide mixed opinions on their utility: some report performance gains when using expert personas for certain domains and their contribution to data diversity in synthetic data creation, while others find near-zero or negative impact on general utility. To fully leverage the benefits of the LLM persona and avoid its harmfulness, a more comprehensive investigation of the mechanism is crucial. In this work, we study how model optimization, task type, prompt length, and placement can impact expert persona effectiveness across instruction-tuned and reasoning LLMs, and provide insight into conditions under which expert personas fail and succeed. Based on our findings, we developed a pipeline to fully leverage the benefits of an expert persona, named PRISM (Persona Routing via Intent-based Self-Modeling), which self-distills an intent-conditioned expert persona into a gated LoRA adapter through a bootstrapping process that requires no external data, models, or knowledge. PRISM enhances human preference and safety alignment on generative tasks while maintaining accuracy on discriminative tasks across all models, with minimal memory and computing overhead. 

---
# ARIADNE: A Perception-Reasoning Synergy Framework for Trustworthy Coronary Angiography Analysis 

**Authors**: Zhan Jin, Yu Luo, Yizhou Zhang, Ziyang Cui, Yuqing Wei, Xianchao Liu, Xueying Zeng, Qing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19169)  

**Abstract**: Conventional pixel-wise loss functions fail to enforce topological constraints in coronary vessel segmentation, producing fragmented vascular trees despite high pixel-level accuracy. We present ARIADNE, a two-stage framework coupling preference-aligned perception with RL-based diagnostic reasoning for topologically coherent stenosis detection. The perception module employs DPO to fine-tune the Sa2VA vision-language foundation model using Betti number constraints as preference signals, aligning the policy toward geometrically complete vessel structures rather than pixel-wise overlap metrics. The reasoning module formulates stenosis localization as a Markov Decision Process with an explicit rejection mechanism that autonomously defers ambiguous anatomical candidates such as bifurcations and vessel crossings, shifting from coverage maximization to reliability optimization. On 1,400 clinical angiograms, ARIADNE achieves state-of-the-art centerline Dice of 0.838, reduces false positives by 41% compared to geometric baselines. External validation on multi-center benchmarks ARCADE and XCAD confirms generalization across acquisition protocols. This represents the first application of DPO for topological alignment in medical imaging, demonstrating that preference-based learning over structural constraints mitigates topological violations while maintaining diagnostic sensitivity in interventional cardiology workflows. 

---
# MolRGen: A Training and Evaluation Setting for De Novo Molecular Generation with Reasonning Models 

**Authors**: Philippe Formont, Maxime Darrin, Ismail Ben Ayed, Pablo Piantanida  

**Link**: [PDF](https://arxiv.org/pdf/2603.18256)  

**Abstract**: Recent advances in reasoning-based large language models (LLMs) have demonstrated substantial improvements in complex problem-solving tasks. Motivated by these advances, several works have explored the application of reasoning LLMs to drug discovery and molecular design. However, most existing approaches either focus on evaluation or rely on training setups that require ground-truth labels, such as molecule pairs with known property modifications. Such supervision is unavailable in \textit{de novo} molecular generation, where the objective is to generate novel molecules that optimize a desirability score without prior knowledge of high-scoring candidates. To bridge this gap, we introduce MolRGen, a large-scale benchmark and dataset for training and evaluating reasoning-based LLMs on \textit{de novo} molecular generation. Our contributions are threefold. First, we propose a setting to evaluate and train models for \textit{de novo} molecular generation and property prediction. Second, we introduce a novel diversity-aware top-$k$ score that captures both the quality and diversity of generated molecules. Third, we show our setting can be used to train LLMs for molecular generation, training a 24B LLM with reinforcement learning, and we provide a detailed analysis of its performance and limitations. 

---
# VC-Soup: Value-Consistency Guided Multi-Value Alignment for Large Language Models 

**Authors**: Hefei Xu, Le Wu, Yu Wang, Min Hou, Han Wu, Zhen Zhang, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.18113)  

**Abstract**: As large language models (LLMs) increasingly shape content generation, interaction, and decision-making across the Web, aligning them with human values has become a central objective in trustworthy AI. This challenge becomes even more pronounced when aligning multiple, potentially conflicting human values. Although recent approaches, such as reward reweighting, prompt-based supervised fine-tuning, and model merging, attempt to tackle multi-value alignment, they still face two major limitations: (1) training separate models for each value combination is prohibitively expensive; (2) value conflicts substantially degrade alignment performance. These limitations make it difficult to achieve favorable trade-offs across diverse human values. To address these challenges, we revisit multi-value alignment from the perspective of value consistency in data and propose VC-soup, a data filtering and parameter merging framework grounded in value-consistent learning. We first design a value consistency metric based on the cosine similarity between the reward-gap vector of each preference pair and an all-ones vector, which quantifies its cross-value coherence. We then filter out low-consistency preference pairs in each value dataset and train on the remaining data to obtain smooth, value-consistent policy models that better preserve linear mode connectivity. Finally, we linearly combine these policies and apply Pareto filtering across values to obtain solutions with balanced multi-value performance. Extensive experiments and theoretical analysis demonstrate that VC-soup effectively mitigates conflicts and consistently outperforms existing multi-value alignment methods. 

---
# Insight-V++: Towards Advanced Long-Chain Visual Reasoning with Multimodal Large Language Models 

**Authors**: Yuhao Dong, Zuyan Liu, Shulin Tian, Yongming Rao, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.18118)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable reliability and advanced capabilities through extended test-time reasoning. However, extending these capabilities to Multi-modal Large Language Models (MLLMs) remains a significant challenge due to a critical scarcity of high-quality, long-chain reasoning data and optimized training pipelines. To bridge this gap, we present a unified multi-agent visual reasoning framework that systematically evolves from our foundational image-centric model, Insight-V, into a generalized spatial-temporal architecture, Insight-V++. We first propose a scalable data generation pipeline equipped with multi-granularity assessment that autonomously synthesizes structured, complex reasoning trajectories across image and video domains without human intervention. Recognizing that directly supervising MLLMs with such intricate data yields sub-optimal results, we design a dual-agent architecture comprising a reasoning agent to execute extensive analytical chains, and a summary agent to critically evaluate and distill final outcomes. While our initial framework utilized Direct Preference Optimization (DPO), its off-policy nature fundamentally constrained reinforcement learning potential. To overcome these limitations, particularly for long-horizon video understanding, Insight-V++ introduces two novel algorithms, ST-GRPO and J-GRPO, which enhance spatial-temporal reasoning and improve evaluative robustness. Crucially, by leveraging reliable feedback from the summary agent, we guide an iterative reasoning path generation process, retraining the entire multi-agent system in a continuous, self-improving loop. Extensive experiments on base models like LLaVA-NeXT and Qwen2.5-VL demonstrate significant performance gains across challenging image and video reasoning benchmarks while preserving strong capabilities on traditional perception-focused tasks. 

---
