# ChatShopBuddy: Towards Reliable Conversational Shopping Agents via Reinforcement Learning 

**Authors**: Yiruo Cheng, Kelong Mao, Tianhao Li, Jiejun Tan, Ji-Rong Wen, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2603.06065)  

**Abstract**: Conversational shopping agents represent a critical consumer-facing application of Large Language Model (LLM)-powered agents, yet how to effectively apply post-training Reinforcement Learning (RL) to optimize such agents remains underexplored. This work investigates RL-based optimization for shopping agents in real-world scenarios, where agents must simultaneously satisfy multiple interdependent objectives spanning objective metrics (product correctness), subjective qualities (persuasiveness), outcome rewards (final response quality), and process rewards (tool efficiency). We present a complete methodology to address this challenge. Specifically, we first construct SmartShopBench, a benchmark that captures diverse shopping intents with a hierarchical evaluation that decomposes complex quality requirements into measurable levels. Building on this evaluation framework, we design Hierarchical Reward Modeling (HRM) to structure mixed reward types through conditional gating that reflects their logical dependencies. To enable efficient training, we further propose Dynamic Contrastive Policy Optimization (DCPO), which balances response quality with operational efficiency through dynamic trajectory selection based on reward and reasoning length. Extensive experiments demonstrate that our RL-trained agent, namely ChatShopBuddy, consistently outperforms larger models relying on generic reasoning, achieving superior stability rather than merely higher peaks. Our work provides valuable guidance for applying RL to real-world conversational agents. 

---
# MAPO: Mixed Advantage Policy Optimization for Long-Horizon Multi-Turn Dialogue 

**Authors**: Naifan Zhang, Ruihan Sun, Jinwei Su, Hengjie Yang, Zhengyuan Pan, Zhaohan Chen, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06194)  

**Abstract**: Subjective multi-turn dialogue tasks, such as emotional support, require conversational policies that adapt to evolving user states and optimize long-horizon interaction quality. However, reinforcement learning (RL) for such settings remains challenging due to the absence of reliable process supervision. Outcome-only training collapses credit assignment across turns into a single trajectory-level reward, while naïve turn-level group sampling incurs prohibitive rollout costs in interactive environments. We propose a critic-free and efficient RL algorithm named MAPO that leverages dense process feedback from a judge model and propagates long-horizon effects through Monte Carlo returns. To stabilize optimization, we introduce a mixed advantage estimator that combines turn-level normalization with batch-level normalization, enabling fine-grained yet scalable credit assignment. Across multiple subjective dialogue benchmarks, including EMPA, EmoBench, and EQ-Bench, and model scales ranging from 7B to 32B, our method consistently improves both training stability and final performance over outcome-only GRPO and single-level normalization baselines. On EMPA, we improve rates by up to 9 points and increase dialogue scores by as much as +43.2 over the 7B base model. Despite training only on EMPA-style environments, our approach generalizes well, yielding consistent improvements on unseen emotional-intelligence benchmarks, including up to +4 points on EmoBench and +3.5 on EQ-Bench. Together, these results demonstrate that dense process supervision combined with mixed-level normalization enables effective and scalable RL for subjective, open-ended multi-turn dialogue. 

---
# ViewFusion: Structured Spatial Thinking Chains for Multi-View Reasoning 

**Authors**: Xingjian Tao, Yiwei Wang, Yujun Cai, Yifan Song, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.06024)  

**Abstract**: Multi-view spatial reasoning remains difficult for current vision-language models. Even when multiple viewpoints are available, models often underutilize cross-view relations and instead rely on single-image shortcuts, leading to fragile performance on viewpoint transformation and occlusion-sensitive cases. We present ViewFusion, a two-stage framework that explicitly separates cross-view spatial pre-alignment from question answering. In the first stage, the model performs deliberate spatial pre-thinking to infer viewpoint relations and spatial transformations across views, forming an intermediate workspace that goes beyond a simple re-description. In the second stage, the model conducts question-driven reasoning conditioned on this workspace to produce the final prediction. We train ViewFusion with synthetic reasoning supervision followed by reinforcement learning using GRPO, which improves answer correctness while stabilizing the intended two-stage generation behavior. On MMSI-Bench, ViewFusion improves accuracy by 5.3\% over Qwen3-VL-4B-Instruct, with the largest gains on examples that require genuine cross-view alignment. 

---
# ReflexiCoder: Teaching Large Language Models to Self-Reflect on Generated Code and Self-Correct It via Reinforcement Learning 

**Authors**: Juyong Jiang, Jiasi Shen, Sunghun Kim, Kang Min Yoo, Jeonghoon Kim, Sungju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.05863)  

**Abstract**: While Large Language Models (LLMs) have revolutionized code generation, standard "System 1" approaches, generating solutions in a single forward pass, often hit a performance ceiling when faced with complex algorithmic tasks. Existing iterative refinement strategies attempt to bridge this gap at inference time, yet they predominantly rely on external oracles, execution feedback, or computationally expensive prompt-response cycles. In this work, we propose ReflexiCoder, a novel reinforcement learning (RL) framework that internalizes the structured reasoning trajectory, encompassing initial generation, bug and optimization aware reflection, and self-correction, directly into the model's weights. Unlike prior methods, ReflexiCoder shifts the paradigm from external-dependent refinement to an intrinsic, fully autonomous self-reflection and self-correction capabilities at inference time. We utilize an RL-zero training paradigm with granular reward functions to optimize the entire reflection-correction trajectory, teaching the model how to debug without reliance on ground-truth feedback or execution engines at inference time. Extensive experiments across seven benchmarks demonstrate that our ReflexiCoder-8B establishes a new state-of-the-art (SOTA) among leading open-source models in the 1.5B-14B range, achieving 94.51% (87.20%) on HumanEval (Plus), 81.80% (78.57%) on MBPP (Plus), 35.00% on BigCodeBench, 52.21% on LiveCodeBench, and 37.34% on CodeForces in a single-attempt setting, rivaling or surpassing proprietary models like GPT-5.1. Notably, our framework is significantly more token-efficient than base models, reducing inference-time compute overhead by approximately 40% through disciplined, high-speed reasoning and reflection patterns. Source code is available at this https URL. 

---
# Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation 

**Authors**: Changcheng Li, Jiancan Wu, Hengheng Zhang, Zhengsu Chen, Guo An, Junxiang Qiu, Xiang Wang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.05881)  

**Abstract**: Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy.
We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications. 

---
# SAHOO: Safeguarded Alignment for High-Order Optimization Objectives in Recursive Self-Improvement 

**Authors**: Subramanyam Sahoo, Aman Chadha, Vinija Jain, Divya Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2603.06333)  

**Abstract**: Recursive self-improvement is moving from theory to practice: modern systems can critique, revise, and evaluate their own outputs, yet iterative self-modification risks subtle alignment drift. We introduce SAHOO, a practical framework to monitor and control drift through three safeguards: (i) the Goal Drift Index (GDI), a learned multi-signal detector combining semantic, lexical, structural, and distributional measures; (ii) constraint preservation checks that enforce safety-critical invariants such as syntactic correctness and non-hallucination; and (iii) regression-risk quantification to flag improvement cycles that undo prior gains. Across 189 tasks in code generation, mathematical reasoning, and truthfulness, SAHOO produces substantial quality gains, including 18.3 percent improvement in code tasks and 16.8 percent in reasoning, while preserving constraints in two domains and maintaining low violations in truthfulness. Thresholds are calibrated on a small validation set of 18 tasks across three cycles. We further map the capability-alignment frontier, showing efficient early improvement cycles but rising alignment costs later and exposing domain-specific tensions such as fluency versus factuality. SAHOO therefore makes alignment preservation during recursive self-improvement measurable, deployable, and systematically validated at scale. 

---
# From Entropy to Calibrated Uncertainty: Training Language Models to Reason About Uncertainty 

**Authors**: Azza Jenane, Nassim Walha, Lukas Kuhn, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2603.06317)  

**Abstract**: Large Language Models (LLMs) that can express interpretable and calibrated uncertainty are crucial in high-stakes domains. While methods to compute uncertainty post-hoc exist, they are often sampling-based and therefore computationally expensive or lack calibration. We propose a three-stage pipeline to post-train LLMs to efficiently infer calibrated uncertainty estimates for their responses. First, we compute fine-grained entropy-based uncertainty scores on the training data, capturing the distributional variability of model outputs in embedding space. Second, these scores are calibrated via Platt scaling, producing reliable and human-interpretable uncertainty signals. Finally, the target LLM is post-trained via reinforcement learning to align its policy with these calibrated signals through a verifiable reward function. Unlike post-hoc uncertainty estimation methods, our approach provides interpretable and computationally efficient uncertainty estimates at test time. Experiments show that models trained with our pipeline achieve better calibration than baselines and generalize to unseen tasks without further processing, suggesting that they learn a robust uncertainty reasoning behavior. 

---
# Place-it-R1: Unlocking Environment-aware Reasoning Potential of MLLM for Video Object Insertion 

**Authors**: Bohai Gu, Taiyi Wu, Dazhao Du, Jian Liu, Shuai Yang, Xiaotong Zhao, Alan Zhao, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.06140)  

**Abstract**: Modern video editing techniques have achieved high visual fidelity when inserting video objects. However, they focus on optimizing visual fidelity rather than physical causality, leading to edits that are physically inconsistent with their environment. In this work, we present Place-it-R$1$, an end-to-end framework for video object insertion that unlocks the environment-aware reasoning potential of Multimodal Large Language Models (MLLMs). Our framework leverages the Chain-of-Thought (CoT) reasoning of MLLMs to orchestrate video diffusion, following a Think-then-Place paradigm. To bridge cognitive reasoning and generative execution, we introduce three key innovations: First, MLLM performs physical scene understanding and interaction reasoning, generating environment-aware chain-of-thought tokens and inferring valid insertion regions to explicitly guide the diffusion toward physically plausible insertion. Then, we introduce MLLM-guided Spatial Direct Preference Optimization (DPO), where diffusion outputs are fed back to the MLLM for scoring, enabling visual naturalness. During inference, the MLLM iteratively triggers refinement cycles and elicits adaptive adjustments from the diffusion model, forming a closed-loop that progressively enhances editing quality. Furthermore, we provide two user-selectable modes: a plausibility-oriented flexible mode that permits environment modifications (\eg, generating support structures) to enhance physical plausibility, and a fidelity-oriented standard mode that preserves scene integrity for maximum fidelity, offering users explicit control over the plausibility-fidelity trade-off. Extensive experiments demonstrate Place-it-R1 achieves physically-coherent video object insertion compared with state-of-the-art solutions and commercial models. 

---
# CORE-Seg: Reasoning-Driven Segmentation for Complex Lesions via Reinforcement Learning 

**Authors**: Yuxin Xie, Yuming Chen, Yishan Yang, Yi Zhou, Tao Zhou, Zhen Zhao, Jiacheng Liu, Huazhu Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.05911)  

**Abstract**: Medical image segmentation is undergoing a paradigm shift from conventional visual pattern matching to cognitive reasoning analysis. Although Multimodal Large Language Models (MLLMs) have shown promise in integrating linguistic and visual knowledge, significant gaps remain: existing general MLLMs possess broad common sense but lack the specialized visual reasoning required for complex lesions, whereas traditional segmentation models excel at pixel-level segmentation but lack logical interpretability. In this paper, we introduce ComLesion-14K, the first diverse Chain-of-Thought (CoT) benchmark for reasoning-driven complex lesion segmentation. To accomplish this task, we propose CORE-Seg, an end-to-end framework integrating reasoning with segmentation through a Semantic-Guided Prompt Adapter. We design a progressive training strategy from SFT to GRPO, equipped with an adaptive dual-granularity reward mechanism to mitigate reward sparsity. Our Method achieves state-of-the-art results with a mean Dice of 37.06\% (14.89\% higher than the second-best baseline), while reducing the failure rate to 18.42\%. Project Page: this https URL 

---
# Reference-guided Policy Optimization for Molecular Optimization via LLM Reasoning 

**Authors**: Xuan Li, Zhanke Zhou, Zongze Li, Jiangchao Yao, Yu Rong, Lu Zhang, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.05900)  

**Abstract**: Large language models (LLMs) benefit substantially from supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) in reasoning tasks. However, these recipes perform poorly in instruction-based molecular optimization, where each data point typically provides only a single optimized reference molecule and no step-by-step optimization trajectory. We reveal that answer-only SFT on the reference molecules collapses reasoning, and RLVR provides sparse feedback under similarity constraints due to the model's lack of effective exploration, which slows learning and limits optimization. To encourage the exploration of new molecules while balancing the exploitation of the reference molecules, we introduce Reference-guided Policy Optimization (RePO), an optimization approach that learns from reference molecules without requiring trajectory data. At each update, RePO samples candidate molecules with their intermediate reasoning trajectories from the model and trains the model using verifiable rewards that measure property satisfaction under similarity constraints in an RL manner. Meanwhile, it applies reference guidance by keeping the policy's intermediate reasoning trajectory as context and training only the answer in a supervised manner. Together, the RL term promotes exploration, while the guidance term mitigates reward sparsity and stabilizes training by grounding outputs to references when many valid molecular edits exist. Across molecular optimization benchmarks, RePO consistently outperforms SFT and RLVR baselines (e.g., GRPO), achieving improvements on the optimization metric (Success Rate $\times$ Similarity), improving balance across competing objectives, and generalizing better to unseen instruction styles. Our code is publicly available at this https URL. 

---
