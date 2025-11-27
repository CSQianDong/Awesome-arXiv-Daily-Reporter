# ICPO: Intrinsic Confidence-Driven Group Relative Preference Optimization for Efficient Reinforcement Learning 

**Authors**: Jinpeng Wang, Chao Li, Ting Ye, Mengyuan Zhang, Wei Liu, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2511.21005)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates significant potential in enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing RLVR methods are often constrained by issues such as coarse-grained rewards, reward noise, and inefficient exploration, which lead to unstable training and entropy collapse. To address this challenge, we propose the Intrinsic Confidence-Driven Group Relative Preference Optimization method (ICPO). The intuition behind it lies in the fact that the probabilities of an LLM generating different responses can inherently and directly reflect its self-assessment of the reasoning process. Inspired by the idea of preference modeling, ICPO calculates a preference advantage score for each response by comparing the relative generation probabilities of multiple responses under the same input prompt, and integrates this score with verifiable rewards to guide the exploration process. We have discovered that the preference advantage score not only alleviates the issues of coarse-grained rewards and reward noise but also effectively curbs overconfident errors, enhances the relative superiority of undervalued high-quality responses, and prevents the model from overfitting to specific strategies, thereby facilitating more thorough exploration. Comprehensive experiments across four general-domain benchmarks and three mathematical benchmarks demonstrate that ICPO steadily boosts reasoning compared to GRPO. 

---
# Escaping the Verifier: Learning to Reason via Demonstrations 

**Authors**: Locke Cai, Ivan Provilkov  

**Link**: [PDF](https://arxiv.org/pdf/2511.21667)  

**Abstract**: Training Large Language Models (LLMs) to reason often relies on Reinforcement Learning (RL) with task-specific verifiers. However, many real-world reasoning-intensive tasks lack verifiers, despite offering abundant expert demonstrations that remain under-utilized for reasoning-focused training. We introduce RARO (Relativistic Adversarial Reasoning Optimization) that learns strong reasoning capabilities from only expert demonstrations via Inverse Reinforcement Learning. Our method sets up an adversarial interaction between a policy (generator) and a relativistic critic (discriminator): the policy learns to mimic expert answers, while the critic learns to compare and distinguish between policy and expert answers. Our method trains both the policy and the critic jointly and continuously via RL, and we identify the key stabilization techniques required for robust learning. Empirically, RARO significantly outperforms strong verifier-free baselines on all of our evaluation tasks -- Countdown, DeepMath, and Poetry Writing -- and enjoys the same robust scaling trends as RL on verifiable tasks. These results demonstrate that our method effectively elicits strong reasoning performance from expert demonstrations alone, enabling robust reasoning learning even when task-specific verifiers are unavailable. 

---
# ToolOrchestra: Elevating Intelligence via Efficient Model and Tool Orchestration 

**Authors**: Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter Belcak, Hanrong Ye, Hongxu Yin, Yi Dong, Evelina Bakhturina, Tao Yu, Yejin Choi, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.21689)  

**Abstract**: Large language models are powerful generalists, yet solving deep and complex problems such as those of the Humanity's Last Exam (HLE) remains both conceptually challenging and computationally expensive. We show that small orchestrators managing other models and a variety of tools can both push the upper bound of intelligence and improve efficiency in solving difficult agentic tasks. We introduce ToolOrchestra, a method for training small orchestrators that coordinate intelligent tools. ToolOrchestra explicitly uses reinforcement learning with outcome-, efficiency-, and user-preference-aware rewards. Using ToolOrchestra, we produce Orchestrator, an 8B model that achieves higher accuracy at lower cost than previous tool-use agents while aligning with user preferences on which tools are to be used for a given query. On HLE, Orchestrator achieves a score of 37.1%, outperforming GPT-5 (35.1%) while being 2.5x more efficient. On tau2-Bench and FRAMES, Orchestrator surpasses GPT-5 by a wide margin while using only about 30% of the cost. Extensive analysis shows that Orchestrator achieves the best trade-off between performance and cost under multiple metrics, and generalizes robustly to unseen tools. These results demonstrate that composing diverse tools with a lightweight orchestration model is both more efficient and more effective than existing methods, paving the way for practical and scalable tool-augmented reasoning systems. 

---
# Monet: Reasoning in Latent Visual Space Beyond Images and Language 

**Authors**: Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21395)  

**Abstract**: "Thinking with images" has emerged as an effective paradigm for advancing visual reasoning, extending beyond text-only chains of thought by injecting visual evidence into intermediate reasoning steps. However, existing methods fall short of human-like abstract visual thinking, as their flexibility is fundamentally limited by external tools. In this work, we introduce Monet, a training framework that enables multimodal large language models (MLLMs) to reason directly within the latent visual space by generating continuous embeddings that function as intermediate visual thoughts. We identify two core challenges in training MLLMs for latent visual reasoning: high computational cost in latent-vision alignment and insufficient supervision over latent embeddings, and address them with a three-stage distillation-based supervised fine-tuning (SFT) pipeline. We further reveal a limitation of applying GRPO to latent reasoning: it primarily enhances text-based reasoning rather than latent reasoning. To overcome this, we propose VLPO (Visual-latent Policy Optimization), a reinforcement learning method that explicitly incorporates latent embeddings into policy gradient updates. To support SFT, we construct Monet-SFT-125K, a high-quality text-image interleaved CoT dataset containing 125K real-world, chart, OCR, and geometry CoTs. Our model, Monet-7B, shows consistent gains across real-world perception and reasoning benchmarks and exhibits strong out-of-distribution generalization on challenging abstract visual reasoning tasks. We also empirically analyze the role of each training component and discuss our early unsuccessful attempts, providing insights for future developments in visual latent reasoning. Our model, data, and code are available at this https URL. 

---
# Self-Guided Defense: Adaptive Safety Alignment for Reasoning Models via Synthesized Guidelines 

**Authors**: Yuhang Wang, Yanxu Zhu, Dongyuan Lu, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21214)  

**Abstract**: Reasoning models have demonstrated remarkable capabilities in complex reasoning tasks. However, ensuring their safety against adversarial jailbreak prompts remains a critical challenge. Due to the covert and deceptive nature of such prompts, they can often evade built-in safety mechanisms and lead to the generation of harmful content. This underscores the need for an adaptive safety alignment approach that enables models to autonomously reinforce their defenses in response to adversarial inputs. This paper introduces the Synthesized Guideline-based Adaptive Safety Alignment (SGASA) framework, which internalizes model-generated safety guidelines to strengthen models' ability to enhance robustness against harmful adversarial prompts while minimizing unnecessary refusals of benign requests. SGASA consists of two key stages: Data Pre-synthesis, which generates safety guidelines and augmented prompts; and Alignment Fine-tuning, which leverages Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO) to embed these guidelines into the model. Extensive experiments across multiple datasets demonstrate that SGASA significantly improves model safety, validating its adaptive and scalable effectiveness. 

---
# Breaking the Safety-Capability Tradeoff: Reinforcement Learning with Verifiable Rewards Maintains Safety Guardrails in LLMs 

**Authors**: Dongkyu Derek Cho, Huan Song, Arijit Ghosh Chowdhury, Haotian An, Yawei Wang, Rohit Thekkanal, Negin Sokhandan, Sharlina Keshava, Hannah Marlowe  

**Link**: [PDF](https://arxiv.org/pdf/2511.21050)  

**Abstract**: Fine-tuning large language models (LLMs) for downstream tasks typically exhibit a fundamental safety-capability tradeoff, where improving task performance degrades safety alignment even on benign datasets. This degradation persists across standard approaches including supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF). While reinforcement learning with verifiable rewards (RLVR) has emerged as a promising alternative that optimizes models on objectively measurable tasks, its safety implications remain unexplored. We present the first comprehensive theoretical and empirical analysis of safety properties in RLVR. Theoretically, we derive upper bounds on safety drift under KL-constrained optimization and prove conditions under which safety degradation is eliminated. Empirically, we conduct extensive experiments across five adversarial safety benchmarks, demonstrating that RLVR can simultaneously enhance reasoning capabilities while maintaining or improving safety guardrails. Our comprehensive ablation studies examine the effects of optimization algorithms, model scale, and task domains. Our findings challenge the prevailing assumption of an inevitable safety capability trade-off, and establish that a specific training methodology can achieve both objectives simultaneously, providing insights for the safe deployment of reasoning-capable LLMs. 

---
# ST-PPO: Stabilized Off-Policy Proximal Policy Optimization for Multi-Turn Agents Training 

**Authors**: Chenliang Li, Adel Elmahdy, Alex Boyd, Zhongruo Wang, Alfredo Garcia, Parminder Bhatia, Taha Kass-Hout, Cao Xiao, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.20718)  

**Abstract**: PPO has been widely adopted for training large language models (LLMs) at the token level in multi-turn dialogue and reasoning tasks. However, its performance is often unstable and prone to collapse. Through empirical analysis, we identify two main sources of instability in this setting: (1)~token-level importance sampling, which is misaligned with the natural granularity of multi-turn environments that have distinct turn-level stages, and (2) inaccurate advantage estimates from off-policy samples, where the critic has not learned to evaluate certain state-action pairs, resulting in high-variance gradients and unstable updates. To address these challenges, we introduce two complementary stabilization techniques: (1) turn-level importance sampling, which aligns optimization with the natural structure of multi-turn reasoning, and (2) clipping-bias correction, which normalizes gradients by downweighting unreliable, highly off-policy samples. Depending on how these components are combined, we obtain three variants: Turn-PPO (turn-level sampling only), S-PPO (clipping-bias correction applied to token-level PPO), and ST-PPO (turn-level sampling combined with clipping-bias correction). In our experiments, we primarily study ST-PPO and S-PPO, which together demonstrate how the two stabilization mechanisms address complementary sources of instability. Experiments on multi-turn search tasks across general QA, multi-hop QA, and medical multiple-choice QA benchmarks show that ST-PPO and S-PPO consistently prevent the performance collapses observed in large-model training, maintain lower clipping ratios throughout optimization, and achieve higher task performance than standard token-level PPO. These results demonstrate that combining turn-level importance sampling with clipping-bias correction provides a practical and scalable solution for stabilizing multi-turn LLM agent training. 

---
# PIRA: Preference-Oriented Instruction-Tuned Reward Models with Dual Aggregation 

**Authors**: Yongfu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2511.20668)  

**Abstract**: Reward models are crucial for aligning Large Language Models (LLMs) with human preferences but face two representative challenges. First, traditional discriminative reward models usually concatenate questions and responses directly as input, resulting in low data efficiency. Second, reward models are vulnerable to reward overoptimization. We propose PIRA, a training paradigm addressing these issues through three strategies: (1) Reformulating question-answer pairs into preference-based instructions for clearer and more explicit task specification, (2) aggregating rewards from diverse preference tasks to reduce bias and improve robustness, and (3) averaging value-head outputs under varying dropout rates to stabilize rewards. Extensive experiments have demonstrated the effectiveness of PIRA. 

---
