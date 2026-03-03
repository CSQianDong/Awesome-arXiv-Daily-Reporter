# EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training 

**Authors**: Aleksei Dorkin, Taido Purason, Emil Kalbaliyev, Hele-Andra Kuulmets, Marii Ojastu, Mark Fišel, Tanel Alumäe, Eleri Aedmaa, Krister Kruusmaa, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2603.02041)  

**Abstract**: Large language models (LLMs) are predominantly trained on English-centric data, resulting in uneven performance for smaller languages. We study whether continued pretraining (CPT) can substantially improve Estonian capabilities in a pretrained multilingual LLM while preserving its English and general reasoning performance. Using Llama 3.1 8B as the main base model, we perform CPT on a mixture that increases Estonian exposure while approximating the original training distribution through English replay and the inclusion of code, mathematics, and instruction-like data. We subsequently apply supervised fine-tuning, preference optimization, and chat vector merging to introduce robust instruction-following behavior. Evaluation on a comprehensive suite of Estonian benchmarks shows consistent gains in linguistic competence, knowledge, reasoning, translation quality, and instruction-following compared to the original base model and its instruction-tuned variant, while maintaining competitive performance on English benchmarks. These findings indicate that CPT, with an appropriately balanced data mixture, together with post-training alignment, can substantially improve single-language capabilities in pretrained multilingual LLMs. 

---
# LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards 

**Authors**: Guanzheng Chen, Michael Qizhe Shieh, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2603.02146)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capabilities of Large Language Models (LLMs) by optimizing them against factual outcomes. However, this paradigm falters in long-context scenarios, as its reliance on internal parametric knowledge is ill-suited for tasks requiring contextual grounding--the ability to find and reason over externally provided information. We identify a key reason for this failure: a reward based solely on the final answer is too sparse to effectively guide the model for identifying relevant evidence. We formally prove that the outcome-only reward leads to significant vanishing gradients for the context grounding process, rendering learning intractable. To overcome this bottleneck, we introduce LongRLVR to augment the sparse answer reward with a dense and verifiable context reward. This auxiliary signal directly incentivizes the model for selecting the correct grounding information, providing a robust learning gradient that solves the underlying optimization challenge. We validate our method on challenging long-context benchmarks using Qwen and LLaMA models. LongRLVR consistently and significantly outperforms the standard RLVR across all models and benchmarks, e.g., boosting a 14B model's scores on RULER-QA from 73.17 to 88.90 and on LongBench v2 from 39.8 to 46.5. Our work demonstrates that explicitly rewarding the grounding process is a critical and effective strategy for unlocking the full reasoning potential of LLMs in long-context applications. Our code is available at this https URL. 

---
# CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production 

**Authors**: Yixin Nie, Lin Guan, Zhongyao Ma, Anchit Gupta, Yipin Zhou, Xiao Li, Zhengping Zhou, Raymond Zeng, Gelin Zhou, Shigan Chu, Ajay Thampi, Wancen Mu, Nathan Shuster, Ketong Wang, Lin Chen, Jason Brewer, Derek Hao Hu, Alexander McCauley, Jason Weston, Sem Park, Na Zhang, Kevin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01973)  

**Abstract**: This report presents CharacterFlywheel, an iterative flywheel process for improving large language models (LLMs) in production social chat applications across Instagram, WhatsApp, and Messenger. Starting from LLaMA 3.1, we refined models across 15 generations using data from both internal and external real-user traffic. Through continuous deployments from July 2024 to April 2025, we conducted controlled 7-day A/B tests showing consistent engagement improvements: 7 of 8 newly deployed models demonstrated positive lift over the baseline, with the strongest performers achieving up to 8.8% improvement in engagement breadth and 19.4% in engagement depth. We also observed substantial gains in steerability, with instruction following increasing from 59.2% to 84.8% and instruction violations decreasing from 26.6% to 5.8%. We detail the CharacterFlywheel process which integrates data curation, reward modeling to estimate and interpolate the landscape of engagement metrics, supervised fine-tuning (SFT), reinforcement learning (RL), and both offline and online evaluation to ensure reliable progress at each optimization step. We also discuss our methods for overfitting prevention and navigating production dynamics at scale. These contributions advance the scientific rigor and understanding of LLMs in social applications serving millions of users. 

---
# Surgical Post-Training: Cutting Errors, Keeping Knowledge 

**Authors**: Wenye Lin, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.01683)  

**Abstract**: Enhancing the reasoning capabilities of Large Language Models (LLMs) via post-training is often constrained by the trade-off between efficiency and catastrophic forgetting. While prior research emphasizes the role of on-policy data in mitigating forgetting, we uncover--and validate both theoretically and empirically--an overlooked yet critical mechanism: the implicit regularization inherent in Direct Preference Optimization's (DPO) reward estimate. This motivates our Surgical Post-Training (SPoT), a new paradigm designed to optimize reasoning efficiently while preserving learned prior knowledge. SPoT consists of: (1) a data rectification pipeline that employs an Oracle to surgically correct erroneous steps via minimal edits, generating data proximal to the model's distribution; and (2) a reward-based binary cross-entropy objective. Unlike the relative ranking in DPO, this objective treats reasoning correctness as a binary classification problem, enforcing decoupled supervision signals. Empirically, with only 4k rectified math data pairs, SPoT improves Qwen3-8B's accuracy by 6.2% on average across in-domain and OOD tasks, requiring merely 28 minutes of training on 8x H800 GPUs. Code: this https URL 

---
# Reasoning Boosts Opinion Alignment in LLMs 

**Authors**: Frédéric Berdoz, Yann Billeter, Yann Vonlanthen, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2603.01214)  

**Abstract**: Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment. 

---
# Can Thinking Models Think to Detect Hateful Memes? 

**Authors**: Mohamed Bayan Kmainasi, Mucahid Kutlu, Ali Ezzat Shahroor, Abul Hasnat, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2603.01225)  

**Abstract**: Hateful memes often require compositional multimodal reasoning: the image and text may appear benign in isolation, yet their interaction conveys harmful intent. Although thinking-based multimodal large language models (MLLMs) have recently advanced vision-language understanding, their capabilities remain underexplored for hateful meme analysis. We propose a reinforcement learning based post-training framework that improves reasoning in thinking-based MLLMs through task-specific rewards and a novel Group Relative Policy Optimization (GRPO) objective. Specifically, we (i) conduct a systematic empirical study of off-the-shelf MLLMs for hateful meme understanding, (ii) extend an existing hateful meme dataset by generating weakly or pseudo-supervised chain-of-thought rationales via distillation, and (iii) introduce a GRPO-based objective that jointly optimizes meme classification and explanation quality to encourage fine-grained, step-by-step reasoning. Experiments on the Hateful Memes benchmark show that our approach achieves state-of-the-art performance, improving accuracy and F1 by approximately 1 percent and explanation quality by approximately 3 percent. We will publicly release our code, dataset extensions, and evaluation resources to support reproducibility. 

---
# How RL Unlocks the Aha Moment in Geometric Interleaved Reasoning 

**Authors**: Xiangxiang Zhang, Caijun Jia, Siyuan Li, Dingyu He, Xiya Xiong, Zheng Sun, Honghao He, Yuchen Wu, Bihui Yu, Linzhuang Sun, Cheng Tan, Jingxuan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2603.01070)  

**Abstract**: Solving complex geometric problems inherently requires interleaved reasoning: a tight alternation between constructing diagrams and performing logical deductions. Although recent Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in visual generation and plotting, we identify a counter-intuitive and underexplored phenomenon. Naively applying Supervised Fine-Tuning (SFT) on interleaved plot-solution data leads to a substantial degradation in reasoning performance compared to text-only baselines. We argue that this failure stems from a fundamental limitation of SFT, which primarily induces distributional alignment: the model learns to reproduce the surface format of interleaved plotting but fails to internalize the causal dependency between the generated plot and reasoning steps. To overcome this limitation, we propose Faire (Functional alignment for interleaved reasoning), a reinforcement learning framework that enforces three casual constraints to move beyond superficial imitation toward functional alignment. Extensive experiments show that Faire induces a qualitative shift in model behavior in which the plotting is effectively internalized, yielding competitive performance on challenging geometric reasoning benchmarks. 

---
# Qwen3-Coder-Next Technical Report 

**Authors**: Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, Zeyao Ma, Kashun Shum, Xuwu Wang, Jinxi Wei, Jiaxi Yang, Jiajun Zhang, Lei Zhang, Zongmeng Zhang, Wenting Zhao, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.00729)  

**Abstract**: We present Qwen3-Coder-Next, an open-weight language model specialized for coding agents. Qwen3-Coder-Next is an 80-billion-parameter model that activates only 3 billion parameters during inference, enabling strong coding capability with efficient inference. In this work, we explore how far strong training recipes can push the capability limits of models with small parameter footprints. To achieve this, we perform agentic training through large-scale synthesis of verifiable coding tasks paired with executable environments, allowing learning directly from environment feedback via mid-training and reinforcement learning. Across agent-centric benchmarks including SWE-Bench and Terminal-Bench, Qwen3-Coder-Next achieves competitive performance relative to its active parameter count. We release both base and instruction-tuned open-weight versions to support research and real-world coding agent development. 

---
# RLAR: An Agentic Reward System for Multi-task Reinforcement Learning on Large Language Models 

**Authors**: Andrew Zhuoer Feng, Cunxiang Wang, Bosi Wen, Yidong Wang, Yu Luo, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.00724)  

**Abstract**: Large language model alignment via reinforcement learning depends critically on reward function quality. However, static, domain-specific reward models are often costly to train and exhibit poor generalization in out-of-distribution scenarios encountered during RL iterations. We present RLAR (Reinforcement Learning from Agent Rewards), an agent-driven framework that dynamically assigns tailored reward functions to individual queries. Specifically, RLAR transforms reward acquisition into a dynamic tool synthesis and invocation task. It leverages LLM agents to autonomously retrieve optimal reward models from the Internet and synthesize programmatic verifiers through code generation. This allows the reward system to self-evolve with the shifting data distributions during training. Experimental results demonstrate that RLAR yields consistent performance gains ranging from 10 to 60 across mathematics, coding, translation, and dialogue tasks. On RewardBench-V2, RLAR significantly outperforms static baselines and approaches the performance upper bound, demonstrating superior generalization through dynamic reward orchestration. The data and code are available on this link: this https URL. 

---
# TAB-PO: Preference Optimization with a Token-Level Adaptive Barrier for Token-Critical Structured Generation 

**Authors**: Samah Fodeh, Linhai Ma, Ganesh Puthiaraju, Srivani Talakokkul, Afshan Khan, Ashley Hagaman, Sarah R. Lowe, Aimee Kendall Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2603.00025)  

**Abstract**: Direct Preference Optimization is an offline post-SFT method for aligning language models from preference pairs, with strong results in instruction following and summarization. However, DPO's sequence-level implicit reward can be brittle for token-critical structured prediction settings such as medical annotation, which often exhibit (i) low-separation preference pairs, where chosen and rejected completions differ by minimal edit distance (often 1-3 tokens), and (ii) token-importance skew, where sparse semantic tokens (hierarchical labels and evidence Spans) carry disproportionate task importance relative to high-frequency structural tokens (JSON scaffolding). In this regime, standard DPO suffers from margin collapse (insufficient log-probability separation between near-identical preferences), likelihood squeezing (the margin objective shifts the absolute likelihoods of both completions together), and gradient dilution, where uniform sequence-level weighting diffuses learning signal across shared scaffolding while rare, confusable label tokens receive weak, noisy updates. We introduce Token-Adaptive Barrier Preference Optimization (TAB-PO), which augments DPO with token-weighted, reference-adjusted advantages that prioritize high-value semantic tokens, and a conditional token-level barrier that regularizes under-confident tokens balancing SFT-anchored likelihood and preference-driven separation in low-separation, importance-skewed regimes. We evaluate TAB-PO on medical communication annotation, a task requiring joint prediction of hierarchical labels and evidence Spans from patient-provider messages. TAB-PO achieves a ~ 4% relative improvement in micro-F1 over SFT and consistently outperforms recent preference-optimization baselines. 

---
# Learning from Synthetic Data Improves Multi-hop Reasoning 

**Authors**: Anmol Kabra, Yilun Yin, Albert Gong, Kamilė Stankevičiūtė, Dongyoung Go, Johann Lee, Katie Z. Luo, Carla P. Gomes, Kilian Q. Weinberger  

**Link**: [PDF](https://arxiv.org/pdf/2603.02091)  

**Abstract**: Reinforcement Learning (RL) has been shown to significantly boost reasoning capabilities of large language models (LLMs) in math, coding, and multi-hop reasoning tasks. However, RL fine-tuning requires abundant high-quality verifiable data, often sourced from human annotations, generated from frontier LLMs, or scored by LLM-based verifiers. All three have considerable limitations: human-annotated datasets are small and expensive to curate, LLM-generated data is hallucination-prone and costly, and LLM-based verifiers are inaccurate and slow. In this work, we investigate a cheaper alternative: RL fine-tuning on rule-generated synthetic data for multi-hop reasoning tasks. We discover that LLMs fine-tuned on synthetic data perform significantly better on popular real-world question-answering benchmarks, despite the synthetic data containing only fictional knowledge. On stratifying performance by question difficulty, we find that synthetic data teaches LLMs to compose knowledge -- a fundamental and generalizable reasoning skill. Our work highlights rule-generated synthetic reasoning data as a free and scalable resource to improve LLM reasoning capabilities. 

---
# Efficient RLVR Training via Weighted Mutual Information Data Selection 

**Authors**: Xinyu Zhou, Boyu Zhu, Haotian Zhang, Huiming Wang, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.01907)  

**Abstract**: Reinforcement learning (RL) plays a central role in improving the reasoning and alignment of large language models, yet its efficiency critically depends on how training data are selected. Existing online selection strategies predominantly rely on difficulty-based heuristics, favouring datapoints with intermediate success rates, implicitly equating difficulty with informativeness and neglecting epistemic uncertainty arising from limited evidence. We introduce InSight, an INformation-guided data SamplInG metHod for RL Training, grounded in a weighted mutual information objective. By modeling data outcomes with Bayesian latent success rates, we show that expected uncertainty reduction decomposes into complementary difficulty- and evidence-dependent components, revealing a fundamental limitation of difficulty-only selection. Leveraging this observation, InSight constructs a stable acquisition score based on the mean belief of datapoints' success rather than noisy sampled outcomes, and naturally extends to multi-rollout settings common in reinforcement learning with verifiable rewards (RLVR). Extensive experiments demonstrate that InSight consistently achieves state-of-the-art performance and improves training efficiency, including a +1.41 average gain on Planning & Mathmatics benchmarks, +1.01 improvement on general reasoning, and up to ~2.2x acceleration, with negligible additional computational overhead. 

---
# ProtRLSearch: A Multi-Round Multimodal Protein Search Agent with Large Language Models Trained via Reinforcement Learning 

**Authors**: Congying Liu, Taihao Li, Ming Huang, Xingyuan Wei, Peipei Liu, Yiqing Shen, Yanxu Mao, Tiehan Cui  

**Link**: [PDF](https://arxiv.org/pdf/2603.01464)  

**Abstract**: Protein analysis tasks arising in healthcare settings often require accurate reasoning under protein sequence constraints, involving tasks such as functional interpretation of disease-related variants, protein-level analysis for clinical research, and similar scenarios. To address such tasks, search agents are introduced to search protein-related information, providing support for disease-related variant analysis and protein function reasoning in protein-centric inference. However, such search agents are mostly limited to single-round, text-only modality search, which prevents the protein sequence modality from being incorporated as a multimodal input into the search decision-making process. Meanwhile, their reliance on reinforcement learning (RL) supervision that focuses solely on the final answer results in a lack of search process constraints, making deviations in keyword selection and reasoning directions difficult to identify and correct in a timely manner. To address these limitations, we propose ProtRLSearch, a multi-round protein search agent trained with multi-dimensional reward based RL, which jointly leverages protein sequence and text as multimodal inputs during real-time search to produce high quality reports. To evaluate the ability of models to integrate protein sequence information and text-based multimodal inputs in realistic protein query settings, we construct ProtMCQs, a benchmark of 3,000 multiple choice questions (MCQs) organized into three difficulty levels. The benchmark evaluates protein query tasks that range from sequence constrained reasoning about protein function and phenotype changes to comprehensive protein reasoning that integrates multi-dimensional sequence features with signal pathways and regulatory networks. 

---
# Learn Hard Problems During RL with Reference Guided Fine-tuning 

**Authors**: Yangzhen Wu, Shanda Li, Zixin Wen, Xin Zhou, Ameet Talwalkar, Yiming Yang, Wenhao Huang, Tianle Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.01223)  

**Abstract**: Reinforcement learning (RL) for mathematical reasoning can suffer from reward sparsity: for challenging problems, LLM fails to sample any correct trajectories, preventing RL from receiving meaningful positive feedback. At the same time, there often exist human-written reference solutions along with the problem (e.g., problems from AoPS), but directly fine-tuning on these solutions offers no benefit because models often cannot imitate human proofs that lie outside their own reasoning distribution.
We introduce Reference-Guided Fine-Tuning (ReGFT), a simple and effective method that utilizes human-written reference solutions to synthesize positive trajectories on hard problems and train on them before RL. For each problem, we provide the model with a partial reference solution and let it generate its own reasoning trace, ensuring the resulting trajectories remain in the model's reasoning space while still benefiting from reference guidance.
Fine-tuning on these reference-guided trajectories increases the number of solvable problems and produces a checkpoint that receives more positive rewards during RL. Across three benchmarks (AIME24, AIME25, BeyondAIME), ReGFT consistently improves supervised accuracy, accelerates DAPO training, and raises the final performance plateau of RL. Our results show that ReGFT effectively overcomes reward sparsity and unlocks stronger RL-based mathematical reasoning. 

---
# Pencil Puzzle Bench: A Benchmark for Multi-Step Verifiable Reasoning 

**Authors**: Justin Waugh  

**Link**: [PDF](https://arxiv.org/pdf/2603.02119)  

**Abstract**: We introduce Pencil Puzzle Bench, a framework for evaluating large language model reasoning through pencil puzzles, a family of constraint-satisfaction problems closely related to NP-complete problems, with deterministic, step-level verification. From a database of 62,231 puzzles across 94 varieties with verified unique solutions, we select a benchmark of 300 puzzles spanning 20 varieties and evaluate 51 models from 11 providers in two modes: direct ask (single-shot) and agentic (multi-turn with iterative verification). A key differentiator of our benchmark is that every intermediate board state can be checked against variety-specific constraints, localizing errors to the exact rule violated, providing the infrastructure for dense, per-move reward signals for process supervision and reinforcement learning.
Our evaluation reveals two distinct axes of capability: (1) reasoning effort scaling, where GPT-5.2 improves 81x from no reasoning to maximum effort; and (2) agentic iteration, where Claude Opus 4.6 rises from 0.3% to 30.0% through iterative checking, while GPT-5.2@xhigh improves from 20.2% to 56.0%. Agentic attempts span a median of 29 turns over 17 minutes, with the longest exceeding 1,221 turns and 14.3 hours - a demanding test of long-context utilization, not just reasoning. 

---
# ToolRLA: Fine-Grained Reward Decomposition for Tool-Integrated Reinforcement Learning Alignment in Domain-Specific Agents 

**Authors**: Pengbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01620)  

**Abstract**: Tool-integrated reasoning agents interleaving natural language deliberation with external API calls show promise for complex multi-step tasks. However, aligning such agents for high-stakes domain-specific deployment is challenging, as existing reinforcement learning uses coarse binary rewards (success/failure) that insufficiently guide nuanced tool invocation in production. We present ToolRLA, a three-stage post-training pipeline (Supervised Fine-Tuning, Group Relative Policy Optimization, Direct Preference Optimization) for domain-specific tool-integrated agents. Its core is a fine-grained reward function with multiplicative correctness decomposition, evaluating tool invocation across four dimensions: format validity, tool selection correctness, invocation efficiency, and domain constraint compliance. Multiplicative composition prioritizes correct tool selection (a prerequisite for meaningful parameter evaluation), while a large negative compliance penalty ({\lambda}=10) ensures regulatory adherence. Deployed on a real-world financial advisory copilot (80+ advisors, 1,200+ daily queries, 15+ heterogeneous APIs), ToolRLA achieves 47% higher end-to-end task completion (62% to 91%), 63% lower tool invocation error (38% to 14%), 93% lower regulatory violation (12% to 0.8%), and sub-2-second latency after three months. Ablation studies confirm fine-grained reward decomposition contributes 7 percentage points over coarse additive rewards; generalizability is validated on ToolBench and API-Bank. 

---
# CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework 

**Authors**: Yuexi Du, Jinglu Wang, Shujie Liu, Nicha C. Dvornek, Yan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.01607)  

**Abstract**: Large visual language models (VLMs) have shown strong multi-modal medical reasoning ability, but most operate as end-to-end black boxes, diverging from clinicians' evidence-based, staged workflows and hindering clinical accountability. Complementarily, expert visual grounding models can accurately localize regions of interest (ROIs), providing explicit, reliable evidence that improves both reasoning accuracy and trust. In this paper, we introduce CARE, advancing Clinical Accountability in multi-modal medical Reasoning with an Evidence-grounded agentic framework. Unlike existing approaches that couple grounding and reasoning within a single generalist model, CARE decomposes the task into coordinated sub-modules to reduce shortcut learning and hallucination: a compact VLM proposes relevant medical entities; an expert entity-referring segmentation model produces pixel-level ROI evidence; and a grounded VLM reasons over the full image augmented by ROI hints. The VLMs are optimized with reinforcement learning with verifiable rewards to align answers with supporting evidence. Furthermore, a VLM coordinator plans tool invocation and reviews evidence-answer consistency, providing agentic control and final verification. Evaluated on standard medical VQA benchmarks, our CARE-Flow (coordinator-free) improves average accuracy by 10.9% over the same size (10B) state-of-the-art (SOTA). With dynamic planning and answer review, our CARE-Coord yields a further gain, outperforming the heavily pre-trained SOTA by 5.2%. Our experiments demonstrate that an agentic framework that emulates clinical workflows, incorporating decoupled specialized models and explicit evidence, yields more accurate and accountable medical AI. 

---
# Learning Structured Reasoning via Tractable Trajectory Control 

**Authors**: Po-Nien Kung, Zhen Yang, Jeffrey Luo, Cheng-Fu Yang, Haikang Deng, Zi-Yi Dou, Yinfei Yang, Nanyun Peng, Zhe Gan, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01641)  

**Abstract**: Large language models can exhibit emergent reasoning behaviors, often manifested as recurring lexical patterns (e.g., "wait," indicating verification). However, complex reasoning trajectories remain sparse in unconstrained sampling, and standard RL often fails to guarantee the acquisition of diverse reasoning behaviors. We propose a systematic discovery and reinforcement of diverse reasoning patterns through structured reasoning, a paradigm that requires targeted exploration of specific reasoning patterns during the RL process. To this end, we propose Ctrl-R, a framework for learning structured reasoning via tractable trajectory control that actively guides the rollout process, incentivizing the exploration of diverse reasoning patterns that are critical for complex problem-solving. The resulting behavior policy enables accurate importance-sampling estimation, supporting unbiased on-policy optimization. We further introduce a power-scaling factor on the importance-sampling weights, allowing the policy to selectively learn from exploratory, out-of-distribution trajectories while maintaining stable optimization. Experiments demonstrate that Ctrl-R enables effective exploration and internalization of previously unattainable reasoning patterns, yielding consistent improvements across language and vision-language models on mathematical reasoning tasks. 

---
# RubricBench: Aligning Model-Generated Rubrics with Human Standards 

**Authors**: Qiyuan Zhang, Junyi Zhou, Yufei Wang, Fuyuan Lyu, Yidong Ming, Can Xu, Qingfeng Sun, Kai Zheng, Peng Kang, Xue Liu, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2603.01562)  

**Abstract**: As Large Language Model (LLM) alignment evolves from simple completions to complex, highly sophisticated generation, Reward Models are increasingly shifting toward rubric-guided evaluation to mitigate surface-level biases. However, the community lacks a unified benchmark to assess this evaluation paradigm, as existing benchmarks lack both the discriminative complexity and the ground-truth rubric annotations required for rigorous analysis. To bridge this gap, we introduce RubricBench, a curated benchmark with 1,147 pairwise comparisons specifically designed to assess the reliability of rubric-based evaluation. Our construction employs a multi-dimensional filtration pipeline to target hard samples featuring nuanced input complexity and misleading surface bias, augmenting each with expert-annotated, atomic rubrics derived strictly from instructions. Comprehensive experiments reveal a substantial capability gap between human-annotated and model-generated rubrics, indicating that even state-of-the-art models struggle to autonomously specify valid evaluation criteria, lagging considerably behind human-guided performance. 

---
# DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent 

**Authors**: Tongzhou Wu, Yuhao Wang, Xinyu Ma, Xiuqiang He, Shuaiqiang Wang, Dawei Yin, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.01152)  

**Abstract**: Deep-research agents are capable of executing multi-step web exploration, targeted retrieval, and sophisticated question answering. Despite their powerful capabilities, deep-research agents face two critical bottlenecks: (1) the lack of large-scale, challenging datasets with real-world difficulty, and (2) the absence of accessible, open-source frameworks for data synthesis and agent training. To bridge these gaps, we first construct DeepResearch-9K, a large-scale challenging dataset specifically designed for deep-research scenarios built from open-source multi-hop question-answering (QA) datasets via a low-cost autonomous pipeline. Notably, it consists of (1) 9000 questions spanning three difficulty levels from L1 to L3 (2) high-quality search trajectories with reasoning chains from Tongyi-DeepResearch-30B-A3B, a state-of-the-art deep-research agent, and (3) verifiable answers. Furthermore, we develop an open-source training framework DeepResearch-R1 that supports (1) multi-turn web interactions, (2) different reinforcement learning (RL) approaches, and (3) different reward models such as rule-based outcome reward and LLM-as-judge feedback. Finally, empirical results demonstrate that agents trained on DeepResearch-9K under our DeepResearch-R1 achieve state-of-the-art results on challenging deep-research benchmarks. We release the DeepResearch-9K dataset on this https URL and the code of DeepResearch-R1 on this https URL. 

---
# DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage 

**Authors**: Haowen Gao, Zhenyu Zhang, Liang Pang, Fangda Guo, Hongjian Dou, Guannan Lv, Shaoguo Liu, Tingting Gao, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.01106)  

**Abstract**: Reinforcement learning (RL) with group relative policy optimization (GRPO) has become a widely adopted approach for enhancing the reasoning capabilities of multimodal large language models (MLLMs). While GRPO enables long-chain reasoning without a critic, it often suffers from sparse rewards on difficult problems and advantage vanishing when group-level rewards are too consistent for overly easy or hard problems. Existing solutions (sample expansion, selective utilization, and indirect reward design) often fail to maintain enough variance in within-group reward distributions to yield clear optimization signals. To address this, we propose DIVA-GRPO, a difficulty-adaptive variant advantage method that adjusts variant difficulty distributions from a global perspective. DIVA-GRPO dynamically assesses problem difficulty, samples variants with appropriate difficulty levels, and calculates advantages across local and global groups using difficulty-weighted and normalized scaling. This alleviates reward sparsity and advantage vanishing while improving training stability. Extensive experiments on six reasoning benchmarks demonstrate that DIVA-GRPO outperforms existing approaches in training efficiency and reasoning performance. Code: this https URL 

---
# Co-Evolutionary Multi-Modal Alignment via Structured Adversarial Evolution 

**Authors**: Guoxin Shi, Haoyu Wang, Zaihui Yang, Yuxing Wang, Yongzhe Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01784)  

**Abstract**: Adversarial behavior plays a central role in aligning large language models with human values. However, existing alignment methods largely rely on static adversarial settings, which fundamentally limit robustness, particularly in multimodal settings with a larger attack surface. In this work, we move beyond static adversarial supervision and introduce co-evolutionary alignment with evolving attacks, instantiated by CEMMA (Co-Evolutionary Multi-Modal Alignment), an automated and adaptive framework for multimodal safety alignment. We introduce an Evolutionary Attacker that decomposes adversarial prompts into method templates and harmful intents. By employing genetic operators, including mutation, crossover, and differential evolution, it enables simple seed attacks to inherit the structural efficacy of sophisticated jailbreaks. The Adaptive Defender is iteratively updated on the synthesized hard negatives, forming a closed-loop process that adapts alignment to evolving attacks. Experiments show that the Evolutionary Attacker substantially increases red-teaming jailbreak attack success rate (ASR), while the Adaptive Defender improves robustness and generalization across benchmarks with higher data efficiency, without inducing excessive benign refusal, and remains compatible with inference-time defenses such as AdaShield. 

---
# Provable and Practical In-Context Policy Optimization for Self-Improvement 

**Authors**: Tianrun Yu, Yuxiao Yang, Zhaoyang Wang, Kaixiang Zhao, Porter Jenkins, Xuchao Zhang, Chetan Bansal, Huaxiu Yao, Weitong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.01335)  

**Abstract**: We study test-time scaling, where a model improves its answer through multi-round self-reflection at inference. We introduce In-Context Policy Optimization (ICPO), in which an agent optimizes its response in context using self-assessed or externally observed rewards without modifying its parameters. To explain this ICPO process, we theoretically show that with sufficient pretraining under a novel Fisher-weighted logit-matching objective, a single-layer linear self-attention model can provably imitate policy-optimization algorithm for linear bandits. Building on this theory, we propose Minimum-Entropy ICPO (ME-ICPO), a practical algorithm that iteratively uses its response and self-assessed reward to refine its response in-context at inference time. By selecting the responses and their rewards with minimum entropy, ME-ICPO ensures the robustness of the self-assessed rewards via majority voting. Across standard mathematical reasoning tasks, ME-ICPO attains competitive, top-tier performance while keeping inference costs affordable compared with other inference-time algorithms. Overall, ICPO provides a principled understanding of self-reflection in LLMs and yields practical benefits for test-time scaling for mathematical reasoning. 

---
# ClinCoT: Clinical-Aware Visual Chain-of-Thought for Medical Vision Language Models 

**Authors**: Xiwei Liu, Yulong Li, Xinlin Zhuang, Xuhui Li, Jianxu Chen, Haolin Yang, Imran Razzak, Yutong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.01124)  

**Abstract**: Medical Vision-Language Models have shown promising potential in clinical decision support, yet they remain prone to factual hallucinations due to insufficient grounding in localized pathological evidence. Existing medical alignment methods primarily operate at the response level through preference optimization, improving output correctness but leaving intermediate reasoning weakly connected to visual regions. Although chain-of-thought (CoT) enhances multimodal reasoning, it remains largely text-centric, limiting effective integration of clinical visual cues. To address this gap, we propose ClinCoT, a clinical-aware visual chain-of-thought framework that transforms preference optimization from response-level correction to visual-driven reasoning. We introduce an automatic data generation pipeline that constructs clinically grounded preference pairs through reasoning with hypotheses-driven region proposals. Multiple Med-LLMs evaluators rank and assign scores to each response, and these rankings serve as supervision to train the target model. We further introduce a scoring-based margin-aware optimization strategy that incorporates both preference ranking and score difference to refine region-level reasoning trajectories. To maintain alignment as the model's policy evolves during training, we adopt an iterative learning scheme that dynamically regenerates preference data. Extensive experiments on three medical VQA and report generation benchmarks demonstrate that ClinCoT consistently improves factual grounding and achieves superior performance compared with existing preference-based alignment methods. 

---
# Seeing Beyond 8bits: Subjective and Objective Quality Assessment of HDR-UGC Videos 

**Authors**: Shreshth Saini, Bowen Chen, Neil Birkbeck, Yilin Wang, Balu Adsumilli, Alan C. Bovik  

**Link**: [PDF](https://arxiv.org/pdf/2603.00938)  

**Abstract**: High Dynamic Range (HDR) user-generated (UGC) videos are rapidly proliferating across social platforms, yet most perceptual video quality assessment (VQA) systems remain tailored to Standard Dynamic Range (SDR). HDR has a higher bit depth, wide color gamut, and elevated luminance range, exposing distortions such as near-black crushing, highlight clipping, banding, and exposure flicker that amplify UGC artifacts and challenge SDR models. To catalyze progress, we curate Beyond8Bits, a large-scale subjective dataset of 44K videos from 6.5K sources with over 1.5M crowd ratings, spanning diverse scenes, capture conditions, and compression settings. We further introduce HDR-Q, the first Multimodal Large Language Model (MLLM) for HDR-UGC VQA. We propose (i) a novel HDR-aware vision encoder to produce HDR-sensitive embeddings, and (ii) HDR-Aware Policy Optimization (HAPO), an RL finetuning framework that anchors reasoning to HDR cues. HAPO augments GRPO via an HDR-SDR contrastive KL that encourages token reliance on HDR inputs and a Gaussian weighted regression reward for fine-grained MOS calibration. Across Beyond8Bits and public HDR-VQA benchmarks, HDR-Q delivers state-of-the-art performance. 

---
