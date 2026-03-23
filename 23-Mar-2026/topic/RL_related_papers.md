# Experience is the Best Teacher: Motivating Effective Exploration in Reinforcement Learning for LLMs 

**Authors**: Wenjian Zhang, Kongcheng Zhang, Jiaxin Qi, Baisheng Lai, Jianqiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20046)  

**Abstract**: Reinforcement Learning (RL) with rubric-based rewards has recently shown remarkable progress in enhancing general reasoning capabilities of Large Language Models (LLMs), yet still suffers from ineffective exploration confined to curent policy distribution. In fact, RL optimization can be viewed as steering the policy toward an ideal distribution that maximizes the rewards, while effective exploration should align efforts with desired target. Leveraging this insight, we propose HeRL, a Hindsight experience guided Reinforcement Learning framework to bootstrap effective exploration by explicitly telling LLMs the desired behaviors specified in rewards. Concretely, HeRL treats failed trajectories along with their unmet rubrics as hindsight experience, which serves as in-context guidance for the policy to explore desired responses beyond its current distribution. Additionally, we introduce a bonus reward to incentivize responses with greater potential for improvement under such guidance. HeRL facilitates effective learning from desired high quality samples without repeated trial-and-error from scratch, yielding a more accurate estimation of the expected gradient theoretically. Extensive experiments across various benchmarks demonstrate that HeRL achieves superior performance gains over baselines, and can further benefit from experience guided self-improvement at test time. Our code is available at this https URL. 

---
# A Subgoal-driven Framework for Improving Long-Horizon LLM Agents 

**Authors**: Taiyi Wang, Sian Gooding, Florian Hartmann, Oriana Riva, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2603.19685)  

**Abstract**: Large language model (LLM)-based agents have emerged as powerful autonomous controllers for digital environments, including mobile interfaces, operating systems, and web browsers. Web navigation, for example, requires handling dynamic content and long sequences of actions, making it particularly challenging. Existing LLM-based agents struggle with long-horizon planning in two main ways. During online execution, they often lose track as new information arrives, lacking a clear and adaptive path toward the final goal. This issue is further exacerbated during reinforcement learning (RL) fine-tuning, where sparse and delayed rewards make it difficult for agents to identify which actions lead to success, preventing them from maintaining coherent reasoning over extended tasks. To address these challenges, we propose two contributions. First, we introduce an agent framework that leverages proprietary models for online planning through subgoal decomposition. Second, we present MiRA (Milestoning your Reinforcement Learning Enhanced Agent), an RL training framework that uses dense, milestone-based reward signals. The real-time planning mechanism improves proprietary models such as Gemini by approximately a 10% absolute increase in success rate (SR) on the WebArena-Lite benchmark. Meanwhile, applying MiRA to the open Gemma3-12B model increases its success rate from 6.4% to 43.0%. This performance surpasses proprietary systems such as GPT-4-Turbo (17.6%) and GPT-4o (13.9%), as well as the previous open-model state of the art, WebRL (38.4%). Overall, our findings demonstrate that combining explicit inference-time planning with milestone-based rewards significantly improves an agent's long-horizon capabilities, paving the way for more robust and general-purpose autonomous systems. 

---
# Chain-of-Adaptation: Surgical Vision-Language Adaptation with Reinforcement Learning 

**Authors**: Jiajie Li, Chenhui Xu, Meihuan Liu, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.20116)  

**Abstract**: Conventional fine-tuning on domain-specific datasets can inadvertently alter a model's pretrained multimodal priors, leading to reduced generalization. To address this, we propose Chain-of-Adaptation (CoA), an adaptation framework designed to integrate domain knowledge while maintaining the model's inherent reasoning and perceptual capabilities. CoA introduces a structured reasoning format that enhances domain alignment without sacrificing general multimodal competence by reinforcement learning. Experiments on standard surgical benchmarks, under both in-distribution and out-of-distribution settings, demonstrate that CoA achieves higher accuracy, stronger generalization, and more stable behavior than supervised fine-tuning. Furthermore, ablation studies confirm that CoA effectively preserves the model's core visual-language abilities, providing a reliable pathway for domain specialization in VLMs. 

---
# An Empirical Study of SFT-DPO Interaction and Parameterization in Small Language Models 

**Authors**: Yuming Feng, Christy Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20100)  

**Abstract**: Direct Preference Optimization (DPO) is widely used after supervised fine-tuning (SFT) to align language models, yet empirical behavior under small backbones and modest data is under-specified. We systematically compare SFT-only, DPO-only, and staged SFT-to-DPO training alongside full fine-tuning (FFT) versus LoRA on a GPT-2-scale decoder, evaluating paraphrase detection and Shakespearean sonnet continuation. DPO yields small, task-dependent gains over strong SFT and can match competitive SFT accuracy without a warm start when the preference construction closely parallels the supervised objective. In contrast, parameterization dominates: FFT consistently outperforms LoRA at matched training depth, and LoRA does not reduce wall-clock time on our hardware. These findings indicate that, in this small-scale regime, supervised full-parameter adaptation remains the primary performance lever, while preference optimization and low-rank adaptation provide limited marginal returns. 

---
# What If Consensus Lies? Selective-Complementary Reinforcement Learning at Test Time 

**Authors**: Dong Yan, Jian Liang, Yanbo Wang, Shuo Lu, Ran He, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2603.19880)  

**Abstract**: Test-Time Reinforcement Learning (TTRL) enables Large Language Models (LLMs) to enhance reasoning capabilities on unlabeled test streams by deriving pseudo-rewards from majority voting consensus. However, existing TTRL methods rely exclusively on positive pseudo-labeling strategies. Such reliance becomes vulnerable under challenging scenarios where answer distributions are highly dispersed, resulting in weak consensus that inadvertently reinforces incorrect trajectories as supervision signals. In this paper, we propose SCRL (Selective-Complementary Reinforcement Learning), a robust test-time reinforcement learning framework that effectively mitigates label noise amplification. SCRL develops Selective Positive Pseudo-Labeling, which enforces strict consensus criteria to filter unreliable majorities. Complementarily, SCRL introduces Entropy-Gated Negative Pseudo-Labeling, the first negative supervision mechanism in TTRL, to reliably prune incorrect trajectories based on generation uncertainty. Extensive experiments on multiple reasoning benchmarks demonstrate that SCRL achieves substantial improvements over baselines, while maintaining robust generalization and training stability under constrained rollout budgets. Our code is available at this https URL. 

---
# PolicySim: An LLM-Based Agent Social Simulation Sandbox for Proactive Policy Optimization 

**Authors**: Renhong Huang, Ning Tang, Jiarong Xu, Yuxuan Cao, Qingqian Tu, Sheng Guo, Bo Zheng, Huiyuan Liu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19649)  

**Abstract**: Social platforms serve as central hubs for information exchange, where user behaviors and platform interventions jointly shape opinions. However, intervention policies like recommendation and content filtering, can unintentionally amplify echo chambers and polarization, posing significant societal risks. Proactively evaluating the impact of such policies is therefore crucial. Existing approaches primarily rely on reactive online A/B testing, where risks are identified only after deployment, making risk identification delayed and costly. LLM-based social simulations offer a promising pre-deployment alternative, but current methods fall short in realistically modeling platform interventions and incorporating feedback from the platform. Bridging these gaps is essential for building actionable frameworks to assess and optimize platform policies. To this end, we propose PolicySim, an LLM-based social simulation sandbox for the proactive assessment and optimization of intervention policies. PolicySim models the bidirectional dynamics between user behavior and platform interventions through two key components: (1) a user agent module refined via supervised fine-tuning (SFT) and direct preference optimization (DPO) to achieve platform-specific behavioral realism; and (2) an adaptive intervention module that employs a contextual bandit with message passing to capture dynamic network structures. Experiments show that PolicySim can accurately simulate platform ecosystems at both micro and macro levels and support effective intervention policy. 

---
# Adaptive Layerwise Perturbation: Unifying Off-Policy Corrections for LLM RL 

**Authors**: Chenlu Ye, Xuanchang Zhang, Yifan Hao, Zhou Yu, Ziji Zhang, Abhinav Gullapalli, Hao Chen, Jing Huang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19470)  

**Abstract**: Off-policy problems such as policy staleness and training-inference mismatch, has become a major bottleneck for training stability and further exploration for LLM RL. To enhance inference efficiency, the distribution gap between the inference and updated policy grows, leading to heavy-tailed importance ratios. Heavy-tailed ratios arise when the policy is locally sharp, which further inflates sharp gradients and can push updates outside the trust region. To address this, we propose Adaptive Layerwise Perturbation(ALP) by injecting small learnable perturbations into input hidden states of each layer during updates, which is used as the numerator of the importance ratio against the unchanged inference policy in the objective. Intuitively, by adding controlled noise to intermediate representations, ALP prevents the updated policy from deviating too sharply from the inference policy, and enlarges the policy family to cover the inference policy family with mismatch noises. Hence, the flattened distribution can naturally tighten the updated and inference policy gap and reduce the tail of importance ratios, thus maintaining training stability. This is further validated empirically. Experiments on single-turn math and multi-turn tool-integrated reasoning tasks show that ALP not only improves final performance, but also avoid blow up of importance ratio tail and KL spikes during iterative training, along with boosted exploration. Ablations show that representation-level perturbations across all layers are most effective, substantially outperforming partial-layer and logits-only variants. 

---
# Goedel-Code-Prover: Hierarchical Proof Search for Open State-of-the-Art Code Verification 

**Authors**: Zenan Li, Ziran Yang, Deyuan, Haoyu Zhao, Andrew Zhao, Shange Tang, Kaiyu Yang, Aarti Gupta, Zhendong Su, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2603.19329)  

**Abstract**: Large language models (LLMs) can generate plausible code but offer limited guarantees of correctness. Formally verifying that implementations satisfy specifications requires constructing machine-checkable proofs, a task that remains beyond current automation. We propose a hierarchical proof search framework for automated code verification in Lean~4 that decomposes complex verification goals into structurally simpler subgoals before attempting tactic-level proving. Central to our approach is a principled decomposition score that combines constructive justification with structural effectiveness. Crucially, this score serves as both the training reward and the inference-time ranking criterion, ensuring strict alignment between optimization and deployment. We train Goedel-Code-Prover-8B, a single unified policy for both decomposition and completion, via supervised initialization followed by hybrid reinforcement learning, where a continuous decomposition reward drives planning exploration while supervised replay stabilizes proof generation. On three Lean-based code verification benchmarks comprising 427 tasks, our 8B-parameter model achieves a 62.0\% prove success rate, a 2.6$\times$ improvement over the strongest baseline, surpassing neural provers up to 84$\times$ larger. We further observe consistent inference-time scaling: success rates improve monotonically with search iterations and sampling budget, with our trained model achieving greater efficiency than frontier off-the-shelf models of comparable scale. 

---
# MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels 

**Authors**: Tianyang Luo, Tao Feng, Zhigang Hua, Yan Xie, Shuang Yang, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2603.19310)  

**Abstract**: Training large language models (LLMs) for complex reasoning via reinforcement learning requires reward labels that specify whether the generated rollouts are correct. However, obtaining reward labels at scale often requires expensive human labeling or time-consuming verification procedures; for instance, evaluating mathematical proofs demands expert review, while open-ended question answering lacks definitive ground truth. When reward labels are limited, the effectiveness of reinforcement learning fine-tuning is constrained by the scarcity of reward labels. We introduce MemReward, a graph-based experience memory framework: an initial LLM policy generates rollouts for each query, each comprising a thinking process and a final answer, and these rollouts are stored as experience memory. Queries, thinking processes, and answers form nodes in a heterogeneous graph with similarity and structural edges; a GNN trained on labeled nodes propagates rewards to unlabeled rollouts during online optimization. Experiments on Qwen2.5-3B and 1.5B across mathematics, question answering, and code generation demonstrate that MemReward, with only 20% labels, achieves 97.3% of Oracle performance on 3B and 96.6% on 1.5B, surpassing Oracle on out-of-domain tasks. Performance scales smoothly with label budget, reaching 99.4% of Oracle at 70% labels. 

---
# Probing to Refine: Reinforcement Distillation of LLMs via Explanatory Inversion 

**Authors**: Zhen Tan, Chengshuai Zhao, Song Wang, Jundong Li, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19266)  

**Abstract**: Distilling robust reasoning capabilities from large language models (LLMs) into smaller, computationally efficient student models remains an unresolved challenge. Despite recent advances, distilled models frequently suffer from superficial pattern memorization and subpar generalization. To overcome these limitations, we introduce a novel distillation framework that moves beyond simple mimicry to instill a deeper conceptual understanding. Our framework features two key innovations. \underline{\textit{First}}, to address pattern memorization, Explanatory Inversion (EI) generates targeted ``explanatory probes'' that compel the student to articulate the underlying logic behind an answer, rather than just memorizing it. \underline{\textit{Second}}, to improve generalization, Explanatory GRPO (\texttt{EXGRPO}) uses a reinforcement learning algorithm with a novel Dialogue Structure Utility Bonus, which explicitly rewards the student for maintaining a coherent reasoning process across these probes. Extensive evaluations on 12 datasets demonstrate significant improvements. Using Gemma-7b as the student model, our method yields an average \textbf{20.39\%} increase over zero-shot performance and a \textbf{6.02\%} improvement over the state-of-the-art distillation baselines. Moreover, models distilled with our method show remarkable training efficiency (e.g., surpassing vanilla fine-tuning with \textbf{10-25\%} training data) and strong generalization to out-of-distribution tasks. Implementation is released at this https URL. 

---
# LARFT: Closing the Cognition-Action Gap for Length Instruction Following in Large Language Models 

**Authors**: Wei Zhang, Lintong Du, Yuanhe Zhang, Zhenhong Zhou, Kun Wang, Li Sun, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.19255)  

**Abstract**: Despite the strong performance of Large Language Models (LLMs) on complex instruction-following tasks, precise control of output length remains a persistent challenge. Existing methods primarily attempt to enforce length constraints by externally imposing length signals or optimization objectives, while largely overlooking the underlying limitation: the model's intrinsic deficit in length cognition. To address this, we propose LARFT (Length-Aware Reinforcement Fine-Tuning), a training framework that aligns the model's length cognition with its action. Specifically, LARFT integrates length-oriented reinforcement learning with a hindsight length awareness. By transforming on-policy data into hindsight self-awareness tasks where the model learns to identify the actual length of its own generation, LARFT jointly optimizes the model's internal representation of length information and refines its policy to satisfy length constraints, thereby achieving precise and reliable length instruction following. Extensive experiments across four base models demonstrate that LARFT outperforms existing baselines, achieving an average improvement of +20.92 points across three length instruction following benchmarks with only a marginal decline of -1.45 points on four general capability benchmarks. 

---
# EvidenceRL: Reinforcing Evidence Consistency for Trustworthy Language Models 

**Authors**: J. Ben Tamo, Yuxing Lu, Benoit L. Marteau, Micky C. Nnamdi, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19532)  

**Abstract**: Large Language Models (LLMs) are fluent but prone to hallucinations, producing answers that appear plausible yet are unsupported by available evidence. This failure is especially problematic in high-stakes domains where decisions must be justified by verifiable information. We introduce \textbf{EvidenceRL}, a reinforcement learning framework that enforces evidence adherence during training. EvidenceRL scores candidate responses for grounding (entailment with retrieved evidence and context) and correctness (agreement with reference answers) and optimizes the generator using Group Relative Policy Optimization (GRPO). We evaluate across two high-stakes domains, cardiac diagnosis and legal reasoning, where EvidenceRL consistently improves evidence grounding and faithfulness without sacrificing task accuracy. On cardiac diagnosis, F1@3 increases from 37.0 to 54.5 on Llama-3.2-3B while grounding ($G_{\max}@3$) rises from 47.6 to 78.2; hallucinations drop nearly 5$\times$ and evidence-supported diagnoses increase from 31.8\% to 61.6\%. On legal reasoning, EvidenceRL raises Faithfulness from 32.8\% to 67.6\% on Llama-3.1-8B, demonstrating consistent behavioral change across domains. Our code is open-sourced at this https URL. 

---
# SAGE: Sustainable Agent-Guided Expert-tuning for Culturally Attuned Translation in Low-Resource Southeast Asia 

**Authors**: Zhixiang Lu, Chong Zhang, Yulong Li, Angelos Stefanidis, Anh Nguyen, Imran Razzak, Jionglong Su, Zhengyong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.19931)  

**Abstract**: The vision of an inclusive World Wide Web is impeded by a severe linguistic divide, particularly for communities in low-resource regions of Southeast Asia. While large language models (LLMs) offer a potential solution for translation, their deployment in data-poor contexts faces a dual challenge: the scarcity of high-quality, culturally relevant data and the prohibitive energy costs of training on massive, noisy web corpora. To resolve the tension between digital inclusion and environmental sustainability, we introduce Sustainable Agent-Guided Expert-tuning (SAGE). This framework pioneers an energy-aware paradigm that prioritizes the "right data" over "big data". Instead of carbon-intensive training on unfiltered datasets, SAGE employs a reinforcement learning (RL) agent, optimized via Group Relative Policy Optimization (GRPO), to autonomously curate a compact training set. The agent utilizes a semantic reward signal derived from a small, expert-constructed set of community dialogues to filter out noise and cultural misalignment. We then efficiently fine-tune open-source LLMs on this curated data using Low-Rank Adaptation (LoRA). We applied SAGE to translation tasks between English and seven low-resource languages (LRLs) in Southeast Asia. Our approach establishes new state-of-the-art performance on BLEU-4 and COMET-22 metrics, effectively capturing local linguistic nuances. Crucially, SAGE surpasses baselines trained on full datasets while reducing data usage by 97.1% and training energy consumption by 95.2%. By delivering high-performance models with a minimal environmental footprint, SAGE offers a scalable and responsible pathway to bridge the digital divide in the Global South. 

---
# LoopRPT: Reinforcement Pre-Training for Looped Language Models 

**Authors**: Guo Tang, Shixin Jiang, Heng Chang, Nuo Chen, Yuhan Li, Huiming Fan, Jia Li, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2603.19714)  

**Abstract**: Looped language models (LoopLMs) perform iterative latent computation to refine internal representations, offering a promising alternative to explicit chain-of-thought (CoT) reasoning. However, existing reinforcement learning (RL) paradigms primarily target output tokens, creating a structural mismatch with looped architectures whose reasoning unfolds implicitly. In this work, we propose LoopRPT, a reinforcement pre-training framework tailored for LoopLMs. By reframing next-token prediction as a next-token reasoning task, LoopRPT assigns reinforcement signals directly to latent steps using an EMA teacher reference and noisy latent rollouts. This formulation enables RL to directly shape intermediate representations, compressing effective reasoning into fewer iterations. We instantiate LoopRPT on the Ouro architecture across multiple model scales. Results demonstrate that LoopRPT consistently improves per-step representation quality, achieving Pareto dominance in accuracy-computation trade-offs. Notably, significant gains on hard tokens indicate that LoopRPT enhances early-stage reasoning rather than merely encouraging premature exits. Our findings highlight reinforcement pre-training as a principled paradigm for learning efficient latent reasoning in LoopLMs. 

---
# PrefPO: Pairwise Preference Prompt Optimization 

**Authors**: Rahul Singhal, Pradyumna Tambwekar, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2603.19311)  

**Abstract**: Prompt engineering is effective but labor-intensive, motivating automated optimization methods. Existing methods typically require labeled datasets, which are often unavailable, and produce verbose, repetitive prompts. We introduce PrefPO, a minimal prompt optimization approach inspired by reinforcement learning from human feedback (RLHF). Its preference-based approach reduces the need for labeled data and hyperparameter tuning-only a starting prompt and natural language criteria are needed. PrefPO uses an LLM discriminator to express pairwise preferences over model outputs and provide feedback to an LLM optimizer, iteratively improving performance. We evaluate PrefPO on 9 BIG-Bench Hard (BBH) tasks and IFEval-Hard, a newly-curated, challenging subset of IFEval. PrefPO matches or exceeds SOTA methods, including GEPA, MIPRO, and TextGrad, on 6/9 tasks and performs comparably to TextGrad on IFEval-Hard (82.4% vs 84.5%). Unlike other methods, PrefPO can optimize in both labeled and unlabeled settings. Without labels, PrefPO closely matches its labeled performance on 6/9 tasks, proving effective without ground truth. PrefPO also improves prompt hygiene: we find existing methods produce prompts 14.7x their original length or with 34% repetitive content; PrefPO reduces these issues by 3-5x. Furthermore, both LLM and human judges rate PrefPO's prompts higher than TextGrad's. Finally, we identify prompt hacking in prompt optimizers, where methods game evaluation criteria, and find PrefPO is susceptible at half the rate of TextGrad (37% vs 86%), generating fewer brittle, misaligned prompts. 

---
# Enhancing Legal LLMs through Metadata-Enriched RAG Pipelines and Direct Preference Optimization 

**Authors**: Suyash Maniyar, Deepali Singh, Rohith Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2603.19251)  

**Abstract**: Large Language Models (LLMs) perform well in short contexts but degrade on long legal documents, often producing hallucinations such as incorrect clauses or precedents. In the legal domain, where precision is critical, such errors undermine reliability and trust.
Retrieval Augmented Generation (RAG) helps ground outputs but remains limited in legal settings, especially with small, locally deployed models required for data privacy. We identify two failure modes: retrieval errors due to lexical redundancy in legal corpora, and decoding errors where models generate answers despite insufficient context.
To address this, we propose Metadata Enriched Hybrid RAG to improve document level retrieval, and apply Direct Preference Optimization (DPO) to enforce safe refusal when context is inadequate. Together, these methods improve grounding, reliability, and safety in legal language models. 

---
# ReViSQL: Achieving Human-Level Text-to-SQL 

**Authors**: Yuxuan Zhu, Tengjun Jin, Yoojin Choi, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20004)  

**Abstract**: Translating natural language to SQL (Text-to-SQL) is a critical challenge in both database research and data analytics applications. Recent efforts have focused on enhancing SQL reasoning by developing large language models and AI agents that decompose Text-to-SQL tasks into manually designed, step-by-step pipelines. However, despite these extensive architectural engineering efforts, a significant gap remains: even state-of-the-art (SOTA) AI agents have not yet achieved the human-level accuracy on the BIRD benchmark. In this paper, we show that closing this gap does not require further architectural complexity, but rather clean training data to improve SQL reasoning of the underlying models.
We introduce ReViSQL, a streamlined framework that achieves human-level accuracy on BIRD for the first time. Instead of complex AI agents, ReViSQL leverages reinforcement learning with verifiable rewards (RLVR) on BIRD-Verified, a dataset we curated comprising 2.5k verified Text-to-SQL instances based on the BIRD Train set. To construct BIRD-Verified, we design a data correction and verification workflow involving SQL experts. We identified and corrected data errors in 61.1% of a subset of BIRD Train. By training on BIRD-Verified, we show that improving data quality alone boosts the single-generation accuracy by 8.2-13.9% under the same RLVR algorithm. To further enhance performance, ReViSQL performs inference-time scaling via execution-based reconciliation and majority voting. Empirically, we demonstrate the superiority of our framework with two model scales: ReViSQL-235B-A22B and ReViSQL-30B-A3B. On an expert-verified BIRD Mini-Dev set, ReViSQL-235B-A22B achieves 93.2% execution accuracy, exceeding the proxy human-level accuracy (92.96%) and outperforming the prior open-source SOTA method by 9.8%. Our lightweight ReViSQL-30B-A3B matches the prior SOTA at a 7.5$\times$ lower per-query cost. 

---
# FedPDPO: Federated Personalized Direct Preference Optimization for Large Language Model Alignment 

**Authors**: Kewen Zhu, Liping Yi, Zhiming Zhao, Zhuang Qi, Han Yu, Qinghua Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.19741)  

**Abstract**: Aligning large language models (LLMs) with human preferences in federated learning (FL) is challenging due to decentralized, privacy-sensitive, and highly non-IID preference data. Direct Preference Optimization (DPO) offers an efficient alternative to reinforcement learning with human feedback (RLHF), but its direct application in FL suffers from severe performance degradation under non-IID data and limited generalization of implicit rewards. To bridge this gap, we propose FedPDPO (Federated Personalized Direct Preference Optimization), a personalized federated framework for preference alignment of LLMs. It adopts a parameter-efficient fine-tuning architecture where each client maintains a frozen pretrained LLM backbone augmented with a Low-Rank Adaptation (LoRA) adapter, enabling communication-efficient aggregation. To address non-IID heterogeneity, we devise (1) the globally shared LoRA adapter with the personalized client-specific LLM head. Moreover, we introduce (2) a personalized DPO training strategy with a client-specific explicit reward head to complement implicit rewards and further alleviate non-IID heterogeneity, and (3) a bottleneck adapter to balance global and local features. We provide theoretical analysis establishing the probabilistic foundation and soundness. Extensive experiments on multiple preference datasets demonstrate state-of-the-art performance, achieving up to 4.80% average accuracy improvements in federated intra-domain and cross-domain settings. 

---
# Maximizing mutual information between user-contexts and responses improve LLM personalization with no additional data 

**Authors**: Hyunji Nam, Haoran Li, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2603.19294)  

**Abstract**: While post-training has successfully improved large language models (LLMs) across a variety of domains, these gains heavily rely on human-labeled data or external verifiers. Existing data has already been exploited, and new high-quality data is expensive to collect. More fundamentally, true intelligence goes far beyond tasks that are easily verifiable. Therefore, we need self-improvement frameworks that allow models to improve without external oversight. We propose *Mutual Information Preference Optimization (MIPO)*, a contrastive data augmentation method that constructs preference pairs by generating a positive response conditioning on the correct prompt, and a negative response by conditioning on a random, unrelated prompt. We show that using Direct Preference Optimization (DPO) to learn from this paired data maximizes pointwise conditional mutual information (MI) (under the base LLM) between prompts and model responses. Empirical results with various-sized Llama- and Qwen-Instruct models show that when used to maximize MI between user context and response, MIPO provides an effective personalization technique, achieving 3-40% improvements on personalization tasks using real-user datasets compared to strong baselines. Surprisingly, MIPO can also be applied to improve performance on math and multiple-choice problems, yielding 1-18% **without any additional data or human supervision**. These results suggest a promising direction for self-improvement. 

---
