# From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models 

**Authors**: Chenchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09459)  

**Abstract**: Reinforcement learning (RL) for large language models (LLMs) increasingly relies on sparse, outcome-level rewards -- yet determining which actions within a long trajectory caused the outcome remains difficult. This credit assignment (CA) problem manifests in two regimes: reasoning RL, where credit must be distributed across tokens and steps within a single chain-of-thought generation (500--30K+ tokens); and agentic RL, where multi-turn environment interaction introduces stochastic transitions, partial observability, and horizons of 100+ turns (100K--1M tokens), making episode-level credit increasingly uninformative.
We survey 47 CA methods (41 core, 6 adjacent enablers) published between 2024 and early 2026, organizing them in a two-dimensional taxonomy by assignment granularity (token, segment, step, turn, multi-agent) and methodology (Monte Carlo, temporal difference, model-based, game-theoretic, information-theoretic). Beyond the survey itself, we contribute three reusable resources: (1) a structured, machine-readable paper inventory with taxonomy labels, baseline families, and evidence levels; (2) a reporting checklist for future CA papers, validated against the reviewed literature to identify systematic methodological gaps; and (3) a benchmark protocol specification with task families, metadata requirements, and controlled bifurcation tasks, accompanied by a method selection decision tree.
Our synthesis suggests that the shift from reasoning to agentic RL complicates and reshapes the credit assignment landscape: reasoning CA is maturing around process reward models and critic-free group comparison, while agentic CA is driving genuinely new approaches -- hindsight counterfactual analysis, privileged asymmetric critics, and turn-level MDP reformulations -- that have no direct precedent in reasoning RL. 

---
# Think Less, Know More: State-Aware Reasoning Compression with Knowledge Guidance for Efficient Reasoning 

**Authors**: Yi Sui, Chaozhuo Li, Dawei Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.09150)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance on complex tasks by leveraging long Chain-of-Thought (CoT), but often suffer from overthinking, leading to excessive reasoning steps and high inference latency. Existing CoT compression methods struggle to balance accuracy and efficiency, and lack fine-grained, step-level adaptation to redundancy and reasoning bias. Therefore, we propose State-Aware Reasoning Compression with Knowledge Guidance (STACK), a framework that performs step-wise CoT compression by explicitly modeling stage-specific redundancy sources and integrating with a retrieval-augmented guidance. STACK constructs online long-short contrastive samples and dynamically switches between knowledge-guided compression for uncertain or biased reasoning state and self-prompted compression for overly long but confident state, complemented by an answer-convergence-based early stopping mechanism to suppress redundant verification. We further propose a reward-difference-driven training strategy by combining Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO), enabling models to learn state-conditioned compression strategies. Experiments on three mathematical reasoning benchmarks show that STACK achieves a superior accuracy-efficiency balance, reducing average response length by 59.9% while improving accuracy by 4.8 points over existing methods. 

---
# PerMix-RLVR: Preserving Persona Expressivity under Verifiable-Reward Alignment 

**Authors**: Jihwan Oh, Soowon Oh, Murad Aghazada, Minchan Jeong, Sungnyun Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2604.08986)  

**Abstract**: Persona prompting has been widely adopted to steer large language models (LLMs) behavior and improve their instruction performance by assigning specific characters. However, identifying an optimal persona is time-consuming, and its impact on output quality remains poorly understood. Prior work has mainly addressed this issue at the prompt level via inference-time strategies, incurring additional computation. In this work, we avoid inference-time prompt search by tackling persona sensitivity during training, aiming to train models that adapt their behavior to diverse personas while preserving task performance. In particular, we find that reinforcement learning with verifiable rewards (RLVR) systematically reduces sensitivity to persona prompts, but also reveals an inherent trade-off of outcome-based optimization: while RLVR improves robustness on tasks with verifiable goals, it can also degrade persona expressivity when needed, e.g., in-character role-playing. To address this limitation, we propose PerMix-RLVR, a persona-mixed RLVR strategy that mitigates the persona robustness-fidelity trade-off, preserving strong robustness to harmful persona variation while enabling faithful persona adoption when required. Concretely, PerMix-RLVR improves persona stability score (PSS) over RLVR by +21.2% on MATH500, while also enhancing persona fidelity by +11.4% on PersonaGym. 

---
# Decomposing the Delta: What Do Models Actually Learn from Preference Pairs? 

**Authors**: Chia-Hsuan Lee, Mingyang Zhou, Renkun Ni, Zelei Cheng, Sihui Dai, Supriyo Chakraborty, Shixiong Zhang, Sambit Sahu, William Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2604.08723)  

**Abstract**: Preference optimization methods such as DPO and KTO are widely used for aligning language models, yet little is understood about what properties of preference data drive downstream reasoning gains. We ask: what aspects of a preference pair improve a reasoning model's performance on general reasoning tasks? We investigate two distinct notions of quality delta in preference data: generator-level delta, arising from the differences in capability between models that generate chosen and rejected reasoning traces, and sample-level delta, arising from differences in judged quality differences within an individual preference pair. To study generator-level delta, we vary the generator's scale and model family, and to study sample-level delta, we employ an LLM-as-a-judge to rate the quality of generated traces along multiple reasoning-quality dimensions. We find that increasing generator-level delta steadily improves performance on out-of-domain reasoning tasks and filtering data by sample-level delta can enable more data-efficient training. Our results suggest a twofold recipe for improving reasoning performance through preference optimization: maximize generator-level delta when constructing preference pairs and exploit sample-level delta to select the most informative training examples. 

---
# VL-Calibration: Decoupled Confidence Calibration for Large Vision-Language Models Reasoning 

**Authors**: Wenyi Xiao, Xinchi Xu, Leilei Gan  

**Link**: [PDF](https://arxiv.org/pdf/2604.09529)  

**Abstract**: Large Vision Language Models (LVLMs) achieve strong multimodal reasoning but frequently exhibit hallucinations and incorrect responses with high certainty, which hinders their usage in high-stakes domains. Existing verbalized confidence calibration methods, largely developed for text-only LLMs, typically optimize a single holistic confidence score using binary answer-level correctness. This design is mismatched to LVLMs: an incorrect prediction may arise from perceptual failures or from reasoning errors given correct perception, and a single confidence conflates these sources while visual uncertainty is often dominated by language priors. To address these issues, we propose VL-Calibration, a reinforcement learning framework that explicitly decouples confidence into visual and reasoning confidence. To supervise visual confidence without ground-truth perception labels, we introduce an intrinsic visual certainty estimation that combines (i) visual grounding measured by KL-divergence under image perturbations and (ii) internal certainty measured by token entropy. We further propose token-level advantage reweighting to focus optimization on tokens based on visual certainty, suppressing ungrounded hallucinations while preserving valid perception. Experiments on thirteen benchmarks show that VL-Calibration effectively improves calibration while boosting visual reasoning accuracy, and it generalizes to out-of-distribution benchmarks across model scales and architectures. 

---
# Visually-Guided Policy Optimization for Multimodal Reasoning 

**Authors**: Zengbin Wang, Feng Xiong, Liang Lin, Xuecai Hu, Yong Wang, Yanlin Wang, Man Zhang, Xiangxiang Chu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09349)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning ability of vision-language models (VLMs). However, the inherent text-dominated nature of VLMs often leads to insufficient visual faithfulness, characterized by sparse attention activation to visual tokens. More importantly, our empirical analysis reveals that temporal visual forgetting along reasoning steps exacerbates this deficiency. To bridge this gap, we propose Visually-Guided Policy Optimization (VGPO), a novel framework to reinforce visual focus during policy optimization. Specifically, VGPO initially introduces a Visual Attention Compensation mechanism that leverages visual similarity to localize and amplify visual cues, while progressively elevating visual expectations in later steps to counteract visual forgetting. Building on this mechanism, we implement a dual-grained advantage re-weighting strategy: the intra-trajectory level highlights tokens exhibiting relatively high visual activation, while the inter-trajectory level prioritizes trajectories demonstrating superior visual accumulation. Extensive experiments demonstrate that VGPO achieves better visual activation and superior performance in mathematical multimodal reasoning and visual-dependent tasks. 

---
# Skip-Connected Policy Optimization for Implicit Advantage 

**Authors**: Fengwei Teng, Jinyi Bai, Xinhao Yao, Demi Ruohan Wang, Jiahao Zhao, Zhijiang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.08690)  

**Abstract**: Group Relative Policy Optimization (GRPO) has proven effective in RLVR by using outcome-based rewards. While fine-grained dense rewards can theoretically improve performance, we reveal that under practical sampling budgets, Monte Carlo estimation yields high-variance and sign-inconsistent advantages for early reasoning tokens, paradoxically underperforming outcome-only GRPO. We propose Skip-Connected Optimization (SKPO), which decomposes reasoning into upstream and downstream phases: upstream receives dense rewards from downstream Monte Carlo sampling with single-stream optimization; downstream maintains group-relative optimization, where a skip connection concatenates the upstream segment with the original problem, enabling the model to leverage helpful upstream reasoning while preserving the freedom to bypass flawed reasoning through direct problem access. Experiments demonstrate improvements of 3.91% and 6.17% relative gains over the strongest baselines on Qwen2.5-Math-7B and Llama-3.2-3B respectively across mathematical benchmarks and out-of-domain tasks including general reasoning and code generation. Further analysis reveals an implicit advantage: SKPO generates trajectories with higher intermediate-step quality even when matched for final correctness. 

---
# SUPERNOVA: Eliciting General Reasoning in LLMs with Reinforcement Learning on Natural Instructions 

**Authors**: Ashima Suvarna, Kendrick Phan, Mehrab Beikzadeh, Hritik Bansal, Saadia Gabriel  

**Link**: [PDF](https://arxiv.org/pdf/2604.08477)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly improved large language model (LLM) reasoning in formal domains such as mathematics and code. Despite these advancements, LLMs still struggle with general reasoning tasks requiring capabilities such as causal inference and temporal understanding. Extending RLVR to general reasoning is fundamentally constrained by the lack of high-quality, verifiable training data that spans diverse reasoning skills. To address this challenge, we propose SUPERNOVA, a data curation framework for RLVR aimed at enhancing general reasoning. Our key insight is that instruction-tuning datasets containing expert-annotated ground-truth encode rich reasoning patterns that can be systematically adapted for RLVR. To study this, we conduct 100+ controlled RL experiments to analyze how data design choices impact downstream reasoning performance. In particular, we investigate three key factors: (i) source task selection, (ii) task mixing strategies, and (iii) synthetic interventions for improving data quality. Our analysis reveals that source task selection is non-trivial and has a significant impact on downstream reasoning performance. Moreover, selecting tasks based on their performance for individual target tasks outperforms strategies based on overall average performance. Finally, models trained on SUPERNOVA outperform strong baselines (e.g., Qwen3.5) on challenging reasoning benchmarks including BBEH, Zebralogic, and MMLU-Pro. In particular, training on SUPERNOVA yields relative improvements of up to 52.8\% on BBEH across model sizes, demonstrating the effectiveness of principled data curation for RLVR. Our findings provide practical insights for curating human-annotated resources to extend RLVR to general reasoning. The code and data is available at this https URL. 

---
# Process Reward Agents for Steering Knowledge-Intensive Reasoning 

**Authors**: Jiwoong Sohn, Tomasz Sternal, Kenneth Styppa, Torsten Hoefler, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2604.09482)  

**Abstract**: Reasoning in knowledge-intensive domains remains challenging as intermediate steps are often not locally verifiable: unlike math or code, evaluating step correctness may require synthesizing clues across large external knowledge sources. As a result, subtle errors can propagate through reasoning traces, potentially never to be detected. Prior work has proposed process reward models (PRMs), including retrieval-augmented variants, but these methods operate post hoc, scoring completed trajectories, which prevents their integration into dynamic inference procedures. Here, we introduce Process Reward Agents (PRA), a test-time method for providing domain-grounded, online, step-wise rewards to a frozen policy. In contrast to prior retrieval-augmented PRMs, PRA enables search-based decoding to rank and prune candidate trajectories at every generation step. Experiments on multiple medical reasoning benchmarks demonstrate that PRA consistently outperforms strong baselines, achieving 80.8% accuracy on MedQA with Qwen3-4B, a new state of the art at the 4B scale. Importantly, PRA generalizes to unseen frozen policy models ranging from 0.5B to 8B parameters, improving their accuracy by up to 25.7% without any policy model updates. More broadly, PRA suggests a paradigm in which frozen reasoners are decoupled from domain-specific reward modules, allowing the deployment of new backbones in complex domains without retraining. 

---
# HiL-Bench (Human-in-Loop Benchmark): Do Agents Know When to Ask for Help? 

**Authors**: Mohamed Elfeki, Tu Trinh, Kelvin Luu, Guangze Luo, Nathan Hunt, Ernesto Montoya, Nandan Marwaha, Yannis He, Charles Wang, Fernando Crabedo, Alessa Castilo, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.09408)  

**Abstract**: Frontier coding agents solve complex tasks when given complete context but collapse when specifications are incomplete or ambiguous. The bottleneck is not raw capability, but judgment: knowing when to act autonomously and when to ask for help. Current benchmarks are blind to this failure mode. They supply unambiguous detailed instructions and solely reward execution correctness, so an agent that makes a lucky guess for a missing requirement will score identically to one that would have asked to be certain.
We present HiL-Bench (Human-in-the-Loop Benchmark) to measure this selective escalation skill. Each task contains human-validated blockers (missing information, ambiguous requests, contradictory information) that surface only through progressive exploration, not upfront inspection. Our core metric, Ask-F1, the harmonic mean of question precision and blocker recall, captures the tension between over-asking and silent guessing; its structure architecturally prevents gaming through question spam.
Evaluation across SWE and text-to-SQL domains reveals a large universal judgment gap: no frontier model recovers more than a fraction of its full-information performance when deciding whether to ask. Failure analysis identifies three key help-seeking patterns: overconfident wrong beliefs with no gap detection; high uncertainty detection yet persistent errors; broad, imprecise escalation without self-correction. These consistent patterns confirm poor help-seeking is a model-level flaw, not task-specific. RL training on shaped Ask-F1 reward shows judgment is trainable: a 32B model improves both help-seeking quality and task pass rate, with gains that transfer across domains. The model does not learn domain-specific heuristics for when to ask; it learns to detect unresolvable uncertainty and act on it. 

---
# Mind the Gap Between Spatial Reasoning and Acting! Step-by-Step Evaluation of Agents With Spatial-Gym 

**Authors**: Lars Benedikt Kaesberg, Tianyu Yang, Niklas Bauer, Terry Ruas, Jan Philip Wahle, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2604.09338)  

**Abstract**: Spatial reasoning is central to navigation and robotics, yet measuring model capabilities on these tasks remains difficult. Existing benchmarks evaluate models in a one-shot setting, requiring full solution generation in a single response, unlike humans, who work in interactive environments step-by-step. We introduce Spatial-Gym, a Gymnasium environment that isolates spatial constraint reasoning by testing pathfinding in 2D-grid puzzles as a sequential decision task with optional backtracking. We evaluate eight models in three settings (one-shot, step-by-step, step-by-step with backtracking) against human, random, and A* baselines on 500 episodes. The best model, GPT-OSS 120B, achieves a solve rate of 16.0%, 82 points below the human baseline (98.0%). Step-by-step format helps weaker models (up to +5.4%) by removing formatting errors, but hurts stronger models (up to 5.6%) by constraining global planning. Backtracking improves episode completion, but increases solve rate only for weaker models; stronger models rarely backtrack and do not benefit from it. Our experiments have three key findings: (1) models fail to scale reasoning effort with difficulty, (2) vision models receiving images of the spatial environment reduce solve rate by 73%, and (3) extended chain-of-thought reasoning retains a 3-5x accuracy advantage over standard inference even in the step-by-step setting. Spatial-Gym enables diagnosis of model limitations and provides a framework for improving spatial reasoning through reinforcement learning. 

---
# E3-TIR: Enhanced Experience Exploitation for Tool-Integrated Reasoning 

**Authors**: Weiyang Guo, Zesheng Shi, Liye Zhao, Jiayuan Ma, Zeen Zhu, Junxian He, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.09455)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant potential in Tool-Integrated Reasoning (TIR), existing training paradigms face significant limitations: Zero-RL suffers from inefficient exploration and mode degradation due to a lack of prior guidance, while SFT-then-RL is limited by high data costs and capability plateaus caused by low-entropy collapse. To address these challenges, we propose E3-TIR (Enhanced Experience Exploitation), a warm-up paradigm for the early stages of agent training. Specifically, we formulate training as the dynamic integration of three experience types: Expert Prefixes, Expert Guided, and Self-Exploration. By executing diverse branching exploration around expert "anchors" and employing a mix policy optimization mechanism, we effectively mitigate distribution shifts and resolve optimization conflicts arising from shared prefixes. Our method dynamically adapts the model's knowledge boundaries, effectively balancing exploration diversity with training this http URL results demonstrate that E3-TIR achieves a 6 performance improvement over traditional paradigms on tool-use tasks, while requiring less than 10 of the synthetic data. Furthermore, in terms of ROI, a comprehensive metric integrating performance, data cost, and training efficiency we achieve a 1.46x gain compared to baselines. Code is available at this https URL. 

---
# StaRPO: Stability-Augmented Reinforcement Policy Optimization 

**Authors**: Jinghan Zhang, Fengran Mo, Tharindu Cyril Weerasooriya, Ruimin Dai, Xiaoyan Han, Yanjie Fu, Dakuo Wang, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.08905)  

**Abstract**: Reinforcement learning (RL) is effective in enhancing the accuracy of large language models in complex reasoning tasks. Existing RL policy optimization frameworks rely on final-answer correctness as feedback signals and rarely capture the internal logical structure of the reasoning process. Consequently, the models would generate fluent and semantically relevant responses but logically inconsistent, structurally erratic, or redundant. To this end, we propose StaRPO, a stability-augmented reinforcement learning framework that explicitly incorporates reasoning stability into the optimization objective. Our StaRPO decomposes stability into two computable lightweight metrics: the Autocorrelation Function (ACF) to evaluate local step-to-step coherence, and Path Efficiency (PE) to evaluate global goal-directedness of the reasoning trajectory. These stability rewards are combined with task rewards to provide complementary and process-aware feedback. We validate the effectiveness of using ACF and PE rewards by showing their correlation with logic errors on two backbone models. Experiments on four reasoning benchmarks show that StaRPO consistently outperforms compared baselines and can enhance both final-answer accuracy and logical stability. 

---
# SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks 

**Authors**: Tianyi Wang, Yixia Li, Long Li, Yibiao Chen, Shaohan Huang, Yun Chen, Peng Li, Yang Liu, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.08865)  

**Abstract**: Proximal Policy Optimization (PPO) is central to aligning Large Language Models (LLMs) in reasoning tasks with verifiable rewards. However, standard token-level PPO struggles in this setting due to the instability of temporal credit assignment over long Chain-of-Thought (CoT) horizons and the prohibitive memory cost of the value model. While critic-free alternatives like GRPO mitigate these issues, they incur significant computational overhead by requiring multiple samples for baseline estimation, severely limiting training throughput. In this paper, we introduce Sequence-Level PPO (SPPO), a scalable algorithm that harmonizes the sample efficiency of PPO with the stability of outcome-based updates. SPPO reformulates the reasoning process as a Sequence-Level Contextual Bandit problem, employing a decoupled scalar value function to derive low-variance advantage signals without multi-sampling. Extensive experiments on mathematical benchmarks demonstrate that SPPO significantly surpasses standard PPO and matches the performance of computation-heavy group-based methods, offering a resource-efficient framework for aligning reasoning LLMs. 

---
# VISOR: Agentic Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning 

**Authors**: Yucheng Shen, Jiulong Wu, Jizhou Huang, Dawei Yin, Lingyong Yan, Min Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.09508)  

**Abstract**: Visual Retrieval-Augmented Generation (VRAG) empowers Vision-Language Models to retrieve and reason over visually rich documents. To tackle complex queries requiring multi-step reasoning, agentic VRAG systems interleave reasoning with iterative retrieval.. However, existing agentic VRAG faces two critical bottlenecks. (1) Visual Evidence Sparsity: key evidence is scattered across pages yet processed in isolation, hindering cross-page reasoning; moreover, fine-grained intra-image evidence often requires precise visual actions, whose misuse degrades retrieval quality; (2) Search Drift in Long Horizons: the accumulation of visual tokens across retrieved pages dilutes context and causes cognitive overload, leading agents to deviate from their search objective. To address these challenges, we propose VISOR (Visual Retrieval-Augmented Generation via Iterative Search and Over-horizon Reasoning), a unified single-agent framework. VISOR features a structured Evidence Space for progressive cross-page reasoning, coupled with a Visual Action Evaluation and Correction mechanism to manage visual actions. Additionally, we introduce a Dynamic Trajectory with Sliding Window and Intent Injection to mitigate search drift. They anchor the evidence space while discarding earlier raw interactions, preventing context from being overwhelmed by visual tokens. We train VISOR using a Group Relative Policy Optimization-based Reinforcement Learning (GRPO-based RL) pipeline with state masking and credit assignment tailored for dynamic context reconstruction. Extensive experiments on ViDoSeek, SlideVQA, and MMLongBench demonstrate that VISOR achieves state-of-the-art performance with superior efficiency for long-horizon visual reasoning tasks. 

---
# SkillMOO: Multi-Objective Optimization of Agent Skills for Software Engineering 

**Authors**: Jingzhi Gong, Ruizhen Gu, Zhiwei Fei, Yazhuo Cao, Lukas Twist, Alina Geiger, Shuo Han, Dominik Sobania, Federica Sarro, Jie M. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.09297)  

**Abstract**: Agent skills provide modular, task-specific guidance for LLM- based coding agents, but manually tuning skill bundles to balance success rate, cost, and runtime is expensive and fragile. We present SkillMOO, a multi-objective optimization framework that automatically evolves skill bundles using LLM-proposed edits and NSGA-II survivor selection: a solver agent evaluates candidate skill bundles on coding tasks and an optimizer agent proposes bundle edits based on failure analysis. On three SkillsBench software engineering tasks, SkillMOO improves pass rate by up to 131% while reducing cost up to 32% relative to the best baseline per task at low optimization overhead. Pattern analysis reveals pruning and substitution as primary drivers of improvement, suggesting effective bundles favor minimal, focused content over accumulated instructions. 

---
# NyayaMind- A Framework for Transparent Legal Reasoning and Judgment Prediction in the Indian Legal System 

**Authors**: Parjanya Aditya Shukla, Shubham Kumar Nigam, Debtanu Datta, Balaramamahanthi Deepak Patnaik, Noel Shallum, Pradeep Reddy Vanga, Saptarshi Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2604.09069)  

**Abstract**: Court Judgment Prediction and Explanation (CJPE) aims to predict a judicial decision and provide a legally grounded explanation for a given case based on the facts, legal issues, arguments, cited statutes, and relevant precedents. For such systems to be practically useful in judicial or legal research settings, they must not only achieve high predictive performance but also generate transparent and structured legal reasoning that aligns with established judicial practices. In this work, we present NyayaMind, an open-source framework designed to enable transparent and scalable legal reasoning for the Indian judiciary. The proposed framework integrates retrieval, reasoning, and verification mechanisms to emulate the structured decision-making process typically followed in courts. Specifically, NyayaMind consists of two main components: a Retrieval Module and a Prediction Module. The Retrieval Module employs a RAG pipeline to identify legally relevant statutes and precedent cases from large-scale legal corpora, while the Prediction Module utilizes reasoning-oriented LLMs fine-tuned for the Indian legal domain to generate structured outputs including issues, arguments, rationale, and the final decision. Our extensive results and expert evaluation demonstrate that NyayaMind significantly improves the quality of explanation and evidence alignment compared to existing CJPE approaches, providing a promising step toward trustworthy AI-assisted legal decision support systems. 

---
# Distributionally Robust Token Optimization in RLHF 

**Authors**: Yeping Jin, Jiaming Hu, Ioannis Ch. Paschalidis  

**Link**: [PDF](https://arxiv.org/pdf/2604.08577)  

**Abstract**: Large Language Models (LLMs) tend to respond correctly to prompts that align to the data they were trained and fine-tuned on. Yet, small shifts in wording, format, or language can trigger surprisingly large failures, especially on multi-step reasoning problems. To address this problem, we propose a Distributionally Robust Token Optimization (DRTO) approach, which combines token-level Reinforcement Learning from Human Feedback (RLHF) with Distributionally Robust Optimization (DRO). DRTO bounds worst case token-wise rewards by constructing an f-divergence ambiguity set over a loss minibatch, leading to a theoretical robustness. Empirically, DRTO enhances consistency under distribution shifts in mathematical reasoning benchmarks, achieving 9.17\% improvement on GSM8K and 2.49% improvement on MathQA. 

---
