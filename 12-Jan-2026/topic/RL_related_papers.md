# WildSci: Advancing Scientific Reasoning from In-the-Wild Literature 

**Authors**: Tengxiao Liu, Deepak Nathani, Zekun Li, Kevin Yang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05567)  

**Abstract**: Recent progress in large language model (LLM) reasoning has focused on domains like mathematics and coding, where abundant high-quality data and objective evaluation metrics are readily available. In contrast, progress in LLM reasoning models remains limited in scientific domains such as medicine and materials science due to limited dataset coverage and the inherent complexity of open-ended scientific questions. To address these challenges, we introduce WildSci, a new dataset of domain-specific science questions automatically synthesized from peer-reviewed literature, covering 9 scientific disciplines and 26 subdomains. By framing complex scientific reasoning tasks in a multiple-choice format, we enable scalable training with well-defined reward signals. We further apply reinforcement learning to finetune models on these data and analyze the resulting training dynamics, including domain-specific performance changes, response behaviors, and generalization trends. Experiments on a suite of scientific benchmarks demonstrate the effectiveness of our dataset and approach. We release WildSci to enable scalable and sustainable research in scientific reasoning, available at this https URL. 

---
# Do LLMs Need Inherent Reasoning Before Reinforcement Learning? A Study in Korean Self-Correction 

**Authors**: Hongjin Kim, Jaewook Lee, Kiyoung Lee, Jong-hun Shin, Soojong Lim, Oh-Woog Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2601.05459)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning and self-correction abilities in high-resource languages like English, but their performance remains limited in low-resource languages such as Korean. In this study, we investigate whether reinforcement learning (RL) can enhance Korean reasoning abilities to a degree comparable to English. Our findings reveal that RL alone yields limited improvements when applied to models lacking inherent Korean reasoning capabilities. To address this, we explore several fine-tuning strategies and show that aligning the model's internal reasoning processes with Korean inputs-particularly by tuning Korean-specific neurons in early layers-is key to unlocking RL's effectiveness. We introduce a self-correction code-switching dataset to facilitate this alignment and observe significant performance gains in both mathematical reasoning and self-correction tasks. Ultimately, we conclude that the crucial factor in multilingual reasoning enhancement is not injecting new linguistic knowledge, but effectively eliciting and aligning existing reasoning capabilities. Our study provides a new perspective on how internal translation and neuron-level tuning contribute to multilingual reasoning alignment in LLMs. 

---
# Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric Rewards 

**Authors**: Jiajie Zhang, Xin Lv, Ling Feng, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.06021)  

**Abstract**: Reinforcement learning (RL) has emerged as a critical technique for enhancing LLM-based deep search agents. However, existing approaches primarily rely on binary outcome rewards, which fail to capture the comprehensiveness and factuality of agents' reasoning process, and often lead to undesirable behaviors such as shortcut exploitation and hallucinations. To address these limitations, we propose \textbf{Citation-aware Rubric Rewards (CaRR)}, a fine-grained reward framework for deep search agents that emphasizes reasoning comprehensiveness, factual grounding, and evidence connectivity. CaRR decomposes complex questions into verifiable single-hop rubrics and requires agents to satisfy these rubrics by explicitly identifying hidden entities, supporting them with correct citations, and constructing complete evidence chains that link to the predicted answer. We further introduce \textbf{Citation-aware Group Relative Policy Optimization (C-GRPO)}, which combines CaRR and outcome rewards for training robust deep search agents. Experiments show that C-GRPO consistently outperforms standard outcome-based RL baselines across multiple deep search benchmarks. Our analysis also validates that C-GRPO effectively discourages shortcut exploitation, promotes comprehensive, evidence-grounded reasoning, and exhibits strong generalization to open-ended deep research tasks. Our code and data are available at this https URL. 

---
# GIFT: Games as Informal Training for Generalizable LLMs 

**Authors**: Nuoyan Lyu, Bingbing Xu, Weihao Meng, Yige Yuan, Yang Zhang, Zhiyong Huang, Tat-Seng Chua, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.05633)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in formal learning tasks such as mathematics and code generation, they still struggle with the "practical wisdom" and generalizable intelligence, such as strategic creativity and social reasoning, that characterize human cognition. This gap arises from a lack of informal learning, which thrives on interactive feedback rather than goal-oriented instruction. In this paper, we propose treating Games as a primary environment for LLM informal learning, leveraging their intrinsic reward signals and abstracted complexity to cultivate diverse competencies. To address the performance degradation observed in multi-task learning, we introduce a Nested Training Framework. Unlike naive task mixing optimizing an implicit "OR" objective, our framework employs sequential task composition to enforce an explicit "AND" objective, compelling the model to master multiple abilities simultaneously to achieve maximal rewards. Using GRPO-based reinforcement learning across Matrix Games, TicTacToe, and Who's the Spy games, we demonstrate that integrating game-based informal learning not only prevents task interference but also significantly bolsters the model's generalization across broad ability-oriented benchmarks. The framework and implementation are publicly available. 

---
# iReasoner: Trajectory-Aware Intrinsic Reasoning Supervision for Self-Evolving Large Multimodal Models 

**Authors**: Meghana Sunil, Manikandarajan Venmathimaran, Muthu Subash Kavitha  

**Link**: [PDF](https://arxiv.org/pdf/2601.05877)  

**Abstract**: Recent work shows that large multimodal models (LMMs) can self-improve from unlabeled data via self-play and intrinsic feedback. Yet existing self-evolving frameworks mainly reward final outcomes, leaving intermediate reasoning weakly constrained despite its importance for visually grounded decision making. We propose iReasoner, a self-evolving framework that improves an LMM's implicit reasoning by explicitly eliciting chain-of-thought (CoT) and rewarding its internal agreement. In a Proposer--Solver loop over unlabeled images, iReasoner augments outcome-level intrinsic rewards with a trajectory-aware signal defined over intermediate reasoning steps, providing learning signals that distinguish reasoning paths leading to the same answer without ground-truth labels or external judges. Starting from Qwen2.5-VL-7B, iReasoner yields up to $+2.1$ points across diverse multimodal reasoning benchmarks under fully unsupervised post-training. We hope this work serves as a starting point for reasoning-aware self-improvement in LMMs in purely unsupervised settings. 

---
# MemBuilder: Reinforcing LLMs for Long-Term Memory Construction via Attributed Dense Rewards 

**Authors**: Zhiyu Shen, Ziming Wu, Fuming Lai, Shaobing Lian, Yanghui Rao  

**Link**: [PDF](https://arxiv.org/pdf/2601.05488)  

**Abstract**: Maintaining consistency in long-term dialogues remains a fundamental challenge for LLMs, as standard retrieval mechanisms often fail to capture the temporal evolution of historical states. While memory-augmented frameworks offer a structured alternative, current systems rely on static prompting of closed-source models or suffer from ineffective training paradigms with sparse rewards. We introduce MemBuilder, a reinforcement learning framework that trains models to orchestrate multi-dimensional memory construction with attributed dense rewards. MemBuilder addresses two key challenges: (1) Sparse Trajectory-Level Rewards: we employ synthetic session-level question generation to provide dense intermediate rewards across extended trajectories; and (2) Multi-Dimensional Memory Attribution: we introduce contribution-aware gradient weighting that scales policy updates based on each component's downstream impact. Experimental results show that MemBuilder enables a 4B-parameter model to outperform state-of-the-art closed-source baselines, exhibiting strong generalization across long-term dialogue benchmarks. 

---
# Closing the Modality Reasoning Gap for Speech Large Language Models 

**Authors**: Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.05543)  

**Abstract**: Although speech large language models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs. 

---
# SceneAlign: Aligning Multimodal Reasoning to Scene Graphs in Complex Visual Scenes 

**Authors**: Chuhan Wang, Xintong Li, Jennifer Yuntong Zhang, Junda Wu, Chengkai Huang, Lina Yao, Julian McAuley, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2601.05600)  

**Abstract**: Multimodal large language models often struggle with faithful reasoning in complex visual scenes, where intricate entities and relations require precise visual grounding at each step. This reasoning unfaithfulness frequently manifests as hallucinated entities, mis-grounded relations, skipped steps, and over-specified reasoning. Existing preference-based approaches, typically relying on textual perturbations or answer-conditioned rationales, fail to address this challenge as they allow models to exploit language priors to bypass visual grounding. To address this, we propose SceneAlign, a framework that leverages scene graphs as structured visual information to perform controllable structural interventions. By identifying reasoning-critical nodes and perturbing them through four targeted strategies that mimic typical grounding failures, SceneAlign constructs hard negative rationales that remain linguistically plausible but are grounded in inaccurate visual facts. These contrastive pairs are used in Direct Preference Optimization to steer models toward fine-grained, structure-faithful reasoning. Across seven visual reasoning benchmarks, SceneAlign consistently improves answer accuracy and reasoning faithfulness, highlighting the effectiveness of grounding-aware alignment for multimodal reasoning. 

---
# MaxCode: A Max-Reward Reinforcement Learning Framework for Automated Code Optimization 

**Authors**: Jiefu Ou, Sapana Chaudhary, Kaj Bostrom, Nathaniel Weir, Shuai Zhang, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2601.05475)  

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in general coding tasks but encounter two key challenges when optimizing code: (i) the complexity of writing optimized code (such as performant CUDA kernels and competition-level CPU code) requires expertise in systems, algorithms and specific languages and (ii) requires interpretation of performance metrics like timing and device utilization beyond binary correctness. In this work, we explore inference-time search algorithms that guide the LLM to discover better solutions through iterative refinement based on execution feedback. Our approach, called MaxCode unifies existing search methods under a max-reward reinforcement learning framework, making the observation and action-value functions modular for modification. To enhance the observation space, we integrate a natural language critique model that converts raw execution feedback into diagnostic insights about errors and performance bottlenecks, and the best-discounted reward seen so far. Together, these provide richer input to the code proposal function. To improve exploration during search, we train a generative reward-to-go model using action values from rollouts to rerank potential solutions. Testing on the KernelBench (CUDA) and PIE (C++) optimization benchmarks shows that MaxCode improves optimized code performance compared to baselines, achieving 20.3% and 10.1% relative improvements in absolute speedup value and relative speedup ranking, respectively. 

---
# TIME: Temporally Intelligent Meta-reasoning Engine for Context Triggered Explicit Reasoning 

**Authors**: Susmit Das  

**Link**: [PDF](https://arxiv.org/pdf/2601.05300)  

**Abstract**: Reasoning oriented large language models often expose explicit "thinking" as long, turn-global traces at the start of every response, either always on or toggled externally at inference time. While useful for arithmetic, programming, and problem solving, this design is costly, blurs claim level auditability, and cannot re-trigger explicit reasoning once the model begins presenting. Dialogue models are also largely blind to temporal structure, treating replies after seconds and replies after weeks as equivalent unless time is stated in text. We introduce TIME, the Temporally Intelligent Meta-reasoning Engine, a behavioral alignment framework that treats explicit reasoning as a context sensitive resource driven by discourse and temporal cues. TIME augments dialogue with optional ISO 8601 <time> tags, tick turns that represent silent gaps, and short <think> blocks that can appear anywhere in a reply. A four-phase curriculum including a small, maximally diverse full-batch alignment step trains Qwen3 dense models to invoke brief, in-place reasoning bursts and keep user facing text compact. We evaluate with TIMEBench, a temporally grounded dialogue benchmark probing chronology, commonsense under gaps and offsets, anomaly detection, and continuity. Across 4B to 32B scales, TIME improves TIMEBench scores over base Qwen3 in both thinking and no-thinking modes while reducing reasoning tokens by about an order of magnitude. Our training data and code are available at this https URL and TIMEBench is available at this https URL 

---
