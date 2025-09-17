# The Anatomy of Alignment: Decomposing Preference Optimization by Steering Sparse Features 

**Authors**: Jeremias Ferrao, Matthijs van der Lende, Ilija Lichkovski, Clement Neo  

**Link**: [PDF](https://arxiv.org/pdf/2509.12934)  

**Abstract**: Aligning large language models is critical for their usability and safety. However, the prevailing approach of Reinforcement Learning from Human Feedback (RLHF) induces diffuse, opaque parameter changes, making it difficult to discern what the model has internalized. Hence, we introduce Feature Steering with Reinforcement Learning (FSRL), a transparent alignment framework that trains a lightweight adapter to steer behavior by modulating interpretable features from a Sparse Autoencoder (SAE). First, we demonstrate that FSRL is an effective method for preference optimization and is comparable with current RLHF methods. We then perform mechanistic analysis on the trained adapter, and find that its policy systematically promotes style features over explicit alignment concepts, suggesting that the preference optimization process rewards stylistic presentation as a proxy for quality. Ultimately, we hope that FSRL provides a tool for both interpretable model control and diagnosing the internal mechanisms of alignment. 

---
# Single-stream Policy Optimization 

**Authors**: Zhongwen Xu, Zihan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.13232)  

**Abstract**: We revisit policy-gradient optimization for Large Language Models (LLMs) from a single-stream perspective. Prevailing group-based methods like GRPO reduce variance with on-the-fly baselines but suffer from critical flaws: frequent degenerate groups erase learning signals, and synchronization barriers hinder scalability. We introduce Single-stream Policy Optimization (SPO), which eliminates these issues by design. SPO replaces per-group baselines with a persistent, KL-adaptive value tracker and normalizes advantages globally across the batch, providing a stable, low-variance learning signal for every sample. Being group-free, SPO enables higher throughput and scales effectively in long-horizon or tool-integrated settings where generation times vary. Furthermore, the persistent value tracker naturally enables an adaptive curriculum via prioritized sampling. Experiments using Qwen3-8B show that SPO converges more smoothly and attains higher accuracy than GRPO, while eliminating computation wasted on degenerate groups. Ablation studies confirm that SPO's gains stem from its principled approach to baseline estimation and advantage normalization, offering a more robust and efficient path for LLM reasoning. Across five hard math benchmarks with Qwen3 8B, SPO improves the average maj@32 by +3.4 percentage points (pp) over GRPO, driven by substantial absolute point gains on challenging datasets, including +7.3 pp on BRUMO 25, +4.4 pp on AIME 25, +3.3 pp on HMMT 25, and achieves consistent relative gain in pass@$k$ across the evaluated $k$ values. SPO's success challenges the prevailing trend of adding incidental complexity to RL algorithms, highlighting a path where fundamental principles, not architectural workarounds, drive the next wave of progress in LLM reasoning. 

---
# Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO 

**Authors**: Francesco Pappone, Ruggero Marino Lazzaroni, Federico Califano, Niccolò Gentile, Roberto Marras  

**Link**: [PDF](https://arxiv.org/pdf/2509.13081)  

**Abstract**: While Large Language Models (LLMs) excel at generating human-like text, aligning their outputs with complex, qualitative goals like pedagogical soundness remains a significant challenge. Standard reinforcement learning techniques often rely on slow and expensive LLM-as-a-judge evaluations or on brittle, keyword-based metrics like ROUGE, which fail to capture the semantic essence of a high-quality explanation. In this work, we introduce a novel approach to reward shaping within the Group Relative Policy Optimisation (GRPO) framework. Our central contribution is the use of a small, efficient encoder-only transformer as a semantic reward model. This model provides a dense, semantically rich reward signal based on the cosine similarity between a generated explanation and a ground-truth reference, guiding the policy towards explanations that are not just factually correct but also structurally and conceptually aligned with expert reasoning. We apply this method to the task of training a model for the Italian medical-school entrance examinations, following standard domain-adaptive continued pre-training (CPT) and supervised fine-tuning (SFT). Our results demonstrate that GRPO with our proposed semantic reward significantly improves explanation faithfulness and clarity over a strong SFT baseline, showcasing the power of using lightweight encoder models for nuanced reward shaping in complex generation tasks 

---
# Building Coding Agents via Entropy-Enhanced Multi-Turn Preference Optimization 

**Authors**: Jiahao Yu, Zelei Cheng, Xian Wu, Xinyu Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.12434)  

**Abstract**: Software engineering presents complex, multi-step challenges for Large Language Models (LLMs), requiring reasoning over large codebases and coordinated tool use. The difficulty of these tasks is exemplified by benchmarks like SWE-bench, where current LLMs still struggle to resolve real-world issues.
A promising approach to enhance performance is test-time scaling (TTS), but its gains are heavily dependent on the diversity of model outputs.
While standard alignment methods such as Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) are effective at aligning model outputs with human preferences, this process can come at the cost of reduced diversity, limiting the effectiveness of TTS.
Additionally, existing preference optimization algorithms are typically designed for single-turn tasks and do not fully address the complexities of multi-turn reasoning and tool integration required for interactive coding agents.
To bridge this gap, we introduce \sys, an entropy-enhanced framework that adapts existing preference optimization algorithms to the multi-turn, tool-assisted setting.
\sys augments the preference objective to explicitly preserve policy entropy and generalizes learning to optimize over multi-turn interactions rather than single-turn responses.
We validate \sys by fine-tuning a diverse suite of models from different families and sizes (up to 106B parameters).
To maximize performance gains from TTS, we further propose a hybrid best-trajectory selection scheme combining a learned verifier model with model free approaches.
On the \swebench leaderboard, our approach establishes new state-of-the-art results among open-weight models. A 30B parameter model trained with \sys ranks 1st on \lite and 4th on \verified on the open-weight leaderboard, surpassed only by models with over 10x more parameters(\eg$>$350B). 

---
# Mitigating Strategy Preference Bias in Emotional Support Conversation via Uncertainty Estimations 

**Authors**: Yougen Zhou, Qin Chen, Ningning Zhou, Jie Zhou, Xingjiao Wu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.12661)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate distress through empathetic dialogue, yet large language models (LLMs) face persistent challenges in delivering effective ESC due to low accuracy in strategy planning. Moreover, there is a considerable preference bias towards specific strategies. Prior methods using fine-tuned strategy planners have shown potential in reducing such bias, while the underlying causes of the preference bias in LLMs have not well been studied. To address these issues, we first reveal the fundamental causes of the bias by identifying the knowledge boundaries of LLMs in strategy planning. Then, we propose an approach to mitigate the bias by reinforcement learning with a dual reward function, which optimizes strategy planning via both accuracy and entropy-based confidence for each region according to the knowledge boundaries. Experiments on the ESCov and ExTES datasets with multiple LLM backbones show that our approach outperforms the baselines, confirming the effectiveness of our approach. 

---
# Audited Reasoning Refinement: Fine-Tuning Language Models via LLM-Guided Step-Wise Evaluation and Correction 

**Authors**: Sumanta Bhattacharyya, Sara Riaz, Pedram Rooshenas  

**Link**: [PDF](https://arxiv.org/pdf/2509.12476)  

**Abstract**: Training a task-specific small reasoning model is challenging when direct human supervision or high-quality labels are scarce. However, LLMs with reasoning capabilities produce abundant intermediate reasoning traces that can be systematically refined to create effective supervision signals. We propose Reason-Refine-then-Align (R2tA), which turns refined model rationales into supervision for training task-specific reasoning models. Our method generates initial reasoning and responses from an open-source base model on task-specific inputs, then refines these traces, fixing hallucinations and inconsistencies, to form a high-fidelity dataset. We perform a two-stage alignment, supervised fine-tuning (SFT), followed by direct preference optimization (DPO) to calibrate the model's intermediate reasoning with human-validated conceptual preferences and then condition the final output on that aligned reasoning. As a case study, we apply R2tA to evaluate extended entity relationship diagrams (EERDs) in database system design, a structurally complex task where prompt-only methods miss or hallucinate errors. We curated a dataset of 600 EERD variants (train/test split of 450/150, respectively) with induced mistakes spanning 11 categories. Empirical evaluation suggests R2tA provides a practical, cost-effective path to scalable LLM adaptation in data-scarce domains, enabling reproducible AI tools for education and beyond. 

---
# When Inverse Data Outperforms: Exploring the Pitfalls of Mixed Data in Multi-Stage Fine-Tuning 

**Authors**: Mengyi Deng, Xin Li, Tingyu Zhu, Zhicheng Yang, Zhijiang Guo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.13079)  

**Abstract**: Existing work has shown that o1-level performance can be achieved with limited data distillation, but most existing methods focus on unidirectional supervised fine-tuning (SFT), overlooking the intricate interplay between diverse reasoning patterns. In this paper, we construct r1k, a high-quality reverse reasoning dataset derived by inverting 1,000 forward examples from s1k, and examine how SFT and Direct Preference Optimization (DPO) affect alignment under bidirectional reasoning objectives. SFT on r1k yields a 1.6%--6.8% accuracy improvement over s1k across evaluated benchmarks. However, naively mixing forward and reverse data during SFT weakens the directional distinction. Although DPO can partially recover this distinction, it also suppresses less preferred reasoning paths by shifting the probability mass toward irrelevant outputs. These findings suggest that mixed reasoning data introduce conflicting supervision signals, underscoring the need for robust and direction-aware alignment strategies. 

---
# Rethinking the Evaluation of Alignment Methods: Insights into Diversity, Generalisation, and Safety 

**Authors**: Denis Janiak, Julia Moska, Dawid Motyka, Karolina Seweryn, Paweł Walkowiak, Bartosz Żuk, Arkadiusz Janz  

**Link**: [PDF](https://arxiv.org/pdf/2509.12936)  

**Abstract**: Large language models (LLMs) require careful alignment to balance competing objectives - factuality, safety, conciseness, proactivity, and diversity. Existing studies focus on individual techniques or specific dimensions, lacking a holistic assessment of the inherent trade-offs. We propose a unified evaluation framework that compares LLM alignment methods (PPO, DPO, ORPO, KTO) across these five axes, using both in-distribution and out-of-distribution datasets. Leveraging a specialized LLM-as-Judge prompt, validated through human studies, we reveal that DPO and KTO excel in factual accuracy, PPO and DPO lead in safety, and PPO best balances conciseness with proactivity. Our findings provide insights into trade-offs of common alignment methods, guiding the development of more balanced and reliable LLMs. 

---
# WebSailor-V2: Bridging the Chasm to Proprietary Agents via Synthetic Data and Scalable Reinforcement Learning 

**Authors**: Kuan Li, Zhongwang Zhang, Huifeng Yin, Rui Ye, Yida Zhao, Liwen Zhang, Litu Ou, Dingchu Zhang, Xixi Wu, Jialong Wu, Xinyu Wang, Zile Qiao, Zhen Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.13305)  

**Abstract**: Transcending human cognitive limitations represents a critical frontier in LLM training. Proprietary agentic systems like DeepResearch have demonstrated superhuman capabilities on extremely complex information-seeking benchmarks such as BrowseComp, a feat previously unattainable. We posit that their success hinges on a sophisticated reasoning pattern absent in open-source models: the ability to systematically reduce extreme uncertainty when navigating vast information landscapes. Based on this insight, we introduce WebSailor, a complete post-training methodology designed to instill this crucial capability. Our approach involves generating novel, high-uncertainty tasks through structured sampling and information obfuscation, RFT cold start, and an efficient agentic RL training algorithm, Duplicating Sampling Policy Optimization (DUPO). With this integrated pipeline, WebSailor significantly outperforms all open-source agents in complex information-seeking tasks, matching proprietary agents' performance and closing the capability gap. 

---
