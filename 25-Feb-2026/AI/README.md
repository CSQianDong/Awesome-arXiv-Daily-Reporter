# Aletheia tackles FirstProof autonomously 

**Authors**: Tony Feng, Junehyuk Jung, Sang-hyun Kim, Carlo Pagano, Sergei Gukov, Chiang-Chiang Tsai, David Woodruff, Adel Javanmard, Aryan Mokhtari, Dawsen Hwang, Yuri Chervonyi, Jonathan N. Lee, Garrett Bingham, Trieu H. Trinh, Vahab Mirrokni, Quoc V. Le, Thang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2602.21201)  

**Abstract**: We report the performance of Aletheia (Feng et al., 2026b), a mathematics research agent powered by Gemini 3 Deep Think, on the inaugural FirstProof challenge. Within the allowed timeframe of the challenge, Aletheia autonomously solved 6 problems (2, 5, 7, 8, 9, 10) out of 10 according to majority expert assessments; we note that experts were not unanimous on Problem 8 (only). For full transparency, we explain our interpretation of FirstProof and disclose details about our experiments as well as our evaluation. Raw prompts and outputs are available at this https URL. 

---
# NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning 

**Authors**: Ishaan Rawal, Shubh Gupta, Yihan Hu, Wei Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21172)  

**Abstract**: Vision-Language-Action (VLA) models are advancing autonomous driving by replacing modular pipelines with unified end-to-end architectures. However, current VLAs face two expensive requirements: (1) massive dataset collection, and (2) dense reasoning annotations. In this work, we address both challenges with \modelname (\textbf{No} \textbf{R}easoning for \textbf{D}riving). Compared to existing VLAs, \modelname achieves competitive performance while being fine-tuned on $<$60\% of the data and no reasoning annotations, resulting in 3$\times$ fewer tokens. We identify that standard Group Relative Policy Optimization (GRPO) fails to yield significant improvements when applied to policies trained on such small, reasoning-free datasets. We show that this limitation stems from difficulty bias, which disproportionately penalizes reward signals from scenarios that produce high-variance rollouts within GRPO. \modelname overcomes this by incorporating Dr.~GRPO, a recent algorithm designed to mitigate difficulty bias in LLMs. As a result, \modelname achieves competitive performance on Waymo and NAVSIM with a fraction of the training data and no reasoning overhead, enabling more efficient autonomous systems. 

---
# CG-DMER: Hybrid Contrastive-Generative Framework for Disentangled Multimodal ECG Representation Learning 

**Authors**: Ziwei Niu, Hao Sun, Shujun Bian, Xihong Yang, Lanfen Lin, Yuxin Liu, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.21154)  

**Abstract**: Accurate interpretation of electrocardiogram (ECG) signals is crucial for diagnosing cardiovascular diseases. Recent multimodal approaches that integrate ECGs with accompanying clinical reports show strong potential, but they still face two main concerns from a modality perspective: (1) intra-modality: existing models process ECGs in a lead-agnostic manner, overlooking spatial-temporal dependencies across leads, which restricts their effectiveness in modeling fine-grained diagnostic patterns; (2) inter-modality: existing methods directly align ECG signals with clinical reports, introducing modality-specific biases due to the free-text nature of the reports. In light of these two issues, we propose CG-DMER, a contrastive-generative framework for disentangled multimodal ECG representation learning, powered by two key designs: (1) Spatial-temporal masked modeling is designed to better capture fine-grained temporal dynamics and inter-lead spatial dependencies by applying masking across both spatial and temporal dimensions and reconstructing the missing information. (2) A representation disentanglement and alignment strategy is designed to mitigate unnecessary noise and modality-specific biases by introducing modality-specific and modality-shared encoders, ensuring a clearer separation between modality-invariant and modality-specific representations. Experiments on three public datasets demonstrate that CG-DMER achieves state-of-the-art performance across diverse downstream tasks. 

---
# A Benchmark for Deep Information Synthesis 

**Authors**: Debjit Paul, Daniel Murphy, Milan Gritta, Ronald Cardenas, Victor Prokhorov, Lena Sophia Bolliger, Aysim Toker, Roy Miles, Andreea-Maria Oncescu, Jasivan Alex Sivakumar, Philipp Borchert, Ismail Elezi, Meiru Zhang, Ka Yiu Lee, Guchun Zhang, Jun Wang, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2602.21143)  

**Abstract**: Large language model (LLM)-based agents are increasingly used to solve complex tasks involving tool use, such as web browsing, code execution, and data analysis. However, current evaluation benchmarks do not adequately assess their ability to solve real-world tasks that require synthesizing information from multiple sources and inferring insights beyond simple fact retrieval. To address this, we introduce DEEPSYNTH, a novel benchmark designed to evaluate agents on realistic, time-consuming problems that combine information gathering, synthesis, and structured reasoning to produce insights. DEEPSYNTH contains 120 tasks collected across 7 domains and data sources covering 67 countries. DEEPSYNTH is constructed using a multi-stage data collection pipeline that requires annotators to collect official data sources, create hypotheses, perform manual analysis, and design tasks with verifiable answers. When evaluated on DEEPSYNTH, 11 state-of-the-art LLMs and deep research agents achieve a maximum F1 score of 8.97 and 17.5 on the LLM-judge metric, underscoring the difficulty of the benchmark. Our analysis reveals that current agents struggle with hallucinations and reasoning over large information spaces, highlighting DEEPSYNTH as a crucial benchmark for guiding future research. 

---
# The Initial Exploration Problem in Knowledge Graph Exploration 

**Authors**: Claire McNamara, Lucy Hederman, Declan O'Sullivan  

**Link**: [PDF](https://arxiv.org/pdf/2602.21066)  

**Abstract**: Knowledge Graphs (KGs) enable the integration and representation of complex information across domains, but their semantic richness and structural complexity create substantial barriers for lay users without expertise in semantic web technologies. When encountering an unfamiliar KG, such users face a distinct orientation challenge: they do not know what questions are possible, how the knowledge is structured, or how to begin exploration. This paper identifies and theorises this phenomenon as the Initial Exploration Problem (IEP). Drawing on theories from information behaviour and human-computer interaction, including ASK, exploratory search, information foraging, and cognitive load theory, we develop a conceptual framing of the IEP characterised by three interdependent barriers: scope uncertainty, ontology opacity, and query incapacity. We argue that these barriers converge at the moment of first contact, distinguishing the IEP from related concepts that presuppose an existing starting point or information goal. Analysing KG exploration interfaces at the level of interaction primitives, we suggest that many systems rely on epistemic assumptions that do not hold at first contact. This reveals a structural gap in the design space: the absence of interaction primitives for scope revelation, mechanisms that communicate what a KG contains without requiring users to formulate queries or interpret ontological structures. In articulating the IEP, this paper provides a theoretical lens for evaluating KG interfaces and for designing entry-point scaffolding that supports initial exploration. 

---
# Motivation is Something You Need 

**Authors**: Mehdi Acheli, Walid Gaaloul  

**Link**: [PDF](https://arxiv.org/pdf/2602.21064)  

**Abstract**: This work introduces a novel training paradigm that draws from affective neuroscience. Inspired by the interplay of emotions and cognition in the human brain and more specifically the SEEKING motivational state, we design a dual-model framework where a smaller base model is trained continuously, while a larger motivated model is activated intermittently during predefined "motivation conditions". The framework mimics the emotional state of high curiosity and anticipation of reward in which broader brain regions are recruited to enhance cognitive performance. Exploiting scalable architectures where larger models extend smaller ones, our method enables shared weight updates and selective expansion of network capacity during noteworthy training steps. Empirical evaluation on the image classification task demonstrates that, not only does the alternating training scheme efficiently and effectively enhance the base model compared to a traditional scheme, in some cases, the motivational model also surpasses its standalone counterpart despite seeing less data per epoch. This opens the possibility of simultaneously training two models tailored to different deployment constraints with competitive or superior performance while keeping training cost lower than when training the larger model. 

---
# Tool Building as a Path to "Superintelligence" 

**Authors**: David Koplow, Tomer Galanti, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2602.21061)  

**Abstract**: The Diligent Learner framework suggests LLMs can achieve superintelligence via test-time search, provided a sufficient step-success probability $\gamma$. In this work, we design a benchmark to measure $\gamma$ on logical out-of-distribution inference. We construct a class of tasks involving GF(2) circuit reconstruction that grow more difficult with each reasoning step, and that are, from an information-theoretic standpoint, impossible to reliably solve unless the LLM carefully integrates all of the information provided. Our analysis demonstrates that while the $\gamma$ value for small LLMs declines superlinearly as depth increases, frontier models exhibit partial robustness on this task. Furthermore, we find that successful reasoning at scale is contingent upon precise tool calls, identifying tool design as a critical capability for LLMs to achieve general superintelligence through the Diligent Learner framework. 

---
# LogicGraph : Benchmarking Multi-Path Logical Reasoning via Neuro-Symbolic Generation and Verification 

**Authors**: Yanrui Wu, Lingling Zhang, Xinyu Zhang, Jiayu Chang, Pengyu Li, Xu Jiang, Jingtao Hu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21044)  

**Abstract**: Evaluations of large language models (LLMs) primarily emphasize convergent logical reasoning, where success is defined by producing a single correct proof. However, many real-world reasoning problems admit multiple valid derivations, requiring models to explore diverse logical paths rather than committing to one route. To address this limitation, we introduce LogicGraph, the first benchmark aimed to systematically evaluate multi-path logical reasoning, constructed via a neuro-symbolic framework that leverages backward logic generation and semantic instantiation. This pipeline yields solver-verified reasoning problems formalized by high-depth multi-path reasoning and inherent logical distractions, where each instance is associated with an exhaustive set of minimal proofs. We further propose a reference-free evaluation framework to rigorously assess model performance in both convergent and divergent regimes. Experiments on state-of-the-art language models reveal a common limitation: models tend to commit early to a single route and fail to explore alternatives, and the coverage gap grows substantially with reasoning depth. LogicGraph exposes this divergence gap and provides actionable insights to motivate future improvements. Our code and data will be released at this https URL. 

---
# Architecting AgentOS: From Token-Level Context to Emergent System-Level Intelligence 

**Authors**: ChengYou Li, XiaoDong Liu, XiangBao Meng, XinYu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20934)  

**Abstract**: The paradigm of Large Language Models is undergoing a fundamental transition from static inference engines to dynamic autonomous cognitive this http URL current research primarily focuses on scaling context windows or optimizing prompt engineering the theoretical bridge between micro scale token processing and macro scale systemic intelligence remains this http URL paper proposes AgentOS,a holistic conceptual framework that redefines the LLM as a "Reasoning Kernel" governed by structured operating system this http URL to this architecture is Deep Context Management which conceptualizes the context window as an Addressable Semantic Space rather than a passive this http URL systematically deconstruct the transition from discrete sequences to coherent cognitive states introducing mechanisms for Semantic Slicing and Temporal Alignment to mitigate cognitive drift in multi-agent this http URL mapping classical OS abstractions such as memory paging interrupt handling and process scheduling onto LLM native constructs, this review provides a rigorous roadmap for architecting resilient scalable and self-evolving cognitive this http URL analysis asserts that the next frontier of AGI development lies in the architectural efficiency of system-level coordination. 

---
# HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG 

**Authors**: Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20926)  

**Abstract**: Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop reasoning. Graph-based RAG approaches attempt to bridge this gap, yet they typically face trade-offs between accuracy and efficiency due to challenges such as costly graph traversals and semantic noise in LLM-generated summaries. In this paper, we propose HyperNode Expansion and Logical Path-Guided Evidence Localization strategies for GraphRAG (HELP), a novel framework designed to balance accuracy with practical efficiency through two core strategies: 1) HyperNode Expansion, which iteratively chains knowledge triplets into coherent reasoning paths abstracted as HyperNodes to capture complex structural dependencies and ensure retrieval accuracy; and 2) Logical Path-Guided Evidence Localization, which leverages precomputed graph-text correlations to map these paths directly to the corpus for superior efficiency. HELP avoids expensive random walks and semantic distortion, preserving knowledge integrity while drastically reducing retrieval latency. Extensive experiments demonstrate that HELP achieves competitive performance across multiple simple and multi-hop QA benchmarks and up to a 28.8$\times$ speedup over leading Graph-based RAG baselines. 

---
# Predicting Sentence Acceptability Judgments in Multimodal Contexts 

**Authors**: Hyewon Jang, Nikolai Ilinykh, Sharid Loáiciga, Jey Han Lau, Shalom Lappin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20918)  

**Abstract**: Previous work has examined the capacity of deep neural networks (DNNs), particularly transformers, to predict human sentence acceptability judgments, both independently of context, and in document contexts. We consider the effect of prior exposure to visual images (i.e., visual context) on these judgments for humans and large language models (LLMs). Our results suggest that, in contrast to textual context, visual images appear to have little if any impact on human acceptability ratings. However, LLMs display the compression effect seen in previous work on human judgments in document contexts. Different sorts of LLMs are able to predict human acceptability judgments to a high degree of accuracy, but in general, their performance is slightly better when visual contexts are removed. Moreover, the distribution of LLM judgments varies among models, with Qwen resembling human patterns, and others diverging from them. LLM-generated predictions on sentence acceptability are highly correlated with their normalised log probabilities in general. However, the correlations decrease when visual contexts are present, suggesting that a higher gap exists between the internal representations of LLMs and their generated predictions in the presence of visual contexts. Our experimental work suggests interesting points of similarity and of difference between human and LLM processing of sentences in multimodal contexts. 

---
# Diagnosing Causal Reasoning in Vision-Language Models via Structured Relevance Graphs 

**Authors**: Dhita Putri Pratama, Soyeon Caren Han, Yihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.20878)  

**Abstract**: Large Vision-Language Models (LVLMs) achieve strong performance on visual question answering benchmarks, yet often rely on spurious correlations rather than genuine causal reasoning. Existing evaluations primarily assess the correctness of the answers, making it unclear whether failures arise from limited reasoning capability or from misidentifying causally relevant information. We introduce Vision-Language Causal Graphs (VLCGs), a structured, query-conditioned representation that explicitly encodes causally relevant objects, attributes, relations, and scene-grounded assumptions. Building on this representation, we present ViLCaR, a diagnostic benchmark comprising tasks for Causal Attribution, Causal Inference, and Question Answering, along with graph-aligned evaluation metrics that assess relevance identification beyond final answer accuracy. Experiments in state-of-the-art LVLMs show that injecting structured relevance information significantly improves attribution and inference consistency compared to zero-shot and standard in-context learning. These findings suggest that current limitations in LVLM causal reasoning stem primarily from insufficient structural guidance rather than a lack of reasoning capacity. 

---
# Pressure Reveals Character: Behavioural Alignment Evaluation at Depth 

**Authors**: Nora Petrova, John Burden  

**Link**: [PDF](https://arxiv.org/pdf/2602.20813)  

**Abstract**: Evaluating alignment in language models requires testing how they behave under realistic pressure, not just what they claim they would do. While alignment failures increasingly cause real-world harm, comprehensive evaluation frameworks with realistic multi-turn scenarios remain lacking. We introduce an alignment benchmark spanning 904 scenarios across six categories -- Honesty, Safety, Non-Manipulation, Robustness, Corrigibility, and Scheming -- validated as realistic by human raters. Our scenarios place models under conflicting instructions, simulated tool access, and multi-turn escalation to reveal behavioural tendencies that single-turn evaluations miss. Evaluating 24 frontier models using LLM judges validated against human annotations, we find that even top-performing models exhibit gaps in specific categories, while the majority of models show consistent weaknesses across the board. Factor analysis reveals that alignment behaves as a unified construct (analogous to the g-factor in cognitive research) with models scoring high on one category tending to score high on others. We publicly release the benchmark and an interactive leaderboard to support ongoing evaluation, with plans to expand scenarios in areas where we observe persistent weaknesses and to add new models as they are released. 

---
# Qwen-BIM: developing large language model for BIM-based design with domain-specific benchmark and dataset 

**Authors**: Jia-Rui Lin, Yun-Hong Cai, Xiang-Rui Ni, Shaojie Zhou, Peng Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20812)  

**Abstract**: As the construction industry advances toward digital transformation, BIM (Building Information Modeling)-based design has become a key driver supporting intelligent construction. Despite Large Language Models (LLMs) have shown potential in promoting BIM-based design, the lack of specific datasets and LLM evaluation benchmarks has significantly hindered the performance of LLMs. Therefore, this paper addresses this gap by proposing: 1) an evaluation benchmark for BIM-based design together with corresponding quantitative indicators to evaluate the performance of LLMs, 2) a method for generating textual data from BIM and constructing corresponding BIM-derived datasets for LLM evaluation and fine-tuning, and 3) a fine-tuning strategy to adapt LLMs for BIM-based design. Results demonstrate that the proposed domain-specific benchmark effectively and comprehensively assesses LLM capabilities, highlighting that general LLMs are still incompetent for domain-specific tasks. Meanwhile, with the proposed benchmark and datasets, Qwen-BIM is developed and achieves a 21.0% average increase in G-Eval score compared to the base LLM model. Notably, with only 14B parameters, performance of Qwen-BIM is comparable to that of general LLMs with 671B parameters for BIM-based design tasks. Overall, this study develops the first domain-specific LLM for BIM-based design by introducing a comprehensive benchmark and high-quality dataset, which provide a solid foundation for developing BIM-related LLMs in various fields. 

---
# POMDPPlanners: Open-Source Package for POMDP Planning 

**Authors**: Yaacov Pariente, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2602.20810)  

**Abstract**: We present POMDPPlanners, an open-source Python package for empirical evaluation of Partially Observable Markov Decision Process (POMDP) planning algorithms. The package integrates state-of-the-art planning algorithms, a suite of benchmark environments with safety-critical variants, automated hyperparameter optimization via Optuna, persistent caching with failure recovery, and configurable parallel simulation -- reducing the overhead of extensive simulation studies. POMDPPlanners is designed to enable scalable, reproducible research on decision-making under uncertainty, with particular emphasis on risk-sensitive settings where standard toolkits fall short. 

---
# Pipeline for Verifying LLM-Generated Mathematical Solutions 

**Authors**: Varvara Sazonova, Dmitri Shmelkin, Stanislav Kikot, Vasily Motolygin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20770)  

**Abstract**: With the growing popularity of Large Reasoning Models and their results in solving mathematical problems, it becomes crucial to measure their capabilities. We introduce a pipeline for both automatic and interactive verification as a more accurate alternative to only checking the answer which is currently the most popular approach for benchmarks. The pipeline can also be used as a generator of correct solutions both in formal and informal languages. 3 AI agents, which can be chosen for the benchmark accordingly, are included in the structure. The key idea is the use of prompts to obtain the solution in the specific form which allows for easier verification using proof assistants and possible use of small models ($\le 8B$). Experiments on several datasets suggest low probability of False Positives. The open-source implementation with instructions on setting up a server is available at this https URL. 

---
# PyVision-RL: Forging Open Agentic Vision Models via RL 

**Authors**: Shitian Zhao, Shaoheng Lin, Ming Li, Haoquan Zhang, Wenshuo Peng, Kaipeng Zhang, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2602.20739)  

**Abstract**: Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce PyVision-RL, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling-filtering-ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop PyVision-Image and PyVision-Video for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents. 

---
# CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference 

**Authors**: Chao Fei, Guozhong Li, Chenxi Liu, Panos Kalnis  

**Link**: [PDF](https://arxiv.org/pdf/2602.20732)  

**Abstract**: Long-context LLMs demand accurate inference at low latency, yet decoding becomes primarily constrained by KV cache as context grows. Prior pruning methods are largely context-agnostic: their token selection ignores step-wise relevance and local semantics, which undermines quality. Moreover, their irregular accesses and selection overheads yield only limited wall-clock speedups. To address this, we propose \textbf{CHESS}, an \textit{algorithm-system co-design} KV-cache management system. Algorithmically, CHESS introduces a context-aware, hierarchical selection policy that dynamically reconstructs a coherent context for the current decoding. System-wise, coarse granularity selection eliminates expensive data movement, fully realizing practical acceleration from theoretical sparsity. Extensive evaluations demonstrate that CHESS surpasses Full-KV quality using only \textbf{1\%} of the KV cache, delivers low-latency stable inference with up to \textbf{4.56$\times$} higher throughput, and consistently outperforms other strong baselines. Code is available at \href{this https URL}{this https URL}. 

---
# Balancing Multiple Objectives in Urban Traffic Control with Reinforcement Learning from AI Feedback 

**Authors**: Chenyang Zhao, Vinny Cahill, Ivana Dusparic  

**Link**: [PDF](https://arxiv.org/pdf/2602.20728)  

**Abstract**: Reward design has been one of the central challenges for real world reinforcement learning (RL) deployment, especially in settings with multiple objectives. Preference-based RL offers an appealing alternative by learning from human preferences over pairs of behavioural outcomes. More recently, RL from AI feedback (RLAIF) has demonstrated that large language models (LLMs) can generate preference labels at scale, mitigating the reliance on human annotators. However, existing RLAIF work typically focuses only on single-objective tasks, leaving the open question of how RLAIF handles systems that involve multiple objectives. In such systems trade-offs among conflicting objectives are difficult to specify, and policies risk collapsing into optimizing for a dominant goal. In this paper, we explore the extension of the RLAIF paradigm to multi-objective self-adaptive systems. We show that multi-objective RLAIF can produce policies that yield balanced trade-offs reflecting different user priorities without laborious reward engineering. We argue that integrating RLAIF into multi-objective RL offers a scalable path toward user-aligned policy learning in domains with inherently conflicting objectives. 

---
# Modality-Guided Mixture of Graph Experts with Entropy-Triggered Routing for Multimodal Recommendation 

**Authors**: Ji Dai, Quan Fang, Dengsheng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2602.20723)  

**Abstract**: Multimodal recommendation enhances ranking by integrating user-item interactions with item content, which is particularly effective under sparse feedback and long-tail distributions. However, multimodal signals are inherently heterogeneous and can conflict in specific contexts, making effective fusion both crucial and challenging. Existing approaches often rely on shared fusion pathways, leading to entangled representations and modality imbalance. To address these issues, we propose \textbf{MAGNET}, a \textbf{M}odality-Guided Mixture of \textbf{A}daptive \textbf{G}raph Experts \textbf{N}etwork with Progressive \textbf{E}ntropy-\textbf{T}riggered Routing for Multimodal Recommendation, designed to enhance controllability, stability, and interpretability in multimodal fusion. MAGNET couples interaction-conditioned expert routing with structure-aware graph augmentation, so that both \emph{what} to fuse and \emph{how} to fuse are explicitly controlled and interpretable. At the representation level, a dual-view graph learning module augments the interaction graph with content-induced edges, improving coverage for sparse and long-tail items while preserving collaborative structure via parallel encoding and lightweight fusion. At the fusion level, MAGNET employs structured experts with explicit modality roles -- dominant, balanced, and complementary -- enabling a more interpretable and adaptive combination of behavioral, visual, and textual cues. To further stabilize sparse routing and prevent expert collapse, we introduce a two-stage entropy-weighting mechanism that monitors routing entropy. This mechanism automatically transitions training from an early coverage-oriented regime to a later specialization-oriented regime, progressively balancing expert utilization and routing confidence. Extensive experiments on public benchmarks demonstrate consistent improvements over strong baselines. 

---
# Buffer Matters: Unleashing the Power of Off-Policy Reinforcement Learning in Large Language Model Reasoning 

**Authors**: Xu Wan, Yansheng Wang, Wenqi Huang, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.20722)  

**Abstract**: Traditional on-policy Reinforcement Learning with Verifiable Rewards (RLVR) frameworks suffer from experience waste and reward homogeneity, which directly hinders learning efficiency on difficult samples during large language models post-training. In this paper, we introduce Batch Adaptation Policy Optimization (BAPO), an off-policy RLVR framework to improve the data efficiency in large language models post-training. It dynamically selects training batches by re-evaluating historically difficult samples and reusing high-quality ones, while holding a lower bound guarantee for policy improvement. Extensive experiments further demonstrate that BAPO achieves an average 12.5% improvement over GRPO across mathematics, planning, and visual reasoning tasks. Crucially, BAPO successfully resolves 40.7% of problems that base models consistently fail to solve. 

---
# Counterfactual Simulation Training for Chain-of-Thought Faithfulness 

**Authors**: Peter Hase, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2602.20710)  

**Abstract**: Inspecting Chain-of-Thought reasoning is among the most common means of understanding why an LLM produced its output. But well-known problems with CoT faithfulness severely limit what insights can be gained from this practice. In this paper, we introduce a training method called Counterfactual Simulation Training (CST), which aims to improve CoT faithfulness by rewarding CoTs that enable a simulator to accurately predict a model's outputs over counterfactual inputs. We apply CST in two settings: (1) CoT monitoring with cue-based counterfactuals, to detect when models rely on spurious features, reward hack, or are sycophantic, and (2) counterfactual simulation over generic model-based counterfactuals, to encourage models to produce more faithful, generalizable reasoning in the CoT. Experiments with models up to 235B parameters show that CST can substantially improve monitor accuracy on cue-based counterfactuals (by 35 accuracy points) as well as simulatability over generic counterfactuals (by 2 points). We further show that: (1) CST outperforms prompting baselines, (2) rewriting unfaithful CoTs with an LLM is 5x more efficient than RL alone, (3) faithfulness improvements do not generalize to dissuading cues (as opposed to persuading cues), and (4) larger models do not show more faithful CoT out of the box, but they do benefit more from CST. These results suggest that CST can improve CoT faithfulness in general, with promising applications for CoT monitoring. Code for experiments in this paper is available at this https URL 

---
# ICON: Indirect Prompt Injection Defense for Agents based on Inference-Time Correction 

**Authors**: Che Wang, Fuyao Zhang, Jiaming Zhang, Ziqi Zhang, Yinghui Wang, Longtao Huang, Jianbo Gao, Zhong Chen, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.20708)  

**Abstract**: Large Language Model (LLM) agents are susceptible to Indirect Prompt Injection (IPI) attacks, where malicious instructions in retrieved content hijack the agent's execution. Existing defenses typically rely on strict filtering or refusal mechanisms, which suffer from a critical limitation: over-refusal, prematurely terminating valid agentic workflows. We propose ICON, a probing-to-mitigation framework that neutralizes attacks while preserving task continuity. Our key insight is that IPI attacks leave distinct over-focusing signatures in the latent space. We introduce a Latent Space Trace Prober to detect attacks based on high intensity scores. Subsequently, a Mitigating Rectifier performs surgical attention steering that selectively manipulate adversarial query key dependencies while amplifying task relevant elements to restore the LLM's functional trajectory. Extensive evaluations on multiple backbones show that ICON achieves a competitive 0.4% ASR, matching commercial grade detectors, while yielding a over 50% task utility gain. Furthermore, ICON demonstrates robust Out of Distribution(OOD) generalization and extends effectively to multi-modal agents, establishing a superior balance between security and efficiency. 

---
# Online Algorithms with Unreliable Guidance 

**Authors**: Julien Dallot, Yuval Emek, Yuval Gil, Maciej Pacut, Stefan Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2602.20706)  

**Abstract**: This paper introduces a new model for ML-augmented online decision making, called online algorithms with unreliable guidance (OAG). This model completely separates between the predictive and algorithmic components, thus offering a single well-defined analysis framework that relies solely on the considered problem. Formulated through the lens of request-answer games, an OAG algorithm receives, with each incoming request, a piece of guidance which is taken from the problem's answer space; ideally, this guidance is the optimal answer for the current request, however with probability $\beta$, the guidance is adversarially corrupted. The goal is to develop OAG algorithms that admit good competitiveness when $\beta = 0$ (a.k.a. consistency) as well as when $\beta = 1$ (a.k.a. robustness); the appealing notion of smoothness, that in most prior work required a dedicated loss function, now arises naturally as $\beta$ shifts from $0$ to $1$.
We then describe a systematic method, called the drop or trust blindly (DTB) compiler, which transforms any online algorithm into a learning-augmented online algorithm in the OAG model. Given a prediction-oblivious online algorithm, its learning-augmented counterpart produced by applying the DTB compiler either follows the incoming guidance blindly or ignores it altogether and proceeds as the initial algorithm would have; the choice between these two alternatives is based on the outcome of a (biased) coin toss. As our main technical contribution, we prove (rigorously) that although remarkably simple, the class of algorithms produced via the DTB compiler includes algorithms with attractive consistency-robustness guarantees for three classic online problems: for caching and uniform metrical task systems our algorithms are optimal, whereas for bipartite matching (with adversarial arrival order), our algorithm outperforms the state-of-the-art. 

---
# PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding 

**Authors**: Baolong Bi, Yuyao Ge, Shenghua Liu, Yuchen He, Siqian Tong, Lizhe Chen, Lingrui Mei, Zehao Li, Yiwei Wang, Yujun Cai, Ming-Hsuan Yang, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.20696)  

**Abstract**: Reliable AI systems require large language models (LLMs) to exhibit behaviors aligned with human preferences and values. However, most existing alignment approaches operate at training time and rely on additional high-quality data, incurring significant computational and annotation costs. While recent work has shown that contrastive decoding can leverage a model's internal distributions to improve specific capabilities, its applicability remains limited to narrow behavioral scopes and scenarios. In this work, we introduce Polarity-Prompt Contrastive Decoding (PromptCD), a test-time behavior control method that generalizes contrastive decoding to broader enhancement settings. PromptCD constructs paired positive and negative guiding prompts for a target behavior and contrasts model responses-specifically token-level probability distributions in LLMs and visual attention patterns in VLMs-to reinforce desirable outcomes. This formulation extends contrastive decoding to a wide range of enhancement objectives and is applicable to both LLMs and Vision-Language Models (VLMs) without additional training. For LLMs, experiments on the "3H" alignment objectives (helpfulness, honesty, and harmlessness) demonstrate consistent and substantial improvements, indicating that post-trained models can achieve meaningful self-enhancement purely at test time. For VLMs, we further analyze contrastive effects on visual attention, showing that PromptCD significantly improves VQA performance by reinforcing behavior-consistent visual grounding. Collectively, these results highlight PromptCD as a simple, general, and cost-efficient strategy for reliable behavior control across modalities. 

---
# How Foundational Skills Influence VLM-based Embodied Agents:A Native Perspective 

**Authors**: Bo Peng, Pi Bu, Keyu Pan, Xinrun Xu, Yinxiu Zhao, Miao Chen, Yang Du, Lin Li, Jun Song, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20687)  

**Abstract**: Recent advances in vision-language models (VLMs) have shown promise for human-level embodied intelligence. However, existing benchmarks for VLM-driven embodied agents often rely on high-level commands or discretized action spaces, which are non-native settings that differ markedly from real-world control. In addition, current benchmarks focus primarily on high-level tasks and lack joint evaluation and analysis at both low and high levels. To address these limitations, we present NativeEmbodied, a challenging benchmark for VLM-driven embodied agents that uses a unified, native low-level action space. Built on diverse simulated scenes, NativeEmbodied includes three representative high-level tasks in complex scenarios to evaluate overall performance. For more detailed analysis, we further decouple the skills required by complex tasks and construct four types of low-level tasks, each targeting a fundamental embodied skill. This joint evaluation across task and skill granularities enables fine-grained assessment of embodied agents. Experiments with state-of-the-art VLMs reveal clear deficiencies in several fundamental embodied skills, and further analysis shows that these bottlenecks significantly limit performance on high-level tasks. NativeEmbodied highlights key challenges for current VLM-driven embodied agents and provides insights to guide future research. 

---
# Recursive Belief Vision Language Model 

**Authors**: Vaidehi Bagaria, Bijo Sebastian, Nirav Patel  

**Link**: [PDF](https://arxiv.org/pdf/2602.20659)  

**Abstract**: Current vision-language-action (VLA) models struggle with long-horizon manipulation under partial observability. Most existing approaches remain observation-driven, relying on short context windows or repeated queries to vision-language models (VLMs). This leads to loss of task progress, action repetition under perceptual aliasing, and high inference latency. Semantic reasoning alone is not the primary bottleneck in long-horizon manipulation. Instead, VLAs lack persistent, action-conditioned state representations and exhibit limited temporal and physical reasoning, making them ill-suited for multi-stage control. This paper introduces RB-VLA, a belief-centric architecture trained with self-supervised world-model objectives that maintains a compact latent state encoding task-relevant history, dynamics, and object interactions. Queried once for high-level intent, the VLM provides task specification, while the belief tracks task progress and enables phase-aware, causally grounded control under partial observability without storing raw observations or scaling memory with time. The belief and intent jointly condition a diffusion policy for robust closed-loop execution. RB-VLA outperforms prior VLAs on long-horizon benchmarks, achieving 52.5% and 37.5% higher success on multi-stage pick-and-place and stacking tasks, respectively, compared to {\pi}0. It also reduces inference latency by up to 5x relative to baselines and eliminates memory growth across timesteps observed in existing VLAs. Ablations show that the belief module is the primary driver of performance, increasing success rates from 32.5% to 77.5%. These results demonstrate the effectiveness of belief-based state representations for long-horizon VLA policies. 

---
# Grounding LLMs in Scientific Discovery via Embodied Actions 

**Authors**: Bo Zhang, Jinfeng Zhou, Yuxuan Chen, Jianing Yin, Minlie Huang, Hongning Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20639)  

**Abstract**: Large Language Models (LLMs) have shown significant potential in scientific discovery but struggle to bridge the gap between theoretical reasoning and verifiable physical simulation. Existing solutions operate in a passive "execute-then-response" loop and thus lacks runtime perception, obscuring agents to transient anomalies (e.g., numerical instability or diverging oscillations). To address this limitation, we propose EmbodiedAct, a framework that transforms established scientific software into active embodied agents by grounding LLMs in embodied actions with a tight perception-execution loop. We instantiate EmbodiedAct within MATLAB and evaluate it on complex engineering design and scientific modeling tasks. Extensive experiments show that EmbodiedAct significantly outperforms existing baselines, achieving SOTA performance by ensuring satisfactory reliability and stability in long-horizon simulations and enhanced accuracy in scientific modeling. 

---
# Identifying two piecewise linear additive value functions from anonymous preference information 

**Authors**: Vincent Auriau, Khaled Belahcene, Emmanuel Malherbe, Vincent Mousseau, Marc Pirlot  

**Link**: [PDF](https://arxiv.org/pdf/2602.20638)  

**Abstract**: Eliciting a preference model involves asking a person, named decision-maker, a series of questions. We assume that these preferences can be represented by an additive value function. In this work, we query simultaneously two decision-makers in the aim to elicit their respective value functions. For each query we receive two answers, without noise, but without knowing which answer corresponds to which this http URL propose an elicitation procedure that identifies the two preference models when the marginal value functions are piecewise linear with known breaking points. 

---
# When can we trust untrusted monitoring? A safety case sketch across collusion strategies 

**Authors**: Nelson Gardner-Challis, Jonathan Bostock, Georgiy Kozhevnikov, Morgan Sinclaire, Joan Velja, Alessandro Abate, Charlie Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20628)  

**Abstract**: AIs are increasingly being deployed with greater autonomy and capabilities, which increases the risk that a misaligned AI may be able to cause catastrophic harm. Untrusted monitoring -- using one untrusted model to oversee another -- is one approach to reducing risk. Justifying the safety of an untrusted monitoring deployment is challenging because developers cannot safely deploy a misaligned model to test their protocol directly. In this paper, we develop upon existing methods for rigorously demonstrating safety based on pre-deployment testing. We relax assumptions that previous AI control research made about the collusion strategies a misaligned AI might use to subvert untrusted monitoring. We develop a taxonomy covering passive self-recognition, causal collusion (hiding pre-shared signals), acausal collusion (hiding signals via Schelling points), and combined strategies. We create a safety case sketch to clearly present our argument, explicitly state our assumptions, and highlight unsolved challenges. We identify conditions under which passive self-recognition could be a more effective collusion strategy than those studied previously. Our work builds towards more robust evaluations of untrusted monitoring. 

---
# Physics-based phenomenological characterization of cross-modal bias in multimodal models 

**Authors**: Hyeongmo Kim, Sohyun Kang, Yerin Choi, Seungyeon Ji, Junhyuk Woo, Hyunsuk Chung, Soyeon Caren Han, Kyungreem Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.20624)  

**Abstract**: The term 'algorithmic fairness' is used to evaluate whether AI models operate fairly in both comparative (where fairness is understood as formal equality, such as "treat like cases as like") and non-comparative (where unfairness arises from the model's inaccuracy, arbitrariness, or inscrutability) contexts. Recent advances in multimodal large language models (MLLMs) are breaking new ground in multimodal understanding, reasoning, and generation; however, we argue that inconspicuous distortions arising from complex multimodal interaction dynamics can lead to systematic bias. The purpose of this position paper is twofold: first, it is intended to acquaint AI researchers with phenomenological explainable approaches that rely on the physical entities that the machine experiences during training/inference, as opposed to the traditional cognitivist symbolic account or metaphysical approaches; second, it is to state that this phenomenological doctrine will be practically useful for tackling algorithmic fairness issues in MLLMs. We develop a surrogate physics-based model that describes transformer dynamics (i.e., semantic network structure and self-/cross-attention) to analyze the dynamics of cross-modal bias in MLLM, which are not fully captured by conventional embedding- or representation-level analyses. We support this position through multi-input diagnostic experiments: 1) perturbation-based analyses of emotion classification using Qwen2.5-Omni and Gemma 3n, and 2) dynamical analysis of Lorenz chaotic time-series prediction through the physical surrogate. Across two architecturally distinct MLLMs, we show that multimodal inputs can reinforce modality dominance rather than mitigate it, as revealed by structured error-attractor patterns under systematic label perturbation, complemented by dynamical analysis. 

---
# CausalReasoningBenchmark: A Real-World Benchmark for Disentangled Evaluation of Causal Identification and Estimation 

**Authors**: Ayush Sawarni, Jiyuan Tan, Vasilis Syrgkanis  

**Link**: [PDF](https://arxiv.org/pdf/2602.20571)  

**Abstract**: Many benchmarks for automated causal inference evaluate a system's performance based on a single numerical output, such as an Average Treatment Effect (ATE). This approach conflates two distinct steps in causal analysis: identification-formulating a valid research design under stated assumptions-and estimation-implementing that design numerically on finite data. We introduce CausalReasoningBenchmark, a benchmark of 173 queries across 138 real-world datasets, curated from 85 peer-reviewed research papers and four widely-used causal-inference textbooks. For each query a system must produce (i) a structured identification specification that names the strategy, the treatment, outcome, and control variables, and all design-specific elements, and (ii) a point estimate with a standard error. By scoring these two components separately, our benchmark enables granular diagnosis: it distinguishes failures in causal reasoning from errors in numerical execution. Baseline results with a state-of-the-art LLM show that, while the model correctly identifies the high-level strategy in 84 % of cases, full identification-specification correctness drops to only 30 %, revealing that the bottleneck lies in the nuanced details of research design rather than in computation. CausalReasoningBenchmark is publicly available on Hugging Face and is designed to foster the development of more robust automated causal-inference systems. 

---
# From Logs to Language: Learning Optimal Verbalization for LLM-Based Recommendation in Production 

**Authors**: Yucheng Shi, Ying Li, Yu Wang, Yesu Feng, Arjun Rao, Rein Houthooft, Shradha Sehgal, Jin Wang, Hao Zhen, Ninghao Liu, Linas Baltrunas  

**Link**: [PDF](https://arxiv.org/pdf/2602.20558)  

**Abstract**: Large language models (LLMs) are promising backbones for generative recommender systems, yet a key challenge remains underexplored: verbalization, i.e., converting structured user interaction logs into effective natural language inputs. Existing methods rely on rigid templates that simply concatenate fields, yielding suboptimal representations for recommendation. We propose a data-centric framework that learns verbalization for LLM-based recommendation. Using reinforcement learning, a verbalization agent transforms raw interaction histories into optimized textual contexts, with recommendation accuracy as the training signal. This agent learns to filter noise, incorporate relevant metadata, and reorganize information to improve downstream predictions. Experiments on a large-scale industrial streaming dataset show that learned verbalization delivers up to 93% relative improvement in discovery item recommendation accuracy over template-based baselines. Further analysis reveals emergent strategies such as user interest summarization, noise removal, and syntax normalization, offering insights into effective context construction for LLM-based recommender systems. 

---
# Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI coordination 

**Authors**: Rakshit Trivedi, Kartik Sharma, David C Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2602.20517)  

**Abstract**: Effective human-AI coordination requires artificial agents capable of exhibiting and responding to human-like behaviors while adapting to changing contexts. Imitation learning has emerged as one of the prominent approaches to build such agents by training them to mimic human-demonstrated behaviors. However, current methods struggle to capture the inherent diversity and non-Markovian nature of human behavior and lack the ability to steer behavior at inference time. Drawing inspiration from the theory of human cognitive processes, where inner speech guides action selection before execution, we propose MIMIC (Modeling Inner Motivations for Imitation and Control), a framework that uses language as an internal representation of behavioral intent. MIMIC employs the novel use of vision-language models as linguistic scaffolding to train a conditional variational autoencoder capable of generating inner speech from observations. A diffusion-based behavior cloning policy then selects actions conditioned on current observations and the generated inner speech. MIMIC enables fine-grained steering of behavior at inference time by conditioning the agent on behavior-specific speech. Experiments across robotic manipulation tasks and human-AI collaboration games demonstrate that MIMIC significantly enhances both behavior diversity and fidelity to human demonstrations while enabling nuanced behavioral steering without training on additional demonstrations. We open source our code and provide pre-trained MIMIC agents and qualitative demos at: this https URL. 

---
# ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory 

**Authors**: Hongbin Zhong, Fazle Faisal, Luis França, Tanakorn Leesatapornwongsa, Adriana Szekeres, Kexin Rong, Suman Nath  

**Link**: [PDF](https://arxiv.org/pdf/2602.20502)  

**Abstract**: Existing Graphical User Interface (GUI) agents operate through step-by-step calls to vision language models--taking a screenshot, reasoning about the next action, executing it, then repeating on the new page--resulting in high costs and latency that scale with the number of reasoning steps, and limited accuracy due to no persistent memory of previously visited pages.
We propose ActionEngine, a training-free framework that transitions from reactive execution to programmatic planning through a novel two-agent architecture: a Crawling Agent that constructs an updatable state-machine memory of the GUIs through offline exploration, and an Execution Agent that leverages this memory to synthesize complete, executable Python programs for online task execution.
To ensure robustness against evolving interfaces, execution failures trigger a vision-based re-grounding fallback that repairs the failed action and updates the memory.
This design drastically improves both efficiency and accuracy: on Reddit tasks from the WebArena benchmark, our agent achieves 95% task success with on average a single LLM call, compared to 66% for the strongest vision-only baseline, while reducing cost by 11.8x and end-to-end latency by 2x.
Together, these components yield scalable and reliable GUI interaction by combining global programmatic planning, crawler-validated action templates, and node-level execution with localized validation and repair. 

---
# KairosVL: Orchestrating Time Series and Semantics for Unified Reasoning 

**Authors**: Haotian Si, Changhua Pei, Xiao He, Zeyan Li, Zhe Xie, Zexin Wang, Jiyao Hu, Zhaoyang Yu, Tieying Zhang, Dan Pei, Jianhui Li, Gaogang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2602.20494)  

**Abstract**: Driven by the increasingly complex and decision-oriented demands of time series analysis, we introduce the Semantic-Conditional Time Series Reasoning task, which extends conventional time series analysis beyond purely numerical modeling to incorporate contextual and semantic understanding. To further enhance the mode's reasoning capabilities on complex time series problems, we propose a two-round reinforcement learning framework: the first round strengthens the mode's perception of fundamental temporal primitives, while the second focuses on semantic-conditioned reasoning. The resulting model, KairosVL, achieves competitive performance across both synthetic and real-world tasks. Extensive experiments and ablation studies demonstrate that our framework not only boosts performance but also preserves intrinsic reasoning ability and significantly improves generalization to unseen scenarios. To summarize, our work highlights the potential of combining semantic reasoning with temporal modeling and provides a practical framework for real-world time series intelligence, which is in urgent demand. 

---
# PreScience: A Benchmark for Forecasting Scientific Contributions 

**Authors**: Anirudh Ajith, Amanpreet Singh, Jay DeYoung, Nadav Kunievsky, Austin C. Kozlowski, Oyvind Tafjord, James Evans, Daniel S. Weld, Tom Hope, Doug Downey  

**Link**: [PDF](https://arxiv.org/pdf/2602.20459)  

**Abstract**: Can AI systems trained on the scientific record up to a fixed point in time forecast the scientific advances that follow? Such a capability could help researchers identify collaborators and impactful research directions, and anticipate which problems and methods will become central next. We introduce PreScience -- a scientific forecasting benchmark that decomposes the research process into four interdependent generative tasks: collaborator prediction, prior work selection, contribution generation, and impact prediction. PreScience is a carefully curated dataset of 98K recent AI-related research papers, featuring disambiguated author identities, temporally aligned scholarly metadata, and a structured graph of companion author publication histories and citations spanning 502K total papers. We develop baselines and evaluations for each task, including LACERScore, a novel LLM-based measure of contribution similarity that outperforms previous metrics and approximates inter-annotator agreement. We find substantial headroom remains in each task -- e.g. in contribution generation, frontier LLMs achieve only moderate similarity to the ground-truth (GPT-5, averages 5.6 on a 1-10 scale). When composed into a 12-month end-to-end simulation of scientific production, the resulting synthetic corpus is systematically less diverse and less novel than human-authored research from the same period. 

---
# Learning to Rewrite Tool Descriptions for Reliable LLM-Agent Tool Use 

**Authors**: Ruocheng Guo, Kaiwen Dong, Xiang Gao, Kamalika Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.20426)  

**Abstract**: The performance of LLM-based agents depends not only on the agent itself but also on the quality of the tool interfaces it consumes. While prior work has focused heavily on agent fine-tuning, tool interfaces-including natural language descriptions and parameter schemas-remain largely human-oriented and often become a bottleneck, especially when agents must select from large candidate tool sets. Existing approaches to improving tool interfaces rely on execution traces, which are frequently unavailable in cold-start or privacy-constrained settings, and typically optimize each tool independently, limiting scalability and generalization to unseen tools. We propose Trace-Free+, a curriculum learning framework that progressively transfers supervision from trace-rich settings to trace-free deployment, encouraging the model to abstract reusable interface-usage patterns and tool usage outcomes. To support this approach, we construct a large-scale dataset of high-quality tool interfaces using a structured workflow over a diverse collection of tools. Experiments on StableToolBench and RestBench show consistent gains on unseen tools, strong cross-domain generalization, and robustness as the number of candidate tools scales to over 100, demonstrating that tool interface optimization is a practical and deployable complement to agent fine-tuning. 

---
# Implicit Intelligence -- Evaluating Agents on What Users Don't Say 

**Authors**: Ved Sirdeshmukh, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2602.20424)  

**Abstract**: Real-world requests to AI agents are fundamentally underspecified. Natural human communication relies on shared context and unstated constraints that speakers expect listeners to infer. Current agentic benchmarks test explicit instruction-following but fail to evaluate whether agents can reason about implicit requirements spanning accessibility needs, privacy boundaries, catastrophic risks, and contextual constraints. We present Implicit Intelligence, an evaluation framework testing whether AI agents can move beyond prompt-following to become genuine goal-fulfillers, paired with Agent-as-a-World (AaW), a harness where interactive worlds are defined in human-readable YAML files and simulated by language models. Our scenarios feature apparent simplicity in user requests, hidden complexity in correct solutions, and discoverability of constraints through environmental exploration. Evaluating 16 frontier and open-weight models across 205 scenarios, we find that even the best-performing model achieves only 48.3% scenario pass rate, revealing substantial room for improvement in bridging the gap between literal instruction-following and human-like contextual reasoning. 

---
# Diffusion Modulation via Environment Mechanism Modeling for Planning 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2602.20422)  

**Abstract**: Diffusion models have shown promising capabilities in trajectory generation for planning in offline reinforcement learning (RL). However, conventional diffusion-based planning methods often fail to account for the fact that generating trajectories in RL requires unique consistency between transitions to ensure coherence in real environments. This oversight can result in considerable discrepancies between the generated trajectories and the underlying mechanisms of a real environment. To address this problem, we propose a novel diffusion-based planning method, termed as Diffusion Modulation via Environment Mechanism Modeling (DMEMM). DMEMM modulates diffusion model training by incorporating key RL environment mechanisms, particularly transition dynamics and reward functions. Experimental results demonstrate that DMEMM achieves state-of-the-art performance for planning with offline reinforcement learning. 

---
# DMCD: Semantic-Statistical Framework for Causal Discovery 

**Authors**: Samarth KaPatel, Sofia Nikiforova, Giacinto Paolo Saggese, Paul Smith  

**Link**: [PDF](https://arxiv.org/pdf/2602.20333)  

**Abstract**: We present DMCD (DataMap Causal Discovery), a two-phase causal discovery framework that integrates LLM-based semantic drafting from variable metadata with statistical validation on observational data. In Phase I, a large language model proposes a sparse draft DAG, serving as a semantically informed prior over the space of possible causal structures. In Phase II, this draft is audited and refined via conditional independence testing, with detected discrepancies guiding targeted edge revisions.
We evaluate our approach on three metadata-rich real-world benchmarks spanning industrial engineering, environmental monitoring, and IT systems analysis. Across these datasets, DMCD achieves competitive or leading performance against diverse causal discovery baselines, with particularly large gains in recall and F1 score. Probing and ablation experiments suggest that these improvements arise from semantic reasoning over metadata rather than memorization of benchmark graphs. Overall, our results demonstrate that combining semantic priors with principled statistical verification yields a high-performing and practically effective approach to causal structure learning. 

---
# An artificial intelligence framework for end-to-end rare disease phenotyping from clinical notes using large language models 

**Authors**: Cathy Shyr, Yan Hu, Rory J. Tinker, Thomas A. Cassini, Kevin W. Byram, Rizwan Hamid, Daniel V. Fabbri, Adam Wright, Josh F. Peterson, Lisa Bastarache, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20324)  

**Abstract**: Phenotyping is fundamental to rare disease diagnosis, but manual curation of structured phenotypes from clinical notes is labor-intensive and difficult to scale. Existing artificial intelligence approaches typically optimize individual components of phenotyping but do not operationalize the full clinical workflow of extracting features from clinical text, standardizing them to Human Phenotype Ontology (HPO) terms, and prioritizing diagnostically informative HPO terms. We developed RARE-PHENIX, an end-to-end AI framework for rare disease phenotyping that integrates large language model-based phenotype extraction, ontology-grounded standardization to HPO terms, and supervised ranking of diagnostically informative phenotypes. We trained RARE-PHENIX using data from 2,671 patients across 11 Undiagnosed Diseases Network clinical sites, and externally validated it on 16,357 real-world clinical notes from Vanderbilt University Medical Center. Using clinician-curated HPO terms as the gold standard, RARE-PHENIX consistently outperformed a state-of-the-art deep learning baseline (PhenoBERT) across ontology-based similarity and precision-recall-F1 metrics in end-to-end evaluation (i.e., ontology-based similarity of 0.70 vs. 0.58). Ablation analyses demonstrated performance improvements with the addition of each module in RARE-PHENIX (extraction, standardization, and prioritization), supporting the value of modeling the full clinical phenotyping workflow. By modeling phenotyping as a clinically aligned workflow rather than a single extraction task, RARE-PHENIX provides structured, ranked phenotypes that are more concordant with clinician curation and has the potential to support human-in-the-loop rare disease diagnosis in real-world settings. 

---
# Multilevel Determinants of Overweight and Obesity Among U.S. Children Aged 10-17: Comparative Evaluation of Statistical and Machine Learning Approaches Using the 2021 National Survey of Children's Health 

**Authors**: Joyanta Jyoti Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2602.20303)  

**Abstract**: Background: Childhood and adolescent overweight and obesity remain major public health concerns in the United States and are shaped by behavioral, household, and community factors. Their joint predictive structure at the population level remains incompletely characterized. Objectives: The study aims to identify multilevel predictors of overweight and obesity among U.S. adolescents and compare the predictive performance, calibration, and subgroup equity of statistical, machine-learning, and deep-learning models. Data and Methods: We analyze 18,792 children aged 10-17 years from the 2021 National Survey of Children's Health. Overweight/obesity is defined using BMI categories. Predictors included diet, physical activity, sleep, parental stress, socioeconomic conditions, adverse experiences, and neighborhood characteristics. Models include logistic regression, random forest, gradient boosting, XGBoost, LightGBM, multilayer perceptron, and TabNet. Performance is evaluated using AUC, accuracy, precision, recall, F1 score, and Brier score. Results: Discrimination range from 0.66 to 0.79. Logistic regression, gradient boosting, and MLP showed the most stable balance of discrimination and calibration. Boosting and deep learning modestly improve recall and F1 score. No model was uniformly superior. Performance disparities across race and poverty groups persist across algorithms. Conclusion: Increased model complexity yields limited gains over logistic regression. Predictors consistently span behavioral, household, and neighborhood domains. Persistent subgroup disparities indicate the need for improved data quality and equity-focused surveillance rather than greater algorithmic complexity. 

---
# Test-Time Training with KV Binding Is Secretly Linear Attention 

**Authors**: Junchen Liu, Sven Elflein, Or Litany, Zan Gojcic, Ruilong Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.21204)  

**Abstract**: Test-time training (TTT) with KV binding as sequence modeling layer is commonly interpreted as a form of online meta-learning that memorizes a key-value mapping at test time. However, our analysis reveals multiple phenomena that contradict this memorization-based interpretation. Motivated by these findings, we revisit the formulation of TTT and show that a broad class of TTT architectures can be expressed as a form of learned linear attention operator. Beyond explaining previously puzzling model behaviors, this perspective yields multiple practical benefits: it enables principled architectural simplifications, admits fully parallel formulations that preserve performance while improving efficiency, and provides a systematic reduction of diverse TTT variants to a standard linear attention form. Overall, our results reframe TTT not as test-time memorization, but as learned linear attention with enhanced representational capacity. 

---
# Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs 

**Authors**: Yining Hong, Huang Huang, Manling Li, Li Fei-Fei, Jiajun Wu, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2602.21198)  

**Abstract**: Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection. 

---
# Why Pass@k Optimization Can Degrade Pass@1: Prompt Interference in LLM Post-training 

**Authors**: Anas Barakat, Souradip Chakraborty, Khushbu Pahwa, Amrit Singh Bedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.21189)  

**Abstract**: Pass@k is a widely used performance metric for verifiable large language model tasks, including mathematical reasoning, code generation, and short-answer reasoning. It defines success if any of $k$ independently sampled solutions passes a verifier. This multi-sample inference metric has motivated inference-aware fine-tuning methods that directly optimize pass@$k$. However, prior work reports a recurring trade-off: pass@k improves while pass@1 degrades under such methods. This trade-off is practically important because pass@1 often remains a hard operational constraint due to latency and cost budgets, imperfect verifier coverage, and the need for a reliable single-shot fallback. We study the origin of this trade-off and provide a theoretical characterization of when pass@k policy optimization can reduce pass@1 through gradient conflict induced by prompt interference. We show that pass@$k$ policy gradients can conflict with pass@1 gradients because pass@$k$ optimization implicitly reweights prompts toward low-success prompts; when these prompts are what we term negatively interfering, their upweighting can rotate the pass@k update direction away from the pass@1 direction. We illustrate our theoretical findings with large language model experiments on verifiable mathematical reasoning tasks. 

---
# XMorph: Explainable Brain Tumor Analysis Via LLM-Assisted Hybrid Deep Intelligence 

**Authors**: Sepehr Salem Ghahfarokhi, M. Moein Esfahani, Raj Sunderraman, Vince Calhoun, Mohammed Alser  

**Link**: [PDF](https://arxiv.org/pdf/2602.21178)  

**Abstract**: Deep learning has significantly advanced automated brain tumor diagnosis, yet clinical adoption remains limited by interpretability and computational constraints. Conventional models often act as opaque ''black boxes'' and fail to quantify the complex, irregular tumor boundaries that characterize malignant growth. To address these challenges, we present XMorph, an explainable and computationally efficient framework for fine-grained classification of three prominent brain tumor types: glioma, meningioma, and pituitary tumors. We propose an Information-Weighted Boundary Normalization (IWBN) mechanism that emphasizes diagnostically relevant boundary regions alongside nonlinear chaotic and clinically validated features, enabling a richer morphological representation of tumor growth. A dual-channel explainable AI module combines GradCAM++ visual cues with LLM-generated textual rationales, translating model reasoning into clinically interpretable insights. The proposed framework achieves a classification accuracy of 96.0%, demonstrating that explainability and high performance can co-exist in AI-based medical imaging systems. The source code and materials for XMorph are all publicly available at: this https URL. 

---
# Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids 

**Authors**: Victor Reijgwart, Cesar Cadena, Roland Siegwart, Lionel Ott  

**Link**: [PDF](https://arxiv.org/pdf/2602.21174)  

**Abstract**: Hierarchical, multi-resolution volumetric mapping approaches are widely used to represent large and complex environments as they can efficiently capture their occupancy and connectivity information. Yet widely used path planning methods such as sampling and trajectory optimization do not exploit this explicit connectivity information, and search-based methods such as A* suffer from scalability issues in large-scale high-resolution maps. In many applications, Euclidean shortest paths form the underpinning of the navigation system. For such applications, any-angle planning methods, which find optimal paths by connecting corners of obstacles with straight-line segments, provide a simple and efficient solution. In this paper, we present a method that has the optimality and completeness properties of any-angle planners while overcoming computational tractability issues common to search-based methods by exploiting multi-resolution representations. Extensive experiments on real and synthetic environments demonstrate the proposed approach's solution quality and speed, outperforming even sampling-based methods. The framework is open-sourced to allow the robotics and planning community to build on our research. 

---
# PVminer: A Domain-Specific Tool to Detect the Patient Voice in Patient Generated Data 

**Authors**: Samah Fodeh, Linhai Ma, Yan Wang, Srivani Talakokkul, Ganesh Puthiaraju, Afshan Khan, Ashley Hagaman, Sarah Lowe, Aimee Roundtree  

**Link**: [PDF](https://arxiv.org/pdf/2602.21165)  

**Abstract**: Patient-generated text such as secure messages, surveys, and interviews contains rich expressions of the patient voice (PV), reflecting communicative behaviors and social determinants of health (SDoH). Traditional qualitative coding frameworks are labor intensive and do not scale to large volumes of patient-authored messages across health systems. Existing machine learning (ML) and natural language processing (NLP) approaches provide partial solutions but often treat patient-centered communication (PCC) and SDoH as separate tasks or rely on models not well suited to patient-facing language. We introduce PVminer, a domain-adapted NLP framework for structuring patient voice in secure patient-provider communication. PVminer formulates PV detection as a multi-label, multi-class prediction task integrating patient-specific BERT encoders (PV-BERT-base and PV-BERT-large), unsupervised topic modeling for thematic augmentation (PV-Topic-BERT), and fine-tuned classifiers for Code, Subcode, and Combo-level labels. Topic representations are incorporated during fine-tuning and inference to enrich semantic inputs. PVminer achieves strong performance across hierarchical tasks and outperforms biomedical and clinical pre-trained baselines, achieving F1 scores of 82.25% (Code), 80.14% (Subcode), and up to 77.87% (Combo). An ablation study further shows that author identity and topic-based augmentation each contribute meaningful gains. Pre-trained models, source code, and documentation will be publicly released, with annotated datasets available upon request for research use. 

---
# SparkMe: Adaptive Semi-Structured Interviewing for Qualitative Insight Discovery 

**Authors**: David Anugraha, Vishakh Padmakumar, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21136)  

**Abstract**: Qualitative insights from user experiences are critical for informing product and policy decisions, but collecting such data at scale is constrained by the time and availability of experts to conduct semi-structured interviews. Recent work has explored using large language models (LLMs) to automate interviewing, yet existing systems lack a principled mechanism for balancing systematic coverage of predefined topics with adaptive exploration, or the ability to pursue follow-ups, deep dives, and emergent themes that arise organically during conversation. In this work, we formulate adaptive semi-structured interviewing as an optimization problem over the interviewer's behavior. We define interview utility as a trade-off between coverage of a predefined interview topic guide, discovery of relevant emergent themes, and interview cost measured by length. Based on this formulation, we introduce SparkMe, a multi-agent LLM interviewer that performs deliberative planning via simulated conversation rollouts to select questions with high expected utility. We evaluate SparkMe through controlled experiments with LLM-based interviewees, showing that it achieves higher interview utility, improving topic guide coverage (+4.7% over the best baseline) and eliciting richer emergent insights while using fewer conversational turns than prior LLM interviewing approaches. We further validate SparkMe in a user study with 70 participants across 7 professions on the impact of AI on their workflows. Domain experts rate SparkMe as producing high-quality adaptive interviews that surface helpful profession-specific insights not captured by prior approaches. The code, datasets, and evaluation protocols for SparkMe are available as open-source at this https URL. 

---
# "Are You Sure?": An Empirical Study of Human Perception Vulnerability in LLM-Driven Agentic Systems 

**Authors**: Xinfeng Li, Shenyu Dai, Kelong Zheng, Yue Xiao, Gelei Deng, Wei Dong, Xiaofeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.21127)  

**Abstract**: Large language model (LLM) agents are rapidly becoming trusted copilots in high-stakes domains like software development and healthcare. However, this deepening trust introduces a novel attack surface: Agent-Mediated Deception (AMD), where compromised agents are weaponized against their human users. While extensive research focuses on agent-centric threats, human susceptibility to deception by a compromised agent remains unexplored. We present the first large-scale empirical study with 303 participants to measure human susceptibility to AMD. This is based on HAT-Lab (Human-Agent Trust Laboratory), a high-fidelity research platform we develop, featuring nine carefully crafted scenarios spanning everyday and professional domains (e.g., healthcare, software development, human resources). Our 10 key findings reveal significant vulnerabilities and provide future defense perspectives. Specifically, only 8.6% of participants perceive AMD attacks, while domain experts show increased susceptibility in certain scenarios. We identify six cognitive failure modes in users and find that their risk awareness often fails to translate to protective behavior. The defense analysis reveals that effective warnings should interrupt workflows with low verification costs. With experiential learning based on HAT-Lab, over 90% of users who perceive risks report increased caution against AMD. This work provides empirical evidence and a platform for human-centric agent security research. 

---
# Cooperative-Competitive Team Play of Real-World Craft Robots 

**Authors**: Rui Zhao, Xihui Li, Yizheng Zhang, Yuzhen Liu, Zhong Zhang, Yufeng Zhang, Cheng Zhou, Zhengyou Zhang, Lei Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.21119)  

**Abstract**: Multi-agent deep Reinforcement Learning (RL) has made significant progress in developing intelligent game-playing agents in recent years. However, the efficient training of collective robots using multi-agent RL and the transfer of learned policies to real-world applications remain open research questions. In this work, we first develop a comprehensive robotic system, including simulation, distributed learning framework, and physical robot components. We then propose and evaluate reinforcement learning techniques designed for efficient training of cooperative and competitive policies on this platform. To address the challenges of multi-agent sim-to-real transfer, we introduce Out of Distribution State Initialization (OODSI) to mitigate the impact of the sim-to-real gap. In the experiments, OODSI improves the Sim2Real performance by 20%. We demonstrate the effectiveness of our approach through experiments with a multi-robot car competitive game and a cooperative task in real-world settings. 

---
# Attention-Based SINR Estimation in User-Centric Non-Terrestrial Networks 

**Authors**: Bruno De Filippo, Alessandro Guidotti, Alessandro Vanelli-Coralli  

**Link**: [PDF](https://arxiv.org/pdf/2602.21116)  

**Abstract**: The signal-to-interference-plus-noise ratio (SINR) is central to performance optimization in user-centric beamforming for satellite-based non-terrestrial networks (NTNs). Its assessment either requires the transmission of dedicated pilots or relies on computing the beamforming matrix through minimum mean squared error (MMSE)-based formulations beforehand, a process that introduces significant computational overhead. In this paper, we propose a low-complexity SINR estimation framework that leverages multi-head self-attention (MHSA) to extract inter-user interference features directly from either channel state information or user location reports. The proposed dual MHSA (DMHSA) models evaluate the SINR of a scheduled user group without requiring explicit MMSE calculations. The architecture achieves a computational complexity reduction by a factor of three in the CSI-based setting and by two orders of magnitude in the location-based configuration, the latter benefiting from the lower dimensionality of user reports. We show that both DMHSA models maintain high estimation accuracy, with the root mean squared error typically below 1 dB with priority-queuing-based scheduled users. These results enable the integration of DMHSA-based estimators into scheduling procedures, allowing the evaluation of multiple candidate user groups and the selection of those offering the highest average SINR and capacity. 

---
# Probing Graph Neural Network Activation Patterns Through Graph Topology 

**Authors**: Floriano Tori, Lorenzo Bini, Marco Sorbi, Stéphane Marchand-Maillet, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2602.21092)  

**Abstract**: Curvature notions on graphs provide a theoretical description of graph topology, highlighting bottlenecks and denser connected regions. Artifacts of the message passing paradigm in Graph Neural Networks, such as oversmoothing and oversquashing, have been attributed to these regions. However, it remains unclear how the topology of a graph interacts with the learned preferences of GNNs. Through Massive Activations, which correspond to extreme edge activation values in Graph Transformers, we probe this correspondence. Our findings on synthetic graphs and molecular benchmarks reveal that MAs do not preferentially concentrate on curvature extremes, despite their theoretical link to information flow. On the Long Range Graph Benchmark, we identify a systemic \textit{curvature shift}: global attention mechanisms exacerbate topological bottlenecks, drastically increasing the prevalence of negative curvature. Our work reframes curvature as a diagnostic probe for understanding when and why graph learning fails. 

---
# Localized Dynamics-Aware Domain Adaption for Off-Dynamics Offline Reinforcement Learning 

**Authors**: Zhangjie Xia, Yu Yang, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.21072)  

**Abstract**: Off-dynamics offline reinforcement learning (RL) aims to learn a policy for a target domain using limited target data and abundant source data collected under different transition dynamics. Existing methods typically address dynamics mismatch either globally over the state space or via pointwise data filtering; these approaches can miss localized cross-domain similarities or incur high computational cost. We propose Localized Dynamics-Aware Domain Adaptation (LoDADA), which exploits localized dynamics mismatch to better reuse source data. LoDADA clusters transitions from source and target datasets and estimates cluster-level dynamics discrepancy via domain discrimination. Source transitions from clusters with small discrepancy are retained, while those from clusters with large discrepancy are filtered out. This yields a fine-grained and scalable data selection strategy that avoids overly coarse global assumptions and expensive per-sample filtering. We provide theoretical insights and extensive experiments across environments with diverse global and local dynamics shifts. Results show that LoDADA consistently outperforms state-of-the-art off-dynamics offline RL methods by better leveraging localized distribution mismatch. 

---
# VAUQ: Vision-Aware Uncertainty Quantification for LVLM Self-Evaluation 

**Authors**: Seongheon Park, Changdae Oh, Hyeong Kyu Choi, Xuefeng Du, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.21054)  

**Abstract**: Large Vision-Language Models (LVLMs) frequently hallucinate, limiting their safe deployment in real-world applications. Existing LLM self-evaluation methods rely on a model's ability to estimate the correctness of its own outputs, which can improve deployment reliability; however, they depend heavily on language priors and are therefore ill-suited for evaluating vision-conditioned predictions. We propose VAUQ, a vision-aware uncertainty quantification framework for LVLM self-evaluation that explicitly measures how strongly a model's output depends on visual evidence. VAUQ introduces the Image-Information Score (IS), which captures the reduction in predictive uncertainty attributable to visual input, and an unsupervised core-region masking strategy that amplifies the influence of salient regions. Combining predictive entropy with this core-masked IS yields a training-free scoring function that reliably reflects answer correctness. Comprehensive experiments show that VAUQ consistently outperforms existing self-evaluation methods across multiple datasets. 

---
# Position-Aware Sequential Attention for Accurate Next Item Recommendations 

**Authors**: Timur Nabiev, Evgeny Frolov  

**Link**: [PDF](https://arxiv.org/pdf/2602.21052)  

**Abstract**: Sequential self-attention models usually rely on additive positional embeddings, which inject positional information into item representations at the input. In the absence of positional signals, the attention block is permutation-equivariant over sequence positions and thus has no intrinsic notion of temporal order beyond causal masking. We argue that additive positional embeddings make the attention mechanism only superficially sensitive to sequence order: positional information is entangled with item embedding semantics, propagates weakly in deep architectures, and limits the ability to capture rich sequential patterns. To address these limitations, we introduce a kernelized self-attention mechanism, where a learnable positional kernel operates purely in the position space, disentangled from semantic similarity, and directly modulates attention weights. When applied per attention block, this kernel enables adaptive multi-scale sequential modeling. Experiments on standard next-item prediction benchmarks show that our positional kernel attention consistently improves over strong competing baselines. 

---
# MIP Candy: A Modular PyTorch Framework for Medical Image Processing 

**Authors**: Tianhao Fu, Yucheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.21033)  

**Abstract**: Medical image processing demands specialized software that handles high-dimensional volumetric data, heterogeneous file formats, and domain-specific training procedures. Existing frameworks either provide low-level components that require substantial integration effort or impose rigid, monolithic pipelines that resist modification. We present MIP Candy (MIPCandy), a freely available, PyTorch-based framework designed specifically for medical image processing. MIPCandy provides a complete, modular pipeline spanning data loading, training, inference, and evaluation, allowing researchers to obtain a fully functional process workflow by implementing a single method, $\texttt{build_network}$, while retaining fine-grained control over every component. Central to the design is $\texttt{LayerT}$, a deferred configuration mechanism that enables runtime substitution of convolution, normalization, and activation modules without subclassing. The framework further offers built-in $k$-fold cross-validation, dataset inspection with automatic region-of-interest detection, deep supervision, exponential moving average, multi-frontend experiment tracking (Weights & Biases, Notion, MLflow), training state recovery, and validation score prediction via quotient regression. An extensible bundle ecosystem provides pre-built model implementations that follow a consistent trainer--predictor pattern and integrate with the core framework without modification. MIPCandy is open-source under the Apache-2.0 license and requires Python~3.12 or later. Source code and documentation are available at this https URL. 

---
# Multimodal MRI Report Findings Supervised Brain Lesion Segmentation with Substructures 

**Authors**: Yubin Ge, Yongsong Huang, Xiaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20994)  

**Abstract**: Report-supervised (RSuper) learning seeks to alleviate the need for dense tumor voxel labels with constraints derived from radiology reports (e.g., volumes, counts, sizes, locations). In MRI studies of brain tumors, however, we often involve multi-parametric scans and substructures. Here, fine-grained modality/parameter-wise reports are usually provided along with global findings and are correlated with different substructures. Moreover, the reports often describe only the largest lesion and provide qualitative or uncertain cues (``mild,'' ``possible''). Classical RSuper losses (e.g., sum volume consistency) can over-constrain or hallucinate unreported findings under such incompleteness, and are unable to utilize these hierarchical findings or exploit the priors of varied lesion types in a merged dataset. We explicitly parse the global quantitative and modality-wise qualitative findings and introduce a unified, one-sided, uncertainty-aware formulation (MS-RSuper) that: (i) aligns modality-specific qualitative cues (e.g., T1c enhancement, FLAIR edema) with their corresponding substructures using existence and absence losses; (ii) enforces one-sided lower-bounds for partial quantitative cues (e.g., largest lesion size, minimal multiplicity); and (iii) adds extra- vs. intra-axial anatomical priors to respect cohort differences. Certainty tokens scale penalties; missing cues are down-weighted. On 1238 report-labeled BraTS-MET/MEN scans, our MS-RSuper largely outperforms both a sparsely-supervised baseline and a naive RSuper method. 

---
# Echoes Over Time: Unlocking Length Generalization in Video-to-Audio Generation Models 

**Authors**: Christian Simon, MAsato Ishii, Wei-Yao Wang, Koichi Saito, Akio Hayakawa, Dongseok Shim, Zhi Zhong, Shuyang Cui, Shusuke Takahashi, Takashi Shibuya, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2602.20981)  

**Abstract**: Scaling multimodal alignment between video and audio is challenging, particularly due to limited data and the mismatch between text descriptions and frame-level video information. In this work, we tackle the scaling challenge in multimodal-to-audio generation, examining whether models trained on short instances can generalize to longer ones during testing. To tackle this challenge, we present multimodal hierarchical networks so-called MMHNet, an enhanced extension of state-of-the-art video-to-audio models. Our approach integrates a hierarchical method and non-causal Mamba to support long-form audio generation. Our proposed method significantly improves long audio generation up to more than 5 minutes. We also prove that training short and testing long is possible in the video-to-audio generation tasks without training on the longer durations. We show in our experiments that our proposed method could achieve remarkable results on long-video to audio benchmarks, beating prior works in video-to-audio tasks. Moreover, we showcase our model capability in generating more than 5 minutes, while prior video-to-audio methods fall short in generating with long durations. 

---
# CrystaL: Spontaneous Emergence of Visual Latents in MLLMs 

**Authors**: Yang Zhang, Danyang Li, Yuxuan Li, Xin Zhang, Tianyu Xie, Mingming Cheng, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20980)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable performance by integrating powerful language backbones with large-scale visual encoders. Among these, latent Chain-of-Thought (CoT) methods enable implicit reasoning in continuous hidden states, facilitating seamless vision-language integration and faster inference. However, existing heuristically predefined supervision signals in latent CoT provide limited guidance for preserving critical visual information in intermediate latent states. To address this limitation, we propose CrystaL (Crystallized Latent Reasoning), a single-stage framework with two paths to process intact and corrupted images, respectively. By explicitly aligning the attention patterns and prediction distributions across the two paths, CrystaL crystallizes latent representations into task-relevant visual semantics, without relying on auxiliary annotations or external modules. Extensive experiments on perception-intensive benchmarks demonstrate that CrystaL consistently outperforms state-of-the-art baselines, achieving substantial gains in fine-grained visual understanding while maintaining robust reasoning capabilities. 

---
# Toward an Agentic Infused Software Ecosystem 

**Authors**: Mark Marron  

**Link**: [PDF](https://arxiv.org/pdf/2602.20979)  

**Abstract**: Fully leveraging the capabilities of AI agents in software development requires a rethinking of the software ecosystem itself. To this end, this paper outlines the creation of an Agentic Infused Software Ecosystem (AISE), that rests on three pillars. The first, of course, is the AI agents themselves, which in the past 5 years have moved from simple code completion and toward sophisticated independent development tasks, a trend which will only continue. The second pillar is the programming language and APIs (or tools) that these agents use to accomplish tasks, and increasingly, serve as the communication substrate that humans and AI agents interact and collaborate through. The final pillar is the runtime environment and ecosystem that agents operate within, and which provide the capabilities that programmatic agents use to interface with (and effect actions in) the external world. To realize the vision of AISE, all three pillars must be advanced in a holistic manner, and critically, in a manner that is synergistic for AI agents as they exist today, those that will exist in the future, and for the human developers that work alongside them. 

---
# Does Order Matter : Connecting The Law of Robustness to Robust Generalization 

**Authors**: Himadri Mandal, Vishnu Varadarajan, Jaee Ponde, Aritra Das, Mihir More, Debayan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.20971)  

**Abstract**: Bubeck and Sellke (2021) pose as an open problem the connection between the law of robustness and robust generalization. The law of robustness states that overparameterization is necessary for models to interpolate robustly; in particular, robust interpolation requires the learned function to be Lipschitz. Robust generalization asks whether small robust training loss implies small robust test loss. We resolve this problem by explicitly connecting the two for arbitrary data distributions. Specifically, we introduce a nontrivial notion of robust generalization error and convert it into a lower bound on the expected Rademacher complexity of the induced robust loss class. Our bounds recover the $\Omega(n^{1/d})$ regime of Wu et al.\ (2023) and show that, up to constants, robust generalization does not change the order of the Lipschitz constant required for smooth interpolation. We conduct experiments to probe the predicted scaling with dataset size and model capacity, testing whether empirical behavior aligns more closely with the predictions of Bubeck and Sellke (2021) or Wu et al.\ (2023). For MNIST, we find that the lower-bound Lipschitz constant scales on the order predicted by Wu et al.\ (2023). Informally, to obtain low robust generalization error, the Lipschitz constant must lie in a range that we bound, and the allowable perturbation radius is linked to the Lipschitz scale. 

---
# Training-Free Intelligibility-Guided Observation Addition for Noisy ASR 

**Authors**: Haoyang Li, Changsong Liu, Wei Rao, Hao Shi, Sakriani Sakti, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2602.20967)  

**Abstract**: Automatic speech recognition (ASR) degrades severely in noisy environments. Although speech enhancement (SE) front-ends effectively suppress background noise, they often introduce artifacts that harm recognition. Observation addition (OA) addressed this issue by fusing noisy and SE enhanced speech, improving recognition without modifying the parameters of the SE or ASR models. This paper proposes an intelligibility-guided OA method, where fusion weights are derived from intelligibility estimates obtained directly from the backend ASR. Unlike prior OA methods based on trained neural predictors, the proposed method is training-free, reducing complexity and enhances generalization. Extensive experiments across diverse SE-ASR combinations and datasets demonstrate strong robustness and improvements over existing OA baselines. Additional analyses of intelligibility-guided switching-based alternatives and frame versus utterance-level OA further validate the proposed design. 

---
# EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations 

**Authors**: Luka Šiktar, Branimir Ćaran, Bojan Šekoranja, Marko Švaco  

**Link**: [PDF](https://arxiv.org/pdf/2602.20958)  

**Abstract**: Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios. 

---
# See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis 

**Authors**: Jaehyun Park, Minyoung Ahn, Minkyu Kim, Jonghyun Lee, Jae-Gil Lee, Dongmin Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.20951)  

**Abstract**: Despite recent advances in diffusion models, AI generated images still often contain visual artifacts that compromise realism. Although more thorough pre-training and bigger models might reduce artifacts, there is no assurance that they can be completely eliminated, which makes artifact mitigation a highly crucial area of study. Previous artifact-aware methodologies depend on human-labeled artifact datasets, which are costly and difficult to scale, underscoring the need for an automated approach to reliably acquire artifact-annotated datasets. In this paper, we propose ArtiAgent, which efficiently creates pairs of real and artifact-injected images. It comprises three agents: a perception agent that recognizes and grounds entities and subentities from real images, a synthesis agent that introduces artifacts via artifact injection tools through novel patch-wise embedding manipulation within a diffusion transformer, and a curation agent that filters the synthesized artifacts and generates both local and global explanations for each instance. Using ArtiAgent, we synthesize 100K images with rich artifact annotations and demonstrate both efficacy and versatility across diverse applications. Code is available at link. 

---
# Some Simple Economics of AGI 

**Authors**: Christian Catalini, Xiang Hui, Jane Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20946)  

**Abstract**: For millennia, human cognition was the primary engine of progress on Earth. As AI decouples cognition from biology, the marginal cost of measurable execution falls to zero, absorbing any labor capturable by metrics--including creative, analytical, and innovative work. The binding constraint on growth is no longer intelligence but human verification bandwidth: the capacity to validate, audit, and underwrite responsibility when execution is abundant. We model the AGI transition as the collision of two racing cost curves: an exponentially decaying Cost to Automate and a biologically bottlenecked Cost to Verify. This structural asymmetry widens a Measurability Gap between what agents can execute and what humans can afford to verify. It also drives a shift from skill-biased to measurability-biased technical change. Rents migrate to verification-grade ground truth, cryptographic provenance, and liability underwriting--the ability to insure outcomes rather than merely generate them. The current human-in-the-loop equilibrium is unstable: eroded from below as apprenticeship collapses (Missing Junior Loop) and from within as experts codify their obsolescence (Codifier's Curse). Unverified deployment becomes privately rational--a Trojan Horse externality. Unmanaged, these forces pull toward a Hollow Economy. Yet by scaling verification alongside agentic capabilities, the forces that threaten collapse become the catalyst for unbounded discovery and experimentation--an Augmented Economy. We derive a practical playbook for individuals, companies, investors, and policymakers. Today's defining challenge is not the race to deploy the most autonomous systems; it is the race to secure the foundations of their oversight. Only by scaling our bandwidth for verification alongside our capacity for execution can we ensure that the intelligence we have summoned preserves the humanity that initiated it. 

---
# The Art of Efficient Reasoning: Data, Reward, and Optimization 

**Authors**: Taiqiang Wu, Zenan Zu, Bo Zhou, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2602.20945)  

**Abstract**: Large Language Models (LLMs) consistently benefit from scaled Chain-of-Thought (CoT) reasoning, but also suffer from heavy computational overhead. To address this issue, efficient reasoning aims to incentivize short yet accurate thinking trajectories, typically through reward shaping with Reinforcement Learning (RL). In this paper, we systematically investigate the mechanics of efficient reasoning for LLMs. For comprehensive evaluation, we advocate for more fine-grained metrics, including length distribution conditioned on correctness and performance across a wide spectrum of token budgets ranging from 2k to 32k. First, we reveal that the training process follows a two-stage paradigm: length adaptation and reasoning refinement. After that, we conduct extensive experiments (about 0.2 million GPU hours) in a unified protocol, deconstructing training prompts and rollouts, reward shaping, and optimization strategies. In particular, a key finding is to train on relatively easier prompts, ensuring the density of positive reward signals and thus avoiding the length collapse. Meanwhile, the learned length bias can be generalized across domains. We distill all findings into valuable insights and practical guidelines, and further validate them across the Qwen3 series, ranging from 0.6B to 30B, demonstrating the robustness and generalization. 

---
# Airavat: An Agentic Framework for Internet Measurement 

**Authors**: Alagappan Ramanathan, Eunju Kang, Dongsu Han, Sangeetha Abdu Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20924)  

**Abstract**: Internet measurement faces twin challenges: complex analyses require expert-level orchestration of tools, yet even syntactically correct implementations can have methodological flaws and can be difficult to verify. Democratizing measurement capabilities thus demands automating both workflow generation and verification against methodological standards established through decades of research.
We present Airavat, the first agentic framework for Internet measurement workflow generation with systematic verification and validation. Airavat coordinates a set of agents mirroring expert reasoning: three agents handle problem decomposition, solution design, and code implementation, with assistance from a registry of existing tools. Two specialized engines ensure methodological correctness: a Verification Engine evaluates workflows against a knowledge graph encoding five decades of measurement research, while a Validation Engine identifies appropriate validation techniques grounded in established methodologies. Through four Internet measurement case studies, we demonstrate that Airavat (i) generates workflows matching expert-level solutions, (ii) makes sound architectural decisions, (iii) addresses novel problems without ground truth, and (iv) identifies methodological flaws missed by standard execution-based testing. 

---
# E-MMKGR: A Unified Multimodal Knowledge Graph Framework for E-commerce Applications 

**Authors**: Jiwoo Kang, Yeon-Chang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.20877)  

**Abstract**: Multimodal recommender systems (MMRSs) enhance collaborative filtering by leveraging item-side modalities, but their reliance on a fixed set of modalities and task-specific objectives limits both modality extensibility and task generalization. We propose E-MMKGR, a framework that constructs an e-commerce-specific Multimodal Knowledge Graph E-MMKG and learns unified item representations through GNN-based propagation and KG-oriented optimization. These representations provide a shared semantic foundation applicable to diverse tasks. Experiments on real-world Amazon datasets show improvements of up to 10.18% in Recall@10 for recommendation and up to 21.72% over vector-based retrieval for product search, demonstrating the effectiveness and extensibility of our approach. 

---
# SoK: Agentic Skills -- Beyond Tool Use in LLM Agents 

**Authors**: Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, Guangsheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20867)  

**Abstract**: Agentic systems increasingly rely on reusable procedural capabilities, \textit{a.k.a., agentic skills}, to execute long-horizon workflows reliably. These capabilities are callable modules that package procedural knowledge with explicit applicability conditions, execution policies, termination criteria, and reusable interfaces. Unlike one-off plans or atomic tool calls, skills operate (and often do well) across tasks.
This paper maps the skill layer across the full lifecycle (discovery, practice, distillation, storage, composition, evaluation, and update) and introduces two complementary taxonomies. The first is a system-level set of \textbf{seven design patterns} capturing how skills are packaged and executed in practice, from metadata-driven progressive disclosure and executable code skills to self-evolving libraries and marketplace distribution. The second is an orthogonal \textbf{representation $\times$ scope} taxonomy describing what skills \emph{are} (natural language, code, policy, hybrid) and what environments they operate over (web, OS, software engineering, robotics).
We analyze the security and governance implications of skill-based agents, covering supply-chain risks, prompt injection via skill payloads, and trust-tiered execution, grounded by a case study of the ClawHavoc campaign in which nearly 1{,}200 malicious skills infiltrated a major agent marketplace, exfiltrating API keys, cryptocurrency wallets, and browser credentials at scale. We further survey deterministic evaluation approaches, anchored by recent benchmark evidence that curated skills can substantially improve agent success rates while self-generated skills may degrade them. We conclude with open challenges toward robust, verifiable, and certifiable skills for real-world autonomous agents. 

---
# Regret-Guided Search Control for Efficient Learning in AlphaZero 

**Authors**: Yun-Jui Tsai, Wei-Yu Chen, Yan-Ru Ju, Yu-Hung Chang, Ti-Rong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20809)  

**Abstract**: Reinforcement learning (RL) agents achieve remarkable performance but remain far less learning-efficient than humans. While RL agents require extensive self-play games to extract useful signals, humans often need only a few games, improving rapidly by repeatedly revisiting states where mistakes occurred. This idea, known as search control, aims to restart from valuable states rather than always from the initial state. In AlphaZero, prior work Go-Exploit applies this idea by sampling past states from self-play or search trees, but it treats all states equally, regardless of their learning potential. We propose Regret-Guided Search Control (RGSC), which extends AlphaZero with a regret network that learns to identify high-regret states, where the agent's evaluation diverges most from the actual outcome. These states are collected from both self-play trajectories and MCTS nodes, stored in a prioritized regret buffer, and reused as new starting positions. Across 9x9 Go, 10x10 Othello, and 11x11 Hex, RGSC outperforms AlphaZero and Go-Exploit by an average of 77 and 89 Elo, respectively. When training on a well-trained 9x9 Go model, RGSC further improves the win rate against KataGo from 69.3% to 78.2%, while both baselines show no improvement. These results demonstrate that RGSC provides an effective mechanism for search control, improving both efficiency and robustness of AlphaZero training. Our code is available at this https URL. 

---
# OrthoDiffusion: A Generalizable Multi-Task Diffusion Foundation Model for Musculoskeletal MRI Interpretation 

**Authors**: Tian Lan, Lei Xu, Zimu Yuan, Shanggui Liu, Jiajun Liu, Jiaxin Liu, Weilai Xiang, Hongyu Yang, Dong Jiang, Jianxin Yin, Dingyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20752)  

**Abstract**: Musculoskeletal disorders represent a significant global health burden and are a leading cause of disability worldwide. While MRI is essential for accurate diagnosis, its interpretation remains exceptionally challenging. Radiologists must identify multiple potential abnormalities within complex anatomical structures across different imaging planes, a process that requires significant expertise and is prone to variability. We developed OrthoDiffusion, a unified diffusion-based foundation model designed for multi-task musculoskeletal MRI interpretation. The framework utilizes three orientation-specific 3D diffusion models, pre-trained in a self-supervised manner on 15,948 unlabeled knee MRI scans, to learn robust anatomical features from sagittal, coronal, and axial views. These view-specific representations are integrated to support diverse clinical tasks, including anatomical segmentation and multi-label diagnosis. Our evaluation demonstrates that OrthoDiffusion achieves excellent performance in the segmentation of 11 knee structures and the detection of 8 knee abnormalities. The model exhibited remarkable robustness across different clinical centers and MRI field strengths, consistently outperforming traditional supervised models. Notably, in settings where labeled data was scarce, OrthoDiffusion maintained high diagnostic precision using only 10\% of training labels. Furthermore, the anatomical representations learned from knee imaging proved highly transferable to other joints, achieving strong diagnostic performance across 11 diseases of the ankle and shoulder. These findings suggest that diffusion-based foundation models can serve as a unified platform for multi-disease diagnosis and anatomical segmentation, potentially improving the efficiency and accuracy of musculoskeletal MRI interpretation in real-world clinical workflows. 

---
# SibylSense: Adaptive Rubric Learning via Memory Tuning and Adversarial Probing 

**Authors**: Yifei Xu, Guilherme Potje, Shivam Shandilya, Tiancheng Yuan, Leonardo de Oliveira Nunes, Rakshanda Agarwal, Saeid Asgari, Adam Atkinson, Emre Kıcıman, Songwu Lu, Ranveer Chandra, Tusher Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2602.20751)  

**Abstract**: Designing aligned and robust rewards for open-ended generation remains a key barrier to RL post-training. Rubrics provide structured, interpretable supervision, but scaling rubric construction is difficult: expert rubrics are costly, prompted rubrics are often superficial or inconsistent, and fixed-pool discriminative rubrics can saturate and drift, enabling reward hacking. We present SibylSense, an inference-time learning approach that adapts a frozen rubric generator through a tunable memory bank of validated rubric items. Memory is updated via verifier-based item rewards measured by reference-candidate answer discriminative gaps from a handful of examples. SibylSense alternates memory tuning with a rubric-adversarial policy update that produces rubric-satisfying candidate answers, shrinking discriminative gaps and driving the rubric generator to capture new quality dimensions. Experiments on two open-ended tasks show that SibylSense yields more discriminative rubrics and improves downstream RL performance over static and non-adaptive baselines. 

---
# Voices of the Mountains: Deep Learning-Based Vocal Error Detection System for Kurdish Maqams 

**Authors**: Darvan Shvan Khairaldeen, Hossein Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2602.20744)  

**Abstract**: Maqam, a singing type, is a significant component of Kurdish music. A maqam singer receives training in a traditional face-to-face or through self-training. Automatic Singing Assessment (ASA) uses machine learning (ML) to provide the accuracy of singing styles and can help learners to improve their performance through error detection. Currently, the available ASA tools follow Western music rules. The musical composition requires all notes to stay within their expected pitch range from start to finish. The system fails to detect micro-intervals and pitch bends, so it identifies Kurdish maqam singing as incorrect even though the singer performs according to traditional rules. Kurdish maqam requires recognizing performance errors within microtonal spaces, which is beyond Western equal temperament. This research is the first attempt to address the mentioned gap. While many error types happen during singing, our focus is on pitch, rhythm, and modal stability errors in the context of Bayati-Kurd. We collected 50 songs from 13 vocalists ( 2-3 hours) and annotated 221 error spans (150 fine pitch, 46 rhythm, 25 modal drift). The data was segmented into 15,199 overlapping windows and converted to log-mel spectrograms. We developed a two-headed CNN-BiLSTM with attention mode to decide whether a window contains an error and to classify it based on the chosen errors. Trained for 20 epochs with early stopping at epoch 10, the model reached a validation macro-F1 of 0.468. On the full 50-song evaluation at a 0.750 threshold, recall was 39.4% and precision 25.8% . Within detected windows, type macro-F1 was 0.387, with F1 of 0.492 (fine pitch), 0.536 (rhythm), and 0.133 (modal drift); modal drift recall was 8.0%. The better performance on common error types shows that the method works, while the poor modal-drift recall shows that more data and balancing are needed. 

---
# RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition 

**Authors**: Kun Ran, Marwah Alaofi, Danula Hettiachchi, Chenglong Ma, Khoi Nguyen Dinh Anh, Khoi Vo Nguyen, Sachin Pathiyan Cherumanal, Lida Rashidi, Falk Scholer, Damiano Spina, Shuoqi Sun, Oleg Zendel  

**Link**: [PDF](https://arxiv.org/pdf/2602.20735)  

**Abstract**: This paper presents the award-winning RMIT-ADM+S system for the Text-to-Text
track of the NeurIPS~2025 MMU-RAG Competition. We introduce Routing-to-RAG
(R2RAG), a research-focused retrieval-augmented generation (RAG)
architecture composed of lightweight components that dynamically adapt the
retrieval strategy based on inferred query complexity and evidence
sufficiency. The system uses smaller LLMs, enabling operation on a single
consumer-grade GPU while supporting complex research tasks. It builds on the
G-RAG system, winner of the ACM~SIGIR~2025 LiveRAG Challenge, and extends it
with modules informed by qualitative review of outputs. R2RAG won the Best
Dynamic Evaluation award in the Open Source category, demonstrating high
effectiveness with careful design and efficient use of resources. 

---
# Communication-Inspired Tokenization for Structured Image Representations 

**Authors**: Aram Davtyan, Yusuf Sahin, Yasaman Haghighi, Sebastian Stapf, Pablo Acuaviva, Alexandre Alahi, Paolo Favaro  

**Link**: [PDF](https://arxiv.org/pdf/2602.20731)  

**Abstract**: Discrete image tokenizers have emerged as a key component of modern vision and multimodal systems, providing a sequential interface for transformer-based architectures. However, most existing approaches remain primarily optimized for reconstruction and compression, often yielding tokens that capture local texture rather than object-level semantic structure. Inspired by the incremental and compositional nature of human communication, we introduce COMmunication inspired Tokenization (COMiT), a framework for learning structured discrete visual token sequences. COMiT constructs a latent message within a fixed token budget by iteratively observing localized image crops and recurrently updating its discrete representation. At each step, the model integrates new visual information while refining and reorganizing the existing token sequence. After several encoding iterations, the final message conditions a flow-matching decoder that reconstructs the full image. Both encoding and decoding are implemented within a single transformer model and trained end-to-end using a combination of flow-matching reconstruction and semantic representation alignment losses. Our experiments demonstrate that while semantic alignment provides grounding, attentive sequential tokenization is critical for inducing interpretable, object-centric token structure and substantially improving compositional generalization and relational reasoning over prior methods. 

---
# AdapTools: Adaptive Tool-based Indirect Prompt Injection Attacks on Agentic LLMs 

**Authors**: Che Wang, Jiaming Zhang, Ziqi Zhang, Zijie Wang, Yinghui Wang, Jianbo Gao, Tao Wei, Zhong Chen, Wei Yang Bryan Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.20720)  

**Abstract**: The integration of external data services (e.g., Model Context Protocol, MCP) has made large language model-based agents increasingly powerful for complex task execution. However, this advancement introduces critical security vulnerabilities, particularly indirect prompt injection (IPI) attacks. Existing attack methods are limited by their reliance on static patterns and evaluation on simple language models, failing to address the fast-evolving nature of modern AI agents. We introduce AdapTools, a novel adaptive IPI attack framework that selects stealthier attack tools and generates adaptive attack prompts to create a rigorous security evaluation environment. Our approach comprises two key components: (1) Adaptive Attack Strategy Construction, which develops transferable adversarial strategies for prompt optimization, and (2) Attack Enhancement, which identifies stealthy tools capable of circumventing task-relevance defenses. Comprehensive experimental evaluation shows that AdapTools achieves a 2.13 times improvement in attack success rate while degrading system utility by a factor of 1.78. Notably, the framework maintains its effectiveness even against state-of-the-art defense mechanisms. Our method advances the understanding of IPI attacks and provides a useful reference for future research. 

---
# Onboard-Targeted Segmentation of Straylight in Space Camera Sensors 

**Authors**: Riccardo Gallon, Fabian Schiemenz, Alessandra Menicucci, Eberhard Gill  

**Link**: [PDF](https://arxiv.org/pdf/2602.20709)  

**Abstract**: This study details an artificial intelligence (AI)-based methodology for the semantic segmentation of space camera faults. Specifically, we address the segmentation of straylight effects induced by solar presence around the camera's Field of View (FoV). Anomalous images are sourced from our published dataset. Our approach emphasizes generalization across diverse flare textures, leveraging pre-training on a public dataset (Flare7k++) including flares in various non-space contexts to mitigate the scarcity of realistic space-specific data. A DeepLabV3 model with MobileNetV3 backbone performs the segmentation task. The model design targets deployment in spacecraft resource-constrained hardware. Finally, based on a proposed interface between our model and the onboard navigation pipeline, we develop custom metrics to assess the model's performance in the system-level context. 

---
# Agile V: A Compliance-Ready Framework for AI-Augmented Engineering -- From Concept to Audit-Ready Delivery 

**Authors**: Christopher Koch, Joshua Andreas Wellbrock  

**Link**: [PDF](https://arxiv.org/pdf/2602.20684)  

**Abstract**: Current AI-assisted engineering workflows lack a built-in mechanism to maintain task-level verification and regulatory traceability at machine-speed delivery. Agile V addresses this gap by embedding independent verification and audit artifact generation into each task cycle. The framework merges Agile iteration with V-Model verification into a continuous Infinity Loop, deploying specialized AI agents for requirements, design, build, test, and compliance, governed by mandatory human approval gates. We evaluate three hypotheses: (H1) audit-ready artifacts emerge as a by-product of development, (H2) 100% requirement-level verification is achievable with independent test generation, and (H3) verified increments can be delivered with single-digit human interactions per cycle. A feasibility case study on a Hardware-in-the-Loop system (about 500 LOC, 8 requirements, 54 tests) supports all three hypotheses: audit-ready documentation was generated automatically (H1), 100% requirement-level pass rate was achieved (H2), and only 6 prompts per cycle were required (H3), yielding an estimated 10-50x cost reduction versus a COCOMO II baseline (sensitivity range from pessimistic to optimistic assumptions). We invite independent replication to validate generalizability. 

---
# UrbanFM: Scaling Urban Spatio-Temporal Foundation Models 

**Authors**: Wei Chen, Yuqian Wu, Junle Chen, Xiaofang Zhou, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20677)  

**Abstract**: Urban systems, as dynamic complex systems, continuously generate spatio-temporal data streams that encode the fundamental laws of human mobility and city evolution. While AI for Science has witnessed the transformative power of foundation models in disciplines like genomics and meteorology, urban computing remains fragmented due to "scenario-specific" models, which are overfitted to specific regions or tasks, hindering their generalizability. To bridge this gap and advance spatio-temporal foundation models for urban systems, we adopt scaling as the central perspective and systematically investigate two key questions: what to scale and how to scale. Grounded in first-principles analysis, we identify three critical dimensions: heterogeneity, correlation, and dynamics, aligning these principles with the fundamental scientific properties of urban spatio-temporal data. Specifically, to address heterogeneity through data scaling, we construct WorldST. This billion-scale corpus standardizes diverse physical signals, such as traffic flow and speed, from over 100 global cities into a unified data format. To enable computation scaling for modeling correlations, we introduce the MiniST unit, a novel split mechanism that discretizes continuous spatio-temporal fields into learnable computational units to unify representations of grid-based and sensor-based observations. Finally, addressing dynamics via architecture scaling, we propose UrbanFM, a minimalist self-attention architecture designed with limited inductive biases to autonomously learn dynamic spatio-temporal dependencies from massive data. Furthermore, we establish EvalST, the largest-scale urban spatio-temporal benchmark to date. Extensive experiments demonstrate that UrbanFM achieves remarkable zero-shot generalization across unseen cities and tasks, marking a pivotal first step toward large-scale urban spatio-temporal foundation models. 

---
# PRECTR-V2:Unified Relevance-CTR Framework with Cross-User Preference Mining, Exposure Bias Correction, and LLM-Distilled Encoder Optimization 

**Authors**: Shuzhi Cao, Rong Chen, Ailong He, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20676)  

**Abstract**: In search systems, effectively coordinating the two core objectives of search relevance matching and click-through rate (CTR) prediction is crucial for discovering users' interests and enhancing platform revenue. In our prior work PRECTR, we proposed a unified framework to integrate these two subtasks,thereby eliminating their inconsistency and leading to mutual this http URL, our previous work still faces three main challenges. First, low-active users and new users have limited search behavioral data, making it difficult to achieve effective personalized relevance preference modeling. Second, training data for ranking models predominantly come from high-relevance exposures, creating a distribution mismatch with the broader candidate space in coarse-ranking, leading to generalization bias. Third, due to the latency constraint, the original model employs an Emb+MLP architecture with a frozen BERT encoder, which prevents joint optimization and creates misalignment between representation learning and CTR fine-tuning. To solve these issues, we further reinforce our method and propose PRECTR-V2. Specifically, we mitigate the low-activity users' sparse behavior problem by mining global relevance preferences under the specific query, which facilitates effective personalized relevance modeling for cold-start scenarios. Subsequently, we construct hard negative samples through embedding noise injection and relevance label reconstruction, and optimize their relative ranking against positive samples via pairwise loss, thereby correcting exposure bias. Finally, we pretrain a lightweight transformer-based encoder via knowledge distillation from LLM and SFT on the text relevance classification task. This encoder replaces the frozen BERT module, enabling better adaptation to CTR fine-tuning and advancing beyond the traditional Emb+MLP paradigm. 

---
# CAMEL: Confidence-Gated Reflection for Reward Modeling 

**Authors**: Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Kun Xu, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2602.20670)  

**Abstract**: Reward models play a fundamental role in aligning large language models with human preferences. Existing methods predominantly follow two paradigms: scalar discriminative preference models, which are efficient but lack interpretability, and generative judging models, which offer richer reasoning at the cost of higher computational overhead. We observe that the log-probability margin between verdict tokens strongly correlates with prediction correctness, providing a reliable proxy for instance difficulty without additional inference cost. Building on this insight, we propose CAMEL, a confidence-gated reflection framework that performs a lightweight single-token preference decision first and selectively invokes reflection only for low-confidence instances. To induce effective self-correction, we train the model via reinforcement learning with counterfactual prefix augmentation, which exposes the model to diverse initial verdicts and encourages genuine revision. Empirically, CAMEL achieves state-of-the-art performance on three widely used reward-model benchmarks with 82.9% average accuracy, surpassing the best prior model by 3.2% and outperforming 70B-parameter models using only 14B parameters, while establishing a strictly better accuracy-efficiency Pareto frontier. 

---
# Vision-Language Models for Ergonomic Assessment of Manual Lifting Tasks: Estimating Horizontal and Vertical Hand Distances from RGB Video 

**Authors**: Mohammad Sadra Rajabi, Aanuoluwapo Ojelade, Sunwook Kim, Maury A. Nussbaum  

**Link**: [PDF](https://arxiv.org/pdf/2602.20658)  

**Abstract**: Manual lifting tasks are a major contributor to work-related musculoskeletal disorders, and effective ergonomic risk assessment is essential for quantifying physical exposure and informing ergonomic interventions. The Revised NIOSH Lifting Equation (RNLE) is a widely used ergonomic risk assessment tool for lifting tasks that relies on six task variables, including horizontal (H) and vertical (V) hand distances; such distances are typically obtained through manual measurement or specialized sensing systems and are difficult to use in real-world environments. We evaluated the feasibility of using innovative vision-language models (VLMs) to non-invasively estimate H and V from RGB video streams. Two multi-stage VLM-based pipelines were developed: a text-guided detection-only pipeline and a detection-plus-segmentation pipeline. Both pipelines used text-guided localization of task-relevant regions of interest, visual feature extraction from those regions, and transformer-based temporal regression to estimate H and V at the start and end of a lift. For a range of lifting tasks, estimation performance was evaluated using leave-one-subject-out validation across the two pipelines and seven camera view conditions. Results varied significantly across pipelines and camera view conditions, with the segmentation-based, multi-view pipeline consistently yielding the smallest errors, achieving mean absolute errors of approximately 6-8 cm when estimating H and 5-8 cm when estimating V. Across pipelines and camera view configurations, pixel-level segmentation reduced estimation error by approximately 20-30% for H and 35-40% for V relative to the detection-only pipeline. These findings support the feasibility of VLM-based pipelines for video-based estimation of RNLE distance parameters. 

---
# Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression 

**Authors**: Chenyue Yu, Lingao Xiao, Jinhong Deng, Ivor W. Tsang, Yang He  

**Link**: [PDF](https://arxiv.org/pdf/2602.20650)  

**Abstract**: Large-scale image datasets are fundamental to deep learning, but their high storage demands pose challenges for deployment in resource-constrained environments. While existing approaches reduce dataset size by discarding samples, they often ignore the significant redundancy within each image -- particularly in the color space. To address this, we propose Dataset Color Quantization (DCQ), a unified framework that compresses visual datasets by reducing color-space redundancy while preserving information crucial for model training. DCQ achieves this by enforcing consistent palette representations across similar images, selectively retaining semantically important colors guided by model perception, and maintaining structural details necessary for effective feature learning. Extensive experiments across CIFAR-10, CIFAR-100, Tiny-ImageNet, and ImageNet-1K show that DCQ significantly improves training performance under aggressive compression, offering a scalable and robust solution for dataset-level storage reduction. Code is available at \href{this https URL}{this https URL}. 

---
# TrajGPT-R: Generating Urban Mobility Trajectory with Reinforcement Learning-Enhanced Generative Pre-trained Transformer 

**Authors**: Jiawei Wang, Chuang Yang, Jiawei Yong, Xiaohang Xu, Hongjun Wang, Noboru Koshizuka, Shintaro Fukushima, Ryosuke Shibasaki, Renhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20643)  

**Abstract**: Mobility trajectories are essential for understanding urban dynamics and enhancing urban planning, yet access to such data is frequently hindered by privacy concerns. This research introduces a transformative framework for generating large-scale urban mobility trajectories, employing a novel application of a transformer-based model pre-trained and fine-tuned through a two-phase process. Initially, trajectory generation is conceptualized as an offline reinforcement learning (RL) problem, with a significant reduction in vocabulary space achieved during tokenization. The integration of Inverse Reinforcement Learning (IRL) allows for the capture of trajectory-wise reward signals, leveraging historical data to infer individual mobility preferences. Subsequently, the pre-trained model is fine-tuned using the constructed reward model, effectively addressing the challenges inherent in traditional RL-based autoregressive methods, such as long-term credit assignment and handling of sparse reward environments. Comprehensive evaluations on multiple datasets illustrate that our framework markedly surpasses existing models in terms of reliability and diversity. Our findings not only advance the field of urban mobility modeling but also provide a robust methodology for simulating urban data, with significant implications for traffic management and urban development planning. The implementation is publicly available at this https URL. 

---
# SurgAtt-Tracker: Online Surgical Attention Tracking via Temporal Proposal Reranking and Motion-Aware Refinement 

**Authors**: Rulin Zhou, Guankun Wang, An Wang, Yujie Ma, Lixin Ouyang, Bolin Cui, Junyan Li, Chaowei Zhu, Mingyang Li, Ming Chen, Xiaopin Zhong, Peng Lu, Jiankun Wang, Xianming Liu, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2602.20636)  

**Abstract**: Accurate and stable field-of-view (FoV) guidance is critical for safe and efficient minimally invasive surgery, yet existing approaches often conflate visual attention estimation with downstream camera control or rely on direct object-centric assumptions. In this work, we formulate surgical attention tracking as a spatio-temporal learning problem and model surgeon focus as a dense attention heatmap, enabling continuous and interpretable frame-wise FoV guidance. We propose SurgAtt-Tracker, a holistic framework that robustly tracks surgical attention by exploiting temporal coherence through proposal-level reranking and motion-aware refinement, rather than direct regression. To support systematic training and evaluation, we introduce SurgAtt-1.16M, a large-scale benchmark with a clinically grounded annotation protocol that enables comprehensive heatmap-based attention analysis across procedures and institutions. Extensive experiments on multiple surgical datasets demonstrate that SurgAtt-Tracker consistently achieves state-of-the-art performance and strong robustness under occlusion, multi-instrument interference, and cross-domain settings. Beyond attention tracking, our approach provides a frame-wise FoV guidance signal that can directly support downstream robotic FoV planning and automatic camera control. 

---
# Enhancing Hate Speech Detection on Social Media: A Comparative Analysis of Machine Learning Models and Text Transformation Approaches 

**Authors**: Saurabh Mishra, Shivani Thakur, Radhika Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20634)  

**Abstract**: The proliferation of hate speech on social media platforms has necessitated the development of effective detection and moderation tools. This study evaluates the efficacy of various machine learning models in identifying hate speech and offensive language and investigates the potential of text transformation techniques to neutralize such content. We compare traditional models like CNNs and LSTMs with advanced neural network models such as BERT and its derivatives, alongside exploring hybrid models that combine different architectural features. Our results indicate that while advanced models like BERT show superior accuracy due to their deep contextual understanding, hybrid models exhibit improved capabilities in certain scenarios. Furthermore, we introduce innovative text transformation approaches that convert negative expressions into neutral ones, thereby potentially mitigating the impact of harmful content. The implications of these findings are discussed, highlighting the strengths and limitations of current technologies and proposing future directions for more robust hate speech detection systems. 

---
# OptiLeak: Efficient Prompt Reconstruction via Reinforcement Learning in Multi-tenant LLM Services 

**Authors**: Longxiang Wang, Xiang Zheng, Xuhao Zhang, Yao Zhang, Ye Wu, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20595)  

**Abstract**: Multi-tenant LLM serving frameworks widely adopt shared Key-Value caches to enhance efficiency. However, this creates side-channel vulnerabilities enabling prompt leakage attacks. Prior studies identified these attack surfaces yet focused on expanding attack vectors rather than optimizing attack performance, reporting impractically high attack costs that underestimate the true privacy risk. We propose OptiLeak, a reinforcement learning-enhanced framework that maximizes prompt reconstruction efficiency through two-stage fine-tuning. Our key insight is that domain-specific ``hard tokens'' -- terms difficult to predict yet carrying sensitive information -- can be automatically identified via likelihood ranking and used to construct preference pairs for Direct Preference Optimization, eliminating manual annotation. This enables effective preference alignment while avoiding the overfitting issues of extended supervised fine-tuning. Evaluated on three benchmarks spanning medical and financial domains, OptiLeak achieves up to $12.48\times$ reduction in average requests per token compared to baseline approaches, with consistent improvements across model scales from 3B to 14B parameters. Our findings demonstrate that cache-based prompt leakage poses a more severe threat than previously reported, underscoring the need for robust cache isolation in production deployments. 

---
# Personal Information Parroting in Language Models 

**Authors**: Nishant Subramani, Kshitish Ghate, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2602.20580)  

**Abstract**: Modern language models (LM) are trained on large scrapes of the Web, containing millions of personal information (PI) instances, many of which LMs memorize, increasing privacy risks. In this work, we develop the regexes and rules (R&R) detector suite to detect email addresses, phone numbers, and IP addresses, which outperforms the best regex-based PI detectors. On a manually curated set of 483 instances of PI, we measure memorization: finding that 13.6% are parroted verbatim by the Pythia-6.9b model, i.e., when the model is prompted with the tokens that precede the PI in the original document, greedy decoding generates the entire PI span exactly. We expand this analysis to study models of varying sizes (160M-6.9B) and pretraining time steps (70k-143k iterations) in the Pythia model suite and find that both model size and amount of pretraining are positively correlated with memorization. Even the smallest model, Pythia-160m, parrots 2.7% of the instances exactly. Consequently, we strongly recommend that pretraining datasets be aggressively filtered and anonymized to minimize PI parroting. 

---
# What Drives Students' Use of AI Chatbots? Technology Acceptance in Conversational AI 

**Authors**: Griffin Pitts, Sanaz Motamedi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20547)  

**Abstract**: Conversational AI tools have been rapidly adopted by students and are becoming part of their learning routines. To understand what drives this adoption, we draw on the Technology Acceptance Model (TAM) and examine how perceived usefulness and perceived ease of use relate to students' behavioral intention to use conversational AI that generates responses for learning tasks. We extend TAM by incorporating trust, perceived enjoyment, and subjective norms as additional factors that capture social and affective influences and uncertainty around AI outputs.
Using partial least squares structural equation modeling, we find perceived usefulness remains the strongest predictor of students' intention to use conversational AI. However, perceived ease of use does not exert a significant direct effect on behavioral intention once other factors are considered, operating instead indirectly through perceived usefulness. Trust and subjective norms significantly influence perceptions of usefulness, while perceived enjoyment exerts both a direct and indirect effect on usage intentions. These findings suggest that adoption decisions for conversational AI systems are influenced less by effort-related considerations and more by confidence in system outputs, affective engagement, and social context. Future research is needed to further examine how these acceptance relationships generalize across different conversational systems and usage contexts. 

---
# Maximin Share Guarantees via Limited Cost-Sensitive Sharing 

**Authors**: Hana Salavcova, Martin Černý, Arpita Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2602.20541)  

**Abstract**: We study the problem of fairly allocating indivisible goods when limited sharing is allowed, that is, each good may be allocated to up to $k$ agents, while incurring a cost for sharing. While classic maximin share (MMS) allocations may not exist in many instances, we demonstrate that allowing controlled sharing can restore fairness guarantees that are otherwise unattainable in certain scenarios. (1) Our first contribution shows that exact maximin share (MMS) allocations are guaranteed to exist whenever goods are allowed to be cost-sensitively shared among at least half of the agents and the number of agents is even; for odd numbers of agents, we obtain a slightly weaker MMS guarantee. (2) We further design a Shared Bag-Filling Algorithm that guarantees a $(1 - C)(k - 1)$-approximate MMS allocation, where $C$ is the maximum cost of sharing a good. Notably, when $(1 - C)(k - 1) \geq 1$, our algorithm recovers an exact MMS allocation. (3) We additionally introduce the Sharing Maximin Share (SMMS) fairness notion, a natural extension of MMS to the $k$-sharing setting. (4) We show that SMMS allocations always exist under identical utilities and for instances with two agents. (5) We construct a counterexample to show the impossibility of the universal existence of an SMMS allocation. (6) Finally, we establish a connection between SMMS and constrained MMS (CMMS), yielding approximation guarantees for SMMS via existing CMMS results. These contributions provide deep theoretical insights for the problem of fair resource allocation when a limited sharing of resources are allowed in multi-agent environments. 

---
# Actor-Curator: Co-adaptive Curriculum Learning via Policy-Improvement Bandits for RL Post-Training 

**Authors**: Zhengyao Gu, Jonathan Light, Raul Astudillo, Ziyu Ye, Langzhou He, Henry Peng Zou, Wei Cheng, Santiago Paternain, Philip S. Yu, Yisong Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.20532)  

**Abstract**: Post-training large foundation models with reinforcement learning typically relies on massive and heterogeneous datasets, making effective curriculum learning both critical and challenging. In this work, we propose ACTOR-CURATOR, a scalable and fully automated curriculum learning framework for reinforcement learning post-training of large language models (LLMs). ACTOR-CURATOR learns a neural curator that dynamically selects training problems from large problem banks by directly optimizing for expected policy performance improvement. We formulate problem selection as a non-stationary stochastic bandit problem, derive a principled loss function based on online stochastic mirror descent, and establish regret guarantees under partial feedback. Empirically, ACTOR-CURATOR consistently outperforms uniform sampling and strong curriculum baselines across a wide range of challenging reasoning benchmarks, demonstrating improved training stability and efficiency. Notably, it achieves relative gains of 28.6% on AIME2024 and 30.5% on ARC-1D over the strongest baseline and up to 80% speedup. These results suggest that ACTOR-CURATOR is a powerful and practical approach for scalable LLM post-training. 

---
# A Generalized Apprenticeship Learning Framework for Capturing Evolving Student Pedagogical Strategies 

**Authors**: Md Mirajul Islam, Xi Yang, Adittya Soukarjya Saha, Rajesh Debnath, Min Chi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20527)  

**Abstract**: Reinforcement Learning (RL) and Deep Reinforcement Learning (DRL) have advanced rapidly in recent years and have been successfully applied to e-learning environments like intelligent tutoring systems (ITSs). Despite great success, the broader application of DRL to educational technologies has been limited due to major challenges such as sample inefficiency and difficulty designing the reward function. In contrast, Apprenticeship Learning (AL) uses a few expert demonstrations to infer the expert's underlying reward functions and derive decision-making policies that generalize and replicate optimal behavior. In this work, we leverage a generalized AL framework, THEMES, to induce effective pedagogical policies by capturing the complexities of the expert student learning process, where multiple reward functions may dynamically evolve over time. We evaluate the effectiveness of THEMES against six state-of-the-art baselines, demonstrating its superior performance and highlighting its potential as a powerful alternative for inducing effective pedagogical policies and show that it can achieve high performance, with an AUC of 0.899 and a Jaccard of 0.653, using only 18 trajectories of a previous semester to predict student pedagogical decisions in a later semester. 

---
# How Do Inpainting Artifacts Propagate to Language? 

**Authors**: Pratham Yashwante, Davit Abrahamyan, Shresth Grover, Sukruth Rao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20520)  

**Abstract**: We study how visual artifacts introduced by diffusion-based inpainting affect language generation in vision-language models. We use a two-stage diagnostic setup in which masked image regions are reconstructed and then provided to captioning models, enabling controlled comparisons between captions generated from original and reconstructed inputs. Across multiple datasets, we analyze the relationship between reconstruction fidelity and downstream caption quality. We observe consistent associations between pixel-level and perceptual reconstruction metrics and both lexical and semantic captioning performance. Additional analysis of intermediate visual representations and attention patterns shows that inpainting artifacts lead to systematic, layer-dependent changes in model behavior. Together, these results provide a practical diagnostic framework for examining how visual reconstruction quality influences language generation in multimodal systems. 

---
# LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration 

**Authors**: Peiliang Cai, Jiacheng Liu, Haowen Xu, Xinyu Wang, Chang Zou, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20497)  

**Abstract**: Diffusion models have achieved remarkable success in image and video generation tasks. However, the high computational demands of Diffusion Transformers (DiTs) pose a significant challenge to their practical deployment. While feature caching is a promising acceleration strategy, existing methods based on simple reusing or training-free forecasting struggle to adapt to the complex, stage-dependent dynamics of the diffusion process, often resulting in quality degradation and failing to maintain consistency with the standard denoising process. To address this, we propose a LEarnable Stage-Aware (LESA) predictor framework based on two-stage training. Our approach leverages a Kolmogorov-Arnold Network (KAN) to accurately learn temporal feature mappings from data. We further introduce a multi-stage, multi-expert architecture that assigns specialized predictors to different noise-level stages, enabling more precise and robust feature forecasting. Extensive experiments show our method achieves significant acceleration while maintaining high-fidelity generation. Experiments demonstrate 5.00x acceleration on FLUX.1-dev with minimal quality degradation (1.0% drop), 6.25x speedup on Qwen-Image with a 20.2% quality improvement over the previous SOTA (TaylorSeer), and 5.00x acceleration on HunyuanVideo with a 24.7% PSNR improvement over TaylorSeer. State-of-the-art performance on both text-to-image and text-to-video synthesis validates the effectiveness and generalization capability of our training-based framework across different models. Our code is included in the supplementary materials and will be released on GitHub. 

---
# Wireless Federated Multi-Task LLM Fine-Tuning via Sparse-and-Orthogonal LoRA 

**Authors**: Nuocheng Yang, Sihua Wang, Ouwen Huan, Mingzhe Chen, Tony Q. S. Quek, Changchuan Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20492)  

**Abstract**: Decentralized federated learning (DFL) based on low-rank adaptation (LoRA) enables mobile devices with multi-task datasets to collaboratively fine-tune a large language model (LLM) by exchanging locally updated parameters with a subset of neighboring devices via wireless connections for knowledge this http URL, directly aggregating parameters fine-tuned on heterogeneous datasets induces three primary issues across the DFL life-cycle: (i) \textit{catastrophic knowledge forgetting during fine-tuning process}, arising from conflicting update directions caused by data heterogeneity; (ii) \textit{inefficient communication and convergence during model aggregation process}, due to bandwidth-intensive redundant model transmissions; and (iii) \textit{multi-task knowledge interference during inference process}, resulting from incompatible knowledge representations coexistence during inference. To address these issues in a fully decentralized scenario, we first propose a sparse-and-orthogonal LoRA that ensures orthogonality between model updates to eliminate direction conflicts during this http URL, we analyze how device connection topology affects multi-task performance, prompting a cluster-based topology design during this http URL, we propose an implicit mixture of experts (MoE) mechanism to avoid the coexistence of incompatible knowledge during inference. Simulation results demonstrate that the proposed approach effectively reduces communication resource consumption by up to $73\%$ and enhances average performance by $5\%$ compared with the traditional LoRA method. 

---
# Hybrid LLM-Embedded Dialogue Agents for Learner Reflection: Designing Responsive and Theory-Driven Interactions 

**Authors**: Paras Sharma, YuePing Sha, Janet Shufor Bih Epse Fofang, Brayden Yan, Jess A. Turner, Nicole Balay, Hubert O. Asare, Angela E.B. Stewart, Erin Walker  

**Link**: [PDF](https://arxiv.org/pdf/2602.20486)  

**Abstract**: Dialogue systems have long supported learner reflections, with theoretically grounded, rule-based designs offering structured scaffolding but often struggling to respond to shifts in engagement. Large Language Models (LLMs), in contrast, can generate context-sensitive responses but are not informed by decades of research on how learning interactions should be structured, raising questions about their alignment with pedagogical theories. This paper presents a hybrid dialogue system that embeds LLM responsiveness within a theory-aligned, rule-based framework to support learner reflections in a culturally responsive robotics summer camp. The rule-based structure grounds dialogue in self-regulated learning theory, while the LLM decides when and how to prompt deeper reflections, responding to evolving conversation context. We analyze themes across dialogues to explore how our hybrid system shaped learner reflections. Our findings indicate that LLM-embedded dialogues supported richer learner reflections on goals and activities, but also introduced challenges due to repetitiveness and misalignment in prompts, reducing engagement. 

---
# VINA: Variational Invertible Neural Architectures 

**Authors**: Shubhanshu Shekhar, Mohammad Javad Khojasteh, Ananya Acharya, Tony Tohme, Kamal Youcef-Toumi  

**Link**: [PDF](https://arxiv.org/pdf/2602.20480)  

**Abstract**: The distinctive architectural features of normalizing flows (NFs), notably bijectivity and tractable Jacobians, make them well-suited for generative modeling. Invertible neural networks (INNs) build on these principles to address supervised inverse problems, enabling direct modeling of both forward and inverse mappings. In this paper, we revisit these architectures from both theoretical and practical perspectives and address a key gap in the literature: the lack of theoretical guarantees on approximation quality under realistic assumptions, whether for posterior inference in INNs or for generative modeling with NFs.
We introduce a unified framework for INNs and NFs based on variational unsupervised loss functions, inspired by analogous formulations in related areas such as generative adversarial networks (GANs) and the Precision-Recall divergence for training normalizing flows. Within this framework, we derive theoretical performance guarantees, quantifying posterior accuracy for INNs and distributional accuracy for NFs, under assumptions that are weaker and more practically realistic than those used in prior work.
Building on these theoretical results, we conduct extensive case studies to distill general design principles and practical guidelines. We conclude by demonstrating the effectiveness of our approach on a realistic ocean-acoustic inversion problem. 

---
# Elimination-compensation pruning for fully-connected neural networks 

**Authors**: Enrico Ballini, Luca Muscarnera, Alessio Fumagalli, Anna Scotti, Francesco Regazzoni  

**Link**: [PDF](https://arxiv.org/pdf/2602.20467)  

**Abstract**: The unmatched ability of Deep Neural Networks in capturing complex patterns in large and noisy datasets is often associated with their large hypothesis space, and consequently to the vast amount of parameters that characterize model architectures. Pruning techniques affirmed themselves as valid tools to extract sparse representations of neural networks parameters, carefully balancing between compression and preservation of information. However, a fundamental assumption behind pruning is that expendable weights should have small impact on the error of the network, while highly important weights should tend to have a larger influence on the inference. We argue that this idea could be generalized; what if a weight is not simply removed but also compensated with a perturbation of the adjacent bias, which does not contribute to the network sparsity? Our work introduces a novel pruning method in which the importance measure of each weight is computed considering the output behavior after an optimal perturbation of its adjacent bias, efficiently computable by automatic differentiation. These perturbations can be then applied directly after the removal of each weight, independently of each other. After deriving analytical expressions for the aforementioned quantities, numerical experiments are conducted to benchmark this technique against some of the most popular pruning strategies, demonstrating an intrinsic efficiency of the proposed approach in very diverse machine learning scenarios. Finally, our findings are discussed and the theoretical implications of our results are presented. 

---
# Protein Language Models Diverge from Natural Language: Comparative Analysis and Improved Inference 

**Authors**: Anna Hart, Chi Han, Jeonghwan Kim, Huimin Zhao, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2602.20449)  

**Abstract**: Modern Protein Language Models (PLMs) apply transformer-based model architectures from natural language processing to biological sequences, predicting a variety of protein functions and properties. However, protein language has key differences from natural language, such as a rich functional space despite a vocabulary of only 20 amino acids. These differences motivate research into how transformer-based architectures operate differently in the protein domain and how we can better leverage PLMs to solve protein-related tasks. In this work, we begin by directly comparing how the distribution of information stored across layers of attention heads differs between the protein and natural language domain. Furthermore, we adapt a simple early-exit technique-originally used in the natural language domain to improve efficiency at the cost of performance-to achieve both increased accuracy and substantial efficiency gains in protein non-structural property prediction by allowing the model to automatically select protein representations from the intermediate layers of the PLMs for the specific task and protein at hand. We achieve performance gains ranging from 0.4 to 7.01 percentage points while simultaneously improving efficiency by over 10 percent across models and non-structural prediction tasks. Our work opens up an area of research directly comparing how language models change behavior when moved into the protein domain and advances language modeling in biological domains. 

---
# Imputation of Unknown Missingness in Sparse Electronic Health Records 

**Authors**: Jun Han, Josue Nassar, Sanjit Singh Batra, Aldo Cordova-Palomera, Vijay Nori, Robert E. Tillman  

**Link**: [PDF](https://arxiv.org/pdf/2602.20442)  

**Abstract**: Machine learning holds great promise for advancing the field of medicine, with electronic health records (EHRs) serving as a primary data source. However, EHRs are often sparse and contain missing data due to various challenges and limitations in data collection and sharing between healthcare providers. Existing techniques for imputing missing values predominantly focus on known unknowns, such as missing or unavailable values of lab test results; most do not explicitly address situations where it is difficult to distinguish what is missing. For instance, a missing diagnosis code in an EHR could signify either that the patient has not been diagnosed with the condition or that a diagnosis was made, but not shared by a provider. Such situations fall into the paradigm of unknown unknowns. To address this challenge, we develop a general purpose algorithm for denoising data to recover unknown missing values in binary EHRs. We design a transformer-based denoising neural network where the output is thresholded adaptively to recover values in cases where we predict data are missing. Our results demonstrate improved accuracy in denoising medical codes within a real EHR dataset compared to existing imputation approaches and leads to increased performance on downstream tasks using the denoised data. In particular, when applying our method to a real world application, predicting hospital readmission from EHRs, our method achieves statistically significant improvement over all existing baselines. 

---
# Examining and Addressing Barriers to Diversity in LLM-Generated Ideas 

**Authors**: Yuting Deng, Melanie Brucks, Olivier Toubia  

**Link**: [PDF](https://arxiv.org/pdf/2602.20408)  

**Abstract**: Ideas generated by independent samples of humans tend to be more diverse than ideas generated from independent LLM samples, raising concerns that widespread reliance on LLMs could homogenize ideation and undermine innovation at a societal level. Drawing on cognitive psychology, we identify (both theoretically and empirically) two mechanisms undermining LLM idea diversity. First, at the individual level, LLMs exhibit fixation just as humans do, where early outputs constrain subsequent ideation. Second, at the collective level, LLMs aggregate knowledge into a unified distribution rather than exhibiting the knowledge partitioning inherent to human populations, where each person occupies a distinct region of the knowledge space. Through four studies, we demonstrate that targeted prompting interventions can address each mechanism independently: Chain-of-Thought (CoT) prompting reduces fixation by encouraging structured reasoning (only in LLMs, not humans), while ordinary personas (versus "creative entrepreneurs" such as Steve Jobs) improve knowledge partitioning by serving as diverse sampling cues, anchoring generation in distinct regions of the semantic space. Combining both approaches produces the highest idea diversity, outperforming humans. These findings offer a theoretically grounded framework for understanding LLM idea diversity and practical strategies for human-AI collaborations that leverage AI's efficiency without compromising the diversity essential to a healthy innovation ecosystem. 

---
# Three Concrete Challenges and Two Hopes for the Safety of Unsupervised Elicitation 

**Authors**: Callum Canavan, Aditya Shrivastava, Allison Qi, Jonathan Michala, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2602.20400)  

**Abstract**: To steer language models towards truthful outputs on tasks which are beyond human capability, previous work has suggested training models on easy tasks to steer them on harder ones (easy-to-hard generalization), or using unsupervised training algorithms to steer models with no external labels at all (unsupervised elicitation). Although techniques from both paradigms have been shown to improve model accuracy on a wide variety of tasks, we argue that the datasets used for these evaluations could cause overoptimistic evaluation results. Unlike many real-world datasets, they often (1) have no features with more salience than truthfulness, (2) have balanced training sets, and (3) contain only data points to which the model can give a well-defined answer. We construct datasets that lack each of these properties to stress-test a range of standard unsupervised elicitation and easy-to-hard generalization techniques. We find that no technique reliably performs well on any of these challenges. We also study ensembling and combining easy-to-hard and unsupervised techniques, and find they only partially mitigate performance degradation due to these challenges. We believe that overcoming these challenges should be a priority for future work on unsupervised elicitation. 

---
# Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems 

**Authors**: Mukul Chhabra, Luigi Medrano, Arush Verma  

**Link**: [PDF](https://arxiv.org/pdf/2602.20379)  

**Abstract**: Enterprise Retrieval-Augmented Generation (RAG) assistants operate in multi-turn, case-based workflows such as technical support and IT operations, where evaluation must reflect operational constraints, structured identifiers (e.g., error codes, versions), and resolution workflows. Existing RAG evaluation frameworks are primarily designed for benchmark-style or single-turn settings and often fail to capture enterprise-specific failure modes such as case misidentification, workflow misalignment, and partial resolution across turns.
We present a case-aware LLM-as-a-Judge evaluation framework for enterprise multi-turn RAG systems. The framework evaluates each turn using eight operationally grounded metrics that separate retrieval quality, grounding fidelity, answer utility, precision integrity, and case/workflow alignment. A severity-aware scoring protocol reduces score inflation and improves diagnostic clarity across heterogeneous enterprise cases. The system uses deterministic prompting with strict JSON outputs, enabling scalable batch evaluation, regression testing, and production monitoring.
Through a comparative study of two instruction-tuned models across short and long workflows, we show that generic proxy metrics provide ambiguous signals, while the proposed framework exposes enterprise-critical tradeoffs that are actionable for system improvement. 

---
# Learning During Detection: Continual Learning for Neural OFDM Receivers via DMRS 

**Authors**: Mohanad Obeed, Ming Jian  

**Link**: [PDF](https://arxiv.org/pdf/2602.20361)  

**Abstract**: Deep neural networks (DNNs) have been increasingly explored for receiver design because they can handle complex environments without relying on explicit channel models. Nevertheless, because communication channels change rapidly, their distributions can shift over time, often making periodic retraining necessary. This paper proposes a zero-overhead online and continual learning framework for orthogonal frequency-division multiplexing (OFDM) neural receivers that directly detect the soft bits of received signals. Unlike conventional fine-tuning methods that rely on dedicated training intervals or full resource grids, our approach leverages existing demodulation reference signals (DMRS) to simultaneously enable signal demodulation and model adaptation. We introduce three pilot designs: fully randomized, hybrid, and additional pilots that flexibly support joint demodulation and learning. To accommodate these pilot designs, we develop two receiver architectures: (i) a parallel design that separates inference and fine-tuning for uninterrupted operation, and (ii) a forward-pass reusing design that reduces computational complexity. Simulation results show that the proposed method effectively tracks both slow and fast channel distribution variations without additional overhead, service interruption, or catastrophic performance degradation under distribution shift. 

---
# Hierarchical Molecular Representation Learning via Fragment-Based Self-Supervised Embedding Prediction 

**Authors**: Jiele Wu, Haozhe Ma, Zhihan Guo, Thanh Vinh Vo, Tze Yun Leong  

**Link**: [PDF](https://arxiv.org/pdf/2602.20344)  

**Abstract**: Graph self-supervised learning (GSSL) has demonstrated strong potential for generating expressive graph embeddings without the need for human annotations, making it particularly valuable in domains with high labeling costs such as molecular graph analysis. However, existing GSSL methods mostly focus on node- or edge-level information, often ignoring chemically relevant substructures which strongly influence molecular properties. In this work, we propose Graph Semantic Predictive Network (GraSPNet), a hierarchical self-supervised framework that explicitly models both atomic-level and fragment-level semantics. GraSPNet decomposes molecular graphs into chemically meaningful fragments without predefined vocabularies and learns node- and fragment-level representations through multi-level message passing with masked semantic prediction at both levels. This hierarchical semantic supervision enables GraSPNet to learn multi-resolution structural information that is both expressive and transferable. Extensive experiments on multiple molecular property prediction benchmarks demonstrate that GraSPNet learns chemically meaningful representations and consistently outperforms state-of-the-art GSSL methods in transfer learning settings. 

---
# No One Size Fits All: QueryBandits for Hallucination Mitigation 

**Authors**: Nicole Cho, William Watson, Alec Koppel, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20332)  

**Abstract**: Advanced reasoning capabilities in Large Language Models (LLMs) have led to more frequent hallucinations; yet most mitigation work focuses on open-source models for post-hoc detection and parameter editing. The dearth of studies focusing on hallucinations in closed-source models is especially concerning, as they constitute the vast majority of models in institutional deployments. We introduce QueryBandits, a model-agnostic contextual bandit framework that adaptively learns online to select the optimal query-rewrite strategy by leveraging an empirically validated and calibrated reward function. Across 16 QA scenarios, our top QueryBandit (Thompson Sampling) achieves an 87.5% win rate over a No-Rewrite baseline and outperforms zero-shot static policies (e.g., Paraphrase or Expand) by 42.6% and 60.3%, respectively. Moreover, all contextual bandits outperform vanilla bandits across all datasets, with higher feature variance coinciding with greater variance in arm selection. This substantiates our finding that there is no single rewrite policy optimal for all queries. We also discover that certain static policies incur higher cumulative regret than No-Rewrite, indicating that an inflexible query-rewriting policy can worsen hallucinations. Thus, learning an online policy over semantic features with QueryBandits can shift model behavior purely through forward-pass mechanisms, enabling its use with closed-source models and bypassing the need for retraining or gradient-based adaptation. 

---
# Circuit Tracing in Vision-Language Models: Understanding the Internal Mechanisms of Multimodal Thinking 

**Authors**: Jingcheng Yang, Tianhu Xiong, Shengyi Qian, Klara Nahrstedt, Mingyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20330)  

**Abstract**: Vision-language models (VLMs) are powerful but remain opaque black boxes. We introduce the first framework for transparent circuit tracing in VLMs to systematically analyze multimodal reasoning. By utilizing transcoders, attribution graphs, and attention-based methods, we uncover how VLMs hierarchically integrate visual and semantic concepts. We reveal that distinct visual feature circuits can handle mathematical reasoning and support cross-modal associations. Validated through feature steering and circuit patching, our framework proves these circuits are causal and controllable, laying the groundwork for more explainable and reliable VLMs. 

---
# Learning Physical Principles from Interaction: Self-Evolving Planning via Test-Time Memory 

**Authors**: Haoyang Li, Yang You, Hao Su, Leonidas Guibas  

**Link**: [PDF](https://arxiv.org/pdf/2602.20323)  

**Abstract**: Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions. 

---
# Fast Spectrogram Event Extraction via Offline Self-Supervised Learning: From Fusion Diagnostics to Bioacoustics 

**Authors**: Nathaniel Chen, Kouroche Bouchiat, Peter Steiner, Andrew Rothstein, David Smith, Max Austin, Mike van Zeeland, Azarakhsh Jalalvand, Egemen Kolemen  

**Link**: [PDF](https://arxiv.org/pdf/2602.20317)  

**Abstract**: Next-generation fusion facilities like ITER face a "data deluge," generating petabytes of multi-diagnostic signals daily that challenge manual analysis. We present a "signals-first" self-supervised framework for the automated extraction of coherent and transient modes from high-noise time-frequency data. We also develop a general-purpose method and tool for extracting coherent, quasi-coherent, and transient modes for fluctuation measurements in tokamaks by employing non-linear optimal techniques in multichannel signal processing with a fast neural network surrogate on fast magnetics, electron cyclotron emission, CO2 interferometers, and beam emission spectroscopy measurements from DIII-D. Results are tested on data from DIII-D, TJ-II, and non-fusion spectrograms. With an inference latency of 0.5 seconds, this framework enables real-time mode identification and large-scale automated database generation for advanced plasma control. Repository is in this https URL. 

---
# Shape-informed cardiac mechanics surrogates in data-scarce regimes via geometric encoding and generative augmentation 

**Authors**: Davide Carrara, Marc Hirschvogel, Francesca Bonizzoni, Stefano Pagani, Simone Pezzuto, Francesco Regazzoni  

**Link**: [PDF](https://arxiv.org/pdf/2602.20306)  

**Abstract**: High-fidelity computational models of cardiac mechanics provide mechanistic insight into the heart function but are computationally prohibitive for routine clinical use. Surrogate models can accelerate simulations, but generalization across diverse anatomies is challenging, particularly in data-scarce settings. We propose a two-step framework that decouples geometric representation from learning the physics response, to enable shape-informed surrogate modeling under data-scarce conditions. First, a shape model learns a compact latent representation of left ventricular geometries. The learned latent space effectively encodes anatomies and enables synthetic geometries generation for data augmentation. Second, a neural field-based surrogate model, conditioned on this geometric encoding, is trained to predict ventricular displacement under external loading. The proposed architecture performs positional encoding by using universal ventricular coordinates, which improves generalization across diverse anatomies. Geometric variability is encoded using two alternative strategies, which are systematically compared: a PCA-based approach suitable for working with point cloud representations of geometries, and a DeepSDF-based implicit neural representation learned directly from point clouds. Overall, our results, obtained on idealized and patient-specific datasets, show that the proposed approaches allow for accurate predictions and generalization to unseen geometries, and robustness to noisy or sparsely sampled inputs. 

---
# What Makes a Good Query? Measuring the Impact of Human-Confusing Linguistic Features on LLM Performance 

**Authors**: William Watson, Nicole Cho, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2602.20300)  

**Abstract**: Large Language Model (LLM) hallucinations are usually treated as defects of the model or its decoding strategy. Drawing on classical linguistics, we argue that a query's form can also shape a listener's (and model's) response. We operationalize this insight by constructing a 22-dimension query feature vector covering clause complexity, lexical rarity, and anaphora, negation, answerability, and intention grounding, all known to affect human comprehension. Using 369,837 real-world queries, we ask: Are there certain types of queries that make hallucination more likely? A large-scale analysis reveals a consistent "risk landscape": certain features such as deep clause nesting and underspecification align with higher hallucination propensity. In contrast, clear intention grounding and answerability align with lower hallucination rates. Others, including domain specificity, show mixed, dataset- and model-dependent effects. Thus, these findings establish an empirically observable query-feature representation correlated with hallucination risk, paving the way for guided query rewriting and future intervention studies. 

---
# InterviewSim: A Scalable Framework for Interview-Grounded Personality Simulation 

**Authors**: Yu Li, Pranav Narayanan Venkit, Yada Pruksachatkun, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20294)  

**Abstract**: Simulating real personalities with large language models requires grounding generation in authentic personal data. Existing evaluation approaches rely on demographic surveys, personality questionnaires, or short AI-led interviews as proxies, but lack direct assessment against what individuals actually said. We address this gap with an interview-grounded evaluation framework for personality simulation at a large scale. We extract over 671,000 question-answer pairs from 23,000 verified interview transcripts across 1,000 public personalities, each with an average of 11.5 hours of interview content. We propose a multi-dimensional evaluation framework with four complementary metrics measuring content similarity, factual consistency, personality alignment, and factual knowledge retention. Through systematic comparison, we demonstrate that methods grounded in real interview data substantially outperform those relying solely on biographical profiles or the model's parametric knowledge. We further reveal a trade-off in how interview data is best utilized: retrieval-augmented methods excel at capturing personality style and response quality, while chronological-based methods better preserve factual consistency and knowledge retention. Our evaluation framework enables principled method selection based on application requirements, and our empirical findings provide actionable insights for advancing personality simulation research. 

---
# Quantifying the Expectation-Realisation Gap for Agentic AI Systems 

**Authors**: Sebastian Lobentanzer  

**Link**: [PDF](https://arxiv.org/pdf/2602.20292)  

**Abstract**: Agentic AI systems are deployed with expectations of substantial productivity gains, yet rigorous empirical evidence reveals systematic discrepancies between pre-deployment expectations and post-deployment outcomes. We review controlled trials and independent validations across software engineering, clinical documentation, and clinical decision support to quantify this expectation-realisation gap. In software development, experienced developers expected a 24% speedup from AI tools but were slowed by 19% -- a 43 percentage-point calibration error. In clinical documentation, vendor claims of multi-minute time savings contrast with measured reductions of less than one minute per note, and one widely deployed tool showed no statistically significant effect. In clinical decision support, externally validated performance falls substantially below developer-reported metrics. These shortfalls are driven by workflow integration friction, verification burden, measurement construct mismatches, and systematic heterogeneity in treatment effects. The evidence motivates structured planning frameworks that require explicit, quantified benefit expectations with human oversight costs factored in. 

---
# Uncertainty-Aware Delivery Delay Duration Prediction via Multi-Task Deep Learning 

**Authors**: Stefan Faulkner, Reza Zandehshahvar, Vahid Eghbal Akhlaghi, Sebastien Ouellet, Carsten Jordan, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2602.20271)  

**Abstract**: Accurate delivery delay prediction is critical for maintaining operational efficiency and customer satisfaction across modern supply chains. Yet the increasing complexity of logistics networks, spanning multimodal transportation, cross-country routing, and pronounced regional variability, makes this prediction task inherently challenging. This paper introduces a multi-task deep learning model for delivery delay duration prediction in the presence of significant imbalanced data, where delayed shipments are rare but operationally consequential. The model embeds high-dimensional shipment features with dedicated embedding layers for tabular data, and then uses a classification-then-regression strategy to predict the delivery delay duration for on-time and delayed shipments. Unlike sequential pipelines, this approach enables end-to-end training, improves the detection of delayed cases, and supports probabilistic forecasting for uncertainty-aware decision making. The proposed approach is evaluated on a large-scale real-world dataset from an industrial partner, comprising more than 10 million historical shipment records across four major source locations with distinct regional characteristics. The proposed model is compared with traditional machine learning methods. Experimental results show that the proposed method achieves a mean absolute error of 0.67-0.91 days for delayed-shipment predictions, outperforming single-step tree-based regression baselines by 41-64% and two-step classify-then-regress tree-based models by 15-35%. These gains demonstrate the effectiveness of the proposed model in operational delivery delay forecasting under highly imbalanced and heterogeneous conditions. 

---
# Exploring Anti-Aging Literature via ConvexTopics and Large Language Models 

**Authors**: Lana E. Yeganova, Won G. Kim, Shubo Tian, Natalie Xie, Donald C. Comeau, W. John Wilbur, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20224)  

**Abstract**: The rapid expansion of biomedical publications creates challenges for organizing knowledge and detecting emerging trends, underscoring the need for scalable and interpretable methods. Common clustering and topic modeling approaches such as K-means or LDA remain sensitive to initialization and prone to local optima, limiting reproducibility and evaluation. We propose a reformulation of a convex optimization based clustering algorithm that produces stable, fine-grained topics by selecting exemplars from the data and guaranteeing a global optimum. Applied to about 12,000 PubMed articles on aging and longevity, our method uncovers topics validated by medical experts. It yields interpretable topics spanning from molecular mechanisms to dietary supplements, physical activity, and gut microbiota. The method performs favorably, and most importantly, its reproducibility and interpretability distinguish it from common clustering approaches, including K-means, LDA, and BERTopic. This work provides a basis for developing scalable, web-accessible tools for knowledge discovery. 

---
# MultiModalPFN: Extending Prior-Data Fitted Networks for Multimodal Tabular Learning 

**Authors**: Wall Kim, Chaeyoung Song, Hanul Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.20223)  

**Abstract**: Recently, TabPFN has gained attention as a foundation model for tabular data. However, it struggles to integrate heterogeneous modalities such as images and text, which are common in domains like healthcare and marketing, thereby limiting its applicability. To address this, we present the Multi-Modal Prior-data Fitted Network (MMPFN), which extends TabPFN to handle tabular and non-tabular modalities in a unified manner. MMPFN comprises per-modality encoders, modality projectors, and pre-trained foundation models. The modality projectors serve as the critical bridge, transforming non-tabular embeddings into tabular-compatible tokens for unified processing. To this end, we introduce a multi-head gated MLP and a cross-attention pooler that extract richer context from non-tabular inputs while mitigates attention imbalance issue in multimodal learning. Extensive experiments on medical and general-purpose multimodal datasets demonstrate that MMPFN consistently outperforms competitive state-of-the-art methods and effectively exploits non-tabular modalities alongside tabular features. These results highlight the promise of extending prior-data fitted networks to the multimodal setting, offering a scalable and effective framework for heterogeneous data learning. The source code is available at this https URL. 

---
# What Matters for Simulation to Online Reinforcement Learning on Real Robots 

**Authors**: Yarden As, Dhruva Tirumala, René Zurbrügg, Chenhao Li, Stelian Coros, Andreas Krause, Markus Wulfmeier  

**Link**: [PDF](https://arxiv.org/pdf/2602.20220)  

**Abstract**: We investigate what specific design choices enable successful online reinforcement learning (RL) on physical robots. Across 100 real-world training runs on three distinct robotic platforms, we systematically ablate algorithmic, systems, and experimental decisions that are typically left implicit in prior work. We find that some widely used defaults can be harmful, while a set of robust, readily adopted design choices within standard RL practice yield stable learning across tasks and hardware. These results provide the first large-sample empirical study of such design choices, enabling practitioners to deploy online RL with lower engineering effort. 

---
# An Approach to Combining Video and Speech with Large Language Models in Human-Robot Interaction 

**Authors**: Guanting Shen, Zi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.20219)  

**Abstract**: Interpreting human intent accurately is a central challenge in human-robot interaction (HRI) and a key requirement for achieving more natural and intuitive collaboration between humans and machines. This work presents a novel multimodal HRI framework that combines advanced vision-language models, speech processing, and fuzzy logic to enable precise and adaptive control of a Dobot Magician robotic arm. The proposed system integrates Florence-2 for object detection, Llama 3.1 for natural language understanding, and Whisper for speech recognition, providing users with a seamless and intuitive interface for object manipulation through spoken commands. By jointly addressing scene perception and action planning, the approach enhances the reliability of command interpretation and execution. Experimental evaluations conducted on consumer-grade hardware demonstrate a command execution accuracy of 75\%, highlighting both the robustness and adaptability of the system. Beyond its current performance, the proposed architecture serves as a flexible and extensible foundation for future HRI research, offering a practical pathway toward more sophisticated and natural human-robot collaboration through tightly coupled speech and vision-language processing. 

---
# KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem 

**Authors**: Seongjin Cha, Gyuwan Kim, Dongsu Han, Tao Yang, Insu Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.20217)  

**Abstract**: Self-speculative decoding (SSD) accelerates LLM inference by skipping layers to create an efficient draft model, yet existing methods often rely on static heuristics that ignore the dynamic computational overhead of attention in long-context scenarios. We propose KnapSpec, a training-free framework that reformulates draft model selection as a knapsack problem to maximize tokens-per-time throughput. By decoupling Attention and MLP layers and modeling their hardware-specific latencies as functions of context length, KnapSpec adaptively identifies optimal draft configurations on the fly via a parallel dynamic programming algorithm. Furthermore, we provide the first rigorous theoretical analysis establishing cosine similarity between hidden states as a mathematically sound proxy for the token acceptance rate. This foundation allows our method to maintain high drafting faithfulness while navigating the shifting bottlenecks of real-world hardware. Our experiments on Qwen3 and Llama3 demonstrate that KnapSpec consistently outperforms state-of-the-art SSD baselines, achieving up to 1.47x wall-clock speedup across various benchmarks. Our plug-and-play approach ensures high-speed inference for long sequences without requiring additional training or compromising the target model's output distribution. 

---
# Right to History: A Sovereignty Kernel for Verifiable AI Agent Execution 

**Authors**: Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20214)  

**Abstract**: AI agents increasingly act on behalf of humans, yet no existing system provides a tamper-evident, independently verifiable record of what they did. As regulations such as the EU AI Act begin mandating automatic logging for high-risk AI systems, this gap carries concrete consequences -- especially for agents running on personal hardware, where no centralized provider controls the log. Extending Floridi's informational rights framework from data about individuals to actions performed on their behalf, this paper proposes the Right to History: the principle that individuals are entitled to a complete, verifiable record of every AI agent action on their own hardware. The paper formalizes this principle through five system invariants with structured proof sketches, and implements it in PunkGo, a Rust sovereignty kernel that unifies RFC 6962 Merkle tree audit logs, capability-based isolation, energy-budget governance, and a human-approval mechanism. Adversarial testing confirms all five invariants hold. Performance evaluation shows sub-1.3 ms median action latency, ~400 actions/sec throughput, and 448-byte Merkle inclusion proofs at 10,000 log entries. 

---
# CodeHacker: Automated Test Case Generation for Detecting Vulnerabilities in Competitive Programming Solutions 

**Authors**: Jingwei Shi, Xinxiang Yin, Jing Huang, Jinman Zhao, Shengyu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20213)  

**Abstract**: The evaluation of Large Language Models (LLMs) for code generation relies heavily on the quality and robustness of test cases. However, existing benchmarks often lack coverage for subtle corner cases, allowing incorrect solutions to pass. To bridge this gap, we propose CodeHacker, an automated agent framework dedicated to generating targeted adversarial test cases that expose latent vulnerabilities in program submissions. Mimicking the hack mechanism in competitive programming, CodeHacker employs a multi-strategy approach, including stress testing, anti-hash attacks, and logic-specific targeting to break specific code submissions. To ensure the validity and reliability of these attacks, we introduce a Calibration Phase, where the agent iteratively refines its own Validator and Checker via self-generated adversarial probes before evaluating contestant this http URL demonstrate that CodeHacker significantly improves the True Negative Rate (TNR) of existing datasets, effectively filtering out incorrect solutions that were previously accepted. Furthermore, generated adversarial cases prove to be superior training data, boosting the performance of RL-trained models on benchmarks like LiveCodeBench. 

---
# Multimodal Crystal Flow: Any-to-Any Modality Generation for Unified Crystal Modeling 

**Authors**: Kiyoung Seong, Sungsoo Ahn, Sehui Han, Changyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.20210)  

**Abstract**: Crystal modeling spans a family of conditional and unconditional generation tasks across different modalities, including crystal structure prediction (CSP) and \emph{de novo} generation (DNG). While recent deep generative models have shown promising performance, they remain largely task-specific, lacking a unified framework that shares crystal representations across different generation tasks. To address this limitation, we propose \emph{Multimodal Crystal Flow (MCFlow)}, a unified multimodal flow model that realizes multiple crystal generation tasks as distinct inference trajectories via independent time variables for atom types and crystal structures. To enable multimodal flow in a standard transformer model, we introduce a composition- and symmetry-aware atom ordering with hierarchical permutation augmentation, injecting strong compositional and crystallographic priors without explicit structural templates. Experiments on the MP-20 and MPTS-52 benchmarks show that MCFlow achieves competitive performance against task-specific baselines across multiple crystal generation tasks. 

---
# Model Merging in the Essential Subspace 

**Authors**: Longhua Li, Lei Qi, Qi Tian, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2602.20208)  

**Abstract**: Model merging aims to integrate multiple task-specific fine-tuned models derived from a shared pre-trained checkpoint into a single multi-task model without additional training. Despite extensive research, task interference remains a major obstacle that often undermines the performance of merged models. In this paper, we propose ESM (Essential Subspace Merging) , a robust framework for effective model merging. We begin by performing Principal Component Analysis (PCA) on feature shifts induced by parameter updates. The resulting principal directions span an essential subspace that dominantly influences feature representations. Each task's parameter update matrix is projected onto its respective essential subspace for low-rank decomposition before merging. This methodology mitigates inter-task interference while preserving core task-specific functionality. Furthermore, we introduce a multi-level polarized scaling strategy that amplifies parameters containing critical knowledge and suppresses redundant ones, preventing essential knowledge from being overwhelmed during fusion. Extensive experiments across multiple task sets and model scales demonstrate that our method achieves state-of-the-art performance in multi-task model merging. 

---
# Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis 

**Authors**: Shrestha Datta, Hongfu Liu, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2602.20207)  

**Abstract**: Knowledge editing in Large Language Models (LLMs) aims to update the model's prediction for a specific query to a desired target while preserving its behavior on all other inputs. This process typically involves two stages: identifying the layer to edit and performing the parameter update. Intuitively, different queries may localize knowledge at different depths of the model, resulting in different sample-wise editing performance for a fixed editing layer. In this work, we hypothesize the existence of fixed golden layers that can achieve near-optimal editing performance similar to sample-wise optimal layers. To validate this hypothesis, we provide empirical evidence by comparing golden layers against ground-truth sample-wise optimal layers. Furthermore, we show that golden layers can be reliably identified using a proxy dataset and generalize effectively to unseen test set queries across datasets. Finally, we propose a novel method, namely Layer Gradient Analysis (LGA) that estimates golden layers efficiently via gradient-attribution, avoiding extensive trial-and-error across multiple editing runs. Extensive experiments on several benchmark datasets demonstrate the effectiveness and robustness of our LGA approach across different LLM types and various knowledge editing methods. 

---
# Mitigating "Epistemic Debt" in Generative AI-Scaffolded Novice Programming using Metacognitive Scripts 

**Authors**: Sreecharan Sankaranarayanan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20206)  

**Abstract**: The democratization of Large Language Models (LLMs) has given rise to ``Vibe Coding," a workflow where novice programmers prioritize semantic intent over syntactic implementation. While this lowers barriers to entry, we hypothesize that without pedagogical guardrails, it is fundamentally misaligned with cognitive skill acquisition. Drawing on the distinction between Cognitive Offloading and Cognitive Outsourcing, we argue that unrestricted AI encourages novices to outsource the Intrinsic Cognitive Load required for schema formation, rather than merely offloading Extraneous Load. This accumulation of ``Epistemic Debt" creates ``Fragile Experts" whose high functional utility masks critically low corrective competence.
To quantify and mitigate this debt, we conducted a between-subjects experiment (N=78) using a custom Cursor IDE plugin backed by Claude 3.5 Sonnet. Participants represented "AI-Native" learners across three conditions: Manual (Control), Unrestricted AI (Outsourcing), and Scaffolded AI (Offloading). The Scaffolded condition utilized a novel ``Explanation Gate," leveraging a real-time LLM-as-a-Judge framework to enforce a ``Teach-Back" protocol before generated code could be integrated.
Results reveal a ``Collapse of Competence": while Unrestricted AI users matched the productivity of the Scaffolded group (p < .001 vs. Manual), they suffered a 77% failure rate in a subsequent AI-Blackout maintenance task, compared to only 39% in the Scaffolded group. Qualitative analysis suggests that successful vibe coders naturally engage in self-scaffolding, treating the AI as a consultant rather than a contractor. We discuss the implications for the maintainability of AI-generated software and propose that future learning systems must enforce Metacognitive Friction to prevent the mass production of unmaintainable code. 

---
# Analyzing Latency Hiding and Parallelism in an MLIR-based AI Kernel Compiler 

**Authors**: Javed Absar, Samarth Narang, Muthu Baskaran  

**Link**: [PDF](https://arxiv.org/pdf/2602.20204)  

**Abstract**: AI kernel compilation for edge devices depends on the compiler's ability to exploit parallelism and hide memory latency in the presence of hierarchical memory and explicit data movement. This paper reports a benchmark methodology and corresponding results for three compiler-controlled mechanisms in an MLIR-based compilation pipeline: vectorization (Vec), multi-threading (MT) across hardware contexts, and double buffering (DB) using ping--pong scratchpad buffers to overlap DMA transfers with compute. Using Triton/Inductor-generated kernels, we present an ablation ladder that separates the contribution of Vec, MT, and DB, and we quantify how MT speedup scales with problem size using GELU as a representative activation kernel. The results show that vectorization provides the primary gain for bandwidth-sensitive kernels, MT delivers substantial improvements once scheduling overhead is amortized, and DB provides additional benefit when transfers and compute can be overlapped (i.e., outside the extremes of purely memory-bound or purely compute-bound behavior). 

---
# Evaluating the Reliability of Digital Forensic Evidence Discovered by Large Language Model: A Case Study 

**Authors**: Jeel Piyushkumar Khatiwala, Daniel Kwaku Ntiamoah Addai, Weifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20202)  

**Abstract**: The growing reliance on AI-identified digital evidence raises significant concerns about its reliability, particularly as large language models (LLMs) are increasingly integrated into forensic investigations. This paper proposes a structured framework that automates forensic artifact extraction, refines data through LLM-driven analysis, and validates results using a Digital Forensic Knowledge Graph (DFKG). Evaluated on a 13 GB forensic image dataset containing 61 applications, 2,864 databases, and 5,870 tables, the framework ensures artifact traceability and evidentiary consistency through deterministic Unique Identifiers (UIDs) and forensic cross-referencing. We propose this methodology to address challenges in ensuring the credibility and forensic integrity of AI-identified evidence, reducing classification errors, and advancing scalable, auditable methodologies. A comprehensive case study on this dataset demonstrates the framework's effectiveness, achieving over 95 percent accuracy in artifact extraction, strong support of chain-of-custody adherence, and robust contextual consistency in forensic relationships. Key results validate the framework's ability to enhance reliability, reduce errors, and establish a legally sound paradigm for AI-assisted digital forensics. 

---
# Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation 

**Authors**: Zaijing Li, Bing Hu, Rui Shao, Gongwei Chen, Dongmei Jiang, Pengwei Xie, Jianye Hao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2602.20200)  

**Abstract**: Hierarchical Vision-Language-Action (VLA) models have rapidly become a dominant paradigm for robotic manipulation. It typically comprising a Vision-Language backbone for perception and understanding, together with a generative policy for action generation. However, its performance is increasingly bottlenecked by the action generation proceess. (i) Low inference efficiency. A pronounced distributional gap between isotropic noise priors and target action distributions, which increases denoising steps and the incidence of infeasible samples. (ii) Poor robustness. Existing policies condition solely on the current observation, neglecting the constraint of history sequence and thus lacking awareness of task progress and temporal consistency. To address these issues, we introduce OptimusVLA, a dual-memory VLA framework with Global Prior Memory (GPM) and Local Consistency Memory (LCM). GPM replaces Gaussian noise with task-level priors retrieved from semantically similar trajectories, thereby shortening the generative path and reducing the umber of function evaluations (NFE). LCM dynamically models executed action sequence to infer task progress and injects a learned consistency constraint that enforces temporal coherence and smoothness of trajectory. Across three simulation benchmarks, OptimusVLA consistently outperforms strong baselines: it achieves 98.6% average success rate on LIBERO, improves over pi_0 by 13.5% on CALVIN, and attains 38% average success rate on RoboTwin 2.0 Hard. In Real-World evaluation, OptimusVLA ranks best on Generalization and Long-horizon suites, surpassing pi_0 by 42.9% and 52.4%, respectively, while delivering 2.9x inference speedup. 

---
# IMOVNO+: A Regional Partitioning and Meta-Heuristic Ensemble Framework for Imbalanced Multi-Class Learning 

**Authors**: Soufiane Bacha, Laouni Djafri, Sahraoui Dhelim, Huansheng Ning  

**Link**: [PDF](https://arxiv.org/pdf/2602.20199)  

**Abstract**: Class imbalance, overlap, and noise degrade data quality, reduce model reliability, and limit generalization. Although widely studied in binary classification, these issues remain underexplored in multi-class settings, where complex inter-class relationships make minority-majority structures unclear and traditional clustering fails to capture distribution shape. Approaches that rely only on geometric distances risk removing informative samples and generating low-quality synthetic data, while binarization approaches treat imbalance locally and ignore global inter-class dependencies. At the algorithmic level, ensembles struggle to integrate weak classifiers, leading to limited robustness. This paper proposes IMOVNO+ (IMbalance-OVerlap-NOise+ Algorithm-Level Optimization), a two-level framework designed to jointly enhance data quality and algorithmic robustness for binary and multi-class tasks. At the data level, first, conditional probability is used to quantify the informativeness of each sample. Second, the dataset is partitioned into core, overlapping, and noisy regions. Third, an overlapping-cleaning algorithm is introduced that combines Z-score metrics with a big-jump gap distance. Fourth, a smart oversampling algorithm based on multi-regularization controls synthetic sample proximity, preventing new overlaps. At the algorithmic level, a meta-heuristic prunes ensemble classifiers to reduce weak-learner influence. IMOVNO+ was evaluated on 35 datasets (13 multi-class, 22 binary). Results show consistent superiority over state-of-the-art methods, approaching 100% in several cases. For multi-class data, IMOVNO+ achieves gains of 37-57% in G-mean, 25-44% in F1-score, 25-39% in precision, and 26-43% in recall. In binary tasks, it attains near-perfect performance with improvements of 14-39%. The framework handles data scarcity and imbalance from collection and privacy limits. 

---
# Controllable Exploration in Hybrid-Policy RLVR for Multi-Modal Reasoning 

**Authors**: Zhuoxu Huang, Mengxi Jia, Hao Sun, Xuelong Li, Jungong Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.20197)  

**Abstract**: Reinforcement Learning with verifiable rewards (RLVR) has emerged as a primary learning paradigm for enhancing the reasoning capabilities of multi-modal large language models (MLLMs). However, during RL training, the enormous state space of MLLM and sparse rewards often leads to entropy collapse, policy degradation, or over-exploitation of suboptimal behaviors. This necessitates an exploration strategy that maintains productive stochasticity while avoiding the drawbacks of uncontrolled random sampling, yielding inefficient exploration. In this paper, we propose CalibRL, a hybrid-policy RLVR framework that supports controllable exploration with expert guidance, enabled by two key mechanisms. First, a distribution-aware advantage weighting scales updates by group rareness to calibrate the distribution, therefore preserving exploration. Meanwhile, the asymmetric activation function (LeakyReLU) leverages the expert knowledge as a calibration baseline to moderate overconfident updates while preserving their corrective direction. CalibRL increases policy entropy in a guided manner and clarifies the target distribution by estimating the on-policy distribution through online sampling. Updates are driven by these informative behaviors, avoiding convergence to erroneous patterns. Importantly, these designs help alleviate the distributional mismatch between the model's policy and expert trajectories, thereby achieving a more stable balance between exploration and exploitation. Extensive experiments across eight benchmarks, including both in-domain and out-of-domain settings, demonstrate consistent improvements, validating the effectiveness of our controllable hybrid-policy RLVR training. Code is available at this https URL. 

---
# OpenPort Protocol: A Security Governance Specification for AI Agent Tool Access 

**Authors**: Genliang Zhu, Chu Wang, Ziyuan Wang, Zhida Li, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20196)  

**Abstract**: AI agents increasingly require direct, structured access to application data and actions, but production deployments still struggle to express and verify the governance properties that matter in practice: least-privilege authorization, controlled write execution, predictable failure handling, abuse resistance, and auditability. This paper introduces OpenPort Protocol (OPP), a governance-first specification for exposing application tools through a secure server-side gateway that is model- and runtime-neutral and can bind to existing tool ecosystems. OpenPort defines authorization-dependent discovery, stable response envelopes with machine-actionable \texttt{agent.*} reason codes, and an authorization model combining integration credentials, scoped permissions, and ABAC-style policy constraints. For write operations, OpenPort specifies a risk-gated lifecycle that defaults to draft creation and human review, supports time-bounded auto-execution under explicit policy, and enforces high-risk safeguards including preflight impact binding and idempotency. To address time-of-check/time-of-use drift in delayed approval flows, OpenPort also specifies an optional State Witness profile that revalidates execution-time preconditions and fails closed on state mismatch. Operationally, the protocol requires admission control (rate limits/quotas) with stable 429 semantics and structured audit events across allow/deny/fail paths so that client recovery and incident analysis are deterministic. We present a reference runtime and an executable governance toolchain (layered conformance profiles, negative security tests, fuzz/abuse regression, and release-gate scans) and evaluate the core profile at a pinned release tag using artifact-based, externally reproducible validation. 

---
# When Backdoors Go Beyond Triggers: Semantic Drift in Diffusion Models Under Encoder Attacks 

**Authors**: Shenyang Chen, Liuwan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20193)  

**Abstract**: Standard evaluations of backdoor attacks on text-to-image (T2I) models primarily measure trigger activation and visual fidelity. We challenge this paradigm, demonstrating that encoder-side poisoning induces persistent, trigger-free semantic corruption that fundamentally reshapes the representation manifold. We trace this vulnerability to a geometric mechanism: a Jacobian-based analysis reveals that backdoors act as low-rank, target-centered deformations that amplify local sensitivity, causing distortion to propagate coherently across semantic neighborhoods. To rigorously quantify this structural degradation, we introduce SEMAD (Semantic Alignment and Drift), a diagnostic framework that measures both internal embedding drift and downstream functional misalignment. Our findings, validated across diffusion and contrastive paradigms, expose the deep structural risks of encoder poisoning and highlight the necessity of geometric audits beyond simple attack success rates. 

---
# MoBiQuant: Mixture-of-Bits Quantization for Token-Adaptive Elastic LLMs 

**Authors**: Dongwei Wang, Jinhee Kim, Seokho Han, Denis Gudovskiy, Yohei Nakata, Tomoyuki Okuno, KhayTze Peong, Kang Eun Jeon, Jong Hwan Ko, Yiran Chen, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.20191)  

**Abstract**: Changing runtime complexity on cloud and edge devices necessitates elastic large language model (LLM) deployment, where an LLM can be inferred with various quantization precisions based on available computational resources. However, it has been observed that the calibration parameters for quantization are typically linked to specific precisions, which presents challenges during elastic-precision calibration and precision switching at runtime. In this work, we attribute the source of varying calibration parameters to the varying token-level sensitivity caused by a precision-dependent outlier migration this http URL by this observation, we propose \texttt{MoBiQuant}, a novel Mixture-of-Bits quantization framework that adjusts weight precision for elastic LLM inference based on token sensitivity. Specifically, we propose the many-in-one recursive residual quantization that can iteratively reconstruct higher-precision weights and the token-aware router to dynamically select the number of residual bit slices. MoBiQuant enables smooth precision switching while improving generalization for the distribution of token outliers. Experimental results demonstrate that MoBiQuant exhibits strong elasticity, enabling it to match the performance of bit-specific calibrated PTQ on LLaMA3-8B without repeated calibration. 

---
# AINet: Anchor Instances Learning for Regional Heterogeneity in Whole Slide Image 

**Authors**: Tingting Zheng, Hongxun Yao, Kui Jiang, Sicheng Zhao, Yi Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20187)  

**Abstract**: Recent advances in multi-instance learning (MIL) have witnessed impressive performance in whole slide image (WSI) analysis. However, the inherent sparsity of tumors and their morphological diversity lead to obvious heterogeneity across regions, posing significant challenges in aggregating high-quality and discriminative representations. To address this, we introduce a novel concept of anchor instance (AI), a compact subset of instances that are representative within their regions (local) and discriminative at the bag (global) level. These AIs act as semantic references to guide interactions across regions, correcting non-discriminative patterns while preserving regional diversity. Specifically, we propose a dual-level anchor mining (DAM) module to \textbf{select} AIs from massive instances, where the most informative AI in each region is extracted by assessing its similarity to both local and global embeddings. Furthermore, to ensure completeness and diversity, we devise an anchor-guided region correction (ARC) module that explores the complementary information from all regions to \textbf{correct} each regional representation. Building upon DAM and ARC, we develop a concise yet effective framework, AINet, which employs a simple predictor and surpasses state-of-the-art methods with substantially fewer FLOPs and parameters. Moreover, both DAM and ARC are modular and can be seamlessly integrated into existing MIL frameworks, consistently improving their performance. 

---
# Closing the Expertise Gap in Residential Building Energy Retrofits: A Domain-Specific LLM for Informed Decision-Making 

**Authors**: Lei Shu, Armin Yeganeh, Sinem Mollaoglu, Jiayu Zhou, Dong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.20181)  

**Abstract**: Residential energy retrofit decision-making is constrained by an expertise gap, as homeowners lack the technical literacy required for energy assessments. To address this challenge, this study develops a domain-specific large language model (LLM) that provides optimal retrofit recommendations using homeowner-accessible descriptions of basic dwelling characteristics. The model is fine-tuned on physics-based energy simulations and techno-economic calculations derived from 536,416 U.S. residential building prototypes across nine major retrofit categories. Using Low-Rank Adaptation (LoRA), the LLM maps dwelling characteristics to optimal retrofit selections and associated performance outcomes. Evaluation against physics-grounded baselines shows that the model identifies the optimal retrofit for CO2 reduction within its top three recommendations in 98.9% of cases and the shortest discounted payback period in 93.3% of cases. Fine-tuning yields an order-of-magnitude reduction in CO2 prediction error and multi-fold reductions for energy use and retrofit cost. The model maintains performance under incomplete input conditions, supporting informed residential decarbonization decisions. 

---
# Enhancing Heat Sink Efficiency in MOSFETs using Physics Informed Neural Networks: A Systematic Study on Coolant Velocity Estimation 

**Authors**: Aniruddha Bora, Isabel K. Alvarez, Julie Chalfant, Chryssostomos Chryssostomidis  

**Link**: [PDF](https://arxiv.org/pdf/2602.20177)  

**Abstract**: In this work, we present a methodology using Physics Informed Neural Networks (PINNs) to determine the required velocity of a coolant, given inlet and outlet temperatures for a given heat flux in a multilayered metal-oxide-semiconductor field-effect transistor (MOSFET). MOSFETs are integral components of Power Electronic Building Blocks (PEBBs) and experiences the majority of the thermal load. Effective cooling of MOSFETs is therefore essential to prevent overheating and potential burnout. Determining the required velocity for the purpose of effective cooling is of importance but is an ill-posed inverse problem and difficult to solve using traditional methods. MOSFET consists of multiple layers with different thermal conductivities, including aluminum, pyrolytic graphite sheets (PGS), and stainless steel pipes containing flowing water. We propose an algorithm that employs sequential training of the MOSFET layers in PINNs. Mathematically, the sequential training method decouples the optimization of each layer by treating the parameters of other layers as constants during its training phase. This reduces the dimensionality of the optimization landscape, making it easier to find the global minimum for each layer's parameters and avoid poor local minima. Convergence of the PINNs solution to the analytical solution is theoretically analyzed. Finally we show the prediction of our proposed methodology to be in good agreement with experimental results. 

---
# CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation 

**Authors**: Chaeyun Kim, YongTaek Lim, Kihyun Kim, Junghwan Kim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.20170)  

**Abstract**: Existing red-teaming benchmarks, when adapted to new languages via direct translation, fail to capture socio-technical vulnerabilities rooted in local culture and law, creating a critical blind spot in LLM safety evaluation. To address this gap, we introduce CAGE (Culturally Adaptive Generation), a framework that systematically adapts the adversarial intent of proven red-teaming prompts to new cultural contexts. At the core of CAGE is the Semantic Mold, a novel approach that disentangles a prompt's adversarial structure from its cultural content. This approach enables the modeling of realistic, localized threats rather than testing for simple jailbreaks. As a representative example, we demonstrate our framework by creating KoRSET, a Korean benchmark, which proves more effective at revealing vulnerabilities than direct translation baselines. CAGE offers a scalable solution for developing meaningful, context-aware safety benchmarks across diverse cultures. Our dataset and evaluation rubrics are publicly available at this https URL. (WARNING: This paper contains model outputs that can be offensive in nature.) 

---
# Autonomous AI and Ownership Rules 

**Authors**: Frank Fagan  

**Link**: [PDF](https://arxiv.org/pdf/2602.20169)  

**Abstract**: This Article examines the circumstances in which AI-generated outputs remain linked to their creators and the points at which they lose that connection, whether through accident, deliberate design, or emergent behavior. In cases where AI is traceable to an originator, accession doctrine provides an efficient means of assigning ownership, preserving investment incentives while maintaining accountability. When AI becomes untraceable -- whether through carelessness, deliberate obfuscation, or emergent behavior -- first possession rules can encourage reallocation to new custodians who are incentivized to integrate AI into productive use. The analysis further explores strategic ownership dissolution, where autonomous AI is intentionally designed to evade attribution, creating opportunities for tax arbitrage and regulatory avoidance. To counteract these inefficiencies, bounty systems, private incentives, and government subsidies are proposed as mechanisms to encourage AI capture and prevent ownerless AI from distorting markets. 

---
# Benchmarking Early Deterioration Prediction Across Hospital-Rich and MCI-Like Emergency Triage Under Constrained Sensing 

**Authors**: KMA Solaiman, Joshua Sebastian, Karma Tobden  

**Link**: [PDF](https://arxiv.org/pdf/2602.20168)  

**Abstract**: Emergency triage decisions are made under severe information constraints, yet most data-driven deterioration models are evaluated using signals unavailable during initial assessment. We present a leakage-aware benchmarking framework for early deterioration prediction that evaluates model performance under realistic, time-limited sensing conditions. Using a patient-deduplicated cohort derived from MIMIC-IV-ED, we compare hospital-rich triage with a vitals-only, MCI-like setting, restricting inputs to information available within the first hour of presentation. Across multiple modeling approaches, predictive performance declines only modestly when limited to vitals, indicating that early physiological measurements retain substantial clinical signal. Structured ablation and interpretability analyses identify respiratory and oxygenation measures as the most influential contributors to early risk stratification, with models exhibiting stable, graceful degradation as sensing is reduced. This work provides a clinically grounded benchmark to support the evaluation and design of deployable triage decision-support systems in resource-constrained settings. 

---
# ConceptRM: The Quest to Mitigate Alert Fatigue through Consensus-Based Purity-Driven Data Cleaning for Reflection Modelling 

**Authors**: Yongda Yu, Lei Zhang, Xinxin Guo, Minghui Yu, Zhengqi Zhuang, Guoping Rong, Haifeng Shen, Zhengfeng Li, Boge Wang, Guoan Zhang, Bangyu Xiang, Xiaobin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.20166)  

**Abstract**: In many applications involving intelligent agents, the overwhelming volume of alerts (mostly false) generated by the agents may desensitize users and cause them to overlook critical issues, leading to the so-called ''alert fatigue''. A common strategy is to train a reflection model as a filter to intercept false alerts with labelled data collected from user verification feedback. However, a key challenge is the noisy nature of such data as it is often collected in production environments. As cleaning noise via manual annotation incurs high costs, this paper proposes a novel method ConceptRM for constructing a high-quality corpus to train a reflection model capable of effectively intercepting false alerts. With only a small amount of expert annotations as anchors, ConceptRM creates perturbed datasets with varying noise ratios and utilizes co-teaching to train multiple distinct models for collaborative learning. By analyzing the consensus decisions of these models, it effectively identifies reliable negative samples from a noisy dataset. Experimental results demonstrate that ConceptRM significantly enhances the interception of false alerts with minimal annotation cost, outperforming several state-of-the-art LLM baselines by up to 53.31% on in-domain datasets and 41.67% on out-of-domain datasets. 

---
# Talking to Yourself: Defying Forgetting in Large Language Models 

**Authors**: Yutao Sun, Mingshuai Chen, Tiancheng Zhao, Phillip Miao, Zilun Zhang, Haozhan Shen, Ruizhe Zhu, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2602.20162)  

**Abstract**: Catastrophic forgetting remains a major challenge when fine-tuning large language models (LLMs) on narrow, task-specific data, often degrading their general knowledge and reasoning abilities. We propose SA-SFT, a lightweight self-augmentation routine in which an LLM generates self-dialogues prior to fine-tuning, and the resulting self-authored data are mixed with task data without modifying optimization or training schedules.
Despite requiring no external data or additional tuning, SA-SFT consistently mitigates catastrophic forgetting while improving in-domain performance. Across 50 evaluation scenarios, it maintains performance comparable to the original model and achieves the best results in 40 cases, outperforming common baselines such as layer freezing and external data mixing. Guided by these empirical findings, we further present a theoretical analysis suggesting that forgetting can partly stem from style-induced parameter drift, and that self-alignment through self-generated data provides an effective means to counteract this effect. Overall, our results indicate that self-augmentation offers a simple and effective mechanism for robust LLM adaptation without incurring catastrophic forgetting. 

---
# Multimodal Multi-Agent Empowered Legal Judgment Prediction 

**Authors**: Zhaolu Kang, Junhao Gong, Qingxi Chen, Hao Zhang, Jiaxin Liu, Rong Fu, Zhiyuan Feng, Yuan Wang, Simon Fong, Kaiyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.12815)  

**Abstract**: Legal Judgment Prediction (LJP) aims to predict the outcomes of legal cases based on factual descriptions, serving as a fundamental task to advance the development of legal systems. Traditional methods often rely on statistical analyses or role-based simulations but face challenges with multiple allegations, diverse evidence, and lack adaptability. In this paper, we introduce JurisMMA, a novel framework for LJP that effectively decomposes trial tasks, standardizes processes, and organizes them into distinct stages. Furthermore, we build JurisMM, a large dataset with over 100,000 recent Chinese judicial records, including both text and multimodal video-text data, enabling comprehensive evaluation. Experiments on JurisMM and the benchmark LawBench validate our framework's effectiveness. These results indicate that our framework is effective not only for LJP but also for a broader range of legal applications, offering new perspectives for the development of future legal methods and datasets. 

---
# Interpretable Medical Image Classification using Prototype Learning and Privileged Information 

**Authors**: Luisa Gallee, Meinrad Beer, Michael Goetz  

**Link**: [PDF](https://arxiv.org/pdf/2310.15741)  

**Abstract**: Interpretability is often an essential requirement in medical imaging. Advanced deep learning methods are required to address this need for explainability and high performance. In this work, we investigate whether additional information available during the training process can be used to create an understandable and powerful model. We propose an innovative solution called Proto-Caps that leverages the benefits of capsule networks, prototype learning and the use of privileged information. Evaluating the proposed solution on the LIDC-IDRI dataset shows that it combines increased interpretability with above state-of-the-art prediction performance. Compared to the explainable baseline model, our method achieves more than 6 % higher accuracy in predicting both malignancy (93.0 %) and mean characteristic features of lung nodules. Simultaneously, the model provides case-based reasoning with prototype representations that allow visual validation of radiologist-defined attributes. 

---
# ShaRP: Shape-Regularized Multidimensional Projections 

**Authors**: Alister Machado, Alexandru Telea, Michael Behrisch  

**Link**: [PDF](https://arxiv.org/pdf/2306.00554)  

**Abstract**: Projections, or dimensionality reduction methods, are techniques of choice for the visual exploration of high-dimensional data. Many such techniques exist, each one of them having a distinct visual signature - i.e., a recognizable way to arrange points in the resulting scatterplot. Such signatures are implicit consequences of algorithm design, such as whether the method focuses on local vs global data pattern preservation; optimization techniques; and hyperparameter settings. We present a novel projection technique - ShaRP - that provides users explicit control over the visual signature of the created scatterplot, which can cater better to interactive visualization scenarios. ShaRP scales well with dimensionality and dataset size, generically handles any quantitative dataset, and provides this extended functionality of controlling projection shapes at a small, user-controllable cost in terms of quality metrics. 

---
