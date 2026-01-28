# Routing End User Queries to Enterprise Databases 

**Authors**: Saikrishna Sudarshan, Tanay Kulkarni, Manasi Patwardhan, Lovekesh Vig, Ashwin Srinivasan, Tanmay Tulsidas Verlekar  

**Link**: [PDF](https://arxiv.org/pdf/2601.19825)  

**Abstract**: We address the task of routing natural language queries in multi-database enterprise environments. We construct realistic benchmarks by extending existing NL-to-SQL datasets. Our study shows that routing becomes increasingly challenging with larger, domain-overlapping DB repositories and ambiguous queries, motivating the need for more structured and robust reasoning-based solutions. By explicitly modelling schema coverage, structural connectivity, and fine-grained semantic alignment, the proposed modular, reasoning-driven reranking strategy consistently outperforms embedding-only and direct LLM-prompting baselines across all the metrics. 

---
# Benchmarks Saturate When The Model Gets Smarter Than The Judge 

**Authors**: Marthe Ballon, Andres Algaba, Brecht Verbeken, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2601.19532)  

**Abstract**: Benchmarks are important tools to track progress in the development of Large Language Models (LLMs), yet inaccuracies in datasets and evaluation methods consistently undermine their effectiveness. Here, we present Omni-MATH-2, a manually revised version of the Omni-MATH dataset comprising a clean, exact-answer subset ($n{=}4181$) and a tagged, non-standard subset ($n{=}247$). Each problem was audited to ensure LaTeX compilability, solvability and verifiability, which involved adding missing figures or information, labeling problems requiring a proof, estimation or image, and removing clutter. This process significantly reduces dataset-induced noise, thereby providing a more precise assessment of model performance. The annotated dataset also allows us to evaluate judge-induced noise by comparing GPT-5 mini with the original Omni-Judge, revealing substantial discrepancies between judges on both the clean and tagged problem subsets. Expert annotations reveal that Omni-Judge is wrong in $96.4\%$ of the judge disagreements, indicating its inability to differentiate between models' abilities, even well before saturation of the benchmark occurs. As problems become more challenging, we find that increasingly competent judges become essential in order to prevent judge errors from masking genuine differences between models. Finally, neither judge identifies the present failure modes for the subset of tagged problems, demonstrating that dataset quality and judge reliability are both critical to develop accurate benchmarks of model performance. 

---
# ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks 

**Authors**: Haoyun Li, Ming Xiao, Kezhi Wang, Robert Schober, Dong In Kim, Yong Liang Guan  

**Link**: [PDF](https://arxiv.org/pdf/2601.19607)  

**Abstract**: Emerging 6G networks rely on complex cross-layer optimization, yet manually translating high-level intents into mathematical formulations remains a bottleneck. While Large Language Models (LLMs) offer promise, monolithic approaches often lack sufficient domain grounding, constraint awareness, and verification capabilities. To address this, we present ComAgent, a multi-LLM agentic AI framework. ComAgent employs a closed-loop Perception-Planning-Action-Reflection cycle, coordinating specialized agents for literature search, coding, and scoring to autonomously generate solver-ready formulations and reproducible simulations. By iteratively decomposing problems and self-correcting errors, the framework effectively bridges the gap between user intent and execution. Evaluations demonstrate that ComAgent achieves expert-comparable performance in complex beamforming optimization and outperforms monolithic LLMs across diverse wireless tasks, highlighting its potential for automating design in emerging wireless networks. 

---
# Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search 

**Authors**: Thomas Bömer, Nico Koltermann, Max Disselnmeyer, Bastian Amberg, Anne Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2601.19622)  

**Abstract**: Heuristic functions are essential to the performance of tree search algorithms such as A*, where their accuracy and efficiency directly impact search outcomes. Traditionally, such heuristics are handcrafted, requiring significant expertise. Recent advances in large language models (LLMs) and evolutionary frameworks have opened the door to automating heuristic design. In this paper, we extend the Evolution of Heuristics (EoH) framework to investigate the automated generation of guiding heuristics for A* search. We introduce a novel domain-agnostic prompt augmentation strategy that includes the A* code into the prompt to leverage in-context learning, named Algorithmic - Contextual EoH (A-CEoH). To evaluate the effectiveness of A-CeoH, we study two problem domains: the Unit-Load Pre-Marshalling Problem (UPMP), a niche problem from warehouse logistics, and the classical sliding puzzle problem (SPP). Our computational experiments show that A-CEoH can significantly improve the quality of the generated heuristics and even outperform expert-designed heuristics. 

---
# GLOVE: Global Verifier for LLM Memory-Environment Realignment 

**Authors**: Xingkun Yin, Hongyang Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.19249)  

**Abstract**: Most existing memory-enhanced Large Language Model (LLM) approaches implicitly assume that memory validity can be established either through external evaluators that provide task-specific success signals or through internal model cognition, such as reflection, for editing memory entries. However, these assumptions often break down in practical environments with dynamic drifts. We propose the Global Verifier (GLOVE), a framework that introduces a new design dimension for LLM memory systems by establishing a relative notion of truth. Through active probing to detect inconsistencies between retrieved memories and fresh observations, GLOVE enables memory-environment realignment by verifying and updating memory without access to ground-truth supervision or strong reliance on model introspection. We evaluate GLOVE on diverse benchmarks spanning web navigation, planning, and control, augmented with controlled environmental drifts that introduce non-stationarity beyond the original benchmark settings. Our results show that GLOVE substantially improves agent success rates, suggesting a robust pathway to cognitive agents capable of self-evolving. 

---
# Beyond In-Domain Detection: SpikeScore for Cross-Domain Hallucination Detection 

**Authors**: Yongxin Deng, Zhen Fang, Yixuan Li, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.19245)  

**Abstract**: Hallucination detection is critical for deploying large language models (LLMs) in real-world applications. Existing hallucination detection methods achieve strong performance when the training and test data come from the same domain, but they suffer from poor cross-domain generalization. In this paper, we study an important yet overlooked problem, termed generalizable hallucination detection (GHD), which aims to train hallucination detectors on data from a single domain while ensuring robust performance across diverse related domains. In studying GHD, we simulate multi-turn dialogues following LLMs initial response and observe an interesting phenomenon: hallucination-initiated multi-turn dialogues universally exhibit larger uncertainty fluctuations than factual ones across different domains. Based on the phenomenon, we propose a new score SpikeScore, which quantifies abrupt fluctuations in multi-turn dialogues. Through both theoretical analysis and empirical validation, we demonstrate that SpikeScore achieves strong cross-domain separability between hallucinated and non-hallucinated responses. Experiments across multiple LLMs and benchmarks demonstrate that the SpikeScore-based detection method outperforms representative baselines in cross-domain generalization and surpasses advanced generalization-oriented methods, verifying the effectiveness of our method in cross-domain hallucination detection. 

---
# Multi-Agent Procedural Graph Extraction with Structural and Logical Refinement 

**Authors**: Wangyang Ying, Yanchi Liu, Xujiang Zhao, Wei Cheng, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.19170)  

**Abstract**: Automatically extracting workflows as procedural graphs from natural language is promising yet underexplored, demanding both structural validity and logical alignment. While recent large language models (LLMs) show potential for procedural graph extraction, they often produce ill-formed structures or misinterpret logical flows. We present \model{}, a multi-agent framework that formulates procedural graph extraction as a multi-round reasoning process with dedicated structural and logical refinement. The framework iterates through three stages: (1) a graph extraction phase with the graph builder agent, (2) a structural feedback phase in which a simulation agent diagnoses and explains structural defects, and (3) a logical feedback phase in which a semantic agent aligns semantics between flow logic and linguistic cues in the source text. Important feedback is prioritized and expressed in natural language, which is injected into subsequent prompts, enabling interpretable and controllable refinement. This modular design allows agents to target distinct error types without supervision or parameter updates. Experiments demonstrate that \model{} achieves substantial improvements in both structural correctness and logical consistency over strong baselines. 

---
# More at Stake: How Payoff and Language Shape LLM Agent Strategies in Cooperation Dilemmas 

**Authors**: Trung-Kiet Huynh, Dao-Sy Duy-Minh, Thanh-Bang Cao, Phong-Hao Le, Hong-Dan Nguyen, Nguyen Lam Phu Quy, Minh-Luan Nguyen-Vo, Hong-Phat Pham, Pham Phu Hoa, Thien-Kim Than, Chi-Nguyen Tran, Huy Tran, Gia-Thoai Tran-Le, Alessio Buscemi, Le Hong Trang, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.19082)  

**Abstract**: As LLMs increasingly act as autonomous agents in interactive and multi-agent settings, understanding their strategic behavior is critical for safety, coordination, and AI-driven social and economic systems. We investigate how payoff magnitude and linguistic context shape LLM strategies in repeated social dilemmas, using a payoff-scaled Prisoner's Dilemma to isolate sensitivity to incentive strength. Across models and languages, we observe consistent behavioral patterns, including incentive-sensitive conditional strategies and cross-linguistic divergence. To interpret these dynamics, we train supervised classifiers on canonical repeated-game strategies and apply them to LLM decisions, revealing systematic, model- and language-dependent behavioral intentions, with linguistic framing sometimes matching or exceeding architectural effects. Our results provide a unified framework for auditing LLMs as strategic agents and highlight cooperation biases with direct implications for AI governance and multi-agent system design. 

---
# TS-Debate: Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning 

**Authors**: Patara Trirat, Jin Myung Kwak, Jay Heo, Heejun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19151)  

**Abstract**: Recent progress at the intersection of large language models (LLMs) and time series (TS) analysis has revealed both promise and fragility. While LLMs can reason over temporal structure given carefully engineered context, they often struggle with numeric fidelity, modality interference, and principled cross-modal integration. We present TS-Debate, a modality-specialized, collaborative multi-agent debate framework for zero-shot time series reasoning. TS-Debate assigns dedicated expert agents to textual context, visual patterns, and numerical signals, preceded by explicit domain knowledge elicitation, and coordinates their interaction via a structured debate protocol. Reviewer agents evaluate agent claims using a verification-conflict-calibration mechanism, supported by lightweight code execution and numerical lookup for programmatic verification. This architecture preserves modality fidelity, exposes conflicting evidence, and mitigates numeric hallucinations without task-specific fine-tuning. Across 20 tasks spanning three public benchmarks, TS-Debate achieves consistent and significant performance improvements over strong baselines, including standard multimodal debate in which all agents observe all inputs. 

---
# RPO:Reinforcement Fine-Tuning with Partial Reasoning Optimization 

**Authors**: Hongzhu Yi, Xinming Wang, Zhenghao zhang, Tianyu Zong, Yuanxiang Wang, Jun Xie, Tao Yu, Haopeng Jin, Zhepeng Wang, Kaixin Xu, Feng Chen, Jiahuan Chen, Yujia Yang, Zhenyu Guan, Bingkang Shi, Jungang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19404)  

**Abstract**: Within the domain of large language models, reinforcement fine-tuning algorithms necessitate the generation of a complete reasoning trajectory beginning from the input query, which incurs significant computational overhead during the rollout phase of training. To address this issue, we analyze the impact of different segments of the reasoning path on the correctness of the final result and, based on these insights, propose Reinforcement Fine-Tuning with Partial Reasoning Optimization (RPO), a plug-and-play reinforcement fine-tuning algorithm. Unlike traditional reinforcement fine-tuning algorithms that generate full reasoning paths, RPO trains the model by generating suffixes of the reasoning path using experience cache. During the rollout phase of training, RPO reduces token generation in this phase by approximately 95%, greatly lowering the theoretical time overhead. Compared with full-path reinforcement fine-tuning algorithms, RPO reduces the training time of the 1.5B model by 90% and the 7B model by 72%. At the same time, it can be integrated with typical algorithms such as GRPO and DAPO, enabling them to achieve training acceleration while maintaining performance comparable to the original algorithms. Our code is open-sourced at this https URL. 

---
# LLM Driven Design of Continuous Optimization Problems with Controllable High-level Properties 

**Authors**: Urban Skvorc, Niki van Stein, Moritz Seiler, Britta Grimme, Thomas Bäck, Heike Trautmann  

**Link**: [PDF](https://arxiv.org/pdf/2601.18846)  

**Abstract**: Benchmarking in continuous black-box optimisation is hindered by the limited structural diversity of existing test suites such as BBOB. We explore whether large language models embedded in an evolutionary loop can be used to design optimisation problems with clearly defined high-level landscape characteristics. Using the LLaMEA framework, we guide an LLM to generate problem code from natural-language descriptions of target properties, including multimodality, separability, basin-size homogeneity, search-space homogeneity and globallocal optima contrast. Inside the loop we score candidates through ELA-based property predictors. We introduce an ELA-space fitness-sharing mechanism that increases population diversity and steers the generator away from redundant landscapes. A complementary basin-of-attraction analysis, statistical testing and visual inspection, verifies that many of the generated functions indeed exhibit the intended structural traits. In addition, a t-SNE embedding shows that they expand the BBOB instance space rather than forming an unrelated cluster. The resulting library provides a broad, interpretable, and reproducible set of benchmark problems for landscape analysis and downstream tasks such as automated algorithm selection. 

---
# HARMONI: Multimodal Personalization of Multi-User Human-Robot Interactions with LLMs 

**Authors**: Jeanne Malécot, Hamed Rahimi, Jeanne Cattoni, Marie Samson, Mouad Abrini, Mahdi Khoramshahi, Maribel Pino, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2601.19839)  

**Abstract**: Existing human-robot interaction systems often lack mechanisms for sustained personalization and dynamic adaptation in multi-user environments, limiting their effectiveness in real-world deployments. We present HARMONI, a multimodal personalization framework that leverages large language models to enable socially assistive robots to manage long-term multi-user interactions. The framework integrates four key modules: (i) a perception module that identifies active speakers and extracts multimodal input; (ii) a world modeling module that maintains representations of the environment and short-term conversational context; (iii) a user modeling module that updates long-term speaker-specific profiles; and (iv) a generation module that produces contextually grounded and ethically informed responses. Through extensive evaluation and ablation studies on four datasets, as well as a real-world scenario-driven user-study in a nursing home environment, we demonstrate that HARMONI supports robust speaker identification, online memory updating, and ethically aligned personalization, outperforming baseline LLM-driven approaches in user modeling accuracy, personalization quality, and user satisfaction. 

---
# Neural Theorem Proving for Verification Conditions: A Real-World Benchmark 

**Authors**: Qiyuan Xu, Xiaokun Luan, Renxi Wang, Joshua Ong Jun Leang, Peixin Wang, Haonan Li, Wenda Li, Conrad Watt  

**Link**: [PDF](https://arxiv.org/pdf/2601.18944)  

**Abstract**: Theorem proving is fundamental to program verification, where the automated proof of Verification Conditions (VCs) remains a primary bottleneck. Real-world program verification frequently encounters hard VCs that existing Automated Theorem Provers (ATPs) cannot prove, leading to a critical need for extensive manual proofs that burden practical application. While Neural Theorem Proving (NTP) has achieved significant success in mathematical competitions, demonstrating the potential of machine learning approaches to formal reasoning, its application to program verification--particularly VC proving--remains largely unexplored. Despite existing work on annotation synthesis and verification-related theorem proving, no benchmark has specifically targeted this fundamental bottleneck: automated VC proving. This work introduces Neural Theorem Proving for Verification Conditions (NTP4VC), presenting the first real-world multi-language benchmark for this task. From real-world projects such as Linux and Contiki-OS kernel, our benchmark leverages industrial pipelines (Why3 and Frama-C) to generate semantically equivalent test cases across formal languages of Isabelle, Lean, and Rocq. We evaluate large language models (LLMs), both general-purpose and those fine-tuned for theorem proving, on NTP4VC. Results indicate that although LLMs show promise in VC proving, significant challenges remain for program verification, highlighting a large gap and opportunity for future research. 

---
# When Iterative RAG Beats Ideal Evidence: A Diagnostic Study in Scientific Multi-hop Question Answering 

**Authors**: Mahdi Astaraki, Mohammad Arshi Saloot, Ali Shiraee Kasmaee, Hamidreza Mahyar, Soheila Samiee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19827)  

**Abstract**: Retrieval-Augmented Generation (RAG) extends large language models (LLMs) beyond parametric knowledge, yet it is unclear when iterative retrieval-reasoning loops meaningfully outperform static RAG, particularly in scientific domains with multi-hop reasoning, sparse domain knowledge, and heterogeneous evidence. We provide the first controlled, mechanism-level diagnostic study of whether synchronized iterative retrieval and reasoning can surpass an idealized static upper bound (Gold Context) RAG. We benchmark eleven state-of-the-art LLMs under three regimes: (i) No Context, measuring reliance on parametric memory; (ii) Gold Context, where all oracle evidence is supplied at once; and (iii) Iterative RAG, a training-free controller that alternates retrieval, hypothesis refinement, and evidence-aware stopping. Using the chemistry-focused ChemKGMultiHopQA dataset, we isolate questions requiring genuine retrieval and analyze behavior with diagnostics spanning retrieval coverage gaps, anchor-carry drop, query quality, composition fidelity, and control calibration. Across models, Iterative RAG consistently outperforms Gold Context, with gains up to 25.6 percentage points, especially for non-reasoning fine-tuned models. Staged retrieval reduces late-hop failures, mitigates context overload, and enables dynamic correction of early hypothesis drift, but remaining failure modes include incomplete hop coverage, distractor latch trajectories, early stopping miscalibration, and high composition failure rates even with perfect retrieval. Overall, staged retrieval is often more influential than the mere presence of ideal evidence; we provide practical guidance for deploying and diagnosing RAG systems in specialized scientific settings and a foundation for more reliable, controllable iterative retrieval-reasoning frameworks. 

---
# Out-of-Distribution Generalization via Invariant Trajectories for Multimodal Large Language Model Editing 

**Authors**: Jiajie Su, Haoyuan Wang, Xiaohua Feng, Yunshan Ma, Xiaobo Xia, Yuyuan Li, Xiaolin Zheng, Jianmao Xiao, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.19700)  

**Abstract**: Knowledge editing emerges as a crucial technique for efficiently correcting incorrect or outdated knowledge in large language models (LLM). Existing editing methods for unimodal LLM rely on a rigid parameter-to-output mapping, which causes causal-underfit and causal-overfit in cascaded reasoning for Multimodal LLM (MLLM). In this paper, we reformulate MLLM editing as an out-of-distribution (OOD) generalization problem, where the goal is to discern semantic shift with factual shift and thus achieve robust editing among diverse cross-modal prompting. The key challenge of this OOD problem lies in identifying invariant causal trajectories that generalize accurately while suppressing spurious correlations. To address it, we propose ODEdit, a plug-and-play invariant learning based framework that optimizes the tripartite OOD risk objective to simultaneously enhance editing reliability, locality, and this http URL further introduce an edit trajectory invariant learning method, which integrates a total variation penalty into the risk minimization objective to stabilize edit trajectories against environmental variations. Theoretical analysis and extensive experiments demonstrate the effectiveness of ODEdit. 

---
# Component-Level Lesioning of Language Models Reveals Clinically Aligned Aphasia Phenotypes 

**Authors**: Yifan Wang, Jichen Zheng, Jingyuan Sun, Yunhao Zhang, Chunyu Ye, Jixing Li, Chengqing Zong, Shaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19723)  

**Abstract**: Large language models (LLMs) increasingly exhibit human-like linguistic behaviors and internal representations that they could serve as computational simulators of language cognition. We ask whether LLMs can be systematically manipulated to reproduce language-production impairments characteristic of aphasia following focal brain lesions. Such models could provide scalable proxies for testing rehabilitation hypotheses, and offer a controlled framework for probing the functional organization of language. We introduce a clinically grounded, component-level framework that simulates aphasia by selectively perturbing functional components in LLMs, and apply it to both modular Mixture-of-Experts models and dense Transformers using a unified intervention interface. Our pipeline (i) identifies subtype-linked components for Broca's and Wernicke's aphasia, (ii) interprets these components with linguistic probing tasks, and (iii) induces graded impairments by progressively perturbing the top-k subtype-linked components, evaluating outcomes with Western Aphasia Battery (WAB) subtests summarized by Aphasia Quotient (AQ). Across architectures and lesioning strategies, subtype-targeted perturbations yield more systematic, aphasia-like regressions than size-matched random perturbations, and MoE modularity supports more localized and interpretable phenotype-to-component mappings. These findings suggest that modular LLMs, combined with clinically informed component perturbations, provide a promising platform for simulating aphasic language production and studying how distinct language functions degrade under targeted disruptions. 

---
# R^3: Replay, Reflection, and Ranking Rewards for LLM Reinforcement Learning 

**Authors**: Zhizheng Jiang, Kang Zhao, Weikai Xu, Xinkui Lin, Wei Liu, Jian Luan, Shuo Shang, Peng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.19620)  

**Abstract**: Large reasoning models (LRMs) aim to solve diverse and complex problems through structured reasoning. Recent advances in group-based policy optimization methods have shown promise in enabling stable advantage estimation without reliance on process-level annotations. However, these methods rely on advantage gaps induced by high-quality samples within the same batch, which makes the training process fragile and inefficient when intra-group advantages collapse under challenging tasks. To address these problems, we propose a reinforcement learning mechanism named \emph{\textbf{R^3}} that along three directions: (1) a \emph{cross-context \underline{\textbf{R}}eplay} strategy that maintains the intra-group advantage by recalling valuable examples from historical trajectories of the same query, (2) an \emph{in-context self-\underline{\textbf{R}}eflection} mechanism enabling models to refine outputs by leveraging past failures, and (3) a \emph{structural entropy \underline{\textbf{R}}anking reward}, which assigns relative rewards to truncated or failed samples by ranking responses based on token-level entropy patterns, capturing both local exploration and global stability. We implement our method on Deepseek-R1-Distill-Qwen-1.5B and train it on the DeepscaleR-40k in the math domain. Experiments demonstrate our method achieves SoTA performance on several math benchmarks, representing significant improvements and fewer reasoning tokens over the base models. Code and model will be released. 

---
# Up to 36x Speedup: Mask-based Parallel Inference Paradigm for Key Information Extraction in MLLMs 

**Authors**: Xinzhong Wang, Ya Guo, Jing Li, Huan Chen, Yi Tu, Yijie Hong, Gongshen Liu, Huijia Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19613)  

**Abstract**: Key Information Extraction (KIE) from visually-rich documents (VrDs) is a critical task, for which recent Large Language Models (LLMs) and Multi-Modal Large Language Models (MLLMs) have demonstrated strong potential. However, their reliance on autoregressive inference, which generates outputs sequentially, creates a significant efficiency bottleneck, especially as KIE tasks often involve extracting multiple, semantically independent fields. To overcome this limitation, we introduce PIP: a Parallel Inference Paradigm for KIE. Our approach reformulates the problem by using "[mask]" tokens as placeholders for all target values, enabling their simultaneous generation in a single forward pass. To facilitate this paradigm, we develop a tailored mask pre-training strategy and construct large-scale supervised datasets. Experimental results show that our PIP-models achieve a 5-36x inference speedup with negligible performance degradation compared to traditional autoregressive base models. By substantially improving efficiency while maintaining high accuracy, PIP paves the way for scalable and practical real-world KIE solutions. 

---
# Explicit Multi-head Attention for Inter-head Interaction in Large Language Models 

**Authors**: Runyu Peng, Yunhua Zhou, Demin Song, Kai Lv, Bo Wang, Qipeng Guo, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19611)  

**Abstract**: In large language models built upon the Transformer architecture, recent studies have shown that inter-head interaction can enhance attention performance. Motivated by this, we propose Multi-head Explicit Attention (MEA), a simple yet effective attention variant that explicitly models cross-head interaction. MEA consists of two key components: a Head-level Linear Composition (HLC) module that separately applies learnable linear combinations to the key and value vectors across heads, thereby enabling rich inter-head communication; and a head-level Group Normalization layer that aligns the statistical properties of the recombined heads. MEA shows strong robustness in pretraining, which allows the use of larger learning rates that lead to faster convergence, ultimately resulting in lower validation loss and improved performance across a range of tasks. Furthermore, we explore the parameter efficiency of MEA by reducing the number of attention heads and leveraging HLC to reconstruct them using low-rank "virtual heads". This enables a practical key-value cache compression strategy that reduces KV-cache memory usage by 50% with negligible performance loss on knowledge-intensive and scientific reasoning tasks, and only a 3.59% accuracy drop for Olympiad-level mathematical benchmarks. 

---
# LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation 

**Authors**: Chongjun Xia, Yanchun Peng, Xianzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19585)  

**Abstract**: Interactive recommender systems can dynamically adapt to user feedback, but often suffer from content homogeneity and filter bubble effects due to overfitting short-term user preferences. While recent efforts aim to improve content diversity, they predominantly operate in static or one-shot settings, neglecting the long-term evolution of user interests. Reinforcement learning provides a principled framework for optimizing long-term user satisfaction by modeling sequential decision-making processes. However, its application in recommendation is hindered by sparse, long-tailed user-item interactions and limited semantic planning capabilities. In this work, we propose LLM-Enhanced Reinforcement Learning (LERL), a novel hierarchical recommendation framework that integrates the semantic planning power of LLM with the fine-grained adaptability of RL. LERL consists of a high-level LLM-based planner that selects semantically diverse content categories, and a low-level RL policy that recommends personalized items within the selected semantic space. This hierarchical design narrows the action space, enhances planning efficiency, and mitigates overexposure to redundant content. Extensive experiments on real-world datasets demonstrate that LERL significantly improves long-term user satisfaction when compared with state-of-the-art baselines. The implementation of LERL is available at this https URL. 

---
# From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation 

**Authors**: Yongqi Wang, Xiaofeng Ji, Jie Wang, Qingbin Li, Xiao Xiong, Zheming Yang, Jian Xu, Minghui Qiu, Xinxiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19588)  

**Abstract**: Adapting Large Language Models (LLMs) to specialized domains without human-annotated data is a crucial yet formidable challenge. Widely adopted knowledge distillation methods often devolve into coarse-grained mimicry, where the student model inefficiently targets its own weaknesses and risks inheriting the teacher's reasoning flaws. This exposes a critical pedagogical dilemma: how to devise a reliable curriculum when the teacher itself is not an infallible expert. Our work resolves this by capitalizing on a key insight: while LLMs may exhibit fallibility in complex, holistic reasoning, they often exhibit high fidelity on focused, atomic sub-problems. Based on this, we propose Divergence-Guided Reasoning Curriculum (DGRC), which constructs a learning path from atomic knowledge to reasoning chains by dynamically deriving two complementary curricula from disagreements in reasoning pathways. When a student and teacher produce conflicting results, DGRC directs the teacher to perform a diagnostic analysis: it analyzes both reasoning paths to formulate atomic queries that target the specific points of divergence, and then self-answers these queries to create high-confidence atomic question-answer pairs. These pairs then serve a dual purpose: (1) providing an atomic curriculum to rectify the student's knowledge gaps, and (2) serving as factual criteria to filter the teacher's original reasoning chains, yielding a verified CoT curriculum that teaches the student how to integrate atomic knowledge into complete reasoning paths. Experiments across the medical and legal domains on student models of various sizes demonstrate the effectiveness of our DGRC framework. Notably, our method achieves a 7.76% relative improvement for the 1.5B student model in the medical domain over strong unlabeled baseline. 

---
# AACR-Bench: Evaluating Automatic Code Review with Holistic Repository-Level Context 

**Authors**: Lei Zhang, Yongda Yu, Minghui Yu, Xinxin Guo, Zhengqi Zhuang, Guoping Rong, Dong Shao, Haifeng Shen, Hongyu Kuang, Zhengfeng Li, Boge Wang, Guoan Zhang, Bangyu Xiang, Xiaobing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19494)  

**Abstract**: High-quality evaluation benchmarks are pivotal for deploying Large Language Models (LLMs) in Automated Code Review (ACR). However, existing benchmarks suffer from two critical limitations: first, the lack of multi-language support in repository-level contexts, which restricts the generalizability of evaluation results; second, the reliance on noisy, incomplete ground truth derived from raw Pull Request (PR) comments, which constrains the scope of issue detection. To address these challenges, we introduce AACR-Bench a comprehensive benchmark that provides full cross-file context across multiple programming languages. Unlike traditional datasets, AACR-Bench employs an "AI-assisted, Expert-verified" annotation pipeline to uncover latent defects often overlooked in original PRs, resulting in a 285\% increase in defect coverage. Extensive evaluations of mainstream LLMs on AACR-Bench reveal that previous assessments may have either misjudged or only partially captured model capabilities due to data limitations. Our work establishes a more rigorous standard for ACR evaluation and offers new insights on LLM based ACR, i.e., the granularity/level of context and the choice of retrieval methods significantly impact ACR performance, and this influence varies depending on the LLM, programming language, and the LLM usage paradigm e.g., whether an Agent architecture is employed. The code, data, and other artifacts of our evaluation set are available at this https URL . 

---
# LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment 

**Authors**: Haonan Zhang, Dongxia Wang, Yi Liu, Kexin Chen, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19487)  

**Abstract**: Safety-aligned LLMs suffer from two failure modes: jailbreak (answering harmful inputs) and over-refusal (declining benign queries). Existing vector steering methods adjust the magnitude of answer vectors, but this creates a fundamental trade-off -- reducing jailbreak increases over-refusal and vice versa. We identify the root cause: LLMs encode the decision to answer (answer vector $v_a$) and the judgment of input safety (benign vector $v_b$) as nearly orthogonal directions, treating them as independent processes. We propose LLM-VA, which aligns $v_a$ with $v_b$ through closed-form weight updates, making the model's willingness to answer causally dependent on its safety assessment -- without fine-tuning or architectural changes. Our method identifies vectors at each layer using SVMs, selects safety-relevant layers, and iteratively aligns vectors via minimum-norm weight modifications. Experiments on 12 LLMs demonstrate that LLM-VA achieves 11.45% higher F1 than the best baseline while preserving 95.92% utility, and automatically adapts to each model's safety bias without manual tuning. Code and models are available at this https URL. 

---
# KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking 

**Authors**: Vítor N. Lourenço, Aline Paes, Tillman Weyde, Audrey Depeige, Mohnish Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.19447)  

**Abstract**: Claim verification is a core component of automated fact-checking systems, aimed at determining the truthfulness of a statement by assessing it against reliable evidence sources such as documents or knowledge bases. This work presents KG-CRAFT, a method that improves automatic claim verification by leveraging large language models (LLMs) augmented with contrastive questions grounded in a knowledge graph. KG-CRAFT first constructs a knowledge graph from claims and associated reports, then formulates contextually relevant contrastive questions based on the knowledge graph structure. These questions guide the distillation of evidence-based reports, which are synthesised into a concise summary that is used for veracity assessment by LLMs. Extensive evaluations on two real-world datasets (LIAR-RAW and RAWFC) demonstrate that our method achieves a new state-of-the-art in predictive performance. Comprehensive analyses validate in detail the effectiveness of our knowledge graph-based contrastive reasoning approach in improving LLMs' fact-checking capabilities. 

---
# Selective Steering: Norm-Preserving Control Through Discriminative Layer Selection 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2601.19375)  

**Abstract**: Despite significant progress in alignment, large language models (LLMs) remain vulnerable to adversarial attacks that elicit harmful behaviors. Activation steering techniques offer a promising inference-time intervention approach, but existing methods suffer from critical limitations: activation addition requires careful coefficient tuning and is sensitive to layer-specific norm variations, while directional ablation provides only binary control. Recent work on Angular Steering introduces continuous control via rotation in a 2D subspace, but its practical implementation violates norm preservation, causing distribution shift and generation collapse, particularly in models below 7B parameters. We propose Selective Steering, which addresses these limitations through two key innovations: (1) a mathematically rigorous norm-preserving rotation formulation that maintains activation distribution integrity, and (2) discriminative layer selection that applies steering only where feature representations exhibit opposite-signed class alignment. Experiments across nine models demonstrate that Selective Steering achieves 5.5x higher attack success rates than prior methods while maintaining zero perplexity violations and approximately 100\% capability retention on standard benchmarks. Our approach provides a principled, efficient framework for controllable and stable LLM behavior modification. Code: this https URL 

---
# When Benchmarks Leak: Inference-Time Decontamination for LLMs 

**Authors**: Jianzhe Chai, Yu Zhe, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2601.19334)  

**Abstract**: Benchmark-based evaluation is the de facto standard for comparing large language models (LLMs). However, its reliability is increasingly threatened by test set contamination, where test samples or their close variants leak into training data and artificially inflate reported performance. To address this issue, prior work has explored two main lines of mitigation. One line attempts to identify and remove contaminated benchmark items before evaluation, but this inevitably alters the evaluation set itself and becomes unreliable when contamination is moderate or severe. The other line preserves the benchmark and instead suppresses contaminated behavior at evaluation time; however, such interventions often interfere with normal inference and lead to noticeable performance degradation on clean inputs. We propose DeconIEP, a decontamination framework that operates entirely during evaluation by applying small, bounded perturbations in the input embedding space. Guided by a relatively less-contaminated reference model, DeconIEP learns an instance-adaptive perturbation generator that steers the evaluated model away from memorization-driven shortcut pathways. Across multiple open-weight LLMs and benchmarks, extensive empirical results show that DeconIEP achieves strong decontamination effectiveness while incurring only minimal degradation in benign utility. 

---
# Innovator-VL: A Multimodal Large Language Model for Scientific Discovery 

**Authors**: Zichen Wen, Boxue Yang, Shuang Chen, Yaojie Zhang, Yuhang Han, Junlong Ke, Cong Wang, Yicheng Fu, Jiawang Zhao, Jiangchao Yao, Xi Fang, Zhen Wang, Henxing Cai, Lin Yao, Zhifeng Gao, Yanhui Hong, Nang Yuan, Yixuan Li, Guojiang Zhao, Haoyi Tao, Nan Wang, Han Lyu, Guolin Ke, Ning Liao, Xiaoxing Wang, Kai Chen, Zhiyu Li, Feiyu Xiong, Sihan Hu, Kun Chen, Yanfeng Wang, Weinan E, Linfeng Zhang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19325)  

**Abstract**: We present Innovator-VL, a scientific multimodal large language model designed to advance understanding and reasoning across diverse scientific domains while maintaining excellent performance on general vision tasks. Contrary to the trend of relying on massive domain-specific pretraining and opaque pipelines, our work demonstrates that principled training design and transparent methodology can yield strong scientific intelligence with substantially reduced data requirements. (i) First, we provide a fully transparent, end-to-end reproducible training pipeline, covering data collection, cleaning, preprocessing, supervised fine-tuning, reinforcement learning, and evaluation, along with detailed optimization recipes. This facilitates systematic extension by the community. (ii) Second, Innovator-VL exhibits remarkable data efficiency, achieving competitive performance on various scientific tasks using fewer than five million curated samples without large-scale pretraining. These results highlight that effective reasoning can be achieved through principled data selection rather than indiscriminate scaling. (iii) Third, Innovator-VL demonstrates strong generalization, achieving competitive performance on general vision, multimodal reasoning, and scientific benchmarks. This indicates that scientific alignment can be integrated into a unified model without compromising general-purpose capabilities. Our practices suggest that efficient, reproducible, and high-performing scientific multimodal models can be built even without large-scale data, providing a practical foundation for future research. 

---
# Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning 

**Authors**: Kishan Panaganti, Zhenwen Liang, Wenhao Yu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19280)  

**Abstract**: Recent progress in Large Language Model (LLM) reasoning is increasingly driven by the refinement of post-training loss functions and alignment strategies. However, standard Reinforcement Learning (RL) paradigms like Group Relative Policy Optimization (GRPO) remain constrained by static uniformity: uniform prompt sampling and a fixed number of rollouts per prompt. For heterogeneous, heavy-tailed reasoning data, this creates structural inefficiencies that waste compute on already-solved patterns while under-training the long tail of hard problems. To address this, we propose Multi-Adversary Group Distributionally Robust Optimization (GDRO), an optimization-first framework that moves beyond uniform reasoning models by dynamically adapting the training distribution.
We introduce an Online Difficulty Classifier that partitions prompts into dynamic pass@k difficulty groups. We then propose two independent GDRO games for post-training: (1) Prompt-GDRO, which employs an EMA-debiased multiplicative-weights bandit sampler to target the intensive difficulty margin and upweight persistently hard groups without frequency bias; and (2) Rollout-GDRO, which uses a shadow-price controller to reallocate rollouts across groups, maximizing gradient variance reduction on hard tasks under a fixed mean budget (compute-neutral). We provide no-regret guarantees for both controllers and additionally a variance-proxy analysis motivating a square-root optimal rollout allocation for Rollout-GDRO. We validate our framework on the DAPO 14.1k dataset using Qwen3-Base models. Prompt-GDRO and Rollout-GDRO achieve average relative gains of +10.6% and +10.1%, respectively, in pass@8 accuracy across 1.7B, 4B, and 8B scales compared to the GRPO baseline. Qualitative analysis shows an emergent curriculum: the adversaries shift resources to the evolving reasoning frontier, enhancing the reasoning model's performance. 

---
# Riddle Quest : The Enigma of Words 

**Authors**: Niharika Sri Parasa, Chaitali Diwan, Srinath Srinivasa  

**Link**: [PDF](https://arxiv.org/pdf/2601.19273)  

**Abstract**: Riddles are concise linguistic puzzles that describe an object or idea through indirect, figurative, or playful clues. They are a longstanding form of creative expression, requiring the solver to interpret hints, recognize patterns, and draw inferences to identify the answers. In this work, we introduce a simple pipeline for creating and evaluating analogy-based riddles. The system includes a triples creator that builds structured facts about a concept, a semantic mapper that selects attributes useful for analogy, a stylized generator that turns them into riddle clues, and a validator that collects all possible answers the riddle could point to. We use this validator to study whether large language models can recover the full answer set for different riddle types. Our case study shows that while models often guess the main intended answer, they frequently miss other valid interpretations. This highlights the value of riddles as a lightweight tool for examining reasoning coverage and ambiguity handling in language models. 

---
# LLM-Assisted Logic Rule Learning: Scaling Human Expertise for Time Series Anomaly Detection 

**Authors**: Haoting Zhang, Shekhar Jain  

**Link**: [PDF](https://arxiv.org/pdf/2601.19255)  

**Abstract**: Time series anomaly detection is critical for supply chain management to take proactive operations, but faces challenges: classical unsupervised anomaly detection based on exploiting data patterns often yields results misaligned with business requirements and domain knowledge, while manual expert analysis cannot scale to millions of products in the supply chain. We propose a framework that leverages large language models (LLMs) to systematically encode human expertise into interpretable, logic-based rules for detecting anomaly patterns in supply chain time series data. Our approach operates in three stages: 1) LLM-based labeling of training data instructed by domain knowledge, 2) automated generation and iterative improvements of symbolic rules through LLM-driven optimization, and 3) rule augmentation with business-relevant anomaly categories supported by LLMs to enhance interpretability. The experiment results showcase that our approach outperforms the unsupervised learning methods in both detection accuracy and interpretability. Furthermore, compared to direct LLM deployment for time series anomaly detection, our approach provides consistent, deterministic results with low computational latency and cost, making it ideal for production deployment. The proposed framework thus demonstrates how LLMs can bridge the gap between scalable automation and expert-driven decision-making in operational settings. 

---
# RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering 

**Authors**: Kaehyun Um, KyuHwan Yeom, Haerim Yang, Minyoung Choi, Hyeongjun Yang, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.19225)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications. 

---
# A Hybrid Supervised-LLM Pipeline for Actionable Suggestion Mining in Unstructured Customer Reviews 

**Authors**: Aakash Trivedi, Aniket Upadhyay, Pratik Narang, Dhruv Kumar, Praveen Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.19214)  

**Abstract**: Extracting actionable suggestions from customer reviews is essential for operational decision-making, yet these directives are often embedded within mixed-intent, unstructured text. Existing approaches either classify suggestion-bearing sentences or generate high-level summaries, but rarely isolate the precise improvement instructions businesses need. We evaluate a hybrid pipeline combining a high-recall RoBERTa classifier trained with a precision-recall surrogate to reduce unrecoverable false negatives with a controlled, instruction-tuned LLM for suggestion extraction, categorization, clustering, and summarization. Across real-world hospitality and food datasets, the hybrid system outperforms prompt-only, rule-based, and classifier-only baselines in extraction accuracy and cluster coherence. Human evaluations further confirm that the resulting suggestions and summaries are clear, faithful, and interpretable. Overall, our results show that hybrid reasoning architectures achieve meaningful improvements fine-grained actionable suggestion mining while highlighting challenges in domain adaptation and efficient local deployment. 

---
# HELM: A Human-Centered Evaluation Framework for LLM-Powered Recommender Systems 

**Authors**: Sushant Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2601.19197)  

**Abstract**: The integration of Large Language Models (LLMs) into recommendation systems has introduced unprecedented capabilities for natural language understanding, explanation generation, and conversational interactions. However, existing evaluation methodologies focus predominantly on traditional accuracy metrics, failing to capture the multifaceted human-centered qualities that determine the real-world user experience. We introduce \framework{} (\textbf{H}uman-centered \textbf{E}valuation for \textbf{L}LM-powered reco\textbf{M}menders), a comprehensive evaluation framework that systematically assesses LLM-powered recommender systems across five human-centered dimensions: \textit{Intent Alignment}, \textit{Explanation Quality}, \textit{Interaction Naturalness}, \textit{Trust \& Transparency}, and \textit{Fairness \& Diversity}. Through extensive experiments involving three state-of-the-art LLM-based recommenders (GPT-4, LLaMA-3.1, and P5) across three domains (movies, books, and restaurants), and rigorous evaluation by 12 domain experts using 847 recommendation scenarios, we demonstrate that \framework{} reveals critical quality dimensions invisible to traditional metrics. Our results show that while GPT-4 achieves superior explanation quality (4.21/5.0) and interaction naturalness (4.35/5.0), it exhibits a significant popularity bias (Gini coefficient 0.73) compared to traditional collaborative filtering (0.58). We release \framework{} as an open-source toolkit to advance human-centered evaluation practices in the recommender systems community. 

---
# SHIELD: An Auto-Healing Agentic Defense Framework for LLM Resource Exhaustion Attacks 

**Authors**: Nirhoshan Sivaroopan, Kanchana Thilakarathna, Albert Zomaya, Manu, Yi Guo, Jo Plested, Tim Lynar, Jack Yang, Wangli Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.19174)  

**Abstract**: Sponge attacks increasingly threaten LLM systems by inducing excessive computation and DoS. Existing defenses either rely on statistical filters that fail on semantically meaningful attacks or use static LLM-based detectors that struggle to adapt as attack strategies evolve. We introduce SHIELD, a multi-agent, auto-healing defense framework centered on a three-stage Defense Agent that integrates semantic similarity retrieval, pattern matching, and LLM-based reasoning. Two auxiliary agents, a Knowledge Updating Agent and a Prompt Optimization Agent, form a closed self-healing loop, when an attack bypasses detection, the system updates an evolving knowledgebase, and refines defense instructions. Extensive experiments show that SHIELD consistently outperforms perplexity-based and standalone LLM defenses, achieving high F1 scores across both non-semantic and semantic sponge attacks, demonstrating the effectiveness of agentic self-healing against evolving resource-exhaustion threats. 

---
# AgenticSCR: An Autonomous Agentic Secure Code Review for Immature Vulnerabilities Detection 

**Authors**: Wachiraphan Charoenwet, Kla Tantithamthavorn, Patanamon Thongtanunam, Hong Yi Lin, Minwoo Jeong, Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19138)  

**Abstract**: Secure code review is critical at the pre-commit stage, where vulnerabilities must be caught early under tight latency and limited-context constraints. Existing SAST-based checks are noisy and often miss immature, context-dependent vulnerabilities, while standalone Large Language Models (LLMs) are constrained by context windows and lack explicit tool use. Agentic AI, which combine LLMs with autonomous decision-making, tool invocation, and code navigation, offer a promising alternative, but their effectiveness for pre-commit secure code review is not yet well understood. In this work, we introduce AgenticSCR, an agentic AI for secure code review for detecting immature vulnerabilities during the pre-commit stage, augmented by security-focused semantic memories. Using our own curated benchmark of immature vulnerabilities, tailored to the pre-commit secure code review, we empirically evaluate how accurate is our AgenticSCR for localizing, detecting, and explaining immature vulnerabilities. Our results show that AgenticSCR achieves at least 153% relatively higher percentage of correct code review comments than the static LLM-based baseline, and also substantially surpasses SAST tools. Moreover, AgenticSCR generates more correct comments in four out of five vulnerability types, consistently and significantly outperforming all other baselines. These findings highlight the importance of Agentic Secure Code Review, paving the way towards an emerging research area of immature vulnerability detection. 

---
# LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems 

**Authors**: Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19121)  

**Abstract**: Recommendation systems must optimize multiple objectives while satisfying hard business constraints such as fairness and coverage. For example, an e-commerce platform may require every recommendation list to include items from multiple sellers and at least one newly listed product; violating such constraints--even once--is unacceptable in production. Prior work on multi-objective recommendation and recent LLM-based recommender agents largely treat constraints as soft penalties or focus on item scoring and interaction, leading to frequent violations in real-world deployments. How to leverage LLMs for coordinating constrained optimization in recommendation systems remains underexplored. We propose DualAgent-Rec, an LLM-coordinated dual-agent framework for constrained multi-objective e-commerce recommendation. The framework separates optimization into an Exploitation Agent that prioritizes accuracy under hard constraints and an Exploration Agent that promotes diversity through unconstrained Pareto search. An LLM-based coordinator adaptively allocates resources between agents based on optimization progress and constraint satisfaction, while an adaptive epsilon-relaxation mechanism guarantees feasibility of final solutions. Experiments on the Amazon Reviews 2023 dataset demonstrate that DualAgent-Rec achieves 100% constraint satisfaction and improves Pareto hypervolume by 4-6% over strong baselines, while maintaining competitive accuracy-diversity trade-offs. These results indicate that LLMs can act as effective orchestration agents for deployable and constraint-compliant recommendation systems. 

---
# RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation 

**Authors**: Guilin Zhang, Kai Zhao, Jeffrey Friedman, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19120)  

**Abstract**: Large Language Models (LLMs) are increasingly used to generate natural-language explanations in recommender systems, acting as explanation agents that reason over user behavior histories. While prior work has focused on explanation fluency and relevance under fixed inputs, the robustness of LLM-generated explanations to realistic user behavior noise remains largely unexplored. In real-world web platforms, interaction histories are inherently noisy due to accidental clicks, temporal inconsistencies, missing values, and evolving preferences, raising concerns about explanation stability and user trust. We present RobustExplain, the first systematic evaluation framework for measuring the robustness of LLM-generated recommendation explanations. RobustExplain introduces five realistic user behavior perturbations evaluated across multiple severity levels and a multi-dimensional robustness metric capturing semantic, keyword, structural, and length consistency. Our goal is to establish a principled, task-level evaluation framework and initial robustness baselines, rather than to provide a comprehensive leaderboard across all available LLMs. Experiments on four representative LLMs (7B--70B) show that current models exhibit only moderate robustness, with larger models achieving up to 8% higher stability. Our results establish the first robustness benchmarks for explanation agents and highlight robustness as a critical dimension for trustworthy, agent-driven recommender systems at web scale. 

---
# Detecting and Correcting Hallucinations in LLM-Generated Code via Deterministic AST Analysis 

**Authors**: Dipin Khati, Daniel Rodriguez-Cardenas, Paul Pantzer, Denys Poshyvanyk  

**Link**: [PDF](https://arxiv.org/pdf/2601.19106)  

**Abstract**: Large Language Models (LLMs) for code generation boost productivity but frequently introduce Knowledge Conflicting Hallucinations (KCHs), subtle, semantic errors, such as non-existent API parameters, that evade linters and cause runtime failures. Existing mitigations like constrained decoding or non-deterministic LLM-in-the-loop repair are often unreliable for these errors. This paper investigates whether a deterministic, static-analysis framework can reliably detect \textit{and} auto-correct KCHs. We propose a post-processing framework that parses generated code into an Abstract Syntax Tree (AST) and validates it against a dynamically-generated Knowledge Base (KB) built via library introspection. This non-executing approach uses deterministic rules to find and fix both API and identifier-level conflicts. On a manually-curated dataset of 200 Python snippets, our framework detected KCHs with 100\% precision and 87.6\% recall (0.934 F1-score), and successfully auto-corrected 77.0\% of all identified hallucinations. Our findings demonstrate that this deterministic post-processing approach is a viable and reliable alternative to probabilistic repair, offering a clear path toward trustworthy code generation. 

---
# HalluJudge: A Reference-Free Hallucination Detection for Context Misalignment in Code Review Automation 

**Authors**: Kla Tantithamthavorn, Hong Yi Lin, Patanamon Thongtanunam, Wachiraphan Charoenwet, Minwoo Jeong, Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.19072)  

**Abstract**: Large Language models (LLMs) have shown strong capabilities in code review automation, such as review comment generation, yet they suffer from hallucinations -- where the generated review comments are ungrounded in the actual code -- poses a significant challenge to the adoption of LLMs in code review workflows. To address this, we explore effective and scalable methods for a hallucination detection in LLM-generated code review comments without the reference. In this work, we design HalluJudge that aims to assess the grounding of generated review comments based on the context alignment. HalluJudge includes four key strategies ranging from direct assessment to structured multi-branch reasoning (e.g., Tree-of-Thoughts). We conduct a comprehensive evaluation of these assessment strategies across Atlassian's enterprise-scale software projects to examine the effectiveness and cost-efficiency of HalluJudge. Furthermore, we analyze the alignment between HalluJudge's judgment and developer preference of the actual LLM-generated code review comments in the real-world production. Our results show that the hallucination assessment in HalluJudge is cost-effective with an F1 score of 0.85 and an average cost of $0.009. On average, 67% of the HalluJudge assessments are aligned with the developer preference of the actual LLM-generated review comments in the online production. Our results suggest that HalluJudge can serve as a practical safeguard to reduce developers' exposure to hallucinated comments, fostering trust in AI-assisted code reviews. 

---
# Principled Fine-tuning of LLMs from User-Edits: A Medley of Preference, Supervision, and Reward 

**Authors**: Dipendra Misra, Aldo Pacchiano, Ta-Chung Chi, Ge Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19055)  

**Abstract**: We study how to fine-tune LLMs using user-edit deployment data consisting of a set of context, an agent's response, and user edits. This deployment data is naturally generated by users in applications such as LLMs-based writing assistants and coding agents. The _natural_ origin of user edits makes it a desired source for adapting and personalizing LLMs. In this setup, there emerges a unification of various feedback types namely preferences, supervised labels, and cost that are typically studied separately in the literature. In this paper, we initiate the theoretical investigation of learning from user edits. We first derive bounds for learning algorithms that learn from each of these feedback types. We prove that these algorithms have different trade-offs depending upon the user, data distribution, and model class. We then propose a simple ensembling procedure to jointly learn from these feedback types. On two domains adapted from Gao et al. 2024, we show our ensembling procedure outperforms these methods that learn from individual feedback. Further, we show that our proposed procedure can robustly adapt to different user-edit distributions at test time. 

---
# From Answer Givers to Design Mentors: Guiding LLMs with the Cognitive Apprenticeship Model 

**Authors**: Yongsu Ahn, Lejun R Liao, Benjamin Bach, Nam Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.19053)  

**Abstract**: Design feedback helps practitioners improve their artifacts while also fostering reflection and design reasoning. Large Language Models (LLMs) such as ChatGPT can support design work, but often provide generic, one-off suggestions that limit reflective engagement. We investigate how to guide LLMs to act as design mentors by applying the Cognitive Apprenticeship Model, which emphasizes demonstrating reasoning through six methods: modeling, coaching, scaffolding, articulation, reflection, and exploration. We operationalize these instructional methods through structured prompting and evaluate them in a within-subjects study with data visualization practitioners. Participants interacted with both a baseline LLM and an instructional LLM designed with cognitive apprenticeship prompts. Surveys, interviews, and conversational log analyses compared experiences across conditions. Our findings show that cognitively informed prompts elicit deeper design reasoning and more reflective feedback exchanges, though the baseline is sometimes preferred depending on task types or experience levels. We distill design considerations for AI-assisted feedback systems that foster reflective practice. 

---
# LLMs versus the Halting Problem: Revisiting Program Termination Prediction 

**Authors**: Oren Sultan, Jordi Armengol-Estape, Pascal Kesseli, Julien Vanegue, Dafna Shahaf, Yossi Adi, Peter O'Hearn  

**Link**: [PDF](https://arxiv.org/pdf/2601.18987)  

**Abstract**: Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems. 

---
# Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries 

**Authors**: Yuchen Zhang, Ravi Shekhar, Haralambos Mouratidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.18899)  

**Abstract**: Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment. 

---
# Reducing False Positives in Static Bug Detection with LLMs: An Empirical Study in Industry 

**Authors**: Xueying Du, Jiayi Feng, Yi Zou, Wei Xu, Jie Ma, Wei Zhang, Sisi Liu, Xin Peng, Yiling Lou  

**Link**: [PDF](https://arxiv.org/pdf/2601.18844)  

**Abstract**: Static analysis tools (SATs) are widely adopted in both academia and industry for improving software quality, yet their practical use is often hindered by high false positive rates, especially in large-scale enterprise systems. These false alarms demand substantial manual inspection, creating severe inefficiencies in industrial code review. While recent work has demonstrated the potential of large language models (LLMs) for false alarm reduction on open-source benchmarks, their effectiveness in real-world enterprise settings remains unclear. To bridge this gap, we conduct the first comprehensive empirical study of diverse LLM-based false alarm reduction techniques in an industrial context at Tencent, one of the largest IT companies in China. Using data from Tencent's enterprise-customized SAT on its large-scale Advertising and Marketing Services software, we construct a dataset of 433 alarms (328 false positives, 105 true positives) covering three common bug types. Through interviewing developers and analyzing the data, our results highlight the prevalence of false positives, which wastes substantial manual effort (e.g., 10-20 minutes of manual inspection per alarm). Meanwhile, our results show the huge potential of LLMs for reducing false alarms in industrial settings (e.g., hybrid techniques of LLM and static analysis eliminate 94-98% of false positives with high recall). Furthermore, LLM-based techniques are cost-effective, with per-alarm costs as low as 2.1-109.5 seconds and $0.0011-$0.12, representing orders-of-magnitude savings compared to manual review. Finally, our case analysis further identifies key limitations of LLM-based false alarm reduction in industrial settings. 

---
# MulVul: Retrieval-augmented Multi-Agent Code Vulnerability Detection via Cross-Model Prompt Evolution 

**Authors**: Zihan Wu, Jie Xu, Yun Peng, Chun Yong Chong, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.18847)  

**Abstract**: Large Language Models (LLMs) struggle to automate real-world vulnerability detection due to two key limitations: the heterogeneity of vulnerability patterns undermines the effectiveness of a single unified model, and manual prompt engineering for massive weakness categories is unscalable.
To address these challenges, we propose \textbf{MulVul}, a retrieval-augmented multi-agent framework designed for precise and broad-coverage vulnerability detection. MulVul adopts a coarse-to-fine strategy: a \emph{Router} agent first predicts the top-$k$ coarse categories and then forwards the input to specialized \emph{Detector} agents, which identify the exact vulnerability types. Both agents are equipped with retrieval tools to actively source evidence from vulnerability knowledge bases to mitigate hallucinations.
Crucially, to automate the generation of specialized prompts, we design \emph{Cross-Model Prompt Evolution}, a prompt optimization mechanism where a generator LLM iteratively refines candidate prompts while a distinct executor LLM validates their effectiveness. This decoupling mitigates the self-correction bias inherent in single-model optimization.
Evaluated on 130 CWE types, MulVul achieves 34.79\% Macro-F1, outperforming the best baseline by 41.5\%. Ablation studies validate cross-model prompt evolution, which boosts performance by 51.6\% over manual prompts by effectively handling diverse vulnerability patterns. 

---
# Encoder-Free ECG-Language Models 

**Authors**: William Han, Tony Chen, Chaojing Duan, Xiaoyu Song, Yihang Yao, Yuzhe Yang, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18798)  

**Abstract**: ECG-Language Models (ELMs) extend recent progress in Multimodal Large Language Models (MLLMs) to automated ECG interpretation. However, most ELMs follow Vision-Language Model (VLM) designs and depend on pretrained ECG encoders, adding architectural and training complexity. Inspired by encoder-free VLMs, we introduce ELF, an encoder-free ELM that replaces the ECG encoder with a single projection layer trained jointly with the LLM. Across five datasets, ELF matches or exceeds state-of-the-art ELMs that use far more complex encoders and training pipelines. We also test whether adding architectural biases to ELF improves performance and find that the single linear projection remains competitive. Finally, we show that ELF, and potentially other ELMs, often rely more on benchmark artifacts and language priors than ECG-derived information, highlighting limitations in current evaluation practices and ELM design. All data and code is available at this https URL. 

---
# SynCABEL: Synthetic Contextualized Augmentation for Biomedical Entity Linking 

**Authors**: Adam Remaki, Christel Gérardin, Eulàlia Farré-Maduell, Martin Krallinger, Xavier Tannier  

**Link**: [PDF](https://arxiv.org/pdf/2601.19667)  

**Abstract**: We present SynCABEL (Synthetic Contextualized Augmentation for Biomedical Entity Linking), a framework that addresses a central bottleneck in supervised biomedical entity linking (BEL): the scarcity of expert-annotated training data. SynCABEL leverages large language models to generate context-rich synthetic training examples for all candidate concepts in a target knowledge base, providing broad supervision without manual annotation. We demonstrate that SynCABEL, when combined with decoder-only models and guided inference establish new state-of-the-art results across three widely used multilingual benchmarks: MedMentions for English, QUAERO for French, and SPACCC for Spanish. Evaluating data efficiency, we show that SynCABEL reaches the performance of full human supervision using up to 60% less annotated data, substantially reducing reliance on labor-intensive and costly expert labeling. Finally, acknowledging that standard evaluation based on exact code matching often underestimates clinically valid predictions due to ontology redundancy, we introduce an LLM-as-a-judge protocol. This analysis reveals that SynCABEL significantly improves the rate of clinically valid predictions. Our synthetic datasets, models, and code are released to support reproducibility and future research. 

---
# Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering 

**Authors**: Fangan Dong, Zuming Yan, Xuri Ge, Zhiwei Xu, Mengqi Zhang, Xuanang Chen, Ben He, Xin Xin, Zhumin Chen, Ying Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.19847)  

**Abstract**: Despite the strong reasoning capabilities of recent large language models (LLMs), achieving reliable performance on challenging tasks often requires post-training or computationally expensive sampling strategies, limiting their practical efficiency. In this work, we first show that a small subset of neurons in LLMs exhibits strong predictive correlations with reasoning correctness. Based on this observation, we propose AdaRAS (Adaptive Reasoning Activation Steering), a lightweight test-time framework that improves reasoning reliability by selectively intervening on neuron activations. AdaRAS identifies Reasoning-Critical Neurons (RCNs) via a polarity-aware mean-difference criterion and adaptively steers their activations during inference, enhancing incorrect reasoning traces while avoiding degradation on already-correct cases. Experiments on 10 mathematics and coding benchmarks demonstrate consistent improvements, including over 13% gains on AIME-24 and AIME-25. Moreover, AdaRAS exhibits strong transferability across datasets and scalability to stronger models, outperforming post-training methods without additional training or sampling cost. 

---
# Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection 

**Authors**: Nicholas Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.19871)  

**Abstract**: Low-resource languages such as isiZulu and isiXhosa face persistent challenges in machine translation due to limited parallel data and linguistic resources. Recent advances in large language models suggest that self-reflection, prompting a model to critique and revise its own outputs, can improve reasoning quality and factual consistency. Building on this idea, this paper introduces Reflective Translation, a prompt-based framework in which a model generates an initial translation, produces a structured self-critique, and then uses this reflection to generate a refined translation. The approach is evaluated on English-isiZulu and English-isiXhosa translation using OPUS-100 and NTREX-African, across multiple prompting strategies and confidence thresholds. Results show consistent improvements in both BLEU and COMET scores between first- and second-pass translations, with average gains of up to +0.22 BLEU and +0.18 COMET. Statistical significance testing using paired nonparametric tests confirms that these improvements are robust. The proposed method is model-agnostic, requires no fine-tuning, and introduces a reflection-augmented dataset that can support future supervised or analysis-driven work. These findings demonstrate that structured self-reflection is a practical and effective mechanism for improving translation quality in low-resource settings. 

---
# Zero-Shot Stance Detection in the Wild: Dynamic Target Generation and Multi-Target Adaptation 

**Authors**: Aohua Li, Yuanshuo Zhang, Ge Gao, Bo Chen, Xiaobing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.19802)  

**Abstract**: Current stance detection research typically relies on predicting stance based on given targets and text. However, in real-world social media scenarios, targets are neither predefined nor static but rather complex and dynamic. To address this challenge, we propose a novel task: zero-shot stance detection in the wild with Dynamic Target Generation and Multi-Target Adaptation (DGTA), which aims to automatically identify multiple target-stance pairs from text without prior target knowledge. We construct a Chinese social media stance detection dataset and design multi-dimensional evaluation metrics. We explore both integrated and two-stage fine-tuning strategies for large language models (LLMs) and evaluate various baseline models. Experimental results demonstrate that fine-tuned LLMs achieve superior performance on this task: the two-stage fine-tuned Qwen2.5-7B attains the highest comprehensive target recognition score of 66.99%, while the integrated fine-tuned DeepSeek-R1-Distill-Qwen-7B achieves a stance detection F1 score of 79.26%. 

---
# Decompose-and-Formalise: Recursively Verifiable Natural Language Inference 

**Authors**: Xin Quan, Marco Valentino, Louise A. Dennis, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2601.19605)  

**Abstract**: Recent work has shown that integrating large language models (LLMs) with theorem provers (TPs) in neuro-symbolic pipelines helps with entailment verification and proof-guided refinement of explanations for natural language inference (NLI). However, scaling such refinement to naturalistic NLI remains difficult: long, syntactically rich inputs and deep multi-step arguments amplify autoformalisation errors, where a single local mismatch can invalidate the proof. Moreover, current methods often handle failures via costly global regeneration due to the difficulty of localising the responsible span or step from prover diagnostics. Aiming to address these problems, we propose a decompose-and-formalise framework that (i) decomposes premise-hypothesis pairs into an entailment tree of atomic steps, (ii) verifies the tree bottom-up to isolate failures to specific nodes, and (iii) performs local diagnostic-guided refinement instead of regenerating the whole explanation. Moreover, to improve faithfulness of autoformalisation, we introduce $\theta$-substitution in an event-based logical form to enforce consistent argument-role bindings. Across a range of reasoning tasks using five LLM backbones, our method achieves the highest explanation verification rates, improving over the state-of-the-art by 26.2%, 21.7%, 21.6% and 48.9%, while reducing refinement iterations and runtime and preserving strong NLI accuracy. 

---
# Evaluation of Oncotimia: An LLM based system for supporting tumour boards 

**Authors**: Luis Lorenzo, Marcos Montana-Mendez, Sergio Figueiras, Miguel Boubeta, Cristobal Bernardo-Castineira  

**Link**: [PDF](https://arxiv.org/pdf/2601.19899)  

**Abstract**: Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality. 

---
# Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing? 

**Authors**: Ahrii Kim, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.19410)  

**Abstract**: Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency.
Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement. 

---
# DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference 

**Authors**: Fuliang Liu, Xue Li, Ketai Zhao, Yinxi Gao, Ziyan Zhou, Zhonghui Zhang, Zhibin Wang, Wanchun Dou, Sheng Zhong, Chen Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.19278)  

**Abstract**: Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03x--3.44x wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework. Code is released at this https URL. 

---
# ReToP: Learning to Rewrite Electronic Health Records for Clinical Prediction 

**Authors**: Jesus Lovon-Melgarejo, Jose G. Moreno, Christine Damase-Michel, Lynda Tamine  

**Link**: [PDF](https://arxiv.org/pdf/2601.19286)  

**Abstract**: Electronic Health Records (EHRs) provide crucial information for clinical decision-making. However, their high-dimensionality, heterogeneity, and sparsity make clinical prediction challenging. Large Language Models (LLMs) allowed progress towards addressing this challenge by leveraging parametric medical knowledge to enhance EHR data for clinical prediction tasks. Despite the significant achievements made so far, most of the existing approaches are fundamentally task-agnostic in the sense that they deploy LLMs as EHR encoders or EHR completion modules without fully integrating signals from the prediction tasks. This naturally hinders task performance accuracy. In this work, we propose Rewrite-To-Predict (ReToP), an LLM-based framework that addresses this limitation through an end-to-end training of an EHR rewriter and a clinical predictor. To cope with the lack of EHR rewrite training data, we generate synthetic pseudo-labels using clinical-driven feature selection strategies to create diverse patient rewrites for fine-tuning the EHR rewriter. ReToP aligns the rewriter with prediction objectives using a novel Classifier Supervised Contribution (CSC) score that enables the EHR rewriter to generate clinically relevant rewrites that directly enhance prediction. Our ReToP framework surpasses strong baseline models across three clinical tasks on MIMIC-IV. Moreover, the analysis of ReToP shows its generalizability to unseen datasets and tasks with minimal fine-tuning while preserving faithful rewrites and emphasizing task-relevant predictive features. 

---
# Formula-One Prompting: Adaptive Reasoning Through Equations For Applied Mathematics 

**Authors**: Natapong Nitarach, Pittawat Taveekitworachai, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2601.19302)  

**Abstract**: Prompting techniques such as Chain-of-Thought (CoT) and Program-of-Thought (PoT) improve LLM mathematical reasoning by structuring intermediate steps in natural language or code. However, applied mathematics problems in domains like finance, physics, and cryptography often require recalling or deriving governing equations, a step that current approaches do not explicitly leverage. We propose Formula-One Prompting (F-1), a two-phase approach that uses mathematical equations as an intermediate representation before adaptive solving. F-1 first formulates governing equations from problem descriptions, then selects a solving strategy among CoT, PoT, or direct computation based on the generated equations, all within a single LLM call. Results across five models and four benchmarks show F-1 outperforms CoT by +5.76% and PoT by +8.42% on average. Crucially, gains are largest in applied domains: +13.30% on FinanceMath over CoT, and within OlympiadBench, larger gains on physics (+2.55%) than pure math (+0.44%). This demonstrates that F-1 is more effective than CoT in applied mathematics problems. 

---
# Self-Aware Knowledge Probing: Evaluating Language Models' Relational Knowledge through Confidence Calibration 

**Authors**: Christopher Kissling, Elena Merdjanovska, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2601.18901)  

**Abstract**: Knowledge probing quantifies how much relational knowledge a language model (LM) has acquired during pre-training. Existing knowledge probes evaluate model capabilities through metrics like prediction accuracy and precision. Such evaluations fail to account for the model's reliability, reflected in the calibration of its confidence scores. In this paper, we propose a novel calibration probing framework for relational knowledge, covering three modalities of model confidence: (1) intrinsic confidence, (2) structural consistency and (3) semantic grounding. Our extensive analysis of ten causal and six masked language models reveals that most models, especially those pre-trained with the masking objective, are overconfident. The best-calibrated scores come from confidence estimates that account for inconsistencies due to statement rephrasing. Moreover, even the largest pre-trained models fail to encode the semantics of linguistic confidence expressions accurately. 

---
# Malicious Repurposing of Open Science Artefacts by Using Large Language Models 

**Authors**: Zahra Hashemi, Zhiqiang Zhong, Jun Pang, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18998)  

**Abstract**: The rapid evolution of large language models (LLMs) has fuelled enthusiasm about their role in advancing scientific discovery, with studies exploring LLMs that autonomously generate and evaluate novel research ideas. However, little attention has been given to the possibility that such models could be exploited to produce harmful research by repurposing open science artefacts for malicious ends. We fill the gap by introducing an end-to-end pipeline that first bypasses LLM safeguards through persuasion-based jailbreaking, then reinterprets NLP papers to identify and repurpose their artefacts (datasets, methods, and tools) by exploiting their vulnerabilities, and finally assesses the safety of these proposals using our evaluation framework across three dimensions: harmfulness, feasibility of misuse, and soundness of technicality. Overall, our findings demonstrate that LLMs can generate harmful proposals by repurposing ethically designed open artefacts; however, we find that LLMs acting as evaluators strongly disagree with one another on evaluation outcomes: GPT-4.1 assigns higher scores (indicating greater potential harms, higher soundness and feasibility of misuse), Gemini-2.5-pro is markedly stricter, and Grok-3 falls between these extremes. This indicates that LLMs cannot yet serve as reliable judges in a malicious evaluation setup, making human evaluation essential for credible dual-use risk assessment. 

---
# ALRM: Agentic LLM for Robotic Manipulation 

**Authors**: Vitor Gaboardi dos Santos, Ibrahim Khadraoui, Ibrahim Farhat, Hamza Yous, Samy Teffahi, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2601.19510)  

**Abstract**: Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP. 

---
# Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning 

**Authors**: Haolin Liu, Dian Yu, Sidi Lu, Yujun Zhou, Rui Liu, Zhenwen Liang, Haitao Mi, Chen-Yu Wei, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18984)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful framework for improving the reasoning capabilities of large language models (LLMs). However, most existing RL approaches rely on sparse outcome rewards, which fail to credit correct intermediate steps in partially successful solutions. Process reward models (PRMs) offer fine-grained step-level supervision, but their scores are often noisy and difficult to evaluate. As a result, recent PRM benchmarks focus on a more objective capability: detecting the first incorrect step in a reasoning path. However, this evaluation target is misaligned with how PRMs are typically used in RL, where their step-wise scores are treated as raw rewards to maximize. To bridge this gap, we propose Verifiable Prefix Policy Optimization (VPPO), which uses PRMs only to localize the first error during RL. Given an incorrect rollout, VPPO partitions the trajectory into a verified correct prefix and an erroneous suffix based on the first error, rewarding the former while applying targeted penalties only after the detected mistake. This design yields stable, interpretable learning signals and improves credit assignment. Across multiple reasoning benchmarks, VPPO consistently outperforms sparse-reward RL and prior PRM-guided baselines on both Pass@1 and Pass@K. 

---
# Intent2QoS: Language Model-Driven Automation of Traffic Shaping Configurations 

**Authors**: Sudipta Acharya, Burak Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2601.18974)  

**Abstract**: Traffic shaping and Quality of Service (QoS) enforcement are critical for managing bandwidth, latency, and fairness in networks. These tasks often rely on low-level traffic control settings, which require manual setup and technical expertise. This paper presents an automated framework that converts high-level traffic shaping intents in natural or declarative language into valid and correct traffic control rules. To the best of our knowledge, we present the first end-to-end pipeline that ties intent translation in a queuing-theoretic semantic model and, with a rule-based critic, yields deployable Linux traffic control configuration sets. The framework has three steps: (1) a queuing simulation with priority scheduling and Active Queue Management (AQM) builds a semantic model; (2) a language model, using this semantic model and a traffic profile, generates sub-intents and configuration rules; and (3) a rule-based critic checks and adjusts the rules for correctness and policy compliance. We evaluate multiple language models by generating traffic control commands from business intents that comply with relevant standards for traffic control protocols. Experimental results on 100 intents show significant gains, with LLaMA3 reaching 0.88 semantic similarity and 0.87 semantic coverage, outperforming other models by over 30\. A thorough sensitivity study demonstrates that AQM-guided prompting reduces variability threefold compared to zero-shot baselines. 

---
# Post-LayerNorm Is Back: Stable, ExpressivE, and Deep 

**Authors**: Chen Chen, Lai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.19895)  

**Abstract**: Large language model (LLM) scaling is hitting a wall. Widening models yields diminishing returns, and extending context length does not improve fundamental expressivity. In contrast, depth scaling offers theoretically superior expressivity, yet current Transformer architectures struggle to train reliably at extreme depths. We revisit the Post-LayerNorm (Post-LN) formulation, whose instability at scale caused its replacement by Pre-LN in modern LLMs. We show that the central failure mode of Post-LN arises from the ResNet-style residual pathway, which introduces gradient vanishing in deep networks. We present Keel, a Post-LN Transformer that replaces this residual path with a Highway-style connection. This modification preserves the gradient flow through the residual branch, preventing signal vanishing from the top layers to the bottom. Unlike prior methods, Keel enables stable training at extreme depths without requiring specialized initialization or complex optimization tricks. Keel trains robustly at depths exceeding 1000 layers and consistently improves perplexity and depth-scaling characteristics over Pre-LN. These findings indicate that Post-LN, when paired with a Highway-style connection, provides a simple and effective foundation for building deeply scalable LLMs, opening the possibility for future infinite-depth architectures. 

---
# UniRec: Unified Multimodal Encoding for LLM-Based Recommendations 

**Authors**: Zijie Lei, Tao Feng, Zhigang Hua, Yan Xie, Guanyu Lin, Shuang Yang, Ge Liu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2601.19423)  

**Abstract**: Large language models have recently shown promise for multimodal recommendation, particularly with text and image inputs. Yet real-world recommendation signals extend far beyond these modalities. To reflect this, we formalize recommendation features into four modalities: text, images, categorical features, and numerical attributes, and highlight the unique challenges this heterogeneity poses for LLMs in understanding multimodal information. In particular, these challenges arise not only across modalities but also within them, as attributes such as price, rating, and time may all be numeric yet carry distinct semantic meanings. Beyond this intra-modality ambiguity, another major challenge is the nested structure of recommendation signals, where user histories are sequences of items, each associated with multiple attributes. To address these challenges, we propose UniRec, a unified multimodal encoder for LLM-based recommendation. UniRec first employs modality-specific encoders to produce consistent embeddings across heterogeneous signals. It then adopts a triplet representation, comprising attribute name, type, and value, to separate schema from raw inputs and preserve semantic distinctions. Finally, a hierarchical Q-Former models the nested structure of user interactions while maintaining their layered organization. Across multiple real-world benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based recommenders by up to 15%, and extensive ablation studies further validate the contributions of each component. 

---
# Comparing how Large Language Models perform against keyword-based searches for social science research data discovery 

**Authors**: Mark Green, Maura Halstead, Caroline Jay, Richard Kingston, Alex Singleton, David Topping  

**Link**: [PDF](https://arxiv.org/pdf/2601.19559)  

**Abstract**: This paper evaluates the performance of a large language model (LLM) based semantic search tool relative to a traditional keyword-based search for data discovery. Using real-world search behaviour, we compare outputs from a bespoke semantic search system applied to UKRI data services with the Consumer Data Research Centre (CDRC) keyword search. Analysis is based on 131 of the most frequently used search terms extracted from CDRC search logs between December 2023 and October 2024. We assess differences in the volume, overlap, ranking, and relevance of returned datasets using descriptive statistics, qualitative inspection, and quantitative similarity measures, including exact dataset overlap, Jaccard similarity, and cosine similarity derived from BERT embeddings. Results show that the semantic search consistently returns a larger number of results than the keyword search and performs particularly well for place based, misspelled, obscure, or complex queries. While the semantic search does not capture all keyword based results, the datasets returned are overwhelmingly semantically similar, with high cosine similarity scores despite lower exact overlap. Rankings of the most relevant results differ substantially between tools, reflecting contrasting prioritisation strategies. Case studies demonstrate that the LLM based tool is robust to spelling errors, interprets geographic and contextual relevance effectively, and supports natural-language queries that keyword search fails to resolve. Overall, the findings suggest that LLM driven semantic search offers a substantial improvement for data discovery, complementing rather than fully replacing traditional keyword-based approaches. 

---
# Reimagining Social Robots as Recommender Systems: Foundations, Framework, and Applications 

**Authors**: Jin Huang, Fethiye Irmak Doğan, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2601.19761)  

**Abstract**: Personalization in social robots refers to the ability of the robot to meet the needs and/or preferences of an individual user. Existing approaches typically rely on large language models (LLMs) to generate context-aware responses based on user metadata and historical interactions or on adaptive methods such as reinforcement learning (RL) to learn from users' immediate reactions in real time. However, these approaches fall short of comprehensively capturing user preferences-including long-term, short-term, and fine-grained aspects-, and of using them to rank and select actions, proactively personalize interactions, and ensure ethically responsible adaptations. To address the limitations, we propose drawing on recommender systems (RSs), which specialize in modeling user preferences and providing personalized recommendations. To ensure the integration of RS techniques is well-grounded and seamless throughout the social robot pipeline, we (i) align the paradigms underlying social robots and RSs, (ii) identify key techniques that can enhance personalization in social robots, and (iii) design them as modular, plug-and-play components. This work not only establishes a framework for integrating RS techniques into social robots but also opens a pathway for deep collaboration between the RS and HRI communities, accelerating innovation in both fields. 

---
# Benchmarking Multimodal Large Language Models for Missing Modality Completion in Product Catalogues 

**Authors**: Junchen Fu, Wenhao Deng, Kaiwen Zheng, Alexandros Karatzoglou, Ioannis Arapakis, Yu Ye, Yongxin Ni, Joemon M. Jose, Xuri Ge  

**Link**: [PDF](https://arxiv.org/pdf/2601.19750)  

**Abstract**: Missing-modality information on e-commerce platforms, such as absent product images or textual descriptions, often arises from annotation errors or incomplete metadata, impairing both product presentation and downstream applications such as recommendation systems. Motivated by the multimodal generative capabilities of recent Multimodal Large Language Models (MLLMs), this work investigates a fundamental yet underexplored question: can MLLMs generate missing modalities for products in e-commerce scenarios? We propose the Missing Modality Product Completion Benchmark (MMPCBench), which consists of two sub-benchmarks: a Content Quality Completion Benchmark and a Recommendation Benchmark.
We further evaluate six state-of-the-art MLLMs from the Qwen2.5-VL and Gemma-3 model families across nine real-world e-commerce categories, focusing on image-to-text and text-to-image completion tasks. Experimental results show that while MLLMs can capture high-level semantics, they struggle with fine-grained word-level and pixel- or patch-level alignment. In addition, performance varies substantially across product categories and model scales, and we observe no trivial correlation between model size and performance, in contrast to trends commonly reported in mainstream benchmarks. We also explore Group Relative Policy Optimization (GRPO) to better align MLLMs with this task. GRPO improves image-to-text completion but does not yield gains for text-to-image completion. Overall, these findings expose the limitations of current MLLMs in real-world cross-modal generation and represent an early step toward more effective missing-modality product completion. 

---
