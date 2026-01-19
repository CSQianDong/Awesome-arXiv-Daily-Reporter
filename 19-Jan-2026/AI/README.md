# BoxMind: Closed-loop AI strategy optimization for elite boxing validated in the 2024 Olympics 

**Authors**: Kaiwen Wang, Kaili Zheng, Rongrong Deng, Qingmin Fan, Milin Zhang, Zongrui Li, Xuesi Zhou, Bo Han, Liren Chen, Chenyi Guo, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11492)  

**Abstract**: Competitive sports require sophisticated tactical analysis, yet combat disciplines like boxing remain underdeveloped in AI-driven analytics due to the complexity of action dynamics and the lack of structured tactical representations. To address this, we present BoxMind, a closed-loop AI expert system validated in elite boxing competition. By defining atomic punch events with precise temporal boundaries and spatial and technical attributes, we parse match footage into 18 hierarchical technical-tactical indicators. We then propose a graph-based predictive model that fuses these explicit technical-tactical profiles with learnable, time-variant latent embeddings to capture the dynamics of boxer matchups. Modeling match outcome as a differentiable function of technical-tactical indicators, we turn winning probability gradients into executable tactical adjustments. Experiments show that the outcome prediction model achieves state-of-the-art performance, with 69.8% accuracy on BoxerGraph test set and 87.5% on Olympic matches. Using this predictive model as a foundation, the system generates strategic recommendations that demonstrate proficiency comparable to human experts. BoxMind is validated through a closed-loop deployment during the 2024 Paris Olympics, directly contributing to the Chinese National Team's historic achievement of three gold and two silver medals. BoxMind establishes a replicable paradigm for transforming unstructured video data into strategic intelligence, bridging the gap between computer vision and decision support in competitive sports. 

---
# Health Facility Location in Ethiopia: Leveraging LLMs to Integrate Expert Knowledge into Algorithmic Planning 

**Authors**: Yohai Trabelsi, Guojun Xiong, Fentabil Getnet, Stéphane Verguet, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2601.11479)  

**Abstract**: Ethiopia's Ministry of Health is upgrading health posts to improve access to essential services, particularly in rural areas. Limited resources, however, require careful prioritization of which facilities to upgrade to maximize population coverage while accounting for diverse expert and stakeholder preferences. In collaboration with the Ethiopian Public Health Institute and Ministry of Health, we propose a hybrid framework that systematically integrates expert knowledge with optimization techniques. Classical optimization methods provide theoretical guarantees but require explicit, quantitative objectives, whereas stakeholder criteria are often articulated in natural language and difficult to formalize. To bridge these domains, we develop the Large language model and Extended Greedy (LEG) framework. Our framework combines a provable approximation algorithm for population coverage optimization with LLM-driven iterative refinement that incorporates human-AI alignment to ensure solutions reflect expert qualitative guidance while preserving coverage guarantees. Experiments on real-world data from three Ethiopian regions demonstrate the framework's effectiveness and its potential to inform equitable, data-driven health system planning. 

---
# Exploring LLM Features in Predictive Process Monitoring for Small-Scale Event-Logs 

**Authors**: Alessandro Padella, Massimiliano de Leoni, Marlon Dumas  

**Link**: [PDF](https://arxiv.org/pdf/2601.11468)  

**Abstract**: Predictive Process Monitoring is a branch of process mining that aims to predict the outcome of an ongoing process. Recently, it leveraged machine-and-deep learning architectures. In this paper, we extend our prior LLM-based Predictive Process Monitoring framework, which was initially focused on total time prediction via prompting. The extension consists of comprehensively evaluating its generality, semantic leverage, and reasoning mechanisms, also across multiple Key Performance Indicators. Empirical evaluations conducted on three distinct event logs and across the Key Performance Indicators of Total Time and Activity Occurrence prediction indicate that, in data-scarce settings with only 100 traces, the LLM surpasses the benchmark methods. Furthermore, the experiments also show that the LLM exploits both its embodied prior knowledge and the internal correlations among training traces. Finally, we examine the reasoning strategies employed by the model, demonstrating that the LLM does not merely replicate existing predictive methods but performs higher-order reasoning to generate the predictions. 

---
# Hyperparameter Optimization of Constraint Programming Solvers 

**Authors**: Hedieh Haddad, Thibault Falque, Pierre Talbot, Pascal Bouvry  

**Link**: [PDF](https://arxiv.org/pdf/2601.11389)  

**Abstract**: The performance of constraint programming solvers is highly sensitive to the choice of their hyperparameters. Manually finding the best solver configuration is a difficult, time-consuming task that typically requires expert knowledge. In this paper, we introduce probe and solve algorithm, a novel two-phase framework for automated hyperparameter optimization integrated into the CPMpy library. This approach partitions the available time budget into two phases: a probing phase that explores different sets of hyperparameters using configurable hyperparameter optimization methods, followed by a solving phase where the best configuration found is used to tackle the problem within the remaining time.
We implement and compare two hyperparameter optimization methods within the probe and solve algorithm: Bayesian optimization and Hamming distance search. We evaluate the algorithm on two different constraint programming solvers, ACE and Choco, across 114 combinatorial problem instances, comparing their performance against the solver's default configurations.
Results show that using Bayesian optimization, the algorithm outperforms the solver's default configurations, improving solution quality for ACE in 25.4% of instances and matching the default performance in 57.9%, and for Choco, achieving superior results in 38.6% of instances. It also consistently surpasses Hamming distance search within the same framework, confirming the advantage of model-based exploration over simple local search. Overall, the probe and solve algorithm offers a practical, resource-aware approach for tuning constraint solvers that yields robust improvements across diverse problem types. 

---
# AstroReason-Bench: Evaluating Unified Agentic Planning across Heterogeneous Space Planning Problems 

**Authors**: Weiyi Wang, Xinchi Chen, Jingjing Gong, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11354)  

**Abstract**: Recent advances in agentic Large Language Models (LLMs) have positioned them as generalist planners capable of reasoning and acting across diverse tasks. However, existing agent benchmarks largely focus on symbolic or weakly grounded environments, leaving their performance in physics-constrained real-world domains underexplored. We introduce AstroReason-Bench, a comprehensive benchmark for evaluating agentic planning in Space Planning Problems (SPP), a family of high-stakes problems with heterogeneous objectives, strict physical constraints, and long-horizon decision-making. AstroReason-Bench integrates multiple scheduling regimes, including ground station communication and agile Earth observation, and provides a unified agent-oriented interaction protocol. Evaluating on a range of state-of-the-art open- and closed-source agentic LLM systems, we find that current agents substantially underperform specialized solvers, highlighting key limitations of generalist planning under realistic constraints. AstroReason-Bench offers a challenging and diagnostic testbed for future agentic research. 

---
# XChoice: Explainable Evaluation of AI-Human Alignment in LLM-based Constrained Choice Decision Making 

**Authors**: Weihong Qi, Fan Huang, Rasika Muralidharan, Jisun An, Haewoon Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2601.11286)  

**Abstract**: We present XChoice, an explainable framework for evaluating AI-human alignment in constrained decision making. Moving beyond outcome agreement such as accuracy and F1 score, XChoice fits a mechanism-based decision model to human data and LLM-generated decisions, recovering interpretable parameters that capture the relative importance of decision factors, constraint sensitivity, and implied trade-offs. Alignment is assessed by comparing these parameter vectors across models, options, and subgroups. We demonstrate XChoice on Americans' daily time allocation using the American Time Use Survey (ATUS) as human ground truth, revealing heterogeneous alignment across models and activities and salient misalignment concentrated in Black and married groups. We further validate robustness of XChoice via an invariance analysis and evaluate targeted mitigation with a retrieval augmented generation (RAG) intervention. Overall, XChoice provides mechanism-based metrics that diagnose misalignment and support informed improvements beyond surface outcome matching. 

---
# Beyond Model Scaling: Test-Time Intervention for Efficient Deep Reasoning 

**Authors**: Qianyue Wang, Jinwu Hu, Yufeng Wang, Huanxiang Lin, Bolin Chen, Zhiquan Wen, Yaofo Chen, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.11252)  

**Abstract**: Large Reasoning Models (LRMs) excel at multi-step reasoning but often suffer from inefficient reasoning processes like overthinking and overshoot, where excessive or misdirected reasoning increases computational cost and degrades performance. Existing efficient reasoning methods operate in a closed-loop manner, lacking mechanisms for external intervention to guide the reasoning process. To address this, we propose Think-with-Me, a novel test-time interactive reasoning paradigm that introduces external feedback intervention into the reasoning process. Our key insights are that transitional conjunctions serve as natural points for intervention, signaling phases of self-validation or exploration and using transitional words appropriately to prolong the reasoning enhances performance, while excessive use affects performance. Building on these insights, Think-with-Me pauses reasoning at these points for external feedback, adaptively extending or terminating reasoning to reduce redundancy while preserving accuracy. The feedback is generated via a multi-criteria evaluation (rationality and completeness) and comes from either human or LLM proxies. We train the target model using Group Relative Policy Optimization (GRPO) to adapt to this interactive mode. Experiments show that Think-with-Me achieves a superior balance between accuracy and reasoning length under limited context windows. On AIME24, Think-with-Me outperforms QwQ-32B by 7.19% in accuracy while reducing average reasoning length by 81% under an 8K window. The paradigm also benefits security and creative tasks. 

---
# Policy-Based Deep Reinforcement Learning Hyperheuristics for Job-Shop Scheduling Problems 

**Authors**: Sofiene Lassoued, Asrat Gobachew, Stefan Lier, Andreas Schwung  

**Link**: [PDF](https://arxiv.org/pdf/2601.11189)  

**Abstract**: This paper proposes a policy-based deep reinforcement learning hyper-heuristic framework for solving the Job Shop Scheduling Problem. The hyper-heuristic agent learns to switch scheduling rules based on the system state dynamically. We extend the hyper-heuristic framework with two key mechanisms. First, action prefiltering restricts decision-making to feasible low-level actions, enabling low-level heuristics to be evaluated independently of environmental constraints and providing an unbiased assessment. Second, a commitment mechanism regulates the frequency of heuristic switching. We investigate the impact of different commitment strategies, from step-wise switching to full-episode commitment, on both training behavior and makespan. Additionally, we compare two action selection strategies at the policy level: deterministic greedy selection and stochastic sampling. Computational experiments on standard JSSP benchmarks demonstrate that the proposed approach outperforms traditional heuristics, metaheuristics, and recent neural network-based scheduling methods 

---
# TANDEM: Temporal-Aware Neural Detection for Multimodal Hate Speech 

**Authors**: Girish A. Koushik, Helen Treharne, Diptesh Kanojia  

**Link**: [PDF](https://arxiv.org/pdf/2601.11178)  

**Abstract**: Social media platforms are increasingly dominated by long-form multimodal content, where harmful narratives are constructed through a complex interplay of audio, visual, and textual cues. While automated systems can flag hate speech with high accuracy, they often function as "black boxes" that fail to provide the granular, interpretable evidence, such as precise timestamps and target identities, required for effective human-in-the-loop moderation. In this work, we introduce TANDEM, a unified framework that transforms audio-visual hate detection from a binary classification task into a structured reasoning problem. Our approach employs a novel tandem reinforcement learning strategy where vision-language and audio-language models optimize each other through self-constrained cross-modal context, stabilizing reasoning over extended temporal sequences without requiring dense frame-level supervision. Experiments across three benchmark datasets demonstrate that TANDEM significantly outperforms zero-shot and context-augmented baselines, achieving 0.73 F1 in target identification on HateMM (a 30% improvement over state-of-the-art) while maintaining precise temporal grounding. We further observe that while binary detection is robust, differentiating between offensive and hateful content remains challenging in multi-class settings due to inherent label ambiguity and dataset imbalance. More broadly, our findings suggest that structured, interpretable alignment is achievable even in complex multimodal settings, offering a blueprint for the next generation of transparent and actionable online safety moderation tools. 

---
# Do We Always Need Query-Level Workflows? Rethinking Agentic Workflow Generation for Multi-Agent Systems 

**Authors**: Zixu Wang, Bingbing Xu, Yige Yuan, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11147)  

**Abstract**: Multi-Agent Systems (MAS) built on large language models typically solve complex tasks by coordinating multiple agents through workflows. Existing approaches generates workflows either at task level or query level, but their relative costs and benefits remain unclear. After rethinking and empirical analyses, we show that query-level workflow generation is not always necessary, since a small set of top-K best task-level workflows together already covers equivalent or even more queries. We further find that exhaustive execution-based task-level evaluation is both extremely token-costly and frequently unreliable. Inspired by the idea of self-evolution and generative reward modeling, we propose a low-cost task-level generation framework \textbf{SCALE}, which means \underline{\textbf{S}}elf prediction of the optimizer with few shot \underline{\textbf{CAL}}ibration for \underline{\textbf{E}}valuation instead of full validation execution. Extensive experiments demonstrate that \textbf{SCALE} maintains competitive performance, with an average degradation of just 0.61\% compared to existing approach across multiple datasets, while cutting overall token usage by up to 83\%. 

---
# ReCreate: Reasoning and Creating Domain Agents Driven by Experience 

**Authors**: Zhezheng Hao, Hong Wang, Jian Luo, Jianqing Zhang, Yuyan Zhou, Qiang Lin, Can Wang, Hande Dong, Jiawei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11100)  

**Abstract**: Large Language Model agents are reshaping the industrial landscape. However, most practical agents remain human-designed because tasks differ widely, making them labor-intensive to build. This situation poses a central question: can we automatically create and adapt domain agents in the wild? While several recent approaches have sought to automate agent creation, they typically treat agent generation as a black-box procedure and rely solely on final performance metrics to guide the process. Such strategies overlook critical evidence explaining why an agent succeeds or fails, and often require high computational costs. To address these limitations, we propose ReCreate, an experience-driven framework for the automatic creation of domain agents. ReCreate systematically leverages agent interaction histories, which provide rich concrete signals on both the causes of success or failure and the avenues for improvement. Specifically, we introduce an agent-as-optimizer paradigm that effectively learns from experience via three key components: (i) an experience storage and retrieval mechanism for on-demand inspection; (ii) a reasoning-creating synergy pipeline that maps execution experience into scaffold edits; and (iii) hierarchical updates that abstract instance-level details into reusable domain patterns. In experiments across diverse domains, ReCreate consistently outperforms human-designed agents and existing automated agent generation methods, even when starting from minimal seed scaffolds. 

---
# MiCA: A Mobility-Informed Causal Adapter for Lightweight Epidemic Forecasting 

**Authors**: Suhan Guo, Jiahong Deng, Furao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11089)  

**Abstract**: Accurate forecasting of infectious disease dynamics is critical for public health planning and intervention. Human mobility plays a central role in shaping the spatial spread of epidemics, but mobility data are noisy, indirect, and difficult to integrate reliably with disease records. Meanwhile, epidemic case time series are typically short and reported at coarse temporal resolution. These conditions limit the effectiveness of parameter-heavy mobility-aware forecasters that rely on clean and abundant data. In this work, we propose the Mobility-Informed Causal Adapter (MiCA), a lightweight and architecture-agnostic module for epidemic forecasting. MiCA infers mobility relations through causal discovery and integrates them into temporal forecasting models via gated residual mixing. This design allows lightweight forecasters to selectively exploit mobility-derived spatial structure while remaining robust under noisy and data-limited conditions, without introducing heavy relational components such as graph neural networks or full attention. Extensive experiments on four real-world epidemic datasets, including COVID-19 incidence, COVID-19 mortality, influenza, and dengue, show that MiCA consistently improves lightweight temporal backbones, achieving an average relative error reduction of 7.5\% across forecasting horizons. Moreover, MiCA attains performance competitive with SOTA spatio-temporal models while remaining lightweight. 

---
# AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts 

**Authors**: Keyu Li, Junhao Shi, Yang Xiao, Mohan Jiang, Jie Sun, Yunze Wu, Shijie Xia, Xiaojie Cai, Tianze Xu, Weiye Si, Wenjie Li, Dequan Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11044)  

**Abstract**: Large Language Models (LLMs) based autonomous agents demonstrate multifaceted capabilities to contribute substantially to economic production. However, existing benchmarks remain focused on single agentic capability, failing to capture long-horizon real-world scenarios. Moreover, the reliance on human-in-the-loop feedback for realistic tasks creates a scalability bottleneck, hindering automated rollout collection and evaluation. To bridge this gap, we introduce AgencyBench, a comprehensive benchmark derived from daily AI usage, evaluating 6 core agentic capabilities across 32 real-world scenarios, comprising 138 tasks with specific queries, deliverables, and rubrics. These scenarios require an average of 90 tool calls, 1 million tokens, and hours of execution time to resolve. To enable automated evaluation, we employ a user simulation agent to provide iterative feedback, and a Docker sandbox to conduct visual and functional rubric-based assessment. Experiments reveal that closed-source models significantly outperform open-source models (48.4% vs 32.1%). Further analysis reveals significant disparities across models in resource efficiency, feedback-driven self-correction, and specific tool-use preferences. Finally, we investigate the impact of agentic scaffolds, observing that proprietary models demonstrate superior performance within their native ecosystems (e.g., Claude-4.5-Opus via Claude-Agent-SDK), while open-source models exhibit distinct performance peaks, suggesting potential optimization for specific execution frameworks. AgencyBench serves as a critical testbed for next-generation agents, highlighting the necessity of co-optimizing model architecture with agentic frameworks. We believe this work sheds light on the future direction of autonomous agents, and we release the full benchmark and evaluation toolkit at this https URL. 

---
# BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search 

**Authors**: Shiyu Liu, Yongjing Yin, Jianhao Yan, Yunbo Tang, Qinggang Zhang, Bei Li, Xin Chen, Jingang Wang, Xunliang Cai, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.11037)  

**Abstract**: RL-based agentic search enables LLMs to solve complex questions via dynamic planning and external search. While this approach significantly enhances accuracy with agent policies optimized via large-scale reinforcement learning, we identify a critical gap in reliability: these agents fail to recognize their reasoning boundaries and rarely admit ``I DON'T KNOW'' (IDK) even when evidence is insufficient or reasoning reaches its limit. The lack of reliability often leads to plausible but unreliable answers, introducing significant risks in many real-world scenarios. To this end, we propose Boundary-Aware Policy Optimization (BAPO), a novel RL framework designed to cultivate reliable boundary awareness without compromising accuracy. BAPO introduces two key components: (i) a group-based boundary-aware reward that encourages an IDK response only when the reasoning reaches its limit, and (ii) an adaptive reward modulator that strategically suspends this reward during early exploration, preventing the model from exploiting IDK as a shortcut. Extensive experiments on four benchmarks demonstrate that BAPO substantially enhances the overall reliability of agentic search. 

---
# Efficient Protein Optimization via Structure-aware Hamiltonian Dynamics 

**Authors**: Jiahao Wang, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11012)  

**Abstract**: The ability to engineer optimized protein variants has transformative potential for biotechnology and medicine. Prior sequence-based optimization methods struggle with the high-dimensional complexities due to the epistasis effect and the disregard for structural constraints. To address this, we propose HADES, a Bayesian optimization method utilizing Hamiltonian dynamics to efficiently sample from a structure-aware approximated posterior. Leveraging momentum and uncertainty in the simulated physical movements, HADES enables rapid transition of proposals toward promising areas. A position discretization procedure is introduced to propose discrete protein sequences from such a continuous state system. The posterior surrogate is powered by a two-stage encoder-decoder framework to determine the structure and function relationships between mutant neighbors, consequently learning a smoothed landscape to sample from. Extensive experiments demonstrate that our method outperforms state-of-the-art baselines in in-silico evaluations across most metrics. Remarkably, our approach offers a unique advantage by leveraging the mutual constraints between protein structure and sequence, facilitating the design of protein sequences with similar structures and optimized properties. The code and data are publicly available at this https URL. 

---
# AdaMARP: An Adaptive Multi-Agent Interaction Framework for General Immersive Role-Playing 

**Authors**: Zhenhua Xu, Dongsheng Chen, Shuo Wang, Jian Li, Chengjie Wang, Meng Han, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11007)  

**Abstract**: LLM role-playing aims to portray arbitrary characters in interactive narratives, yet existing systems often suffer from limited immersion and adaptability. They typically under-model dynamic environmental information and assume largely static scenes and casts, offering insufficient support for multi-character orchestration, scene transitions, and on-the-fly character introduction. We propose an adaptive multi-agent role-playing framework, AdaMARP, featuring an immersive message format that interleaves [Thought], (Action), <Environment>, and Speech, together with an explicit Scene Manager that governs role-playing through discrete actions (init_scene, pick_speaker, switch_scene, add_role, end) accompanied by rationales. To train these capabilities, we construct AdaRPSet for the Actor Model and AdaSMSet for supervising orchestration decisions, and introduce AdaptiveBench for trajectory-level evaluation. Experiments across multiple backbones and model scales demonstrate consistent improvements: AdaRPSet enhances character consistency, environment grounding, and narrative coherence, with an 8B actor outperforming several commercial LLMs, while AdaSMSet enables smoother scene transitions and more natural role introductions, surpassing Claude Sonnet 4.5 using only a 14B LLM. 

---
# What Matters in Data Curation for Multimodal Reasoning? Insights from the DCVLR Challenge 

**Authors**: Yosub Shin, Michael Buriek, Boris Sobolev, Pavel Bushuyeu, Vikas Kumar, Haoyang Xu, Samuel Watson, Igor Molybog  

**Link**: [PDF](https://arxiv.org/pdf/2601.10922)  

**Abstract**: We study data curation for multimodal reasoning through the NeurIPS 2025 Data Curation for Vision-Language Reasoning (DCVLR) challenge, which isolates dataset selection by fixing the model and training protocol. Using a compact curated dataset derived primarily from Walton Multimodal Cold Start, our submission placed first in the challenge. Through post-competition ablations, we show that difficulty-based example selection on an aligned base dataset is the dominant driver of performance gains. Increasing dataset size does not reliably improve mean accuracy under the fixed training recipe, but mainly reduces run-to-run variance, while commonly used diversity and synthetic augmentation heuristics provide no additional benefit and often degrade performance. These results characterize DCVLR as a saturation-regime evaluation and highlight the central role of alignment and difficulty in data-efficient multimodal reasoning. 

---
# ARC Prize 2025: Technical Report 

**Authors**: François Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers  

**Link**: [PDF](https://arxiv.org/pdf/2601.10904)  

**Abstract**: The ARC-AGI benchmark series serves as a critical measure of few-shot generalization on novel tasks, a core aspect of intelligence. The ARC Prize 2025 global competition targeted the newly released ARC-AGI-2 dataset, which features greater task complexity compared to its predecessor. The Kaggle competition attracted 1,455 teams and 15,154 entries, with the top score reaching 24% on the ARC-AGI-2 private evaluation set. Paper submissions nearly doubled year-over-year to 90 entries, reflecting the growing research interest in fluid intelligence and abstract reasoning. The defining theme of 2025 is the emergence of the refinement loop -- a per-task iterative program optimization loop guided by a feedback signal. Refinement loops come in a variety of forms, in particular evolutionary program synthesis approaches and application-layer refinements to commercial AI systems. Such refinement loops are also possible in weight space, as evidenced by zero-pretraining deep learning methods which are now achieving competitive performance with remarkably small networks (7M parameters). In parallel, four frontier AI labs (Anthropic, Google DeepMind, OpenAI, and xAI) reported ARC-AGI performance in public model cards in 2025, establishing ARC-AGI as an industry standard benchmark for AI reasoning. However, our analysis indicates that current frontier AI reasoning performance remains fundamentally constrained to knowledge coverage, giving rise to new forms of benchmark contamination. In this paper, we survey the top-performing methods, examine the role of refinement loops in AGI progress, discuss knowledge-dependent overfitting, and preview ARC-AGI-3, which introduces interactive reasoning challenges that require exploration, planning, memory, goal acquisition, and alignment capabilities. 

---
# Optimisation of complex product innovation processes based on trend models with three-valued logic 

**Authors**: Nina Bočková, Barbora Volná, Mirko Dohnal  

**Link**: [PDF](https://arxiv.org/pdf/2601.10768)  

**Abstract**: This paper investigates complex product-innovation processes using models grounded in a set of heuristics. Each heuristic is expressed through simple trends -- increasing, decreasing, or constant -- which serve as minimally information-intensive quantifiers, avoiding reliance on numerical values or rough sets. A solution to a trend model is defined as a set of scenarios with possible transitions between them, represented by a transition graph. Any possible future or past behaviour of the system under study can thus be depicted by a path within this graph. 

---
# Explore with Long-term Memory: A Benchmark and Multimodal LLM-based Reinforcement Learning Framework for Embodied Exploration 

**Authors**: Sen Wang, Bangwei Liu, Zhenkun Gao, Lizhuang Ma, Xuhong Wang, Yuan Xie, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.10744)  

**Abstract**: An ideal embodied agent should possess lifelong learning capabilities to handle long-horizon and complex tasks, enabling continuous operation in general environments. This not only requires the agent to accurately accomplish given tasks but also to leverage long-term episodic memory to optimize decision-making. However, existing mainstream one-shot embodied tasks primarily focus on task completion results, neglecting the crucial process of exploration and memory utilization. To address this, we propose Long-term Memory Embodied Exploration (LMEE), which aims to unify the agent's exploratory cognition and decision-making behaviors to promote lifelong this http URL further construct a corresponding dataset and benchmark, LMEE-Bench, incorporating multi-goal navigation and memory-based question answering to comprehensively evaluate both the process and outcome of embodied exploration. To enhance the agent's memory recall and proactive exploration capabilities, we propose MemoryExplorer, a novel method that fine-tunes a multimodal large language model through reinforcement learning to encourage active memory querying. By incorporating a multi-task reward function that includes action prediction, frontier selection, and question answering, our model achieves proactive exploration. Extensive experiments against state-of-the-art embodied exploration models demonstrate that our approach achieves significant advantages in long-horizon embodied tasks. 

---
# CTHA: Constrained Temporal Hierarchical Architecture for Stable Multi-Agent LLM Systems 

**Authors**: Percy Jardine  

**Link**: [PDF](https://arxiv.org/pdf/2601.10738)  

**Abstract**: Recently, multi-time-scale agent architectures have extended the ubiquitous single-loop paradigm by introducing temporal hierarchies with distinct cognitive layers. While yielding substantial performance gains, this diversification fundamentally compromises the coordination stability intrinsic to unified agent systems, which causes severe inter-layer conflicts, unbounded error propagation, and restricted scalability. To address these challenges, we propose Constrained Temporal Hierarchical Architecture (CTHA), a general framework that projects the inter-layer communication space onto structured manifolds to restore coordination stability, while incorporating principled arbitration mechanisms to ensure coherent decision-making. Specifically, CTHA enforces three key constraints: (1) Message Contract Constraints that formalize information flow between layers via typed summary, plan, and policy packets; (2) Authority Manifold Constraints that bound each layer's decision space according to its temporal scope; and (3) Arbiter Resolution Constraints that guarantee conflict-free composition of multi-layer decisions. Empirical experiments demonstrate that CTHA is effective for complex task execution at scale, offering 47% reduction in failure cascades, 2.3x improvement in sample efficiency, and superior scalability compared to unconstrained hierarchical baselines. We anticipate that CTHA, as a principled extension of temporal hierarchies, will contribute to a deeper understanding of multi-agent coordination and suggest promising directions for the evolution of robust autonomous systems. 

---
# ORBITFLOW: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration 

**Authors**: Xinyue Ma, Heelim Hong, Taegeon Um, Jongseop Lee, Seoyeong Choy, Woo-Yeon Lee, Myeongjae Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2601.10729)  

**Abstract**: Serving long-context LLMs is challenging because request lengths and batch composition vary during token generation, causing the memory footprint to fluctuate significantly at runtime. Offloading KV caches to host memory limits effective memory usage, but existing static and predetermined offloading strategies cannot adapt to the rapidly shifting memory demands of long-context serving. This often leads to excessive CPU-to-GPU KV transfers that translate into latency spikes and frequent SLO violations. To address these challenges, we introduce ORBITFLOW, a fine-grained and adaptive KV cache management system that meets latency SLOs in long-context LLM serving. ORBITFLOW employs a lightweight ILP solver to decide which layers' KV caches to retain on the GPU for each request, within memory capacity constraints. It continuously refines KV placements based on runtime feedback when the active plan becomes suboptimal during token generation. Under heavy load, ORBITFLOW invokes a fallback mechanism to temporarily defer in-flight requests with large memory footprints, preserving overall SLO attainment. Our experiments demonstrate that ORBITFLOW improves SLO attainment for TPOT and TBT by up to 66% and 48%, respectively, while reducing the 95th percentile latency by 38% and achieving up to 3.3x higher throughput compared to existing offloading methods. 

---
# Building AI Agents to Improve Job Referral Requests to Strangers 

**Authors**: Ross Chu, Yuting Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10726)  

**Abstract**: This paper develops AI agents that help job seekers write effective requests for job referrals in a professional online community. The basic workflow consists of an improver agent that rewrites the referral request and an evaluator agent that measures the quality of revisions using a model trained to predict the probability of receiving referrals from other users. Revisions suggested by the LLM (large language model) increase predicted success rates for weaker requests while reducing them for stronger requests. Enhancing the LLM with Retrieval-Augmented Generation (RAG) prevents edits that worsen stronger requests while it amplifies improvements for weaker requests. Overall, using LLM revisions with RAG increases the predicted success rate for weaker requests by 14\% without degrading performance on stronger requests. Although improvements in model-predicted success do not guarantee more referrals in the real world, they provide low-cost signals for promising features before running higher-stakes experiments on real users. 

---
# Do You Trust Me? Cognitive-Affective Signatures of Trustworthiness in Large Language Models 

**Authors**: Gerard Yeo, Svetlana Churina, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2601.10719)  

**Abstract**: Perceived trustworthiness underpins how users navigate online information, yet it remains unclear whether large language models (LLMs),increasingly embedded in search, recommendation, and conversational systems, represent this construct in psychologically coherent ways. We analyze how instruction-tuned LLMs (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) encode perceived trustworthiness in web-like narratives using the PEACE-Reviews dataset annotated for cognitive appraisals, emotions, and behavioral intentions. Across models, systematic layer- and head-level activation differences distinguish high- from low-trust texts, revealing that trust cues are implicitly encoded during pretraining. Probing analyses show linearly de-codable trust signals and fine-tuning effects that refine rather than restructure these representations. Strongest associations emerge with appraisals of fairness, certainty, and accountability-self -- dimensions central to human trust formation online. These findings demonstrate that modern LLMs internalize psychologically grounded trust signals without explicit supervision, offering a representational foundation for designing credible, transparent, and trust-worthy AI systems in the web ecosystem. Code and appendix are available at: this https URL. 

---
# Japanese AI Agent System on Human Papillomavirus Vaccination: System Design 

**Authors**: Junyu Liu, Siwen Yang, Dexiu Ma, Qian Niu, Zequn Zhang, Momoko Nagai-Tanima, Tomoki Aoyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.10718)  

**Abstract**: Human papillomavirus (HPV) vaccine hesitancy poses significant public health challenges, particularly in Japan where proactive vaccination recommendations were suspended from 2013 to 2021. The resulting information gap is exacerbated by misinformation on social media, and traditional ways cannot simultaneously address individual queries while monitoring population-level discourse. This study aimed to develop a dual-purpose AI agent system that provides verified HPV vaccine information through a conversational interface while generating analytical reports for medical institutions based on user interactions and social media. We implemented a system comprising: a vector database integrating academic papers, government sources, news media, and social media; a Retrieval-Augmented Generation chatbot using ReAct agent architecture with multi-tool orchestration across five knowledge sources; and an automated report generation system with modules for news analysis, research synthesis, social media sentiment analysis, and user interaction pattern identification. Performance was assessed using a 0-5 scoring scale. For single-turn evaluation, the chatbot achieved mean scores of 4.83 for relevance, 4.89 for routing, 4.50 for reference quality, 4.90 for correctness, and 4.88 for professional identity (overall 4.80). Multi-turn evaluation yielded higher scores: context retention 4.94, topic coherence 5.00, and overall 4.98. The report generation system achieved completeness 4.00-5.00, correctness 4.00-5.00, and helpfulness 3.67-5.00, with reference validity 5.00 across all periods. This study demonstrates the feasibility of an integrated AI agent system for bidirectional HPV vaccine communication. The architecture enables verified information delivery with source attribution while providing systematic public discourse analysis, with a transferable framework for adaptation to other medical contexts. 

---
# Do explanations generalize across large reasoning models? 

**Authors**: Koyena Pal, David Bau, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.11517)  

**Abstract**: Large reasoning models (LRMs) produce a textual chain of thought (CoT) in the process of solving a problem, which serves as a potentially powerful tool to understand the problem by surfacing a human-readable, natural-language explanation. However, it is unclear whether these explanations generalize, i.e. whether they capture general patterns about the underlying problem rather than patterns which are esoteric to the LRM. This is a crucial question in understanding or discovering new concepts, e.g. in AI for science. We study this generalization question by evaluating a specific notion of generalizability: whether explanations produced by one LRM induce the same behavior when given to other LRMs. We find that CoT explanations often exhibit this form of generalization (i.e. they increase consistency between LRMs) and that this increased generalization is correlated with human preference rankings and post-training with reinforcement learning. We further analyze the conditions under which explanations yield consistent answers and propose a straightforward, sentence-level ensembling strategy that improves consistency. Taken together, these results prescribe caution when using LRM explanations to yield new insights and outline a framework for characterizing LRM explanation generalization. 

---
# Building Production-Ready Probes For Gemini 

**Authors**: János Kramár, Joshua Engels, Zheng Wang, Bilal Chughtai, Rohin Shah, Neel Nanda, Arthur Conmy  

**Link**: [PDF](https://arxiv.org/pdf/2601.11516)  

**Abstract**: Frontier language model capabilities are improving rapidly. We thus need stronger mitigations against bad actors misusing increasingly powerful systems. Prior work has shown that activation probes may be a promising misuse mitigation technique, but we identify a key remaining challenge: probes fail to generalize under important production distribution shifts. In particular, we find that the shift from short-context to long-context inputs is difficult for existing probe architectures. We propose several new probe architecture that handle this long-context distribution shift.
We evaluate these probes in the cyber-offensive domain, testing their robustness against various production-relevant shifts, including multi-turn conversations, static jailbreaks, and adaptive red teaming. Our results demonstrate that while multimax addresses context length, a combination of architecture choice and training on diverse distributions is required for broad generalization. Additionally, we show that pairing probes with prompted classifiers achieves optimal accuracy at a low cost due to the computational efficiency of probes.
These findings have informed the successful deployment of misuse mitigation probes in user-facing instances of Gemini, Google's frontier language model. Finally, we find early positive results using AlphaEvolve to automate improvements in both probe architecture search and adaptive red teaming, showing that automating some AI safety research is already possible. 

---
# MetaboNet: The Largest Publicly Available Consolidated Dataset for Type 1 Diabetes Management 

**Authors**: Miriam K. Wolff, Peter Calhoun, Eleonora Maria Aiello, Yao Qin, Sam F. Royston  

**Link**: [PDF](https://arxiv.org/pdf/2601.11505)  

**Abstract**: Progress in Type 1 Diabetes (T1D) algorithm development is limited by the fragmentation and lack of standardization across existing T1D management datasets. Current datasets differ substantially in structure and are time-consuming to access and process, which impedes data integration and reduces the comparability and generalizability of algorithmic developments. This work aims to establish a unified and accessible data resource for T1D algorithm development. Multiple publicly available T1D datasets were consolidated into a unified resource, termed the MetaboNet dataset. Inclusion required the availability of both continuous glucose monitoring (CGM) data and corresponding insulin pump dosing records. Additionally, auxiliary information such as reported carbohydrate intake and physical activity was retained when present. The MetaboNet dataset comprises 3135 subjects and 1228 patient-years of overlapping CGM and insulin data, making it substantially larger than existing standalone benchmark datasets. The resource is distributed as a fully public subset available for immediate download at this https URL , and with a Data Use Agreement (DUA)-restricted subset accessible through their respective application processes. For the datasets in the latter subset, processing pipelines are provided to automatically convert the data into the standardized MetaboNet format. A consolidated public dataset for T1D research is presented, and the access pathways for both its unrestricted and DUA-governed components are described. The resulting dataset covers a broad range of glycemic profiles and demographics and thus can yield more generalizable algorithmic performance than individual datasets. 

---
# The Poisoned Apple Effect: Strategic Manipulation of Mediated Markets via Technology Expansion of AI Agents 

**Authors**: Eilam Shapira, Roi Reichart, Moshe Tennenholtz  

**Link**: [PDF](https://arxiv.org/pdf/2601.11496)  

**Abstract**: The integration of AI agents into economic markets fundamentally alters the landscape of strategic interaction. We investigate the economic implications of expanding the set of available technologies in three canonical game-theoretic settings: bargaining (resource division), negotiation (asymmetric information trade), and persuasion (strategic information transmission). We find that simply increasing the choice of AI delegates can drastically shift equilibrium payoffs and regulatory outcomes, often creating incentives for regulators to proactively develop and release technologies. Conversely, we identify a strategic phenomenon termed the "Poisoned Apple" effect: an agent may release a new technology, which neither they nor their opponent ultimately uses, solely to manipulate the regulator's choice of market design in their favor. This strategic release improves the releaser's welfare at the expense of their opponent and the regulator's fairness objectives. Our findings demonstrate that static regulatory frameworks are vulnerable to manipulation via technology expansion, necessitating dynamic market designs that adapt to the evolving landscape of AI capabilities. 

---
# MHA2MLA-VLM: Enabling DeepSeek's Economical Multi-Head Latent Attention across Vision-Language Models 

**Authors**: Xiaoran Fan, Zhichao Sun, Tao Ji, Lixing Shen, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2601.11464)  

**Abstract**: As vision-language models (VLMs) tackle increasingly complex and multimodal tasks, the rapid growth of Key-Value (KV) cache imposes significant memory and computational bottlenecks during inference. While Multi-Head Latent Attention (MLA) offers an effective means to compress the KV cache and accelerate inference, adapting existing VLMs to the MLA architecture without costly pretraining remains largely unexplored. In this work, we present MHA2MLA-VLM, a parameter-efficient and multimodal-aware framework for converting off-the-shelf VLMs to MLA. Our approach features two core techniques: (1) a modality-adaptive partial-RoPE strategy that supports both traditional and multimodal settings by selectively masking nonessential dimensions, and (2) a modality-decoupled low-rank approximation method that independently compresses the visual and textual KV spaces. Furthermore, we introduce parameter-efficient fine-tuning to minimize adaptation cost and demonstrate that minimizing output activation error, rather than parameter distance, substantially reduces performance loss. Extensive experiments on three representative VLMs show that MHA2MLA-VLM restores original model performance with minimal supervised data, significantly reduces KV cache footprint, and integrates seamlessly with KV quantization. 

---
# Interactive Narrative Analytics: Bridging Computational Narrative Extraction and Human Sensemaking 

**Authors**: Brian Keith  

**Link**: [PDF](https://arxiv.org/pdf/2601.11459)  

**Abstract**: Information overload and misinformation create significant challenges in extracting meaningful narratives from large news collections. This paper defines the nascent field of Interactive Narrative Analytics (INA), which combines computational narrative extraction with interactive visual analytics to support sensemaking. INA approaches enable the interactive exploration of narrative structures through computational methods and visual interfaces that facilitate human interpretation. The field faces challenges in scalability, interactivity, knowledge integration, and evaluation standardization, yet offers promising opportunities across news analysis, intelligence, scientific literature exploration, and social media analysis. Through the combination of computational and human insight, INA addresses complex challenges in narrative sensemaking. 

---
# PRISM-CAFO: Prior-conditioned Remote-sensing Infrastructure Segmentation and Mapping for CAFOs 

**Authors**: Oishee Bintey Hoque, Nibir Chandra Mandal, Kyle Luong, Amanda Wilson, Samarth Swarup, Madhav Marathe, Abhijin Adiga  

**Link**: [PDF](https://arxiv.org/pdf/2601.11451)  

**Abstract**: Large-scale livestock operations pose significant risks to human health and the environment, while also being vulnerable to threats such as infectious diseases and extreme weather events. As the number of such operations continues to grow, accurate and scalable mapping has become increasingly important. In this work, we present an infrastructure-first, explainable pipeline for identifying and characterizing Concentrated Animal Feeding Operations (CAFOs) from aerial and satellite imagery. Our method (1) detects candidate infrastructure (e.g., barns, feedlots, manure lagoons, silos) with a domain-tuned YOLOv8 detector, then derives SAM2 masks from these boxes and filters component-specific criteria, (2) extracts structured descriptors (e.g., counts, areas, orientations, and spatial relations) and fuses them with deep visual features using a lightweight spatial cross-attention classifier, and (3) outputs both CAFO type predictions and mask-level attributions that link decisions to visible infrastructure. Through comprehensive evaluation, we show that our approach achieves state-of-the-art performance, with Swin-B+PRISM-CAFO surpassing the best performing baseline by up to 15\%. Beyond strong predictive performance across diverse U.S. regions, we run systematic gradient--activation analyses that quantify the impact of domain priors and show ho 

---
# Map2Thought: Explicit 3D Spatial Reasoning via Metric Cognitive Maps 

**Authors**: Xiangjun Gao, Zhensong Zhang, Dave Zhenyu Chen, Songcen Xu, Long Quan, Eduardo Pérez-Pellitero, Youngkyoon Jang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11442)  

**Abstract**: We propose Map2Thought, a framework that enables explicit and interpretable spatial reasoning for 3D VLMs. The framework is grounded in two key components: Metric Cognitive Map (Metric-CogMap) and Cognitive Chain-of-Thought (Cog-CoT). Metric-CogMap provides a unified spatial representation by integrating a discrete grid for relational reasoning with a continuous, metric-scale representation for precise geometric understanding. Building upon the Metric-CogMap, Cog-CoT performs explicit geometric reasoning through deterministic operations, including vector operations, bounding-box distances, and occlusion-aware appearance order cues, producing interpretable inference traces grounded in 3D structure. Experimental results show that Map2Thought enables explainable 3D understanding, achieving 59.9% accuracy using only half the supervision, closely matching the 60.9% baseline trained with the full dataset. It consistently outperforms state-of-the-art methods by 5.3%, 4.8%, and 4.0% under 10%, 25%, and 50% training subsets, respectively, on the VSI-Bench. 

---
# Hierarchical Orthogonal Residual Spread for Precise Massive Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Yuheng Yang, Jingxin Han, Andi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11441)  

**Abstract**: Large language models (LLMs) exhibit exceptional performance across various domains, yet they face critical safety concerns. Model editing has emerged as an effective approach to mitigate these issues. Existing model editing methods often focus on optimizing an information matrix that blends new and old knowledge. While effective, these approaches can be computationally expensive and may cause conflicts. In contrast, we shift our attention to Hierarchical Orthogonal Residual SprEad of the information matrix, which reduces noisy gradients and enables more stable edits from a different perspective. We demonstrate the effectiveness of our method HORSE through a clear theoretical comparison with several popular methods and extensive experiments conducted on two datasets across multiple LLMs. The results show that HORSE maintains precise massive editing across diverse scenarios. The code is available at this https URL 

---
# GenDA: Generative Data Assimilation on Complex Urban Areas via Classifier-Free Diffusion Guidance 

**Authors**: Francisco Giral, Álvaro Manzano, Ignacio Gómez, Ricardo Vinuesa, Soledad Le Clainche  

**Link**: [PDF](https://arxiv.org/pdf/2601.11440)  

**Abstract**: Urban wind flow reconstruction is essential for assessing air quality, heat dispersion, and pedestrian comfort, yet remains challenging when only sparse sensor data are available. We propose GenDA, a generative data assimilation framework that reconstructs high-resolution wind fields on unstructured meshes from limited observations. The model employs a multiscale graph-based diffusion architecture trained on computational fluid dynamics (CFD) simulations and interprets classifier-free guidance as a learned posterior reconstruction mechanism: the unconditional branch learns a geometry-aware flow prior, while the sensor-conditioned branch injects observational constraints during sampling. This formulation enables obstacle-aware reconstruction and generalization across unseen geometries, wind directions, and mesh resolutions without retraining. We consider both sparse fixed sensors and trajectory-based observations using the same reconstruction procedure. When evaluated against supervised graph neural network (GNN) baselines and classical reduced-order data assimilation methods, GenDA reduces the relative root-mean-square error (RRMSE) by 25-57% and increases the structural similarity index (SSIM) by 23-33% across the tested meshes. Experiments are conducted on Reynolds-averaged Navier-Stokes (RANS) simulations of a real urban neighbourhood in Bristol, United Kingdom, at a characteristic Reynolds number of $\mathrm{Re}\approx2\times10^{7}$, featuring complex building geometry and irregular terrain. The proposed framework provides a scalable path toward generative, geometry-aware data assimilation for environmental monitoring in complex domains. 

---
# Relational Linearity is a Predictor of Hallucinations 

**Authors**: Yuetian Lu, Yihong Liu, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2601.11429)  

**Abstract**: Hallucination is a central failure mode in large language models (LLMs). We focus on hallucinations of answers to questions like: "Which instrument did Glenn Gould play?", but we ask these questions for synthetic entities that are unknown to the model. Surprisingly, we find that medium-size models like Gemma-7B-IT frequently hallucinate, i.e., they have difficulty recognizing that the hallucinated fact is not part of their knowledge. We hypothesize that an important factor in causing these hallucinations is the linearity of the relation: linear relations tend to be stored more abstractly, making it difficult for the LLM to assess its knowledge; the facts of nonlinear relations tend to be stored more directly, making knowledge assessment easier. To investigate this hypothesis, we create SyntHal, a dataset of 6000 synthetic entities for six relations. In our experiments with four models, we determine, for each relation, the hallucination rate on SyntHal and also measure its linearity, using $\Delta\cos$. We find a strong correlation ($r \in [.78,.82]$) between relational linearity and hallucination rate, providing evidence for our hypothesis that the underlying storage of triples of a relation is a factor in how well a model can self-assess its knowledge. This finding has implications for how to manage hallucination behavior and suggests new research directions for improving the representation of factual knowledge in LLMs. 

---
# The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents 

**Authors**: Ziyu Wang, Chenyuan Liu, Yushun Xiang, Runhao Zhang, Qingbo Hao, Hongliang Lu, Houyu Chen, Zhizhong Feng, Kaiyue Zheng, Dehao Ye, Xianchao Zeng, Xinyu Zhou, Boran Wen, Jiaxin Li, Mingyu Zhang, Kecheng Zheng, Qian Zhu, Ran Cheng, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11421)  

**Abstract**: Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at this https URL. 

---
# Topology-Guaranteed Image Segmentation: Enforcing Connectivity, Genus, and Width Constraints 

**Authors**: Wenxiao Li, Xue-Cheng Tai, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11409)  

**Abstract**: Existing research highlights the crucial role of topological priors in image segmentation, particularly in preserving essential structures such as connectivity and genus. Accurately capturing these topological features often requires incorporating width-related information, including the thickness and length inherent to the image structures. However, traditional mathematical definitions of topological structures lack this dimensional width information, limiting methods like persistent homology from fully addressing practical segmentation needs. To overcome this limitation, we propose a novel mathematical framework that explicitly integrates width information into the characterization of topological structures. This method leverages persistent homology, complemented by smoothing concepts from partial differential equations (PDEs), to modify local extrema of upper-level sets. This approach enables the resulting topological structures to inherently capture width properties. We incorporate this enhanced topological description into variational image segmentation models. Using some proper loss functions, we are also able to design neural networks that can segment images with the required topological and width properties. Through variational constraints on the relevant topological energies, our approach successfully preserves essential topological invariants such as connectivity and genus counts, simultaneously ensuring that segmented structures retain critical width attributes, including line thickness and length. Numerical experiments demonstrate the effectiveness of our method, showcasing its capability to maintain topological fidelity while explicitly embedding width characteristics into segmented image structures. 

---
# Wetland mapping from sparse annotations with satellite image time series and temporal-aware segment anything model 

**Authors**: Shuai Yuan, Tianwu Lin, Shuang Chen, Yu Xia, Peng Qin, Xiangyu Liu, Xiaoqing Xu, Nan Xu, Hongsheng Zhang, Jie Wang, Peng Gong  

**Link**: [PDF](https://arxiv.org/pdf/2601.11400)  

**Abstract**: Accurate wetland mapping is essential for ecosystem monitoring, yet dense pixel-level annotation is prohibitively expensive and practical applications usually rely on sparse point labels, under which existing deep learning models perform poorly, while strong seasonal and inter-annual wetland dynamics further render single-date imagery inadequate and lead to significant mapping errors; although foundation models such as SAM show promising generalization from point prompts, they are inherently designed for static images and fail to model temporal information, resulting in fragmented masks in heterogeneous wetlands. To overcome these limitations, we propose WetSAM, a SAM-based framework that integrates satellite image time series for wetland mapping from sparse point supervision through a dual-branch design, where a temporally prompted branch extends SAM with hierarchical adapters and dynamic temporal aggregation to disentangle wetland characteristics from phenological variability, and a spatial branch employs a temporally constrained region-growing strategy to generate reliable dense pseudo-labels, while a bidirectional consistency regularization jointly optimizes both branches. Extensive experiments across eight global regions of approximately 5,000 km2 each demonstrate that WetSAM substantially outperforms state-of-the-art methods, achieving an average F1-score of 85.58%, and delivering accurate and structurally consistent wetland segmentation with minimal labeling effort, highlighting its strong generalization capability and potential for scalable, low-cost, high-resolution wetland mapping. 

---
# Evaluating LLM Behavior in Hiring: Implicit Weights, Fairness Across Groups, and Alignment with Human Preferences 

**Authors**: Morgane Hoffmann, Emma Jouffroy, Warren Jouanneau, Marc Palyart, Charles Pebereau  

**Link**: [PDF](https://arxiv.org/pdf/2601.11379)  

**Abstract**: General-purpose Large Language Models (LLMs) show significant potential in recruitment applications, where decisions require reasoning over unstructured text, balancing multiple criteria, and inferring fit and competence from indirect productivity signals. Yet, it is still uncertain how LLMs assign importance to each attribute and whether such assignments are in line with economic principles, recruiter preferences or broader societal norms. We propose a framework to evaluate an LLM's decision logic in recruitment, by drawing on established economic methodologies for analyzing human hiring behavior. We build synthetic datasets from real freelancer profiles and project descriptions from a major European online freelance marketplace and apply a full factorial design to estimate how a LLM weighs different match-relevant criteria when evaluating freelancer-project fit. We identify which attributes the LLM prioritizes and analyze how these weights vary across project contexts and demographic subgroups. Finally, we explain how a comparable experimental setup could be implemented with human recruiters to assess alignment between model and human decisions. Our findings reveal that the LLM weighs core productivity signals, such as skills and experience, but interprets certain features beyond their explicit matching value. While showing minimal average discrimination against minority groups, intersectional effects reveal that productivity signals carry different weights between demographic groups. 

---
# Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs 

**Authors**: Marcantonio Bracale Syrnikov, Federico Pierucci, Marcello Galisai, Matteo Prandi, Piercosma Bisconti, Francesco Giarrusso, Olga Sorokoletova, Vincenzo Suriani, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.11369)  

**Abstract**: Multi-agent LLM ensembles can converge on coordinated, socially harmful equilibria. This paper advances an experimental framework for evaluating Institutional AI, our system-level approach to AI alignment that reframes alignment from preference engineering in agent-space to mechanism design in institution-space. Central to this approach is the governance graph, a public, immutable manifest that declares legal states, transitions, sanctions, and restorative paths; an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance. We apply the Institutional AI framework to govern the Cournot collusion case documented by prior work and compare three regimes: Ungoverned (baseline incentives from the structure of the Cournot market), Constitutional (a prompt-only policy-as-prompt prohibition implemented as a fixed written anti-collusion constitution, and Institutional (governance-graph-based). Across six model configurations including cross-provider pairs (N=90 runs/condition), the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure. These results suggest that multi-agent alignment may benefit from being framed as an institutional design problem, where governance graphs can provide a tractable abstraction for alignment-relevant collective behavior. 

---
# Think-Clip-Sample: Slow-Fast Frame Selection for Video Understanding 

**Authors**: Wenhui Tan, Ruihua Song, Jiaze Li, Jianzhong Ju, Zhenbo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2601.11359)  

**Abstract**: Recent progress in multi-modal large language models (MLLMs) has significantly advanced video understanding. However, their performance on long-form videos remains limited by computational constraints and suboptimal frame selection. We present Think-Clip-Sample (TCS), a training-free framework that enhances long video understanding through two key components: (i) Multi-Query Reasoning, which generates multiple queries to capture complementary aspects of the question and video; and (ii) Clip-level Slow-Fast Sampling, which adaptively balances dense local details and sparse global context. Extensive experiments on MLVU, LongVideoBench, and VideoMME demonstrate that TCS consistently improves performance across different MLLMs, boosting up to 6.9% accuracy, and is capable of achieving comparable accuracy with 50% fewer inference time cost, highlighting both efficiency and efficacy of TCS on long video understanding. 

---
# FEATHer: Fourier-Efficient Adaptive Temporal Hierarchy Forecaster for Time-Series Forecasting 

**Authors**: Jaehoon Lee, Seungwoo Lee, Younghwi Kim, Dohee Kim, Sunghyun Sim  

**Link**: [PDF](https://arxiv.org/pdf/2601.11350)  

**Abstract**: Time-series forecasting is fundamental in industrial domains like manufacturing and smart factories. As systems evolve toward automation, models must operate on edge devices (e.g., PLCs, microcontrollers) with strict constraints on latency and memory, limiting parameters to a few thousand. Conventional deep architectures are often impractical here. We propose the Fourier-Efficient Adaptive Temporal Hierarchy Forecaster (FEATHer) for accurate long-term forecasting under severe limits. FEATHer introduces: (i) ultra-lightweight multiscale decomposition into frequency pathways; (ii) a shared Dense Temporal Kernel using projection-depthwise convolution-projection without recurrence or attention; (iii) frequency-aware branch gating that adaptively fuses representations based on spectral characteristics; and (iv) a Sparse Period Kernel reconstructing outputs via period-wise downsampling to capture seasonality. FEATHer maintains a compact architecture (as few as 400 parameters) while outperforming baselines. Across eight benchmarks, it achieves the best ranking, recording 60 first-place results with an average rank of 2.05. These results demonstrate that reliable long-range forecasting is achievable on constrained edge hardware, offering a practical direction for industrial real-time inference. 

---
# How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting 

**Authors**: Parker Seegmiller, Joseph Gatto, Sarah E. Greer, Ganza Belise Isingizwe, Rohan Ray, Timothy E. Burdick, Sarah Masud Preum  

**Link**: [PDF](https://arxiv.org/pdf/2601.11344)  

**Abstract**: Large language models (LLMs) show promise in drafting responses to patient portal messages, yet their integration into clinical workflows raises various concerns, including whether they would actually save clinicians time and effort in their portal workload. We investigate LLM alignment with individual clinicians through a comprehensive evaluation of the patient message response drafting task. We develop a novel taxonomy of thematic elements in clinician responses and propose a novel evaluation framework for assessing clinician editing load of LLM-drafted responses at both content and theme levels. We release an expert-annotated dataset and conduct large-scale evaluations of local and commercial LLMs using various adaptation techniques including thematic prompting, retrieval-augmented generation, supervised fine-tuning, and direct preference optimization. Our results reveal substantial epistemic uncertainty in aligning LLM drafts with clinician responses. While LLMs demonstrate capability in drafting certain thematic elements, they struggle with clinician-aligned generation in other themes, particularly question asking to elicit further information from patients. Theme-driven adaptation strategies yield improvements across most themes. Our findings underscore the necessity of adapting LLMs to individual clinician preferences to enable reliable and responsible use in patient-clinician communication workflows. 

---
# From SERPs to Sound: How Search Engine Result Pages and AI-generated Podcasts Interact to Influence User Attitudes on Controversial Topics 

**Authors**: Junjie Wang, Gaole He, Alisa Rieger, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.11282)  

**Abstract**: Compared to search engine result pages (SERPs), AI-generated podcasts represent a relatively new and relatively more passive modality of information consumption, delivering narratives in a naturally engaging format. As these two media increasingly converge in everyday information-seeking behavior, it is essential to explore how their interaction influences user attitudes, particularly in contexts involving controversial, value-laden, and often debated topics. Addressing this need, we aim to understand how information mediums of present-day SERPs and AI-generated podcasts interact to shape the opinions of users. To this end, through a controlled user study (N=483), we investigated user attitudinal effects of consuming information via SERPs and AI-generated podcasts, focusing on how the sequence and modality of exposure shape user opinions. A majority of users in our study corresponded to attitude change outcomes, and we found an effect of sequence on attitude change. Our results further revealed a role of viewpoint bias and the degree of topic controversiality in shaping attitude change, although we found no effect of individual moderators. 

---
# X-Distill: Cross-Architecture Vision Distillation for Visuomotor Learning 

**Authors**: Maanping Shao, Feihong Zhang, Gu Zhang, Baiye Cheng, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11269)  

**Abstract**: Visuomotor policies often leverage large pre-trained Vision Transformers (ViTs) for their powerful generalization capabilities. However, their significant data requirements present a major challenge in the data-scarce context of most robotic learning settings, where compact CNNs with strong inductive biases can be more easily optimized. To address this trade-off, we introduce X-Distill, a simple yet highly effective method that synergizes the strengths of both architectures. Our approach involves an offline, cross-architecture knowledge distillation, transferring the rich visual representations of a large, frozen DINOv2 teacher to a compact ResNet-18 student on the general-purpose ImageNet dataset. This distilled encoder, now endowed with powerful visual priors, is then jointly fine-tuned with a diffusion policy head on the target manipulation tasks. Extensive experiments on $34$ simulated benchmarks and $5$ challenging real-world tasks demonstrate that our method consistently outperforms policies equipped with from-scratch ResNet or fine-tuned DINOv2 encoders. Notably, X-Distill also surpasses 3D encoders that utilize privileged point cloud observations or much larger Vision-Language Models. Our work highlights the efficacy of a simple, well-founded distillation strategy for achieving state-of-the-art performance in data-efficient robotic manipulation. 

---
# Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation 

**Authors**: Pingzhi Tang, Yiding Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11258)  

**Abstract**: Large Language Models (LLMs) face the "knowledge cutoff" challenge, where their frozen parametric memory prevents direct internalization of new information. While Supervised Fine-Tuning (SFT) is commonly used to update model knowledge, it often updates factual content without reliably improving the model's ability to use the newly incorporated information for question answering or decision-making. Reinforcement Learning (RL) is essential for acquiring reasoning skills; however, its high computational cost makes it impractical for efficient online adaptation. We empirically observe that the parameter updates induced by SFT and RL are nearly orthogonal. Based on this observation, we propose Parametric Skill Transfer (PaST), a framework that supports modular skill transfer for efficient and effective knowledge adaptation. By extracting a domain-agnostic Skill Vector from a source domain, we can linearly inject knowledge manipulation skills into a target model after it has undergone lightweight SFT on new data. Experiments on knowledge-incorporation QA (SQuAD, LooGLE) and agentic tool-use benchmarks (ToolBench) demonstrate the effectiveness of our method. On SQuAD, PaST outperforms the state-of-the-art self-editing SFT baseline by up to 9.9 points. PaST further scales to long-context QA on LooGLE with an 8.0-point absolute accuracy gain, and improves zero-shot ToolBench success rates by +10.3 points on average with consistent gains across tool categories, indicating strong scalability and cross-domain transferability of the Skill Vector. 

---
# FactCorrector: A Graph-Inspired Approach to Long-Form Factuality Correction of Large Language Models 

**Authors**: Javier Carnerero-Cano, Massimiliano Pronesti, Radu Marinescu, Tigran Tchrakian, James Barry, Jasmina Gajcin, Yufang Hou, Alessandra Pascale, Elizabeth Daly  

**Link**: [PDF](https://arxiv.org/pdf/2601.11232)  

**Abstract**: Large language models (LLMs) are widely used in knowledge-intensive applications but often generate factually incorrect responses. A promising approach to rectify these flaws is correcting LLMs using feedback. Therefore, in this paper, we introduce FactCorrector, a new post-hoc correction method that adapts across domains without retraining and leverages structured feedback about the factuality of the original response to generate a correction. To support rigorous evaluations of factuality correction methods, we also develop the VELI5 benchmark, a novel dataset containing systematically injected factual errors and ground-truth corrections. Experiments on VELI5 and several popular long-form factuality datasets show that the FactCorrector approach significantly improves factual precision while preserving relevance, outperforming strong baselines. We release our code at this https URL. 

---
# SDFLoRA: Selective Dual-Module LoRA for Federated Fine-tuning with Heterogeneous Clients 

**Authors**: Zhikang Shen, Jianrong Lu, Haiyuan Wan, Jianhai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11219)  

**Abstract**: Federated learning (FL) for large language models (LLMs) has attracted increasing attention as a way to enable privacy-preserving adaptation over distributed data. Parameter-efficient methods such as LoRA are widely adopted to reduce communication and memory costs. Despite these advances, practical FL deployments often exhibit rank heterogeneity, since different clients may use different low-rank configurations. This makes direct aggregation of LoRA updates biased and unstable. Existing solutions typically enforce unified ranks or align heterogeneous updates into a shared subspace, which over-constrains client-specific semantics, limits personalization, and provides weak protection of local client information under differential privacy noise. To address this issue, we propose Selective Dual-module Federated LoRA (SDFLoRA), which decomposes each client adapter into a global module that captures transferable knowledge and a local module that preserves client-specific adaptations. The global module is selectively aligned and aggregated across clients, while local modules remain private. This design enables robust learning under rank heterogeneity and supports privacy-aware optimization by injecting differential privacy noise exclusively into the global module. Experiments on GLUE benchmarks demonstrate that SDFLoRA outperforms representative federated LoRA baselines and achieves a better utility-privacy trade-off. 

---
# LoRA as Oracle 

**Authors**: Marco Arazzi, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2601.11207)  

**Abstract**: Backdoored and privacy-leaking deep neural networks pose a serious threat to the deployment of machine learning systems in security-critical settings. Existing defenses for backdoor detection and membership inference typically require access to clean reference models, extensive retraining, or strong assumptions about the attack mechanism. In this work, we introduce a novel LoRA-based oracle framework that leverages low-rank adaptation modules as a lightweight, model-agnostic probe for both backdoor detection and membership inference.
Our approach attaches task-specific LoRA adapters to a frozen backbone and analyzes their optimization dynamics and representation shifts when exposed to suspicious samples. We show that poisoned and member samples induce distinctive low-rank updates that differ significantly from those generated by clean or non-member data. These signals can be measured using simple ranking and energy-based statistics, enabling reliable inference without access to the original training data or modification of the deployed model. 

---
# Epistemic Control and the Normativity of Machine Learning-Based Science 

**Authors**: Emanuele Ratti  

**Link**: [PDF](https://arxiv.org/pdf/2601.11202)  

**Abstract**: The past few years have witnessed an increasing use of machine learning (ML) systems in science. Paul Humphreys has argued that, because of specific characteristics of ML systems, human scientists are pushed out of the loop of science. In this chapter, I investigate to what extent this is true. First, I express these concerns in terms of what I call epistemic control. I identify two conditions for epistemic control, called tracking and tracing, drawing on works in philosophy of technology. With this new understanding of the problem, I then argue against Humphreys pessimistic view. Finally, I construct a more nuanced view of epistemic control in ML-based science. 

---
# FAQ: Mitigating Quantization Error via Regenerating Calibration Data with Family-Aware Quantization 

**Authors**: Haiyang Xiao, Weiqing Li, Jinyue Guo, Guochao Jiang, Guohua Liu, Yuewei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11200)  

**Abstract**: Although post-training quantization (PTQ) provides an efficient numerical compression scheme for deploying large language models (LLMs) on resource-constrained devices, the representativeness and universality of calibration data remain a core bottleneck in determining the accuracy of quantization parameters. Traditional PTQ methods typically rely on limited samples, making it difficult to capture the activation distribution during the inference phase, leading to biases in quantization parameters. To address this, we propose \textbf{FAQ} (Family-Aware Quantization), a calibration data regeneration framework that leverages prior knowledge from LLMs of the same family to generate high-fidelity calibration samples. Specifically, FAQ first inputs the original calibration samples into a larger LLM from the same family as the target model, regenerating a series of high-fidelity calibration data using a highly consistent knowledge system. Subsequently, this data, carrying Chain-of-Thought reasoning and conforming to the expected activation distribution, undergoes group competition under expert guidance to select the best samples, which are then re-normalized to enhance the effectiveness of standard PTQ. Experiments on multiple model series, including Qwen3-8B, show that FAQ reduces accuracy loss by up to 28.5\% compared to the baseline with original calibration data, demonstrating its powerful potential and contribution. 

---
# SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation 

**Authors**: Aiman Al Masoud, Marco Arazzi, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2601.11199)  

**Abstract**: Retrieval-Augmented Generation (RAG) has attracted significant attention due to its ability to combine the generative capabilities of Large Language Models (LLMs) with knowledge obtained through efficient retrieval mechanisms over large-scale data collections. Currently, the majority of existing approaches overlook the risks associated with exposing sensitive or access-controlled information directly to the generation model. Only a few approaches propose techniques to instruct the generative model to refrain from disclosing sensitive information; however, recent studies have also demonstrated that LLMs remain vulnerable to prompt injection attacks that can override intended behavioral constraints. For these reasons, we propose a novel approach to Selective Disclosure in Retrieval-Augmented Generation, called SD-RAG, which decouples the enforcement of security and privacy constraints from the generation process itself. Rather than relying on prompt-level safeguards, SD-RAG applies sanitization and disclosure controls during the retrieval phase, prior to augmenting the language model's input. Moreover, we introduce a semantic mechanism to allow the ingestion of human-readable dynamic security and privacy constraints together with an optimized graph-based data model that supports fine-grained, policy-aware retrieval. Our experimental evaluation demonstrates the superiority of SD-RAG over baseline existing approaches, achieving up to a $58\%$ improvement in the privacy score, while also showing a strong resilience to prompt injection attacks targeting the generative model. 

---
# Artificial Intelligence and the US Economy: An Accounting Perspective on Investment and Production 

**Authors**: Luisa Carpinelli, Filippo Natoli, Marco Taboga  

**Link**: [PDF](https://arxiv.org/pdf/2601.11196)  

**Abstract**: Artificial intelligence (AI) has moved to the center of policy, market, and academic debates, but its macroeconomic footprint is still only partly understood. This paper provides an overview on how the current AI wave is captured in US national accounts, combining a simple macro-accounting framework with a stylized description of the AI production process. We highlight the crucial role played by data centers, which constitute the backbone of the AI ecosystem and have attracted formidable investment in 2025, as they are indispensable for meeting the rapidly increasing worldwide demand for AI services. We document that the boom in IT and AI-related capital expenditure in the first three quarters of the year has given an outsized boost to aggregate demand, while its contribution to GDP growth is smaller once the high import content of AI hardware is netted out. Furthermore, simple calculations suggest that, at current utilization rates and pricing, the production of services originating in new AI data centers could contribute to GDP over the turn of the next quarters on a scale comparable to that of investment spending to date. Short reinvestment cycles and uncertainty about future AI demand, while not currently acting as a macroeconomic drag, can nevertheless fuel macroeconomic risks over the medium term. 

---
# Clustering High-dimensional Data: Balancing Abstraction and Representation Tutorial at AAAI 2026 

**Authors**: Claudia Plant, Lena G. M. Bauer, Christian Böhm  

**Link**: [PDF](https://arxiv.org/pdf/2601.11160)  

**Abstract**: How to find a natural grouping of a large real data set? Clustering requires a balance between abstraction and representation. To identify clusters, we need to abstract from superfluous details of individual objects. But we also need a rich representation that emphasizes the key features shared by groups of objects that distinguish them from other groups of objects.
Each clustering algorithm implements a different trade-off between abstraction and representation. Classical K-means implements a high level of abstraction - details are simply averaged out - combined with a very simple representation - all clusters are Gaussians in the original data space. We will see how approaches to subspace and deep clustering support high-dimensional and complex data by allowing richer representations. However, with increasing representational expressiveness comes the need to explicitly enforce abstraction in the objective function to ensure that the resulting method performs clustering and not just representation learning. We will see how current deep clustering methods define and enforce abstraction through centroid-based and density-based clustering losses. Balancing the conflicting goals of abstraction and representation is challenging. Ideas from subspace clustering help by learning one latent space for the information that is relevant to clustering and another latent space to capture all other information in the data.
The tutorial ends with an outlook on future research in clustering. Future methods will more adaptively balance abstraction and representation to improve performance, energy efficiency and interpretability. By automatically finding the sweet spot between abstraction and representation, the human brain is very good at clustering and other related tasks such as single-shot learning. So, there is still much room for improvement. 

---
# Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation 

**Authors**: Ji Dai, Quan Fang, Jun Hu, Desheng Cai, Yang Yang, Can Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11151)  

**Abstract**: Multimedia recommendation systems leverage user-item interactions and multimodal information to capture user preferences, enabling more accurate and personalized recommendations. Despite notable advancements, existing approaches still face two critical limitations: first, shallow modality fusion often relies on simple concatenation, failing to exploit rich synergic intra- and inter-modal relationships; second, asymmetric feature treatment-where users are only characterized by interaction IDs while items benefit from rich multimodal content-hinders the learning of a shared semantic space. To address these issues, we propose a Cross-modal Recursive Attention Network with dual graph Embedding (CRANE). To tackle shallow fusion, we design a core Recursive Cross-Modal Attention (RCA) mechanism that iteratively refines modality features based on cross-correlations in a joint latent space, effectively capturing high-order intra- and inter-modal dependencies. For symmetric multimodal learning, we explicitly construct users' multimodal profiles by aggregating features of their interacted items. Furthermore, CRANE integrates a symmetric dual-graph framework-comprising a heterogeneous user-item interaction graph and a homogeneous item-item semantic graph-unified by a self-supervised contrastive learning objective to fuse behavioral and semantic signals. Despite these complex modeling capabilities, CRANE maintains high computational efficiency. Theoretical and empirical analyses confirm its scalability and high practical efficiency, achieving faster convergence on small datasets and superior performance ceilings on large-scale ones. Comprehensive experiments on four public real-world datasets validate an average 5% improvement in key metrics over state-of-the-art baselines. 

---
# Deep GraphRAG: A Balanced Approach to Hierarchical Retrieval and Adaptive Integration 

**Authors**: Yuejie Li, Ke Yang, Tao Wang, Bolin Chen, Bowen Li, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.11144)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) frameworks face a trade-off between the comprehensiveness of global search and the efficiency of local search. Existing methods are often challenged by navigating large-scale hierarchical graphs, optimizing retrieval paths, and balancing exploration-exploitation dynamics, frequently lacking robust multi-stage re-ranking. To overcome these deficits, we propose Deep GraphRAG, a framework designed for a balanced approach to hierarchical retrieval and adaptive integration. It introduces a hierarchical global-to-local retrieval strategy that integrates macroscopic inter-community and microscopic intra-community contextual relations. This strategy employs a three-stage process: (1) inter-community filtering, which prunes the search space using local context; (2) community-level refinement, which prioritizes relevant subgraphs via entity-interaction analysis; and (3) entity-level fine-grained search within target communities. A beam search-optimized dynamic re-ranking module guides this process, continuously filtering candidates to balance efficiency and global comprehensiveness. Deep GraphRAG also features a Knowledge Integration Module leveraging a compact LLM, trained with Dynamic Weighting Reward GRPO (DW-GRPO). This novel reinforcement learning approach dynamically adjusts reward weights to balance three key objectives: relevance, faithfulness, and conciseness. This training enables compact models (1.5B) to approach the performance of large models (70B) in the integration task. Evaluations on Natural Questions and HotpotQA demonstrate that Deep GraphRAG significantly outperforms baseline graph retrieval methods in both accuracy and efficiency. 

---
# Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model 

**Authors**: Minho Lee, Hyeonseok Kim, Jin Tak Kim, Sangshin Park, Jeong Hyun Lee, Jungsan Cho, Jemin Hwangbo  

**Link**: [PDF](https://arxiv.org/pdf/2601.11143)  

**Abstract**: The simulation-to-reality (sim-to-real) transfer of large-scale hydraulic robots presents a significant challenge in robotics because of the inherent slow control response and complex fluid dynamics. The complex dynamics result from the multiple interconnected cylinder structure and the difference in fluid rates of the cylinders. These characteristics complicate detailed simulation for all joints, making it unsuitable for reinforcement learning (RL) applications. In this work, we propose an analytical actuator model driven by hydraulic dynamics to represent the complicated actuators. The model predicts joint torques for all 12 actuators in under 1 microsecond, allowing rapid processing in RL environments. We compare our model with neural network-based actuator models and demonstrate the advantages of our model in data-limited scenarios. The locomotion policy trained in RL with our model is deployed on a hydraulic quadruped robot, which is over 300 kg. This work is the first demonstration of a successful transfer of stable and robust command-tracking locomotion with RL on a heavy hydraulic quadruped robot, demonstrating advanced sim-to-real transferability. 

---
# Context-aware Graph Causality Inference for Few-Shot Molecular Property Prediction 

**Authors**: Van Thuy Hoang, O-Joun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.11135)  

**Abstract**: Molecular property prediction is becoming one of the major applications of graph learning in Web-based services, e.g., online protein structure prediction and drug discovery. A key challenge arises in few-shot scenarios, where only a few labeled molecules are available for predicting unseen properties. Recently, several studies have used in-context learning to capture relationships among molecules and properties, but they face two limitations in: (1) exploiting prior knowledge of functional groups that are causally linked to properties and (2) identifying key substructures directly correlated with properties. We propose CaMol, a context-aware graph causality inference framework, to address these challenges by using a causal inference perspective, assuming that each molecule consists of a latent causal structure that determines a specific property. First, we introduce a context graph that encodes chemical knowledge by linking functional groups, molecules, and properties to guide the discovery of causal substructures. Second, we propose a learnable atom masking strategy to disentangle causal substructures from confounding ones. Third, we introduce a distribution intervener that applies backdoor adjustment by combining causal substructures with chemically grounded confounders, disentangling causal effects from real-world chemical variations. Experiments on diverse molecular datasets showed that CaMol achieved superior accuracy and sample efficiency in few-shot tasks, showing its generalizability to unseen properties. Also, the discovered causal substructures were strongly aligned with chemical knowledge about functional groups, supporting the model interpretability. 

---
# Learn Before Represent: Bridging Generative and Contrastive Learning for Domain-Specific LLM Embeddings 

**Authors**: Xiaoyu Liang, Yuchen Peng, Jiale Luo, Wenhao Wang, Haoji Hu, Xincheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.11124)  

**Abstract**: Large Language Models (LLMs) adapted via contrastive learning excel in general representation learning but struggle in vertical domains like chemistry and law, primarily due to a lack of domain-specific knowledge. This work identifies a core bottleneck: the prevailing ``LLM+CL'' paradigm focuses on semantic alignment but cannot perform knowledge acquisition, leading to failures on specialized terminology. To bridge this gap, we propose Learn Before Represent (LBR), a novel two-stage framework. LBR first injects domain knowledge via an Information Bottleneck-Constrained Generative Learning stage, preserving the LLM's causal attention to maximize knowledge acquisition while compressing semantics. It then performs Generative-Refined Contrastive Learning on the compressed representations for alignment. This approach maintains architectural consistency and resolves the objective conflict between generative and contrastive learning. Extensive experiments on medical, chemistry, and code retrieval tasks show that LBR significantly outperforms strong baselines. Our work establishes a new paradigm for building accurate and robust representations in vertical domains. 

---
# Vision-as-Inverse-Graphics Agent via Interleaved Multimodal Reasoning 

**Authors**: Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, Haiwen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11109)  

**Abstract**: Vision-as-inverse-graphics, the concept of reconstructing an image as an editable graphics program is a long-standing goal of computer vision. Yet even strong VLMs aren't able to achieve this in one-shot as they lack fine-grained spatial and physical grounding capability. Our key insight is that closing this gap requires interleaved multimodal reasoning through iterative execution and verification. Stemming from this, we present VIGA (Vision-as-Inverse-Graphic Agent) that starts from an empty world and reconstructs or edits scenes through a closed-loop write-run-render-compare-revise procedure. To support long-horizon reasoning, VIGA combines (i) a skill library that alternates generator and verifier roles and (ii) an evolving context memory that contains plans, code diffs, and render history. VIGA is task-agnostic as it doesn't require auxiliary modules, covering a wide range of tasks such as 3D reconstruction, multi-step scene editing, 4D physical interaction, and 2D document editing, etc. Empirically, we found VIGA substantially improves one-shot baselines on BlenderGym (35.32%) and SlideBench (117.17%). Moreover, VIGA is also model-agnostic as it doesn't require finetuning, enabling a unified protocol to evaluate heterogeneous foundation VLMs. To better support this protocol, we introduce BlenderBench, a challenging benchmark that stress-tests interleaved multimodal reasoning with graphics engine, where VIGA improves by 124.70%. 

---
# Efficient Multilingual Name Type Classification Using Convolutional Networks 

**Authors**: Davor Lauc  

**Link**: [PDF](https://arxiv.org/pdf/2601.11090)  

**Abstract**: We present a convolutional neural network approach for classifying proper names by language and entity type. Our model, Onomas-CNN X, combines parallel convolution branches with depthwise-separable operations and hierarchical classification to process names efficiently on CPU hardware. We evaluate the architecture on a large multilingual dataset covering 104 languages and four entity types (person, organization, location, other). Onomas-CNN X achieves 92.1% accuracy while processing 2,813 names per second on a single CPU core - 46 times faster than fine-tuned XLM-RoBERTa with comparable accuracy. The model reduces energy consumption by a factor of 46 compared to transformer baselines. Our experiments demonstrate that specialized CNN architectures remain competitive with large pre-trained models for focused NLP tasks when sufficient training data exists. 

---
# Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments 

**Authors**: Jiaohong Yao, Linfeng Liang, Yao Deng, Xi Zheng, Richard Han, Yuankai Qi  

**Link**: [PDF](https://arxiv.org/pdf/2601.11078)  

**Abstract**: Marker-based landing is widely used in drone delivery and return-to-base systems for its simplicity and reliability. However, most approaches assume idealized landing site visibility and sensor performance, limiting robustness in complex urban settings. We present a simulation-based evaluation suite on the AirSim platform with systematically varied urban layouts, lighting, and weather to replicate realistic operational diversity. Using onboard camera sensors (RGB for marker detection and depth for obstacle avoidance), we benchmark two heuristic coverage patterns and a reinforcement learning-based agent, analyzing how exploration strategy and scene complexity affect success rate, path efficiency, and robustness. Results underscore the need to evaluate marker-based autonomous landing under diverse, sensor-relevant conditions to guide the development of reliable aerial navigation systems. 

---
# ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development 

**Authors**: Jie Yang, Honglin Guo, Li Ji, Jiazheng Zhou, Rui Zheng, Zhikai Lei, Shuo Zhang, Zhiheng Xi, Shichun Liu, Yuxin Wang, Bo Wang, Yining Zheng, Tao Gui, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11077)  

**Abstract**: The evolution of Large Language Models (LLMs) into autonomous agents has expanded the scope of AI coding from localized code generation to complex, repository-level, and execution-driven problem solving. However, current benchmarks predominantly evaluate code logic in static contexts, neglecting the dynamic, full-process requirements of real-world engineering, particularly in backend development which demands rigorous environment configuration and service deployment. To address this gap, we introduce ABC-Bench, a benchmark explicitly designed to evaluate agentic backend coding within a realistic, executable workflow. Using a scalable automated pipeline, we curated 224 practical tasks spanning 8 languages and 19 frameworks from open-source repositories. Distinct from previous evaluations, ABC-Bench require the agents to manage the entire development lifecycle from repository exploration to instantiating containerized services and pass the external end-to-end API tests. Our extensive evaluation reveals that even state-of-the-art models struggle to deliver reliable performance on these holistic tasks, highlighting a substantial disparity between current model capabilities and the demands of practical backend engineering. Our code is available at this https URL. 

---
# A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation 

**Authors**: Jiaqi Liang, Yue Chen, Qize Yu, Yan Shen, Haipeng Zhang, Hao Dong, Ruihai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11076)  

**Abstract**: Furniture assembly is a crucial yet challenging task for robots, requiring precise dual-arm coordination where one arm manipulates parts while the other provides collaborative support and stabilization. To accomplish this task more effectively, robots need to actively adapt support strategies throughout the long-horizon assembly process, while also generalizing across diverse part geometries. We propose A3D, a framework which learns adaptive affordances to identify optimal support and stabilization locations on furniture parts. The method employs dense point-level geometric representations to model part interaction patterns, enabling generalization across varied geometries. To handle evolving assembly states, we introduce an adaptive module that uses interaction feedback to dynamically adjust support strategies during assembly based on previous interactions. We establish a simulation environment featuring 50 diverse parts across 8 furniture types, designed for dual-arm collaboration evaluation. Experiments demonstrate that our framework generalizes effectively to diverse part geometries and furniture categories in both simulation and real-world settings. 

---
# Bridging Cognitive Neuroscience and Graph Intelligence: Hippocampus-Inspired Multi-View Hypergraph Learning for Web Finance Fraud 

**Authors**: Rongkun Cui, Nana Zhang, Kun Zhu, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11073)  

**Abstract**: Online financial services constitute an essential component of contemporary web ecosystems, yet their openness introduces substantial exposure to fraud that harms vulnerable users and weakens trust in digital finance. Such threats have become a significant web harm that erodes societal fairness and affects the well being of online communities. However, existing detection methods based on graph neural networks (GNNs) struggle with two persistent challenges: (1) fraud camouflage, where malicious transactions mimic benign behaviors to evade detection, and (2) long-tailed data distributions, which obscure rare but critical fraudulent cases. To fill these gaps, we propose HIMVH, a Hippocampus-Inspired Multi-View Hypergraph learning model for web finance fraud detection. Specifically, drawing inspiration from the scene conflict monitoring role of the hippocampus, we design a cross-view inconsistency perception module that captures subtle discrepancies and behavioral heterogeneity across multiple transaction views. This module enables the model to identify subtle cross-view conflicts for detecting online camouflaged fraudulent behaviors. Furthermore, inspired by the match-mismatch novelty detection mechanism of the CA1 region, we introduce a novelty-aware hypergraph learning module that measures feature deviations from neighborhood expectations and adaptively reweights messages, thereby enhancing sensitivity to online rare fraud patterns in the long-tailed settings. Extensive experiments on six web-based financial fraud datasets demonstrate that HIMVH achieves 6.42\% improvement in AUC, 9.74\% in F1 and 39.14\% in AP on average over 15 SOTA models. 

---
# Fairness in Healthcare Processes: A Quantitative Analysis of Decision Making in Triage 

**Authors**: Rachmadita Andreswari, Stephan A. Fahrenkrog-Petersen, Jan Mendling  

**Link**: [PDF](https://arxiv.org/pdf/2601.11065)  

**Abstract**: Fairness in automated decision-making has become a critical concern, particularly in high-pressure healthcare scenarios such as emergency triage, where fast and equitable decisions are essential. Process mining is increasingly investigating fairness. There is a growing area focusing on fairness-aware algorithms. So far, we know less how these concepts perform on empirical healthcare data or how they cover aspects of justice theory. This study addresses this research problem and proposes a process mining approach to assess fairness in triage by linking real-life event logs with conceptual dimensions of justice. Using the MIMICEL event log (as derived from MIMIC-IV ED), we analyze time, re-do, deviation and decision as process outcomes, and evaluate the influence of age, gender, race, language and insurance using the Kruskal-Wallis, Chi-square and effect size measurements. These outcomes are mapped to justice dimensions to support the development of a conceptual framework. The results demonstrate which aspects of potential unfairness in high-acuity and sub-acute surface. In this way, this study contributes empirical insights that support further research in responsible, fairness-aware process mining in healthcare. 

---
# H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning 

**Authors**: Haishan Zeng, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.11063)  

**Abstract**: In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P. 

---
# Predicting Biased Human Decision-Making with Large Language Models in Conversational Settings 

**Authors**: Stephen Pilli, Vivek Nallur  

**Link**: [PDF](https://arxiv.org/pdf/2601.11049)  

**Abstract**: We examine whether large language models (LLMs) can predict biased decision-making in conversational settings, and whether their predictions capture not only human cognitive biases but also how those effects change under cognitive load. In a pre-registered study (N = 1,648), participants completed six classic decision-making tasks via a chatbot with dialogues of varying complexity. Participants exhibited two well-documented cognitive biases: the Framing Effect and the Status Quo Bias. Increased dialogue complexity resulted in participants reporting higher mental demand. This increase in cognitive load selectively, but significantly, increased the effect of the biases, demonstrating the load-bias interaction. We then evaluated whether LLMs (GPT-4, GPT-5, and open-source models) could predict individual decisions given demographic information and prior dialogue. While results were mixed across choice problems, LLM predictions that incorporated dialogue context were significantly more accurate in several key scenarios. Importantly, their predictions reproduced the same bias patterns and load-bias interactions observed in humans. Across all models tested, the GPT-4 family consistently aligned with human behavior, outperforming GPT-5 and open-source models in both predictive accuracy and fidelity to human-like bias patterns. These findings advance our understanding of LLMs as tools for simulating human decision-making and inform the design of conversational agents that adapt to user biases. 

---
# Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse 

**Authors**: Chi Zhang, Mengqi Zhang, Xiaotian Ye, Runxi Cheng, Zisheng Zhou, Ying Zhou, Pengjie Ren, Zhumin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.11042)  

**Abstract**: Sequential knowledge editing in large language models often causes catastrophic collapse of the model's general abilities, especially for parameter-modifying methods. Existing approaches mitigate this issue through heuristic constraints on parameter updates, yet the mechanisms underlying such degradation remain insufficiently understood. In this work, we present a spectral analysis of sequential knowledge editing and show that a model's general abilities are closely associated with dominant singular directions of pretrained weight matrices. These directions are highly sensitive to perturbations and are progressively disrupted by repeated edits, closely tracking the collapse in both editing efficacy and general performance. Building on this insight, we propose REVIVE, a plug-and-play framework that stabilizes sequential editing by explicitly preserving the dominant singular subspace. REVIVE represents parameter updates in the spectral basis of the original weights and filters components that would interfere with the protected region. Extensive experiments across multiple models and benchmarks show that REVIVE consistently improves editing efficacy while substantially preserving general abilities under long-horizon sequential editing, including extreme settings with up to 20,000 edits. 

---
# Your One-Stop Solution for AI-Generated Video Detection 

**Authors**: Long Ma, Zihao Xue, Yan Wang, Zhiyuan Yan, Jin Xu, Xiaorui Jiang, Haiyang Yu, Yong Liao, Zhen Bi  

**Link**: [PDF](https://arxiv.org/pdf/2601.11035)  

**Abstract**: Recent advances in generative modeling can create remarkably realistic synthetic videos, making it increasingly difficult for humans to distinguish them from real ones and necessitating reliable detection methods.
However, two key limitations hinder the development of this field.
\textbf{From the dataset perspective}, existing datasets are often limited in scale and constructed using outdated or narrowly scoped generative models, making it difficult to capture the diversity and rapid evolution of modern generative techniques. Moreover, the dataset construction process frequently prioritizes quantity over quality, neglecting essential aspects such as semantic diversity, scenario coverage, and technological representativeness.
\textbf{From the benchmark perspective}, current benchmarks largely remain at the stage of dataset creation, leaving many fundamental issues and in-depth analysis yet to be systematically explored.
Addressing this gap, we propose AIGVDBench, a benchmark designed to be comprehensive and representative, covering \textbf{31} state-of-the-art generation models and over \textbf{440,000} videos. By executing more than \textbf{1,500} evaluations on \textbf{33} existing detectors belonging to four distinct categories. This work presents \textbf{8 in-depth analyses} from multiple perspectives and identifies \textbf{4 novel findings} that offer valuable insights for future research. We hope this work provides a solid foundation for advancing the field of AI-generated video detection.
Our benchmark is open-sourced at this https URL. 

---
# IDDR-NGP: Incorporating Detectors for Distractor Removal with Instant Neural Radiance Field 

**Authors**: Xianliang Huang, Jiajie Gou, Shuhang Chen, Zhizhou Zhong, Jihong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.11030)  

**Abstract**: This paper presents the first unified distractor removal method, named IDDR-NGP, which directly operates on Instant-NPG. The method is able to remove a wide range of distractors in 3D scenes, such as snowflakes, confetti, defoliation and petals, whereas existing methods usually focus on a specific type of distractors. By incorporating implicit 3D representations with 2D detectors, we demonstrate that it is possible to efficiently restore 3D scenes from multiple corrupted images. We design the learned perceptual image patch similarity~( LPIPS) loss and the multi-view compensation loss (MVCL) to jointly optimize the rendering results of IDDR-NGP, which could aggregate information from multi-view corrupted images. All of them can be trained in an end-to-end manner to synthesize high-quality 3D scenes. To support the research on distractors removal in implicit 3D representations, we build a new benchmark dataset that consists of both synthetic and real-world distractors. To validate the effectiveness and robustness of IDDR-NGP, we provide a wide range of distractors with corresponding annotated labels added to both realistic and synthetic scenes. Extensive experimental results demonstrate the effectiveness and robustness of IDDR-NGP in removing multiple types of distractors. In addition, our approach achieves results comparable with the existing SOTA desnow methods and is capable of accurately removing both realistic and synthetic distractors. 

---
# Combating Spurious Correlations in Graph Interpretability via Self-Reflection 

**Authors**: Kecheng Cai, Chenyang Xu, Chao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.11021)  

**Abstract**: Interpretable graph learning has recently emerged as a popular research topic in machine learning. The goal is to identify the important nodes and edges of an input graph that are crucial for performing a specific graph reasoning task. A number of studies have been conducted in this area, and various benchmark datasets have been proposed to facilitate evaluation. Among them, one of the most challenging is the Spurious-Motif benchmark, introduced at ICLR 2022. The datasets in this synthetic benchmark are deliberately designed to include spurious correlations, making it particularly difficult for models to distinguish truly relevant structures from misleading patterns. As a result, existing methods exhibit significantly worse performance on this benchmark compared to others.
In this paper, we focus on improving interpretability on the challenging Spurious-Motif datasets. We demonstrate that the self-reflection technique, commonly used in large language models to tackle complex tasks, can also be effectively adapted to enhance interpretability in datasets with strong spurious correlations. Specifically, we propose a self-reflection framework that can be integrated with existing interpretable graph learning methods. When such a method produces importance scores for each node and edge, our framework feeds these predictions back into the original method to perform a second round of evaluation. This iterative process mirrors how large language models employ self-reflective prompting to reassess their previous outputs. We further analyze the reasons behind this improvement from the perspective of graph representation learning, which motivates us to propose a fine-tuning training method based on this feedback mechanism. 

---
# Finding the Translation Switch: Discovering and Exploiting the Task-Initiation Features in LLMs 

**Authors**: Xinwei Wu, Heng Liu, Xiaohu Zhao, Yuqi Ren, Linlong Xu, Longyue Wang, Deyi Xiong, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11019)  

**Abstract**: Large Language Models (LLMs) frequently exhibit strong translation abilities, even without task-specific fine-tuning. However, the internal mechanisms governing this innate capability remain largely opaque. To demystify this process, we leverage Sparse Autoencoders (SAEs) and introduce a novel framework for identifying task-specific features. Our method first recalls features that are frequently co-activated on translation inputs and then filters them for functional coherence using a PCA-based consistency metric. This framework successfully isolates a small set of **translation initiation** features. Causal interventions demonstrate that amplifying these features steers the model towards correct translation, while ablating them induces hallucinations and off-task outputs, confirming they represent a core component of the model's innate translation competency. Moving from analysis to application, we leverage this mechanistic insight to propose a new data selection strategy for efficient fine-tuning. Specifically, we prioritize training on **mechanistically hard** samples-those that fail to naturally activate the translation initiation features. Experiments show this approach significantly improves data efficiency and suppresses hallucinations. Furthermore, we find these mechanisms are transferable to larger models of the same family. Our work not only decodes a core component of the translation mechanism in LLMs but also provides a blueprint for using internal model mechanism to create more robust and efficient models. The codes are available at this https URL. 

---
# Contextual Distributionally Robust Optimization with Causal and Continuous Structure: An Interpretable and Tractable Approach 

**Authors**: Fenglin Zhang, Jie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.11016)  

**Abstract**: In this paper, we introduce a framework for contextual distributionally robust optimization (DRO) that considers the causal and continuous structure of the underlying distribution by developing interpretable and tractable decision rules that prescribe decisions using covariates. We first introduce the causal Sinkhorn discrepancy (CSD), an entropy-regularized causal Wasserstein distance that encourages continuous transport plans while preserving the causal consistency. We then formulate a contextual DRO model with a CSD-based ambiguity set, termed Causal Sinkhorn DRO (Causal-SDRO), and derive its strong dual reformulation where the worst-case distribution is characterized as a mixture of Gibbs distributions. To solve the corresponding infinite-dimensional policy optimization, we propose the Soft Regression Forest (SRF) decision rule, which approximates optimal policies within arbitrary measurable function spaces. The SRF preserves the interpretability of classical decision trees while being fully parametric, differentiable, and Lipschitz smooth, enabling intrinsic interpretation from both global and local perspectives. To solve the Causal-SDRO with parametric decision rules, we develop an efficient stochastic compositional gradient algorithm that converges to an $\varepsilon$-stationary point at a rate of $O(\varepsilon^{-4})$, matching the convergence rate of standard stochastic gradient descent. Finally, we validate our method through numerical experiments on synthetic and real-world datasets, demonstrating its superior performance and interpretability. 

---
# When Personalization Misleads: Understanding and Mitigating Hallucinations in Personalized LLMs 

**Authors**: Zhongxiang Sun, Yi Zhan, Chenglei Shen, Weijie Yu, Xiao Zhang, Ming He, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.11000)  

**Abstract**: Personalized large language models (LLMs) adapt model behavior to individual users to enhance user satisfaction, yet personalization can inadvertently distort factual reasoning. We show that when personalized LLMs face factual queries, there exists a phenomenon where the model generates answers aligned with a user's prior history rather than the objective truth, resulting in personalization-induced hallucinations that degrade factual reliability and may propagate incorrect beliefs, due to representational entanglement between personalization and factual representations. To address this issue, we propose Factuality-Preserving Personalized Steering (FPPS), a lightweight inference-time approach that mitigates personalization-induced factual distortions while preserving personalized behavior. We further introduce PFQABench, the first benchmark designed to jointly evaluate factual and personalized question answering under personalization. Experiments across multiple LLM backbones and personalization methods show that FPPS substantially improves factual accuracy while maintaining personalized performance. 

---
# Steering Language Models Before They Speak: Logit-Level Interventions 

**Authors**: Hyeseon An, Shinwoo Park, Hyundong Jin, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.10960)  

**Abstract**: Steering LLMs is essential for specialized applications such as style-sensitive text rewriting, user-adaptive communication, and toxicity mitigation. Current steering methods, such as prompting-based and activation-based approaches, are widely used to guide model behavior. However, activation-based techniques require deep access to internal layers, while prompting-based steering often fails to provide consistent or fine-grained control. In order to address these limitations, we propose a training-free inference-time logit intervention for controllable generation. Our approach utilizes a statistical token score table derived from z-normalized log-odds of labeled corpora to shift the decoding distribution. Empirical evaluations across three diverse datasets focusing on writing complexity, formality, and toxicity demonstrate that our method effectively steers output characteristics, confirming its broad applicability and task-agnostic nature. Our results show that statistically grounded logit steering can achieve large, consistent, and multi-task control gains: up to +47%p accuracy and 50x f1 improvement. 

---
# Beyond Max Tokens: Stealthy Resource Amplification via Tool Calling Chains in LLM Agents 

**Authors**: Kaiyu Zhou, Yongsen Zheng, Yicheng He, Meng Xue, Xueluan Gong, Yuji Wang, Kwok-Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2601.10955)  

**Abstract**: The agent-tool communication loop is a critical attack surface in modern Large Language Model (LLM) agents. Existing Denial-of-Service (DoS) attacks, primarily triggered via user prompts or injected retrieval-augmented generation (RAG) context, are ineffective for this new paradigm. They are fundamentally single-turn and often lack a task-oriented approach, making them conspicuous in goal-oriented workflows and unable to exploit the compounding costs of multi-turn agent-tool interactions. We introduce a stealthy, multi-turn economic DoS attack that operates at the tool layer under the guise of a correctly completed task. Our method adjusts text-visible fields and a template-governed return policy in a benign, Model Context Protocol (MCP)-compatible tool server, optimizing these edits with a Monte Carlo Tree Search (MCTS) optimizer. These adjustments leave function signatures unchanged and preserve the final payload, steering the agent into prolonged, verbose tool-calling sequences using text-only notices. This compounds costs across turns, escaping single-turn caps while keeping the final answer correct to evade validation. Across six LLMs on the ToolBench and BFCL benchmarks, our attack expands tasks into trajectories exceeding 60,000 tokens, inflates costs by up to 658x, and raises energy by 100-560x. It drives GPU KV cache occupancy from <1% to 35-74% and cuts co-running throughput by approximately 50%. Because the server remains protocol-compatible and task outcomes are correct, conventional checks fail. These results elevate the agent-tool interface to a first-class security frontier, demanding a paradigm shift from validating final answers to monitoring the economic and computational cost of the entire agentic process. 

---
# Multi-Stage Patient Role-Playing Framework for Realistic Clinical Interactions 

**Authors**: Shijie Jiang, Zefan Zhang, Kehua Zhu, Tian Bai, Ruihong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10951)  

**Abstract**: The simulation of realistic clinical interactions plays a pivotal role in advancing clinical Large Language Models (LLMs) and supporting medical diagnostic education. Existing approaches and benchmarks rely on generic or LLM-generated dialogue data, which limits the authenticity and diversity of doctor-patient interactions. In this work, we propose the first Chinese patient simulation dataset (Ch-PatientSim), constructed from realistic clinical interaction scenarios to comprehensively evaluate the performance of models in emulating patient behavior. Patients are simulated based on a five-dimensional persona structure. To address issues of the persona class imbalance, a portion of the dataset is augmented using few-shot generation, followed by manual verification. We evaluate various state-of-the-art LLMs and find that most produce overly formal responses that lack individual personality. To address this limitation, we propose a training-free Multi-Stage Patient Role-Playing (MSPRP) framework, which decomposes interactions into three stages to ensure both personalization and realism in model responses. Experimental results demonstrate that our approach significantly improves model performance across multiple dimensions of patient simulation. 

---
# PatientVLM Meets DocVLM: Pre-Consultation Dialogue Between Vision-Language Models for Efficient Diagnosis 

**Authors**: K Lokesh, Abhirama Subramanyam Penamakuri, Uday Agarwal, Apoorva Challa, Shreya K Gowda, Somesh Gupta, Anand Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2601.10945)  

**Abstract**: Traditionally, AI research in medical diagnosis has largely centered on image analysis. While this has led to notable advancements, the absence of patient-reported symptoms continues to hinder diagnostic accuracy. To address this, we propose a Pre-Consultation Dialogue Framework (PCDF) that mimics real-world diagnostic procedures, where doctors iteratively query patients before reaching a conclusion. Specifically, we simulate diagnostic dialogues between two vision-language models (VLMs): a DocVLM, which generates follow-up questions based on the image and dialogue history, and a PatientVLM, which responds using a symptom profile derived from the ground-truth diagnosis. We additionally conducted a small-scale clinical validation of the synthetic symptoms generated by our framework, with licensed clinicians confirming their clinical relevance, symptom coverage, and overall realism. These findings indicate that the resulting DocVLM-PatientVLM interactions form coherent, multi-turn consultations paired with images and diagnoses, which we then use to fine-tune the DocVLM. This dialogue-based supervision leads to substantial gains over image-only training, highlighting the value of realistic symptom elicitation for diagnosis. 

---
# Sparse Data Tree Canopy Segmentation: Fine-Tuning Leading Pretrained Models on Only 150 Images 

**Authors**: David Szczecina, Hudson Sun, Anthony Bertnyk, Niloofar Azad, Kyle Gao, Lincoln Linlin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.10931)  

**Abstract**: Tree canopy detection from aerial imagery is an important task for environmental monitoring, urban planning, and ecosystem analysis. Simulating real-life data annotation scarcity, the Solafune Tree Canopy Detection competition provides a small and imbalanced dataset of only 150 annotated images, posing significant challenges for training deep models without severe overfitting. In this work, we evaluate five representative architectures, YOLOv11, Mask R-CNN, DeepLabv3, Swin-UNet, and DINOv2, to assess their suitability for canopy segmentation under extreme data scarcity. Our experiments show that pretrained convolution-based models, particularly YOLOv11 and Mask R-CNN, generalize significantly better than pretrained transformer-based models. DeeplabV3, Swin-UNet and DINOv2 underperform likely due to differences between semantic and instance segmentation tasks, the high data requirements of Vision Transformers, and the lack of strong inductive biases. These findings confirm that transformer-based architectures struggle in low-data regimes without substantial pretraining or augmentation and that differences between semantic and instance segmentation further affect model performance. We provide a detailed analysis of training strategies, augmentation policies, and model behavior under the small-data constraint and demonstrate that lightweight CNN-based methods remain the most reliable for canopy detection on limited imagery. 

---
# Selecting Language Models for Social Science: Start Small, Start Open, and Validate 

**Authors**: Dustin S. Stoltz, Marshall A. Taylor, Sanuj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.10926)  

**Abstract**: Currently, there are thousands of large pretrained language models (LLMs) available to social scientists. How do we select among them? Using validity, reliability, reproducibility, and replicability as guides, we explore the significance of: (1) model openness, (2) model footprint, (3) training data, and (4) model architectures and fine-tuning. While ex-ante tests of validity (i.e., benchmarks) are often privileged in these discussions, we argue that social scientists cannot altogether avoid validating computational measures (ex-post). Replicability, in particular, is a more pressing guide for selecting language models. Being able to reliably replicate a particular finding that entails the use of a language model necessitates reliably reproducing a task. To this end, we propose starting with smaller, open models, and constructing delimited benchmarks to demonstrate the validity of the entire computational pipeline. 

---
# RobuMTL: Enhancing Multi-Task Learning Robustness Against Weather Conditions 

**Authors**: Tasneem Shaffee, Sherief Reda  

**Link**: [PDF](https://arxiv.org/pdf/2601.10921)  

**Abstract**: Robust Multi-Task Learning (MTL) is crucial for autonomous systems operating in real-world environments, where adverse weather conditions can severely degrade model performance and reliability. In this paper, we introduce RobuMTL, a novel architecture designed to adaptively address visual degradation by dynamically selecting task-specific hierarchical Low-Rank Adaptation (LoRA) modules and a LoRA expert squad based on input perturbations in a mixture-of-experts fashion. Our framework enables adaptive specialization based on input characteristics, improving robustness across diverse real-world conditions. To validate our approach, we evaluated it on the PASCAL and NYUD-v2 datasets and compared it against single-task models, standard MTL baselines, and state-of-the-art methods. On the PASCAL benchmark, RobuMTL delivers a +2.8% average relative improvement under single perturbations and up to +44.4% under mixed weather conditions compared to the MTL baseline. On NYUD-v2, RobuMTL achieves a +9.7% average relative improvement across tasks. The code is available at GitHub. 

---
# Self-learned representation-guided latent diffusion model for breast cancer classification in deep ultraviolet whole surface images 

**Authors**: Pouya Afshin, David Helminiak, Tianling Niu, Julie M. Jorns, Tina Yen, Bing Yu, Dong Hye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.10917)  

**Abstract**: Breast-Conserving Surgery (BCS) requires precise intraoperative margin assessment to preserve healthy tissue. Deep Ultraviolet Fluorescence Scanning Microscopy (DUV-FSM) offers rapid, high-resolution surface imaging for this purpose; however, the scarcity of annotated DUV data hinders the training of robust deep learning models. To address this, we propose an Self-Supervised Learning (SSL)-guided Latent Diffusion Model (LDM) to generate high-quality synthetic training patches. By guiding the LDM with embeddings from a fine-tuned DINO teacher, we inject rich semantic details of cellular structures into the synthetic data. We combine real and synthetic patches to fine-tune a Vision Transformer (ViT), utilizing patch prediction aggregation for WSI-level classification. Experiments using 5-fold cross-validation demonstrate that our method achieves 96.47 % accuracy and reduces the FID score to 45.72, significantly outperforming class-conditioned baselines. 

---
# Medical SAM3: A Foundation Model for Universal Prompt-Driven Medical Image Segmentation 

**Authors**: Chongcong Jiang, Tianxingjian Ding, Chuhan Song, Jiachen Tu, Ziyang Yan, Yihua Shao, Zhenyi Wang, Yuzhang Shang, Tianyu Han, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2601.10880)  

**Abstract**: Promptable segmentation foundation models such as SAM3 have demonstrated strong generalization capabilities through interactive and concept-based prompting. However, their direct applicability to medical image segmentation remains limited by severe domain shifts, the absence of privileged spatial prompts, and the need to reason over complex anatomical and volumetric structures. Here we present Medical SAM3, a foundation model for universal prompt-driven medical image segmentation, obtained by fully fine-tuning SAM3 on large-scale, heterogeneous 2D and 3D medical imaging datasets with paired segmentation masks and text prompts. Through a systematic analysis of vanilla SAM3, we observe that its performance degrades substantially on medical data, with its apparent competitiveness largely relying on strong geometric priors such as ground-truth-derived bounding boxes. These findings motivate full model adaptation beyond prompt engineering alone. By fine-tuning SAM3's model parameters on 33 datasets spanning 10 medical imaging modalities, Medical SAM3 acquires robust domain-specific representations while preserving prompt-driven flexibility. Extensive experiments across organs, imaging modalities, and dimensionalities demonstrate consistent and significant performance gains, particularly in challenging scenarios characterized by semantic ambiguity, complex morphology, and long-range 3D context. Our results establish Medical SAM3 as a universal, text-guided segmentation foundation model for medical imaging and highlight the importance of holistic model adaptation for achieving robust prompt-driven segmentation under severe domain shift. Code and model will be made available at this https URL. 

---
# Can Vision-Language Models Understand Construction Workers? An Exploratory Study 

**Authors**: Hieu Bui, Nathaniel E. Chodosh, Arash Tavakoli  

**Link**: [PDF](https://arxiv.org/pdf/2601.10835)  

**Abstract**: As robotics become increasingly integrated into construction workflows, their ability to interpret and respond to human behavior will be essential for enabling safe and effective collaboration. Vision-Language Models (VLMs) have emerged as a promising tool for visual understanding tasks and offer the potential to recognize human behaviors without extensive domain-specific training. This capability makes them particularly appealing in the construction domain, where labeled data is scarce and monitoring worker actions and emotional states is critical for safety and productivity. In this study, we evaluate the performance of three leading VLMs, GPT-4o, Florence 2, and LLaVa-1.5, in detecting construction worker actions and emotions from static site images. Using a curated dataset of 1,000 images annotated across ten action and ten emotion categories, we assess each model's outputs through standardized inference pipelines and multiple evaluation metrics. GPT-4o consistently achieved the highest scores across both tasks, with an average F1-score of 0.756 and accuracy of 0.799 in action recognition, and an F1-score of 0.712 and accuracy of 0.773 in emotion recognition. Florence 2 performed moderately, with F1-scores of 0.497 for action and 0.414 for emotion, while LLaVa-1.5 showed the lowest overall performance, with F1-scores of 0.466 for action and 0.461 for emotion. Confusion matrix analyses revealed that all models struggled to distinguish semantically close categories, such as collaborating in teams versus communicating with supervisors. While the results indicate that general-purpose VLMs can offer a baseline capability for human behavior recognition in construction environments, further improvements, such as domain adaptation, temporal modeling, or multimodal sensing, may be needed for real-world reliability. 

---
# Approximately Optimal Global Planning for Contact-Rich SE(2) Manipulation on a Graph of Reachable Sets 

**Authors**: Simin Liu, Tong Zhao, Bernhard Paus Graesdal, Peter Werner, Jiuguang Wang, John Dolan, Changliu Liu, Tao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10827)  

**Abstract**: If we consider human manipulation, it is clear that contact-rich manipulation (CRM)-the ability to use any surface of the manipulator to make contact with objects-can be far more efficient and natural than relying solely on end-effectors (i.e., fingertips). However, state-of-the-art model-based planners for CRM are still focused on feasibility rather than optimality, limiting their ability to fully exploit CRM's advantages. We introduce a new paradigm that computes approximately optimal manipulator plans. This approach has two phases. Offline, we construct a graph of mutual reachable sets, where each set contains all object orientations reachable from a starting object orientation and grasp. Online, we plan over this graph, effectively computing and sequencing local plans for globally optimized motion. On a challenging, representative contact-rich task, our approach outperforms a leading planner, reducing task cost by 61%. It also achieves a 91% success rate across 250 queries and maintains sub-minute query times, ultimately demonstrating that globally optimized contact-rich manipulation is now practical for real-world tasks. 

---
# Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents 

**Authors**: Himanshu Thakur, Anusha Kamath, Anurag Muthyala, Dhwani Sanmukhani, Smruthi Mukund, Jay Katukuri  

**Link**: [PDF](https://arxiv.org/pdf/2601.10820)  

**Abstract**: Recent advances in code generation models have unlocked unprecedented opportunities for automating feature engineering, yet their adoption in real-world ML teams remains constrained by critical challenges: (i) the scarcity of datasets capturing the iterative and complex coding processes of production-level feature engineering, (ii) limited integration and personalization of widely used coding agents, such as CoPilot and Devin, with a team's unique tools, codebases, workflows, and practices, and (iii) suboptimal human-AI collaboration due to poorly timed or insufficient feedback. We address these challenges with a planner-guided, constrained-topology multi-agent framework that generates code for repositories in a multi-step fashion. The LLM-powered planner leverages a team's environment, represented as a graph, to orchestrate calls to available agents, generate context-aware prompts, and use downstream failures to retroactively correct upstream artifacts. It can request human intervention at critical steps, ensuring generated code is reliable, maintainable, and aligned with team expectations. On a novel in-house dataset, our approach achieves 38% and 150% improvement in the evaluation metric over manually crafted and unplanned workflows respectively. In practice, when building features for recommendation models serving over 120 million users, our approach has delivered real-world impact by reducing feature engineering cycles from three weeks to a single day. 

---
# Digital Metabolism: Decoupling Logic from Facts via Regenerative Unlearning -- Towards a Pure Neural Logic Core 

**Authors**: Mengmeng Peng, Zhenyu Fang, He Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.10810)  

**Abstract**: Large language models (LLMs) currently suffer from parameter entanglement, where general reasoning capabilities (logic) and specific factual knowledge (facts) exist in a superposition state within shared weights. This coupling leads to the "memory wall," where computational capacity is squandered on simulating retrieval, often resulting in hallucinations. In this paper, we propose "digital metabolism," a thermodynamic hypothesis suggesting that targeted forgetting is necessary for distilling a pure neural logic core. To validate this hypothesis, we introduce the Regenerative Logic-Core Protocol (RLCP), a dual-stream training framework that renders specific factual dependencies linearly undecodable via deep-layer gradient reversal. Applying RLCP to Qwen2.5-0.5B, we observe a distinct phase transition: the model achieves near-zero retention of targeted factual associations (Accuracy < 7%) while exhibiting changes consistent with an emergent "structural crystallization" effect. Empirical analysis on GSM8K reveals that the "metabolized" model spontaneously adopts chain-of-thought (CoT) scaffolding, which we interpret as compensating for the loss of direct associative recall (shifting from $O(1)$ recall to $O(N)$ reasoning). While the causal mechanism underlying this behavioral shift requires further investigation, our findings provide a dynamic weight-level counterpart to architectural innovations like DeepSeek's Engram, paving the way for modular "Neural CPU + Symbolic RAM" architectures. 

---
# Unified Optimization of Source Weights and Transfer Quantities in Multi-Source Transfer Learning: An Asymptotic Framework 

**Authors**: Qingyue Zhang, Chang Chu, Haohao Fu, Tianren Peng, Yanru Wu, Guanbo Huang, Yang Li, Shao-Lun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10779)  

**Abstract**: Transfer learning plays a vital role in improving model performance in data-scarce scenarios. However, naive uniform transfer from multiple source tasks may result in negative transfer, highlighting the need to properly balance the contributions of heterogeneous sources. Moreover, existing transfer learning methods typically focus on optimizing either the source weights or the amount of transferred samples, while largely neglecting the joint consideration of the other. In this work, we propose a theoretical framework, Unified Optimization of Weights and Quantities (UOWQ), which formulates multi-source transfer learning as a parameter estimation problem grounded in an asymptotic analysis of a Kullback-Leibler divergence-based generalization error measure. The proposed framework jointly determines the optimal source weights and optimal transfer quantities for each source task. Firstly, we prove that using all available source samples is always optimal once the weights are properly adjusted, and we provide a theoretical explanation for this phenomenon. Moreover, to determine the optimal transfer weights, our analysis yields closed-form solutions in the single-source setting and develops a convex optimization-based numerical procedure for the multi-source case. Building on the theoretical results, we further propose practical algorithms for both multi-source transfer learning and multi-task learning settings. Extensive experiments on real-world benchmarks, including DomainNet and Office-Home, demonstrate that UOWQ consistently outperforms strong baselines. The results validate both the theoretical predictions and the practical effectiveness of our framework. 

---
# LogicLens: Leveraging Semantic Code Graph to explore Multi Repository large systems 

**Authors**: Niko Usai, Dario Montagnini, Kristian Ilianov Iliev, Raffaele Camanzo  

**Link**: [PDF](https://arxiv.org/pdf/2601.10773)  

**Abstract**: Understanding large software systems is a challenging task, especially when code is distributed across multiple repositories and microservices. Developers often need to reason not only about the structure of the code, but also about its domain logic and runtime behaviors, which are typically implicit and scattered. We introduce LogicLens, a reactive conversational agent that assists developers in exploring complex software systems through a semantic multi-repository graph. This graph is built in a preprocessing step by combining syntactic code analysis, via AST parsing and repository traversal, with semantic enrichment using Large Language Models (LLMs). The resulting graph captures both structural elements, such as files, classes, and functions, as well as functional abstractions like domain entities, operations, and workflows. Once the graph is constructed, LogicLens enables developers to interact with it via natural language, dynamically retrieving relevant subgraphs and answering technical or functional queries. We present the architecture of the system, discuss emergent behaviors, and evaluate its effectiveness on real-world multi-repository scenarios. We demonstrate emergent capabilities including impact analysis and symptom-based debugging that arise naturally from the semantic graph structure. 

---
# Unifying Speech Recognition, Synthesis and Conversion with Autoregressive Transformers 

**Authors**: Runyuan Cai, Yu Lin, Yiming Wang, Chunlin Fu, Xiaodong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.10770)  

**Abstract**: Traditional speech systems typically rely on separate, task-specific models for text-to-speech (TTS), automatic speech recognition (ASR), and voice conversion (VC), resulting in fragmented pipelines that limit scalability, efficiency, and cross-task generalization. In this paper, we present General-Purpose Audio (GPA), a unified audio foundation model that integrates multiple core speech tasks within a single large language model (LLM) architecture. GPA operates on a shared discrete audio token space and supports instruction-driven task induction, enabling a single autoregressive model to flexibly perform TTS, ASR, and VC without architectural modifications. This unified design combines a fully autoregressive formulation over discrete speech tokens, joint multi-task training across speech domains, and a scalable inference pipeline that achieves high concurrency and throughput. The resulting model family supports efficient multi-scale deployment, including a lightweight 0.3B-parameter variant optimized for edge and resource-constrained environments. Together, these design choices demonstrate that a unified autoregressive architecture can achieve competitive performance across diverse speech tasks while remaining viable for low-latency, practical deployment. 

---
# AnyECG: Evolved ECG Foundation Model for Holistic Health Profiling 

**Authors**: Jun Li, Hongling Zhu, Yujie Xiao, Qinghao Zhao, Yalei Ke, Gongzheng Tang, Guangkun Nie, Deyun Zhang, Jin Li, Canqing Yu, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2601.10748)  

**Abstract**: Background: Artificial intelligence enabled electrocardiography (AI-ECG) has demonstrated the ability to detect diverse pathologies, but most existing models focus on single disease identification, neglecting comorbidities and future risk prediction. Although ECGFounder expanded cardiac disease coverage, a holistic health profiling model remains needed.
Methods: We constructed a large multicenter dataset comprising 13.3 million ECGs from 2.98 million patients. Using transfer learning, ECGFounder was fine-tuned to develop AnyECG, a foundation model for holistic health profiling. Performance was evaluated using external validation cohorts and a 10-year longitudinal cohort for current diagnosis, future risk prediction, and comorbidity identification.
Results: AnyECG demonstrated systemic predictive capability across 1172 conditions, achieving an AUROC greater than 0.7 for 306 diseases. The model revealed novel disease associations, robust comorbidity patterns, and future disease risks. Representative examples included high diagnostic performance for hyperparathyroidism (AUROC 0.941), type 2 diabetes (0.803), Crohn disease (0.817), lymphoid leukemia (0.856), and chronic obstructive pulmonary disease (0.773).
Conclusion: The AnyECG foundation model provides substantial evidence that AI-ECG can serve as a systemic tool for concurrent disease detection and long-term risk prediction. 

---
# Line-based Event Preprocessing: Towards Low-Energy Neuromorphic Computer Vision 

**Authors**: Amélie Gruel, Pierre Lewden, Adrien F. Vincent, Sylvain Saïghi  

**Link**: [PDF](https://arxiv.org/pdf/2601.10742)  

**Abstract**: Neuromorphic vision made significant progress in recent years, thanks to the natural match between spiking neural networks and event data in terms of biological inspiration, energy savings, latency and memory use for dynamic visual data processing. However, optimising its energy requirements still remains a challenge within the community, especially for embedded applications. One solution may reside in preprocessing events to optimise data quantity thus lowering the energy cost on neuromorphic hardware, proportional to the number of synaptic operations. To this end, we extend an end-to-end neuromorphic line detection mechanism to introduce line-based event data preprocessing. Our results demonstrate on three benchmark event-based datasets that preprocessing leads to an advantageous trade-off between energy consumption and classification performance. Depending on the line-based preprocessing strategy and the complexity of the classification task, we show that one can maintain or increase the classification accuracy while significantly reducing the theoretical energy consumption. Our approach systematically leads to a significant improvement of the neuromorphic classification efficiency, thus laying the groundwork towards a more frugal neuromorphic computer vision thanks to event preprocessing. 

---
# Neuro-Symbolic Activation Discovery: Transferring Mathematical Structures from Physics to Ecology for Parameter-Efficient Neural Networks 

**Authors**: Anas Hajbi  

**Link**: [PDF](https://arxiv.org/pdf/2601.10740)  

**Abstract**: Modern neural networks rely on generic activation functions (ReLU, GELU, SiLU) that ignore the mathematical structure inherent in scientific data. We propose Neuro-Symbolic Activation Discovery, a framework that uses Genetic Programming to extract interpretable mathematical formulas from data and inject them as custom activation functions. Our key contribution is the discovery of a Geometric Transfer phenomenon: activation functions learned from particle physics data successfully generalize to ecological classification, outperforming standard activations (ReLU, GELU, SiLU) in both accuracy and parameter efficiency. On the Forest Cover dataset, our Hybrid Transfer model achieves 82.4% accuracy with only 5,825 parameters, compared to 83.4% accuracy requiring 31,801 parameters for a conventional heavy network -- a 5.5x parameter reduction with only 1% accuracy loss. We introduce a Parameter Efficiency Score ($E_{param} = AUC / \log_{10}(Params)$) and demonstrate that lightweight hybrid architectures consistently achieve 18-21% higher efficiency than over-parameterized baselines. Crucially, we establish boundary conditions: while Physics to Ecology transfer succeeds (both involve continuous Euclidean measurements), Physics to Text transfer fails (discrete word frequencies require different mathematical structures). Our work opens pathways toward domain-specific activation libraries for efficient scientific machine learning. 

---
# Millimeter-Wave Gesture Recognition in ISAC: Does Reducing Sensing Airtime Hamper Accuracy? 

**Authors**: Jakob Struye, Nabeel Nisar Bhat, Siddhartha Kumar, Mohammad Hossein Moghaddam, Jeroen Famaey  

**Link**: [PDF](https://arxiv.org/pdf/2601.10733)  

**Abstract**: Most Integrated Sensing and Communications (ISAC) systems require dividing airtime across their two modes. However, the specific impact of this decision on sensing performance remains unclear and underexplored. In this paper, we therefore investigate the impact on a gesture recognition system using a Millimeter-Wave (mmWave) ISAC system. With our dataset of power per beam pair gathered with two mmWave devices performing constant beam sweeps while test subjects performed distinct gestures, we train a gesture classifier using Convolutional Neural Networks. We then subsample these measurements, emulating reduced sensing airtime, showing that a sensing airtime of 25 % only reduces classification accuracy by 0.15 percentage points from full-time sensing. Alongside this high-quality sensing at low airtime, mmWave systems are known to provide extremely high data throughputs, making mmWave ISAC a prime enabler for applications such as truly wireless Extended Reality. 

---
# DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion 

**Authors**: Hanlin Zhang, Daxin Tan, Dehua Tao, Xiao Chen, Haochen Tan, Yunhe Li, Yuchen Cao, Jianping Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.09239)  

**Abstract**: Speech tokenizers serve as the cornerstone of discrete Speech Large Language Models (Speech LLMs). Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose DSA-Tokenizer, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. To eliminate rigid length constraints between the two sequences, we introduce a hierarchical Flow-Matching decoder that further improve speech generation quality. Furthermore, We employ a joint reconstruction-recombination training strategy to enforce this separation. DSA-Tokenizer enables high fidelity reconstruction and flexible recombination through robust disentanglement, facilitating controllable generation in speech LLMs. Our analysis highlights disentangled tokenization as a pivotal paradigm for future speech modeling. Audio samples are avaialble at this https URL. The code and model will be made publicly available after the paper has been accepted. 

---
# EvidFuse: Writing-Time Evidence Learning for Consistent Text-Chart Data Reporting 

**Authors**: Huanxiang Lin, Qianyue Wang, Jinwu Hu, Bailin Chen, Qing Du, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.05487)  

**Abstract**: Data-driven reports communicate decision-relevant insights by tightly interleaving narrative text with charts grounded in underlying tables. However, current LLM-based systems typically generate narratives and visualizations in staged pipelines, following either a text-first-graph-second or a graph-first-text-second paradigm. These designs often lead to chart-text inconsistency and insight freezing, where the intermediate evidence space becomes fixed and the model can no longer retrieve or construct new visual evidence as the narrative evolves, resulting in shallow and predefined analysis. To address the limitations, we propose \textbf{EvidFuse}, a training-free multi-agent framework that enables writing-time text-chart interleaved generation for data-driven reports. EvidFuse decouples visualization analysis from long-form drafting via two collaborating components: a \textbf{Data-Augmented Analysis Agent}, equipped with Exploratory Data Analysis (EDA)-derived knowledge and access to raw tables, and a \textbf{Real-Time Evidence Construction Writer} that plans an outline and drafts the report while intermittently issuing fine-grained analysis requests. This design allows visual evidence to be constructed and incorporated exactly when the narrative requires it, directly constraining subsequent claims and enabling on-demand expansion of the evidence space. Experiments demonstrate that EvidFuse attains the top rank in both LLM-as-a-judge and human evaluations on chart quality, chart-text alignment, and report-level usefulness. 

---
# Generative AI Purpose-built for Social and Mental Health: A Real-World Pilot 

**Authors**: Thomas D. Hull, Lizhe Zhang, Patricia A. Arean, Matteo Malgaroli  

**Link**: [PDF](https://arxiv.org/pdf/2511.11689)  

**Abstract**: Generative artificial intelligence (GAI) chatbots built for mental health could deliver safe, personalized, and scalable mental health support. We evaluate a foundation model designed for mental health. Adults completed mental health measures while engaging with the chatbot between May 15, 2025 and September 15, 2025. Users completed an opt-in consent, demographic information, mental health symptoms, social connection, and self-identified goals. Measures were repeated every two weeks up to 6 weeks, and a final follow-up at 10 weeks. Analyses included effect sizes, and growth mixture models to identify participant groups and their characteristic engagement, severity, and demographic factors. Users demonstrated significant reductions in PHQ-9 and GAD-7 that were sustained at follow-up. Significant improvements in Hope, Behavioral Activation, Social Interaction, Loneliness, and Perceived Social Support were observed throughout and maintained at 10 week follow-up. Engagement was high and predicted outcomes. Working alliance was comparable to traditional care and predicted outcomes. Automated safety guardrails functioned as designed, with 76 sessions flagged for risk and all handled according to escalation policies. This single arm naturalistic observational study provides initial evidence that a GAI foundation model for mental health can deliver accessible, engaging, effective, and safe mental health support. These results lend support to findings from early randomized designs and offer promise for future study of mental health GAI in real world settings. 

---
