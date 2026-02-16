# Optimal Take-off under Fuzzy Clearances 

**Authors**: Hugo Henry, Arthur Tsai, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13166)  

**Abstract**: This paper presents a hybrid obstacle avoidance architecture that integrates Optimal Control under clearance with a Fuzzy Rule Based System (FRBS) to enable adaptive constraint handling for unmanned aircraft. Motivated by the limitations of classical optimal control under uncertainty and the need for interpretable decision making in safety critical aviation systems, we design a three stage Takagi Sugeno Kang fuzzy layer that modulates constraint radii, urgency levels, and activation decisions based on regulatory separation minima and airworthiness guidelines from FAA and EASA. These fuzzy-derived clearances are then incorporated as soft constraints into an optimal control problem solved using the FALCON toolbox and IPOPT. The framework aims to reduce unnecessary recomputations by selectively activating obstacle avoidance updates while maintaining compliance with aviation procedures. A proof of concept implementation using a simplified aircraft model demonstrates that the approach can generate optimal trajectories with computation times of 2,3 seconds per iteration in a single threaded MATLAB environment, suggesting feasibility for near real time applications. However, our experiments revealed a critical software incompatibility in the latest versions of FALCON and IPOPT, in which the Lagrangian penalty term remained identically zero, preventing proper constraint enforcement. This behavior was consistent across scenarios and indicates a solver toolbox regression rather than a modeling flaw. Future work includes validating this effect by reverting to earlier software versions, optimizing the fuzzy membership functions using evolutionary methods, and extending the system to higher fidelity aircraft models and stochastic obstacle environments. 

---
# Constrained Assumption-Based Argumentation Frameworks 

**Authors**: Emanuele De Angelis, Fabio Fioravanti, Maria Chiara Meo, Alberto Pettorossi, Maurizio Proietti, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2602.13135)  

**Abstract**: Assumption-based Argumentation (ABA) is a well-established form of structured argumentation. ABA frameworks with an underlying atomic language are widely studied, but their applicability is limited by a representational restriction to ground (variable-free) arguments and attacks built from propositional atoms. In this paper, we lift this restriction and propose a novel notion of constrained ABA (CABA), whose components, as well as arguments built from them, may include constrained variables, ranging over possibly infinite domains. We define non-ground semantics for CABA, in terms of various notions of non-ground attacks. We show that the new semantics conservatively generalise standard ABA semantics. 

---
# Consistency of Large Reasoning Models Under Multi-Turn Attacks 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2602.13093)  

**Abstract**: Large reasoning models with reasoning capabilities achieve state-of-the-art performance on complex tasks, but their robustness under multi-turn adversarial pressure remains underexplored. We evaluate nine frontier reasoning models under adversarial attacks. Our findings reveal that reasoning confers meaningful but incomplete robustness: most reasoning models studied significantly outperform instruction-tuned baselines, yet all exhibit distinct vulnerability profiles, with misleading suggestions universally effective and social pressure showing model-specific efficacy. Through trajectory analysis, we identify five failure modes (Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, and Reasoning Fatigue) with the first two accounting for 50% of failures. We further demonstrate that Confidence-Aware Response Generation (CARG), effective for standard LLMs, fails for reasoning models due to overconfidence induced by extended reasoning traces; counterintuitively, random confidence embedding outperforms targeted extraction. Our results highlight that reasoning capabilities do not automatically confer adversarial robustness and that confidence-based defenses require fundamental redesign for reasoning models. 

---
# Information-theoretic analysis of world models in optimal reward maximizers 

**Authors**: Alfred Harwood, Jose Faustino, Alex Altair  

**Link**: [PDF](https://arxiv.org/pdf/2602.12963)  

**Abstract**: An important question in the field of AI is the extent to which successful behaviour requires an internal representation of the world. In this work, we quantify the amount of information an optimal policy provides about the underlying environment. We consider a Controlled Markov Process (CMP) with $n$ states and $m$ actions, assuming a uniform prior over the space of possible transition dynamics. We prove that observing a deterministic policy that is optimal for any non-constant reward function then conveys exactly $n \log m$ bits of information about the environment. Specifically, we show that the mutual information between the environment and the optimal policy is $n \log m$ bits. This bound holds across a broad class of objectives, including finite-horizon, infinite-horizon discounted, and time-averaged reward maximization. These findings provide a precise information-theoretic lower bound on the "implicit world model'' necessary for optimality. 

---
# BrowseComp-$V^3$: A Visual, Vertical, and Verifiable Benchmark for Multimodal Browsing Agents 

**Authors**: Huanyao Zhang, Jiepeng Zhou, Bo Li, Bowen Zhou, Yanzhe Dan, Haishan Lu, Zhiyong Cao, Jiaoyang Chen, Yuqian Han, Zinan Sheng, Zhengwei Tao, Hao Liang, Jialong Wu, Yang Shi, Yuanpeng He, Jiaye Lin, Qintong Zhang, Guochen Yan, Runhao Zhao, Zhengpin Li, Xiaohan Yu, Lang Mei, Chong Chen, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2602.12876)  

**Abstract**: Multimodal large language models (MLLMs), equipped with increasingly advanced planning and tool-use capabilities, are evolving into autonomous agents capable of performing multimodal web browsing and deep search in open-world environments. However, existing benchmarks for multimodal browsing remain limited in task complexity, evidence accessibility, and evaluation granularity, hindering comprehensive and reproducible assessments of deep search capabilities. To address these limitations, we introduce BrowseComp-$V^3$, a novel benchmark consisting of 300 carefully curated and challenging questions spanning diverse domains. The benchmark emphasizes deep, multi-level, and cross-modal multi-hop reasoning, where critical evidence is interleaved across textual and visual modalities within and across web pages. All supporting evidence is strictly required to be publicly searchable, ensuring fairness and reproducibility. Beyond final-answer accuracy, we incorporate an expert-validated, subgoal-driven process evaluation mechanism that enables fine-grained analysis of intermediate reasoning behaviors and systematic characterization of capability boundaries. In addition, we propose OmniSeeker, a unified multimodal browsing agent framework integrating diverse web search and visual perception tools. Comprehensive experiments demonstrate that even state-of-the-art models achieve only 36% accuracy on our benchmark, revealing critical bottlenecks in multimodal information integration and fine-grained perception. Our results highlight a fundamental gap between current model capabilities and robust multimodal deep search in real-world settings. 

---
# WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning 

**Authors**: Junjie Wang, Zequn Xie, Dan Yang, Jie Feng, Yue Shen, Duolin Sun, Meixiu Long, Yihan Jiao, Zhehao Tan, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12852)  

**Abstract**: Deep Research systems based on web agents have shown strong potential in solving complex information-seeking tasks, yet their search efficiency remains underexplored. We observe that many state-of-the-art open-source web agents rely on long tool-call trajectories with cyclic reasoning loops and exploration of unproductive branches. To address this, we propose WebClipper, a framework that compresses web agent trajectories via graph-based pruning. Concretely, we model the agent's search process as a state graph and cast trajectory optimization as a minimum-necessary Directed Acyclic Graph (DAG) mining problem, yielding pruned trajectories that preserve essential reasoning while eliminating redundant steps. Continued training on these refined trajectories enables the agent to evolve toward more efficient search patterns and reduces tool-call rounds by about 20% while improving accuracy. Furthermore, we introduce a new metric called F-AE Score to measure the model's overall performance in balancing accuracy and efficiency. Experiments demonstrate that WebClipper compresses tool-call rounds under excellent performance, providing practical insight into balancing effectiveness and efficiency in web agent design. 

---
# X-SYS: A Reference Architecture for Interactive Explanation Systems 

**Authors**: Tobias Labarta, Nhi Hoang, Maximilian Dreyer, Jim Berend, Oleg Hein, Jackie Ma, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2602.12748)  

**Abstract**: The explainable AI (XAI) research community has proposed numerous technical methods, yet deploying explainability as systems remains challenging: Interactive explanation systems require both suitable algorithms and system capabilities that maintain explanation usability across repeated queries, evolving models and data, and governance constraints. We argue that operationalizing XAI requires treating explainability as an information systems problem where user interaction demands induce specific system requirements. We introduce X-SYS, a reference architecture for interactive explanation systems, that guides (X)AI researchers, developers and practitioners in connecting interactive explanation user interfaces (XUI) with system capabilities. X-SYS organizes around four quality attributes named STAR (scalability, traceability, responsiveness, and adaptability), and specifies a five-component decomposition (XUI Services, Explanation Services, Model Services, Data Services, Orchestration and Governance). It maps interaction patterns to system capabilities to decouple user interface evolution from backend computation. We implement X-SYS through SemanticLens, a system for semantic search and activation steering in vision-language models. SemanticLens demonstrates how contract-based service boundaries enable independent evolution, offline/online separation ensures responsiveness, and persistent state management supports traceability. Together, this work provides a reusable blueprint and concrete instantiation for interactive explanation systems supporting end-to-end design under operational constraints. 

---
# SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks 

**Authors**: Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianyu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, Han-chung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12670)  

**Abstract**: Agent Skills are structured packages of procedural knowledge that augment LLM agents at inference time. Despite rapid adoption, there is no standard way to measure whether they actually help. We present SkillsBench, a benchmark of 86 tasks across 11 domains paired with curated Skills and deterministic verifiers. Each task is evaluated under three conditions: no Skills, curated Skills, and self-generated Skills. We test 7 agent-model configurations over 7,308 trajectories. Curated Skills raise average pass rate by 16.2 percentage points(pp), but effects vary widely by domain (+4.5pp for Software Engineering to +51.9pp for Healthcare) and 16 of 84 tasks show negative deltas. Self-generated Skills provide no benefit on average, showing that models cannot reliably author the procedural knowledge they benefit from consuming. Focused Skills with 2--3 modules outperform comprehensive documentation, and smaller models with Skills can match larger models without them. 

---
# Evaluating Robustness of Reasoning Models on Parameterized Logical Problems 

**Authors**: Naïm Es-sebbani, Esteban Marquer, Yakoub Salhi, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2602.12665)  

**Abstract**: Logic provides a controlled testbed for evaluating LLM-based reasoners, yet standard SAT-style benchmarks often conflate surface difficulty (length, wording, clause order) with the structural phenomena that actually determine satisfiability. We introduce a diagnostic benchmark for 2-SAT built from parameterized families of structured 2--CNF formulas, where satisfiability is characterized by the implication graph and can be tuned along interpretable axes. Our generators isolate distinct competencies and failure modes: (i) contradiction-cycle UNSAT cores with controllable size and imbalance, (ii) SAT instances with a prescribed fraction of free variables to control solution multiplicity, (iii) planted backbones that modulate propagation, (iv) late bridge clauses that couple otherwise monotone regions to probe sensitivity to ordering and revision, and (v) symmetry/duplication variants that test abstraction under renaming and redundant structure. We evaluate LLM-based reasoners on decision accuracy and assignment validity, and quantify robustness under semantics-preserving perturbations such as clause reordering, filler clauses, and variable renaming. Across models, we observe sharp performance transitions under targeted structural interventions even when surface statistics are held fixed, revealing brittleness regimes that are invisible to aggregate SAT accuracy. 

---
# Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents 

**Authors**: Ruihan Yang, Fanghua Ye, Xiang We, Ruoqing Zhao, Kang Luo, Xinbo Xu, Bo Zhao, Ruotian Ma, Shanyi Wang, Zhaopeng Tu, Xiaolong Li, Deqing Yang, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2602.12662)  

**Abstract**: Large language models (LLMs) are increasingly deployed as autonomous agents for multi-turn decision-making tasks. However, current agents typically rely on fixed cognitive patterns: non-thinking models generate immediate responses, while thinking models engage in deep reasoning uniformly. This rigidity is inefficient for long-horizon tasks, where cognitive demands vary significantly from step to step, with some requiring strategic planning and others only routine execution. In this paper, we introduce CogRouter, a framework that trains agents to dynamically adapt cognitive depth at each step. Grounded in ACT-R theory, we design four hierarchical cognitive levels ranging from instinctive responses to strategic planning. Our two-stage training approach includes Cognition-aware Supervised Fine-tuning (CoSFT) to instill stable level-specific patterns, and Cognition-aware Policy Optimization (CoPO) for step-level credit assignment via confidence-aware advantage reweighting. The key insight is that appropriate cognitive depth should maximize the confidence of the resulting action. Experiments on ALFWorld and ScienceWorld demonstrate that CogRouter achieves state-of-the-art performance with superior efficiency. With Qwen2.5-7B, it reaches an 82.3% success rate, outperforming GPT-4o (+40.3%), OpenAI-o3 (+18.3%), and GRPO (+14.0%), while using 62% fewer tokens. 

---
# AI Agents for Inventory Control: Human-LLM-OR Complementarity 

**Authors**: Jackie Baek, Yaopeng Fu, Will Ma, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2602.12631)  

**Abstract**: Inventory control is a fundamental operations problem in which ordering decisions are traditionally guided by theoretically grounded operations research (OR) algorithms. However, such algorithms often rely on rigid modeling assumptions and can perform poorly when demand distributions shift or relevant contextual information is unavailable. Recent advances in large language models (LLMs) have generated interest in AI agents that can reason flexibly and incorporate rich contextual signals, but it remains unclear how best to incorporate LLM-based methods into traditional decision-making pipelines.
We study how OR algorithms, LLMs, and humans can interact and complement each other in a multi-period inventory control setting. We construct InventoryBench, a benchmark of over 1,000 inventory instances spanning both synthetic and real-world demand data, designed to stress-test decision rules under demand shifts, seasonality, and uncertain lead times. Through this benchmark, we find that OR-augmented LLM methods outperform either method in isolation, suggesting that these methods are complementary rather than substitutes.
We further investigate the role of humans through a controlled classroom experiment that embeds LLM recommendations into a human-in-the-loop decision pipeline. Contrary to prior findings that human-AI collaboration can degrade performance, we show that, on average, human-AI teams achieve higher profits than either humans or AI agents operating alone. Beyond this population-level finding, we formalize an individual-level complementarity effect and derive a distribution-free lower bound on the fraction of individuals who benefit from AI collaboration; empirically, we find this fraction to be substantial. 

---
# GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics 

**Authors**: Modi Jin, Yiming Zhang, Boyuan Sun, Dingwen Zhang, MingMing Cheng, Qibin Hou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12617)  

**Abstract**: This paper presents GeoAgent, a model capable of reasoning closely with humans and deriving fine-grained address conclusions. Previous RL-based methods have achieved breakthroughs in performance and interpretability but still remain concerns because of their reliance on AI-generated chain-of-thought (CoT) data and training strategies, which conflict with geographic characteristics. To address these issues, we first introduce GeoSeek, a new geolocation dataset comprising CoT data annotated by geographic experts and professional players. We further thoroughly explore the inherent characteristics of geographic tasks and propose a geo-similarity reward and a consistency reward assessed by a consistency agent to assist training. This encourages the model to converge towards correct answers from a geographic perspective while ensuring the integrity and consistency of its reasoning process. Experimental results show that GeoAgent outperforms existing methods and a series of general VLLMs across multiple grains, while generating reasoning that closely aligns with humans. 

---
# Can I Have Your Order? Monte-Carlo Tree Search for Slot Filling Ordering in Diffusion Language Models 

**Authors**: Joshua Ong Jun Leang, Yu Zhao, Mihaela Cătălina Stoian, Wenda Li, Shay B. Cohen, Eleonora Giunchiglia  

**Link**: [PDF](https://arxiv.org/pdf/2602.12586)  

**Abstract**: While plan-and-infill decoding in Masked Diffusion Models (MDMs) shows promise for mathematical and code reasoning, performance remains highly sensitive to slot infilling order, often yielding substantial output variance. We introduce McDiffuSE, a framework that formulates slot selection as decision making and optimises infilling orders through Monte Carlo Tree Search (MCTS). McDiffuSE uses look-ahead simulations to evaluate partial completions before commitment, systematically exploring the combinatorial space of generation orders. Experiments show an average improvement of 3.2% over autoregressive baselines and 8.0% over baseline plan-and-infill, with notable gains of 19.5% on MBPP and 4.9% on MATH500. Our analysis reveals that while McDiffuSE predominantly follows sequential ordering, incorporating non-sequential generation is essential for maximising performance. We observe that larger exploration constants, rather than increased simulations, are necessary to overcome model confidence biases and discover effective orderings. These findings establish MCTS-based planning as an effective approach for enhancing generation quality in MDMs. 

---
# To Mix or To Merge: Toward Multi-Domain Reinforcement Learning for Large Language Models 

**Authors**: Haoqing Wang, Xiang Long, Ziheng Li, Yilong Xu, Tingguang Li, Yehui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12566)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) plays a key role in stimulating the explicit reasoning capability of Large Language Models (LLMs). We can achieve expert-level performance in some specific domains via RLVR, such as coding or math. When a general multi-domain expert-level model is required, we need to carefully consider the collaboration of RLVR across different domains. The current state-of-the-art models mainly employ two different training paradigms for multi-domain RLVR: mixed multi-task RLVR and separate RLVR followed by model merging. However, most of the works did not provide a detailed comparison and analysis about these paradigms. To this end, we choose multiple commonly used high-level tasks (e.g., math, coding, science, and instruction following) as our target domains and design extensive qualitative and quantitative experiments using open-source datasets. We find the RLVR across domains exhibits few mutual interferences, and reasoning-intensive domains demonstrate mutually synergistic effects. Furthermore, we analyze the internal mechanisms of mutual gains from the perspectives of weight space geometry, model prediction behavior, and information constraints. This project is named as M2RL that means Mixed multi-task training or separate training followed by model Merging for Reinforcement Learning, and the homepage is at this https URL 

---
# Scaling Web Agent Training through Automatic Data Generation and Fine-grained Evaluation 

**Authors**: Lajanugen Logeswaran, Jaekyeom Kim, Sungryull Sohn, Creighton Glasscock, Honglak Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12544)  

**Abstract**: We present a scalable pipeline for automatically generating high-quality training data for web agents. In particular, a major challenge in identifying high-quality training instances is trajectory evaluation - quantifying how much progress was made towards task completion. We introduce a novel constraint-based evaluation framework that provides fine-grained assessment of progress towards task completion. This enables us to leverage partially successful trajectories, which significantly expands the amount of usable training data. We evaluate our method on a new benchmark we propose called BookingArena, which consists of complex booking tasks across 20 popular websites, and demonstrate that our distilled student model outperforms open-source approaches and matches or exceeds commercial systems, while being a significantly smaller model. Our work addresses the challenge of efficiently creating diverse, realistic web interaction datasets and provides a systematic evaluation methodology for complex structured web tasks. 

---
# Intent-Driven Smart Manufacturing Integrating Knowledge Graphs and Large Language Models 

**Authors**: Takoua Jradi, John Violos, Dimitrios Spatharakis, Lydia Mavraidi, Ioannis Dimolitsas, Aris Leivadeas, Symeon Papavassiliou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12419)  

**Abstract**: The increasing complexity of smart manufacturing environments demands interfaces that can translate high-level human intents into machine-executable actions. This paper presents a unified framework that integrates instruction-tuned Large Language Models (LLMs) with ontology-aligned Knowledge Graphs (KGs) to enable intent-driven interaction in Manufacturing-as-a-Service (MaaS) ecosystems. We fine-tune Mistral-7B-Instruct-V02 on a domain-specific dataset, enabling the translation of natural language intents into structured JSON requirement models. These models are semantically mapped to a Neo4j-based knowledge graph grounded in the ISA-95 standard, ensuring operational alignment with manufacturing processes, resources, and constraints. Our experimental results demonstrate significant performance gains over zero-shot and 3-shots baselines, achieving 89.33\% exact match accuracy and 97.27\% overall accuracy. This work lays the foundation for scalable, explainable, and adaptive human-machine 

---
# Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting 

**Authors**: Siyuan Li, Yunjia Wu, Yiyong Xiao, Pingyang Huang, Peize Li, Ruitong Liu, Yan Wen, Te Sun, Fangyi Pei  

**Link**: [PDF](https://arxiv.org/pdf/2602.12389)  

**Abstract**: Temporal knowledge graph (TKG) forecasting requires predicting future facts by jointly modeling structural dependencies within each snapshot and temporal evolution across snapshots. However, most existing methods are stateless: they recompute entity representations at each timestamp from a limited query window, leading to episodic amnesia and rapid decay of long-term dependencies. To address this limitation, we propose Entity State Tuning (EST), an encoder-agnostic framework that endows TKG forecasters with persistent and continuously evolving entity states. EST maintains a global state buffer and progressively aligns structural evidence with sequential signals via a closed-loop design. Specifically, a topology-aware state perceiver first injects entity-state priors into structural encoding. Then, a unified temporal context module aggregates the state-enhanced events with a pluggable sequence backbone. Subsequently, a dual-track evolution mechanism writes the updated context back to the global entity state memory, balancing plasticity against stability. Experiments on multiple benchmarks show that EST consistently improves diverse backbones and achieves state-of-the-art performance, highlighting the importance of state persistence for long-horizon TKG forecasting. The code is published at this https URL 

---
# A Theoretical Framework for Adaptive Utility-Weighted Benchmarking 

**Authors**: Philip Waggoner  

**Link**: [PDF](https://arxiv.org/pdf/2602.12356)  

**Abstract**: Benchmarking has long served as a foundational practice in machine learning and, increasingly, in modern AI systems such as large language models, where shared tasks, metrics, and leaderboards offer a common basis for measuring progress and comparing approaches. As AI systems are deployed in more varied and consequential settings, though, there is growing value in complementing these established practices with a more holistic conceptualization of what evaluation should represent. Of note, recognizing the sociotechnical contexts in which these systems operate invites an opportunity for a deeper view of how multiple stakeholders and their unique priorities might inform what we consider meaningful or desirable model behavior. This paper introduces a theoretical framework that reconceptualizes benchmarking as a multilayer, adaptive network linking evaluation metrics, model components, and stakeholder groups through weighted interactions. Using conjoint-derived utilities and a human-in-the-loop update rule, we formalize how human tradeoffs can be embedded into benchmark structure and how benchmarks can evolve dynamically while preserving stability and interpretability. The resulting formulation generalizes classical leaderboards as a special case and provides a foundation for building evaluation protocols that are more context aware, resulting in new robust tools for analyzing the structural properties of benchmarks, which opens a path toward more accountable and human-aligned evaluation. 

---
# GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory 

**Authors**: Pepijn Cobben, Xuanqiang Angelo Huang, Thao Amelia Pham, Isabel Dahlgren, Terry Jingchen Zhang, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2602.12316)  

**Abstract**: Frontier AI systems are increasingly capable and deployed in high-stakes multi-agent environments. However, existing AI safety benchmarks largely evaluate single agents, leaving multi-agent risks such as coordination failure and conflict poorly understood. We introduce GT-HarmBench, a benchmark of 2,009 high-stakes scenarios spanning game-theoretic structures such as the Prisoner's Dilemma, Stag Hunt and Chicken. Scenarios are drawn from realistic AI risk contexts in the MIT AI Risk Repository. Across 15 frontier models, agents choose socially beneficial actions in only 62% of cases, frequently leading to harmful outcomes. We measure sensitivity to game-theoretic prompt framing and ordering, and analyze reasoning patterns driving failures. We further show that game-theoretic interventions improve socially beneficial outcomes by up to 18%. Our results highlight substantial reliability gaps and provide a broad standardized testbed for studying alignment in multi-agent environments. The benchmark and code are available at this https URL. 

---
# Semantic Chunking and the Entropy of Natural Language 

**Authors**: Weishun Zhong, Doron Sivan, Tankut Can, Mikhail Katkov, Misha Tsodyks  

**Link**: [PDF](https://arxiv.org/pdf/2602.13194)  

**Abstract**: The entropy rate of printed English is famously estimated to be about one bit per character, a benchmark that modern large language models (LLMs) have only recently approached. This entropy rate implies that English contains nearly 80 percent redundancy relative to the five bits per character expected for random text. We introduce a statistical model that attempts to capture the intricate multi-scale structure of natural language, providing a first-principles account of this redundancy level. Our model describes a procedure of self-similarly segmenting text into semantically coherent chunks down to the single-word level. The semantic structure of the text can then be hierarchically decomposed, allowing for analytical treatment. Numerical experiments with modern LLMs and open datasets suggest that our model quantitatively captures the structure of real texts at different levels of the semantic hierarchy. The entropy rate predicted by our model agrees with the estimated entropy rate of printed English. Moreover, our theory further reveals that the entropy rate of natural language is not fixed but should increase systematically with the semantic complexity of corpora, which are captured by the only free parameter in our model. 

---
# CoPE-VideoLM: Codec Primitives For Efficient Video Language Models 

**Authors**: Sayan Deb Sarkar, Rémi Pautrat, Ondrej Miksik, Marc Pollefeys, Iro Armeni, Mahdi Rad, Mihai Dusmanu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13191)  

**Abstract**: Video Language Models (VideoLMs) empower AI systems to understand temporal dynamics in videos. To fit to the maximum context window constraint, current methods use keyframe sampling which can miss both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. To address these limitations, we propose to leverage video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach reduces the time-to-first-token by up to $86\%$ and token usage by up to $93\%$ compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we are able to maintain or exceed performance on $14$ diverse video understanding benchmarks spanning general question answering, temporal reasoning, long-form understanding, and spatial scene understanding. 

---
# Asynchronous Verified Semantic Caching for Tiered LLM Architectures 

**Authors**: Asmit Kumar Singh, Haozhe Wang, Laxmi Naga Santosh Attaluri, Tak Chiam, Weihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13165)  

**Abstract**: Large language models (LLMs) now sit in the critical path of search, assistance, and agentic workflows, making semantic caching essential for reducing inference cost and latency. Production deployments typically use a tiered static-dynamic design: a static cache of curated, offline vetted responses mined from logs, backed by a dynamic cache populated online. In practice, both tiers are commonly governed by a single embedding similarity threshold, which induces a hard tradeoff: conservative thresholds miss safe reuse opportunities, while aggressive thresholds risk serving semantically incorrect responses. We introduce \textbf{Krites}, an asynchronous, LLM-judged caching policy that expands static coverage without changing serving decisions. On the critical path, Krites behaves exactly like a standard static threshold policy. When the nearest static neighbor of the prompt falls just below the static threshold, Krites asynchronously invokes an LLM judge to verify whether the static response is acceptable for the new prompt. Approved matches are promoted into the dynamic cache, allowing future repeats and paraphrases to reuse curated static answers and expanding static reach over time. In trace-driven simulations on conversational and search workloads, Krites increases the fraction of requests served with curated static answers (direct static hits plus verified promotions) by up to $\textbf{3.9}$ times for conversational traffic and search-style queries relative to tuned baselines, with unchanged critical path latency. 

---
# In-Context Autonomous Network Incident Response: An End-to-End Large Language Model Agent Approach 

**Authors**: Yiran Gao, Kim Hammar, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.13156)  

**Abstract**: Rapidly evolving cyberattacks demand incident response systems that can autonomously learn and adapt to changing threats. Prior work has extensively explored the reinforcement learning approach, which involves learning response strategies through extensive simulation of the incident. While this approach can be effective, it requires handcrafted modeling of the simulator and suppresses useful semantics from raw system logs and alerts. To address these limitations, we propose to leverage large language models' (LLM) pre-trained security knowledge and in-context learning to create an end-to-end agentic solution for incident response planning. Specifically, our agent integrates four functionalities, perception, reasoning, planning, and action, into one lightweight LLM (14b model). Through fine-tuning and chain-of-thought reasoning, our LLM agent is capable of processing system logs and inferring the underlying network state (perception), updating its conjecture of attack models (reasoning), simulating consequences under different response strategies (planning), and generating an effective response (action). By comparing LLM-simulated outcomes with actual observations, the LLM agent repeatedly refines its attack conjecture and corresponding response, thereby demonstrating in-context adaptation. Our agentic approach is free of modeling and can run on commodity hardware. When evaluated on incident logs reported in the literature, our agent achieves recovery up to 23% faster than those of frontier LLMs. 

---
# SCOPE: Selective Conformal Optimized Pairwise LLM Judging 

**Authors**: Sher Badshah, Ali Emami, Hassan Sajjad  

**Link**: [PDF](https://arxiv.org/pdf/2602.13110)  

**Abstract**: Large language models (LLMs) are increasingly used as judges to replace costly human preference labels in pairwise evaluation. Despite their practicality, LLM judges remain prone to miscalibration and systematic biases. This paper proposes SCOPE (Selective Conformal Optimized Pairwise Evaluation), a framework for selective pairwise judging with finite-sample statistical guarantees. Under exchangeability, SCOPE calibrates an acceptance threshold such that the error rate among non-abstained judgments is at most a user-specified level $\alpha$. To provide SCOPE with a bias-neutral uncertainty signal, we introduce Bidirectional Preference Entropy (BPE), which queries the judge under both response positions, aggregates the implied preference probabilities to enforce invariance to response order, and converts the aggregated probability into an entropy-based uncertainty score. Across MT-Bench, RewardBench, and Chatbot Arena, BPE improves uncertainty quality over standard confidence proxies, providing a stronger selection signal that enables SCOPE to consistently meet the target risk level while retaining good coverage across judge scales. In particular, at $\alpha = 0.10$, \textsc{Scope} consistently satisfies the risk bound across all benchmarks and judge scales (empirical risk $\approx 0.097$ to $0.099$), while retaining substantial coverage, reaching $0.89$ on RewardBench with Qwen-14B and $0.98$ on RewardBench with Qwen-32B. Compared to naïve baselines, \textsc{Scope} accepts up to $2.4\times$ more judgments on MT-Bench with Qwen-7B under the same target risk constraint, demonstrating that BPE enables reliable and high-coverage LLM-based evaluation. 

---
# Which Algorithms Can Graph Neural Networks Learn? 

**Authors**: Solveig Wittig, Antonis Vasileiou, Robert R. Nerem, Timo Stoll, Floris Geerts, Yusu Wang, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2602.13106)  

**Abstract**: In recent years, there has been growing interest in understanding neural architectures' ability to learn to execute discrete algorithms, a line of work often referred to as neural algorithmic reasoning. The goal is to integrate algorithmic reasoning capabilities into larger neural pipelines. Many such architectures are based on (message-passing) graph neural networks (MPNNs), owing to their permutation equivariance and ability to deal with sparsity and variable-sized inputs. However, existing work is either largely empirical and lacks formal guarantees or it focuses solely on expressivity, leaving open the question of when and how such architectures generalize beyond a finite training set. In this work, we propose a general theoretical framework that characterizes the sufficient conditions under which MPNNs can learn an algorithm from a training set of small instances and provably approximate its behavior on inputs of arbitrary size. Our framework applies to a broad class of algorithms, including single-source shortest paths, minimum spanning trees, and general dynamic programming problems, such as the $0$-$1$ knapsack problem. In addition, we establish impossibility results for a wide range of algorithmic tasks, showing that standard MPNNs cannot learn them, and we derive more expressive MPNN-like architectures that overcome these limitations. Finally, we refine our analysis for the Bellman-Ford algorithm, yielding a substantially smaller required training set and significantly extending the recent work of Nerem et al. [2025] by allowing for a differentiable regularization loss. Empirical results largely support our theoretical findings. 

---
# How cyborg propaganda reshapes collective action 

**Authors**: Jonas R. Kunst, Kinga Bierwiaczonek, Meeyoung Cha, Omid V. Ebrahimi, Marc Fawcett-Atkinson, Asbjørn Følstad, Anton Gollwitzer, Nils Köbis, Gary Marcus, Jon Roozenbeek, Daniel Thilo Schroeder, Jay J. Van Bavel, Sander van der Linden, Rory White, Live Leonhardsen Wilhelmsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13088)  

**Abstract**: The distinction between genuine grassroots activism and automated influence operations is collapsing. While policy debates focus on bot farms, a distinct threat to democracy is emerging via partisan coordination apps and artificial intelligence-what we term 'cyborg propaganda.' This architecture combines large numbers of verified humans with adaptive algorithmic automation, enabling a closed-loop system. AI tools monitor online sentiment to optimize directives and generate personalized content for users to post online. Cyborg propaganda thereby exploits a critical legal shield: by relying on verified citizens to ratify and disseminate messages, these campaigns operate in a regulatory gray zone, evading liability frameworks designed for automated botnets. We explore the collective action paradox of this technology: does it democratize power by 'unionizing' influence (pooling the reach of dispersed citizens to overcome the algorithmic invisibility of isolated voices), or does it reduce citizens to 'cognitive proxies' of a central directive? We argue that cyborg propaganda fundamentally alters the digital public square, shifting political discourse from a democratic contest of individual ideas to a battle of algorithmic campaigns. We outline a research agenda to distinguish organic from coordinated information diffusion and propose governance frameworks to address the regulatory challenges of AI-assisted collective expression. 

---
# EXCODER: EXplainable Classification Of DiscretE time series Representations 

**Authors**: Yannik Hahn, Antonin Königsfeld, Hasan Tercan, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13087)  

**Abstract**: Deep learning has significantly improved time series classification, yet the lack of explainability in these models remains a major challenge. While Explainable AI (XAI) techniques aim to make model decisions more transparent, their effectiveness is often hindered by the high dimensionality and noise present in raw time series data. In this work, we investigate whether transforming time series into discrete latent representations-using methods such as Vector Quantized Variational Autoencoders (VQ-VAE) and Discrete Variational Autoencoders (DVAE)-not only preserves but enhances explainability by reducing redundancy and focusing on the most informative patterns. We show that applying XAI methods to these compressed representations leads to concise and structured explanations that maintain faithfulness without sacrificing classification performance. Additionally, we propose Similar Subsequence Accuracy (SSA), a novel metric that quantitatively assesses the alignment between XAI-identified salient subsequences and the label distribution in the training data. SSA provides a systematic way to validate whether the features highlighted by XAI methods are truly representative of the learned classification patterns. Our findings demonstrate that discrete latent representations not only retain the essential characteristics needed for classification but also offer a pathway to more compact, interpretable, and computationally efficient explanations in time series analysis. 

---
# Bus-Conditioned Zero-Shot Trajectory Generation via Task Arithmetic 

**Authors**: Shuai Liu, Ning Cao, Yile Chen, Yue Jiang, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2602.13071)  

**Abstract**: Mobility trajectory data provide essential support for smart city applications. However, such data are often difficult to obtain. Meanwhile, most existing trajectory generation methods implicitly assume that at least a subset of real mobility data from target city is available, which limits their applicability in data-inaccessible scenarios. In this work, we propose a new problem setting, called bus-conditioned zero-shot trajectory generation, where no mobility trajectories from a target city are accessible. The generation process relies solely on source city mobility data and publicly available bus timetables from both cities. Under this setting, we propose MobTA, the first approach to introduce task arithmetic into trajectory generation. MobTA models the parameter shift from bus-timetable-based trajectory generation to mobility trajectory generation in source city, and applies this shift to target city through arithmetic operations on task vectors. This enables trajectory generation that reflects target-city mobility patterns without requiring any real mobility data from it. Furthermore, we theoretically analyze MobTA's stability across base and instruction-tuned LLMs. Extensive experiments show that MobTA significantly outperforms existing methods, and achieves performance close to models finetuned using target city mobility trajectories. 

---
# Diverging Flows: Detecting Extrapolations in Conditional Generation 

**Authors**: Constantinos Tsakonas, Serena Ivaldi, Jean-Baptiste Mouret  

**Link**: [PDF](https://arxiv.org/pdf/2602.13061)  

**Abstract**: The ability of Flow Matching (FM) to model complex conditional distributions has established it as the state-of-the-art for prediction tasks (e.g., robotics, weather forecasting). However, deployment in safety-critical settings is hindered by a critical extrapolation hazard: driven by smoothness biases, flow models yield plausible outputs even for off-manifold conditions, resulting in silent failures indistinguishable from valid predictions. In this work, we introduce Diverging Flows, a novel approach that enables a single model to simultaneously perform conditional generation and native extrapolation detection by structurally enforcing inefficient transport for off-manifold inputs. We evaluate our method on synthetic manifolds, cross-domain style transfer, and weather temperature forecasting, demonstrating that it achieves effective detection of extrapolations without compromising predictive fidelity or inference latency. These results establish Diverging Flows as a robust solution for trustworthy flow models, paving the way for reliable deployment in domains such as medicine, robotics, and climate science. 

---
# Curriculum-DPO++: Direct Preference Optimization via Data and Model Curricula for Text-to-Image Generation 

**Authors**: Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Nicu Sebe, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2602.13055)  

**Abstract**: Direct Preference Optimization (DPO) has been proposed as an effective and efficient alternative to reinforcement learning from human feedback (RLHF). However, neither RLHF nor DPO take into account the fact that learning certain preferences is more difficult than learning other preferences, rendering the optimization process suboptimal. To address this gap in text-to-image generation, we recently proposed Curriculum-DPO, a method that organizes image pairs by difficulty. In this paper, we introduce Curriculum-DPO++, an enhanced method that combines the original data-level curriculum with a novel model-level curriculum. More precisely, we propose to dynamically increase the learning capacity of the denoising network as training advances. We implement this capacity increase via two mechanisms. First, we initialize the model with only a subset of the trainable layers used in the original Curriculum-DPO. As training progresses, we sequentially unfreeze layers until the configuration matches the full baseline architecture. Second, as the fine-tuning is based on Low-Rank Adaptation (LoRA), we implement a progressive schedule for the dimension of the low-rank matrices. Instead of maintaining a fixed capacity, we initialize the low-rank matrices with a dimension significantly smaller than that of the baseline. As training proceeds, we incrementally increase their rank, allowing the capacity to grow until it converges to the same rank value as in Curriculum-DPO. Furthermore, we propose an alternative ranking strategy to the one employed by Curriculum-DPO. Finally, we compare Curriculum-DPO++ against Curriculum-DPO and other state-of-the-art preference optimization approaches on nine benchmarks, outperforming the competing methods in terms of text alignment, aesthetics and human preference. Our code is available at this https URL. 

---
# Can we trust AI to detect healthy multilingual English speakers among the cognitively impaired cohort in the UK? An investigation using real-world conversational speech 

**Authors**: Madhurananda Pahar, Caitlin Illingworth, Dorota Braun, Bahman Mirheidari, Lise Sproson, Daniel Blackburn, Heidi Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2602.13047)  

**Abstract**: Conversational speech often reveals early signs of cognitive decline, such as dementia and MCI. In the UK, one in four people belongs to an ethnic minority, and dementia prevalence is expected to rise most rapidly among Black and Asian communities. This study examines the trustworthiness of AI models, specifically the presence of bias, in detecting healthy multilingual English speakers among the cognitively impaired cohort, to make these tools clinically beneficial. For experiments, monolingual participants were recruited nationally (UK), and multilingual speakers were enrolled from four community centres in Sheffield and Bradford. In addition to a non-native English accent, multilinguals spoke Somali, Chinese, or South Asian languages, who were further divided into two Yorkshire accents (West and South) to challenge the efficiency of the AI tools thoroughly. Although ASR systems showed no significant bias across groups, classification and regression models using acoustic and linguistic features exhibited bias against multilingual speakers, particularly in memory, fluency, and reading tasks. This bias was more pronounced when models were trained on the publicly available DementiaBank dataset. Moreover, multilinguals were more likely to be misclassified as having cognitive decline. This study is the first of its kind to discover that, despite their strong overall performance, current AI models show bias against multilingual individuals from ethnic minority backgrounds in the UK, and they are also more likely to misclassify speakers with a certain accent (South Yorkshire) as living with a more severe cognitive decline. In this pilot study, we conclude that the existing AI tools are therefore not yet reliable for diagnostic use in these populations, and we aim to address this in future work by developing more generalisable, bias-mitigated models. 

---
# Geometric Manifold Rectification for Imbalanced Learning 

**Authors**: Xubin Wang, Qing Li, Weijia Jia  

**Link**: [PDF](https://arxiv.org/pdf/2602.13045)  

**Abstract**: Imbalanced classification presents a formidable challenge in machine learning, particularly when tabular datasets are plagued by noise and overlapping class boundaries. From a geometric perspective, the core difficulty lies in the topological intrusion of the majority class into the minority manifold, which obscures the true decision boundary. Traditional undersampling techniques, such as Edited Nearest Neighbours (ENN), typically employ symmetric cleaning rules and uniform voting, failing to capture the local manifold structure and often inadvertently removing informative minority samples. In this paper, we propose GMR (Geometric Manifold Rectification), a novel framework designed to robustly handle imbalanced structured data by exploiting local geometric priors. GMR makes two contributions: (1) Geometric confidence estimation that uses inverse-distance weighted kNN voting with an adaptive distance metric to capture local reliability; and (2) asymmetric cleaning that is strict on majority samples while conservatively protecting minority samples via a safe-guarding cap on minority removal. Extensive experiments on multiple benchmark datasets show that GMR is competitive with strong sampling baselines. 

---
# Look Inward to Explore Outward: Learning Temperature Policy from LLM Internal States via Hierarchical RL 

**Authors**: Yixiao Zhou, Yang Li, Dongzhou Cheng, Hehe Fan, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.13035)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) trains large language models (LLMs) from sampled trajectories, making decoding strategy a core component of learning rather than a purely inference-time choice. Sampling temperature directly controls the exploration--exploitation trade-off by modulating policy entropy, yet existing methods rely on static values or heuristic adaptations that are decoupled from task-level rewards. We propose Introspective LLM, a hierarchical reinforcement learning framework that learns to control sampling temperature during generation. At each decoding step, the model selects a temperature based on its hidden state and samples the next token from the resulting distribution. Temperature and token policies are jointly optimized from downstream rewards using a coordinate ascent scheme. Experiments on mathematical reasoning benchmarks show that learned temperature policies outperform fixed and heuristic baselines, while exhibiting interpretable exploration behaviors aligned with reasoning uncertainty. 

---
# Buy versus Build an LLM: A Decision Framework for Governments 

**Authors**: Jiahao Lu, Ziwei Xu, William Tjhi, Junnan Li, Antoine Bosselut, Pang Wei Koh, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2602.13033)  

**Abstract**: Large Language Models (LLMs) represent a new frontier of digital infrastructure that can support a wide range of public-sector applications, from general purpose citizen services to specialized and sensitive state functions. When expanding AI access, governments face a set of strategic choices over whether to buy existing services, build domestic capabilities, or adopt hybrid approaches across different domains and use cases. These are critical decisions especially when leading model providers are often foreign corporations, and LLM outputs are increasingly treated as trusted inputs to public decision-making and public discourse. In practice, these decisions are not intended to mandate a single approach across all domains; instead, national AI strategies are typically pluralistic, with sovereign, commercial and open-source models coexisting to serve different purposes. Governments may rely on commercial models for non-sensitive or commodity tasks, while pursuing greater control for critical, high-risk or strategically important applications.
This paper provides a strategic framework for making this decision by evaluating these options across dimensions including sovereignty, safety, cost, resource capability, cultural fit, and sustainability. Importantly, "building" does not imply that governments must act alone: domestic capabilities may be developed through public research institutions, universities, state-owned enterprises, joint ventures, or broader national ecosystems. By detailing the technical requirements and practical challenges of each pathway, this work aims to serve as a reference for policy-makers to determine whether a buy or build approach best aligns with their specific national needs and societal goals. 

---
# Prior-Guided Symbolic Regression: Towards Scientific Consistency in Equation Discovery 

**Authors**: Jing Xiao, Xinhai Chen, Jiaming Peng, Qinglin Wang, Menghan Jia, Zhiquan Lai, Guangping Yu, Dongsheng Li, Tiejun Li, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13021)  

**Abstract**: Symbolic Regression (SR) aims to discover interpretable equations from observational data, with the potential to reveal underlying principles behind natural phenomena. However, existing approaches often fall into the Pseudo-Equation Trap: producing equations that fit observations well but remain inconsistent with fundamental scientific principles. A key reason is that these approaches are dominated by empirical risk minimization, lacking explicit constraints to ensure scientific consistency. To bridge this gap, we propose PG-SR, a prior-guided SR framework built upon a three-stage pipeline consisting of warm-up, evolution, and refinement. Throughout the pipeline, PG-SR introduces a prior constraint checker that explicitly encodes domain priors as executable constraint programs, and employs a Prior Annealing Constrained Evaluation (PACE) mechanism during the evolution stage to progressively steer discovery toward scientifically consistent regions. Theoretically, we prove that PG-SR reduces the Rademacher complexity of the hypothesis space, yielding tighter generalization bounds and establishing a guarantee against pseudo-equations. Experimentally, PG-SR outperforms state-of-the-art baselines across diverse domains, maintaining robustness to varying prior quality, noisy data, and data scarcity. 

---
# Synaptic Activation and Dual Liquid Dynamics for Interpretable Bio-Inspired Models 

**Authors**: Mónika Farsang, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2602.13017)  

**Abstract**: In this paper, we present a unified framework for various bio-inspired models to better understand their structural and functional differences. We show that liquid-capacitance-extended models lead to interpretable behavior even in dense, all-to-all recurrent neural network (RNN) policies. We further demonstrate that incorporating chemical synapses improves interpretability and that combining chemical synapses with synaptic activation yields the most accurate and interpretable RNN models. To assess the accuracy and interpretability of these RNN policies, we consider the challenging lane-keeping control task and evaluate performance across multiple metrics, including turn-weighted validation loss, neural activity during driving, absolute correlation between neural activity and road trajectory, saliency maps of the networks' attention, and the robustness of their saliency maps measured by the structural similarity index. 

---
# Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in Large Language Models 

**Authors**: Hao Chen, Ye He, Yuchun Fan, Yukun Yan, Zhenghao Liu, Qingfu Zhu, Maosong Sun, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2602.12996)  

**Abstract**: Knowledge augmentation has significantly enhanced the performance of Large Language Models (LLMs) in knowledge-intensive tasks. However, existing methods typically operate on the simplistic premise that model performance equates with internal knowledge, overlooking the knowledge-confidence gaps that lead to overconfident errors or uncertain truths. To bridge this gap, we propose a novel meta-cognitive framework for reliable knowledge augmentation via differentiated intervention and alignment. Our approach leverages internal cognitive signals to partition the knowledge space into mastered, confused, and missing regions, guiding targeted knowledge expansion. Furthermore, we introduce a cognitive consistency mechanism to synchronize subjective certainty with objective accuracy, ensuring calibrated knowledge boundaries. Extensive experiments demonstrate the our framework consistently outperforms strong baselines, validating its rationality in not only enhancing knowledge capabilities but also fostering cognitive behaviors that better distinguish knowns from unknowns. 

---
# Detecting Object Tracking Failure via Sequential Hypothesis Testing 

**Authors**: Alejandro Monroy Muñoz, Rajeev Verma, Alexander Timans  

**Link**: [PDF](https://arxiv.org/pdf/2602.12983)  

**Abstract**: Real-time online object tracking in videos constitutes a core task in computer vision, with wide-ranging applications including video surveillance, motion capture, and robotics. Deployed tracking systems usually lack formal safety assurances to convey when tracking is reliable and when it may fail, at best relying on heuristic measures of model confidence to raise alerts. To obtain such assurances we propose interpreting object tracking as a sequential hypothesis test, wherein evidence for or against tracking failures is gradually accumulated over time. Leveraging recent advancements in the field, our sequential test (formalized as an e-process) quickly identifies when tracking failures set in whilst provably containing false alerts at a desired rate, and thus limiting potentially costly re-calibration or intervention steps. The approach is computationally light-weight, requires no extra training or fine-tuning, and is in principle model-agnostic. We propose both supervised and unsupervised variants by leveraging either ground-truth or solely internal tracking information, and demonstrate its effectiveness for two established tracking models across four video benchmarks. As such, sequential testing can offer a statistically grounded and efficient mechanism to incorporate safety assurances into real-time tracking systems. 

---
# Learning Native Continuation for Action Chunking Flow Policies 

**Authors**: Yufeng Liu, Hang Yu, Juntu Zhao, Bocheng Li, Di Zhang, Mingzhu Li, Wenxuan Wu, Yingdong Hu, Junyuan Xie, Junliang Guo, Dequan Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12978)  

**Abstract**: Action chunking enables Vision Language Action (VLA) models to run in real time, but naive chunked execution often exhibits discontinuities at chunk boundaries. Real-Time Chunking (RTC) alleviates this issue but is external to the policy, leading to spurious multimodal switching and trajectories that are not intrinsically smooth. We propose Legato, a training-time continuation method for action-chunked flow-based VLA policies. Specifically, Legato initializes denoising from a schedule-shaped mixture of known actions and noise, exposing the model to partial action information. Moreover, Legato reshapes the learned flow dynamics to ensure that the denoising process remains consistent between training and inference under per-step guidance. Legato further uses randomized schedule condition during training to support varying inference delays and achieve controllable smoothness. Empirically, Legato produces smoother trajectories and reduces spurious multimodal switching during execution, leading to less hesitation and shorter task completion time. Extensive real-world experiments show that Legato consistently outperforms RTC across five manipulation tasks, achieving approximately 10% improvements in both trajectory smoothness and task completion time. 

---
# Drift-Aware Variational Autoencoder-based Anomaly Detection with Two-level Ensembling 

**Authors**: Jin Li, Kleanthis Malialis, Christos G. Panayiotou, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12976)  

**Abstract**: In today's digital world, the generation of vast amounts of streaming data in various domains has become ubiquitous. However, many of these data are unlabeled, making it challenging to identify events, particularly anomalies. This task becomes even more formidable in nonstationary environments where model performance can deteriorate over time due to concept drift. To address these challenges, this paper presents a novel method, VAE++ESDD, which employs incremental learning and two-level ensembling: an ensemble of Variational AutoEncoder(VAEs) for anomaly prediction, along with an ensemble of concept drift detectors. Each drift detector utilizes a statistical-based concept drift mechanism. To evaluate the effectiveness of VAE++ESDD, we conduct a comprehensive experimental study using real-world and synthetic datasets characterized by severely or extremely low anomalous rates and various drift characteristics. Our study reveals that the proposed method significantly outperforms both strong baselines and state-of-the-art methods. 

---
# Extending confidence calibration to generalised measures of variation 

**Authors**: Andrew Thompson, Vivek Desai  

**Link**: [PDF](https://arxiv.org/pdf/2602.12975)  

**Abstract**: We propose the Variation Calibration Error (VCE) metric for assessing the calibration of machine learning classifiers. The metric can be viewed as an extension of the well-known Expected Calibration Error (ECE) which assesses the calibration of the maximum probability or confidence. Other ways of measuring the variation of a probability distribution exist which have the advantage of taking into account the full probability distribution, for example the Shannon entropy. We show how the ECE approach can be extended from assessing confidence calibration to assessing the calibration of any metric of variation. We present numerical examples upon synthetic predictions which are perfectly calibrated by design, demonstrating that, in this scenario, the VCE has the desired property of approaching zero as the number of data samples increases, in contrast to another entropy-based calibration metric (the UCE) which has been proposed in the literature. 

---
# RGAlign-Rec: Ranking-Guided Alignment for Latent Query Reasoning in Recommendation Systems 

**Authors**: Junhua Liu, Yang Jihao, Cheng Chang, Kunrong LI, Bin Fu, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2602.12968)  

**Abstract**: Proactive intent prediction is a critical capability in modern e-commerce chatbots, enabling "zero-query" recommendations by anticipating user needs from behavioral and contextual signals. However, existing industrial systems face two fundamental challenges: (1) the semantic gap between discrete user features and the semantic intents within the chatbot's Knowledge Base, and (2) the objective misalignment between general-purpose LLM outputs and task-specific ranking utilities. To address these issues, we propose RGAlign-Rec, a closed-loop alignment framework that integrates an LLM-based semantic reasoner with a Query-Enhanced (QE) ranking model. We also introduce Ranking-Guided Alignment (RGA), a multi-stage training paradigm that utilizes downstream ranking signals as feedback to refine the LLM's latent reasoning. Extensive experiments on a large-scale industrial dataset from Shopee demonstrate that RGAlign-Rec achieves a 0.12% gain in GAUC, leading to a significant 3.52% relative reduction in error rate, and a 0.56% improvement in Recall@3. Online A/B testing further validates the cumulative effectiveness of our framework: the Query-Enhanced model (QE-Rec) initially yields a 0.98% improvement in CTR, while the subsequent Ranking-Guided Alignment stage contributes an additional 0.13% gain. These results indicate that ranking-aware alignment effectively synchronizes semantic reasoning with ranking objectives, significantly enhancing both prediction accuracy and service quality in real-world proactive recommendation systems. 

---
# TriGen: NPU Architecture for End-to-End Acceleration of Large Language Models based on SW-HW Co-Design 

**Authors**: Jonghun Lee, Junghoon Lee, Hyeonjin Kim, Seoho Jeon, Jisup Yoon, Hyunbin Park, Meejeong Park, Heonjae Ha  

**Link**: [PDF](https://arxiv.org/pdf/2602.12962)  

**Abstract**: Recent studies have extensively explored NPU architectures for accelerating AI inference in on-device environments, which are inherently resource-constrained. Meanwhile, transformer-based large language models (LLMs) have become dominant, with rapidly increasing model sizes but low degree of parameter reuse compared to conventional CNNs, making end-to-end execution on resource-limited devices extremely challenging. To address these challenges, we propose TriGen, a novel NPU architecture tailored for resource-constrained environments through software-hardware co-design. Firstly, TriGen adopts low-precision computation using microscaling (MX) to enable additional optimization opportunities while preserving accuracy, and resolves the issues that arise by employing such precision. Secondly, to jointly optimize both nonlinear and linear operations, TriGen eliminates the need for specialized hardware for essential nonlinear operations by using fast and accurate LUT, thereby maximizing performance gains and reducing hardware-cost in on-device environments, and finally, by taking practical hardware constraints into account, further employs scheduling techniques to maximize computational utilization even under limited on-chip memory capacity. We evaluate the performance of TriGen on various LLMs and show that TriGen achieves an average 2.73x performance speedup and 52% less memory transfer over the baseline NPU design with negligible accuracy loss. 

---
# Transporting Task Vectors across Different Architectures without Training 

**Authors**: Filippo Rinaldi, Aniello Panariello, Giacomo Salici, Angelo Porrello, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2602.12952)  

**Abstract**: Adapting large pre-trained models to downstream tasks often produces task-specific parameter updates that are expensive to relearn for every model variant. While recent work has shown that such updates can be transferred between models with identical architectures, transferring them across models of different widths remains largely unexplored. In this work, we introduce Theseus, a training-free method for transporting task-specific updates across heterogeneous models. Rather than matching parameters directly, we characterize a task update by the functional effect it induces on intermediate representations. We formalize task-vector transport as a functional matching problem on observed activations and show that, after aligning representation spaces via orthogonal Procrustes analysis, it admits a stable closed-form solution that preserves the geometry of the update. We evaluate Theseus on vision and language models across different widths, showing consistent improvements over strong baselines without additional training or backpropagation. Our results show that task updates can be meaningfully transferred across architectures when task identity is defined functionally rather than parametrically. 

---
# Deep-Learning Atlas Registration for Melanoma Brain Metastases: Preserving Pathology While Enabling Cohort-Level Analyses 

**Authors**: Nanna E. Wielenberg, Ilinca Popp, Oliver Blanck, Lucas Zander, Jan C. Peeken, Stephanie E. Combs, Anca-Ligia Grosu, Dimos Baltas, Tobias Fechter  

**Link**: [PDF](https://arxiv.org/pdf/2602.12933)  

**Abstract**: Melanoma brain metastases (MBM) are common and spatially heterogeneous lesions, complicating cohort-level analyses due to anatomical variability and differing MRI protocols. We propose a fully differentiable, deep-learning-based deformable registration framework that aligns individual pathological brains to a common atlas while preserving metastatic tissue without requiring lesion masks or preprocessing.
Missing anatomical correspondences caused by metastases are handled through a forward-model similarity metric based on distance-transformed anatomical labels, combined with a volume-preserving regularization term to ensure deformation plausibility. Registration performance was evaluated using Dice coefficient (DSC), Hausdorff distance (HD), average symmetric surface distance (ASSD), and Jacobian-based measures. The method was applied to 209 MBM patients from three centres, enabling standardized mapping of metastases to anatomical, arterial, and perfusion atlases.
The framework achieved high registration accuracy across datasets (DSC 0.89-0.92, HD 6.79-7.60 mm, ASSD 0.63-0.77 mm) while preserving metastatic volumes. Spatial analysis demonstrated significant over-representation of MBM in the cerebral cortex and putamen, under-representation in white matter, and consistent localization near the gray-white matter junction. No arterial territory showed increased metastasis frequency after volume correction.
This approach enables robust atlas registration of pathological brain MRI without lesion masks and supports reproducible multi-centre analyses. Applied to MBM, it confirms and refines known spatial predilections, particularly preferential seeding near the gray-white matter junction and cortical regions. The publicly available implementation facilitates reproducible research and extension to other brain tumours and neurological pathologies. 

---
# Never say never: Exploring the effects of available knowledge on agent persuasiveness in controlled physiotherapy motivation dialogues 

**Authors**: Stephan Vonschallen, Rahel Häusler, Theresa Schmiedel, Friederike Eyssel  

**Link**: [PDF](https://arxiv.org/pdf/2602.12924)  

**Abstract**: Generative Social Agents (GSAs) are increasingly impacting human users through persuasive means. On the one hand, they might motivate users to pursue personal goals, such as healthier lifestyles. On the other hand, they are associated with potential risks like manipulation and deception, which are induced by limited control over probabilistic agent outputs. However, as GSAs manifest communicative patterns based on available knowledge, their behavior may be regulated through their access to such knowledge. Following this approach, we explored persuasive ChatGPT-generated messages in the context of human-robot physiotherapy motivation. We did so by comparing ChatGPT-generated responses to predefined inputs from a hypothetical physiotherapy patient. In Study 1, we qualitatively analyzed 13 ChatGPT-generated dialogue scripts with varying knowledge configurations regarding persuasive message characteristics. In Study 2, third-party observers (N = 27) rated a selection of these dialogues in terms of the agent's expressiveness, assertiveness, and persuasiveness. Our findings indicate that LLM-based GSAs can adapt assertive and expressive personality traits -- significantly enhancing perceived persuasiveness. Moreover, persuasiveness significantly benefited from the availability of information about the patients' age and past profession, mediated by perceived assertiveness and expressiveness. Contextual knowledge about physiotherapy benefits did not significantly impact persuasiveness, possibly because the LLM had inherent knowledge about such benefits even without explicit prompting. Overall, the study highlights the importance of empirically studying behavioral patterns of GSAs, specifically in terms of what information generative AI systems require for consistent and responsible communication. 

---
# EPRBench: A High-Quality Benchmark Dataset for Event Stream Based Visual Place Recognition 

**Authors**: Xiao Wang, Xingxing Xiong, Jinfeng Gao, Xufeng Lou, Bo Jiang, Si-bao Chen, Yaowei Wang, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2602.12919)  

**Abstract**: Event stream-based Visual Place Recognition (VPR) is an emerging research direction that offers a compelling solution to the instability of conventional visible-light cameras under challenging conditions such as low illumination, overexposure, and high-speed motion. Recognizing the current scarcity of dedicated datasets in this domain, we introduce EPRBench, a high-quality benchmark specifically designed for event stream-based VPR. EPRBench comprises 10K event sequences and 65K event frames, collected using both handheld and vehicle-mounted setups to comprehensively capture real-world challenges across diverse viewpoints, weather conditions, and lighting scenarios. To support semantic-aware and language-integrated VPR research, we provide LLM-generated scene descriptions, subsequently refined through human annotation, establishing a solid foundation for integrating LLMs into event-based perception pipelines. To facilitate systematic evaluation, we implement and benchmark 15 state-of-the-art VPR algorithms on EPRBench, offering a strong baseline for future algorithmic comparisons. Furthermore, we propose a novel multi-modal fusion paradigm for VPR: leveraging LLMs to generate textual scene descriptions from raw event streams, which then guide spatially attentive token selection, cross-modal feature fusion, and multi-scale representation learning. This framework not only achieves highly accurate place recognition but also produces interpretable reasoning processes alongside its predictions, significantly enhancing model transparency and explainability. The dataset and source code will be released on this https URL 

---
# Ultrasound-Guided Real-Time Spinal Motion Visualization for Spinal Instability Assessment 

**Authors**: Feng Li, Yuan Bi, Tianyu Song, Zhongliang Jiang, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2602.12917)  

**Abstract**: Purpose: Spinal instability is a widespread condition that causes pain, fatigue, and restricted mobility, profoundly affecting patients' quality of life. In clinical practice, the gold standard for diagnosis is dynamic X-ray imaging. However, X-ray provides only 2D motion information, while 3D modalities such as computed tomography (CT) or cone beam computed tomography (CBCT) cannot efficiently capture motion. Therefore, there is a need for a system capable of visualizing real-time 3D spinal motion while minimizing radiation exposure.
Methods: We propose ultrasound as an auxiliary modality for 3D spine visualization. Due to acoustic limitations, ultrasound captures only the superficial spinal surface. Therefore, the partially compounded ultrasound volume is registered to preoperative 3D imaging. In this study, CBCT provides the neutral spine configuration, while robotic ultrasound acquisition is performed at maximal spinal bending. A kinematic model is applied to the CBCT-derived spine model for coarse registration, followed by ICP for fine registration, with kinematic parameters optimized based on the registration results. Real-time ultrasound motion tracking is then used to estimate continuous 3D spinal motion by interpolating between the neutral and maximally bent states.
Results: The pipeline was evaluated on a bendable 3D-printed lumbar spine phantom. The registration error was $1.941 \pm 0.199$ mm and the interpolated spinal motion error was $2.01 \pm 0.309$ mm (median).
Conclusion: The proposed robotic ultrasound framework enables radiation-reduced, real-time 3D visualization of spinal motion, offering a promising 3D alternative to conventional dynamic X-ray imaging for assessing spinal instability. 

---
# Robustness of Object Detection of Autonomous Vehicles in Adverse Weather Conditions 

**Authors**: Fox Pettersen, Hong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12902)  

**Abstract**: As self-driving technology advances toward widespread adoption, determining safe operational thresholds across varying environmental conditions becomes critical for public safety. This paper proposes a method for evaluating the robustness of object detection ML models in autonomous vehicles under adverse weather conditions. It employs data augmentation operators to generate synthetic data that simulates different severance degrees of the adverse operation conditions at progressive intensity levels to find the lowest intensity of the adverse conditions at which the object detection model fails. The robustness of the object detection model is measured by the average first failure coefficients (AFFC) over the input images in the benchmark. The paper reports an experiment with four object detection models: YOLOv5s, YOLOv11s, Faster R-CNN, and Detectron2, utilising seven data augmentation operators that simulate weather conditions fog, rain, and snow, and lighting conditions of dark, bright, flaring, and shadow. The experiment data show that the method is feasible, effective, and efficient to evaluate and compare the robustness of object detection models in various adverse operation conditions. In particular, the Faster R-CNN model achieved the highest robustness with an overall average AFFC of 71.9% over all seven adverse conditions, while YOLO variants showed the AFFC values of 43%. The method is also applied to assess the impact of model training that targets adverse operation conditions using synthetic data on model robustness. It is observed that such training can improve robustness in adverse conditions but may suffer from diminishing returns and forgetting phenomena (i.e., decline in robustness) if overtrained. 

---
# RADAR: Revealing Asymmetric Development of Abilities in MLLM Pre-training 

**Authors**: Yunshuang Nie, Bingqian Lin, Minzhe Niu, Kun Xiang, Jianhua Han, Guowei Huang, Xingyue Quan, Hang Xu, Bokui Chen, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12892)  

**Abstract**: Pre-trained Multi-modal Large Language Models (MLLMs) provide a knowledge-rich foundation for post-training by leveraging their inherent perception and reasoning capabilities to solve complex tasks. However, the lack of an efficient evaluation framework impedes the diagnosis of their performance bottlenecks. Current evaluation primarily relies on testing after supervised fine-tuning, which introduces laborious additional training and autoregressive decoding costs. Meanwhile, common pre-training metrics cannot quantify a model's perception and reasoning abilities in a disentangled manner. Furthermore, existing evaluation benchmarks are typically limited in scale or misaligned with pre-training objectives. Thus, we propose RADAR, an efficient ability-centric evaluation framework for Revealing Asymmetric Development of Abilities in MLLM pRe-training. RADAR involves two key components: (1) Soft Discrimination Score, a novel metric for robustly tracking ability development without fine-tuning, based on quantifying nuanced gradations of the model preference for the correct answer over distractors; and (2) Multi-Modal Mixture Benchmark, a new 15K+ sample benchmark for comprehensively evaluating pre-trained MLLMs' perception and reasoning abilities in a 0-shot manner, where we unify authoritative benchmark datasets and carefully collect new datasets, extending the evaluation scope and addressing the critical gaps in current benchmarks. With RADAR, we comprehensively reveal the asymmetric development of perceptual and reasoning capabilities in pretrained MLLMs across diverse factors, including data volume, model size, and pretraining strategy. Our RADAR underscores the need for a decomposed perspective on pre-training ability bottlenecks, informing targeted interventions to advance MLLMs efficiently. Our code is publicly available at this https URL. 

---
# A Microservice-Based Platform for Sustainable and Intelligent SLO Fulfilment and Service Management 

**Authors**: Juan Luis Herrera, Daniel Wang, Schahram Dustdar  

**Link**: [PDF](https://arxiv.org/pdf/2602.12875)  

**Abstract**: The Microservices Architecture (MSA) design pattern has become a staple for modern applications, allowing functionalities to be divided across fine-grained microservices, fostering reusability, distribution, and interoperability. As MSA-based applications are deployed to the Computing Continuum (CC), meeting their Service Level Objectives (SLOs) becomes a challenge. Trading off performance and sustainability SLOs is especially challenging. This challenge can be addressed with intelligent decision systems, able to reconfigure the services during runtime to meet the SLOs. However, developing these agents while adhering to the MSA pattern is complex, especially because CC providers, who have key know-how and information to fulfill these SLOs, must comply with the privacy requirements of application developers. This work presents the Carbon-Aware SLO and Control plAtform (CASCA), an open-source MSA-based platform that allows CC providers to reconfigure services and fulfill their SLOs while maintaining the privacy of developers. CASCA is architected to be highly reusable, distributable, and easy to use, extend, and modify. CASCA has been evaluated in a real CC testbed for a media streaming service, where decision systems implemented in Bash, Rust, and Python successfully reconfigured the service, unaffected by upholding privacy. 

---
# Knowledge-Based Design Requirements for Generative Social Robots in Higher Education 

**Authors**: Stephan Vonschallen, Dominique Oberle, Theresa Schmiedel, Friederike Eyssel  

**Link**: [PDF](https://arxiv.org/pdf/2602.12873)  

**Abstract**: Generative social robots (GSRs) powered by large language models enable adaptive, conversational tutoring but also introduce risks such as hallucina-tions, overreliance, and privacy violations. Existing frameworks for educa-tional technologies and responsible AI primarily define desired behaviors, yet they rarely specify the knowledge prerequisites that enable generative systems to express these behaviors reliably. To address this gap, we adopt a knowledge-based design perspective and investigate what information tutor-ing-oriented GSRs require to function responsibly and effectively in higher education. Based on twelve semi-structured interviews with university stu-dents and lecturers, we identify twelve design requirements across three knowledge types: self-knowledge (assertive, conscientious and friendly per-sonality with customizable role), user-knowledge (personalized information about student learning goals, learning progress, motivation type, emotional state and background), and context-knowledge (learning materials, educa-tional strategies, course-related information, and physical learning environ-ment). By identifying these knowledge requirements, this work provides a structured foundation for the design of tutoring GSRs and future evaluations, aligning generative system capabilities with pedagogical and ethical expecta-tions. 

---
# X-VORTEX: Spatio-Temporal Contrastive Learning for Wake Vortex Trajectory Forecasting 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2602.12869)  

**Abstract**: Wake vortices are strong, coherent air turbulences created by aircraft, and they pose a major safety and capacity challenge for air traffic management. Tracking how vortices move, weaken, and dissipate over time from LiDAR measurements is still difficult because scans are sparse, vortex signatures fade as the flow breaks down under atmospheric turbulence and instabilities, and point-wise annotation is prohibitively expensive. Existing approaches largely treat each scan as an independent, fully supervised segmentation problem, which overlooks temporal structure and does not scale to the vast unlabeled archives collected in practice. We present X-VORTEX, a spatio-temporal contrastive learning framework grounded in Augmentation Overlap Theory that learns physics-aware representations from unlabeled LiDAR point cloud sequences. X-VORTEX addresses two core challenges: sensor sparsity and time-varying vortex dynamics. It constructs paired inputs from the same underlying flight event by combining a weakly perturbed sequence with a strongly augmented counterpart produced via temporal subsampling and spatial masking, encouraging the model to align representations across missing frames and partial observations. Architecturally, a time-distributed geometric encoder extracts per-scan features and a sequential aggregator models the evolving vortex state across variable-length sequences. We evaluate on a real-world dataset of over one million LiDAR scans. X-VORTEX achieves superior vortex center localization while using only 1% of the labeled data required by supervised baselines, and the learned representations support accurate trajectory forecasting. 

---
# Chimera: Neuro-Symbolic Attention Primitives for Trustworthy Dataplane Intelligence 

**Authors**: Rong Fu, Wenxin Zhang, Xiaowen Ma, Kun Liu, Wangyu Wu, Ziyu Kong, Jia Yee Tan, Tailong Luo, Xianda Li, Zeli Su, Youjin Wang, Yongtai Liu, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.12851)  

**Abstract**: Deploying expressive learning models directly on programmable dataplanes promises line-rate, low-latency traffic analysis but remains hindered by strict hardware constraints and the need for predictable, auditable behavior. Chimera introduces a principled framework that maps attention-oriented neural computations and symbolic constraints onto dataplane primitives, enabling trustworthy inference within the match-action pipeline. Chimera combines a kernelized, linearized attention approximation with a two-layer key-selection hierarchy and a cascade fusion mechanism that enforces hard symbolic guarantees while preserving neural expressivity. The design includes a hardware-aware mapping protocol and a two-timescale update scheme that together permit stable, line-rate operation under realistic dataplane budgets. The paper presents the Chimera architecture, a hardware mapping strategy, and empirical evidence showing that neuro-symbolic attention primitives can achieve high-fidelity inference within the resource envelope of commodity programmable switches. 

---
# Amortized Reasoning Tree Search: Decoupling Proposal and Decision in Large Language Models 

**Authors**: Zesheng Hong, Jiadong Yu, Hui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12846)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has established itself as the dominant paradigm for instilling rigorous reasoning capabilities in Large Language Models. While effective at amplifying dominant behaviors, we identify a critical pathology in this alignment process: the systematic suppression of valid but rare (low-likelihood under the base model distribution) reasoning paths. We theoretically characterize this phenomenon as a "Normalization Squeeze," where the interplay between mode-seeking policy gradients and finite sampling acts as a high-pass likelihood filter, driving the probability of rare correct traces to statistical extinction. To counteract this collapse without discarding the base model's latent diversity, we propose Amortized Reasoning Tree Search (ARTS). Unlike standard approaches that force internalization via parameter updates, ARTS prioritizes deliberation by decoupling generation from verification. We introduce a Flow Matching objective that repurposes the verifier to estimate the conservation of probability flow, enabling robust navigation through sparse, high-entropy search spaces where traditional discriminative objectives fail. Extensive experiments on the MATH-500 benchmark demonstrate that ARTS achieves a performance of 74.6% (BoN@16), effectively matching fully fine-tuned policies (74.7%) without modifying the generative backbone. Crucially, on the long-tail subset where coupled RL optimization collapses to 0% pass@k, ARTS uniquely recovers significant performance, suggesting that disentangling verification from generation offers a more robust pathway for solving complex reasoning tasks. 

---
# TRACE: Temporal Reasoning via Agentic Context Evolution for Streaming Electronic Health Records (EHRs) 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2602.12833)  

**Abstract**: Large Language Models (LLMs) encode extensive medical knowledge but struggle to apply it reliably to longitudinal patient trajectories, where evolving clinical states, irregular timing, and heterogeneous events degrade performance over time. Existing adaptation strategies rely on fine-tuning or retrieval-based augmentation, which introduce computational overhead, privacy constraints, or instability under long contexts. We introduce TRACE (Temporal Reasoning via Agentic Context Evolution), a framework that enables temporal clinical reasoning with frozen LLMs by explicitly structuring and maintaining context rather than extending context windows or updating parameters. TRACE operates over a dual-memory architecture consisting of a static Global Protocol encoding institutional clinical rules and a dynamic Individual Protocol tracking patient-specific state. Four agentic components, Router, Reasoner, Auditor, and Steward, coordinate over this structured memory to support temporal inference and state evolution. The framework maintains bounded inference cost via structured state compression and selectively audits safety-critical clinical decisions. Evaluated on longitudinal clinical event streams from MIMIC-IV, TRACE significantly improves next-event prediction accuracy, protocol adherence, and clinical safety over long-context and retrieval-augmented baselines, while producing interpretable and auditable reasoning traces. 

---
# FLAC: Maximum Entropy RL via Kinetic Energy Regularized Bridge Matching 

**Authors**: Lei Lv, Yunfei Li, Yu Luo, Fuchun Sun, Xiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2602.12829)  

**Abstract**: Iterative generative policies, such as diffusion models and flow matching, offer superior expressivity for continuous control but complicate Maximum Entropy Reinforcement Learning because their action log-densities are not directly accessible. To address this, we propose Field Least-Energy Actor-Critic (FLAC), a likelihood-free framework that regulates policy stochasticity by penalizing the kinetic energy of the velocity field. Our key insight is to formulate policy optimization as a Generalized Schrödinger Bridge (GSB) problem relative to a high-entropy reference process (e.g., uniform). Under this view, the maximum-entropy principle emerges naturally as staying close to a high-entropy reference while optimizing return, without requiring explicit action densities. In this framework, kinetic energy serves as a physically grounded proxy for divergence from the reference: minimizing path-space energy bounds the deviation of the induced terminal action distribution. Building on this view, we derive an energy-regularized policy iteration scheme and a practical off-policy algorithm that automatically tunes the kinetic energy via a Lagrangian dual mechanism. Empirically, FLAC achieves superior or comparable performance on high-dimensional benchmarks relative to strong baselines, while avoiding explicit density estimation. 

---
# GRAIL: Geometry-Aware Retrieval-Augmented Inference with LLMs over Hyperbolic Representations of Patient Trajectories 

**Authors**: Zhan Qu, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2602.12828)  

**Abstract**: Predicting future clinical events from longitudinal electronic health records (EHRs) is challenging due to sparse multi-type clinical events, hierarchical medical vocabularies, and the tendency of large language models (LLMs) to hallucinate when reasoning over long structured histories. We study next-visit event prediction, which aims to forecast a patient's upcoming clinical events based on prior visits. We propose GRAIL, a framework that models longitudinal EHRs using structured geometric representations and structure-aware retrieval. GRAIL constructs a unified clinical graph by combining deterministic coding-system hierarchies with data-driven temporal associations across event types, embeds this graph in hyperbolic space, and summarizes each visit as a probabilistic Central Event that denoises sparse observations. At inference time, GRAIL retrieves a structured set of clinically plausible future events aligned with hierarchical and temporal progression, and optionally refines their ranking using an LLM as a constrained inference-time reranker. Experiments on MIMIC-IV show that GRAIL consistently improves multi-type next-visit prediction and yields more hierarchy-consistent forecasts. 

---
# Left-right asymmetry in predicting brain activity from LLMs' representations emerges with their formal linguistic competence 

**Authors**: Laurent Bonnasse-Gahot, Christophe Pallier  

**Link**: [PDF](https://arxiv.org/pdf/2602.12811)  

**Abstract**: When humans and large language models (LLMs) process the same text, activations in the LLMs correlate with brain activity measured, e.g., with functional magnetic resonance imaging (fMRI). Moreover, it has been shown that, as the training of an LLM progresses, the performance in predicting brain activity from its internal activations improves more in the left hemisphere than in the right one. The aim of the present work is to understand which kind of competence acquired by the LLMs underlies the emergence of this left-right asymmetry. Using the OLMo-2 7B language model at various training checkpoints and fMRI data from English participants, we compare the evolution of the left-right asymmetry in brain scores alongside performance on several benchmarks. We observe that the asymmetry co-emerges with the formal linguistic abilities of the LLM. These abilities are demonstrated in two ways: by the model's capacity to assign a higher probability to an acceptable sentence than to a grammatically unacceptable one within a minimal contrasting pair, or its ability to produce well-formed text. On the opposite, the left-right asymmetry does not correlate with the performance on arithmetic or Dyck language tasks; nor with text-based tasks involving world knowledge and reasoning. We generalize these results to another family of LLMs (Pythia) and another language, namely French. Our observations indicate that the left-right asymmetry in brain predictivity matches the progress in formal linguistic competence (knowledge of linguistic patterns). 

---
# RAT-Bench: A Comprehensive Benchmark for Text Anonymization 

**Authors**: Nataša Krčo, Zexi Yao, Matthieu Meeus, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2602.12806)  

**Abstract**: Data containing personal information is increasingly used to train, fine-tune, or query Large Language Models (LLMs). Text is typically scrubbed of identifying information prior to use, often with tools such as Microsoft's Presidio or Anthropic's PII purifier. These tools have traditionally been evaluated on their ability to remove specific identifiers (e.g., names), yet their effectiveness at preventing re-identification remains unclear. We introduce RAT-Bench, a comprehensive benchmark for text anonymization tools based on re-identification risk. Using U.S. demographic statistics, we generate synthetic text containing various direct and indirect identifiers across domains, languages, and difficulty levels. We evaluate a range of NER- and LLM-based text anonymization tools and, based on the attributes an LLM-based attacker is able to correctly infer from the anonymized text, we report the risk of re-identification in the U.S. population, while properly accounting for the disparate impact of identifiers. We find that, while capabilities vary widely, even the best tools are far from perfect in particular when direct identifiers are not written in standard ways and when indirect identifiers enable re-identification. Overall we find LLM-based anonymizers, including new iterative anonymizers, to provide a better privacy-utility trade-off albeit at a higher computational cost. Importantly, we also find them to work well across languages. We conclude with recommendations for future anonymization tools and will release the benchmark and encourage community efforts to expand it, in particular to other geographies. 

---
# Can Neural Networks Provide Latent Embeddings for Telemetry-Aware Greedy Routing? 

**Authors**: Andreas Boltres, Niklas Freymuth, Gerhard Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2602.12798)  

**Abstract**: Telemetry-Aware routing promises to increase efficacy and responsiveness to traffic surges in computer networks. Recent research leverages Machine Learning to deal with the complex dependency between network state and routing, but sacrifices explainability of routing decisions due to the black-box nature of the proposed neural routing modules. We propose \emph{Placer}, a novel algorithm using Message Passing Networks to transform network states into latent node embeddings. These embeddings facilitate quick greedy next-hop routing without directly solving the all-pairs shortest paths problem, and let us visualize how certain network events shape routing decisions. 

---
# SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise 

**Authors**: Yuejie Li, Ke Yang, Yueying Hua, Berlin Chen, Jianhao Nie, Yueping He, Caixin Kang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12783)  

**Abstract**: Spoken query retrieval is an important interaction mode in modern information retrieval. However, existing evaluation datasets are often limited to simple queries under constrained noise conditions, making them inadequate for assessing the robustness of spoken query retrieval systems under complex acoustic perturbations. To address this limitation, we present SQuTR, a robustness benchmark for spoken query retrieval that includes a large-scale dataset and a unified evaluation protocol. SQuTR aggregates 37,317 unique queries from six commonly used English and Chinese text retrieval datasets, spanning multiple domains and diverse query types. We synthesize speech using voice profiles from 200 real speakers and mix 17 categories of real-world environmental noise under controlled SNR levels, enabling reproducible robustness evaluation from quiet to highly noisy conditions. Under the unified protocol, we conduct large-scale evaluations on representative cascaded and end-to-end retrieval systems. Experimental results show that retrieval performance decreases as noise increases, with substantially different drops across systems. Even large-scale retrieval models struggle under extreme noise, indicating that robustness remains a critical bottleneck. Overall, SQuTR provides a reproducible testbed for benchmarking and diagnostic analysis, and facilitates future research on robustness in spoken query to text retrieval. 

---
# "Not Human, Funnier": How Machine Identity Shapes Humor Perception in Online AI Stand-up Comedy 

**Authors**: Xuehan Huang, Canwen Wang, Yifei Hao, Daijin Yang, Ray LC  

**Link**: [PDF](https://arxiv.org/pdf/2602.12763)  

**Abstract**: Chatbots are increasingly applied to domains previously reserved for human actors. One such domain is comedy, whereby both the general public working with ChatGPT and research-based LLM-systems have tried their hands on making humor. In formative interviews with professional comedians and video analyses of stand-up comedy in humans, we found that human performers often use their ethnic, gender, community, and demographic-based identity to enable joke-making. This suggests whether the identity of AI itself can empower AI humor generation for human audiences. We designed a machine-identity-based agent that uses its own status as AI to tell jokes in online performance format. Studies with human audiences (N=32) showed that machine-identity-based agents were seen as funnier than baseline-GPT agent. This work suggests the design of human-AI integrated systems that explicitly utilize AI as its own unique identity apart from humans. 

---
# VineetVC: Adaptive Video Conferencing Under Severe Bandwidth Constraints Using Audio-Driven Talking-Head Reconstruction 

**Authors**: Vineet Kumar Rakesh, Soumya Mazumdar, Tapas Samanta, Hemendra Kumar Pandey, Amitabha Das, Sarbajit Pal  

**Link**: [PDF](https://arxiv.org/pdf/2602.12758)  

**Abstract**: Intense bandwidth depletion within consumer and constrained networks has the potential to undermine the stability of real-time video conferencing: encoder rate management becomes saturated, packet loss escalates, frame rates deteriorate, and end-to-end latency significantly increases. This work delineates an adaptive conferencing system that integrates WebRTC media delivery with a supplementary audio-driven talking-head reconstruction pathway and telemetry-driven mode regulation. The system consists of a WebSocket signaling service, an optional SFU for multi-party transmission, a browser client capable of real-time WebRTC statistics extraction and CSV telemetry export, and an AI REST service that processes a reference face image and recorded audio to produce a synthesized MP4; the browser can substitute its outbound camera track with the synthesized stream with a median bandwidth of 32.80 kbps. The solution incorporates a bandwidth-mode switching strategy and a client-side mode-state logger. 

---
# MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs 

**Authors**: Baorong Shi, Bo Cui, Boyuan Jiang, Deli Yu, Fang Qian, Haihua Yang, Huichao Wang, Jiale Chen, Jianfei Pan, Jieqiong Cao, Jinghao Lin, Kai Wu, Lin Yang, Shengsheng Yao, Tao Chen, Xiaojun Xiao, Xiaozhong Ji, Xu Wang, Yijun He, Zhixiong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12705)  

**Abstract**: We present MedXIAOHE, a medical vision-language foundation model designed to advance general-purpose medical understanding and reasoning in real-world clinical applications. MedXIAOHE achieves state-of-the-art performance across diverse medical benchmarks and surpasses leading closed-source multimodal systems on multiple capabilities. To achieve this, we propose an entity-aware continual pretraining framework that organizes heterogeneous medical corpora to broaden knowledge coverage and reduce long-tail gaps (e.g., rare diseases). For medical expert-level reasoning and interaction, MedXIAOHE incorporates diverse medical reasoning patterns via reinforcement learning and tool-augmented agentic training, enabling multi-step diagnostic reasoning with verifiable decision traces. To improve reliability in real-world use, MedXIAOHE integrates user-preference rubrics, evidence-grounded reasoning, and low-hallucination long-form report generation, with improved adherence to medical instructions. We release this report to document our practical design choices, scaling insights, and evaluation framework, hoping to inspire further research. 

---
# ALOE: Action-Level Off-Policy Evaluation for Vision-Language-Action Model Post-Training 

**Authors**: Rushuai Yang, Hecheng Wang, Chiming Liu, Xiaohan Yan, Yunlong Wang, Xuan Du, Shuoyu Yue, Yongcheng Liu, Chuheng Zhang, Lizhe Qi, Yi Chen, Wei Shan, Maoqing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12691)  

**Abstract**: We study how to improve large foundation vision-language-action (VLA) systems through online reinforcement learning (RL) in real-world settings. Central to this process is the value function, which provides learning signals to guide VLA learning from experience. In practice, the value function is estimated from trajectory fragments collected from different data sources, including historical policies and intermittent human interventions. Estimating the value function of current behavior quality from the mixture data is inherently an off-policy evaluation problem. However, prior work often adopts conservative on-policy estimation for stability, which avoids direct evaluation of the current high-capacity policy and limits learning effectiveness. In this paper, we propose ALOE, an action-level off-policy evaluation framework for VLA post-training. ALOE applies chunking-based temporal-difference bootstrapping to evaluate individual action sequences instead of predicting final task outcomes. This design improves effective credit assignment to critical action chunks under sparse rewards and supports stable policy improvement. We evaluate our method on three real-world manipulation tasks, including smartphone packing as a high-precision task, laundry folding as a long-horizon deformable-object task, and bimanual pick-and-place involving multi-object perception. Across all tasks, ALOE improves learning efficiency without compromising execution speed, showing that off-policy RL can be reintroduced in a reliable manner for real-world VLA post-training. Videos and additional materials are available at our project website. 

---
# Trust the uncertain teacher: distilling dark knowledge via calibrated uncertainty 

**Authors**: Jeonghyun Kim, SooKyung Kim, Richeng Xuan, Hyunsoo Cho  

**Link**: [PDF](https://arxiv.org/pdf/2602.12687)  

**Abstract**: The core of knowledge distillation lies in transferring the teacher's rich 'dark knowledge'-subtle probabilistic patterns that reveal how classes are related and the distribution of uncertainties. While this idea is well established, teachers trained with conventional cross-entropy often fail to preserve such signals. Their distributions collapse into sharp, overconfident peaks that appear decisive but are in fact brittle, offering little beyond the hard label or subtly hindering representation-level transfer. This overconfidence is especially problematic in high-cardinality tasks, where the nuances among many plausible classes matter most for guiding a compact student. Moreover, such brittle targets reduce robustness under distribution shift, leaving students vulnerable to miscalibration in real-world conditions. To address this limitation, we revisit distillation from a distributional perspective and propose Calibrated Uncertainty Distillation (CUD), a framework designed to make dark knowledge more faithfully accessible. Instead of uncritically adopting the teacher's overconfidence, CUD encourages teachers to reveal uncertainty where it is informative and guides students to learn from targets that are calibrated rather than sharpened certainty. By directly shaping the teacher's predictive distribution before transfer, our approach balances accuracy and calibration, allowing students to benefit from both confident signals on easy cases and structured uncertainty on hard ones. Across diverse benchmarks, CUD yields students that are not only more accurate, but also more calibrated under shift and more reliable on ambiguous, long-tail inputs. 

---
# SLA2: Sparse-Linear Attention with Learnable Routing and QAT 

**Authors**: Jintao Zhang, Haoxu Wang, Kai Jiang, Kaiwen Zheng, Youhe Jiang, Ion Stoica, Jianfei Chen, Jun Zhu, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2602.12675)  

**Abstract**: Sparse-Linear Attention (SLA) combines sparse and linear attention to accelerate diffusion models and has shown strong performance in video generation. However, (i) SLA relies on a heuristic split that assigns computations to the sparse or linear branch based on attention-weight magnitude, which can be suboptimal. Additionally, (ii) after formally analyzing the attention error in SLA, we identify a mismatch between SLA and a direct decomposition into sparse and linear attention. We propose SLA2, which introduces (I) a learnable router that dynamically selects whether each attention computation should use sparse or linear attention, (II) a more faithful and direct sparse-linear attention formulation that uses a learnable ratio to combine the sparse and linear attention branches, and (III) a sparse + low-bit attention design, where low-bit attention is introduced via quantization-aware fine-tuning to reduce quantization error. Experiments show that on video diffusion models, SLA2 can achieve 97% attention sparsity and deliver an 18.6x attention speedup while preserving generation quality. 

---
# IndicFairFace: Balanced Indian Face Dataset for Auditing and Mitigating Geographical Bias in Vision-Language Models 

**Authors**: Aarish Shah Mohsin, Mohammed Tayyab Ilyas Khan, Mohammad Nadeem, Shahab Saquib Sohail, Erik Cambria, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.12659)  

**Abstract**: Vision-Language Models (VLMs) are known to inherit and amplify societal biases from their web-scale training data with Indian being particularly misrepresented. Existing fairness-aware datasets have significantly improved demographic balance across global race and gender groups, yet they continue to treat Indian as a single monolithic category. The oversimplification ignores the vast intra-national diversity across 28 states and 8 Union Territories of India and leads to representational and geographical bias. To address the limitation, we present IndicFairFace, a novel and balanced face dataset comprising 14,400 images representing geographical diversity of India. Images were sourced ethically from Wikimedia Commons and open-license web repositories and uniformly balanced across states and gender. Using IndicFairFace, we quantify intra-national geographical bias in prominent CLIP-based VLMs and reduce it using post-hoc Iterative Nullspace Projection debiasing approach. We also show that the adopted debiasing approach does not adversely impact the existing embedding space as the average drop in retrieval accuracy on benchmark datasets is less than 1.5 percent. Our work establishes IndicFairFace as the first benchmark to study geographical bias in VLMs for the Indian context. 

---
# PMG: Parameterized Motion Generator for Human-like Locomotion Control 

**Authors**: Chenxi Han, Yuheng Min, Zihao Huang, Ao Hong, Hang Liu, Yi Cheng, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12656)  

**Abstract**: Recent advances in data-driven reinforcement learning and motion tracking have substantially improved humanoid locomotion, yet critical practical challenges remain. In particular, while low-level motion tracking and trajectory-following controllers are mature, whole-body reference-guided methods are difficult to adapt to higher-level command interfaces and diverse task contexts: they require large, high-quality datasets, are brittle across speed and pose regimes, and are sensitive to robot-specific calibration. To address these limitations, we propose the Parameterized Motion Generator (PMG), a real-time motion generator grounded in an analysis of human motion structure that synthesizes reference trajectories using only a compact set of parameterized motion data together with High-dimensional control commands. Combined with an imitation-learning pipeline and an optimization-based sim-to-real motor parameter identification module, we validate the complete approach on our humanoid prototype ZERITH Z1 and show that, within a single integrated system, PMG produces natural, human-like locomotion, responds precisely to high-dimensional control inputs-including VR-based teleoperation-and enables efficient, verifiable sim-to-real transfer. Together, these results establish a practical, experimentally validated pathway toward natural and deployable humanoid control. 

---
# Multi-Task Learning with Additive U-Net for Image Denoising and Classification 

**Authors**: Vikram Lakkavalli, Neelam Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2602.12649)  

**Abstract**: We investigate additive skip fusion in U-Net architectures for image denoising and denoising-centric multi-task learning (MTL). By replacing concatenative skips with gated additive fusion, the proposed Additive U-Net (AddUNet) constrains shortcut capacity while preserving fixed feature dimensionality across depth. This structural regularization induces controlled encoder-decoder information flow and stabilizes joint optimization. Across single-task denoising and joint denoising-classification settings, AddUNet achieves competitive reconstruction performance with improved training stability. In MTL, learned skip weights exhibit systematic task-aware redistribution: shallow skips favor reconstruction, while deeper features support discrimination. Notably, reconstruction remains robust even under limited classification capacity, indicating implicit task decoupling through additive fusion. These findings show that simple constraints on skip connections act as an effective architectural regularizer for stable and scalable multi-task learning without increasing model complexity. 

---
# Unifying Model-Free Efficiency and Model-Based Representations via Latent Dynamics 

**Authors**: Jashaswimalya Acharjee, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2602.12643)  

**Abstract**: We present Unified Latent Dynamics (ULD), a novel reinforcement learning algorithm that unifies the efficiency of model-free methods with the representational strengths of model-based approaches, without incurring planning overhead. By embedding state-action pairs into a latent space in which the true value function is approximately linear, our method supports a single set of hyperparameters across diverse domains -- from continuous control with low-dimensional and pixel inputs to high-dimensional Atari games. We prove that, under mild conditions, the fixed point of our embedding-based temporal-difference updates coincides with that of a corresponding linear model-based value expansion, and we derive explicit error bounds relating embedding fidelity to value approximation quality. In practice, ULD employs synchronized updates of encoder, value, and policy networks, auxiliary losses for short-horizon predictive dynamics, and reward-scale normalization to ensure stable learning under sparse rewards. Evaluated on 80 environments spanning Gym locomotion, DeepMind Control (proprioceptive and visual), and Atari, our approach matches or exceeds the performance of specialized model-free and general model-based baselines -- achieving cross-domain competence with minimal tuning and a fraction of the parameter footprint. These results indicate that value-aligned latent representations alone can deliver the adaptability and sample efficiency traditionally attributed to full model-based planning. 

---
# Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR 

**Authors**: Dohyung Kim, Minbeom Kim, Jeonghye Kim, Sangmook Lee, Sojeong Rhee, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2602.12642)  

**Abstract**: Reward-maximizing RL methods enhance the reasoning performance of LLMs, but often reduce the diversity among outputs. Recent works address this issue by adopting GFlowNets, training LLMs to match a target distribution while jointly learning its partition function. In contrast to prior works that treat this partition function solely as a normalizer, we reinterpret it as a per-prompt expected-reward (i.e., online accuracy) signal, leveraging this unused information to improve sample efficiency. Specifically, we first establish a theoretical relationship between the partition function and per-prompt accuracy estimates. Building on this key insight, we propose Partition Function-Guided RL (PACED-RL), a post-training framework that leverages accuracy estimates to prioritize informative question prompts during training, and further improves sample efficiency through an accuracy estimate error-prioritized replay. Crucially, both components reuse information already produced during GFlowNet training, effectively amortizing the compute overhead into the existing optimization process. Extensive experiments across diverse benchmarks demonstrate strong performance improvements over GRPO and prior GFlowNet approaches, highlighting PACED-RL as a promising direction for a more sample efficient distribution-matching training for LLMs. 

---
# Artic: AI-oriented Real-time Communication for MLLM Video Assistant 

**Authors**: Jiangkai Wu, Zhiyuan Ren, Junquan Zhong, Liming Liu, Xinggong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12641)  

**Abstract**: AI Video Assistant emerges as a new paradigm for Real-time Communication (RTC), where one peer is a Multimodal Large Language Model (MLLM) deployed in the cloud. This makes interaction between humans and AI more intuitive, akin to chatting with a real person. However, a fundamental mismatch exists between current RTC frameworks and AI Video Assistants, stemming from the drastic shift in Quality of Experience (QoE) and more challenging networks. Measurements on our production prototype also confirm that current RTC fails, causing latency spikes and accuracy drops.
To address these challenges, we propose Artic, an AI-oriented RTC framework for MLLM Video Assistants, exploring the shift from "humans watching video" to "AI understanding video." Specifically, Artic proposes: (1) Response Capability-aware Adaptive Bitrate, which utilizes MLLM accuracy saturation to proactively cap bitrate, reserving bandwidth headroom to absorb future fluctuations for latency reduction; (2) Zero-overhead Context-aware Streaming, which allocates limited bitrate to regions most important for the response, maintaining accuracy even under ultra-low bitrates; and (3) Degraded Video Understanding Benchmark, the first benchmark evaluating how RTC-induced video degradation affects MLLM accuracy. Prototype experiments using real-world uplink traces show that compared with existing methods, Artic significantly improves accuracy by 15.12% and reduces latency by 135.31 ms. We will release the benchmark and codes at this https URL. 

---
# Unleashing Low-Bit Inference on Ascend NPUs: A Comprehensive Evaluation of HiFloat Formats 

**Authors**: Pengxiang Zhao, Hui-Ling Zhen, Xing Li, Han Bao, Weizhe Lin, Zhiyuan Yang, Ziwei Yu, Xin Wang, Mingxuan Yuan, Xianzhi Yu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2602.12635)  

**Abstract**: As LLMs scale, low-bit floating-point formats like MXFP and NVFP4 offer new opportunities for precision and efficiency. In this work, we evaluate HiFloat (HiF8 and HiF4), a family of formats tailored for Ascend NPUs. Through rigorous comparison across weight-activation and KV-cache tasks, we provide three key insights: (1) INT8 suits narrow-range data, while floating-point formats excel with high-variance data; (2) in 4-bit regimes, HiF4's hierarchical scaling prevents the accuracy collapse seen in integer formats; and (3) HiFloat is fully compatible with state-of-the-art post-training quantization frameworks. Overall, HiFloat provides a solution for high-efficiency LLM inference on NPUs. 

---
# TensorCommitments: A Lightweight Verifiable Inference for Language Models 

**Authors**: Oguzhan Baser, Elahe Sadeghi, Eric Wang, David Ribeiro Alves, Sam Kazemian, Hong Kang, Sandeep P. Chinchali, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2602.12630)  

**Abstract**: Most large language models (LLMs) run on external clouds: users send a prompt, pay for inference, and must trust that the remote GPU executes the LLM without any adversarial tampering. We critically ask how to achieve verifiable LLM inference, where a prover (the service) must convince a verifier (the client) that an inference was run correctly without rerunning the LLM. Existing cryptographic works are too slow at the LLM scale, while non-cryptographic ones require a strong verifier GPU. We propose TensorCommitments (TCs), a tensor-native proof-of-inference scheme. TC binds the LLM inference to a commitment, an irreversible tag that breaks under tampering, organized in our multivariate Terkle Trees. For LLaMA2, TC adds only 0.97% prover and 0.12% verifier time over inference while improving robustness to tailored LLM attacks by up to 48% over the best prior work requiring a verifier GPU. 

---
# Vision Token Reduction via Attention-Driven Self-Compression for Efficient Multimodal Large Language Models 

**Authors**: Omer Faruk Deniz, Ruiyu Mao, Ruochen Li, Yapeng Tian, Latifur Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12618)  

**Abstract**: Multimodal Large Language Models (MLLMs) incur significant computational cost from processing numerous vision tokens through all LLM layers. Prior pruning methods operate either before the LLM, limiting generality due to diverse encoder-projector designs or within the LLM using heuristics that are incompatible with FlashAttention. We take a different approach: rather than identifying unimportant tokens, we treat the LLM itself as the optimal guide for compression. Observing that deeper layers naturally transmit vision-to-text information, we introduce Attention-Driven Self-Compression (ADSC), a simple, broadly applicable method that progressively reduces vision tokens using only the LLM's attention mechanism. Our method applies uniform token downsampling at selected layers, forming bottlenecks that encourage the model to reorganize and compress information into the remaining tokens. It requires no score computation, auxiliary modules, or attention modification, and remains fully compatible with FlashAttention. Applied to LLaVA-1.5, ADSC reduces FLOPs by 53.7% and peak KV-cache memory by 56.7%, while preserving 98.2% of the original model performance. Across multiple benchmarks, it outperforms prior pruning approaches in both efficiency and accuracy. Crucially, under high compression ratios, our method remains robust while heuristic-based techniques degrade sharply. 

---
# Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback 

**Authors**: Sein Kim, Sangwu Park, Hongseok Kang, Wonjoong Kim, Jimin Seo, Yeonjun In, Kanghoon Yoon, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2602.12612)  

**Abstract**: Traditional methods for automating recommender system design, such as Neural Architecture Search (NAS), are often constrained by a fixed search space defined by human priors, limiting innovation to pre-defined operators. While recent LLM-driven code evolution frameworks shift fixed search space target to open-ended program spaces, they primarily rely on scalar metrics (e.g., NDCG, Hit Ratio) that fail to provide qualitative insights into model failures or directional guidance for improvement. To address this, we propose Self-EvolveRec, a novel framework that establishes a directional feedback loop by integrating a User Simulator for qualitative critiques and a Model Diagnosis Tool for quantitative internal verification. Furthermore, we introduce a Diagnosis Tool - Model Co-Evolution strategy to ensure that evaluation criteria dynamically adapt as the recommendation architecture evolves. Extensive experiments demonstrate that Self-EvolveRec significantly outperforms state-of-the-art NAS and LLM-driven code evolution baselines in both recommendation performance and user satisfaction. Our code is available at this https URL. 

---
# QuEPT: Quantized Elastic Precision Transformers with One-Shot Calibration for Multi-Bit Switching 

**Authors**: Ke Xu, Yixin Wang, Zhongcheng Li, Hao Cui, Jinshui Hu, Xingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12609)  

**Abstract**: Elastic precision quantization enables multi-bit deployment via a single optimization pass, fitting diverse quantization this http URL, the high storage and optimization costs associated with the Transformer architecture, research on elastic quantization remains limited, particularly for large language this http URL paper proposes QuEPT, an efficient post-training scheme that reconstructs block-wise multi-bit errors with one-shot calibration on a small data slice. It can dynamically adapt to various predefined bit-widths by cascading different low-rank adapters, and supports real-time switching between uniform quantization and mixed precision quantization without repeated optimization. To enhance accuracy and robustness, we introduce Multi-Bit Token Merging (MB-ToMe) to dynamically fuse token features across different bit-widths, improving robustness during bit-width switching. Additionally, we propose Multi-Bit Cascaded Low-Rank adapters (MB-CLoRA) to strengthen correlations between bit-width groups, further improve the overall performance of QuEPT. Extensive experiments demonstrate that QuEPT achieves comparable or better performance to existing state-of-the-art post-training quantization this http URL code is available at this https URL 

---
# HyperMLP: An Integrated Perspective for Sequence Modeling 

**Authors**: Jiecheng Lu, Shihao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12601)  

**Abstract**: Self-attention is often viewed as probabilistic query-key lookup, motivating designs that preserve normalized attention scores and fixed positional semantics. We advocate a simpler and more unified perspective: an autoregressive attention head can be viewed as a dynamic two-layer MLP whose weights are instantiated from the context history. From this view, attention scores form an ever-growing hidden representation, and standard MLP activations such as ReLU or GLU naturally implement input-conditioned selection over a context-dependent memory pool rather than a probability distribution. Based on this formulation, we introduce HyperMLP and HyperGLU, which learn dynamic mixing in both feature space and sequence space, using a reverse-offset (lag) layout to align temporal mixing with autoregressive semantics. We provide theoretical characterizations of the expressivity and implications of this structure, and empirically show that HyperMLP/HyperGLU consistently outperform strong softmax-attention baselines under matched parameter budgets. 

---
# RQ-GMM: Residual Quantized Gaussian Mixture Model for Multimodal Semantic Discretization in CTR Prediction 

**Authors**: Ziye Tong, Jiahao Liu, Weimin Zhang, Hongji Ruan, Derick Tang, Zhanpeng Zeng, Qinsong Zeng, Peng Zhang, Tun Lu, Ning Gu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12593)  

**Abstract**: Multimodal content is crucial for click-through rate (CTR) prediction. However, directly incorporating continuous embeddings from pre-trained models into CTR models yields suboptimal results due to misaligned optimization objectives and convergence speed inconsistency during joint training. Discretizing embeddings into semantic IDs before feeding them into CTR models offers a more effective solution, yet existing methods suffer from limited codebook utilization, reconstruction accuracy, and semantic discriminability. We propose RQ-GMM (Residual Quantized Gaussian Mixture Model), which introduces probabilistic modeling to better capture the statistical structure of multimodal embedding spaces. Through Gaussian Mixture Models combined with residual quantization, RQ-GMM achieves superior codebook utilization and reconstruction accuracy. Experiments on public datasets and online A/B tests on a large-scale short-video platform serving hundreds of millions of users demonstrate substantial improvements: RQ-GMM yields a 1.502% gain in Advertiser Value over strong baselines. The method has been fully deployed, serving daily recommendations for hundreds of millions of users. 

---
# Power Interpretable Causal ODE Networks: A Unified Model for Explainable Anomaly Detection and Root Cause Analysis in Power Systems 

**Authors**: Yue Sun, Likai Wang, Rick S. Blum, Parv Venkitasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2602.12592)  

**Abstract**: Anomaly detection and root cause analysis (RCA) are critical for ensuring the safety and resilience of cyber-physical systems such as power grids. However, existing machine learning models for time series anomaly detection often operate as black boxes, offering only binary outputs without any explanation, such as identifying anomaly type and origin. To address this challenge, we propose Power Interpretable Causality Ordinary Differential Equation (PICODE) Networks, a unified, causality-informed architecture that jointly performs anomaly detection along with the explanation why it is detected as an anomaly, including root cause localization, anomaly type classification, and anomaly shape characterization. Experimental results in power systems demonstrate that PICODE achieves competitive detection performance while offering improved interpretability and reduced reliance on labeled data or external causal graphs. We provide theoretical results demonstrating the alignment between the shape of anomaly functions and the changes in the weights of the extracted causal graphs. 

---
# VI-CuRL: Stabilizing Verifier-Independent RL Reasoning via Confidence-Guided Variance Reduction 

**Authors**: Xin-Qiang Cai, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2602.12579)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a dominant paradigm for enhancing Large Language Models (LLMs) reasoning, yet its reliance on external verifiers limits its scalability. Recent findings suggest that RLVR primarily functions by eliciting latent capabilities, motivating the development of verifier-free algorithms. However, in such settings, standard methods like Group Relative Policy Optimization face a critical challenge: destructive gradient variance that often leads to training collapse. To address this issue, we introduceVerifier-Independent Curriculum Reinforcement Learning (VI-CuRL), a framework that leverages the model's intrinsic confidence to construct a curriculum independent from external verifiers. By prioritizing high-confidence samples, VI-CuRL effectively manages the bias-variance trade-off, specifically targeting the reduction of action and problem variance. We provide a rigorous theoretical analysis, proving that our estimator guarantees asymptotic unbiasedness. Empirically, VI-CuRL promotes stability and consistently outperforms verifier-independent baselines across six challenging benchmarks with/without verifiers. 

---
# Monte Carlo Tree Search with Reasoning Path Refinement for Small Language Models in Conversational Text-to-NoSQL 

**Authors**: Xubang Xiong, Raymond Chi-Wing Wong, Yuanfeng Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.12574)  

**Abstract**: NoSQL databases have been widely adopted in big data analytics, geospatial applications, and healthcare services, due to their flexibility and scalability. However, querying NoSQL databases requires specialized technical expertise, creating a high barrier for users. While recent studies have explored text-to-NoSQL problem, they primarily focus on single-turn interactions, ignoring the conversational nature of real-world queries. To bridge this gap, we introduce the Conversational Text-to-NoSQL task, which generates NoSQL queries given a natural language question, a NoSQL database, and the dialogue history. To address this task, we propose Stage-MCTS, a framework that endows small language models (SLMs) with NoSQL-specific reasoning capabilities by formulating query generation as a search problem. The framework employs Monte Carlo Tree Search (MCTS) guided by a rule-based reward to produce stepwise reasoning data, followed by progressive supervised fine-tuning (SFT) and self-training strategies. We further construct CoNoSQL, a cross-domain dataset with over 2,000 dialogues and 150 databases, to support evaluation. Experiments demonstrate that our approach outperforms state-of-the-art large reasoning models, improving execution value match (EVM) accuracy by up to 7.93%. 

---
# SD-MoE: Spectral Decomposition for Effective Expert Specialization 

**Authors**: Ruijun Huang, Fang Dong, Xin Zhang, Hengjie Cao, Zhendong Huang, Anrui Chen, Jixian Zhou, Mengyi Chen, Yifeng Yang, Mingzhi Dong, Yujiang Wang, Jinlong Hou, Qin Lv, Robert P. Dick, Yuan Cheng, Fan Yang, Tun Lu, Chun Zhang, Li Shang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12556)  

**Abstract**: Mixture-of-Experts (MoE) architectures scale Large Language Models via expert specialization induced by conditional computation. In practice, however, expert specialization often fails: some experts become functionally similar, while others functioning as de facto shared experts, limiting the effective capacity and model performance. In this work, we analysis from a spectral perspective on parameter and gradient spaces, uncover that (1) experts share highly overlapping dominant spectral components in their parameters, (2) dominant gradient subspaces are strongly aligned across experts, driven by ubiquitous low-rank structure in human corpus, and (3) gating mechanisms preferentially route inputs along these dominant directions, further limiting specialization. To address this, we propose Spectral-Decoupled MoE (SD-MoE), which decomposes both parameter and gradient in the spectral space. SD-MoE improves performance across downstream tasks, enables effective expert specialization, incurring minimal additional computation, and can be seamlessly integrated into a wide range of existing MoE architectures, including Qwen and DeepSeek. 

---
# A consequence of failed sequential learning: A computational account of developmental amnesia 

**Authors**: Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12547)  

**Abstract**: Developmental amnesia, featured with severely impaired episodic memory and almost normal semantic memory, has been discovered to occur in children with hippocampal atrophy. This unique combination of characteristics seems to challenge the understanding that early loss of episodic memory may impede cognitive development and result in severe mental retardation. Although a few underlying mechanisms have been suggested, no computational model has been reported that is able to mimic the unique combination of characteristics. In this study, a cognitive system is presented, and developmental amnesia is demonstrated computationally in terms of impaired episodic recall, spared recognition and spared semantic learning. Impaired sequential/spatial learning ability of the hippocampus is suggested to be the cause of such amnesia. Simulation shows that impaired sequential leaning may only result in severe impairment of episodic recall, but affect neither recognition ability nor semantic learning. The spared semantic learning is inline with the view that semantic learning is largely associated with the consolidation of episodic memory, a process in which episodic memory may be mostly activated randomly, instead of sequentially. Furthermore, retrograded amnesia is also simulated, and the result and its mechanism are in agreement with most computational models of amnesia reported previously. 

---
# Decoder-only Conformer with Modality-aware Sparse Mixtures of Experts for ASR 

**Authors**: Jaeyoung Lee, Masato Mimura  

**Link**: [PDF](https://arxiv.org/pdf/2602.12546)  

**Abstract**: We present a decoder-only Conformer for automatic speech recognition (ASR) that processes speech and text in a single stack without external speech encoders or pretrained large language models (LLM). The model uses a modality-aware sparse mixture of experts (MoE): disjoint expert pools for speech and text with hard routing and top-1 selection, embedded in hybrid-causality Conformer blocks (bidirectional for speech, causal for text). Training combines CTC on speech positions with label-smoothed cross-entropy for text generation. Our 113M-parameter model consistently improves WER over a 139M AED baseline on Librispeech (2.8% vs. 3.2% test-clean; 5.6% vs. 6.0% test-other). On Common Voice 16.1 with a single multilingual model across five languages, our approach reduces average WER from 12.2% to 10.6%. To our knowledge, this is the first randomly initialized decoder-only ASR that surpasses strong AED baselines via modality-aware routing and sparse MoE, achieving better accuracy with fewer active parameters and without alignment/adaptation modules. 

---
# Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference 

**Authors**: Pengfei Hu, Chang Lu, Feifan Liu, Yue Ning  

**Link**: [PDF](https://arxiv.org/pdf/2602.12542)  

**Abstract**: Deep learning models for clinical event prediction on electronic health records (EHR) often suffer performance degradation when deployed under different data distributions. While domain adaptation (DA) methods can mitigate such shifts, its "black-box" nature prevents widespread adoption in clinical practice where transparency is essential for trust and safety. We propose ExtraCare to decompose patient representations into invariant and covariant components. By supervising these two components and enforcing their orthogonality during training, our model preserves label information while exposing domain-specific variation at the same time for more accurate predictions than most feature alignment models. More importantly, it offers human-understandable explanations by mapping sparse latent dimensions to medical concepts and quantifying their contributions via targeted ablations. ExtraCare is evaluated on two real-world EHR datasets across multiple domain partition settings, demonstrating superior performance along with enhanced transparency, as evidenced by its accurate predictions and explanations from extensive case studies. 

---
# Bench-MFG: A Benchmark Suite for Learning in Stationary Mean Field Games 

**Authors**: Lorenzo Magnino, Jiacheng Shen, Matthieu Geist, Olivier Pietquin, Mathieu Laurière  

**Link**: [PDF](https://arxiv.org/pdf/2602.12517)  

**Abstract**: The intersection of Mean Field Games (MFGs) and Reinforcement Learning (RL) has fostered a growing family of algorithms designed to solve large-scale multi-agent systems. However, the field currently lacks a standardized evaluation protocol, forcing researchers to rely on bespoke, isolated, and often simplistic environments. This fragmentation makes it difficult to assess the robustness, generalization, and failure modes of emerging methods. To address this gap, we propose a comprehensive benchmark suite for MFGs (Bench-MFG), focusing on the discrete-time, discrete-space, stationary setting for the sake of clarity. We introduce a taxonomy of problem classes, ranging from no-interaction and monotone games to potential and dynamics-coupled games, and provide prototypical environments for each. Furthermore, we propose MF-Garnets, a method for generating random MFG instances to facilitate rigorous statistical testing. We benchmark a variety of learning algorithms across these environments, including a novel black-box approach (MF-PSO) for exploitability minimization. Based on our extensive empirical results, we propose guidelines to standardize future experimental comparisons. Code available at \href{this https URL}{this https URL}. 

---
# Monocular Reconstruction of Neural Tactile Fields 

**Authors**: Pavan Mantripragada, Siddhanth Deshmukh, Eadom Dessalene, Manas Desai, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2602.12508)  

**Abstract**: Robots operating in the real world must plan through environments that deform, yield, and reconfigure under contact, requiring interaction-aware 3D representations that extend beyond static geometric occupancy. To address this, we introduce neural tactile fields, a novel 3D representation that maps spatial locations to the expected tactile response upon contact. Our model predicts these neural tactile fields from a single monocular RGB image -- the first method to do so. When integrated with off-the-shelf path planners, neural tactile fields enable robots to generate paths that avoid high-resistance objects while deliberately routing through low-resistance regions (e.g. foliage), rather than treating all occupied space as equally impassable. Empirically, our learning framework improves volumetric 3D reconstruction by $85.8\%$ and surface reconstruction by $26.7\%$ compared to state-of-the-art monocular 3D reconstruction methods (LRM and Direct3D). 

---
# Favia: Forensic Agent for Vulnerability-fix Identification and Analysis 

**Authors**: André Storhaug, Jiamou Sun, Jingyue Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.12500)  

**Abstract**: Identifying vulnerability-fixing commits corresponding to disclosed CVEs is essential for secure software maintenance but remains challenging at scale, as large repositories contain millions of commits of which only a small fraction address security issues. Existing automated approaches, including traditional machine learning techniques and recent large language model (LLM)-based methods, often suffer from poor precision-recall trade-offs. Frequently evaluated on randomly sampled commits, we uncover that they are substantially underestimating real-world difficulty, where candidate commits are already security-relevant and highly similar. We propose Favia, a forensic, agent-based framework for vulnerability-fix identification that combines scalable candidate ranking with deep and iterative semantic reasoning. Favia first employs an efficient ranking stage to narrow the search space of commits. Each commit is then rigorously evaluated using a ReAct-based LLM agent. By providing the agent with a pre-commit repository as environment, along with specialized tools, the agent tries to localize vulnerable components, navigates the codebase, and establishes causal alignment between code changes and vulnerability root causes. This evidence-driven process enables robust identification of indirect, multi-file, and non-trivial fixes that elude single-pass or similarity-based methods. We evaluate Favia on CVEVC, a large-scale dataset we made that comprises over 8 million commits from 3,708 real-world repositories, and show that it consistently outperforms state-of-the-art traditional and LLM-based baselines under realistic candidate selection, achieving the strongest precision-recall trade-offs and highest F1-scores. 

---
# Human-Like Coarse Object Representations in Vision Models 

**Authors**: Andrey Gizdov, Andrea Procopio, Yichen Li, Daniel Harari, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2602.12486)  

**Abstract**: Humans appear to represent objects for intuitive physics with coarse, volumetric bodies'' that smooth concavities - trading fine visual details for efficient physical predictions - yet their internal structure is largely unknown. Segmentation models, in contrast, optimize pixel-accurate masks that may misalign with such bodies. We ask whether and when these models nonetheless acquire human-like bodies. Using a time-to-collision (TTC) behavioral paradigm, we introduce a comparison pipeline and alignment metric, then vary model training time, size, and effective capacity via pruning. Across all manipulations, alignment with human behavior follows an inverse U-shaped curve: small/briefly trained/pruned models under-segment into blobs; large/fully trained models over-segment with boundary wiggles; and an intermediate ideal body granularity'' best matches humans. This suggests human-like coarse bodies emerge from resource constraints rather than bespoke biases, and points to simple knobs - early checkpoints, modest architectures, light pruning - for eliciting physics-efficient representations. We situate these results within resource-rational accounts balancing recognition detail against physical affordances. 

---
# A Lightweight and Explainable DenseNet-121 Framework for Grape Leaf Disease Classification 

**Authors**: Md. Ehsanul Haque, Md.Saymon Hosen Polash, Rakib Hasan Ovi, Aminul Kader Bulbul, Md Kamrul Siam, Tamim Hasan Saykat  

**Link**: [PDF](https://arxiv.org/pdf/2602.12484)  

**Abstract**: Grapes are among the most economically and culturally significant fruits on a global scale, and table grapes and wine are produced in significant quantities in Europe and Asia. The production and quality of grapes are significantly impacted by grape diseases such as Bacterial Rot, Downy Mildew, and Powdery Mildew. Consequently, the sustainable management of a vineyard necessitates the early and precise identification of these diseases. Current automated methods, particularly those that are based on the YOLO framework, are often computationally costly and lack interpretability that makes them unsuitable for real-world scenarios. This study proposes grape leaf disease classification using Optimized DenseNet 121. Domain-specific preprocessing and extensive connectivity reveal disease-relevant characteristics, including veins, edges, and lesions. An extensive comparison with baseline CNN models, including ResNet18, VGG16, AlexNet, and SqueezeNet, demonstrates that the proposed model exhibits superior performance. It achieves an accuracy of 99.27%, an F1 score of 99.28%, a specificity of 99.71%, and a Kappa of 98.86%, with an inference time of 9 seconds. The cross-validation findings show a mean accuracy of 99.12%, indicating strength and generalizability across all classes. We also employ Grad-CAM to highlight disease-related regions to guarantee the model is highlighting physiologically relevant aspects and increase transparency and confidence. Model optimization reduces processing requirements for real-time deployment, while transfer learning ensures consistency on smaller and unbalanced samples. An effective architecture, domain-specific preprocessing, and interpretable outputs make the proposed framework scalable, precise, and computationally inexpensive for detecting grape leaf diseases. 

---
# Not a Silver Bullet for Loneliness: How Attachment and Age Shape Intimacy with AI Companions 

**Authors**: Raffaele Ciriello, Uri Gal, Ofir Turel  

**Link**: [PDF](https://arxiv.org/pdf/2602.12476)  

**Abstract**: Artificial intelligence (AI) companions are increasingly promoted as solutions for loneliness, often overlooking how personal dispositions and life-stage conditions shape artificial intimacy. Because intimacy is a primary coping mechanism for loneliness that varies by attachment style and age, we examine how different types of users form intimate relationships with AI companions in response to loneliness. Drawing on a hermeneutic literature review and a survey of 277 active AI companion users, we develop and test a model in which loneliness predicts intimacy, moderated by attachment insecurity and conditioned by age. Although the cross-sectional data limits causal inference, the results reveal a differentiated pattern. Loneliness is paradoxically associated with reduced intimacy for securely attached users but with increased intimacy for avoidant and ambivalent users, while anxious users show mixed effects. Older adults report higher intimacy even at lower loneliness levels. These findings challenge portrayals of AI companions as universal remedies for loneliness. Instead, artificial intimacy emerges as a sociotechnical process shaped by psychological dispositions and demographic conditions. The study clarifies who is most likely to form intimate relationships with AI companions and highlights ethical risks in commercial models that may capitalise on user vulnerability. 

---
# Designing RNAs with Language Models 

**Authors**: Milan Gautam, Ning Dai, Tianshuo Zhou, Bowen Xie, David Mathews, Liang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12470)  

**Abstract**: RNA design, the task of finding a sequence that folds into a target secondary structure, has broad biological and biomedical impact but remains computationally challenging due to the exponentially large sequence space and exponentially many competing folds. Traditional approaches treat it as an optimization problem, relying on per-instance heuristics or constraint-based search. We instead reframe RNA design as conditional sequence generation and introduce a reusable neural approximator, instantiated as an autoregressive language model (LM), that maps target structures directly to sequences. We first train our model in a supervised setting on random-induced structure-sequence pairs, and then use reinforcement learning (RL) to optimize end-to-end metrics. We also propose methods to select a small subset for RL that greatly improves RL efficiency and quality. Across four datasets, our approach outperforms state-of-the-art systems on key metrics such as Boltzmann probability while being 1.7x faster, establishing conditional LM generation as a scalable, task-agnostic alternative to per-instance optimization for RNA design. Our code and data are available at this https URL. 

---
# Correctness, Artificial Intelligence, and the Epistemic Value of Mathematical Proof 

**Authors**: James Owen Weatherall, Jesse Wolfson  

**Link**: [PDF](https://arxiv.org/pdf/2602.12463)  

**Abstract**: We argue that it is neither necessary nor sufficient for a mathematical proof to have epistemic value that it be "correct", in the sense of formalizable in a formal proof system. We then present a view on the relationship between mathematics and logic that clarifies the role of formal correctness in mathematics. Finally, we discuss the significance of these arguments for recent discussions about automated theorem provers and applications of AI to mathematics. 

---
# Safe Reinforcement Learning via Recovery-based Shielding with Gaussian Process Dynamics Models 

**Authors**: Alexander W. Goodall, Francesco Belardinelli  

**Link**: [PDF](https://arxiv.org/pdf/2602.12444)  

**Abstract**: Reinforcement learning (RL) is a powerful framework for optimal decision-making and control but often lacks provable guarantees for safety-critical applications. In this paper, we introduce a novel recovery-based shielding framework that enables safe RL with a provable safety lower bound for unknown and non-linear continuous dynamical systems. The proposed approach integrates a backup policy (shield) with the RL agent, leveraging Gaussian process (GP) based uncertainty quantification to predict potential violations of safety constraints, dynamically recovering to safe trajectories only when necessary. Experience gathered by the 'shielded' agent is used to construct the GP models, with policy optimization via internal model-based sampling - enabling unrestricted exploration and sample efficient learning, without compromising safety. Empirically our approach demonstrates strong performance and strict safety-compliance on a suite of continuous control environments. 

---
# Agent Skills for Large Language Models: Architecture, Acquisition, Security, and the Path Forward 

**Authors**: Renjun Xu, Yang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12430)  

**Abstract**: The transition from monolithic language models to modular, skill-equipped agents marks a defining shift in how large language models (LLMs) are deployed in practice. Rather than encoding all procedural knowledge within model weights, agent skills -- composable packages of instructions, code, and resources that agents load on demand -- enable dynamic capability extension without retraining. It is formalized in a paradigm of progressive disclosure, portable skill definitions, and integration with the Model Context Protocol (MCP). This survey provides a comprehensive treatment of the agent skills landscape, as it has rapidly evolved during the last few months. We organize the field along four axes: (i) architectural foundations, examining the this http URL specification, progressive context loading, and the complementary roles of skills and MCP; (ii) skill acquisition, covering reinforcement learning with skill libraries (SAGE), autonomous skill discovery (SEAgent), and compositional skill synthesis; (iii) deployment at scale, including the computer-use agent (CUA) stack, GUI grounding advances, and benchmark progress on OSWorld and SWE-bench; and (iv) security, where recent empirical analyses reveal that 26.1\% of community-contributed skills contain vulnerabilities, motivating our proposed Skill Trust and Lifecycle Governance Framework -- a four-tier, gate-based permission model that maps skill provenance to graduated deployment capabilities. We identify seven open challenges -- from cross-platform skill portability to capability-based permission models -- and propose a research agenda for realizing trustworthy, self-improving skill ecosystems. Unlike prior surveys that broadly cover LLM agents or tool use, this work focuses specifically on the emerging skill abstraction layer and its implications for the next generation of agentic systems. Project repo: this https URL. 

---
# RankLLM: Weighted Ranking of LLMs by Quantifying Question Difficulty 

**Authors**: Ziqian Zhang, Xingjian Hu, Yue Huang, Kai Zhang, Ruoxi Chen, Yixin Liu, Qingsong Wen, Kaidi Xu, Xiangliang Zhang, Neil Zhenqiang Gong, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2602.12424)  

**Abstract**: Benchmarks establish a standardized evaluation framework to systematically assess the performance of large language models (LLMs), facilitating objective comparisons and driving advancements in the field. However, existing benchmarks fail to differentiate question difficulty, limiting their ability to effectively distinguish models' capabilities. To address this limitation, we propose RankLLM, a novel framework designed to quantify both question difficulty and model competency. RankLLM introduces difficulty as the primary criterion for differentiation, enabling a more fine-grained evaluation of LLM capabilities. RankLLM's core mechanism facilitates bidirectional score propagation between models and questions. The core intuition of RankLLM is that a model earns a competency score when it correctly answers a question, while a question's difficulty score increases when it challenges a model. Using this framework, we evaluate 30 models on 35,550 questions across multiple domains. RankLLM achieves 90% agreement with human judgments and consistently outperforms strong baselines such as IRT. It also exhibits strong stability, fast convergence, and high computational efficiency, making it a practical solution for large-scale, difficulty-aware LLM evaluation. 

---
# CacheMind: From Miss Rates to Why -- Natural-Language, Trace-Grounded Reasoning for Cache Replacement 

**Authors**: Kaushal Mhapsekar, Azam Ghanbari, Bita Aslrousta, Samira Mirbagher-Ajorpaz  

**Link**: [PDF](https://arxiv.org/pdf/2602.12422)  

**Abstract**: Cache replacement remains a challenging problem in CPU microarchitecture, often addressed using hand-crafted heuristics, limiting cache performance. Cache data analysis requires parsing millions of trace entries with manual filtering, making the process slow and non-interactive. To address this, we introduce CacheMind, a conversational tool that uses Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to enable semantic reasoning over cache traces. Architects can now ask natural language questions like, "Why is the memory access associated with PC X causing more evictions?", and receive trace-grounded, human-readable answers linked to program semantics for the first time. To evaluate CacheMind, we present CacheMindBench, the first verified benchmark suite for LLM-based reasoning for the cache replacement problem. Using the SIEVE retriever, CacheMind achieves 66.67% on 75 unseen trace-grounded questions and 84.80% on 25 unseen policy-specific reasoning tasks; with RANGER, it achieves 89.33% and 64.80% on the same evaluations. Additionally, with RANGER, CacheMind achieves 100% accuracy on 4 out of 6 categories in the trace-grounded tier of CacheMindBench. Compared to LlamaIndex (10% retrieval success), SIEVE achieves 60% and RANGER achieves 90%, demonstrating that existing Retrieval-Augmented Generation (RAGs) are insufficient for precise, trace-grounded microarchitectural reasoning. We provided four concrete actionable insights derived using CacheMind, wherein bypassing use case improved cache hit rate by 7.66% and speedup by 2.04%, software fix use case gives speedup of 76%, and Mockingjay replacement policy use case gives speedup of 0.7%; showing the utility of CacheMind on non-trivial queries that require a natural-language interface. 

---
# Soft Contamination Means Benchmarks Test Shallow Generalization 

**Authors**: Ari Spiesberger, Juan J. Vazquez, Nicky Pochinkov, Tomáš Gavenčiak, Peli Grietzer, Gavin Leech, Nandi Schoots  

**Link**: [PDF](https://arxiv.org/pdf/2602.12413)  

**Abstract**: If LLM training data is polluted with benchmark test data, then benchmark performance gives biased estimates of out-of-distribution (OOD) generalization. Typical decontamination filters use n-gram matching which fail to detect semantic duplicates: sentences with equivalent (or near-equivalent) content that are not close in string space. We study this soft contamination of training data by semantic duplicates. Among other experiments, we embed the Olmo3 training corpus and find that: 1) contamination remains widespread, e.g. we find semantic duplicates for 78% of CodeForces and exact duplicates for 50% of ZebraLogic problems; 2) including semantic duplicates of benchmark data in training does improve benchmark performance; and 3) when finetuning on duplicates of benchmark datapoints, performance also improves on truly-held-out datapoints from the same benchmark. We argue that recent benchmark gains are thus confounded: the prevalence of soft contamination means gains reflect both genuine capability improvements and the accumulation of test data and effective test data in growing training corpora. 

---
# AstRL: Analog and Mixed-Signal Circuit Synthesis with Deep Reinforcement Learning 

**Authors**: Felicia B. Guo, Ken T. Ho, Andrei Vladimirescu, Borivoje Nikolic  

**Link**: [PDF](https://arxiv.org/pdf/2602.12402)  

**Abstract**: Analog and mixed-signal (AMS) integrated circuits (ICs) lie at the core of modern computing and communications systems. However, despite the continued rise in design complexity, advances in AMS automation remain limited. This reflects the central challenge in developing a generalized optimization method applicable across diverse circuit design spaces, many of which are distinct, constrained, and non-differentiable. To address this, our work casts circuit design as a graph generation problem and introduces a novel method of AMS synthesis driven by deep reinforcement learning (AstRL). Based on a policy-gradient approach, AstRL generates circuits directly optimized for user-specified targets within a simulator-embedded environment that provides ground-truth feedback during training. Through behavioral-cloning and discriminator-based similarity rewards, our method demonstrates, for the first time, an expert-aligned paradigm for generalized circuit generation validated in simulation. Importantly, the proposed approach operates at the level of individual transistors, enabling highly expressive, fine-grained topology generation. Strong inductive biases encoded in the action space and environment further drive structurally consistent and valid generation. Experimental results for three realistic design tasks illustrate substantial improvements in conventional design metrics over state-of-the-art baselines, with 100% of generated designs being structurally correct and over 90% demonstrating required functionality. 

---
# What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis 

**Authors**: Xirui Li, Ming Li, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.12395)  

**Abstract**: Reinforcement learning (RL) with verifiable rewards has become a standard post-training stage for boosting visual reasoning in vision-language models, yet it remains unclear what capabilities RL actually improves compared with supervised fine-tuning as cold-start initialization (IN). End-to-end benchmark gains conflate multiple factors, making it difficult to attribute improvements to specific skills. To bridge the gap, we propose a Frankenstein-style analysis framework including: (i) functional localization via causal probing; (ii) update characterization via parameter comparison; and (iii) transferability test via model merging. Instead, RL induces a consistent inference-time shift primarily in mid-to-late layers, and these mid-to-late refinements are both transferable (via merging) and necessary (via freezing) for RL gains. Overall, our results suggest that RL's reliable contribution in visual reasoning is not a uniform enhancement of visual perception, but a systematic refinement of mid-to-late transformer computation that improves vision-to-reasoning alignment and reasoning performance, highlighting the limitations of benchmark-only evaluation for understanding multimodal reasoning improvements. 

---
# Reproducing DragDiffusion: Interactive Point-Based Editing with Diffusion Models 

**Authors**: Ali Subhan, Ashir Raza  

**Link**: [PDF](https://arxiv.org/pdf/2602.12393)  

**Abstract**: DragDiffusion is a diffusion-based method for interactive point-based image editing that enables users to manipulate images by directly dragging selected points. The method claims that accurate spatial control can be achieved by optimizing a single diffusion latent at an intermediate timestep, together with identity-preserving fine-tuning and spatial regularization. This work presents a reproducibility study of DragDiffusion using the authors' released implementation and the DragBench benchmark. We reproduce the main ablation studies on diffusion timestep selection, LoRA-based fine-tuning, mask regularization strength, and UNet feature supervision, and observe close agreement with the qualitative and quantitative trends reported in the original work. At the same time, our experiments show that performance is sensitive to a small number of hyperparameter assumptions, particularly the optimized timestep and the feature level used for motion supervision, while other components admit broader operating ranges. We further evaluate a multi-timestep latent optimization variant and find that it does not improve spatial accuracy while substantially increasing computational cost. Overall, our findings support the central claims of DragDiffusion while clarifying the conditions under which they are reliably reproducible. Code is available at this https URL. 

---
# Rational Neural Networks have Expressivity Advantages 

**Authors**: Maosen Tang, Alex Townsend  

**Link**: [PDF](https://arxiv.org/pdf/2602.12390)  

**Abstract**: We study neural networks with trainable low-degree rational activation functions and show that they are more expressive and parameter-efficient than modern piecewise-linear and smooth activations such as ELU, LeakyReLU, LogSigmoid, PReLU, ReLU, SELU, CELU, Sigmoid, SiLU, Mish, Softplus, Tanh, Softmin, Softmax, and LogSoftmax. For an error target of $\varepsilon>0$, we establish approximation-theoretic separations: Any network built from standard fixed activations can be uniformly approximated on compact domains by a rational-activation network with only $\mathrm{poly}(\log\log(1/\varepsilon))$ overhead in size, while the converse provably requires $\Omega(\log(1/\varepsilon))$ parameters in the worst case. This exponential gap persists at the level of full networks and extends to gated activations and transformer-style nonlinearities. In practice, rational activations integrate seamlessly into standard architectures and training pipelines, allowing rationals to match or outperform fixed activations under identical architectures and optimizers. 

---
# Why Deep Jacobian Spectra Separate: Depth-Induced Scaling and Singular-Vector Alignment 

**Authors**: Nathanaël Haas, Francçois Gatine, Augustin M Cosse, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2602.12384)  

**Abstract**: Understanding why gradient-based training in deep networks exhibits strong implicit bias remains challenging, in part because tractable singular-value dynamics are typically available only for balanced deep linear models. We propose an alternative route based on two theoretically grounded and empirically testable signatures of deep Jacobians: depth-induced exponential scaling of ordered singular values and strong spectral separation. Adopting a fixed-gates view of piecewise-linear networks, where Jacobians reduce to products of masked linear maps within a single activation region, we prove the existence of Lyapunov exponents governing the top singular values at initialization, give closed-form expressions in a tractable masked model, and quantify finite-depth corrections. We further show that sufficiently strong separation forces singular-vector alignment in matrix products, yielding an approximately shared singular basis for intermediate Jacobians. Together, these results motivate an approximation regime in which singular-value dynamics become effectively decoupled, mirroring classical balanced deep-linear analyses without requiring balancing. Experiments in fixed-gates settings validate the predicted scaling, alignment, and resulting dynamics, supporting a mechanistic account of emergent low-rank Jacobian structure as a driver of implicit bias. 

---
# TFT-ACB-XML: Decision-Level Integration of Customized Temporal Fusion Transformer and Attention-BiLSTM with XGBoost Meta-Learner for BTC Price Forecasting 

**Authors**: Raiz Ud Din, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.12380)  

**Abstract**: Accurate forecasting of Bitcoin (BTC) has always been a challenge because decentralized markets are non-linear, highly volatile, and have temporal irregularities. Existing deep learning models often struggle with interpretability and generalization across diverse market conditions. This research presents a hybrid stacked-generalization framework, TFT-ACB-XML, for BTC closing price prediction. The framework integrates two parallel base learners: a customized Temporal Fusion Transformer (TFT) and an Attention-Customized Bidirectional Long Short-Term Memory network (ACB), followed by an XGBoost regressor as the meta-learner. The customized TFT model handles long-range dependencies and global temporal dynamics via variable selection networks and interpretable single-head attention. The ACB module uses a new attention mechanism alongside the customized BiLSTM to capture short-term sequential dependencies. Predictions from both customized TFT and ACB are weighted through an error-reciprocal weighting strategy. These weights are derived from validation performance, where a model showing lower prediction error receives a higher weight. Finally, the framework concatenates these weighted outputs into a feature vector and feeds the vector to an XGBoost regressor, which captures non-linear residuals and produces the final BTC closing price prediction. Empirical validation using BTC data from October 1, 2014, to January 5, 2026, shows improved performance of the proposed framework compared to recent Deep Learning and Transformer baseline models. The results show a MAPE of 0.65%, an MAE of 198.15, and an RMSE of 258.30 for one-step-ahead out-of-sample under a walk-forward evaluation on the test block. The evaluation period spans the 2024 BTC halving and the spot ETFs (exchange-traded funds) period, which coincide with major liquidity and volatility shifts. 

---
# Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning 

**Authors**: Abdul Wahab, Raksha Kumaraswamy, Martha White  

**Link**: [PDF](https://arxiv.org/pdf/2602.12375)  

**Abstract**: Optimistic value estimates provide one mechanism for directed exploration in reinforcement learning (RL). The agent acts greedily with respect to an estimate of the value plus what can be seen as a value bonus. The value bonus can be learned by estimating a value function on reward bonuses, propagating local uncertainties around rewards. However, this approach only increases the value bonus for an action retroactively, after seeing a higher reward bonus from that state and action. Such an approach does not encourage the agent to visit a state and action for the first time. In this work, we introduce an algorithm for exploration called Value Bonuses with Ensemble errors (VBE), that maintains an ensemble of random action-value functions (RQFs). VBE uses the errors in the estimation of these RQFs to design value bonuses that provide first-visit optimism and deep exploration. The key idea is to design the rewards for these RQFs in such a way that the value bonus can decrease to zero. We show that VBE outperforms Bootstrap DQN and two reward bonus approaches (RND and ACB) on several classic environments used to test exploration and provide demonstrative experiments that it can scale easily to more complex environments like Atari. 

---
# Policy4OOD: A Knowledge-Guided World Model for Policy Intervention Simulation against the Opioid Overdose Crisis 

**Authors**: Yijun Ma, Zehong Wang, Weixiang Sun, Zheyuan Zhang, Kaiwen Shi, Nitesh Chawla, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2602.12373)  

**Abstract**: The opioid epidemic remains one of the most severe public health crises in the United States, yet evaluating policy interventions before implementation is difficult: multiple policies interact within a dynamic system where targeting one risk pathway may inadvertently amplify another. We argue that effective opioid policy evaluation requires three capabilities -- forecasting future outcomes under current policies, counterfactual reasoning about alternative past decisions, and optimization over candidate interventions -- and propose to unify them through world modeling. We introduce Policy4OOD, a knowledge-guided spatio-temporal world model that addresses three core challenges: what policies prescribe, where effects manifest, and when effects unfold.Policy4OOD jointly encodes policy knowledge graphs, state-level spatial dependencies, and socioeconomic time series into a policy-conditioned Transformer that forecasts future opioid this http URL trained, the world model serves as a simulator: forecasting requires only a forward pass, counterfactual analysis substitutes alternative policy encodings in the historical sequence, and policy optimization employs Monte Carlo Tree Search over the learned simulator. To support this framework, we construct a state-level monthly dataset (2019--2024) integrating opioid mortality, socioeconomic indicators, and structured policy encodings. Experiments demonstrate that spatial dependencies and structured policy knowledge significantly improve forecasting accuracy, validating each architectural component and the potential of world modeling for data-driven public health decision support. 

---
# Intrinsic Credit Assignment for Long Horizon Interaction 

**Authors**: Ilze Amanda Auzina, Joschka Strüber, Sergio Hernández-Gutiérrez, Shashwat Goel, Ameya Prabhu, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2602.12342)  

**Abstract**: How can we train agents to navigate uncertainty over long horizons? In this work, we propose {\Delta}Belief-RL, which leverages a language model's own intrinsic beliefs to reward intermediate progress. Our method utilizes the change in the probability an agent assigns to the target solution for credit assignment. By training on synthetic interaction data, {\Delta}Belief-RL teaches information-seeking capabilities that consistently outperform purely outcome-based rewards for Reinforcement Learning, with improvements generalizing to out-of-distribution applications ranging from customer service to personalization. Notably, the performance continues to improve as we scale test-time interactions beyond the training horizon, with interaction-efficiency increasing even on Pass@k metrics. Overall, our work introduces a scalable training strategy for navigating uncertainty over a long-horizon, by enabling credit assignment to intermediate actions via intrinsic {\Delta}Belief rewards. 

---
# ForeAct: Steering Your VLA with Efficient Visual Foresight Planning 

**Authors**: Zhuoyang Zhang, Shang Yang, Qinghao Hu, Luke J. Huang, James Hou, Yufei Sun, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.12322)  

**Abstract**: Vision-Language-Action (VLA) models convert high-level language instructions into concrete, executable actions, a task that is especially challenging in open-world environments. We present Visual Foresight Planning (ForeAct), a general and efficient planner that guides a VLA step-by-step using imagined future observations and subtask descriptions. With an imagined future observation, the VLA can focus on visuo-motor inference rather than high-level semantic reasoning, leading to improved accuracy and generalization. Our planner comprises a highly efficient foresight image generation module that predicts a high-quality 640$\times$480 future observation from the current visual input and language instruction within only 0.33s on an H100 GPU, together with a vision-language model that reasons over the task and produces subtask descriptions for both the generator and the VLA. Importantly, state-of-the-art VLAs can integrate our planner seamlessly by simply augmenting their visual inputs, without any architectural modification. The foresight generator is pretrained on over 1 million multi-task, cross-embodiment episodes, enabling it to learn robust embodied dynamics. We evaluate our framework on a benchmark that consists of 11 diverse, multi-step real-world tasks. It achieves an average success rate of 87.4%, demonstrating a +40.9% absolute improvement over the $\pi_0$ baseline (46.5%) and a +30.3% absolute improvement over $\pi_0$ augmented with textual subtask guidance (57.1%). 

---
# Free Lunch in Medical Image Foundation Model Pre-training via Randomized Synthesis and Disentanglement 

**Authors**: Yuhan Wei, Yuting He, Linshan Wu, Fuxiang Huang, Junlin Hou, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.12317)  

**Abstract**: Medical image foundation models (MIFMs) have demonstrated remarkable potential for a wide range of clinical tasks, yet their development is constrained by the scarcity, heterogeneity, and high cost of large-scale annotated datasets. Here, we propose RaSD (Randomized Synthesis and Disentanglement), a scalable framework for pre-training MIFMs entirely on synthetic data. By modeling anatomical structures and appearance variations with randomized Gaussian distributions, RaSD exposes models to sufficient multi-scale structural and appearance perturbations, forcing them to rely on invariant and task-relevant anatomical cues rather than dataset-specific textures, thereby enabling robust and transferable representation learning. We pre-trained RaSD on 1.2 million 3D volumes and 9.6 million 2D images, and extensively evaluated the resulting models across 6 imaging modalities, 48 datasets, and 56 downstream tasks. Across all evaluated downstream tasks, RaSD consistently outperforms training-from-scratch models, achieves the best performance on 17 tasks, and remains comparable to models pre-trained on large real datasets in most others. These results demonstrate that the capacity of synthetic data alone to drive robust representation learning. Our findings establish a paradigm shift in medical AI, demonstrating that synthetic data can serve as a "free lunch" for scalable, privacy-preserving, and clinically generalizable foundation models. 

---
# AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping 

**Authors**: Sunghwan Kim, Ryang Heo, Yongsik Seo, Jinyoung Yeo, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.12315)  

**Abstract**: The proliferation of e-commerce has made web shopping platforms key gateways for customers navigating the vast digital marketplace. Yet this rapid expansion has led to a noisy and fragmented information environment, increasing cognitive burden as shoppers explore and purchase products online. With promising potential to alleviate this challenge, agentic systems have garnered growing attention for automating user-side tasks in web shopping. Despite significant advancements, existing benchmarks fail to comprehensively evaluate how well agentic systems can curate products in open-web settings. Specifically, they have limited coverage of shopping scenarios, focusing only on simplified single-platform lookups rather than exploratory search. Moreover, they overlook personalization in evaluation, leaving unclear whether agents can adapt to diverse user preferences in realistic shopping contexts. To address this gap, we present AgenticShop, the first benchmark for evaluating agentic systems on personalized product curation in open-web environment. Crucially, our approach features realistic shopping scenarios, diverse user profiles, and a verifiable, checklist-driven personalization evaluation framework. Through extensive experiments, we demonstrate that current agentic systems remain largely insufficient, emphasizing the need for user-side systems that effectively curate tailored products across the modern web. 

---
# Visible and Hyperspectral Imaging for Quality Assessment of Milk: Property Characterisation and Identification 

**Authors**: Massimo Martinelli, Elena Tomassi, Nafiou Arouna, Morena Gabriele, Laryssa Perez Fabbri, Luisa Pozzo, Giuseppe Conte, Davide Moroni, Laura Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2602.12313)  

**Abstract**: Rapid and non-destructive assessment of milk quality is crucial to ensuring both nutritional value and food safety. In this study, we investigated the potential of visible and hyperspectral imaging as cost-effective and quick-response alternatives to conventional chemical analyses for characterizing key properties of cowś milk. A total of 52 milk samples were analysed to determine their biochemical composition (polyphenols, antioxidant capacity, and fatty acids) using spectrophotometer methods and standard gas-liquid and high-performance liquid chromatography (GLC/HPLC). Concurrently, visible (RGB) images were captured using a standard smartphone, and hyperspectral data were acquired in the near-infrared range. A comprehensive analytical framework, including eleven different machine learning algorithms, was employed to correlate imaging features with biochemical measurements. Analysis of visible images accurately distinguished between fresh samples and those stored for 12 days (100 percent accuracy) and achieved perfect discrimination between antibiotic-treated and untreated groups (100 percent accuracy). Moreover, image-derived features enabled perfect prediction of the polyphenols content and the antioxidant capacity using an XGBoost model. Hyperspectral imaging further achieved classification accuracies exceeding 95 percent for several individual fatty acids and 94.8 percent for treatment groups using a Random Forest model. These findings demonstrate that both visible and hyperspectral imaging, when coupled with machine learning, are powerful, non-invasive tools for the rapid assessment of milkś chemical and nutritional profiles, highlighting the strong potential of imaging-based approaches for milk quality assessment. 

---
# Perceptual Self-Reflection in Agentic Physics Simulation Code Generation 

**Authors**: Prashant Shende, Bradley Camburn  

**Link**: [PDF](https://arxiv.org/pdf/2602.12311)  

**Abstract**: We present a multi-agent framework for generating physics simulation code from natural language descriptions, featuring a novel perceptual self-reflection mechanism for validation. The system employs four specialized agents: a natural language interpreter that converts user requests into physics-based descriptions; a technical requirements generator that produces scaled simulation parameters; a physics code generator with automated self-correction; and a physics validator that implements perceptual self-reflection. The key innovation is perceptual validation, which analyzes rendered animation frames using a vision-capable language model rather than inspecting code structure directly. This approach addresses the ``oracle gap'' where syntactically correct code produces physically incorrect behavior--a limitation that conventional testing cannot detect. We evaluate the system across seven domains including classical mechanics, fluid dynamics, thermodynamics, electromagnetics, wave physics, reaction-diffusion systems, and non-physics data visualization. The perceptual self-reflection architecture demonstrates substantial improvement over single-shot generation baselines, with the majority of tested scenarios achieving target physics accuracy thresholds. The system exhibits robust pipeline stability with consistent code self-correction capability, operating at approximately \$0.20 per animation. These results validate our hypothesis that feeding visual simulation outputs back to a vision-language model for iterative refinement significantly outperforms single-shot code generation for physics simulation tasks and highlights the potential of agentic AI to support engineering workflows and physics data generation pipelines. 

---
# Quantum walk inspired JPEG compression of images 

**Authors**: Abhishek Verma, Sahil Tomar, Sandeep Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2602.12306)  

**Abstract**: This work proposes a quantum inspired adaptive quantization framework that enhances the classical JPEG compression by introducing a learned, optimized Qtable derived using a Quantum Walk Inspired Optimization (QWIO) search strategy. The optimizer searches a continuous parameter space of frequency band scaling factors under a unified rate distortion objective that jointly considers reconstruction fidelity and compression efficiency. The proposed framework is evaluated on MNIST, CIFAR10, and ImageNet subsets, using Peak Signal to Noise Ratio (PSNR), Structural Similarity Index (SSIM), Bits Per Pixel (BPP), and error heatmap visual analysis as evaluation metrics. Experimental results show average gains ranging from 3 to 6 dB PSNR, along with better structural preservation of edges, contours, and luminance transitions, without modifying decoder compatibility. The structure remains JPEG compliant and can be implemented using accessible scientific packages making it ideal for deployment and practical research use. 

---
# OptiML: An End-to-End Framework for Program Synthesis and CUDA Kernel Optimization 

**Authors**: Arijit Bhattacharjee, Heng Ping, Son Vu Le, Paul Bogdan, Nesreen K. Ahmed, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2602.12305)  

**Abstract**: Generating high-performance CUDA kernels remains challenging due to the need to navigate a combinatorial space of low-level transformations under noisy and expensive hardware feedback. Although large language models can synthesize functionally correct CUDA code, achieving competitive performance requires systematic exploration and verification of optimization choices. We present OptiML, an end-to-end framework that maps either natural-language intent or input CUDA code to performance-optimized CUDA kernels by formulating kernel optimization as search under verification. OptiML consists of two decoupled stages. When the input is natural language, a Mixture-of-Thoughts generator (OptiML-G) acts as a proposal policy over kernel implementation strategies, producing an initial executable program. A search-based optimizer (OptiML-X) then refines either synthesized or user-provided kernels using Monte Carlo Tree Search over LLM-driven edits, guided by a hardware-aware reward derived from profiler feedback. Each candidate transformation is compiled, verified, and profiled with Nsight Compute, and evaluated by a composite objective that combines runtime with hardware bottleneck proxies and guardrails against regressions. We evaluate OptiML in both synthesis-and-optimize and optimization-only settings on a diverse suite of CUDA kernels. Results show that OptiML consistently discovers verified performance improvements over strong LLM baselines and produces interpretable optimization trajectories grounded in profiler evidence. 

---
# OmniCustom: Sync Audio-Video Customization Via Joint Audio-Video Generation Model 

**Authors**: Maomao Li, Zhen Li, Kaipeng Zhang, Guosheng Yin, Zhifeng Li, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12304)  

**Abstract**: Existing mainstream video customization methods focus on generating identity-consistent videos based on given reference images and textual prompts. Benefiting from the rapid advancement of joint audio-video generation, this paper proposes a more compelling new task: sync audio-video customization, which aims to synchronously customize both video identity and audio timbre. Specifically, given a reference image $I^{r}$ and a reference audio $A^{r}$, this novel task requires generating videos that maintain the identity of the reference image while imitating the timbre of the reference audio, with spoken content freely specifiable through user-provided textual prompts. To this end, we propose OmniCustom, a powerful DiT-based audio-video customization framework that can synthesize a video following reference image identity, audio timbre, and text prompts all at once in a zero-shot manner. Our framework is built on three key contributions. First, identity and audio timbre control are achieved through separate reference identity and audio LoRA modules that operate through self-attention layers within the base audio-video generation model. Second, we introduce a contrastive learning objective alongside the standard flow matching objective. It uses predicted flows conditioned on reference inputs as positive examples and those without reference conditions as negative examples, thereby enhancing the model ability to preserve identity and timbre. Third, we train OmniCustom on our constructed large-scale, high-quality audio-visual human dataset. Extensive experiments demonstrate that OmniCustom outperforms existing methods in generating audio-video content with consistent identity and timbre fidelity. 

---
# Adaptive traffic signal control optimization using a novel road partition and multi-channel state representation method 

**Authors**: Maojiang Deng, Shoufeng Lu, Jiazhao Shi, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.12296)  

**Abstract**: This study proposes a novel adaptive traffic signal control method leveraging a Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to optimize signal timing by integrating variable cell length and multi-channel state representation. A road partition formula consisting of the sum of logarithmic and linear functions was proposed. The state variables are a vector composed of three channels: the number of vehicles, the average speed, and space occupancy. The set of available signal phases constitutes the action space, the selected phase is executed with a fixed green time. The reward function is formulated using the absolute values of key traffic state metrics - waiting time, speed, and fuel consumption. Each metric is normalized by a typical maximum value and assigned a weight that reflects its priority and optimization direction. The simulation results, using Sumo-TensorFlow-Python, demonstrate a cross-range transferability evaluation and show that the proposed variable cell length and multi-channel state representation method excels compared to fixed cell length in optimization performance. 

---
# Energy-Aware Reinforcement Learning for Robotic Manipulation of Articulated Components in Infrastructure Operation and Maintenance 

**Authors**: Xiaowen Tao, Yinuo Wang, Haitao Ding, Yuanyang Qi, Ziyu Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.12288)  

**Abstract**: With the growth of intelligent civil infrastructure and smart cities, operation and maintenance (O&M) increasingly requires safe, efficient, and energy-conscious robotic manipulation of articulated components, including access doors, service drawers, and pipeline valves. However, existing robotic approaches either focus primarily on grasping or target object-specific articulated manipulation, and they rarely incorporate explicit actuation energy into multi-objective optimisation, which limits their scalability and suitability for long-term deployment in real O&M settings. Therefore, this paper proposes an articulation-agnostic and energy-aware reinforcement learning framework for robotic manipulation in intelligent infrastructure O&M. The method combines part-guided 3D perception, weighted point sampling, and PointNet-based encoding to obtain a compact geometric representation that generalises across heterogeneous articulated objects. Manipulation is formulated as a Constrained Markov Decision Process (CMDP), in which actuation energy is explicitly modelled and regulated via a Lagrangian-based constrained Soft Actor-Critic scheme. The policy is trained end-to-end under this CMDP formulation, enabling effective articulated-object operation while satisfying a long-horizon energy budget. Experiments on representative O&M tasks demonstrate 16%-30% reductions in energy consumption, 16%-32% fewer steps to success, and consistently high success rates, indicating a scalable and sustainable solution for infrastructure O&M manipulation. 

---
# Retrieval-Augmented Self-Taught Reasoning Model with Adaptive Chain-of-Thought for ASR Named Entity Correction 

**Authors**: Junjie An, Jingguang Tian, Tianyi Wang, Yu Gao, Xiaofeng Mou, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.12287)  

**Abstract**: End-to-end automatic speech recognition (ASR) systems frequently misrecognize domain-specific phrases like named entities, which can cause catastrophic failures in downstream tasks. A new family of named entity correction methods based on large language models (LLMs) has recently emerged. However, these approaches have yet to fully exploit the sophisticated reasoning capabilities inherent to LLMs. To bridge this gap, we propose a novel retrieval-augmented generation framework for correcting named entity errors in ASR. Our approach consists of two key components: (1) a rephrasing language model (RLM) for named entity recognition, followed by candidate retrieval using a phonetic-level edit distance; and (2) a novel self-taught reasoning model with adaptive chain-of-thought (A-STAR) that dynamically adjusts the depth of its reasoning based on task difficulty. Experiments on the AISHELL-1 and Homophone datasets demonstrate the effectiveness of our method, which achieves relative reductions in the named entity character error rate of 17.96\% and 34.42\%, respectively, compared to a strong baseline. 

---
# From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness 

**Authors**: Linbo Cao, Lihao Sun, Yang Yue  

**Link**: [PDF](https://arxiv.org/pdf/2602.12285)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed as autonomous agents capable of actions with real-world impacts beyond text generation. While persona-induced biases in text generation are well documented, their effects on agent task performance remain largely unexplored, even though such effects pose more direct operational risks. In this work, we present the first systematic case study showing that demographic-based persona assignments can alter LLM agents' behavior and degrade performance across diverse domains. Evaluating widely deployed models on agentic benchmarks spanning strategic reasoning, planning, and technical operations, we uncover substantial performance variations - up to 26.2% degradation, driven by task-irrelevant persona cues. These shifts appear across task types and model architectures, indicating that persona conditioning and simple prompt injections can distort an agent's decision-making reliability. Our findings reveal an overlooked vulnerability in current LLM agentic systems: persona assignments can introduce implicit biases and increase behavioral volatility, raising concerns for the safe and robust deployment of LLM agents. 

---
# A Lightweight LLM Framework for Disaster Humanitarian Information Classification 

**Authors**: Han Jinzhen, Kim Jisung, Yang Jong Soo, Yun Hong Sik  

**Link**: [PDF](https://arxiv.org/pdf/2602.12284)  

**Abstract**: Timely classification of humanitarian information from social media is critical for effective disaster response. However, deploying large language models (LLMs) for this task faces challenges in resource-constrained emergency settings. This paper develops a lightweight, cost-effective framework for disaster tweet classification using parameter-efficient fine-tuning. We construct a unified experimental corpus by integrating and normalizing the HumAID dataset (76,484 tweets across 19 disaster events) into a dual-task benchmark: humanitarian information categorization and event type identification. Through systematic evaluation of prompting strategies, LoRA fine-tuning, and retrieval-augmented generation (RAG) on Llama 3.1 8B, we demonstrate that: (1) LoRA achieves 79.62% humanitarian classification accuracy (+37.79% over zero-shot) while training only ~2% of parameters; (2) QLoRA enables efficient deployment with 99.4% of LoRA performance at 50% memory cost; (3) contrary to common assumptions, RAG strategies degrade fine-tuned model performance due to label noise from retrieved examples. These findings establish a practical, reproducible pipeline for building reliable crisis intelligence systems with limited computational resources. 

---
# Peak + Accumulation: A Proxy-Level Scoring Formula for Multi-Turn LLM Attack Detection 

**Authors**: J Alex Corll  

**Link**: [PDF](https://arxiv.org/pdf/2602.11247)  

**Abstract**: Multi-turn prompt injection attacks distribute malicious intent across multiple conversation turns, exploiting the assumption that each turn is evaluated independently. While single-turn detection has been extensively studied, no published formula exists for aggregating per-turn pattern scores into a conversation-level risk score at the proxy layer -- without invoking an LLM. We identify a fundamental flaw in the intuitive weighted-average approach: it converges to the per-turn score regardless of turn count, meaning a 20-turn persistent attack scores identically to a single suspicious turn. Drawing on analogies from change-point detection (CUSUM), Bayesian belief updating, and security risk-based alerting, we propose peak + accumulation scoring -- a formula combining peak single-turn risk, persistence ratio, and category diversity. Evaluated on 10,654 multi-turn conversations -- 588 attacks sourced from WildJailbreak adversarial prompts and 10,066 benign conversations from WildChat -- the formula achieves 90.8% recall at 1.20% false positive rate with an F1 of 85.9%. A sensitivity analysis over the persistence parameter reveals a phase transition at rho ~ 0.4, where recall jumps 12 percentage points with negligible FPR increase. We release the scoring algorithm, pattern library, and evaluation harness as open source. 

---
# Language-Guided Invariance Probing of Vision-Language Models 

**Authors**: Jae Joong Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.13494)  

**Abstract**: Recent vision-language models (VLMs) such as CLIP, OpenCLIP, EVA02-CLIP and SigLIP achieve strong zero-shot performance, but it is unclear how reliably they respond to controlled linguistic perturbations. We introduce Language-Guided Invariance Probing (LGIP), a benchmark that measures (i) invariance to meaning-preserving paraphrases and (ii) sensitivity to meaning-changing semantic flips in image-text matching. Using 40k MS COCO images with five human captions each, we automatically generate paraphrases and rule-based flips that alter object category, color or count, and summarize model behavior with an invariance error, a semantic sensitivity gap and a positive-rate statistic.
Across nine VLMs, EVA02-CLIP and large OpenCLIP variants lie on a favorable invariance-sensitivity frontier, combining low paraphrase-induced variance with consistently higher scores for original captions than for their flipped counterparts. In contrast, SigLIP and SigLIP2 show much larger invariance error and often prefer flipped captions to the human descriptions, especially for object and color edits. These failures are largely invisible to standard retrieval metrics, indicating that LGIP provides a model-agnostic diagnostic for the linguistic robustness of VLMs beyond conventional accuracy scores. 

---
