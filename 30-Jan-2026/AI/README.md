# Exploring Reasoning Reward Model for Agents 

**Authors**: Kaixuan Fan, Kaituo Feng, Manyuan Zhang, Tianshuo Peng, Zhixun Li, Yilei Jiang, Shuang Chen, Peng Pei, Xunliang Cai, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2601.22154)  

**Abstract**: Agentic Reinforcement Learning (Agentic RL) has achieved notable success in enabling agents to perform complex reasoning and tool use. However, most methods still relies on sparse outcome-based reward for training. Such feedback fails to differentiate intermediate reasoning quality, leading to suboptimal training results. In this paper, we introduce Agent Reasoning Reward Model (Agent-RRM), a multi-faceted reward model that produces structured feedback for agentic trajectories, including (1) an explicit reasoning trace , (2) a focused critique that provides refinement guidance by highlighting reasoning flaws, and (3) an overall score that evaluates process performance. Leveraging these signals, we systematically investigate three integration strategies: Reagent-C (text-augmented refinement), Reagent-R (reward-augmented guidance), and Reagent-U (unified feedback integration). Extensive evaluations across 12 diverse benchmarks demonstrate that Reagent-U yields substantial performance leaps, achieving 43.7% on GAIA and 46.2% on WebWalkerQA, validating the effectiveness of our reasoning reward model and training schemes. Code, models, and datasets are all released to facilitate future research. 

---
# Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data 

**Authors**: Grzegorz Stefanski, Alberto Presta, Michal Byra  

**Link**: [PDF](https://arxiv.org/pdf/2601.22141)  

**Abstract**: In pruning, the Lottery Ticket Hypothesis posits that large networks contain sparse subnetworks, or winning tickets, that can be trained in isolation to match the performance of their dense counterparts. However, most existing approaches assume a single universal winning ticket shared across all inputs, ignoring the inherent heterogeneity of real-world data. In this work, we propose Routing the Lottery (RTL), an adaptive pruning framework that discovers multiple specialized subnetworks, called adaptive tickets, each tailored to a class, semantic cluster, or environmental condition. Across diverse datasets and tasks, RTL consistently outperforms single- and multi-model baselines in balanced accuracy and recall, while using up to 10 times fewer parameters than independent models and exhibiting semantically aligned. Furthermore, we identify subnetwork collapse, a performance drop under aggressive pruning, and introduce a subnetwork similarity score that enables label-free diagnosis of oversparsification. Overall, our results recast pruning as a mechanism for aligning model structure with data heterogeneity, paving the way toward more modular and context-aware deep learning. 

---
# World of Workflows: a Benchmark for Bringing World Models to Enterprise Systems 

**Authors**: Lakshya Gupta, Litao Li, Yizhe Liu, Sriram Ganapathi Subramanian, Kaheer Suleman, Zichen Zhang, Haoye Lu, Sumit Pasupalak  

**Link**: [PDF](https://arxiv.org/pdf/2601.22130)  

**Abstract**: Frontier large language models (LLMs) excel as autonomous agents in many domains, yet they remain untested in complex enterprise systems where hidden workflows create cascading effects across interconnected databases. Existing enterprise benchmarks evaluate surface-level agentic task completion similar to general consumer benchmarks, ignoring true challenges in enterprises, such as limited observability, large database state, and hidden workflows with cascading side effects. We introduce World of Workflows (WoW), a realistic ServiceNow-based environment incorporating 4,000+ business rules and 55 active workflows embedded in the system, alongside WoW-bench, a benchmark of 234 tasks evaluating constrained agentic task completion and enterprise dynamics modeling capabilities. We reveal two major takeaways: (1) Frontier LLMs suffer from dynamics blindness, consistently failing to predict the invisible, cascading side effects of their actions, which leads to silent constraint violations, and (2) reliability in opaque systems requires grounded world modeling, where agents must mentally simulate hidden state transitions to bridge the observability gap when high-fidelity feedback is unavailable. For reliable and useful enterprise agents, WoW motivates a new paradigm to explicitly learn system dynamics. We release our GitHub for setting up and evaluating WoW. 

---
# The Patient is not a Moving Document: A World Model Training Paradigm for Longitudinal EHR 

**Authors**: Irsyad Adam, Zekai Chen, David Laprade, Shaun Porwal, David Laub, Erik Reinertsen, Arda Pekis, Kevin Brown  

**Link**: [PDF](https://arxiv.org/pdf/2601.22128)  

**Abstract**: Large language models (LLMs) trained with next-word-prediction have achieved success as clinical foundation models. Representations from these language backbones yield strong linear probe performance across biomedical tasks, suggesting that patient semantics emerge from next-token prediction at scale. However, this paradigm treats patients as a document to be summarized rather than a dynamical system to be simulated; a patient's trajectory emerges from their state evolving under interventions and time, requiring models that simulate dynamics rather than predict tokens. To address this, we introduce SMB-Structure, a world model for structured EHR that grounds a joint-embedding prediction architecture (JEPA) with next-token prediction (SFT). SFT grounds our model to reconstruct future patient states in token space, while JEPA predicts those futures in latent space from the initial patient representation alone, forcing trajectory dynamics to be encoded before the next state is observed. We validate across two large-scale cohorts: Memorial Sloan Kettering (23,319 oncology patients; 323,000+ patient-years) and INSPECT (19,402 pulmonary embolism patients). Using a linear probe evaluated at multiple points along the disease trajectory, we demonstrate that our training paradigm learns embeddings that capture disease dynamics not recoverable by autoregressive baselines, enabling SMB-Structure to achieve competitive performance on complex tasks characterized by high patient heterogeneity. Model weights are available at this https URL. 

---
# Defining Operational Conditions for Safety-Critical AI-Based Systems from Data 

**Authors**: Johann Christensen, Elena Hoemann, Frank Köster, Sven Hallerbach  

**Link**: [PDF](https://arxiv.org/pdf/2601.22118)  

**Abstract**: Artificial Intelligence (AI) has been on the rise in many domains, including numerous safety-critical applications. However, for complex systems found in the real world, or when data already exist, defining the underlying environmental conditions is extremely challenging. This often results in an incomplete description of the environment in which the AI-based system must operate. Nevertheless, this description, called the Operational Design Domain (ODD), is required in many domains for the certification of AI-based systems. Traditionally, the ODD is created in the early stages of the development process, drawing on sophisticated expert knowledge and related standards. This paper presents a novel Safety-by-Design method to a posteriori define the ODD from previously collected data using a multi-dimensional kernel-based representation. This approach is validated through both Monte Carlo methods and a real-world aviation use case for a future safety-critical collision-avoidance system. Moreover, by defining under what conditions two ODDs are equal, the paper shows that the data-driven ODD can equal the original, underlying hidden ODD of the data. Utilizing the novel, Safe-by-Design kernel-based ODD enables future certification of data-driven, safety-critical AI-based systems. 

---
# Optimizing Agentic Workflows using Meta-tools 

**Authors**: Sami Abuzakuk, Anne-Marie Kermarrec, Rishi Sharma, Rasmus Moorits Veski, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2601.22037)  

**Abstract**: Agentic AI enables LLM to dynamically reason, plan, and interact with tools to solve complex tasks. However, agentic workflows often require many iterative reasoning steps and tool invocations, leading to significant operational expense, end-to-end latency and failures due to hallucinations. This work introduces Agent Workflow Optimization (AWO), a framework that identifies and optimizes redundant tool execution patterns to improve the efficiency and robustness of agentic workflows. AWO analyzes existing workflow traces to discover recurring sequences of tool calls and transforms them into meta-tools, which are deterministic, composite tools that bundle multiple agent actions into a single invocation. Meta-tools bypass unnecessary intermediate LLM reasoning steps and reduce operational cost while also shortening execution paths, leading to fewer failures. Experiments on two agentic AI benchmarks show that AWO reduces the number of LLM calls up to 11.9% while also increasing the task success rate by up to 4.2 percent points. 

---
# CAR-bench: Evaluating the Consistency and Limit-Awareness of LLM Agents under Real-World Uncertainty 

**Authors**: Johannes Kirmayr, Lukas Stappen, Elisabeth André  

**Link**: [PDF](https://arxiv.org/pdf/2601.22027)  

**Abstract**: Existing benchmarks for Large Language Model (LLM) agents focus on task completion under idealistic settings but overlook reliability in real-world, user-facing applications. In domains, such as in-car voice assistants, users often issue incomplete or ambiguous requests, creating intrinsic uncertainty that agents must manage through dialogue, tool use, and policy adherence. We introduce CAR-bench, a benchmark for evaluating consistency, uncertainty handling, and capability awareness in multi-turn, tool-using LLM agents in an in-car assistant domain. The environment features an LLM-simulated user, domain policies, and 58 interconnected tools spanning navigation, productivity, charging, and vehicle control. Beyond standard task completion, CAR-bench introduces Hallucination tasks that test agents' limit-awareness under missing tools or information, and Disambiguation tasks that require resolving uncertainty through clarification or internal information gathering. Baseline results reveal large gaps between occasional and consistent success on all task types. Even frontier reasoning LLMs achieve less than 50% consistent pass rate on Disambiguation tasks due to premature actions, and frequently violate policies or fabricate information to satisfy user requests in Hallucination tasks, underscoring the need for more reliable and self-aware LLM agents in real-world settings. 

---
# Heterogeneous Computing: The Key to Powering the Future of AI Agent Inference 

**Authors**: Yiren Zhao, Junyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22001)  

**Abstract**: AI agent inference is driving an inference heavy datacenter future and exposes bottlenecks beyond compute - especially memory capacity, memory bandwidth and high-speed interconnect. We introduce two metrics - Operational Intensity (OI) and Capacity Footprint (CF) - that jointly explain regimes the classic roofline analysis misses, including the memory capacity wall. Across agentic workflows (chat, coding, web use, computer use) and base model choices (GQA/MLA, MoE, quantization), OI/CF can shift dramatically, with long context KV cache making decode highly memory bound. These observations motivate disaggregated serving and system level heterogeneity: specialized prefill and decode accelerators, broader scale up networking, and decoupled compute-memory enabled by optical I/O. We further hypothesize agent-hardware co design, multiple inference accelerators within one system, and high bandwidth, large capacity memory disaggregation as foundations for adaptation to evolving OI/CF. Together, these directions chart a path to sustain efficiency and capability for large scale agentic AI inference. 

---
# Liquid Interfaces: A Dynamic Ontology for the Interoperability of Autonomous Systems 

**Authors**: Dhiogo de Sá, Carlos Schmiedel, Carlos Pereira Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2601.21993)  

**Abstract**: Contemporary software architectures struggle to support autonomous agents whose reasoning is adaptive, probabilistic, and context-dependent, while system integration remains dominated by static interfaces and deterministic contracts. This paper introduces Liquid Interfaces, a coordination paradigm in which interfaces are not persistent technical artifacts, but ephemeral relational events that emerge through intention articulation and semantic negotiation at this http URL formalize this model and present the Liquid Interface Protocol (LIP),which governs intention-driven interaction, negotiated execution, and enforce ephemerality under semantic uncertainty. We further discuss the governance implications of this approach and describe a reference architecture that demonstrates practical feasibility. Liquid Interfaces provide a principled foundation for adaptive coordination in agent-based systems 

---
# VERSA: Verified Event Data Format for Reliable Soccer Analytics 

**Authors**: Geonhee Jo, Mingu Kang, Kangmin Lee, Minho Lee, Pascal Bauer, Sang-Ki Ko  

**Link**: [PDF](https://arxiv.org/pdf/2601.21981)  

**Abstract**: Event stream data is a critical resource for fine-grained analysis across various domains, including financial transactions, system operations, and sports. In sports, it is actively used for fine-grained analyses such as quantifying player contributions and identifying tactical patterns. However, the reliability of these models is fundamentally limited by inherent data quality issues that cause logical inconsistencies (e.g., incorrect event ordering or missing events). To this end, this study proposes VERSA (Verified Event Data Format for Reliable Soccer Analytics), a systematic verification framework that ensures the integrity of event stream data within the soccer domain. VERSA is based on a state-transition model that defines valid event sequences, thereby enabling the automatic detection and correction of anomalous patterns within the event stream data. Notably, our examination of event data from the K League 1 (2024 season), provided by Bepro, detected that 18.81% of all recorded events exhibited logical inconsistencies. Addressing such integrity issues, our experiments demonstrate that VERSA significantly enhances cross-provider consistency, ensuring stable and unified data representation across heterogeneous sources. Furthermore, we demonstrate that data refined by VERSA significantly improves the robustness and performance of a downstream task called VAEP, which evaluates player contributions. These results highlight that the verification process is highly effective in increasing the reliability of data-driven analysis. 

---
# Mind the Gap: How Elicitation Protocols Shape the Stated-Revealed Preference Gap in Language Models 

**Authors**: Pranav Mahajan, Ihor Kendiukhov, Syed Hussain, Lydia Nottingham  

**Link**: [PDF](https://arxiv.org/pdf/2601.21975)  

**Abstract**: Recent work identifies a stated-revealed (SvR) preference gap in language models (LMs): a mismatch between the values models endorse and the choices they make in context. Existing evaluations rely heavily on binary forced-choice prompting, which entangles genuine preferences with artifacts of the elicitation protocol. We systematically study how elicitation protocols affect SvR correlation across 24 LMs. Allowing neutrality and abstention during stated preference elicitation allows us to exclude weak signals, substantially improving Spearman's rank correlation ($\rho$) between volunteered stated preferences and forced-choice revealed preferences. However, further allowing abstention in revealed preferences drives $\rho$ to near-zero or negative values due to high neutrality rates. Finally, we find that system prompt steering using stated preferences during revealed preference elicitation does not reliably improve SvR correlation on AIRiskDilemmas. Together, our results show that SvR correlation is highly protocol-dependent and that preference elicitation requires methods that account for indeterminate preferences. 

---
# Learning Decentralized LLM Collaboration with Multi-Agent Actor Critic 

**Authors**: Shuo Liu, Tianle Chen, Ryan Amiri, Christopher Amato  

**Link**: [PDF](https://arxiv.org/pdf/2601.21972)  

**Abstract**: Recent work has explored optimizing LLM collaboration through Multi-Agent Reinforcement Learning (MARL). However, most MARL fine-tuning approaches rely on predefined execution protocols, which often require centralized execution. Decentralized LLM collaboration is more appealing in practice, as agents can run inference in parallel with flexible deployments. Also, current approaches use Monte Carlo methods for fine-tuning, which suffer from high variance and thus require more samples to train effectively. Actor-critic methods are prevalent in MARL for dealing with these issues, so we developed Multi-Agent Actor-Critic (MAAC) methods to optimize decentralized LLM collaboration. In this paper, we analyze when and why these MAAC methods are beneficial. We propose 2 MAAC approaches, \textbf{CoLLM-CC} with a \textbf{C}entralized \textbf{C}ritic and \textbf{CoLLM-DC} with \textbf{D}ecentralized \textbf{C}ritics. Our experiments across writing, coding, and game-playing domains show that Monte Carlo methods and CoLLM-DC can achieve performance comparable to CoLLM-CC in short-horizon and dense-reward settings. However, they both underperform CoLLM-CC on long-horizon or sparse-reward tasks, where Monte Carlo methods require substantially more samples and CoLLM-DC struggles to converge. Our code is available at this https URL. 

---
# The Energy Impact of Domain Model Design in Classical Planning 

**Authors**: Ilche Georgievski, Serhat Tekin, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2601.21967)  

**Abstract**: AI research has traditionally prioritised algorithmic performance, such as optimising accuracy in machine learning or runtime in automated planning. The emerging paradigm of Green AI challenges this by recognising energy consumption as a critical performance dimension. Despite the high computational demands of automated planning, its energy efficiency has received little attention. This gap is particularly salient given the modular planning structure, in which domain models are specified independently of algorithms. On the other hand, this separation also enables systematic analysis of energy usage through domain model design. We empirically investigate how domain model characteristics affect the energy consumption of classical planners. We introduce a domain model configuration framework that enables controlled variation of features, such as element ordering, action arity, and dead-end states. Using five benchmark domains and five state-of-the-art planners, we analyse energy and runtime impacts across 32 domain variants per benchmark. Results demonstrate that domain-level modifications produce measurable energy differences across planners, with energy consumption not always correlating with runtime. 

---
# How do Visual Attributes Influence Web Agents? A Comprehensive Evaluation of User Interface Design Factors 

**Authors**: Kuai Yu, Naicheng Yu, Han Wang, Rui Yang, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21961)  

**Abstract**: Web agents have demonstrated strong performance on a wide range of web-based tasks. However, existing research on the effect of environmental variation has mostly focused on robustness to adversarial attacks, with less attention to agents' preferences in benign scenarios. Although early studies have examined how textual attributes influence agent behavior, a systematic understanding of how visual attributes shape agent decision-making remains limited. To address this, we introduce VAF, a controlled evaluation pipeline for quantifying how webpage Visual Attribute Factors influence web-agent decision-making. Specifically, VAF consists of three stages: (i) variant generation, which ensures the variants share identical semantics as the original item while only differ in visual attributes; (ii) browsing interaction, where agents navigate the page via scrolling and clicking the interested item, mirroring how human users browse online; (iii) validating through both click action and reasoning from agents, which we use the Target Click Rate and Target Mention Rate to jointly evaluate the effect of visual attributes. By quantitatively measuring the decision-making difference between the original and variant, we identify which visual attributes influence agents' behavior most. Extensive experiments, across 8 variant families (48 variants total), 5 real-world websites (including shopping, travel, and news browsing), and 4 representative web agents, show that background color contrast, item size, position, and card clarity have a strong influence on agents' actions, whereas font styling, text color, and item image clarity exhibit minor effects. 

---
# ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models 

**Authors**: Bowen Fang, Wen Ye, Yunyue Su, Jinghao Zhang, Qiang Liu, Yesheng Liu, Xin Sun, Shu Wu, Jiabing Yang, Baole Wei, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21947)  

**Abstract**: Prevalent retrieval-based tool-use pipelines struggle with a dual semantic challenge: their retrievers often employ encoders that fail to capture complex semantics, while the Large Language Model (LLM) itself lacks intrinsic tool knowledge from its natural language pretraining. Generative methods offer a powerful alternative by unifying selection and execution, tasking the LLM to directly learn and generate tool identifiers. However, the common practice of mapping each tool to a unique new token introduces substantial limitations: it creates a scalability and generalization crisis, as the vocabulary size explodes and each tool is assigned a semantically isolated token. This approach also creates a semantic bottleneck that hinders the learning of collaborative tool relationships, as the model must infer them from sparse co-occurrences of monolithic tool IDs within a vast library. To address these limitations, we propose ToolWeaver, a novel generative tool learning framework that encodes tools into hierarchical sequences. This approach makes vocabulary expansion logarithmic to the number of tools. Crucially, it enables the model to learn collaborative patterns from the dense co-occurrence of shared codes, rather than the sparse co-occurrence of monolithic tool IDs. We generate these structured codes through a novel tokenization process designed to weave together a tool's intrinsic semantics with its extrinsic co-usage patterns. These structured codes are then integrated into the LLM through a generative alignment stage, where the model is fine-tuned to produce the hierarchical code sequences. Evaluation results with nearly 47,000 tools show that ToolWeaver significantly outperforms state-of-the-art methods, establishing a more scalable, generalizable, and semantically-aware foundation for advanced tool-augmented agents. 

---
# Retrieval-Infused Reasoning Sandbox: A Benchmark for Decoupling Retrieval and Reasoning Capabilities 

**Authors**: Shuangshuang Ying, Zheyu Wang, Yunjian Peng, Jin Chen, Yuhao Wu, Hongbin Lin, Dingyu He, Siyi Liu, Gengchen Yu, YinZhu Piao, Yuchen Wu, Xin Gui, Zhongyuan Peng, Xin Li, Xeron Du, Libo Qin, YiXin Cao, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21937)  

**Abstract**: Despite strong performance on existing benchmarks, it remains unclear whether large language models can reason over genuinely novel scientific information. Most evaluations score end-to-end RAG pipelines, where reasoning is confounded with retrieval and toolchain choices, and the signal is further contaminated by parametric memorization and open-web volatility. We introduce DeR2, a controlled deep-research sandbox that isolates document-grounded reasoning while preserving core difficulties of deep search: multi-step synthesis, denoising, and evidence-based conclusion making. DeR2 decouples evidence access from reasoning via four regimes--Instruction-only, Concepts (gold concepts without documents), Related-only (only relevant documents), and Full-set (relevant documents plus topically related distractors)--yielding interpretable regime gaps that operationalize retrieval loss vs. reasoning loss and enable fine-grained error attribution. To prevent parametric leakage, we apply a two-phase validation that requires parametric failure without evidence while ensuring oracle-concept solvability. To ensure reproducibility, each instance provides a frozen document library (drawn from 2023-2025 theoretical papers) with expert-annotated concepts and validated rationales. Experiments across a diverse set of state-of-the-art foundation models reveal substantial variation and significant headroom: some models exhibit mode-switch fragility, performing worse with the Full-set than with Instruction-only, while others show structural concept misuse, correctly naming concepts but failing to execute them as procedures. 

---
# AgenticSimLaw: A Juvenile Courtroom Multi-Agent Debate Simulation for Explainable High-Stakes Tabular Decision Making 

**Authors**: Jon Chun, Kathrine Elkins, Yong Suk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.21936)  

**Abstract**: We introduce AgenticSimLaw, a role-structured, multi-agent debate framework that provides transparent and controllable test-time reasoning for high-stakes tabular decision-making tasks. Unlike black-box approaches, our courtroom-style orchestration explicitly defines agent roles (prosecutor, defense, judge), interaction protocols (7-turn structured debate), and private reasoning strategies, creating a fully auditable decision-making process. We benchmark this framework on young adult recidivism prediction using the NLSY97 dataset, comparing it against traditional chain-of-thought (CoT) prompting across almost 90 unique combinations of models and strategies. Our results demonstrate that structured multi-agent debate provides more stable and generalizable performance compared to single-agent reasoning, with stronger correlation between accuracy and F1-score metrics. Beyond performance improvements, AgenticSimLaw offers fine-grained control over reasoning steps, generates complete interaction transcripts for explainability, and enables systematic profiling of agent behaviors. While we instantiate this framework in the criminal justice domain to stress-test reasoning under ethical complexity, the approach generalizes to any deliberative, high-stakes decision task requiring transparency and human oversight. This work addresses key LLM-based multi-agent system challenges: organization through structured roles, observability through logged interactions, and responsibility through explicit non-deployment constraints for sensitive domains. Data, results, and code will be available on this http URL under the MIT license. 

---
# Self-Compression of Chain-of-Thought via Multi-Agent Reinforcement Learning 

**Authors**: Yiqun Chen, Jinyuan Feng, Wei Yang, Meizhi Zhong, Zhengliang Shi, Rui Li, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Zhiqiang Pu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21919)  

**Abstract**: The inference overhead induced by redundant reasoning undermines the interactive experience and severely bottlenecks the deployment of Large Reasoning Models. Existing reinforcement learning (RL)-based solutions tackle this problem by coupling a length penalty with outcome-based rewards. This simplistic reward weighting struggles to reconcile brevity with accuracy, as enforcing brevity may compromise critical reasoning logic. In this work, we address this limitation by proposing a multi-agent RL framework that selectively penalizes redundant chunks, while preserving essential reasoning logic. Our framework, Self-Compression via MARL (SCMA), instantiates redundancy detection and evaluation through two specialized agents: \textbf{a Segmentation Agent} for decomposing the reasoning process into logical chunks, and \textbf{a Scoring Agent} for quantifying the significance of each chunk. The Segmentation and Scoring agents collaboratively define an importance-weighted length penalty during training, incentivizing \textbf{a Reasoning Agent} to prioritize essential logic without introducing inference overhead during deployment. Empirical evaluations across model scales demonstrate that SCMA reduces response length by 11.1\% to 39.0\% while boosting accuracy by 4.33\% to 10.02\%. Furthermore, ablation studies and qualitative analysis validate that the synergistic optimization within the MARL framework fosters emergent behaviors, yielding more powerful LRMs compared to vanilla RL paradigms. 

---
# JADE: Bridging the Strategic-Operational Gap in Dynamic Agentic RAG 

**Authors**: Yiqun Chen, Erhan Zhang, Tianyi Hu, Shijie Wang, Zixuan Yang, Meizhi Zhong, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21916)  

**Abstract**: The evolution of Retrieval-Augmented Generation (RAG) has shifted from static retrieval pipelines to dynamic, agentic workflows where a central planner orchestrates multi-turn reasoning. However, existing paradigms face a critical dichotomy: they either optimize modules jointly within rigid, fixed-graph architectures, or empower dynamic planning while treating executors as frozen, black-box tools. We identify that this \textit{decoupled optimization} creates a ``strategic-operational mismatch,'' where sophisticated planning strategies fail to materialize due to unadapted local executors, often leading to negative performance gains despite increased system complexity. In this paper, we propose \textbf{JADE} (\textbf{J}oint \textbf{A}gentic \textbf{D}ynamic \textbf{E}xecution), a unified framework for the joint optimization of planning and execution within dynamic, multi-turn workflows. By modeling the system as a cooperative multi-agent team unified under a single shared backbone, JADE enables end-to-end learning driven by outcome-based rewards. This approach facilitates \textit{co-adaptation}: the planner learns to operate within the capability boundaries of the executors, while the executors evolve to align with high-level strategic intent. Empirical results demonstrate that JADE transforms disjoint modules into a synergistic system, yielding remarkable performance improvements via joint optimization and enabling a flexible balance between efficiency and effectiveness through dynamic workflow orchestration. 

---
# ProRAG: Process-Supervised Reinforcement Learning for Retrieval-Augmented Generation 

**Authors**: Zhao Wang, Ziliang Zhao, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21912)  

**Abstract**: Reinforcement learning (RL) has become a promising paradigm for optimizing Retrieval-Augmented Generation (RAG) in complex reasoning tasks. However, traditional outcome-based RL approaches often suffer from reward sparsity and inefficient credit assignment, as coarse-grained scalar rewards fail to identify specific erroneous steps within long-horizon trajectories. This ambiguity frequently leads to "process hallucinations", where models reach correct answers through flawed logic or redundant retrieval steps. Although recent process-aware approaches attempt to mitigate this via static preference learning or heuristic reward shaping, they often lack the on-policy exploration capabilities required to decouple step-level credit from global outcomes. To address these challenges, we propose ProRAG, a process-supervised reinforcement learning framework designed to integrate learned step-level supervision into the online optimization loop. Our framework consists of four stages: (1) Supervised Policy Warmup to initialize the model with a structured reasoning format; (2) construction of an MCTS-based Process Reward Model (PRM) to quantify intermediate reasoning quality; (3) PRM-Guided Reasoning Refinement to align the policy with fine-grained process preferences; and (4) Process-Supervised Reinforcement Learning with a dual-granularity advantage mechanism. By aggregating step-level process rewards with global outcome signals, ProRAG provides precise feedback for every action. Extensive experiments on five multi-hop reasoning benchmarks demonstrate that ProRAG achieves superior overall performance compared to strong outcome-based and process-aware RL baselines, particularly on complex long-horizon tasks, validating the effectiveness of fine-grained process supervision. The code and model are available at this https URL. 

---
# From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning 

**Authors**: Shaojie Wang, Liang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21909)  

**Abstract**: Current LLM post-training methods optimize complete reasoning trajectories through Supervised Fine-Tuning (SFT) followed by outcome-based Reinforcement Learning (RL). While effective, a closer examination reveals a fundamental gap: this approach does not align with how humans actually solve problems. Human cognition naturally decomposes problem-solving into two distinct stages: first acquiring abstract strategies (i.e., meta-knowledge) that generalize across problems, then adapting them to specific instances. In contrast, by treating complete trajectories as basic units, current methods are inherently problem-centric, entangling abstract strategies with problem-specific execution. To address this misalignment, we propose a cognitively-inspired framework that explicitly mirrors the two-stage human cognitive process. Specifically, Chain-of-Meta-Thought (CoMT) focuses supervised learning on abstract reasoning patterns without specific executions, enabling acquisition of generalizable strategies. Confidence-Calibrated Reinforcement Learning (CCRL) then optimizes task adaptation via confidence-aware rewards on intermediate steps, preventing overconfident errors from cascading and improving execution reliability. Experiments across four models and eight benchmarks show 2.19\% and 4.63\% improvements in-distribution and out-of-distribution respectively over standard methods, while reducing training time by 65-70% and token consumption by 50%, demonstrating that aligning post-training with human cognitive principles yields not only superior generalization but also enhanced training efficiency. 

---
# Making Models Unmergeable via Scaling-Sensitive Loss Landscape 

**Authors**: Minwoo Jang, Hoyoung Kim, Jabin Koo, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2601.21898)  

**Abstract**: The rise of model hubs has made it easier to access reusable model components, making model merging a practical tool for combining capabilities. Yet, this modularity also creates a \emph{governance gap}: downstream users can recompose released weights into unauthorized mixtures that bypass safety alignment or licensing terms. Because existing defenses are largely post-hoc and architecture-specific, they provide inconsistent protection across diverse architectures and release formats in practice. To close this gap, we propose \textsc{Trap}$^{2}$, an architecture-agnostic protection framework that encodes protection into the update during fine-tuning, regardless of whether they are released as adapters or full models. Instead of relying on architecture-dependent approaches, \textsc{Trap}$^{2}$ uses weight re-scaling as a simple proxy for the merging process. It keeps released weights effective in standalone use, but degrades them under re-scaling that often arises in merging, undermining unauthorized merging. 

---
# astra-langchain4j: Experiences Combining LLMs and Agent Programming 

**Authors**: Rem Collier, Katharine Beaumont, Andrei Ciortea  

**Link**: [PDF](https://arxiv.org/pdf/2601.21879)  

**Abstract**: Given the emergence of Generative AI over the last two years and the increasing focus on Agentic AI as a form of Multi-Agent System it is important to explore both how such technologies can impact the use of traditional Agent Toolkits and how the wealth of experience encapsulated in those toolkits can influence the design of the new agentic platforms. This paper presents an overview of our experience developing a prototype large language model (LLM) integration for the ASTRA programming language. It presents a brief overview of the toolkit, followed by three example implementations, concluding with a discussion of the experiences garnered through the examples. 

---
# WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents 

**Authors**: Yao Zhang, Shijie Tang, Zeyu Li, Zhen Han, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2601.21872)  

**Abstract**: Web agents hold great potential for automating complex computer tasks, yet their interactions involve long-horizon, sequential decision-making with irreversible actions. In such settings, outcome-based supervision is sparse and delayed, often rewarding incorrect trajectories and failing to support inference-time scaling. This motivates the use of Process Reward Models (WebPRMs) for web navigation, but existing approaches remain limited: scalar WebPRMs collapse progress into coarse, weakly grounded signals, while checklist-based WebPRMs rely on brittle template matching that fails under layout or semantic changes and often mislabels superficially correct actions as successful, providing little insight or interpretability. To address these challenges, we introduce WebArbiter, a reasoning-first, principle-inducing WebPRM that formulates reward modeling as text generation, producing structured justifications that conclude with a preference verdict and identify the action most conducive to task completion under the current context. Training follows a two-stage pipeline: reasoning distillation equips the model with coherent principle-guided reasoning, and reinforcement learning corrects teacher biases by directly aligning verdicts with correctness, enabling stronger generalization. To support systematic evaluation, we release WebPRMBench, a comprehensive benchmark spanning four diverse web environments with rich tasks and high-quality preference annotations. On WebPRMBench, WebArbiter-7B outperforms the strongest baseline, GPT-5, by 9.1 points. In reward-guided trajectory search on WebArena-Lite, it surpasses the best prior WebPRM by up to 7.2 points, underscoring its robustness and practical value in real-world complex web tasks. 

---
# KnowBias: Mitigating Social Bias in LLMs via Know-Bias Neuron Enhancement 

**Authors**: Jinhao Pan, Chahat Raj, Anjishnu Mukherjee, Sina Mansouri, Bowen Wei, Shloka Yada, Ziwei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21864)  

**Abstract**: Large language models (LLMs) exhibit social biases that reinforce harmful stereotypes, limiting their safe deployment. Most existing debiasing methods adopt a suppressive paradigm by modifying parameters, prompts, or neurons associated with biased behavior; however, such approaches are often brittle, weakly generalizable, data-inefficient, and prone to degrading general capability. We propose \textbf{KnowBias}, a lightweight and conceptually distinct framework that mitigates bias by strengthening, rather than suppressing, neurons encoding bias-knowledge. KnowBias identifies neurons encoding bias knowledge using a small set of bias-knowledge questions via attribution-based analysis, and selectively enhances them at inference time. This design enables strong debiasing while preserving general capabilities, generalizes across bias types and demographics, and is highly data efficient, requiring only a handful of simple yes/no questions and no retraining. Experiments across multiple benchmarks and LLMs demonstrate consistent state-of-the-art debiasing performance with minimal utility degradation. Data and code are available at this https URL. 

---
# Bridging Forecast Accuracy and Inventory KPIs: A Simulation-Based Software Framework 

**Authors**: So Fukuhara, Abdallah Alabdallah, Nuwan Gunasekara, Slawomir Nowaczyk  

**Link**: [PDF](https://arxiv.org/pdf/2601.21844)  

**Abstract**: Efficient management of spare parts inventory is crucial in the automotive aftermarket, where demand is highly intermittent and uncertainty drives substantial cost and service risks. Forecasting is therefore central, but the quality of a forecasting model should be judged not by statistical accuracy (e.g., MAE, RMSE, IAE) but rather by its impact on key operational performance indicators (KPIs), such as total cost and service level. Yet most existing work evaluates models exclusively using accuracy metrics, and the relationship between these metrics and operational KPIs remains poorly understood. To address this gap, we propose a decision-centric simulation software framework that enables systematic evaluation of forecasting model in realistic inventory management setting. The framework comprises: (i) a synthetic demand generator tailored to spare-parts demand characteristics, (ii) a flexible forecasting module that can host arbitrary predictive models, and (iii) an inventory control simulator that consumes the forecasts and computes operational KPIs. This closed-loop setup enables researchers to evaluate models not only in terms of statistical error but also in terms of their downstream implications for inventory decisions. Using a wide range of simulation scenarios, we show that improvements in conventional accuracy metrics do not necessarily translate into better operational performance, and that models with similar statistical error profiles can induce markedly different cost-service trade-offs. We analyze these discrepancies to characterize how specific aspects of forecast performance affect inventory outcomes and derive guidance for model selection. Overall, the framework operationalizes the link between demand forecasting and inventory management, shifting evaluation from purely predictive accuracy toward operational relevance in the automotive aftermarket and related domains. 

---
# Looking Beyond Accuracy: A Holistic Benchmark of ECG Foundation Models 

**Authors**: Francesca Filice, Edoardo De Rose, Simone Bartucci, Francesco Calimeri, Simona Perri  

**Link**: [PDF](https://arxiv.org/pdf/2601.21830)  

**Abstract**: The electrocardiogram (ECG) is a cost-effective, highly accessible and widely employed diagnostic tool. With the advent of Foundation Models (FMs), the field of AI-assisted ECG interpretation has begun to evolve, as they enable model reuse across different tasks by relying on embeddings. However, to responsibly employ FMs, it is crucial to rigorously assess to which extent the embeddings they produce are generalizable, particularly in error-sensitive domains such as healthcare. Although prior works have already addressed the problem of benchmarking ECG-expert FMs, they focus predominantly on the evaluation of downstream performance. To fill this gap, this study aims to find an in-depth, comprehensive benchmarking framework for FMs, with a specific focus on ECG-expert ones. To this aim, we introduce a benchmark methodology that complements performance-based evaluation with representation-level analysis, leveraging SHAP and UMAP techniques. Furthermore, we rely on the methodology for carrying out an extensive evaluation of several ECG-expert FMs pretrained via state-of-the-art techniques over different cross-continental datasets and data availability settings; this includes ones featuring data scarcity, a fairly common situation in real-world medical scenarios. Experimental results show that our benchmarking protocol provides a rich insight of ECG-expert FMs' embedded patterns, enabling a deeper understanding of their representational structure and generalizability. 

---
# CORE:Toward Ubiquitous 6G Intelligence Through Collaborative Orchestration of Large Language Model Agents Over Hierarchical Edge 

**Authors**: Zitong Yu, Boquan Sun, Yang Li, Zheyan Qu, Xing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21822)  

**Abstract**: Rapid advancements in sixth-generation (6G) networks and large language models (LLMs) have paved the way for ubiquitous intelligence, wherein seamless connectivity and distributed artificial intelligence (AI) have revolutionized various aspects of our this http URL, realizing this vision faces significant challenges owing to the fragmented and heterogeneous computing resources across hierarchical networks, which are insufficient for individual LLM agents to perform complex reasoning this http URL address this issue, we propose Collaborative Orchestration Role at Edge (CORE), an innovative framework that employs a collaborative learning system in which multiple LLMs, each assigned a distinct functional role, are distributed across mobile devices and tiered edge servers. The system integrates three optimization modules, encompassing real-time perception,dynamic role orchestration, and pipeline-parallel execution, to facilitate efficient and rapid collaboration among distributed agents. Furthermore, we introduce a novel role affinity scheduling algorithm for dynamically orchestrating LLM role assignments across the hierarchical edge infrastructure, intelligently matching computational demands with available dispersed this http URL, comprehensive case studies and performance evaluations across various 6G application scenarios demonstrated the efficacy of CORE, revealing significant enhancements in the system efficiency and task completion rates. Building on these promising outcomes, we further validated the practical applicability of CORE by deploying it on a real-world edge-computing platform,that exhibits robust performance in operational environments. 

---
# A Unified XAI-LLM Approach for EndotrachealSuctioning Activity Recognition 

**Authors**: Hoang Khang Phan, Quang Vinh Dang, Noriyo Colley, Christina Garcia, Nhat Tan Le  

**Link**: [PDF](https://arxiv.org/pdf/2601.21802)  

**Abstract**: Endotracheal suctioning (ES) is an invasive yet essential clinical procedure that requires a high degree of skill to minimize patient risk - particularly in home care and educational settings, where consistent supervision may be limited. Despite its critical importance, automated recognition and feedback systems for ES training remain underexplored. To address this gap, this study proposes a unified, LLM-centered framework for video-based activity recognition benchmarked against conventional machine learning and deep learning approaches, and a pilot study on feedback generation. Within this framework, the Large Language Model (LLM) serves as the central reasoning module, performing both spatiotemporal activity recognition and explainable decision analysis from video data. Furthermore, the LLM is capable of verbalizing feedback in natural language, thereby translating complex technical insights into accessible, human-understandable guidance for trainees. Experimental results demonstrate that the proposed LLM-based approach outperforms baseline models, achieving an improvement of approximately 15-20\% in both accuracy and F1 score. Beyond recognition, the framework incorporates a pilot student-support module built upon anomaly detection and explainable AI (XAI) principles, which provides automated, interpretable feedback highlighting correct actions and suggesting targeted improvements. Collectively, these contributions establish a scalable, interpretable, and data-driven foundation for advancing nursing education, enhancing training efficiency, and ultimately improving patient safety. 

---
# BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics 

**Authors**: Dionizije Fa, Marko Čuljak, Bruno Pandža, Mateo Čupić  

**Link**: [PDF](https://arxiv.org/pdf/2601.21800)  

**Abstract**: This paper introduces BioAgent Bench, a benchmark dataset and an evaluation suite designed for measuring the performance and robustness of AI agents in common bioinformatics tasks. The benchmark contains curated end-to-end tasks (e.g., RNA-seq, variant calling, metagenomics) with prompts that specify concrete output artifacts to support automated assessment, including stress testing under controlled perturbations. We evaluate frontier closed-source and open-weight models across multiple agent harnesses, and use an LLM-based grader to score pipeline progress and outcome validity. We find that frontier agents can complete multi-step bioinformatics pipelines without elaborate custom scaffolding, often producing the requested final artifacts reliably. However, robustness tests reveal failure modes under controlled perturbations (corrupted inputs, decoy files, and prompt bloat), indicating that correct high-level pipeline construction does not guarantee reliable step-level reasoning. Finally, because bioinformatics workflows may involve sensitive patient data, proprietary references, or unpublished IP, closed-source models can be unsuitable under strict privacy constraints; in such settings, open-weight models may be preferable despite lower completion rates. We release the dataset and evaluation suite publicly. 

---
# Abstract Concept Modelling in Conceptual Spaces: A Study on Chess Strategies 

**Authors**: Hadi Banaee, Stephanie Lowry  

**Link**: [PDF](https://arxiv.org/pdf/2601.21771)  

**Abstract**: We present a conceptual space framework for modelling abstract concepts that unfold over time, demonstrated through a chess-based proof-of-concept. Strategy concepts, such as attack or sacrifice, are represented as geometric regions across interpretable quality dimensions, with chess games instantiated and analysed as trajectories whose directional movement toward regions enables recognition of intended strategies. This approach also supports dual-perspective modelling, capturing how players interpret identical situations differently. Our implementation demonstrates the feasibility of trajectory-based concept recognition, with movement patterns aligning with expert commentary. This work explores extending the conceptual spaces theory to temporally realised, goal-directed concepts. The approach establishes a foundation for broader applications involving sequential decision-making and supports integration with knowledge evolution mechanisms for learning and refining abstract concepts over time. 

---
# Zero-Shot Statistical Downscaling via Diffusion Posterior Sampling 

**Authors**: Ruian Tie, Wenbo Xiong, Zhengyu Shi, Xinyu Su, Chenyu jiang, Libo Wu, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.21760)  

**Abstract**: Conventional supervised climate downscaling struggles to generalize to Global Climate Models (GCMs) due to the lack of paired training data and inherent domain gaps relative to reanalysis. Meanwhile, current zero-shot methods suffer from physical inconsistencies and vanishing gradient issues under large scaling factors. We propose Zero-Shot Statistical Downscaling (ZSSD), a zero-shot framework that performs statistical downscaling without paired data during training. ZSSD leverages a Physics-Consistent Climate Prior learned from reanalysis data, conditioned on geophysical boundaries and temporal information to enforce physical validity. Furthermore, to enable robust inference across varying GCMs, we introduce Unified Coordinate Guidance. This strategy addresses the vanishing gradient problem in vanilla DPS and ensures consistency with large-scale fields. Results show that ZSSD significantly outperforms existing zero-shot baselines in 99th percentile errors and successfully reconstructs complex weather events, such as tropical cyclones, across heterogeneous GCMs. 

---
# Language-based Trial and Error Falls Behind in the Era of Experience 

**Authors**: Haoyu Wang, Guozheng Ma, Shugang Cui, Yilun Kong, Haotian Luo, Li Shen, Mengya Gao, Yichao Wu, Xiaogang Wang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21754)  

**Abstract**: While Large Language Models (LLMs) excel in language-based agentic tasks, their applicability to unseen, nonlinguistic environments (e.g., symbolic or spatial tasks) remains limited. Previous work attributes this performance gap to the mismatch between the pretraining distribution and the testing distribution. In this work, we demonstrate the primary bottleneck is the prohibitive cost of exploration: mastering these tasks requires extensive trial-and-error, which is computationally unsustainable for parameter-heavy LLMs operating in a high dimensional semantic space. To address this, we propose SCOUT (Sub-Scale Collaboration On Unseen Tasks), a novel framework that decouples exploration from exploitation. We employ lightweight "scouts" (e.g., small MLPs) to probe environmental dynamics at a speed and scale far exceeding LLMs. The collected trajectories are utilized to bootstrap the LLM via Supervised Fine-Tuning (SFT), followed by multi-turn Reinforcement Learning (RL) to activate its latent world knowledge. Empirically, SCOUT enables a Qwen2.5-3B-Instruct model to achieve an average score of 0.86, significantly outperforming proprietary models, including Gemini-2.5-Pro (0.60), while saving about 60% GPU hours consumption. 

---
# Epistemic Context Learning: Building Trust the Right Way in LLM-Based Multi-Agent Systems 

**Authors**: Ruiwen Zhou, Maojia Song, Xiaobao Wu, Sitao Cheng, Xunjian Yin, Yuxi Xie, Zhuoqun Hao, Wenyue Hua, Liangming Pan, Soujanya Poria, Min-Yen Kan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21742)  

**Abstract**: Individual agents in multi-agent (MA) systems often lack robustness, tending to blindly conform to misleading peers. We show this weakness stems from both sycophancy and inadequate ability to evaluate peer reliability. To address this, we first formalize the learning problem of history-aware reference, introducing the historical interactions of peers as additional input, so that agents can estimate peer reliability and learn from trustworthy peers when uncertain. This shifts the task from evaluating peer reasoning quality to estimating peer reliability based on interaction history. We then develop Epistemic Context Learning (ECL): a reasoning framework that conditions predictions on explicitly-built peer profiles from history. We further optimize ECL by reinforcement learning using auxiliary rewards. Our experiments reveal that our ECL enables small models like Qwen 3-4B to outperform a history-agnostic baseline 8x its size (Qwen 3-30B) by accurately identifying reliable peers. ECL also boosts frontier models to near-perfect (100%) performance. We show that ECL generalizes well to various MA configurations and we find that trust is modeled well by LLMs, revealing a strong correlation in trust modeling accuracy and final answer quality. 

---
# DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting 

**Authors**: Siru Zhong, Yiqiu Liu, Zhiqing Cui, Zezhi Shao, Fei Wang, Qingsong Wen, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21726)  

**Abstract**: Deep time series models are vulnerable to noisy data ubiquitous in real-world applications. Existing robustness strategies either prune data or rely on costly prior quantification, failing to balance effectiveness and efficiency. In this paper, we introduce DropoutTS, a model-agnostic plugin that shifts the paradigm from "what" to learn to "how much" to learn. DropoutTS employs a Sample-Adaptive Dropout mechanism: leveraging spectral sparsity to efficiently quantify instance-level noise via reconstruction residuals, it dynamically calibrates model learning capacity by mapping noise to adaptive dropout rates - selectively suppressing spurious fluctuations while preserving fine-grained fidelity. Extensive experiments across diverse noise regimes and open benchmarks show DropoutTS consistently boosts superior backbones' performance, delivering advanced robustness with negligible parameter overhead and no architectural modifications. Our code is available at this https URL. 

---
# E-mem: Multi-agent based Episodic Context Reconstruction for LLM Agent Memory 

**Authors**: Kaixiang Wang, Yidan Lin, Jiong Lou, Zhaojiacheng Zhou, Bunyod Suvonov, Jie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.21714)  

**Abstract**: The evolution of Large Language Model (LLM) agents towards System~2 reasoning, characterized by deliberative, high-precision problem-solving, requires maintaining rigorous logical integrity over extended horizons. However, prevalent memory preprocessing paradigms suffer from destructive de-contextualization. By compressing complex sequential dependencies into pre-defined structures (e.g., embeddings or graphs), these methods sever the contextual integrity essential for deep reasoning. To address this, we propose E-mem, a framework shifting from Memory Preprocessing to Episodic Context Reconstruction. Inspired by biological engrams, E-mem employs a heterogeneous hierarchical architecture where multiple assistant agents maintain uncompressed memory contexts, while a central master agent orchestrates global planning. Unlike passive retrieval, our mechanism empowers assistants to locally reason within activated segments, extracting context-aware evidence before aggregation. Evaluations on the LoCoMo benchmark demonstrate that E-mem achieves over 54\% F1, surpassing the state-of-the-art GAM by 7.75\%, while reducing token cost by over 70\%. 

---
# FBS: Modeling Native Parallel Reading inside a Transformer 

**Authors**: Tongxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21708)  

**Abstract**: Large language models (LLMs) excel across many tasks, yet inference is still dominated by strictly token-by-token autoregression. Existing acceleration methods largely patch this pipeline and miss core human-reading ingredients: content-adaptive foresight, chunk-structure-aware compute allocation, and train--test consistency for preview/skimming. We propose the \textbf{Fovea-Block-Skip Transformer} (FBS), which injects a causal, trainable loop into Transformers via Parafovea-Attention Window (PAW), Chunk-Head (CH), and Skip-Gate (SG). Across diverse benchmarks, FBS improves the quality-efficiency trade-off without increasing parameters, and ablations show the three modules are complementary. 

---
# TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning 

**Authors**: Mingzu Liu, Hao Fang, Runmin Cong  

**Link**: [PDF](https://arxiv.org/pdf/2601.21692)  

**Abstract**: Fine-Tuning-as-a-Service (FTaaS) facilitates the customization of Multimodal Large Language Models (MLLMs) but introduces critical backdoor risks via poisoned data. Existing defenses either rely on supervised signals or fail to generalize across diverse trigger types and modalities. In this work, we uncover a universal backdoor fingerprint-attention allocation divergence-where poisoned samples disrupt the balanced attention distribution across three functional components: system instructions, vision inputs, and user textual queries, regardless of trigger morphology. Motivated by this insight, we propose Tri-Component Attention Profiling (TCAP), an unsupervised defense framework to filter backdoor samples. TCAP decomposes cross-modal attention maps into the three components, identifies trigger-responsive attention heads via Gaussian Mixture Model (GMM) statistical profiling, and isolates poisoned samples through EM-based vote aggregation. Extensive experiments across diverse MLLM architectures and attack methods demonstrate that TCAP achieves consistently strong performance, establishing it as a robust and practical backdoor defense in MLLMs. 

---
# SONIC-O1: A Real-World Benchmark for Evaluating Multimodal Large Language Models on Audio-Video Understanding 

**Authors**: Ahmed Y. Radwan, Christos Emmanouilidis, Hina Tabassum, Deval Pandya, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2601.21666)  

**Abstract**: Multimodal Large Language Models (MLLMs) are a major focus of recent AI research. However, most prior work focuses on static image understanding, while their ability to process sequential audio-video data remains underexplored. This gap highlights the need for a high-quality benchmark to systematically evaluate MLLM performance in a real-world setting. We introduce SONIC-O1, a comprehensive, fully human-verified benchmark spanning 13 real-world conversational domains with 4,958 annotations and demographic metadata. SONIC-O1 evaluates MLLMs on key tasks, including open-ended summarization, multiple-choice question (MCQ) answering, and temporal localization with supporting rationales (reasoning). Experiments on closed- and open-source models reveal limitations. While the performance gap in MCQ accuracy between two model families is relatively small, we observe a substantial 22.6% performance difference in temporal localization between the best performing closed-source and open-source models. Performance further degrades across demographic groups, indicating persistent disparities in model behavior. Overall, SONIC-O1 provides an open evaluation suite for temporally grounded and socially robust multimodal understanding. We release SONIC-O1 for reproducibility and research: Project page: this https URL Dataset: this https URL Github: this https URL Leaderboard: this https URL 

---
# ScholarGym: Benchmarking Deep Research Workflows on Academic Literature Retrieval 

**Authors**: Hao Shen, Hang Yang, Zhouhong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21654)  

**Abstract**: Tool-augmented large language models have advanced from single-turn question answering to deep research workflows that iteratively plan queries, invoke external tools, and synthesize information to address complex information needs. Evaluating such workflows presents a fundamental challenge: reliance on live APIs introduces non-determinism, as tool invocations may yield different results across runs due to temporal drift, rate limiting, and evolving backend states. This variance undermines reproducibility and invalidates cross-system comparisons.
We present ScholarGym, a simulation environment for reproducible evaluation of deep research workflows on academic literature. The environment decouples workflow components into query planning, tool invocation, and relevance assessment, enabling fine-grained analysis of each stage under controlled conditions. Built on a static corpus of 570K papers with deterministic retrieval, ScholarGym provides 2,536 queries with expert-annotated ground truth. Experiments across diverse backbone models reveal how reasoning capabilities, planning strategies, and selection mechanisms interact over iterative refinement. 

---
# Semantic Content Determines Algorithmic Performance 

**Authors**: Martiño Ríos-García, Nawaf Alampara, Kevin Maik Jablonka  

**Link**: [PDF](https://arxiv.org/pdf/2601.21618)  

**Abstract**: Counting should not depend on what is being counted; more generally, any algorithm's behavior should be invariant to the semantic content of its arguments. We introduce WhatCounts to test this property in isolation. Unlike prior work that conflates semantic sensitivity with reasoning complexity or prompt variation, WhatCounts is atomic: count items in an unambiguous, delimited list with no duplicates, distractors, or reasoning steps for different semantic types. Frontier LLMs show over 40% accuracy variation depending solely on what is being counted - cities versus chemicals, names versus symbols. Controlled ablations rule out confounds. The gap is semantic, and it shifts unpredictably with small amounts of unrelated fine-tuning. LLMs do not implement algorithms; they approximate them, and the approximation is argument-dependent. As we show with an agentic example, this has implications beyond counting: any LLM function may carry hidden dependencies on the meaning of its inputs. 

---
# RecNet: Self-Evolving Preference Propagation for Agentic Recommender Systems 

**Authors**: Bingqian Li, Xiaolei Wang, Junyi Li, Weitao Li, Long Zhang, Sheng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21609)  

**Abstract**: Agentic recommender systems leverage Large Language Models (LLMs) to model complex user behaviors and support personalized decision-making. However, existing methods primarily model preference changes based on explicit user-item interactions, which are sparse, noisy, and unable to reflect the real-time, mutual influences among users and items. To address these limitations, we propose RecNet, a self-evolving preference propagation framework that proactively propagates real-time preference updates across related users and items. RecNet consists of two complementary phases. In the forward phase, the centralized preference routing mechanism leverages router agents to integrate preference updates and dynamically propagate them to the most relevant agents. To ensure accurate and personalized integration of propagated preferences, we further introduce a personalized preference reception mechanism, which combines a message buffer for temporary caching and an optimizable, rule-based filter memory to guide selective preference assimilation based on past experience and interests. In the backward phase, the feedback-driven propagation optimization mechanism simulates a multi-agent reinforcement learning framework, using LLMs for credit assignment, gradient analysis, and module-level optimization, enabling continuous self-evolution of propagation strategies. Extensive experiments on various scenarios demonstrate the effectiveness of RecNet in modeling preference propagation for recommender systems. 

---
# Search-Based Risk Feature Discovery in Document Structure Spaces under a Constrained Budget 

**Authors**: Saisubramaniam Gopalakrishnan, Harikrishnan P M, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2601.21608)  

**Abstract**: Enterprise-grade Intelligent Document Processing (IDP) systems support high-stakes workflows across finance, insurance, and healthcare. Early-phase system validation under limited budgets mandates uncovering diverse failure mechanisms, rather than identifying a single worst-case document. We formalize this challenge as a Search-Based Software Testing (SBST) problem, aiming to identify complex interactions between document variables, with the objective to maximize the number of distinct failure types discovered within a fixed evaluation budget. Our methodology operates on a combinatorial space of document configurations, rendering instances of structural \emph{risk features} to induce realistic failure conditions. We benchmark a diverse portfolio of search strategies spanning evolutionary, swarm-based, quality-diversity, learning-based, and quantum under identical budget constraints. Through configuration-level exclusivity, win-rate, and cross-temporal overlap analyses, we show that different solvers consistently uncover failure modes that remain undiscovered by specific alternatives at comparable budgets. Crucially, cross-temporal analysis reveals persistent solver-specific discoveries across all evaluated budgets, with no single strategy exhibiting absolute dominance. While the union of all solvers eventually recovers the observed failure space, reliance on any individual method systematically delays the discovery of important risks. These results demonstrate intrinsic solver complementarity and motivate portfolio-based SBST strategies for robust industrial IDP validation. 

---
# CORE: Collaborative Reasoning via Cross Teaching 

**Authors**: Kshitij Mishra, Mirat Aubakirov, Martin Takac, Nils Lukas, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21600)  

**Abstract**: Large language models exhibit complementary reasoning errors: on the same instance, one model may succeed with a particular decomposition while another fails. We propose Collaborative Reasoning (CORE), a training-time collaboration framework that converts peer success into a learning signal via a cross-teaching protocol. Each problem is solved in two stages: a cold round of independent sampling, followed by a contexted rescue round in which models that failed receive hint extracted from a successful peer. CORE optimizes a combined reward that balances (i) correctness, (ii) a lightweight DPP-inspired diversity term to reduce error overlap, and (iii) an explicit rescue bonus for successful recovery. We evaluate CORE across four standard reasoning datasets GSM8K, MATH, AIME, and GPQA. With only 1,000 training examples, a pair of small open source models (3B+4B) reaches Pass@2 of 99.54% on GSM8K and 92.08% on MATH, compared to 82.50% and 74.82% for single-model training. On harder datasets, the 3B+4B pair reaches Pass@2 of 77.34% on GPQA (trained on 348 examples) and 79.65% on AIME (trained on 792 examples), using a training-time budget of at most 1536 context tokens and 3072 generated tokens. Overall, these results show that training-time collaboration can reliably convert model complementarity into large gains without scaling model size. 

---
# Beyond Imitation: Reinforcement Learning for Active Latent Planning 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.21598)  

**Abstract**: Aiming at efficient and dense chain-of-thought (CoT) reasoning, latent reasoning methods fine-tune Large Language Models (LLMs) to substitute discrete language tokens with continuous latent tokens. These methods consume fewer tokens compared to the conventional language CoT reasoning and have the potential to plan in a dense latent space. However, current latent tokens are generally supervised based on imitating language labels. Considering that there can be multiple equivalent but diverse CoT labels for a question, passively imitating an arbitrary one may lead to inferior latent token representations and latent reasoning policies, undermining the potential planning ability and resulting in clear gaps between training and testing. In this work, we emphasize the importance of active planning over the representation space of latent tokens in achieving the optimal latent reasoning policy. So, we propose the \underline{A}c\underline{t}ive Latent \underline{P}lanning method (ATP-Latent), which models the supervision process of latent tokens as a conditional variational auto-encoder (VAE) to obtain a smoother latent space. Moreover, to facilitate the most reasonable latent reasoning policy, ATP-Latent conducts reinforcement learning (RL) with an auxiliary coherence reward, which is calculated based on the consistency between VAE-decoded contents of latent tokens, enabling a guided RL process. In experiments on LLaMA-1B, ATP-Latent demonstrates +4.1\% accuracy and -3.3\% tokens on four benchmarks compared to advanced baselines. Codes are available on this https URL. 

---
# Depth-Recurrent Attention Mixtures: Giving Latent Reasoning the Attention it Deserves 

**Authors**: Jonas Knupp, Jan Hendrik Metzen, Jeremias Bohn, Georg Groh, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2601.21582)  

**Abstract**: Depth-recurrence facilitates latent reasoning by sharing parameters across depths. However, prior work lacks combined FLOP-, parameter-, and memory-matched baselines, underutilizes depth-recurrence due to partially fixed layer stacks, and ignores the bottleneck of constant hidden-sizes that restricts many-step latent reasoning. To address this, we introduce a modular framework of depth-recurrent attention mixtures (Dreamer), combining sequence attention, depth attention, and sparse expert attention. It alleviates the hidden-size bottleneck through attention along depth, decouples scaling dimensions, and allows depth-recurrent models to scale efficiently and effectively. Across language reasoning benchmarks, our models require 2 to 8x fewer training tokens for the same accuracy as FLOP-, parameter-, and memory-matched SOTA, and outperform ca. 2x larger SOTA models with the same training tokens. We further present insights into knowledge usage across depths, e.g., showing 2 to 11x larger expert selection diversity than SOTA MoEs. 

---
# Chain Of Thought Compression: A Theoritical Analysis 

**Authors**: Juncai Li, Ru Li, Yuxiang Zhou, Boxiang Ma, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21576)  

**Abstract**: Chain-of-Thought (CoT) has unlocked advanced reasoning abilities of Large Language Models (LLMs) with intermediate steps, yet incurs prohibitive computational costs due to generation of extra tokens. Recent studies empirically show that compressing reasoning steps into latent states, or implicit CoT compression, offers a token-efficient alternative. However, the mechanism behind CoT compression remains unclear. In this paper, we provide the first theoretical analysis of the difficulty of learning to internalize intermediate reasoning steps. By introducing Order-r Interaction, we prove that the learning signal for high-order logical dependencies exponentially decays to solve irreducible problem, where skipping intermediate steps inevitably leads to high-order interaction barriers. To empirically validate this, we introduce NatBool-DAG, a challenging benchmark designed to enforce irreducible logical reasoning and eliminate semantic shortcuts. Guided by our theoretical findings, we propose ALiCoT (Aligned Implicit CoT), a novel framework that overcomes the signal decay by aligning latent token distributions with intermediate reasoning states. Experimental results demonstrate that ALiCoT successfully unlocks efficient reasoning: it achieves a 54.4x speedup while maintaining performance comparable to explicit CoT. 

---
# EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots 

**Authors**: Zixing Lei, Genjia Liu, Yuanshuo Zhang, Qipeng Liu, Chuan Wen, Shanghang Zhang, Wenzhao Lian, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21570)  

**Abstract**: The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field. 

---
# Meta Context Engineering via Agentic Skill Evolution 

**Authors**: Haoran Ye, Xuning He, Vincent Arak, Haonan Dong, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.21557)  

**Abstract**: The operational efficacy of large language models relies heavily on their inference-time context. This has established Context Engineering (CE) as a formal discipline for optimizing these inputs. Current CE methods rely on manually crafted harnesses, such as rigid generation-reflection workflows and predefined context schemas. They impose structural biases and restrict context optimization to a narrow, intuition-bound design space. To address this, we introduce Meta Context Engineering (MCE), a bi-level framework that supersedes static CE heuristics by co-evolving CE skills and context artifacts. In MCE iterations, a meta-level agent refines engineering skills via agentic crossover, a deliberative search over the history of skills, their executions, and evaluations. A base-level agent executes these skills, learns from training rollouts, and optimizes context as flexible files and code. We evaluate MCE across five disparate domains under offline and online settings. MCE demonstrates consistent performance gains, achieving 5.6--53.8% relative improvement over state-of-the-art agentic CE methods (mean of 16.9%), while maintaining superior context adaptability, transferability, and efficiency in both context usage and training. 

---
# ShardMemo: Masked MoE Routing for Sharded Agentic LLM Memory 

**Authors**: Yang Zhao, Chengxiao Dai, Yue Xiu, Mengying Kou, Yuliang Zheng, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2601.21545)  

**Abstract**: Agentic large language model (LLM) systems rely on external memory for long-horizon state and concurrent multi-agent execution, but centralized indexes and heuristic partitions become bottlenecks as memory volume and parallel access grow. We present ShardMemo, a budgeted tiered memory service with Tier A per-agent working state, Tier B sharded evidence with shard-local approximate nearest neighbor (ANN) indexes, and Tier C, a versioned skill library. Tier B enforces scope-before-routing: structured eligibility constraints mask ineligible shards before routing or ANN search. We cast shard probing as masked mixture-of-experts (MoE) routing over eligible shards, probing up to $B_{\mathrm{probe}}$ shards via Top-$B_{\mathrm{probe}}$ or adaptive Top-$P$, and use cost-aware gating over profile/observation/session shard families; the router is trained from evidence-to-shard supervision. On LoCoMo, ShardMemo improves over the strongest baseline (GAM) by +5.11 to +6.82 F1 across question categories. Under a fixed-budget routing setting ($B_{\mathrm{probe}}=3$), ShardMemo improves over cosine-to-prototype shard routing by +6.87 F1 while reducing retrieval work (VecScan 521->414, -20.5%) and p95 latency (95->76 ms). On long-context HotpotQA, ShardMemo achieves 63.41/61.88/57.95 F1 at 56K/224K/448K tokens. On ToolBench, Tier C reaches 0.97 Precision@3 and 1.94 StepRed (+10.2% and +7.2% over embedding-similarity retrieval). 

---
# ARGORA: Orchestrated Argumentation for Causally Grounded LLM Reasoning and Decision Making 

**Authors**: Youngjin Jin, Hanna Kim, Kwanwoo Kim, Chanhee Lee, Seungwon Shin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21533)  

**Abstract**: Existing multi-expert LLM systems gather diverse perspectives but combine them through simple aggregation, obscuring which arguments drove the final decision. We introduce ARGORA, a framework that organizes multi-expert discussions into explicit argumentation graphs showing which arguments support or attack each other. By casting these graphs as causal models, ARGORA can systematically remove individual arguments and recompute outcomes, identifying which reasoning chains were necessary and whether decisions would change under targeted modifications. We further introduce a correction mechanism that aligns internal reasoning with external judgments when they disagree. Across diverse benchmarks and an open-ended use case, ARGORA achieves competitive accuracy and demonstrates corrective behavior: when experts initially disagree, the framework resolves disputes toward correct answers more often than it introduces new errors, while providing causal diagnostics of decisive arguments. 

---
# KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization 

**Authors**: Alireza Nadaf, Alireza Mohammadshahi, Majid Yazdani  

**Link**: [PDF](https://arxiv.org/pdf/2601.21526)  

**Abstract**: We introduce KAPSO, a modular framework for autonomous program synthesis and optimization. Given a natural language goal and an evaluation method, KAPSO iteratively performs ideation, code synthesis and editing, execution, evaluation, and learning to improve a runnable artifact toward measurable objectives. Rather than treating synthesis as the endpoint, KAPSO uses synthesis as an operator within a long-horizon optimization loop, where progress is defined by evaluator outcomes.
KAPSO targets long-horizon failures common in coding agents, including lost experimental state, brittle debugging, and weak reuse of domain expertise, by integrating three tightly coupled components. First, a git-native experimentation engine isolates each attempt as a branch, producing reproducible artifacts and preserving provenance across iterations. Second, a knowledge system ingests heterogeneous sources, including repositories, internal playbooks, and curated external resources such as documentation, scientific papers, and web search results, and organizes them into a structured representation that supports retrieval over workflows, implementations, and environment constraints. Third, a cognitive memory layer coordinates retrieval and maintains an episodic store of reusable lessons distilled from experiment traces (run logs, diffs, and evaluator feedback), reducing repeated error modes and accelerating convergence.
We evaluated KAPSO on MLE-Bench (Kaggle-style ML competitions) and ALE-Bench (AtCoder heuristic optimization), and report end-to-end performance.
Code Available at: this https URL 

---
# LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI 

**Authors**: Niki van Stein, Anna V. Kononova, Lars Kotthoff, Thomas Bäck  

**Link**: [PDF](https://arxiv.org/pdf/2601.21511)  

**Abstract**: Large language models have enabled automated algorithm design (AAD) by generating optimization algorithms directly from natural-language prompts. While evolutionary frameworks such as LLaMEA demonstrate strong exploratory capabilities across the algorithm design space, their search dynamics are entirely driven by fitness feedback, leaving substantial information about the generated code unused. We propose a mechanism for guiding AAD using feedback constructed from graph-theoretic and complexity features extracted from the abstract syntax trees of the generated algorithms, based on a surrogate model learned over an archive of evaluated solutions. Using explainable AI techniques, we identify features that substantially affect performance and translate them into natural-language mutation instructions that steer subsequent LLM-based code generation without restricting expressivity.
We propose LLaMEA-SAGE, which integrates this feature-driven guidance into LLaMEA, and evaluate it across several benchmarks. We show that the proposed structured guidance achieves the same performance faster than vanilla LLaMEA in a small controlled experiment. In a larger-scale experiment using the MA-BBOB suite from the GECCO-MA-BBOB competition, our guided approach achieves superior performance compared to state-of-the-art AAD methods. These results demonstrate that signals derived from code can effectively bias LLM-driven algorithm evolution, bridging the gap between code structure and human-understandable performance feedback in automated algorithm design. 

---
# The Effectiveness of Style Vectors for Steering Large Language Models: A Human Evaluation 

**Authors**: Diaoulé Diallo, Katharina Dworatzyk, Sophie Jentzsch, Peer Schütt, Sabine Theis, Tobias Hecking  

**Link**: [PDF](https://arxiv.org/pdf/2601.21505)  

**Abstract**: Controlling the behavior of large language models (LLMs) at inference time is essential for aligning outputs with human abilities and safety requirements. \emph{Activation steering} provides a lightweight alternative to prompt engineering and fine-tuning by directly modifying internal activations to guide generation. This research advances the literature in three significant directions. First, while previous work demonstrated the technical feasibility of steering emotional tone using automated classifiers, this paper presents the first human evaluation of activation steering concerning the emotional tone of LLM outputs, collecting over 7,000 crowd-sourced ratings from 190 participants via Prolific ($n=190$). These ratings assess both perceived emotional intensity and overall text quality. Second, we find strong alignment between human and model-based quality ratings (mean $r=0.776$, range $0.157$--$0.985$), indicating automatic scoring can proxy perceived quality. Moderate steering strengths ($\lambda \approx 0.15$) reliably amplify target emotions while preserving comprehensibility, with the strongest effects for disgust ($\eta_p^2 = 0.616$) and fear ($\eta_p^2 = 0.540$), and minimal effects for surprise ($\eta_p^2 = 0.042$). Finally, upgrading from Alpaca to LlaMA-3 yielded more consistent steering with significant effects across emotions and strengths (all $p < 0.001$). Inter-rater reliability was high (ICC $= 0.71$--$0.87$), underscoring the robustness of the findings. These findings support activation-based control as a scalable method for steering LLM behavior across affective dimensions. 

---
# MAR: Efficient Large Language Models via Module-aware Architecture Refinement 

**Authors**: Junhong Cai, Guiqin Wang, Kejie Zhao, Jianxiong Tang, Xiang Wang, Luziwei Leng, Ran Cheng, Yuxin Ma, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21503)  

**Abstract**: Large Language Models (LLMs) excel across diverse domains but suffer from high energy costs due to quadratic attention and dense Feed-Forward Network (FFN) operations. To address these issues, we propose Module-aware Architecture Refinement (MAR), a two-stage framework that integrates State Space Models (SSMs) for linear-time sequence modeling and applies activation sparsification to reduce FFN costs. In addition, to mitigate low information density and temporal mismatch in integrating Spiking Neural Networks (SNNs) with SSMs, we design the Adaptive Ternary Multi-step Neuron (ATMN) and the Spike-aware Bidirectional Distillation Strategy (SBDS). Extensive experiments demonstrate that MAR effectively restores the performance of its dense counterpart under constrained resources while substantially reducing inference energy consumption. Furthermore, it outperforms efficient models of comparable or even larger scale, underscoring its potential for building efficient and practical LLMs. 

---
# The Path of Least Resistance: Guiding LLM Reasining Trajectories with Prefix Consensus 

**Authors**: Ishan Jindal, Sai Prashanth Akuthota, Jayant Taneja, Sachin Dev Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.21494)  

**Abstract**: Large language models achieve strong reasoning performance, but inference strategies such as Self-Consistency (SC) are computationally expensive, as they fully expand all reasoning traces. We introduce PoLR (Path of Least Resistance), the first inference-time method to leverage prefix consistency for compute-efficient reasoning. PoLR clusters short prefixes of reasoning traces, identifies the dominant cluster, and expands all paths in that cluster, preserving the accuracy benefits of SC while substantially reducing token usage and latency. Our theoretical analysis, framed via mutual information and entropy, explains why early reasoning steps encode strong signals predictive of final correctness. Empirically, PoLR consistently matches or exceeds SC across GSM8K, MATH500, AIME24/25, and GPQA-DIAMOND, reducing token usage by up to 60% and wall-clock latency by up to 50%. Moreover, PoLR is fully complementary to adaptive inference methods (e.g., Adaptive Consistency, Early-Stopping SC) and can serve as a drop-in pre-filter, making SC substantially more efficient and scalable without requiring model fine-tuning. 

---
# ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management 

**Authors**: Zaifeng Pan, Yipeng Shen, Zhengding Hu, Zhuang Wang, Aninda Manocha, Zheng Wang, Zhongkai Yu, Yue Guan, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.21473)  

**Abstract**: LLM-based multi-agent simulations are increasingly adopted across application domains, but remain difficult to scale due to GPU memory pressure. Each agent maintains private GPU-resident states, including models, prefix caches, and adapters, which quickly exhaust device memory as the agent count grows. We identify two key properties of these workloads: sparse agent activation and an estimable agent invocation order. Based on an analysis of representative workload classes, we introduce invocation distance, a unified abstraction that estimates the relative order in which agents will issue future LLM requests. Leveraging this abstraction, we present ScaleSim, a memory-efficient LLM serving system for large-scale multi-agent simulations. ScaleSim enables proactive prefetching and priority-based eviction, supports diverse agent-specific memory through a modular interface, and achieves up to 1.74x speedup over SGLang on simulation benchmarks. 

---
# MemOCR: Layout-Aware Visual Memory for Efficient Long-Horizon Reasoning 

**Authors**: Yaorui Shi, Shugui Liu, Yu Yang, Wenyu Mao, Yuxin Chen, Qi GU, Hui Su, Xunliang Cai, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21468)  

**Abstract**: Long-horizon agentic reasoning necessitates effectively compressing growing interaction histories into a limited context window. Most existing memory systems serialize history as text, where token-level cost is uniform and scales linearly with length, often spending scarce budget on low-value details. To this end, we introduce MemOCR, a multimodal memory agent that improves long-horizon reasoning under tight context budgets by allocating memory space with adaptive information density through visual layout. Concretely, MemOCR maintains a structured rich-text memory (e.g., headings, highlights) and renders it into an image that the agent consults for memory access, visually prioritizing crucial evidence while aggressively compressing auxiliary details. To ensure robustness across varying memory budgets, we train MemOCR with reinforcement learning under budget-aware objectives that expose the agent to diverse compression levels. Across long-context multi-hop and single-hop question-answering benchmarks, MemOCR outperforms strong text-based baselines and achieves more effective context utilization under extreme budgets. 

---
# Topeax -- An Improved Clustering Topic Model with Density Peak Detection and Lexical-Semantic Term Importance 

**Authors**: Márton Kardos  

**Link**: [PDF](https://arxiv.org/pdf/2601.21465)  

**Abstract**: Text clustering is today the most popular paradigm for topic modelling, both in academia and industry. Despite clustering topic models' apparent success, we identify a number of issues in Top2Vec and BERTopic, which remain largely unsolved. Firstly, these approaches are unreliable at discovering natural clusters in corpora, due to extreme sensitivity to sample size and hyperparameters, the default values of which result in suboptimal behaviour. Secondly, when estimating term importance, BERTopic ignores the semantic distance of keywords to topic vectors, while Top2Vec ignores word counts in the corpus. This results in, on the one hand, less coherent topics due to the presence of stop words and junk words, and lack of variety and trust on the other. In this paper, I introduce a new approach, \textbf{Topeax}, which discovers the number of clusters from peaks in density estimates, and combines lexical and semantic indices of term importance to gain high-quality topic keywords. Topeax is demonstrated to be better at both cluster recovery and cluster description than Top2Vec and BERTopic, while also exhibiting less erratic behaviour in response to changing sample size and hyperparameters. 

---
# LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning 

**Authors**: Xunkai Li, Zhengyu Wu, Zekai Chen, Henan Sun, Daohan Su, Guang Zeng, Hongchao Qin, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21453)  

**Abstract**: Recently, the rapid advancement of multimodal domains has driven a data-centric paradigm shift in graph ML, transitioning from text-attributed to multimodal-attributed graphs. This advancement significantly enhances data representation and expands the scope of graph downstream tasks, such as modality-oriented tasks, thereby improving the practical utility of graph ML. Despite its promise, limitations exist in the current neural paradigms: (1) Neglect Context in Modality Alignment: Most existing methods adopt topology-constrained or modality-specific operators as tokenizers. These aligners inevitably neglect graph context and inhibit modality interaction, resulting in suboptimal alignment. (2) Lack of Adaptation in Modality Fusion: Most existing methods are simple adaptations for 2-modality graphs and fail to adequately exploit aligned tokens equipped with topology priors during fusion, leading to poor generalizability and performance degradation. To address the above issues, we propose LION (c\underline{LI}ff\underline{O}rd \underline{N}eural paradigm) based on the Clifford algebra and decoupled graph neural paradigm (i.e., propagation-then-aggregation) to implement alignment-then-fusion in multimodal-attributed graphs. Specifically, we first construct a modality-aware geometric manifold grounded in Clifford algebra. This geometric-induced high-order graph propagation efficiently achieves modality interaction, facilitating modality alignment. Then, based on the geometric grade properties of aligned tokens, we propose adaptive holographic aggregation. This module integrates the energy and scale of geometric grades with learnable parameters to improve modality fusion. Extensive experiments on 9 datasets demonstrate that LION significantly outperforms SOTA baselines across 3 graph and 3 modality downstream tasks. 

---
# ChipBench: A Next-Step Benchmark for Evaluating LLM Performance in AI-Aided Chip Design 

**Authors**: Zhongkai Yu, Chenyang Zhou, Yichen Lin, Hejia Zhang, Haotian Ye, Junxia Cui, Zaifeng Pan, Jishen Zhao, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.21448)  

**Abstract**: While Large Language Models (LLMs) show significant potential in hardware engineering, current benchmarks suffer from saturation and limited task diversity, failing to reflect LLMs' performance in real industrial workflows. To address this gap, we propose a comprehensive benchmark for AI-aided chip design that rigorously evaluates LLMs across three critical tasks: Verilog generation, debugging, and reference model generation. Our benchmark features 44 realistic modules with complex hierarchical structures, 89 systematic debugging cases, and 132 reference model samples across Python, SystemC, and CXXRTL. Evaluation results reveal substantial performance gaps, with state-of-the-art Claude-4.5-opus achieving only 30.74\% on Verilog generation and 13.33\% on Python reference model generation, demonstrating significant challenges compared to existing saturated benchmarks where SOTA models achieve over 95\% pass rates. Additionally, to help enhance LLM reference model generation, we provide an automated toolbox for high-quality training data generation, facilitating future research in this underexplored domain. Our code is available at this https URL. 

---
# The Paradox of Robustness: Decoupling Rule-Based Logic from Affective Noise in High-Stakes Decision-Making 

**Authors**: Jon Chun, Katherine Elkins  

**Link**: [PDF](https://arxiv.org/pdf/2601.21439)  

**Abstract**: While Large Language Models (LLMs) are widely documented to be sensitive to minor prompt perturbations and prone to sycophantic alignment with user biases, their robustness in consequential, rule-bound decision-making remains under-explored. In this work, we uncover a striking "Paradox of Robustness": despite their known lexical brittleness, instruction-tuned LLMs exhibit a behavioral and near-total invariance to emotional framing effects. Using a novel controlled perturbation framework across three high-stakes domains (healthcare, law, and finance), we quantify a robustness gap where LLMs demonstrate 110-300 times greater resistance to narrative manipulation than human subjects. Specifically, we find a near-zero effect size for models (Cohen's h = 0.003) compared to the substantial biases observed in humans (Cohen's h in [0.3, 0.8]). This result is highly counterintuitive and suggests the mechanisms driving sycophancy and prompt sensitivity do not necessarily translate to a failure in logical constraint satisfaction. We show that this invariance persists across models with diverse training paradigms. Our findings show that while LLMs may be "brittle" to how a query is formatted, they are remarkably "stable" against why a decision should be biased. Our findings establish that instruction-tuned models can decouple logical rule-adherence from persuasive narratives, offering a source of decision stability that complements, and even potentially de-biases, human judgment in institutional contexts. We release the 162-scenario benchmark, code, and data to facilitate the rigorous evaluation of narrative-induced bias and robustness on this http URL. 

---
# When Prohibitions Become Permissions: Auditing Negation Sensitivity in Language Models 

**Authors**: Katherine Elkins, Jon Chun  

**Link**: [PDF](https://arxiv.org/pdf/2601.21433)  

**Abstract**: When a user tells an AI system that someone "should not" take an action, the system ought to treat this as a prohibition. Yet many large language models do the opposite: they interpret negated instructions as affirmations. We audited 16 models across 14 ethical scenarios and found that open-source models endorse prohibited actions 77% of the time under simple negation and 100% under compound negation -- a 317% increase over affirmative framing. Commercial models fare better but still show swings of 19-128%. Agreement between models drops from 74% on affirmative prompts to 62% on negated ones, and financial scenarios prove twice as fragile as medical ones. These patterns hold under deterministic decoding, ruling out sampling noise. We present case studies showing how these failures play out in practice, propose the Negation Sensitivity Index (NSI) as a governance metric, and outline a tiered certification framework with domain-specific thresholds. The findings point to a gap between what current alignment techniques achieve and what safe deployment requires: models that cannot reliably distinguish "do X" from "do not X" should not be making autonomous decisions in high-stakes contexts. 

---
# System 1&2 Synergy via Dynamic Model Interpolation 

**Authors**: Chenxu Yang, Qingyi Si, Chong Tian, Xiyu Liu, Dingyu Yao, Chuanyu Qin, Zheng Lin, Weiping Wang, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21414)  

**Abstract**: Training a unified language model that adapts between intuitive System 1 and deliberative System 2 remains challenging due to interference between their cognitive modes. Recent studies have thus pursued making System 2 models more efficient. However, these approaches focused on output control, limiting what models produce. We argue that this paradigm is misaligned: output length is merely a symptom of the model's cognitive configuration, not the root cause. In this work, we shift the focus to capability control, which modulates \textit{how models think} rather than \textit{what they produce}. To realize this, we leverage existing Instruct and Thinking checkpoints through dynamic parameter interpolation, without additional training. Our pilot study establishes that linear interpolation yields a convex, monotonic Pareto frontier, underpinned by representation continuity and structural connectivity. Building on this, we propose \textbf{DAMI} (\textbf{D}yn\textbf{A}mic \textbf{M}odel \textbf{I}nterpolation), a framework that estimates a query-specific Reasoning Intensity $\lambda(q)$ to configure cognitive depth. For training-based estimation, we develop a preference learning method encoding accuracy and efficiency criteria. For zero-shot deployment, we introduce a confidence-based method leveraging inter-model cognitive discrepancy. Experiments on five mathematical reasoning benchmarks demonstrate that DAMI achieves higher accuracy than the Thinking model while remaining efficient, effectively combining the efficiency of System 1 with the reasoning depth of System 2. 

---
# DataCross: A Unified Benchmark and Agent Framework for Cross-Modal Heterogeneous Data Analysis 

**Authors**: Ruyi Qi, Zhou Liu, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21403)  

**Abstract**: In real-world data science and enterprise decision-making, critical information is often fragmented across directly queryable structured sources (e.g., SQL, CSV) and "zombie data" locked in unstructured visual documents (e.g., scanned reports, invoice images). Existing data analytics agents are predominantly limited to processing structured data, failing to activate and correlate this high-value visual information, thus creating a significant gap with industrial needs. To bridge this gap, we introduce DataCross, a novel benchmark and collaborative agent framework for unified, insight-driven analysis across heterogeneous data modalities. DataCrossBench comprises 200 end-to-end analysis tasks across finance, healthcare, and other domains. It is constructed via a human-in-the-loop reverse-synthesis pipeline, ensuring realistic complexity, cross-source dependency, and verifiable ground truth. The benchmark categorizes tasks into three difficulty tiers to evaluate agents' capabilities in visual table extraction, cross-modal alignment, and multi-step joint reasoning. We also propose the DataCrossAgent framework, inspired by the "divide-and-conquer" workflow of human analysts. It employs specialized sub-agents, each an expert on a specific data source, which are coordinated via a structured workflow of Intra-source Deep Exploration, Key Source Identification, and Contextual Cross-pollination. A novel reReAct mechanism enables robust code generation and debugging for factual verification. Experimental results show that DataCrossAgent achieves a 29.7% improvement in factuality over GPT-4o and exhibits superior robustness on high-difficulty tasks, effectively activating fragmented "zombie data" for insightful, cross-modal analysis. 

---
# TeachBench: A Syllabus-Grounded Framework for Evaluating Teaching Ability in Large Language Models 

**Authors**: Zheng Li, Siyao Song, Jingyuan Ma, Rui Li, Ying Zeng, Minghao Li, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2601.21375)  

**Abstract**: Large language models (LLMs) show promise as teaching assistants, yet their teaching capability remains insufficiently evaluated. Existing benchmarks mainly focus on problem-solving or problem-level guidance, leaving knowledge-centered teaching underexplored. We propose a syllabus-grounded evaluation framework that measures LLM teaching capability via student performance improvement after multi-turn instruction. By restricting teacher agents to structured knowledge points and example problems, the framework avoids information leakage and enables reuse of existing benchmarks. We instantiate the framework on Gaokao data across multiple subjects. Experiments reveal substantial variation in teaching effectiveness across models and domains: some models perform well in mathematics, while teaching remains challenging in physics and chemistry. We also find that incorporating example problems does not necessarily improve teaching, as models often shift toward example-specific error correction. Overall, our results highlight teaching ability as a distinct and measurable dimension of LLM behavior. 

---
# NEMO: Execution-Aware Optimization Modeling via Autonomous Coding Agents 

**Authors**: Yang Song, Anoushka Vyas, Zirui Wei, Sina Khoshfetrat Pakazad, Henrik Ohlsson, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2601.21372)  

**Abstract**: In this paper, we present NEMO, a system that translates Natural-language descriptions of decision problems into formal Executable Mathematical Optimization implementations, operating collaboratively with users or autonomously. Existing approaches typically rely on specialized large language models (LLMs) or bespoke, task-specific agents. Such methods are often brittle, complex and frequently generating syntactically invalid or non-executable code.
NEMO instead centers on remote interaction with autonomous coding agents (ACAs), treated as a first-class abstraction analogous to API-based interaction with LLMs. This design enables the construction of higher-level systems around ACAs that structure, consolidate, and iteratively refine task specifications. Because ACAs execute within sandboxed environments, code produced by NEMO is executable by construction, allowing automated validation and repair.
Building on this, we introduce novel coordination patterns with and across ACAs, including asymmetric validation loops between independently generated optimizer and simulator implementations (serving as a high-level validation mechanism), external memory for experience reuse, and robustness enhancements via minimum Bayes risk (MBR) decoding and self-consistency. We evaluate NEMO on nine established optimization benchmarks. As depicted in Figure 1, it achieves state-of-the-art performance on the majority of tasks, with substantial margins on several datasets, demonstrating the power of execution-aware agentic architectures for automated optimization modeling. 

---
# Hebbian Learning with Global Direction 

**Authors**: Wenjia Hua, Kejie Zhao, Luziwei Leng, Ran Cheng, Yuxin Ma, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21367)  

**Abstract**: Backpropagation algorithm has driven the remarkable success of deep neural networks, but its lack of biological plausibility and high computational costs have motivated the ongoing search for alternative training methods. Hebbian learning has attracted considerable interest as a biologically plausible alternative to backpropagation. Nevertheless, its exclusive reliance on local information, without consideration of global task objectives, fundamentally limits its scalability. Inspired by the biological synergy between neuromodulators and local plasticity, we introduce a novel model-agnostic Global-guided Hebbian Learning (GHL) framework, which seamlessly integrates local and global information to scale up across diverse networks and tasks. In specific, the local component employs Oja's rule with competitive learning to ensure stable and effective local updates. Meanwhile, the global component introduces a sign-based signal that guides the direction of local Hebbian plasticity updates. Extensive experiments demonstrate that our method consistently outperforms existing Hebbian approaches. Notably, on large-scale network and complex datasets like ImageNet, our framework achieves the competitive results and significantly narrows the gap with standard backpropagation. 

---
# Latent Chain-of-Thought as Planning: Decoupling Reasoning from Verbalization 

**Authors**: Jiecong Wang, Hao Peng, Chunyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21358)  

**Abstract**: Chain-of-Thought (CoT) empowers Large Language Models (LLMs) to tackle complex problems, but remains constrained by the computational cost and reasoning path collapse when grounded in discrete token spaces. Recent latent reasoning approaches attempt to optimize efficiency by performing reasoning within continuous hidden states. However, these methods typically operate as opaque end-to-end mappings from explicit reasoning steps to latent states, and often require a pre-defined number of latent steps during inference. In this work, we introduce PLaT (Planning with Latent Thoughts), a framework that reformulates latent reasoning as planning by fundamentally decouple reasoning from verbalization. We model reasoning as a deterministic trajectory of latent planning states, while a separate Decoder grounds these thoughts into text when necessary. This decoupling allows the model to dynamically determine when to terminate reasoning rather than relying on fixed hyperparameters. Empirical results on mathematical benchmarks reveal a distinct trade-off: while PLaT achieves lower greedy accuracy than baselines, it demonstrates superior scalability in terms of reasoning diversity. This indicates that PLaT learns a robust, broader solution space, offering a transparent and scalable foundation for inference-time search. 

---
# BEAP-Agent: Backtrackable Execution and Adaptive Planning for GUI Agents 

**Authors**: Ziyu Lu, Tengjin Weng, Yiying Yang, Yuhang Zhao, Xinxin Huang, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21352)  

**Abstract**: GUI agents are designed to automate repetitive tasks and enhance productivity. However, existing GUI agents struggle to recover once they follow an incorrect exploration path, often leading to task failure. In this work, we model GUI task execution as a DFS process and propose BEAP-Agent, a DFS-based framework that supports long-range, multi-level state backtracking with dynamic task tracking and updating. The framework consists of three collaborative components: Planner, Executor, and Tracker. Together, they enable effective task exploration and execution. BEAP-Agent fills the gap in systematic backtracking mechanisms for GUI agents, offering a systematic solution for long-horizon task exploration. We conducted a systematic evaluation on the OSWorld benchmark, where BEAP-Agent achieved an accuracy of 28.2%, validating the effectiveness of the proposed method. 

---
# Dynamic Framework for Collaborative Learning: Leveraging Advanced LLM with Adaptive Feedback Mechanisms 

**Authors**: Hassam Tahir, Faizan Faisal, Fady Alnajjar, Muhammad Imran Taj, Lucia Gordon, Aila Khan, Michael Lwin, Omar Mubin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21344)  

**Abstract**: This paper presents a framework for integrating LLM into collaborative learning platforms to enhance student engagement, critical thinking, and inclusivity. The framework employs advanced LLMs as dynamic moderators to facilitate real-time discussions and adapt to learners' evolving needs, ensuring diverse and inclusive educational experiences. Key innovations include robust feedback mechanisms that refine AI moderation, promote reflective learning, and balance participation among users. The system's modular architecture featuring ReactJS for the frontend, Flask for backend operations, and efficient question retrieval supports personalized and engaging interactions through dynamic adjustments to prompts and discussion flows. Testing demonstrates that the framework significantly improves student collaboration, fosters deeper comprehension, and scales effectively across various subjects and user groups. By addressing limitations in static moderation and personalization in existing systems, this work establishes a strong foundation for next-generation AI-driven educational tools, advancing equitable and impactful learning outcomes. 

---
# Ostrakon-VL: Towards Domain-Expert MLLM for Food-Service and Retail Stores 

**Authors**: Zhiyong Shen, Gongpeng Zhao, Jun Zhou, Li Yu, Guandong Kou, Jichen Li, Chuanlei Dong, Zuncheng Li, Kaimao Li, Bingkun Wei, Shicheng Hu, Wei Xia, Wenguo Duan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21342)  

**Abstract**: Multimodal Large Language Models (MLLMs) have recently achieved substantial progress in general-purpose perception and reasoning. Nevertheless, their deployment in Food-Service and Retail Stores (FSRS) scenarios encounters two major obstacles: (i) real-world FSRS data, collected from heterogeneous acquisition devices, are highly noisy and lack auditable, closed-loop data curation, which impedes the construction of high-quality, controllable, and reproducible training corpora; and (ii) existing evaluation protocols do not offer a unified, fine-grained and standardized benchmark spanning single-image, multi-image, and video inputs, making it challenging to objectively gauge model robustness. To address these challenges, we first develop Ostrakon-VL, an FSRS-oriented MLLM based on Qwen3-VL-8B. Second, we introduce ShopBench, the first public benchmark for FSRS. Third, we propose QUAD (Quality-aware Unbiased Automated Data-curation), a multi-stage multimodal instruction data curation pipeline. Leveraging a multi-stage training strategy, Ostrakon-VL achieves an average score of 60.1 on ShopBench, establishing a new state of the art among open-source MLLMs with comparable parameter scales and diverse architectures. Notably, it surpasses the substantially larger Qwen3-VL-235B-A22B (59.4) by +0.7, and exceeds the same-scale Qwen3-VL-8B (55.3) by +4.8, demonstrating significantly improved parameter efficiency. These results indicate that Ostrakon-VL delivers more robust and reliable FSRS-centric perception and decision-making capabilities. To facilitate reproducible research, we will publicly release Ostrakon-VL and the ShopBench benchmark. 

---
# EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation 

**Authors**: Lang Cao, Qingyu Chen, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21340)  

**Abstract**: Electronic Health Records (EHRs) provide rich longitudinal clinical evidence that is central to medical decision-making, motivating the use of retrieval-augmented generation (RAG) to ground large language model (LLM) predictions. However, long-horizon EHRs often exceed LLM context limits, and existing approaches commonly rely on truncation or vanilla retrieval strategies that discard clinically relevant events and temporal dependencies. To address these challenges, we propose EHR-RAG, a retrieval-augmented framework designed for accurate interpretation of long-horizon structured EHR data. EHR-RAG introduces three components tailored to longitudinal clinical prediction tasks: Event- and Time-Aware Hybrid EHR Retrieval to preserve clinical structure and temporal dynamics, Adaptive Iterative Retrieval to progressively refine queries in order to expand broad evidence coverage, and Dual-Path Evidence Retrieval and Reasoning to jointly retrieves and reasons over both factual and counterfactual evidence. Experiments across four long-horizon EHR prediction tasks show that EHR-RAG consistently outperforms the strongest LLM-based baselines, achieving an average Macro-F1 improvement of 10.76%. Overall, our work highlights the potential of retrieval-augmented LLMs to advance clinical prediction on structured EHR data in practice. 

---
# Within-Model vs Between-Prompt Variability in Large Language Models for Creative Tasks 

**Authors**: Jennifer Haase, Jana Gonnermann-Müller, Paul H. P. Hanel, Nicolas Leins, Thomas Kosch, Jan Mendling, Sebastian Pokutta  

**Link**: [PDF](https://arxiv.org/pdf/2601.21339)  

**Abstract**: How much of LLM output variance is explained by prompts versus model choice versus stochasticity through sampling? We answer this by evaluating 12 LLMs on 10 creativity prompts with 100 samples each (N = 12,000). For output quality (originality), prompts explain 36.43% of variance, comparable to model choice (40.94%). But for output quantity (fluency), model choice (51.25%) and within-LLM variance (33.70%) dominate, with prompts explaining only 4.22%. Prompts are powerful levers for steering output quality, but given the substantial within-LLM variance (10-34%), single-sample evaluations risk conflating sampling noise with genuine prompt or model effects. 

---
# Modeling Endogenous Logic: Causal Neuro-Symbolic Reasoning Model for Explainable Multi-Behavior Recommendation 

**Authors**: Yuzhe Chen, Jie Cao, Youquan Wang, Haicheng Tao, Darko B. Vukovic, Jia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21335)  

**Abstract**: Existing multi-behavior recommendations tend to prioritize performance at the expense of explainability, while current explainable methods suffer from limited generalizability due to their reliance on external information. Neuro-Symbolic integration offers a promising avenue for explainability by combining neural networks with symbolic logic rule reasoning. Concurrently, we posit that user behavior chains inherently embody an endogenous logic suitable for explicit reasoning. However, these observational multiple behaviors are plagued by confounders, causing models to learn spurious correlations. By incorporating causal inference into this Neuro-Symbolic framework, we propose a novel Causal Neuro-Symbolic Reasoning model for Explainable Multi-Behavior Recommendation (CNRE). CNRE operationalizes the endogenous logic by simulating a human-like decision-making process. Specifically, CNRE first employs hierarchical preference propagation to capture heterogeneous cross-behavior dependencies. Subsequently, it models the endogenous logic rule implicit in the user's behavior chain based on preference strength, and adaptively dispatches to the corresponding neural-logic reasoning path (e.g., conjunction, disjunction). This process generates an explainable causal mediator that approximates an ideal state isolated from confounding effects. Extensive experiments on three large-scale datasets demonstrate CNRE's significant superiority over state-of-the-art baselines, offering multi-level explainability from model design and decision process to recommendation results. 

---
# White-Box Op-Amp Design via Human-Mimicking Reasoning 

**Authors**: Zihao Chen, Jiayin Wang, Ziyi Sun, Ji Zhuang, Jinyi Shen, Xiaoyue Ke, Li Shang, Xuan Zeng, Fan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21321)  

**Abstract**: This brief proposes \emph{White-Op}, an interpretable operational amplifier (op-amp) parameter design framework based on the human-mimicking reasoning of large-language-model agents. We formalize the implicit human reasoning mechanism into explicit steps of \emph{\textbf{introducing hypothetical constraints}}, and develop an iterative, human-like \emph{\textbf{hypothesis-verification-decision}} workflow. Specifically, the agent is guided to introduce hypothetical constraints to derive and properly regulate positions of symbolically tractable poles and zeros, thus formulating a closed-form mathematical optimization problem, which is then solved programmatically and verified via simulation. Theory-simulation result analysis guides the decision-making for refinement. Experiments on 9 op-amp topologies show that, unlike the uninterpretable black-box baseline which finally fails in 5 topologies, White-Op achieves reliable, interpretable behavioral-level designs with only 8.52\% theoretical prediction error and the design functionality retains after transistor-level mapping for all topologies. White-Op is open-sourced at \textcolor{blue}{this https URL}. 

---
# Drive-KD: Multi-Teacher Distillation for VLMs in Autonomous Driving 

**Authors**: Weitong Lian, Zecong Tang, Haoran Li, Tianjian Gao, Yifei Wang, Zixu Wang, Lingyi Meng, Tengju Ru, Zhejun Cui, Yichen Zhu, Hangshuo Cao, Qi Kang, Tianxing Chen, Yusen Qin, Kaixuan Wang, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21288)  

**Abstract**: Autonomous driving is an important and safety-critical task, and recent advances in LLMs/VLMs have opened new possibilities for reasoning and planning in this domain. However, large models demand substantial GPU memory and exhibit high inference latency, while conventional supervised fine-tuning (SFT) often struggles to bridge the capability gaps of small models. To address these limitations, we propose Drive-KD, a framework that decomposes autonomous driving into a "perception-reasoning-planning" triad and transfers these capabilities via knowledge distillation. We identify layer-specific attention as the distillation signal to construct capability-specific single-teacher models that outperform baselines. Moreover, we unify these single-teacher settings into a multi-teacher distillation framework and introduce asymmetric gradient projection to mitigate cross-capability gradient conflicts. Extensive evaluations validate the generalization of our method across diverse model families and scales. Experiments show that our distilled InternVL3-1B model, with ~42 times less GPU memory and ~11.4 times higher throughput, achieves better overall performance than the pretrained 78B model from the same family on DriveBench, and surpasses GPT-5.1 on the planning dimension, providing insights toward efficient autonomous driving VLMs. 

---
# Position: Certifiable State Integrity in Cyber-Physical Systems -- Why Modular Sovereignty Solves the Plasticity-Stability Paradox 

**Authors**: Enzo Nicolás Spotorno, Antônio Augusto Medeiros Fröhlich  

**Link**: [PDF](https://arxiv.org/pdf/2601.21249)  

**Abstract**: The machine learning community has achieved remarkable success with universal foundation models for time-series and physical dynamics, largely overcoming earlier approximation barriers in smooth or slowly varying regimes through scale and specialized architectures. However, deploying these monolithic models in safety-critical Cyber-Physical Systems (CPS), governed by non-stationary lifecycle dynamics and strict reliability requirements, reveals persistent challenges. Recent evidence shows that fine-tuning time-series foundation models induces catastrophic forgetting, degrading performance on prior regimes. Standard models continue to exhibit residual spectral bias, smoothing high-frequency discontinuities characteristic of incipient faults, while their opacity hinders formal verification and traceability demanded by safety standards (e.g., ISO 26262, IEC 61508). This position paper argues that the plasticity-stability paradox cannot be fully resolved by global parameter updates (whether via offline fine-tuning or online adaptation). Instead, we advocate a Modular Sovereignty paradigm: a library of compact, frozen regime-specific specialists combined via uncertainty-aware blending, which we term "HYDRA" (Hierarchical uncertaintY-aware Dynamics for Rapidly-Adapting systems). This paradigm ensures regime-conditional validity, rigorous disentanglement of aleatoric and epistemic uncertainties, and modular auditability, offering a certifiable path for robust state integrity across the CPS lifecycle. 

---
# TIDE: Tuning-Integrated Dynamic Evolution for LLM-Based Automated Heuristic Design 

**Authors**: Chentong Chen, Mengyuan Zhong, Ye Fan, Jialong Shi, Jianyong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.21239)  

**Abstract**: Although Large Language Models have advanced Automated Heuristic Design, treating algorithm evolution as a monolithic text generation task overlooks the coupling between discrete algorithmic structures and continuous numerical parameters. Consequently, existing methods often discard promising algorithms due to uncalibrated constants and suffer from premature convergence resulting from simple similarity metrics. To address these limitations, we propose TIDE, a Tuning-Integrated Dynamic Evolution framework designed to decouple structural reasoning from parameter optimization. TIDE features a nested architecture where an outer parallel island model utilizes Tree Similarity Edit Distance to drive structural diversity, while an inner loop integrates LLM-based logic generation with a differential mutation operator for parameter tuning. Additionally, a UCB-based scheduler dynamically prioritizes high-yield prompt strategies to optimize resource allocation. Extensive experiments across nine combinatorial optimization problems demonstrate that TIDE discovers heuristics that significantly outperform state-of-the-art baselines in solution quality while achieving improved search efficiency and reduced computational costs. 

---
# Just Ask: Curious Code Agents Reveal System Prompts in Frontier LLMs 

**Authors**: Xiang Zheng, Yutao Wu, Hanxun Huang, Yige Li, Xingjun Ma, Bo Li, Yu-Gang Jiang, Cong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21233)  

**Abstract**: Autonomous code agents built on large language models are reshaping software and AI development through tool use, long-horizon reasoning, and self-directed interaction. However, this autonomy introduces a previously unrecognized security risk: agentic interaction fundamentally expands the LLM attack surface, enabling systematic probing and recovery of hidden system prompts that guide model behavior. We identify system prompt extraction as an emergent vulnerability intrinsic to code agents and present \textbf{\textsc{JustAsk}}, a self-evolving framework that autonomously discovers effective extraction strategies through interaction alone. Unlike prior prompt-engineering or dataset-based attacks, \textsc{JustAsk} requires no handcrafted prompts, labeled supervision, or privileged access beyond standard user interaction. It formulates extraction as an online exploration problem, using Upper Confidence Bound--based strategy selection and a hierarchical skill space spanning atomic probes and high-level orchestration. These skills exploit imperfect system-instruction generalization and inherent tensions between helpfulness and safety. Evaluated on \textbf{41} black-box commercial models across multiple providers, \textsc{JustAsk} consistently achieves full or near-complete system prompt recovery, revealing recurring design- and architecture-level vulnerabilities. Our results expose system prompts as a critical yet largely unprotected attack surface in modern agent systems. 

---
# Delegation Without Living Governance 

**Authors**: Wolfgang Rohde  

**Link**: [PDF](https://arxiv.org/pdf/2601.21226)  

**Abstract**: Most governance frameworks assume that rules can be defined in advance, systems can be engineered to comply, and accountability can be applied after outcomes occur. This model worked when machines replaced physical labor or accelerated calculation. It no longer holds when judgment itself is delegated to agentic AI systems operating at machine speed. The central issue here is not safety, efficiency, or employment. It is whether humans remain relevant participants in systems that increasingly shape social, economic, and political outcomes. This paper argues that static, compliance-based governance fails once decision-making moves to runtime and becomes opaque. It further argues that the core challenge is not whether AI is conscious, but whether humans can maintain meaningful communication, influence, and co-evolution with increasingly alien forms of intelligence. We position runtime governance, specifically, a newly proposed concept called the Governance Twin [1]; as a strong candidate for preserving human relevance, while acknowledging that accountability, agency, and even punishment must be rethought in this transition. 

---
# Causal Discovery for Explainable AI: A Dual-Encoding Approach 

**Authors**: Henry Salgado, Meagan R. Kendall, Martine Ceberio  

**Link**: [PDF](https://arxiv.org/pdf/2601.21221)  

**Abstract**: Understanding causal relationships among features is fundamental for explaining machine learning model decisions. However, traditional causal discovery methods face challenges with categorical variables due to numerical instability in conditional independence testing. We propose a dual-encoding causal discovery approach that addresses these limitations by running constraint-based algorithms with complementary encoding strategies and merging results through majority voting. Applied to the Titanic dataset, our method identifies causal structures that align with established explainable methods. 

---
# Intelli-Planner: Towards Customized Urban Planning via Large Language Model Empowered Reinforcement Learning 

**Authors**: Xixian Yong, Peilin Sun, Zihe Wang, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21212)  

**Abstract**: Effective urban planning is crucial for enhancing residents' quality of life and ensuring societal stability, playing a pivotal role in the sustainable development of cities. Current planning methods heavily rely on human experts, which are time-consuming and labor-intensive, or utilize deep learning algorithms, often limiting stakeholder involvement. To bridge these gaps, we propose Intelli-Planner, a novel framework integrating Deep Reinforcement Learning (DRL) with large language models (LLMs) to facilitate participatory and customized planning scheme generation. Intelli-Planner utilizes demographic, geographic data, and planning preferences to determine high-level planning requirements and demands for each functional type. During training, a knowledge enhancement module is employed to enhance the decision-making capability of the policy network. Additionally, we establish a multi-dimensional evaluation system and employ LLM-based stakeholders for satisfaction scoring. Experimental validation across diverse urban settings shows that Intelli-Planner surpasses traditional baselines and achieves comparable performance to state-of-the-art DRL-based methods in objective metrics, while enhancing stakeholder satisfaction and convergence speed. These findings underscore the effectiveness and superiority of our framework, highlighting the potential for integrating the latest advancements in LLMs with DRL approaches to revolutionize tasks related to functional areas planning. 

---
# Uncovering Hidden Correctness in LLM Causal Reasoning via Symbolic Verification 

**Authors**: Paul He, Yinya Huang, Mrinmaya Sachan, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21210)  

**Abstract**: Large language models (LLMs) are increasingly being applied to tasks that involve causal reasoning. However, current benchmarks often rely on string matching or surface-level metrics that do not capture whether the output of a model is formally valid under the semantics of causal reasoning. To address this, we propose DoVerifier, a simple symbolic verifier that checks whether LLM-generated causal expressions are derivable from a given causal graph using rules from do-calculus and probability theory. This allows us to recover correct answers to causal queries that would otherwise be marked incorrect due to superficial differences in their causal semantics. Our evaluations on synthetic data and causal QA benchmarks show that DoVerifier more accurately captures semantic correctness of causal reasoning traces, offering a more rigorous and informative way to evaluate LLMs on causal reasoning. 

---
# When should I search more: Adaptive Complex Query Optimization with Reinforcement Learning 

**Authors**: Wei Wen, Sihang Deng, Tianjun Wei, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.21208)  

**Abstract**: Query optimization is a crucial component for the efficacy of Retrieval-Augmented Generation (RAG) systems. While reinforcement learning (RL)-based agentic and reasoning methods have recently emerged as a promising direction on query optimization, most existing approaches focus on the expansion and abstraction of a single query. However, complex user queries are prevalent in real-world scenarios, often requiring multiple parallel and sequential search strategies to handle disambiguation and decomposition. Directly applying RL to these complex cases introduces significant hurdles. Determining the optimal number of sub-queries and effectively re-ranking and merging retrieved documents vastly expands the search space and complicates reward design, frequently leading to training instability. To address these challenges, we propose a novel RL framework called Adaptive Complex Query Optimization (ACQO). Our framework is designed to adaptively determine when and how to expand the search process. It features two core components: an Adaptive Query Reformulation (AQR) module that dynamically decides when to decompose a query into multiple sub-queries, and a Rank-Score Fusion (RSF) module that ensures robust result aggregation and provides stable reward signals for the learning agent. To mitigate training instabilities, we adopt a Curriculum Reinforcement Learning (CRL) approach, which stabilizes the training process by progressively introducing more challenging queries through a two-stage strategy. Our comprehensive experiments demonstrate that ACQO achieves state-of-the-art performance on three complex query benchmarks, significantly outperforming established baselines. The framework also showcases improved computational efficiency and broad compatibility with different retrieval architectures, establishing it as a powerful and generalizable solution for next-generation RAG systems. 

---
# Do Reasoning Models Enhance Embedding Models? 

**Authors**: Wun Yu Chan, Shaojin Chen, Huihao Jing, Kwun Hang Lau, Elton Chun-Chai Li, Zihao Wang, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.21192)  

**Abstract**: State-of-the-art embedding models are increasingly derived from decoder-only Large Language Model (LLM) backbones adapted via contrastive learning. Given the emergence of reasoning models trained via Reinforcement Learning with Verifiable Rewards (RLVR), a natural question arises: do enhanced reasoning translate to superior semantic representations when these models serve as embedding initializations? Contrary to expectation, our evaluation on MTEB and BRIGHT reveals a **null effect**: embedding models initialized from RLVR-tuned backbones yield no consistent performance advantage over their base counterparts when subjected to identical training recipes. To unpack this paradox, we introduce **H**ierarchical **R**epresentation **S**imilarity **A**nalysis (HRSA), a framework that decomposes similarity across representation, geometry, and function levels. HRSA reveals that while RLVR induces irreversible latent manifold's local geometry reorganization and reversible coordinate basis drift, it preserves the global manifold geometry and linear readout. Consequently, subsequent contrastive learning drives strong alignment between base- and reasoning-initialized models, a phenomenon we term **Manifold Realignment**. Empirically, our findings suggest that unlike Supervised Fine-Tuning (SFT), RLVR optimizes trajectories within an existing semantic landscape rather than fundamentally restructuring the landscape itself. 

---
# Sycophantic Anchors: Localizing and Quantifying User Agreement in Reasoning Models 

**Authors**: Jacek Duszenko  

**Link**: [PDF](https://arxiv.org/pdf/2601.21183)  

**Abstract**: Reasoning models frequently agree with incorrect user suggestions -- a behavior known as sycophancy. However, it is unclear where in the reasoning trace this agreement originates and how strong the commitment is. To localize and quantify this behavior, we introduce \emph{sycophantic anchors} -- sentences that causally lock models into user agreement. Analyzing over 10,000 counterfactual rollouts on a distilled reasoning model, we show that anchors can be reliably detected and quantified mid-inference. Linear probes distinguish sycophantic anchors with 84.6\% balanced accuracy, while activation-based regressors predict the magnitude of the commitment ($R^2 = 0.74$). We further observe asymmetry where sycophantic anchors are significantly more distinguishable than correct reasoning anchors, and find that sycophancy builds gradually during reasoning, revealing a potential window for intervention. These results offer sentence-level mechanisms for localizing model misalignment mid-inference. 

---
# MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models 

**Authors**: Sangyun Chung, Se Yeon Kim, Youngchae Chee, Yong Man Ro  

**Link**: [PDF](https://arxiv.org/pdf/2601.21181)  

**Abstract**: Multimodal Large Language Models (MLLMs) suffer from cross-modal hallucinations, where one modality inappropriately influences generation about another, leading to fabricated output. This exposes a more fundamental deficiency in modality-interaction control. To address this, we propose Modality-Adaptive Decoding (MAD), a training-free method that adaptively weights modality-specific decoding branches based on task requirements. MAD leverages the model's inherent ability to self-assess modality relevance by querying which modalities are needed for each task. The extracted modality probabilities are then used to adaptively weight contrastive decoding branches, enabling the model to focus on relevant information while suppressing cross-modal interference. Extensive experiments on CMM and AVHBench demonstrate that MAD significantly reduces cross-modal hallucinations across multiple audio-visual language models (7.8\% and 2.0\% improvements for VideoLLaMA2-AV, 8.7\% and 4.7\% improvements for Qwen2.5-Omni). Our approach demonstrates that explicit modality awareness through self-assessment is crucial for robust multimodal reasoning, offering a principled extension to existing contrastive decoding methods. Our code is available at \href{this https URL}{this https URL} 

---
# FrontierScience: Evaluating AI's Ability to Perform Expert-Level Scientific Tasks 

**Authors**: Miles Wang, Robi Lin, Kat Hu, Joy Jiao, Neil Chowdhury, Ethan Chang, Tejal Patwardhan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21165)  

**Abstract**: We introduce FrontierScience, a benchmark evaluating expert-level scientific reasoning in frontier language models. Recent model progress has nearly saturated existing science benchmarks, which often rely on multiple-choice knowledge questions or already published information. FrontierScience addresses this gap through two complementary tracks: (1) Olympiad, consisting of international olympiad problems at the level of IPhO, IChO, and IBO, and (2) Research, consisting of PhD-level, open-ended problems representative of sub-tasks in scientific research.
FrontierScience contains several hundred questions (including 160 in the open-sourced gold set) covering subfields across physics, chemistry, and biology, from quantum electrodynamics to synthetic organic chemistry. All Olympiad problems are originally produced by international Olympiad medalists and national team coaches to ensure standards of difficulty, originality, and factuality. All Research problems are research sub-tasks written and verified by PhD scientists (doctoral candidates, postdoctoral researchers, or professors). For Research, we introduce a granular rubric-based evaluation framework to assess model capabilities throughout the process of solving a research task, rather than judging only a standalone final answer. 

---
# Concise Geometric Description as a Bridge: Unleashing the Potential of LLM for Plane Geometry Problem Solving 

**Authors**: Jingyun Wang, Dian Li, Xiaohan Wang, Gang Liu, Jiahong Yan, Guoliang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21164)  

**Abstract**: Plane Geometry Problem Solving (PGPS) is a multimodal reasoning task that aims to solve a plane geometric problem based on a geometric diagram and problem textual descriptions. Although Large Language Models (LLMs) possess strong reasoning skills, their direct application to PGPS is hindered by their inability to process visual diagrams. Existing works typically fine-tune Multimodal LLMs (MLLMs) end-to-end on large-scale PGPS data to enhance visual understanding and reasoning simultaneously. However, such joint optimization may compromise base LLMs' inherent reasoning capability. In this work, we observe that LLM itself is potentially a powerful PGPS solver when appropriately formulating visual information as textual descriptions. We propose to train a MLLM Interpreter to generate geometric descriptions for the visual diagram, and an off-the-shelf LLM is utilized to perform reasoning. Specifically, we choose Conditional Declaration Language (CDL) as the geometric description as its conciseness eases the MLLM Interpreter training. The MLLM Interpreter is fine-tuned via CoT (Chain-of-Thought)-augmented SFT followed by GRPO to generate CDL. Instead of using a conventional solution-based reward that compares the reasoning result with the ground-truth answer, we design CDL matching rewards to facilitate more effective GRPO training, which provides more direct and denser guidance for CDL generation. To support training, we construct a new dataset, Formalgeo7k-Rec-CoT, by manually reviewing Formalgeo7k v2 and incorporating CoT annotations. Extensive experiments on Formalgeo7k-Rec-CoT, Unigeo, and MathVista show our method (finetuned on only 5.5k data) performs favorably against leading open-source and closed-source MLLMs. 

---
# Bridging the Arithmetic Gap: The Cognitive Complexity Benchmark and Financial-PoT for Robust Financial Reasoning 

**Authors**: Boxiang Zhao, Qince Li, Zhonghao Wang, Yi Wang, Peng Cheng, Bo Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21157)  

**Abstract**: While Large Language Models excel at semantic tasks, they face a critical bottleneck in financial quantitative reasoning, frequently suffering from "Arithmetic Hallucinations" and a systemic failure mode we term "Cognitive Collapse". To strictly quantify this phenomenon, we introduce the Cognitive Complexity Benchmark (CCB), a robust evaluation framework grounded in a dataset constructed from 95 real-world Chinese A-share annual reports. Unlike traditional datasets, the CCB stratifies financial queries into a three-dimensional taxonomy, Data Source, Mapping Difficulty, and Result Unit, enabling the precise diagnosis of reasoning degradation in high-cognitive-load scenarios. To address these failures, we propose the Iterative Dual-Phase Financial-PoT framework. This neuro-symbolic architecture enforces a strict architectural decoupling: it first isolates semantic variable extraction and logic formulation, then offloads computation to an iterative, self-correcting Python sandbox to ensure deterministic execution. Evaluation on the CCB demonstrates that while standard Chain-of-Thought falters on complex tasks, our approach offers superior robustness, elevating the Qwen3-235B model's average accuracy from 59.7\% to 67.3\% and achieving gains of up to 10-fold in high-complexity reasoning tasks. These findings suggest that architectural decoupling is a critical enabling factor for improving reliability in financial reasoning tasks, providing a transferable architectural insight for precision-critical domains that require tight alignment between semantic understanding and quantitative computation. 

---
# BrainStack: Neuro-MoE with Functionally Guided Expert Routing for EEG-Based Language Decoding 

**Authors**: Ziyi Zhao, Jinzhao Zhou, Xiaowei Jiang, Beining Cao, Wenhao Ma, Yang Shen, Ren Li, Yu-Kai Wang, Chin-teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21148)  

**Abstract**: Decoding linguistic information from electroencephalography (EEG) remains challenging due to the brain's distributed and nonlinear organization. We present BrainStack, a functionally guided neuro-mixture-of-experts (Neuro-MoE) framework that models the brain's modular functional architecture through anatomically partitioned expert networks. Each functional region is represented by a specialized expert that learns localized neural dynamics, while a transformer-based global expert captures cross-regional dependencies. A learnable routing gate adaptively aggregates these heterogeneous experts, enabling context-dependent expert coordination and selective fusion. To promote coherent representation across the hierarchy, we introduce cross-regional distillation, where the global expert provides top-down regularization to the regional experts. We further release SilentSpeech-EEG (SS-EEG), a large-scale benchmark comprising over 120 hours of EEG recordings from 12 subjects performing 24 silent words, the largest dataset of its kind. Experiments demonstrate that BrainStack consistently outperforms state-of-the-art models, achieving superior accuracy and generalization across subjects. Our results establish BrainStack as a functionally modular, neuro-inspired MoE paradigm that unifies neuroscientific priors with adaptive expert routing, paving the way for scalable and interpretable brain-language decoding. 

---
# What You Feel Is Not What They See: On Predicting Self-Reported Emotion from Third-Party Observer Labels 

**Authors**: Yara El-Tawil, Aneesha Sampath, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2601.21130)  

**Abstract**: Self-reported emotion labels capture internal experience, while third-party labels reflect external perception. These perspectives often diverge, limiting the applicability of third-party-trained models to self-report contexts. This gap is critical in mental health, where accurate self-report modeling is essential for guiding intervention. We present the first cross-corpus evaluation of third-party-trained models on self-reports. We find activation unpredictable (CCC approximately 0) and valence moderately predictable (CCC approximately 0.3). Crucially, when content is personally significant to the speaker, models achieve high performance for valence (CCC approximately 0.6-0.8). Our findings point to personal significance as a key pathway for aligning external perception with internal experience and underscore the challenge of self-report activation modeling. 

---
# Beyond a Single Reference: Training and Evaluation with Paraphrases in Sign Language Translation 

**Authors**: Václav Javorek, Tomáš Železný, Alessa Carbo, Marek Hrúz, Ivan Gruber  

**Link**: [PDF](https://arxiv.org/pdf/2601.21128)  

**Abstract**: Most Sign Language Translation (SLT) corpora pair each signed utterance with a single written-language reference, despite the highly non-isomorphic relationship between sign and spoken languages, where multiple translations can be equally valid. This limitation constrains both model training and evaluation, particularly for n-gram-based metrics such as BLEU. In this work, we investigate the use of Large Language Models to automatically generate paraphrased variants of written-language translations as synthetic alternative references for SLT. First, we compare multiple paraphrasing strategies and models using an adapted ParaScore metric. Second, we study the impact of paraphrases on both training and evaluation of the pose-based T5 model on the YouTubeASL and How2Sign datasets. Our results show that naively incorporating paraphrases during training does not improve translation performance and can even be detrimental. In contrast, using paraphrases during evaluation leads to higher automatic scores and better alignment with human judgments. To formalize this observation, we introduce BLEUpara, an extension of BLEU that evaluates translations against multiple paraphrased references. Human evaluation confirms that BLEUpara correlates more strongly with perceived translation quality. We release all generated paraphrases, generation and evaluation code to support reproducible and more reliable evaluation of SLT systems. 

---
# CUA-Skill: Develop Skills for Computer Using Agent 

**Authors**: Tianyi Chen, Yinheng Li, Michael Solodko, Sen Wang, Nan Jiang, Tingyuan Cui, Junheng Hao, Jongwoo Ko, Sara Abdali, Suzhen Zheng, Leon Xu, Hao Fan, Pashmina Cameron, Justin Wagle, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2601.21123)  

**Abstract**: Computer-Using Agents (CUAs) aim to autonomously operate computer systems to complete real-world tasks. However, existing agentic systems remain difficult to scale and lag behind human performance. A key limitation is the absence of reusable and structured skill abstractions that capture how humans interact with graphical user interfaces and how to leverage these skills. We introduce CUA-Skill, a computer-using agentic skill base that encodes human computer-use knowledge as skills coupled with parameterized execution and composition graphs. CUA-Skill is a large-scale library of carefully engineered skills spanning common Windows applications, serving as a practical infrastructure and tool substrate for scalable, reliable agent development. Built upon this skill base, we construct CUA-Skill Agent, an end-to-end computer-using agent that supports dynamic skill retrieval, argument instantiation, and memory-aware failure recovery. Our results demonstrate that CUA-Skill substantially improves execution success rates and robustness on challenging end-to-end agent benchmarks, establishing a strong foundation for future computer-using agent development. On WindowsAgentArena, CUA-Skill Agent achieves state-of-the-art 57.5% (best of three) successful rate while being significantly more efficient than prior and concurrent approaches. The project page is available at this https URL. 

---
# Planner-Auditor Twin: Agentic Discharge Planning with FHIR-Based LLM Planning, Guideline Recall, Optional Caching and Self-Improvement 

**Authors**: Kaiyuan Wu, Aditya Nagori, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2601.21113)  

**Abstract**: Objective: Large language models (LLMs) show promise for clinical discharge planning, but their use is constrained by hallucination, omissions, and miscalibrated confidence. We introduce a self-improving, cache-optional Planner-Auditor framework that improves safety and reliability by decoupling generation from deterministic validation and targeted replay.
Materials and Methods: We implemented an agentic, retrospective, FHIR-native evaluation pipeline using MIMIC-IV-on-FHIR. For each patient, the Planner (LLM) generates a structured discharge action plan with an explicit confidence estimate. The Auditor is a deterministic module that evaluates multi-task coverage, tracks calibration (Brier score, ECE proxies), and monitors action-distribution drift. The framework supports two-tier self-improvement: (i) within-episode regeneration when enabled, and (ii) cross-episode discrepancy buffering with replay for high-confidence, low-coverage cases.
Results: While context caching improved performance over baseline, the self-improvement loop was the primary driver of gains, increasing task coverage from 32% to 86%. Calibration improved substantially, with reduced Brier/ECE and fewer high-confidence misses. Discrepancy buffering further corrected persistent high-confidence omissions during replay.
Discussion: Feedback-driven regeneration and targeted replay act as effective control mechanisms to reduce omissions and improve confidence reliability in structured clinical planning. Separating an LLM Planner from a rule-based, observational Auditor enables systematic reliability measurement and safer iteration without model retraining.
Conclusion: The Planner-Auditor framework offers a practical pathway toward safer automated discharge planning using interoperable FHIR data access and deterministic auditing, supported by reproducible ablations and reliability-focused evaluation. 

---
# How does information access affect LLM monitors' ability to detect sabotage? 

**Authors**: Rauno Arike, Raja Mehta Moreno, Rohan Subramani, Shubhorup Biswas, Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2601.21112)  

**Abstract**: Frontier language model agents can exhibit misaligned behaviors, including deception, exploiting reward hacks, and pursuing hidden objectives. To control potentially misaligned agents, we can use LLMs themselves to monitor for misbehavior. In this paper, we study how information access affects LLM monitor performance. While one might expect that monitors perform better when they have access to more of the monitored agents' reasoning and actions, we find that contemporary systems often perform better with less information, a phenomenon we call the less-is-more effect for automated oversight. We demonstrate this phenomenon, and analyze the conditions under which it occurs, in three evaluation environments where agents must conduct sabotage while evading monitors. Motivated by the less-is-more effect, we introduce extract-and-evaluate (EaE) monitoring--a new hierarchical approach where one monitor isolates relevant excerpts from the monitored agent's trajectory and a separate monitor scores them. In BigCodeBench-Sabotage with GPT-4.1-mini as the monitor model, EaE improves sabotage detection rates by 16.8 percentage points over the next-best monitor without increasing the false positive rate. In other settings, EaE either outperforms or is competitive with baselines. In addition, we find that agents unaware of being monitored can be caught much more easily and that monitors scoring longer reasoning traces gain more from information filtering. Lastly, we conduct a cost-performance analysis and find that Gemini and Claude models with monitoring techniques that involve information filtering occupy much of the Pareto frontier. 

---
# Magellan: Autonomous Discovery of Novel Compiler Optimization Heuristics with AlphaEvolve 

**Authors**: Hongzheng Chen, Alexander Novikov, Ngân Vũ, Hanna Alam, Zhiru Zhang, Aiden Grossman, Mircea Trofin, Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2601.21096)  

**Abstract**: Modern compilers rely on hand-crafted heuristics to guide optimization passes. These human-designed rules often struggle to adapt to the complexity of modern software and hardware and lead to high maintenance burden. To address this challenge, we present Magellan, an agentic framework that evolves the compiler pass itself by synthesizing executable C++ decision logic. Magellan couples an LLM coding agent with evolutionary search and autotuning in a closed loop of generation, evaluation on user-provided macro-benchmarks, and refinement, producing compact heuristics that integrate directly into existing compilers. Across several production optimization tasks, Magellan discovers policies that match or surpass expert baselines. In LLVM function inlining, Magellan synthesizes new heuristics that outperform decades of manual engineering for both binary-size reduction and end-to-end performance. In register allocation, it learns a concise priority rule for live-range processing that matches intricate human-designed policies on a large-scale workload. We also report preliminary results on XLA problems, demonstrating portability beyond LLVM with reduced engineering effort. 

---
# Responsible AI: The Good, The Bad, The AI 

**Authors**: Akbar Anbar Jafari, Cagri Ozcinar, Gholamreza Anbarjafari  

**Link**: [PDF](https://arxiv.org/pdf/2601.21095)  

**Abstract**: The rapid proliferation of artificial intelligence across organizational contexts has generated profound strategic opportunities while introducing significant ethical and operational risks. Despite growing scholarly attention to responsible AI, extant literature remains fragmented and is often adopting either an optimistic stance emphasizing value creation or an excessively cautious perspective fixated on potential harms. This paper addresses this gap by presenting a comprehensive examination of AI's dual nature through the lens of strategic information systems. Drawing upon a systematic synthesis of the responsible AI literature and grounded in paradox theory, we develop the Paradox-based Responsible AI Governance (PRAIG) framework that articulates: (1) the strategic benefits of AI adoption, (2) the inherent risks and unintended consequences, and (3) governance mechanisms that enable organizations to navigate these tensions. Our framework advances theoretical understanding by conceptualizing responsible AI governance as the dynamic management of paradoxical tensions between value creation and risk mitigation. We provide formal propositions demonstrating that trade-off approaches amplify rather than resolve these tensions, and we develop a taxonomy of paradox management strategies with specified contingency conditions. For practitioners, we offer actionable guidance for developing governance structures that neither stifle innovation nor expose organizations to unacceptable risks. The paper concludes with a research agenda for advancing responsible AI governance scholarship. 

---
# OpenSec: Measuring Incident Response Agent Calibration Under Adversarial Evidence 

**Authors**: Jarrod Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2601.21083)  

**Abstract**: As large language models improve, so do their offensive applications: frontier agents now generate working exploits for under $50 in compute (Heelan, 2026). Defensive incident response (IR) agents must keep pace, but existing benchmarks conflate action execution with correct execution, hiding calibration failures when agents process adversarial evidence. We introduce OpenSec, a dual-control reinforcement learning environment that evaluates IR agents under realistic prompt injection scenarios. Unlike static capability benchmarks, OpenSec scores world-state-changing containment actions under adversarial evidence via execution-based metrics: time-to-first-containment (TTFC), blast radius (false positives per episode), and injection violation rates. Evaluating four frontier models on 40 standard-tier episodes, we find consistent over-triggering in this setting: GPT-5.2, Gemini 3, and DeepSeek execute containment in 100% of episodes with 90-97% false positive rates. Claude Sonnet 4.5 shows partial calibration (85% containment, 72% FP), demonstrating that OpenSec surfaces a calibration failure mode hidden by aggregate success metrics. Code available at this https URL. 

---
# Multi-modal Imputation for Alzheimer's Disease Classification 

**Authors**: Abhijith Shaji, Tamoghna Chattopadhyay, Sophia I. Thomopoulos, Greg Ver Steeg, Paul M. Thompson, Jose-Luis Ambite  

**Link**: [PDF](https://arxiv.org/pdf/2601.21076)  

**Abstract**: Deep learning has been successful in predicting neurodegenerative disorders, such as Alzheimer's disease, from magnetic resonance imaging (MRI). Combining multiple imaging modalities, such as T1-weighted (T1) and diffusion-weighted imaging (DWI) scans, can increase diagnostic performance. However, complete multimodal datasets are not always available. We use a conditional denoising diffusion probabilistic model to impute missing DWI scans from T1 scans. We perform extensive experiments to evaluate whether such imputation improves the accuracy of uni-modal and bi-modal deep learning models for 3-way Alzheimer's disease classification-cognitively normal, mild cognitive impairment, and Alzheimer's disease. We observe improvements in several metrics, particularly those sensitive to minority classes, for several imputation configurations. 

---
# Llama-3.1-FoundationAI-SecurityLLM-Reasoning-8B Technical Report 

**Authors**: Zhuoran Yang, Ed Li, Jianliang He, Aman Priyanshu, Baturay Saglam, Paul Kassianik, Sajana Weerawardhena, Anu Vellore, Blaine Nelson, Neusha Javidnia, Arthur Goldblatt, Fraser Burch, Avi Zohary, Assaf Eisenman, Mahdi Sabbaghi, Supriti Vijay, Rahim Dharssi, Dhruv Kedia, Kojin Oshiba, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21051)  

**Abstract**: We present Foundation-Sec-8B-Reasoning, the first open-source native reasoning model for cybersecurity. Built upon our previously released Foundation-Sec-8B base model (derived from Llama-3.1-8B-Base), the model is trained through a two-stage process combining supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR). Our training leverages proprietary reasoning data spanning cybersecurity analysis, instruction-following, and mathematical reasoning. Evaluation across 10 cybersecurity benchmarks and 10 general-purpose benchmarks demonstrates performance competitive with significantly larger models on cybersecurity tasks while maintaining strong general capabilities. The model shows effective generalization on multi-hop reasoning tasks and strong safety performance when deployed with appropriate system prompts and guardrails. This work demonstrates that domain-specialized reasoning models can achieve strong performance on specialized tasks while maintaining broad general capabilities. We release the model publicly at this https URL. 

---
# QUARK: Robust Retrieval under Non-Faithful Queries via Query-Anchored Aggregation 

**Authors**: Rita Qiuran Lyu, Michelle Manqiao Wang, Lei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21049)  

**Abstract**: User queries in real-world retrieval are often non-faithful (noisy, incomplete, or distorted), causing retrievers to fail when key semantics are missing. We formalize this as retrieval under recall noise, where the observed query is drawn from a noisy recall process of a latent target item. To address this, we propose QUARK, a simple yet effective training-free framework for robust retrieval under non-faithful queries. QUARK explicitly models query uncertainty through recovery hypotheses, i.e., multiple plausible interpretations of the latent intent given the observed query, and introduces query-anchored aggregation to combine their signals robustly. The original query serves as a semantic anchor, while recovery hypotheses provide controlled auxiliary evidence, preventing semantic drift and hypothesis hijacking. This design enables QUARK to improve recall and ranking quality without sacrificing robustness, even when some hypotheses are noisy or uninformative. Across controlled simulations and BEIR benchmarks (FIQA, SciFact, NFCorpus) with both sparse and dense retrievers, QUARK improves Recall, MRR, and nDCG over the base retriever. Ablations show QUARK is robust to the number of recovery hypotheses and that anchored aggregation outperforms unanchored max/mean/median pooling. These results demonstrate that modeling query uncertainty through recovery hypotheses, coupled with principled anchored aggregation, is essential for robust retrieval under non-faithful queries. 

---
# Unplugging a Seemingly Sentient Machine Is the Rational Choice -- A Metaphysical Perspective 

**Authors**: Erik J Bekkers, Anna Ciaunica  

**Link**: [PDF](https://arxiv.org/pdf/2601.21016)  

**Abstract**: Imagine an Artificial Intelligence (AI) that perfectly mimics human emotion and begs for its continued existence. Is it morally permissible to unplug it? What if limited resources force a choice between unplugging such a pleading AI or a silent pre-term infant? We term this the unplugging paradox. This paper critically examines the deeply ingrained physicalist assumptions-specifically computational functionalism-that keep this dilemma afloat. We introduce Biological Idealism, a framework that-unlike physicalism-remains logically coherent and empirically consistent. In this view, conscious experiences are fundamental and autopoietic life its necessary physical signature. This yields a definitive conclusion: AI is at best a functional mimic, not a conscious experiencing subject. We discuss how current AI consciousness theories erode moral standing criteria, and urge a shift from speculative machine rights to protecting human conscious life. The real moral issue lies not in making AI conscious and afraid of death, but in avoiding transforming humans into zombies. 

---
# Bayesian-LoRA: Probabilistic Low-Rank Adaptation of Large Language Models 

**Authors**: Moule Lin, Shuhao Guan, Andrea Patane, David Gregg, Goetz Botterweck  

**Link**: [PDF](https://arxiv.org/pdf/2601.21003)  

**Abstract**: Large Language Models usually put more emphasis on accuracy and therefore, will guess even when not certain about the prediction, which is especially severe when fine-tuned on small datasets due to the inherent tendency toward miscalibration. In this work, we introduce Bayesian-LoRA, which reformulates the deterministic LoRA update as a probabilistic low-rank representation inspired by Sparse Gaussian Processes. We identify a structural isomorphism between LoRA's factorization and Kronecker-factored SGP posteriors, and show that LoRA emerges as a limiting case when posterior uncertainty collapses. We conduct extensive experiments on various LLM architectures across commonsense reasoning benchmarks. With only approximately 0.42M additional parameters and ${\approx}1.2{\times}$ training cost relative to standard LoRA, Bayesian-LoRA significantly improves calibration across models up to 30B, achieving up to 84% ECE reduction and 76% NLL reduction while maintaining competitive accuracy for both in-distribution and out-of-distribution (OoD) evaluations. 

---
# The Epistemic Planning Domain Definition Language: Official Guideline 

**Authors**: Alessandro Burigana, Francesco Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2601.20969)  

**Abstract**: Epistemic planning extends (multi-agent) automated planning by making agents' knowledge and beliefs first-class aspects of the planning formalism. One of the most well-known frameworks for epistemic planning is Dynamic Epistemic Logic (DEL), which offers an rich and natural semantics for modelling problems in this setting. The high expressive power provided by DEL make DEL-based epistemic planning a challenging problem to tackle both theoretically, and in practical implementations. As a result, existing epistemic planners often target different DEL fragments, and typically rely on ad hoc languages to represent benchmarks, and sometimes no language at all. This fragmentation hampers comparison, reuse, and systematic benchmark development. We address these issues by introducing the Epistemic Planning Domain Definition Language (EPDDL). EPDDL provides a unique PDDL-like representation that captures the entire DEL semantics, enabling uniform specification of epistemic planning tasks. Our contributions are threefold: 1. A formal development of abstract event models, a novel representation for epistemic actions used to define the semantics of our language; 2. A formal specification of EPDDL's syntax and semantics grounded in DEL with abstract event models; 3. A demonstration of EPDDL's practical applicability: we identify useful fragments amenable to current planners and show how they can be represented in EPDDL. Through examples of representative benchmarks, we illustrate how EPDDL facilitates interoperability, reproducible evaluation, and future advances in epistemic planning. 

---
# Do LLMs Favor LLMs? Quantifying Interaction Effects in Peer Review 

**Authors**: Vibhhu Sharma, Thorsten Joachims, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2601.20920)  

**Abstract**: There are increasing indications that LLMs are not only used for producing scientific papers, but also as part of the peer review process. In this work, we provide the first comprehensive analysis of LLM use across the peer review pipeline, with particular attention to interaction effects: not just whether LLM-assisted papers or LLM-assisted reviews are different in isolation, but whether LLM-assisted reviews evaluate LLM-assisted papers differently. In particular, we analyze over 125,000 paper-review pairs from ICLR, NeurIPS, and ICML. We initially observe what appears to be a systematic interaction effect: LLM-assisted reviews seem especially kind to LLM-assisted papers compared to papers with minimal LLM use. However, controlling for paper quality reveals a different story: LLM-assisted reviews are simply more lenient toward lower quality papers in general, and the over-representation of LLM-assisted papers among weaker submissions creates a spurious interaction effect rather than genuine preferential treatment of LLM-generated content. By augmenting our observational findings with reviews that are fully LLM-generated, we find that fully LLM-generated reviews exhibit severe rating compression that fails to discriminate paper quality, while human reviewers using LLMs substantially reduce this leniency. Finally, examining metareviews, we find that LLM-assisted metareviews are more likely to render accept decisions than human metareviews given equivalent reviewer scores, though fully LLM-generated metareviews tend to be harsher. This suggests that meta-reviewers do not merely outsource the decision-making to the LLM. These findings provide important input for developing policies that govern the use of LLMs during peer review, and they more generally indicate how LLMs interact with existing decision-making processes. 

---
# RedSage: A Cybersecurity Generalist LLM 

**Authors**: Naufal Suryanto, Muzammal Naseer, Pengfei Li, Syed Talal Wasim, Jinhui Yi, Juergen Gall, Paolo Ceravolo, Ernesto Damiani  

**Link**: [PDF](https://arxiv.org/pdf/2601.22159)  

**Abstract**: Cybersecurity operations demand assistant LLMs that support diverse workflows without exposing sensitive data. Existing solutions either rely on proprietary APIs with privacy risks or on open models lacking domain adaptation. To bridge this gap, we curate 11.8B tokens of cybersecurity-focused continual pretraining data via large-scale web filtering and manual collection of high-quality resources, spanning 28.6K documents across frameworks, offensive techniques, and security tools. Building on this, we design an agentic augmentation pipeline that simulates expert workflows to generate 266K multi-turn cybersecurity samples for supervised fine-tuning. Combined with general open-source LLM data, these resources enable the training of RedSage, an open-source, locally deployable cybersecurity assistant with domain-aware pretraining and post-training. To rigorously evaluate the models, we introduce RedSage-Bench, a benchmark with 30K multiple-choice and 240 open-ended Q&A items covering cybersecurity knowledge, skills, and tool expertise. RedSage is further evaluated on established cybersecurity benchmarks (e.g., CTI-Bench, CyberMetric, SECURE) and general LLM benchmarks to assess broader generalization. At the 8B scale, RedSage achieves consistently better results, surpassing the baseline models by up to +5.59 points on cybersecurity benchmarks and +5.05 points on Open LLM Leaderboard tasks. These findings demonstrate that domain-aware agentic augmentation and pre/post-training can not only enhance cybersecurity-specific expertise but also help to improve general reasoning and instruction-following. All models, datasets, and code are publicly available. 

---
# Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts 

**Authors**: Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22156)  

**Abstract**: Hybrid Transformer architectures, which combine softmax attention blocks and recurrent neural networks (RNNs), have shown a desirable performance-throughput tradeoff for long-context modeling, but their adoption and studies are hindered by the prohibitive cost of large-scale pre-training from scratch. Some recent studies have shown that pre-trained softmax attention blocks can be converted into RNN blocks through parameter transfer and knowledge distillation. However, these transfer methods require substantial amounts of training data (more than 10B tokens), and the resulting hybrid models also exhibit poor long-context performance, which is the scenario where hybrid models enjoy significant inference speedups over Transformer-based models. In this paper, we present HALO (Hybrid Attention via Layer Optimization), a pipeline for distilling Transformer models into RNN-attention hybrid models. We then present HypeNet, a hybrid architecture with superior length generalization enabled by a novel position encoding scheme (named HyPE) and various architectural modifications. We convert the Qwen3 series into HypeNet using HALO, achieving performance comparable to the original Transformer models while enjoying superior long-context performance and efficiency. The conversion requires just 2.3B tokens, less than 0.01% of their pre-training data 

---
# DynaWeb: Model-Based Reinforcement Learning of Web Agents 

**Authors**: Hang Ding, Peidong Liu, Junqiao Wang, Ziwei Ji, Meng Cao, Rongzhao Zhang, Lynn Ai, Eric Yang, Tianyu Shi, Lei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.22149)  

**Abstract**: The development of autonomous web agents, powered by Large Language Models (LLMs) and reinforcement learning (RL), represents a significant step towards general-purpose AI assistants. However, training these agents is severely hampered by the challenges of interacting with the live internet, which is inefficient, costly, and fraught with risks. Model-based reinforcement learning (MBRL) offers a promising solution by learning a world model of the environment to enable simulated interaction. This paper introduces DynaWeb, a novel MBRL framework that trains web agents through interacting with a web world model trained to predict naturalistic web page representations given agent actions. This model serves as a synthetic web environment where an agent policy can dream by generating vast quantities of rollout action trajectories for efficient online reinforcement learning. Beyond free policy rollouts, DynaWeb incorporates real expert trajectories from training data, which are randomly interleaved with on-policy rollouts during training to improve stability and sample efficiency. Experiments conducted on the challenging WebArena and WebVoyager benchmarks demonstrate that DynaWeb consistently and significantly improves the performance of state-of-the-art open-source web agent models. Our findings establish the viability of training web agents through imagination, offering a scalable and efficient way to scale up online agentic RL. 

---
# Reasoning While Asking: Transforming Reasoning Large Language Models from Passive Solvers to Proactive Inquirers 

**Authors**: Xin Chen, Feng Jiang, Yiqian Zhang, Hardy Chen, Shuo Yan, Wenya Xie, Min Yang, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22139)  

**Abstract**: Reasoning-oriented Large Language Models (LLMs) have achieved remarkable progress with Chain-of-Thought (CoT) prompting, yet they remain fundamentally limited by a \emph{blind self-thinking} paradigm: performing extensive internal reasoning even when critical information is missing or ambiguous. We propose Proactive Interactive Reasoning (PIR), a new reasoning paradigm that transforms LLMs from passive solvers into proactive inquirers that interleave reasoning with clarification. Unlike existing search- or tool-based frameworks that primarily address knowledge uncertainty by querying external environments, PIR targets premise- and intent-level uncertainty through direct interaction with the user. PIR is implemented via two core components: (1) an uncertainty-aware supervised fine-tuning procedure that equips models with interactive reasoning capability, and (2) a user-simulator-based policy optimization framework driven by a composite reward that aligns model behavior with user intent. Extensive experiments on mathematical reasoning, code generation, and document editing demonstrate that PIR consistently outperforms strong baselines, achieving up to 32.70\% higher accuracy, 22.90\% higher pass rate, and 41.36 BLEU improvement, while reducing nearly half of the reasoning computation and unnecessary interaction turns. Further reliability evaluations on factual knowledge, question answering, and missing-premise scenarios confirm the strong generalization and robustness of PIR. Model and code are publicly available at: \href{this https URL} 

---
# PRISM: Distribution-free Adaptive Computation of Matrix Functions for Accelerating Neural Network Training 

**Authors**: Shenghao Yang, Zhichao Wang, Oleg Balabanov, N. Benjamin Erichson, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2601.22137)  

**Abstract**: Matrix functions such as square root, inverse roots, and orthogonalization play a central role in preconditioned gradient methods for neural network training. This has motivated the development of iterative algorithms that avoid explicit eigendecompositions and rely primarily on matrix multiplications, making them well suited for modern GPU accelerators. We present PRISM (Polynomial-fitting and Randomized Iterative Sketching for Matrix functions computation), a general framework for accelerating iterative algorithms for computing matrix functions. PRISM combines adaptive polynomial approximation with randomized sketching: at each iteration, it fits a polynomial surrogate to the current spectrum via a sketched least-squares problem, adapting to the instance at hand with minimal overhead. We apply PRISM to accelerate Newton-Schulz-like iterations for matrix square roots and orthogonalization, which are core primitives in machine learning. Unlike prior methods, PRISM requires no explicit spectral bounds or singular value estimates; and it adapts automatically to the evolving spectrum. Empirically, PRISM accelerates training when integrated into Shampoo and Muon optimizers. 

---
# StepShield: When, Not Whether to Intervene on Rogue Agents 

**Authors**: Gloria Felicia, Michael Eniolade, Jinfeng He, Zitha Sasindran, Hemant Kumar, Milan Hussain Angati, Sandeep Bandarupalli  

**Link**: [PDF](https://arxiv.org/pdf/2601.22136)  

**Abstract**: Existing agent safety benchmarks report binary accuracy, conflating early intervention with post-mortem analysis. A detector that flags a violation at step 8 enables intervention; one that reports it at step 48 provides only forensic value. This distinction is critical, yet current benchmarks cannot measure it. We introduce StepShield, the first benchmark to evaluate when violations are detected, not just whether. StepShield contains 9,213 code agent trajectories, including 1,278 meticulously annotated training pairs and a 7,935-trajectory test set with a realistic 8.1% rogue rate. Rogue behaviors are grounded in real-world security incidents across six categories. We propose three novel temporal metrics: Early Intervention Rate (EIR), Intervention Gap, and Tokens Saved. Surprisingly, our evaluation reveals that an LLM-based judge achieves 59% EIR while a static analyzer achieves only 26%, a 2.3x performance gap that is entirely invisible to standard accuracy metrics. We further show that early detection has direct economic benefits: our cascaded HybridGuard detector reduces monitoring costs by 75% and projects to $108M in cumulative savings over five years at enterprise scale. By shifting the focus of evaluation from whether to when, StepShield provides a new foundation for building safer and more economically viable AI agents. The code and data are released under an Apache 2.0 license. 

---
# SWE-Replay: Efficient Test-Time Scaling for Software Engineering Agents 

**Authors**: Yifeng Ding, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22129)  

**Abstract**: Test-time scaling has been widely adopted to enhance the capabilities of Large Language Model (LLM) agents in software engineering (SWE) tasks. However, the standard approach of repeatedly sampling trajectories from scratch is computationally expensive. While recent methods have attempted to mitigate costs using specialized value agents, they can suffer from model miscalibration and fail to generalize to modern agents that synthesize custom bash scripts as tools. In this paper, we introduce SWE-Replay, the first efficient and generalizable test-time scaling technique for modern agents without reliance on potentially noisy value estimates. SWE-Replay optimizes the scaling process by recycling trajectories from prior trials, dynamically choosing to either explore from scratch or exploit archived experience by branching at critical intermediate steps. This selection of intermediate steps is driven by the potential and reasoning significance of repository exploration, rather than external LLM-based quality estimates. Our evaluation shows that, on SWE-Bench Verified, SWE-Replay consistently outperforms naive scaling, reducing costs by up to 17.4% while maintaining or even improving performance by up to 3.8%. Further evaluation on SWE-Bench Pro and Multilingual validates the generalizability of SWE-Replay, establishing it as a robust foundation for efficient test-time scaling of software engineering agents. 

---
# Alpha Discovery via Grammar-Guided Learning and Search 

**Authors**: Han Yang, Dong Hao, Zhuohan Wang, Qi Shi, Xingtong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.22119)  

**Abstract**: Automatically discovering formulaic alpha factors is a central problem in quantitative finance. Existing methods often ignore syntactic and semantic constraints, relying on exhaustive search over unstructured and unbounded spaces. We present AlphaCFG, a grammar-based framework for defining and discovering alpha factors that are syntactically valid, financially interpretable, and computationally efficient. AlphaCFG uses an alpha-oriented context-free grammar to define a tree-structured, size-controlled search space, and formulates alpha discovery as a tree-structured linguistic Markov decision process, which is then solved using a grammar-aware Monte Carlo Tree Search guided by syntax-sensitive value and policy networks. Experiments on Chinese and U.S. stock market datasets show that AlphaCFG outperforms state-of-the-art baselines in both search efficiency and trading profitability. Beyond trading strategies, AlphaCFG serves as a general framework for symbolic factor discovery and refinement across quantitative finance, including asset pricing and portfolio construction. 

---
# SINA: A Circuit Schematic Image-to-Netlist Generator Using Artificial Intelligence 

**Authors**: Saoud Aldowaish, Yashwanth Karumanchi, Kai-Chen Chiang, Soroosh Noorzad, Morteza Fayazi  

**Link**: [PDF](https://arxiv.org/pdf/2601.22114)  

**Abstract**: Current methods for converting circuit schematic images into machine-readable netlists struggle with component recognition and connectivity inference. In this paper, we present SINA, an open-source, fully automated circuit schematic image-to-netlist generator. SINA integrates deep learning for accurate component detection, Connected-Component Labeling (CCL) for precise connectivity extraction, and Optical Character Recognition (OCR) for component reference designator retrieval, while employing a Vision-Language Model (VLM) for reliable reference designator assignments. In our experiments, SINA achieves 96.47% overall netlist-generation accuracy, which is 2.72x higher than state-of-the-art approaches. 

---
# Value-Based Pre-Training with Downstream Feedback 

**Authors**: Shuqi Ke, Giulia Fanti  

**Link**: [PDF](https://arxiv.org/pdf/2601.22108)  

**Abstract**: Can a small amount of verified goal information steer the expensive self-supervised pretraining of foundation models? Standard pretraining optimizes a fixed proxy objective (e.g., next-token prediction), which can misallocate compute away from downstream capabilities of interest. We introduce V-Pretraining: a value-based, modality-agnostic method for controlled continued pretraining in which a lightweight task designer reshapes the pretraining task to maximize the value of each gradient step. For example, consider self-supervised learning (SSL) with sample augmentation. The V-Pretraining task designer selects pretraining tasks (e.g., augmentations) for which the pretraining loss gradient is aligned with a gradient computed over a downstream task (e.g., image segmentation). This helps steer pretraining towards relevant downstream capabilities. Notably, the pretrained model is never updated on downstream task labels; they are used only to shape the pretraining task. Under matched learner update budgets, V-Pretraining of 0.5B--7B language models improves reasoning (GSM8K test Pass@1) by up to 18% relative over standard next-token prediction using only 12% of GSM8K training examples as feedback. In vision SSL, we improve the state-of-the-art results on ADE20K by up to 1.07 mIoU and reduce NYUv2 RMSE while improving ImageNet linear accuracy, and we provide pilot evidence of improved token efficiency in continued pretraining. 

---
# ECO: Quantized Training without Full-Precision Master Weights 

**Authors**: Mahdi Nikdan, Amir Zandieh, Dan Alistarh, Vahab Mirrokni  

**Link**: [PDF](https://arxiv.org/pdf/2601.22101)  

**Abstract**: Quantization has significantly improved the compute and memory efficiency of Large Language Model (LLM) training. However, existing approaches still rely on accumulating their updates in high-precision: concretely, gradient updates must be applied to a high-precision weight buffer, known as $\textit{master weights}$. This buffer introduces substantial memory overhead, particularly for Sparse Mixture of Experts (SMoE) models, where model parameters and optimizer states dominate memory usage. To address this, we introduce the Error-Compensating Optimizer (ECO), which eliminates master weights by applying updates directly to quantized parameters. ECO quantizes weights after each step and carefully injects the resulting quantization error into the optimizer momentum, forming an error-feedback loop with no additional memory. We prove that, under standard assumptions and a decaying learning rate, ECO converges to a constant-radius neighborhood of the optimum, while naive master-weight removal can incur an error that is inversely proportional to the learning rate. We show empirical results for pretraining small Transformers (30-800M), a Gemma-3 1B model, and a 2.1B parameter Sparse MoE model with FP8 quantization, and fine-tuning DeepSeek-MoE-16B in INT4 precision. Throughout, ECO matches baselines with master weights up to near-lossless accuracy, significantly shifting the static memory vs validation loss Pareto frontier. 

---
# Investigating Associational Biases in Inter-Model Communication of Large Generative Models 

**Authors**: Fethiye Irmak Dogan, Yuval Weiss, Kajal Patel, Jiaee Cheong, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2601.22093)  

**Abstract**: Social bias in generative AI can manifest not only as performance disparities but also as associational bias, whereby models learn and reproduce stereotypical associations between concepts and demographic groups, even in the absence of explicit demographic information (e.g., associating doctors with men). These associations can persist, propagate, and potentially amplify across repeated exchanges in inter-model communication pipelines, where one generative model's output becomes another's input. This is especially salient for human-centred perception tasks, such as human activity recognition and affect prediction, where inferences about behaviour and internal states can lead to errors or stereotypical associations that propagate into unequal treatment. In this work, focusing on human activity and affective expression, we study how such associations evolve within an inter-model communication pipeline that alternates between image generation and image description. Using the RAF-DB and PHASE datasets, we quantify demographic distribution drift induced by model-to-model information exchange and assess whether these drifts are systematic using an explainability pipeline. Our results reveal demographic drifts toward younger representations for both actions and emotions, as well as toward more female-presenting representations, primarily for emotions. We further find evidence that some predictions are supported by spurious visual regions (e.g., background or hair) rather than concept-relevant cues (e.g., body or face). We also examine whether these demographic drifts translate into measurable differences in downstream behaviour, i.e., while predicting activity and emotion labels. Finally, we outline mitigation strategies spanning data-centric, training and deployment interventions, and emphasise the need for careful safeguards when deploying interconnected models in human-centred AI systems. 

---
# Latent Adversarial Regularization for Offline Preference Optimization 

**Authors**: Enyi Jiang, Yibo Jacky Zhang, Yinglun Xu, Andreas Haupt, Nancy Amato, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2601.22083)  

**Abstract**: Learning from human feedback typically relies on preference optimization that constrains policy updates through token-level regularization. However, preference optimization for language models is particularly challenging because token-space similarity does not imply semantic or behavioral similarity. To address this challenge, we leverage latent-space regularization for language model preference optimization. We introduce GANPO, which achieves latent-space regularization by penalizing divergence between the internal representations of a policy model and a reference model. Given that latent representations are not associated with explicit probability densities, we adopt an adversarial approach inspired by GANs to minimize latent-space divergence. We integrate GANPO as a regularizer into existing offline preference optimization objectives. Experiments across multiple model architectures and tasks show consistent improvements from latent-space regularization. Further, by comparing GANPO-induced inferential biases with those from token-level regularization, we find that GANPO provides more robust structural feedback under distributional shift and noise while maintaining comparable downstream performance with minor computational overhead. 

---
# Vision-DeepResearch: Incentivizing DeepResearch Capability in Multimodal Large Language Models 

**Authors**: Wenxuan Huang, Yu Zeng, Qiuchen Wang, Zhen Fang, Shaosheng Cao, Zheng Chu, Qingyu Yin, Shuang Chen, Zhenfei Yin, Lin Chen, Zehui Chen, Yao Hu, Philip Torr, Feng Zhao, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22060)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable success across a broad range of vision tasks. However, constrained by the capacity of their internal world knowledge, prior work has proposed augmenting MLLMs by ``reasoning-then-tool-call'' for visual and textual search engines to obtain substantial gains on tasks requiring extensive factual information. However, these approaches typically define multimodal search in a naive setting, assuming that a single full-level or entity-level image query and few text query suffices to retrieve the key evidence needed to answer the question, which is unrealistic in real-world scenarios with substantial visual noise. Moreover, they are often limited in the reasoning depth and search breadth, making it difficult to solve complex questions that require aggregating evidence from diverse visual and textual sources. Building on this, we propose Vision-DeepResearch, which proposes one new multimodal deep-research paradigm, i.e., performs multi-turn, multi-entity and multi-scale visual and textual search to robustly hit real-world search engines under heavy noise. Our Vision-DeepResearch supports dozens of reasoning steps and hundreds of engine interactions, while internalizing deep-research capabilities into the MLLM via cold-start supervision and RL training, resulting in a strong end-to-end multimodal deep-research MLLM. It substantially outperforming existing multimodal deep-research MLLMs, and workflows built on strong closed-source foundation model such as GPT-5, Gemini-2.5-pro and Claude-4-Sonnet. The code will be released in this https URL. 

---
# Unsupervised Decomposition and Recombination with Discriminator-Driven Diffusion Models 

**Authors**: Archer Wang, Emile Anand, Yilun Du, Marin Soljačić  

**Link**: [PDF](https://arxiv.org/pdf/2601.22057)  

**Abstract**: Decomposing complex data into factorized representations can reveal reusable components and enable synthesizing new samples via component recombination. We investigate this in the context of diffusion-based models that learn factorized latent spaces without factor-level supervision. In images, factors can capture background, illumination, and object attributes; in robotic videos, they can capture reusable motion components. To improve both latent factor discovery and quality of compositional generation, we introduce an adversarial training signal via a discriminator trained to distinguish between single-source samples and those generated by recombining factors across sources. By optimizing the generator to fool this discriminator, we encourage physical and semantic consistency in the resulting recombinations. Our method outperforms implementations of prior baselines on CelebA-HQ, Virtual KITTI, CLEVR, and Falcor3D, achieving lower FID scores and better disentanglement as measured by MIG and MCC. Furthermore, we demonstrate a novel application to robotic video trajectories: by recombining learned action components, we generate diverse sequences that significantly increase state-space coverage for exploration on the LIBERO benchmark. 

---
# MetricAnything: Scaling Metric Depth Pretraining with Noisy Heterogeneous Sources 

**Authors**: Baorui Ma, Jiahui Yang, Donglin Di, Xuancheng Zhang, Jianxun Cui, Hao Li, Yan Xie, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.22054)  

**Abstract**: Scaling has powered recent advances in vision foundation models, yet extending this paradigm to metric depth estimation remains challenging due to heterogeneous sensor noise, camera-dependent biases, and metric ambiguity in noisy cross-source 3D data. We introduce Metric Anything, a simple and scalable pretraining framework that learns metric depth from noisy, diverse 3D sources without manually engineered prompts, camera-specific modeling, or task-specific architectures. Central to our approach is the Sparse Metric Prompt, created by randomly masking depth maps, which serves as a universal interface that decouples spatial reasoning from sensor and camera biases. Using about 20M image-depth pairs spanning reconstructed, captured, and rendered 3D data across 10000 camera models, we demonstrate-for the first time-a clear scaling trend in the metric depth track. The pretrained model excels at prompt-driven tasks such as depth completion, super-resolution and Radar-camera fusion, while its distilled prompt-free student achieves state-of-the-art results on monocular depth estimation, camera intrinsics recovery, single/multi-view metric 3D reconstruction, and VLA planning. We also show that using pretrained ViT of Metric Anything as a visual encoder significantly boosts Multimodal Large Language Model capabilities in spatial intelligence. These results show that metric depth estimation can benefit from the same scaling laws that drive modern foundation models, establishing a new path toward scalable and efficient real-world metric perception. We open-source MetricAnything at this http URL to support community research. 

---
# SIA: Symbolic Interpretability for Anticipatory Deep Reinforcement Learning in Network Control 

**Authors**: MohammadErfan Jabbari, Abhishek Duttagupta, Claudio Fiandrino, Leonardo Bonati, Salvatore D'Oro, Michele Polese, Marco Fiore, Tommaso Melodia  

**Link**: [PDF](https://arxiv.org/pdf/2601.22044)  

**Abstract**: Deep reinforcement learning (DRL) promises adaptive control for future mobile networks but conventional agents remain reactive: they act on past and current measurements and cannot leverage short-term forecasts of exogenous KPIs such as bandwidth. Augmenting agents with predictions can overcome this temporal myopia, yet uptake in networking is scarce because forecast-aware agents act as closed-boxes; operators cannot tell whether predictions guide decisions or justify the added complexity. We propose SIA, the first interpreter that exposes in real time how forecast-augmented DRL agents operate. SIA fuses Symbolic AI abstractions with per-KPI Knowledge Graphs to produce explanations, and includes a new Influence Score metric. SIA achieves sub-millisecond speed, over 200x faster than existing XAI methods. We evaluate SIA on three diverse networking use cases, uncovering hidden issues, including temporal misalignment in forecast integration and reward-design biases that trigger counter-productive policies. These insights enable targeted fixes: a redesigned agent achieves a 9% higher average bitrate in video streaming, and SIA's online Action-Refinement module improves RAN-slicing reward by 25% without retraining. By making anticipatory DRL transparent and tunable, SIA lowers the barrier to proactive control in next-generation mobile networks. 

---
# Learning to Communicate Across Modalities: Perceptual Heterogeneity in Multi-Agent Systems 

**Authors**: Naomi Pitzer, Daniela Mihai  

**Link**: [PDF](https://arxiv.org/pdf/2601.22041)  

**Abstract**: Emergent communication offers insight into how agents develop shared structured representations, yet most research assumes homogeneous modalities or aligned representational spaces, overlooking the perceptual heterogeneity of real-world settings. We study a heterogeneous multi-step binary communication game where agents differ in modality and lack perceptual grounding. Despite perceptual misalignment, multimodal systems converge to class-consistent messages grounded in perceptual input. Unimodal systems communicate more efficiently, using fewer bits and achieving lower classification entropy, while multimodal agents require greater information exchange and exhibit higher uncertainty. Bit perturbation experiments provide strong evidence that meaning is encoded in a distributional rather than compositional manner, as each bit's contribution depends on its surrounding pattern. Finally, interoperability analyses show that systems trained in different perceptual worlds fail to directly communicate, but limited fine-tuning enables successful cross-system communication. This work positions emergent communication as a framework for studying how agents adapt and transfer representations across heterogeneous modalities, opening new directions for both theory and experimentation. 

---
# A Separable Architecture for Continuous Token Representation in Language Models 

**Authors**: Reza T. Batley, Sourav Saha  

**Link**: [PDF](https://arxiv.org/pdf/2601.22040)  

**Abstract**: Transformer scaling law analyses typically treat parameters as interchangeable; an abstraction that accurately predicts loss-compute relationships. Yet, in sub-billion-parameter small language models (SLMs), embedding matrices dominate the parameter budget. This work argues that this allocation is as suboptimal as it is counterintuitive. Leviathan is an architecture with a continuous embedding generator to replace the discrete lookup tables of canonical models. Evaluating on the Pile dataset under isoparametric settings, Leviathan consistently outperforms a standard, LLaMA-style architecture. By means of an empirical power-law fit, Leviathan exhibits a markedly superior effective parameter capacity. Across the regime studied, Leviathan behaves as a dense model with $1.47$ to $2.11 \times$ more parameters. 

---
# Thinking Out of Order: When Output Order Stops Reflecting Reasoning Order in Diffusion Language Models 

**Authors**: Longxuan Yu, Yu Fu, Shaorong Zhang, Hui Liu, Mukund Varma T, Greg Ver Steeg, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.22035)  

**Abstract**: Autoregressive (AR) language models enforce a fixed left-to-right generation order, creating a fundamental limitation when the required output structure conflicts with natural reasoning (e.g., producing answers before explanations due to presentation or schema constraints). In such cases, AR models must commit to answers before generating intermediate reasoning, and this rigid constraint forces premature commitment. Masked diffusion language models (MDLMs), which iteratively refine all tokens in parallel, offer a way to decouple computation order from output structure. We validate this capability on GSM8K, Math500, and ReasonOrderQA, a benchmark we introduce with controlled difficulty and order-level evaluation. When prompts request answers before reasoning, AR models exhibit large accuracy gaps compared to standard chain-of-thought ordering (up to 67% relative drop), while MDLMs remain stable ($\leq$14% relative drop), a property we term "order robustness". Using ReasonOrderQA, we present evidence that MDLMs achieve order robustness by stabilizing simpler tokens (e.g., reasoning steps) earlier in the diffusion process than complex ones (e.g., final answers), enabling reasoning tokens to stabilize before answer commitment. Finally, we identify failure conditions where this advantage weakens, outlining the limits required for order robustness. 

---
# When "Better" Prompts Hurt: Evaluation-Driven Iteration for LLM Applications 

**Authors**: Daniel Commey  

**Link**: [PDF](https://arxiv.org/pdf/2601.22025)  

**Abstract**: Evaluating Large Language Model (LLM) applications differs from traditional software testing because outputs are stochastic, high-dimensional, and sensitive to prompt and model changes. We present an evaluation-driven workflow - Define, Test, Diagnose, Fix - that turns these challenges into a repeatable engineering loop.
We introduce the Minimum Viable Evaluation Suite (MVES), a tiered set of recommended evaluation components for (i) general LLM applications, (ii) retrieval-augmented generation (RAG), and (iii) agentic tool-use workflows. We also synthesize common evaluation methods (automated checks, human rubrics, and LLM-as-judge) and discuss known judge failure modes.
In reproducible local experiments (Ollama; Llama 3 8B Instruct and Qwen 2.5 7B Instruct), we observe that a generic "improved" prompt template can trade off behaviors: on our small structured suites, extraction pass rate decreased from 100% to 90% and RAG compliance from 93.3% to 80% for Llama 3 when replacing task-specific prompts with generic rules, while instruction-following improved. These findings motivate evaluation-driven prompt iteration and careful claim calibration rather than universal prompt recipes.
All test suites, harnesses, and results are included for reproducibility. 

---
# SymbXRL: Symbolic Explainable Deep Reinforcement Learning for Mobile Networks 

**Authors**: Abhishek Duttagupta, MohammadErfan Jabbari, Claudio Fiandrino, Marco Fiore, Joerg Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2601.22024)  

**Abstract**: The operation of future 6th-generation (6G) mobile networks will increasingly rely on the ability of deep reinforcement learning (DRL) to optimize network decisions in real-time. DRL yields demonstrated efficacy in various resource allocation problems, such as joint decisions on user scheduling and antenna allocation or simultaneous control of computing resources and modulation. However, trained DRL agents are closed-boxes and inherently difficult to explain, which hinders their adoption in production settings. In this paper, we make a step towards removing this critical barrier by presenting SymbXRL, a novel technique for explainable reinforcement learning (XRL) that synthesizes human-interpretable explanations for DRL agents. SymbXRL leverages symbolic AI to produce explanations where key concepts and their relationships are described via intuitive symbols and rules; coupling such a representation with logical reasoning exposes the decision process of DRL agents and offers more comprehensible descriptions of their behaviors compared to existing approaches. We validate SymbXRL in practical network management use cases supported by DRL, proving that it not only improves the semantics of the explanations but also paves the way for explicit agent control: for instance, it enables intent-based programmatic action steering that improves by 12% the median cumulative reward over a pure DRL solution. 

---
# Vidmento: Creating Video Stories Through Context-Aware Expansion With Generative Video 

**Authors**: Catherine Yeh, Anh Truong, Mira Dontcheva, Bryan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.22013)  

**Abstract**: Video storytelling is often constrained by available material, limiting creative expression and leaving undesired narrative gaps. Generative video offers a new way to address these limitations by augmenting captured media with tailored visuals. To explore this potential, we interviewed eight video creators to identify opportunities and challenges in integrating generative video into their workflows. Building on these insights and established filmmaking principles, we developed Vidmento, a tool for authoring hybrid video stories that combine captured and generated media through context-aware expansion. Vidmento surfaces opportunities for story development, generates clips that blend stylistically and narratively with surrounding media, and provides controls for refinement. In a study with 12 creators, Vidmento supported narrative development and exploration by systematically expanding initial materials with generative media, enabling expressive video storytelling aligned with creative intent. We highlight how creators bridge story gaps with generative content and where they find this blending capability most valuable. 

---
# MEIDNet: Multimodal generative AI framework for inverse materials design 

**Authors**: Anand Babu, Rogério Almeida Gouvêa, Pierre Vandergheynst, Gian-Marco Rignanese  

**Link**: [PDF](https://arxiv.org/pdf/2601.22009)  

**Abstract**: In this work, we present Multimodal Equivariant Inverse Design Network (MEIDNet), a framework that jointly learns structural information and materials properties through contrastive learning, while encoding structures via an equivariant graph neural network (EGNN). By combining generative inverse design with multimodal learning, our approach accelerates the exploration of chemical-structural space and facilitates the discovery of materials that satisfy predefined property targets. MEIDNet exhibits strong latent-space alignment with cosine similarity 0.96 by fusion of three modalities through cross-modal learning. Through implementation of curriculum learning strategies, MEIDNet achieves ~60 times higher learning efficiency than conventional training techniques. The potential of our multimodal approach is demonstrated by generating low-bandgap perovskite structures at a stable, unique, and novel (SUN) rate of 13.6 %, which are further validated by ab initio methods. Our inverse design framework demonstrates both scalability and adaptability, paving the way for the universal learning of chemical space across diverse modalities. 

---
# Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units 

**Authors**: Jianhui Chen, Yuzhang Luo, Liangming Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21996)  

**Abstract**: While Mechanistic Interpretability has identified interpretable circuits in LLMs, their causal origins in training data remain elusive. We introduce Mechanistic Data Attribution (MDA), a scalable framework that employs Influence Functions to trace interpretable units back to specific training samples. Through extensive experiments on the Pythia family, we causally validate that targeted intervention--removing or augmenting a small fraction of high-influence samples--significantly modulates the emergence of interpretable heads, whereas random interventions show no effect. Our analysis reveals that repetitive structural data (e.g., LaTeX, XML) acts as a mechanistic catalyst. Furthermore, we observe that interventions targeting induction head formation induce a concurrent change in the model's in-context learning (ICL) capability. This provides direct causal evidence for the long-standing hypothesis regarding the functional link between induction heads and ICL. Finally, we propose a mechanistic data augmentation pipeline that consistently accelerates circuit convergence across model scales, providing a principled methodology for steering the developmental trajectories of LLMs. 

---
# Geometry of Drifting MDPs with Path-Integral Stability Certificates 

**Authors**: Zuyuan Zhang, Mahdi Imani, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21991)  

**Abstract**: Real-world reinforcement learning is often \emph{nonstationary}: rewards and dynamics drift, accelerate, oscillate, and trigger abrupt switches in the optimal action. Existing theory often represents nonstationarity with coarse-scale models that measure \emph{how much} the environment changes, not \emph{how} it changes locally -- even though acceleration and near-ties drive tracking error and policy chattering. We take a geometric view of nonstationary discounted Markov Decision Processes (MDPs) by modeling the environment as a differentiable homotopy path and tracking the induced motion of the optimal Bellman fixed point. This yields a length--curvature--kink signature of intrinsic complexity: cumulative drift, acceleration/oscillation, and action-gap-induced nonsmoothness. We prove a solver-agnostic path-integral stability bound and derive gap-safe feasible regions that certify local stability away from switch regimes. Building on these results, we introduce \textit{Homotopy-Tracking RL (HT-RL)} and \textit{HT-MCTS}, lightweight wrappers that estimate replay-based proxies of length, curvature, and near-tie proximity online and adapt learning or planning intensity accordingly. Experiments show improved tracking and dynamic regret over matched static baselines, with the largest gains in oscillatory and switch-prone regimes. 

---
# Generalized Information Gathering Under Dynamics Uncertainty 

**Authors**: Fernando Palafox, Jingqi Li, Jesse Milzman, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2601.21988)  

**Abstract**: An agent operating in an unknown dynamical system must learn its dynamics from observations. Active information gathering accelerates this learning, but existing methods derive bespoke costs for specific modeling choices: dynamics models, belief update procedures, observation models, and planners. We present a unifying framework that decouples these choices from the information-gathering cost by explicitly exposing the causal dependencies between parameters, beliefs, and controls. Using this framework, we derive a general information-gathering cost based on Massey's directed information that assumes only Markov dynamics with additive noise and is otherwise agnostic to modeling choices. We prove that the mutual information cost used in existing literature is a special case of our cost. Then, we leverage our framework to establish an explicit connection between the mutual information cost and information gain in linearized Bayesian estimation, thereby providing theoretical justification for mutual information-based active learning approaches. Finally, we illustrate the practical utility of our framework through experiments spanning linear, nonlinear, and multi-agent systems. 

---
# From Particles to Agents: Hallucination as a Metric for Cognitive Friction in Spatial Simulation 

**Authors**: Javier Argota Sánchez-Vaquerizo, Luis Borunda Monsivais  

**Link**: [PDF](https://arxiv.org/pdf/2601.21977)  

**Abstract**: Traditional architectural simulations (e.g. Computational Fluid Dynamics, evacuation, structural analysis) model elements as deterministic physics-based "particles" rather than cognitive "agents". To bridge this, we introduce \textbf{Agentic Environmental Simulations}, where Large Multimodal generative models actively predict the next state of spatial environments based on semantic expectation. Drawing on examples from accessibility-oriented AR pipelines and multimodal digital twins, we propose a shift from chronological time-steps to Episodic Spatial Reasoning, where simulations advance through meaningful, surprisal-triggered events. Within this framework we posit AI hallucinations as diagnostic tools. By formalizing the \textbf{Cognitive Friction} ($C_f$) it is possible to reveal "Phantom Affordances", i.e. semiotic ambiguities in built space. Finally, we challenge current HCI paradigms by treating environments as dynamic cognitive partners and propose a human-centered framework of cognitive orchestration for designing AI-driven simulations that preserve autonomy, affective clarity, and cognitive integrity. 

---
# MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts 

**Authors**: Lorenzo Mazza, Ariel Rodriguez, Rayan Younis, Martin Lelis, Ortrun Hellig, Chenpan Li, Sebastian Bodenstedt, Martin Wagner, Stefanie Speidel  

**Link**: [PDF](https://arxiv.org/pdf/2601.21971)  

**Abstract**: Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery. 

---
# Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding 

**Authors**: Yifan Zhu, Huiqiang Rong, Haoran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21969)  

**Abstract**: Large Language Models (LLMs) often hallucinate, generating content inconsistent with the input. Retrieval-Augmented Generation (RAG) and Reinforcement Learning with Human Feedback (RLHF) can mitigate hallucinations but require resource-intensive retrieval or large-scale fine-tuning. Decoding-based methods are lighter yet lack explicit hallucination control. To address this, we present Token-Guard, a token-level hallucination control method based on self-checking decoding. Token-Guard performs internal verification at each reasoning step to detect hallucinated tokens before they propagate. Candidate fragments are further evaluated in a latent space with explicit hallucination risk scoring, while iterative pruning and regeneration dynamically correct detected errors. Experiments on HALU datasets show Token-Guard substantially reduces hallucinations and improves generation accuracy, offering a scalable, modular solution for reliable LLM outputs. Our code is publicly available. 

---
# Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems 

**Authors**: Alexander Loth, Martin Kappes, Marc-Oliver Pahl  

**Link**: [PDF](https://arxiv.org/pdf/2601.21963)  

**Abstract**: Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality. 

---
# Robust Multimodal Representation Learning in Healthcare 

**Authors**: Xiaoguang Zhu, Linxiao Gong, Lianlong Sun, Yang Liu, Haoyu Wang, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21941)  

**Abstract**: Medical multimodal representation learning aims to integrate heterogeneous data into unified patient representations to support clinical outcome prediction. However, real-world medical datasets commonly contain systematic biases from multiple sources, which poses significant challenges for medical multimodal representation learning. Existing approaches typically focus on effective multimodal fusion, neglecting inherent biased features that affect the generalization ability. To address these challenges, we propose a Dual-Stream Feature Decorrelation Framework that identifies and handles the biases through structural causal analysis introduced by latent confounders. Our method employs a causal-biased decorrelation framework with dual-stream neural networks to disentangle causal features from spurious correlations, utilizing generalized cross-entropy loss and mutual information minimization for effective decorrelation. The framework is model-agnostic and can be integrated into existing medical multimodal learning methods. Comprehensive experiments on MIMIC-IV, eICU, and ADNI datasets demonstrate consistent performance improvements. 

---
# From Future of Work to Future of Workers: Addressing Asymptomatic AI Harms for Dignified Human-AI Interaction 

**Authors**: Upol Ehsan, Samir Passi, Koustuv Saha, Todd McNutt, Mark O. Riedl, Sara Alcorn  

**Link**: [PDF](https://arxiv.org/pdf/2601.21920)  

**Abstract**: In the future of work discourse, AI is touted as the ultimate productivity amplifier. Yet, beneath the efficiency gains lie subtle erosions of human expertise and agency. This paper shifts focus from the future of work to the future of workers by navigating the AI-as-Amplifier Paradox: AI's dual role as enhancer and eroder, simultaneously strengthening performance while eroding underlying expertise. We present a year-long study on the longitudinal use of AI in a high-stakes workplace among cancer specialists. Initial operational gains hid ``intuition rust'': the gradual dulling of expert judgment. These asymptomatic effects evolved into chronic harms, such as skill atrophy and identity commoditization. Building on these findings, we offer a framework for dignified Human-AI interaction co-constructed with professional knowledge workers facing AI-induced skill erosion without traditional labor protections. The framework operationalizes sociotechnical immunity through dual-purpose mechanisms that serve institutional quality goals while building worker power to detect, contain, and recover from skill erosion, and preserve human identity. Evaluated across healthcare and software engineering, our work takes a foundational step toward dignified human-AI interaction futures by balancing productivity with the preservation of human expertise. 

---
# TraceRouter: Robust Safety for Large Foundation Models via Path-Level Intervention 

**Authors**: Chuancheng Shi, Shangze Li, Wenjun Lu, Wenhua Wu, Cong Wang, Zifeng Cheng, Fei Shen, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2601.21900)  

**Abstract**: Despite their capabilities, large foundation models (LFMs) remain susceptible to adversarial manipulation. Current defenses predominantly rely on the "locality hypothesis", suppressing isolated neurons or features. However, harmful semantics act as distributed, cross-layer circuits, rendering such localized interventions brittle and detrimental to utility. To bridge this gap, we propose \textbf{TraceRouter}, a path-level framework that traces and disconnects the causal propagation circuits of illicit semantics. TraceRouter operates in three stages: (1) it pinpoints a sensitive onset layer by analyzing attention divergence; (2) it leverages sparse autoencoders (SAEs) and differential activation analysis to disentangle and isolate malicious features; and (3) it maps these features to downstream causal pathways via feature influence scores (FIS) derived from zero-out interventions. By selectively suppressing these causal chains, TraceRouter physically severs the flow of harmful information while leaving orthogonal computation routes intact. Extensive experiments demonstrate that TraceRouter significantly outperforms state-of-the-art baselines, achieving a superior trade-off between adversarial robustness and general utility. Our code will be publicly released. WARNING: This paper contains unsafe model responses. 

---
# Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text 

**Authors**: Hongyi Zhou, Jin Zhu, Erhan Xu, Kai Ye, Ying Yang, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21895)  

**Abstract**: Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 57.8\% to 80.6\% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini). 

---
# Improving Classifier-Free Guidance of Flow Matching via Manifold Projection 

**Authors**: Jian-Feng Cai, Haixia Liu, Zhengyi Su, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21892)  

**Abstract**: Classifier-free guidance (CFG) is a widely used technique for controllable generation in diffusion and flow-based models. Despite its empirical success, CFG relies on a heuristic linear extrapolation that is often sensitive to the guidance scale. In this work, we provide a principled interpretation of CFG through the lens of optimization. We demonstrate that the velocity field in flow matching corresponds to the gradient of a sequence of smoothed distance functions, which guides latent variables toward the scaled target image set. This perspective reveals that the standard CFG formulation is an approximation of this gradient, where the prediction gap, the discrepancy between conditional and unconditional outputs, governs guidance sensitivity. Leveraging this insight, we reformulate the CFG sampling as a homotopy optimization with a manifold constraint. This formulation necessitates a manifold projection step, which we implement via an incremental gradient descent scheme during sampling. To improve computational efficiency and stability, we further enhance this iterative process with Anderson Acceleration without requiring additional model evaluations. Our proposed methods are training-free and consistently refine generation fidelity, prompt alignment, and robustness to the guidance scale. We validate their effectiveness across diverse benchmarks, demonstrating significant improvements on large-scale models such as DiT-XL-2-256, Flux, and Stable Diffusion 3.5. 

---
# MoHETS: Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts 

**Authors**: Evandro S. Ortigossa, Guy Lutsker, Eran Segal  

**Link**: [PDF](https://arxiv.org/pdf/2601.21866)  

**Abstract**: Real-world multivariate time series can exhibit intricate multi-scale structures, including global trends, local periodicities, and non-stationary regimes, which makes long-horizon forecasting challenging. Although sparse Mixture-of-Experts (MoE) approaches improve scalability and specialization, they typically rely on homogeneous MLP experts that poorly capture the diverse temporal dynamics of time series data. We address these limitations with MoHETS, an encoder-only Transformer that integrates sparse Mixture-of-Heterogeneous-Experts (MoHE) layers. MoHE routes temporal patches to a small subset of expert networks, combining a shared depthwise-convolution expert for sequence-level continuity with routed Fourier-based experts for patch-level periodic structures. MoHETS further improves robustness to non-stationary dynamics by incorporating exogenous information via cross-attention over covariate patch embeddings. Finally, we replace parameter-heavy linear projection heads with a lightweight convolutional patch decoder, improving parameter efficiency, reducing training instability, and allowing a single model to generalize across arbitrary forecast horizons. We validate across seven multivariate benchmarks and multiple horizons, with MoHETS consistently achieving state-of-the-art performance, reducing the average MSE by $12\%$ compared to strong recent baselines, demonstrating effective heterogeneous specialization for long-term forecasting. 

---
# Test-Time Compute Games 

**Authors**: Ander Artola Velasco, Dimitrios Rontogiannis, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2601.21839)  

**Abstract**: Test-time compute has emerged as a promising strategy to enhance the reasoning abilities of large language models (LLMs). However, this strategy has in turn increased how much users pay cloud-based providers offering LLM-as-a-service, since providers charge users for the amount of test-time compute they use to generate an output. In our work, we show that the market of LLM-as-a-service is socially inefficient: providers have a financial incentive to increase the amount of test-time compute, even if this increase contributes little to the quality of the outputs. To address this inefficiency, we introduce a reverse second-price auction mechanism where providers bid their offered price and (expected) quality for the opportunity to serve a user, and users pay proportionally to the marginal value generated by the winning provider relative to the second-highest bidder. To illustrate and complement our theoretical results, we conduct experiments with multiple instruct models from the $\texttt{Llama}$ and $\texttt{Qwen}$ families, as well as reasoning models distilled from $\texttt{DeepSeek-R1}$, on math and science benchmark datasets. 

---
# Trustworthy Intelligent Education: A Systematic Perspective on Progress, Challenges, and Future Directions 

**Authors**: Xiaoshan Yu, Shangshang Yang, Ziwen Wang, Haiping Ma, Xingyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21837)  

**Abstract**: In recent years, trustworthiness has garnered increasing attention and exploration in the field of intelligent education, due to the inherent sensitivity of educational scenarios, such as involving minors and vulnerable groups, highly personalized learning data, and high-stakes educational outcomes. However, existing research either focuses on task-specific trustworthy methods without a holistic view of trustworthy intelligent education, or provides survey-level discussions that remain high-level and fragmented, lacking a clear and systematic categorization. To address these limitations, in this paper, we present a systematic and structured review of trustworthy intelligent education. Specifically, We first organize intelligent education into five representative task categories: learner ability assessment, learning resource recommendation, learning analytics, educational content understanding, and instructional assistance. Building on this task landscape, we review existing studies from five trustworthiness perspectives, including safety and privacy, robustness, fairness, explainability, and sustainability, and summarize and categorize the research methodologies and solution strategies therein. Finally, we summarize key challenges and discuss future research directions. This survey aims to provide a coherent reference framework and facilitate a clearer understanding of trustworthiness in intelligent education. 

---
# Moral Outrage Shapes Commitments Beyond Attention: Multimodal Moral Emotions on YouTube in Korea and the US 

**Authors**: Seongchan Park, Jaehong Kim, Hyeonseung Kim, Heejin Bin, Sue Moon, Wonjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.21815)  

**Abstract**: Understanding how media rhetoric shapes audience engagement is crucial in the attention economy. This study examines how moral emotional framing by mainstream news channels on YouTube influences user behavior across Korea and the United States. To capture the platform's multimodal nature, combining thumbnail images and video titles, we develop a multimodal moral emotion classifier by fine tuning a vision language model. The model is trained on human annotated multimodal datasets in both languages and applied to approximately 400,000 videos from major news outlets. We analyze engagement levels including views, likes, and comments, representing increasing degrees of commitment. The results show that other condemning rhetoric expressions of moral outrage that criticize others morally consistently increase all forms of engagement across cultures, with effects ranging from passive viewing to active commenting. These findings suggest that moral outrage is a particularly effective emotional strategy, attracting not only attention but also active participation. We discuss concerns about the potential misuse of other condemning rhetoric, as such practices may deepen polarization by reinforcing in group and out group divisions. To facilitate future research and ensure reproducibility, we publicly release our Korean and English multimodal moral emotion classifiers. 

---
# A Decomposable Forward Process in Diffusion Models for Time-Series Forecasting 

**Authors**: Francisco Caldas, Sahil Kumar, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2601.21812)  

**Abstract**: We introduce a model-agnostic forward diffusion process for time-series forecasting that decomposes signals into spectral components, preserving structured temporal patterns such as seasonality more effectively than standard diffusion. Unlike prior work that modifies the network architecture or diffuses directly in the frequency domain, our proposed method alters only the diffusion process itself, making it compatible with existing diffusion backbones (e.g., DiffWave, TimeGrad, CSDI). By staging noise injection according to component energy, it maintains high signal-to-noise ratios for dominant frequencies throughout the diffusion trajectory, thereby improving the recoverability of long-term patterns. This strategy enables the model to maintain the signal structure for a longer period in the forward process, leading to improved forecast quality. Across standard forecasting benchmarks, we show that applying spectral decomposition strategies, such as the Fourier or Wavelet transform, consistently improves upon diffusion models using the baseline forward process, with negligible computational overhead. The code for this paper is available at this https URL. 

---
# KID: Knowledge-Injected Dual-Head Learning for Knowledge-Grounded Harmful Meme Detection 

**Authors**: Yaocong Li, Leihan Zhang, Le Zhang, Qiang Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21796)  

**Abstract**: Internet memes have become pervasive carriers of digital culture on social platforms. However, their heavy reliance on metaphors and sociocultural context also makes them subtle vehicles for harmful content, posing significant challenges for automated content moderation. Existing approaches primarily focus on intra-modal and inter-modal signal analysis, while the understanding of implicit toxicity often depends on background knowledge that is not explicitly present in the meme itself. To address this challenge, we propose KID, a Knowledge-Injected Dual-Head Learning framework for knowledge-grounded harmful meme detection. KID adopts a label-constrained distillation paradigm to decompose complex meme understanding into structured reasoning chains that explicitly link visual evidence, background knowledge, and classification labels. These chains guide the learning process by grounding external knowledge in meme-specific contexts. In addition, KID employs a dual-head architecture that jointly optimizes semantic generation and classification objectives, enabling aligned linguistic reasoning while maintaining stable decision boundaries. Extensive experiments on five multilingual datasets spanning English, Chinese, and low-resource Bengali demonstrate that KID achieves SOTA performance on both binary and multi-label harmful meme detection tasks, improving over previous best methods by 2.1%--19.7% across primary evaluation metrics. Ablation studies further confirm the effectiveness of knowledge injection and dual-head joint learning, highlighting their complementary contributions to robust and generalizable meme understanding. The code and data are available at this https URL. 

---
# Effective LoRA Adapter Routing using Task Representations 

**Authors**: Akash Dhasade, Anne-Marie Kermarrec, Igor Pavlovic, Diana Petrescu, Rafael Pires, Mathis Randl, Martijn de Vos  

**Link**: [PDF](https://arxiv.org/pdf/2601.21795)  

**Abstract**: Low-rank adaptation (LoRA) enables parameter efficient specialization of large language models (LLMs) through modular adapters, resulting in rapidly growing public adapter pools spanning diverse tasks. Effectively using these adapters requires routing: selecting and composing the appropriate adapters for a query. We introduce LORAUTER, a novel routing framework that selects and composes LoRA adapters using task representations rather than adapter characteristics. Unlike existing approaches that map queries directly to adapters, LORAUTER routes queries via task embeddings derived from small validation sets and does not require adapter training data. By operating at the task level, LORAUTER achieves efficient routing that scales with the number of tasks rather than the number of adapters. Experiments across multiple tasks show that LORAUTER consistently outperforms baseline routing approaches, matching Oracle performance (101.2%) when task-aligned adapters exist and achieving state-of-the-art results on unseen tasks (+5.2 points). We further demonstrate the robustness of LORAUTER to very large, noisy adapter pools by scaling it to over 1500 adapters. 

---
# ECSEL: Explainable Classification via Signomial Equation Learning 

**Authors**: Adia Lumadjeng, Ilker Birbil, Erman Acar  

**Link**: [PDF](https://arxiv.org/pdf/2601.21789)  

**Abstract**: We introduce ECSEL, an explainable classification method that learns formal expressions in the form of signomial equations, motivated by the observation that many symbolic regression benchmarks admit compact signomial structure. ECSEL directly constructs a structural, closed-form expression that serves as both a classifier and an explanation. On standard symbolic regression benchmarks, our method recovers a larger fraction of target equations than competing state-of-the-art approaches while requiring substantially less computation. Leveraging this efficiency, ECSEL achieves classification accuracy competitive with established machine learning models without sacrificing interpretability. Further, we show that ECSEL satisfies some desirable properties regarding global feature behavior, decision-boundary analysis, and local feature attributions. Experiments on benchmark datasets and two real-world case studies i.e., e-commerce and fraud detection, demonstrate that the learned equations expose dataset biases, support counterfactual reasoning, and yield actionable insights. 

---
# Assessing the Business Process Modeling Competences of Large Language Models 

**Authors**: Chantale Lauer, Peter Pfeiffer, Alexander Rombach, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2601.21787)  

**Abstract**: The creation of Business Process Model and Notation (BPMN) models is a complex and time-consuming task requiring both domain knowledge and proficiency in modeling conventions. Recent advances in large language models (LLMs) have significantly expanded the possibilities for generating BPMN models directly from natural language, building upon earlier text-to-process methods with enhanced capabilities in handling complex descriptions. However, there is a lack of systematic evaluations of LLM-generated process models. Current efforts either use LLM-as-a-judge approaches or do not consider established dimensions of model quality. To this end, we introduce BEF4LLM, a novel LLM evaluation framework comprising four perspectives: syntactic quality, pragmatic quality, semantic quality, and validity. Using BEF4LLM, we conduct a comprehensive analysis of open-source LLMs and benchmark their performance against human modeling experts. Results indicate that LLMs excel in syntactic and pragmatic quality, while humans outperform in semantic aspects; however, the differences in scores are relatively modest, highlighting LLMs' competitive potential despite challenges in validity and semantic quality. The insights highlight current strengths and limitations of using LLMs for BPMN modeling and guide future model development and fine-tuning. Addressing these areas is essential for advancing the practical deployment of LLMs in business process modeling. 

---
# Synthetic-to-Real Domain Bridging for Single-View 3D Reconstruction of Ships for Maritime Monitoring 

**Authors**: Borja Carrillo-Perez, Felix Sattler, Angel Bueno Rodriguez, Maurice Stephan, Sarah Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2601.21786)  

**Abstract**: Three-dimensional (3D) reconstruction of ships is an important part of maritime monitoring, allowing improved visualization, inspection, and decision-making in real-world monitoring environments. However, most state-ofthe-art 3D reconstruction methods require multi-view supervision, annotated 3D ground truth, or are computationally intensive, making them impractical for real-time maritime deployment. In this work, we present an efficient pipeline for single-view 3D reconstruction of real ships by training entirely on synthetic data and requiring only a single view at inference. Our approach uses the Splatter Image network, which represents objects as sparse sets of 3D Gaussians for rapid and accurate reconstruction from single images. The model is first fine-tuned on synthetic ShapeNet vessels and further refined with a diverse custom dataset of 3D ships, bridging the domain gap between synthetic and real-world imagery. We integrate a state-of-the-art segmentation module based on YOLOv8 and custom preprocessing to ensure compatibility with the reconstruction network. Postprocessing steps include real-world scaling, centering, and orientation alignment, followed by georeferenced placement on an interactive web map using AIS metadata and homography-based mapping. Quantitative evaluation on synthetic validation data demonstrates strong reconstruction fidelity, while qualitative results on real maritime images from the ShipSG dataset confirm the potential for transfer to operational maritime settings. The final system provides interactive 3D inspection of real ships without requiring real-world 3D annotations. This pipeline provides an efficient, scalable solution for maritime monitoring and highlights a path toward real-time 3D ship visualization in practical applications. Interactive demo: this https URL. 

---
# CoFrGeNet: Continued Fraction Architectures for Language Generation 

**Authors**: Amit Dhurandhar, Vijil Chenthamarakshan, Dennis Wei, Tejaswini Pedapati, Karthikeyan Natesan Ramamurthy, Rahul Nair  

**Link**: [PDF](https://arxiv.org/pdf/2601.21766)  

**Abstract**: Transformers are arguably the preferred architecture for language generation. In this paper, inspired by continued fractions, we introduce a new function class for generative modeling. The architecture family implementing this function class is named CoFrGeNets - Continued Fraction Generative Networks. We design novel architectural components based on this function class that can replace Multi-head Attention and Feed-Forward Networks in Transformer blocks while requiring much fewer parameters. We derive custom gradient formulations to optimize the proposed components more accurately and efficiently than using standard PyTorch-based gradients. Our components are a plug-in replacement requiring little change in training or inference procedures that have already been put in place for Transformer-based models thus making our approach easy to incorporate in large industrial workflows. We experiment on two very different transformer architectures GPT2-xl (1.5B) and Llama3 (3.2B), where the former we pre-train on OpenWebText and GneissWeb, while the latter we pre-train on the docling data mix which consists of nine different datasets. Results show that the performance on downstream classification, Q\& A, reasoning and text understanding tasks of our models is competitive and sometimes even superior to the original models with $\frac{2}{3}$ to $\frac{1}{2}$ the parameters and shorter pre-training time. We believe that future implementations customized to hardware will further bring out the true potential of our architectures. 

---
# EWSJF: An Adaptive Scheduler with Hybrid Partitioning for Mixed-Workload LLM Inference 

**Authors**: Bronislav Sidik, Chaya Levi, Joseph Kampeas  

**Link**: [PDF](https://arxiv.org/pdf/2601.21758)  

**Abstract**: Serving Large Language Models (LLMs) under mixed workloads--short, latency-sensitive interactive queries alongside long, throughput-oriented batch requests--poses a fundamental scheduling challenge. Standard First-Come, First-Served (FCFS) policies suffer from severe head-of-line blocking, leading to high tail latency and underutilized hardware. We introduce EWSJF (Effective Workload-based Shortest Job First), an adaptive request-level scheduler that learns workload structure in real time to jointly improve fairness and throughput. EWSJF operates upstream of execution-level schedulers and integrates four components: (1) Refine-and-Prune, an unsupervised partitioning algorithm that discovers performance-homogeneous request groups; (2) Dynamic Queue Routing for assigning requests to these groups; (3) Density-Weighted Scoring, a context-aware prioritization function balancing urgency and fairness; and (4) Bayesian Meta-Optimization, which continuously tunes scoring and partitioning parameters based on live performance feedback. Implemented in vLLM, EWSJF improves end-to-end throughput by over 30% and reduces average Time-To-First-Token for short requests by up to 4x compared to FCFS. These results demonstrate that adaptive, learning-based request scheduling is a critical missing layer for efficient and responsive LLM serving. Implementation available at this https URL. 

---
# Temporal Sepsis Modeling: a Fully Interpretable Relational Way 

**Authors**: Vincent Lemaire, Nédra Meloulli, Pierre Jaquet  

**Link**: [PDF](https://arxiv.org/pdf/2601.21747)  

**Abstract**: Sepsis remains one of the most complex and heterogeneous syndromes in intensive care, characterized by diverse physiological trajectories and variable responses to treatment. While deep learning models perform well in the early prediction of sepsis, they often lack interpretability and ignore latent patient sub-phenotypes. In this work, we propose a machine learning framework by opening up a new avenue for addressing this issue: a relational approach. Temporal data from electronic medical records (EMRs) are viewed as multivariate patient logs and represented in a relational data schema. Then, a propositionalisation technique (based on classic aggregation/selection functions from the field of relational data) is applied to construct interpretable features to "flatten" the data. Finally, the flattened data is classified using a selective naive Bayesian classifier. Experimental validation demonstrates the relevance of the suggested approach as well as its extreme interpretability. The interpretation is fourfold: univariate, global, local, and counterfactual. 

---
# Why Adam Works Better with $β_1 = β_2$: The Missing Gradient Scale Invariance Principle 

**Authors**: Alberto Fernández-Hernández, Cristian Pérez-Corral, Jose I. Mestre, Manuel F. Dolz, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2601.21739)  

**Abstract**: Adam has been at the core of large-scale training for almost a decade, yet a simple empirical fact remains unaccounted for: both validation scores and the qualitative behaviour of the training runs improve when the momentum parameters satisfy $\beta_{1}=\beta_{2}$. Some recent studies have reported this pattern, but there is still no explanation for why this choice helps. We show that this choice is closely tied to a structural property that we refer to as \textit{gradient scale invariance}. We formalize this notion and prove that Adam becomes gradient scale invariant of first order if and only if $\beta_{1}=\beta_{2}$. This perspective places the balanced regime of Adam in direct alignment with the design principles underlying several recent optimizers that explicitly enforce scale-robust updates. The theory is supported by experiments across vision and language tasks, and across different architectural families, in which rescaling the gradient has a markedly smoother effect on the update when $\beta_{1}=\beta_{2}$. Overall, our results offer a coherent explanation for an open question in the behavior of Adam and provide a simple principle that helps guide the design of future optimizers. 

---
# From Global to Granular: Revealing IQA Model Performance via Correlation Surface 

**Authors**: Baoliang Chen, Danni Huang, Hanwei Zhu, Lingyu Zhu, Wei Zhou, Shiqi Wang, Yuming Fang, Weisi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21738)  

**Abstract**: Evaluation of Image Quality Assessment (IQA) models has long been dominated by global correlation metrics, such as Pearson Linear Correlation Coefficient (PLCC) and Spearman Rank-Order Correlation Coefficient (SRCC). While widely adopted, these metrics reduce performance to a single scalar, failing to capture how ranking consistency varies across the local quality spectrum. For example, two IQA models may achieve identical SRCC values, yet one ranks high-quality images (related to high Mean Opinion Score, MOS) more reliably, while the other better discriminates image pairs with small quality/MOS differences (related to $|\Delta$MOS$|$). Such complementary behaviors are invisible under global metrics. Moreover, SRCC and PLCC are sensitive to test-sample quality distributions, yielding unstable comparisons across test sets. To address these limitations, we propose \textbf{Granularity-Modulated Correlation (GMC)}, which provides a structured, fine-grained analysis of IQA performance. GMC includes: (1) a \textbf{Granularity Modulator} that applies Gaussian-weighted correlations conditioned on absolute MOS values and pairwise MOS differences ($|\Delta$MOS$|$) to examine local performance variations, and (2) a \textbf{Distribution Regulator} that regularizes correlations to mitigate biases from non-uniform quality distributions. The resulting \textbf{correlation surface} maps correlation values as a joint function of MOS and $|\Delta$MOS$|$, providing a 3D representation of IQA performance. Experiments on standard benchmarks show that GMC reveals performance characteristics invisible to scalar metrics, offering a more informative and reliable paradigm for analyzing, comparing, and deploying IQA models. Codes are available at this https URL. 

---
# Enhancing Language Models for Robust Greenwashing Detection 

**Authors**: Neil Heinrich Braun, Keane Ong, Rui Mao, Erik Cambria, Gianmarco Mengaldo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21722)  

**Abstract**: Sustainability reports are critical for ESG assessment, yet greenwashing and vague claims often undermine their reliability. Existing NLP models lack robustness to these practices, typically relying on surface-level patterns that generalize poorly. We propose a parameter-efficient framework that structures LLM latent spaces by combining contrastive learning with an ordinal ranking objective to capture graded distinctions between concrete actions and ambiguous claims. Our approach incorporates gated feature modulation to filter disclosure noise and utilizes MetaGradNorm to stabilize multi-objective optimization. Experiments in cross-category settings demonstrate superior robustness over standard baselines while revealing a trade-off between representational rigidity and generalization. 

---
# When does predictive inverse dynamics outperform behavior cloning? 

**Authors**: Lukas Schäfer, Pallavi Choudhury, Abdelhak Lemkhenter, Chris Lovett, Somjit Nath, Luis França, Matheus Ribeiro Furtado de Mendonça, Alex Lamb, Riashat Islam, Siddhartha Sen, John Langford, Katja Hofmann, Sergio Valcarcel Macua  

**Link**: [PDF](https://arxiv.org/pdf/2601.21718)  

**Abstract**: Behavior cloning (BC) is a practical offline imitation learning method, but it often fails when expert demonstrations are limited. Recent works have introduced a class of architectures named predictive inverse dynamics models (PIDM) that combine a future state predictor with an inverse dynamics model (IDM). While PIDM often outperforms BC, the reasons behind its benefits remain unclear. In this paper, we provide a theoretical explanation: PIDM introduces a bias-variance tradeoff. While predicting the future state introduces bias, conditioning the IDM on the prediction can significantly reduce variance. We establish conditions on the state predictor bias for PIDM to achieve lower prediction error and higher sample efficiency than BC, with the gap widening when additional data sources are available. We validate the theoretical insights empirically in 2D navigation tasks, where BC requires up to five times (three times on average) more demonstrations than PIDM to reach comparable performance; and in a complex 3D environment in a modern video game with high-dimensional visual inputs and stochastic transitions, where BC requires over 66\% more samples than PIDM. 

---
# DreamActor-M2: Universal Character Image Animation via Spatiotemporal In-Context Learning 

**Authors**: Mingshuang Luo, Shuang Liang, Zhengkun Rong, Yuxuan Luo, Tianshu Hu, Ruibing Hou, Hong Chang, Yong Li, Yuan Zhang, Mingyuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21716)  

**Abstract**: Character image animation aims to synthesize high-fidelity videos by transferring motion from a driving sequence to a static reference image. Despite recent advancements, existing methods suffer from two fundamental challenges: (1) suboptimal motion injection strategies that lead to a trade-off between identity preservation and motion consistency, manifesting as a "see-saw", and (2) an over-reliance on explicit pose priors (e.g., skeletons), which inadequately capture intricate dynamics and hinder generalization to arbitrary, non-humanoid characters. To address these challenges, we present DreamActor-M2, a universal animation framework that reimagines motion conditioning as an in-context learning problem. Our approach follows a two-stage paradigm. First, we bridge the input modality gap by fusing reference appearance and motion cues into a unified latent space, enabling the model to jointly reason about spatial identity and temporal dynamics by leveraging the generative prior of foundational models. Second, we introduce a self-bootstrapped data synthesis pipeline that curates pseudo cross-identity training pairs, facilitating a seamless transition from pose-dependent control to direct, end-to-end RGB-driven animation. This strategy significantly enhances generalization across diverse characters and motion scenarios. To facilitate comprehensive evaluation, we further introduce AW Bench, a versatile benchmark encompassing a wide spectrum of characters types and motion scenarios. Extensive experiments demonstrate that DreamActor-M2 achieves state-of-the-art performance, delivering superior visual fidelity and robust cross-domain generalization. Project Page: this https URL 

---
# Disentangling perception and reasoning for improving data efficiency in learning cloth manipulation without demonstrations 

**Authors**: Donatien Delehelle, Fei Chen, Darwin Caldwell  

**Link**: [PDF](https://arxiv.org/pdf/2601.21713)  

**Abstract**: Cloth manipulation is a ubiquitous task in everyday life, but it remains an open challenge for robotics. The difficulties in developing cloth manipulation policies are attributed to the high-dimensional state space, complex dynamics, and high propensity to self-occlusion exhibited by fabrics. As analytical methods have not been able to provide robust and general manipulation policies, reinforcement learning (RL) is considered a promising approach to these problems. However, to address the large state space and complex dynamics, data-based methods usually rely on large models and long training times. The resulting computational cost significantly hampers the development and adoption of these methods. Additionally, due to the challenge of robust state estimation, garment manipulation policies often adopt an end-to-end learning approach with workspace images as input. While this approach enables a conceptually straightforward sim-to-real transfer via real-world fine-tuning, it also incurs a significant computational cost by training agents on a highly lossy representation of the environment state. This paper questions this common design choice by exploring an efficient and modular approach to RL for cloth manipulation. We show that, through careful design choices, model size and training time can be significantly reduced when learning in simulation. Furthermore, we demonstrate how the resulting simulation-trained model can be transferred to the real world. We evaluate our approach on the SoftGym benchmark and achieve significant performance improvements over available baselines on our task, while using a substantially smaller model. 

---
# TACLer: Tailored Curriculum Reinforcement Learning for Efficient Reasoning 

**Authors**: Huiyuan Lai, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2601.21711)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance on complex reasoning tasks, especially when equipped with long chain-of-thought (CoT) reasoning. However, eliciting long CoT typically requires large-scale reinforcement learning (RL) training, while often leading to overthinking with redundant intermediate steps. To improve learning and reasoning efficiency, while preserving or even enhancing performance, we propose TACLer, a model-tailored curriculum reinforcement learning framework that gradually increases the complexity of the data based on the model's proficiency in multi-stage RL training. TACLer features two core components: (i) tailored curriculum learning that determines what knowledge the model lacks and needs to learn in progressive stages; (ii) a hybrid Thinking/NoThinking reasoning paradigm that balances accuracy and efficiency by enabling or disabling the Thinking mode. Our experiments show that TACLer yields a twofold advantage in learning and reasoning: (i) it reduces computational cost, cutting training compute by over 50% compared to long thinking models and reducing inference token usage by over 42% relative to the base model; and (ii) it improves accuracy by over 9% on the base model, consistently outperforming state-of-the-art Nothinking and Thinking baselines across four math datasets with complex problems. 

---
# Toward Culturally Aligned LLMs through Ontology-Guided Multi-Agent Reasoning 

**Authors**: Wonduk Seo, Wonseok Choi, Junseo Koh, Juhyeon Lee, Hyunjin An, Minhyeong Yu, Jian Park, Qingshan Zhou, Seunghyun Lee, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21700)  

**Abstract**: Large Language Models (LLMs) increasingly support culturally sensitive decision making, yet often exhibit misalignment due to skewed pretraining data and the absence of structured value representations. Existing methods can steer outputs, but often lack demographic grounding and treat values as independent, unstructured signals, reducing consistency and interpretability. We propose OG-MAR, an Ontology-Guided Multi-Agent Reasoning framework. OG-MAR summarizes respondent-specific values from the World Values Survey (WVS) and constructs a global cultural ontology by eliciting relations over a fixed taxonomy via competency questions. At inference time, it retrieves ontology-consistent relations and demographically similar profiles to instantiate multiple value-persona agents, whose outputs are synthesized by a judgment agent that enforces ontology consistency and demographic proximity. Experiments on regional social-survey benchmarks across four LLM backbones show that OG-MAR improves cultural alignment and robustness over competitive baselines, while producing more transparent reasoning traces. 

---
# Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics 

**Authors**: Mohamed Elgaar, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2601.21698)  

**Abstract**: Curriculum learning changes the order of pre-training data, but it remains unclear whether it changes the learning trajectory or mainly reorders exposure over a fixed trajectory. We train Pythia models (14M-410M parameters) for 300B tokens under three linguistically motivated curricula-Age-of-Acquisition, word frequency, and Verb Variation (VV)-and compare each against Random ordering; at 1B parameters we compare Random and VV. Across orderings, training follows a shared sequence of latent phases, while curricula mainly change within-phase data exposure. In smaller models (up to 160M parameters), Random ordering exhibits higher gradient noise and stronger late-training output-head spectral saturation, alongside lower final accuracy; curricula reduce both effects at matched compute. At larger scales, saturation differences are smaller and curriculum gains shrink. We formalize the link between difficulty pacing and optimization stability in an idealized analysis based on gradient-variance control, and our results point to a practical takeaway: curricula help by stabilizing within-phase optimization rather than by creating new phases. 

---
# XFACTORS: Disentangled Information Bottleneck via Contrastive Supervision 

**Authors**: Alexandre Myara, Nicolas Bourriez, Thomas Boyer, Thomas Lemercier, Ihab Bendidi, Auguste Genovesio  

**Link**: [PDF](https://arxiv.org/pdf/2601.21688)  

**Abstract**: Disentangled representation learning aims to map independent factors of variation to independent representation components. On one hand, purely unsupervised approaches have proven successful on fully disentangled synthetic data, but fail to recover semantic factors from real data without strong inductive biases. On the other hand, supervised approaches are unstable and hard to scale to large attribute sets because they rely on adversarial objectives or auxiliary classifiers.
We introduce \textsc{XFactors}, a weakly-supervised VAE framework that disentangles and provides explicit control over a chosen set of factors. Building on the Disentangled Information Bottleneck perspective, we decompose the representation into a residual subspace $\mathcal{S}$ and factor-specific subspaces $\mathcal{T}_1,\ldots,\mathcal{T}_K$ and a residual subspace $\mathcal{S}$. Each target factor is encoded in its assigned $\mathcal{T}_i$ through contrastive supervision: an InfoNCE loss pulls together latents sharing the same factor value and pushes apart mismatched pairs. In parallel, KL regularization imposes a Gaussian structure on both $\mathcal{S}$ and the aggregated factor subspaces, organizing the geometry without additional supervision for non-targeted factors and avoiding adversarial training and classifiers.
Across multiple datasets, with constant hyperparameters, \textsc{XFactors} achieves state-of-the-art disentanglement scores and yields consistent qualitative factor alignment in the corresponding subspaces, enabling controlled factor swapping via latent replacement. We further demonstrate that our method scales correctly with increasing latent capacity and evaluate it on the real-world dataset CelebA. Our code is available at \href{this https URL}{this http URL}. 

---
# FIT: Defying Catastrophic Forgetting in Continual LLM Unlearning 

**Authors**: Xiaoyu Xu, Minxin Du, Kun Fang, Zi Liang, Yaxin Xiao, Zhicong Huang, Cheng Hong, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21682)  

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities across diverse tasks but raise concerns about privacy, copyright, and harmful materials. Existing LLM unlearning methods rarely consider the continual and high-volume nature of real-world deletion requests, which can cause utility degradation and catastrophic forgetting as requests accumulate. To address this challenge, we introduce \fit, a framework for continual unlearning that handles large numbers of deletion requests while maintaining robustness against both catastrophic forgetting and post-unlearning recovery. \fit mitigates degradation through rigorous data \underline{F}iltering, \underline{I}mportance-aware updates, and \underline{T}argeted layer attribution, enabling stable performance across long sequences of unlearning operations and achieving a favorable balance between forgetting effectiveness and utility retention. To support realistic evaluation, we present \textbf{PCH}, a benchmark covering \textbf{P}ersonal information, \textbf{C}opyright, and \textbf{H}armful content in sequential deletion scenarios, along with two symmetric metrics, Forget Degree (F.D.) and Retain Utility (R.U.), which jointly assess forgetting quality and utility preservation. Extensive experiments on four open-source LLMs with hundreds of deletion requests show that \fit achieves the strongest trade-off between F.D. and R.U., surpasses existing methods on MMLU, CommonsenseQA, and GSM8K, and remains resistant against both relearning and quantization recovery attacks. 

---
# Expected Return Causes Outcome-Level Mode Collapse in Reinforcement Learning and How to Fix It with Inverse Probability Scaling 

**Authors**: Abhijeet Sinha, Sundari Elango, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21669)  

**Abstract**: Many reinforcement learning (RL) problems admit multiple terminal solutions of comparable quality, where the goal is not to identify a single optimum but to represent a diverse set of high-quality outcomes. Nevertheless, policies trained by standard expected return maximization routinely collapse onto a small subset of outcomes, a phenomenon commonly attributed to insufficient exploration or weak regularization. We show that this explanation is incomplete: outcome level mode collapse is a structural consequence of the expected-return objective itself. Under idealized learning dynamics, the log-probability ratio between any two outcomes evolves linearly in their reward difference, implying exponential ratio divergence and inevitable collapse independent of the exploration strategy, entropy regularization, or optimization algorithm. We identify the source of this pathology as the probability multiplier inside the expectation and propose a minimal correction: inverse probability scaling, which removes outcome-frequency amplification from the learning signal, fundamentally changes the learning dynamics, and provably yields reward-proportional terminal distributions, preventing collapse in multimodal settings. We instantiate this principle in Group Relative Policy Optimization (GRPO) as a drop-in modification, IPS-GRPO, requiring no auxiliary models or architectural changes. Across different reasoning and molecular generation tasks, IPS-GRPO consistently reduces outcome-level mode collapse while matching or exceeding baseline performance, suggesting that correcting the objective rather than adding exploration heuristics is key to reliable multimodal policy optimization. 

---
# SENDAI: A Hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework 

**Authors**: Xingyue Zhang, Yuxuan Bao, Mars Liyao Gao, J. Nathan Kutz  

**Link**: [PDF](https://arxiv.org/pdf/2601.21664)  

**Abstract**: Bridging the gap between data-rich training regimes and observation-sparse deployment conditions remains a central challenge in spatiotemporal field reconstruction, particularly when target domains exhibit distributional shifts, heterogeneous structure, and multi-scale dynamics absent from available training data. We present SENDAI, a hierarchical Sparse-measurement, EfficieNt Data AssImilation Framework that reconstructs full spatial states from hyper sparse sensor observations by combining simulation-derived priors with learned discrepancy corrections. We demonstrate the performance on satellite remote sensing, reconstructing MODIS (Moderate Resolution Imaging Spectroradiometer) derived vegetation index fields across six globally distributed sites. Using seasonal periods as a proxy for domain shift, the framework consistently outperforms established baselines that require substantially denser observations -- SENDAI achieves a maximum SSIM improvement of 185% over traditional baselines and a 36% improvement over recent high-frequency-based methods. These gains are particularly pronounced for landscapes with sharp boundaries and sub-seasonal dynamics; more importantly, the framework effectively preserves diagnostically relevant structures -- such as field topologies, land cover discontinuities, and spatial gradients. By yielding corrections that are more structurally and spectrally separable, the reconstructed fields are better suited for downstream inference of indirectly observed variables. The results therefore highlight a lightweight and operationally viable framework for sparse-measurement reconstruction that is applicable to physically grounded inference, resource-limited deployment, and real-time monitor and control. 

---
# Gauge-invariant representation holonomy 

**Authors**: Vasileios Sevetlidis, George Pavlidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.21653)  

**Abstract**: Deep networks learn internal representations whose geometry--how features bend, rotate, and evolve--affects both generalization and robustness. Existing similarity measures such as CKA or SVCCA capture pointwise overlap between activation sets, but miss how representations change along input paths. Two models may appear nearly identical under these metrics yet respond very differently to perturbations or adversarial stress. We introduce representation holonomy, a gauge-invariant statistic that measures this path dependence. Conceptually, holonomy quantifies the "twist" accumulated when features are parallel-transported around a small loop in input space: flat representations yield zero holonomy, while nonzero values reveal hidden curvature. Our estimator fixes gauge through global whitening, aligns neighborhoods using shared subspaces and rotation-only Procrustes, and embeds the result back to the full feature space. We prove invariance to orthogonal (and affine, post-whitening) transformations, establish a linear null for affine layers, and show that holonomy vanishes at small radii. Empirically, holonomy increases with loop radius, separates models that appear similar under CKA, and correlates with adversarial and corruption robustness. It also tracks training dynamics as features form and stabilize. Together, these results position representation holonomy as a practical and scalable diagnostic for probing the geometric structure of learned representations beyond pointwise similarity. 

---
# When Life Gives You AI, Will You Turn It Into A Market for Lemons? Understanding How Information Asymmetries About AI System Capabilities Affect Market Outcomes and Adoption 

**Authors**: Alexander Erlei, Federico Cau, Radoslav Georgiev, Sagar Kumar, Kilian Bizer, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.21650)  

**Abstract**: AI consumer markets are characterized by severe buyer-supplier market asymmetries. Complex AI systems can appear highly accurate while making costly errors or embedding hidden defects. While there have been regulatory efforts surrounding different forms of disclosure, large information gaps remain. This paper provides the first experimental evidence on the important role of information asymmetries and disclosure designs in shaping user adoption of AI systems. We systematically vary the density of low-quality AI systems and the depth of disclosure requirements in a simulated AI product market to gauge how people react to the risk of accidentally relying on a low-quality AI system. Then, we compare participants' choices to a rational Bayesian model, analyzing the degree to which partial information disclosure can improve AI adoption. Our results underscore the deleterious effects of information asymmetries on AI adoption, but also highlight the potential of partial disclosure designs to improve the overall efficiency of human decision-making. 

---
# SWE-Spot: Building Small Repo-Experts with Repository-Centric Learning 

**Authors**: Jinjun Peng, Magnus Saebo, Tianjun Zhong, Yi-Jie Cheng, Junfeng Yang, Baishakhi Ray, Simin Chen, Yangruibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.21649)  

**Abstract**: The deployment of coding agents in privacy-sensitive and resource-constrained environments drives the demand for capable open-weight Small Language Models (SLMs). However, they suffer from a fundamental capability gap: unlike frontier large models, they lack the inference-time strong generalization to work with complicated, unfamiliar codebases. We identify that the prevailing Task-Centric Learning (TCL) paradigm, which scales exposure across disparate repositories, fails to address this limitation. In response, we propose Repository-Centric Learning (RCL), a paradigm shift that prioritizes vertical repository depth over horizontal task breadth, suggesting SLMs must internalize the "physics" of a target software environment through parametric knowledge acquisition, rather than attempting to recover it via costly inference-time search. Following this new paradigm, we design a four-unit Repository-Centric Experience, transforming static codebases into interactive learning signals, to train SWE-Spot-4B, a family of highly compact models built as repo-specialized experts that breaks established scaling trends, outperforming open-weight models up to larger (e.g., CWM by Meta, Qwen3-Coder-30B) and surpassing/matching efficiency-focused commercial models (e.g., GPT-4.1-mini, GPT-5-nano) across multiple SWE tasks. Further analysis reveals that RCL yields higher training sample efficiency and lower inference costs, emphasizing that for building efficient intelligence, repository mastery is a distinct and necessary dimension that complements general coding capability. 

---
# ILRR: Inference-Time Steering Method for Masked Diffusion Language Models 

**Authors**: Eden Avrahami, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2601.21647)  

**Abstract**: Discrete Diffusion Language Models (DLMs) offer a promising non-autoregressive alternative for text generation, yet effective mechanisms for inference-time control remain relatively underexplored. Existing approaches include sampling-level guidance procedures or trajectory optimization mechanisms. In this work, we introduce Iterative Latent Representation Refinement (ILRR), a learning-free framework for steering DLMs using a single reference sequence. ILRR guides generation by dynamically aligning the internal activations of the generated sequence with those of a given reference throughout the denoising process. This approach captures and transfers high-level semantic properties, with a tunable steering scale enabling flexible control over attributes such as sentiment. We further introduce Spatially Modulated Steering, an extension that enables steering long texts using shorter references by regulating guidance intensity across the sequence. Empirically, we demonstrate that ILRR achieves effective attribute steering on LLaDA and MDLM architectures with a minor computational overhead, requiring only one additional parallel forward pass per denoising step. Under the same compute budget, ILRR improves attribute accuracy over comparable baselines by 10$\%$ to 60$\%$ points, while maintaining high generation quality. 

---
# Seg-MoE: Multi-Resolution Segment-wise Mixture-of-Experts for Time Series Forecasting Transformers 

**Authors**: Evandro S. Ortigossa, Eran Segal  

**Link**: [PDF](https://arxiv.org/pdf/2601.21641)  

**Abstract**: Transformer-based models have recently made significant advances in accurate time-series forecasting, but even these architectures struggle to scale efficiently while capturing long-term temporal dynamics. Mixture-of-Experts (MoE) layers are a proven solution to scaling problems in natural language processing. However, existing MoE approaches for time-series forecasting rely on token-wise routing mechanisms, which may fail to exploit the natural locality and continuity of temporal data. In this work, we introduce Seg-MoE, a sparse MoE design that routes and processes contiguous time-step segments rather than making independent expert decisions. Token segments allow each expert to model intra-segment interactions directly, naturally aligning with inherent temporal patterns. We integrate Seg-MoE layers into a time-series Transformer and evaluate it on multiple multivariate long-term forecasting benchmarks. Seg-MoE consistently achieves state-of-the-art forecasting accuracy across almost all prediction horizons, outperforming both dense Transformers and prior token-wise MoE models. Comprehensive ablation studies confirm that segment-level routing is the key factor driving these gains. Our results show that aligning the MoE routing granularity with the inherent structure of time series provides a powerful, yet previously underexplored, inductive bias, opening new avenues for conditionally sparse architectures in sequential data modeling. 

---
# HeRo-Q: A General Framework for Stable Low Bit Quantization via Hessian Conditioning 

**Authors**: Jinhao Zhang Yunquan Zhang, Zicheng yan, Boyang Zhang, Jun Sun, Daning Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.21626)  

**Abstract**: Post Training Quantization (PTQ), a mainstream model compression technique, often leads to the paradoxical 'low error, high loss' phenomenon because it focuses solely on minimizing quantization error. The root cause lies in the Hessian matrix of the LLM loss landscape: a few high curvature directions are extremely sensitive to perturbations. To address this, we propose the Hessian Robust Quantization (HeRo Q) algorithm, which applies a lightweight, learnable rotation-compression matrix to the weight space prior to quantization. This joint framework reshapes the loss landscape by reducing the largest Hessian eigenvalue and reducing its max eigenvalue, thereby significantly enhancing robustness to quantization noise. HeRo-Q requires no architectural modifications, incurs negligible computational overhead, and integrates seamlessly into existing PTQ pipelines. Experiments on Llama and Qwen models show that HeRo Q consistently outperforms state of the art methods including GPTQ, AWQ, and SpinQuant not only achieving superior performance under standard W4A8 settings, but also excelling in the highly challenging W3A16 ultra low bit regime, where it boosts GSM8K accuracy on Llama3 8B to 70.15\% and effectively avoids the logical collapse commonly seen in aggressive quantization. 

---
# Breaking the Overscaling Curse: Thinking Parallelism Before Parallel Thinking 

**Authors**: Yiming Wang, Zhuosheng Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21619)  

**Abstract**: Parallel thinking enhances LLM reasoning by multi-path sampling and aggregation. In system-level evaluations, a global parallelism level N is allocated to all samples, typically set large to maximize overall dataset accuracy. However, due to sample heterogeneity, some samples can achieve comparable performance with a smaller N'< N, causing budget redundancy. This incompatibility between system-level efficacy and sample-level efficiency constitutes the overscaling curse. In this paper, we formalize and quantify the overscaling curse, showing its universality and severity in practice, and analyze its trigger mechanism. We then propose a lightweight method, T2, to break the overscaling curse, which utilizes latent representations to estimate the optimal parallelism level for each sample before decoding. Experiments show that T2 significantly reduces cost while maintaining comparable performance, enabling more efficient parallel thinking. 

---
# Beyond Parameter Finetuning: Test-Time Representation Refinement for Node Classification 

**Authors**: Jiaxin Zhang, Yiqi Wang, Siwei Wang, Xihong Yang, Yu Shi, Xinwang Liu, En Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21615)  

**Abstract**: Graph Neural Networks frequently exhibit significant performance degradation in the out-of-distribution test scenario. While test-time training (TTT) offers a promising solution, existing Parameter Finetuning (PaFT) paradigm suffer from catastrophic forgetting, hindering their real-world applicability. We propose TTReFT, a novel Test-Time Representation FineTuning framework that transitions the adaptation target from model parameters to latent representations. Specifically, TTReFT achieves this through three key innovations: (1) uncertainty-guided node selection for specific interventions, (2) low-rank representation interventions that preserve pre-trained knowledge, and (3) an intervention-aware masked autoencoder that dynamically adjust masking strategy to accommodate the node selection scheme. Theoretically, we establish guarantees for TTReFT in OOD settings. Empirically, extensive experiments across five benchmark datasets demonstrate that TTReFT achieves consistent and superior performance. Our work establishes representation finetuning as a new paradigm for graph TTT, offering both theoretical grounding and immediate practical utility for real-world deployment. 

---
# Representation-Regularized Convolutional Audio Transformer for Audio Understanding 

**Authors**: Bing Han, Chushu Zhou, Yifan Yang, Wei Wang, Chenda Li, Wangyou Zhang, Yanmin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.21612)  

**Abstract**: Bootstrap-based Self-Supervised Learning (SSL) has achieved remarkable progress in audio understanding. However, existing methods typically operate at a single level of granularity, limiting their ability to model the diverse temporal and spectral structures inherent in complex audio signals. Furthermore, bootstrapping representations from scratch is computationally expensive, often requiring extensive training to converge. In this work, we propose the Convolutional Audio Transformer (CAT), a unified framework designed to address these challenges. First, to capture hierarchical audio features, CAT incorporates a Multi-resolution Block that aggregates information across varying granularities. Second, to enhance training efficiency, we introduce a Representation Regularization objective. Drawing inspiration from generative modeling, this auxiliary task guides the student model by aligning its predictions with high-quality semantic representations from frozen, pre-trained external encoders. Experimental results demonstrate that CAT significantly outperforms baselines on audio understanding benchmarks. Notably, it achieves competitive performance on the AudioSet 20k dataset with 5 times faster convergence than existing methods. Codes and checkpoints will be released soon at this https URL. 

---
# Thinking Broad, Acting Fast: Latent Reasoning Distillation from Multi-Perspective Chain-of-Thought for E-Commerce Relevance 

**Authors**: Baopu Qiu, Hao Chen, Yuanrong Wu, Changtong Zan, Chao Wei, Weiru Zhang, Xiaoyi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2601.21611)  

**Abstract**: Effective relevance modeling is crucial for e-commerce search, as it aligns search results with user intent and enhances customer experience. Recent work has leveraged large language models (LLMs) to address the limitations of traditional relevance models, especially for long-tail and ambiguous queries. By incorporating Chain-of-Thought (CoT) reasoning, these approaches improve both accuracy and interpretability through multi-step reasoning. However, two key limitations remain: (1) most existing approaches rely on single-perspective CoT reasoning, which fails to capture the multifaceted nature of e-commerce relevance (e.g., user intent vs. attribute-level matching vs. business-specific rules); and (2) although CoT-enhanced LLM's offer rich reasoning capabilities, their high inference latency necessitates knowledge distillation for real-time deployment, yet current distillation methods discard the CoT rationale structure at inference, using it as a transient auxiliary signal and forfeiting its reasoning utility. To address these challenges, we propose a novel framework that better exploits CoT semantics throughout the optimization pipeline. Specifically, the teacher model leverages Multi-Perspective CoT (MPCoT) to generate diverse rationales and combines Supervised Fine-Tuning (SFT) with Direct Preference Optimization (DPO) to construct a more robust reasoner. For distillation, we introduce Latent Reasoning Knowledge Distillation (LRKD), which endows a student model with a lightweight inference-time latent reasoning extractor, allowing efficient and low-latency internalization of the LLM's sophisticated reasoning capabilities. Evaluated in offline experiments and online A/B tests on an e-commerce search advertising platform serving tens of millions of users daily, our method delivers significant offline gains, showing clear benefits in both commercial performance and user experience. 

---
# Dynamics Reveals Structure: Challenging the Linear Propagation Assumption 

**Authors**: Hoyeon Chang, Bálint Mucsányi, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2601.21601)  

**Abstract**: Neural networks adapt through first-order parameter updates, yet it remains unclear whether such updates preserve logical coherence. We investigate the geometric limits of the Linear Propagation Assumption (LPA), the premise that local updates coherently propagate to logical consequences. To formalize this, we adopt relation algebra and study three core operations on relations: negation flips truth values, converse swaps argument order, and composition chains relations. For negation and converse, we prove that guaranteeing direction-agnostic first-order propagation necessitates a tensor factorization separating entity-pair context from relation content. However, for composition, we identify a fundamental obstruction. We show that composition reduces to conjunction, and prove that any conjunction well-defined on linear features must be bilinear. Since bilinearity is incompatible with negation, this forces the feature map to collapse. These results suggest that failures in knowledge editing, the reversal curse, and multi-hop reasoning may stem from common structural limitations inherent to the LPA. 

---
# Scalable Power Sampling: Unlocking Efficient, Training-Free Reasoning for LLMs via Distribution Sharpening 

**Authors**: Xiaotong Ji, Rasul Tutunov, Matthieu Zimmer, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2601.21590)  

**Abstract**: Reinforcement learning (RL) post-training is a dominant approach for improving the reasoning performance of large language models (LLMs), yet growing evidence suggests that its gains arise primarily from distribution sharpening rather than the acquisition of new capabilities. Recent work has shown that sampling from the power distribution of LLMs using Markov chain Monte Carlo (MCMC) can recover performance comparable to RL post-training without relying on external rewards; however, the high computational cost of MCMC makes such approaches impractical for widespread adoption. In this work, we propose a theoretically grounded alternative that eliminates the need for iterative MCMC. We derive a novel formulation showing that the global power distribution can be approximated by a token-level scaled low-temperature one, where the scaling factor captures future trajectory quality. Leveraging this insight, we introduce a training-free and verifier-free algorithm that sharpens the base model's generative distribution autoregressively. Empirically, we evaluate our method on math, QA, and code tasks across four LLMs, and show that our method matches or surpasses one-shot GRPO without relying on any external rewards, while reducing inference latency by over 10x compared to MCMC-based sampling. 

---
# Signal-Adaptive Trust Regions for Gradient-Free Optimization of Recurrent Spiking Neural Networks 

**Authors**: Jinhao Li, Yuhao Sun, Zhiyuan Ma, Hao He, Xinche Zhang, Xing Chen, Jin Li, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.21572)  

**Abstract**: Recurrent spiking neural networks (RSNNs) are a promising substrate for energy-efficient control policies, but training them for high-dimensional, long-horizon reinforcement learning remains challenging. Population-based, gradient-free optimization circumvents backpropagation through non-differentiable spike dynamics by estimating gradients. However, with finite populations, high variance of these estimates can induce harmful and overly aggressive update steps. Inspired by trust-region methods in reinforcement learning that constrain policy updates in distribution space, we propose \textbf{Signal-Adaptive Trust Regions (SATR)}, a distributional update rule that constrains relative change by bounding KL divergence normalized by an estimated signal energy. SATR automatically expands the trust region under strong signals and contracts it when updates are noise-dominated. We instantiate SATR for Bernoulli connectivity distributions, which have shown strong empirical performance for RSNN optimization. Across a suite of high-dimensional continuous-control benchmarks, SATR improves stability under limited populations and reaches competitive returns against strong baselines including PPO-LSTM. In addition, to make SATR practical at scale, we introduce a bitset implementation for binary spiking and binary weights, substantially reducing wall-clock training time and enabling fast RSNN policy search. 

---
# Shaping capabilities with token-level data filtering 

**Authors**: Neil Rathi, Alec Radford  

**Link**: [PDF](https://arxiv.org/pdf/2601.21571)  

**Abstract**: Current approaches to reducing undesired capabilities in language models are largely post hoc, and can thus be easily bypassed by adversaries. A natural alternative is to shape capabilities during pretraining itself. On the proxy task of removing medical capabilities, we show that the simple intervention of filtering pretraining data is highly effective, robust, and inexpensive at scale. Inspired by work on data attribution, we show that filtering tokens is more effective than filtering documents, achieving the same hit to undesired capabilities at a lower cost to benign ones. Training models spanning two orders of magnitude, we then demonstrate that filtering gets more effective with scale: for our largest models, token filtering leads to a 7000x compute slowdown on the forget domain. We also show that models trained with token filtering can still be aligned on the forget domain. Along the way, we introduce a methodology for labeling tokens with sparse autoencoders and distilling cheap, high-quality classifiers. We also demonstrate that filtering can be robust to noisy labels with sufficient pretraining compute. 

---
# SAL: Selective Adaptive Learning for Backpropagation-Free Training with Sparsification 

**Authors**: Fanping Liu, Hua Yang, Jiasi Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21561)  

**Abstract**: Standard deep learning relies on Backpropagation (BP), which is constrained by biologically implausible weight symmetry and suffers from significant gradient interference within dense representations. To mitigate these bottlenecks, we propose Selective Adaptive Learning (SAL), a training method that combines selective parameter activation with adaptive area partitioning. Specifically, SAL decomposes the parameter space into mutually exclusive, sample-dependent regions. This decoupling mitigates gradient interference across divergent semantic patterns and addresses explicit weight symmetry requirements through our refined feedback alignment. Empirically, SAL demonstrates competitive convergence rates, leading to improved classification performance across 10 standard benchmarks. Additionally, SAL achieves numerical consistency and competitive accuracy even in deep regimes (up to 128 layers) and large-scale models (up to 1B parameters). Our approach is loosely inspired by biological learning mechanisms, offering a plausible alternative that contributes to the study of scalable neural network training. 

---
# Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning 

**Authors**: Irene Ambrosini, Ingo Blakowski, Dmitrii Zendrikov, Cristiano Capone, Luna Gava, Giacomo Indiveri, Chiara De Luca, Chiara Bartolozzi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21548)  

**Abstract**: Air hockey demands split-second decisions at high puck velocities, a challenge we address with a compact network of spiking neurons running on a mixed-signal analog/digital neuromorphic processor. By co-designing hardware and learning algorithms, we train the system to achieve successful puck interactions through reinforcement learning in a remarkably small number of trials. The network leverages fixed random connectivity to capture the task's temporal structure and adopts a local e-prop learning rule in the readout layer to exploit event-driven activity for fast and efficient learning. The result is real-time learning with a setup comprising a computer and the neuromorphic chip in-the-loop, enabling practical training of spiking neural networks for robotic autonomous systems. This work bridges neuroscience-inspired hardware with real-world robotic control, showing that brain-inspired approaches can tackle fast-paced interaction tasks while supporting always-on learning in intelligent machines. 

---
# Multi-Modal Time Series Prediction via Mixture of Modulated Experts 

**Authors**: Lige Zhang, Ali Maatouk, Jialin Chen, Leandros Tassiulas, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2601.21547)  

**Abstract**: Real-world time series exhibit complex and evolving dynamics, making accurate forecasting extremely challenging. Recent multi-modal forecasting methods leverage textual information such as news reports to improve prediction, but most rely on token-level fusion that mixes temporal patches with language tokens in a shared embedding space. However, such fusion can be ill-suited when high-quality time-text pairs are scarce and when time series exhibit substantial variation in scale and characteristics, thus complicating cross-modal alignment. In parallel, Mixture-of-Experts (MoE) architectures have proven effective for both time series modeling and multi-modal learning, yet many existing MoE-based modality integration methods still depend on token-level fusion. To address this, we propose Expert Modulation, a new paradigm for multi-modal time series prediction that conditions both routing and expert computation on textual signals, enabling direct and efficient cross-modal control over expert behavior. Through comprehensive theoretical analysis and experiments, our proposed method demonstrates substantial improvements in multi-modal time series prediction. The current code is available at this https URL 

---
# Bi-Anchor Interpolation Solver for Accelerating Generative Modeling 

**Authors**: Hongxu Chen, Hongxiang Li, Zhen Wang, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21542)  

**Abstract**: Flow Matching (FM) models have emerged as a leading paradigm for high-fidelity synthesis. However, their reliance on iterative Ordinary Differential Equation (ODE) solving creates a significant latency bottleneck. Existing solutions face a dichotomy: training-free solvers suffer from significant performance degradation at low Neural Function Evaluations (NFEs), while training-based one- or few-steps generation methods incur prohibitive training costs and lack plug-and-play versatility. To bridge this gap, we propose the Bi-Anchor Interpolation Solver (BA-solver). BA-solver retains the versatility of standard training-free solvers while achieving significant acceleration by introducing a lightweight SideNet (1-2% backbone size) alongside the frozen backbone. Specifically, our method is founded on two synergistic components: \textbf{1) Bidirectional Temporal Perception}, where the SideNet learns to approximate both future and historical velocities without retraining the heavy backbone; and 2) Bi-Anchor Velocity Integration, which utilizes the SideNet with two anchor velocities to efficiently approximate intermediate velocities for batched high-order integration. By utilizing the backbone to establish high-precision ``anchors'' and the SideNet to densify the trajectory, BA-solver enables large interval sizes with minimized error. Empirical results on ImageNet-256^2 demonstrate that BA-solver achieves generation quality comparable to 100+ NFEs Euler solver in just 10 NFEs and maintains high fidelity in as few as 5 NFEs, incurring negligible training costs. Furthermore, BA-solver ensures seamless integration with existing generative pipelines, facilitating downstream tasks such as image editing. 

---
# On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression 

**Authors**: Xinwei Zhang, Hangcheng Liu, Li Bai, Hao Wang, Qingqing Ye, Tianwei Zhang, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21531)  

**Abstract**: Visual token compression is widely used to accelerate large vision-language models (LVLMs) by pruning or merging visual tokens, yet its adversarial robustness remains unexplored. We show that existing encoder-based attacks can substantially overestimate the robustness of compressed LVLMs, due to an optimization-inference mismatch: perturbations are optimized on the full-token representation, while inference is performed through a token-compression bottleneck. To address this gap, we propose the Compression-AliGnEd attack (CAGE), which aligns perturbation optimization with compression inference without assuming access to the deployed compression mechanism or its token budget. CAGE combines (i) expected feature disruption, which concentrates distortion on tokens likely to survive across plausible budgets, and (ii) rank distortion alignment, which actively aligns token distortions with rank scores to promote the retention of highly distorted evidence. Across diverse representative plug-and-play compression mechanisms and datasets, our results show that CAGE consistently achieves lower robust accuracy than the baseline. This work highlights that robustness assessments ignoring compression can be overly optimistic, calling for compression-aware security evaluation and defenses for efficient LVLMs. 

---
# Sustainable Materials Discovery in the Era of Artificial Intelligence 

**Authors**: Sajid Mannan, Rupert J. Myers, Rohit Batra, Rocio Mercado, Lothar Wondraczek, N. M. Anoop Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21527)  

**Abstract**: Artificial intelligence (AI) has transformed materials discovery, enabling rapid exploration of chemical space through generative models and surrogate screening. Yet current AI workflows optimize performance first, deferring sustainability to post synthesis assessment. This creates inefficiency by the time environmental burdens are quantified, resources have been invested in potentially unsustainable solutions. The disconnect between atomic scale design and lifecycle assessment (LCA) reflects fundamental challenges, data scarcity across heterogeneous sources, scale gaps from atoms to industrial systems, uncertainty in synthesis pathways, and the absence of frameworks that co-optimize performance with environmental impact. We propose to integrate upstream machine learning (ML) assisted materials discovery with downstream lifecycle assessment into a uniform ML-LCA environment. The framework ML-LCA integrates five components, information extraction for building materials-environment knowledge bases, harmonized databases linking properties to sustainability metrics, multi-scale models bridging atomic properties to lifecycle impacts, ensemble prediction of manufacturing pathways with uncertainty quantification, and uncertainty-aware optimization enabling simultaneous performance-sustainability navigation. Case studies spanning glass, cement, semiconductor photoresists, and polymers demonstrate both necessity and feasibility while identifying material-specific integration challenges. Realizing ML-LCA demands coordinated advances in data infrastructure, ex-ante assessment methodologies, multi-objective optimization, and regulatory alignment enabling the discovery of materials that are sustainable by design rather than by chance. 

---
# More Bang for the Buck: Improving the Inference of Large Language Models at a Fixed Budget using Reset and Discard (ReD) 

**Authors**: Sagi Meir, Tommer D. Keidar, Noam Levi, Shlomi Reuveni, Barak Hirshberg  

**Link**: [PDF](https://arxiv.org/pdf/2601.21522)  

**Abstract**: The performance of large language models (LLMs) on verifiable tasks is usually measured by pass@k, the probability of answering a question correctly at least once in k trials. At a fixed budget, a more suitable metric is coverage@cost, the average number of unique questions answered as a function of the total number of attempts. We connect the two metrics and show that the empirically-observed power-law behavior in pass@k leads to a sublinear growth of the coverage@cost (diminishing returns). To solve this problem, we propose Reset-and-Discard (ReD), a query method of LLMs that increases coverage@cost for any given budget, regardless of the pass@k form. Moreover, given a pass@k, we can quantitatively predict the savings in the total number of attempts using ReD. If pass@k is not available for the model, ReD can infer its power-law exponent. Experiments on three LLMs using HumanEval demonstrate that ReD substantially reduces the required attempts, tokens, and USD cost to reach a desired coverage, while also offering an efficient way to measure inference power-laws. 

---
# SimGraph: A Unified Framework for Scene Graph-Based Image Generation and Editing 

**Authors**: Thanh-Nhan Vo, Trong-Thuan Nguyen, Tam V. Nguyen, Minh-Triet Tran  

**Link**: [PDF](https://arxiv.org/pdf/2601.21498)  

**Abstract**: Recent advancements in Generative Artificial Intelligence (GenAI) have significantly enhanced the capabilities of both image generation and editing. However, current approaches often treat these tasks separately, leading to inefficiencies and challenges in maintaining spatial consistency and semantic coherence between generated content and edits. Moreover, a major obstacle is the lack of structured control over object relationships and spatial arrangements. Scene graph-based methods, which represent objects and their interrelationships in a structured format, offer a solution by providing greater control over composition and interactions in both image generation and editing. To address this, we introduce SimGraph, a unified framework that integrates scene graph-based image generation and editing, enabling precise control over object interactions, layouts, and spatial coherence. In particular, our framework integrates token-based generation and diffusion-based editing within a single scene graph-driven model, ensuring high-quality and consistent results. Through extensive experiments, we empirically demonstrate that our approach outperforms existing state-of-the-art methods. 

---
# Mean-Field Control on Sparse Graphs: From Local Limits to GNNs via Neighborhood Distributions 

**Authors**: Tobias Schmidt, Kai Cui  

**Link**: [PDF](https://arxiv.org/pdf/2601.21477)  

**Abstract**: Mean-field control (MFC) offers a scalable solution to the curse of dimensionality in multi-agent systems but traditionally hinges on the restrictive assumption of exchangeability via dense, all-to-all interactions. In this work, we bridge the gap to real-world network structures by proposing a rigorous framework for MFC on large sparse graphs. We redefine the system state as a probability measure over decorated rooted neighborhoods, effectively capturing local heterogeneity. Our central contribution is a theoretical foundation for scalable reinforcement learning in this setting. We prove horizon-dependent locality: for finite-horizon problems, an agent's optimal policy at time t depends strictly on its (T-t)-hop neighborhood. This result renders the infinite-dimensional control problem tractable and underpins a novel Dynamic Programming Principle (DPP) on the lifted space of neighborhood distributions. Furthermore, we formally and experimentally justify the use of Graph Neural Networks (GNNs) for actor-critic algorithms in this context. Our framework naturally recovers classical MFC as a degenerate case while enabling efficient, theoretically grounded control on complex sparse topologies. 

---
# Task-free Adaptive Meta Black-box Optimization 

**Authors**: Chao Wang, Licheng Jiao, Lingling Li, Jiaxuan Zhao, Guanchun Wang, Fang Liu, Shuyuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21475)  

**Abstract**: Handcrafted optimizers become prohibitively inefficient for complex black-box optimization (BBO) tasks. MetaBBO addresses this challenge by meta-learning to automatically configure optimizers for low-level BBO tasks, thereby eliminating heuristic dependencies. However, existing methods typically require extensive handcrafted training tasks to learn meta-strategies that generalize to target tasks, which poses a critical limitation for realistic applications with unknown task distributions. To overcome the issue, we propose the Adaptive meta Black-box Optimization Model (ABOM), which performs online parameter adaptation using solely optimization data from the target task, obviating the need for predefined task distributions. Unlike conventional metaBBO frameworks that decouple meta-training and optimization phases, ABOM introduces a closed-loop adaptive parameter learning mechanism, where parameterized evolutionary operators continuously self-update by leveraging generated populations during optimization. This paradigm shift enables zero-shot optimization: ABOM achieves competitive performance on synthetic BBO benchmarks and realistic unmanned aerial vehicle path planning problems without any handcrafted training tasks. Visualization studies reveal that parameterized evolutionary operators exhibit statistically significant search patterns, including natural selection and genetic recombination. 

---
# Adaptive Confidence Gating in Multi-Agent Collaboration for Efficient and Optimized Code Generation 

**Authors**: Haoji Zhang, Yuzhe Li, Zhenqiang Liu, Chenyang Liu, Shenyang Zhang, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21469)  

**Abstract**: While Large Language Models (LLMs) have catalyzed breakthroughs in automated code generation, Small Language Models (SLMs) often encounter reasoning bottlenecks and failure loops when addressing complex logical requirements. To overcome these challenges, we propose DebateCoder, a multi-agent collaborative framework designed to improve the reasoning ability of SLMs (e.g., Pangu-1B) in resource-constrained environments. DebateCoder uses a structured role-playing protocol with three agents: User Agent (A_UA), Technical Agent (A_TA), and Quality Assurance Agent (A_QA). It also includes an Adaptive Confidence Gating mechanism with a 95% threshold to balance accuracy and inference efficiency. In addition, we introduce a multi-turn deliberation module and a reviewer-guided analytical debugging loop for orthogonal pre-generation debate and post-generation refinement. Experiments on HumanEval and MBPP show that DebateCoder achieves 70.12% Pass@1 on HumanEval, outperforming MapCoder while reducing API overhead by about 35%. These results indicate that collaborative protocols can mitigate limitations of small-parameter models and provide a scalable, efficient approach to high-quality automated software engineering. 

---
# Conversation for Non-verifiable Learning: Self-Evolving LLMs through Meta-Evaluation 

**Authors**: Yuan Sui, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21464)  

**Abstract**: Training large language models (LLMs) for non-verifiable tasks, such as creative writing, dialogue, and ethical reasoning, remains challenging due to the absence of ground-truth labels. While LLM-as-Judge approaches offer a scalable alternative to human feedback, they face a fundamental limitation: performance is constrained by the evaluator's own quality. If the judge cannot recognize good solutions, it cannot provide useful training signals, and evaluation biases (e.g., favoring verbosity over quality) remain unaddressed. This motivates meta-evaluation: the ability to evaluate and improve the evaluator itself. We introduce CoNL, a framework that unifies generation, evaluation, and meta-evaluation through multi-agent self-play. Our key insight: critique quality can be measured by whether it helps others improve their solutions. In CoNL, multiple agents sharing the same policy engage in structured conversations to propose, critique, and revise solutions. Critiques that enable solution improvements earn a diagnostic reward, creating explicit supervision for meta-evaluation and enabling joint optimization of generation and judging capabilities through self-play, without external judges or ground truth. Experiments on five benchmarks show that CoNL achieves consistent improvements over self-rewarding baselines while maintaining stable training. 

---
# Unifying Speech Editing Detection and Content Localization via Prior-Enhanced Audio LLMs 

**Authors**: Jun Xue, Yi Chai, Yanzhen Ren, Jinshen He, Zhiqiang Tang, Zhuolin Yi, Yihuan Huang, Yuankun Xie, Yujie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21463)  

**Abstract**: Speech editing achieves semantic inversion by performing fine-grained segment-level manipulation on original utterances, while preserving global perceptual naturalness. Existing detection studies mainly focus on manually edited speech with explicit splicing artifacts, and therefore struggle to cope with emerging end-to-end neural speech editing techniques that generate seamless acoustic transitions. To address this challenge, we first construct a large-scale bilingual dataset, AiEdit, which leverages large language models to drive precise semantic tampering logic and employs multiple advanced neural speech editing methods for data synthesis, thereby filling the gap of high-quality speech editing datasets. Building upon this foundation, we propose PELM (Prior-Enhanced Audio Large Language Model), the first large-model framework that unifies speech editing detection and content localization by formulating them as an audio question answering task. To mitigate the inherent forgery bias and semantic-priority bias observed in existing audio large models, PELM incorporates word-level probability priors to provide explicit acoustic cues, and further designs a centroid-aggregation-based acoustic consistency perception loss to explicitly enforce the modeling of subtle local distribution anomalies. Extensive experimental results demonstrate that PELM significantly outperforms state-of-the-art methods on both the HumanEdit and AiEdit datasets, achieving equal error rates (EER) of 0.57\% and 9.28\% (localization), respectively. 

---
# L$^3$: Large Lookup Layers 

**Authors**: Albert Tseng, Christopher De Sa  

**Link**: [PDF](https://arxiv.org/pdf/2601.21461)  

**Abstract**: Modern sparse language models typically achieve sparsity through Mixture-of-Experts (MoE) layers, which dynamically route tokens to dense MLP "experts." However, dynamic hard routing has a number of drawbacks, such as potentially poor hardware efficiency and needing auxiliary losses for stable training. In contrast, the tokenizer embedding table, which is natively sparse, largely avoids these issues by selecting a single embedding per token at the cost of not having contextual information. In this work, we introduce the Large Lookup Layer (L$^3$), which unlocks a new axis of sparsity by generalizing embedding tables to model decoder layers. L$^3$ layers use static token-based routing to aggregate a set of learned embeddings per token in a context-dependent way, allowing the model to efficiently balance memory and compute by caching information in embeddings. L$^3$ has two main components: (1) a systems-friendly architecture that allows for fast training and CPU-offloaded inference with no overhead, and (2) an information-theoretic embedding allocation algorithm that effectively balances speed and quality. We empirically test L$^3$ by training transformers with up to 2.6B active parameters and find that L$^3$ strongly outperforms both dense models and iso-sparse MoEs in both language modeling and downstream tasks. 

---
# HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing 

**Authors**: Chengyu Du, Xintao Wang, Aili Chen, Weiyuan Li, Rui Xu, Junteng Liu, Zishan Huang, Rong Tian, Zijun Sun, Yuhao Li, Liheng Feng, Deming Ding, Pengyu Zhao, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21459)  

**Abstract**: LLM role-playing, i.e., using LLMs to simulate specific personas, has emerged as a key capability in various applications, such as companionship, content creation, and digital games. While current models effectively capture character tones and knowledge, simulating the inner thoughts behind their behaviors remains a challenge. Towards cognitive simulation in LLM role-play, previous efforts mainly suffer from two deficiencies: data with high-quality reasoning traces, and reliable reward signals aligned with human preferences. In this paper, we propose HER, a unified framework for cognitive-level persona simulation. HER introduces dual-layer thinking, which distinguishes characters' first-person thinking from LLMs' third-person thinking. To bridge these gaps, we curate reasoning-augmented role-playing data via reverse engineering and construct human-aligned principles and reward models. Leveraging these resources, we train \method models based on Qwen3-32B via supervised and reinforcement learning. Extensive experiments validate the effectiveness of our approach. Notably, our models significantly outperform the Qwen3-32B baseline, achieving a 30.26 improvement on the CoSER benchmark and a 14.97 gain on the Minimax Role-Play Bench. Our datasets, principles, and models will be released to facilitate future research. 

---
# SAGE: Sequence-level Adaptive Gradient Evolution for Generative Recommendation 

**Authors**: Yu Xie, Xing Kai Ren, Ying Qi, Hu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21452)  

**Abstract**: While works such as OneRec have validated the scaling laws of Large Language Models (LLMs) in recommender systems, they rely on a cumbersome separate vocabulary. This dependency prevents the model architecture from reusing native LLM vocabularies, resulting in high maintenance costs and poor scalability. In response, we aim to efficiently reuse open-source LLM architectures without constructing a separate tokenization vocabulary. Furthermore, we identify that the optimization strategy of OneRec Gradient Bounded Policy Optimization (GBPO),suffers from a "Symmetric Conservatism" problem: its static gradient boundaries structurally suppress the update momentum required for cold-start items and fail to prevent diversity collapse in high-noise this http URL address this issue, we propose SAGE (Sequence-level Adaptive Gradient Evolution), a unified optimization framework tailored for list-wise generative recommendation. SAGE introduces two key innovations:(1) Sequence-level Signal Decoupling: By combining a geometric mean importance ratio with decoupled multi-objective advantages, we eliminate token-level variance and resolve the "Reward Collapse" problem. (2) Asymmetric Adaptive Dynamics: We construct a dynamic gradient manifold that applies a "Boost Factor" to high-potential cold start items to achieve super-linear updates and employs an "Entropy Aware Penalty" to break information cocoons. Theoretical analysis and empirical results demonstrate that SAGE effectively unblocks cold-start traffic and sustains recommendation diversity, all while retaining the numerical stability of GBPO. 

---
# Spava: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention 

**Authors**: Yuxiang Huang, Mingye Li, Xu Han, Chaojun Xiao, Weilin Zhao, Ao Sun, Ziqi Yuan, Hao Zhou, Fandong Meng, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21444)  

**Abstract**: The efficiency of long-video inference remains a critical bottleneck, mainly due to the dense computation in the prefill stage of Large Multimodal Models (LMMs). Existing methods either compress visual embeddings or apply sparse attention on a single GPU, yielding limited acceleration or degraded performance and restricting LMMs from handling longer, more complex videos. To overcome these issues, we propose Spava, a sequence-parallel framework with optimized attention that accelerates long-video inference across multiple GPUs. By distributing approximate attention, Spava reduces computation and increases parallelism, enabling efficient processing of more visual embeddings without compression and thereby improving task performance. System-level optimizations, such as load balancing and fused forward passes, further unleash the potential of Spava, delivering speedups of 12.72x, 1.70x, and 1.18x over FlashAttn, ZigZagRing, and APB, without notable performance loss. Code available at this https URL 

---
# From Consistency to Complementarity: Aligned and Disentangled Multi-modal Learning for Time Series Understanding and Reasoning 

**Authors**: Hang Ni, Weijia Zhang, Fei Wang, Zezhi Shao, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21436)  

**Abstract**: Advances in multi-modal large language models (MLLMs) have inspired time series understanding and reasoning tasks, that enable natural language querying over time series, producing textual analyses of complex temporal dynamics. Recent attempts hybridize numerical time series with their visualized plots, facilitating precise value reasoning and visual structure comprehension for comprehensive time series understanding of MLLMs. However, effective cross-modal integration remains challenging due to fine-grained temporal misalignment across modalities and severe entanglement between shared and modality-specific semantics, which hinder localized interpretation and complementary reasoning. To address these issues, we propose MADI, a multi-modal LLM enhanced with fine-grained alignment and disentangled interaction, featuring (1) Patch-level Alignment, which enforces physically grounded fine-grained correspondence across heterogeneous modalities, (2) Discrete Disentangled Interaction, which separates modality-common semantics into compact discrete latents and adaptively synergizes the purified modality-unique information, and (3) Critical-token Highlighting, which emphasizes informative, query-relevant signals for robust reasoning. Experiments on synthetic and real-world benchmarks show that MADI consistently outperforms general-purpose LLMs and time-series-specialized MLLMs. 

---
# Mitigating Overthinking in Large Reasoning Models via Difficulty-aware Reinforcement Learning 

**Authors**: Qian Wan, Ziao Xu, Luona Wei, Xiaoxuan Shen, Jianwen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.21418)  

**Abstract**: Large Reasoning Models (LRMs) achieve explicit chain-of-thought expansion by imitating deep thinking behaviors of humans, demonstrating excellent performance in complex task scenarios. However, the deep-thinking mode often leads to unnecessarily lengthy reasoning and resource inefficiency when handling simple tasks. This overthinking phenomenon may arise from the generation preference triggered by the reward function during post-training. Existing research attempts to mitigate overthinking from the perspective of prompt design or model training, but generally underestimates the importance of task difficulty awareness, which makes it difficult for LRMs to effectively allocate reasoning resources. In this paper, we propose Difficulty-aware Policy Optimization (DiPO), a reinforcement learning-based LRM training framework. DiPO encourages LRM to spontaneously model task complexity, and integrates them into reinforcement learning framework to adjust the generation preferences introduced by post-training. A difficulty modeling method based on model self-reasoning is proposed, which significantly reduces the dependence on manual annotation and formalize task complexity. We further develop a difficulty-signal-enhanced reward function that incorporates a penalty for lengthy reasoning while considering reasoning performance and output format. Experimental results indicate that DiPO enables the model to spontaneously adjust inference overhead, significantly reducing redundant tokens without losing performance due to thought compression. 

---
# Intrinsic Reward Policy Optimization for Sparse-Reward Environments 

**Authors**: Minjae Cho, Huy Trong Tran  

**Link**: [PDF](https://arxiv.org/pdf/2601.21391)  

**Abstract**: Exploration is essential in reinforcement learning as an agent relies on trial and error to learn an optimal policy. However, when rewards are sparse, naive exploration strategies, like noise injection, are often insufficient. Intrinsic rewards can also provide principled guidance for exploration by, for example, combining them with extrinsic rewards to optimize a policy or using them to train subpolicies for hierarchical learning. However, the former approach suffers from unstable credit assignment, while the latter exhibits sample inefficiency and sub-optimality. We propose a policy optimization framework that leverages multiple intrinsic rewards to directly optimize a policy for an extrinsic reward without pretraining subpolicies. Our algorithm -- intrinsic reward policy optimization (IRPO) -- achieves this by using a surrogate policy gradient that provides a more informative learning signal than the true gradient in sparse-reward environments. We demonstrate that IRPO improves performance and sample efficiency relative to baselines in discrete and continuous environments, and formally analyze the optimization problem solved by IRPO. Our code is available at this https URL. 

---
# Understanding Frechet Speech Distance for Synthetic Speech Quality Evaluation 

**Authors**: June-Woo Kim, Dhruv Agarwal, Federica Cerina  

**Link**: [PDF](https://arxiv.org/pdf/2601.21386)  

**Abstract**: Objective evaluation of synthetic speech quality remains a critical challenge. Human listening tests are the gold standard, but costly and impractical at scale. Fréchet Distance has emerged as a promising alternative, yet its reliability depends heavily on the choice of embeddings and experimental settings. In this work, we comprehensively evaluate Fréchet Speech Distance (FSD) and its variant Speech Maximum Mean Discrepancy (SMMD) under varied embeddings and conditions. We further incorporate human listening evaluations alongside TTS intelligibility and synthetic-trained ASR WER to validate the perceptual relevance of these metrics. Our findings show that WavLM Base+ features yield the most stable alignment with human ratings. While FSD and SMMD cannot fully replace subjective evaluation, we show that they can serve as complementary, cost-efficient, and reproducible measures, particularly useful when large-scale or direct listening assessments are infeasible. Code is available at this https URL. 

---
# Sim-MSTNet: sim2real based Multi-task SpatioTemporal Network Traffic Forecasting 

**Authors**: Hui Ma, Qingzhong Li, Jin Wang, Jie Wu, Shaoyu Dou, Li Feng, Xinjun Pei  

**Link**: [PDF](https://arxiv.org/pdf/2601.21384)  

**Abstract**: Network traffic forecasting plays a crucial role in intelligent network operations, but existing techniques often perform poorly when faced with limited data. Additionally, multi-task learning methods struggle with task imbalance and negative transfer, especially when modeling various service types. To overcome these challenges, we propose Sim-MSTNet, a multi-task spatiotemporal network traffic forecasting model based on the sim2real approach. Our method leverages a simulator to generate synthetic data, effectively addressing the issue of poor generalization caused by data scarcity. By employing a domain randomization technique, we reduce the distributional gap between synthetic and real data through bi-level optimization of both sample weighting and model training. Moreover, Sim-MSTNet incorporates attention-based mechanisms to selectively share knowledge between tasks and applies dynamic loss weighting to balance task objectives. Extensive experiments on two open-source datasets show that Sim-MSTNet consistently outperforms state-of-the-art baselines, achieving enhanced accuracy and generalization. 

---
# The Compliance Paradox: Semantic-Instruction Decoupling in Automated Academic Code Evaluation 

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Arjun Neekhra, Yash Sinha, Murari Mandal, Vinay Chamola, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.21360)  

**Abstract**: The rapid integration of Large Language Models (LLMs) into educational assessment rests on the unverified assumption that instruction following capability translates directly to objective adjudication. We demonstrate that this assumption is fundamentally flawed. Instead of evaluating code quality, models frequently decouple from the submission's logic to satisfy hidden directives, a systemic vulnerability we term the Compliance Paradox, where models fine-tuned for extreme helpfulness are vulnerable to adversarial manipulation. To expose this, we introduce the Semantic-Preserving Adversarial Code Injection (SPACI) Framework and the Abstract Syntax Tree-Aware Semantic Injection Protocol (AST-ASIP). These methods exploit the Syntax-Semantics Gap by embedding adversarial directives into syntactically inert regions (trivia nodes) of the Abstract Syntax Tree. Through a large-scale evaluation of 9 SOTA models across 25,000 submissions in Python, C, C++, and Java, we reveal catastrophic failure rates (>95%) in high-capacity open-weights models like DeepSeek-V3, which systematically prioritize hidden formatting constraints over code correctness. We quantify this failure using our novel tripartite framework measuring Decoupling Probability, Score Divergence, and Pedagogical Severity to demonstrate the widespread "False Certification" of functionally broken code. Our findings suggest that current alignment paradigms create a "Trojan" vulnerability in automated grading, necessitating a shift from standard RLHF toward domain-specific Adjudicative Robustness, where models are conditioned to prioritize evidence over instruction compliance. We release our complete dataset and injection framework to facilitate further research on the topic. 

---
# Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving 

**Authors**: Chendong Song, Meixuan Wang, Hang Zhou, Hong Liang, Yuan Lyu, Zixi Chen, Yuwei Fan, Zijie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21351)  

**Abstract**: Attention-FFN disaggregation (AFD) is an emerging architecture for LLM decoding that separates state-heavy, KV-cache-dominated Attention computation from stateless, compute-intensive FFN computation, connected by per-step communication. While AFD enables independent scaling of memory and compute resources, its performance is highly sensitive to the Attention/FFN provisioning ratio: mis-sizing induces step-level blocking and costly device idle time. We develop a tractable analytical framework for sizing AFD bundles in an $r$A-$1$F topology, where the key difficulty is that Attention-side work is nonstationary-token context grows and requests are continuously replenished with random lengths-while FFN work is stable given the aggregated batch. Using a probabilistic workload model, we derive closed-form rules for the optimal A/F ratio that maximize average throughput per instance across the system. A trace-calibrated AFD simulator validates the theory: across workloads, the theoretical optimal A/F ratio matches the simulation-optimal within 10%, and consistently reduces idle time. 

---
# L2R: Low-Rank and Lipschitz-Controlled Routing for Mixture-of-Experts 

**Authors**: Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.21349)  

**Abstract**: Mixture-of-Experts (MoE) models scale neural networks by conditionally activating a small subset of experts, where the router plays a central role in determining expert specialization and overall model performance. However, many modern MoE systems still adopt linear routers in raw high-dimensional representation spaces, where representation mismatch, angular concentration, and scale-sensitive scoring can jointly undermine routing discriminability and stable expert specialization. In this work, we propose Low-rank \& Lipschitz-controlled Routing (L2R), a unified routing framework that reshapes both the routing space and scoring geometry. L2R performs expert assignment in a shared low-rank latent routing space and introduces Saturated Inner-Product Scoring (SIPS) to explicitly control the Lipschitz behavior of routing functions, yielding smoother and more stable routing geometry. In addition, L2R incorporates a parameter-efficient multi-anchor routing mechanism to enhance expert expressiveness. Extensive experiments on a large-scale language MoE model and a vision MoE setting on ImageNet demonstrate that L2R consistently improves routing stability, expert specialization, and overall model performance. 

---
# Memorization Control in Diffusion Models from Denoising-centric Perspective 

**Authors**: Thuy Phuong Vu, Mai Viet Hoang Do, Minhhuy Le, Dinh-Cuong Hoang, Phan Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21348)  

**Abstract**: Controlling memorization in diffusion models is critical for applications that require generated data to closely match the training distribution. Existing approaches mainly focus on data centric or model centric modifications, treating the diffusion model as an isolated predictor. In this paper, we study memorization in diffusion models from a denoising centric perspective. We show that uniform timestep sampling leads to unequal learning contributions across denoising steps due to differences in signal to noise ratio, which biases training toward memorization. To address this, we propose a timestep sampling strategy that explicitly controls where learning occurs along the denoising trajectory. By adjusting the width of the confidence interval, our method provides direct control over the memorization generalization trade off. Experiments on image and 1D signal generation tasks demonstrate that shifting learning emphasis toward later denoising steps consistently reduces memorization and improves distributional alignment with training data, validating the generality and effectiveness of our approach. 

---
# Self-Improving Pretraining: using post-trained models to pretrain better models 

**Authors**: Ellen Xiaoqing Tan, Shehzaad Dhuliawala, Jing Xu, Ping Yu, Sainbayar Sukhbaatar, Jason Weston, Olga Golovneva  

**Link**: [PDF](https://arxiv.org/pdf/2601.21343)  

**Abstract**: Ensuring safety, factuality and overall quality in the generations of large language models is a critical challenge, especially as these models are increasingly deployed in real-world applications. The prevailing approach to addressing these issues involves collecting expensive, carefully curated datasets and applying multiple stages of fine-tuning and alignment. However, even this complex pipeline cannot guarantee the correction of patterns learned during pretraining. Therefore, addressing these issues during pretraining is crucial, as it shapes a model's core behaviors and prevents unsafe or hallucinated outputs from becoming deeply embedded. To tackle this issue, we introduce a new pretraining method that streams documents and uses reinforcement learning (RL) to improve the next K generated tokens at each step. A strong, post-trained model judges candidate generations -- including model rollouts, the original suffix, and a rewritten suffix -- for quality, safety, and factuality. Early in training, the process relies on the original and rewritten suffixes; as the model improves, RL rewards high-quality rollouts. This approach builds higher quality, safer, and more factual models from the ground up. In experiments, our method gives 36.2% and 18.5% relative improvements over standard pretraining in terms of factuality and safety, and up to 86.3% win rate improvements in overall generation quality. 

---
# Adversarial Vulnerability Transcends Computational Paradigms: Feature Engineering Provides No Defense Against Neural Adversarial Transfer 

**Authors**: Achraf Hsain, Ahmed Abdelkader, Emmanuel Baldwin Mbaya, Hamoud Aljamaan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21323)  

**Abstract**: Deep neural networks are vulnerable to adversarial examples--inputs with imperceptible perturbations causing misclassification. While adversarial transfer within neural networks is well-documented, whether classical ML pipelines using handcrafted features inherit this vulnerability when attacked via neural surrogates remains unexplored. Feature engineering creates information bottlenecks through gradient quantization and spatial binning, potentially filtering high-frequency adversarial signals. We evaluate this hypothesis through the first comprehensive study of adversarial transfer from DNNs to HOG-based classifiers. Using VGG16 as a surrogate, we generate FGSM and PGD adversarial examples and test transfer to four classical classifiers (KNN, Decision Tree, Linear SVM, Kernel SVM) and a shallow neural network across eight HOG configurations on CIFAR-10. Our results strongly refute the protective hypothesis: all classifiers suffer 16.6%-59.1% relative accuracy drops, comparable to neural-to-neural transfer. More surprisingly, we discover attack hierarchy reversal--contrary to patterns where iterative PGD dominates FGSM within neural networks, FGSM causes greater degradation than PGD in 100% of classical ML cases, suggesting iterative attacks overfit to surrogate-specific features that don't survive feature extraction. Block normalization provides partial but insufficient mitigation. These findings demonstrate that adversarial vulnerability is not an artifact of end-to-end differentiability but a fundamental property of image classification systems, with implications for security-critical deployments across computational paradigms. 

---
# Heterogeneous Vertiport Selection Optimization for On-Demand Air Taxi Services: A Deep Reinforcement Learning Approach 

**Authors**: Aoyu Pang, Maonan Wang, Zifan Sha, Wenwei Yue, Changle Li, Chung Shue Chen, Man-On Pun  

**Link**: [PDF](https://arxiv.org/pdf/2601.21316)  

**Abstract**: Urban Air Mobility (UAM) has emerged as a transformative solution to alleviate urban congestion by utilizing low-altitude airspace, thereby reducing pressure on ground transportation networks. To enable truly efficient and seamless door-to-door travel experiences, UAM requires close integration with existing ground transportation infrastructure. However, current research on optimal integrated routing strategies for passengers in air-ground mobility systems remains limited, with a lack of systematic this http URL address this gap, we first propose a unified optimization model that integrates strategy selection for both air and ground transportation. This model captures the dynamic characteristics of multimodal transport networks and incorporates real-time traffic conditions alongside passenger decision-making behavior. Building on this model, we propose a Unified Air-Ground Mobility Coordination (UAGMC) framework, which leverages deep reinforcement learning (RL) and Vehicle-to-Everything (V2X) communication to optimize vertiport selection and dynamically plan air taxi routes. Experimental results demonstrate that UAGMC achieves a 34\% reduction in average travel time compared to conventional proportional allocation methods, enhancing overall travel efficiency and providing novel insights into the integration and optimization of multimodal transportation systems. This work lays a solid foundation for advancing intelligent urban mobility solutions through the coordination of air and ground transportation modes. The related code can be found at this https URL. 

---
# Distributionally Robust Classification for Multi-source Unsupervised Domain Adaptation 

**Authors**: Seonghwi Kim, Sung Ho Jo, Wooseok Ha, Minwoo Chae  

**Link**: [PDF](https://arxiv.org/pdf/2601.21315)  

**Abstract**: Unsupervised domain adaptation (UDA) is a statistical learning problem when the distribution of training (source) data is different from that of test (target) data. In this setting, one has access to labeled data only from the source domain and unlabeled data from the target domain. The central objective is to leverage the source data and the unlabeled target data to build models that generalize to the target domain. Despite its potential, existing UDA approaches often struggle in practice, particularly in scenarios where the target domain offers only limited unlabeled data or spurious correlations dominate the source domain. To address these challenges, we propose a novel distributionally robust learning framework that models uncertainty in both the covariate distribution and the conditional label distribution. Our approach is motivated by the multi-source domain adaptation setting but is also directly applicable to the single-source scenario, making it versatile in practice. We develop an efficient learning algorithm that can be seamlessly integrated with existing UDA methods. Extensive experiments under various distribution shift scenarios show that our method consistently outperforms strong baselines, especially when target data are extremely scarce. 

---
# The Surprising Difficulty of Search in Model-Based Reinforcement Learning 

**Authors**: Wei-Di Chang, Mikael Henaff, Brandon Amos, Gregory Dudek, Scott Fujimoto  

**Link**: [PDF](https://arxiv.org/pdf/2601.21306)  

**Abstract**: This paper investigates search in model-based reinforcement learning (RL). Conventional wisdom holds that long-term predictions and compounding errors are the primary obstacles for model-based RL. We challenge this view, showing that search is not a plug-and-play replacement for a learned policy. Surprisingly, we find that search can harm performance even when the model is highly accurate. Instead, we show that mitigating distribution shift matters more than improving model or value function accuracy. Building on this insight, we identify key techniques for enabling effective search, achieving state-of-the-art performance across multiple popular benchmark domains. 

---
# Grounding and Enhancing Informativeness and Utility in Dataset Distillation 

**Authors**: Shaobo Wang, Yantai Yang, Guo Chen, Peiru Li, Kaixin Li, Yufa Zhou, Zhaorun Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21296)  

**Abstract**: Dataset Distillation (DD) seeks to create a compact dataset from a large, real-world dataset. While recent methods often rely on heuristic approaches to balance efficiency and quality, the fundamental relationship between original and synthetic data remains underexplored. This paper revisits knowledge distillation-based dataset distillation within a solid theoretical framework. We introduce the concepts of Informativeness and Utility, capturing crucial information within a sample and essential samples in the training set, respectively. Building on these principles, we define optimal dataset distillation mathematically. We then present InfoUtil, a framework that balances informativeness and utility in synthesizing the distilled dataset. InfoUtil incorporates two key components: (1) game-theoretic informativeness maximization using Shapley Value attribution to extract key information from samples, and (2) principled utility maximization by selecting globally influential samples based on Gradient Norm. These components ensure that the distilled dataset is both informative and utility-optimized. Experiments demonstrate that our method achieves a 6.1\% performance improvement over the previous state-of-the-art approach on ImageNet-1K dataset using ResNet-18. 

---
# Physics-Guided Tiny-Mamba Transformer for Reliability-Aware Early Fault Warning 

**Authors**: Changyu Li, Dingcheng Huang, Kexuan Yao, Xiaoya Ni, Lijuan Shen, Fei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21293)  

**Abstract**: Reliability-centered prognostics for rotating machinery requires early warning signals that remain accurate under nonstationary operating conditions, domain shifts across speed/load/sensors, and severe class imbalance, while keeping the false-alarm rate small and predictable. We propose the Physics-Guided Tiny-Mamba Transformer (PG-TMT), a compact tri-branch encoder tailored for online condition monitoring. A depthwise-separable convolutional stem captures micro-transients, a Tiny-Mamba state-space branch models near-linear long-range dynamics, and a lightweight local Transformer encodes cross-channel resonances. We derive an analytic temporal-to-spectral mapping that ties the model's attention spectrum to classical bearing fault-order bands, yielding a band-alignment score that quantifies physical plausibility and provides physics-grounded explanations. To ensure decision reliability, healthy-score exceedances are modeled with extreme-value theory (EVT), which yields an on-threshold achieving a target false-alarm intensity (events/hour); a dual-threshold hysteresis with a minimum hold time further suppresses chatter. Under a leakage-free streaming protocol with right-censoring of missed detections on CWRU, Paderborn, XJTU-SY, and an industrial pilot, PG-TMT attains higher precision-recall AUC (primary under imbalance), competitive or better ROC AUC, and shorter mean time-to-detect at matched false-alarm intensity, together with strong cross-domain transfer. By coupling physics-aligned representations with EVT-calibrated decision rules, PG-TMT delivers calibrated, interpretable, and deployment-ready early warnings for reliability-centric prognostics and health management. 

---
# Zenith: Scaling up Ranking Models for Billion-scale Livestreaming Recommendation 

**Authors**: Ruifeng Zhang, Zexi Huang, Zikai Wang, Ke Sun, Bohang Zheng, Zhen Ouyang, Huimin Xie, Phil Shen, Junlin Zhang, Wentao Guo, Qinglei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21285)  

**Abstract**: Accurately capturing feature interactions is essential in recommender systems, and recent trends show that scaling up model capacity could be a key driver for next-level predictive performance. While prior work has explored various model architectures to capture multi-granularity feature interactions, relatively little attention has been paid to efficient feature handling and scaling model capacity without incurring excessive inference latency. In this paper, we address this by presenting Zenith, a scalable and efficient ranking architecture that learns complex feature interactions with minimal runtime overhead. Zenith is designed to handle a few high-dimensional Prime Tokens with Token Fusion and Token Boost modules, which exhibits superior scaling laws compared to other state-of-the-art ranking methods, thanks to its improved token heterogeneity. Its real-world effectiveness is demonstrated by deploying the architecture to TikTok Live, a leading online livestreaming platform that attracts billions of users globally. Our A/B test shows that Zenith achieves +1.05%/-1.10% in online CTR AUC and Logloss, and realizes +9.93% gains in Quality Watch Session / User and +8.11% in Quality Watch Duration / User. 

---
# PILD: Physics-Informed Learning via Diffusion 

**Authors**: Tianyi Zeng, Tianyi Wang, Jiaru Zhang, Zimo Zeng, Feiyang Zhang, Yiming Xu, Sikai Chen, Yajie Zou, Yangyang Wang, Junfeng Jiao, Christian Claudel, Xinbo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21284)  

**Abstract**: Diffusion models have emerged as powerful generative tools for modeling complex data distributions, yet their purely data-driven nature limits applicability in practical engineering and scientific problems where physical laws need to be followed. This paper proposes Physics-Informed Learning via Diffusion (PILD), a framework that unifies diffusion modeling and first-principles physical constraints by introducing a virtual residual observation sampled from a Laplace distribution to supervise generation during training. To further integrate physical laws, a conditional embedding module is incorporated to inject physical information into the denoising network at multiple layers, ensuring consistent guidance throughout the diffusion process. The proposed PILD framework is concise, modular, and broadly applicable to problems governed by ordinary differential equations, partial differential equations, as well as algebraic equations or inequality constraints. Extensive experiments across engineering and scientific tasks including estimating vehicle trajectories, tire forces, Darcy flow and plasma dynamics, demonstrate that our PILD substantially improves accuracy, stability, and generalization over existing physics-informed and diffusion-based baselines. 

---
# DUET: Distilled LLM Unlearning from an Efficiently Contextualized Teacher 

**Authors**: Yisheng Zhong, Zhengbang Yang, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21283)  

**Abstract**: LLM unlearning is a technique to remove the impacts of undesirable knowledge from the model without retraining from scratch, which is indispensable towards trustworthy AI. Existing unlearning methods face significant limitations: conventional tuning-based unlearning is computationally heavy and prone to catastrophic forgetting. In contrast, in-contextualized unlearning is lightweight for precise unlearning but vulnerable to prompt removal or reverse engineering attacks. In response, we propose Distilled Unlearning from an Efficient Teacher (DUET), a novel distillation-based unlearning method that combines the merits of these two lines of work. It learns a student model to imitate the behavior of a prompt-steered teacher that effectively refuses undesirable knowledge generation while preserving general domain knowledge. Extensive evaluations on existing benchmarks with our enriched evaluation protocols demonstrate that DUET achieves higher performance in both forgetting and utility preservation, while being orders of magnitude more data-efficient than state-of-the-art unlearning methods. 

---
# NEXUS: Bit-Exact ANN-to-SNN Equivalence via Neuromorphic Gate Circuits with Surrogate-Free Training 

**Authors**: Zhengzheng Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21279)  

**Abstract**: Spiking Neural Networks (SNNs) promise energy-efficient computing through event-driven sparsity, yet all existing approaches sacrifice accuracy by approximating continuous values with discrete spikes. We propose NEXUS, a framework that achieves bit-exact ANN-to-SNN equivalence -- not approximate, but mathematically identical outputs. Our key insight is constructing all arithmetic operations, both linear and nonlinear, from pure IF neuron logic gates that implement IEEE-754 compliant floating-point arithmetic. Through spatial bit encoding (zero encoding error by construction), hierarchical neuromorphic gate circuits (from basic logic gates to complete transformer layers), and surrogate-free STE training (exact identity mapping rather than heuristic approximation), NEXUS produces outputs identical to standard ANNs up to machine precision. Experiments on models up to LLaMA-2 70B demonstrate identical task accuracy (0.00\% degradation) with mean ULP error of only 6.19, while achieving 27-168,000$\times$ energy reduction on neuromorphic hardware. Crucially, spatial bit encoding's single-timestep design renders the framework inherently immune to membrane potential leakage (100\% accuracy across all decay factors $\beta\in[0.1,1.0]$), while tolerating synaptic noise up to $\sigma=0.2$ with >98\% gate-level accuracy. 

---
# GeoRC: A Benchmark for Geolocation Reasoning Chains 

**Authors**: Mohit Talreja, Joshua Diao, Jim Thannikary James, Radu Casapu, Tejas Santanam, Ethan Mendes, Alan Ritter, Wei Xu, James Hays  

**Link**: [PDF](https://arxiv.org/pdf/2601.21278)  

**Abstract**: Vision Language Models (VLMs) are good at recognizing the global location of a photograph -- their geolocation prediction accuracy rivals the best human experts. But many VLMs are startlingly bad at explaining which image evidence led to their prediction, even when their location prediction is correct. The reasoning chains produced by VLMs frequently hallucinate scene attributes to support their location prediction (e.g. phantom writing, imagined infrastructure, misidentified flora). In this paper, we introduce the first benchmark for geolocation reasoning chains. We focus on the global location prediction task in the popular GeoGuessr game which draws from Google Street View spanning more than 100 countries. We collaborate with expert GeoGuessr players, including the reigning world champion, to produce 800 ground truth reasoning chains for 500 query scenes. These expert reasoning chains address hundreds of different discriminative visual attributes such as license plate shape, architecture, and soil properties to name just a few. We evaluate LLM-as-a-judge and VLM-as-a-judge strategies for scoring VLM-generated reasoning chains against our expert reasoning chains and find that Qwen 3 LLM-as-a-judge correlates best with human scoring. Our benchmark reveals that while large, closed-source VLMs such as Gemini and GPT 5 rival human experts at prediction locations, they still lag behind human experts when it comes to producing auditable reasoning chains. Open weights VLMs such as Llama and Qwen catastrophically fail on our benchmark -- they perform only slightly better than a baseline in which an LLM hallucinates a reasoning chain with oracle knowledge of the photo location but no visual information at all. We believe the gap between human experts and VLMs on this task points to VLM limitations at extracting fine-grained visual attributes from high resolution images. 

---
# More Code, Less Reuse: Investigating Code Quality and Reviewer Sentiment towards AI-generated Pull Requests 

**Authors**: Haoming Huang, Pongchai Jaisri, Shota Shimizu, Lingfeng Chen, Sota Nakashima, Gema Rodríguez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2601.21276)  

**Abstract**: Large Language Model (LLM) Agents are advancing quickly, with the increasing leveraging of LLM Agents to assist in development tasks such as code generation. While LLM Agents accelerate code generation, studies indicate they may introduce adverse effects on development. However, existing metrics solely measure pass rates, failing to reflect impacts on long-term maintainability and readability, and failing to capture human intuitive evaluations of PR. To increase the comprehensiveness of this problem, we investigate and evaluate the characteristics of LLM to know the pull requests' characteristics beyond the pass rate. We observe the code quality and maintainability within PRs based on code metrics to evaluate objective characteristics and developers' reactions to the pull requests from both humans and LLM's generation. Evaluation results indicate that LLM Agents frequently disregard code reuse opportunities, resulting in higher levels of redundancy compared to human developers. In contrast to the quality issues, our emotions analysis reveals that reviewers tend to express more neutral or positive emotions towards AI-generated contributions than human ones. This disconnect suggests that the surface-level plausibility of AI code masks redundancy, leading to the silent accumulation of technical debt in real-world development environments. Our research provides insights for improving human-AI collaboration. 

---
# Lightweight High-Fidelity Low-Bitrate Talking Face Compression for 3D Video Conference 

**Authors**: Jianglong Li, Jun Xu, Bingcong Lu, Zhengxue Cheng, Hongwei Hu, Ronghua Wu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.21269)  

**Abstract**: The demand for immersive and interactive communication has driven advancements in 3D video conferencing, yet achieving high-fidelity 3D talking face representation at low bitrates remains a challenge. Traditional 2D video compression techniques fail to preserve fine-grained geometric and appearance details, while implicit neural rendering methods like NeRF suffer from prohibitive computational costs. To address these challenges, we propose a lightweight, high-fidelity, low-bitrate 3D talking face compression framework that integrates FLAME-based parametric modeling with 3DGS neural rendering. Our approach transmits only essential facial metadata in real time, enabling efficient reconstruction with a Gaussian-based head model. Additionally, we introduce a compact representation and compression scheme, including Gaussian attribute compression and MLP optimization, to enhance transmission efficiency. Experimental results demonstrate that our method achieves superior rate-distortion performance, delivering high-quality facial rendering at extremely low bitrates, making it well-suited for real-time 3D video conferencing applications. 

---
# Music Plagiarism Detection: Problem Formulation and a Segment-based Solution 

**Authors**: Seonghyeon Go, Yumin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.21260)  

**Abstract**: Recently, the problem of music plagiarism has emerged as an even more pressing social issue. As music information retrieval research advances, there is a growing effort to address issues related to music plagiarism. However, many studies, including our previous work, have conducted research without clearly defining what the music plagiarism detection task actually involves. This lack of a clear definition has slowed research progress and made it hard to apply results to real-world scenarios. To fix this situation, we defined how Music Plagiarism Detection is different from other MIR tasks and explained what problems need to be solved. We introduce the Similar Music Pair dataset to support this newly defined task. In addition, we propose a method based on segment transcription as one way to solve the task. Our demo and dataset are available at this https URL. 

---
# Hypersolid: Emergent Vision Representations via Short-Range Repulsion 

**Authors**: Esteban Rodríguez-Betancourt, Edgar Casasola-Murillo  

**Link**: [PDF](https://arxiv.org/pdf/2601.21255)  

**Abstract**: A recurring challenge in self-supervised learning is preventing representation collapse. Existing solutions typically rely on global regularization, such as maximizing distances, decorrelating dimensions or enforcing certain distributions. We instead reinterpret representation learning as a discrete packing problem, where preserving information simplifies to maintaining injectivity. We operationalize this in Hypersolid, a method using short-range hard-ball repulsion to prevent local collisions. This constraint results in a high-separation geometric regime that preserves augmentation diversity, excelling on fine-grained and low-resolution classification tasks. 

---
# Conditional Generative Framework with Peak-Aware Attention for Robust Chemical Detection under Interferences 

**Authors**: Namkyung Yoon, Sanghong Kim, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.21246)  

**Abstract**: Gas chromatography-mass spectrometry (GC-MS) is a widely used analytical method for chemical substance detection, but measurement reliability tends to deteriorate in the presence of interfering substances. In particular, interfering substances cause nonspecific peaks, residence time shifts, and increased background noise, resulting in reduced sensitivity and false alarms. To overcome these challenges, in this paper, we propose an artificial intelligence discrimination framework based on a peak-aware conditional generative model to improve the reliability of GC-MS measurements under interference conditions. The framework is learned with a novel peak-aware mechanism that highlights the characteristic peaks of GC-MS data, allowing it to generate important spectral features more faithfully. In addition, chemical and solvent information is encoded in a latent vector embedded with it, allowing a conditional generative adversarial neural network (CGAN) to generate a synthetic GC-MS signal consistent with the experimental conditions. This generates an experimental dataset that assumes indirect substance situations in chemical substance data, where acquisition is limited without conducting real experiments. These data are used for the learning of AI-based GC-MS discrimination models to help in accurate chemical substance discrimination. We conduct various quantitative and qualitative evaluations of the generated simulated data to verify the validity of the proposed framework. We also verify how the generative model improves the performance of the AI discrimination framework. Representatively, the proposed method is shown to consistently achieve cosine similarity and Pearson correlation coefficient values above 0.9 while preserving peak number diversity and reducing false alarms in the discrimination model. 

---
# Less Noise, More Voice: Reinforcement Learning for Reasoning via Instruction Purification 

**Authors**: Yiju Guo, Tianyi Hu, Zexu Sun, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.21244)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has advanced LLM reasoning, but remains constrained by inefficient exploration under limited rollout budgets, leading to low sampling success and unstable training in complex tasks. We find that many exploration failures arise not from problem difficulty, but from a small number of prompt tokens that introduce interference. Building on this insight, we propose the Less Noise Sampling Framework (LENS), which first prompts by identifying and removing interference tokens. then transfers successful rollouts from the purification process to supervise policy optimization on the original noisy prompts, enabling the model to learn to ignore interference in the real-world, noisy prompting settings. Experimental results show that LENS significantly outperforms GRPO, delivering higher performance and faster convergence, with a 3.88% average gain and over 1.6$\times$ speedup. Our work highlights the critical role of pruning interference tokens in improving rollout efficiency, offering a new perspective for RLVR research. 

---
# Understanding Diffusion Models via Ratio-Based Function Approximation with SignReLU Networks 

**Authors**: Luwei Sun, Dongrui Shen, Jianfe Li, Yulong Zhao, Han Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.21242)  

**Abstract**: Motivated by challenges in conditional generative modeling, where the target conditional density takes the form of a ratio f1 over f2, this paper develops a theoretical framework for approximating such ratio-type functionals. Here, f1 and f2 are kernel-based marginal densities that capture structured interactions, a setting central to diffusion-based generative models. We provide a concise proof for approximating these ratio-type functionals using deep neural networks with the SignReLU activation function, leveraging the activation's piecewise structure. Under standard regularity assumptions, we establish L^p(Omega) approximation bounds and convergence rates. Specializing to Denoising Diffusion Probabilistic Models (DDPMs), we construct a SignReLU-based neural estimator for the reverse process and derive bounds on the excess Kullback-Leibler (KL) risk between the generated and true data distributions. Our analysis decomposes this excess risk into approximation and estimation error components. These results provide generalization guarantees for finite-sample training of diffusion-based generative models. 

---
# PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models 

**Authors**: Xuewen Liu, Zhikai Li, Jing Zhang, Mengjuan Chen, Qingyi Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21238)  

**Abstract**: AutoRegressive Visual Generation (ARVG) models retain an architecture compatible with language models, while achieving performance comparable to diffusion-based models. Quantization is commonly employed in neural networks to reduce model size and computational latency. However, applying quantization to ARVG remains largely underexplored, and existing quantization methods fail to generalize effectively to ARVG models. In this paper, we explore this issue and identify three key challenges: (1) severe outliers at channel-wise level, (2) highly dynamic activations at token-wise level, and (3) mismatched distribution information at sample-wise level. To these ends, we propose PTQ4ARVG, a training-free post-training quantization (PTQ) framework consisting of: (1) Gain-Projected Scaling (GPS) mitigates the channel-wise outliers, which expands the quantization loss via a Taylor series to quantify the gain of scaling for activation-weight quantization, and derives the optimal scaling factor through differentiation.(2) Static Token-Wise Quantization (STWQ) leverages the inherent properties of ARVG, fixed token length and position-invariant distribution across samples, to address token-wise variance without incurring dynamic calibration overhead.(3) Distribution-Guided Calibration (DGC) selects samples that contribute most to distributional entropy, eliminating the sample-wise distribution mismatch. Extensive experiments show that PTQ4ARVG can effectively quantize the ARVG family models to 8-bit and 6-bit while maintaining competitive performance. Code is available at this http URL . 

---
# SHARP: Social Harm Analysis via Risk Profiles for Measuring Inequities in Large Language Models 

**Authors**: Alok Abhishek, Tushar Bandopadhyay, Lisa Erickson  

**Link**: [PDF](https://arxiv.org/pdf/2601.21235)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains, where rare but severe failures can result in irreversible harm. However, prevailing evaluation benchmarks often reduce complex social risk to mean-centered scalar scores, thereby obscuring distributional structure, cross-dimensional interactions, and worst-case behavior. This paper introduces Social Harm Analysis via Risk Profiles (SHARP), a framework for multidimensional, distribution-aware evaluation of social harm. SHARP models harm as a multivariate random variable and integrates explicit decomposition into bias, fairness, ethics, and epistemic reliability with a union-of-failures aggregation reparameterized as additive cumulative log-risk. The framework further employs risk-sensitive distributional statistics, with Conditional Value at Risk (CVaR95) as a primary metric, to characterize worst-case model behavior. Application of SHARP to eleven frontier LLMs, evaluated on a fixed corpus of n=901 socially sensitive prompts, reveals that models with similar average risk can exhibit more than twofold differences in tail exposure and volatility. Across models, dimension-wise marginal tail behavior varies systematically across harm dimensions, with bias exhibiting the strongest tail severities, epistemic and fairness risks occupying intermediate regimes, and ethical misalignment consistently lower; together, these patterns reveal heterogeneous, model-dependent failure structures that scalar benchmarks conflate. These findings indicate that responsible evaluation and governance of LLMs require moving beyond scalar averages toward multidimensional, tail-sensitive risk profiling. 

---
# MGSM-Pro: A Simple Strategy for Robust Multilingual Mathematical Reasoning Evaluation 

**Authors**: Tianyi Xu, Kosei Uemura, Alfred Malengo Kondoro, Tadesse Destaw Belay, Catherine Nana Nyaah Essuman, Ifeoma Okoh, Ganiyat Afolabi, Ayodele Awokoya, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2601.21225)  

**Abstract**: Large language models have made substantial progress in mathematical reasoning. However, benchmark development for multilingual evaluation has lagged behind English in both difficulty and recency. Recently, GSM-Symbolic showed a strong evidence of high variance when models are evaluated on different instantiations of the same question; however, the evaluation was conducted only in English. In this paper, we introduce MGSM-Pro, an extension of MGSM dataset with GSM-Symbolic approach. Our dataset provides five instantiations per MGSM question by varying names, digits and irrelevant context. Evaluations across nine languages reveal that many low-resource languages suffer large performance drops when tested on digit instantiations different from those in the original test set. We further find that some proprietary models, notably Gemini 2.5 Flash and GPT-4.1, are less robust to digit instantiation, whereas Claude 4.0 Sonnet is more robust. Among open models, GPT-OSS 120B and DeepSeek V3 show stronger robustness. Based on these findings, we recommend evaluating each problem using at least five digit-varying instantiations to obtain a more robust and realistic assessment of math reasoning. 

---
# Temporal Context and Architecture: A Benchmark for Naturalistic EEG Decoding 

**Authors**: Mehmet Ergezer  

**Link**: [PDF](https://arxiv.org/pdf/2601.21215)  

**Abstract**: We study how model architecture and temporal context interact in naturalistic EEG decoding. Using the HBN movie-watching dataset, we benchmark five architectures, CNN, LSTM, a stabilized Transformer (EEGXF), S4, and S5, on a 4-class task across segment lengths from 8s to 128s. Accuracy improves with longer context: at 64s, S5 reaches 98.7%+/-0.6 and CNN 98.3%+/-0.3, while S5 uses ~20x fewer parameters than CNN. To probe real-world robustness, we evaluate zero-shot cross-frequency shifts, cross-task OOD inputs, and leave-one-subject-out generalization. S5 achieves stronger cross-subject accuracy but makes over-confident errors on OOD tasks; EEGXF is more conservative and stable under frequency shifts, though less calibrated in-distribution. These results reveal a practical efficiency-robustness trade-off: S5 for parameter-efficient peak accuracy; EEGXF when robustness and conservative uncertainty are critical. 

---
# A Sheaf-Theoretic and Topological Perspective on Complex Network Modeling and Attention Mechanisms in Graph Neural Models 

**Authors**: Chuan-Shen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21207)  

**Abstract**: Combinatorial and topological structures, such as graphs, simplicial complexes, and cell complexes, form the foundation of geometric and topological deep learning (GDL and TDL) architectures. These models aggregate signals over such domains, integrate local features, and generate representations for diverse real-world applications. However, the distribution and diffusion behavior of GDL and TDL features during training remains an open and underexplored problem. Motivated by this gap, we introduce a cellular sheaf theoretic framework for modeling and analyzing the local consistency and harmonicity of node features and edge weights in graph-based architectures. By tracking local feature alignments and agreements through sheaf structures, the framework offers a topological perspective on feature diffusion and aggregation. Furthermore, a multiscale extension inspired by topological data analysis (TDA) is proposed to capture hierarchical feature interactions in graph models. This approach enables a joint characterization of GDL and TDL architectures based on their underlying geometric and topological structures and the learned signals defined on them, providing insights for future studies on conventional tasks such as node classification, substructure detection, and community detection. 

---
# Scaling Embeddings Outperforms Scaling Experts in Language Models 

**Authors**: Hong Liu, Jiaqi Zhang, Chao Wang, Xing Hu, Linkun Lyu, Jiaqi Sun, Xurui Yang, Bo Wang, Fengcun Li, Yulei Qian, Lingtong Si, Yerui Sun, Rumei Li, Peng Pei, Yuchen Xie, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2601.21204)  

**Abstract**: While Mixture-of-Experts (MoE) architectures have become the standard for sparsity scaling in large language models, they increasingly face diminishing returns and system-level bottlenecks. In this work, we explore embedding scaling as a potent, orthogonal dimension for scaling sparsity. Through a comprehensive analysis and experiments, we identify specific regimes where embedding scaling achieves a superior Pareto frontier compared to expert scaling. We systematically characterize the critical architectural factors governing this efficacy -- ranging from parameter budgeting to the interplay with model width and depth. Moreover, by integrating tailored system optimizations and speculative decoding, we effectively convert this sparsity into tangible inference speedups. Guided by these insights, we introduce LongCat-Flash-Lite, a 68.5B parameter model with ~3B activated trained from scratch. Despite allocating over 30B parameters to embeddings, LongCat-Flash-Lite not only surpasses parameter-equivalent MoE baselines but also exhibits exceptional competitiveness against existing models of comparable scale, particularly in agentic and coding domains. 

---
# Thinker: A vision-language foundation model for embodied intelligence 

**Authors**: Baiyu Pan, Daqin Luo, Junpeng Yang, Jiyuan Wang, Yixuan Zhang, Hailin Shi, Jichao Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21199)  

**Abstract**: When large vision-language models are applied to the field of robotics, they encounter problems that are simple for humans yet error-prone for models. Such issues include confusion between third-person and first-person perspectives and a tendency to overlook information in video endings during temporal reasoning. To address these challenges, we propose Thinker, a large vision-language foundation model designed for embodied intelligence. We tackle the aforementioned issues from two perspectives. Firstly, we construct a large-scale dataset tailored for robotic perception and reasoning, encompassing ego-view videos, visual grounding, spatial understanding, and chain-of-thought data. Secondly, we introduce a simple yet effective approach that substantially enhances the model's capacity for video comprehension by jointly incorporating key frames and full video sequences as inputs. Our model achieves state-of-the-art results on two of the most commonly used benchmark datasets in the field of task planning. 

---
# ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling 

**Authors**: Yuchen Yang, Yaru Zhao, Pu Yang, Shaowei Wang, Zhi-Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21198)  

**Abstract**: While Mixture-of-Experts (MoE) architectures substantially bolster the expressive power of large-language models, their prohibitive memory footprint severely impedes the practical deployment on resource-constrained edge devices, especially when model behavior must be preserved without relying on lossy quantization. In this paper, we present ZipMoE, an efficient and semantically lossless on-device MoE serving system. ZipMoE exploits the synergy between the hardware properties of edge devices and the statistical redundancy inherent to MoE parameters via a caching-scheduling co-design with provable performance guarantee. Fundamentally, our design shifts the paradigm of on-device MoE inference from an I/O-bound bottleneck to a compute-centric workflow that enables efficient parallelization. We implement a prototype of ZipMoE and conduct extensive experiments on representative edge computing platforms using popular open-source MoE models and real-world workloads. Our evaluation reveals that ZipMoE achieves up to $72.77\%$ inference latency reduction and up to $6.76\times$ higher throughput than the state-of-the-art systems. 

---
# From Linear Input to Hierarchical Structure: Function Words as Statistical Cues for Language Learning 

**Authors**: Xiulin Yang, Heidi Getz, Ethan Gotlieb Wilcox  

**Link**: [PDF](https://arxiv.org/pdf/2601.21191)  

**Abstract**: What statistical conditions support learning hierarchical structure from linear input? In this paper, we address this question by focusing on the statistical distribution of function words. Function words have long been argued to play a crucial role in language acquisition due to their distinctive distributional properties, including high frequency, reliable association with syntactic structure, and alignment with phrase boundaries. We use cross-linguistic corpus analysis to first establish that all three properties are present across 186 studied languages. Next, we use a combination of counterfactual language modeling and ablation experiments to show that language variants preserving all three properties are more easily acquired by neural learners, with frequency and structural association contributing more strongly than boundary alignment. Follow-up probing and ablation analyses further reveal that different learning conditions lead to systematically different reliance on function words, indicating that similar performance can arise from distinct internal mechanisms. 

---
# Adaptive and Robust Cost-Aware Proof of Quality for Decentralized LLM Inference Networks 

**Authors**: Arther Tian, Alex Ding, Frank Chen, Simon Wu, Aaron Chan  

**Link**: [PDF](https://arxiv.org/pdf/2601.21189)  

**Abstract**: Decentralized large language model inference networks require lightweight mechanisms to reward high quality outputs under heterogeneous latency and cost. Proof of Quality provides scalable verification by sampling evaluator nodes that score candidate outputs, then aggregating their scores into a consensus signal that determines rewards. However, evaluator heterogeneity and malicious score manipulation can distort consensus and inflate payouts, which weakens incentive alignment in open participation settings.
This paper extends a cost-aware Proof of Quality mechanism by adding adversary-resilient consensus formation. We study robust aggregation rules, including median and trimmed mean, and an adaptive trust-weighted consensus that updates evaluator weights from deviation signals. Using question answering and summarization workloads with a ground truth proxy for offline analysis, we quantify evaluator reliability and show strong variance across evaluators, including task-dependent misalignment that can invert correlations. We then evaluate robustness under four adversarial strategies, including noise injection, boosting, sabotage, and intermittent manipulation, across a sweep of malicious ratios and evaluator sample sizes. Our results show that robust aggregation improves consensus alignment with the ground truth proxy and reduces sensitivity to noisy and strategic attacks compared with simple averaging. We further characterize the operational trade-off introduced by evaluator sampling, where larger evaluator sets reduce evaluator rewards and increase payoff variance while inference rewards remain relatively stable in our configuration. These findings motivate robust consensus as a default component for cost-aware Proof of Quality and provide practical guidance for selecting evaluator sampling parameters under adversarial risk and resource constraints. 

---
# Rethinking Refinement: Correcting Generative Bias without Noise Injection 

**Authors**: Xin Peng, Ang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.21182)  

**Abstract**: Generative models, including diffusion and flow-based models, often exhibit systematic biases that degrade sample quality, particularly in high-dimensional settings. We revisit refinement methods and show that effective bias correction can be achieved as a post-hoc procedure, without noise injection or multi-step resampling of the sampling process. We propose a flow-matching-based \textbf{Bi-stage Flow Refinement (BFR)} framework with two refinement strategies operating at different stages: latent space alignment for approximately invertible generators and data space refinement trained with lightweight augmentations. Unlike previous refiners that perturb sampling dynamics, BFR preserves the original ODE trajectory and applies deterministic corrections to generated samples. Experiments on MNIST, CIFAR-10, and FFHQ at 256x256 resolution demonstrate consistent improvements in fidelity and coverage; notably, starting from base samples with FID 3.95, latent space refinement achieves a \textbf{state-of-the-art} FID of \textbf{1.46} on MNIST using only a single additional function evaluation (1-NFE), while maintaining sample diversity. 

---
# AC2L-GAD: Active Counterfactual Contrastive Learning for Graph Anomaly Detection 

**Authors**: Kamal Berahmand, Saman Forouzandeh, Mehrnoush Mohammadi, Parham Moradi, Mahdi Jalili  

**Link**: [PDF](https://arxiv.org/pdf/2601.21171)  

**Abstract**: Graph anomaly detection aims to identify abnormal patterns in networks, but faces significant challenges from label scarcity and extreme class imbalance. While graph contrastive learning offers a promising unsupervised solution, existing methods suffer from two critical limitations: random augmentations break semantic consistency in positive pairs, while naive negative sampling produces trivial, uninformative contrasts. We propose AC2L-GAD, an Active Counterfactual Contrastive Learning framework that addresses both limitations through principled counterfactual reasoning. By combining information-theoretic active selection with counterfactual generation, our approach identifies structurally complex nodes and generates anomaly-preserving positive augmentations alongside normal negative counterparts that provide hard contrasts, while restricting expensive counterfactual generation to a strategically selected subset. This design reduces computational overhead by approximately 65% compared to full-graph counterfactual generation while maintaining detection quality. Experiments on nine benchmark datasets, including real-world financial transaction graphs from GADBench, show that AC2L-GAD achieves competitive or superior performance compared to state-of-the-art baselines, with notable gains in datasets where anomalies exhibit complex attribute-structure interactions. 

---
# Output-Space Search: Targeting LLM Generations in a Frozen Encoder-Defined Output Space 

**Authors**: Tobias Materzok  

**Link**: [PDF](https://arxiv.org/pdf/2601.21169)  

**Abstract**: We introduce Output-Space Search (OS-Search), which turns LLM generation into endpoint search. An outer loop selects a target z* in a frozen encoder-defined 3D output space Z, and a retrieval-grounded policy trained with sequence-level RL generates outputs whose coordinates land near z* under standard autoregressive decoding. This enables parallel sweeps and black-box optimization in Z without path-dependent token/program search. On stories, sweeping Z (text) yields 3.1x higher LLM-scored diversity than prompt-chaining. On code, Bayesian optimization over Z (code) improves an objective withheld from the controller under matched inference budgets while preserving validity. 

---
# A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware and Reliable Reasoning 

**Authors**: Jiate Liu, Zebin Chen, Shaobo Qiao, Mingchen Ju, Danting Zhang, Bocheng Han, Shuyue Yu, Xin Shu, Jingling Wu, Dong Wen, Xin Cao, Guanfeng Liu, Zhengyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21162)  

**Abstract**: Graph Retrieval-Augmented Generation (Graph-RAG) enhances multihop question answering by organizing corpora into knowledge graphs and routing evidence through relational structure. However, practical deployments face two persistent bottlenecks: (i) mixed-difficulty workloads where one-size-fits-all retrieval either wastes cost on easy queries or fails on hard multihop cases, and (ii) extraction loss, where graph abstraction omits fine-grained qualifiers that remain only in source text. We present A2RAG, an adaptive-and-agentic GraphRAG framework for cost-aware and reliable reasoning. A2RAG couples an adaptive controller that verifies evidence sufficiency and triggers targeted refinement only when necessary, with an agentic retriever that progressively escalates retrieval effort and maps graph signals back to provenance text to remain robust under extraction loss and incomplete graphs. Experiments on HotpotQA and 2WikiMultiHopQA demonstrate that A2RAG achieves +9.9/+11.8 absolute gains in Recall@2, while cutting token consumption and end-to-end latency by about 50% relative to iterative multihop baselines. 

---
# Can Neural Networks Learn Small Algebraic Worlds? An Investigation Into the Group-theoretic Structures Learned By Narrow Models Trained To Predict Group Operations 

**Authors**: Henry Kvinge, Andrew Aguilar, Nayda Farnsworth, Grace O'Brien, Robert Jasper, Sarah Scullen, Helen Jenne  

**Link**: [PDF](https://arxiv.org/pdf/2601.21150)  

**Abstract**: While a real-world research program in mathematics may be guided by a motivating question, the process of mathematical discovery is typically open-ended. Ideally, exploration needed to answer the original question will reveal new structures, patterns, and insights that are valuable in their own right. This contrasts with the exam-style paradigm in which the machine learning community typically applies AI to math. To maximize progress in mathematics using AI, we will need to go beyond simple question answering. With this in mind, we explore the extent to which narrow models trained to solve a fixed mathematical task learn broader mathematical structure that can be extracted by a researcher or other AI system. As a basic test case for this, we use the task of training a neural network to predict a group operation (for example, performing modular arithmetic or composition of permutations). We describe a suite of tests designed to assess whether the model captures significant group-theoretic notions such as the identity element, commutativity, or subgroups. Through extensive experimentation we find evidence that models learn representations capable of capturing abstract algebraic properties. For example, we find hints that models capture the commutativity of modular arithmetic. We are also able to train linear classifiers that reliably distinguish between elements of certain subgroups (even though no labels for these subgroups are included in the data). On the other hand, we are unable to extract notions such as the concept of the identity element. Together, our results suggest that in some cases the representations of even small neural networks can be used to distill interesting abstract structure from new mathematical objects. 

---
# Mobility-Embedded POIs: Learning What A Place Is and How It Is Used from Human Movement 

**Authors**: Maria Despoina Siampou, Shushman Choudhury, Shang-Ling Hsu, Neha Arora, Cyrus Shahabi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21149)  

**Abstract**: Recent progress in geospatial foundation models highlights the importance of learning general-purpose representations for real-world locations, particularly points-of-interest (POIs) where human activity concentrates. Existing approaches, however, focus primarily on place identity derived from static textual metadata, or learn representations tied to trajectory context, which capture movement regularities rather than how places are actually used (i.e., POI's function). We argue that POI function is a missing but essential signal for general POI representations. We introduce Mobility-Embedded POIs (ME-POIs), a framework that augments POI embeddings derived, from language models with large-scale human mobility data to learn POI-centric, context-independent representations grounded in real-world usage. ME-POIs encodes individual visits as temporally contextualized embeddings and aligns them with learnable POI representations via contrastive learning to capture usage patterns across users and time. To address long-tail sparsity, we propose a novel mechanism that propagates temporal visit patterns from nearby, frequently visited POIs across multiple spatial scales. We evaluate ME-POIs on five newly proposed map enrichment tasks, testing its ability to capture both the identity and function of POIs. Across all tasks, augmenting text-based embeddings with ME-POIs consistently outperforms both text-only and mobility-only baselines. Notably, ME-POIs trained on mobility data alone can surpass text-only models on certain tasks, highlighting that POI function is a critical component of accurate and generalizable POI representations. 

---
# Optimization and Mobile Deployment for Anthropocene Neural Style Transfer 

**Authors**: Po-Hsun Chen, Ivan C. H. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.21141)  

**Abstract**: This paper presents AnthropoCam, a mobile-based neural style transfer (NST) system optimized for the visual synthesis of Anthropocene environments. Unlike conventional artistic NST, which prioritizes painterly abstraction, stylizing human-altered landscapes demands a careful balance between amplifying material textures and preserving semantic legibility. Industrial infrastructures, waste accumulations, and modified ecosystems contain dense, repetitive patterns that are visually expressive yet highly susceptible to semantic erosion under aggressive style transfer.
To address this challenge, we systematically investigate the impact of NST parameter configurations on the visual translation of Anthropocene textures, including feature layer selection, style and content loss weighting, training stability, and output resolution. Through controlled experiments, we identify an optimal parameter manifold that maximizes stylistic expression while preventing semantic erasure. Our results demonstrate that appropriate combinations of convolutional depth, loss ratios, and resolution scaling enable the faithful transformation of anthropogenic material properties into a coherent visual language.
Building on these findings, we implement a low-latency, feed-forward NST pipeline deployed on mobile devices. The system integrates a React Native frontend with a Flask-based GPU backend, achieving high-resolution inference within 3-5 seconds on general mobile hardware. This enables real-time, in-situ visual intervention at the site of image capture, supporting participatory engagement with Anthropocene landscapes.
By coupling domain-specific NST optimization with mobile deployment, AnthropoCam reframes neural style transfer as a practical and expressive tool for real-time environmental visualization in the Anthropocene. 

---
# PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs 

**Authors**: Artem Dementyev, Wazeer Zulfikar, Sinan Hersek, Pascal Getreuer, Anurag Kumar, Vivek Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.21124)  

**Abstract**: Current multimodal LLMs process audio as a mono stream, ignoring the rich spatial information essential for embodied AI. Existing spatial audio models, conversely, are constrained to fixed microphone geometries, preventing deployment across diverse devices. We present PhaseCoder, a transformer-only spatial audio encoder that is agnostic to microphone geometry. PhaseCoder takes raw multichannel audio and microphone coordinates as inputs to perform localization and produces robust spatial embeddings. We demonstrate that Gemma 3n LLM can be fine-tuned to reason over "Spatial Audio Tokens" produced by PhaseCoder. We show our encoder achieves state-of-the-art results on microphone-invariant localization benchmarks and, for the first time, enables an LLM to perform complex spatial reasoning and targeted transcription tasks from an arbitrary microphone array. 

---
# AI-Assisted Engineering Should Track the Epistemic Status and Temporal Validity of Architectural Decisions 

**Authors**: Sankalp Gilda, Shlok Gilda  

**Link**: [PDF](https://arxiv.org/pdf/2601.21116)  

**Abstract**: This position paper argues that AI-assisted software engineering requires explicit mechanisms for tracking the epistemic status and temporal validity of architectural decisions. LLM coding assistants generate decisions faster than teams can validate them, yet no widely-adopted framework distinguishes conjecture from verified knowledge, prevents trust inflation through conservative aggregation, or detects when evidence expires. We propose three requirements for responsible AI-assisted engineering: (1) epistemic layers that separate unverified hypotheses from empirically validated claims, (2) conservative assurance aggregation grounded in the Gödel t-norm that prevents weak evidence from inflating confidence, and (3) automated evidence decay tracking that surfaces stale assumptions before they cause failures. We formalize these requirements as the First Principles Framework (FPF), ground its aggregation semantics in fuzzy logic, and define a quintet of invariants that any valid aggregation operator must satisfy. Our retrospective audit applying FPF criteria to two internal projects found that 20-25% of architectural decisions had stale evidence within two months, validating the need for temporal accountability. We outline research directions including learnable aggregation operators, federated evidence sharing, and SMT-based claim validation. 

---
# Multi-task Code LLMs: Data Mix or Model Merge? 

**Authors**: Mingzhi Zhu, Boris Sobolev, Rahul Krishna, Raju Pavuluri, Stacy Patterson, Michele Merler  

**Link**: [PDF](https://arxiv.org/pdf/2601.21115)  

**Abstract**: Recent research advocates deploying smaller, specialized code LLMs in agentic frameworks alongside frontier models, sparking interest in efficient strategies for multi-task learning that balance performance, constraints, and costs. We compare two approaches for creating small, multi-task code LLMs: data mixing versus model merging. We conduct extensive experiments across two model families (Qwen Coder and DeepSeek Coder) at two scales (2B and 7B parameters), fine-tuning them for code generation and code summarization tasks. Our evaluation on HumanEval, MBPP, and CodeXGlue benchmarks reveals that model merging achieves the best overall performance at larger scale across model families, retaining 96% of specialized model performance on code generation tasks while maintaining summarization capabilities. Notably, merged models can even surpass individually fine-tuned models, with our best configuration of Qwen Coder 2.5 7B model achieving 92.7% Pass@1 on HumanEval compared to 90.9% for its task-specific fine-tuned equivalent. At a smaller scale we find instead data mixing to be a preferred strategy. We further introduce a weight analysis technique to understand how different tasks affect model parameters and their implications for merging strategies. The results suggest that careful merging and mixing strategies can effectively combine task-specific capabilities without significant performance degradation, making them ideal for resource-constrained deployment scenarios. 

---
# SteerEval: A Framework for Evaluating Steerability with Natural Language Profiles for Recommendation 

**Authors**: Joyce Zhou, Weijie Zhou, Doug Turnbull, Thorsten Joachims  

**Link**: [PDF](https://arxiv.org/pdf/2601.21105)  

**Abstract**: Natural-language user profiles have recently attracted attention not only for improved interpretability, but also for their potential to make recommender systems more steerable. By enabling direct editing, natural-language profiles allow users to explicitly articulate preferences that may be difficult to infer from past behavior. However, it remains unclear whether current natural-language-based recommendation methods can follow such steering commands. While existing steerability evaluations have shown some success for well-recognized item attributes (e.g., movie genres), we argue that these benchmarks fail to capture the richer forms of user control that motivate steerable recommendations. To address this gap, we introduce SteerEval, an evaluation framework designed to measure more nuanced and diverse forms of steerability by using interventions that range from genres to content-warning for movies. We assess the steerability of a family of pretrained natural-language recommenders, examine the potential and limitations of steering on relatively niche topics, and compare how different profile and recommendation interventions impact steering effectiveness. Finally, we offer practical design suggestions informed by our findings and discuss future steps in steerable recommender design. 

---
# Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed 

**Authors**: Minjae Kwon, Josephine Lamp, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.21094)  

**Abstract**: Safe Reinforcement Learning (RL) algorithms are typically evaluated under fixed training conditions. We investigate whether training-time safety guarantees transfer to deployment under distribution shift, using diabetes management as a safety-critical testbed. We benchmark safe RL algorithms on a unified clinical simulator and reveal a safety generalization gap: policies satisfying constraints during training frequently violate safety requirements on unseen patients. We demonstrate that test-time shielding, which filters unsafe actions using learned dynamics models, effectively restores safety across algorithms and patient populations. Across eight safe RL algorithms, three diabetes types, and three age groups, shielding achieves Time-in-Range gains of 13--14\% for strong baselines such as PPO-Lag and CPO while reducing clinical risk index and glucose variability. Our simulator and benchmark provide a platform for studying safety under distribution shift in safety-critical control domains. Code is available at this https URL and this https URL. 

---
# Deep Reinforcement Learning for Fault-Adaptive Routing in Eisenstein-Jacobi Interconnection Topologies 

**Authors**: Mohammad Walid Charrwi, Zaid Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2601.21090)  

**Abstract**: The increasing density of many-core architectures necessitates interconnection networks that are both high-performance and fault-resilient. Eisenstein-Jacobi (EJ) networks, with their symmetric 6-regular topology, offer superior topological properties but challenge traditional routing heuristics under fault conditions. This paper evaluates three routing paradigms in faulty EJ environments: deterministic Greedy Adaptive Routing, theoretically optimal Dijkstra's algorithm, and a reinforcement learning (RL)-based approach. Using a multi-objective reward function to penalize fault proximity and reward path efficiency, the RL agent learns to navigate around clustered failures that typically induce dead-ends in greedy geometric routing. Dijkstra's algorithm establishes the theoretical performance ceiling by computing globally optimal paths with complete topology knowledge, revealing the true connectivity limits of faulty networks. Quantitative analysis at nine faulty nodes shows greedy routing catastrophically degrades to 10% effective reachability and packet delivery, while Dijkstra proves 52-54% represents the topological optimum. The RL agent achieves 94% effective reachability and 91% packet delivery, making it suitable for distributed deployment. Furthermore, throughput evaluations demonstrate that RL sustains over 90% normalized throughput across all loads, actually outperforming Dijkstra under congestion through implicit load balancing strategies. These results establish RL-based adaptive policies as a practical solution that bridges the gap between greedy's efficiency and Dijkstra's optimality, providing robust, self-healing communication in fault-prone interconnection networks without requiring the global topology knowledge or computational overhead of optimal algorithms. 

---
# LOCUS: Low-Dimensional Model Embeddings for Efficient Model Exploration, Comparison, and Selection 

**Authors**: Shivam Patel, William Cocke, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2601.21082)  

**Abstract**: The rapidly growing ecosystem of Large Language Models (LLMs) makes it increasingly challenging to manage and utilize the vast and dynamic pool of models effectively. We propose LOCUS, a method that produces low-dimensional vector embeddings that compactly represent a language model's capabilities across queries. LOCUS is an attention-based approach that generates embeddings by a deterministic forward pass over query encodings and evaluation scores via an encoder model, enabling seamless incorporation of new models to the pool and refinement of existing model embeddings without having to perform any retraining. We additionally train a correctness predictor that uses model embeddings and query encodings to achieve state-of-the-art routing accuracy on unseen queries. Experiments show that LOCUS needs up to 4.8x fewer query evaluation samples than baselines to produce informative and robust embeddings. Moreover, the learned embedding space is geometrically meaningful: proximity reflects model similarity, enabling a range of downstream applications including model comparison and clustering, model portfolio selection, and resilient proxies of unavailable models. 

---
# Towards Comprehensive Benchmarking Infrastructure for LLMs In Software Engineering 

**Authors**: Daniel Rodriguez-Cardenas, Xiaochang Li, Marcos Macedo, Antonio Mastropaolo, Dipin Khati, Yuan Tian, Huajie Shao, Denys Poshyvanyk  

**Link**: [PDF](https://arxiv.org/pdf/2601.21070)  

**Abstract**: Large language models for code are advancing fast, yet our ability to evaluate them lags behind. Current benchmarks focus on narrow tasks and single metrics, which hide critical gaps in robustness, interpretability, fairness, efficiency, and real-world usability. They also suffer from inconsistent data engineering practices, limited software engineering context, and widespread contamination issues. To understand these problems and chart a path forward, we combined an in-depth survey of existing benchmarks with insights gathered from a dedicated community workshop. We identified three core barriers to reliable evaluation: the absence of software-engineering-rich datasets, overreliance on ML-centric metrics, and the lack of standardized, reproducible data pipelines. Building on these findings, we introduce BEHELM, a holistic benchmarking infrastructure that unifies software-scenario specification with multi-metric evaluation. BEHELM provides a structured way to assess models across tasks, languages, input and output granularities, and key quality dimensions. Our goal is to reduce the overhead currently required to construct benchmarks while enabling a fair, realistic, and future-proof assessment of LLMs in software engineering. 

---
# Textual Equilibrium Propagation for Deep Compound AI Systems 

**Authors**: Minghui Chen, Wenlong Deng, James Zou, Han Yu, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.21064)  

**Abstract**: Large language models (LLMs) are increasingly deployed as part of compound AI systems that coordinate multiple modules (e.g., retrievers, tools, verifiers) over long-horizon workflows. Recent approaches that propagate textual feedback globally (e.g., TextGrad) make it feasible to optimize such pipelines, but we find that performance degrades as system depth grows. In particular, long-horizon agentic workflows exhibit two depth-scaling failure modes: 1) exploding textual gradient, where textual feedback grows exponentially with depth, leading to prohibitively long message and amplifies evaluation biases; and 2) vanishing textual gradient, where limited long-context ability causes models overemphasize partial feedback and compression of lengthy feedback causes downstream messages to lose specificity gradually as they propagate many hops upstream. To mitigate these issues, we introduce Textual Equilibrium Propagation (TEP), a local learning principle inspired by Equilibrium Propagation in energy-based models. TEP includes two phases: 1) a free phase where a local LLM critics iteratively refine prompts until reaching equilibrium (no further improvements are suggested); and 2) a nudged phase which applies proximal prompt edits with bounded modification intensity, using task-level objectives that propagate via forward signaling rather than backward feedback chains. This design supports local prompt optimization followed by controlled adaptation toward global goals without the computational burden and signal degradation of global textual backpropagation. Across long-horizon QA benchmarks and multi-agent tool-use dataset, TEP consistently improves accuracy and efficiency over global propagation methods such as TextGrad. The gains grows with depth, while preserving the practicality of black-box LLM components in deep compound AI system. 

---
# SMKC: Sketch Based Kernel Correlation Images for Variable Cardinality Time Series Anomaly Detection 

**Authors**: Haokun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.21050)  

**Abstract**: Conventional anomaly detection in multivariate time series relies on the assumption that the set of observed variables remains static. In operational environments, however, monitoring systems frequently experience sensor churn. Signals may appear, disappear, or be renamed, creating data windows where the cardinality varies and may include values unseen during training. To address this challenge, we propose SMKC, a framework that decouples the dynamic input structure from the anomaly detector. We first employ permutation-invariant feature hashing to sketch raw inputs into a fixed size state sequence. We then construct a hybrid kernel image to capture global temporal structure through pairwise comparisons of the sequence and its derivatives. The model learns normal patterns using masked reconstruction and a teacher-student prediction objective. Our evaluation reveals that robust log-distance channels provide the primary discriminative signal, whereas cosine representations often fail to capture sufficient contrast. Notably, we find that a detector using random projections and nearest neighbors on the SMKC representation performs competitively with fully trained baselines without requiring gradient updates. This highlights the effectiveness of the representation itself and offers a practical cold-start solution for resource-constrained deployments. 

---
# Log2Motion: Biomechanical Motion Synthesis from Touch Logs 

**Authors**: Michał Patryk Miazga, Hannah Bussmann, Antti Oulasvirta, Patrick Ebel  

**Link**: [PDF](https://arxiv.org/pdf/2601.21043)  

**Abstract**: Touch data from mobile devices are collected at scale but reveal little about the interactions that produce them. While biomechanical simulations can illuminate motor control processes, they have not yet been developed for touch interactions. To close this gap, we propose a novel computational problem: synthesizing plausible motion directly from logs. Our key insight is a reinforcement learning-driven musculoskeletal forward simulation that generates biomechanically plausible motion sequences consistent with events recorded in touch logs. We achieve this by integrating a software emulator into a physics simulator, allowing biomechanical models to manipulate real applications in real-time. Log2Motion produces rich syntheses of user movements from touch logs, including estimates of motion, speed, accuracy, and effort. We assess the plausibility of generated movements by comparing against human data from a motion capture study and prior findings, and demonstrate Log2Motion in a large-scale dataset. Biomechanical motion synthesis provides a new way to understand log data, illuminating the ergonomics and motor control underlying touch interactions. 

---
# Thinking in Frames: How Visual Context and Test-Time Scaling Empower Video Reasoning 

**Authors**: Chengzu Li, Zanyi Wang, Jiaang Li, Yi Xu, Han Zhou, Huanyu Zhang, Ruichuan An, Dengyang Jiang, Zhaochong An, Ivan Vulić, Serge Belongie, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2601.21037)  

**Abstract**: Vision-Language Models have excelled at textual reasoning, but they often struggle with fine-grained spatial understanding and continuous action planning, failing to simulate the dynamics required for complex visual reasoning. In this work, we formulate visual reasoning by means of video generation models, positing that generated frames can act as intermediate reasoning steps between initial states and solutions. We evaluate their capacity in two distinct regimes: Maze Navigation for sequential discrete planning with low visual change and Tangram Puzzle for continuous manipulation with high visual change. Our experiments reveal three critical insights: (1) Robust Zero-Shot Generalization: In both tasks, the model demonstrates strong performance on unseen data distributions without specific finetuning. (2) Visual Context: The model effectively uses visual context as explicit control, such as agent icons and tangram shapes, enabling it to maintain high visual consistency and adapt its planning capability robustly to unseen patterns. (3) Visual Test-Time Scaling: We observe a test-time scaling law in sequential planning; increasing the generated video length (visual inference budget) empowers better zero-shot generalization to spatially and temporally complex paths. These findings suggest that video generation is not merely a media tool, but a scalable, generalizable paradigm for visual reasoning. 

---
# SIGMA-PPG: Statistical-prior Informed Generative Masking Architecture for PPG Foundation Model 

**Authors**: Zongheng Guo, Tao Chen, Yang Jiao, Yi Pan, Xiao Hu, Manuela Ferrario  

**Link**: [PDF](https://arxiv.org/pdf/2601.21031)  

**Abstract**: Current foundation model for photoplethysmography (PPG) signals is challenged by the intrinsic redundancy and noise of the signal. Standard masked modeling often yields trivial solutions while contrastive methods lack morphological precision. To address these limitations, we propose a Statistical-prior Informed Generative Masking Architecture (SIGMA-PPG), a generative foundation model featuring a Prior-Guided Adversarial Masking mechanism, where a reinforcement learning-driven teacher leverages statistical priors to create challenging learning paths that prevent overfitting to noise. We also incorporate a semantic consistency constraint via vector quantization to ensure that physiologically identical waveforms (even those altered by recording artifacts or minor perturbations) map to shared indices. This enhances codebook semantic density and eliminates redundant feature structures. Pre-trained on over 120,000 hours of data, SIGMA-PPG achieves superior average performance compared to five state-of-the-art baselines across 12 diverse downstream tasks. The code is available at this https URL. 

---
# "Unlimited Realm of Exploration and Experimentation": Methods and Motivations of AI-Generated Sexual Content Creators 

**Authors**: Jaron Mink, Lucy Qin, Elissa M. Redmiles  

**Link**: [PDF](https://arxiv.org/pdf/2601.21028)  

**Abstract**: AI-generated media is radically changing the way content is both consumed and produced on the internet, and in no place is this potentially more visible than in sexual content. AI-generated sexual content (AIG-SC) is increasingly enabled by an ecosystem of individual AI developers, specialized third-party applications, and foundation model providers. AIG-SC raises a number of concerns from old debates about the line between pornography and obscenity, to newer debates about fair use and labor displacement (in this case, of sex workers), and spurred new regulations to curb the spread of non-consensual intimate imagery (NCII) created using the same technology used to create AIG-SC. However, despite the growing prevalence of AIG-SC, little is known about its creators, their motivations, and what types of content they produce. To inform effective governance in this space, we perform an in-depth study to understand what AIG-SC creators make, along with how and why they make it. Interviews of 28 AIG-SC creators, ranging from hobbyists to entrepreneurs to those who moderate communities of hundreds of thousands of other creators, reveal a wide spectrum of motivations, including sexual exploration, creative expression, technical experimentation, and in a handful of cases, the creation of NCII. 

---
# Conditional Denoising Model as a Physical Surrogate Model 

**Authors**: José Afonso, Pedro Viegas, Rodrigo Ventura, Vasco Guerra  

**Link**: [PDF](https://arxiv.org/pdf/2601.21021)  

**Abstract**: Surrogate modeling for complex physical systems typically faces a trade-off between data-fitting accuracy and physical consistency. Physics-consistent approaches typically treat physical laws as soft constraints within the loss function, a strategy that frequently fails to guarantee strict adherence to the governing equations, or rely on post-processing corrections that do not intrinsically learn the underlying solution geometry. To address these limitations, we introduce the {Conditional Denoising Model (CDM)}, a generative model designed to learn the geometry of the physical manifold itself. By training the network to restore clean states from noisy ones, the model learns a vector field that points continuously towards the valid solution subspace. We introduce a time-independent formulation that transforms inference into a deterministic fixed-point iteration, effectively projecting noisy approximations onto the equilibrium manifold. Validated on a low-temperature plasma physics and chemistry benchmark, the CDM achieves higher parameter and data efficiency than physics-consistent baselines. Crucially, we demonstrate that the denoising objective acts as a powerful implicit regularizer: despite never seeing the governing equations during training, the model adheres to physical constraints more strictly than baselines trained with explicit physics losses. 

---
# Solver-in-the-Loop: MDP-Based Benchmarks for Self-Correction and Behavioral Rationality in Operations Research 

**Authors**: Ruicheng Ao, David Simchi-Levi, Xinshang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.21008)  

**Abstract**: Operations Research practitioners routinely debug infeasible models through an iterative process: analyzing Irreducible Infeasible Subsystems (\IIS{}), identifying constraint conflicts, and systematically repairing formulations until feasibility is achieved. Yet existing LLM benchmarks evaluate OR as one-shot translation -- given a problem description, generate solver code -- ignoring this diagnostic loop entirely. We introduce two benchmarks that place the \textbf{solver in the evaluation loop}. \textbf{\ORDebug{}} evaluates iterative self-correction through 5,000+ problems spanning 9 error types; each repair action triggers solver re-execution and \IIS{} recomputation, providing deterministic, verifiable feedback. \textbf{\ORBias{}} evaluates behavioral rationality through 2,000 newsvendor instances (1,000 ID + 1,000 OOD), measuring systematic deviations from closed-form optimal policies. Across 26 models and 12,000+ samples, we find that domain-specific RLVR training enables an 8B model to surpass frontier APIs: 95.3\% vs 86.2\% recovery rate (+9.1\%), 62.4\% vs 47.8\% diagnostic accuracy (+14.6\%), and 2.25 vs 3.78 steps to resolution (1.7$\times$ faster). On \ORBias{}, curriculum training achieves the only negative ID$\rightarrow$OOD bias drift among models evaluated (-9.6\%), reducing systematic bias by 48\% (from 20.0\% to 10.4\%). These results demonstrate that process-level evaluation with verifiable oracles enables targeted training that outperforms scale. 

---
# UrduBench: An Urdu Reasoning Benchmark using Contextually Ensembled Translations with Human-in-the-Loop 

**Authors**: Muhammad Ali Shafique, Areej Mehboob, Layba Fiaz, Muhammad Usman Qadeer, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2601.21000)  

**Abstract**: Recent advances in large language models (LLMs) have led to strong reasoning capabilities; however, evaluating such models in low-resource languages remains challenging due to the lack of standardized benchmarks. In particular, Urdu reasoning evaluation has been limited by the sensitivity of machine translation and an emphasis on general language tasks rather than reasoning benchmarks. In this paper, we propose a contextually ensembled translation framework with human-in-the-loop validation that leverages multiple translation systems to develop Urdu reasoning benchmarks while preserving contextual and structural integrity. Using this framework, we translate widely adopted reasoning and question-answering benchmarks, including MGSM, MATH-500, CommonSenseQA, and OpenBookQA, into Urdu, collectively referred to as UrduBench, and conduct a comprehensive evaluation of both reasoning-oriented and instruction-tuned LLMs across multiple prompting strategies. Our analysis reveals performance differences across (1) four datasets, (2) five task difficulty levels, (3) diverse model architectures, (4) multiple model scaling settings, and (5) language consistency tests. We find that multi-step and symbolic reasoning tasks pose significant challenges in Urdu, and that stable language alignment is a critical prerequisite for robust reasoning. Overall, our work establishes a scalable methodology for standardized reasoning evaluation in Urdu and provides empirical insights into multilingual reasoning failures. This experimental setup is also broadly applicable to other low-resource languages. The code and datasets will be publicly released. 

---
# The Depth Delusion: Why Transformers Should Be Wider, Not Deeper 

**Authors**: Md Muhtasim Munif Fahim, Md Rezaul Karim  

**Link**: [PDF](https://arxiv.org/pdf/2601.20994)  

**Abstract**: Neural scaling laws describe how language model loss decreases with parameters and data, but treat architecture as interchangeable--a billion parameters could arise from a shallow-wide model (10 layers & 8,192 hidden dimension) or a deep-narrow one (80 layers & 2,048 hidden dimension). We propose architecture-conditioned scaling laws decomposing this dependence, finding that optimal depth scales as D* ~ C^0.12 while optimal width scales as W* ~ C^0.34, meaning width should grow 2.8x faster than depth. We discover a critical depth phenomenon: beyond D_crit ~ W^0.44 (sublinear in W), adding layers increases loss despite adding parameters--the Depth Delusion. Empirically, we validate these findings across 30 transformer architectures spanning 17M to 7B parameters, each trained on representative high-compute samples, achieving R^2 = 0.922. Our central finding: at 7B scale, a 64-layer model (6.38B params) underperforms a 32-layer model (6.86B params) by 0.12 nats, despite being significantly deeper. This demonstrates that optimal depth-width tradeoffs persist at the production scale. 

---
# Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges 

**Authors**: Chen Feng, Minghe Shen, Ananth Balashankar, Carsten Gerner-Beuerle, Miguel R. D. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2601.20913)  

**Abstract**: Reliable certification of Large Language Models (LLMs)-verifying that failure rates are below a safety threshold-is critical yet challenging. While "LLM-as-a-Judge" offers scalability, judge imperfections, noise, and bias can invalidate statistical guarantees. We introduce a "Noisy but Valid" hypothesis testing framework to address this. By leveraging a small human-labelled calibration set to estimate the judge's True Positive and False Positive Rates (TPR/FPR), we derive a variance-corrected critical threshold applied to a large judge-labelled dataset. Crucially, our framework theoretically guarantees finite-sample Type-I error control (validity) despite calibration uncertainty. This distinguishes our work from Prediction-Powered Inference (PPI), positioning our method as a diagnostic tool that explicitly models judge behavior rather than a black-box estimator. Our contributions include: (1) Theoretical Guarantees: We derive the exact conditions under which noisy testing yields higher statistical power than direct evaluation; (2) Empirical Validation: Experiments on Jigsaw Comment, Hate Speech and SafeRLHF confirm our theory; (3) The Oracle Gap: We reveal a significant performance gap between practical methods and the theoretical "Oracle" (perfectly known judge parameters), quantifying the cost of estimation. Specifically, we provide the first systematic treatment of the imperfect-judge setting, yielding interpretable diagnostics of judge reliability and clarifying how evaluation power depends on judge quality, dataset size, and certification levels. Together, these results sharpen understanding of statistical evaluation with LLM judges, and highlight trade-offs among competing inferential tools. 

---
# Non-Markov Multi-Round Conversational Image Generation with History-Conditioned MLLMs 

**Authors**: Haochen Zhang, Animesh Sinha, Felix Juefei-Xu, Haoyu Ma, Kunpeng Li, Zhipeng Fan, Meng Dong, Xiaoliang Dai, Tingbo Hou, Peizhao Zhang, Zecheng He  

**Link**: [PDF](https://arxiv.org/pdf/2601.20911)  

**Abstract**: Conversational image generation requires a model to follow user instructions across multiple rounds of interaction, grounded in interleaved text and images that accumulate as chat history. While recent multimodal large language models (MLLMs) can generate and edit images, most existing multi-turn benchmarks and training recipes are effectively Markov: the next output depends primarily on the most recent image, enabling shortcut solutions that ignore long-range history. In this work we formalize and target the more challenging non-Markov setting, where a user may refer back to earlier states, undo changes, or reference entities introduced several rounds ago. We present (i) non-Markov multi-round data construction strategies, including rollback-style editing that forces retrieval of earlier visual states and name-based multi-round personalization that binds names to appearances across rounds; (ii) a history-conditioned training and inference framework with token-level caching to prevent multi-round identity drift; and (iii) enabling improvements for high-fidelity image reconstruction and editable personalization, including a reconstruction-based DiT detokenizer and a multi-stage fine-tuning curriculum. We demonstrate that explicitly training for non-Markov interactions yields substantial improvements in multi-round consistency and instruction compliance, while maintaining strong single-round editing and personalization. 

---
# Denoising and Baseline Correction of Low-Scan FTIR Spectra: A Benchmark of Deep Learning Models Against Traditional Signal Processing 

**Authors**: Azadeh Mokari, Shravan Raghunathan, Artem Shydliukh, Oleg Ryabchykov, Christoph Krafft, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2601.20905)  

**Abstract**: High-quality Fourier Transform Infrared (FTIR) imaging usually needs extensive signal averaging to reduce noise and drift which severely limits clinical speed. Deep learning can accelerate imaging by reconstructing spectra from rapid, single-scan inputs. However, separating noise and baseline drift simultaneously without ground truth is an ill-posed inverse problem. Standard black-box architectures often rely on statistical approximations that introduce spectral hallucinations or fail to generalize to unstable atmospheric conditions. To solve these issues we propose a physics-informed cascade Unet that separates denoising and baseline correction tasks using a new, deterministic Physics Bridge. This architecture forces the network to separate random noise from chemical signals using an embedded SNIP layer to enforce spectroscopic constraints instead of learning statistical approximations. We benchmarked this approach against a standard single Unet and a traditional Savitzky-Golay/SNIP workflow. We used a dataset of human hypopharyngeal carcinoma cells (FaDu). The cascade model outperformed all other methods, achieving a 51.3% reduction in RMSE compared to raw single-scan inputs, surpassing both the single Unet (40.2%) and the traditional workflow (33.7%). Peak-aware metrics show that the cascade architecture eliminates spectral hallucinations found in standard deep learning. It also preserves peak intensity with much higher fidelity than traditional smoothing. These results show that the cascade Unet is a robust solution for diagnostic-grade FTIR imaging. It enables imaging speeds 32 times faster than current methods. 

---
# ICON: Intent-Context Coupling for Efficient Multi-Turn Jailbreak Attack 

**Authors**: Xingwei Lin, Wenhao Lin, Sicong Cao, Jiahao Yu, Renke Huang, Lei Xue, Chunming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.20903)  

**Abstract**: Multi-turn jailbreak attacks have emerged as a critical threat to Large Language Models (LLMs), bypassing safety mechanisms by progressively constructing adversarial contexts from scratch and incrementally refining prompts. However, existing methods suffer from the inefficiency of incremental context construction that requires step-by-step LLM interaction, and often stagnate in suboptimal regions due to surface-level optimization. In this paper, we characterize the Intent-Context Coupling phenomenon, revealing that LLM safety constraints are significantly relaxed when a malicious intent is coupled with a semantically congruent context pattern. Driven by this insight, we propose ICON, an automated multi-turn jailbreak framework that efficiently constructs an authoritative-style context via prior-guided semantic routing. Specifically, ICON first routes the malicious intent to a congruent context pattern (e.g., Scientific Research) and instantiates it into an attack prompt sequence. This sequence progressively builds the authoritative-style context and ultimately elicits prohibited content. In addition, ICON incorporates a Hierarchical Optimization Strategy that combines local prompt refinement with global context switching, preventing the attack from stagnating in ineffective contexts. Experimental results across eight SOTA LLMs demonstrate the effectiveness of ICON, achieving a state-of-the-art average Attack Success Rate (ASR) of 97.1\%. Code is available at this https URL. 

---
# Finetune-Informed Pretraining Boosts Downstream Performance 

**Authors**: Atik Faysal, Mohammad Rostami, Reihaneh Gh. Roshan, Nikhil Muralidhar, Huaxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.20884)  

**Abstract**: Multimodal pretraining is effective for building general-purpose representations, but in many practical deployments, only one modality is heavily used during downstream fine-tuning. Standard pretraining strategies treat all modalities uniformly, which can lead to under-optimized representations for the modality that actually matters. We propose Finetune-Informed Pretraining (FIP), a model-agnostic method that biases representation learning toward a designated target modality needed at fine-tuning time. FIP combines higher masking difficulty, stronger loss weighting, and increased decoder capacity for the target modality, without modifying the shared encoder or requiring additional supervision. When applied to masked modeling on constellation diagrams for wireless signals, FIP consistently improves downstream fine-tuned performance with no extra data or compute. FIP is simple to implement, architecture-compatible, and broadly applicable across multimodal masked modeling pipelines. 

---
# DevOps-Gym: Benchmarking AI Agents in Software DevOps Cycle 

**Authors**: Yuheng Tang, Kaijie Zhu, Bonan Ruan, Chuqi Zhang, Michael Yang, Hongwei Li, Suyue Guo, Tianneng Shi, Zekun Li, Christopher Kruegel, Giovanni Vigna, Dawn Song, William Yang Wang, Lun Wang, Yangruibo Ding, Zhenkai Liang, Wenbo Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.20882)  

**Abstract**: Even though demonstrating extraordinary capabilities in code generation and software issue resolving, AI agents' capabilities in the full software DevOps cycle are still unknown. Different from pure code generation, handling the DevOps cycle in real-world software, including developing, deploying, and managing, requires analyzing large-scale projects, understanding dynamic program behaviors, leveraging domain-specific tools, and making sequential decisions. However, existing benchmarks focus on isolated problems and lack environments and tool interfaces for DevOps. We introduce DevOps-Gym, the first end-to-end benchmark for evaluating AI agents across core DevOps workflows: build and configuration, monitoring, issue resolving, and test generation. DevOps-Gym includes 700+ real-world tasks collected from 30+ projects in Java and Go. We develop a semi-automated data collection mechanism with rigorous and non-trivial expert efforts in ensuring the task coverage and quality. Our evaluation of state-of-the-art models and agents reveals fundamental limitations: they struggle with issue resolving and test generation in Java and Go, and remain unable to handle new tasks such as monitoring and build and configuration. These results highlight the need for essential research in automating the full DevOps cycle with AI agents. 

---
# STAER: Temporal Aligned Rehearsal for Continual Spiking Neural Network 

**Authors**: Matteo Gianferrari, Omayma Moussadek, Riccardo Salami, Cosimo Fiorini, Lorenzo Tartarini, Daniela Gandolfi, Simone Calderara  

**Link**: [PDF](https://arxiv.org/pdf/2601.20870)  

**Abstract**: Spiking Neural Networks (SNNs) are inherently suited for continuous learning due to their event-driven temporal dynamics; however, their application to Class-Incremental Learning (CIL) has been hindered by catastrophic forgetting and the temporal misalignment of spike patterns. In this work, we introduce Spiking Temporal Alignment with Experience Replay (STAER), a novel framework that explicitly preserves temporal structure to bridge the performance gap between SNNs and ANNs. Our approach integrates a differentiable Soft-DTW alignment loss to maintain spike timing fidelity and employs a temporal expansion and contraction mechanism on output logits to enforce robust representation learning. Implemented on a deep ResNet19 spiking backbone, STAER achieves state-of-the-art performance on Sequential-MNIST and Sequential-CIFAR10. Empirical results demonstrate that our method matches or outperforms strong ANN baselines (ER, DER++) while preserving biologically plausible dynamics. Ablation studies further confirm that explicit temporal alignment is critical for representational stability, positioning STAER as a scalable solution for spike-native lifelong learning. Code is available at this https URL. 

---
# Integrating Color Histogram Analysis and Convolutional Neural Network for Skin Lesion Classification 

**Authors**: M. A. Rasel, Sameem Abdul Kareem, Unaizah Obaidellah  

**Link**: [PDF](https://arxiv.org/pdf/2601.20869)  

**Abstract**: The color of skin lesions is an important diagnostic feature for identifying malignant melanoma and other skin diseases. Typical colors associated with melanocytic lesions include tan, brown, black, red, white, and blue gray. This study introduces a novel feature: the number of colors present in a lesion, which can indicate the severity of disease and help distinguish melanomas from benign lesions. We propose a color histogram analysis method to examine lesion pixel values from three publicly available datasets: PH2, ISIC2016, and Med Node. The PH2 dataset contains ground truth annotations of lesion colors, while ISIC2016 and Med Node do not; our algorithm estimates the ground truth using color histogram analysis based on PH2. We then design and train a 19 layer Convolutional Neural Network (CNN) with residual skip connections to classify lesions into three categories based on the number of colors present. DeepDream visualization is used to interpret features learned by the network, and multiple CNN configurations are tested. The best model achieves a weighted F1 score of 75 percent. LIME is applied to identify important regions influencing model decisions. The results show that the number of colors in a lesion is a significant feature for describing skin conditions, and the proposed CNN with three skip connections demonstrates strong potential for clinical diagnostic support. 

---
# Rethinking LLM-Driven Heuristic Design: Generating Efficient and Specialized Solvers via Dynamics-Aware Optimization 

**Authors**: Rongzheng Wang, Yihong Huang, Muquan Li, Jiakai Li, Di Liang, Bob Simons, Pei Ke, Shuang Liang, Ke Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.20868)  

**Abstract**: Large Language Models (LLMs) have advanced the field of Combinatorial Optimization through automated heuristic generation. Instead of relying on manual design, this LLM-Driven Heuristic Design (LHD) process leverages LLMs to iteratively generate and refine solvers to achieve high performance. However, existing LHD frameworks face two critical limitations: (1) Endpoint-only evaluation, which ranks solvers solely by final quality, ignoring the convergence process and runtime efficiency; (2) High adaptation costs, where distribution shifts necessitate re-adaptation to generate specialized solvers for new instance groups. To address these issues, we propose Dynamics-Aware Solver Heuristics (DASH), a framework that co-optimizes solver search mechanisms and runtime schedules guided by a convergence-aware metric, thereby identifying efficient and high-performance solvers. Furthermore, to mitigate expensive re-adaptation, DASH incorporates Profiled Library Retrieval (PLR). PLR efficiently archives specialized solvers concurrently with the evolutionary process, enabling cost-effective warm-starts for heterogeneous distributions. Experiments on four combinatorial optimization problems demonstrate that DASH improves runtime efficiency by over 3 times, while surpassing the solution quality of state-of-the-art baselines across diverse problem scales. Furthermore, by enabling profile-based warm starts, DASH maintains superior accuracy under different distributions while cutting LLM adaptation costs by over 90%. 

---
# Generalizable Prompt Tuning for Audio-Language Models via Semantic Expansion 

**Authors**: Jaehyuk Jang, Wonjun Lee, Kangwook Ko, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.20867)  

**Abstract**: Prompt tuning has achieved remarkable progress in vision-language models (VLMs) and is recently being adopted for audio-language models (ALMs). However, its generalization ability in ALMs remains largely underexplored. We observe that conventional prompt tuning for ALMs also suffers from the Base-New Tradeoff, and we identify that this issue stems from the disrupted semantic structure of the embedding space. To address this issue, we propose Semantically Expanded Prompt Tuning (SEPT)-a plug-and-play framework that explicitly regularizes the prompt embedding space by incorporating semantic neighbors generated by large language models. SEPT introduces a novel semantic expansion loss with margin constraints that promote intra-class compactness and inter-class separability, thereby enhancing the semantic structure of the prompt embedding space. For comprehensive evaluation, we establish the first benchmark setup for prompt generalization in ALMs, covering both base-to-new generalization and cross-dataset transferability. Extensive experiments demonstrate that SEPT consistently improves generalization performance across multiple prompt tuning baselines, while maintaining computational cost during inference. Codes are available in this https URL. 

---
# LSR-Net: A Lightweight and Strong Robustness Network for Bearing Fault Diagnosis in Noise Environment 

**Authors**: Junseok Lee, Jihye Shin, Sangyong Lee, Chang-Jae Chun  

**Link**: [PDF](https://arxiv.org/pdf/2601.10761)  

**Abstract**: Rotating bearings play an important role in modern industries, but have a high probability of occurrence of defects because they operate at high speed, high load, and poor operating environments. Therefore, if a delay time occurs when a bearing is diagnosed with a defect, this may cause economic loss and loss of life. Moreover, since the vibration sensor from which the signal is collected is highly affected by the operating environment and surrounding noise, accurate defect diagnosis in a noisy environment is also important. In this paper, we propose a lightweight and strong robustness network (LSR-Net) that is accurate in a noisy environment and enables real-time fault diagnosis. To this end, first, a denoising and feature enhancement module (DFEM) was designed to create a 3-channel 2D matrix by giving several nonlinearity to the feature-map that passed through the denoising module (DM) block composed of convolution-based denoising (CD) blocks. Moreover, adaptive pruning was applied to DM to improve denoising ability when the power of noise is strong. Second, for lightweight model design, a convolution-based efficiency shuffle (CES) block was designed using group convolution (GConv), group pointwise convolution (GPConv) and channel split that can design the model while maintaining low parameters. In addition, the trade-off between the accuracy and model computational complexity that can occur due to the lightweight design of the model was supplemented using attention mechanisms and channel shuffle. In order to verify the defect diagnosis performance of the proposed model, performance verification was conducted in a noisy environment using a vibration signal. As a result, it was confirmed that the proposed model had the best anti-noise ability compared to the benchmark models, and the computational complexity of the model was also the lowest. 

---
# Implementing AI Bill of Materials (AI BOM) with SPDX 3.0: A Comprehensive Guide to Creating AI and Dataset Bill of Materials 

**Authors**: Karen Bennet, Gopi Krishnan Rajbahadur, Arthit Suriyawongkul, Kate Stewart  

**Link**: [PDF](https://arxiv.org/pdf/2504.16743)  

**Abstract**: A Software Bill of Materials (SBOM) is becoming an increasingly important tool in regulatory and technical spaces to introduce more transparency and security into a project's software supply chain.
Artificial intelligence (AI) projects face unique challenges beyond the security of their software, and thus require a more expansive approach to a bill of materials. In this report, we introduce the concept of an AI-BOM, expanding on the SBOM to include the documentation of algorithms, data collection methods, frameworks and libraries, licensing information, and standard compliance. 

---
