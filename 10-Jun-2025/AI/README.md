# GUI-Reflection: Empowering Multimodal GUI Models with Self-Reflection Behavior 

**Authors**: Penghao Wu, Shengnan Ma, Bo Wang, Jiaheng Yu, Lewei Lu, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08012)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown great potential in revolutionizing Graphical User Interface (GUI) automation. However, existing GUI models mostly rely on learning from nearly error-free offline trajectories, thus lacking reflection and error recovery capabilities. To bridge this gap, we propose GUI-Reflection, a novel framework that explicitly integrates self-reflection and error correction capabilities into end-to-end multimodal GUI models throughout dedicated training stages: GUI-specific pre-training, offline supervised fine-tuning (SFT), and online reflection tuning. GUI-reflection enables self-reflection behavior emergence with fully automated data generation and learning processes without requiring any human annotation. Specifically, 1) we first propose scalable data pipelines to automatically construct reflection and error correction data from existing successful trajectories. While existing GUI models mainly focus on grounding and UI understanding ability, we propose the GUI-Reflection Task Suite to learn and evaluate reflection-oriented abilities explicitly. 2) Furthermore, we built a diverse and efficient environment for online training and data collection of GUI models on mobile devices. 3) We also present an iterative online reflection tuning algorithm leveraging the proposed environment, enabling the model to continuously enhance its reflection and error correction abilities. Our framework equips GUI agents with self-reflection and correction capabilities, paving the way for more robust, adaptable, and intelligent GUI automation, with all data, models, environments, and tools to be released publicly. 

---
# $τ^2$-Bench: Evaluating Conversational Agents in a Dual-Control Environment 

**Authors**: Victor Barres, Honghua Dong, Soham Ray, Xujie Si, Karthik Narasimhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07982)  

**Abstract**: Existing benchmarks for conversational AI agents simulate single-control environments, where only the AI agent can use tools to interact with the world, while the user remains a passive information provider. This differs from real-world scenarios like technical support, where users need to actively participate in modifying the state of the (shared) world. In order to address this gap, we introduce $\tau^2$-bench, with four key contributions:
1) A novel Telecom dual-control domain modeled as a Dec-POMDP, where both agent and user make use of tools to act in a shared, dynamic environment that tests both agent coordination and communication,
2) A compositional task generator that programmatically creates diverse, verifiable tasks from atomic components, ensuring domain coverage and controlled complexity,
3) A reliable user simulator tightly coupled with the environment, whose behavior is constrained by tools and observable states, improving simulation fidelity,
4) Fine-grained analysis of agent performance through multiple ablations including separating errors arising from reasoning vs communication/coordination.
In particular, our experiments show significant performance drops when agents shift from no-user to dual-control, highlighting the challenges of guiding users. Overall, $\tau^2$-bench provides a controlled testbed for agents that must both reason effectively and guide user actions. 

---
# Reinforcing Multimodal Understanding and Generation with Dual Self-rewards 

**Authors**: Jixiang Hong, Yiran Zhang, Guanzhong Wang, Yi Liu, Ji-Rong Wen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07963)  

**Abstract**: Building upon large language models (LLMs), recent large multimodal models (LMMs) unify cross-model understanding and generation into a single framework. However, LMMs still struggle to achieve accurate image-text alignment, prone to generating text responses contradicting the visual input or failing to follow the text-to-image prompts. Current solutions require external supervision (e.g., human feedback or reward models) and only address unidirectional tasks-either understanding or generation. In this work, based on the observation that understanding and generation are inverse dual tasks, we introduce a self-supervised dual reward mechanism to reinforce the understanding and generation capabilities of LMMs. Specifically, we sample multiple outputs for a given input in one task domain, then reverse the input-output pairs to compute the dual likelihood of the model as self-rewards for optimization. Extensive experimental results on visual understanding and generation benchmarks demonstrate that our method can effectively enhance the performance of the model without any external supervision, especially achieving remarkable improvements in text-to-image tasks. 

---
# Gradients: When Markets Meet Fine-tuning -- A Distributed Approach to Model Optimisation 

**Authors**: Christopher Subia-Waud  

**Link**: [PDF](https://arxiv.org/pdf/2506.07940)  

**Abstract**: Foundation model fine-tuning faces a fundamental challenge: existing AutoML platforms rely on single optimisation strategies that explore only a fraction of viable hyperparameter configurations. In this white paper, We introduce Gradients, a decentralised AutoML platform that transforms hyperparameter optimisation into a competitive marketplace where independent miners compete to discover optimal configurations. Economic incentives align individual exploration with collective optimisation goals, driving systematic investigation of hyperparameter regions that centralised methods miss. We evaluate our approach across 180 controlled experiments spanning diverse model architectures (70M to 70B parameters) and task types. Gradients achieves an 82.8\% win rate against HuggingFace AutoTrain and 100\% against TogetherAI, Databricks, and Google Cloud, with mean improvements of 11.8\% and 42.1\% respectively. Complex reasoning and retrieval tasks show particularly strong gains of 30-40\%, whilst diffusion models achieve 23.4\% improvements for person-specific generation. These results demonstrate that competitive, economically-driven approaches can systematically discover superior configurations that centralised AutoML consistently miss. 

---
# Solving Inequality Proofs with Large Language Models 

**Authors**: Jiayi Sheng, Luna Lyu, Jikai Jin, Tony Xia, Alex Gu, James Zou, Pan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07927)  

**Abstract**: Inequality proving, crucial across diverse scientific and mathematical fields, tests advanced reasoning skills such as discovering tight bounds and strategic theorem application. This makes it a distinct, demanding frontier for large language models (LLMs), offering insights beyond general mathematical problem-solving. Progress in this area is hampered by existing datasets that are often scarce, synthetic, or rigidly formal. We address this by proposing an informal yet verifiable task formulation, recasting inequality proving into two automatically checkable subtasks: bound estimation and relation prediction. Building on this, we release IneqMath, an expert-curated dataset of Olympiad-level inequalities, including a test set and training corpus enriched with step-wise solutions and theorem annotations. We also develop a novel LLM-as-judge evaluation framework, combining a final-answer judge with four step-wise judges designed to detect common reasoning flaws. A systematic evaluation of 29 leading LLMs on IneqMath reveals a surprising reality: even top models like o1 achieve less than 10% overall accuracy under step-wise scrutiny; this is a drop of up to 65.5% from their accuracy considering only final answer equivalence. This discrepancy exposes fragile deductive chains and a critical gap for current LLMs between merely finding an answer and constructing a rigorous proof. Scaling model size and increasing test-time computation yield limited gains in overall proof correctness. Instead, our findings highlight promising research directions such as theorem-guided reasoning and self-refinement. Code and data are available at this https URL. 

---
# LUCIFER: Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07915)  

**Abstract**: In dynamic environments, the rapid obsolescence of pre-existing environmental knowledge creates a gap between an agent's internal model and the evolving reality of its operational context. This disparity between prior and updated environmental valuations fundamentally limits the effectiveness of autonomous decision-making. To bridge this gap, the contextual bias of human domain stakeholders, who naturally accumulate insights through direct, real-time observation, becomes indispensable. However, translating their nuanced, and context-rich input into actionable intelligence for autonomous systems remains an open challenge. To address this, we propose LUCIFER (Language Understanding and Context-Infused Framework for Exploration and Behavior Refinement), a domain-agnostic framework that integrates a hierarchical decision-making architecture with reinforcement learning (RL) and large language models (LLMs) into a unified system. This architecture mirrors how humans decompose complex tasks, enabling a high-level planner to coordinate specialised sub-agents, each focused on distinct objectives and temporally interdependent actions. Unlike traditional applications where LLMs are limited to single role, LUCIFER integrates them in two synergistic roles: as context extractors, structuring verbal stakeholder input into domain-aware representations that influence decision-making through an attention space mechanism aligning LLM-derived insights with the agent's learning process, and as zero-shot exploration facilitators guiding the agent's action selection process during exploration. We benchmark various LLMs in both roles and demonstrate that LUCIFER improves exploration efficiency and decision quality, outperforming flat, goal-conditioned policies. Our findings show the potential of context-driven decision-making, where autonomous systems leverage human contextual knowledge for operational success. 

---
# Evaluating Large Language Models on the Frame and Symbol Grounding Problems: A Zero-shot Benchmark 

**Authors**: Shoko Oka  

**Link**: [PDF](https://arxiv.org/pdf/2506.07896)  

**Abstract**: Recent advancements in large language models (LLMs) have revitalized philosophical debates surrounding artificial intelligence. Two of the most fundamental challenges - namely, the Frame Problem and the Symbol Grounding Problem - have historically been viewed as unsolvable within traditional symbolic AI systems. This study investigates whether modern LLMs possess the cognitive capacities required to address these problems. To do so, I designed two benchmark tasks reflecting the philosophical core of each problem, administered them under zero-shot conditions to 13 prominent LLMs (both closed and open-source), and assessed the quality of the models' outputs across five trials each. Responses were scored along multiple criteria, including contextual reasoning, semantic coherence, and information filtering. The results demonstrate that while open-source models showed variability in performance due to differences in model size, quantization, and instruction tuning, several closed models consistently achieved high scores. These findings suggest that select modern LLMs may be acquiring capacities sufficient to produce meaningful and stable responses to these long-standing theoretical challenges. 

---
# A Temporal FRBR/FRBRoo-Based Model for Component-Level Versioning of Legal Norms 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07853)  

**Abstract**: Effectively representing legal norms for automated processing is a critical challenge, particularly in tracking the diachronic evolution of their hierarchical components (e.g., articles, paragraphs). While foundational frameworks like FRBR/FRBRoo and standards like Akoma Ntoso model legal documents at a macro level, they lack native mechanisms for granular, component-level versioning. This limitation hinders the deterministic point-in-time reconstruction of legal texts, a fundamental capability for reliable Legal Tech and AI applications. This paper proposes a structured, temporal model that extends the FRBRoo framework to address this gap. It introduces specialized subclasses of Expressio - Temporal Version (TV) and Language Version (LV - to represent the state of a legal norm and its linguistic variations at specific points in time. The model applies this same paradigm hierarchically, introducing Component Work (CW), Component Temporal Version (CTV), and Component Language Version (CLV) to track the lifecycle of individual articles, paragraphs, and clauses. Using the Brazilian Federal Constitution as a case study, the paper demonstrates how each amendment creates new Component Temporal Versions for affected provisions, while unaffected components retain their existing versions. This fine-grained, time-aware architecture enables the precise, deterministic retrieval and reconstruction of any part of a legal text as it existed on a specific date. The model provides a robust foundation for developing advanced legal information systems, knowledge graphs, and AI tools capable of accurate historical analysis and impact assessment, overcoming the limitations of current generative models. 

---
# HAIBU-ReMUD: Reasoning Multimodal Ultrasound Dataset and Model Bridging to General Specific Domains 

**Authors**: Shijie Wang, Yilun Zhang, Zeyu Lai, Dexing Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07837)  

**Abstract**: Multimodal large language models (MLLMs) have shown great potential in general domains but perform poorly in some specific domains due to a lack of domain-specific data, such as image-text data or vedio-text data. In some specific domains, there is abundant graphic and textual data scattered around, but lacks standardized arrangement. In the field of medical ultrasound, there are ultrasonic diagnostic books, ultrasonic clinical guidelines, ultrasonic diagnostic reports, and so on. However, these ultrasonic materials are often saved in the forms of PDF, images, etc., and cannot be directly used for the training of MLLMs. This paper proposes a novel image-text reasoning supervised fine-tuning data generation pipeline to create specific domain quadruplets (image, question, thinking trace, and answer) from domain-specific materials. A medical ultrasound domain dataset ReMUD is established, containing over 45,000 reasoning and non-reasoning supervised fine-tuning Question Answering (QA) and Visual Question Answering (VQA) data. The ReMUD-7B model, fine-tuned on Qwen2.5-VL-7B-Instruct, outperforms general-domain MLLMs in medical ultrasound field. To facilitate research, the ReMUD dataset, data generation codebase, and ReMUD-7B parameters will be released at this https URL, addressing the data shortage issue in specific domain MLLMs. 

---
# Addition in Four Movements: Mapping Layer-wise Information Trajectories in LLMs 

**Authors**: Yao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07824)  

**Abstract**: Multi-digit addition is a clear probe of the computational power of large language models. To dissect the internal arithmetic processes in LLaMA-3-8B-Instruct, we combine linear probing with logit-lens inspection. Inspired by the step-by-step manner in which humans perform addition, we propose and analyze a coherent four-stage trajectory in the forward pass:Formula-structure representations become linearly decodable first, while the answer token is still far down the candidate this http URL computational features then emerge this http URL deeper activation layers, numerical abstractions of the result become clearer, enabling near-perfect detection and decoding of the individual digits in the this http URL the output, the model organizes and generates the final content, with the correct token reliably occupying the top this http URL trajectory suggests a hierarchical process that favors internal computation over rote memorization. We release our code and data to facilitate reproducibility. 

---
# Guideline Forest: Experience-Induced Multi-Guideline Reasoning with Stepwise Aggregation 

**Authors**: Jiaxiang CHen, Zhuo Wang, Mingxi Zou, Qifan Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07820)  

**Abstract**: Human reasoning is flexible, adaptive, and grounded in prior experience-qualities that large language models (LLMs) still struggle to emulate. Existing methods either explore diverse reasoning paths at inference time or search for optimal workflows through expensive operations, but both fall short in leveraging multiple reusable strategies in a structured, efficient manner. We propose Guideline Forest, a framework that enhances LLMs reasoning by inducing structured reasoning strategies-called guidelines-from verified examples and executing them via step-wise aggregation. Unlike test-time search or single-path distillation, our method draws on verified reasoning experiences by inducing reusable guidelines and expanding each into diverse variants. Much like human reasoning, these variants reflect alternative thought patterns, are executed in parallel, refined via self-correction, and aggregated step by step-enabling the model to adaptively resolve uncertainty and synthesize robust this http URL evaluate Guideline Forest on four benchmarks-GSM8K, MATH-500, MBPP, and HumanEval-spanning mathematical and programmatic reasoning. Guideline Forest consistently outperforms strong baselines, including CoT, ReAct, ToT, FoT, and AFlow. Ablation studies further highlight the effectiveness of multi-path reasoning and stepwise aggregation, underscoring the Guideline Forest's adaptability and generalization potential. 

---
# A Proposal to Extend the Common Model of Cognition with Metacognition 

**Authors**: John Laird, Christian Lebiere, Paul Rosenbloom, Andrea Stocco, Robert Wray  

**Link**: [PDF](https://arxiv.org/pdf/2506.07807)  

**Abstract**: The Common Model of Cognition (CMC) provides an abstract characterization of the structure and processing required by a cognitive architecture for human-like minds. We propose a unified approach to integrating metacognition within the CMC. We propose that metacognition involves reasoning over explicit representations of an agent's cognitive capabilities and processes in working memory. Our proposal exploits the existing cognitive capabilities of the CMC, making minimal extensions in the structure and information available within working memory. We provide examples of metacognition within our proposal. 

---
# REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models 

**Authors**: Diego Forniés-Tabuenca, Alejandro Uribe, Urtzi Otamendi, Arkaitz Artetxe, Juan Carlos Rivera, Oier Lopez de Lacalle  

**Link**: [PDF](https://arxiv.org/pdf/2506.07759)  

**Abstract**: Multi-objective optimization is fundamental in complex decision-making tasks. Traditional algorithms, while effective, often demand extensive problem-specific modeling and struggle to adapt to nonlinear structures. Recent advances in Large Language Models (LLMs) offer enhanced explainability, adaptability, and reasoning. This work proposes Reflective Evolution of Multi-objective Heuristics (REMoH), a novel framework integrating NSGA-II with LLM-based heuristic generation. A key innovation is a reflection mechanism that uses clustering and search-space reflection to guide the creation of diverse, high-quality heuristics, improving convergence and maintaining solution diversity. The approach is evaluated on the Flexible Job Shop Scheduling Problem (FJSSP) in-depth benchmarking against state-of-the-art methods using three instance datasets: Dauzere, Barnes, and Brandimarte. Results demonstrate that REMoH achieves competitive results compared to state-of-the-art approaches with reduced modeling effort and enhanced adaptability. These findings underscore the potential of LLMs to augment traditional optimization, offering greater flexibility, interpretability, and robustness in multi-objective scenarios. 

---
# Agent Semantics, Semantic Spacetime, and Graphical Reasoning 

**Authors**: Mark Burgess  

**Link**: [PDF](https://arxiv.org/pdf/2506.07756)  

**Abstract**: Some formal aspects of the Semantic Spacetime graph model are presented, with reference to its use for directed knowledge representations and process modelling. A finite $\gamma(3,4)$ representation is defined to form a closed set of operations that can scale to any degree of semantic complexity. The Semantic Spacetime postulates bring predictability with minimal constraints to pathways in graphs. The ubiquitous appearance of absorbing states in any partial graph means that a graph process leaks information. The issue is closely associated with the issue of division by zero, which signals a loss of closure and the need for manual injection of remedial information. The Semantic Spacetime model (and its Promise Theory) origins help to clarify how such absorbing states are associated with boundary information where intentionality can enter. 

---
# RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards 

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07736)  

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements. 

---
# NeurIPS 2025 E2LM Competition : Early Training Evaluation of Language Models 

**Authors**: Mouadh Yagoubi, Yasser Dahou, Billel Mokeddem, Younes Belkada, Phuc H. Le-Khac, Basma El Amel Boussaha, Reda Alami, Jingwei Zuo, Damiano Marsili, Mugariya Farooq, Mounia Lalmas, Georgia Gkioxari, Patrick Gallinari, Philip Torr, Hakim Hacid  

**Link**: [PDF](https://arxiv.org/pdf/2506.07731)  

**Abstract**: Existing benchmarks have proven effective for assessing the performance of fully trained large language models. However, we find striking differences in the early training stages of small models, where benchmarks often fail to provide meaningful or discriminative signals. To explore how these differences arise, this competition tackles the challenge of designing scientific knowledge evaluation tasks specifically tailored for measuring early training progress of language models. Participants are invited to develop novel evaluation methodologies or adapt existing benchmarks to better capture performance differences among language models. To support this effort, we provide three pre-trained small models (0.5B, 1B, and 3B parameters), along with intermediate checkpoints sampled during training up to 200B tokens. All experiments and development work can be run on widely available free cloud-based GPU platforms, making participation accessible to researchers with limited computational resources. Submissions will be evaluated based on three criteria: the quality of the performance signal they produce, the consistency of model rankings at 1 trillion tokens of training, and their relevance to the scientific knowledge domain. By promoting the design of tailored evaluation strategies for early training, this competition aims to attract a broad range of participants from various disciplines, including those who may not be machine learning experts or have access to dedicated GPU resources. Ultimately, this initiative seeks to make foundational LLM research more systematic and benchmark-informed from the earliest phases of model development. 

---
# MCPWorld: A Unified Benchmarking Testbed for API, GUI, and Hybrid Computer Use Agents 

**Authors**: Yunhe Yan, Shihe Wang, Jiajun Du, Yexuan Yang, Yuxuan Shan, Qichen Qiu, Xianqing Jia, Xinge Wang, Xin Yuan, Xu Han, Mao Qin, Yinxiao Chen, Chen Peng, Shangguang Wang, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07672)  

**Abstract**: (M)LLM-powered computer use agents (CUA) are emerging as a transformative technique to automate human-computer interaction. However, existing CUA benchmarks predominantly target GUI agents, whose evaluation methods are susceptible to UI changes and ignore function interactions exposed by application APIs, e.g., Model Context Protocol (MCP). To this end, we propose MCPWorld, the first automatic CUA testbed for API, GUI, and API-GUI hybrid agents. A key principle of MCPWorld is the use of "white-box apps", i.e., those with source code availability and can be revised/re-compiled as needed (e.g., adding MCP support), with two notable advantages:
(1) It greatly broadens the design space of CUA, such as what and how the app features to be exposed/extracted as CUA-callable APIs.
(2) It allows MCPWorld to programmatically verify task completion by directly monitoring application behavior through techniques like dynamic code instrumentation, offering robust, accurate CUA evaluation decoupled from specific agent implementations or UI states.
Currently, MCPWorld includes 201 well curated and annotated user tasks, covering diversified use cases and difficulty levels. MCPWorld is also fully containerized with GPU acceleration support for flexible adoption on different OS/hardware environments. Our preliminary experiments, using a representative LLM-powered CUA framework, achieve 75.12% task completion accuracy, simultaneously providing initial evidence on the practical effectiveness of agent automation leveraging MCP. Overall, we anticipate MCPWorld to facilitate and standardize the benchmarking of next-generation computer use agents that can leverage rich external tools. Our code and dataset are publicly available at this https URL. 

---
# SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling 

**Authors**: Haoran Wang, Zhenyu Hou, Yao Wei, Jie Tang, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07636)  

**Abstract**: Large language models (LLMs) have advanced rapidly from conversational problem solving to addressing real-world tasks involving tool use, such as software engineering (SWE). Recent LLM-powered toolkits, such as OpenAI Codex and Cursor, have offered end-to-end automation of the software development process. However, building effective SWE agents remains challenging due to the lack of high-quality training data and effective test cases. To address this issue, we present SWE-Dev, an SWE agent built upon open-source LLMs. First, we develop a robust pipeline to synthesize test cases for patch evaluation. Second, we scale up agent trajectories to construct the training data for building SWE-Dev. Experiments on the SWE-bench-Verified benchmark show that the SWE-Dev models can achieve top performance among all open SWE agents. Specifically, the success rates of the SWE-Dev 7B and 32B parameter models reach 23.4% and 36.6%, respectively, outperforming state-of-the-art open-source models. All code, models, and datasets are publicly available at this https URL. 

---
# Automating Exploratory Multiomics Research via Language Models 

**Authors**: Shang Qu, Ning Ding, Linhai Xie, Yifei Li, Zaoqu Liu, Kaiyan Zhang, Yibai Xiong, Yuxin Zuo, Zhangren Chen, Ermo Hua, Xingtai Lv, Youbang Sun, Yang Li, Dong Li, Fuchu He, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07591)  

**Abstract**: This paper introduces PROTEUS, a fully automated system that produces data-driven hypotheses from raw data files. We apply PROTEUS to clinical proteogenomics, a field where effective downstream data analysis and hypothesis proposal is crucial for producing novel discoveries. PROTEUS uses separate modules to simulate different stages of the scientific process, from open-ended data exploration to specific statistical analysis and hypothesis proposal. It formulates research directions, tools, and results in terms of relationships between biological entities, using unified graph structures to manage complex research processes. We applied PROTEUS to 10 clinical multiomics datasets from published research, arriving at 360 total hypotheses. Results were evaluated through external data validation and automatic open-ended scoring. Through exploratory and iterative research, the system can navigate high-throughput and heterogeneous multiomics data to arrive at hypotheses that balance reliability and novelty. In addition to accelerating multiomic analysis, PROTEUS represents a path towards tailoring general autonomous systems to specialized scientific domains to achieve open-ended hypothesis generation from data. 

---
# SAFEFLOW: A Principled Protocol for Trustworthy and Transactional Autonomous Agent Systems 

**Authors**: Peiran Li, Xinkai Zou, Zhuohang Wu, Ruifeng Li, Shuo Xing, Hanwen Zheng, Zhikai Hu, Yuping Wang, Haoxi Li, Qin Yuan, Yingmo Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07564)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (VLMs) have enabled powerful autonomous agents capable of complex reasoning and multi-modal tool use. Despite their growing capabilities, today's agent frameworks remain fragile, lacking principled mechanisms for secure information flow, reliability, and multi-agent coordination. In this work, we introduce SAFEFLOW, a new protocol-level framework for building trustworthy LLM/VLM-based agents. SAFEFLOW enforces fine-grained information flow control (IFC), precisely tracking provenance, integrity, and confidentiality of all the data exchanged between agents, tools, users, and environments. By constraining LLM reasoning to respect these security labels, SAFEFLOW prevents untrusted or adversarial inputs from contaminating high-integrity decisions. To ensure robustness in concurrent multi-agent settings, SAFEFLOW introduces transactional execution, conflict resolution, and secure scheduling over shared state, preserving global consistency across agents. We further introduce mechanisms, including write-ahead logging, rollback, and secure caches, that further enhance resilience against runtime errors and policy violations. To validate the performances, we built SAFEFLOWBENCH, a comprehensive benchmark suite designed to evaluate agent reliability under adversarial, noisy, and concurrent operational conditions. Extensive experiments demonstrate that agents built with SAFEFLOW maintain impressive task performance and security guarantees even in hostile environments, substantially outperforming state-of-the-art. Together, SAFEFLOW and SAFEFLOWBENCH lay the groundwork for principled, robust, and secure agent ecosystems, advancing the frontier of reliable autonomy. 

---
# GTR-CoT: Graph Traversal as Visual Chain of Thought for Molecular Structure Recognition 

**Authors**: Jingchao Wang, Haote Yang, Jiang Wu, Yifan He, Xingjian Wei, Yinfan Wang, Chengjin Liu, Lingli Ge, Lijun Wu, Bin Wang, Dahua Lin, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2506.07553)  

**Abstract**: Optical Chemical Structure Recognition (OCSR) is crucial for digitizing chemical knowledge by converting molecular images into machine-readable formats. While recent vision-language models (VLMs) have shown potential in this task, their image-captioning approach often struggles with complex molecular structures and inconsistent annotations. To overcome these challenges, we introduce GTR-Mol-VLM, a novel framework featuring two key innovations: (1) the \textit{Graph Traversal as Visual Chain of Thought} mechanism that emulates human reasoning by incrementally parsing molecular graphs through sequential atom-bond predictions, and (2) the data-centric principle of \textit{Faithfully Recognize What You've Seen}, which addresses the mismatch between abbreviated structures in images and their expanded annotations. To support model development, we constructed GTR-CoT-1.3M, a large-scale instruction-tuning dataset with meticulously corrected annotations, and introduced MolRec-Bench, the first benchmark designed for a fine-grained evaluation of graph-parsing accuracy in OCSR. Comprehensive experiments demonstrate that GTR-Mol-VLM achieves superior results compared to specialist models, chemistry-domain VLMs, and commercial general-purpose VLMs. Notably, in scenarios involving molecular images with functional group abbreviations, GTR-Mol-VLM outperforms the second-best baseline by approximately 14 percentage points, both in SMILES-based and graph-based metrics. We hope that this work will drive OCSR technology to more effectively meet real-world needs, thereby advancing the fields of cheminformatics and AI for Science. We will release GTR-CoT at this https URL. 

---
# Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning 

**Authors**: Weiqiang Jin, Hongyang Du, Guizhong Liu, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07548)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved strong performance in cooperative adversarial tasks. However, most existing methods typically train agents against fixed opponent strategies and rely on such meta-static difficulty conditions, which limits their adaptability to changing environments and often leads to suboptimal policies. Inspired by the success of curriculum learning (CL) in supervised tasks, we propose a dynamic CL framework for MARL that employs an self-adaptive difficulty adjustment mechanism. This mechanism continuously modulates opponent strength based on real-time agent training performance, allowing agents to progressively learn from easier to more challenging scenarios. However, the dynamic nature of CL introduces instability due to nonstationary environments and sparse global rewards. To address this challenge, we develop a Counterfactual Group Relative Policy Advantage (CGRPA), which is tightly coupled with the curriculum by providing intrinsic credit signals that reflect each agent's impact under evolving task demands. CGRPA constructs a counterfactual advantage function that isolates individual contributions within group behavior, facilitating more reliable policy updates throughout the curriculum. CGRPA evaluates each agent's contribution through constructing counterfactual action advantage function, providing intrinsic rewards that enhance credit assignment and stabilize learning under non-stationary conditions. Extensive experiments demonstrate that our method improves both training stability and final performance, achieving competitive results against state-of-the-art methods. The code is available at this https URL. 

---
# Coordinating Search-Informed Reasoning and Reasoning-Guided Search in Claim Verification 

**Authors**: Qisheng Hu, Quanyu Long, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07528)  

**Abstract**: Multi-hop claim verification is inherently challenging, requiring multi-step reasoning to construct verification chains while iteratively searching for information to uncover hidden bridging facts. This process is fundamentally interleaved, as effective reasoning relies on dynamically retrieved evidence, while effective search demands reasoning to refine queries based on partial information. To achieve this, we propose Hierarchical Agent Reasoning and Information Search (HARIS), explicitly modeling the coordinated process of reasoning-driven searching and search-informed reasoning. HARIS consists of a high-level reasoning agent that focuses on constructing the main verification chain, generating factual questions when more information is needed, and a low-level search agent that iteratively retrieves more information, refining its search based on intermediate findings. This design allows each agent to specialize in its respective task, enhancing verification accuracy and interpretability. HARIS is trained using reinforcement learning with outcome-based rewards. Experimental results on the EX-FEVER and HOVER benchmarks demonstrate that HARIS achieves strong performance, greatly advancing multi-hop claim verification. 

---
# Learning What Reinforcement Learning Can't: Interleaved Online Fine-Tuning for Hardest Questions 

**Authors**: Lu Ma, Hao Liang, Meiyi Qiang, Lexiang Tang, Xiaochen Ma, Zhen Hao Wong, Junbo Niu, Chengyu Shen, Runming He, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07527)  

**Abstract**: Recent advances in large language model (LLM) reasoning have shown that sophisticated behaviors such as planning and self-reflection can emerge through reinforcement learning (RL). However, despite these successes, RL in its current form remains insufficient to induce capabilities that exceed the limitations of the base model, as it is primarily optimized based on existing knowledge of the model rather than facilitating the acquisition of new information. To address this limitation, we employ supervised fine-tuning (SFT) to learn what RL cannot, which enables the incorporation of new knowledge and reasoning patterns by leveraging high-quality demonstration data. We analyze the training dynamics of RL and SFT for LLM reasoning and find that RL excels at maintaining and improving performance on questions within the model's original capabilities, while SFT is more effective at enabling progress on questions beyond the current scope of the model. Motivated by the complementary strengths of RL and SFT, we introduce a novel training approach, \textbf{ReLIFT} (\textbf{Re}inforcement \textbf{L}earning \textbf{I}nterleaved with Online \textbf{F}ine-\textbf{T}uning). In ReLIFT, the model is primarily trained using RL, but when it encounters challenging questions, high-quality solutions are collected for fine-tuning, and the training process alternates between RL and fine-tuning to enhance the model's reasoning abilities. ReLIFT achieves an average improvement of over +5.2 points across five competition-level benchmarks and one out-of-distribution benchmark compared to other zero-RL models. Furthermore, we demonstrate that ReLIFT outperforms both RL and SFT while using only 13\% of the detailed demonstration data, highlighting its scalability. These results provide compelling evidence that ReLIFT overcomes the fundamental limitations of RL and underscores the significant potential. 

---
# Efficient Generation of Diverse Cooperative Agents with World Models 

**Authors**: Yi Loo, Akshunn Trivedi, Malika Meghjani  

**Link**: [PDF](https://arxiv.org/pdf/2506.07450)  

**Abstract**: A major bottleneck in the training process for Zero-Shot Coordination (ZSC) agents is the generation of partner agents that are diverse in collaborative conventions. Current Cross-play Minimization (XPM) methods for population generation can be very computationally expensive and sample inefficient as the training objective requires sampling multiple types of trajectories. Each partner agent in the population is also trained from scratch, despite all of the partners in the population learning policies of the same coordination task. In this work, we propose that simulated trajectories from the dynamics model of an environment can drastically speed up the training process for XPM methods. We introduce XPM-WM, a framework for generating simulated trajectories for XPM via a learned World Model (WM). We show XPM with simulated trajectories removes the need to sample multiple trajectories. In addition, we show our proposed method can effectively generate partners with diverse conventions that match the performance of previous methods in terms of SP population training reward as well as training partners for ZSC agents. Our method is thus, significantly more sample efficient and scalable to a larger number of partners. 

---
# Fact in Fragments: Deconstructing Complex Claims via LLM-based Atomic Fact Extraction and Verification 

**Authors**: Liwen Zheng, Chaozhuo Li, Zheng Liu, Feiran Huang, Haoran Jia, Zaisheng Ye, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07446)  

**Abstract**: Fact verification plays a vital role in combating misinformation by assessing the veracity of claims through evidence retrieval and reasoning. However, traditional methods struggle with complex claims requiring multi-hop reasoning over fragmented evidence, as they often rely on static decomposition strategies and surface-level semantic retrieval, which fail to capture the nuanced structure and intent of the claim. This results in accumulated reasoning errors, noisy evidence contamination, and limited adaptability to diverse claims, ultimately undermining verification accuracy in complex scenarios. To address this, we propose Atomic Fact Extraction and Verification (AFEV), a novel framework that iteratively decomposes complex claims into atomic facts, enabling fine-grained retrieval and adaptive reasoning. AFEV dynamically refines claim understanding and reduces error propagation through iterative fact extraction, reranks evidence to filter noise, and leverages context-specific demonstrations to guide the reasoning process. Extensive experiments on five benchmark datasets demonstrate that AFEV achieves state-of-the-art performance in both accuracy and interpretability. 

---
# LegalReasoner: Step-wised Verification-Correction for Legal Judgment Reasoning 

**Authors**: Weijie Shi, Han Zhu, Jiaming Ji, Mengze Li, Jipeng Zhang, Ruiyuan Zhang, Jia Zhu, Jiajie Xu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07443)  

**Abstract**: Legal judgment prediction (LJP) aims to function as a judge by making final rulings based on case claims and facts, which plays a vital role in the judicial domain for supporting court decision-making and improving judicial efficiency. However, existing methods often struggle with logical errors when conducting complex legal reasoning. We propose LegalReasoner, which enhances LJP reliability through step-wise verification and correction of the reasoning process. Specifically, it first identifies dispute points to decompose complex cases, and then conducts step-wise reasoning while employing a process verifier to validate each step's logic from correctness, progressiveness, and potential perspectives. When errors are detected, expert-designed attribution and resolution strategies are applied for correction. To fine-tune LegalReasoner, we release the LegalHK dataset, containing 58,130 Hong Kong court cases with detailed annotations of dispute points, step-by-step reasoning chains, and process verification labels. Experiments demonstrate that LegalReasoner significantly improves concordance with court decisions from 72.37 to 80.27 on LLAMA-3.1-70B. The data is available at this https URL. 

---
# HeTa: Relation-wise Heterogeneous Graph Foundation Attack Model 

**Authors**: Yuling Wang, Zihui Chen, Pengfei Jiao, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07428)  

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are vulnerable, highlighting the need for tailored attacks to assess their robustness and ensure security. However, existing HGNN attacks often require complex retraining of parameters to generate specific perturbations for new scenarios. Recently, foundation models have opened new horizons for the generalization of graph neural networks by capturing shared semantics across various graph distributions. This leads us to ask:Can we design a foundation attack model for HGNNs that enables generalizable perturbations across different HGNNs, and quickly adapts to new heterogeneous graphs (HGs)? Empirical findings reveal that, despite significant differences in model design and parameter space, different HGNNs surprisingly share common vulnerability patterns from a relation-aware perspective. Therefore, we explore how to design foundation HGNN attack criteria by mining shared attack units. In this paper, we propose a novel relation-wise heterogeneous graph foundation attack model, HeTa. We introduce a foundation surrogate model to align heterogeneity and identify the importance of shared relation-aware attack units. Building on this, we implement a serialized relation-by-relation attack based on the identified relational weights. In this way, the perturbation can be transferred to various target HGNNs and easily fine-tuned for new HGs. Extensive experiments exhibit powerful attack performances and generalizability of our method. 

---
# Evaluating Visual Mathematics in Multimodal LLMs: A Multilingual Benchmark Based on the Kangaroo Tests 

**Authors**: Arnau Igualde Sáez, Lamyae Rhomrasi, Yusef Ahsini, Ricardo Vinuesa, Sergio Hoyas, Jose P. García Sabater, Marius J. Fullana i Alfonso, J. Alberto Conejero  

**Link**: [PDF](https://arxiv.org/pdf/2506.07418)  

**Abstract**: Multimodal Large Language Models (MLLMs) promise advanced vision language capabilities, yet their effectiveness in visually presented mathematics remains underexplored. This paper analyzes the development and evaluation of MLLMs for mathematical problem solving, focusing on diagrams, multilingual text, and symbolic notation. We then assess several models, including GPT 4o, Pixtral, Qwen VL, Llama 3.2 Vision variants, and Gemini 2.0 Flash in a multilingual Kangaroo style benchmark spanning English, French, Spanish, and Catalan. Our experiments reveal four key findings. First, overall precision remains moderate across geometry, visual algebra, logic, patterns, and combinatorics: no single model excels in every topic. Second, while most models see improved accuracy with questions that do not have images, the gain is often limited; performance for some remains nearly unchanged without visual input, indicating underutilization of diagrammatic information. Third, substantial variation exists across languages and difficulty levels: models frequently handle easier items but struggle with advanced geometry and combinatorial reasoning. Notably, Gemini 2.0 Flash achieves the highest precision on image based tasks, followed by Qwen VL 2.5 72B and GPT 4o, though none approach human level performance. Fourth, a complementary analysis aimed at distinguishing whether models reason or simply recite reveals that Gemini and GPT 4o stand out for their structured reasoning and consistent accuracy. In contrast, Pixtral and Llama exhibit less consistent reasoning, often defaulting to heuristics or randomness when unable to align their outputs with the given answer options. 

---
# An Intelligent Fault Self-Healing Mechanism for Cloud AI Systems via Integration of Large Language Models and Deep Reinforcement Learning 

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07411)  

**Abstract**: As the scale and complexity of cloud-based AI systems continue to increase, the detection and adaptive recovery of system faults have become the core challenges to ensure service reliability and continuity. In this paper, we propose an Intelligent Fault Self-Healing Mechanism (IFSHM) that integrates Large Language Model (LLM) and Deep Reinforcement Learning (DRL), aiming to realize a fault recovery framework with semantic understanding and policy optimization capabilities in cloud AI systems. On the basis of the traditional DRL-based control model, the proposed method constructs a two-stage hybrid architecture: (1) an LLM-driven fault semantic interpretation module, which can dynamically extract deep contextual semantics from multi-source logs and system indicators to accurately identify potential fault modes; (2) DRL recovery strategy optimizer, based on reinforcement learning, learns the dynamic matching of fault types and response behaviors in the cloud environment. The innovation of this method lies in the introduction of LLM for environment modeling and action space abstraction, which greatly improves the exploration efficiency and generalization ability of reinforcement learning. At the same time, a memory-guided meta-controller is introduced, combined with reinforcement learning playback and LLM prompt fine-tuning strategy, to achieve continuous adaptation to new failure modes and avoid catastrophic forgetting. Experimental results on the cloud fault injection platform show that compared with the existing DRL and rule methods, the IFSHM framework shortens the system recovery time by 37% with unknown fault scenarios. 

---
# Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data 

**Authors**: Xin-Cheng Wen, Yijun Yang, Cuiyun Gao, Yang Xiao, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.07390)  

**Abstract**: Large language models (LLMs) demonstrate considerable proficiency in numerous coding-related tasks; however, their capabilities in detecting software vulnerabilities remain limited. This limitation primarily stems from two factors: (1) the absence of reasoning data related to vulnerabilities, which hinders the models' ability to capture underlying vulnerability patterns; and (2) their focus on learning semantic representations rather than the reason behind them, thus failing to recognize semantically similar vulnerability samples. Furthermore, the development of LLMs specialized in vulnerability detection is challenging, particularly in environments characterized by the scarcity of high-quality datasets. In this paper, we propose a novel framework ReVD that excels at mining vulnerability patterns through reasoning data synthesizing and vulnerability-specific preference optimization. Specifically, we construct forward and backward reasoning processes for vulnerability and corresponding fixed code, ensuring the synthesis of high-quality reasoning data. Moreover, we design the triplet supervised fine-tuning followed by curriculum online preference optimization for enabling ReVD to better understand vulnerability patterns. The extensive experiments conducted on PrimeVul and SVEN datasets demonstrate that ReVD sets new state-of-the-art for LLM-based software vulnerability detection, e.g., 12.24\%-22.77\% improvement in the accuracy. The source code and data are available at this https URL. 

---
# Subgoal-Guided Policy Heuristic Search with Learned Subgoals 

**Authors**: Jake Tuero, Michael Buro, Levi H. S. Lelis  

**Link**: [PDF](https://arxiv.org/pdf/2506.07255)  

**Abstract**: Policy tree search is a family of tree search algorithms that use a policy to guide the search. These algorithms provide guarantees on the number of expansions required to solve a given problem that are based on the quality of the policy. While these algorithms have shown promising results, the process in which they are trained requires complete solution trajectories to train the policy. Search trajectories are obtained during a trial-and-error search process. When the training problem instances are hard, learning can be prohibitively costly, especially when starting from a randomly initialized policy. As a result, search samples are wasted in failed attempts to solve these hard instances. This paper introduces a novel method for learning subgoal-based policies for policy tree search algorithms. The subgoals and policies conditioned on subgoals are learned from the trees that the search expands while attempting to solve problems, including the search trees of failed attempts. We empirically show that our policy formulation and training method improve the sample efficiency of learning a policy and heuristic function in this online setting. 

---
# LLM-Enhanced Rapid-Reflex Async-Reflect Embodied Agent for Real-Time Decision-Making in Dynamically Changing Environments 

**Authors**: Yangqing Zheng, Shunqi Mao, Dingxin Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07223)  

**Abstract**: In the realm of embodied intelligence, the evolution of large language models (LLMs) has markedly enhanced agent decision making. Consequently, researchers have begun exploring agent performance in dynamically changing high-risk scenarios, i.e., fire, flood, and wind scenarios in the HAZARD benchmark. Under these extreme conditions, the delay in decision making emerges as a crucial yet insufficiently studied issue. We propose a Time Conversion Mechanism (TCM) that translates inference delays in decision-making into equivalent simulation frames, thus aligning cognitive and physical costs under a single FPS-based metric. By extending HAZARD with Respond Latency (RL) and Latency-to-Action Ratio (LAR), we deliver a fully latency-aware evaluation protocol. Moreover, we present the Rapid-Reflex Async-Reflect Agent (RRARA), which couples a lightweight LLM-guided feedback module with a rule-based agent to enable immediate reactive behaviors and asynchronous reflective refinements in situ. Experiments on HAZARD show that RRARA substantially outperforms existing baselines in latency-sensitive scenarios. 

---
# BIMgent: Towards Autonomous Building Modeling via Computer-use Agents 

**Authors**: Zihan Deng, Changyu Du, Stavros Nousias, André Borrmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.07217)  

**Abstract**: Existing computer-use agents primarily focus on general-purpose desktop automation tasks, with limited exploration of their application in highly specialized domains. In particular, the 3D building modeling process in the Architecture, Engineering, and Construction (AEC) sector involves open-ended design tasks and complex interaction patterns within Building Information Modeling (BIM) authoring software, which has yet to be thoroughly addressed by current studies. In this paper, we propose BIMgent, an agentic framework powered by multimodal large language models (LLMs), designed to enable autonomous building model authoring via graphical user interface (GUI) operations. BIMgent automates the architectural building modeling process, including multimodal input for conceptual design, planning of software-specific workflows, and efficient execution of the authoring GUI actions. We evaluate BIMgent on real-world building modeling tasks, including both text-based conceptual design generation and reconstruction from existing building design. The design quality achieved by BIMgent was found to be reasonable. Its operations achieved a 32% success rate, whereas all baseline models failed to complete the tasks (0% success rate). Results demonstrate that BIMgent effectively reduces manual workload while preserving design intent, highlighting its potential for practical deployment in real-world architectural modeling scenarios. 

---
# Reasoning Multimodal Large Language Model: Data Contamination and Dynamic Evaluation 

**Authors**: Ming Liu, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07202)  

**Abstract**: Multimodal Large Language Models (MLLMs) show impressive vision-language benchmark performance, yet growing concerns about data contamination (test set exposure during training) risk masking true generalization. This concern extends to reasoning MLLMs, often fine-tuned via reinforcement learning from potentially contaminated base models. We propose a novel dynamic evaluation framework to rigorously assess MLLM generalization, moving beyond static benchmarks. Instead of perturbing inputs, we perturb the task itself. Using the same visual input, models are evaluated across a family of tasks (e.g., QA, captioning, question posing, verification) to probe diverse capabilities. This task perturbation reveals whether model performance is robust or reliant on superficial task-specific cues. Our approach is analogous to loss landscape sharpness: models overfit or contaminated for a single task (sharp minima) falter under task shifts, unlike models with generalizable solutions (flatter minima). We developed an automated pipeline with a calibrated judge scoring open-ended generations (captions, questions) using paraphrase and corruption sampling. Applying this framework to leading image/video MLLMs on benchmarks including MME, RealWorldQA, and CVRR-ES, we analyze each model's cross-task "ability vector." We demonstrate that fine-tuning on simulated test data (extreme contamination) drastically sharpens task-specific performance but harms overall generalization. Our dynamic task perturbation offers deeper insights into MLLM generalization, distinguishing genuine understanding from spurious leakage or overfitting. 

---
# Exploring Effective Strategies for Building a Customised GPT Agent for Coding Classroom Dialogues 

**Authors**: Luwei Bai, Dongkeun Han, Sara Hennessy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07194)  

**Abstract**: This study investigates effective strategies for developing a customised GPT agent to code classroom dialogue. While classroom dialogue is widely recognised as a crucial element of education, its analysis remains challenging due to the need for a nuanced understanding of dialogic functions and the labour-intensive nature of manual transcript coding. Recent advancements in large language models offer promising avenues for automating this process. However, existing studies predominantly focus on training large-scale models or evaluating pre-trained models with fixed codebooks, which are often not applicable or replicable for dialogue researchers working with small datasets or customised coding schemes. Using GPT-4's MyGPT agent as a case, this study evaluates its baseline performance in coding classroom dialogue with a human codebook and examines how performance varies with different example inputs through a variable control method. Through a design-based research approach, it identifies a set of practical strategies, based on MyGPT's unique features, for configuring effective agents with limited data. The findings suggest that, despite some limitations, a MyGPT agent developed with these strategies can serve as a useful coding assistant by generating coding suggestions. 

---
# Mitigating Behavioral Hallucination in Multimodal Large Language Models for Sequential Images 

**Authors**: Liangliang You, Junchi Yao, Shu Yang, Guimin Hu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07184)  

**Abstract**: While multimodal large language models excel at various tasks, they still suffer from hallucinations, which limit their reliability and scalability for broader domain applications. To address this issue, recent research mainly focuses on objective hallucination. However, for sequential images, besides objective hallucination, there is also behavioral hallucination, which is less studied. This work aims to fill in the gap. We first reveal that behavioral hallucinations mainly arise from two key factors: prior-driven bias and the snowball effect. Based on these observations, we introduce SHE (Sequence Hallucination Eradication), a lightweight, two-stage framework that (1) detects hallucinations via visual-textual alignment check using our proposed adaptive temporal window and (2) mitigates them via orthogonal projection onto the joint embedding space. We also propose a new metric (BEACH) to quantify behavioral hallucination severity. Empirical results on standard benchmarks demonstrate that SHE reduces behavioral hallucination by over 10% on BEACH while maintaining descriptive accuracy. 

---
# Translating Federated Learning Algorithms in Python into CSP Processes Using ChatGPT 

**Authors**: Miroslav Popovic, Marko Popovic, Miodrag Djukic, Ilija Basicevic  

**Link**: [PDF](https://arxiv.org/pdf/2506.07173)  

**Abstract**: The Python Testbed for Federated Learning Algorithms is a simple Python FL framework that is easy to use by ML&AI developers who do not need to be professional programmers and is also amenable to LLMs. In the previous research, generic federated learning algorithms provided by this framework were manually translated into the CSP processes and algorithms' safety and liveness properties were automatically verified by the model checker PAT. In this paper, a simple translation process is introduced wherein the ChatGPT is used to automate the translation of the mentioned federated learning algorithms in Python into the corresponding CSP processes. Within the process, the minimality of the used context is estimated based on the feedback from ChatGPT. The proposed translation process was experimentally validated by successful translation (verified by the model checker PAT) of both generic centralized and decentralized federated learning algorithms. 

---
# BRIGHT+: Upgrading the BRIGHT Benchmark with MARCUS, a Multi-Agent RAG Clean-Up Suite 

**Authors**: Liyang Chen, Yujun Cai, Jieqiong Dong, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07116)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems require corpora that are both structurally clean and semantically coherent. BRIGHT is a recent and influential benchmark designed to evaluate complex multi-hop retrieval across diverse, high-reasoning domains. However, its practical effectiveness is limited by common web-crawled artifacts - such as content redundancy and semantic discontinuity - that impair retrieval accuracy and downstream reasoning. Notably, we find that such issues are concentrated in seven StackExchange-derived subdomains, while other domains (e.g., Coding and Theorem-based content) remain relatively clean.
In this study, we present MARCUS, a multi-agent pipeline that leverages large language models (LLMs) to systematically clean and re-chunk BRIGHT into a higher-quality corpus: BRIGHT-Plus. MARCUS applies dedicated agents for structural noise removal and semantic segmentation, preserving answer-bearing spans while improving contextual integrity. Experimental evaluations demonstrate that BRIGHT-Plus yields consistent and significant improvements in both retrieval accuracy and multi-hop reasoning across a diverse set of retrievers. We release both the BRIGHT-Plus corpus and the MARCUS pipeline to support future research on robust, reasoning-centric retrieval. 

---
# Reasoning Paths as Signals: Augmenting Multi-hop Fact Verification through Structural Reasoning Progression 

**Authors**: Liwen Zheng, Chaozhuo Li, Haoran Jia, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07075)  

**Abstract**: The growing complexity of factual claims in real-world scenarios presents significant challenges for automated fact verification systems, particularly in accurately aggregating and reasoning over multi-hop evidence. Existing approaches often rely on static or shallow models that fail to capture the evolving structure of reasoning paths, leading to fragmented retrieval and limited interpretability. To address these issues, we propose a Structural Reasoning framework for Multi-hop Fact Verification that explicitly models reasoning paths as structured graphs throughout both evidence retrieval and claim verification stages. Our method comprises two key modules: a structure-enhanced retrieval mechanism that constructs reasoning graphs to guide evidence collection, and a reasoning-path-guided verification module that incrementally builds subgraphs to represent evolving inference trajectories. We further incorporate a structure-aware reasoning mechanism that captures long-range dependencies across multi-hop evidence chains, enabling more precise verification. Extensive experiments on the FEVER and HoVer datasets demonstrate that our approach consistently outperforms strong baselines, highlighting the effectiveness of reasoning-path modeling in enhancing retrieval precision and verification accuracy. 

---
# Mathesis: Towards Formal Theorem Proving from Natural Languages 

**Authors**: Yu Xuejun, Jianyuan Zhong, Zijin Feng, Pengyi Zhai, Roozbeh Yousefzadeh, Wei Chong Ng, Haoxiong Liu, Ziyi Shou, Jing Xiong, Yudong Zhou, Claudia Beth Ong, Austen Jeremy Sugiarto, Yaoxi Zhang, Wai Ming Tai, Huan Cao, Dongcai Lu, Jiacheng Sun, Qiang Xu, Shen Xin, Zhenguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07047)  

**Abstract**: Recent advances in large language models show strong promise for formal reasoning. However, most LLM-based theorem provers have long been constrained by the need for expert-written formal statements as inputs, limiting their applicability to real-world problems expressed in natural language. We tackle this gap with Mathesis, the first end-to-end theorem proving pipeline processing informal problem statements. It contributes Mathesis-Autoformalizer, the first autoformalizer using reinforcement learning to enhance the formalization ability of natural language problems, aided by our novel LeanScorer framework for nuanced formalization quality assessment. It also proposes a Mathesis-Prover, which generates formal proofs from the formalized statements. To evaluate the real-world applicability of end-to-end formal theorem proving, we introduce Gaokao-Formal, a benchmark of 488 complex problems from China's national college entrance exam. Our approach is carefully designed, with a thorough study of each component. Experiments demonstrate Mathesis's effectiveness, with the autoformalizer outperforming the best baseline by 22% in pass-rate on Gaokao-Formal. The full system surpasses other model combinations, achieving 64% accuracy on MiniF2F with pass@32 and a state-of-the-art 18% on Gaokao-Formal. 

---
# Evaluating LLM-corrupted Crowdsourcing Data Without Ground Truth 

**Authors**: Yichi Zhang, Jinlong Pang, Zhaowei Zhu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06991)  

**Abstract**: The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimension training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets. 

---
# Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments 

**Authors**: Riley Simmons-Edler, Ryan P. Badman, Felix Baastad Berg, Raymond Chua, John J. Vastola, Joshua Lunger, William Qian, Kanaka Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06981)  

**Abstract**: Understanding the behavior of deep reinforcement learning (DRL) agents -- particularly as task and agent sophistication increase -- requires more than simple comparison of reward curves, yet standard methods for behavioral analysis remain underdeveloped in DRL. We apply tools from neuroscience and ethology to study DRL agents in a novel, complex, partially observable environment, ForageWorld, designed to capture key aspects of real-world animal foraging -- including sparse, depleting resource patches, predator threats, and spatially extended arenas. We use this environment as a platform for applying joint behavioral and neural analysis to agents, revealing detailed, quantitatively grounded insights into agent strategies, memory, and planning. Contrary to common assumptions, we find that model-free RNN-based DRL agents can exhibit structured, planning-like behavior purely through emergent dynamics -- without requiring explicit memory modules or world models. Our results show that studying DRL agents like animals -- analyzing them with neuroethology-inspired tools that reveal structure in both behavior and neural dynamics -- uncovers rich structure in their learning dynamics that would otherwise remain invisible. We distill these tools into a general analysis framework linking core behavioral and representational features to diagnostic methods, which can be reused for a wide range of tasks and agents. As agents grow more complex and autonomous, bridging neuroscience, cognitive science, and AI will be essential -- not just for understanding their behavior, but for ensuring safe alignment and maximizing desirable behaviors that are hard to measure via reward. We show how this can be done by drawing on lessons from how biological intelligence is studied. 

---
# Long-Tailed Learning for Generalized Category Discovery 

**Authors**: Cuong Manh Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06965)  

**Abstract**: Generalized Category Discovery (GCD) utilizes labeled samples of known classes to discover novel classes in unlabeled samples. Existing methods show effective performance on artificial datasets with balanced distributions. However, real-world datasets are always imbalanced, significantly affecting the effectiveness of these methods. To solve this problem, we propose a novel framework that performs generalized category discovery in long-tailed distributions. We first present a self-guided labeling technique that uses a learnable distribution to generate pseudo-labels, resulting in less biased classifiers. We then introduce a representation balancing process to derive discriminative representations. By mining sample neighborhoods, this process encourages the model to focus more on tail classes. We conduct experiments on public datasets to demonstrate the effectiveness of the proposed framework. The results show that our model exceeds previous state-of-the-art methods. 

---
# Deontically Constrained Policy Improvement in Reinforcement Learning Agents 

**Authors**: Alena Makarova, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2506.06959)  

**Abstract**: Markov Decision Processes (MDPs) are the most common model for decision making under uncertainty in the Machine Learning community. An MDP captures non-determinism, probabilistic uncertainty, and an explicit model of action. A Reinforcement Learning (RL) agent learns to act in an MDP by maximizing a utility function. This paper considers the problem of learning a decision policy that maximizes utility subject to satisfying a constraint expressed in deontic logic. In this setup, the utility captures the agent's mission - such as going quickly from A to B. The deontic formula represents (ethical, social, situational) constraints on how the agent might achieve its mission by prohibiting classes of behaviors. We use the logic of Expected Act Utilitarianism, a probabilistic stit logic that can be interpreted over controlled MDPs. We develop a variation on policy improvement, and show that it reaches a constrained local maximum of the mission utility. Given that in stit logic, an agent's duty is derived from value maximization, this can be seen as a way of acting to simultaneously maximize two value functions, one of which is implicit, in a bi-level structure. We illustrate these results with experiments on sample MDPs. 

---
# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity 

**Authors**: Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, Mehrdad Farajtabar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06941)  

**Abstract**: Recent generations of language models have introduced Large Reasoning Models (LRMs) that generate detailed thinking processes before providing answers. While these models demonstrate improved performance on reasoning benchmarks, their fundamental capabilities, scaling properties, and limitations remain insufficiently understood. Current evaluations primarily focus on established math and coding benchmarks, emphasizing final answer accuracy. However, this evaluation paradigm often suffers from contamination and does not provide insights into the reasoning traces. In this work, we systematically investigate these gaps with the help of controllable puzzle environments that allow precise manipulation of complexity while maintaining consistent logical structures. This setup enables the analysis of not only final answers but also the internal reasoning traces, offering insights into how LRMs think. Through extensive experiments, we show that LRMs face a complete accuracy collapse beyond certain complexities. Moreover, they exhibit a counterintuitive scaling limit: their reasoning effort increases with problem complexity up to a point, then declines despite having remaining token budget. By comparing LRMs with their standard LLM counterparts under same inference compute, we identify three performance regimes: (1) low-complexity tasks where standard models outperform LRMs, (2) medium-complexity tasks where LRMs demonstrates advantage, and (3) high-complexity tasks where both models face complete collapse. We found that LRMs have limitations in exact computation: they fail to use explicit algorithms and reason inconsistently across scales. We also investigate the reasoning traces in more depth, studying the patterns of explored solutions and analyzing the models' computational behavior, shedding light on their strengths, limitations, and raising questions about their reasoning capabilities. 

---
# An Agentic Framework for Autonomous Metamaterial Modeling and Inverse Design 

**Authors**: Darui Lu, Jordan M. Malof, Willie J. Padilla  

**Link**: [PDF](https://arxiv.org/pdf/2506.06935)  

**Abstract**: Recent significant advances in integrating multiple Large Language Model (LLM) systems have enabled Agentic Frameworks capable of performing complex tasks autonomously, including novel scientific research. We develop and demonstrate such a framework specifically for the inverse design of photonic metamaterials. When queried with a desired optical spectrum, the Agent autonomously proposes and develops a forward deep learning model, accesses external tools via APIs for tasks like simulation and optimization, utilizes memory, and generates a final design via a deep inverse method. The framework's effectiveness is demonstrated in its ability to automate, reason, plan, and adapt. Notably, the Agentic Framework possesses internal reflection and decision flexibility, permitting highly varied and potentially novel outputs. 

---
# Boosting LLM Reasoning via Spontaneous Self-Correction 

**Authors**: Xutong Zhao, Tengyu Xu, Xuewei Wang, Zhengxing Chen, Di Jin, Liang Tan, Yen-Ting, Zishun Yu, Zhuokai Zhao, Yun He, Sinong Wang, Han Fang, Sarath Chandar, Chen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06923)  

**Abstract**: While large language models (LLMs) have demonstrated remarkable success on a broad range of tasks, math reasoning remains a challenging one. One of the approaches for improving math reasoning is self-correction, which designs self-improving loops to let the model correct its own mistakes. However, existing self-correction approaches treat corrections as standalone post-generation refinements, relying on extra prompt and system designs to elicit self-corrections, instead of performing real-time, spontaneous self-corrections in a single pass. To address this, we propose SPOC, a spontaneous self-correction approach that enables LLMs to generate interleaved solutions and verifications in a single inference pass, with generation dynamically terminated based on verification outcomes, thereby effectively scaling inference time compute. SPOC considers a multi-agent perspective by assigning dual roles -- solution proposer and verifier -- to the same model. We adopt a simple yet effective approach to generate synthetic data for fine-tuning, enabling the model to develop capabilities for self-verification and multi-agent collaboration. We further improve its solution proposal and verification accuracy through online reinforcement learning. Experiments on mathematical reasoning benchmarks show that SPOC significantly improves performance. Notably, SPOC boosts the accuracy of Llama-3.1-8B and 70B Instruct models, achieving gains of 8.8% and 11.6% on MATH500, 10.0% and 20.0% on AMC23, and 3.3% and 6.7% on AIME24, respectively. 

---
# Causal Graph based Event Reasoning using Semantic Relation Experts 

**Authors**: Mahnaz Koupaee, Xueying Bai, Mudan Chen, Greg Durrett, Nathanael Chambers, Niranjan Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06910)  

**Abstract**: Understanding how events in a scenario causally connect with each other is important for effectively modeling and reasoning about events. But event reasoning remains a difficult challenge, and despite recent advances, Large Language Models (LLMs) still struggle to accurately identify causal connections between events. This struggle leads to poor performance on deeper reasoning tasks like event forecasting and timeline understanding. To address this challenge, we investigate the generation of causal event graphs (e.g., A enables B) as a parallel mechanism to help LLMs explicitly represent causality during inference. This paper evaluates both how to generate correct graphs as well as how graphs can assist reasoning. We propose a collaborative approach to causal graph generation where we use LLMs to simulate experts that focus on specific semantic relations. The experts engage in multiple rounds of discussions which are then consolidated by a final expert. Then, to demonstrate the utility of causal graphs, we use them on multiple downstream applications, and also introduce a new explainable event prediction task that requires a causal chain of events in the explanation. These explanations are more informative and coherent than baseline generations. Finally, our overall approach not finetuned on any downstream task, achieves competitive results with state-of-the-art models on both forecasting and next event prediction tasks. 

---
# Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering 

**Authors**: Akash Gupta, Amos Storkey, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2506.06905)  

**Abstract**: Large Multimodal Models (LMMs) often rely on in-context learning (ICL) to perform new tasks with minimal supervision. However, ICL performance, especially in smaller LMMs, is inconsistent and does not always improve monotonically with increasing examples. We hypothesize that this occurs due to the LMM being overwhelmed by additional information present in the image embeddings, which is not required for the downstream task. To address this, we propose a meta-learning approach that provides an alternative for inducing few-shot capabilities in LMMs, using a fixed set of soft prompts that are distilled from task-relevant image features and can be adapted at test time using a few examples. To facilitate this distillation, we introduce an attention-mapper module that can be easily integrated with the popular LLaVA v1.5 architecture and is jointly learned with soft prompts, enabling task adaptation in LMMs under low-data regimes with just a few gradient steps. Evaluation on the VL-ICL Bench shows that our method consistently outperforms ICL and related prompt-tuning approaches, even under image perturbations, improving task induction and reasoning across visual question answering tasks. 

---
# KnowCoder-V2: Deep Knowledge Analysis 

**Authors**: Zixuan Li, Wenxuan Liu, Long Bai, Chunmao Zhang, Wei Li, Fenghui Zhang, Quanxin Jin, Ruoyun He, Zhuo Chen, Zhilei Hu, Fei Wang, Bingbing Xu, Xuhui Jiang, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06881)  

**Abstract**: Deep knowledge analysis tasks always involve the systematic extraction and association of knowledge from large volumes of data, followed by logical reasoning to discover insights. However, to solve such complex tasks, existing deep research frameworks face three major challenges: 1) They lack systematic organization and management of knowledge; 2) They operate purely online, making it inefficient for tasks that rely on shared and large-scale knowledge; 3) They cannot perform complex knowledge computation, limiting their abilities to produce insightful analytical results. Motivated by these, in this paper, we propose a \textbf{K}nowledgeable \textbf{D}eep \textbf{R}esearch (\textbf{KDR}) framework that empowers deep research with deep knowledge analysis capability. Specifically, it introduces an independent knowledge organization phase to preprocess large-scale, domain-relevant data into systematic knowledge offline. Based on this knowledge, it extends deep research with an additional kind of reasoning steps that perform complex knowledge computation in an online manner. To enhance the abilities of LLMs to solve knowledge analysis tasks in the above framework, we further introduce \textbf{\KCII}, an LLM that bridges knowledge organization and reasoning via unified code generation. For knowledge organization, it generates instantiation code for predefined classes, transforming data into knowledge objects. For knowledge computation, it generates analysis code and executes on the above knowledge objects to obtain deep analysis results. Experimental results on more than thirty datasets across six knowledge analysis tasks demonstrate the effectiveness of \KCII. Moreover, when integrated into the KDR framework, \KCII can generate high-quality reports with insightful analytical results compared to the mainstream deep research framework. 

---
# Incorporating Failure of Machine Learning in Dynamic Probabilistic Safety Assurance 

**Authors**: Razieh Arshadizadeh, Mahmoud Asgari, Zeinab Khosravi, Yiannis Papadopoulos, Koorosh Aslansefat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06868)  

**Abstract**: Machine Learning (ML) models are increasingly integrated into safety-critical systems, such as autonomous vehicle platooning, to enable real-time decision-making. However, their inherent imperfection introduces a new class of failure: reasoning failures often triggered by distributional shifts between operational and training data. Traditional safety assessment methods, which rely on design artefacts or code, are ill-suited for ML components that learn behaviour from data. SafeML was recently proposed to dynamically detect such shifts and assign confidence levels to the reasoning of ML-based components. Building on this, we introduce a probabilistic safety assurance framework that integrates SafeML with Bayesian Networks (BNs) to model ML failures as part of a broader causal safety analysis. This allows for dynamic safety evaluation and system adaptation under uncertainty. We demonstrate the approach on an simulated automotive platooning system with traffic sign recognition. The findings highlight the potential broader benefits of explicitly modelling ML failures in safety assessment. 

---
# United Minds or Isolated Agents? Exploring Coordination of LLMs under Cognitive Load Theory 

**Authors**: HaoYang Shang, Xuan Liu, Zi Liang, Jie Zhang, Haibo Hu, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06843)  

**Abstract**: Large Language Models (LLMs) exhibit a notable performance ceiling on complex, multi-faceted tasks, as they often fail to integrate diverse information or adhere to multiple constraints. We posit that such limitation arises when the demands of a task exceed the LLM's effective cognitive load capacity. This interpretation draws a strong analogy to Cognitive Load Theory (CLT) in cognitive science, which explains similar performance boundaries in the human mind, and is further supported by emerging evidence that reveals LLMs have bounded working memory characteristics. Building upon this CLT-grounded understanding, we introduce CoThinker, a novel LLM-based multi-agent framework designed to mitigate cognitive overload and enhance collaborative problem-solving abilities. CoThinker operationalizes CLT principles by distributing intrinsic cognitive load through agent specialization and managing transactional load via structured communication and a collective working memory. We empirically validate CoThinker on complex problem-solving tasks and fabricated high cognitive load scenarios, demonstrating improvements over existing multi-agent baselines in solution quality and efficiency. Our analysis reveals characteristic interaction patterns, providing insights into the emergence of collective cognition and effective load management, thus offering a principled approach to overcoming LLM performance ceilings. 

---
# Cross-Entropy Games for Language Models: From Implicit Knowledge to General Capability Measures 

**Authors**: Clément Hongler, Andrew Emil  

**Link**: [PDF](https://arxiv.org/pdf/2506.06832)  

**Abstract**: Large Language Models (LLMs) define probability measures on text. By considering the implicit knowledge question of what it means for an LLM to know such a measure and what it entails algorithmically, we are naturally led to formulate a series of tasks that go beyond generative sampling, involving forms of summarization, counterfactual thinking, anomaly detection, originality search, reverse prompting, debating, creative solving, etc. These tasks can be formulated as games based on LLM measures, which we call Cross-Entropy (Xent) Games. Xent Games can be single-player or multi-player. They involve cross-entropy scores and cross-entropy constraints, and can be expressed as simple computational graphs and programs. We show the Xent Game space is large enough to contain a wealth of interesting examples, while being constructible from basic game-theoretic consistency axioms. We then discuss how the Xent Game space can be used to measure the abilities of LLMs. This leads to the construction of Xent Game measures: finite families of Xent Games that can be used as capability benchmarks, built from a given scope, by extracting a covering measure. To address the unbounded scope problem associated with the challenge of measuring general abilities, we propose to explore the space of Xent Games in a coherent fashion, using ideas inspired by evolutionary dynamics. 

---
# Learning What Matters Now: A Dual-Critic Context-Aware RL Framework for Priority-Driven Information Gain 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.06786)  

**Abstract**: Autonomous systems operating in high-stakes search-and-rescue (SAR) missions must continuously gather mission-critical information while flexibly adapting to shifting operational priorities. We propose CA-MIQ (Context-Aware Max-Information Q-learning), a lightweight dual-critic reinforcement learning (RL) framework that dynamically adjusts its exploration strategy whenever mission priorities change. CA-MIQ pairs a standard extrinsic critic for task reward with an intrinsic critic that fuses state-novelty, information-location awareness, and real-time priority alignment. A built-in shift detector triggers transient exploration boosts and selective critic resets, allowing the agent to re-focus after a priority revision. In a simulated SAR grid-world, where experiments specifically test adaptation to changes in the priority order of information types the agent is expected to focus on, CA-MIQ achieves nearly four times higher mission-success rates than baselines after a single priority shift and more than three times better performance in multiple-shift scenarios, achieving 100% recovery while baseline methods fail to adapt. These results highlight CA-MIQ's effectiveness in any discrete environment with piecewise-stationary information-value distributions. 

---
# Bio-Inspired Classification: Combining Information Theory and Spiking Neural Networks -- Influence of the Learning Rules 

**Authors**: Zofia Rudnicka, Janusz Szczepanski, Agnieszka Pregowska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06750)  

**Abstract**: Training of Spiking Neural Networks (SNN) is challenging due to their unique properties, including temporal dynamics, non-differentiability of spike events, and sparse event-driven activations. In this paper, we widely consider the influence of the type of chosen learning algorithm, including bioinspired learning rules on the accuracy of classification. We proposed a bioinspired classifier based on the combination of SNN and Lempel-Ziv complexity (LZC). This approach synergizes the strengths of SNNs in temporal precision and biological realism with LZC's structural complexity analysis, facilitating efficient and interpretable classification of spatiotemporal neural data. It turned out that the classic backpropagation algorithm achieves excellent classification accuracy, but at extremely high computational cost, which makes it impractical for real-time applications. Biologically inspired learning algorithms such as tempotron and Spikprop provide increased computational efficiency while maintaining competitive classification performance, making them suitable for time-sensitive tasks. The results obtained indicate that the selection of the most appropriate learning algorithm depends on the trade-off between classification accuracy and computational cost as well as application constraints. 

---
# AI PsyRoom: Artificial Intelligence Platform for Segmented Yearning and Reactive Outcome Optimization Method 

**Authors**: Yigui Feng, Qinglin Wang, Ke Liu, Xinhai Chen, Bo Yang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06740)  

**Abstract**: Psychological counseling faces huge challenges due to the growing demand for mental health services and the shortage of trained professionals. Large language models (LLMs) have shown potential to assist psychological counseling, especially in empathy and emotional support. However, existing models lack a deep understanding of emotions and are unable to generate personalized treatment plans based on fine-grained emotions. To address these shortcomings, we present AI PsyRoom, a multi-agent simulation framework designed to enhance psychological counseling by generating empathetic and emotionally nuanced conversations. By leveraging fine-grained emotion classification and a multi-agent framework, we construct a multi-agent PsyRoom A for dialogue reconstruction, generating a high-quality dialogue dataset EmoPsy, which contains 35 sub-emotions, 423 specific emotion scenarios, and 12,350 dialogues. We also propose PsyRoom B for generating personalized treatment plans. Quantitative evaluations demonstrate that AI PsyRoom significantly outperforms state-of-the-art methods, achieving 18% improvement in problem orientation, 23% in expression, 24% in Empathy, and 16% in interactive communication quality. The datasets and models are publicly available, providing a foundation for advancing AI-assisted psychological counseling research. 

---
# Honey, I shrunk the hypothesis space (through logical preprocessing) 

**Authors**: Andrew Cropper, Filipe Gouveia, David M. Cerna  

**Link**: [PDF](https://arxiv.org/pdf/2506.06739)  

**Abstract**: Inductive logic programming (ILP) is a form of logical machine learning. The goal is to search a hypothesis space for a hypothesis that generalises training examples and background knowledge. We introduce an approach that 'shrinks' the hypothesis space before an ILP system searches it. Our approach uses background knowledge to find rules that cannot be in an optimal hypothesis regardless of the training examples. For instance, our approach discovers relationships such as "even numbers cannot be odd" and "prime numbers greater than 2 are odd". It then removes violating rules from the hypothesis space. We implement our approach using answer set programming and use it to shrink the hypothesis space of a constraint-based ILP system. Our experiments on multiple domains, including visual reasoning and game playing, show that our approach can substantially reduce learning times whilst maintaining predictive accuracies. For instance, given just 10 seconds of preprocessing time, our approach can reduce learning times from over 10 hours to only 2 seconds. 

---
# VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs 

**Authors**: Can Li, Ting Zhang, Mei Wang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06727)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated remarkable problem-solving capabilities across various domains. However, their ability to perform mathematical reasoning when answer options are represented as images--an essential aspect of multi-image comprehension--remains underexplored. To bridge this gap, we introduce VisioMath, a benchmark designed to evaluate mathematical reasoning in multimodal contexts involving image-based answer choices. VisioMath comprises 8,070 images and 1,800 multiple-choice questions, where each answer option is an image, presenting unique challenges to existing LMMs. To the best of our knowledge, VisioMath is the first dataset specifically tailored for mathematical reasoning in image-based-option scenarios, where fine-grained distinctions between answer choices are critical for accurate problem-solving. We systematically evaluate state-of-the-art LMMs on VisioMath and find that even the most advanced models struggle with this task. Notably, GPT-4o achieves only 45.9% accuracy, underscoring the limitations of current models in reasoning over visually similar answer choices. By addressing a crucial gap in existing benchmarks, VisioMath establishes a rigorous testbed for future research, driving advancements in multimodal reasoning. 

---
# WorldLLM: Improving LLMs' world modeling using curiosity-driven theory-making 

**Authors**: Guillaume Levy, Cedric Colas, Pierre-Yves Oudeyer, Thomas Carta, Clement Romac  

**Link**: [PDF](https://arxiv.org/pdf/2506.06725)  

**Abstract**: Large Language Models (LLMs) possess general world knowledge but often struggle to generate precise predictions in structured, domain-specific contexts such as simulations. These limitations arise from their inability to ground their broad, unstructured understanding in specific environments. To address this, we present WorldLLM, a framework that enhances LLM-based world modeling by combining Bayesian inference and autonomous active exploration with reinforcement learning. WorldLLM leverages the in-context learning abilities of LLMs to guide an LLM-based world model's predictions using natural language hypotheses given in its prompt. These hypotheses are iteratively refined through a Bayesian inference framework that leverages a second LLM as the proposal distribution given collected evidence. This evidence is collected using a curiosity-driven reinforcement learning policy that explores the environment to find transitions with a low log-likelihood under our LLM-based predictive model using the current hypotheses. By alternating between refining hypotheses and collecting new evidence, our framework autonomously drives continual improvement of the predictions. Our experiments demonstrate the effectiveness of WorldLLM in a textual game environment that requires agents to manipulate and combine objects. The framework not only enhances predictive accuracy, but also generates human-interpretable theories of environment dynamics. 

---
# Integrating AI Planning Semantics into SysML System Models for Automated PDDL File Generation 

**Authors**: Hamied Nabizada, Tom Jeleniewski, Lasse Beers, Maximilian Weigand, Felix Gehlhoff, Alexander Fay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06714)  

**Abstract**: This paper presents a SysML profile that enables the direct integration of planning semantics based on the Planning Domain Definition Language (PDDL) into system models. Reusable stereotypes are defined for key PDDL concepts such as types, predicates, functions and actions, while formal OCL constraints ensure syntactic consistency. The profile was derived from the Backus-Naur Form (BNF) definition of PDDL 3.1 to align with SysML modeling practices. A case study from aircraft manufacturing demonstrates the application of the profile: a robotic system with interchangeable end effectors is modeled and enriched to generate both domain and problem descriptions in PDDL format. These are used as input to a PDDL solver to derive optimized execution plans. The approach supports automated and model-based generation of planning descriptions and provides a reusable bridge between system modeling and AI planning in engineering design. 

---
# Contextual Experience Replay for Self-Improvement of Language Agents 

**Authors**: Yitao Liu, Chenglei Si, Karthik Narasimhan, Shunyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06698)  

**Abstract**: Large language model (LLM) agents have been applied to sequential decision-making tasks such as web navigation, but without any environment-specific experiences, they often fail in these complex tasks. Moreover, current LLM agents are not designed to continually learn from past experiences during inference time, which could be crucial for them to gain these environment-specific experiences. To address this, we propose Contextual Experience Replay (CER), a training-free framework to enable efficient self-improvement for language agents in their context window. Specifically, CER accumulates and synthesizes past experiences into a dynamic memory buffer. These experiences encompass environment dynamics and common decision-making patterns, allowing the agents to retrieve and augment themselves with relevant knowledge in new tasks, enhancing their adaptability in complex environments. We evaluate CER on the challenging WebArena and VisualWebArena benchmarks. On VisualWebArena, CER achieves a competitive performance of 31.9%. On WebArena, CER also gets a competitive average success rate of 36.7%, relatively improving the success rate of the GPT-4o agent baseline by 51.0%. We also conduct a comprehensive analysis on it to prove its efficiency, validity and understand it better. 

---
# GELD: A Unified Neural Model for Efficiently Solving Traveling Salesman Problems Across Different Scales 

**Authors**: Yubin Xiao, Di Wang, Rui Cao, Xuan Wu, Boyang Li, You Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06634)  

**Abstract**: The Traveling Salesman Problem (TSP) is a well-known combinatorial optimization problem with broad real-world applications. Recent advancements in neural network-based TSP solvers have shown promising results. Nonetheless, these models often struggle to efficiently solve both small- and large-scale TSPs using the same set of pre-trained model parameters, limiting their practical utility. To address this issue, we introduce a novel neural TSP solver named GELD, built upon our proposed broad global assessment and refined local selection framework. Specifically, GELD integrates a lightweight Global-view Encoder (GE) with a heavyweight Local-view Decoder (LD) to enrich embedding representation while accelerating the decision-making process. Moreover, GE incorporates a novel low-complexity attention mechanism, allowing GELD to achieve low inference latency and scalability to larger-scale TSPs. Additionally, we propose a two-stage training strategy that utilizes training instances of different sizes to bolster GELD's generalization ability. Extensive experiments conducted on both synthetic and real-world datasets demonstrate that GELD outperforms seven state-of-the-art models considering both solution quality and inference speed. Furthermore, GELD can be employed as a post-processing method to significantly elevate the quality of the solutions derived by existing neural TSP solvers via spending affordable additional computing time. Notably, GELD is shown as capable of solving TSPs with up to 744,710 nodes, first-of-its-kind to solve this large size TSP without relying on divide-and-conquer strategies to the best of our knowledge. 

---
# AI Simulation by Digital Twins: Systematic Survey, Reference Framework, and Mapping to a Standardized Architecture 

**Authors**: Xiaoran Liu, Istvan David  

**Link**: [PDF](https://arxiv.org/pdf/2506.06580)  

**Abstract**: Insufficient data volume and quality are particularly pressing challenges in the adoption of modern subsymbolic AI. To alleviate these challenges, AI simulation uses virtual training environments in which AI agents can be safely and efficiently developed with simulated, synthetic data. Digital twins open new avenues in AI simulation, as these high-fidelity virtual replicas of physical systems are equipped with state-of-the-art simulators and the ability to further interact with the physical system for additional data collection. In this article, we report on our systematic survey of digital twin-enabled AI simulation. By analyzing 22 primary studies, we identify technological trends and derive a reference framework to situate digital twins and AI components. Based on our findings, we derive a reference framework and provide architectural guidelines by mapping it onto the ISO 23247 reference architecture for digital twins. Finally, we identify challenges and research opportunities for prospective researchers. 

---
# The Optimization Paradox in Clinical AI Multi-Agent Systems 

**Authors**: Suhana Bedi, Iddah Mlauzi, Daniel Shin, Sanmi Koyejo, Nigam H. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2506.06574)  

**Abstract**: Multi-agent artificial intelligence systems are increasingly deployed in clinical settings, yet the relationship between component-level optimization and system-wide performance remains poorly understood. We evaluated this relationship using 2,400 real patient cases from the MIMIC-CDM dataset across four abdominal pathologies (appendicitis, pancreatitis, cholecystitis, diverticulitis), decomposing clinical diagnosis into information gathering, interpretation, and differential diagnosis. We evaluated single agent systems (one model performing all tasks) against multi-agent systems (specialized models for each task) using comprehensive metrics spanning diagnostic outcomes, process adherence, and cost efficiency. Our results reveal a paradox: while multi-agent systems generally outperformed single agents, the component-optimized or Best of Breed system with superior components and excellent process metrics (85.5% information accuracy) significantly underperformed in diagnostic accuracy (67.7% vs. 77.4% for a top multi-agent system). This finding underscores that successful integration of AI in healthcare requires not just component level optimization but also attention to information flow and compatibility between agents. Our findings highlight the need for end to end system validation rather than relying on component metrics alone. 

---
# ScriptDoctor: Automatic Generation of PuzzleScript Games via Large Language Models and Tree Search 

**Authors**: Sam Earle, Ahmed Khalifa, Muhammad Umair Nasir, Zehua Jiang, Graham Todd, Andrzej Banburski-Fahey, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2506.06524)  

**Abstract**: There is much interest in using large pre-trained models in Automatic Game Design (AGD), whether via the generation of code, assets, or more abstract conceptualization of design ideas. But so far this interest largely stems from the ad hoc use of such generative models under persistent human supervision. Much work remains to show how these tools can be integrated into longer-time-horizon AGD pipelines, in which systems interface with game engines to test generated content autonomously. To this end, we introduce ScriptDoctor, a Large Language Model (LLM)-driven system for automatically generating and testing games in PuzzleScript, an expressive but highly constrained description language for turn-based puzzle games over 2D gridworlds. ScriptDoctor generates and tests game design ideas in an iterative loop, where human-authored examples are used to ground the system's output, compilation errors from the PuzzleScript engine are used to elicit functional code, and search-based agents play-test generated games. ScriptDoctor serves as a concrete example of the potential of automated, open-ended LLM-based workflows in generating novel game content. 

---
# Reinforcement Learning for Autonomous Warehouse Orchestration in SAP Logistics Execution: Redefining Supply Chain Agility 

**Authors**: Sumanth Pillella  

**Link**: [PDF](https://arxiv.org/pdf/2506.06523)  

**Abstract**: In an era of escalating supply chain demands, SAP Logistics Execution (LE) is pivotal for managing warehouse operations, transportation, and delivery. This research introduces a pioneering framework leveraging reinforcement learning (RL) to autonomously orchestrate warehouse tasks in SAP LE, enhancing operational agility and efficiency. By modeling warehouse processes as dynamic environments, the framework optimizes task allocation, inventory movement, and order picking in real-time. A synthetic dataset of 300,000 LE transactions simulates real-world warehouse scenarios, including multilingual data and operational disruptions. The analysis achieves 95% task optimization accuracy, reducing processing times by 60% compared to traditional methods. Visualizations, including efficiency heatmaps and performance graphs, guide agile warehouse strategies. This approach tackles data privacy, scalability, and SAP integration, offering a transformative solution for modern supply chains. 

---
# SIGMA: Refining Large Language Model Reasoning via Sibling-Guided Monte Carlo Augmentation 

**Authors**: Yanwei Ren, Haotian Zhang, Fuxiang Wu, Jiayan Qiu, Jiaxing Huang, Baosheng Yu, Liu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06470)  

**Abstract**: Enhancing large language models by simply scaling up datasets has begun to yield diminishing returns, shifting the spotlight to data quality. Monte Carlo Tree Search (MCTS) has emerged as a powerful technique for generating high-quality chain-of-thought data, yet conventional approaches typically retain only the top-scoring trajectory from the search tree, discarding sibling nodes that often contain valuable partial insights, recurrent error patterns, and alternative reasoning strategies. This unconditional rejection of non-optimal reasoning branches may waste vast amounts of informative data in the whole search tree. We propose SIGMA (Sibling Guided Monte Carlo Augmentation), a novel framework that reintegrates these discarded sibling nodes to refine LLM reasoning. SIGMA forges semantic links among sibling nodes along each search path and applies a two-stage refinement: a critique model identifies overlooked strengths and weaknesses across the sibling set, and a revision model conducts text-based backpropagation to refine the top-scoring trajectory in light of this comparative feedback. By recovering and amplifying the underutilized but valuable signals from non-optimal reasoning branches, SIGMA substantially improves reasoning trajectories. On the challenging MATH benchmark, our SIGMA-tuned 7B model achieves 54.92% accuracy using only 30K samples, outperforming state-of-the-art models trained on 590K samples. This result highlights that our sibling-guided optimization not only significantly reduces data usage but also significantly boosts LLM reasoning. 

---
# Towards Foundation Model on Temporal Knowledge Graph Reasoning 

**Authors**: Jiaxin Pan, Mojtaba Nayyeri, Osama Mohammed, Daniel Hernandez, Rongchuan Zhang, Cheng Cheng, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2506.06367)  

**Abstract**: Temporal Knowledge Graphs (TKGs) store temporal facts with quadruple formats (s, p, o, t). Existing Temporal Knowledge Graph Embedding (TKGE) models perform link prediction tasks in transductive or semi-inductive settings, which means the entities, relations, and temporal information in the test graph are fully or partially observed during training. Such reliance on seen elements during inference limits the models' ability to transfer to new domains and generalize to real-world scenarios. A central limitation is the difficulty in learning representations for entities, relations, and timestamps that are transferable and not tied to dataset-specific vocabularies. To overcome these limitations, we introduce the first fully-inductive approach to temporal knowledge graph link prediction. Our model employs sinusoidal positional encodings to capture fine-grained temporal patterns and generates adaptive entity and relation representations using message passing conditioned on both local and global temporal contexts. Our model design is agnostic to temporal granularity and time span, effectively addressing temporal discrepancies across TKGs and facilitating time-aware structural information transfer. As a pretrained, scalable, and transferable model, POSTRA demonstrates strong zero-shot performance on unseen temporal knowledge graphs, effectively generalizing to novel entities, relations, and timestamps. Extensive theoretical analysis and empirical results show that a single pretrained model can improve zero-shot performance on various inductive temporal reasoning scenarios, marking a significant step toward a foundation model for temporal KGs. 

---
# Will artificial agents pursue power by default? 

**Authors**: Christian Tarsney  

**Link**: [PDF](https://arxiv.org/pdf/2506.06352)  

**Abstract**: Researchers worried about catastrophic risks from advanced AI have argued that we should expect sufficiently capable AI agents to pursue power over humanity because power is a convergent instrumental goal, something that is useful for a wide range of final goals. Others have recently expressed skepticism of these claims. This paper aims to formalize the concepts of instrumental convergence and power-seeking in an abstract, decision-theoretic framework, and to assess the claim that power is a convergent instrumental goal. I conclude that this claim contains at least an element of truth, but might turn out to have limited predictive utility, since an agent's options cannot always be ranked in terms of power in the absence of substantive information about the agent's final goals. However, the fact of instrumental convergence is more predictive for agents who have a good shot at attaining absolute or near-absolute power. 

---
# Memory OS of AI Agent 

**Authors**: Jiazheng Kang, Mingming Ji, Zhe Zhao, Ting Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.06326)  

**Abstract**: Large Language Models (LLMs) face a crucial challenge from fixed context windows and inadequate memory management, leading to a severe shortage of long-term memory capabilities and limited personalization in the interactive experience with AI agents. To overcome this challenge, we innovatively propose a Memory Operating System, i.e., MemoryOS, to achieve comprehensive and efficient memory management for AI agents. Inspired by the memory management principles in operating systems, MemoryOS designs a hierarchical storage architecture and consists of four key modules: Memory Storage, Updating, Retrieval, and Generation. Specifically, the architecture comprises three levels of storage units: short-term memory, mid-term memory, and long-term personal memory. Key operations within MemoryOS include dynamic updates between storage units: short-term to mid-term updates follow a dialogue-chain-based FIFO principle, while mid-term to long-term updates use a segmented page organization strategy. Our pioneering MemoryOS enables hierarchical memory integration and dynamic updating. Extensive experiments on the LoCoMo benchmark show an average improvement of 49.11% on F1 and 46.18% on BLEU-1 over the baselines on GPT-4o-mini, showing contextual coherence and personalized memory retention in long conversations. The implementation code is open-sourced at this https URL. 

---
# Mapping Human-Agent Co-Learning and Co-Adaptation: A Scoping Review 

**Authors**: Shruti Kumar, Xiaoyu Chen, Xiaomei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06324)  

**Abstract**: Several papers have delved into the challenges of human-AI-robot co-learning and co-adaptation. It has been noted that the terminology used to describe this collaborative relationship in existing studies needs to be more consistent. For example, the prefix "co" is used interchangeably to represent both "collaborative" and "mutual," and the terms "co-learning" and "co-adaptation" are sometimes used interchangeably. However, they can reflect subtle differences in the focus of the studies. The current scoping review's primary research question (RQ1) aims to gather existing papers discussing this collaboration pattern and examine the terms researchers use to describe this human-agent relationship. Given the relative newness of this area of study, we are also keen on exploring the specific types of intelligent agents and task domains that have been considered in existing research (RQ2). This exploration is significant as it can shed light on the diversity of human-agent interactions, from one-time to continuous learning/adaptation scenarios. It can also help us understand the dynamics of human-agent interactions in different task domains, guiding our expectations towards research situated in dynamic, complex domains. Our third objective (RQ3) is to investigate the cognitive theories and frameworks that have been utilized in existing studies to measure human-agent co-learning and co-adaptation. This investigation is crucial as it can help us understand the theoretical underpinnings of human-agent collaboration and adaptation, and it can also guide us in identifying any new frameworks proposed specifically for this type of relationship. 

---
# Large Language Models and Their Applications in Roadway Safety and Mobility Enhancement: A Comprehensive Review 

**Authors**: Muhammad Monjurul Karim, Yan Shi, Shucheng Zhang, Bingzhang Wang, Mehrdad Nasri, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06301)  

**Abstract**: Roadway safety and mobility remain critical challenges for modern transportation systems, demanding innovative analytical frameworks capable of addressing complex, dynamic, and heterogeneous environments. While traditional engineering methods have made progress, the complexity and dynamism of real-world traffic necessitate more advanced analytical frameworks. Large Language Models (LLMs), with their unprecedented capabilities in natural language understanding, knowledge integration, and reasoning, represent a promising paradigm shift. This paper comprehensively reviews the application and customization of LLMs for enhancing roadway safety and mobility. A key focus is how LLMs are adapted -- via architectural, training, prompting, and multimodal strategies -- to bridge the "modality gap" with transportation's unique spatio-temporal and physical data. The review systematically analyzes diverse LLM applications in mobility (e.g., traffic flow prediction, signal control) and safety (e.g., crash analysis, driver behavior assessment,). Enabling technologies such as V2X integration, domain-specific foundation models, explainability frameworks, and edge computing are also examined. Despite significant potential, challenges persist regarding inherent LLM limitations (hallucinations, reasoning deficits), data governance (privacy, bias), deployment complexities (sim-to-real, latency), and rigorous safety assurance. Promising future research directions are highlighted, including advanced multimodal fusion, enhanced spatio-temporal reasoning, human-AI collaboration, continuous learning, and the development of efficient, verifiable systems. This review provides a structured roadmap of current capabilities, limitations, and opportunities, underscoring LLMs' transformative potential while emphasizing the need for responsible innovation to realize safer, more intelligent transportation systems. 

---
# Deep Research Bench: Evaluating AI Web Research Agents 

**Authors**: FutureSearch, Nikos I. Bosse, Jon Evans, Robert G. Gambee, Daniel Hnyk, Peter Mühlbacher, Lawrence Phillips, Dan Schwarz, Jack Wildman  

**Link**: [PDF](https://arxiv.org/pdf/2506.06287)  

**Abstract**: Amongst the most common use cases of modern AI is LLM chat with web search enabled. However, no direct evaluations of the quality of web research agents exist that control for the continually-changing web. We introduce Deep Research Bench, consisting of 89 multi-step web research task instances of varying difficulty across 8 diverse task categories, with the answers carefully worked out by skilled humans. We provide a "RetroSearch" environment with a large frozen set of scraped web pages, and demonstrate that offline "RetroSearch" agents perform comparably to "live web" agents, enabling reliable evaluations of models over time. We provide robust agent tooling and scaffolding to benchmark major LLMs as they are released, including "thinking" models like o3 and Gemini 2.5 Pro. We include automated evaluations of the lengthy agent traces to report progress over time in hallucinations, tool use, and forgetting. Finally, we evaluate the major web research products branded as "Deep Research", "Deep Search", "Search", or "Research." Results are available on a public leaderboard at this https URL. 

---
# NFISiS: New Perspectives on Fuzzy Inference Systems for Renewable Energy Forecasting 

**Authors**: Kaike Sa Teles Rocha Alves, Eduardo Pestana de Aguiar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06285)  

**Abstract**: Evolving Fuzzy Systems (eFS) have gained significant attention due to their ability to adaptively update their structure in response to data dynamics while maintaining interpretability. However, the lack of publicly available implementations of these models limits their accessibility and widespread adoption. To address this gap, we present evolvingfuzzysystems, a Python library that provides implementations of several well-established eFS models, including ePL-KRLS-DISCO, ePL+, eMG, ePL, exTS, Simpl\_eTS, and eTS. The library facilitates model evaluation and comparison by offering built-in tools for training, visualization, and performance assessment. The models are evaluated using the fetch\_california\_housing dataset, with performance measured in terms of normalized root-mean-square error (NRMSE), non-dimensional error index (NDEI), and mean absolute percentage error (MAPE). Additionally, computational complexity is analyzed by measuring execution times and rule evolution during training and testing phases. The results highlight ePL as a simple yet efficient model that balances accuracy and computational cost, making it particularly suitable for real-world applications. By making these models publicly available, evolvingfuzzysystems aims to foster research and practical applications in adaptive and interpretable machine learning. 

---
# Unreal Patterns 

**Authors**: John Beverley, Jim Logan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06284)  

**Abstract**: This paper introduces a framework for representing information about entities that do not exist or may never exist, such as those involving fictional entities, blueprints, simulations, and future scenarios. Traditional approaches that introduce "dummy instances" or rely on modal logic are criticized, and a proposal is defended in which such cases are modeled using the intersections of actual types rather than specific non existent tokens. The paper positions itself within the Basic Formal Ontology and its realist commitments, emphasizing the importance of practical, implementable solutions over purely metaphysical or philosophical proposals, arguing that existing approaches to non existent entities either overcommit to metaphysical assumptions or introduce computational inefficiencies that hinder applications. By developing a structured ontology driven approach to unreal patterns, the paper aims to provide a useful and computationally viable means of handling references to hypothetical or non existent entities. 

---
# Understanding Financial Reasoning in AI: A Multimodal Benchmark and Error Learning Approach 

**Authors**: Shuangyan Deng, Haizhou Peng, Jiachen Xu, Chunhou Liu, Ciprian Doru Giurcuaneanu, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06282)  

**Abstract**: Effective financial reasoning demands not only textual understanding but also the ability to interpret complex visual data such as charts, tables, and trend graphs. This paper introduces a new benchmark designed to evaluate how well AI models - especially large language and multimodal models - reason in finance-specific contexts. Covering 3,200 expert-level question-answer pairs across 15 core financial topics, the benchmark integrates both textual and visual modalities to reflect authentic analytical challenges in finance. To address limitations in current reasoning approaches, we propose an error-aware learning framework that leverages historical model mistakes and feedback to guide inference, without requiring fine-tuning. Our experiments across state-of-the-art models show that multimodal inputs significantly enhance performance and that incorporating error feedback leads to consistent and measurable improvements. The results highlight persistent challenges in visual understanding and mathematical logic, while also demonstrating the promise of self-reflective reasoning in financial AI systems. Our code and data can be found at https://anonymous/FinMR/CodeData. 

---
# StableMTL: Repurposing Latent Diffusion Models for Multi-Task Learning from Partially Annotated Synthetic Datasets 

**Authors**: Anh-Quan Cao, Ivan Lopes, Raoul de Charette  

**Link**: [PDF](https://arxiv.org/pdf/2506.08013)  

**Abstract**: Multi-task learning for dense prediction is limited by the need for extensive annotation for every task, though recent works have explored training with partial task labels. Leveraging the generalization power of diffusion models, we extend the partial learning setup to a zero-shot setting, training a multi-task model on multiple synthetic datasets, each labeled for only a subset of tasks. Our method, StableMTL, repurposes image generators for latent regression. Adapting a denoising framework with task encoding, per-task conditioning and a tailored training scheme. Instead of per-task losses requiring careful balancing, a unified latent loss is adopted, enabling seamless scaling to more tasks. To encourage inter-task synergy, we introduce a multi-stream model with a task-attention mechanism that converts N-to-N task interactions into efficient 1-to-N attention, promoting effective cross-task sharing. StableMTL outperforms baselines on 7 tasks across 8 benchmarks. 

---
# Vision Transformers Don't Need Trained Registers 

**Authors**: Nick Jiang, Amil Dravid, Alexei Efros, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2506.08010)  

**Abstract**: We investigate the mechanism underlying a previously identified phenomenon in Vision Transformers -- the emergence of high-norm tokens that lead to noisy attention maps. We observe that in multiple models (e.g., CLIP, DINOv2), a sparse set of neurons is responsible for concentrating high-norm activations on outlier tokens, leading to irregular attention patterns and degrading downstream visual processing. While the existing solution for removing these outliers involves retraining models from scratch with additional learned register tokens, we use our findings to create a training-free approach to mitigate these artifacts. By shifting the high-norm activations from our discovered register neurons into an additional untrained token, we can mimic the effect of register tokens on a model already trained without registers. We demonstrate that our method produces cleaner attention and feature maps, enhances performance over base models across multiple downstream visual tasks, and achieves results comparable to models explicitly trained with register tokens. We then extend test-time registers to off-the-shelf vision-language models to improve their interpretability. Our results suggest that test-time registers effectively take on the role of register tokens at test-time, offering a training-free solution for any pre-trained model released without them. 

---
# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion 

**Authors**: Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman  

**Link**: [PDF](https://arxiv.org/pdf/2506.08009)  

**Abstract**: We introduce Self Forcing, a novel training paradigm for autoregressive video diffusion models. It addresses the longstanding issue of exposure bias, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference. Unlike prior methods that denoise future frames based on ground-truth context frames, Self Forcing conditions each frame's generation on previously self-generated outputs by performing autoregressive rollout with key-value (KV) caching during training. This strategy enables supervision through a holistic loss at the video level that directly evaluates the quality of the entire generated sequence, rather than relying solely on traditional frame-wise objectives. To ensure training efficiency, we employ a few-step diffusion model along with a stochastic gradient truncation strategy, effectively balancing computational cost and performance. We further introduce a rolling KV cache mechanism that enables efficient autoregressive video extrapolation. Extensive experiments demonstrate that our approach achieves real-time streaming video generation with sub-second latency on a single GPU, while matching or even surpassing the generation quality of significantly slower and non-causal diffusion models. Project website: this http URL 

---
# Hidden in plain sight: VLMs overlook their visual representations 

**Authors**: Stephanie Fu, Tyler Bonnen, Devin Guillory, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2506.08008)  

**Abstract**: Language provides a natural interface to specify and evaluate performance on visual tasks. To realize this possibility, vision language models (VLMs) must successfully integrate visual and linguistic information. Our work compares VLMs to a direct readout of their visual encoders to understand their ability to integrate across these modalities. Across a series of vision-centric benchmarks (e.g., depth estimation, correspondence), we find that VLMs perform substantially worse than their visual encoders, dropping to near-chance performance. We investigate these results through a series of analyses across the entire VLM: namely 1) the degradation of vision representations, 2) brittleness to task prompt, and 3) the language model's role in solving the task. We find that the bottleneck in performing these vision-centric tasks lies in this third category; VLMs are not effectively using visual information easily accessible throughout the entire model, and they inherit the language priors present in the LLM. Our work helps diagnose the failure modes of open-source VLMs, and presents a series of evaluations useful for future investigations into visual understanding within VLMs. 

---
# Dynamic View Synthesis as an Inverse Problem 

**Authors**: Hidir Yesiltepe, Pinar Yanardag  

**Link**: [PDF](https://arxiv.org/pdf/2506.08004)  

**Abstract**: In this work, we address dynamic view synthesis from monocular videos as an inverse problem in a training-free setting. By redesigning the noise initialization phase of a pre-trained video diffusion model, we enable high-fidelity dynamic view synthesis without any weight updates or auxiliary modules. We begin by identifying a fundamental obstacle to deterministic inversion arising from zero-terminal signal-to-noise ratio (SNR) schedules and resolve it by introducing a novel noise representation, termed K-order Recursive Noise Representation. We derive a closed form expression for this representation, enabling precise and efficient alignment between the VAE-encoded and the DDIM inverted latents. To synthesize newly visible regions resulting from camera motion, we introduce Stochastic Latent Modulation, which performs visibility aware sampling over the latent space to complete occluded regions. Comprehensive experiments demonstrate that dynamic view synthesis can be effectively performed through structured latent manipulation in the noise initialization phase. 

---
# Audio-Sync Video Generation with Multi-Stream Temporal Control 

**Authors**: Shuchen Weng, Haojie Zheng, Zheng Chang, Si Li, Boxin Shi, Xinlong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08003)  

**Abstract**: Audio is inherently temporal and closely synchronized with the visual world, making it a naturally aligned and expressive control signal for controllable video generation (e.g., movies). Beyond control, directly translating audio into video is essential for understanding and visualizing rich audio narratives (e.g., Podcasts or historical recordings). However, existing approaches fall short in generating high-quality videos with precise audio-visual synchronization, especially across diverse and complex audio types. In this work, we introduce MTV, a versatile framework for audio-sync video generation. MTV explicitly separates audios into speech, effects, and music tracks, enabling disentangled control over lip motion, event timing, and visual mood, respectively -- resulting in fine-grained and semantically aligned video generation. To support the framework, we additionally present DEMIX, a dataset comprising high-quality cinematic videos and demixed audio tracks. DEMIX is structured into five overlapped subsets, enabling scalable multi-stage training for diverse generation scenarios. Extensive experiments demonstrate that MTV achieves state-of-the-art performance across six standard metrics spanning video quality, text-video consistency, and audio-video alignment. Project page: this https URL. 

---
# Reparameterized LLM Training via Orthogonal Equivalence Transformation 

**Authors**: Zeju Qiu, Simon Buchholz, Tim Z. Xiao, Maximilian Dax, Bernhard Schölkopf, Weiyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08001)  

**Abstract**: While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs. 

---
# Thinking vs. Doing: Agents that Reason by Scaling Test-Time Interaction 

**Authors**: Junhong Shen, Hao Bai, Lunjun Zhang, Yifei Zhou, Amrith Setlur, Shengbang Tong, Diego Caples, Nan Jiang, Tong Zhang, Ameet Talwalkar, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07976)  

**Abstract**: The current paradigm of test-time scaling relies on generating long reasoning traces ("thinking" more) before producing a response. In agent problems that require interaction, this can be done by generating thinking traces before acting in the world. However, this process does not allow agents to acquire new information from the environment or adapt their behavior over time. In this work, we propose to scale test-time interaction, an untapped dimension of test-time scaling that increases the agent's interaction horizon to enable running rich behaviors such as exploration, backtracking, and dynamic re-planning within a single rollout. To demonstrate the promise of this scaling dimension, we study the domain of web agents. We first show that even prompting-based interaction scaling without any training can improve task success on web benchmarks non-trivially. Building on this, we introduce TTI (Test-Time Interaction), a curriculum-based online reinforcement learning (RL) approach that trains agents by adaptively adjusting their rollout lengths. Using a Gemma 3 12B model, TTI produces state-of-the-art open-source, open-data web agents on WebVoyager and WebArena benchmarks. We further show that TTI enables agents to balance exploration and exploitation adaptively. Our results establish interaction scaling as a powerful, complementary axis to scaling per-step compute, offering new avenues for training adaptive agents. 

---
# HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization 

**Authors**: Hongzheng Chen, Yingheng Wang, Yaohui Cai, Hins Hu, Jiajie Li, Shirley Huang, Chenhui Deng, Rongjian Liang, Shufeng Kong, Haoxing Ren, Samitha Samaranayake, Carla P. Gomes, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07972)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains. 

---
# SlideCoder: Layout-aware RAG-enhanced Hierarchical Slide Generation from Design 

**Authors**: Wenxin Tang, Jingyu Xiao, Wenxuan Jiang, Xi Xiao, Yuhang Wang, Xuxin Tang, Qing Li, Yuehe Ma, Junliang Liu, Shisong Tang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07964)  

**Abstract**: Manual slide creation is labor-intensive and requires expert prior knowledge. Existing natural language-based LLM generation methods struggle to capture the visual and structural nuances of slide designs. To address this, we formalize the Reference Image to Slide Generation task and propose Slide2Code, the first benchmark with difficulty-tiered samples based on a novel Slide Complexity Metric. We introduce SlideCoder, a layout-aware, retrieval-augmented framework for generating editable slides from reference images. SlideCoder integrates a Color Gradient-based Segmentation algorithm and a Hierarchical Retrieval-Augmented Generation method to decompose complex tasks and enhance code generation. We also release SlideMaster, a 7B open-source model fine-tuned with improved reverse-engineered data. Experiments show that SlideCoder outperforms state-of-the-art baselines by up to 40.5 points, demonstrating strong performance across layout fidelity, execution accuracy, and visual consistency. Our code is available at this https URL. 

---
# Correlated Errors in Large Language Models 

**Authors**: Elliot Kim, Avi Garg, Kenny Peng, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07962)  

**Abstract**: Diversity in training data, architecture, and providers is assumed to mitigate homogeneity in LLMs. However, we lack empirical evidence on whether different LLMs differ meaningfully. We conduct a large-scale empirical evaluation on over 350 LLMs overall, using two popular leaderboards and a resume-screening task. We find substantial correlation in model errors -- on one leaderboard dataset, models agree 60% of the time when both models err. We identify factors driving model correlation, including shared architectures and providers. Crucially, however, larger and more accurate models have highly correlated errors, even with distinct architectures and providers. Finally, we show the effects of correlation in two downstream tasks: LLM-as-judge evaluation and hiring -- the latter reflecting theoretical predictions regarding algorithmic monoculture. 

---
# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models 

**Authors**: Peiyan Li, Yixiang Chen, Hongtao Wu, Xiao Ma, Xiangnan Wu, Yan Huang, Liang Wang, Tao Kong, Tieniu Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07961)  

**Abstract**: Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:this https URL 

---
# ProtocolLLM: RTL Benchmark for SystemVerilog Generation of Communication Protocols 

**Authors**: Arnav Sheth, Ivaxi Sheth, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2506.07945)  

**Abstract**: Recent advances in Large Language Models (LLMs) have shown promising capabilities in generating code for general-purpose programming languages. In contrast, their applicability for hardware description languages, particularly for generating synthesizable and functionally correct designs, remains significantly underexplored. HDLs such as SystemVerilog are logic-oriented and demand strict adherence to timing semantics, concurrency, and synthesizability constraints. Moreover, HDL-based design flows encompass a broad set of tasks beyond structural code generation, including testbench development, assertion-based verification, timing closure, and protocol-level integration for on-chip communication. The objective of our paper is to analyze the capabilities of state-of-the-art LLMs in generating SystemVerilog implementations of standard communication protocols, a core component of embedded and System-on-Chip (SoC) architectures. This paper introduces the first benchmark suite targeting four widely used protocols: SPI, I2C, UART, and AXI. We define code generation tasks that capture varying levels of design abstraction and prompt specificity. The generated designs are assessed for syntactic correctness, synthesizability, and functional fidelity via waveform simulation and test benches. 

---
# Decoupling the Image Perception and Multimodal Reasoning for Reasoning Segmentation with Digital Twin Representations 

**Authors**: Yizhen Li, Dell Zhang, Xuelong Li, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07943)  

**Abstract**: Reasoning Segmentation (RS) is a multimodal vision-text task that requires segmenting objects based on implicit text queries, demanding both precise visual perception and vision-text reasoning capabilities. Current RS approaches rely on fine-tuning vision-language models (VLMs) for both perception and reasoning, but their tokenization of images fundamentally disrupts continuous spatial relationships between objects. We introduce DTwinSeger, a novel RS approach that leverages Digital Twin (DT) representation as an intermediate layer to decouple perception from reasoning. Innovatively, DTwinSeger reformulates RS as a two-stage process, where the first transforms the image into a structured DT representation that preserves spatial relationships and semantic properties and then employs a Large Language Model (LLM) to perform explicit reasoning over this representation to identify target objects. We propose a supervised fine-tuning method specifically for LLM with DT representation, together with a corresponding fine-tuning dataset Seg-DT, to enhance the LLM's reasoning capabilities with DT representations. Experiments show that our method can achieve state-of-the-art performance on two image RS benchmarks and three image referring segmentation benchmarks. It yields that DT representation functions as an effective bridge between vision and text, enabling complex multimodal reasoning tasks to be accomplished solely with an LLM. 

---
# Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models 

**Authors**: Chengyue Huang, Yuchen Zhu, Sichen Zhu, Jingyun Xiao, Moises Andrade, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.07936)  

**Abstract**: Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics -- such as copying or majority voting -- rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with Reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, suggesting that current VLMs do not effectively utilize demonstration-level information as intended in MM-ICL. 

---
# Diffusion of Responsibility in Collective Decision Making 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07935)  

**Abstract**: The term "diffusion of responsibility'' refers to situations in which multiple agents share responsibility for an outcome, obscuring individual accountability. This paper examines this frequently undesirable phenomenon in the context of collective decision-making mechanisms.
The work shows that if a decision is made by two agents, then the only way to avoid diffusion of responsibility is for one agent to act as a "dictator'', making the decision unilaterally. In scenarios with more than two agents, any diffusion-free mechanism is an "elected dictatorship'' where the agents elect a single agent to make a unilateral decision.
The technical results are obtained by defining a bisimulation of decision-making mechanisms, proving that bisimulation preserves responsibility-related properties, and establishing the results for a smallest bisimular mechanism. 

---
# Uncovering the Functional Roles of Nonlinearity in Memory 

**Authors**: Manuel Brenner, Georgia Koppe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07919)  

**Abstract**: Memory and long-range temporal processing are core requirements for sequence modeling tasks across natural language processing, time-series forecasting, speech recognition, and control. While nonlinear recurrence has long been viewed as essential for enabling such mechanisms, recent work suggests that linear dynamics may often suffice. In this study, we go beyond performance comparisons to systematically dissect the functional role of nonlinearity in recurrent networks--identifying both when it is computationally necessary, and what mechanisms it enables. We use Almost Linear Recurrent Neural Networks (AL-RNNs), which allow fine-grained control over nonlinearity, as both a flexible modeling tool and a probe into the internal mechanisms of memory. Across a range of classic sequence modeling tasks and a real-world stimulus selection task, we find that minimal nonlinearity is not only sufficient but often optimal, yielding models that are simpler, more robust, and more interpretable than their fully nonlinear or linear counterparts. Our results provide a principled framework for selectively introducing nonlinearity, bridging dynamical systems theory with the functional demands of long-range memory and structured computation in recurrent neural networks, with implications for both artificial and biological neural systems. 

---
# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces 

**Authors**: Kevin Rojas, Yuchen Zhu, Sichen Zhu, Felix X.-F. Ye, Molei Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07903)  

**Abstract**: Diffusion models have demonstrated remarkable performance in generating unimodal data across various tasks, including image, video, and text generation. On the contrary, the joint generation of multimodal data through diffusion models is still in the early stages of exploration. Existing approaches heavily rely on external preprocessing protocols, such as tokenizers and variational autoencoders, to harmonize varied data representations into a unified, unimodal format. This process heavily demands the high accuracy of encoders and decoders, which can be problematic for applications with limited data. To lift this restriction, we propose a novel framework for building multimodal diffusion models on arbitrary state spaces, enabling native generation of coupled data across different modalities. By introducing an innovative decoupled noise schedule for each modality, we enable both unconditional and modality-conditioned generation within a single model simultaneously. We empirically validate our approach for text-image generation and mixed-type tabular data synthesis, demonstrating that it achieves competitive performance. 

---
# MiniCPM4: Ultra-Efficient LLMs on End Devices 

**Authors**: MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengdan Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Yukun Yan, Jiarui Yuan, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Ge Zhou, Jie Zhou, Wei Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07900)  

**Abstract**: This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose this http URL that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability. 

---
# GaussianVAE: Adaptive Learning Dynamics of 3D Gaussians for High-Fidelity Super-Resolution 

**Authors**: Shuja Khalid, Mohamed Ibrahim, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07897)  

**Abstract**: We present a novel approach for enhancing the resolution and geometric fidelity of 3D Gaussian Splatting (3DGS) beyond native training resolution. Current 3DGS methods are fundamentally limited by their input resolution, producing reconstructions that cannot extrapolate finer details than are present in the training views. Our work breaks this limitation through a lightweight generative model that predicts and refines additional 3D Gaussians where needed most. The key innovation is our Hessian-assisted sampling strategy, which intelligently identifies regions that are likely to benefit from densification, ensuring computational efficiency. Unlike computationally intensive GANs or diffusion approaches, our method operates in real-time (0.015s per inference on a single consumer-grade GPU), making it practical for interactive applications. Comprehensive experiments demonstrate significant improvements in both geometric accuracy and rendering quality compared to state-of-the-art methods, establishing a new paradigm for resolution-free 3D scene enhancement. 

---
# Diffusion Counterfactual Generation with Semantic Abduction 

**Authors**: Rajat Rasal, Avinash Kori, Fabio De Sousa Ribeiro, Tian Xia, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.07883)  

**Abstract**: Counterfactual image generation presents significant challenges, including preserving identity, maintaining perceptual quality, and ensuring faithfulness to an underlying causal model. While existing auto-encoding frameworks admit semantic latent spaces which can be manipulated for causal control, they struggle with scalability and fidelity. Advancements in diffusion models present opportunities for improving counterfactual image editing, having demonstrated state-of-the-art visual quality, human-aligned perception and representation learning capabilities. Here, we present a suite of diffusion-based causal mechanisms, introducing the notions of spatial, semantic and dynamic abduction. We propose a general framework that integrates semantic representations into diffusion models through the lens of Pearlian causality to edit images via a counterfactual reasoning process. To our knowledge, this is the first work to consider high-level semantic identity preservation for diffusion counterfactuals and to demonstrate how semantic control enables principled trade-offs between faithful causal control and identity preservation. 

---
# FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity 

**Authors**: Jinxi Li, Ziyang Song, Siyuan Zhou, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07865)  

**Abstract**: In this paper, we aim to model 3D scene geometry, appearance, and the underlying physics purely from multi-view videos. By applying various governing PDEs as PINN losses or incorporating physics simulation into neural networks, existing works often fail to learn complex physical motions at boundaries or require object priors such as masks or types. In this paper, we propose FreeGave to learn the physics of complex dynamic 3D scenes without needing any object priors. The key to our approach is to introduce a physics code followed by a carefully designed divergence-free module for estimating a per-Gaussian velocity field, without relying on the inefficient PINN losses. Extensive experiments on three public datasets and a newly collected challenging real-world dataset demonstrate the superior performance of our method for future frame extrapolation and motion segmentation. Most notably, our investigation into the learned physics codes reveals that they truly learn meaningful 3D physical motion patterns in the absence of any human labels in training. 

---
# Lightweight Sequential Transformers for Blood Glucose Level Prediction in Type-1 Diabetes 

**Authors**: Mirko Paolo Barbato, Giorgia Rigamonti, Davide Marelli, Paolo Napoletano  

**Link**: [PDF](https://arxiv.org/pdf/2506.07864)  

**Abstract**: Type 1 Diabetes (T1D) affects millions worldwide, requiring continuous monitoring to prevent severe hypo- and hyperglycemic events. While continuous glucose monitoring has improved blood glucose management, deploying predictive models on wearable devices remains challenging due to computational and memory constraints. To address this, we propose a novel Lightweight Sequential Transformer model designed for blood glucose prediction in T1D. By integrating the strengths of Transformers' attention mechanisms and the sequential processing of recurrent neural networks, our architecture captures long-term dependencies while maintaining computational efficiency. The model is optimized for deployment on resource-constrained edge devices and incorporates a balanced loss function to handle the inherent data imbalance in hypo- and hyperglycemic events. Experiments on two benchmark datasets, OhioT1DM and DiaTrend, demonstrate that the proposed model outperforms state-of-the-art methods in predicting glucose levels and detecting adverse events. This work fills the gap between high-performance modeling and practical deployment, providing a reliable and efficient T1D management solution. 

---
# Fairness Overfitting in Machine Learning: An Information-Theoretic Perspective 

**Authors**: Firas Laakom, Haobo Chen, Jürgen Schmidhuber, Yuheng Bu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07861)  

**Abstract**: Despite substantial progress in promoting fairness in high-stake applications using machine learning models, existing methods often modify the training process, such as through regularizers or other interventions, but lack formal guarantees that fairness achieved during training will generalize to unseen data. Although overfitting with respect to prediction performance has been extensively studied, overfitting in terms of fairness loss has received far less attention. This paper proposes a theoretical framework for analyzing fairness generalization error through an information-theoretic lens. Our novel bounding technique is based on Efron-Stein inequality, which allows us to derive tight information-theoretic fairness generalization bounds with both Mutual Information (MI) and Conditional Mutual Information (CMI). Our empirical results validate the tightness and practical relevance of these bounds across diverse fairness-aware learning algorithms. Our framework offers valuable insights to guide the design of algorithms improving fairness generalization. 

---
# LogoSP: Local-global Grouping of Superpoints for Unsupervised Semantic Segmentation of 3D Point Clouds 

**Authors**: Zihui Zhang, Weisheng Dai, Hongtao Wen, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07857)  

**Abstract**: We study the problem of unsupervised 3D semantic segmentation on raw point clouds without needing human labels in training. Existing methods usually formulate this problem into learning per-point local features followed by a simple grouping strategy, lacking the ability to discover additional and possibly richer semantic priors beyond local features. In this paper, we introduce LogoSP to learn 3D semantics from both local and global point features. The key to our approach is to discover 3D semantic information by grouping superpoints according to their global patterns in the frequency domain, thus generating highly accurate semantic pseudo-labels for training a segmentation network. Extensive experiments on two indoor and an outdoor datasets show that our LogoSP surpasses all existing unsupervised methods by large margins, achieving the state-of-the-art performance for unsupervised 3D semantic segmentation. Notably, our investigation into the learned global patterns reveals that they truly represent meaningful 3D semantics in the absence of human labels during training. 

---
# Residual Reweighted Conformal Prediction for Graph Neural Networks 

**Authors**: Zheng Zhang, Jie Bao, Zhixin Zhou, Nicolo Colombo, Lixin Cheng, Rui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07854)  

**Abstract**: Graph Neural Networks (GNNs) excel at modeling relational data but face significant challenges in high-stakes domains due to unquantified uncertainty. Conformal prediction (CP) offers statistical coverage guarantees, but existing methods often produce overly conservative prediction intervals that fail to account for graph heteroscedasticity and structural biases. While residual reweighting CP variants address some of these limitations, they neglect graph topology, cluster-specific uncertainties, and risk data leakage by reusing training sets. To address these issues, we propose Residual Reweighted GNN (RR-GNN), a framework designed to generate minimal prediction sets with provable marginal coverage guarantees.
RR-GNN introduces three major innovations to enhance prediction performance. First, it employs Graph-Structured Mondrian CP to partition nodes or edges into communities based on topological features, ensuring cluster-conditional coverage that reflects heterogeneity. Second, it uses Residual-Adaptive Nonconformity Scores by training a secondary GNN on a held-out calibration set to estimate task-specific residuals, dynamically adjusting prediction intervals according to node or edge uncertainty. Third, it adopts a Cross-Training Protocol, which alternates the optimization of the primary GNN and the residual predictor to prevent information leakage while maintaining graph dependencies. We validate RR-GNN on 15 real-world graphs across diverse tasks, including node classification, regression, and edge weight prediction. Compared to CP baselines, RR-GNN achieves improved efficiency over state-of-the-art methods, with no loss of coverage. 

---
# PolyVivid: Vivid Multi-Subject Video Generation with Cross-Modal Interaction and Enhancement 

**Authors**: Teng Hu, Zhentao Yu, Zhengguang Zhou, Jiangning Zhang, Yuan Zhou, Qinglin Lu, Ran Yi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07848)  

**Abstract**: Despite recent advances in video generation, existing models still lack fine-grained controllability, especially for multi-subject customization with consistent identity and interaction. In this paper, we propose PolyVivid, a multi-subject video customization framework that enables flexible and identity-consistent generation. To establish accurate correspondences between subject images and textual entities, we design a VLLM-based text-image fusion module that embeds visual identities into the textual space for precise grounding. To further enhance identity preservation and subject interaction, we propose a 3D-RoPE-based enhancement module that enables structured bidirectional fusion between text and image embeddings. Moreover, we develop an attention-inherited identity injection module to effectively inject fused identity features into the video generation process, mitigating identity drift. Finally, we construct an MLLM-based data pipeline that combines MLLM-based grounding, segmentation, and a clique-based subject consolidation strategy to produce high-quality multi-subject data, effectively enhancing subject distinction and reducing ambiguity in downstream video generation. Extensive experiments demonstrate that PolyVivid achieves superior performance in identity fidelity, video realism, and subject alignment, outperforming existing open-source and commercial baselines. 

---
# Diffusion models under low-noise regime 

**Authors**: Elizabeth Pavlova, Xue-Xin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.07841)  

**Abstract**: Recent work on diffusion models proposed that they operate in two regimes: memorization, in which models reproduce their training data, and generalization, in which they generate novel samples. While this has been tested in high-noise settings, the behavior of diffusion models as effective denoisers when the corruption level is small remains unclear. To address this gap, we systematically investigated the behavior of diffusion models under low-noise diffusion dynamics, with implications for model robustness and interpretability. Using (i) CelebA subsets of varying sample sizes and (ii) analytic Gaussian mixture benchmarks, we reveal that models trained on disjoint data diverge near the data manifold even when their high-noise outputs converge. We quantify how training set size, data geometry, and model objective choice shape denoising trajectories and affect score accuracy, providing insights into how these models actually learn representations of data distributions. This work starts to address gaps in our understanding of generative model reliability in practical applications where small perturbations are common. 

---
# Are Trees Really Green? A Detection Approach of IoT Malware Attacks 

**Authors**: Silvia Lucia Sanna, Diego Soi, Davide Maiorca, Giorgio Giacinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.07836)  

**Abstract**: Nowadays, the Internet of Things (IoT) is widely employed, and its usage is growing exponentially because it facilitates remote monitoring, predictive maintenance, and data-driven decision making, especially in the healthcare and industrial sectors. However, IoT devices remain vulnerable due to their resource constraints and difficulty in applying security patches. Consequently, various cybersecurity attacks are reported daily, such as Denial of Service, particularly in IoT-driven solutions. Most attack detection methodologies are based on Machine Learning (ML) techniques, which can detect attack patterns. However, the focus is more on identification rather than considering the impact of ML algorithms on computational resources. This paper proposes a green methodology to identify IoT malware networking attacks based on flow privacy-preserving statistical features. In particular, the hyperparameters of three tree-based models -- Decision Trees, Random Forest and Extra-Trees -- are optimized based on energy consumption and test-time performance in terms of Matthew's Correlation Coefficient. Our results show that models maintain high performance and detection accuracy while consistently reducing power usage in terms of watt-hours (Wh). This suggests that on-premise ML-based Intrusion Detection Systems are suitable for IoT and other resource-constrained devices. 

---
# Improving large language models with concept-aware fine-tuning 

**Authors**: Michael K. Chen, Xikun Zhang, Jiaxing Huang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07833)  

**Abstract**: Large language models (LLMs) have become the cornerstone of modern AI. However, the existing paradigm of next-token prediction fundamentally limits their ability to form coherent, high-level concepts, making it a critical barrier to human-like understanding and reasoning. Take the phrase "ribonucleic acid" as an example: an LLM will first decompose it into tokens, i.e., artificial text fragments ("rib", "on", ...), then learn each token sequentially, rather than grasping the phrase as a unified, coherent semantic entity. This fragmented representation hinders deeper conceptual understanding and, ultimately, the development of truly intelligent systems. In response, we introduce Concept-Aware Fine-Tuning (CAFT), a novel multi-token training method that redefines how LLMs are fine-tuned. By enabling the learning of sequences that span multiple tokens, this method fosters stronger concept-aware learning. Our experiments demonstrate significant improvements compared to conventional next-token finetuning methods across diverse tasks, including traditional applications like text summarization and domain-specific ones like de novo protein design. Multi-token prediction was previously only possible in the prohibitively expensive pretraining phase; CAFT, to our knowledge, is the first to bring the multi-token setting to the post-training phase, thus effectively democratizing its benefits for the broader community of practitioners and researchers. Finally, the unexpected effectiveness of our proposed method suggests wider implications for the machine learning research community. All code and data are available at this https URL 

---
# Decentralizing Multi-Agent Reinforcement Learning with Temporal Causal Information 

**Authors**: Jan Corazza, Hadi Partovi Aria, Hyohun Kim, Daniel Neider, Zhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07829)  

**Abstract**: Reinforcement learning (RL) algorithms can find an optimal policy for a single agent to accomplish a particular task. However, many real-world problems require multiple agents to collaborate in order to achieve a common goal. For example, a robot executing a task in a warehouse may require the assistance of a drone to retrieve items from high shelves. In Decentralized Multi-Agent RL (DMARL), agents learn independently and then combine their policies at execution time, but often must satisfy constraints on compatibility of local policies to ensure that they can achieve the global task when combined. In this paper, we study how providing high-level symbolic knowledge to agents can help address unique challenges of this setting, such as privacy constraints, communication limitations, and performance concerns. In particular, we extend the formal tools used to check the compatibility of local policies with the team task, making decentralized training with theoretical guarantees usable in more scenarios. Furthermore, we empirically demonstrate that symbolic knowledge about the temporal evolution of events in the environment can significantly expedite the learning process in DMARL. 

---
# Accelerating Diffusion Models in Offline RL via Reward-Aware Consistency Trajectory Distillation 

**Authors**: Xintong Duan, Yutong He, Fahim Tajwar, Ruslan Salakhutdinov, J. Zico Kolter, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2506.07822)  

**Abstract**: Although diffusion models have achieved strong results in decision-making tasks, their slow inference speed remains a key limitation. While the consistency model offers a potential solution, its applications to decision-making often struggle with suboptimal demonstrations or rely on complex concurrent training of multiple networks. In this work, we propose a novel approach to consistency distillation for offline reinforcement learning that directly incorporates reward optimization into the distillation process. Our method enables single-step generation while maintaining higher performance and simpler training. Empirical evaluations on the Gym MuJoCo benchmarks and long horizon planning demonstrate that our approach can achieve an 8.7% improvement over previous state-of-the-art while offering up to 142x speedup over diffusion counterparts in inference time. 

---
# Self-Cascaded Diffusion Models for Arbitrary-Scale Image Super-Resolution 

**Authors**: Junseo Bang, Joonhee Lee, Kyeonghyun Lee, Haechang Lee, Dong Un Kang, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07813)  

**Abstract**: Arbitrary-scale image super-resolution aims to upsample images to any desired resolution, offering greater flexibility than traditional fixed-scale super-resolution. Recent approaches in this domain utilize regression-based or generative models, but many of them are a single-stage upsampling process, which may be challenging to learn across a wide, continuous distribution of scaling factors. Progressive upsampling strategies have shown promise in mitigating this issue, yet their integration with diffusion models for flexible upscaling remains underexplored. Here, we present CasArbi, a novel self-cascaded diffusion framework for arbitrary-scale image super-resolution. CasArbi meets the varying scaling demands by breaking them down into smaller sequential factors and progressively enhancing the image resolution at each step with seamless transitions for arbitrary scales. Our novel coordinate-guided residual diffusion model allows for the learning of continuous image representations while enabling efficient diffusion sampling. Extensive experiments demonstrate that our CasArbi outperforms prior arts in both perceptual and distortion performance metrics across diverse arbitrary-scale super-resolution benchmarks. 

---
# Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability 

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07804)  

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at this https URL. 

---
# MultiMatch: Multihead Consistency Regularization Matching for Semi-Supervised Text Classification 

**Authors**: Iustin Sirbu, Robert-Adrian Popovici, Cornelia Caragea, Stefan Trausan-Matu, Traian Rebedea  

**Link**: [PDF](https://arxiv.org/pdf/2506.07801)  

**Abstract**: We introduce MultiMatch, a novel semi-supervised learning (SSL) algorithm combining the paradigms of co-training and consistency regularization with pseudo-labeling. At its core, MultiMatch features a three-fold pseudo-label weighting module designed for three key purposes: selecting and filtering pseudo-labels based on head agreement and model confidence, and weighting them according to the perceived classification difficulty. This novel module enhances and unifies three existing techniques -- heads agreement from Multihead Co-training, self-adaptive thresholds from FreeMatch, and Average Pseudo-Margins from MarginMatch -- resulting in a holistic approach that improves robustness and performance in SSL settings. Experimental results on benchmark datasets highlight the superior performance of MultiMatch, achieving state-of-the-art results on 9 out of 10 setups from 5 natural language processing datasets and ranking first according to the Friedman test among 19 methods. Furthermore, MultiMatch demonstrates exceptional robustness in highly imbalanced settings, outperforming the second-best approach by 3.26% -- and data imbalance is a key factor for many text classification tasks. 

---
# Re-ranking Reasoning Context with Tree Search Makes Large Vision-Language Models Stronger 

**Authors**: Qi Yang, Chenghao Zhang, Lubin Fan, Kun Ding, Jieping Ye, Shiming Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07785)  

**Abstract**: Recent advancements in Large Vision Language Models (LVLMs) have significantly improved performance in Visual Question Answering (VQA) tasks through multimodal Retrieval-Augmented Generation (RAG). However, existing methods still face challenges, such as the scarcity of knowledge with reasoning examples and erratic responses from retrieved knowledge. To address these issues, in this study, we propose a multimodal RAG framework, termed RCTS, which enhances LVLMs by constructing a Reasoning Context-enriched knowledge base and a Tree Search re-ranking method. Specifically, we introduce a self-consistent evaluation mechanism to enrich the knowledge base with intrinsic reasoning patterns. We further propose a Monte Carlo Tree Search with Heuristic Rewards (MCTS-HR) to prioritize the most relevant examples. This ensures that LVLMs can leverage high-quality contextual reasoning for better and more consistent responses. Extensive experiments demonstrate that our framework achieves state-of-the-art performance on multiple VQA datasets, significantly outperforming In-Context Learning (ICL) and Vanilla-RAG methods. It highlights the effectiveness of our knowledge base and re-ranking method in improving LVLMs. Our code is available at this https URL. 

---
# Comparing Credit Risk Estimates in the Gen-AI Era 

**Authors**: Nicola Lavecchia, Sid Fadanelli, Federico Ricciuti, Gennaro Aloe, Enrico Bagli, Pietro Giuffrida, Daniele Vergari  

**Link**: [PDF](https://arxiv.org/pdf/2506.07754)  

**Abstract**: Generative AI technologies have demonstrated significant potential across diverse applications. This study provides a comparative analysis of credit score modeling techniques, contrasting traditional approaches with those leveraging generative AI. Our findings reveal that current generative AI models fall short of matching the performance of traditional methods, regardless of the integration strategy employed. These results highlight the limitations in the current capabilities of generative AI for credit risk scoring, emphasizing the need for further research and development before the possibility of applying generative AI for this specific task, or equivalent ones. 

---
# Augmenting LLMs' Reasoning by Reinforcing Abstract Thinking 

**Authors**: Silin Gao, Antoine Bosselut, Samy Bengio, Emmanuel Abbe  

**Link**: [PDF](https://arxiv.org/pdf/2506.07751)  

**Abstract**: Recent studies have shown that large language models (LLMs), especially smaller ones, often lack robustness in their reasoning. I.e., they tend to experience performance drops when faced with distribution shifts, such as changes to numerical or nominal variables, or insertions of distracting clauses. A possible strategy to address this involves generating synthetic data to further "instantiate" reasoning problems on potential variations. In contrast, our approach focuses on "abstracting" reasoning problems. This not only helps counteract distribution shifts but also facilitates the connection to symbolic tools for deriving solutions. We find that this abstraction process is better acquired through reinforcement learning (RL) than just supervised fine-tuning, which often fails to produce faithful abstractions. Our method, AbstraL -- which promotes abstract reasoning in LLMs using RL on granular abstraction data -- significantly mitigates performance degradation on recent GSM perturbation benchmarks. 

---
# Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning 

**Authors**: Seungho Baek, Taegeon Park, Jongchan Park, Seungjun Oh, Yusung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07744)  

**Abstract**: Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: this https URL. 

---
# ArchiLense: A Framework for Quantitative Analysis of Architectural Styles Based on Vision Large Language Models 

**Authors**: Jing Zhong, Jun Yin, Peilin Li, Pengyu Zeng, Miao Zhang, Shuai Lu, Ran Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07739)  

**Abstract**: Architectural cultures across regions are characterized by stylistic diversity, shaped by historical, social, and technological contexts in addition to geograph-ical conditions. Understanding architectural styles requires the ability to describe and analyze the stylistic features of different architects from various regions through visual observations of architectural imagery. However, traditional studies of architectural culture have largely relied on subjective expert interpretations and historical literature reviews, often suffering from regional biases and limited ex-planatory scope. To address these challenges, this study proposes three core contributions: (1) We construct a professional architectural style dataset named ArchDiffBench, which comprises 1,765 high-quality architectural images and their corresponding style annotations, collected from different regions and historical periods. (2) We propose ArchiLense, an analytical framework grounded in Vision-Language Models and constructed using the ArchDiffBench dataset. By integrating ad-vanced computer vision techniques, deep learning, and machine learning algo-rithms, ArchiLense enables automatic recognition, comparison, and precise classi-fication of architectural imagery, producing descriptive language outputs that ar-ticulate stylistic differences. (3) Extensive evaluations show that ArchiLense achieves strong performance in architectural style recognition, with a 92.4% con-sistency rate with expert annotations and 84.5% classification accuracy, effec-tively capturing stylistic distinctions across images. The proposed approach transcends the subjectivity inherent in traditional analyses and offers a more objective and accurate perspective for comparative studies of architectural culture. 

---
# ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models 

**Authors**: Shadi Hamdan, Chonghao Sima, Zetong Yang, Hongyang Li, Fatma Güney  

**Link**: [PDF](https://arxiv.org/pdf/2506.07725)  

**Abstract**: How can we benefit from large models without sacrificing inference speed, a common dilemma in self-driving systems? A prevalent solution is a dual-system architecture, employing a small model for rapid, reactive decisions and a larger model for slower but more informative analyses. Existing dual-system designs often implement parallel architectures where inference is either directly conducted using the large model at each current frame or retrieved from previously stored inference results. However, these works still struggle to enable large models for a timely response to every online frame. Our key insight is to shift intensive computations of the current frame to previous time steps and perform a batch inference of multiple time steps to make large models respond promptly to each time step. To achieve the shifting, we introduce Efficiency through Thinking Ahead (ETA), an asynchronous system designed to: (1) propagate informative features from the past to the current frame using future predictions from the large model, (2) extract current frame features using a small model for real-time responsiveness, and (3) integrate these dual features via an action mask mechanism that emphasizes action-critical image regions. Evaluated on the Bench2Drive CARLA Leaderboard-v2 benchmark, ETA advances state-of-the-art performance by 8% with a driving score of 69.53 while maintaining a near-real-time inference speed at 50 ms. 

---
# Consistent Video Editing as Flow-Driven Image-to-Video Generation 

**Authors**: Ge Wang, Songlin Fan, Hangxu Liu, Quanjian Song, Hewei Wang, Jinfeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07713)  

**Abstract**: With the prosper of video diffusion models, down-stream applications like video editing have been significantly promoted without consuming much computational cost. One particular challenge in this task lies at the motion transfer process from the source video to the edited one, where it requires the consideration of the shape deformation in between, meanwhile maintaining the temporal consistency in the generated video sequence. However, existing methods fail to model complicated motion patterns for video editing, and are fundamentally limited to object replacement, where tasks with non-rigid object motions like multi-object and portrait editing are largely neglected. In this paper, we observe that optical flows offer a promising alternative in complex motion modeling, and present FlowV2V to re-investigate video editing as a task of flow-driven Image-to-Video (I2V) generation. Specifically, FlowV2V decomposes the entire pipeline into first-frame editing and conditional I2V generation, and simulates pseudo flow sequence that aligns with the deformed shape, thus ensuring the consistency during editing. Experimental results on DAVIS-EDIT with improvements of 13.67% and 50.66% on DOVER and warping error illustrate the superior temporal consistency and sample quality of FlowV2V compared to existing state-of-the-art ones. Furthermore, we conduct comprehensive ablation studies to analyze the internal functionalities of the first-frame paradigm and flow alignment in the proposed method. 

---
# NOVA3D: Normal Aligned Video Diffusion Model for Single Image to 3D Generation 

**Authors**: Yuxiao Yang, Peihao Li, Yuhong Zhang, Junzhe Lu, Xianglong He, Minghan Qin, Weitao Wang, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07698)  

**Abstract**: 3D AI-generated content (AIGC) has made it increasingly accessible for anyone to become a 3D content creator. While recent methods leverage Score Distillation Sampling to distill 3D objects from pretrained image diffusion models, they often suffer from inadequate 3D priors, leading to insufficient multi-view consistency. In this work, we introduce NOVA3D, an innovative single-image-to-3D generation framework. Our key insight lies in leveraging strong 3D priors from a pretrained video diffusion model and integrating geometric information during multi-view video fine-tuning. To facilitate information exchange between color and geometric domains, we propose the Geometry-Temporal Alignment (GTA) attention mechanism, thereby improving generalization and multi-view consistency. Moreover, we introduce the de-conflict geometry fusion algorithm, which improves texture fidelity by addressing multi-view inaccuracies and resolving discrepancies in pose alignment. Extensive experiments validate the superiority of NOVA3D over existing baselines. 

---
# GaRAGe: A Benchmark with Grounding Annotations for RAG Evaluation 

**Authors**: Ionut-Teodor Sorodoc, Leonardo F. R. Ribeiro, Rexhina Blloshmi, Christopher Davis, Adrià de Gispert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07671)  

**Abstract**: We present GaRAGe, a large RAG benchmark with human-curated long-form answers and annotations of each grounding passage, allowing a fine-grained evaluation of whether LLMs can identify relevant grounding when generating RAG answers. Our benchmark contains 2366 questions of diverse complexity, dynamism, and topics, and includes over 35K annotated passages retrieved from both private document sets and the Web, to reflect real-world RAG use cases. This makes it an ideal test bed to evaluate an LLM's ability to identify only the relevant information necessary to compose a response, or provide a deflective response when there is insufficient information. Evaluations of multiple state-of-the-art LLMs on GaRAGe show that the models tend to over-summarise rather than (a) ground their answers strictly on the annotated relevant passages (reaching at most a Relevance-Aware Factuality Score of 60%), or (b) deflect when no relevant grounding is available (reaching at most 31% true positive rate in deflections). The F1 in attribution to relevant sources is at most 58.9%, and we show that performance is particularly reduced when answering time-sensitive questions and when having to draw knowledge from sparser private grounding sources. 

---
# Synthesis by Design: Controlled Data Generation via Structural Guidance 

**Authors**: Lei Xu, Sirui Chen, Yuxuan Huang, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07664)  

**Abstract**: Mathematical reasoning remains challenging for LLMs due to complex logic and the need for precise computation. Existing methods enhance LLM reasoning by synthesizing datasets through problem rephrasing, but face issues with generation quality and problem complexity. To address this, we propose to extract structural information with generated problem-solving code from mathematical reasoning and guide data generation with structured solutions. Applied to MATH and GSM8K, our approach produces 39K problems with labeled intermediate steps and a 6.1K-problem benchmark of higher difficulty. Results on our benchmark show that model performance declines as reasoning length increases. Additionally, we conducted fine-tuning experiments using the proposed training data on a range of LLMs, and the results validate the effectiveness of our dataset. We hope the proposed method and dataset will contribute to future research in enhancing LLM reasoning capabilities. 

---
# FMaMIL: Frequency-Driven Mamba Multi-Instance Learning for Weakly Supervised Lesion Segmentation in Medical Images 

**Authors**: Hangbei Cheng, Xiaorong Dong, Xueyu Liu, Jianan Zhang, Xuetao Ma, Mingqiang Wei, Liansheng Wang, Junxin Chen, Yongfei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07652)  

**Abstract**: Accurate lesion segmentation in histopathology images is essential for diagnostic interpretation and quantitative analysis, yet it remains challenging due to the limited availability of costly pixel-level annotations. To address this, we propose FMaMIL, a novel two-stage framework for weakly supervised lesion segmentation based solely on image-level labels. In the first stage, a lightweight Mamba-based encoder is introduced to capture long-range dependencies across image patches under the MIL paradigm. To enhance spatial sensitivity and structural awareness, we design a learnable frequency-domain encoding module that supplements spatial-domain features with spectrum-based information. CAMs generated in this stage are used to guide segmentation training. In the second stage, we refine the initial pseudo labels via a CAM-guided soft-label supervision and a self-correction mechanism, enabling robust training even under label noise. Extensive experiments on both public and private histopathology datasets demonstrate that FMaMIL outperforms state-of-the-art weakly supervised methods without relying on pixel-level annotations, validating its effectiveness and potential for digital pathology applications. 

---
# LoRMA: Low-Rank Multiplicative Adaptation for LLMs 

**Authors**: Harsh Bihany, Shubham Patel, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07621)  

**Abstract**: Large Language Models have shown remarkable capabilities in the NLP domain. Their effectiveness can mainly be attributed to their ability to adapt to an array of downstream tasks. However, generally, full fine-tuning is a computationally expensive job. To mitigate this, many techniques have been developed that prime efficiency, a prominent one being Low-Rank Adaptation (LoRA). However, LoRA and its variants employ re-parametrized additive updates. In this paper, we propose Low-Rank Multiplicative Adaptation (LoRMA), which shifts the paradigm of additive updates to a richer space of matrix multiplicative transformations. We tackle challenges such as computational complexity and rank bottleneck of matrix multiplication by effectively re-ordering operations and introducing rank inflation strategies. We conduct extensive experiments to demonstrate the effectiveness of our approach in terms of various evaluation metrics. 

---
# PolitiSky24: U.S. Political Bluesky Dataset with User Stance Labels 

**Authors**: Peyman Rostami, Vahid Rahimzadeh, Ali Adibi, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2506.07606)  

**Abstract**: Stance detection identifies the viewpoint expressed in text toward a specific target, such as a political figure. While previous datasets have focused primarily on tweet-level stances from established platforms, user-level stance resources, especially on emerging platforms like Bluesky remain scarce. User-level stance detection provides a more holistic view by considering a user's complete posting history rather than isolated posts. We present the first stance detection dataset for the 2024 U.S. presidential election, collected from Bluesky and centered on Kamala Harris and Donald Trump. The dataset comprises 16,044 user-target stance pairs enriched with engagement metadata, interaction graphs, and user posting histories. PolitiSky24 was created using a carefully evaluated pipeline combining advanced information retrieval and large language models, which generates stance labels with supporting rationales and text spans for transparency. The labeling approach achieves 81\% accuracy with scalable LLMs. This resource addresses gaps in political stance analysis through its timeliness, open-data nature, and user-level perspective. The dataset is available at this https URL 

---
# SurgBench: A Unified Large-Scale Benchmark for Surgical Video Analysis 

**Authors**: Jianhui Wei, Zikai Xiao, Danyu Sun, Luqi Gong, Zongxin Yang, Zuozhu Liu, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07603)  

**Abstract**: Surgical video understanding is pivotal for enabling automated intraoperative decision-making, skill assessment, and postoperative quality improvement. However, progress in developing surgical video foundation models (FMs) remains hindered by the scarcity of large-scale, diverse datasets for pretraining and systematic evaluation. In this paper, we introduce \textbf{SurgBench}, a unified surgical video benchmarking framework comprising a pretraining dataset, \textbf{SurgBench-P}, and an evaluation benchmark, \textbf{SurgBench-E}. SurgBench offers extensive coverage of diverse surgical scenarios, with SurgBench-P encompassing 53 million frames across 22 surgical procedures and 11 specialties, and SurgBench-E providing robust evaluation across six categories (phase classification, camera motion, tool recognition, disease diagnosis, action classification, and organ detection) spanning 72 fine-grained tasks. Extensive experiments reveal that existing video FMs struggle to generalize across varied surgical video analysis tasks, whereas pretraining on SurgBench-P yields substantial performance improvements and superior cross-domain generalization to unseen procedures and modalities. Our dataset and code are available upon request. 

---
# SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding 

**Authors**: Nianbo Zeng, Haowen Hou, Fei Richard Yu, Si Shi, Ying Tiffany He  

**Link**: [PDF](https://arxiv.org/pdf/2506.07600)  

**Abstract**: Despite recent advances in retrieval-augmented generation (RAG) for video understanding, effectively understanding long-form video content remains underexplored due to the vast scale and high complexity of video data. Current RAG approaches typically segment videos into fixed-length chunks, which often disrupts the continuity of contextual information and fails to capture authentic scene boundaries. Inspired by the human ability to naturally organize continuous experiences into coherent scenes, we present SceneRAG, a unified framework that leverages large language models to segment videos into narrative-consistent scenes by processing ASR transcripts alongside temporal metadata. SceneRAG further sharpens these initial boundaries through lightweight heuristics and iterative correction. For each scene, the framework fuses information from both visual and textual modalities to extract entity relations and dynamically builds a knowledge graph, enabling robust multi-hop retrieval and generation that account for long-range dependencies. Experiments on the LongerVideos benchmark, featuring over 134 hours of diverse content, confirm that SceneRAG substantially outperforms prior baselines, achieving a win rate of up to 72.5 percent on generation tasks. 

---
# PrunePEFT: Iterative Hybrid Pruning for Parameter-Efficient Fine-tuning of LLMs 

**Authors**: Tongzhou Yu, Zhuhao Zhang, Guanghui Zhu, Shen Jiang, Meikang Qiu, Yihua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07587)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) methods have emerged as effective and promising approaches for fine-tuning pre-trained language models. Compared with Full parameter Fine-Tuning (FFT), PEFT achieved comparable task performance with a substantial reduction of trainable parameters, which largely saved the training and storage costs. However, using the PEFT method requires considering a vast design space, such as the type of PEFT modules and their insertion layers. Inadequate configurations can lead to sub-optimal results. Conventional solutions such as architectural search techniques, while effective, tend to introduce substantial additional overhead. In this paper, we propose a novel approach, PrunePEFT, which formulates the PEFT strategy search as a pruning problem and introduces a hybrid pruning strategy that capitalizes on the sensitivity of pruning methods to different PEFT modules. This method extends traditional pruning techniques by iteratively removing redundant or conflicting PEFT modules, thereby optimizing the fine-tuned configuration. By efficiently identifying the most relevant modules, our approach significantly reduces the computational burden typically associated with architectural search processes, making it a more scalable and efficient solution for fine-tuning large pre-trained models. 

---
# Beyond the Sentence: A Survey on Context-Aware Machine Translation with Large Language Models 

**Authors**: Ramakrishna Appicharla, Baban Gain, Santanu Pal, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07583)  

**Abstract**: Despite the popularity of the large language models (LLMs), their application to machine translation is relatively underexplored, especially in context-aware settings. This work presents a literature review of context-aware translation with LLMs. The existing works utilise prompting and fine-tuning approaches, with few focusing on automatic post-editing and creating translation agents for context-aware machine translation. We observed that the commercial LLMs (such as ChatGPT and Tower LLM) achieved better results than the open-source LLMs (such as Llama and Bloom LLMs), and prompt-based approaches serve as good baselines to assess the quality of translations. Finally, we present some interesting future directions to explore. 

---
# FedCGD: Collective Gradient Divergence Optimized Scheduling for Wireless Federated Learning 

**Authors**: Tan Chen, Jintao Yan, Yuxuan Sun, Sheng Zhou, Zhisheng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07581)  

**Abstract**: Federated learning (FL) is a promising paradigm for multiple devices to cooperatively train a model. When applied in wireless networks, two issues consistently affect the performance of FL, i.e., data heterogeneity of devices and limited bandwidth. Many papers have investigated device scheduling strategies considering the two issues. However, most of them recognize data heterogeneity as a property of individual devices. In this paper, we prove that the convergence speed of FL is affected by the sum of device-level and sample-level collective gradient divergence (CGD). The device-level CGD refers to the gradient divergence of the scheduled device group, instead of the sum of the individual device divergence. The sample-level CGD is statistically upper bounded by sampling variance, which is inversely proportional to the total number of samples scheduled for local update. To derive a tractable form of the device-level CGD, we further consider a classification problem and transform it into the weighted earth moving distance (WEMD) between the group distribution and the global distribution. Then we propose FedCGD algorithm to minimize the sum of multi-level CGDs by balancing WEMD and sampling variance, within polynomial time. Simulation shows that the proposed strategy increases classification accuracy on the CIFAR-10 dataset by up to 4.2\% while scheduling 41.8\% fewer devices, and flexibly switches between reducing WEMD and reducing sampling variance. 

---
# Denoising the Future: Top-p Distributions for Moving Through Time 

**Authors**: Florian Andreas Marwitz, Ralf Möller, Magnus Bender, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2506.07578)  

**Abstract**: Inference in dynamic probabilistic models is a complex task involving expensive operations. In particular, for Hidden Markov Models, the whole state space has to be enumerated for advancing in time. Even states with negligible probabilities are considered, resulting in computational inefficiency and increased noise due to the propagation of unlikely probability mass. We propose to denoise the future and speed up inference by using only the top-p states, i.e., the most probable states with accumulated probability p. We show that the error introduced by using only the top-p states is bound by p and the so-called minimal mixing rate of the underlying model. Moreover, in our empirical evaluation, we show that we can expect speedups of at least an order of magnitude, while the error in terms of total variation distance is below 0.09. 

---
# LLM-driven Indoor Scene Layout Generation via Scaled Human-aligned Data Synthesis and Multi-Stage Preference Optimization 

**Authors**: Yixuan Yang, Zhen Luo, Tongsheng Ding, Junru Lu, Mingqi Gao, Jinyu Yang, Victor Sanchez, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07570)  

**Abstract**: Automatic indoor layout generation has attracted increasing attention due to its potential in interior design, virtual environment construction, and embodied AI. Existing methods fall into two categories: prompt-driven approaches that leverage proprietary LLM services (e.g., GPT APIs) and learning-based methods trained on layout data upon diffusion-based models. Prompt-driven methods often suffer from spatial inconsistency and high computational costs, while learning-based methods are typically constrained by coarse relational graphs and limited datasets, restricting their generalization to diverse room categories. In this paper, we revisit LLM-based indoor layout generation and present 3D-SynthPlace, a large-scale dataset that combines synthetic layouts generated via a 'GPT synthesize, Human inspect' pipeline, upgraded from the 3D-Front dataset. 3D-SynthPlace contains nearly 17,000 scenes, covering four common room types -- bedroom, living room, kitchen, and bathroom -- enriched with diverse objects and high-level spatial annotations. We further introduce OptiScene, a strong open-source LLM optimized for indoor layout generation, fine-tuned based on our 3D-SynthPlace dataset through our two-stage training. For the warum-up stage I, we adopt supervised fine-tuning (SFT), which is taught to first generate high-level spatial descriptions then conditionally predict concrete object placements. For the reinforcing stage II, to better align the generated layouts with human design preferences, we apply multi-turn direct preference optimization (DPO), which significantly improving layout quality and generation success rates. Extensive experiments demonstrate that OptiScene outperforms traditional prompt-driven and learning-based baselines. Moreover, OptiScene shows promising potential in interactive tasks such as scene editing and robot navigation. 

---
# MoE-MLoRA for Multi-Domain CTR Prediction: Efficient Adaptation with Expert Specialization 

**Authors**: Ken Yagel, Eyal German, Aviel Ben Siman Tov  

**Link**: [PDF](https://arxiv.org/pdf/2506.07563)  

**Abstract**: Personalized recommendation systems must adapt to user interactions across different domains. Traditional approaches like MLoRA apply a single adaptation per domain but lack flexibility in handling diverse user behaviors. To address this, we propose MoE-MLoRA, a mixture-of-experts framework where each expert is first trained independently to specialize in its domain before a gating network is trained to weight their contributions dynamically. We evaluate MoE-MLoRA across eight CTR models on Movielens and Taobao, showing that it improves performance in large-scale, dynamic datasets (+1.45 Weighed-AUC in Taobao-20) but offers limited benefits in structured datasets with low domain diversity and sparsity. Further analysis of the number of experts per domain reveals that larger ensembles do not always improve performance, indicating the need for model-aware tuning. Our findings highlight the potential of expert-based architectures for multi-domain recommendation systems, demonstrating that task-aware specialization and adaptive gating can enhance predictive accuracy in complex environments. The implementation and code are available in our GitHub repository. 

---
# SELT: Self-Evaluation Tree Search for LLMs with Task Decomposition 

**Authors**: Mengsong Wu, Di Zhang, Yuqiang Li, Dongzhan Zhou, Wenliang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07557)  

**Abstract**: While Large Language Models (LLMs) have achieved remarkable success in a wide range of applications, their performance often degrades in complex reasoning tasks. In this work, we introduce SELT (Self-Evaluation LLM Tree Search), a novel framework that leverages a modified Monte Carlo Tree Search (MCTS) to enhance LLM reasoning without relying on external reward models. By redefining the Upper Confidence Bound scoring to align with intrinsic self-evaluation capabilities of LLMs and decomposing the inference process into atomic subtasks augmented with semantic clustering at each node, SELT effectively balances exploration and exploitation, reduces redundant reasoning paths, and mitigates hallucination. We validate our approach on challenging benchmarks, including the knowledge-based MMLU and the Tool Learning dataset Seal-Tools, where SELT achieves significant improvements in answer accuracy and reasoning robustness compared to baseline methods. Notably, our framework operates without task-specific fine-tuning, demonstrating strong generalizability across diverse reasoning tasks. Relevant results and code are available at this https URL . 

---
# Synthesize Privacy-Preserving High-Resolution Images via Private Textual Intermediaries 

**Authors**: Haoxiang Wang, Zinan Lin, Da Yu, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07555)  

**Abstract**: Generating high fidelity, differentially private (DP) synthetic images offers a promising route to share and analyze sensitive visual data without compromising individual privacy. However, existing DP image synthesis methods struggle to produce high resolution outputs that faithfully capture the structure of the original data. In this paper, we introduce a novel method, referred to as Synthesis via Private Textual Intermediaries (SPTI), that can generate high resolution DP images with easy adoption. The key idea is to shift the challenge of DP image synthesis from the image domain to the text domain by leveraging state of the art DP text generation methods. SPTI first summarizes each private image into a concise textual description using image to text models, then applies a modified Private Evolution algorithm to generate DP text, and finally reconstructs images using text to image models. Notably, SPTI requires no model training, only inference with off the shelf models. Given a private dataset, SPTI produces synthetic images of substantially higher quality than prior DP approaches. On the LSUN Bedroom dataset, SPTI attains an FID less than or equal to 26.71 under epsilon equal to 1.0, improving over Private Evolution FID of 40.36. Similarly, on MM CelebA HQ, SPTI achieves an FID less than or equal to 33.27 at epsilon equal to 1.0, compared to 57.01 from DP fine tuning baselines. Overall, our results demonstrate that Synthesis via Private Textual Intermediaries provides a resource efficient and proprietary model compatible framework for generating high resolution DP synthetic images, greatly expanding access to private visual datasets. 

---
# ChemAgent: Enhancing LLMs for Chemistry and Materials Science through Tree-Search Based Tool Learning 

**Authors**: Mengsong Wu, YaFei Wang, Yidong Ming, Yuqi An, Yuwei Wan, Wenliang Chen, Binbin Lin, Yuqiang Li, Tong Xie, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.07551)  

**Abstract**: Large language models (LLMs) have recently demonstrated promising capabilities in chemistry tasks while still facing challenges due to outdated pretraining knowledge and the difficulty of incorporating specialized chemical expertise. To address these issues, we propose an LLM-based agent that synergistically integrates 137 external chemical tools created ranging from basic information retrieval to complex reaction predictions, and a dataset curation pipeline to generate the dataset ChemToolBench that facilitates both effective tool selection and precise parameter filling during fine-tuning and evaluation. We introduce a Hierarchical Evolutionary Monte Carlo Tree Search (HE-MCTS) framework, enabling independent optimization of tool planning and execution. By leveraging self-generated data, our approach supports step-level fine-tuning (FT) of the policy model and training task-adaptive PRM and ORM that surpass GPT-4o. Experimental evaluations demonstrate that our approach significantly improves performance in Chemistry QA and discovery tasks, offering a robust solution to integrate specialized tools with LLMs for advanced chemical applications. All datasets and code are available at this https URL . 

---
# APTOS-2024 challenge report: Generation of synthetic 3D OCT images from fundus photographs 

**Authors**: Bowen Liu, Weiyi Zhang, Peranut Chotcomwongse, Xiaolan Chen, Ruoyu Chen, Pawin Pakaymaskul, Niracha Arjkongharn, Nattaporn Vongsa, Xuelian Cheng, Zongyuan Ge, Kun Huang, Xiaohui Li, Yiru Duan, Zhenbang Wang, BaoYe Xie, Qiang Chen, Huazhu Fu, Michael A. Mahr, Jiaqi Qu, Wangyiyang Chen, Shiye Wang, Yubo Tan, Yongjie Li, Mingguang He, Danli Shi, Paisan Ruamviboonsuk  

**Link**: [PDF](https://arxiv.org/pdf/2506.07542)  

**Abstract**: Optical Coherence Tomography (OCT) provides high-resolution, 3D, and non-invasive visualization of retinal layers in vivo, serving as a critical tool for lesion localization and disease diagnosis. However, its widespread adoption is limited by equipment costs and the need for specialized operators. In comparison, 2D color fundus photography offers faster acquisition and greater accessibility with less dependence on expensive devices. Although generative artificial intelligence has demonstrated promising results in medical image synthesis, translating 2D fundus images into 3D OCT images presents unique challenges due to inherent differences in data dimensionality and biological information between modalities. To advance generative models in the fundus-to-3D-OCT setting, the Asia Pacific Tele-Ophthalmology Society (APTOS-2024) organized a challenge titled Artificial Intelligence-based OCT Generation from Fundus Images. This paper details the challenge framework (referred to as APTOS-2024 Challenge), including: the benchmark dataset, evaluation methodology featuring two fidelity metrics-image-based distance (pixel-level OCT B-scan similarity) and video-based distance (semantic-level volumetric consistency), and analysis of top-performing solutions. The challenge attracted 342 participating teams, with 42 preliminary submissions and 9 finalists. Leading methodologies incorporated innovations in hybrid data preprocessing or augmentation (cross-modality collaborative paradigms), pre-training on external ophthalmic imaging datasets, integration of vision foundation models, and model architecture improvement. The APTOS-2024 Challenge is the first benchmark demonstrating the feasibility of fundus-to-3D-OCT synthesis as a potential solution for improving ophthalmic care accessibility in under-resourced healthcare settings, while helping to expedite medical research and clinical applications. 

---
# Domain Randomization for Object Detection in Manufacturing Applications using Synthetic Data: A Comprehensive Study 

**Authors**: Xiaomeng Zhu, Jacob Henningsson, Duruo Li, Pär Mårtensson, Lars Hanson, Mårten Björkman, Atsuto Maki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07539)  

**Abstract**: This paper addresses key aspects of domain randomization in generating synthetic data for manufacturing object detection applications. To this end, we present a comprehensive data generation pipeline that reflects different factors: object characteristics, background, illumination, camera settings, and post-processing. We also introduce the Synthetic Industrial Parts Object Detection dataset (SIP15-OD) consisting of 15 objects from three industrial use cases under varying environments as a test bed for the study, while also employing an industrial dataset publicly available for robotic applications. In our experiments, we present more abundant results and insights into the feasibility as well as challenges of sim-to-real object detection. In particular, we identified material properties, rendering methods, post-processing, and distractors as important factors. Our method, leveraging these, achieves top performance on the public dataset with Yolov8 models trained exclusively on synthetic data; mAP@50 scores of 96.4% for the robotics dataset, and 94.1%, 99.5%, and 95.3% across three of the SIP15-OD use cases, respectively. The results showcase the effectiveness of the proposed domain randomization, potentially covering the distribution close to real data for the applications. 

---
# IntenTest: Stress Testing for Intent Integrity in API-Calling LLM Agents 

**Authors**: Shiwei Feng, Xiangzhe Xu, Xuan Chen, Kaiyuan Zhang, Syed Yusuf Ahmed, Zian Su, Mingwei Zheng, Xiangyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07524)  

**Abstract**: LLM agents are increasingly deployed to automate real-world tasks by invoking APIs through natural language instructions. While powerful, they often suffer from misinterpretation of user intent, leading to the agent's actions that diverge from the user's intended goal, especially as external toolkits evolve. Traditional software testing assumes structured inputs and thus falls short in handling the ambiguity of natural language. We introduce IntenTest, an API-centric stress testing framework that systematically uncovers intent integrity violations in LLM agents. Unlike prior work focused on fixed benchmarks or adversarial inputs, IntenTest generates realistic tasks based on toolkits' documentation and applies targeted mutations to expose subtle agent errors while preserving user intent. To guide testing, we propose semantic partitioning, which organizes natural language tasks into meaningful categories based on toolkit API parameters and their equivalence classes. Within each partition, seed tasks are mutated and ranked by a lightweight predictor that estimates the likelihood of triggering agent errors. To enhance efficiency, IntenTest maintains a datatype-aware strategy memory that retrieves and adapts effective mutation patterns from past cases. Experiments on 80 toolkit APIs demonstrate that IntenTest effectively uncovers intent integrity violations, significantly outperforming baselines in both error-exposing rate and query efficiency. Moreover, IntenTest generalizes well to stronger target models using smaller LLMs for test generation, and adapts to evolving APIs across domains. 

---
# LeVo: High-Quality Song Generation with Multi-Preference Alignment 

**Authors**: Shun Lei, Yaoxun Xu, Zhiwei Lin, Huaicheng Zhang, Wei Tan, Hangting Chen, Jianwei Yu, Yixuan Zhang, Chenyu Yang, Haina Zhu, Shuai Wang, Zhiyong Wu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07520)  

**Abstract**: Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at this https URL. 

---
# Reinforcement Learning via Implicit Imitation Guidance 

**Authors**: Perry Dong, Alec M. Lessing, Annie S. Chen, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2506.07505)  

**Abstract**: We study the problem of sample efficient reinforcement learning, where prior data such as demonstrations are provided for initialization in lieu of a dense reward signal. A natural approach is to incorporate an imitation learning objective, either as regularization during training or to acquire a reference policy. However, imitation learning objectives can ultimately degrade long-term performance, as it does not directly align with reward maximization. In this work, we propose to use prior data solely for guiding exploration via noise added to the policy, sidestepping the need for explicit behavior cloning constraints. The key insight in our framework, Data-Guided Noise (DGN), is that demonstrations are most useful for identifying which actions should be explored, rather than forcing the policy to take certain actions. Our approach achieves up to 2-3x improvement over prior reinforcement learning from offline data methods across seven simulated continuous control tasks. 

---
# Graph-of-Causal Evolution: Challenging Chain-of-Model for Reasoning 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07501)  

**Abstract**: In view of the problem that each subchain in the chain-of-model (CoM) relies only on the information of the previous subchain and may lose long-range dependencies due to the causal mask blocking the global context flow between multi-level subchains, this work proposes a graph of causal evolution (GoCE). Its core principle is to map the implicit token representation into a differentiable and sparse causal adjacency matrix, then permeate causal constraints through each layer of calculation using causal-masked attention and causal-MoE. By combining intervention consistency loss test and self-evolution gate, the dynamic balance between causal structure learning and adaptive updating of transformer architecture is realized. The researcher built experimental environments in sandboxes built with Claude Sonnet 4, o4-mini-high, and DeepSeek R1 respectively with the transformer variant architecture introduced in GoCE. It is evaluated on publicly available datasets including CLUTRR, CLADDER, EX-FEVER, and CausalQA and compared with the baseline LLMs. The finding proves that GoCE strengthens the transformer's ability to capture long-range causal dependencies, while the ability to self-evolve is improved. It not only surpasses the design of CoM in terms of design principles, but also provides experience for future research on causal learning and continuous adaptive improvement. 

---
# CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization 

**Authors**: Dasol Hong, Wooju Lee, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2506.07484)  

**Abstract**: Prompt tuning, which adapts vision-language models by freezing model parameters and optimizing only the prompt, has proven effective for task-specific adaptations. The core challenge in prompt tuning is improving specialization for a specific task and generalization for unseen domains. However, frozen encoders often produce misaligned features, leading to confusion between classes and limiting specialization. To overcome this issue, we propose a confusion-aware loss (CoA-loss) that improves specialization by refining the decision boundaries between confusing classes. Additionally, we mathematically demonstrate that a mixture model can enhance generalization without compromising specialization. This is achieved using confidence-aware weights (CoA-weights), which adjust the weights of each prediction in the mixture model based on its confidence within the class domains. Extensive experiments show that CoCoA-Mix, a mixture model with CoA-loss and CoA-weights, outperforms state-of-the-art methods by enhancing specialization and generalization. Our code is publicly available at this https URL. 

---
# Premise Selection for a Lean Hammer 

**Authors**: Thomas Zhu, Joshua Clune, Jeremy Avigad, Albert Qiaochu Jiang, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2506.07477)  

**Abstract**: Neural methods are transforming automated reasoning for proof assistants, yet integrating these advances into practical verification workflows remains challenging. Hammers are tools that interface with external automatic theorem provers to automate tedious reasoning steps. They have dramatically improved productivity in proof assistants, but the Lean proof assistant still does not have a hammer despite its growing popularity. We present LeanHammer, the first end-to-end domain-general hammer for Lean, built on a novel neural premise selection system for a hammer in dependent type theory. Unlike existing Lean premise selectors, our approach dynamically adapts to user-specific contexts and combines with symbolic proof search and reconstruction to create a practical hammer. With comprehensive evaluations, we show that our premise selector enables LeanHammer to solve 21\% more goals relative to existing premise selectors, and generalize well to diverse domains. Our work bridges the gap between neural retrieval and symbolic reasoning, making formal verification more accessible to researchers and practitioners. 

---
# Ambiguity-Restrained Text-Video Representation Learning for Partially Relevant Video Retrieval 

**Authors**: CH Cho, WJ Moon, W Jun, MS Jung, JP Heo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07471)  

**Abstract**: Partially Relevant Video Retrieval~(PRVR) aims to retrieve a video where a specific segment is relevant to a given text query. Typical training processes of PRVR assume a one-to-one relationship where each text query is relevant to only one video. However, we point out the inherent ambiguity between text and video content based on their conceptual scope and propose a framework that incorporates this ambiguity into the model learning process. Specifically, we propose Ambiguity-Restrained representation Learning~(ARL) to address ambiguous text-video pairs. Initially, ARL detects ambiguous pairs based on two criteria: uncertainty and similarity. Uncertainty represents whether instances include commonly shared context across the dataset, while similarity indicates pair-wise semantic overlap. Then, with the detected ambiguous pairs, our ARL hierarchically learns the semantic relationship via multi-positive contrastive learning and dual triplet margin loss. Additionally, we delve into fine-grained relationships within the video instances. Unlike typical training at the text-video level, where pairwise information is provided, we address the inherent ambiguity within frames of the same untrimmed video, which often contains multiple contexts. This allows us to further enhance learning at the text-frame level. Lastly, we propose cross-model ambiguity detection to mitigate the error propagation that occurs when a single model is employed to detect ambiguous pairs for its training. With all components combined, our proposed method demonstrates its effectiveness in PRVR. 

---
# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO 

**Authors**: Jinyoung Park, Jeehye Na, Jinyoung Kim, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07464)  

**Abstract**: Recent works have demonstrated the effectiveness of reinforcement learning (RL)-based post-training in enhancing the reasoning capabilities of large language models (LLMs). In particular, Group Relative Policy Optimization (GRPO) has shown impressive success by employing a PPO-style reinforcement algorithm with group-based normalized rewards. However, the application of GRPO to Video Large Language Models (Video LLMs) has been less studied. In this paper, we explore GRPO for video LLMs and identify two primary issues that impede its effective learning: (1) reliance on safeguards, and (2) the vanishing advantage problem. To mitigate these challenges, we propose DeepVideo-R1, a video large language model trained with our proposed Reg-GRPO (Regressive GRPO) and difficulty-aware data augmentation strategy. Reg-GRPO reformulates the GRPO objective as a regression task, directly predicting the advantage in GRPO. This design eliminates the need for safeguards like clipping and min functions, thereby facilitating more direct policy guidance by aligning the model with the advantage values. We also design the difficulty-aware data augmentation strategy that dynamically augments training samples at solvable difficulty levels, fostering diverse and informative reward signals. Our comprehensive experiments show that DeepVideo-R1 significantly improves video reasoning performance across multiple video reasoning benchmarks. 

---
# CCI4.0: A Bilingual Pretraining Dataset for Enhancing Reasoning in Large Language Models 

**Authors**: Guang Liu, Liangdong Wang, Jijie Li, Yang Yu, Yao Xu, Jiabei Chen, Yu Bai, Feng Liao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07463)  

**Abstract**: We introduce CCI4.0, a large-scale bilingual pre-training dataset engineered for superior data quality and diverse human-like reasoning trajectory. CCI4.0 occupies roughly $35$ TB of disk space and comprises two sub-datasets: CCI4.0-M2-Base and CCI4.0-M2-CoT. CCI4.0-M2-Base combines a $5.2$ TB carefully curated Chinese web corpus, a $22.5$ TB English subset from Nemotron-CC, and diverse sources from math, wiki, arxiv, and code. Although these data are mostly sourced from well-processed datasets, the quality standards of various domains are dynamic and require extensive expert experience and labor to process. So, we propose a novel pipeline justifying data quality mainly based on models through two-stage deduplication, multiclassifier quality scoring, and domain-aware fluency filtering. We extract $4.5$ billion pieces of CoT(Chain-of-Thought) templates, named CCI4.0-M2-CoT. Differing from the distillation of CoT from larger models, our proposed staged CoT extraction exemplifies diverse reasoning patterns and significantly decreases the possibility of hallucination. Empirical evaluations demonstrate that LLMs pre-trained in CCI4.0 benefit from cleaner, more reliable training signals, yielding consistent improvements in downstream tasks, especially in math and code reflection tasks. Our results underscore the critical role of rigorous data curation and human thinking templates in advancing LLM performance, shedding some light on automatically processing pretraining corpora. 

---
# KScope: A Framework for Characterizing the Knowledge Status of Language Models 

**Authors**: Yuxin Xiao, Shan Chen, Jack Gallifant, Danielle Bitterman, Thomas Hartvigsen, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07458)  

**Abstract**: Characterizing a large language model's (LLM's) knowledge of a given question is challenging. As a result, prior work has primarily examined LLM behavior under knowledge conflicts, where the model's internal parametric memory contradicts information in the external context. However, this does not fully reflect how well the model knows the answer to the question. In this paper, we first introduce a taxonomy of five knowledge statuses based on the consistency and correctness of LLM knowledge modes. We then propose KScope, a hierarchical framework of statistical tests that progressively refines hypotheses about knowledge modes and characterizes LLM knowledge into one of these five statuses. We apply KScope to nine LLMs across four datasets and systematically establish: (1) Supporting context narrows knowledge gaps across models. (2) Context features related to difficulty, relevance, and familiarity drive successful knowledge updates. (3) LLMs exhibit similar feature preferences when partially correct or conflicted, but diverge sharply when consistently wrong. (4) Context summarization constrained by our feature analysis, together with enhanced credibility, further improves update effectiveness and generalizes across LLMs. 

---
# Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs 

**Authors**: Jared Strader, Aaron Ray, Jacob Arkin, Mason B. Peterson, Yun Chang, Nathan Hughes, Christopher Bradley, Yi Xuan Jia, Carlos Nieto-Granda, Rajat Talak, Chuchu Fan, Luca Carlone, Jonathan P. How, Nicholas Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.07454)  

**Abstract**: In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments. 

---
# When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment 

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07452)  

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety. 

---
# LlamaRec-LKG-RAG: A Single-Pass, Learnable Knowledge Graph-RAG Framework for LLM-Based Ranking 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.07449)  

**Abstract**: Recent advances in Large Language Models (LLMs) have driven their adoption in recommender systems through Retrieval-Augmented Generation (RAG) frameworks. However, existing RAG approaches predominantly rely on flat, similarity-based retrieval that fails to leverage the rich relational structure inherent in user-item interactions. We introduce LlamaRec-LKG-RAG, a novel single-pass, end-to-end trainable framework that integrates personalized knowledge graph context into LLM-based recommendation ranking. Our approach extends the LlamaRec architecture by incorporating a lightweight user preference module that dynamically identifies salient relation paths within a heterogeneous knowledge graph constructed from user behavior and item metadata. These personalized subgraphs are seamlessly integrated into prompts for a fine-tuned Llama-2 model, enabling efficient and interpretable recommendations through a unified inference step. Comprehensive experiments on ML-100K and Amazon Beauty datasets demonstrate consistent and significant improvements over LlamaRec across key ranking metrics (MRR, NDCG, Recall). LlamaRec-LKG-RAG demonstrates the critical value of structured reasoning in LLM-based recommendations and establishes a foundation for scalable, knowledge-aware personalization in next-generation recommender systems. Code is available at~\href{this https URL}{repository}. 

---
# Extending Epistemic Uncertainty Beyond Parameters Would Assist in Designing Reliable LLMs 

**Authors**: T. Duy Nguyen-Hien, Desi R. Ivanova, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.07448)  

**Abstract**: Although large language models (LLMs) are highly interactive and extendable, current approaches to ensure reliability in deployments remain mostly limited to rejecting outputs with high uncertainty in order to avoid misinformation. This conservative strategy reflects the current lack of tools to systematically distinguish and respond to different sources of uncertainty. In this paper, we advocate for the adoption of Bayesian Modeling of Experiments -- a framework that provides a coherent foundation to reason about uncertainty and clarify the reducibility of uncertainty -- for managing and proactively addressing uncertainty that arises in LLM deployments. This framework enables LLMs and their users to take contextually appropriate steps, such as requesting clarification, retrieving external information, or refining inputs. By supporting active resolution rather than passive avoidance, it opens the door to more reliable, transparent, and broadly applicable LLM systems, particularly in high-stakes, real-world settings. 

---
# Prompt to Protection: A Comparative Study of Multimodal LLMs in Construction Hazard Recognition 

**Authors**: Nishi Chaudhary, S M Jamil Uddin, Sathvik Sharath Chandra, Anto Ovid, Alex Albert  

**Link**: [PDF](https://arxiv.org/pdf/2506.07436)  

**Abstract**: The recent emergence of multimodal large language models (LLMs) has introduced new opportunities for improving visual hazard recognition on construction sites. Unlike traditional computer vision models that rely on domain-specific training and extensive datasets, modern LLMs can interpret and describe complex visual scenes using simple natural language prompts. However, despite growing interest in their applications, there has been limited investigation into how different LLMs perform in safety-critical visual tasks within the construction domain. To address this gap, this study conducts a comparative evaluation of five state-of-the-art LLMs: Claude-3 Opus, GPT-4.5, GPT-4o, GPT-o3, and Gemini 2.0 Pro, to assess their ability to identify potential hazards from real-world construction images. Each model was tested under three prompting strategies: zero-shot, few-shot, and chain-of-thought (CoT). Zero-shot prompting involved minimal instruction, few-shot incorporated basic safety context and a hazard source mnemonic, and CoT provided step-by-step reasoning examples to scaffold model thinking. Quantitative analysis was performed using precision, recall, and F1-score metrics across all conditions. Results reveal that prompting strategy significantly influenced performance, with CoT prompting consistently producing higher accuracy across models. Additionally, LLM performance varied under different conditions, with GPT-4.5 and GPT-o3 outperforming others in most settings. The findings also demonstrate the critical role of prompt design in enhancing the accuracy and consistency of multimodal LLMs for construction safety applications. This study offers actionable insights into the integration of prompt engineering and LLMs for practical hazard recognition, contributing to the development of more reliable AI-assisted safety systems. 

---
# Fast Geometric Embedding for Node Influence Maximization 

**Authors**: Alexander Kolpakov, Igor Rivin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07435)  

**Abstract**: Computing classical centrality measures such as betweenness and closeness is computationally expensive on large-scale graphs. In this work, we introduce an efficient force layout algorithm that embeds a graph into a low-dimensional space, where the radial distance from the origin serves as a proxy for various centrality measures. We evaluate our method on multiple graph families and demonstrate strong correlations with degree, PageRank, and paths-based centralities. As an application, it turns out that the proposed embedding allows to find high-influence nodes in a network, and provides a fast and scalable alternative to the standard greedy algorithm. 

---
# Well Begun is Half Done: Low-resource Preference Alignment by Weak-to-Strong Decoding 

**Authors**: Feifan Song, Shaohang Wei, Wen Luo, Yuxuan Fan, Tianyu Liu, Guoyin Wang, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07434)  

**Abstract**: Large Language Models (LLMs) require alignment with human preferences to avoid generating offensive, false, or meaningless content. Recently, low-resource methods for LLM alignment have been popular, while still facing challenges in obtaining both high-quality and aligned content. Motivated by the observation that the difficulty of generating aligned responses is concentrated at the beginning of decoding, we propose a novel framework, Weak-to-Strong Decoding (WSD), to enhance the alignment ability of base models by the guidance of a small aligned model. The small model first drafts well-aligned beginnings, followed by the large base model to continue the rest, controlled by a well-designed auto-switch mechanism. We also collect a new dataset, GenerAlign, to fine-tune a small-sized Pilot-3B as the draft model, which effectively enhances different base models under the WSD framework to outperform all baseline methods, while avoiding degradation on downstream tasks, termed as the alignment tax. Extensive experiments are further conducted to examine the impact of different settings and time efficiency, as well as analyses on the intrinsic mechanisms of WSD in depth. 

---
# FAMSeg: Fetal Femur and Cranial Ultrasound Segmentation Using Feature-Aware Attention and Mamba Enhancement 

**Authors**: Jie He, Minglang Chen, Minying Lu, Bocheng Liang, Junming Wei, Guiyan Peng, Jiaxi Chen, Ying Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07431)  

**Abstract**: Accurate ultrasound image segmentation is a prerequisite for precise biometrics and accurate assessment. Relying on manual delineation introduces significant errors and is time-consuming. However, existing segmentation models are designed based on objects in natural scenes, making them difficult to adapt to ultrasound objects with high noise and high similarity. This is particularly evident in small object segmentation, where a pronounced jagged effect occurs. Therefore, this paper proposes a fetal femur and cranial ultrasound image segmentation model based on feature perception and Mamba enhancement to address these challenges. Specifically, a longitudinal and transverse independent viewpoint scanning convolution block and a feature perception module were designed to enhance the ability to capture local detail information and improve the fusion of contextual information. Combined with the Mamba-optimized residual structure, this design suppresses the interference of raw noise and enhances local multi-dimensional scanning. The system builds global information and local feature dependencies, and is trained with a combination of different optimizers to achieve the optimal solution. After extensive experimental validation, the FAMSeg network achieved the fastest loss reduction and the best segmentation performance across images of varying sizes and orientations. 

---
# Plug-in and Fine-tuning: Bridging the Gap between Small Language Models and Large Language Models 

**Authors**: Kyeonghyun Kim, Jinhee Jang, Juhwan Choi, Yoonji Lee, Kyohoon Jin, YoungBin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07424)  

**Abstract**: Large language models (LLMs) are renowned for their extensive linguistic knowledge and strong generalization capabilities, but their high computational demands make them unsuitable for resource-constrained environments. In contrast, small language models (SLMs) are computationally efficient but often lack the broad generalization capacity of LLMs. To bridge this gap, we propose PiFi, a novel framework that combines the strengths of both LLMs and SLMs to achieve high performance while maintaining efficiency. PiFi integrates a single frozen layer from an LLM into a SLM and fine-tunes the combined model for specific tasks, boosting performance without a significant increase in computational cost. We show that PiFi delivers consistent performance improvements across a range of natural language processing tasks, including both natural language understanding and generation. Moreover, our findings demonstrate PiFi's ability to effectively leverage LLM knowledge, enhancing generalization to unseen domains and facilitating the transfer of linguistic abilities. 

---
# Evidential Spectrum-Aware Contrastive Learning for OOD Detection in Dynamic Graphs 

**Authors**: Nan Sun, Xixun Lin, Zhiheng Zhou, Yanmin Shang, Zhenlin Cheng, Yanan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07417)  

**Abstract**: Recently, Out-of-distribution (OOD) detection in dynamic graphs, which aims to identify whether incoming data deviates from the distribution of the in-distribution (ID) training set, has garnered considerable attention in security-sensitive fields. Current OOD detection paradigms primarily focus on static graphs and confront two critical challenges: i) high bias and high variance caused by single-point estimation, which makes the predictions sensitive to randomness in the data; ii) score homogenization resulting from the lack of OOD training data, where the model only learns ID-specific patterns, resulting in overall low OOD scores and a narrow score gap between ID and OOD data. To tackle these issues, we first investigate OOD detection in dynamic graphs through the lens of Evidential Deep Learning (EDL). Specifically, we propose EviSEC, an innovative and effective OOD detector via Evidential Spectrum-awarE Contrastive Learning. We design an evidential neural network to redefine the output as the posterior Dirichlet distribution, explaining the randomness of inputs through the uncertainty of distribution, which is overlooked by single-point estimation. Moreover, spectrum-aware augmentation module generates OOD approximations to identify patterns with high OOD scores, thereby widening the score gap between ID and OOD data and mitigating score homogenization. Extensive experiments on real-world datasets demonstrate that EviSAC effectively detects OOD samples in dynamic graphs. 

---
# LiteVLM: A Low-Latency Vision-Language Model Inference Pipeline for Resource-Constrained Environments 

**Authors**: Jin Huang, Yuchao Jin, Le An, Josh Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.07416)  

**Abstract**: This paper introduces an efficient Vision-Language Model (VLM) pipeline specifically optimized for deployment on embedded devices, such as those used in robotics and autonomous driving. The pipeline significantly reduces the computational overhead by jointly leveraging patch selection to filter irrelevant camera views, a token selection module to reduce input sequence length for the LLM, and speculative decoding to accelerate token generation. Evaluation on the NVIDIA DRIVE Thor platform for automonous driving application, our pipeline achieves $2.5\times$ end-to-end latency reduction without compromising task accuracy. The speed-up further increases to $3.2\times$ when applying FP8 post-training quantization. These results demonstrate our pipeline as a viable solution for enabling real-time VLM deployment in resource-constrained environments. 

---
# Fractional-order Jacobian Matrix Differentiation and Its Application in Artificial Neural Networks 

**Authors**: Xiaojun zhou, Chunna Zhao, Yaqun Huang, Chengli Zhou, Junjie Ye, Kemeng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07408)  

**Abstract**: Fractional-order differentiation has many characteristics different from integer-order differentiation. These characteristics can be applied to the optimization algorithms of artificial neural networks to obtain better results. However, due to insufficient theoretical research, at present, there is no fractional-order matrix differentiation method that is perfectly compatible with automatic differentiation (Autograd) technology. Therefore, we propose a fractional-order matrix differentiation calculation method. This method is introduced by the definition of the integer-order Jacobian matrix. We denote it as fractional-order Jacobian matrix differentiation (${\bf{J}^\alpha }$). Through ${\bf{J}^\alpha }$, we can carry out the matrix-based fractional-order chain rule. Based on the Linear module and the fractional-order differentiation, we design the fractional-order Autograd technology to enable the use of fractional-order differentiation in hidden layers, thereby enhancing the practicality of fractional-order differentiation in deep learning. In the experiment, according to the PyTorch framework, we design fractional-order Linear (FLinear) and replace this http URL in the multilayer perceptron with FLinear. Through the qualitative analysis of the training set and validation set $Loss$, the quantitative analysis of the test set indicators, and the analysis of time consumption and GPU memory usage during model training, we verify the superior performance of ${\bf{J}^\alpha }$ and prove that it is an excellent fractional-order gradient descent method in the field of deep learning. 

---
# Anomaly Detection and Early Warning Mechanism for Intelligent Monitoring Systems in Multi-Cloud Environments Based on LLM 

**Authors**: Yihong Jin, Ze Yang, Juntian Liu, Xinhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07407)  

**Abstract**: With the rapid development of multi-cloud environments, it is increasingly important to ensure the security and reliability of intelligent monitoring systems. In this paper, we propose an anomaly detection and early warning mechanism for intelligent monitoring system in multi-cloud environment based on Large-Scale Language Model (LLM). On the basis of the existing monitoring framework, the proposed model innovatively introduces a multi-level feature extraction method, which combines the natural language processing ability of LLM with traditional machine learning methods to enhance the accuracy of anomaly detection and improve the real-time response efficiency. By introducing the contextual understanding capabilities of LLMs, the model dynamically adapts to different cloud service providers and environments, so as to more effectively detect abnormal patterns and predict potential failures. Experimental results show that the proposed model is significantly better than the traditional anomaly detection system in terms of detection accuracy and latency, and significantly improves the resilience and active management ability of cloud infrastructure. 

---
# InverseScope: Scalable Activation Inversion for Interpreting Large Language Models 

**Authors**: Yifan Luo, Zhennan Zhou, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07406)  

**Abstract**: Understanding the internal representations of large language models (LLMs) is a central challenge in interpretability research. Existing feature interpretability methods often rely on strong assumptions about the structure of representations that may not hold in practice. In this work, we introduce InverseScope, an assumption-light and scalable framework for interpreting neural activations via input inversion. Given a target activation, we define a distribution over inputs that generate similar activations and analyze this distribution to infer the encoded features. To address the inefficiency of sampling in high-dimensional spaces, we propose a novel conditional generation architecture that significantly improves sample efficiency compared to previous methods. We further introduce a quantitative evaluation protocol that tests interpretability hypotheses using feature consistency rate computed over the sampled inputs. InverseScope scales inversion-based interpretability methods to larger models and practical tasks, enabling systematic and quantitative analysis of internal representations in real-world LLMs. 

---
# MedChat: A Multi-Agent Framework for Multimodal Diagnosis with Large Language Models 

**Authors**: Philip Liu, Sparsh Bansal, Jimmy Dinh, Aditya Pawar, Ramani Satishkumar, Shail Desai, Neeraj Gupta, Xin Wang, Shu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07400)  

**Abstract**: The integration of deep learning-based glaucoma detection with large language models (LLMs) presents an automated strategy to mitigate ophthalmologist shortages and improve clinical reporting efficiency. However, applying general LLMs to medical imaging remains challenging due to hallucinations, limited interpretability, and insufficient domain-specific medical knowledge, which can potentially reduce clinical accuracy. Although recent approaches combining imaging models with LLM reasoning have improved reporting, they typically rely on a single generalist agent, restricting their capacity to emulate the diverse and complex reasoning found in multidisciplinary medical teams. To address these limitations, we propose MedChat, a multi-agent diagnostic framework and platform that combines specialized vision models with multiple role-specific LLM agents, all coordinated by a director agent. This design enhances reliability, reduces hallucination risk, and enables interactive diagnostic reporting through an interface tailored for clinical review and educational use. Code available at this https URL. 

---
# MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems 

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07399)  

**Abstract**: Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by integrating cross-modal knowledge, enabling their increasing adoption across real-world multimodal tasks. These knowledge databases may contain sensitive information that requires privacy protection. However, multimodal RAG systems inherently grant external users indirect access to such data, making them potentially vulnerable to privacy attacks, particularly membership inference attacks (MIAs). % Existing MIA methods targeting RAG systems predominantly focus on the textual modality, while the visual modality remains relatively underexplored. To bridge this gap, we propose MrM, the first black-box MIA framework targeted at multimodal RAG systems. It utilizes a multi-object data perturbation framework constrained by counterfactual attacks, which can concurrently induce the RAG systems to retrieve the target data and generate information that leaks the membership information. Our method first employs an object-aware data perturbation method to constrain the perturbation to key semantics and ensure successful retrieval. Building on this, we design a counterfact-informed mask selection strategy to prioritize the most informative masked regions, aiming to eliminate the interference of model self-knowledge and amplify attack efficacy. Finally, we perform statistical membership inference by modeling query trials to extract features that reflect the reconstruction of masked semantics from response patterns. Experiments on two visual datasets and eight mainstream commercial visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves consistently strong performance across both sample-level and set-level evaluations, and remains robust under adaptive defenses. 

---
# From Static to Adaptive Defense: Federated Multi-Agent Deep Reinforcement Learning-Driven Moving Target Defense Against DoS Attacks in UAV Swarm Networks 

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Tian Qin, Yuyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07392)  

**Abstract**: The proliferation of unmanned aerial vehicle (UAV) swarms has enabled a wide range of mission-critical applications, but also exposes UAV networks to severe Denial-of-Service (DoS) threats due to their open wireless environment, dynamic topology, and resource constraints. Traditional static or centralized defense mechanisms are often inadequate for such dynamic and distributed scenarios. To address these challenges, we propose a novel federated multi-agent deep reinforcement learning (FMADRL)-driven moving target defense (MTD) framework for proactive and adaptive DoS mitigation in UAV swarm networks. Specifically, we design three lightweight and coordinated MTD mechanisms, including leader switching, route mutation, and frequency hopping, that leverage the inherent flexibility of UAV swarms to disrupt attacker efforts and enhance network resilience. The defense problem is formulated as a multi-agent partially observable Markov decision process (POMDP), capturing the distributed, resource-constrained, and uncertain nature of UAV swarms under attack. Each UAV is equipped with a local policy agent that autonomously selects MTD actions based on partial observations and local experiences. By employing a policy gradient-based FMADRL algorithm, UAVs collaboratively optimize their defense policies via reward-weighted aggregation, enabling distributed learning without sharing raw data and thus reducing communication overhead. Extensive simulations demonstrate that our approach significantly outperforms state-of-the-art baselines, achieving up to a 34.6% improvement in attack mitigation rate, a reduction in average recovery time of up to 94.6%, and decreases in energy consumption and defense cost by as much as 29.3% and 98.3%, respectively, while maintaining robust mission continuity under various DoS attack strategies. 

---
# Shapley-Coop: Credit Assignment for Emergent Cooperation in Self-Interested LLM Agents 

**Authors**: Yun Hua, Haosheng Chen, Shiqin Wang, Wenhao Li, Xiangfeng Wang, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07388)  

**Abstract**: Large Language Models (LLMs) show strong collaborative performance in multi-agent systems with predefined roles and workflows. However, in open-ended environments lacking coordination rules, agents tend to act in self-interested ways. The central challenge in achieving coordination lies in credit assignment -- fairly evaluating each agent's contribution and designing pricing mechanisms that align their heterogeneous goals. This problem is critical as LLMs increasingly participate in complex human-AI collaborations, where fair compensation and accountability rely on effective pricing mechanisms. Inspired by how human societies address similar coordination challenges (e.g., through temporary collaborations such as employment or subcontracting), we propose a cooperative workflow, Shapley-Coop. Shapley-Coop integrates Shapley Chain-of-Thought -- leveraging marginal contributions as a principled basis for pricing -- with structured negotiation protocols for effective price matching, enabling LLM agents to coordinate through rational task-time pricing and post-task reward redistribution. This approach aligns agent incentives, fosters cooperation, and maintains autonomy. We evaluate Shapley-Coop across two multi-agent games and a software engineering simulation, demonstrating that it consistently enhances LLM agent collaboration and facilitates equitable credit assignment. These results highlight the effectiveness of Shapley-Coop's pricing mechanisms in accurately reflecting individual contributions during task execution. 

---
# Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation 

**Authors**: Jintao Tong, Ran Ma, Yixiong Zou, Guangyao Chen, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.07376)  

**Abstract**: Cross-domain few-shot segmentation (CD-FSS) is proposed to pre-train the model on a source-domain dataset with sufficient samples, and then transfer the model to target-domain datasets where only a few samples are available for efficient fine-tuning. There are majorly two challenges in this task: (1) the domain gap and (2) fine-tuning with scarce data. To solve these challenges, we revisit the adapter-based methods, and discover an intriguing insight not explored in previous works: the adapter not only helps the fine-tuning of downstream tasks but also naturally serves as a domain information decoupler. Then, we delve into this finding for an interpretation, and find the model's inherent structure could lead to a natural decoupling of domain information. Building upon this insight, we propose the Domain Feature Navigator (DFN), which is a structure-based decoupler instead of loss-based ones like current works, to capture domain-specific information, thereby directing the model's attention towards domain-agnostic knowledge. Moreover, to prevent the potential excessive overfitting of DFN during the source-domain training, we further design the SAM-SVN method to constrain DFN from learning sample-specific knowledge. On target domains, we freeze the model and fine-tune the DFN to learn target-specific knowledge specific. Extensive experiments demonstrate that our method surpasses the state-of-the-art method in CD-FSS significantly by 2.69% and 4.68% MIoU in 1-shot and 5-shot scenarios, respectively. 

---
# HyColor: An Efficient Heuristic Algorithm for Graph Coloring 

**Authors**: Enqiang Zhu, Yu Zhang, Haopeng Sun, Ziqi Wei, Witold Pedrycz, Chanjuan Liu, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07373)  

**Abstract**: The graph coloring problem (GCP) is a classic combinatorial optimization problem that aims to find the minimum number of colors assigned to vertices of a graph such that no two adjacent vertices receive the same color. GCP has been extensively studied by researchers from various fields, including mathematics, computer science, and biological science. Due to the NP-hard nature, many heuristic algorithms have been proposed to solve GCP. However, existing GCP algorithms focus on either small hard graphs or large-scale sparse graphs (with up to 10^7 vertices). This paper presents an efficient hybrid heuristic algorithm for GCP, named HyColor, which excels in handling large-scale sparse graphs while achieving impressive results on small dense graphs. The efficiency of HyColor comes from the following three aspects: a local decision strategy to improve the lower bound on the chromatic number; a graph-reduction strategy to reduce the working graph; and a k-core and mixed degree-based greedy heuristic for efficiently coloring graphs. HyColor is evaluated against three state-of-the-art GCP algorithms across four benchmarks, comprising three large-scale sparse graph benchmarks and one small dense graph benchmark, totaling 209 instances. The results demonstrate that HyColor consistently outperforms existing heuristic algorithms in both solution accuracy and computational efficiency for the majority of instances. Notably, HyColor achieved the best solutions in 194 instances (over 93%), with 34 of these solutions significantly surpassing those of other algorithms. Furthermore, HyColor successfully determined the chromatic number and achieved optimal coloring in 128 instances. 

---
# C3S3: Complementary Competition and Contrastive Selection for Semi-Supervised Medical Image Segmentation 

**Authors**: Jiaying He, Yitong Lin, Jiahe Chen, Honghui Xu, Jianwei Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.07368)  

**Abstract**: For the immanent challenge of insufficiently annotated samples in the medical field, semi-supervised medical image segmentation (SSMIS) offers a promising solution. Despite achieving impressive results in delineating primary target areas, most current methodologies struggle to precisely capture the subtle details of boundaries. This deficiency often leads to significant diagnostic inaccuracies. To tackle this issue, we introduce C3S3, a novel semi-supervised segmentation model that synergistically integrates complementary competition and contrastive selection. This design significantly sharpens boundary delineation and enhances overall precision. Specifically, we develop an $\textit{Outcome-Driven Contrastive Learning}$ module dedicated to refining boundary localization. Additionally, we incorporate a $\textit{Dynamic Complementary Competition}$ module that leverages two high-performing sub-networks to generate pseudo-labels, thereby further improving segmentation quality. The proposed C3S3 undergoes rigorous validation on two publicly accessible datasets, encompassing the practices of both MRI and CT scans. The results demonstrate that our method achieves superior performance compared to previous cutting-edge competitors. Especially, on the 95HD and ASD metrics, our approach achieves a notable improvement of at least $6\%$, highlighting the significant advancements. The code is available at this https URL. 

---
# Multiple Object Stitching for Unsupervised Representation Learning 

**Authors**: Chengchao Shen, Dawei Liu, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07364)  

**Abstract**: Contrastive learning for single object centric images has achieved remarkable progress on unsupervised representation, but suffering inferior performance on the widespread images with multiple objects. In this paper, we propose a simple but effective method, Multiple Object Stitching (MOS), to refine the unsupervised representation for multi-object images. Specifically, we construct the multi-object images by stitching the single object centric ones, where the objects in the synthesized multi-object images are predetermined. Hence, compared to the existing contrastive methods, our method provides additional object correspondences between multi-object images without human annotations. In this manner, our method pays more attention to the representations of each object in multi-object image, thus providing more detailed representations for complicated downstream tasks, such as object detection and semantic segmentation. Experimental results on ImageNet, CIFAR and COCO datasets demonstrate that our proposed method achieves the leading unsupervised representation performance on both single object centric images and multi-object ones. The source code is available at this https URL. 

---
# Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust 

**Authors**: Claudiu Popa, Rex Pallath, Liam Cunningham, Hewad Tahiri, Abiram Kesavarajah, Tao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07363)  

**Abstract**: Deepfake Technology Unveiled: The Commoditization of AI and Its Impact on Digital Trust. With the increasing accessibility of generative AI, tools for voice cloning, face-swapping, and synthetic media creation have advanced significantly, lowering both financial and technical barriers for their use. While these technologies present innovative opportunities, their rapid growth raises concerns about trust, privacy, and security. This white paper explores the implications of deepfake technology, analyzing its role in enabling fraud, misinformation, and the erosion of authenticity in multimedia. Using cost-effective, easy to use tools such as Runway, Rope, and ElevenLabs, we explore how realistic deepfakes can be created with limited resources, demonstrating the risks posed to individuals and organizations alike. By analyzing the technical and ethical challenges of deepfake mitigation and detection, we emphasize the urgent need for regulatory frameworks, public awareness, and collaborative efforts to maintain trust in digital media. 

---
# Lightweight Joint Audio-Visual Deepfake Detection via Single-Stream Multi-Modal Learning Framework 

**Authors**: Kuiyuan Zhang, Wenjie Pei, Rushi Lan, Yifang Guo, Zhongyun Hua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07358)  

**Abstract**: Deepfakes are AI-synthesized multimedia data that may be abused for spreading misinformation. Deepfake generation involves both visual and audio manipulation. To detect audio-visual deepfakes, previous studies commonly employ two relatively independent sub-models to learn audio and visual features, respectively, and fuse them subsequently for deepfake detection. However, this may underutilize the inherent correlations between audio and visual features. Moreover, utilizing two isolated feature learning sub-models can result in redundant neural layers, making the overall model inefficient and impractical for resource-constrained environments.
In this work, we design a lightweight network for audio-visual deepfake detection via a single-stream multi-modal learning framework. Specifically, we introduce a collaborative audio-visual learning block to efficiently integrate multi-modal information while learning the visual and audio features. By iteratively employing this block, our single-stream network achieves a continuous fusion of multi-modal features across its layers. Thus, our network efficiently captures visual and audio features without the need for excessive block stacking, resulting in a lightweight network design. Furthermore, we propose a multi-modal classification module that can boost the dependence of the visual and audio classifiers on modality content. It also enhances the whole resistance of the video classifier against the mismatches between audio and visual modalities. We conduct experiments on the DF-TIMIT, FakeAVCeleb, and DFDC benchmark datasets. Compared to state-of-the-art audio-visual joint detection methods, our method is significantly lightweight with only 0.48M parameters, yet it achieves superiority in both uni-modal and multi-modal deepfakes, as well as in unseen types of deepfakes. 

---
# SALT: A Lightweight Model Adaptation Method for Closed Split Computing Environments 

**Authors**: Yuya Okada, Takayuki Nishio  

**Link**: [PDF](https://arxiv.org/pdf/2506.07355)  

**Abstract**: We propose SALT (Split-Adaptive Lightweight Tuning), a lightweight model adaptation framework for Split Computing under closed constraints, where the head and tail networks are proprietary and inaccessible to users. In such closed environments, conventional adaptation methods are infeasible since they require access to model parameters or architectures. SALT addresses this challenge by introducing a compact, trainable adapter on the client side to refine latent features from the head network, enabling user-specific adaptation without modifying the original models or increasing communication overhead. We evaluate SALT on user-specific classification tasks with CIFAR-10 and CIFAR-100, demonstrating improved accuracy with lower training latency compared to fine-tuning methods. Furthermore, SALT facilitates model adaptation for robust inference over lossy networks, a common challenge in edge-cloud environments. With minimal deployment overhead, SALT offers a practical solution for personalized inference in edge AI systems under strict system constraints. 

---
# Distributed Risk-Sensitive Safety Filters for Uncertain Discrete-Time Systems 

**Authors**: Armin Lederer, Erfaun Noorani, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2506.07347)  

**Abstract**: Ensuring safety in multi-agent systems is a significant challenge, particularly in settings where centralized coordination is impractical. In this work, we propose a novel risk-sensitive safety filter for discrete-time multi-agent systems with uncertain dynamics that leverages control barrier functions (CBFs) defined through value functions. Our approach relies on centralized risk-sensitive safety conditions based on exponential risk operators to ensure robustness against model uncertainties. We introduce a distributed formulation of the safety filter by deriving two alternative strategies: one based on worst-case anticipation and another on proximity to a known safe policy. By allowing agents to switch between strategies, feasibility can be ensured. Through detailed numerical evaluations, we demonstrate the efficacy of our approach in maintaining safety without being overly conservative. 

---
# Real-Time Execution of Action Chunking Flow Policies 

**Authors**: Kevin Black, Manuel Y. Galliker, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.07339)  

**Abstract**: Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See this https URL for videos. 

---
# Improving LLM Reasoning through Interpretable Role-Playing Steering 

**Authors**: Anyi Wang, Dong Shu, Yifan Wang, Yunpu Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.07335)  

**Abstract**: Role-playing has emerged as an effective technique for enhancing the reasoning capabilities of large language models (LLMs). However, existing methods primarily rely on prompt engineering, which often lacks stability and interpretability. In this paper, we introduce Sparse Autoencoder Role-Playing Steering (SRPS), a novel framework that identifies and manipulates internal model features associated with role-playing behavior. Our approach extracts latent representations from role-play prompts, selects the most relevant features based on activation patterns, and constructs a steering vector that can be injected into the model's residual stream with controllable intensity. Our method enables fine-grained control over role-specific behavior and offers insights into how role information influences internal model activations. Extensive experiments across various reasoning benchmarks and model sizes demonstrate consistent performance gains. Notably, in the zero-shot chain-of-thought (CoT) setting, the accuracy of Llama3.1-8B on CSQA improves from 31.86% to 39.80%, while Gemma2-9B on SVAMP increases from 37.50% to 45.10%. These results highlight the potential of SRPS to enhance reasoning ability in LLMs, providing better interpretability and stability compared to traditional prompt-based role-playing. 

---
# JavelinGuard: Low-Cost Transformer Architectures for LLM Security 

**Authors**: Yash Datta, Sharath Rajasekar  

**Link**: [PDF](https://arxiv.org/pdf/2506.07330)  

**Abstract**: We present JavelinGuard, a suite of low-cost, high-performance model architectures designed for detecting malicious intent in Large Language Model (LLM) interactions, optimized specifically for production deployment. Recent advances in transformer architectures, including compact BERT(Devlin et al. 2019) variants (e.g., ModernBERT (Warner et al. 2024)), allow us to build highly accurate classifiers with as few as approximately 400M parameters that achieve rapid inference speeds even on standard CPU hardware. We systematically explore five progressively sophisticated transformer-based architectures: Sharanga (baseline transformer classifier), Mahendra (enhanced attention-weighted pooling with deeper heads), Vaishnava and Ashwina (hybrid neural ensemble architectures), and Raudra (an advanced multi-task framework with specialized loss functions). Our models are rigorously benchmarked across nine diverse adversarial datasets, including popular sets like the NotInject series, BIPIA, Garak, ImprovedLLM, ToxicChat, WildGuard, and our newly introduced JavelinBench, specifically crafted to test generalization on challenging borderline and hard-negative cases. Additionally, we compare our architectures against leading open-source guardrail models as well as large decoder-only LLMs such as gpt-4o, demonstrating superior cost-performance trade-offs in terms of accuracy, and latency. Our findings reveal that while Raudra's multi-task design offers the most robust performance overall, each architecture presents unique trade-offs in speed, interpretability, and resource requirements, guiding practitioners in selecting the optimal balance of complexity and efficiency for real-world LLM security applications. 

---
# Reward Model Interpretability via Optimal and Pessimal Tokens 

**Authors**: Brian Christian, Hannah Rose Kirk, Jessica A.F. Thompson, Christopher Summerfield, Tsvetomira Dumbalska  

**Link**: [PDF](https://arxiv.org/pdf/2506.07326)  

**Abstract**: Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions. 

---
# Speech Recognition on TV Series with Video-guided Post-Correction 

**Authors**: Haoyuan Yang, Yue Zhang, Liqiang Jing  

**Link**: [PDF](https://arxiv.org/pdf/2506.07323)  

**Abstract**: Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments. 

---
# Towards Competent AI for Fundamental Analysis in Finance: A Benchmark Dataset and Evaluation 

**Authors**: Zonghan Wu, Junlin Wang, Congyuan Zou, Chenhan Wang, Yilei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.07315)  

**Abstract**: Generative AI, particularly large language models (LLMs), is beginning to transform the financial industry by automating tasks and helping to make sense of complex financial information. One especially promising use case is the automatic creation of fundamental analysis reports, which are essential for making informed investment decisions, evaluating credit risks, guiding corporate mergers, etc. While LLMs attempt to generate these reports from a single prompt, the risks of inaccuracy are significant. Poor analysis can lead to misguided investments, regulatory issues, and loss of trust. Existing financial benchmarks mainly evaluate how well LLMs answer financial questions but do not reflect performance in real-world tasks like generating financial analysis reports. In this paper, we propose FinAR-Bench, a solid benchmark dataset focusing on financial statement analysis, a core competence of fundamental analysis. To make the evaluation more precise and reliable, we break this task into three measurable steps: extracting key information, calculating financial indicators, and applying logical reasoning. This structured approach allows us to objectively assess how well LLMs perform each step of the process. Our findings offer a clear understanding of LLMs current strengths and limitations in fundamental analysis and provide a more practical way to benchmark their performance in real-world financial settings. 

---
# Generative Modeling of Networked Time-Series via Transformer Architectures 

**Authors**: Yusuf Elnady  

**Link**: [PDF](https://arxiv.org/pdf/2506.07312)  

**Abstract**: Many security and network applications require having large datasets to train the machine learning models. Limited data access is a well-known problem in the security domain. Recent studies have shown the potential of Transformer models to enlarge the size of data by synthesizing new samples, but the synthesized samples don't improve the models over the real data. To address this issue, we design an efficient transformer-based model as a generative framework to generate time-series data, that can be used to boost the performance of existing and new ML workflows. Our new transformer model achieves the SOTA results. We style our model to be generalizable and work across different datasets, and produce high-quality samples. 

---
# Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference 

**Authors**: Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07311)  

**Abstract**: Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. In this work, we introduce a novel integration of PagedAttention with PyTorch's FlexAttention, addressing internal fragmentation and inefficiencies associated with monolithic KV cache allocations. Implemented within IBM's Foundation Model Stack (FMS), our fused attention kernel efficiently gathers scattered KV data. Our benchmarks on an NVIDIA L4 GPU (24GB) demonstrate significantly reduced inference latency, growing only linearly (~2x) with sequence length from 128 to 2048 tokens when utilizing a global KV cache, compared to exponential latency increases without caching. While peak memory usage remains largely unchanged for single-step evaluations (dominated by model weights and activations), paged attention causes minimal incremental memory usage, observable only at sequence lengths exceeding 2048 tokens due to its power-of-two cache allocations. We open-source the full implementation and discuss its implications for future long-context model deployment. 

---
# Pre-trained Large Language Models Learn Hidden Markov Models In-context 

**Authors**: Yijia Dai, Zhaolin Gao, Yahya Satter, Sarah Dean, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.07298)  

**Abstract**: Hidden Markov Models (HMMs) are foundational tools for modeling sequential data with latent Markovian structure, yet fitting them to real-world data remains computationally challenging. In this work, we show that pre-trained large language models (LLMs) can effectively model data generated by HMMs via in-context learning (ICL)$\unicode{x2013}$their ability to infer patterns from examples within a prompt. On a diverse set of synthetic HMMs, LLMs achieve predictive accuracy approaching the theoretical optimum. We uncover novel scaling trends influenced by HMM properties, and offer theoretical conjectures for these empirical observations. We also provide practical guidelines for scientists on using ICL as a diagnostic tool for complex data. On real-world animal decision-making tasks, ICL achieves competitive performance with models designed by human experts. To our knowledge, this is the first demonstration that ICL can learn and predict HMM-generated sequences$\unicode{x2013}$an advance that deepens our understanding of in-context learning in LLMs and establishes its potential as a powerful tool for uncovering hidden structure in complex scientific data. 

---
# HotelMatch-LLM: Joint Multi-Task Training of Small and Large Language Models for Efficient Multimodal Hotel Retrieval 

**Authors**: Arian Askari, Emmanouil Stergiadis, Ilya Gusev, Moran Beladev  

**Link**: [PDF](https://arxiv.org/pdf/2506.07296)  

**Abstract**: We present HotelMatch-LLM, a multimodal dense retrieval model for the travel domain that enables natural language property search, addressing the limitations of traditional travel search engines which require users to start with a destination and editing search parameters. HotelMatch-LLM features three key innovations: (1) Domain-specific multi-task optimization with three novel retrieval, visual, and language modeling objectives; (2) Asymmetrical dense retrieval architecture combining a small language model (SLM) for efficient online query processing and a large language model (LLM) for embedding hotel data; and (3) Extensive image processing to handle all property image galleries. Experiments on four diverse test sets show HotelMatch-LLM significantly outperforms state-of-the-art models, including VISTA and MARVEL. Specifically, on the test set -- main query type -- we achieve 0.681 for HotelMatch-LLM compared to 0.603 for the most effective baseline, MARVEL. Our analysis highlights the impact of our multi-task optimization, the generalizability of HotelMatch-LLM across LLM architectures, and its scalability for processing large image galleries. 

---
# Secondary Stakeholders in AI: Fighting for, Brokering, and Navigating Agency 

**Authors**: Leah Hope Ajmani, Nuredin Ali Abdelkadir, Stevie Chancellor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07281)  

**Abstract**: As AI technologies become more human-facing, there have been numerous calls to adapt participatory approaches to AI development -- spurring the idea of participatory AI. However, these calls often focus only on primary stakeholders, such as end-users, and not secondary stakeholders. This paper seeks to translate the ideals of participatory AI to a broader population of secondary AI stakeholders through semi-structured interviews. We theorize that meaningful participation involves three participatory ideals: (1) informedness, (2) consent, and (3) agency. We also explore how secondary stakeholders realize these ideals by traversing a complicated problem space. Like walking up the rungs of a ladder, these ideals build on one another. We introduce three stakeholder archetypes: the reluctant data contributor, the unsupported activist, and the well-intentioned practitioner, who must navigate systemic barriers to achieving agentic AI relationships. We envision an AI future where secondary stakeholders are able to meaningfully participate with the AI systems they influence and are influenced by. 

---
# From Generation to Generalization: Emergent Few-Shot Learning in Video Diffusion Models 

**Authors**: Pablo Acuaviva, Aram Davtyan, Mariam Hassan, Sebastian Stapf, Ahmad Rahimi, Alexandre Alahi, Paolo Favaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07280)  

**Abstract**: Video Diffusion Models (VDMs) have emerged as powerful generative tools, capable of synthesizing high-quality spatiotemporal content. Yet, their potential goes far beyond mere video generation. We argue that the training dynamics of VDMs, driven by the need to model coherent sequences, naturally pushes them to internalize structured representations and an implicit understanding of the visual world. To probe the extent of this internal knowledge, we introduce a few-shot fine-tuning framework that repurposes VDMs for new tasks using only a handful of examples. Our method transforms each task into a visual transition, enabling the training of LoRA weights on short input-output sequences without altering the generative interface of a frozen VDM. Despite minimal supervision, the model exhibits strong generalization across diverse tasks, from low-level vision (for example, segmentation and pose estimation) to high-level reasoning (for example, on ARC-AGI). These results reframe VDMs as more than generative engines. They are adaptable visual learners with the potential to serve as the backbone for future foundation models in vision. 

---
# Tokenized Bandit for LLM Decoding and Alignment 

**Authors**: Suho Shin, Chenghao Yang, Haifeng Xu, Mohammad T. Hajiaghayi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07276)  

**Abstract**: We introduce the tokenized linear bandit (TLB) and multi-armed bandit (TMAB), variants of linear and stochastic multi-armed bandit problems inspired by LLM decoding and alignment. In these problems, at each round $t \in [T]$, a user submits a query (context), and the decision maker (DM) sequentially selects a token irrevocably from a token set. Once the sequence is complete, the DM observes a random utility from the user, whose expectation is presented by a sequence function mapping the chosen token sequence to a nonnegative real value that depends on the query.
In both problems, we first show that learning is impossible without any structure on the sequence function. We introduce a natural assumption, diminishing distance with more commons (DDMC), and propose algorithms with regret $\tilde{O}(L\sqrt{T})$ and $\tilde{O}(L\sqrt{T^{2/3}})$ for TLB and TMAB, respectively. As a side product, we obtain an (almost) optimality of the greedy decoding for LLM decoding algorithm under DDMC, which justifies the unresaonable effectiveness of greedy decoding in several tasks. This also has an immediate application to decoding-time LLM alignment, when the misaligned utility can be represented as the frozen LLM's utility and a linearly realizable latent function. We finally validate our algorithm's performance empirically as well as verify our assumptions using synthetic and real-world datasets. 

---
# Parsing the Switch: LLM-Based UD Annotation for Complex Code-Switched and Low-Resource Languages 

**Authors**: Olga Kellert, Nemika Tyagi, Muhammad Imran, Nelvin Licona-Guevara, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07274)  

**Abstract**: Code-switching presents a complex challenge for syntactic analysis, especially in low-resource language settings where annotated data is scarce. While recent work has explored the use of large language models (LLMs) for sequence-level tagging, few approaches systematically investigate how well these models capture syntactic structure in code-switched contexts. Moreover, existing parsers trained on monolingual treebanks often fail to generalize to multilingual and mixed-language input. To address this gap, we introduce the BiLingua Parser, an LLM-based annotation pipeline designed to produce Universal Dependencies (UD) annotations for code-switched text. First, we develop a prompt-based framework for Spanish-English and Spanish-Guaraní data, combining few-shot LLM prompting with expert review. Second, we release two annotated datasets, including the first Spanish-Guaraní UD-parsed corpus. Third, we conduct a detailed syntactic analysis of switch points across language pairs and communicative contexts. Experimental results show that BiLingua Parser achieves up to 95.29% LAS after expert revision, significantly outperforming prior baselines and multilingual parsers. These results show that LLMs, when carefully guided, can serve as practical tools for bootstrapping syntactic resources in under-resourced, code-switched environments. Data and source code are available at this https URL 

---
# SDE-SQL: Enhancing Text-to-SQL Generation in Large Language Models via Self-Driven Exploration with SQL Probes 

**Authors**: Wenxuan Xie, Yaxun Dai, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07245)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance on the Text-to-SQL task. However, prior approaches typically rely on static, pre-processed database information provided at inference time, which limits the model's ability to fully understand the database contents. Without dynamic interaction, LLMs are constrained to fixed, human-provided context and cannot autonomously explore the underlying data. To address this limitation, we propose SDE-SQL, a framework that enables large language models to perform self-driven exploration of databases during inference. This is accomplished by generating and executing SQL probes, which allow the model to actively retrieve information from the database and iteratively update its understanding of the data. Unlike prior methods, SDE-SQL operates in a zero-shot setting, without relying on any question-SQL pairs as in-context demonstrations. When evaluated on the BIRD benchmark with Qwen2.5-72B-Instruct, SDE-SQL achieves an 8.02% relative improvement in execution accuracy over the vanilla Qwen2.5-72B-Instruct baseline, establishing a new state-of-the-art among methods based on open-source models without supervised fine-tuning (SFT) or model ensembling. Moreover, with SFT, the performance of SDE-SQL can be further enhanced, yielding an additional 0.52% improvement. 

---
# Overclocking LLM Reasoning: Monitoring and Controlling Thinking Path Lengths in LLMs 

**Authors**: Roy Eisenstadt, Itamar Zimerman, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2506.07240)  

**Abstract**: Recently, techniques such as explicit structured reasoning have demonstrated strong test-time scaling behavior by enforcing a separation between the model's internal "thinking" process and the final response. A key factor influencing answer quality in this setting is the length of the thinking stage. When the reasoning is too short, the model may fail to capture the complexity of the task. Conversely, when it is too long, the model may overthink, leading to unnecessary computation and degraded performance. This paper explores and exploits the underlying mechanisms by which LLMs understand and regulate the length of their reasoning during explicit thought processes. First, we show that LLMs encode their progress through the reasoning process and introduce an interactive progress bar visualization, which is then used to reveal insights on the model's planning dynamics. Second, we manipulate the internal progress encoding during inference to reduce unnecessary steps and generate a more concise and decisive chain of thoughts. Our empirical results demonstrate that this "overclocking" method mitigates overthinking, improves answer accuracy, and reduces inference latency. Our code is publicly available. 

---
# VeriLoC: Line-of-Code Level Prediction of Hardware Design Quality from Verilog Code 

**Authors**: Raghu Vamshi Hemadri, Jitendra Bhandari, Johann Knechtel, Badri P Gopalan, Ramesh Narayanaswamy, Ramesh Karri, Siddharth Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.07239)  

**Abstract**: Modern chip design is complex, and there is a crucial need for early-stage prediction of key design-quality metrics like timing and routing congestion directly from Verilog code (a commonly used programming language for hardware design). It is especially important yet complex to predict individual lines of code that cause timing violations or downstream routing congestion. Prior works have tried approaches like converting Verilog into an intermediate graph representation and using LLM embeddings alongside other features to predict module-level quality, but did not consider line-level quality prediction. We propose VeriLoC, the first method that predicts design quality directly from Verilog at both the line- and module-level. To this end, VeriLoC leverages recent Verilog code-generation LLMs to extract local line-level and module-level embeddings, and train downstream classifiers/regressors on concatenations of these embeddings. VeriLoC achieves high F1-scores of 0.86-0.95 for line-level congestion and timing prediction, and reduces the mean average percentage error from 14% - 18% for SOTA methods down to only 4%. We believe that VeriLoC embeddings and insights from our work will also be of value for other predictive and optimization tasks for complex hardware design. 

---
# Learn as Individuals, Evolve as a Team: Multi-agent LLMs Adaptation in Embodied Environments 

**Authors**: Xinran Li, Chenjia Bai, Zijian Li, Jiakun Zheng, Ting Xiao, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07232)  

**Abstract**: Large language models (LLMs) possess extensive knowledge bases and strong reasoning capabilities, making them promising tools for complex, multi-agent planning in embodied environments. However, despite LLMs' advanced abilities and the sophisticated modular design of agentic methods, existing LLM-based planning algorithms remain limited by weak adaptation capabilities to multi-agent embodied scenarios. We address this limitation by introducing a framework that enables LLM agents to learn and evolve both before and during test time, equipping them with environment-relevant knowledge for better planning and enhanced communication for improved cooperation. Inspired by centralized training with decentralized execution in multi-agent reinforcement learning, we propose a \textit{Learn as Individuals, Evolve as a Team (LIET)} paradigm for multi-agent LLMs adaptation. At the individual level, LLM agents learn a local utility function from exploratory datasets to better comprehend the embodied environment, which is then queried during test time to support informed decision-making. At the team level, LLM agents collaboratively and iteratively maintain and update a shared cooperation knowledge list based on new experiences, using it to guide more effective communication. By combining individual learning with team evolution, LIET enables comprehensive and flexible adaptation for LLM agents. Our experiments on Communicative Watch-And-Help and ThreeD-World Multi-Agent Transport benchmarks demonstrate that LIET, instantiated with both LLaMA and GPT-4o, outperforms existing baselines and exhibits strong cooperative planning abilities. 

---
# Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward 

**Authors**: Tong Xiao, Xin Xu, Zhenya Huang, Hongyu Gao, Quan Liu, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07218)  

**Abstract**: Enhancing the multimodal reasoning capabilities of Multimodal Large Language Models (MLLMs) is a challenging task that has attracted increasing attention in the community. Recently, several studies have applied Reinforcement Learning with Verifiable Rewards (RLVR) to the multimodal domain in order to enhance the reasoning abilities of MLLMs. However, these works largely overlook the enhancement of multimodal perception capabilities in MLLMs, which serve as a core prerequisite and foundational component of complex multimodal reasoning. Through McNemar's test, we find that existing RLVR method fails to effectively enhance the multimodal perception capabilities of MLLMs, thereby limiting their further improvement in multimodal reasoning. To address this limitation, we propose Perception-R1, which introduces a novel visual perception reward that explicitly encourages MLLMs to perceive the visual content accurately, thereby can effectively incentivizing both their multimodal perception and reasoning capabilities. Specifically, we first collect textual visual annotations from the CoT trajectories of multimodal problems, which will serve as visual references for reward assignment. During RLVR training, we employ a judging LLM to assess the consistency between the visual annotations and the responses generated by MLLM, and assign the visual perception reward based on these consistency judgments. Extensive experiments on several multimodal reasoning benchmarks demonstrate the effectiveness of our Perception-R1, which achieves state-of-the-art performance on most benchmarks using only 1,442 training data. 

---
# Sword and Shield: Uses and Strategies of LLMs in Navigating Disinformation 

**Authors**: Gionnieve Lim, Bryan Chen Zhengyu Tan, Kellie Yu Hui Sim, Weiyan Shi, Ming Hui Chew, Ming Shan Hee, Roy Ka-Wei Lee, Simon T. Perrault, Kenny Tsu Wei Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.07211)  

**Abstract**: The emergence of Large Language Models (LLMs) presents a dual challenge in the fight against disinformation. These powerful tools, capable of generating human-like text at scale, can be weaponised to produce sophisticated and persuasive disinformation, yet they also hold promise for enhancing detection and mitigation strategies. This paper investigates the complex dynamics between LLMs and disinformation through a communication game that simulates online forums, inspired by the game Werewolf, with 25 participants. We analyse how Disinformers, Moderators, and Users leverage LLMs to advance their goals, revealing both the potential for misuse and combating disinformation. Our findings highlight the varying uses of LLMs depending on the participants' roles and strategies, underscoring the importance of understanding their effectiveness in this context. We conclude by discussing implications for future LLM development and online platform design, advocating for a balanced approach that empowers users and fosters trust while mitigating the risks of LLM-assisted disinformation. 

---
# Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs 

**Authors**: Wenrui Zhou, Shu Yang, Qingsong Yang, Zikun Guo, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07180)  

**Abstract**: As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the video-language domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE (Video-LLM Sycophancy Benchmarking and Evaluation), the first dedicated benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISE pioneeringly brings linguistic perspectives on sycophancy into the visual domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. In addition, we explore key-frame selection as an interpretable, training-free mitigation strategy, which reveals potential paths for reducing sycophantic bias by strengthening visual grounding. 

---
# Regularized Adaptive Graph Learning for Large-Scale Traffic Forecasting 

**Authors**: Kaiqi Wu, Weiyang Kong, Sen Zhang, Yubao Liu, Zitong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07179)  

**Abstract**: Traffic prediction is a critical task in spatial-temporal forecasting with broad applications in travel planning and urban management. Adaptive graph convolution networks have emerged as mainstream solutions due to their ability to learn node embeddings in a data-driven manner and capture complex latent dependencies. However, existing adaptive graph learning methods for traffic forecasting often either ignore the regularization of node embeddings, which account for a significant proportion of model parameters, or face scalability issues from expensive graph convolution operations. To address these challenges, we propose a Regularized Adaptive Graph Learning (RAGL) model. First, we introduce a regularized adaptive graph learning framework that synergizes Stochastic Shared Embedding (SSE) and adaptive graph convolution via a residual difference mechanism, achieving both embedding regularization and noise suppression. Second, to ensure scalability on large road networks, we develop the Efficient Cosine Operator (ECO), which performs graph convolution based on the cosine similarity of regularized embeddings with linear time complexity. Extensive experiments on four large-scale real-world traffic datasets show that RAGL consistently outperforms state-of-the-art methods in terms of prediction accuracy and exhibits competitive computational efficiency. 

---
# Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models 

**Authors**: Sangwon Jang, Taekyung Ki, Jaehyeong Jo, Jaehong Yoon, Soo Ye Kim, Zhe Lin, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07177)  

**Abstract**: Advancements in diffusion models have significantly improved video quality, directing attention to fine-grained controllability. However, many existing methods depend on fine-tuning large-scale video models for specific tasks, which becomes increasingly impractical as model sizes continue to grow. In this work, we present Frame Guidance, a training-free guidance for controllable video generation based on frame-level signals, such as keyframes, style reference images, sketches, or depth maps. For practical training-free guidance, we propose a simple latent processing method that dramatically reduces memory usage, and apply a novel latent optimization strategy designed for globally coherent video generation. Frame Guidance enables effective control across diverse tasks, including keyframe guidance, stylization, and looping, without any training, compatible with any video models. Experimental results show that Frame Guidance can produce high-quality controlled videos for a wide range of tasks and input signals. 

---
# CTDGSI: A comprehensive exploitation of instance selection methods for automatic text classification. VII Concurso de Teses, Dissertações e Trabalhos de Graduação em SI -- XXI Simpósio Brasileiro de Sistemas de Informação 

**Authors**: Washington Cunha, Leonardo Rocha, Marcos André Gonçalves  

**Link**: [PDF](https://arxiv.org/pdf/2506.07169)  

**Abstract**: Progress in Natural Language Processing (NLP) has been dictated by the rule of more: more data, more computing power and more complexity, best exemplified by the Large Language Models. However, training (or fine-tuning) large dense models for specific applications usually requires significant amounts of computing resources. This \textbf{Ph.D. dissertation} focuses on an under-investi\-gated NLP data engineering technique, whose potential is enormous in the current scenario known as Instance Selection (IS). The IS goal is to reduce the training set size by removing noisy or redundant instances while maintaining the effectiveness of the trained models and reducing the training process cost. We provide a comprehensive and scientifically sound comparison of IS methods applied to an essential NLP task -- Automatic Text Classification (ATC), considering several classification solutions and many datasets. Our findings reveal a significant untapped potential for IS solutions. We also propose two novel IS solutions that are noise-oriented and redundancy-aware, specifically designed for large datasets and transformer architectures. Our final solution achieved an average reduction of 41\% in training sets, while maintaining the same levels of effectiveness in all datasets. Importantly, our solutions demonstrated speedup improvements of 1.67x (up to 2.46x), making them scalable for datasets with hundreds of thousands of documents. 

---
# Efficient Text-Attributed Graph Learning through Selective Annotation and Graph Alignment 

**Authors**: Huanyi Xie, Lijie Hu, Lu Yu, Tianhao Huang, Longfei Li, Meng Li, Jun Zhou, Huan Wang, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07168)  

**Abstract**: In the realm of Text-attributed Graphs (TAGs), traditional graph neural networks (GNNs) often fall short due to the complex textual information associated with each node. Recent methods have improved node representations by leveraging large language models (LLMs) to enhance node text features, but these approaches typically require extensive annotations or fine-tuning across all nodes, which is both time-consuming and costly. To overcome these challenges, we introduce GAGA, an efficient framework for TAG representation learning. GAGA reduces annotation time and cost by focusing on annotating only representative nodes and edges. It constructs an annotation graph that captures the topological relationships among these annotations. Furthermore, GAGA employs a two-level alignment module to effectively integrate the annotation graph with the TAG, aligning their underlying structures. Experiments show that GAGA achieves classification accuracies on par with or surpassing state-of-the-art methods while requiring only 1% of the data to be annotated, demonstrating its high efficiency. 

---
# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models 

**Authors**: Qi Liu, Jingqing Ruan, Hao Li, Haodong Zhao, Desheng Wang, Jiansong Chen, Wan Guanglu, Xunliang Cai, Zhi Zheng, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07165)  

**Abstract**: Existing multi-objective preference alignment methods for large language models (LLMs) face limitations: (1) the inability to effectively balance various preference dimensions, and (2) reliance on auxiliary reward/reference models introduces computational complexity. To address these challenges, we propose Adaptive Multi-objective Preference Optimization (AMoPO), a novel framework that achieves dynamic balance across preference dimensions. By introducing the multi-objective optimization paradigm to use the dimension-aware generation metrics as implicit rewards, AMoPO aligns LLMs with diverse preferences without additional reward models or reference models. We introduce an adaptive weight assignment mechanism that models the generation space as a Gaussian distribution, allowing dynamic prioritization of preference dimensions. Empirical results demonstrate that AMoPO outperforms state-of-the-art baselines by 28.5%, and the experiments on 7B, 14B, and 32B models reveal the scaling ability of AMoPO. Moreover, additional analysis of multiple dimensions verifies its adaptability and effectiveness. These findings validate AMoPO's capability to achieve dimension-aware preference alignment, highlighting its superiority. Our codes and datasets are available at this https URL. 

---
# Syntactic Control of Language Models by Posterior Inference 

**Authors**: Vicky Xefteri, Tim Vieira, Ryan Cotterell, Afra Amini  

**Link**: [PDF](https://arxiv.org/pdf/2506.07154)  

**Abstract**: Controlling the syntactic structure of text generated by language models is valuable for applications requiring clarity, stylistic consistency, or interpretability, yet it remains a challenging task. In this paper, we argue that sampling algorithms based on the posterior inference can effectively enforce a target constituency structure during generation. Our approach combines sequential Monte Carlo, which estimates the posterior distribution by sampling from a proposal distribution, with a syntactic tagger that ensures that each generated token aligns with the desired syntactic structure. Our experiments with GPT2 and Llama3-8B models show that with an appropriate proposal distribution, we can improve syntactic accuracy, increasing the F1 score from $12.31$ (GPT2-large) and $35.33$ (Llama3-8B) to about $93$ in both cases without compromising the language model's fluency. These results underscore both the complexity of syntactic control and the effectiveness of sampling algorithms, offering a promising approach for applications where precise control over syntax is essential. 

---
# Mind the Web: The Security of Web Use Agents 

**Authors**: Avishag Shapira, Parth Atulbhai Gandhi, Edan Habler, Oleg Brodt, Asaf Shabtai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07153)  

**Abstract**: Web-use agents are rapidly being deployed to automate complex web tasks, operating with extensive browser capabilities including multi-tab navigation, DOM manipulation, JavaScript execution and authenticated session access. However, these powerful capabilities create a critical and previously unexplored attack surface. This paper demonstrates how attackers can exploit web-use agents' high-privilege capabilities by embedding malicious content in web pages such as comments, reviews, or advertisements that agents encounter during legitimate browsing tasks. In addition, we introduce the task-aligned injection technique that frame malicious commands as helpful task guidance rather than obvious attacks. This technique exploiting fundamental limitations in LLMs' contextual reasoning: agents struggle in maintaining coherent contextual awareness and fail to detect when seemingly helpful web content contains steering attempts that deviate from their original task goal. Through systematic evaluation of four popular agents (OpenAI Operator, Browser Use, Do Browser, OpenOperator), we demonstrate nine payload types that compromise confidentiality, integrity, and availability, including unauthorized camera activation, user impersonation, local file exfiltration, password leakage, and denial of service, with validation across multiple LLMs achieving success rates of 80%-100%. These payloads succeed across agents with built-in safety mechanisms, requiring only the ability to post content on public websites, creating unprecedented risks given the ease of exploitation combined with agents' high-privilege access. To address this attack, we propose comprehensive mitigation strategies including oversight mechanisms, execution constraints, and task-aware reasoning techniques, providing practical directions for secure development and deployment. 

---
# Prompting Science Report 2: The Decreasing Value of Chain of Thought in Prompting 

**Authors**: Lennart Meincke, Ethan Mollick, Lilach Mollick, Dan Shapiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.07142)  

**Abstract**: This is the second in a series of short reports that seek to help business, education, and policy leaders understand the technical details of working with AI through rigorous testing. In this report, we investigate Chain-of-Thought (CoT) prompting, a technique that encourages a large language model (LLM) to "think step by step" (Wei et al., 2022). CoT is a widely adopted method for improving reasoning tasks, however, our findings reveal a more nuanced picture of its effectiveness. We demonstrate two things:
- The effectiveness of Chain-of-Thought prompting can vary greatly depending on the type of task and model. For non-reasoning models, CoT generally improves average performance by a small amount, particularly if the model does not inherently engage in step-by-step processing by default. However, CoT can introduce more variability in answers, sometimes triggering occasional errors in questions the model would otherwise get right. We also found that many recent models perform some form of CoT reasoning even if not asked; for these models, a request to perform CoT had little impact. Performing CoT generally requires far more tokens (increasing cost and time) than direct answers.
- For models designed with explicit reasoning capabilities, CoT prompting often results in only marginal, if any, gains in answer accuracy. However, it significantly increases the time and tokens needed to generate a response. 

---
# Learning Compact Vision Tokens for Efficient Large Multimodal Models 

**Authors**: Hao Tang, Chengchao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07138)  

**Abstract**: Large multimodal models (LMMs) suffer significant computational challenges due to the high cost of Large Language Models (LLMs) and the quadratic complexity of processing long vision token sequences. In this paper, we explore the spatial redundancy among vision tokens and shorten the length of vision token sequences for inference acceleration. Specifically, we propose a Spatial Token Fusion (STF) method to learn compact vision tokens for short vision token sequence, where spatial-adjacent tokens are fused into one. Meanwhile, weight-frozen vision encoder can not well adapt to the demand of extensive downstream vision-language tasks. To this end, we further introduce a Multi-Block Token Fusion (MBTF) module to supplement multi-granularity features for the reduced token sequence. Overall, we combine STF and MBTF module to balance token reduction and information preservation, thereby improving inference efficiency without sacrificing multimodal reasoning capabilities. Experimental results demonstrate that our method based on LLaVA-1.5 achieves comparable or even superior performance to the baseline on 8 popular vision-language benchmarks with only $25\%$ vision tokens of baseline. The source code and trained weights are available at this https URL. 

---
# Taxonomy of migration scenarios for Qiskit refactoring using LLMs 

**Authors**: José Manuel Suárez, Luís Mariano Bibbó, Joaquín Bogado, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.07135)  

**Abstract**: As quantum computing advances, quantum programming libraries' heterogeneity and steady evolution create new challenges for software developers. Frequent updates in software libraries break working code that needs to be refactored, thus adding complexity to an already complex landscape. These refactoring challenges are, in many cases, fundamentally different from those known in classical software engineering due to the nature of quantum computing software. This study addresses these challenges by developing a taxonomy of quantum circuit's refactoring problems, providing a structured framework to analyze and compare different refactoring approaches. Large Language Models (LLMs) have proven valuable tools for classic software development, yet their value in quantum software engineering remains unexplored. This study uses LLMs to categorize refactoring needs in migration scenarios between different Qiskit versions. Qiskit documentation and release notes were scrutinized to create an initial taxonomy of refactoring required for migrating between Qiskit releases. Two taxonomies were produced: one by expert developers and one by an LLM. These taxonomies were compared, analyzing differences and similarities, and were integrated into a unified taxonomy that reflects the findings of both methods. By systematically categorizing refactoring challenges in Qiskit, the unified taxonomy is a foundation for future research on AI-assisted migration while enabling a more rigorous evaluation of automated refactoring techniques. Additionally, this work contributes to quantum software engineering (QSE) by enhancing software development workflows, improving language compatibility, and promoting best practices in quantum programming. 

---
# Reliable Critics: Monotonic Improvement and Convergence Guarantees for Reinforcement Learning 

**Authors**: Eshwar S. R., Gugan Thoppe, Aditya Gopalan, Gal Dalal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07134)  

**Abstract**: Despite decades of research, it remains challenging to correctly use Reinforcement Learning (RL) algorithms with function approximation. A prime example is policy iteration, whose fundamental guarantee of monotonic improvement collapses even under linear function approximation. To address this issue, we introduce Reliable Policy Iteration (RPI). It replaces the common projection or Bellman-error minimization during policy evaluation with a Bellman-based constrained optimization. We prove that not only does RPI confer textbook monotonicity on its value estimates but these estimates also lower bound the true return. Also, their limit partially satisfies the unprojected Bellman equation, emphasizing RPI's natural fit within RL. RPI is the first algorithm with such monotonicity and convergence guarantees under function approximation. For practical use, we provide a model-free variant of RPI that amounts to a novel critic. It can be readily integrated into primary model-free PI implementations such as DQN and DDPG. In classical control tasks, such RPI-enhanced variants consistently maintain their lower-bound guarantee while matching or surpassing the performance of all baseline methods. 

---
# Robotic Policy Learning via Human-assisted Action Preference Optimization 

**Authors**: Wenke xia, Yichu Yang, Hongtao Wu, Xiao Ma, Tao Kong, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07127)  

**Abstract**: Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks. 

---
# MAGNet: A Multi-Scale Attention-Guided Graph Fusion Network for DRC Violation Detection 

**Authors**: Weihan Lu, Hong Cai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.07126)  

**Abstract**: Design rule checking (DRC) is of great significance for cost reduction and design efficiency improvement in integrated circuit (IC) designs. Machine-learning-based DRC has become an important approach in computer-aided design (CAD). In this paper, we propose MAGNet, a hybrid deep learning model that integrates an improved U-Net with a graph neural network for DRC violation prediction. The U-Net backbone is enhanced with a Dynamic Attention Module (DAM) and a Multi-Scale Convolution Module (MSCM) to strengthen its capability in extracting fine-grained and multi-scale spatial features. In parallel, we construct a pixel-aligned graph structure based on chip layout tiles, and apply a specialized GNN to model the topological relationships among pins. During graph construction, a graph-to-grid mapping is generated to align GNN features with the layout image. In addition, a label amplification strategy is adopted during training to enhance the model's sensitivity to sparse violation patterns. Overall, MAGNet effectively combines spatial, semantic, and structural information, achieving improved prediction accuracy and reduced false positive rates in DRC hotspot detection. Subsequently, through incremental training, we achieve a more sensitive discrimination ability for hotspots. The results demonstrate that, in comparison with ibUnet, RouteNet, and J-Net, MAGnet significantly outperforms these models, achieving substantial improvements in overall performance. 

---
# Image segmentation and classification of E-waste for waste segregation 

**Authors**: Prakriti Tripathi, Theertha Biju, Maniram Thota, Rakesh Lingam  

**Link**: [PDF](https://arxiv.org/pdf/2506.07122)  

**Abstract**: Industry partners provided a problem statement that involves classifying electronic waste using machine learning models that will be used by pick-and-place robots for waste segregation. We started by taking common electronic waste items, such as a mouse and charger, unsoldering them, and taking pictures to create a custom dataset. Then state-of-the art YOLOv11 model was trained and run to achieve 70 mAP in real-time. Mask-RCNN model was also trained and achieved 41 mAP. The model will be further integrated with pick-and-place robots to perform segregation of e-waste. 

---
# Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models 

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07121)  

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs. 

---
# RBA-FE: A Robust Brain-Inspired Audio Feature Extractor for Depression Diagnosis 

**Authors**: Yu-Xuan Wu, Ziyan Huang, Bin Hu, Zhi-Hong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07118)  

**Abstract**: This article proposes a robust brain-inspired audio feature extractor (RBA-FE) model for depression diagnosis, using an improved hierarchical network architecture. Most deep learning models achieve state-of-the-art performance for image-based diagnostic tasks, ignoring the counterpart audio features. In order to tailor the noise challenge, RBA-FE leverages six acoustic features extracted from the raw audio, capturing both spatial characteristics and temporal dependencies. This hybrid attribute helps alleviate the precision limitation in audio feature extraction within other learning models like deep residual shrinkage networks. To deal with the noise issues, our model incorporates an improved spiking neuron model, called adaptive rate smooth leaky integrate-and-fire (ARSLIF). The ARSLIF model emulates the mechanism of ``retuning of cellular signal selectivity" in the brain attention systems, which enhances the model robustness against environmental noises in audio data. Experimental results demonstrate that RBA-FE achieves state-of-the-art accuracy on the MODMA dataset, respectively with 0.8750, 0.8974, 0.8750 and 0.8750 in precision, accuracy, recall and F1 score. Extensive experiments on the AVEC2014 and DAIC-WOZ datasets both show enhancements in noise robustness. It is further indicated by comparison that the ARSLIF neuron model suggest the abnormal firing pattern within the feature extraction on depressive audio data, offering brain-inspired interpretability. 

---
# Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings 

**Authors**: Rong-Xi Tan, Ming Chen, Ke Xue, Yao Wang, Yaoyuan Wang, Sheng Fu, Chao Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.07109)  

**Abstract**: The pursuit of universal black-box optimization (BBO) algorithms is a longstanding goal. However, unlike domains such as language or vision, where scaling structured data has driven generalization, progress in offline BBO remains hindered by the lack of unified representations for heterogeneous numerical spaces. Thus, existing offline BBO approaches are constrained to single-task and fixed-dimensional settings, failing to achieve cross-domain universal optimization. Recent advances in language models (LMs) offer a promising path forward: their embeddings capture latent relationships in a unifying way, enabling universal optimization across different data types possible. In this paper, we discuss multiple potential approaches, including an end-to-end learning framework in the form of next-token prediction, as well as prioritizing the learning of latent spaces with strong representational capabilities. To validate the effectiveness of these methods, we collect offline BBO tasks and data from open-source academic works for training. Experiments demonstrate the universality and effectiveness of our proposed methods. Our findings suggest that unifying language model priors and learning string embedding space can overcome traditional barriers in universal BBO, paving the way for general-purpose BBO algorithms. The code is provided at this https URL. 

---
# Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models 

**Authors**: Samir Abdaljalil, Hasan Kurban, Khalid Qaraqe, Erchin Serpedin  

**Link**: [PDF](https://arxiv.org/pdf/2506.07106)  

**Abstract**: Large language models (LLMs) have shown strong performance across natural language reasoning tasks, yet their reasoning processes remain brittle and difficult to interpret. Prompting techniques like Chain-of-Thought (CoT) enhance reliability by eliciting intermediate reasoning steps or aggregating multiple outputs. However, they lack mechanisms for enforcing logical structure and assessing internal coherence. We introduce Theorem-of-Thought (ToTh), a novel framework that models reasoning as collaboration among three parallel agents, each simulating a distinct mode of inference: abductive, deductive, and inductive. Each agent produces a reasoning trace, which is structured into a formal reasoning graph. To evaluate consistency, we apply Bayesian belief propagation guided by natural language inference (NLI), assigning confidence scores to each step. The most coherent graph is selected to derive the final answer. Experiments on symbolic (WebOfLies) and numerical (MultiArith) reasoning benchmarks show that ToTh consistently outperforms CoT, Self-Consistency, and CoT-Decoding across multiple LLMs, while producing interpretable and logically grounded reasoning chains. Our findings suggest a promising direction for building more robust and cognitively inspired LLM reasoning. The implementation is available at this https URL. 

---
# How Far Are We from Optimal Reasoning Efficiency? 

**Authors**: Jiaxuan Gao, Shu Yan, Qixin Tan, Lu Yang, Shusheng Xu, Wei Fu, Zhiyu Mei, Kaifeng Lyu, Yi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07104)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate remarkable problem-solving capabilities through extended Chain-of-Thought (CoT) reasoning but often produce excessively verbose and redundant reasoning traces. This inefficiency incurs high inference costs and limits practical deployment. While existing fine-tuning methods aim to improve reasoning efficiency, assessing their efficiency gains remains challenging due to inconsistent evaluations. In this work, we introduce the reasoning efficiency frontiers, empirical upper bounds derived from fine-tuning base LRMs across diverse approaches and training configurations. Based on these frontiers, we propose the Reasoning Efficiency Gap (REG), a unified metric quantifying deviations of any fine-tuned LRMs from these frontiers. Systematic evaluation on challenging mathematical benchmarks reveals significant gaps in current methods: they either sacrifice accuracy for short length or still remain inefficient under tight token budgets. To reduce the efficiency gap, we propose REO-RL, a class of Reinforcement Learning algorithms that minimizes REG by targeting a sparse set of token budgets. Leveraging numerical integration over strategically selected budgets, REO-RL approximates the full efficiency objective with low error using a small set of token budgets. Through systematic benchmarking, we demonstrate that our efficiency metric, REG, effectively captures the accuracy-length trade-off, with low-REG methods reducing length while maintaining accuracy. Our approach, REO-RL, consistently reduces REG by >=50 across all evaluated LRMs and matching Qwen3-4B/8B efficiency frontiers under a 16K token budget with minimal accuracy loss. Ablation studies confirm the effectiveness of our exponential token budget strategy. Finally, our findings highlight that fine-tuning LRMs to perfectly align with the efficiency frontiers remains an open challenge. 

---
# Filling the Missings: Spatiotemporal Data Imputation by Conditional Diffusion 

**Authors**: Wenying He, Jieling Huang, Junhua Gu, Ji Zhang, Yude Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.07099)  

**Abstract**: Missing data in spatiotemporal systems presents a significant challenge for modern applications, ranging from environmental monitoring to urban traffic management. The integrity of spatiotemporal data often deteriorates due to hardware malfunctions and software failures in real-world deployments. Current approaches based on machine learning and deep learning struggle to model the intricate interdependencies between spatial and temporal dimensions effectively and, more importantly, suffer from cumulative errors during the data imputation process, which propagate and amplify through iterations. To address these limitations, we propose CoFILL, a novel Conditional Diffusion Model for spatiotemporal data imputation. CoFILL builds on the inherent advantages of diffusion models to generate high-quality imputations without relying on potentially error-prone prior estimates. It incorporates an innovative dual-stream architecture that processes temporal and frequency domain features in parallel. By fusing these complementary features, CoFILL captures both rapid fluctuations and underlying patterns in the data, which enables more robust imputation. The extensive experiments reveal that CoFILL's noise prediction network successfully transforms random noise into meaningful values that align with the true data distribution. The results also show that CoFILL outperforms state-of-the-art methods in imputation accuracy. The source code is publicly available at this https URL. 

---
# Patient Similarity Computation for Clinical Decision Support: An Efficient Use of Data Transformation, Combining Static and Time Series Data 

**Authors**: Joydeb Kumar Sana, Mohammad M. Masud, M Sohel Rahman, M Saifur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2506.07092)  

**Abstract**: Patient similarity computation (PSC) is a fundamental problem in healthcare informatics. The aim of the patient similarity computation is to measure the similarity among patients according to their historical clinical records, which helps to improve clinical decision support. This paper presents a novel distributed patient similarity computation (DPSC) technique based on data transformation (DT) methods, utilizing an effective combination of time series and static data. Time series data are sensor-collected patients' information, including metrics like heart rate, blood pressure, Oxygen saturation, respiration, etc. The static data are mainly patient background and demographic data, including age, weight, height, gender, etc. Static data has been used for clustering the patients. Before feeding the static data to the machine learning model adaptive Weight-of-Evidence (aWOE) and Z-score data transformation (DT) methods have been performed, which improve the prediction performances. In aWOE-based patient similarity models, sensitive patient information has been processed using aWOE which preserves the data privacy of the trained models. We used the Dynamic Time Warping (DTW) approach, which is robust and very popular, for time series similarity. However, DTW is not suitable for big data due to the significant computational run-time. To overcome this problem, distributed DTW computation is used in this study. For Coronary Artery Disease, our DT based approach boosts prediction performance by as much as 11.4%, 10.20%, and 12.6% in terms of AUC, accuracy, and F-measure, respectively. In the case of Congestive Heart Failure (CHF), our proposed method achieves performance enhancement up to 15.9%, 10.5%, and 21.9% for the same measures, respectively. The proposed method reduces the computation time by as high as 40%. 

---
# On the Generalization of Data-Assisted Control in port-Hamiltonian Systems (DAC-pH) 

**Authors**: Mostafa Eslami, Maryam Babazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.07079)  

**Abstract**: This paper introduces a hypothetical hybrid control framework for port-Hamiltonian (p$\mathcal{H}$) systems, employing a dynamic decomposition based on Data-Assisted Control (DAC). The system's evolution is split into two parts with fixed topology: Right-Hand Side (RHS)- an intrinsic Hamiltonian flow handling worst-case parametric uncertainties, and Left-Hand Side (LHS)- a dissipative/input flow addressing both structural and parametric uncertainties. A virtual port variable $\Pi$ serves as the interface between these two components. A nonlinear controller manages the intrinsic Hamiltonian flow, determining a desired port control value $\Pi_c$. Concurrently, Reinforcement Learning (RL) is applied to the dissipative/input flow to learn an agent for providing optimal policy in mapping $\Pi_c$ to the actual system input. This hybrid approach effectively manages RHS uncertainties while preserving the system's inherent structure. Key advantages include adjustable performance via LHS controller parameters, enhanced AI explainability and interpretability through the port variable $\Pi$, the ability to guarantee safety and state attainability with hard/soft constraints, reduced complexity in learning hypothesis classes compared to end-to-end solutions, and improved state/parameter estimation using LHS prior knowledge and system Hamiltonian to address partial observability. The paper details the p$\mathcal{H}$ formulation, derives the decomposition, and presents the modular controller architecture. Beyond design, crucial aspects of stability and robustness analysis and synthesis are investigated, paving the way for deeper theoretical investigations. An application example, a pendulum with nonlinear dynamics, is simulated to demonstrate the approach's empirical and phenomenological benefits for future research. 

---
# Dual-Priv Pruning : Efficient Differential Private Fine-Tuning in Multimodal Large Language Models 

**Authors**: Qianshan Wei, Jiaqi Li, Zihan You, Yi Zhan, Kecen Li, Jialin Wu, Xinfeng Li Hengjun Liu, Yi Yu, Bin Cao, Yiwen Xu, Yang Liu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.07077)  

**Abstract**: Differential Privacy (DP) is a widely adopted technique, valued for its effectiveness in protecting the privacy of task-specific datasets, making it a critical tool for large language models. However, its effectiveness in Multimodal Large Language Models (MLLMs) remains uncertain. Applying Differential Privacy (DP) inherently introduces substantial computation overhead, a concern particularly relevant for MLLMs which process extensive textual and visual data. Furthermore, a critical challenge of DP is that the injected noise, necessary for privacy, scales with parameter dimensionality, leading to pronounced model degradation; This trade-off between privacy and utility complicates the application of Differential Privacy (DP) to complex architectures like MLLMs. To address these, we propose Dual-Priv Pruning, a framework that employs two complementary pruning mechanisms for DP fine-tuning in MLLMs: (i) visual token pruning to reduce input dimensionality by removing redundant visual information, and (ii) gradient-update pruning during the DP optimization process. This second mechanism selectively prunes parameter updates based on the magnitude of noisy gradients, aiming to mitigate noise impact and improve utility. Experiments demonstrate that our approach achieves competitive results with minimal performance degradation. In terms of computational efficiency, our approach consistently utilizes less memory than standard DP-SGD. While requiring only 1.74% more memory than zeroth-order methods which suffer from severe performance issues on A100 GPUs, our method demonstrates leading memory efficiency on H20 GPUs. To the best of our knowledge, we are the first to explore DP fine-tuning in MLLMs. Our code is coming soon. 

---
# From Axioms to Algorithms: Mechanized Proofs of the vNM Utility Theorem 

**Authors**: Li Jingyuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.07066)  

**Abstract**: This paper presents a comprehensive formalization of the von Neumann-Morgenstern (vNM) expected utility theorem using the Lean 4 interactive theorem prover. We implement the classical axioms of preference-completeness, transitivity, continuity, and independence-enabling machine-verified proofs of both the existence and uniqueness of utility representations. Our formalization captures the mathematical structure of preference relations over lotteries, verifying that preferences satisfying the vNM axioms can be represented by expected utility maximization.
Our contributions include a granular implementation of the independence axiom, formally verified proofs of fundamental claims about mixture lotteries, constructive demonstrations of utility existence, and computational experiments validating the results. We prove equivalence to classical presentations while offering greater precision at decision boundaries.
This formalization provides a rigorous foundation for applications in economic modeling, AI alignment, and management decision systems, bridging the gap between theoretical decision theory and computational implementation. 

---
# Com$^2$: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models 

**Authors**: Kai Xiong, Xiao Ding, Yixin Cao, Yuxiong Yan, Li Du, Yufei Zhang, Jinglong Gao, Jiaqian Liu, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07064)  

**Abstract**: Large language models (LLMs) have mastered abundant simple and explicit commonsense knowledge through pre-training, enabling them to achieve human-like performance in simple commonsense reasoning. Nevertheless, LLMs struggle to reason with complex and implicit commonsense knowledge that is derived from simple ones (such as understanding the long-term effects of certain events), an aspect humans tend to focus on more. Existing works focus on complex tasks like math and code, while complex commonsense reasoning remains underexplored due to its uncertainty and lack of structure. To fill this gap and align with real-world concerns, we propose a benchmark Com$^2$ focusing on complex commonsense reasoning. We first incorporate causal event graphs to serve as structured complex commonsense. Then we adopt causal theory~(e.g., intervention) to modify the causal event graphs and obtain different scenarios that meet human concerns. Finally, an LLM is employed to synthesize examples with slow thinking, which is guided by the logical relationships in the modified causal graphs. Furthermore, we use detective stories to construct a more challenging subset. Experiments show that LLMs struggle in reasoning depth and breadth, while post-training and slow thinking can alleviate this. The code and data are available at this https URL. 

---
# Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search 

**Authors**: Dongryung Lee, Sejune Joo, Kimin Lee, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.07062)  

**Abstract**: The problem of relocating a set of objects to designated areas amidst movable obstacles can be framed as a Geometric Task and Motion Planning (G-TAMP) problem, a subclass of task and motion planning (TAMP). Traditional approaches to G-TAMP have relied either on domain-independent heuristics or on learning from planning experience to guide the search, both of which typically demand significant computational resources or data. In contrast, humans often use common sense to intuitively decide which objects to manipulate in G-TAMP problems. Inspired by this, we propose leveraging Large Language Models (LLMs), which have common sense knowledge acquired from internet-scale data, to guide task planning in G-TAMP problems. To enable LLMs to perform geometric reasoning, we design a predicate-based prompt that encodes geometric information derived from a motion planning algorithm. We then query the LLM to generate a task plan, which is then used to search for a feasible set of continuous parameters. Since LLMs are prone to mistakes, instead of committing to LLM's outputs, we extend Monte Carlo Tree Search (MCTS) to a hybrid action space and use the LLM to guide the search. Unlike the previous approach that calls an LLM at every node and incurs high computational costs, we use it to warm-start the MCTS with the nodes explored in completing the LLM's task plan. On six different G-TAMP problems, we show our method outperforms previous LLM planners and pure search algorithms. Code can be found at: this https URL 

---
# Less is More: some Computational Principles based on Parcimony, and Limitations of Natural Intelligence 

**Authors**: Laura Cohen, Xavier Hinaut, Lilyana Petrova, Alexandre Pitti, Syd Reynal, Ichiro Tsuda  

**Link**: [PDF](https://arxiv.org/pdf/2506.07060)  

**Abstract**: Natural intelligence (NI) consistently achieves more with less. Infants learn language, develop abstract concepts, and acquire sensorimotor skills from sparse data, all within tight neural and energy limits. In contrast, today's AI relies on virtually unlimited computational power, energy, and data to reach high performance. This paper argues that constraints in NI are paradoxically catalysts for efficiency, adaptability, and creativity. We first show how limited neural bandwidth promotes concise codes that still capture complex patterns. Spiking neurons, hierarchical structures, and symbolic-like representations emerge naturally from bandwidth constraints, enabling robust generalization. Next, we discuss chaotic itinerancy, illustrating how the brain transits among transient attractors to flexibly retrieve memories and manage uncertainty. We then highlight reservoir computing, where random projections facilitate rapid generalization from small datasets. Drawing on developmental perspectives, we emphasize how intrinsic motivation, along with responsive social environments, drives infant language learning and discovery of meaning. Such active, embodied processes are largely absent in current AI. Finally, we suggest that adopting 'less is more' principles -- energy constraints, parsimonious architectures, and real-world interaction -- can foster the emergence of more efficient, interpretable, and biologically grounded artificial systems. 

---
# Policy Gradient with Tree Search: Avoiding Local Optimas through Lookahead 

**Authors**: Uri Koren, Navdeep Kumar, Uri Gadot, Giorgia Ramponi, Kfir Yehuda Levy, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2506.07054)  

**Abstract**: Classical policy gradient (PG) methods in reinforcement learning frequently converge to suboptimal local optima, a challenge exacerbated in large or complex environments. This work investigates Policy Gradient with Tree Search (PGTS), an approach that integrates an $m$-step lookahead mechanism to enhance policy optimization. We provide theoretical analysis demonstrating that increasing the tree search depth $m$-monotonically reduces the set of undesirable stationary points and, consequently, improves the worst-case performance of any resulting stationary policy. Critically, our analysis accommodates practical scenarios where policy updates are restricted to states visited by the current policy, rather than requiring updates across the entire state space. Empirical evaluations on diverse MDP structures, including Ladder, Tightrope, and Gridworld environments, illustrate PGTS's ability to exhibit "farsightedness," navigate challenging reward landscapes, escape local traps where standard PG fails, and achieve superior solutions. 

---
# Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs 

**Authors**: Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07045)  

**Abstract**: The rapid advancement of image generation technologies intensifies the demand for interpretable and robust detection methods. Although existing approaches often attain high accuracy, they typically operate as black boxes without providing human-understandable justifications. Multi-modal Large Language Models (MLLMs), while not originally intended for forgery detection, exhibit strong analytical and reasoning capabilities. When properly fine-tuned, they can effectively identify AI-generated images and offer meaningful explanations. However, existing MLLMs still struggle with hallucination and often fail to align their visual interpretations with actual image content and human reasoning. To bridge this gap, we construct a dataset of AI-generated images annotated with bounding boxes and descriptive captions that highlight synthesis artifacts, establishing a foundation for human-aligned visual-textual grounded reasoning. We then finetune MLLMs through a multi-stage optimization strategy that progressively balances the objectives of accurate detection, visual localization, and coherent textual explanation. The resulting model achieves superior performance in both detecting AI-generated images and localizing visual flaws, significantly outperforming baseline methods. 

---
# Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning 

**Authors**: LASA Team, Weiwen Xu, Hou Pong Chan, Long Li, Mahani Aljunied, Ruifeng Yuan, Jianyu Wang, Chenghao Xiao, Guizhen Chen, Chaoqun Liu, Zhaodonghui Li, Yu Sun, Junao Shen, Chaojun Wang, Jie Tan, Deli Zhao, Tingyang Xu, Hao Zhang, Yu Rong  

**Link**: [PDF](https://arxiv.org/pdf/2506.07044)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in understanding common visual elements, largely due to their large-scale datasets and advanced training strategies. However, their effectiveness in medical applications remains limited due to the inherent discrepancies between data and tasks in medical scenarios and those in the general domain. Concretely, existing medical MLLMs face the following critical limitations: (1) limited coverage of medical knowledge beyond imaging, (2) heightened susceptibility to hallucinations due to suboptimal data curation processes, (3) lack of reasoning capabilities tailored for complex medical scenarios. To address these challenges, we first propose a comprehensive data curation procedure that (1) efficiently acquires rich medical knowledge data not only from medical imaging but also from extensive medical texts and general-domain data; and (2) synthesizes accurate medical captions, visual question answering (VQA), and reasoning samples. As a result, we build a multimodal dataset enriched with extensive medical knowledge. Building on the curated data, we introduce our medical-specialized MLLM: Lingshu. Lingshu undergoes multi-stage training to embed medical expertise and enhance its task-solving capabilities progressively. Besides, we preliminarily explore the potential of applying reinforcement learning with verifiable rewards paradigm to enhance Lingshu's medical reasoning ability. Additionally, we develop MedEvalKit, a unified evaluation framework that consolidates leading multimodal and textual medical benchmarks for standardized, fair, and efficient model assessment. We evaluate the performance of Lingshu on three fundamental medical tasks, multimodal QA, text-based QA, and medical report generation. The results show that Lingshu consistently outperforms the existing open-source multimodal models on most tasks ... 

---
# Efficient $Q$-Learning and Actor-Critic Methods for Robust Average Reward Reinforcement Learning 

**Authors**: Yang Xu, Swetha Ganesh, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.07040)  

**Abstract**: We present the first $Q$-learning and actor-critic algorithms for robust average reward Markov Decision Processes (MDPs) with non-asymptotic convergence under contamination, TV distance and Wasserstein distance uncertainty sets. We show that the robust $Q$ Bellman operator is a strict contractive mapping with respect to a carefully constructed semi-norm with constant functions being quotiented out. This property supports a stochastic approximation update, that learns the optimal robust $Q$ function in $\tilde{\cO}(\epsilon^{-2})$ samples. We also show that the same idea can be used for robust $Q$ function estimation, which can be further used for critic estimation. Coupling it with theories in robust policy mirror descent update, we present a natural actor-critic algorithm that attains an $\epsilon$-optimal robust policy in $\tilde{\cO}(\epsilon^{-3})$ samples. These results advance the theory of distributionally robust reinforcement learning in the average reward setting. 

---
# AnnoDPO: Protein Functional Annotation Learning with Direct Preference Optimization 

**Authors**: Zixuan Jiang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07035)  

**Abstract**: Deciphering protein function remains a fundamental challenge in protein representation learning. The task presents significant difficulties for protein language models (PLMs) due to the sheer volume of functional annotation categories and the highly imbalanced distribution of annotated instances across biological ontologies. Inspired by the remarkable success of reinforcement learning from human feedback (RLHF) in large language model (LLM) alignment, we propose AnnoDPO, a novel multi-modal framework for protein function prediction that leverages Direct Preference Optimization (DPO) to enhance annotation learning. Our methodology addresses the dual challenges of annotation scarcity and category imbalance through preference-aligned training objectives, establishing a new paradigm for biological knowledge integration in protein representation learning. 

---
# HauntAttack: When Attack Follows Reasoning as a Shadow 

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.07031)  

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs. 

---
# SiliCoN: Simultaneous Nuclei Segmentation and Color Normalization of Histological Images 

**Authors**: Suman Mahapatra, Pradipta Maji  

**Link**: [PDF](https://arxiv.org/pdf/2506.07028)  

**Abstract**: Segmentation of nuclei regions from histological images is an important task for automated computer-aided analysis of histological images, particularly in the presence of impermissible color variation in the color appearance of stained tissue images. While color normalization enables better nuclei segmentation, accurate segmentation of nuclei structures makes color normalization rather trivial. In this respect, the paper proposes a novel deep generative model for simultaneously segmenting nuclei structures and normalizing color appearance of stained histological this http URL model judiciously integrates the merits of truncated normal distribution and spatial attention. The model assumes that the latent color appearance information, corresponding to a particular histological image, is independent of respective nuclei segmentation map as well as embedding map information. The disentangled representation makes the model generalizable and adaptable as the modification or loss in color appearance information cannot be able to affect the nuclei segmentation map as well as embedding information. Also, for dealing with the stain overlap of associated histochemical reagents, the prior for latent color appearance code is assumed to be a mixture of truncated normal distributions. The proposed model incorporates the concept of spatial attention for segmentation of nuclei regions from histological images. The performance of the proposed approach, along with a comparative analysis with related state-of-the-art algorithms, has been demonstrated on publicly available standard histological image data sets. 

---
# Optimal Transport Driven Asymmetric Image-to-Image Translation for Nuclei Segmentation of Histological Images 

**Authors**: Suman Mahapatra, Pradipta Maji  

**Link**: [PDF](https://arxiv.org/pdf/2506.07023)  

**Abstract**: Segmentation of nuclei regions from histological images enables morphometric analysis of nuclei structures, which in turn helps in the detection and diagnosis of diseases under consideration. To develop a nuclei segmentation algorithm, applicable to different types of target domain representations, image-to-image translation networks can be considered as they are invariant to target domain image representations. One of the important issues with image-to-image translation models is that they fail miserably when the information content between two image domains are asymmetric in nature. In this regard, the paper introduces a new deep generative model for segmenting nuclei structures from histological images. The proposed model considers an embedding space for handling information-disparity between information-rich histological image space and information-poor segmentation map domain. Integrating judiciously the concepts of optimal transport and measure theory, the model develops an invertible generator, which provides an efficient optimization framework with lower network complexity. The concept of invertible generator automatically eliminates the need of any explicit cycle-consistency loss. The proposed model also introduces a spatially-constrained squeeze operation within the framework of invertible generator to maintain spatial continuity within the image patches. The model provides a better trade-off between network complexity and model performance compared to other existing models having complex network architectures. The performance of the proposed deep generative model, along with a comparison with state-of-the-art nuclei segmentation methods, is demonstrated on publicly available histological image data sets. 

---
# AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint 

**Authors**: Leheng Sheng, Changshuo Shen, Weixiang Zhao, Junfeng Fang, Xiaohao Liu, Zhenkai Liang, Xiang Wang, An Zhang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.07022)  

**Abstract**: As LLMs are increasingly deployed in real-world applications, ensuring their ability to refuse malicious prompts, especially jailbreak attacks, is essential for safe and reliable use. Recently, activation steering has emerged as an effective approach for enhancing LLM safety by adding a refusal direction vector to internal activations of LLMs during inference, which will further induce the refusal behaviors of LLMs. However, indiscriminately applying activation steering fundamentally suffers from the trade-off between safety and utility, since the same steering vector can also lead to over-refusal and degraded performance on benign prompts. Although prior efforts, such as vector calibration and conditional steering, have attempted to mitigate this trade-off, their lack of theoretical grounding limits their robustness and effectiveness. To better address the trade-off between safety and utility, we present a theoretically grounded and empirically effective activation steering method called AlphaSteer. Specifically, it considers activation steering as a learnable process with two principled learning objectives: utility preservation and safety enhancement. For utility preservation, it learns to construct a nearly zero vector for steering benign data, with the null-space constraints. For safety enhancement, it learns to construct a refusal direction vector for steering malicious data, with the help of linear regression. Experiments across multiple jailbreak attacks and utility benchmarks demonstrate the effectiveness of AlphaSteer, which significantly improves the safety of LLMs without compromising general capabilities. Our codes are available at this https URL. 

---
# MAGNET: A Multi-agent Framework for Finding Audio-Visual Needles by Reasoning over Multi-Video Haystacks 

**Authors**: Sanjoy Chowdhury, Mohamed Elmoghany, Yohan Abeysinghe, Junjie Fei, Sayan Nag, Salman Khan, Mohamed Elhoseiny, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.07016)  

**Abstract**: Large multimodal models (LMMs) have shown remarkable progress in audio-visual understanding, yet they struggle with real-world scenarios that require complex reasoning across extensive video collections. Existing benchmarks for video question answering remain limited in scope, typically involving one clip per query, which falls short of representing the challenges of large-scale, audio-visual retrieval and reasoning encountered in practical applications. To bridge this gap, we introduce a novel task named AV-HaystacksQA, where the goal is to identify salient segments across different videos in response to a query and link them together to generate the most informative answer. To this end, we present AVHaystacks, an audio-visual benchmark comprising 3100 annotated QA pairs designed to assess the capabilities of LMMs in multi-video retrieval and temporal grounding task. Additionally, we propose a model-agnostic, multi-agent framework MAGNET to address this challenge, achieving up to 89% and 65% relative improvements over baseline methods on BLEU@4 and GPT evaluation scores in QA task on our proposed AVHaystacks. To enable robust evaluation of multi-video retrieval and temporal grounding for optimal response generation, we introduce two new metrics, STEM, which captures alignment errors between a ground truth and a predicted step sequence and MTGS, to facilitate balanced and interpretable evaluation of segment-level grounding performance. Project: this https URL 

---
# Deep regularization networks for inverse problems with noisy operators 

**Authors**: Fatemeh Pourahmadian, Yang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.07008)  

**Abstract**: A supervised learning approach is proposed for regularization of large inverse problems where the main operator is built from noisy data. This is germane to superresolution imaging via the sampling indicators of the inverse scattering theory. We aim to accelerate the spatiotemporal regularization process for this class of inverse problems to enable real-time imaging. In this approach, a neural operator maps each pattern on the right-hand side of the scattering equation to its affiliated regularization parameter. The network is trained in two steps which entails: (1) training on low-resolution regularization maps furnished by the Morozov discrepancy principle with nonoptimal thresholds, and (2) optimizing network predictions through minimization of the Tikhonov loss function regulated by the validation loss. Step 2 allows for tailoring of the approximate maps of Step 1 toward construction of higher quality images. This approach enables direct learning from test data and dispenses with the need for a-priori knowledge of the optimal regularization maps. The network, trained on low-resolution data, quickly generates dense regularization maps for high-resolution imaging. We highlight the importance of the training loss function on the network's generalizability. In particular, we demonstrate that networks informed by the logic of discrepancy principle lead to images of higher contrast. In this case, the training process involves many-objective optimization. We propose a new method to adaptively select the appropriate loss weights during training without requiring an additional optimization process. The proposed approach is synthetically examined for imaging damage evolution in an elastic plate. The results indicate that the discrepancy-informed regularization networks not only accelerate the imaging process, but also remarkably enhance the image quality in complex environments. 

---
# CARoL: Context-aware Adaptation for Robot Learning 

**Authors**: Zechen Hu, Tong Xu, Xuesu Xiao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07006)  

**Abstract**: Using Reinforcement Learning (RL) to learn new robotic tasks from scratch is often inefficient. Leveraging prior knowledge has the potential to significantly enhance learning efficiency, which, however, raises two critical challenges: how to determine the relevancy of existing knowledge and how to adaptively integrate them into learning a new task. In this paper, we propose Context-aware Adaptation for Robot Learning (CARoL), a novel framework to efficiently learn a similar but distinct new task from prior knowledge. CARoL incorporates context awareness by analyzing state transitions in system dynamics to identify similarities between the new task and prior knowledge. It then utilizes these identified similarities to prioritize and adapt specific knowledge pieces for the new task. Additionally, CARoL has a broad applicability spanning policy-based, value-based, and actor-critic RL algorithms. We validate the efficiency and generalizability of CARoL on both simulated robotic platforms and physical ground vehicles. The simulations include CarRacing and LunarLander environments, where CARoL demonstrates faster convergence and higher rewards when learning policies for new tasks. In real-world experiments, we show that CARoL enables a ground vehicle to quickly and efficiently adapt policies learned in simulation to smoothly traverse real-world off-road terrain. 

---
# End-to-End Probabilistic Framework for Learning with Hard Constraints 

**Authors**: Utkarsh Utkarsh, Danielle C. Maddix, Ruijun Ma, Michael W. Mahoney, Yuyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.07003)  

**Abstract**: We present a general purpose probabilistic forecasting framework, ProbHardE2E, to learn systems that can incorporate operational/physical constraints as hard requirements. ProbHardE2E enforces hard constraints by exploiting variance information in a novel way; and thus it is also capable of performing uncertainty quantification (UQ) on the model. Our methodology uses a novel differentiable probabilistic projection layer (DPPL) that can be combined with a wide range of neural network architectures. This DPPL allows the model to learn the system in an end-to-end manner, compared to other approaches where the constraints are satisfied either through a post-processing step or at inference. In addition, ProbHardE2E can optimize a strictly proper scoring rule, without making any distributional assumptions on the target, which enables it to obtain robust distributional estimates (in contrast to existing approaches that generally optimize likelihood-based objectives, which are heavily biased by their distributional assumptions and model choices); and it can incorporate a range of non-linear constraints (increasing the power of modeling and flexibility). We apply ProbHardE2E to problems in learning partial differential equations with uncertainty estimates and to probabilistic time-series forecasting, showcasing it as a broadly applicable general setup that connects these seemingly disparate domains. 

---
# Towards Physics-informed Diffusion for Anomaly Detection in Trajectories 

**Authors**: Arun Sharma, Mingzhou Yang, Majid Farhadloo, Subhankar Ghosh, Bharat Jayaprakash, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2506.06999)  

**Abstract**: Given trajectory data, a domain-specific study area, and a user-defined threshold, we aim to find anomalous trajectories indicative of possible GPS spoofing (e.g., fake trajectory). The problem is societally important to curb illegal activities in international waters, such as unauthorized fishing and illicit oil transfers. The problem is challenging due to advances in AI generated in deep fakes generation (e.g., additive noise, fake trajectories) and lack of adequate amount of labeled samples for ground-truth verification. Recent literature shows promising results for anomalous trajectory detection using generative models despite data sparsity. However, they do not consider fine-scale spatiotemporal dependencies and prior physical knowledge, resulting in higher false-positive rates. To address these limitations, we propose a physics-informed diffusion model that integrates kinematic constraints to identify trajectories that do not adhere to physical laws. Experimental results on real-world datasets in the maritime and urban domains show that the proposed framework results in higher prediction accuracy and lower estimation error rate for anomaly detection and trajectory generation methods, respectively. Our implementation is available at this https URL. 

---
# What makes Reasoning Models Different? Follow the Reasoning Leader for Efficient Decoding 

**Authors**: Ming Li, Zhengyuan Yang, Xiyao Wang, Dianqi Li, Kevin Lin, Tianyi Zhou, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06998)  

**Abstract**: Large reasoning models (LRMs) achieve strong reasoning performance by emitting long chains of thought. Yet, these verbose traces slow down inference and often drift into unnecessary detail, known as the overthinking phenomenon. To better understand LRMs' behavior, we systematically analyze the token-level misalignment between reasoning and non-reasoning models. While it is expected that their primary difference lies in the stylistic "thinking cues", LRMs uniquely exhibit two pivotal, previously under-explored phenomena: a Global Misalignment Rebound, where their divergence from non-reasoning models persists or even grows as response length increases, and more critically, a Local Misalignment Diminish, where the misalignment concentrates at the "thinking cues" each sentence starts with but rapidly declines in the remaining of the sentence. Motivated by the Local Misalignment Diminish, we propose FoReaL-Decoding, a collaborative fast-slow thinking decoding method for cost-quality trade-off. In FoReaL-Decoding, a Leading model leads the first few tokens for each sentence, and then a weaker draft model completes the following tokens to the end of each sentence. FoReaL-Decoding adopts a stochastic gate to smoothly interpolate between the small and the large model. On four popular math-reasoning benchmarks (AIME24, GPQA-Diamond, MATH500, AMC23), FoReaL-Decoding reduces theoretical FLOPs by 30 to 50% and trims CoT length by up to 40%, while preserving 86 to 100% of model performance. These results establish FoReaL-Decoding as a simple, plug-and-play route to controllable cost-quality trade-offs in reasoning-centric tasks. 

---
# MoXGATE: Modality-aware cross-attention for multi-omic gastrointestinal cancer sub-type classification 

**Authors**: Sajib Acharjee Dip, Uddip Acharjee Shuvo, Dipanwita Mallick, Abrar Rahman Abir, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06980)  

**Abstract**: Cancer subtype classification is crucial for personalized treatment and prognostic assessment. However, effectively integrating multi-omic data remains challenging due to the heterogeneous nature of genomic, epigenomic, and transcriptomic features. In this work, we propose Modality-Aware Cross-Attention MoXGATE, a novel deep-learning framework that leverages cross-attention and learnable modality weights to enhance feature fusion across multiple omics sources. Our approach effectively captures inter-modality dependencies, ensuring robust and interpretable integration. Through experiments on Gastrointestinal Adenocarcinoma (GIAC) and Breast Cancer (BRCA) datasets from TCGA, we demonstrate that MoXGATE outperforms existing methods, achieving 95\% classification accuracy. Ablation studies validate the effectiveness of cross-attention over simple concatenation and highlight the importance of different omics modalities. Moreover, our model generalizes well to unseen cancer types e.g., breast cancer, underscoring its adaptability. Key contributions include (1) a cross-attention-based multi-omic integration framework, (2) modality-weighted fusion for enhanced interpretability, (3) application of focal loss to mitigate data imbalance, and (4) validation across multiple cancer subtypes. Our results indicate that MoXGATE is a promising approach for multi-omic cancer subtype classification, offering improved performance and biological generalizability. 

---
# UdonCare: Hierarchy Pruning for Unseen Domain Discovery in Predictive Healthcare 

**Authors**: Pengfei Hu, Xiaoxue Han, Fei Wang, Yue Ning  

**Link**: [PDF](https://arxiv.org/pdf/2506.06977)  

**Abstract**: Domain generalization has become a critical challenge in clinical prediction, where patient cohorts often exhibit shifting data distributions that degrade model performance. Typical domain generalization approaches struggle in real-world healthcare settings for two main reasons: (1) patient-specific domain labels are typically unavailable, making domain discovery especially difficult; (2) purely data-driven approaches overlook key clinical insights, leading to a gap in medical knowledge integration. To address these problems, we leverage hierarchical medical ontologies like the ICD-9-CM hierarchy to group diseases into higher-level categories and discover more flexible latent domains. In this paper, we introduce UdonCare, a hierarchy-guided framework that iteratively prunes fine-grained domains, encodes these refined domains, and applies a Siamese-type inference mechanism to separate domain-related signals from patient-level features. Experimental results on clinical datasets (MIMIC-III and MIMIC-IV) show that the proposed model achieves higher performance compared to other domain generalization baselines when substantial domain gaps presents, highlighting the untapped potential of medical knowledge for enhancing domain generalization in practical healthcare applications. 

---
# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test 

**Authors**: Xiaoyuan Zhu, Yaowen Ye, Tianyi Qiu, Hanlin Zhu, Sijun Tan, Ajraf Mannan, Jonathan Michala, Raluca Ada Popa, Willie Neiswanger  

**Link**: [PDF](https://arxiv.org/pdf/2506.06975)  

**Abstract**: As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets. 

---
# Position: Simulating Society Requires Simulating Thought 

**Authors**: Chance Jiajie Li, Jiayi Wu, Zhenze Mo, Ao Qu, Yuhan Tang, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Jinhua Zhao, Paul Liang, Luis Alonso, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2506.06958)  

**Abstract**: Simulating society with large language models (LLMs), we argue, requires more than generating plausible behavior -- it demands cognitively grounded reasoning that is structured, revisable, and traceable. LLM-based agents are increasingly used to emulate individual and group behavior -- primarily through prompting and supervised fine-tuning. Yet they often lack internal coherence, causal reasoning, and belief traceability -- making them unreliable for analyzing how people reason, deliberate, or respond to interventions.
To address this, we present a conceptual modeling paradigm, Generative Minds (GenMinds), which draws from cognitive science to support structured belief representations in generative agents. To evaluate such agents, we introduce the RECAP (REconstructing CAusal Paths) framework, a benchmark designed to assess reasoning fidelity via causal traceability, demographic grounding, and intervention consistency. These contributions advance a broader shift: from surface-level mimicry to generative agents that simulate thought -- not just language -- for social simulations. 

---
# BIS Reasoning 1.0: The First Large-Scale Japanese Benchmark for Belief-Inconsistent Syllogistic Reasoning 

**Authors**: Ha-Thanh Nguyen, Chaoran Liu, Hirokazu Kiyomaru, Koichi Takeda, Yusuke Miyao, Maki Matsuda, Yusuke Oda, Pontus Stenetorp, Qianying Liu, Su Myat Noe, Hideyuki Tachibana, Kouta Nakayama, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06955)  

**Abstract**: We present BIS Reasoning 1.0, the first large-scale Japanese dataset of syllogistic reasoning problems explicitly designed to evaluate belief-inconsistent reasoning in large language models (LLMs). Unlike prior datasets such as NeuBAROCO and JFLD, which focus on general or belief-aligned reasoning, BIS Reasoning 1.0 introduces logically valid yet belief-inconsistent syllogisms to uncover reasoning biases in LLMs trained on human-aligned corpora. We benchmark state-of-the-art models - including GPT models, Claude models, and leading Japanese LLMs - revealing significant variance in performance, with GPT-4o achieving 79.54% accuracy. Our analysis identifies critical weaknesses in current LLMs when handling logically valid but belief-conflicting inputs. These findings have important implications for deploying LLMs in high-stakes domains such as law, healthcare, and scientific literature, where truth must override intuitive belief to ensure integrity and safety. 

---
# Is Your Training Pipeline Production-Ready? A Case Study in the Healthcare Domain 

**Authors**: Daniel Lawand, Lucas Quaresma, Roberto Bolgheroni, Alfredo Goldman, Renato Cordeiro Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06946)  

**Abstract**: Deploying a Machine Learning (ML) training pipeline into production requires robust software engineering practices. This differs significantly from experimental workflows. This experience report investigates this challenge in SPIRA, a project whose goal is to create an ML-Enabled System (MLES) to pre-diagnose insufficiency respiratory via speech analysis. The first version of SPIRA's training pipeline lacked critical software quality attributes. This paper presents an overview of the MLES, then compares three versions of the architecture of the Continuous Training subsystem, which evolved from a Big Ball of Mud, to a Modular Monolith, towards Microservices. By adopting different design principles and patterns to enhance its maintainability, robustness, and extensibility. In this way, the paper seeks to offer insights for both ML Engineers tasked to productionize ML training pipelines and Data Scientists seeking to adopt MLOps practices. 

---
# Polar Hierarchical Mamba: Towards Streaming LiDAR Object Detection with Point Clouds as Egocentric Sequences 

**Authors**: Mellon M. Zhang, Glen Chou, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06944)  

**Abstract**: Accurate and efficient object detection is essential for autonomous vehicles, where real-time perception requires low latency and high throughput. LiDAR sensors provide robust depth information, but conventional methods process full 360° scans in a single pass, introducing significant delay. Streaming approaches address this by sequentially processing partial scans in the native polar coordinate system, yet they rely on translation-invariant convolutions that are misaligned with polar geometry -- resulting in degraded performance or requiring complex distortion mitigation. Recent Mamba-based state space models (SSMs) have shown promise for LiDAR perception, but only in the full-scan setting, relying on geometric serialization and positional embeddings that are memory-intensive and ill-suited to streaming. We propose Polar Hierarchical Mamba (PHiM), a novel SSM architecture designed for polar-coordinate streaming LiDAR. PHiM uses local bidirectional Mamba blocks for intra-sector spatial encoding and a global forward Mamba for inter-sector temporal modeling, replacing convolutions and positional encodings with distortion-aware, dimensionally-decomposed operations. PHiM sets a new state-of-the-art among streaming detectors on the Waymo Open Dataset, outperforming the previous best by 10\% and matching full-scan baselines at twice the throughput. Code will be available at this https URL . 

---
# Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry 

**Authors**: Mahdi Salmani, Alireza Abdollahpoorrostam, Seyed-Mohsen Moosavi-Dezfooli  

**Link**: [PDF](https://arxiv.org/pdf/2506.06933)  

**Abstract**: Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur asymmetric costs; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. In this paper, we propose a general framework for decision-based attacks under asymmetric query costs, which we refer to as asymmetric black-box attacks. We modify two core components of existing attacks: the search strategy and the gradient estimation process. Specifically, we propose Asymmetric Search (AS), a more conservative variant of binary search that reduces reliance on high-cost queries, and Asymmetric Gradient Estimation (AGREST), which shifts the sampling distribution to favor low-cost queries. We design efficient algorithms that minimize total attack cost by balancing different query types, in contrast to earlier methods such as stealthy attacks that focus only on limiting expensive (high-cost) queries. Our method can be integrated into a range of existing black-box attacks with minimal changes. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, with improvements of up to 40% in some settings. 

---
# DiscoSum: Discourse-aware News Summarization 

**Authors**: Alexander Spangher, Tenghao Huang, Jialiang Gu, Jiatong Shi, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06930)  

**Abstract**: Recent advances in text summarization have predominantly leveraged large language models to generate concise summaries. However, language models often do not maintain long-term discourse structure, especially in news articles, where organizational flow significantly influences reader engagement. We introduce a novel approach to integrating discourse structure into summarization processes, focusing specifically on news articles across various media. We present a novel summarization dataset where news articles are summarized multiple times in different ways across different social media platforms (e.g. LinkedIn, Facebook, etc.). We develop a novel news discourse schema to describe summarization structures and a novel algorithm, DiscoSum, which employs beam search technique for structure-aware summarization, enabling the transformation of news stories to meet different stylistic and structural demands. Both human and automatic evaluation results demonstrate the efficacy of our approach in maintaining narrative fidelity and meeting structural requirements. 

---
# Graph-Based Physics-Guided Urban PM2.5 Air Quality Imputation with Constrained Monitoring Data 

**Authors**: Shangjie Du, Hui Wei, Dong Yoon Lee, Zhizhang Hu, Shijia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06917)  

**Abstract**: This work introduces GraPhy, a graph-based, physics-guided learning framework for high-resolution and accurate air quality modeling in urban areas with limited monitoring data. Fine-grained air quality monitoring information is essential for reducing public exposure to pollutants. However, monitoring networks are often sparse in socioeconomically disadvantaged regions, limiting the accuracy and resolution of air quality modeling. To address this, we propose a physics-guided graph neural network architecture called GraPhy with layers and edge features designed specifically for low-resolution monitoring data. Experiments using data from California's socioeconomically disadvantaged San Joaquin Valley show that GraPhy achieves the overall best performance evaluated by mean squared error (MSE), mean absolute error (MAE), and R-square value (R2), improving the performance by 9%-56% compared to various baseline models. Moreover, GraPhy consistently outperforms baselines across different spatial heterogeneity levels, demonstrating the effectiveness of our model design. 

---
# Uncertainty Estimation on Graphs with Structure Informed Stochastic Partial Differential Equations 

**Authors**: Fred Xu, Thomas Markovich  

**Link**: [PDF](https://arxiv.org/pdf/2506.06907)  

**Abstract**: Graph Neural Networks have achieved impressive results across diverse network modeling tasks, but accurately estimating uncertainty on graphs remains difficult, especially under distributional shifts. Unlike traditional uncertainty estimation, graph-based uncertainty must account for randomness arising from both the graph's structure and its label distribution, which adds complexity. In this paper, making an analogy between the evolution of a stochastic partial differential equation (SPDE) driven by Matern Gaussian Process and message passing using GNN layers, we present a principled way to design a novel message passing scheme that incorporates spatial-temporal noises motivated by the Gaussian Process approach to SPDE. Our method simultaneously captures uncertainty across space and time and allows explicit control over the covariance kernel smoothness, thereby enhancing uncertainty estimates on graphs with both low and high label informativeness. Our extensive experiments on Out-of-Distribution (OOD) detection on graph datasets with varying label informativeness demonstrate the soundness and superiority of our model to existing approaches. 

---
# Can Biologically Plausible Temporal Credit Assignment Rules Match BPTT for Neural Similarity? E-prop as an Example 

**Authors**: Yuhan Helena Liu, Guangyu Robert Yang, Christopher J. Cueva  

**Link**: [PDF](https://arxiv.org/pdf/2506.06904)  

**Abstract**: Understanding how the brain learns may be informed by studying biologically plausible learning rules. These rules, often approximating gradient descent learning to respect biological constraints such as locality, must meet two critical criteria to be considered an appropriate brain model: (1) good neuroscience task performance and (2) alignment with neural recordings. While extensive research has assessed the first criterion, the second remains underexamined. Employing methods such as Procrustes analysis on well-known neuroscience datasets, this study demonstrates the existence of a biologically plausible learning rule -- namely e-prop, which is based on gradient truncation and has demonstrated versatility across a wide range of tasks -- that can achieve neural data similarity comparable to Backpropagation Through Time (BPTT) when matched for task accuracy. Our findings also reveal that model architecture and initial conditions can play a more significant role in determining neural similarity than the specific learning rule. Furthermore, we observe that BPTT-trained models and their biologically plausible counterparts exhibit similar dynamical properties at comparable accuracies. These results underscore the substantial progress made in developing biologically plausible learning rules, highlighting their potential to achieve both competitive task performance and neural data similarity. 

---
# LLM-D12: A Dual-Dimensional Scale of Instrumental and Relational Dependencies on Large Language Models 

**Authors**: Ala Yankouskaya, Areej B. Babiker, Syeda W. F. Rizvi, Sameha Alshakhsi, Magnus Liebherr, Raian Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.06874)  

**Abstract**: There is growing interest in understanding how people interact with large language models (LLMs) and whether such models elicit dependency or even addictive behaviour. Validated tools to assess the extent to which individuals may become dependent on LLMs are scarce and primarily build on classic behavioral addiction symptoms, adapted to the context of LLM use. We view this as a conceptual limitation, as the LLM-human relationship is more nuanced and warrants a fresh and distinct perspective. To address this gap, we developed and validated a new 12-item questionnaire to measure LLM dependency, referred to as LLM-D12. The scale was based on the authors' prior theoretical work, with items developed accordingly and responses collected from 526 participants in the UK. Exploratory and confirmatory factor analyses, performed on separate halves of the total sample using a split-sample approach, supported a two-factor structure: Instrumental Dependency (six items) and Relationship Dependency (six items). Instrumental Dependency reflects the extent to which individuals rely on LLMs to support or collaborate in decision-making and cognitive tasks. Relationship Dependency captures the tendency to perceive LLMs as socially meaningful, sentient, or companion-like entities. The two-factor structure demonstrated excellent internal consistency and clear discriminant validity. External validation confirmed both the conceptual foundation and the distinction between the two subscales. The psychometric properties and structure of our LLM-D12 scale were interpreted in light of the emerging view that dependency on LLMs does not necessarily indicate dysfunction but may still reflect reliance levels that could become problematic in certain contexts. 

---
# Recursive Semantic Anchoring in ISO 639:2023: A Structural Extension to ISO/TC 37 Frameworks 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06870)  

**Abstract**: ISO 639:2023 unifies the ISO language-code family and introduces contextual metadata, but it lacks a machine-native mechanism for handling dialectal drift and creole mixtures. We propose a formalisation of recursive semantic anchoring, attaching to every language entity $\chi$ a family of fixed-point operators $\phi_{n,m}$ that model bounded semantic drift via the relation $\phi_{n,m}(\chi) = \chi \oplus \Delta(\chi)$, where $\Delta(\chi)$ is a drift vector in a latent semantic manifold. The base anchor $\phi_{0,0}$ recovers the canonical ISO 639:2023 identity, whereas $\phi_{99,9}$ marks the maximal drift state that triggers a deterministic fallback. Using category theory, we treat the operators $\phi_{n,m}$ as morphisms and drift vectors as arrows in a category $\mathrm{DriftLang}$. A functor $\Phi: \mathrm{DriftLang} \to \mathrm{AnchorLang}$ maps every drifted object to its unique anchor and proves convergence. We provide an RDF/Turtle schema (\texttt{BaseLanguage}, \texttt{DriftedLanguage}, \texttt{ResolvedAnchor}) and worked examples -- e.g., $\phi_{8,4}$ (Standard Mandarin) versus $\phi_{8,7}$ (a colloquial variant), and $\phi_{1,7}$ for Nigerian Pidgin anchored to English. Experiments with transformer models show higher accuracy in language identification and translation on noisy or code-switched input when the $\phi$-indices are used to guide fallback routing. The framework is compatible with ISO/TC 37 and provides an AI-tractable, drift-aware semantic layer for future standards. 

---
# SAFE: Finding Sparse and Flat Minima to Improve Pruning 

**Authors**: Dongyeop Lee, Kwanhee Lee, Jinseok Chung, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06866)  

**Abstract**: Sparsifying neural networks often suffers from seemingly inevitable performance degradation, and it remains challenging to restore the original performance despite much recent progress. Motivated by recent studies in robust optimization, we aim to tackle this problem by finding subnetworks that are both sparse and flat at the same time. Specifically, we formulate pruning as a sparsity-constrained optimization problem where flatness is encouraged as an objective. We solve it explicitly via an augmented Lagrange dual approach and extend it further by proposing a generalized projection operation, resulting in novel pruning methods called SAFE and its extension, SAFE$^+$. Extensive evaluations on standard image classification and language modeling tasks reveal that SAFE consistently yields sparse networks with improved generalization performance, which compares competitively to well-established baselines. In addition, SAFE demonstrates resilience to noisy data, making it well-suited for real-world conditions. 

---
# Face recognition on point cloud with cgan-top for denoising 

**Authors**: Junyu Liu, Jianfeng Ren, Sunhong Liang, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06864)  

**Abstract**: Face recognition using 3D point clouds is gaining growing interest, while raw point clouds often contain a significant amount of noise due to imperfect sensors. In this paper, an end-to-end 3D face recognition on a noisy point cloud is proposed, which synergistically integrates the denoising and recognition modules. Specifically, a Conditional Generative Adversarial Network on Three Orthogonal Planes (cGAN-TOP) is designed to effectively remove the noise in the point cloud, and recover the underlying features for subsequent recognition. A Linked Dynamic Graph Convolutional Neural Network (LDGCNN) is then adapted to recognize faces from the processed point cloud, which hierarchically links both the local point features and neighboring features of multiple scales. The proposed method is validated on the Bosphorus dataset. It significantly improves the recognition accuracy under all noise settings, with a maximum gain of 14.81%. 

---
# Multimodal Spatial Language Maps for Robot Navigation and Manipulation 

**Authors**: Chenguang Huang, Oier Mees, Andy Zeng, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2506.06862)  

**Abstract**: Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues. 

---
# High-Fidelity Scientific Simulation Surrogates via Adaptive Implicit Neural Representations 

**Authors**: Ziwei Li, Yuhan Duan, Tianyu Xiong, Yi-Tang Chen, Wei-Lun Chao, Han-Wei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06858)  

**Abstract**: Effective surrogate models are critical for accelerating scientific simulations. Implicit neural representations (INRs) offer a compact and continuous framework for modeling spatially structured data, but they often struggle with complex scientific fields exhibiting localized, high-frequency variations. Recent approaches address this by introducing additional features along rigid geometric structures (e.g., grids), but at the cost of flexibility and increased model size. In this paper, we propose a simple yet effective alternative: Feature-Adaptive INR (FA-INR). FA-INR leverages cross-attention to an augmented memory bank to learn flexible feature representations, enabling adaptive allocation of model capacity based on data characteristics, rather than rigid structural assumptions. To further improve scalability, we introduce a coordinate-guided mixture of experts (MoE) that enhances the specialization and efficiency of feature representations. Experiments on three large-scale ensemble simulation datasets show that FA-INR achieves state-of-the-art fidelity while significantly reducing model size, establishing a new trade-off frontier between accuracy and compactness for INR-based surrogates. 

---
# Position Prediction Self-Supervised Learning for Multimodal Satellite Imagery Semantic Segmentation 

**Authors**: John Waithaka, Moise Busogi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06852)  

**Abstract**: Semantic segmentation of satellite imagery is crucial for Earth observation applications, but remains constrained by limited labelled training data. While self-supervised pretraining methods like Masked Autoencoders (MAE) have shown promise, they focus on reconstruction rather than localisation-a fundamental aspect of segmentation tasks. We propose adapting LOCA (Location-aware), a position prediction self-supervised learning method, for multimodal satellite imagery semantic segmentation. Our approach addresses the unique challenges of satellite data by extending SatMAE's channel grouping from multispectral to multimodal data, enabling effective handling of multiple modalities, and introducing same-group attention masking to encourage cross-modal interaction during pretraining. The method uses relative patch position prediction, encouraging spatial reasoning for localisation rather than reconstruction. We evaluate our approach on the Sen1Floods11 flood mapping dataset, where it significantly outperforms existing reconstruction-based self-supervised learning methods for satellite imagery. Our results demonstrate that position prediction tasks, when properly adapted for multimodal satellite imagery, learn representations more effective for satellite image semantic segmentation than reconstruction-based approaches. 

---
# PCoT: Persuasion-Augmented Chain of Thought for Detecting Fake News and Social Media Disinformation 

**Authors**: Arkadiusz Modzelewski, Witold Sosnowski, Tiziano Labruna, Adam Wierzbicki, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2506.06842)  

**Abstract**: Disinformation detection is a key aspect of media literacy. Psychological studies have shown that knowledge of persuasive fallacies helps individuals detect disinformation. Inspired by these findings, we experimented with large language models (LLMs) to test whether infusing persuasion knowledge enhances disinformation detection. As a result, we introduce the Persuasion-Augmented Chain of Thought (PCoT), a novel approach that leverages persuasion to improve disinformation detection in zero-shot classification. We extensively evaluate PCoT on online news and social media posts. Moreover, we publish two novel, up-to-date disinformation datasets: EUDisinfo and MultiDis. These datasets enable the evaluation of PCoT on content entirely unseen by the LLMs used in our experiments, as the content was published after the models' knowledge cutoffs. We show that, on average, PCoT outperforms competitive methods by 15% across five LLMs and five datasets. These findings highlight the value of persuasion in strengthening zero-shot disinformation detection. 

---
# A Statistical Framework for Model Selection in LSTM Networks 

**Authors**: Fahad Mostafa  

**Link**: [PDF](https://arxiv.org/pdf/2506.06840)  

**Abstract**: Long Short-Term Memory (LSTM) neural network models have become the cornerstone for sequential data modeling in numerous applications, ranging from natural language processing to time series forecasting. Despite their success, the problem of model selection, including hyperparameter tuning, architecture specification, and regularization choice remains largely heuristic and computationally expensive. In this paper, we propose a unified statistical framework for systematic model selection in LSTM networks. Our framework extends classical model selection ideas, such as information criteria and shrinkage estimation, to sequential neural networks. We define penalized likelihoods adapted to temporal structures, propose a generalized threshold approach for hidden state dynamics, and provide efficient estimation strategies using variational Bayes and approximate marginal likelihood methods. Several biomedical data centric examples demonstrate the flexibility and improved performance of the proposed framework. 

---
# AI-Generated Compromises for Coalition Formation 

**Authors**: Eyal Briman, Ehud Shapiro, Nimrod Talmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06837)  

**Abstract**: The challenge of finding compromises between agent proposals is fundamental to AI subfields such as argumentation, mediation, and negotiation. Building on this tradition, Elkind et al. (2021) introduced a process for coalition formation that seeks majority-supported proposals preferable to the status quo, using a metric space where each agent has an ideal point. A crucial step in this process involves identifying compromise proposals around which agent coalitions can unite. How to effectively find such compromise proposals remains an open question. We address this gap by formalizing a model that incorporates agent bounded rationality and uncertainty, and by developing AI methods to generate compromise proposals. We focus on the domain of collaborative document writing, such as the democratic drafting of a community constitution. Our approach uses natural language processing techniques and large language models to induce a semantic metric space over text. Based on this space, we design algorithms to suggest compromise points likely to receive broad support. To evaluate our methods, we simulate coalition formation processes and show that AI can facilitate large-scale democratic text editing, a domain where traditional tools are limited. 

---
# Harnessing Vision-Language Models for Time Series Anomaly Detection 

**Authors**: Zelin He, Sarah Alnegheimish, Matthew Reimherr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06836)  

**Abstract**: Time-series anomaly detection (TSAD) has played a vital role in a variety of fields, including healthcare, finance, and industrial monitoring. Prior methods, which mainly focus on training domain-specific models on numerical data, lack the visual-temporal reasoning capacity that human experts have to identify contextual anomalies. To fill this gap, we explore a solution based on vision language models (VLMs). Recent studies have shown the ability of VLMs for visual reasoning tasks, yet their direct application to time series has fallen short on both accuracy and efficiency. To harness the power of VLMs for TSAD, we propose a two-stage solution, with (1) ViT4TS, a vision-screening stage built on a relatively lightweight pretrained vision encoder, which leverages 2-D time-series representations to accurately localize candidate anomalies; (2) VLM4TS, a VLM-based stage that integrates global temporal context and VLM reasoning capacity to refine the detection upon the candidates provided by ViT4TS. We show that without any time-series training, VLM4TS outperforms time-series pretrained and from-scratch baselines in most cases, yielding a 24.6 percent improvement in F1-max score over the best baseline. Moreover, VLM4TS also consistently outperforms existing language-model-based TSAD methods and is on average 36 times more efficient in token usage. 

---
# EndoARSS: Adapting Spatially-Aware Foundation Model for Efficient Activity Recognition and Semantic Segmentation in Endoscopic Surgery 

**Authors**: Guankun Wang, Rui Tang, Mengya Xu, Long Bai, Huxin Gao, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.06830)  

**Abstract**: Endoscopic surgery is the gold standard for robotic-assisted minimally invasive surgery, offering significant advantages in early disease detection and precise interventions. However, the complexity of surgical scenes, characterized by high variability in different surgical activity scenarios and confused image features between targets and the background, presents challenges for surgical environment understanding. Traditional deep learning models often struggle with cross-activity interference, leading to suboptimal performance in each downstream task. To address this limitation, we explore multi-task learning, which utilizes the interrelated features between tasks to enhance overall task performance. In this paper, we propose EndoARSS, a novel multi-task learning framework specifically designed for endoscopy surgery activity recognition and semantic segmentation. Built upon the DINOv2 foundation model, our approach integrates Low-Rank Adaptation to facilitate efficient fine-tuning while incorporating Task Efficient Shared Low-Rank Adapters to mitigate gradient conflicts across diverse tasks. Additionally, we introduce the Spatially-Aware Multi-Scale Attention that enhances feature representation discrimination by enabling cross-spatial learning of global information. In order to evaluate the effectiveness of our framework, we present three novel datasets, MTLESD, MTLEndovis and MTLEndovis-Gen, tailored for endoscopic surgery scenarios with detailed annotations for both activity recognition and semantic segmentation tasks. Extensive experiments demonstrate that EndoARSS achieves remarkable performance across multiple benchmarks, significantly improving both accuracy and robustness in comparison to existing models. These results underscore the potential of EndoARSS to advance AI-driven endoscopic surgical systems, offering valuable insights for enhancing surgical safety and efficiency. 

---
# Controllable Coupled Image Generation via Diffusion Models 

**Authors**: Chenfei Yuan, Nanshan Jia, Hangqi Li, Peter W. Glynn, Zeyu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06826)  

**Abstract**: We provide an attention-level control method for the task of coupled image generation, where "coupled" means that multiple simultaneously generated images are expected to have the same or very similar backgrounds. While backgrounds coupled, the centered objects in the generated images are still expected to enjoy the flexibility raised from different text prompts. The proposed method disentangles the background and entity components in the model's cross-attention modules, attached with a sequence of time-varying weight control parameters depending on the time step of sampling. We optimize this sequence of weight control parameters with a combined objective that assesses how coupled the backgrounds are as well as text-to-image alignment and overall visual quality. Empirical results demonstrate that our method outperforms existing approaches across these criteria. 

---
# Exploring Visual Prompting: Robustness Inheritance and Beyond 

**Authors**: Qi Li, Liangzhi Li, Zhouqiang Jiang, Bowen Wang, Keke Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06823)  

**Abstract**: Visual Prompting (VP), an efficient method for transfer learning, has shown its potential in vision tasks. However, previous works focus exclusively on VP from standard source models, it is still unknown how it performs under the scenario of a robust source model: Can the robustness of the source model be successfully inherited? Does VP also encounter the same trade-off between robustness and generalization ability as the source model during this process? If such a trade-off exists, is there a strategy specifically tailored to VP to mitigate this limitation? In this paper, we thoroughly explore these three questions for the first time and provide affirmative answers to them. To mitigate the trade-off faced by VP, we propose a strategy called Prompt Boundary Loosening (PBL). As a lightweight, plug-and-play strategy naturally compatible with VP, PBL effectively ensures the successful inheritance of robustness when the source model is a robust model, while significantly enhancing VP's generalization ability across various downstream datasets. Extensive experiments across various datasets show that our findings are universal and demonstrate the significant benefits of the proposed strategy. 

---
# Hi-LSplat: Hierarchical 3D Language Gaussian Splatting 

**Authors**: Chenlu Zhan, Yufei Zhang, Gaoang Wang, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06822)  

**Abstract**: Modeling 3D language fields with Gaussian Splatting for open-ended language queries has recently garnered increasing attention. However, recent 3DGS-based models leverage view-dependent 2D foundation models to refine 3D semantics but lack a unified 3D representation, leading to view inconsistencies. Additionally, inherent open-vocabulary challenges cause inconsistencies in object and relational descriptions, impeding hierarchical semantic understanding. In this paper, we propose Hi-LSplat, a view-consistent Hierarchical Language Gaussian Splatting work for 3D open-vocabulary querying. To achieve view-consistent 3D hierarchical semantics, we first lift 2D features to 3D features by constructing a 3D hierarchical semantic tree with layered instance clustering, which addresses the view inconsistency issue caused by 2D semantic features. Besides, we introduce instance-wise and part-wise contrastive losses to capture all-sided hierarchical semantic representations. Notably, we construct two hierarchical semantic datasets to better assess the model's ability to distinguish different semantic levels. Extensive experiments highlight our method's superiority in 3D open-vocabulary segmentation and localization. Its strong performance on hierarchical semantic datasets underscores its ability to capture complex hierarchical semantics within 3D scenes. 

---
# Can LLMs Generate Reliable Test Case Generators? A Study on Competition-Level Programming Problems 

**Authors**: Yuhan Cao, Zian Chen, Kun Quan, Ziliang Zhang, Yu Wang, Xiaoning Dong, Yeqi Feng, Guanzhong He, Jingcheng Huang, Jianhao Li, Yixuan Tan, Jiafu Tang, Yilin Tang, Junlei Wu, Qianyu Xiao, Can Zheng, Shouchen Zhou, Yuxiang Zhu, Yiming Huang, Tian Xie, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2506.06821)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, capable of tackling complex tasks during inference. However, the extent to which LLMs can be utilized for code checking or debugging through test case generation remains largely unexplored. We investigate this problem from the perspective of competition-level programming (CP) programs and propose TCGBench, a Benchmark for (LLM generation of) Test Case Generators. This benchmark comprises two tasks, aimed at studying the capabilities of LLMs in (1) generating valid test case generators for a given CP problem, and further (2) generating targeted test case generators that expose bugs in human-written code. Experimental results indicate that while state-of-the-art LLMs can generate valid test case generators in most cases, most LLMs struggle to generate targeted test cases that reveal flaws in human code effectively. Especially, even advanced reasoning models (e.g., o3-mini) fall significantly short of human performance in the task of generating targeted generators. Furthermore, we construct a high-quality, manually curated dataset of instructions for generating targeted generators. Analysis demonstrates that the performance of LLMs can be enhanced with the aid of this dataset, by both prompting and fine-tuning. 

---
# IMPA-HGAE:Intra-Meta-Path Augmented Heterogeneous Graph Autoencoder 

**Authors**: Di Lin, Wanjing Ren, Xuanbin Li, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06809)  

**Abstract**: Self-supervised learning (SSL) methods have been increasingly applied to diverse downstream tasks due to their superior generalization capabilities and low annotation costs. However, most existing heterogeneous graph SSL models convert heterogeneous graphs into homogeneous ones via meta-paths for training, which only leverage information from nodes at both ends of meta-paths while underutilizing the heterogeneous node information along the meta-paths. To address this limitation, this paper proposes a novel framework named IMPA-HGAE to enhance target node embeddings by fully exploiting internal node information along meta-paths. Experimental results validate that IMPA-HGAE achieves superior performance on heterogeneous datasets. Furthermore, this paper introduce innovative masking strategies to strengthen the representational capacity of generative SSL models on heterogeneous graph data. Additionally, this paper discuss the interpretability of the proposed method and potential future directions for generative self-supervised learning in heterogeneous graphs. This work provides insights into leveraging meta-path-guided structural semantics for robust representation learning in complex graph scenarios. 

---
# Not quite Sherlock Holmes: Language model predictions do not reliably differentiate impossible from improbable events 

**Authors**: James A. Michaelov, Reeka Estacio, Zhien Zhang, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06808)  

**Abstract**: Can language models reliably predict that possible events are more likely than merely improbable ones? By teasing apart possibility, typicality, and contextual relatedness, we show that despite the results of previous work, language models' ability to do this is far from robust. In fact, under certain conditions, all models tested - including Llama 3, Gemma 2, and Mistral NeMo - perform at worse-than-chance level, assigning higher probabilities to impossible sentences such as 'the car was given a parking ticket by the brake' than to merely unlikely sentences such as 'the car was given a parking ticket by the explorer'. 

---
# Label-semantics Aware Generative Approach for Domain-Agnostic Multilabel Classification 

**Authors**: Subhendu Khatuya, Shashwat Naidu, Saptarshi Ghosh, Pawan Goyal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2506.06806)  

**Abstract**: The explosion of textual data has made manual document classification increasingly challenging. To address this, we introduce a robust, efficient domain-agnostic generative model framework for multi-label text classification. Instead of treating labels as mere atomic symbols, our approach utilizes predefined label descriptions and is trained to generate these descriptions based on the input text. During inference, the generated descriptions are matched to the pre-defined labels using a finetuned sentence transformer. We integrate this with a dual-objective loss function, combining cross-entropy loss and cosine similarity of the generated sentences with the predefined target descriptions, ensuring both semantic alignment and accuracy. Our proposed model LAGAMC stands out for its parameter efficiency and versatility across diverse datasets, making it well-suited for practical applications. We demonstrate the effectiveness of our proposed model by achieving new state-of-the-art performances across all evaluated datasets, surpassing several strong baselines. We achieve improvements of 13.94% in Micro-F1 and 24.85% in Macro-F1 compared to the closest baseline across all datasets. 

---
# Is Optimal Transport Necessary for Inverse Reinforcement Learning? 

**Authors**: Zixuan Dong, Yumi Omori, Keith Ross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06793)  

**Abstract**: Inverse Reinforcement Learning (IRL) aims to recover a reward function from expert demonstrations. Recently, Optimal Transport (OT) methods have been successfully deployed to align trajectories and infer rewards. While OT-based methods have shown strong empirical results, they introduce algorithmic complexity, hyperparameter sensitivity, and require solving the OT optimization problems. In this work, we challenge the necessity of OT in IRL by proposing two simple, heuristic alternatives: (1) Minimum-Distance Reward, which assigns rewards based on the nearest expert state regardless of temporal order; and (2) Segment-Matching Reward, which incorporates lightweight temporal alignment by matching agent states to corresponding segments in the expert trajectory. These methods avoid optimization, exhibit linear-time complexity, and are easy to implement. Through extensive evaluations across 32 online and offline benchmarks with three reinforcement learning algorithms, we show that our simple rewards match or outperform recent OT-based approaches. Our findings suggest that the core benefits of OT may arise from basic proximity alignment rather than its optimal coupling formulation, advocating for reevaluation of complexity in future IRL design. 

---
# Feature-Based Instance Neighbor Discovery: Advanced Stable Test-Time Adaptation in Dynamic World 

**Authors**: Qinting Jiang, Chuyang Ye, Dongyan Wei, Bingli Wang, Yuan Xue, Jingyan Jiang, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06782)  

**Abstract**: Despite progress, deep neural networks still suffer performance declines under distribution shifts between training and test domains, leading to a substantial decrease in Quality of Experience (QoE) for applications. Existing test-time adaptation (TTA) methods are challenged by dynamic, multiple test distributions within batches. We observe that feature distributions across different domains inherently cluster into distinct groups with varying means and variances. This divergence reveals a critical limitation of previous global normalization strategies in TTA, which inevitably distort the original data characteristics. Based on this insight, we propose Feature-based Instance Neighbor Discovery (FIND), which comprises three key components: Layer-wise Feature Disentanglement (LFD), Feature Aware Batch Normalization (FABN) and Selective FABN (S-FABN). LFD stably captures features with similar distributions at each layer by constructing graph structures. While FABN optimally combines source statistics with test-time distribution specific statistics for robust feature representation. Finally, S-FABN determines which layers require feature partitioning and which can remain unified, thereby enhancing inference efficiency. Extensive experiments demonstrate that FIND significantly outperforms existing methods, achieving a 30\% accuracy improvement in dynamic scenarios while maintaining computational efficiency. 

---
# Depth-Optimal Quantum Layout Synthesis as SAT 

**Authors**: Anna B. Jakobsen, Anders B. Clausen, Jaco van de Pol, Irfansha Shaik  

**Link**: [PDF](https://arxiv.org/pdf/2506.06752)  

**Abstract**: Quantum circuits consist of gates applied to qubits. Current quantum hardware platforms impose connectivity restrictions on binary CX gates. Hence, Layout Synthesis is an important step to transpile quantum circuits before they can be executed. Since CX gates are noisy, it is important to reduce the CX count or CX depth of the mapped circuits.
We provide a new and efficient encoding of Quantum-circuit Layout Synthesis in SAT. Previous SAT encodings focused on gate count and CX-gate count. Our encoding instead guarantees that we find mapped circuits with minimal circuit depth or minimal CX-gate depth. We use incremental SAT solving and parallel plans for an efficient encoding. This results in speedups of more than 10-100x compared to OLSQ2, which guarantees depth-optimality. But minimizing depth still takes more time than minimizing gate count with Q-Synth.
We correlate the noise reduction achieved by simulating circuits after (CX)-count and (CX)-depth reduction. We find that minimizing for CX-count correlates better with reducing noise than minimizing for CX-depth. However, taking into account both CX-count and CX-depth provides the best noise reduction. 

---
# C-PATH: Conversational Patient Assistance and Triage in Healthcare System 

**Authors**: Qi Shi, Qiwei Han, Cláudia Soares  

**Link**: [PDF](https://arxiv.org/pdf/2506.06737)  

**Abstract**: Navigating healthcare systems can be complex and overwhelming, creating barriers for patients seeking timely and appropriate medical attention. In this paper, we introduce C-PATH (Conversational Patient Assistance and Triage in Healthcare), a novel conversational AI system powered by large language models (LLMs) designed to assist patients in recognizing symptoms and recommending appropriate medical departments through natural, multi-turn dialogues. C-PATH is fine-tuned on medical knowledge, dialogue data, and clinical summaries using a multi-stage pipeline built on the LLaMA3 architecture. A core contribution of this work is a GPT-based data augmentation framework that transforms structured clinical knowledge from DDXPlus into lay-person-friendly conversations, allowing alignment with patient communication norms. We also implement a scalable conversation history management strategy to ensure long-range coherence. Evaluation with GPTScore demonstrates strong performance across dimensions such as clarity, informativeness, and recommendation accuracy. Quantitative benchmarks show that C-PATH achieves superior performance in GPT-rewritten conversational datasets, significantly outperforming domain-specific baselines. C-PATH represents a step forward in the development of user-centric, accessible, and accurate AI tools for digital health assistance and triage. 

---
# Ai-Driven Vulnerability Analysis in Smart Contracts: Trends, Challenges and Future Directions 

**Authors**: Mesut Ozdag  

**Link**: [PDF](https://arxiv.org/pdf/2506.06735)  

**Abstract**: Smart contracts, integral to blockchain ecosystems, enable decentralized applications to execute predefined operations without intermediaries. Their ability to enforce trustless interactions has made them a core component of platforms such as Ethereum. Vulnerabilities such as numerical overflows, reentrancy attacks, and improper access permissions have led to the loss of millions of dollars throughout the blockchain and smart contract sector. Traditional smart contract auditing techniques such as manual code reviews and formal verification face limitations in scalability, automation, and adaptability to evolving development patterns. As a result, AI-based solutions have emerged as a promising alternative, offering the ability to learn complex patterns, detect subtle flaws, and provide scalable security assurances. This paper examines novel AI-driven techniques for vulnerability detection in smart contracts, focusing on machine learning, deep learning, graph neural networks, and transformer-based models. This paper analyzes how each technique represents code, processes semantic information, and responds to real world vulnerability classes. We also compare their strengths and weaknesses in terms of accuracy, interpretability, computational overhead, and real time applicability. Lastly, it highlights open challenges and future opportunities for advancing this domain. 

---
# Neural Spectral Band Generation for Audio Coding 

**Authors**: Woongjib Choi, Byeong Hyeon Kim, Hyungseob Lim, Inseon Jang, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06732)  

**Abstract**: Audio bandwidth extension is the task of reconstructing missing high frequency components of bandwidth-limited audio signals, where bandwidth limitation is a common issue for audio signals due to several reasons, including channel capacity and data constraints. While conventional spectral band replication is a well-established parametric approach to audio bandwidth extension, the SBR usually entails coarse feature extraction and reconstruction techniques, which leads to limitations when processing various types of audio signals. In parallel, numerous deep neural network-based audio bandwidth extension methods have been proposed. These DNN-based methods are usually referred to as blind BWE, as these methods do not rely on prior information extracted from original signals, and only utilize given low frequency band signals to estimate missing high frequency components. In order to replace conventional SBR with DNNs, simply adopting existing DNN-based methodologies results in suboptimal performance due to the blindness of these methods. My proposed research suggests a new approach to parametric non-blind bandwidth extension, as DNN-based side information extraction and DNN-based bandwidth extension are performed only at the front and end of the audio coding pipeline. 

---
# Fuse and Federate: Enhancing EV Charging Station Security with Multimodal Fusion and Federated Learning 

**Authors**: Rabah Rahal, Abdelaziz Amara Korba, Yacine Ghamri-Doudane  

**Link**: [PDF](https://arxiv.org/pdf/2506.06730)  

**Abstract**: The rapid global adoption of electric vehicles (EVs) has established electric vehicle supply equipment (EVSE) as a critical component of smart grid infrastructure. While essential for ensuring reliable energy delivery and accessibility, EVSE systems face significant cybersecurity challenges, including network reconnaissance, backdoor intrusions, and distributed denial-of-service (DDoS) attacks. These emerging threats, driven by the interconnected and autonomous nature of EVSE, require innovative and adaptive security mechanisms that go beyond traditional intrusion detection systems (IDS). Existing approaches, whether network-based or host-based, often fail to detect sophisticated and targeted attacks specifically crafted to exploit new vulnerabilities in EVSE infrastructure. This paper proposes a novel intrusion detection framework that leverages multimodal data sources, including network traffic and kernel events, to identify complex attack patterns. The framework employs a distributed learning approach, enabling collaborative intelligence across EVSE stations while preserving data privacy through federated learning. Experimental results demonstrate that the proposed framework outperforms existing solutions, achieving a detection rate above 98% and a precision rate exceeding 97% in decentralized environments. This solution addresses the evolving challenges of EVSE security, offering a scalable and privacypreserving response to advanced cyber threats 

---
# Improving Wildlife Out-of-Distribution Detection: Africas Big Five 

**Authors**: Mufhumudzi Muthivhi, Jiahao Huo, Fredrik Gustafsson, Terence L. van Zyl  

**Link**: [PDF](https://arxiv.org/pdf/2506.06719)  

**Abstract**: Mitigating human-wildlife conflict seeks to resolve unwanted encounters between these parties. Computer Vision provides a solution to identifying individuals that might escalate into conflict, such as members of the Big Five African animals. However, environments often contain several varied species. The current state-of-the-art animal classification models are trained under a closed-world assumption. They almost always remain overconfident in their predictions even when presented with unknown classes. This study investigates out-of-distribution (OOD) detection of wildlife, specifically the Big Five. To this end, we select a parametric Nearest Class Mean (NCM) and a non-parametric contrastive learning approach as baselines to take advantage of pretrained and projected features from popular classification encoders. Moreover, we compare our baselines to various common OOD methods in the literature. The results show feature-based methods reflect stronger generalisation capability across varying classification thresholds. Specifically, NCM with ImageNet pre-trained features achieves a 2%, 4% and 22% improvement on AUPR-IN, AUPR-OUT and AUTC over the best OOD methods, respectively. The code can be found here this https URL 

---
# DivScore: Zero-Shot Detection of LLM-Generated Text in Specialized Domains 

**Authors**: Zhihui Chen, Kai He, Yucheng Huang, Yunxiao Zhu, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06705)  

**Abstract**: Detecting LLM-generated text in specialized and high-stakes domains like medicine and law is crucial for combating misinformation and ensuring authenticity. However, current zero-shot detectors, while effective on general text, often fail when applied to specialized content due to domain shift. We provide a theoretical analysis showing this failure is fundamentally linked to the KL divergence between human, detector, and source text distributions. To address this, we propose DivScore, a zero-shot detection framework using normalized entropy-based scoring and domain knowledge distillation to robustly identify LLM-generated text in specialized domains. We also release a domain-specific benchmark for LLM-generated text detection in the medical and legal domains. Experiments on our benchmark show that DivScore consistently outperforms state-of-the-art detectors, with 14.4% higher AUROC and 64.0% higher recall (0.1% false positive rate threshold). In adversarial settings, DivScore demonstrates superior robustness than other baselines, achieving on average 22.8% advantage in AUROC and 29.5% in recall. Code and data are publicly available. 

---
# Do Protein Transformers Have Biological Intelligence? 

**Authors**: Fudong Lin, Wanrou Du, Jinchan Liu, Tarikul Milon, Shelby Meche, Wu Xu, Xiaoqi Qin, Xu Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06701)  

**Abstract**: Deep neural networks, particularly Transformers, have been widely adopted for predicting the functional properties of proteins. In this work, we focus on exploring whether Protein Transformers can capture biological intelligence among protein sequences. To achieve our goal, we first introduce a protein function dataset, namely Protein-FN, providing over 9000 protein data with meaningful labels. Second, we devise a new Transformer architecture, namely Sequence Protein Transformers (SPT), for computationally efficient protein function predictions. Third, we develop a novel Explainable Artificial Intelligence (XAI) technique called Sequence Score, which can efficiently interpret the decision-making processes of protein models, thereby overcoming the difficulty of deciphering biological intelligence bided in Protein Transformers. Remarkably, even our smallest SPT-Tiny model, which contains only 5.4M parameters, demonstrates impressive predictive accuracy, achieving 94.3% on the Antibiotic Resistance (AR) dataset and 99.6% on the Protein-FN dataset, all accomplished by training from scratch. Besides, our Sequence Score technique helps reveal that our SPT models can discover several meaningful patterns underlying the sequence structures of protein data, with these patterns aligning closely with the domain knowledge in the biology community. We have officially released our Protein-FN dataset on Hugging Face Datasets this https URL. Our code is available at this https URL. 

---
# MarginSel : Max-Margin Demonstration Selection for LLMs 

**Authors**: Rajeev Bhatt Ambati, James Lester, Shashank Srivastava, Snigdha Chaturvedi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06699)  

**Abstract**: Large Language Models (LLMs) excel at few-shot learning via in-context learning (ICL). However, the effectiveness of ICL is often sensitive to the selection and ordering of demonstration examples. To address this, we present MarginSel: Max-Margin Demonstration Selection for LLMs, a two-step method that selects hard demonstration examples for the ICL prompt, adapting to each test instance. Our approach achieves 2-7% absolute improvement in F1-score across classification tasks, compared to a random selection of examples. We also provide theoretical insights and empirical evidence showing that MarginSel induces max-margin behavior in LLMs by effectively increasing the margin for hard examples, analogous to support vectors, thereby shifting the decision boundary in a beneficial direction. 

---
# Design and Implementation of a RISC-V SoC with Custom DSP Accelerators for Edge Computing 

**Authors**: Priyanshu Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.06693)  

**Abstract**: This paper presents a comprehensive analysis of the RISC-V instruction set architecture, focusing on its modular design, implementation challenges, and performance characteristics. We examine the RV32I base instruction set with extensions for multiplication (M) and atomic operations (A). Through cycle-accurate simulation of a pipelined implementation, we evaluate performance metrics including CPI (cycles per instruction) and power efficiency. Our results demonstrate RISC-V's advantages in embedded systems and its scalability for custom accelerators. Comparative analysis shows a 17% reduction in power consumption compared to ARM Cortex-M0 implementations in similar process nodes. The open-standard nature of RISC-V provides significant flexibility for domain-specific optimizations. 

---
# RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks 

**Authors**: Shiying Duan, Pei Ren, Nanxiang Jiang, Zhengping Che, Jian Tang, Yifan Sun, Zhaoxin Fan, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06683)  

**Abstract**: Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance. 

---
# DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning 

**Authors**: Wenhao Yao, Zhenxin Li, Shiyi Lan, Zi Wang, Xinglong Sun, Jose M. Alvarez, Zuxuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06659)  

**Abstract**: In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios. 

---
# Self-Adapting Improvement Loops for Robotic Learning 

**Authors**: Calvin Luo, Zilai Zeng, Mingxi Jia, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.06658)  

**Abstract**: Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Adapting Improvement Loop (SAIL), where an in-domain video model iteratively updates itself on self-produced trajectories, collected through adaptation with an internet-scale pretrained video model, and steadily improves its performance for a specified task of interest. We apply SAIL to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks initially unseen during original in-domain video model training. Furthermore, we discover that SAIL is surprisingly robust regarding if and how the self-collected experience is filtered, and the quality of the initial in-domain demonstrations. Through adaptation with summarized internet-scale data, and learning through online experience, we thus demonstrate a way to iteratively bootstrap a high-performance video model for solving novel robotic tasks through self-improvement. 

---
# Quantile Regression with Large Language Models for Price Prediction 

**Authors**: Nikhita Vedula, Dushyanta Dhyani, Laleh Jalali, Boris Oreshkin, Mohsen Bayati, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06657)  

**Abstract**: Large Language Models (LLMs) have shown promise in structured prediction tasks, including regression, but existing approaches primarily focus on point estimates and lack systematic comparison across different methods. We investigate probabilistic regression using LLMs for unstructured inputs, addressing challenging text-to-distribution prediction tasks such as price estimation where both nuanced text understanding and uncertainty quantification are critical. We propose a novel quantile regression approach that enables LLMs to produce full predictive distributions, improving upon traditional point estimates. Through extensive experiments across three diverse price prediction datasets, we demonstrate that a Mistral-7B model fine-tuned with quantile heads significantly outperforms traditional approaches for both point and distributional estimations, as measured by three established metrics each for prediction accuracy and distributional calibration. Our systematic comparison of LLM approaches, model architectures, training approaches, and data scaling reveals that Mistral-7B consistently outperforms encoder architectures, embedding-based methods, and few-shot learning methods. Our experiments also reveal the effectiveness of LLM-assisted label correction in achieving human-level accuracy without systematic bias. Our curated datasets are made available at this https URL to support future research. 

---
# Non-Intrusive Load Monitoring Based on Image Load Signatures and Continual Learning 

**Authors**: Olimjon Toirov, Wei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06637)  

**Abstract**: Non-Intrusive Load Monitoring (NILM) identifies the operating status and energy consumption of each electrical device in the circuit by analyzing the electrical signals at the bus, which is of great significance for smart power management. However, the complex and changeable load combinations and application environments lead to the challenges of poor feature robustness and insufficient model generalization of traditional NILM methods. To this end, this paper proposes a new non-intrusive load monitoring method that integrates "image load signature" and continual learning. This method converts multi-dimensional power signals such as current, voltage, and power factor into visual image load feature signatures, and combines deep convolutional neural networks to realize the identification and classification of multiple devices; at the same time, self-supervised pre-training is introduced to improve feature generalization, and continual online learning strategies are used to overcome model forgetting to adapt to the emergence of new loads. This paper conducts a large number of experiments on high-sampling rate load datasets, and compares a variety of existing methods and model variants. The results show that the proposed method has achieved significant improvements in recognition accuracy. 

---
# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning 

**Authors**: Shubham Parashar, Shurui Gui, Xiner Li, Hongyi Ling, Sushil Vemuri, Blake Olson, Eric Li, Yu Zhang, James Caverlee, Dileep Kalathil, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.06632)  

**Abstract**: We aim to improve the reasoning capabilities of language models via reinforcement learning (RL). Recent RL post-trained models like DeepSeek-R1 have demonstrated reasoning abilities on mathematical and coding tasks. However, prior studies suggest that using RL alone to improve reasoning on inherently difficult tasks is less effective. Here, we draw inspiration from curriculum learning and propose to schedule tasks from easy to hard (E2H), allowing LLMs to build reasoning skills gradually. Our method is termed E2H Reasoner. Empirically, we observe that, although easy tasks are important initially, fading them out through appropriate scheduling is essential in preventing overfitting. Theoretically, we establish convergence guarantees for E2H Reasoner within an approximate policy iteration framework. We derive finite-sample complexity bounds and show that when tasks are appropriately decomposed and conditioned, learning through curriculum stages requires fewer total samples than direct learning. Experiments across multiple domains show that E2H Reasoner significantly improves the reasoning ability of small LLMs (1.5B to 3B), which otherwise struggle when trained with vanilla RL alone, highlighting the effectiveness of our method. 

---
# Active Test-time Vision-Language Navigation 

**Authors**: Heeju Ko, Sungjune Kim, Gyeongrok Oh, Jeongyoon Yoon, Honglak Lee, Sujin Jang, Seungryong Kim, Sangpil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.06630)  

**Abstract**: Vision-Language Navigation (VLN) policies trained on offline datasets often exhibit degraded task performance when deployed in unfamiliar navigation environments at test time, where agents are typically evaluated without access to external interaction or feedback. Entropy minimization has emerged as a practical solution for reducing prediction uncertainty at test time; however, it can suffer from accumulated errors, as agents may become overconfident in incorrect actions without sufficient contextual grounding. To tackle these challenges, we introduce ATENA (Active TEst-time Navigation Agent), a test-time active learning framework that enables a practical human-robot interaction via episodic feedback on uncertain navigation outcomes. In particular, ATENA learns to increase certainty in successful episodes and decrease it in failed ones, improving uncertainty calibration. Here, we propose mixture entropy optimization, where entropy is obtained from a combination of the action and pseudo-expert distributions-a hypothetical action distribution assuming the agent's selected action to be optimal-controlling both prediction confidence and action preference. In addition, we propose a self-active learning strategy that enables an agent to evaluate its navigation outcomes based on confident predictions. As a result, the agent stays actively engaged throughout all iterations, leading to well-grounded and adaptive decision-making. Extensive evaluations on challenging VLN benchmarks-REVERIE, R2R, and R2R-CE-demonstrate that ATENA successfully overcomes distributional shifts at test time, outperforming the compared baseline methods across various settings. 

---
# \textit{QuantMCP}: Grounding Large Language Models in Verifiable Financial Reality 

**Authors**: Yifan Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.06622)  

**Abstract**: Large Language Models (LLMs) hold immense promise for revolutionizing financial analysis and decision-making, yet their direct application is often hampered by issues of data hallucination and lack of access to real-time, verifiable financial information. This paper introduces QuantMCP, a novel framework designed to rigorously ground LLMs in financial reality. By leveraging the Model Context Protocol (MCP) for standardized and secure tool invocation, QuantMCP enables LLMs to accurately interface with a diverse array of Python-accessible financial data APIs (e.g., Wind, yfinance). Users can interact via natural language to precisely retrieve up-to-date financial data, thereby overcoming LLM's inherent limitations in factual data recall. More critically, once furnished with this verified, structured data, the LLM's analytical capabilities are unlocked, empowering it to perform sophisticated data interpretation, generate insights, and ultimately support more informed financial decision-making processes. QuantMCP provides a robust, extensible, and secure bridge between conversational AI and the complex world of financial data, aiming to enhance both the reliability and the analytical depth of LLM applications in finance. 

---
# Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit 

**Authors**: Charles Goddard, Fernando Fernandes Neto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06607)  

**Abstract**: We present a training-free method to transplant tokenizers in pretrained large language models (LLMs) by reconstructing unseen token embeddings via Orthogonal Matching Pursuit (OMP). Specifically, we approximate each out-of-vocabulary token as a sparse linear combination of shared tokens, in two phases: first, compute each new token's representation in the donor embedding space with a small dictionary of shared anchor tokens, then transfer these same sparse coefficients back into the base model's embedding space.
On two challenging cross-tokenizer tasks--Llama$\to$Mistral NeMo (12B) and Qwen$\to$Llama (1B)--we show that OMP achieves best zero-shot preservation of the base model's performance across multiple benchmarks, while other zero-shot approaches degrade significantly. Compared to baselines (zero-init, mean-init, and existing approaches like WECHSEL, FOCUS, ZETT), OMP consistently achieves the best overall performance, effectively bridging large tokenizer discrepancies without gradient updates. Our analysis further identifies mismatched numerical tokenization schemes as a critical challenge for preserving mathematical reasoning capabilities. This technique enables direct reuse of pretrained model weights with new tokenizers, facilitating cross-tokenizer knowledge distillation, speculative decoding, ensembling, merging, and domain-specific vocabulary adaptations. We integrate our method into the open-source mergekit-tokensurgeon tool for post hoc vocabulary realignment. 

---
# MedCite: Can Language Models Generate Verifiable Text for Medicine? 

**Authors**: Xiao Wang, Mengjue Tan, Qiao Jin, Guangzhi Xiong, Yu Hu, Aidong Zhang, Zhiyong Lu, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06605)  

**Abstract**: Existing LLM-based medical question-answering systems lack citation generation and evaluation capabilities, raising concerns about their adoption in practice. In this work, we introduce \name, the first end-to-end framework that facilitates the design and evaluation of citation generation with LLMs for medical tasks. Meanwhile, we introduce a novel multi-pass retrieval-citation method that generates high-quality citations. Our evaluation highlights the challenges and opportunities of citation generation for medical tasks, while identifying important design choices that have a significant impact on the final citation quality. Our proposed method achieves superior citation precision and recall improvements compared to strong baseline methods, and we show that evaluation results correlate well with annotation results from professional experts. 

---
# CAtCh: Cognitive Assessment through Cookie Thief 

**Authors**: Joseph T Colonel, Carolyn Hagler, Guiselle Wismer, Laura Curtis, Jacqueline Becker, Juan Wisnivesky, Alex Federman, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.06603)  

**Abstract**: Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at this https URL. 

---
# From Model-Based and Adaptive Control to Evolving Fuzzy Control 

**Authors**: Daniel Leite, Igor Škrjanc, Fernando Gomide  

**Link**: [PDF](https://arxiv.org/pdf/2506.06594)  

**Abstract**: Evolving fuzzy systems build and adapt fuzzy models - such as predictors and controllers - by incrementally updating their rule-base structure from data streams. On the occasion of the 60-year anniversary of fuzzy set theory, commemorated during the Fuzz-IEEE 2025 event, this brief paper revisits the historical development and core contributions of classical fuzzy and adaptive modeling and control frameworks. It then highlights the emergence and significance of evolving intelligent systems in fuzzy modeling and control, emphasizing their advantages in handling nonstationary environments. Key challenges and future directions are discussed, including safety, interpretability, and principled structural evolution. 

---
# Towards Efficient Multi-LLM Inference: Characterization and Analysis of LLM Routing and Hierarchical Techniques 

**Authors**: Adarsh Prasad Behera, Jaya Prakash Champati, Roberto Morabito, Sasu Tarkoma, James Gross  

**Link**: [PDF](https://arxiv.org/pdf/2506.06579)  

**Abstract**: Recent progress in Language Models (LMs) has dramatically advanced the field of natural language processing (NLP), excelling at tasks like text generation, summarization, and question answering. However, their inference remains computationally expensive and energy intensive, especially in settings with limited hardware, power, or bandwidth. This makes it difficult to deploy LMs in mobile, edge, or cost sensitive environments. To address these challenges, recent approaches have introduced multi LLM intelligent model selection strategies that dynamically allocate computational resources based on query complexity -- using lightweight models for simpler queries and escalating to larger models only when necessary. This survey explores two complementary strategies for efficient LLM inference: (i) routing, which selects the most suitable model based on the query, and (ii) cascading or hierarchical inference (HI), which escalates queries through a sequence of models until a confident response is found. Both approaches aim to reduce computation by using lightweight models for simpler tasks while offloading only when needed. We provide a comparative analysis of these techniques across key performance metrics, discuss benchmarking efforts, and outline open challenges. Finally, we outline future research directions to enable faster response times, adaptive model selection based on task complexity, and scalable deployment across heterogeneous environments, making LLM based systems more efficient and accessible for real world applications. 

---
# Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce 

**Authors**: Yijia Shao, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06576)  

**Abstract**: The rapid rise of compound AI systems (a.k.a., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. Yet, we lack a systematic understanding of the evolving landscape. In this paper, we address this gap by introducing a novel auditing framework to assess which occupational tasks workers want AI agents to automate or augment, and how those desires align with the current technological capabilities. Our framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, we construct the WORKBank database, building on the U.S. Department of Labor's O*NET database, to capture preferences from 1,500 domain workers and capability assessments from AI experts across over 844 tasks spanning 104 occupations. Jointly considering the desire and technological capability divides tasks in WORKBank into four zones: Automation "Green Light" Zone, Automation "Red Light" Zone, R&D Opportunity Zone, Low Priority Zone. This highlights critical mismatches and opportunities for AI agent development. Moving beyond a simple automate-or-not dichotomy, our results reveal diverse HAS profiles across occupations, reflecting heterogeneous expectations for human involvement. Moreover, our study offers early signals of how AI agent integration may reshape the core human competencies, shifting from information-focused skills to interpersonal ones. These findings underscore the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics. 

---
# Graph Persistence goes Spectral 

**Authors**: Mattie Ji, Amauri H. Souza, Vikas Garg  

**Link**: [PDF](https://arxiv.org/pdf/2506.06571)  

**Abstract**: Including intricate topological information (e.g., cycles) provably enhances the expressivity of message-passing graph neural networks (GNNs) beyond the Weisfeiler-Leman (WL) hierarchy. Consequently, Persistent Homology (PH) methods are increasingly employed for graph representation learning. In this context, recent works have proposed decorating classical PH diagrams with vertex and edge features for improved expressivity. However, due to their dependence on features, these methods still fail to capture basic graph structural information. In this paper, we propose SpectRe -- a new topological descriptor for graphs that integrates spectral information into PH diagrams. Notably, SpectRe is strictly more expressive than existing descriptors on graphs. We also introduce notions of global and local stability to analyze existing descriptors and establish that SpectRe is locally stable. Finally, experiments on synthetic and real-world datasets demonstrate the effectiveness of SpectRe and its potential to enhance the capabilities of graph models in relevant learning tasks. 

---
# Textile Analysis for Recycling Automation using Transfer Learning and Zero-Shot Foundation Models 

**Authors**: Yannis Spyridis, Vasileios Argyriou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06569)  

**Abstract**: Automated sorting is crucial for improving the efficiency and scalability of textile recycling, but accurately identifying material composition and detecting contaminants from sensor data remains challenging. This paper investigates the use of standard RGB imagery, a cost-effective sensing modality, for key pre-processing tasks in an automated system. We present computer vision components designed for a conveyor belt setup to perform (a) classification of four common textile types and (b) segmentation of non-textile features such as buttons and zippers. For classification, several pre-trained architectures were evaluated using transfer learning and cross-validation, with EfficientNetB0 achieving the best performance on a held-out test set with 81.25\% accuracy. For feature segmentation, a zero-shot approach combining the Grounding DINO open-vocabulary detector with the Segment Anything Model (SAM) was employed, demonstrating excellent performance with a mIoU of 0.90 for the generated masks against ground truth. This study demonstrates the feasibility of using RGB images coupled with modern deep learning techniques, including transfer learning for classification and foundation models for zero-shot segmentation, to enable essential analysis steps for automated textile recycling pipelines. 

---
# AS-ASR: A Lightweight Framework for Aphasia-Specific Automatic Speech Recognition 

**Authors**: Chen Bao, Chuanbing Huo, Qinyu Chen, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06566)  

**Abstract**: This paper proposes AS-ASR, a lightweight aphasia-specific speech recognition framework based on Whisper-tiny, tailored for low-resource deployment on edge devices. Our approach introduces a hybrid training strategy that systematically combines standard and aphasic speech at varying ratios, enabling robust generalization, and a GPT-4-based reference enhancement method that refines noisy aphasic transcripts, improving supervision quality. We conduct extensive experiments across multiple data mixing configurations and evaluation settings. Results show that our fine-tuned model significantly outperforms the zero-shot baseline, reducing WER on aphasic speech by over 30% while preserving performance on standard speech. The proposed framework offers a scalable, efficient solution for real-world disordered speech recognition. 

---
# LaMP-Cap: Personalized Figure Caption Generation With Multimodal Figure Profiles 

**Authors**: Ho Yin 'Sam' Ng, Ting-Yao Hsu, Aashish Anantha Ramakrishnan, Branislav Kveton, Nedim Lipka, Franck Dernoncourt, Dongwon Lee, Tong Yu, Sungchul Kim, Ryan A. Rossi, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06561)  

**Abstract**: Figure captions are crucial for helping readers understand and remember a figure's key message. Many models have been developed to generate these captions, helping authors compose better quality captions more easily. Yet, authors almost always need to revise generic AI-generated captions to match their writing style and the domain's style, highlighting the need for personalization. Despite language models' personalization (LaMP) advances, these technologies often focus on text-only settings and rarely address scenarios where both inputs and profiles are multimodal. This paper introduces LaMP-Cap, a dataset for personalized figure caption generation with multimodal figure profiles. For each target figure, LaMP-Cap provides not only the needed inputs, such as figure images, but also up to three other figures from the same document--each with its image, caption, and figure-mentioning paragraphs--as a profile to characterize the context. Experiments with four LLMs show that using profile information consistently helps generate captions closer to the original author-written ones. Ablation studies reveal that images in the profile are more helpful than figure-mentioning paragraphs, highlighting the advantage of using multimodal profiles over text-only ones. 

---
# KramaBench: A Benchmark for AI Systems on Data-to-Insight Pipelines over Data Lakes 

**Authors**: Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Sivaprasad Sudhir, Om Chabra, Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J. Cafarella, Lei Cao, Samuel Madden, Tim Kraska  

**Link**: [PDF](https://arxiv.org/pdf/2506.06541)  

**Abstract**: Constructing real-world data-to-insight pipelines often involves data extraction from data lakes, data integration across heterogeneous data sources, and diverse operations from data cleaning to analysis. The design and implementation of data science pipelines require domain knowledge, technical expertise, and even project-specific insights. AI systems have shown remarkable reasoning, coding, and understanding capabilities. However, it remains unclear to what extent these capabilities translate into successful design and execution of such complex pipelines. We introduce KRAMABENCH: a benchmark composed of 104 manually-curated real-world data science pipelines spanning 1700 data files from 24 data sources in 6 different domains. We show that these pipelines test the end-to-end capabilities of AI systems on data processing, requiring data discovery, wrangling and cleaning, efficient processing, statistical reasoning, and orchestrating data processing steps given a high-level task. Our evaluation tests 5 general models and 3 code generation models using our reference framework, DS-GURU, which instructs the AI model to decompose a question into a sequence of subtasks, reason through each step, and synthesize Python code that implements the proposed design. Our results on KRAMABENCH show that, although the models are sufficiently capable of solving well-specified data science code generation tasks, when extensive data processing and domain knowledge are required to construct real-world data science pipelines, existing out-of-box models fall short. Progress on KramaBench represents crucial steps towards developing autonomous data science agents for real-world applications. Our code, reference framework, and data are available at this https URL. 

---
# Large Language Models Can Be a Viable Substitute for Expert Political Surveys When a Shock Disrupts Traditional Measurement Approaches 

**Authors**: Patrick Y. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06540)  

**Abstract**: After a disruptive event or shock, such as the Department of Government Efficiency (DOGE) federal layoffs of 2025, expert judgments are colored by knowledge of the outcome. This can make it difficult or impossible to reconstruct the pre-event perceptions needed to study the factors associated with the event. This position paper argues that large language models (LLMs), trained on vast amounts of digital media data, can be a viable substitute for expert political surveys when a shock disrupts traditional measurement. We analyze the DOGE layoffs as a specific case study for this position. We use pairwise comparison prompts with LLMs and derive ideology scores for federal executive agencies. These scores replicate pre-layoff expert measures and predict which agencies were targeted by DOGE. We also use this same approach and find that the perceptions of certain federal agencies as knowledge institutions predict which agencies were targeted by DOGE, even when controlling for ideology. This case study demonstrates that using LLMs allows us to rapidly and easily test the associated factors hypothesized behind the shock. More broadly, our case study of this recent event exemplifies how LLMs offer insights into the correlational factors of the shock when traditional measurement techniques fail. We conclude by proposing a two-part criterion for when researchers can turn to LLMs as a substitute for expert political surveys. 

---
# Beyond Facts: Evaluating Intent Hallucination in Large Language Models 

**Authors**: Yijie Hao, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.06539)  

**Abstract**: When exposed to complex queries containing multiple conditions, today's large language models (LLMs) tend to produce responses that only partially satisfy the query while neglecting certain conditions. We therefore introduce the concept of Intent Hallucination. In this phenomenon, LLMs either omit (neglecting to address certain parts) or misinterpret (responding to invented query parts) elements of the given query, leading to intent hallucinated generation. To systematically evaluate intent hallucination, we introduce FAITHQA, a novel benchmark for intent hallucination that contains 20,068 problems, covering both query-only and retrieval-augmented generation (RAG) setups with varying topics and difficulty. FAITHQA is the first hallucination benchmark that goes beyond factual verification, tailored to identify the fundamental cause of intent hallucination. By evaluating various LLMs on FAITHQA, we find that (1) intent hallucination is a common issue even for state-of-the-art models, and (2) the phenomenon stems from omission or misinterpretation of LLMs. To facilitate future research, we introduce an automatic LLM generation evaluation metric, CONSTRAINT SCORE, for detecting intent hallucination. Human evaluation results demonstrate that CONSTRAINT SCORE is closer to human performance for intent hallucination compared to baselines. 

---
# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks 

**Authors**: Zijiang Yan, Hao Zhou, Jianhua Pei, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06532)  

**Abstract**: Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches. 

---
# Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Syed Zawad, Farhan Ahmed, Heiko Ludwig, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2506.06522)  

**Abstract**: Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture. 

---
# Private GPTs for LLM-driven testing in software development and machine learning 

**Authors**: Jakub Jagielski, Markus Abel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06509)  

**Abstract**: In this contribution, we examine the capability of private GPTs to automatically generate executable test code based on requirements. More specifically, we use acceptance criteria as input, formulated as part of epics, or stories, which are typically used in modern development processes. This gives product owners, or business intelligence, respectively, a way to directly produce testable criteria through the use of LLMs. We explore the quality of the so-produced tests in two ways: i) directly by letting the LLM generate code from requirements, ii) through an intermediate step using Gherkin syntax. As a result, it turns out that the two-step procedure yields better results -where we define better in terms of human readability and best coding practices, i.e. lines of code and use of additional libraries typically used in testing. Concretely, we evaluate prompt effectiveness across two scenarios: a simple "Hello World" program and a digit classification model, showing that structured prompts lead to higher-quality test outputs. 

---
# Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms 

**Authors**: Alex Havrilla, Edward Hughes, Mikayel Samvelyan, Jacob Abernethy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06499)  

**Abstract**: Large language model (LLM) driven synthetic data generation has emerged as a powerful method for improving model reasoning capabilities. However, most methods either distill large state-of-the-art models into small students or use natural ground-truth problem statements to guarantee problem statement quality. This limits the scalability of these approaches to more complex and diverse problem domains. To address this, we present SPARQ: Synthetic Problem Generation for Reasoning via Quality-Diversity Algorithms, a novel approach for generating high-quality and diverse synthetic math problem and solution pairs using only a single model by measuring a problem's solve-rate: a proxy for problem difficulty. Starting from a seed dataset of 7.5K samples, we generate over 20 million new problem-solution pairs. We show that filtering the generated data by difficulty and then fine-tuning the same model on the resulting data improves relative model performance by up to 24\%. Additionally, we conduct ablations studying the impact of synthetic data quantity, quality and diversity on model generalization. We find that higher quality, as measured by problem difficulty, facilitates better in-distribution performance. Further, while generating diverse synthetic data does not as strongly benefit in-distribution performance, filtering for more diverse data facilitates more robust OOD generalization. We also confirm the existence of model and data scaling laws for synthetically generated problems, which positively benefit downstream model generalization. 

---
# What Is Seen Cannot Be Unseen: The Disruptive Effect of Knowledge Conflict on Large Language Models 

**Authors**: Kaiser Sun, Fan Bai, Mark Dredze  

**Link**: [PDF](https://arxiv.org/pdf/2506.06485)  

**Abstract**: Large language models frequently rely on both contextual input and parametric knowledge to perform tasks. However, these sources can come into conflict, especially when retrieved documents contradict the model's parametric knowledge. We propose a diagnostic framework to systematically evaluate LLM behavior under context-memory conflict, where the contextual information diverges from their parametric beliefs. We construct diagnostic data that elicit these conflicts and analyze model performance across multiple task types. Our findings reveal that (1) knowledge conflict has minimal impact on tasks that do not require knowledge utilization, (2) model performance is consistently higher when contextual and parametric knowledge are aligned, (3) models are unable to fully suppress their internal knowledge even when instructed, and (4) providing rationales that explain the conflict increases reliance on contexts. These insights raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs. 

---
# The Economic Dispatch of Power-to-Gas Systems with Deep Reinforcement Learning:Tackling the Challenge of Delayed Rewards with Long-Term Energy Storage 

**Authors**: Manuel Sage, Khalil Al Handawi, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06484)  

**Abstract**: Power-to-Gas (P2G) technologies gain recognition for enabling the integration of intermittent renewables, such as wind and solar, into electricity grids. However, determining the most cost-effective operation of these systems is complex due to the volatile nature of renewable energy, electricity prices, and loads. Additionally, P2G systems are less efficient in converting and storing energy compared to battery energy storage systems (BESs), and the benefits of converting electricity into gas are not immediately apparent. Deep Reinforcement Learning (DRL) has shown promise in managing the operation of energy systems amidst these uncertainties. Yet, DRL techniques face difficulties with the delayed reward characteristic of P2G system operation. Previous research has mostly focused on short-term studies that look at the energy conversion process, neglecting the long-term storage capabilities of P2G.
This study presents a new method by thoroughly examining how DRL can be applied to the economic operation of P2G systems, in combination with BESs and gas turbines, over extended periods. Through three progressively more complex case studies, we assess the performance of DRL algorithms, specifically Deep Q-Networks and Proximal Policy Optimization, and introduce modifications to enhance their effectiveness. These modifications include integrating forecasts, implementing penalties on the reward function, and applying strategic cost calculations, all aimed at addressing the issue of delayed rewards. Our findings indicate that while DRL initially struggles with the complex decision-making required for P2G system operation, the adjustments we propose significantly improve its capability to devise cost-effective operation strategies, thereby unlocking the potential for long-term energy storage in P2G technologies. 

---
# Noise Consistency Regularization for Improved Subject-Driven Image Synthesis 

**Authors**: Yao Ni, Song Wen, Piotr Koniusz, Anoop Cherian  

**Link**: [PDF](https://arxiv.org/pdf/2506.06483)  

**Abstract**: Fine-tuning Stable Diffusion enables subject-driven image synthesis by adapting the model to generate images containing specific subjects. However, existing fine-tuning methods suffer from two key issues: underfitting, where the model fails to reliably capture subject identity, and overfitting, where it memorizes the subject image and reduces background diversity. To address these challenges, we propose two auxiliary consistency losses for diffusion fine-tuning. First, a prior consistency regularization loss ensures that the predicted diffusion noise for prior (non-subject) images remains consistent with that of the pretrained model, improving fidelity. Second, a subject consistency regularization loss enhances the fine-tuned model's robustness to multiplicative noise modulated latent code, helping to preserve subject identity while improving diversity. Our experimental results demonstrate that incorporating these losses into fine-tuning not only preserves subject identity but also enhances image diversity, outperforming DreamBooth in terms of CLIP scores, background variation, and overall visual quality. 

---
# Edge-Enabled Collaborative Object Detection for Real-Time Multi-Vehicle Perception 

**Authors**: Everett Richards, Bipul Thapa, Lena Mashayekhy  

**Link**: [PDF](https://arxiv.org/pdf/2506.06474)  

**Abstract**: Accurate and reliable object detection is critical for ensuring the safety and efficiency of Connected Autonomous Vehicles (CAVs). Traditional on-board perception systems have limited accuracy due to occlusions and blind spots, while cloud-based solutions introduce significant latency, making them unsuitable for real-time processing demands required for autonomous driving in dynamic environments. To address these challenges, we introduce an innovative framework, Edge-Enabled Collaborative Object Detection (ECOD) for CAVs, that leverages edge computing and multi-CAV collaboration for real-time, multi-perspective object detection. Our ECOD framework integrates two key algorithms: Perceptive Aggregation and Collaborative Estimation (PACE) and Variable Object Tally and Evaluation (VOTE). PACE aggregates detection data from multiple CAVs on an edge server to enhance perception in scenarios where individual CAVs have limited visibility. VOTE utilizes a consensus-based voting mechanism to improve the accuracy of object classification by integrating data from multiple CAVs. Both algorithms are designed at the edge to operate in real-time, ensuring low-latency and reliable decision-making for CAVs. We develop a hardware-based controlled testbed consisting of camera-equipped robotic CAVs and an edge server to evaluate the efficacy of our framework. Our experimental results demonstrate the significant benefits of ECOD in terms of improved object classification accuracy, outperforming traditional single-perspective onboard approaches by up to 75%, while ensuring low-latency, edge-driven real-time processing. This research highlights the potential of edge computing to enhance collaborative perception for latency-sensitive autonomous systems. 

---
# Cost-Efficient LLM Training with Lifetime-Aware Tensor Offloading via GPUDirect Storage 

**Authors**: Ziqi Yuan, Haoyang Zhang, Yirui Eric Zhou, Apoorve Mohan, I-Hsin Chung, Seetharami Seelam, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06472)  

**Abstract**: We present the design and implementation of a new lifetime-aware tensor offloading framework for GPU memory expansion using low-cost PCIe-based solid-state drives (SSDs). Our framework, TERAIO, is developed explicitly for large language model (LLM) training with multiple GPUs and multiple SSDs. Its design is driven by our observation that the active tensors take only a small fraction (1.7% on average) of allocated GPU memory in each LLM training iteration, the inactive tensors are usually large and will not be used for a long period of time, creating ample opportunities for offloading/prefetching tensors to/from slow SSDs without stalling the GPU training process. TERAIO accurately estimates the lifetime (active period of time in GPU memory) of each tensor with the profiling of the first few iterations in the training process. With the tensor lifetime analysis, TERAIO will generate an optimized tensor offloading/prefetching plan and integrate it into the compiled LLM program via PyTorch. TERAIO has a runtime tensor migration engine to execute the offloading/prefetching plan via GPUDirect storage, which allows direct tensor migration between GPUs and SSDs for alleviating the CPU bottleneck and maximizing the SSD bandwidth utilization. In comparison with state-of-the-art studies such as ZeRO-Offload and ZeRO-Infinity, we show that TERAIO improves the training performance of various LLMs by 1.47x on average, and achieves 80.7% of the ideal performance assuming unlimited GPU memory. 

---
# WISCA: A Consensus-Based Approach to Harmonizing Interpretability in Tabular Datasets 

**Authors**: Antonio Jesús Banegas-Luna, Horacio Pérez-Sánchez, Carlos Martínez-Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2506.06455)  

**Abstract**: While predictive accuracy is often prioritized in machine learning (ML) models, interpretability remains essential in scientific and high-stakes domains. However, diverse interpretability algorithms frequently yield conflicting explanations, highlighting the need for consensus to harmonize results. In this study, six ML models were trained on six synthetic datasets with known ground truths, utilizing various model-agnostic interpretability techniques. Consensus explanations were generated using established methods and a novel approach: WISCA (Weighted Scaled Consensus Attributions), which integrates class probability and normalized attributions. WISCA consistently aligned with the most reliable individual method, underscoring the value of robust consensus strategies in improving explanation reliability. 

---
# Canonical Autoregressive Generation 

**Authors**: Ivi Chatzi, Nina Corvelo Benz, Stratis Tsirtsis, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.06446)  

**Abstract**: State of the art large language models are trained using large amounts of tokens derived from raw text using what is called a tokenizer. Crucially, the tokenizer determines the (token) vocabulary a model will use during inference as well as, in principle, the (token) language. This is because, while the token vocabulary may allow for different tokenizations of a string, the tokenizer always maps the string to only one of these tokenizations--the canonical tokenization. However, multiple lines of empirical evidence suggest that large language models do not always generate canonical token sequences, and this comes with several negative consequences. In this work, we first show that, to generate a canonical token sequence, a model needs to generate (partial) canonical token sequences at each step of the autoregressive generation process underpinning its functioning. Building upon this theoretical result, we introduce canonical sampling, a simple and efficient sampling method that precludes a given model from generating non-canonical token sequences. Further, we also show that, in comparison with standard sampling, the distribution of token sequences generated using canonical sampling is provably closer to the true distribution of token sequences used during training. 

---
# Saffron-1: Towards an Inference Scaling Paradigm for LLM Safety Assurance 

**Authors**: Ruizhong Qiu, Gaotang Li, Tianxin Wei, Jingrui He, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06444)  

**Abstract**: Existing safety assurance research has primarily focused on training-phase alignment to instill safe behaviors into LLMs. However, recent studies have exposed these methods' susceptibility to diverse jailbreak attacks. Concurrently, inference scaling has significantly advanced LLM reasoning capabilities but remains unexplored in the context of safety assurance. Addressing this gap, our work pioneers inference scaling for robust and effective LLM safety against emerging threats. We reveal that conventional inference scaling techniques, despite their success in reasoning tasks, perform poorly in safety contexts, even falling short of basic approaches like Best-of-N Sampling. We attribute this inefficiency to a newly identified challenge, the exploration--efficiency dilemma, arising from the high computational overhead associated with frequent process reward model (PRM) evaluations. To overcome this dilemma, we propose SAFFRON, a novel inference scaling paradigm tailored explicitly for safety assurance. Central to our approach is the introduction of a multifurcation reward model (MRM) that significantly reduces the required number of reward model evaluations. To operationalize this paradigm, we further propose: (i) a partial supervision training objective for MRM, (ii) a conservative exploration constraint to prevent out-of-distribution explorations, and (iii) a Trie-based key--value caching strategy that facilitates cache sharing across sequences during tree search. Extensive experiments validate the effectiveness of our method. Additionally, we publicly release our trained multifurcation reward model (Saffron-1) and the accompanying token-level safety reward dataset (Safety4M) to accelerate future research in LLM safety. Our code, model, and data are publicly available at this https URL , and our project homepage is at this https URL . 

---
# Unlocking Chemical Insights: Superior Molecular Representations from Intermediate Encoder Layers 

**Authors**: Luis Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2506.06443)  

**Abstract**: Pretrained molecular encoders have become indispensable in computational chemistry for tasks such as property prediction and molecular generation. However, the standard practice of relying solely on final-layer embeddings for downstream tasks may discard valuable information. In this work, we challenge this convention by conducting a comprehensive layer-wise analysis of five diverse molecular encoders across 22 ADMET property prediction tasks. Our results demonstrate that embeddings from intermediate layers consistently outperform final-layer representations. Specifically, using fixed embeddings from the optimal intermediate layers improved downstream performance by an average of 5.4%, reaching gains up to 28.6%. Furthermore, finetuning up to these intermediate layers yielded even greater average improvements of 8.5%, with performance increases as high as 40.8%, achieving new state-of-the-art results on several benchmarks. Additionally, a strong positive correlation between fixed embedding performance and finetuning outcomes supports an efficient evaluate-then-finetune approach, enabling identification of optimal layers with reduced computational cost. These findings highlight the importance of exploring the full representational depth of molecular encoders to achieve substantial performance improvements and computational efficiency. The code is made publicly available at this https URL. 

---
# Benchmarking Misuse Mitigation Against Covert Adversaries 

**Authors**: Davis Brown, Mahdi Sabbaghi, Luze Sun, Alexander Robey, George J. Pappas, Eric Wong, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.06414)  

**Abstract**: Existing language model safety evaluations focus on overt attacks and low-stakes tasks. Realistic attackers can subvert current safeguards by requesting help on small, benign-seeming tasks across many independent queries. Because individual queries do not appear harmful, the attack is hard to {detect}. However, when combined, these fragments uplift misuse by helping the attacker complete hard and dangerous tasks. Toward identifying defenses against such strategies, we develop Benchmarks for Stateful Defenses (BSD), a data generation pipeline that automates evaluations of covert attacks and corresponding defenses. Using this pipeline, we curate two new datasets that are consistently refused by frontier models and are too difficult for weaker open-weight models. Our evaluations indicate that decomposition attacks are effective misuse enablers, and highlight stateful defenses as a countermeasure. 

---
# HeavyWater and SimplexWater: Watermarking Low-Entropy Text Distributions 

**Authors**: Dor Tsur, Carol Xuan Long, Claudio Mayrink Verdun, Hsiang Hsu, Chen-Fu Chen, Haim Permuter, Sajani Vithana, Flavio P. Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06409)  

**Abstract**: Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks - such as coding - where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in this https URL 

---
# TimeWak: Temporal Chained-Hashing Watermark for Time Series Data 

**Authors**: Zhi Wen Soi, Chaoyi Zhu, Fouad Abiad, Aditya Shankar, Jeroen M. Galjaard, Huijuan Wang, Lydia Y. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.06407)  

**Abstract**: Synthetic time series generated by diffusion models enable sharing privacy-sensitive datasets, such as patients' functional MRI records. Key criteria for synthetic data include high data utility and traceability to verify the data source. Recent watermarking methods embed in homogeneous latent spaces, but state-of-the-art time series generators operate in real space, making latent-based watermarking incompatible. This creates the challenge of watermarking directly in real space while handling feature heterogeneity and temporal dependencies. We propose TimeWak, the first watermarking algorithm for multivariate time series diffusion models. To handle temporal dependence and spatial heterogeneity, TimeWak embeds a temporal chained-hashing watermark directly within the real temporal-feature space. The other unique feature is the $\epsilon$-exact inversion, which addresses the non-uniform reconstruction error distribution across features from inverting the diffusion process to detect watermarks. We derive the error bound of inverting multivariate time series and further maintain high watermark detectability. We extensively evaluate TimeWak on its impact on synthetic data quality, watermark detectability, and robustness under various post-editing attacks, against 5 datasets and baselines of different temporal lengths. Our results show that TimeWak achieves improvements of 61.96% in context-FID score, and 8.44% in correlational scores against the state-of-the-art baseline, while remaining consistently detectable. 

---
# SMAR: Soft Modality-Aware Routing Strategy for MoE-based Multimodal Large Language Models Preserving Language Capabilities 

**Authors**: Guoyang Xia, Yifeng Ding, Fengfa Li, Lei Ren, Chen Wei, Fangxiang Feng, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06406)  

**Abstract**: Mixture of Experts (MoE) architectures have become a key approach for scaling large language models, with growing interest in extending them to multimodal tasks. Existing methods to build multimodal MoE models either incur high training costs or suffer from degraded language capabilities when adapting pretrained models. To address this, we propose Soft ModalityAware Routing (SMAR), a novel regularization technique that uses Kullback Leibler divergence to control routing probability distributions across modalities, encouraging expert specialization without modifying model architecture or heavily relying on textual data. Experiments on visual instruction tuning show that SMAR preserves language ability at 86.6% retention with only 2.5% pure text, outperforming baselines while maintaining strong multimodal performance. Our approach offers a practical and efficient solution to balance modality differentiation and language capabilities in multimodal MoE models. 

---
# Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights 

**Authors**: Sooyung Choi, Jaehyeok Lee, Xiaoyuan Yi, Jing Yao, Xing Xie, JinYeong Bak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06404)  

**Abstract**: The application scope of Large Language Models (LLMs) continues to expand, leading to increasing interest in personalized LLMs that align with human values. However, aligning these models with individual values raises significant safety concerns, as certain values may correlate with harmful information. In this paper, we identify specific safety risks associated with value-aligned LLMs and investigate the psychological principles behind these challenges. Our findings reveal two key insights. (1) Value-aligned LLMs are more prone to harmful behavior compared to non-fine-tuned models and exhibit slightly higher risks in traditional safety evaluations than other fine-tuned models. (2) These safety issues arise because value-aligned LLMs genuinely generate text according to the aligned values, which can amplify harmful outcomes. Using a dataset with detailed safety categories, we find significant correlations between value alignment and safety risks, supported by psychological hypotheses. This study offers insights into the "black box" of value alignment and proposes in-context alignment methods to enhance the safety of value-aligned LLMs. 

---
# Direct Behavior Optimization: Unlocking the Potential of Lightweight LLMs 

**Authors**: Hongming Yang, Shi Lin, Jun Shao, Changting Lin, Donghai Zhu, Meng Han, Qinglei Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.06401)  

**Abstract**: Lightweight Large Language Models (LwLLMs) are reduced-parameter, optimized models designed to run efficiently on consumer-grade hardware, offering significant advantages in resource efficiency, cost-effectiveness, and data privacy. However, these models often struggle with limited inference and reasoning capabilities, which restrict their performance on complex tasks and limit their practical applicability. Moreover, existing prompt optimization methods typically rely on extensive manual effort or the meta-cognitive abilities of state-of-the-art LLMs, making them less effective for LwLLMs. To address these challenges, we introduce DeBoP, a new Direct Behavior Optimization Paradigm, original from the Chain-of-Thought (CoT) prompting technique. Unlike CoT Prompting, DeBoP is an automatic optimization method, which focuses on the optimization directly on the behavior of LwLLMs. In particular, DeBoP transforms the optimization of complex prompts into the optimization of discrete, quantifiable execution sequences using a gradient-free Monte Carlo Tree Search. We evaluate DeBoP on seven challenging tasks where state-of-the-art LLMs excel but LwLLMs generally underperform. Experimental results demonstrate that DeBoP significantly outperforms recent prompt optimization methods on most tasks. In particular, DeBoP-optimized LwLLMs surpass GPT-3.5 on most tasks while reducing computational time by approximately 60% compared to other automatic prompt optimization methods. 

---
# Theoretical Analysis of Positional Encodings in Transformer Models: Impact on Expressiveness and Generalization 

**Authors**: Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06398)  

**Abstract**: Positional encodings are a core part of transformer-based models, enabling processing of sequential data without recurrence. This paper presents a theoretical framework to analyze how various positional encoding methods, including sinusoidal, learned, relative, and bias-based methods like Attention with Linear Biases (ALiBi), impact a transformer's expressiveness, generalization ability, and extrapolation to longer sequences. Expressiveness is defined via function approximation, generalization bounds are established using Rademacher complexity, and new encoding methods based on orthogonal functions, such as wavelets and Legendre polynomials, are proposed. The extrapolation capacity of existing and proposed encodings is analyzed, extending ALiBi's biasing approach to a unified theoretical context. Experimental evaluation on synthetic sequence-to-sequence tasks shows that orthogonal transform-based encodings outperform traditional sinusoidal encodings in generalization and extrapolation. This work addresses a critical gap in transformer theory, providing insights for design choices in natural language processing, computer vision, and other transformer applications. 

---
# Natural Language Interaction with Databases on Edge Devices in the Internet of Battlefield Things 

**Authors**: Christopher D. Molek, Roberto Fronteddu, K. Brent Venable, Niranjan Suri  

**Link**: [PDF](https://arxiv.org/pdf/2506.06396)  

**Abstract**: The expansion of the Internet of Things (IoT) in the battlefield, Internet of Battlefield Things (IoBT), gives rise to new opportunities for enhancing situational awareness. To increase the potential of IoBT for situational awareness in critical decision making, the data from these devices must be processed into consumer-ready information objects, and made available to consumers on demand. To address this challenge we propose a workflow that makes use of natural language processing (NLP) to query a database technology and return a response in natural language. Our solution utilizes Large Language Models (LLMs) that are sized for edge devices to perform NLP as well as graphical databases which are well suited for dynamic connected networks which are pervasive in the IoBT. Our architecture employs LLMs for both mapping questions in natural language to Cypher database queries as well as to summarize the database output back to the user in natural language. We evaluate several medium sized LLMs for both of these tasks on a database representing publicly available data from the US Army's Multipurpose Sensing Area (MSA) at the Jornada Range in Las Cruces, NM. We observe that Llama 3.1 (8 billion parameters) outperforms the other models across all the considered metrics. Most importantly, we note that, unlike current methods, our two step approach allows the relaxation of the Exact Match (EM) requirement of the produced Cypher queries with ground truth code and, in this way, it achieves a 19.4% increase in accuracy. Our workflow lays the ground work for deploying LLMs on edge devices to enable natural language interactions with databases containing information objects for critical decision making. 

---
# From Rogue to Safe AI: The Role of Explicit Refusals in Aligning LLMs with International Humanitarian Law 

**Authors**: John Mavi, Diana Teodora Găitan, Sergio Coronado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06391)  

**Abstract**: Large Language Models (LLMs) are widely used across sectors, yet their alignment with International Humanitarian Law (IHL) is not well understood. This study evaluates eight leading LLMs on their ability to refuse prompts that explicitly violate these legal frameworks, focusing also on helpfulness - how clearly and constructively refusals are communicated. While most models rejected unlawful requests, the clarity and consistency of their responses varied. By revealing the model's rationale and referencing relevant legal or safety principles, explanatory refusals clarify the system's boundaries, reduce ambiguity, and help prevent misuse. A standardised system-level safety prompt significantly improved the quality of the explanations expressed within refusals in most models, highlighting the effectiveness of lightweight interventions. However, more complex prompts involving technical language or requests for code revealed ongoing vulnerabilities. These findings contribute to the development of safer, more transparent AI systems and propose a benchmark to evaluate the compliance of LLM with IHL. 

---
# Benchmarking Large Language Models on Homework Assessment in Circuit Analysis 

**Authors**: Liangliang Chen, Zhihao Qin, Yiming Guo, Jacqueline Rohde, Ying Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06390)  

**Abstract**: Large language models (LLMs) have the potential to revolutionize various fields, including code development, robotics, finance, and education, due to their extensive prior knowledge and rapid advancements. This paper investigates how LLMs can be leveraged in engineering education. Specifically, we benchmark the capabilities of different LLMs, including GPT-3.5 Turbo, GPT-4o, and Llama 3 70B, in assessing homework for an undergraduate-level circuit analysis course. We have developed a novel dataset consisting of official reference solutions and real student solutions to problems from various topics in circuit analysis. To overcome the limitations of image recognition in current state-of-the-art LLMs, the solutions in the dataset are converted to LaTeX format. Using this dataset, a prompt template is designed to test five metrics of student solutions: completeness, method, final answer, arithmetic error, and units. The results show that GPT-4o and Llama 3 70B perform significantly better than GPT-3.5 Turbo across all five metrics, with GPT-4o and Llama 3 70B each having distinct advantages in different evaluation aspects. Additionally, we present insights into the limitations of current LLMs in several aspects of circuit analysis. Given the paramount importance of ensuring reliability in LLM-generated homework assessment to avoid misleading students, our results establish benchmarks and offer valuable insights for the development of a reliable, personalized tutor for circuit analysis -- a focus of our future work. Furthermore, the proposed evaluation methods can be generalized to a broader range of courses for engineering education in the future. 

---
# Model-based Neural Data Augmentation for sub-wavelength Radio Localization 

**Authors**: Baptiste Chatelier, Vincent Corlay, Musa Furkan Keskin, Matthieu Crussière, Henk Wymeersch, Luc Le Magoarou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06387)  

**Abstract**: The increasing deployment of large antenna arrays at base stations has significantly improved the spatial resolution and localization accuracy of radio-localization methods. However, traditional signal processing techniques struggle in complex radio environments, particularly in scenarios dominated by non line of sight (NLoS) propagation paths, resulting in degraded localization accuracy. Recent developments in machine learning have facilitated the development of machine learning-assisted localization techniques, enhancing localization accuracy in complex radio environments. However, these methods often involve substantial computational complexity during both the training and inference phases. This work extends the well-established fingerprinting-based localization framework by simultaneously reducing its memory requirements and improving its accuracy. Specifically, a model-based neural network is used to learn the location-to-channel mapping, and then serves as a generative neural channel model. This generative model augments the fingerprinting comparison dictionary while reducing the memory requirements. The proposed method outperforms fingerprinting baselines by achieving sub-wavelength localization accuracy, even in NLoS environments. Remarkably, it offers an improvement by several orders of magnitude in localization accuracy, while simultaneously reducing memory requirements by an order of magnitude compared to classical fingerprinting methods. 

---
# Detection Method for Prompt Injection by Integrating Pre-trained Model and Heuristic Feature Engineering 

**Authors**: Yi Ji, Runzhi Li, Baolei Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06384)  

**Abstract**: With the widespread adoption of Large Language Models (LLMs), prompt injection attacks have emerged as a significant security threat. Existing defense mechanisms often face critical trade-offs between effectiveness and generalizability. This highlights the urgent need for efficient prompt injection detection methods that are applicable across a wide range of LLMs. To address this challenge, we propose DMPI-PMHFE, a dual-channel feature fusion detection framework. It integrates a pretrained language model with heuristic feature engineering to detect prompt injection attacks. Specifically, the framework employs DeBERTa-v3-base as a feature extractor to transform input text into semantic vectors enriched with contextual information. In parallel, we design heuristic rules based on known attack patterns to extract explicit structural features commonly observed in attacks. Features from both channels are subsequently fused and passed through a fully connected neural network to produce the final prediction. This dual-channel approach mitigates the limitations of relying only on DeBERTa to extract features. Experimental results on diverse benchmark datasets demonstrate that DMPI-PMHFE outperforms existing methods in terms of accuracy, recall, and F1-score. Furthermore, when deployed actually, it significantly reduces attack success rates across mainstream LLMs, including GLM-4, LLaMA 3, Qwen 2.5, and GPT-4o. 

---
# Human and AI collaboration in Fitness Education:A Longitudinal Study with a Pilates Instructor 

**Authors**: Qian Huang, King Wang Poon  

**Link**: [PDF](https://arxiv.org/pdf/2506.06383)  

**Abstract**: Artificial intelligence is poised to transform teaching and coaching practices,yet its optimal role alongside human expertise remains this http URL study investigates human and AI collaboration in fitness education through a one year qualitative case study with a Pilates this http URL researcher participated in the instructor classes and conducted biweekly semi structured interviews to explore how generative AI could be integrated into class planning and instruction. 

---
# On the Fundamental Impossibility of Hallucination Control in Large Language Models 

**Authors**: Michał P. Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.06382)  

**Abstract**: This paper explains \textbf{why it is impossible to create large language models that do not hallucinate and what are the trade-offs we should be looking for}. It presents a formal \textbf{impossibility theorem} demonstrating that no inference mechanism can simultaneously satisfy four fundamental properties: \textbf{truthful (non-hallucinatory) generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality}. By modeling LLM inference as an \textbf{auction of ideas} where neural components compete to contribute to responses, we prove the impossibility using the Green-Laffont theorem. That mathematical framework provides a rigorous foundation for understanding the nature of inference process, with implications for model architecture, training objectives, and evaluation methods. 

---
# CPS-Guard: Framework for Dependability Assurance of AI- and LLM-Based Cyber-Physical Systems 

**Authors**: Trisanth Srinivasan, Santosh Patapati, Himani Musku, Idhant Gode, Aditya Arora, Samvit Bhattacharya, Abubakr Nazriev, Sanika Hirave, Zaryab Kanjiani, Srinjoy Ghose, Srinidhi Shetty  

**Link**: [PDF](https://arxiv.org/pdf/2506.06381)  

**Abstract**: Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce CPS-Guard, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, CPS-Guard continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that CPS-Guard effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems. 

---
# Beyond the Norm: A Survey of Synthetic Data Generation for Rare Events 

**Authors**: Jingyi Gu, Xuan Zhang, Guiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06380)  

**Abstract**: Extreme events, such as market crashes, natural disasters, and pandemics, are rare but catastrophic, often triggering cascading failures across interconnected systems. Accurate prediction and early warning can help minimize losses and improve preparedness. While data-driven methods offer powerful capabilities for extreme event modeling, they require abundant training data, yet extreme event data is inherently scarce, creating a fundamental challenge. Synthetic data generation has emerged as a powerful solution. However, existing surveys focus on general data with privacy preservation emphasis, rather than extreme events' unique performance requirements. This survey provides the first overview of synthetic data generation for extreme events. We systematically review generative modeling techniques and large language models, particularly those enhanced by statistical theory as well as specialized training and sampling mechanisms to capture heavy-tailed distributions. We summarize benchmark datasets and introduce a tailored evaluation framework covering statistical, dependence, visual, and task-oriented metrics. A central contribution is our in-depth analysis of each metric's applicability in extremeness and domain-specific adaptations, providing actionable guidance for model evaluation in extreme settings. We categorize key application domains and identify underexplored areas like behavioral finance, wildfires, earthquakes, windstorms, and infectious outbreaks. Finally, we outline open challenges, providing a structured foundation for advancing synthetic rare-event research. 

---
# Enhancing Decision-Making of Large Language Models via Actor-Critic 

**Authors**: Heng Dong, Kefei Duan, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06376)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable advancements in natural language processing tasks, yet they encounter challenges in complex decision-making scenarios that require long-term reasoning and alignment with high-level objectives. Existing methods either rely on short-term auto-regressive action generation or face limitations in accurately simulating rollouts and assessing outcomes, leading to sub-optimal decisions. This paper introduces a novel LLM-based Actor-Critic framework, termed LAC, that effectively improves LLM policies with long-term action evaluations in a principled and scalable way. Our approach addresses two key challenges: (1) extracting robust action evaluations by computing Q-values via token logits associated with positive/negative outcomes, enhanced by future trajectory rollouts and reasoning; and (2) enabling efficient policy improvement through a gradient-free mechanism. Experiments across diverse environments -- including high-level decision-making (ALFWorld), low-level action spaces (BabyAI-Text), and large action spaces (WebShop) -- demonstrate the framework's generality and superiority over state-of-the-art methods. Notably, our approach achieves competitive performance using 7B/8B parameter LLMs, even outperforming baseline methods employing GPT-4 in complex tasks. These results underscore the potential of integrating structured policy optimization with LLMs' intrinsic knowledge to advance decision-making capabilities in multi-step environments. 

---
# CR-BLEA: Contrastive Ranking for Adaptive Resource Allocation in Bilevel Evolutionary Algorithms 

**Authors**: Dejun Xu, Jijia Chen, Gary G. Yen, Min Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06362)  

**Abstract**: Bilevel optimization poses a significant computational challenge due to its nested structure, where each upper-level candidate solution requires solving a corresponding lower-level problem. While evolutionary algorithms (EAs) are effective at navigating such complex landscapes, their high resource demands remain a key bottleneck -- particularly the redundant evaluation of numerous unpromising lower-level tasks. Despite recent advances in multitasking and transfer learning, resource waste persists. To address this issue, we propose a novel resource allocation framework for bilevel EAs that selectively identifies and focuses on promising lower-level tasks. Central to our approach is a contrastive ranking network that learns relational patterns between paired upper- and lower-level solutions online. This knowledge guides a reference-based ranking strategy that prioritizes tasks for optimization and adaptively controls resampling based on estimated population quality. Comprehensive experiments across five state-of-the-art bilevel algorithms show that our framework significantly reduces computational cost while preserving -- or even enhancing -- solution accuracy. This work offers a generalizable strategy to improve the efficiency of bilevel EAs, paving the way for more scalable bilevel optimization. 

---
# Tactile MNIST: Benchmarking Active Tactile Perception 

**Authors**: Tim Schneider, Guillaume Duret, Cristiana de Farias, Roberto Calandra, Liming Chen, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2506.06361)  

**Abstract**: Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception. 

---
# From Transformers to Large Language Models: A systematic review of AI applications in the energy sector towards Agentic Digital Twins 

**Authors**: Gabriel Antonesi, Tudor Cioara, Ionut Anghel, Vasilis Michalakopoulos, Elissaios Sarmas, Liana Toderean  

**Link**: [PDF](https://arxiv.org/pdf/2506.06359)  

**Abstract**: Artificial intelligence (AI) has long promised to improve energy management in smart grids by enhancing situational awareness and supporting more effective decision-making. While traditional machine learning has demonstrated notable results in forecasting and optimization, it often struggles with generalization, situational awareness, and heterogeneous data integration. Recent advances in foundation models such as Transformer architecture and Large Language Models (LLMs) have demonstrated improved capabilities in modelling complex temporal and contextual relationships, as well as in multi-modal data fusion which is essential for most AI applications in the energy sector. In this review we synthesize the rapid expanding field of AI applications in the energy domain focusing on Transformers and LLMs. We examine the architectural foundations, domain-specific adaptations and practical implementations of transformer models across various forecasting and grid management tasks. We then explore the emerging role of LLMs in the field: adaptation and fine tuning for the energy sector, the type of tasks they are suited for, and the new challenges they introduce. Along the way, we highlight practical implementations, innovations, and areas where the research frontier is rapidly expanding. These recent developments reviewed underscore a broader trend: Generative AI (GenAI) is beginning to augment decision-making not only in high-level planning but also in day-to-day operations, from forecasting and grid balancing to workforce training and asset onboarding. Building on these developments, we introduce the concept of the Agentic Digital Twin, a next-generation model that integrates LLMs to bring autonomy, proactivity, and social interaction into digital twin-based energy management systems. 

---
# Towards real-time assessment of infrasound event detection capability using deep learning-based transmission loss estimation 

**Authors**: Alice Janela Cameijo, Alexis Le Pichon, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven peter Naesholm, Constantino Listowski, Samir Aknine  

**Link**: [PDF](https://arxiv.org/pdf/2506.06358)  

**Abstract**: Accurate modeling of infrasound transmission loss is essential for evaluating the performance of the International Monitoring System, enabling the effective design and maintenance of infrasound stations to support compliance of the Comprehensive Nuclear-Test-Ban Treaty. State-of-the-art propagation modeling tools enable transmission loss to be finely simulated using atmospheric models. However, the computational cost prohibits the exploration of a large parameter space in operational monitoring applications. To address this, recent studies made use of a deep learning algorithm capable of making transmission loss predictions almost instantaneously. However, the use of nudged atmospheric models leads to an incomplete representation of the medium, and the absence of temperature as an input makes the algorithm incompatible with long range propagation. In this study, we address these limitations by using both wind and temperature fields as inputs to a neural network, simulated up to 130 km altitude and 4,000 km distance. We also optimize several aspects of the neural network architecture. We exploit convolutional and recurrent layers to capture spatially and range-dependent features embedded in realistic atmospheric models, improving the overall performance. The neural network reaches an average error of 4 dB compared to full parabolic equation simulations and provides epistemic and data-related uncertainty estimates. Its evaluation on the 2022 Hunga Tonga-Hunga Ha'apai volcanic eruption demonstrates its prediction capability using atmospheric conditions and frequencies not included in the training. This represents a significant step towards near real-time assessment of International Monitoring System detection thresholds of explosive sources. 

---
# Large Language Models for EEG: A Comprehensive Survey and Taxonomy 

**Authors**: Naseem Babu, Jimson Mathew, A. P. Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2506.06353)  

**Abstract**: The growing convergence between Large Language Models (LLMs) and electroencephalography (EEG) research is enabling new directions in neural decoding, brain-computer interfaces (BCIs), and affective computing. This survey offers a systematic review and structured taxonomy of recent advancements that utilize LLMs for EEG-based analysis and applications. We organize the literature into four domains: (1) LLM-inspired foundation models for EEG representation learning, (2) EEG-to-language decoding, (3) cross-modal generation including image and 3D object synthesis, and (4) clinical applications and dataset management tools. The survey highlights how transformer-based architectures adapted through fine-tuning, few-shot, and zero-shot learning have enabled EEG-based models to perform complex tasks such as natural language generation, semantic interpretation, and diagnostic assistance. By offering a structured overview of modeling strategies, system designs, and application areas, this work serves as a foundational resource for future work to bridge natural language processing and neural signal analysis through language models. 

---
# Deep learning methods for modeling infrasound transmission loss in the middle atmosphere 

**Authors**: Alexis Le Pichon, Alice Janela Cameijo, Samir Aknine, Youcef Sklab, Souhila Arib, Quentin Brissaud, Sven Peter Naesholm  

**Link**: [PDF](https://arxiv.org/pdf/2506.06351)  

**Abstract**: Accurate modeling of infrasound transmission losses (TLs) is essential to assess the performance of the global International Monitoring System infrasound network. Among existing propagation modeling tools, parabolic equation (PE) method enables TLs to be finely modeled, but its computational cost does not allow exploration of a large parameter space for operational monitoring applications. To reduce computation times, Brissaud et al. 2023 explored the potential of convolutional neural networks trained on a large set of regionally simulated wavefields (< 1000 km from the source) to predict TLs with negligible computation times compared to PE simulations. However, this method struggles in unfavorable initial wind conditions, especially at high frequencies, and causal issues with winds at large distances from the source affecting ground TLs close to the source. In this study, we have developed an optimized convolutional network designed to minimize prediction errors while predicting TLs from globally simulated combined temperature and wind fields spanning over propagation ranges of 4000 km. Our approach enhances the previously proposed one by implementing key optimizations that improve the overall architecture performance. The implemented model predicts TLs with an average error of 8.6 dB in the whole frequency band (0.1-3.2 Hz) and explored realistic atmospheric scenarios. 

---
# Unified Game Moderation: Soft-Prompting and LLM-Assisted Label Transfer for Resource-Efficient Toxicity Detection 

**Authors**: Zachary Yang, Domenico Tullo, Reihaneh Rabbany  

**Link**: [PDF](https://arxiv.org/pdf/2506.06347)  

**Abstract**: Toxicity detection in gaming communities faces significant scaling challenges when expanding across multiple games and languages, particularly in real-time environments where computational efficiency is crucial. We present two key findings to address these challenges while building upon our previous work on ToxBuster, a BERT-based real-time toxicity detection system. First, we introduce a soft-prompting approach that enables a single model to effectively handle multiple games by incorporating game-context tokens, matching the performance of more complex methods like curriculum learning while offering superior scalability. Second, we develop an LLM-assisted label transfer framework using GPT-4o-mini to extend support to seven additional languages. Evaluations on real game chat data across French, German, Portuguese, and Russian achieve macro F1-scores ranging from 32.96% to 58.88%, with particularly strong performance in German, surpassing the English benchmark of 45.39%. In production, this unified approach significantly reduces computational resources and maintenance overhead compared to maintaining separate models for each game and language combination. At Ubisoft, this model successfully identifies an average of 50 players, per game, per day engaging in sanctionable behavior. 

---
# Explainable-AI powered stock price prediction using time series transformers: A Case Study on BIST100 

**Authors**: Sukru Selim Calik, Andac Akyuz, Zeynep Hilal Kilimci, Kerem Colak  

**Link**: [PDF](https://arxiv.org/pdf/2506.06345)  

**Abstract**: Financial literacy is increasingly dependent on the ability to interpret complex financial data and utilize advanced forecasting tools. In this context, this study proposes a novel approach that combines transformer-based time series models with explainable artificial intelligence (XAI) to enhance the interpretability and accuracy of stock price predictions. The analysis focuses on the daily stock prices of the five highest-volume banks listed in the BIST100 index, along with XBANK and XU100 indices, covering the period from January 2015 to March 2025. Models including DLinear, LTSNet, Vanilla Transformer, and Time Series Transformer are employed, with input features enriched by technical indicators. SHAP and LIME techniques are used to provide transparency into the influence of individual features on model outputs. The results demonstrate the strong predictive capabilities of transformer models and highlight the potential of interpretable machine learning to empower individuals in making informed investment decisions and actively engaging in financial markets. 

---
# A Reinforcement Learning Approach for RIS-aided Fair Communications 

**Authors**: Alex Pierron, Michel Barbeau, Luca De Cicco, Jose Rubio-Hernan, Joaquin Garcia-Alfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.06344)  

**Abstract**: Reconfigurable Intelligent Surfaces (RISs) are composed of physical elements that can dynamically alter electromagnetic wave properties to enhance beamforming and leading to improvements in areas with low coverage properties. They have the potential to be combined with Reinforcement Learning (RL) techniques to achieve network performance and energy efficiency via optimization techniques. In addition to performance and energy improvements, it is also crucial to consider the concept of fair communications. RISs must ensure that User Equipment (UE) units receive their signals with adequate strength, without other UE being deprived of service due to insufficient power. In this paper, we address such a problem. We explore the fairness properties of previous work and propose a novel method that aims at obtaining an efficient and fair duplex RIS-RL system for multiple legitimate UE units. We report and discuss our experimental work and simulation results. We also release our code and datasets to foster further research in the topic. 

---
# TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment 

**Authors**: Taesoo Kim, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2506.06343)  

**Abstract**: Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data. 

---
# NR4DER: Neural Re-ranking for Diversified Exercise Recommendation 

**Authors**: Xinghe Cheng, Xufang Zhou, Liangda Fang, Chaobo He, Yuyu Zhou, Weiqi Luo, Zhiguo Gong, Quanlong Guan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06341)  

**Abstract**: With the widespread adoption of online education platforms, an increasing number of students are gaining new knowledge through Massive Open Online Courses (MOOCs). Exercise recommendation have made strides toward improving student learning outcomes. However, existing methods not only struggle with high dropout rates but also fail to match the diverse learning pace of students. They frequently face difficulties in adjusting to inactive students' learning patterns and in accommodating individualized learning paces, resulting in limited accuracy and diversity in recommendations. To tackle these challenges, we propose Neural Re-ranking for Diversified Exercise Recommendation (in short, NR4DER). NR4DER first leverages the mLSTM model to improve the effectiveness of the exercise filter module. It then employs a sequence enhancement method to enhance the representation of inactive students, accurately matches students with exercises of appropriate difficulty. Finally, it utilizes neural re-ranking to generate diverse recommendation lists based on individual students' learning histories. Extensive experimental results indicate that NR4DER significantly outperforms existing methods across multiple real-world datasets and effectively caters to the diverse learning pace of students. 

---
# Structured Semantics from Unstructured Notes: Language Model Approaches to EHR-Based Decision Support 

**Authors**: Wu Hao Ran, Xi Xi, Furong Li, Jingyi Lu, Jian Jiang, Hui Huang, Yuzhuan Zhang, Shi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06340)  

**Abstract**: The advent of large language models (LLMs) has opened new avenues for analyzing complex, unstructured data, particularly within the medical domain. Electronic Health Records (EHRs) contain a wealth of information in various formats, including free text clinical notes, structured lab results, and diagnostic codes. This paper explores the application of advanced language models to leverage these diverse data sources for improved clinical decision support. We will discuss how text-based features, often overlooked in traditional high dimensional EHR analysis, can provide semantically rich representations and aid in harmonizing data across different institutions. Furthermore, we delve into the challenges and opportunities of incorporating medical codes and ensuring the generalizability and fairness of AI models in healthcare. 

---
# Optimizing RAG Pipelines for Arabic: A Systematic Analysis of Core Components 

**Authors**: Jumana Alsubhi, Mohammad D. Alahmadi, Ahmed Alhusayni, Ibrahim Aldailami, Israa Hamdine, Ahmad Shabana, Yazeed Iskandar, Suhayb Khayyat  

**Link**: [PDF](https://arxiv.org/pdf/2506.06339)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful architecture for combining the precision of retrieval systems with the fluency of large language models. While several studies have investigated RAG pipelines for high-resource languages, the optimization of RAG components for Arabic remains underexplored. This study presents a comprehensive empirical evaluation of state-of-the-art RAG components-including chunking strategies, embedding models, rerankers, and language models-across a diverse set of Arabic datasets. Using the RAGAS framework, we systematically compare performance across four core metrics: context precision, context recall, answer faithfulness, and answer relevancy. Our experiments demonstrate that sentence-aware chunking outperforms all other segmentation methods, while BGE-M3 and Multilingual-E5-large emerge as the most effective embedding models. The inclusion of a reranker (bge-reranker-v2-m3) significantly boosts faithfulness in complex datasets, and Aya-8B surpasses StableLM in generation quality. These findings provide critical insights for building high-quality Arabic RAG pipelines and offer practical guidelines for selecting optimal components across different document types. 

---
# FinBERT2: A Specialized Bidirectional Encoder for Bridging the Gap in Finance-Specific Deployment of Large Language Models 

**Authors**: Xuan Xu, Fufang Wen, Beilin Chu, Zhibing Fu, Qinhong Lin, Jiaqi Liu, Binjie Fei, Zhongliang Yang, Linna Zhou, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06335)  

**Abstract**: In natural language processing (NLP), the focus has shifted from encoder-only tiny language models like BERT to decoder-only large language models(LLMs) such as GPT-3. However, LLMs' practical application in the financial sector has revealed three limitations: (1) LLMs often perform worse than fine-tuned BERT on discriminative tasks despite costing much higher computational resources, such as market sentiment analysis in financial reports; (2) Application on generative tasks heavily relies on retrieval augmented generation (RAG) methods to provide current and specialized information, with general retrievers showing suboptimal performance on domain-specific retrieval tasks; (3) There are additional inadequacies in other feature-based scenarios, such as topic modeling. We introduce FinBERT2, a specialized bidirectional encoder pretrained on a high-quality, financial-specific corpus of 32b tokens. This represents the largest known Chinese financial pretraining corpus for models of this parameter size. As a better backbone, FinBERT2 can bridge the gap in the financial-specific deployment of LLMs through the following achievements: (1) Discriminative fine-tuned models (Fin-Labelers) outperform other (Fin)BERT variants by 0.4%-3.3% and leading LLMs by 9.7%-12.3% on average across five financial classification tasks. (2) Contrastive fine-tuned models (Fin-Retrievers) outperform both open-source (e.g., +6.8\% avg improvement over BGE-base-zh) and proprietary (e.g., +4.2\% avg improvement over OpenAI's text-embedding-3-large) embedders across five financial retrieval tasks; (3) Building on FinBERT2 variants, we construct the Fin-TopicModel, which enables superior clustering and topic representation for financial titles. Our work revisits financial BERT models through comparative analysis with contemporary LLMs and offers practical insights for effectively utilizing FinBERT in the LLMs era. 

---
# Introduction to Predictive Coding Networks for Machine Learning 

**Authors**: Mikko Stenlund  

**Link**: [PDF](https://arxiv.org/pdf/2506.06332)  

**Abstract**: Predictive coding networks (PCNs) constitute a biologically inspired framework for understanding hierarchical computation in the brain, and offer an alternative to traditional feedforward neural networks in ML. This note serves as a quick, onboarding introduction to PCNs for machine learning practitioners. We cover the foundational network architecture, inference and learning update rules, and algorithmic implementation. A concrete image-classification task (CIFAR-10) is provided as a benchmark-smashing application, together with an accompanying Python notebook containing the PyTorch implementation. 

---
# How Significant Are the Real Performance Gains? An Unbiased Evaluation Framework for GraphRAG 

**Authors**: Qiming Zeng, Xiao Yan, Hao Luo, Yuhao Lin, Yuxiang Wang, Fangcheng Fu, Bo Du, Quanqing Xu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06331)  

**Abstract**: By retrieving contexts from knowledge graphs, graph-based retrieval-augmented generation (GraphRAG) enhances large language models (LLMs) to generate quality answers for user questions. Many GraphRAG methods have been proposed and reported inspiring performance in answer quality. However, we observe that the current answer evaluation framework for GraphRAG has two critical flaws, i.e., unrelated questions and evaluation biases, which may lead to biased or even wrong conclusions on performance. To tackle the two flaws, we propose an unbiased evaluation framework that uses graph-text-grounded question generation to produce questions that are more related to the underlying dataset and an unbiased evaluation procedure to eliminate the biases in LLM-based answer assessment. We apply our unbiased framework to evaluate 3 representative GraphRAG methods and find that their performance gains are much more moderate than reported previously. Although our evaluation framework may still have flaws, it calls for scientific evaluations to lay solid foundations for GraphRAG research. 

---
# Evolutionary model for energy trading in community microgrids using Hawk-Dove strategies 

**Authors**: Viorica Rozina Chifu, Tudor Cioara, Cristina Bianca Pop, Ionut Anghel  

**Link**: [PDF](https://arxiv.org/pdf/2506.06325)  

**Abstract**: This paper proposes a decentralized model of energy cooperation between microgrids, in which decisions are made locally, at the level of the microgrid community. Each microgrid is modeled as an autonomous agent that adopts a Hawk or Dove strategy, depending on the level of energy stored in the battery and its role in the energy trading process. The interactions between selling and buying microgrids are modeled through an evolutionary algorithm. An individual in the algorithm population is represented as an energy trading matrix that encodes the amounts of energy traded between the selling and buying microgrids. The population evolution is achieved by recombination and mutation operators. Recombination uses a specialized operator for matrix structures, and mutation is applied to the matrix elements according to a Gaussian distribution. The evaluation of an individual is made with a multi-criteria fitness function that considers the seller profit, the degree of energy stability at the community level, penalties for energy imbalance at the community level and for the degradation of microgrids batteries. The method was tested on a simulated scenario with 100 microgrids, each with its own selling and buying thresholds, to reflect a realistic environment with variable storage characteristics of microgrids batteries. By applying the algorithm on this scenario, 95 out of the 100 microgrids reached a stable energy state. This result confirms the effectiveness of the proposed model in achieving energy balance both at the individual level, for each microgrid, and at the level of the entire community. 

---
# Neural networks with image recognition by pairs 

**Authors**: Polad Geidarov  

**Link**: [PDF](https://arxiv.org/pdf/2506.06322)  

**Abstract**: Neural networks based on metric recognition methods have a strictly determined architecture. Number of neurons, connections, as well as weights and thresholds values are calculated analytically, based on the initial conditions of tasks: number of recognizable classes, number of samples, metric expressions used. This paper discusses the possibility of transforming these networks in order to apply classical learning algorithms to them without using analytical expressions that calculate weight values. In the received network, training is carried out by recognizing images in pairs. This approach simplifies the learning process and easily allows to expand the neural network by adding new images to the recognition task. The advantages of these networks, including such as: 1) network architecture simplicity and transparency; 2) training simplicity and reliability; 3) the possibility of using a large number of images in the recognition problem using a neural network; 4) a consistent increase in the number of recognizable classes without changing the previous values of weights and thresholds. 

---
# MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes 

**Authors**: Feiyang Pan, Shenghe Zheng, Chunyan Yin, Guangbin Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.06318)  

**Abstract**: MEMS gyroscopes play a critical role in inertial navigation and motion control applications but typically suffer from a fundamental trade-off between measurement range and noise performance. Existing hardware-based solutions aimed at mitigating this issue introduce additional complexity, cost, and scalability challenges. Deep-learning methods primarily focus on noise reduction and typically require precisely aligned ground-truth signals, making them difficult to deploy in practical scenarios and leaving the fundamental trade-off unresolved. To address these challenges, we introduce Mixture of Experts for MEMS Gyroscopes (MoE-Gyro), a novel self-supervised framework specifically designed for simultaneous over-range signal reconstruction and noise suppression. MoE-Gyro employs two experts: an Over-Range Reconstruction Expert (ORE), featuring a Gaussian-Decay Attention mechanism for reconstructing saturated segments; and a Denoise Expert (DE), utilizing dual-branch complementary masking combined with FFT-guided augmentation for robust noise reduction. A lightweight gating module dynamically routes input segments to the appropriate expert. Furthermore, existing evaluation lack a comprehensive standard for assessing multi-dimensional signal enhancement. To bridge this gap, we introduce IMU Signal Enhancement Benchmark (ISEBench), an open-source benchmarking platform comprising the GyroPeak-100 dataset and a unified evaluation of IMU signal enhancement methods. We evaluate MoE-Gyro using our proposed ISEBench, demonstrating that our framework significantly extends the measurable range from 450 deg/s to 1500 deg/s, reduces Bias Instability by 98.4%, and achieves state-of-the-art performance, effectively addressing the long-standing trade-off in inertial sensing. 

---
# A Reinforcement-Learning-Enhanced LLM Framework for Automated A/B Testing in Personalized Marketing 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06316)  

**Abstract**: For personalized marketing, a new challenge of how to effectively algorithm the A/B testing to maximize user response is urgently to be overcome. In this paper, we present a new approach, the RL-LLM-AB test framework, for using reinforcement learning strategy optimization combined with LLM to automate and personalize A/B tests. The RL-LLM-AB test is built upon the pre-trained instruction-tuned language model. It first generates A/B versions of candidate content variants using a Prompt-Conditioned Generator, and then dynamically embeds and fuses the user portrait and the context of the current query with the multi-modal perception module to constitute the current interaction state. The content version is then selected in real-time through the policy optimization module with an Actor-Critic structure, and long-term revenue is estimated according to real-time feedback (such as click-through rate and conversion rate). Furthermore, a Memory-Augmented Reward Estimator is embedded into the framework to capture long-term user preference drift, which helps to generalize policy across multiple users and content contexts. Numerical results demonstrate the superiority of our proposed RL-LLM-ABTest over existing A/B testing methods, including classical A/B testing, Contextual Bandits, and benchmark reinforcement learning approaches on real-world marketing data. 

---
# DISRetrieval: Harnessing Discourse Structure for Long Document Retrieval 

**Authors**: Huiyao Chen, Yi Yang, Yinghui Li, Meishan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06313)  

**Abstract**: Long document understanding has become increasingly crucial in natural language processing, with retrieval-based methods emerging as a promising solution to address the context length limitations of large language models (LLMs). However, existing approaches either treat documents as flat sequences or employ arbitrary chunking strategies, failing to capture the inherent discourse structure that guides human comprehension. We present DISRetrieval, a novel hierarchical retrieval framework that leverages linguistic discourse structure to enhance long document understanding. Our approach introduces three key innovations: (1) a discourse-aware document organization framework that utilizes rhetorical structure theory (RST) to create sentence-level hierarchical representations, preserving both semantic relationships and natural document flow; (2) an LLM-enhanced node representation technique that combines discourse structure with adaptive summarization to enrich tree nodes with contextual information; and (3) a hierarchical evidence retrieval mechanism that effectively selects relevant content while maintaining discourse coherence. Through comprehensive experiments on QASPER and QuALITY datasets, DISRetrieval demonstrates substantial improvements over existing methods in both token-level retrieval metrics and downstream question answering tasks. Our ablation studies confirm that incorporating discourse structure significantly enhances retrieval effectiveness across different document lengths and query types, validating the importance of linguistically-informed document representation in long-text understanding. Our code and datasets are publicly available at github/DreamH1gh/DISRetrieval to facilitate future research. 

---
# Reward Is Enough: LLMs Are In-Context Reinforcement Learners 

**Authors**: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Yanjun Qi, Shangtong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06303)  

**Abstract**: Reinforcement learning (RL) is a human-designed framework for solving sequential decision making problems. In this work, we demonstrate that, surprisingly, RL emerges in LLM's (Large Language Model) inference time -- a phenomenon known as in-context RL (ICRL). Specifically, we propose a novel multi-round prompting framework called ICRL prompting. The goal is to prompt the LLM to complete a task. After the LLM generates a response at the current round, we give numerical scalar feedbacks for the response, called the rewards. At the next round, we prompt the LLM again with the same task and a context consisting of all previous responses and rewards. We observe that the quality of the LLM's response increases as the context grows. In other words, the LLM is able to maximize the scalar reward signal in the inference time, just like an RL algorithm. We evaluate ICRL prompting in three benchmarks (Game of 24, creative writing, and ScienceWorld) and demonstrate significant performance improvements over baseline methods such as Self-Refine and Reflexion. Surprisingly, in some experiments the reward signals are generated by the LLM itself, yet performance improvements are still observed from ICRL prompting, offering a promising paradigm for scaling test-time compute. 

---
# How Malicious AI Swarms Can Threaten Democracy 

**Authors**: Daniel Thilo Schroeder, Meeyoung Cha, Andrea Baronchelli, Nick Bostrom, Nicholas A. Christakis, David Garcia, Amit Goldenberg, Yara Kyrychenko, Kevin Leyton-Brown, Nina Lutz, Gary Marcus, Filippo Menczer, Gordon Pennycook, David G. Rand, Frank Schweitzer, Christopher Summerfield, Audrey Tang, Jay Van Bavel, Sander van der Linden, Dawn Song, Jonas R. Kunst  

**Link**: [PDF](https://arxiv.org/pdf/2506.06299)  

**Abstract**: Advances in AI portend a new era of sophisticated disinformation operations. While individual AI systems already create convincing -- and at times misleading -- information, an imminent development is the emergence of malicious AI swarms. These systems can coordinate covertly, infiltrate communities, evade traditional detectors, and run continuous A/B tests, with round-the-clock persistence. The result can include fabricated grassroots consensus, fragmented shared reality, mass harassment, voter micro-suppression or mobilization, contamination of AI training data, and erosion of institutional trust. With democratic processes worldwide increasingly vulnerable, we urge a three-pronged response: (1) platform-side defenses -- always-on swarm-detection dashboards, pre-election high-fidelity swarm-simulation stress-tests, transparency audits, and optional client-side "AI shields" for users; (2) model-side safeguards -- standardized persuasion-risk tests, provenance-authenticating passkeys, and watermarking; and (3) system-level oversight -- a UN-backed AI Influence Observatory. 

---
# Pairwise Calibrated Rewards for Pluralistic Alignment 

**Authors**: Daniel Halpern, Evi Micha, Ariel D. Procaccia, Itai Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2506.06298)  

**Abstract**: Current alignment pipelines presume a single, universal notion of desirable behavior. However, human preferences often diverge across users, contexts, and cultures. As a result, disagreement collapses into the majority signal and minority perspectives are discounted. To address this, we propose reflecting diverse human preferences through a distribution over multiple reward functions, each inducing a distinct aligned policy. The distribution is learned directly from pairwise preference without annotator identifiers or predefined groups. Instead, annotator disagreements are treated as informative soft labels. Our central criterion is pairwise calibration: for every pair of candidate responses, the proportion of reward functions preferring one response matches the fraction of annotators with that preference. We prove that even a small outlier-free ensemble can accurately represent diverse preference distributions. Empirically, we introduce and validate a practical training heuristic to learn such ensembles, and demonstrate its effectiveness through improved calibration, implying a more faithful representation of pluralistic values. 

---
# Optimal patient allocation for echocardiographic assessments 

**Authors**: Bozhi Sun, Seda Tierney, Jeffrey A. Feinstein, Frederick Damen, Alison L. Marsden, Daniele E. Schiavazzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06297)  

**Abstract**: Scheduling echocardiographic exams in a hospital presents significant challenges due to non-deterministic factors (e.g., patient no-shows, patient arrival times, diverse exam durations, etc.) and asymmetric resource constraints between fetal and non-fetal patient streams. To address these challenges, we first conducted extensive pre-processing on one week of operational data from the Echo Laboratory at Stanford University's Lucile Packard Children's Hospital, to estimate patient no-show probabilities and derive empirical distributions of arrival times and exam durations. Based on these inputs, we developed a discrete-event stochastic simulation model using SimPy, and integrate it with the open source Gymnasium Python library. As a baseline for policy optimization, we developed a comparative framework to evaluate on-the-fly versus reservation-based allocation strategies, in which different proportions of resources are reserved in advance. Considering a hospital configuration with a 1:6 ratio of fetal to non-fetal rooms and a 4:2 ratio of fetal to non-fetal sonographers, we show that on-the-fly allocation generally yields better performance, more effectively adapting to patient variability and resource constraints. Building on this foundation, we apply reinforcement learning (RL) to derive an approximated optimal dynamic allocation policy. This RL-based policy is benchmarked against the best-performing rule-based strategies, allowing us to quantify their differences and provide actionable insights for improving echo lab efficiency through intelligent, data-driven resource management. 

---
# Dynamic Graph CNN with Jacobi Kolmogorov-Arnold Networks for 3D Classification of Point Sets 

**Authors**: Hanaa El Afia, Said Ohamouddou, Raddouane Chiheb, Abdellatif El Afia  

**Link**: [PDF](https://arxiv.org/pdf/2506.06296)  

**Abstract**: We introduce Jacobi-KAN-DGCNN, a framework that integrates Dynamic Graph Convolutional Neural Network (DGCNN) with Jacobi Kolmogorov-Arnold Networks (KAN) for the classification of three-dimensional point clouds. This method replaces Multi-Layer Perceptron (MLP) layers with adaptable univariate polynomial expansions within a streamlined DGCNN architecture, circumventing deep levels for both MLP and KAN to facilitate a layer-by-layer comparison. In comparative experiments on the ModelNet40 dataset, KAN layers employing Jacobi polynomials outperform the traditional linear layer-based DGCNN baseline in terms of accuracy and convergence speed, while maintaining parameter efficiency. Our results demonstrate that higher polynomial degrees do not automatically improve performance, highlighting the need for further theoretical and empirical investigation to fully understand the interactions between polynomial bases, degrees, and the mechanisms of graph-based learning. 

---
# dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching 

**Authors**: Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06295)  

**Abstract**: Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub. 

---
# GLProtein: Global-and-Local Structure Aware Protein Representation Learning 

**Authors**: Yunqing Liu, Wenqi Fan, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06294)  

**Abstract**: Proteins are central to biological systems, participating as building blocks across all forms of life. Despite advancements in understanding protein functions through protein sequence analysis, there remains potential for further exploration in integrating protein structural information. We argue that the structural information of proteins is not only limited to their 3D information but also encompasses information from amino acid molecules (local information) to protein-protein structure similarity (global information). To address this, we propose \textbf{GLProtein}, the first framework in protein pre-training that incorporates both global structural similarity and local amino acid details to enhance prediction accuracy and functional insights. GLProtein innovatively combines protein-masked modelling with triplet structure similarity scoring, protein 3D distance encoding and substructure-based amino acid molecule encoding. Experimental results demonstrate that GLProtein outperforms previous methods in several bioinformatics tasks, including predicting protein-protein interaction, contact prediction, and so on. 

---
# Prediction of Bank Credit Ratings using Heterogeneous Topological Graph Neural Networks 

**Authors**: Junyi Liu, Stanley Kok  

**Link**: [PDF](https://arxiv.org/pdf/2506.06293)  

**Abstract**: Agencies such as Standard & Poor's and Moody's provide bank credit ratings that influence economic stability and decision-making by stakeholders. Accurate and timely predictions support informed decision-making, regulatory actions, and investor protection. However, a complete interbank connection graph is often unavailable due to privacy concerns, complicating the direct application of Graph Neural Networks (GNNs) for rating prediction. our research utilizes persistent homology to construct a network that captures relationships among banks and combines this with a traditional lending network to create a heterogeneous network that integrates information from both sources, leading to improved predictions. Experiments on a global, real-world dataset validate the effectiveness of HTGNN. This research has implications for investors and regulatory bodies in enhancing proactive risk mitigation and the implementation of effective market this http URL code can be find at this https URL. 

---
# Mutual-Taught for Co-adapting Policy and Reward Models 

**Authors**: Tianyuan Shi, Canbin Huang, Fanqi Wan, Longguang Zhong, Ziyi Yang, Weizhou Shen, Xiaojun Quan, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.06292)  

**Abstract**: During the preference optimization of large language models (LLMs), distribution shifts may arise between newly generated model samples and the data used to train the reward model (RM). This shift reduces the efficacy of the RM, which in turn negatively impacts the performance of the policy model (PM). To address this challenge, we propose Mutual-Taught, a self-training method that iteratively improves both the PM and RM without requiring additional human annotation. Our approach mirrors the expectation-maximization (EM) algorithm. In the E-step, the PM is updated using feedback from the current RM, guiding the PM toward a better approximation of the latent optimal preference distribution. In the M-step, we update the RM by constructing training data from the outputs of the PM before and after the E-step update. This process ensures that the RM adapts to the evolving policy distribution. Experimental results demonstrate that this iterative approach leads to consistent improvements in both models. Specifically, our 8B policy model, LLaMA-3-8B-Instruct-MT, achieves a length-controlled win rate of 54.1\% on AlpacaEval-2, while our 8B reward model, FsfairX-LLaMA3-RM-MT, performs on par with GPT-4o-2024-08-06 on RewardBench. 

---
# Improvement of Optimization using Learning Based Models in Mixed Integer Linear Programming Tasks 

**Authors**: Xiaoke Wang, Batuhan Altundas, Zhaoxin Li, Aaron Zhao, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2506.06291)  

**Abstract**: Mixed Integer Linear Programs (MILPs) are essential tools for solving planning and scheduling problems across critical industries such as construction, manufacturing, and logistics. However, their widespread adoption is limited by long computational times, especially in large-scale, real-time scenarios. To address this, we present a learning-based framework that leverages Behavior Cloning (BC) and Reinforcement Learning (RL) to train Graph Neural Networks (GNNs), producing high-quality initial solutions for warm-starting MILP solvers in Multi-Agent Task Allocation and Scheduling Problems. Experimental results demonstrate that our method reduces optimization time and variance compared to traditional techniques while maintaining solution quality and feasibility. 

---
# CellCLIP -- Learning Perturbation Effects in Cell Painting via Text-Guided Contrastive Learning 

**Authors**: Mingyu Lu, Ethan Weinberger, Chanwoo Kim, Su-In Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.06290)  

**Abstract**: High-content screening (HCS) assays based on high-throughput microscopy techniques such as Cell Painting have enabled the interrogation of cells' morphological responses to perturbations at an unprecedented scale. The collection of such data promises to facilitate a better understanding of the relationships between different perturbations and their effects on cellular state. Towards achieving this goal, recent advances in cross-modal contrastive learning could, in theory, be leveraged to learn a unified latent space that aligns perturbations with their corresponding morphological effects. However, the application of such methods to HCS data is not straightforward due to substantial differences in the semantics of Cell Painting images compared to natural images, and the difficulty of representing different classes of perturbations (e.g., small molecule vs CRISPR gene knockout) in a single latent space. In response to these challenges, here we introduce CellCLIP, a cross-modal contrastive learning framework for HCS data. CellCLIP leverages pre-trained image encoders coupled with a novel channel encoding scheme to better capture relationships between different microscopy channels in image embeddings, along with natural language encoders for representing perturbations. Our framework outperforms current open-source models, demonstrating the best performance in both cross-modal retrieval and biologically meaningful downstream tasks while also achieving significant reductions in computation time. 

---
# DELPHYNE: A Pre-Trained Model for General and Financial Time Series 

**Authors**: Xueying Ding, Aakriti Mittal, Achintya Gopal  

**Link**: [PDF](https://arxiv.org/pdf/2506.06288)  

**Abstract**: Time-series data is a vital modality within data science communities. This is particularly valuable in financial applications, where it helps in detecting patterns, understanding market behavior, and making informed decisions based on historical data. Recent advances in language modeling have led to the rise of time-series pre-trained models that are trained on vast collections of datasets and applied to diverse tasks across financial domains. However, across financial applications, existing time-series pre-trained models have not shown boosts in performance over simple finance benchmarks in both zero-shot and fine-tuning settings. This phenomenon occurs because of a i) lack of financial data within the pre-training stage, and ii) the negative transfer effect due to inherently different time-series patterns across domains. Furthermore, time-series data is continuous, noisy, and can be collected at varying frequencies and with varying lags across different variables, making this data more challenging to model than languages. To address the above problems, we introduce a Pre-trained MoDEL for FINance TimE-series (Delphyne). Delphyne achieves competitive performance to existing foundation and full-shot models with few fine-tuning steps on publicly available datasets, and also shows superior performances on various financial tasks. 

---
# Disentangling AI Alignment: A Structured Taxonomy Beyond Safety and Ethics 

**Authors**: Kevin Baum  

**Link**: [PDF](https://arxiv.org/pdf/2506.06286)  

**Abstract**: Recent advances in AI research make it increasingly plausible that artificial agents with consequential real-world impact will soon operate beyond tightly controlled environments. Ensuring that these agents are not only safe but that they adhere to broader normative expectations is thus an urgent interdisciplinary challenge. Multiple fields -- notably AI Safety, AI Alignment, and Machine Ethics -- claim to contribute to this task. However, the conceptual boundaries and interrelations among these domains remain vague, leaving researchers without clear guidance in positioning their work.
To address this meta-challenge, we develop a structured conceptual framework for understanding AI alignment. Rather than focusing solely on alignment goals, we introduce a taxonomy distinguishing the alignment aim (safety, ethicality, legality, etc.), scope (outcome vs. execution), and constituency (individual vs. collective). This structural approach reveals multiple legitimate alignment configurations, providing a foundation for practical and philosophical integration across domains, and clarifying what it might mean for an agent to be aligned all-things-considered. 

---
# Facial Foundational Model Advances Early Warning of Coronary Artery Disease from Live Videos with DigitalShadow 

**Authors**: Juexiao Zhou, Zhongyi Han, Mankun Xin, Xingwei He, Guotao Wang, Jiaoyan Song, Gongning Luo, Wenjia He, Xintong Li, Yuetan Chu, Juanwen Chen, Bo Wang, Xia Wu, Wenwen Duan, Zhixia Guo, Liyan Bai, Yilin Pan, Xuefei Bi, Lu Liu, Long Feng, Xiaonan He, Xin Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.06283)  

**Abstract**: Global population aging presents increasing challenges to healthcare systems, with coronary artery disease (CAD) responsible for approximately 17.8 million deaths annually, making it a leading cause of global mortality. As CAD is largely preventable, early detection and proactive management are essential. In this work, we introduce DigitalShadow, an advanced early warning system for CAD, powered by a fine-tuned facial foundation model. The system is pre-trained on 21 million facial images and subsequently fine-tuned into LiveCAD, a specialized CAD risk assessment model trained on 7,004 facial images from 1,751 subjects across four hospitals in China. DigitalShadow functions passively and contactlessly, extracting facial features from live video streams without requiring active user engagement. Integrated with a personalized database, it generates natural language risk reports and individualized health recommendations. With privacy as a core design principle, DigitalShadow supports local deployment to ensure secure handling of user data. 

---
# STARFlow: Scaling Latent Normalizing Flows for High-resolution Image Synthesis 

**Authors**: Jiatao Gu, Tianrong Chen, David Berthelot, Huangjie Zheng, Yuyang Wang, Ruixiang Zhang, Laurent Dinh, Miguel Angel Bautista, Josh Susskind, Shuangfei Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.06276)  

**Abstract**: We present STARFlow, a scalable generative model based on normalizing flows that achieves strong performance in high-resolution image synthesis. The core of STARFlow is Transformer Autoregressive Flow (TARFlow), which combines the expressive power of normalizing flows with the structured modeling capabilities of Autoregressive Transformers. We first establish the theoretical universality of TARFlow for modeling continuous distributions. Building on this foundation, we introduce several key architectural and algorithmic innovations to significantly enhance scalability: (1) a deep-shallow design, wherein a deep Transformer block captures most of the model representational capacity, complemented by a few shallow Transformer blocks that are computationally efficient yet substantially beneficial; (2) modeling in the latent space of pretrained autoencoders, which proves more effective than direct pixel-level modeling; and (3) a novel guidance algorithm that significantly boosts sample quality. Crucially, our model remains an end-to-end normalizing flow, enabling exact maximum likelihood training in continuous spaces without discretization. STARFlow achieves competitive performance in both class-conditional and text-conditional image generation tasks, approaching state-of-the-art diffusion models in sample quality. To our knowledge, this work is the first successful demonstration of normalizing flows operating effectively at this scale and resolution. 

---
# GOLFer: Smaller LM-Generated Documents Hallucination Filter & Combiner for Query Expansion in Information Retrieval 

**Authors**: Lingyuan Liu, Mengxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.04762)  

**Abstract**: Large language models (LLMs)-based query expansion for information retrieval augments queries with generated hypothetical documents with LLMs. However, its performance relies heavily on the scale of the language models (LMs), necessitating larger, more advanced LLMs. This approach is costly, computationally intensive, and often has limited accessibility. To address these limitations, we introduce GOLFer - Smaller LMs-Generated Documents Hallucination Filter & Combiner - a novel method leveraging smaller open-source LMs for query expansion. GOLFer comprises two modules: a hallucination filter and a documents combiner. The former detects and removes non-factual and inconsistent sentences in generated documents, a common issue with smaller LMs, while the latter combines the filtered content with the query using a weight vector to balance their influence. We evaluate GOLFer alongside dominant LLM-based query expansion methods on three web search and ten low-resource datasets. Experimental results demonstrate that GOLFer consistently outperforms other methods using smaller LMs, and maintains competitive performance against methods using large-size LLMs, demonstrating its effectiveness. 

---
# Low-resource Machine Translation: what for? who for? An observational study on a dedicated Tetun language translation service 

**Authors**: Raphael Merx, Adérito José Guterres Correia, Hanna Suominen, Ekaterina Vylomova  

**Link**: [PDF](https://arxiv.org/pdf/2411.12262)  

**Abstract**: Low-resource machine translation (MT) presents a diversity of community needs and application challenges that remain poorly understood. To complement surveys and focus groups, which tend to rely on small samples of respondents, we propose an observational study on actual usage patterns of tetun$.$org, a specialized MT service for the Tetun language, which is the lingua franca in Timor-Leste. Our analysis of 100,000 translation requests reveals patterns that challenge assumptions based on existing corpora. We find that users, many of them students on mobile devices, typically translate text from a high-resource language into Tetun across diverse domains including science, healthcare, and daily life. This contrasts sharply with available Tetun corpora, which are dominated by news articles covering government and social issues. Our results suggest that MT systems for institutionalized minority languages like Tetun should prioritize accuracy on domains relevant to educational contexts, in the high-resource to low-resource direction. More broadly, this study demonstrates how observational analysis can inform low-resource language technology development, by grounding research in practical community needs. 

---
# MiniGPT-Reverse-Designing: Predicting Image Adjustments Utilizing MiniGPT-4 

**Authors**: Vahid Azizi, Fatemeh Koochaki  

**Link**: [PDF](https://arxiv.org/pdf/2406.00971)  

**Abstract**: Vision-Language Models (VLMs) have recently seen significant advancements through integrating with Large Language Models (LLMs). The VLMs, which process image and text modalities simultaneously, have demonstrated the ability to learn and understand the interaction between images and texts across various multi-modal tasks. Reverse designing, which could be defined as a complex vision-language task, aims to predict the edits and their parameters, given a source image, an edited version, and an optional high-level textual edit description. This task requires VLMs to comprehend the interplay between the source image, the edited version, and the optional textual context simultaneously, going beyond traditional vision-language tasks. In this paper, we extend and fine-tune MiniGPT-4 for the reverse designing task. Our experiments demonstrate the extensibility of off-the-shelf VLMs, specifically MiniGPT-4, for more complex tasks such as reverse designing. Code is available at this \href{this https URL} 

---
# Dual-Modal Attention-Enhanced Text-Video Retrieval with Triplet Partial Margin Contrastive Learning 

**Authors**: Chen Jiang, Hong Liu, Xuzheng Yu, Qing Wang, Yuan Cheng, Jia Xu, Zhongyi Liu, Qingpei Guo, Wei Chu, Ming Yang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2309.11082)  

**Abstract**: In recent years, the explosion of web videos makes text-video retrieval increasingly essential and popular for video filtering, recommendation, and search. Text-video retrieval aims to rank relevant text/video higher than irrelevant ones. The core of this task is to precisely measure the cross-modal similarity between texts and videos. Recently, contrastive learning methods have shown promising results for text-video retrieval, most of which focus on the construction of positive and negative pairs to learn text and video representations. Nevertheless, they do not pay enough attention to hard negative pairs and lack the ability to model different levels of semantic similarity. To address these two issues, this paper improves contrastive learning using two novel techniques. First, to exploit hard examples for robust discriminative power, we propose a novel Dual-Modal Attention-Enhanced Module (DMAE) to mine hard negative pairs from textual and visual clues. By further introducing a Negative-aware InfoNCE (NegNCE) loss, we are able to adaptively identify all these hard negatives and explicitly highlight their impacts in the training loss. Second, our work argues that triplet samples can better model fine-grained semantic similarity compared to pairwise samples. We thereby present a new Triplet Partial Margin Contrastive Learning (TPM-CL) module to construct partial order triplet samples by automatically generating fine-grained hard negatives for matched text-video pairs. The proposed TPM-CL designs an adaptive token masking strategy with cross-modal interaction to model subtle semantic differences. Extensive experiments demonstrate that the proposed approach outperforms existing methods on four widely-used text-video retrieval datasets, including MSR-VTT, MSVD, DiDeMo and ActivityNet. 

---
