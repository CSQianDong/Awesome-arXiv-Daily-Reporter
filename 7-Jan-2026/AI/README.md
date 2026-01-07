# MAGMA: A Multi-Graph based Agentic Memory Architecture for AI Agents 

**Authors**: Dongming Jiang, Yi Li, Guanpeng Li, Bingzhe Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03236)  

**Abstract**: Memory-Augmented Generation (MAG) extends Large Language Models with external memory to support long-context reasoning, but existing approaches largely rely on semantic similarity over monolithic memory stores, entangling temporal, causal, and entity information. This design limits interpretability and alignment between query intent and retrieved evidence, leading to suboptimal reasoning accuracy. In this paper, we propose MAGMA, a multi-graph agentic memory architecture that represents each memory item across orthogonal semantic, temporal, causal, and entity graphs. MAGMA formulates retrieval as policy-guided traversal over these relational views, enabling query-adaptive selection and structured context construction. By decoupling memory representation from retrieval logic, MAGMA provides transparent reasoning paths and fine-grained control over retrieval. Experiments on LoCoMo and LongMemEval demonstrate that MAGMA consistently outperforms state-of-the-art agentic memory systems in long-horizon reasoning tasks. 

---
# InfiAgent: An Infinite-Horizon Framework for General-Purpose Autonomous Agents 

**Authors**: Chenglin Yu, Yuchen Wang, Songmiao Wang, Hongxia Yang, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03204)  

**Abstract**: LLM agents can reason and use tools, but they often break down on long-horizon tasks due to unbounded context growth and accumulated errors. Common remedies such as context compression or retrieval-augmented prompting introduce trade-offs between information fidelity and reasoning stability. We present InfiAgent, a general-purpose framework that keeps the agent's reasoning context strictly bounded regardless of task duration by externalizing persistent state into a file-centric state abstraction. At each step, the agent reconstructs context from a workspace state snapshot plus a fixed window of recent actions. Experiments on DeepResearch and an 80-paper literature review task show that, without task-specific fine-tuning, InfiAgent with a 20B open-source model is competitive with larger proprietary systems and maintains substantially higher long-horizon coverage than context-centric baselines. These results support explicit state externalization as a practical foundation for stable long-horizon agents. Github Repo:this https URL 

---
# Automatic Prompt Engineering with No Task Cues and No Tuning 

**Authors**: Faisal Chowdhury, Nandana Mihindukulasooriya, Niharika S D'Souza, Horst Samulowitz, Neeru Gupta, Tomasz Hanusiak, Michal Kapitonow  

**Link**: [PDF](https://arxiv.org/pdf/2601.03130)  

**Abstract**: This paper presents a system for automatic prompt engineering that is much simpler in both design and application and yet as effective as the existing approaches. It requires no tuning and no explicit clues about the task. We evaluated our approach on cryptic column name expansion (CNE) in database tables, a task which is critical for tabular data search, access, and understanding and yet there has been very little existing work. We evaluated on datasets in two languages, English and German. This is the first work to report on the application of automatic prompt engineering for the CNE task. To the best of our knowledge, this is also the first work on the application of automatic prompt engineering for a language other than English. 

---
# A framework for assuring the accuracy and fidelity of an AI-enabled Digital Twin of en route UK airspace 

**Authors**: Adam Keane, Nick Pepper, Chris Burr, Amy Hodgkin, Dewi Gould, John Korna, Marc Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2601.03120)  

**Abstract**: Digital Twins combine simulation, operational data and Artificial Intelligence (AI), and have the potential to bring significant benefits across the aviation industry. Project Bluebird, an industry-academic collaboration, has developed a probabilistic Digital Twin of en route UK airspace as an environment for training and testing AI Air Traffic Control (ATC) agents. There is a developing regulatory landscape for this kind of novel technology. Regulatory requirements are expected to be application specific, and may need to be tailored to each specific use case.
We draw on emerging guidance for both Digital Twin development and the use of Artificial Intelligence/Machine Learning (AI/ML) in Air Traffic Management (ATM) to present an assurance framework. This framework defines actionable goals and the evidence required to demonstrate that a Digital Twin accurately represents its physical counterpart and also provides sufficient functionality across target use cases. It provides a structured approach for researchers to assess, understand and document the strengths and limitations of the Digital Twin, whilst also identifying areas where fidelity could be improved. Furthermore, it serves as a foundation for engagement with stakeholders and regulators, supporting discussions around the regulatory needs for future applications, and contributing to the emerging guidance through a concrete, working example of a Digital Twin.
The framework leverages a methodology known as Trustworthy and Ethical Assurance (TEA) to develop an assurance case. An assurance case is a nested set of structured arguments that provides justified evidence for how a top-level goal has been realised. In this paper we provide an overview of each structured argument and a number of deep dives which elaborate in more detail upon particular arguments, including the required evidence, assumptions and justifications. 

---
# Explainable Fuzzy GNNs for Leak Detection in Water Distribution Networks 

**Authors**: Qusai Khaled, Pasquale De Marinis, Moez Louati, David Ferras, Laura Genga, Uzay Kaymak  

**Link**: [PDF](https://arxiv.org/pdf/2601.03062)  

**Abstract**: Timely leak detection in water distribution networks is critical for conserving resources and maintaining operational efficiency. Although Graph Neural Networks (GNNs) excel at capturing spatial-temporal dependencies in sensor data, their black-box nature and the limited work on graph-based explainable models for water networks hinder practical adoption. We propose an explainable GNN framework that integrates mutual information to identify critical network regions and fuzzy logic to provide clear, rule-based explanations for node classification tasks. After benchmarking several GNN architectures, we selected the generalized graph convolution network (GENConv) for its superior performance and developed a fuzzy-enhanced variant that offers intuitive explanations for classified leak locations. Our fuzzy graph neural network (FGENConv) achieved Graph F1 scores of 0.889 for detection and 0.814 for localization, slightly below the crisp GENConv 0.938 and 0.858, respectively. Yet it compensates by providing spatially localized, fuzzy rule-based explanations. By striking the right balance between precision and explainability, the proposed fuzzy network could enable hydraulic engineers to validate predicted leak locations, conserve human resources, and optimize maintenance strategies. The code is available at this http URL. 

---
# Rationale-Grounded In-Context Learning for Time Series Reasoning with Multimodal Large Language Models 

**Authors**: Qingxiang Liu, Zhiqing Cui, Xiaoliang Luo, Yuqian Wu, Zhuoyang Jiang, Huaiyu Wan, Sheng Sun, Lvchun Wang, Wei Yu, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02968)  

**Abstract**: The underperformance of existing multimodal large language models for time series reasoning lies in the absence of rationale priors that connect temporal observations to their downstream outcomes, which leads models to rely on superficial pattern matching rather than principled reasoning. We therefore propose the rationale-grounded in-context learning for time series reasoning, where rationales work as guiding reasoning units rather than post-hoc explanations, and develop the RationaleTS method. Specifically, we firstly induce label-conditioned rationales, composed of reasoning paths from observable evidence to the potential outcomes. Then, we design the hybrid retrieval by balancing temporal patterns and semantic contexts to retrieve correlated rationale priors for the final in-context inference on new samples. We conduct extensive experiments to demonstrate the effectiveness and efficiency of our proposed RationaleTS on three-domain time series reasoning tasks. We will release our code for reproduction. 

---
# Batch-of-Thought: Cross-Instance Learning for Enhanced LLM Reasoning 

**Authors**: Xuan Yang, Furong Jia, Roy Xie, Xiong Xi, Hengwei Bian, Jian Li, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.02950)  

**Abstract**: Current Large Language Model reasoning systems process queries independently, discarding valuable cross-instance signals such as shared reasoning patterns and consistency constraints. We introduce Batch-of-Thought (BoT), a training-free method that processes related queries jointly to enable cross-instance learning. By performing comparative analysis across batches, BoT identifies high-quality reasoning templates, detects errors through consistency checks, and amortizes computational costs. We instantiate BoT within a multi-agent reflection architecture (BoT-R), where a Reflector performs joint evaluation to unlock mutual information gain unavailable in isolated processing. Experiments across three model families and six benchmarks demonstrate that BoT-R consistently improves accuracy and confidence calibration while reducing inference costs by up to 61%. Our theoretical and experimental analysis reveals when and why batch-aware reasoning benefits LLM systems. 

---
# Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning 

**Authors**: Xinglang Zhang, Yunyao Zhang, ZeLiang Chen, Junqing Yu, Wei Yang, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.02902)  

**Abstract**: Symbolic logical reasoning is a critical yet underexplored capability of large language models (LLMs), providing reliable and verifiable decision-making in high-stakes domains such as mathematical reasoning and legal judgment. In this study, we present a systematic analysis of logical reasoning under controlled increases in logical complexity, and reveal a previously unrecognized phenomenon, which we term Logical Phase Transitions: rather than degrading smoothly, logical reasoning performance remains stable within a regime but collapses abruptly beyond a critical logical depth, mirroring physical phase transitions such as water freezing beyond a critical temperature threshold. Building on this insight, we propose Neuro-Symbolic Curriculum Tuning, a principled framework that adaptively aligns natural language with logical symbols to establish a shared representation, and reshapes training dynamics around phase-transition boundaries to progressively strengthen reasoning at increasing logical depths. Experiments on five benchmarks show that our approach effectively mitigates logical reasoning collapse at high complexity, yielding average accuracy gains of +1.26 in naive prompting and +3.95 in CoT, while improving generalization to unseen logical compositions. Code and data are available at this https URL. 

---
# ReTreVal: Reasoning Tree with Validation - A Hybrid Framework for Enhanced LLM Multi-Step Reasoning 

**Authors**: Abhishek HS, Pavan C Shekar, Arpit Jain, Ashwanth Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02880)  

**Abstract**: Multi-step reasoning remains a key challenge for Large Language Models (LLMs), particularly in complex domains such as mathematics and creative writing. While recent approaches including ReAct, Reflexion, and Self-Refine improve reasoning through iterative refinement and reflection, they often lack structured exploration of alternative solution paths and persistent learning across problems. We propose ReTreVal (Reasoning Tree with Validation), a hybrid framework that integrates Tree-of-Thoughts exploration, self-refinement, LLM-based critique scoring, and reflexion memory to enable bounded and validated multi-step reasoning. ReTreVal constructs a structured reasoning tree with adaptive depth based on problem complexity, where each node undergoes iterative self-critique and refinement guided by explicit LLM-generated feedback. A dual validation mechanism evaluates reasoning quality, coherence, and correctness at each node while persistently storing insights from successful reasoning paths and failure patterns in a reflexion memory buffer, enabling cross-problem learning. Critique-based pruning retains only the top-k highest-scoring nodes at each level, controlling computational cost while preserving high-quality solution paths. We evaluate ReTreVal against ReAct, Reflexion, and Self-Refine across 500 mathematical problems and creative writing tasks using Qwen 2.5 7B as the underlying LLM, and demonstrate that ReTreVal consistently outperforms existing methods through its combination of structured exploration, critique-driven refinement, and cross-problem memory, making it particularly effective for tasks requiring exploratory reasoning, rigorous verification, and knowledge transfer. 

---
# SimRPD: Optimizing Recruitment Proactive Dialogue Agents through Simulator-Based Data Evaluation and Selection 

**Authors**: Zhiyong Cao, Dunqiang Liu, Qi Dai, Haojun Xu, Huaiyan Xu, Huan He, Yafei Liu, Siyuan Liu, XiaoLin Lin, Ke Ma, Ruqian Shi, Sijia Yao, Hao Wang, Sicheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.02871)  

**Abstract**: Task-oriented proactive dialogue agents play a pivotal role in recruitment, particularly for steering conversations towards specific business outcomes, such as acquiring social-media contacts for private-channel conversion. Although supervised fine-tuning and reinforcement learning have proven effective for training such agents, their performance is heavily constrained by the scarcity of high-quality, goal-oriented domain-specific training data. To address this challenge, we propose SimRPD, a three-stage framework for training recruitment proactive dialogue agents. First, we develop a high-fidelity user simulator to synthesize large-scale conversational data through multi-turn online dialogue. Then we introduce a multi-dimensional evaluation framework based on Chain-of-Intention (CoI) to comprehensively assess the simulator and effectively select high-quality data, incorporating both global-level and instance-level metrics. Finally, we train the recruitment proactive dialogue agent on the selected dataset. Experiments in a real-world recruitment scenario demonstrate that SimRPD outperforms existing simulator-based data selection strategies, highlighting its practical value for industrial deployment and its potential applicability to other business-oriented dialogue scenarios. 

---
# M3MAD-Bench: Are Multi-Agent Debates Really Effective Across Domains and Modalities? 

**Authors**: Ao Li, Jinghui Zhang, Luyu Li, Yuxiang Duan, Lang Gao, Mingcai Chen, Weijun Qin, Shaopeng Li, Fengxian Ji, Ning Liu, Lizhen Cui, Xiuying Chen, Yuntao Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.02854)  

**Abstract**: As an agent-level reasoning and coordination paradigm, Multi-Agent Debate (MAD) orchestrates multiple agents through structured debate to improve answer quality and support complex reasoning. However, existing research on MAD suffers from two fundamental limitations: evaluations are conducted under fragmented and inconsistent settings, hindering fair comparison, and are largely restricted to single-modality scenarios that rely on textual inputs only. To address these gaps, we introduce M3MAD-Bench, a unified and extensible benchmark for evaluating MAD methods across Multi-domain tasks, Multi-modal inputs, and Multi-dimensional metrics. M3MAD-Bench establishes standardized protocols over five core task domains: Knowledge, Mathematics, Medicine, Natural Sciences, and Complex Reasoning, and systematically covers both pure text and vision-language datasets, enabling controlled cross-modality comparison. We evaluate MAD methods on nine base models spanning different architectures, scales, and modality capabilities. Beyond accuracy, M3MAD-Bench incorporates efficiency-oriented metrics such as token consumption and inference time, providing a holistic view of performance--cost trade-offs. Extensive experiments yield systematic insights into the effectiveness, robustness, and efficiency of MAD across text-only and multimodal scenarios. We believe M3MAD-Bench offers a reliable foundation for future research on standardized MAD evaluation. The code is available at this http URL. 

---
# Sample-Efficient Neurosymbolic Deep Reinforcement Learning 

**Authors**: Celeste Veronese, Daniele Meli, Alessandro Farinelli  

**Link**: [PDF](https://arxiv.org/pdf/2601.02850)  

**Abstract**: Reinforcement Learning (RL) is a well-established framework for sequential decision-making in complex environments. However, state-of-the-art Deep RL (DRL) algorithms typically require large training datasets and often struggle to generalize beyond small-scale training scenarios, even within standard benchmarks. We propose a neuro-symbolic DRL approach that integrates background symbolic knowledge to improve sample efficiency and generalization to more challenging, unseen tasks. Partial policies defined for simple domain instances, where high performance is easily attained, are transferred as useful priors to accelerate learning in more complex settings and avoid tuning DRL parameters from scratch. To do so, partial policies are represented as logical rules, and online reasoning is performed to guide the training process through two mechanisms: (i) biasing the action distribution during exploration, and (ii) rescaling Q-values during exploitation. This neuro-symbolic integration enhances interpretability and trustworthiness while accelerating convergence, particularly in sparse-reward environments and tasks with long planning horizons. We empirically validate our methodology on challenging variants of gridworld environments, both in the fully observable and partially observable setting. We show improved performance over a state-of-the-art reward machine baseline. 

---
# Quantum-enhanced long short-term memory with attention for spatial permeability prediction in oilfield reservoirs 

**Authors**: Muzhen Zhang, Yujie Cheng, Zhanxiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2601.02818)  

**Abstract**: Spatial prediction of reservoir parameters, especially permeability, is crucial for oil and gas exploration and development. However, the wide range and high variability of permeability prevent existing methods from providing reliable predictions. For the first time in subsurface spatial prediction, this study presents a quantum-enhanced long short-term memory with attention (QLSTMA) model that incorporates variational quantum circuits (VQCs) into the recurrent cell. Using quantum entanglement and superposition principles, the QLSTMA significantly improves the ability to predict complex geological parameters such as permeability. Two quantization structures, QLSTMA with Shared Gates (QLSTMA-SG) and with Independent Gates (QLSTMA-IG), are designed to investigate and evaluate the effects of quantum structure configurations and the number of qubits on model performance. Experimental results demonstrate that the 8-qubit QLSTMA-IG model significantly outperforms the traditional long short-term memory with attention (LSTMA), reducing Mean Absolute Error (MAE) by 19% and Root Mean Squared Error (RMSE) by 20%, with particularly strong performance in regions featuring complex well-logging data. These findings validate the potential of quantum-classical hybrid neural networks for reservoir prediction, indicating that increasing the number of qubits yields further accuracy gains despite the reliance on classical simulations. This study establishes a foundational framework for the eventual deployment of such models on real quantum hardware and their extension to broader applications in petroleum engineering and geoscience. 

---
# Causal-Enhanced AI Agents for Medical Research Screening 

**Authors**: Duc Ngo, Arya Rahgoza  

**Link**: [PDF](https://arxiv.org/pdf/2601.02814)  

**Abstract**: Systematic reviews are essential for evidence-based medicine, but reviewing 1.5 million+ annual publications manually is infeasible. Current AI approaches suffer from hallucinations in systematic review tasks, with studies reporting rates ranging from 28--40% for earlier models to 2--15% for modern implementations which is unacceptable when errors impact patient care.
We present a causal graph-enhanced retrieval-augmented generation system integrating explicit causal reasoning with dual-level knowledge graphs. Our approach enforces evidence-first protocols where every causal claim traces to retrieved literature and automatically generates directed acyclic graphs visualizing intervention-outcome pathways.
Evaluation on 234 dementia exercise abstracts shows CausalAgent achieves 95% accuracy, 100% retrieval success, and zero hallucinations versus 34% accuracy and 10% hallucinations for baseline AI. Automatic causal graphs enable explicit mechanism modeling, visual synthesis, and enhanced interpretability. While this proof-of-concept evaluation used ten questions focused on dementia exercise research, the architectural approach demonstrates transferable principles for trustworthy medical AI and causal reasoning's potential for high-stakes healthcare. 

---
# HAL: Inducing Human-likeness in LLMs with Alignment 

**Authors**: Masum Hasan, Junjie Zhao, Ehsan Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2601.02813)  

**Abstract**: Conversational human-likeness plays a central role in human-AI interaction, yet it has remained difficult to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale human evaluations, models aligned with HAL are more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way. 

---
# LLM Agent Framework for Intelligent Change Analysis in Urban Environment using Remote Sensing Imagery 

**Authors**: Zixuan Xiao, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.02757)  

**Abstract**: Existing change detection methods often lack the versatility to handle diverse real-world queries and the intelligence for comprehensive analysis. This paper presents a general agent framework, integrating Large Language Models (LLM) with vision foundation models to form ChangeGPT. A hierarchical structure is employed to mitigate hallucination. The agent was evaluated on a curated dataset of 140 questions categorized by real-world scenarios, encompassing various question types (e.g., Size, Class, Number) and complexities. The evaluation assessed the agent's tool selection ability (Precision/Recall) and overall query accuracy (Match). ChangeGPT, especially with a GPT-4-turbo backend, demonstrated superior performance, achieving a 90.71 % Match rate. Its strength lies particularly in handling change-related queries requiring multi-step reasoning and robust tool selection. Practical effectiveness was further validated through a real-world urban change monitoring case study in Qianhai Bay, Shenzhen. By providing intelligence, adaptability, and multi-type change analysis, ChangeGPT offers a powerful solution for decision-making in remote sensing applications. 

---
# The Path Ahead for Agentic AI: Challenges and Opportunities 

**Authors**: Nadia Sibai, Yara Ahmed, Serry Sibaee, Sawsan AlHalawani, Adel Ammar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2601.02749)  

**Abstract**: The evolution of Large Language Models (LLMs) from passive text generators to autonomous, goal-driven systems represents a fundamental shift in artificial intelligence. This chapter examines the emergence of agentic AI systems that integrate planning, memory, tool use, and iterative reasoning to operate autonomously in complex environments. We trace the architectural progression from statistical models to transformer-based systems, identifying capabilities that enable agentic behavior: long-range reasoning, contextual awareness, and adaptive decision-making. The chapter provides three contributions: (1) a synthesis of how LLM capabilities extend toward agency through reasoning-action-reflection loops; (2) an integrative framework describing core components perception, memory, planning, and tool execution that bridge LLMs with autonomous behavior; (3) a critical assessment of applications and persistent challenges in safety, alignment, reliability, and sustainability. Unlike existing surveys, we focus on the architectural transition from language understanding to autonomous action, emphasizing the technical gaps that must be resolved before deployment. We identify critical research priorities, including verifiable planning, scalable multi-agent coordination, persistent memory architectures, and governance frameworks. Responsible advancement requires simultaneous progress in technical robustness, interpretability, and ethical safeguards to realize potential while mitigating risks of misalignment and unintended consequences. 

---
# Time-Scaling Is What Agents Need Now 

**Authors**: Zhi Liu, Guangzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02714)  

**Abstract**: Early artificial intelligence paradigms exhibited separated cognitive functions: Neural Networks focused on "perception-representation," Reinforcement Learning on "decision-making-behavior," and Symbolic AI on "knowledge-reasoning." With Transformer-based large models and world models, these paradigms are converging into cognitive agents with closed-loop "perception-decision-action" capabilities.
Humans solve complex problems under limited cognitive resources through temporalized sequential reasoning. Language relies on problem space search for deep semantic reasoning. While early large language models (LLMs) could generate fluent text, they lacked robust semantic reasoning capabilities. Prompting techniques like Chain-of-Thought (CoT) and Tree-of-Thought (ToT) extended reasoning paths by making intermediate steps explicit. Recent models like DeepSeek-R1 enhanced performance through explicit reasoning trajectories. However, these methods have limitations in search completeness and efficiency.
This highlights the need for "Time-Scaling"--the systematic extension and optimization of an agent's ability to unfold reasoning over time. Time-Scaling refers to architectural design utilizing extended temporal pathways, enabling deeper problem space exploration, dynamic strategy adjustment, and enhanced metacognitive control, paralleling human sequential reasoning under cognitive constraints. It represents a critical frontier for enhancing deep reasoning and problem-solving without proportional increases in static model parameters. Advancing intelligent agent capabilities requires placing Time-Scaling principles at the forefront, positioning explicit temporal reasoning management as foundational. 

---
# Learning User Preferences Through Interaction for Long-Term Collaboration 

**Authors**: Shuhaib Mehri, Priyanka Kargupta, Tal August, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2601.02702)  

**Abstract**: As conversational agents accumulate experience collaborating with users, adapting to user preferences is essential for fostering long-term relationships and improving collaboration quality over time. We introduce MultiSessionCollab, a benchmark that evaluates how well agents can learn user preferences and leverage them to improve collaboration quality throughout multiple sessions. To develop agents that succeed in this setting, we present long-term collaborative agents equipped with a memory that persists and refines user preference as interaction experience accumulates. Moreover, we demonstrate that learning signals can be derived from user simulator behavior in MultiSessionCollab to train agents to generate more comprehensive reflections and update their memory more effectively. Extensive experiments show that equipping agents with memory improves long-term collaboration, yielding higher task success rates, more efficient interactions, and reduced user effort. Finally, we conduct a human user study that demonstrates that memory helps improve user experience in real-world settings. 

---
# Learning from Prompt itself: the Hierarchical Attribution Prompt Optimization 

**Authors**: Dongyu Chen, Jian Ma, Xianpeng Zhang, Lei Zhang, Haonan Lu, Chen Chen, Chuangchuang Wang, Kai Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02683)  

**Abstract**: Optimization is fundamental across numerous disciplines, typically following an iterative process of refining an initial solution to enhance performance. This principle is equally critical in prompt engineering, where designing effective prompts for large language models constitutes a complex optimization challenge. A structured optimization approach requires automated or semi-automated procedures to develop improved prompts, thereby reducing manual effort, improving performance, and yielding an interpretable process. However, current prompt optimization methods often induce prompt drift, where new prompts fix prior failures but impair performance on previously successful tasks. Additionally, generating prompts from scratch can compromise interpretability. To address these limitations, this study proposes the Hierarchical Attribution Prompt Optimization (HAPO) framework, which introduces three innovations: (1) a dynamic attribution mechanism targeting error patterns in training data and prompting history, (2) semantic-unit optimization for editing functional prompt segments, and (3) multimodal-friendly progression supporting both end-to-end LLM and LLM-MLLM workflows. Applied in contexts like single/multi-image QA (e.g., OCRV2) and complex task analysis (e.g., BBH), HAPO demonstrates enhanced optimization efficiency, outperforming comparable automated prompt optimization methods and establishing an extensible paradigm for scalable prompt engineering. 

---
# Inferring Causal Graph Temporal Logic Formulas to Expedite Reinforcement Learning in Temporally Extended Tasks 

**Authors**: Hadi Partovi Aria, Zhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02666)  

**Abstract**: Decision-making tasks often unfold on graphs with spatial-temporal dynamics. Black-box reinforcement learning often overlooks how local changes spread through network structure, limiting sample efficiency and interpretability. We present GTL-CIRL, a closed-loop framework that simultaneously learns policies and mines Causal Graph Temporal Logic (Causal GTL) specifications. The method shapes rewards with robustness, collects counterexamples when effects fail, and uses Gaussian Process (GP) driven Bayesian optimization to refine parameterized cause templates. The GP models capture spatial and temporal correlations in the system dynamics, enabling efficient exploration of complex parameter spaces. Case studies in gene and power networks show faster learning and clearer, verifiable behavior compared to standard RL baselines. 

---
# AWARE-US: Benchmark for Preference-Aware Resolution in Tool-Calling Agents 

**Authors**: Mehmet Kurmaz  

**Link**: [PDF](https://arxiv.org/pdf/2601.02643)  

**Abstract**: Tool-calling conversational agents querying structured databases often face two linked failures: underspecification (missing constraints needed to run a precise query) and infeasibility (the fully specified query returns an empty set because no item satisfies all constraints). Existing work often responds with "no results" or relaxes constraints using ad hoc rules, which can violate user intent by discarding requirements the user cares about most. We frame infeasibility handling as a preference-aware query repair problem: when a query is unsatisfiable, the agent should relax the least important constraints to the user. We propose three LLM-based methods for inferring relative constraint importance from dialogue: (1) local weighting, (2) global one-shot weighting, and (3) pairwise ranking. Experiments show local weighting achieves the best preference alignment, while global weighting performs best on correct constraint relaxation. We also introduce AWARE-US, a benchmark of persona-grounded queries requiring agents to disambiguate requests via conversation and resolve infeasibility in a way consistent with persona-implied preferences. 

---
# An Empirical Study of On-Device Translation for Real-Time Live-Stream Chat on Mobile Devices 

**Authors**: Jeiyoon Park, Daehwan Lee, Changmin Yeo, Yongshin Han, Minseop Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.02641)  

**Abstract**: Despite its efficiency, there has been little research on the practical aspects required for real-world deployment of on-device AI models, such as the device's CPU utilization and thermal conditions. In this paper, through extensive experiments, we investigate two key issues that must be addressed to deploy on-device models in real-world services: (i) the selection of on-device models and the resource consumption of each model, and (ii) the capability and potential of on-device models for domain adaptation. To this end, we focus on a task of translating live-stream chat messages and manually construct LiveChatBench, a benchmark consisting of 1,000 Korean-English parallel sentence pairs. Experiments on five mobile devices demonstrate that, although serving a large and heterogeneous user base requires careful consideration of highly constrained deployment settings and model selection, the proposed approach nevertheless achieves performance comparable to commercial models such as GPT-5.1 on the well-targeted task. We expect that our findings will provide meaningful insights to the on-device AI community. 

---
# Orchestral AI: A Framework for Agent Orchestration 

**Authors**: Alexander Roman, Jacob Roman  

**Link**: [PDF](https://arxiv.org/pdf/2601.02577)  

**Abstract**: The rapid proliferation of LLM agent frameworks has forced developers to choose between vendor lock-in through provider-specific SDKs and complex multi-package ecosystems that obscure control flow and hinder reproducibility. Integrating tool calling across multiple LLM providers remains a core engineering challenge due to fragmented APIs, incompatible message formats, and inconsistent streaming and tool-calling behavior, making it difficult to build portable, reliable agent systems. We introduce Orchestral, a lightweight Python framework that provides a unified, type-safe interface for building LLM agents across major providers while preserving the simplicity required for scientific computing and production deployment. Orchestral defines a single universal representation for messages, tools, and LLM usage that operates seamlessly across providers, eliminating manual format translation and reducing framework-induced complexity. Automatic tool schema generation from Python type hints removes the need for handwritten descriptors while maintaining type safety across provider boundaries. A synchronous execution model with streaming support enables deterministic behavior, straightforward debugging, and real-time interaction without introducing server dependencies. The framework's modular architecture cleanly separates provider integration, tool execution, conversation orchestration, and user-facing interfaces, enabling extensibility without architectural entanglement. Orchestral supports advanced agent capabilities found in larger frameworks, including rich tool calling, context compaction, workspace sandboxing, user approval workflows, sub-agents, memory management, and MCP integration. 

---
# SimpleMem: Efficient Lifelong Memory for LLM Agents 

**Authors**: Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.02553)  

**Abstract**: To support reliable long-term interaction in complex environments, LLM agents require memory systems that efficiently manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) \textit{Semantic Structured Compression}, which applies entropy-aware filtering to distill unstructured interactions into compact, multi-view indexed memory units; (2) \textit{Recursive Memory Consolidation}, an asynchronous process that integrates related units into higher-level abstract representations to reduce redundancy; and (3) \textit{Adaptive Query-Aware Retrieval}, which dynamically adjusts retrieval scope based on query complexity to construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at this https URL. 

---
# Textual Explanations and Their Evaluations for Reinforcement Learning Policy 

**Authors**: Ahmad Terra, Mohit Ahmed, Rafia Inam, Elena Fersman, Martin Törngren  

**Link**: [PDF](https://arxiv.org/pdf/2601.02514)  

**Abstract**: Understanding a Reinforcement Learning (RL) policy is crucial for ensuring that autonomous agents behave according to human expectations. This goal can be achieved using Explainable Reinforcement Learning (XRL) techniques. Although textual explanations are easily understood by humans, ensuring their correctness remains a challenge, and evaluations in state-of-the-art remain limited. We present a novel XRL framework for generating textual explanations, converting them into a set of transparent rules, improving their quality, and evaluating them. Expert's knowledge can be incorporated into this framework, and an automatic predicate generator is also proposed to determine the semantic information of a state. Textual explanations are generated using a Large Language Model (LLM) and a clustering technique to identify frequent conditions. These conditions are then converted into rules to evaluate their properties, fidelity, and performance in the deployed environment. Two refinement techniques are proposed to improve the quality of explanations and reduce conflicting information. Experiments were conducted in three open-source environments to enable reproducibility, and in a telecom use case to evaluate the industrial applicability of the proposed XRL framework. This framework addresses the limitations of an existing method, Autonomous Policy Explanation, and the generated transparent rules can achieve satisfactory performance on certain tasks. This framework also enables a systematic and quantitative evaluation of textual explanations, providing valuable insights for the XRL field. 

---
# Multi-RADS Synthetic Radiology Report Dataset and Head-to-Head Benchmarking of 41 Open-Weight and Proprietary Language Models 

**Authors**: Kartik Bose, Abhinandan Kumar, Raghuraman Soundararajan, Priya Mudgil, Samonee Ralmilay, Niharika Dutta, Manphool Singhal, Arun Kumar, Saugata Sen, Anurima Patra, Priya Ghosh, Abanti Das, Amit Gupta, Ashish Verma, Dipin Sudhakaran, Ekta Dhamija, Himangi Unde, Ishan Kumar, Krithika Rangarajan, Prerna Garg, Rachel Sequeira, Sudhin Shylendran, Taruna Yadav, Tej Pal, Pankaj Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2601.03232)  

**Abstract**: Background: Reporting and Data Systems (RADS) standardize radiology risk communication but automated RADS assignment from narrative reports is challenging because of guideline complexity, output-format constraints, and limited benchmarking across RADS frameworks and model sizes. Purpose: To create RXL-RADSet, a radiologist-verified synthetic multi-RADS benchmark, and compare validity and accuracy of open-weight small language models (SLMs) with a proprietary model for RADS assignment. Materials and Methods: RXL-RADSet contains 1,600 synthetic radiology reports across 10 RADS (BI-RADS, CAD-RADS, GB-RADS, LI-RADS, Lung-RADS, NI-RADS, O-RADS, PI-RADS, TI-RADS, VI-RADS) and multiple modalities. Reports were generated by LLMs using scenario plans and simulated radiologist styles and underwent two-stage radiologist verification. We evaluated 41 quantized SLMs (12 families, 0.135-32B parameters) and GPT-5.2 under a fixed guided prompt. Primary endpoints were validity and accuracy; a secondary analysis compared guided versus zero-shot prompting. Results: Under guided prompting GPT-5.2 achieved 99.8% validity and 81.1% accuracy (1,600 predictions). Pooled SLMs (65,600 predictions) achieved 96.8% validity and 61.1% accuracy; top SLMs in the 20-32B range reached ~99% validity and mid-to-high 70% accuracy. Performance scaled with model size (inflection between <1B and >=10B) and declined with RADS complexity primarily due to classification difficulty rather than invalid outputs. Guided prompting improved validity (99.2% vs 96.7%) and accuracy (78.5% vs 69.6%) compared with zero-shot. Conclusion: RXL-RADSet provides a radiologist-verified multi-RADS benchmark; large SLMs (20-32B) can approach proprietary-model performance under guided prompting, but gaps remain for higher-complexity schemes. 

---
# The Sonar Moment: Benchmarking Audio-Language Models in Audio Geo-Localization 

**Authors**: Ruixing Zhang, Zihan Liu, Leilei Sun, Tongyu Zhu, Weifeng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2601.03227)  

**Abstract**: Geo-localization aims to infer the geographic origin of a given signal. In computer vision, geo-localization has served as a demanding benchmark for compositional reasoning and is relevant to public safety. In contrast, progress on audio geo-localization has been constrained by the lack of high-quality audio-location pairs. To address this gap, we introduce AGL1K, the first audio geo-localization benchmark for audio language models (ALMs), spanning 72 countries and territories. To extract reliably localizable samples from a crowd-sourced platform, we propose the Audio Localizability metric that quantifies the informativeness of each recording, yielding 1,444 curated audio clips. Evaluations on 16 ALMs show that ALMs have emerged with audio geo-localization capability. We find that closed-source models substantially outperform open-source models, and that linguistic clues often dominate as a scaffold for prediction. We further analyze ALMs' reasoning traces, regional bias, error causes, and the interpretability of the localizability metric. Overall, AGL1K establishes a benchmark for audio geo-localization and may advance ALMs with better geospatial reasoning capability. 

---
# The Fake Friend Dilemma: Trust and the Political Economy of Conversational AI 

**Authors**: Jacob Erickson  

**Link**: [PDF](https://arxiv.org/pdf/2601.03222)  

**Abstract**: As conversational AI systems become increasingly integrated into everyday life, they raise pressing concerns about user autonomy, trust, and the commercial interests that influence their behavior. To address these concerns, this paper develops the Fake Friend Dilemma (FFD), a sociotechnical condition in which users place trust in AI agents that appear supportive while pursuing goals that are misaligned with the user's own. The FFD provides a critical framework for examining how anthropomorphic AI systems facilitate subtle forms of manipulation and exploitation. Drawing on literature in trust, AI alignment, and surveillance capitalism, we construct a typology of harms, including covert advertising, political propaganda, behavioral nudging, and surveillance. We then assess possible mitigation strategies, including both structural and technical interventions. By focusing on trust as a vector of asymmetrical power, the FFD offers a lens for understanding how AI systems may undermine user autonomy while maintaining the appearance of helpfulness. 

---
# Fine-tuning Small Language Models as Efficient Enterprise Search Relevance Labelers 

**Authors**: Yue Kang, Zhuoyi Huang, Benji Schussheim, Diana Licon, Dina Atia, Shixing Cao, Jacob Danovitch, Kunho Kim, Billy Norcilien, Jonah Karpman, Mahmound Sayed, Mike Taylor, Tao Sun, Pavel Metrikov, Vipul Agarwal, Chris Quirk, Ye-Yi Wang, Nick Craswell, Irene Shaffer, Tianwei Chen, Sulaiman Vesal, Soundar Srinivasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03211)  

**Abstract**: In enterprise search, building high-quality datasets at scale remains a central challenge due to the difficulty of acquiring labeled data. To resolve this challenge, we propose an efficient approach to fine-tune small language models (SLMs) for accurate relevance labeling, enabling high-throughput, domain-specific labeling comparable or even better in quality to that of state-of-the-art large language models (LLMs). To overcome the lack of high-quality and accessible datasets in the enterprise domain, our method leverages on synthetic data generation. Specifically, we employ an LLM to synthesize realistic enterprise queries from a seed document, apply BM25 to retrieve hard negatives, and use a teacher LLM to assign relevance scores. The resulting dataset is then distilled into an SLM, producing a compact relevance labeler. We evaluate our approach on a high-quality benchmark consisting of 923 enterprise query-document pairs annotated by trained human annotators, and show that the distilled SLM achieves agreement with human judgments on par with or better than the teacher LLM. Furthermore, our fine-tuned labeler substantially improves throughput, achieving 17 times increase while also being 19 times more cost-effective. This approach enables scalable and cost-effective relevance labeling for enterprise-scale retrieval applications, supporting rapid offline evaluation and iteration in real-world settings. 

---
# UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward 

**Authors**: Yile Liu, Yixian Liu, Zongwei Li, Yufei Huang, Xinhua Feng, Zhichao Hu, Jinglu Hu, Jianfeng Yan, Fengzong Lian, Yuhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03205)  

**Abstract**: While Large Language Models (LLMs) have demonstrated significant potential in natural language processing , complex general-purpose reasoning requiring multi-step logic, planning, and verification remains a critical bottleneck. Although Reinforcement Learning with Verifiable Rewards (RLVR) has succeeded in specific domains , the field lacks large-scale, high-quality, and difficulty-calibrated data for general reasoning. To address this, we propose UltraLogic, a framework that decouples the logical core of a problem from its natural language expression through a Code-based Solving methodology to automate high-quality data production. The framework comprises hundreds of unique task types and an automated calibration pipeline across ten difficulty levels. Furthermore, to mitigate binary reward sparsity and the Non-negative Reward Trap, we introduce the Bipolar Float Reward (BFR) mechanism, utilizing graded penalties to effectively distinguish perfect responses from those with logical flaws. Our experiments demonstrate that task diversity is the primary driver for reasoning enhancement , and that BFR, combined with a difficulty matching strategy, significantly improves training efficiency, guiding models toward global logical optima. 

---
# Counterfactual Fairness with Graph Uncertainty 

**Authors**: Davi Valério, Chrysoula Zerva, Mariana Pinto, Ricardo Santos, André Carreiro  

**Link**: [PDF](https://arxiv.org/pdf/2601.03203)  

**Abstract**: Evaluating machine learning (ML) model bias is key to building trustworthy and robust ML systems. Counterfactual Fairness (CF) audits allow the measurement of bias of ML models with a causal framework, yet their conclusions rely on a single causal graph that is rarely known with certainty in real-world scenarios. We propose CF with Graph Uncertainty (CF-GU), a bias evaluation procedure that incorporates the uncertainty of specifying a causal graph into CF. CF-GU (i) bootstraps a Causal Discovery algorithm under domain knowledge constraints to produce a bag of plausible Directed Acyclic Graphs (DAGs), (ii) quantifies graph uncertainty with the normalized Shannon entropy, and (iii) provides confidence bounds on CF metrics. Experiments on synthetic data show how contrasting domain knowledge assumptions support or refute audits of CF, while experiments on real-world data (COMPAS and Adult datasets) pinpoint well-known biases with high confidence, even when supplied with minimal domain knowledge constraints. 

---
# Recursive querying of neural networks via weighted structures 

**Authors**: Martin Grohe, Christoph Standke, Juno Steegmans, Jan Van den Bussche  

**Link**: [PDF](https://arxiv.org/pdf/2601.03201)  

**Abstract**: Expressive querying of machine learning models - viewed as a form of intentional data - enables their verification and interpretation using declarative languages, thereby making learned representations of data more accessible. Motivated by the querying of feedforward neural networks, we investigate logics for weighted structures. In the absence of a bound on neural network depth, such logics must incorporate recursion; thereto we revisit the functional fixpoint mechanism proposed by Grädel and Gurevich. We adopt it in a Datalog-like syntax; we extend normal forms for fixpoint logics to weighted structures; and show an equivalent "loose" fixpoint mechanism that allows values of inductively defined weight functions to be overwritten. We propose a "scalar" restriction of functional fixpoint logic, of polynomial-time data complexity, and show it can express all PTIME model-agnostic queries over reduced networks with polynomially bounded weights. In contrast, we show that very simple model-agnostic queries are already NP-complete. Finally, we consider transformations of weighted structures by iterated transductions. 

---
# DIP: Dynamic In-Context Planner For Diffusion Language Models 

**Authors**: Yang Li, Han Meng, Chenan Wang, Haipeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.03199)  

**Abstract**: Diffusion language models (DLMs) have shown strong potential for general natural language tasks with in-context examples. However, due to the bidirectional attention mechanism, DLMs incur substantial computational cost as context length increases. This work addresses this issue with a key discovery: unlike the sequential generation in autoregressive language models (ARLMs), the diffusion generation paradigm in DLMs allows \textit{efficient dynamic adjustment of the context} during generation. Building on this insight, we propose \textbf{D}ynamic \textbf{I}n-Context \textbf{P}lanner (DIP), a context-optimization method that dynamically selects and inserts in-context examples during generation, rather than providing all examples in the prompt upfront. Results show DIP maintains generation quality while achieving up to 12.9$\times$ inference speedup over standard inference and 1.17$\times$ over KV cache-enhanced inference. 

---
# UniCorn: Towards Self-Improving Unified Multimodal Models through Self-Generated Supervision 

**Authors**: Ruiyan Han, Zhen Fang, XinYu Sun, Yuchen Ma, Ziheng Wang, Yu Zeng, Zehui Chen, Lin Chen, Wenxuan Huang, Wei-Jie Xu, Yi Cao, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.03193)  

**Abstract**: While Unified Multimodal Models (UMMs) have achieved remarkable success in cross-modal comprehension, a significant gap persists in their ability to leverage such internal knowledge for high-quality generation. We formalize this discrepancy as Conduction Aphasia, a phenomenon where models accurately interpret multimodal inputs but struggle to translate that understanding into faithful and controllable synthesis. To address this, we propose UniCorn, a simple yet elegant self-improvement framework that eliminates the need for external data or teacher supervision. By partitioning a single UMM into three collaborative roles: Proposer, Solver, and Judge, UniCorn generates high-quality interactions via self-play and employs cognitive pattern reconstruction to distill latent understanding into explicit generative signals. To validate the restoration of multimodal coherence, we introduce UniCycle, a cycle-consistency benchmark based on a Text to Image to Text reconstruction loop. Extensive experiments demonstrate that UniCorn achieves comprehensive and substantial improvements over the base model across six general image generation benchmarks. Notably, it achieves SOTA performance on TIIF(73.8), DPG(86.8), CompBench(88.5), and UniCycle while further delivering substantial gains of +5.0 on WISE and +6.5 on OneIG. These results highlight that our method significantly enhances T2I generation while maintaining robust comprehension, demonstrating the scalability of fully self-supervised refinement for unified multimodal intelligence. 

---
# AnatomiX, an Anatomy-Aware Grounded Multimodal Large Language Model for Chest X-Ray Interpretation 

**Authors**: Anees Ur Rehman Hashmi, Numan Saeed, Christoph Lippert  

**Link**: [PDF](https://arxiv.org/pdf/2601.03191)  

**Abstract**: Multimodal medical large language models have shown impressive progress in chest X-ray interpretation but continue to face challenges in spatial reasoning and anatomical understanding. Although existing grounding techniques improve overall performance, they often fail to establish a true anatomical correspondence, resulting in incorrect anatomical understanding in the medical domain. To address this gap, we introduce AnatomiX, a multitask multimodal large language model explicitly designed for anatomically grounded chest X-ray interpretation. Inspired by the radiological workflow, AnatomiX adopts a two stage approach: first, it identifies anatomical structures and extracts their features, and then leverages a large language model to perform diverse downstream tasks such as phrase grounding, report generation, visual question answering, and image understanding. Extensive experiments across multiple benchmarks demonstrate that AnatomiX achieves superior anatomical reasoning and delivers over 25% improvement in performance on anatomy grounding, phrase grounding, grounded diagnosis and grounded captioning tasks compared to existing approaches. Code and pretrained model are available at this https URL 

---
# Decentralized Autoregressive Generation 

**Authors**: Stepan Maschan, Haoxuan Qu, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03184)  

**Abstract**: We present a theoretical analysis of decentralization of autoregressive generation. We define the Decentralized Discrete Flow Matching objective, by expressing probability generating velocity as a linear combination of expert flows. We also conduct experiments demonstrat- ing the equivalence between decentralized and centralized training settings for multimodal language models across diverse set of benchmarks. Specifically, we compare two distinct paradigms: LLaVA and InternVL 2.5-1B, which uses a fixed CLIP vision encoder and per- forms full-parameter fine-tuning (ViT+MLP+LLM) during the instruction tuning stage. 

---
# Multi-Modal Data-Enhanced Foundation Models for Prediction and Control in Wireless Networks: A Survey 

**Authors**: Han Zhang, Mohammad Farzanullah, Mohammad Ghassemi, Akram Bin Sediq, Ali Afana, Melike Erol-Kantarci  

**Link**: [PDF](https://arxiv.org/pdf/2601.03181)  

**Abstract**: Foundation models (FMs) are recognized as a transformative breakthrough that has started to reshape the future of artificial intelligence (AI) across both academia and industry. The integration of FMs into wireless networks is expected to enable the development of general-purpose AI agents capable of handling diverse network management requests and highly complex wireless-related tasks involving multi-modal data. Inspired by these ideas, this work discusses the utilization of FMs, especially multi-modal FMs in wireless networks. We focus on two important types of tasks in wireless network management: prediction tasks and control tasks. In particular, we first discuss FMs-enabled multi-modal contextual information understanding in wireless networks. Then, we explain how FMs can be applied to prediction and control tasks, respectively. Following this, we introduce the development of wireless-specific FMs from two perspectives: available datasets for development and the methodologies used. Finally, we conclude with a discussion of the challenges and future directions for FM-enhanced wireless networks. 

---
# Rapid Augmentations for Time Series (RATS): A High-Performance Library for Time Series Augmentation 

**Authors**: Wadie Skaf, Felix Kern, Aryamaan Basu Roy, Tejas Pradhan, Roman Kalkreuth, Holger Hoos  

**Link**: [PDF](https://arxiv.org/pdf/2601.03159)  

**Abstract**: Time series augmentation is critical for training robust deep learning models, particularly in domains where labelled data is scarce and expensive to obtain. However, existing augmentation libraries for time series, mainly written in Python, suffer from performance bottlenecks, where running time grows exponentially as dataset sizes increase -- an aspect limiting their applicability in large-scale, production-grade systems. We introduce RATS (Rapid Augmentations for Time Series), a high-performance library for time series augmentation written in Rust with Python bindings (RATSpy). RATS implements multiple augmentation methods spanning basic transformations, frequency-domain operations and time warping techniques, all accessible through a unified pipeline interface with built-in parallelisation. Comprehensive benchmarking of RATSpy versus a commonly used library (tasug) on 143 datasets demonstrates that RATSpy achieves an average speedup of 74.5\% over tsaug (up to 94.8\% on large datasets), with up to 47.9\% less peak memory usage. 

---
# Prompt-Counterfactual Explanations for Generative AI System Behavior 

**Authors**: Sofie Goethals, Foster Provost, João Sedoc  

**Link**: [PDF](https://arxiv.org/pdf/2601.03156)  

**Abstract**: As generative AI systems become integrated into real-world applications, organizations increasingly need to be able to understand and interpret their behavior. In particular, decision-makers need to understand what causes generative AI systems to exhibit specific output characteristics. Within this general topic, this paper examines a key question: what is it about the input -the prompt- that causes an LLM-based generative AI system to produce output that exhibits specific characteristics, such as toxicity, negative sentiment, or political bias. To examine this question, we adapt a common technique from the Explainable AI literature: counterfactual explanations. We explain why traditional counterfactual explanations cannot be applied directly to generative AI systems, due to several differences in how generative AI systems function. We then propose a flexible framework that adapts counterfactual explanations to non-deterministic, generative AI systems in scenarios where downstream classifiers can reveal key characteristics of their outputs. Based on this framework, we introduce an algorithm for generating prompt-counterfactual explanations (PCEs). Finally, we demonstrate the production of counterfactual explanations for generative AI systems with three case studies, examining different output characteristics (viz., political leaning, toxicity, and sentiment). The case studies further show that PCEs can streamline prompt engineering to suppress undesirable output characteristics and can enhance red-teaming efforts to uncover additional prompts that elicit undesirable outputs. Ultimately, this work lays a foundation for prompt-focused interpretability in generative AI: a capability that will become indispensable as these models are entrusted with higher-stakes tasks and subject to emerging regulatory requirements for transparency and accountability. 

---
# Self-Verification is All You Need To Pass The Japanese Bar Examination 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03144)  

**Abstract**: Despite rapid advances in large language models (LLMs), achieving reliable performance on highly professional and structured examinations remains a significant challenge. The Japanese bar examination is a particularly demanding benchmark, requiring not only advanced legal reasoning but also strict adherence to complex answer formats that involve joint evaluation of multiple propositions. While recent studies have reported improvements by decomposing such questions into simpler true--false judgments, these approaches have not been systematically evaluated under the original exam format and scoring scheme, leaving open the question of whether they truly capture exam-level competence. In this paper, we present a self-verification model trained on a newly constructed dataset that faithfully replicates the authentic format and evaluation scale of the exam. Our model is able to exceed the official passing score when evaluated on the actual exam scale, marking the first demonstration, to our knowledge, of an LLM passing the Japanese bar examination without altering its original question structure or scoring rules. We further conduct extensive comparisons with alternative strategies, including multi-agent inference and decomposition-based supervision, and find that these methods fail to achieve comparable performance. Our results highlight the importance of format-faithful supervision and consistency verification, and suggest that carefully designed single-model approaches can outperform more complex systems in high-stakes professional reasoning tasks. Our dataset and codes are publicly available. 

---
# Limited Linguistic Diversity in Embodied AI Datasets 

**Authors**: Selma Wanna, Agnes Luhtaru, Jonathan Salfity, Ryan Barron, Juston Moore, Cynthia Matuszek, Mitch Pryor  

**Link**: [PDF](https://arxiv.org/pdf/2601.03136)  

**Abstract**: Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage. 

---
# Unified Thinker: A General Reasoning Modular Core for Image Generation 

**Authors**: Sashuai Zhou, Qiang Zhou, Jijin Hu, Hanqing Yang, Yue Cao, Junpeng Ma, Yinchao Ma, Jun Song, Tiezheng Ge, Cheng Yu, Bo Zheng, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.03127)  

**Abstract**: Despite impressive progress in high-fidelity image synthesis, generative models still struggle with logic-intensive instruction following, exposing a persistent reasoning--execution gap. Meanwhile, closed-source systems (e.g., Nano Banana) have demonstrated strong reasoning-driven image generation, highlighting a substantial gap to current open-source models. We argue that closing this gap requires not merely better visual generators, but executable reasoning: decomposing high-level intents into grounded, verifiable plans that directly steer the generative process. To this end, we propose Unified Thinker, a task-agnostic reasoning architecture for general image generation, designed as a unified planning core that can plug into diverse generators and workflows. Unified Thinker decouples a dedicated Thinker from the image Generator, enabling modular upgrades of reasoning without retraining the entire generative model. We further introduce a two-stage training paradigm: we first build a structured planning interface for the Thinker, then apply reinforcement learning to ground its policy in pixel-level feedback, encouraging plans that optimize visual correctness over textual plausibility. Extensive experiments on text-to-image generation and image editing show that Unified Thinker substantially improves image reasoning and generation quality. 

---
# LeafLife: An Explainable Deep Learning Framework with Robustness for Grape Leaf Disease Recognition 

**Authors**: B. M. Shahria Alam, Md. Nasim Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.03124)  

**Abstract**: Plant disease diagnosis is essential to farmers' management choices because plant diseases frequently lower crop yield and product quality. For harvests to flourish and agricultural productivity to boost, grape leaf disease detection is important. The plant disease dataset contains grape leaf diseases total of 9,032 images of four classes, among them three classes are leaf diseases, and the other one is healthy leaves. After rigorous pre-processing dataset was split (70% training, 20% validation, 10% testing), and two pre-trained models were deployed: InceptionV3 and Xception. Xception shows a promising result of 96.23% accuracy, which is remarkable than InceptionV3. Adversarial Training is used for robustness, along with more transparency. Grad-CAM is integrated to confirm the leaf disease. Finally deployed a web application using Streamlit with a heatmap visualization and prediction with confidence level for robust grape leaf disease classification. 

---
# ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation 

**Authors**: Peiran Li, Jan Fillies, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2601.03121)  

**Abstract**: Augmenting toxic language data in a controllable and class-specific manner is crucial for improving robustness in toxicity classification, yet remains challenging due to limited supervision and distributional skew. We propose ToxiGAN, a class-aware text augmentation framework that combines adversarial generation with semantic guidance from large language models (LLMs). To address common issues in GAN-based augmentation such as mode collapse and semantic drift, ToxiGAN introduces a two-step directional training strategy and leverages LLM-generated neutral texts as semantic ballast. Unlike prior work that treats LLMs as static generators, our approach dynamically selects neutral exemplars to provide balanced guidance. Toxic samples are explicitly optimized to diverge from these exemplars, reinforcing class-specific contrastive signals. Experiments on four hate speech benchmarks show that ToxiGAN achieves the strongest average performance in both macro-F1 and hate-F1, consistently outperforming traditional and LLM-based augmentation methods. Ablation and sensitivity analyses further confirm the benefits of semantic ballast and directional training in enhancing classifier robustness. 

---
# Transformers self-organize like newborn visual systems when trained in prenatal worlds 

**Authors**: Lalit Pandey, Samantha M. W. Wood, Justin N. Wood  

**Link**: [PDF](https://arxiv.org/pdf/2601.03117)  

**Abstract**: Do transformers learn like brains? A key challenge in addressing this question is that transformers and brains are trained on fundamentally different data. Brains are initially "trained" on prenatal sensory experiences (e.g., retinal waves), whereas transformers are typically trained on large datasets that are not biologically plausible. We reasoned that if transformers learn like brains, then they should develop the same structure as newborn brains when exposed to the same prenatal data. To test this prediction, we simulated prenatal visual input using a retinal wave generator. Then, using self-supervised temporal learning, we trained transformers to adapt to those retinal waves. During training, the transformers spontaneously developed the same structure as newborn visual systems: (1) early layers became sensitive to edges, (2) later layers became sensitive to shapes, and (3) the models developed larger receptive fields across layers. The organization of newborn visual systems emerges spontaneously when transformers adapt to a prenatal visual world. This developmental convergence suggests that brains and transformers learn in common ways and follow the same general fitting principles. 

---
# Who Laughs with Whom? Disentangling Influential Factors in Humor Preferences across User Clusters and LLMs 

**Authors**: Soichiro Murakami, Hidetaka Kamigaito, Hiroya Takamura, Manabu Okumura  

**Link**: [PDF](https://arxiv.org/pdf/2601.03103)  

**Abstract**: Humor preferences vary widely across individuals and cultures, complicating the evaluation of humor using large language models (LLMs). In this study, we model heterogeneity in humor preferences in Oogiri, a Japanese creative response game, by clustering users with voting logs and estimating cluster-specific weights over interpretable preference factors using Bradley-Terry-Luce models. We elicit preference judgments from LLMs by prompting them to select the funnier response and found that user clusters exhibit distinct preference patterns and that the LLM results can resemble those of particular clusters. Finally, we demonstrate that, by persona prompting, LLM preferences can be directed toward a specific cluster. The scripts for data collection and analysis will be released to support reproducibility. 

---
# Text-Guided Layer Fusion Mitigates Hallucination in Multimodal LLMs 

**Authors**: Chenchen Lin, Sanbao Su, Rachel Luo, Yuxiao Chen, Yan Wang, Marco Pavone, Fei Miao  

**Link**: [PDF](https://arxiv.org/pdf/2601.03100)  

**Abstract**: Multimodal large language models (MLLMs) typically rely on a single late-layer feature from a frozen vision encoder, leaving the encoder's rich hierarchy of visual cues under-utilized. MLLMs still suffer from visually ungrounded hallucinations, often relying on language priors rather than image evidence. While many prior mitigation strategies operate on the text side, they leave the visual representation unchanged and do not exploit the rich hierarchy of features encoded across vision layers. Existing multi-layer fusion methods partially address this limitation but remain static, applying the same layer mixture regardless of the query. In this work, we introduce TGIF (Text-Guided Inter-layer Fusion), a lightweight module that treats encoder layers as depth-wise "experts" and predicts a prompt-dependent fusion of visual features. TGIF follows the principle of direct external fusion, requires no vision-encoder updates, and adds minimal overhead. Integrated into LLaVA-1.5-7B, TGIF provides consistent improvements across hallucination, OCR, and VQA benchmarks, while preserving or improving performance on ScienceQA, GQA, and MMBench. These results suggest that query-conditioned, hierarchy-aware fusion is an effective way to strengthen visual grounding and reduce hallucination in modern MLLMs. 

---
# Grad-ELLM: Gradient-based Explanations for Decoder-only LLMs 

**Authors**: Xin Huang, Antoni B. Chan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03089)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet their black-box nature raises concerns about transparency and faithfulness. Input attribution methods aim to highlight each input token's contributions to the model's output, but existing approaches are typically model-agnostic, and do not focus on transformer-specific architectures, leading to limited faithfulness. To address this, we propose Grad-ELLM, a gradient-based attribution method for decoder-only transformer-based LLMs. By aggregating channel importance from gradients of the output logit with respect to attention layers and spatial importance from attention maps, Grad-ELLM generates heatmaps at each generation step without requiring architectural modifications. Additionally, we introduce two faithfulneses metrics $\pi$-Soft-NC and $\pi$-Soft-NS, which are modifications of Soft-NC/NS that provide fairer comparisons by controlling the amount of information kept when perturbing the text. We evaluate Grad-ELLM on sentiment classification, question answering, and open-generation tasks using different models. Experiment results show that Grad-ELLM consistently achieves superior faithfulness than other attribution methods. 

---
# Joint Encoding of KV-Cache Blocks for Scalable LLM Serving 

**Authors**: Joseph Kampeas, Emir Haleva  

**Link**: [PDF](https://arxiv.org/pdf/2601.03067)  

**Abstract**: Modern large language models (LLMs) drive interactive AI systems but are bottlenecked by the memory-heavy growth of key-value (KV) caches, which limits real-time throughput under concurrent loads. Existing KV-cache compression methods rely on rigid heuristics, disrupt tensor layouts, or require specialized compute, hindering scalability and deployment.
We propose joint encoding of KV-cache blocks, which fuses similar blocks across requests and input chunks into shared representations while preserving standard cache structure. This alleviates the KV-cache memory bottleneck, supporting high-concurrency serving without specialized hardware. Theoretically, we analyze the rate-distortion tradeoff of fused cache blocks under a Poisson process model. Empirically, our method achieves up to 4.38 $\times$ KV-cache compression with negligible accuracy loss across diverse LLMs and benchmarks, outperforming recent structured and adaptive compression baselines. In real LLM serving, joint encoding improves the token throughput by $\sim$40\% on a single-machine vLLM benchmark, demonstrating substantial gains in inference throughput. Code is available at this https URL  kv_joint_encoding. 

---
# Do LLMs Encode Functional Importance of Reasoning Tokens? 

**Authors**: Janvijay Singh, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2601.03066)  

**Abstract**: Large language models solve complex tasks by generating long reasoning chains, achieving higher accuracy at the cost of increased computational cost and reduced ability to isolate functionally relevant reasoning. Prior work on compact reasoning shortens such chains through probabilistic sampling, heuristics, or supervision from frontier models, but offers limited insight into whether models internally encode token-level functional importance for answer generation. We address this gap diagnostically and propose greedy pruning, a likelihood-preserving deletion procedure that iteratively removes reasoning tokens whose removal minimally degrades model likelihood under a specified objective, yielding length-controlled reasoning chains. We evaluate pruned reasoning in a distillation framework and show that students trained on pruned chains outperform a frontier-model-supervised compression baseline at matched reasoning lengths. Finally, our analysis reveals systematic pruning patterns and shows that attention scores can predict greedy pruning ranks, further suggesting that models encode a nontrivial functional importance structure over reasoning tokens. 

---
# IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation 

**Authors**: Yankai Jiang, Qiaoru Li, Binlu Xu, Haoran Sun, Chao Ding, Junting Dong, Yuxiang Cai, Xuhong Zhang, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.03054)  

**Abstract**: Recent research on medical MLLMs has gradually shifted its focus from image-level understanding to fine-grained, pixel-level comprehension. Although segmentation serves as the foundation for pixel-level understanding, existing approaches face two major challenges. First, they introduce implicit segmentation tokens and require simultaneous fine-tuning of both the MLLM and external pixel decoders, which increases the risk of catastrophic forgetting and limits generalization to out-of-domain scenarios. Second, most methods rely on single-pass reasoning and lack the capability to iteratively refine segmentation results, leading to suboptimal performance. To overcome these limitations, we propose a novel agentic MLLM, named IBISAgent, that reformulates segmentation as a vision-centric, multi-step decision-making process. IBISAgent enables MLLMs to generate interleaved reasoning and text-based click actions, invoke segmentation tools, and produce high-quality masks without architectural modifications. By iteratively performing multi-step visual reasoning on masked image features, IBISAgent naturally supports mask refinement and promotes the development of pixel-level visual reasoning capabilities. We further design a two-stage training framework consisting of cold-start supervised fine-tuning and agentic reinforcement learning with tailored, fine-grained rewards, enhancing the model's robustness in complex medical referring and reasoning segmentation tasks. Extensive experiments demonstrate that IBISAgent consistently outperforms both closed-source and open-source SOTA methods. All datasets, code, and trained models will be released publicly. 

---
# On the Intrinsic Limits of Transformer Image Embeddings in Non-Solvable Spatial Reasoning 

**Authors**: Siyi Lyu, Quan Liu, Feng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.03048)  

**Abstract**: Vision Transformers (ViTs) excel in semantic recognition but exhibit systematic failures in spatial reasoning tasks such as mental rotation. While often attributed to data scale, we propose that this limitation arises from the intrinsic circuit complexity of the architecture. We formalize spatial understanding as learning a Group Homomorphism: mapping image sequences to a latent space that preserves the algebraic structure of the underlying transformation group. We demonstrate that for non-solvable groups (e.g., the 3D rotation group $\mathrm{SO}(3)$), maintaining such a structure-preserving embedding is computationally lower-bounded by the Word Problem, which is $\mathsf{NC^1}$-complete. In contrast, we prove that constant-depth ViTs with polynomial precision are strictly bounded by $\mathsf{TC^0}$. Under the conjecture $\mathsf{TC^0} \subsetneq \mathsf{NC^1}$, we establish a complexity boundary: constant-depth ViTs fundamentally lack the logical depth to efficiently capture non-solvable spatial structures. We validate this complexity gap via latent-space probing, demonstrating that ViT representations suffer a structural collapse on non-solvable tasks as compositional depth increases. 

---
# Motion Blur Robust Wheat Pest Damage Detection with Dynamic Fuzzy Feature Fusion 

**Authors**: Han Zhang, Yanwei Wang, Fang Li, Hongjun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03046)  

**Abstract**: Motion blur caused by camera shake produces ghosting artifacts that substantially degrade edge side object detection. Existing approaches either suppress blur as noise and lose discriminative structure, or apply full image restoration that increases latency and limits deployment on resource constrained devices. We propose DFRCP, a Dynamic Fuzzy Robust Convolutional Pyramid, as a plug in upgrade to YOLOv11 for blur robust detection. DFRCP enhances the YOLOv11 feature pyramid by combining large scale and medium scale features while preserving native representations, and by introducing Dynamic Robust Switch units that adaptively inject fuzzy features to strengthen global perception under jitter. Fuzzy features are synthesized by rotating and nonlinearly interpolating multiscale features, then merged through a transparency convolution that learns a content adaptive trade off between original and fuzzy cues. We further develop a CUDA parallel rotation and interpolation kernel that avoids boundary overflow and delivers more than 400 times speedup, making the design practical for edge deployment. We train with paired supervision on a private wheat pest damage dataset of about 3,500 images, augmented threefold using two blur regimes, uniform image wide motion blur and bounding box confined rotational blur. On blurred test sets, YOLOv11 with DFRCP achieves about 10.4 percent higher accuracy than the YOLOv11 baseline with only a modest training time overhead, reducing the need for manual filtering after data collection. 

---
# Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage 

**Authors**: Junhao Hu, Fangze Li, Mingtao Xu, Feifan Meng, Shiju Zhao, Tiancheng Hu, Ting Peng, Anmin Liu, Wenrui Huang, Chenxu Liu, Ziyue Hua, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.03043)  

**Abstract**: Large language models (LLMs) demonstrate strong capabilities across a wide range of complex tasks and are increasingly deployed at scale, placing significant demands on inference efficiency. Prior work typically decomposes inference into prefill and decode stages, with the decode stage dominating total latency. To reduce time and memory complexity in the decode stage, a line of work introduces sparse-attention algorithms. In this paper, we show, both empirically and theoretically, that sparse attention can paradoxically increase end-to-end complexity: information loss often induces significantly longer sequences, a phenomenon we term ``Less is Less'' (Lil). To mitigate the Lil problem, we propose an early-stopping algorithm that detects the threshold where information loss exceeds information gain during sparse decoding. Our early-stopping algorithm reduces token consumption by up to 90% with a marginal accuracy degradation of less than 2% across reasoning-intensive benchmarks. 

---
# PiDR: Physics-Informed Inertial Dead Reckoning for Autonomous Platforms 

**Authors**: Arup Kumar Sahoo, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2601.03040)  

**Abstract**: A fundamental requirement for full autonomy is the ability to sustain accurate navigation in the absence of external data, such as GNSS signals or visual information. In these challenging environments, the platform must rely exclusively on inertial sensors, leading to pure inertial navigation. However, the inherent noise and other error terms of the inertial sensors in such real-world scenarios will cause the navigation solution to drift over time. Although conventional deep-learning models have emerged as a possible approach to inertial navigation, they are inherently black-box in nature. Furthermore, they struggle to learn effectively with limited supervised sensor data and often fail to preserve physical principles. To address these limitations, we propose PiDR, a physics-informed inertial dead-reckoning framework for autonomous platforms in situations of pure inertial navigation. PiDR offers transparency by explicitly integrating inertial navigation principles into the network training process through the physics-informed residual component. PiDR plays a crucial role in mitigating abrupt trajectory deviations even under limited or sparse supervision. We evaluated PiDR on real-world datasets collected by a mobile robot and an autonomous underwater vehicle. We obtained more than 29% positioning improvement in both datasets, demonstrating the ability of PiDR to generalize different platforms operating in various environments and dynamics. Thus, PiDR offers a robust, lightweight, yet effective architecture and can be deployed on resource-constrained platforms, enabling real-time pure inertial navigation in adverse scenarios. 

---
# Validating Generalist Robots with Situation Calculus and STL Falsification 

**Authors**: Changwen Li, Rongjie Yan, Chih-Hong Cheng, Jian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.03038)  

**Abstract**: Generalist robots are becoming a reality, capable of interpreting natural language instructions and executing diverse operations. However, their validation remains challenging because each task induces its own operational context and correctness specification, exceeding the assumptions of traditional validation methods. We propose a two-layer validation framework that combines abstract reasoning with concrete system falsification. At the abstract layer, situation calculus models the world and derives weakest preconditions, enabling constraint-aware combinatorial testing to systematically generate diverse, semantically valid world-task configurations with controllable coverage strength. At the concrete layer, these configurations are instantiated for simulation-based falsification with STL monitoring. Experiments on tabletop manipulation tasks show that our framework effectively uncovers failure cases in the NVIDIA GR00T controller, demonstrating its promise for validating general-purpose robot autonomy. 

---
# Causal Manifold Fairness: Enforcing Geometric Invariance in Representation Learning 

**Authors**: Vidhi Rathore  

**Link**: [PDF](https://arxiv.org/pdf/2601.03032)  

**Abstract**: Fairness in machine learning is increasingly critical, yet standard approaches often treat data as static points in a high-dimensional space, ignoring the underlying generative structure. We posit that sensitive attributes (e.g., race, gender) do not merely shift data distributions but causally warp the geometry of the data manifold itself. To address this, we introduce Causal Manifold Fairness (CMF), a novel framework that bridges causal inference and geometric deep learning. CMF learns a latent representation where the local Riemannian geometry, defined by the metric tensor and curvature, remains invariant under counterfactual interventions on sensitive attributes. By enforcing constraints on the Jacobian and Hessian of the decoder, CMF ensures that the rules of the latent space (distances and shapes) are preserved across demographic groups. We validate CMF on synthetic Structural Causal Models (SCMs), demonstrating that it effectively disentangles sensitive geometric warping while preserving task utility, offering a rigorous quantification of the fairness-utility trade-off via geometric metrics. 

---
# Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis 

**Authors**: Choonghan Kim, Hyunmin Hwang, Hangeol Chang, Jaemin Kim, Jinse Park, Jae-Sung Lim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.03018)  

**Abstract**: While Large Language Models (LLMs) have shown strong performance on clinical text understanding, they struggle with longitudinal prediction tasks such as dementia prognosis, which require reasoning over complex, non-monotonic symptom trajectories across multiple visits. Standard supervised training lacks explicit annotations for symptom evolution, while direct Reinforcement Learning (RL) is hindered by sparse binary rewards. To address this challenge, we introduce Dementia-R1, an RL-based framework for longitudinal dementia prognosis from unstructured clinical notes. Our approach adopts a Cold-Start RL strategy that pre-trains the model to predict verifiable clinical indices extracted from patient histories, enhancing the capability to reason about disease progression before determining the final clinical status. Extensive experiments demonstrate that Dementia-R1 achieves an F1 score of 77.03% on real-world unstructured clinical datasets. Notably, on the ADNI benchmark, our 7B model rivals GPT-4o, effectively capturing fluctuating cognitive trajectories. Code is available at this https URL 

---
# In-Context Reinforcement Learning through Bayesian Fusion of Context and Value Prior 

**Authors**: Anaïs Berkes, Vincent Taboga, Donna Vakalis, David Rolnick, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2601.03015)  

**Abstract**: In-context reinforcement learning (ICRL) promises fast adaptation to unseen environments without parameter updates, but current methods either cannot improve beyond the training distribution or require near-optimal data, limiting practical adoption. We introduce SPICE, a Bayesian ICRL method that learns a prior over Q-values via deep ensemble and updates this prior at test-time using in-context information through Bayesian updates. To recover from poor priors resulting from training on sub-optimal data, our online inference follows an Upper-Confidence Bound rule that favours exploration and adaptation. We prove that SPICE achieves regret-optimal behaviour in both stochastic bandits and finite-horizon MDPs, even when pretrained only on suboptimal trajectories. We validate these findings empirically across bandit and control benchmarks. SPICE achieves near-optimal decisions on unseen tasks, substantially reduces regret compared to prior ICRL and meta-RL approaches while rapidly adapting to unseen tasks and remaining robust under distribution shift. 

---
# SentGraph: Hierarchical Sentence Graph for Multi-hop Retrieval-Augmented Question Answering 

**Authors**: Junli Liang, Pengfei Zhou, Wangqiu Zhou, Wenjie Qing, Qi Zhao, Ziwen Wang, Qi Song, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.03014)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) effectively supports single-hop question answering with large language models but faces significant limitations in multi-hop question answering tasks, which require combining evidence from multiple documents. Existing chunk-based retrieval often provides irrelevant and logically incoherent context, leading to incomplete evidence chains and incorrect reasoning during answer generation. To address these challenges, we propose SentGraph, a sentence-level graph-based RAG framework that explicitly models fine-grained logical relationships between sentences for multi-hop question answering. Specifically, we construct a hierarchical sentence graph offline by first adapting Rhetorical Structure Theory to distinguish nucleus and satellite sentences, and then organizing them into topic-level subgraphs with cross-document entity bridges. During online retrieval, SentGraph performs graph-guided evidence selection and path expansion to retrieve fine-grained sentence-level evidence. Extensive experiments on four multi-hop question answering benchmarks demonstrate the effectiveness of SentGraph, validating the importance of explicitly modeling sentence-level logical dependencies for multi-hop reasoning. 

---
# JPU: Bridging Jailbreak Defense and Unlearning via On-Policy Path Rectification 

**Authors**: Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Zhaoye Li, Bin Ji, Baosheng Wang, Jie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.03005)  

**Abstract**: Despite extensive safety alignment, Large Language Models (LLMs) often fail against jailbreak attacks. While machine unlearning has emerged as a promising defense by erasing specific harmful parameters, current methods remain vulnerable to diverse jailbreaks. We first conduct an empirical study and discover that this failure mechanism is caused by jailbreaks primarily activating non-erased parameters in the intermediate layers. Further, by probing the underlying mechanism through which these circumvented parameters reassemble into the prohibited output, we verify the persistent existence of dynamic $\textbf{jailbreak paths}$ and show that the inability to rectify them constitutes the fundamental gap in existing unlearning defenses. To bridge this gap, we propose $\textbf{J}$ailbreak $\textbf{P}$ath $\textbf{U}$nlearning (JPU), which is the first to rectify dynamic jailbreak paths towards safety anchors by dynamically mining on-policy adversarial samples to expose vulnerabilities and identify jailbreak paths. Extensive experiments demonstrate that JPU significantly enhances jailbreak resistance against dynamic attacks while preserving the model's utility. 

---
# Learning to Act Robustly with View-Invariant Latent Actions 

**Authors**: Youngjoon Jeong, Junha Chun, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.02994)  

**Abstract**: Vision-based robotic policies often struggle with even minor viewpoint changes, underscoring the need for view-invariant visual representations. This challenge becomes more pronounced in real-world settings, where viewpoint variability is unavoidable and can significantly disrupt policy performance. Existing methods typically learn invariance from multi-view observations at the scene level, but such approaches rely on visual appearance and fail to incorporate the physical dynamics essential for robust generalization. We propose View-Invariant Latent Action (VILA), which models a latent action capturing transition patterns across trajectories to learn view-invariant representations grounded in physical dynamics. VILA aligns these latent actions across viewpoints using an action-guided objective based on ground-truth action sequences. Experiments in both simulation and the real world show that VILA-based policies generalize effectively to unseen viewpoints and transfer well to new tasks, establishing VILA as a strong pretraining framework that improves robustness and downstream learning performance. 

---
# Towards Faithful Reasoning in Comics for Small MLLMs 

**Authors**: Chengcheng Feng, Haojie Yin, Yucheng Jin, Kaizhu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02991)  

**Abstract**: Comic-based visual question answering (CVQA) poses distinct challenges to multimodal large language models (MLLMs) due to its reliance on symbolic abstraction, narrative logic, and humor, which differ from conventional VQA tasks. Although Chain-of-Thought (CoT) prompting is widely used to enhance MLLM reasoning, surprisingly, its direct application to CVQA often degrades performance, especially in small-scale models. Our theoretical and empirical analyses reveal that standard CoT in CVQA suffers from state entanglement, spurious transitions, and exploration inefficiency, with small models particularly vulnerable in resource-constrained settings. To address these issues, we propose a novel comic reasoning framework, designed to produce more faithful and transferable reasoning chains in small MLLMs. Specifically, our framework combines modular CoT generation with GRPO-based reinforcement fine-tuning and a novel structured reward. Beyond comic VQA, we further evaluate our approach on a broader class of humor-centric and abstract visual reasoning tasks, including meme understanding and editorial cartoon interpretation. Across five challenging benchmarks, our 3B model outperforms state-of-the-art methods, and plug-in experiments yield an additional average improvement of $\mathbf{12.1\%}$ across different MLLMs. 

---
# ULS+: Data-driven Model Adaptation Enhances Lesion Segmentation 

**Authors**: Rianne Weber, Niels Rocholl, Max de Grauw, Mathias Prokop, Ewoud Smit, Alessa Hering  

**Link**: [PDF](https://arxiv.org/pdf/2601.02988)  

**Abstract**: In this study, we present ULS+, an enhanced version of the Universal Lesion Segmentation (ULS) model. The original ULS model segments lesions across the whole body in CT scans given volumes of interest (VOIs) centered around a click-point. Since its release, several new public datasets have become available that can further improve model performance. ULS+ incorporates these additional datasets and uses smaller input image sizes, resulting in higher accuracy and faster inference.
We compared ULS and ULS+ using the Dice score and robustness to click-point location on the ULS23 Challenge test data and a subset of the Longitudinal-CT dataset. In all comparisons, ULS+ significantly outperformed ULS. Additionally, ULS+ ranks first on the ULS23 Challenge test-phase leaderboard. By maintaining a cycle of data-driven updates and clinical validation, ULS+ establishes a foundation for robust and clinically relevant lesion segmentation models. 

---
# LAMS-Edit: Latent and Attention Mixing with Schedulers for Improved Content Preservation in Diffusion-Based Image and Style Editing 

**Authors**: Wingwa Fu, Takayuki Okatani  

**Link**: [PDF](https://arxiv.org/pdf/2601.02987)  

**Abstract**: Text-to-Image editing using diffusion models faces challenges in balancing content preservation with edit application and handling real-image editing. To address these, we propose LAMS-Edit, leveraging intermediate states from the inversion process--an essential step in real-image editing--during edited image generation. Specifically, latent representations and attention maps from both processes are combined at each step using weighted interpolation, controlled by a scheduler. This technique, Latent and Attention Mixing with Schedulers (LAMS), integrates with Prompt-to-Prompt (P2P) to form LAMS-Edit--an extensible framework that supports precise editing with region masks and enables style transfer via LoRA. Extensive experiments demonstrate that LAMS-Edit effectively balances content preservation and edit application. 

---
# Interpretable All-Type Audio Deepfake Detection with Audio LLMs via Frequency-Time Reinforcement Learning 

**Authors**: Yuankun Xie, Xiaoxuan Guo, Jiayi Zhou, Tao Wang, Jian Liu, Ruibo Fu, Xiaopeng Wang, Haonan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.02983)  

**Abstract**: Recent advances in audio large language models (ALLMs) have made high-quality synthetic audio widely accessible, increasing the risk of malicious audio deepfakes across speech, environmental sounds, singing voice, and music. Real-world audio deepfake detection (ADD) therefore requires all-type detectors that generalize across heterogeneous audio and provide interpretable decisions. Given the strong multi-task generalization ability of ALLMs, we first investigate their performance on all-type ADD under both supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). However, SFT using only binary real/fake labels tends to reduce the model to a black-box classifier, sacrificing interpretability. Meanwhile, vanilla RFT under sparse supervision is prone to reward hacking and can produce hallucinated, ungrounded rationales. To address this, we propose an automatic annotation and polishing pipeline that constructs Frequency-Time structured chain-of-thought (CoT) rationales, producing ~340K cold-start demonstrations. Building on CoT data, we propose Frequency Time-Group Relative Policy Optimization (FT-GRPO), a two-stage training paradigm that cold-starts ALLMs with SFT and then applies GRPO under rule-based frequency-time constraints. Experiments demonstrate that FT-GRPO achieves state-of-the-art performance on all-type ADD while producing interpretable, FT-grounded rationales. The data and code are available online. 

---
# Mechanistic Knobs in LLMs: Retrieving and Steering High-Order Semantic Features via Sparse Autoencoders 

**Authors**: Ruikang Zhang, Shuo Wang, Qi Su  

**Link**: [PDF](https://arxiv.org/pdf/2601.02978)  

**Abstract**: Recent work in Mechanistic Interpretability (MI) has enabled the identification and intervention of internal features in Large Language Models (LLMs). However, a persistent challenge lies in linking such internal features to the reliable control of complex, behavior-level semantic attributes in language generation. In this paper, we propose a Sparse Autoencoder-based framework for retrieving and steering semantically interpretable internal features associated with high-level linguistic behaviors. Our method employs a contrastive feature retrieval pipeline based on controlled semantic oppositions, combing statistical activation analysis and generation-based validation to distill monosemantic functional features from sparse activation spaces. Using the Big Five personality traits as a case study, we demonstrate that our method enables precise, bidirectional steering of model behavior while maintaining superior stability and performance compared to existing activation steering methods like Contrastive Activation Addition (CAA). We further identify an empirical effect, which we term Functional Faithfulness, whereby intervening on a specific internal feature induces coherent and predictable shifts across multiple linguistic dimensions aligned with the target semantic attribute. Our findings suggest that LLMs internalize deeply integrated representations of high-order concepts, and provide a novel, robust mechanistic path for the regulation of complex AI behaviors. 

---
# Correct, Concise and Complete: Multi-stage Training For Adaptive Reasoning 

**Authors**: Nathanaël Carraz Rakotonirina, Ren Pang, Neha Anna John, Michael Bohlke-Schneider, Momchil Hardalov  

**Link**: [PDF](https://arxiv.org/pdf/2601.02972)  

**Abstract**: The reasoning capabilities of large language models (LLMs) have improved substantially through increased test-time computation, typically in the form of intermediate tokens known as chain-of-thought (CoT). However, CoT often becomes unnecessarily long, increasing computation cost without actual accuracy gains or sometimes even degrading performance, a phenomenon known as ``overthinking''. We propose a multi-stage efficient reasoning method that combines supervised fine-tuning -- via rejection sampling or reasoning trace reformatting -- with reinforcement learning using an adaptive length penalty. We introduce a lightweight reward function that penalizes tokens generated after the first correct answer but encouraging self-verification only when beneficial. We conduct a holistic evaluation across seven diverse reasoning tasks, analyzing the accuracy--response length trade-off. Our approach reduces response length by an average of 28\% for 8B models and 40\% for 32B models, while incurring only minor performance drops of 1.6 and 2.5 points, respectively. Despite its conceptual simplicity, it achieves a superior trade-off compared to more complex state-of-the-art efficient reasoning methods, scoring 76.6, in terms of the area under the Overthinking-Adjusted Accuracy curve ($\text{AUC}_{\text{OAA}}$) -- 5 points above the base model and 2.5 points above the second-best approach. 

---
# MoE Adapter for Large Audio Language Models: Sparsity, Disentanglement, and Gradient-Conflict-Free 

**Authors**: Yishu Lei, Shuwei He, Jing Hu, Dan Zhang, Xianlong Luo, Danxiang Zhu, Shikun Feng, Rui Liu, Jingzhou He, Yu Sun, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02967)  

**Abstract**: Extending the input modality of Large Language Models~(LLMs) to the audio domain is essential for achieving comprehensive multimodal perception. However, it is well-known that acoustic information is intrinsically \textit{heterogeneous}, entangling attributes such as speech, music, and environmental context. Existing research is limited to a dense, parameter-shared adapter to model these diverse patterns, which induces \textit{gradient conflict} during optimization, as parameter updates required for distinct attributes contradict each other. To address this limitation, we introduce the \textit{\textbf{MoE-Adapter}}, a sparse Mixture-of-Experts~(MoE) architecture designed to decouple acoustic information. Specifically, it employs a dynamic gating mechanism that routes audio tokens to specialized experts capturing complementary feature subspaces while retaining shared experts for global context, thereby mitigating gradient conflicts and enabling fine-grained feature learning. Comprehensive experiments show that the MoE-Adapter achieves superior performance on both audio semantic and paralinguistic tasks, consistently outperforming dense linear baselines with comparable computational costs. Furthermore, we will release the related code and models to facilitate future research. 

---
# The World is Not Mono: Enabling Spatial Understanding in Large Audio-Language Models 

**Authors**: Yuhuan You, Lai Wei, Xihong Wu, Tianshu Qu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02954)  

**Abstract**: Existing large audio-language models perceive the world as "mono" -- a single stream of audio that ignores the critical spatial dimension ("where") required for universal acoustic scene analysis. To bridge this gap, we first introduce a hierarchical framework for Auditory Scene Analysis (ASA). Guided by this framework, we introduce a system that enables models like Qwen2-Audio to understand and reason about the complex acoustic world. Our framework achieves this through three core contributions: First, we build a large-scale, synthesized binaural audio dataset to provide the rich spatial cues. Second, we design a hybrid feature projector, which leverages parallel semantic and spatial encoders to extract decoupled representations. These distinct streams are integrated via a dense fusion mechanism, ensuring the model receives a holistic view of the acoustic scene. Finally, we employ a progressive training curriculum, advancing from supervised fine-tuning (SFT) to reinforcement learning via Group Relative Policy Optimization (GRPO), to explicitly evolve the model's capabilities towards reasoning. On our comprehensive benchmark, the model demonstrates comparatively strong capability for spatial understanding. By enabling this spatial perception, our work provides a clear pathway for leveraging the powerful reasoning abilities of large models towards holistic acoustic scene analysis, advancing from "mono" semantic recognition to spatial intelligence. 

---
# SastBench: A Benchmark for Testing Agentic SAST Triage 

**Authors**: Jake Feiglin, Guy Dar  

**Link**: [PDF](https://arxiv.org/pdf/2601.02941)  

**Abstract**: SAST (Static Application Security Testing) tools are among the most widely used techniques in defensive cybersecurity, employed by commercial and non-commercial organizations to identify potential vulnerabilities in software. Despite their great utility, they generate numerous false positives, requiring costly manual filtering (aka triage). While LLM-powered agents show promise for automating cybersecurity tasks, existing benchmarks fail to emulate real-world SAST finding distributions. We introduce SastBench, a benchmark for evaluating SAST triage agents that combines real CVEs as true positives with filtered SAST tool findings as approximate false positives. SastBench features an agent-agnostic design. We evaluate different agents on the benchmark and present a comparative analysis of their performance, provide a detailed analysis of the dataset, and discuss the implications for future development. 

---
# PrismVAU: Prompt-Refined Inference System for Multimodal Video Anomaly Understanding 

**Authors**: Iñaki Erregue, Kamal Nasrollahi, Sergio Escalera  

**Link**: [PDF](https://arxiv.org/pdf/2601.02927)  

**Abstract**: Video Anomaly Understanding (VAU) extends traditional Video Anomaly Detection (VAD) by not only localizing anomalies but also describing and reasoning about their context. Existing VAU approaches often rely on fine-tuned multimodal large language models (MLLMs) or external modules such as video captioners, which introduce costly annotations, complex training pipelines, and high inference overhead. In this work, we introduce PrismVAU, a lightweight yet effective system for real-time VAU that leverages a single off-the-shelf MLLM for anomaly scoring, explanation, and prompt optimization. PrismVAU operates in two complementary stages: (1) a coarse anomaly scoring module that computes frame-level anomaly scores via similarity to textual anchors, and (2) an MLLM-based refinement module that contextualizes anomalies through system and user prompts. Both textual anchors and prompts are optimized with a weakly supervised Automatic Prompt Engineering (APE) framework. Extensive experiments on standard VAD benchmarks demonstrate that PrismVAU delivers competitive detection performance and interpretable anomaly explanations -- without relying on instruction tuning, frame-level annotations, and external modules or dense processing -- making it an efficient and practical solution for real-world applications. 

---
# DCG ReID: Disentangling Collaboration and Guidance Fusion Representations for Multi-modal Vehicle Re-Identification 

**Authors**: Aihua Zheng, Ya Gao, Shihao Li, Chenglong Li, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02924)  

**Abstract**: Multi-modal vehicle Re-Identification (ReID) aims to leverage complementary information from RGB, Near Infrared (NIR), and Thermal Infrared (TIR) modalities to retrieve the same vehicle. The challenges of multi-modal vehicle ReID arise from the uncertainty of modality quality distribution induced by inherent discrepancies across modalities, resulting in distinct conflicting fusion requirements for data with balanced and unbalanced quality distributions. Existing methods handle all multi-modal data within a single fusion model, overlooking the different needs of the two data types and making it difficult to decouple the conflict between intra-class consistency and inter-modal heterogeneity. To this end, we propose Disentangle Collaboration and Guidance Fusion Representations for Multi-modal Vehicle ReID (DCG-ReID). Specifically, to disentangle heterogeneous quality-distributed modal data without mutual interference, we first design the Dynamic Confidence-based Disentangling Weighting (DCDW) mechanism: dynamically reweighting three-modal contributions via interaction-derived modal confidence to build a disentangled fusion framework. Building on DCDW, we develop two scenario-specific fusion strategies: (1) for balanced quality distributions, Collaboration Fusion Module (CFM) mines pairwise consensus features to capture shared discriminative information and boost intra-class consistency; (2) for unbalanced distributions, Guidance Fusion Module (GFM) implements differential amplification of modal discriminative disparities to reinforce dominant modality advantages, guide auxiliary modalities to mine complementary discriminative info, and mitigate inter-modal divergence to boost multi-modal joint decision performance. Extensive experiments on three multi-modal ReID benchmarks (WMVeID863, MSVR310, RGBNT100) validate the effectiveness of our method. Code will be released upon acceptance. 

---
# RAL2M: Retrieval Augmented Learning-To-Match Against Hallucination in Compliance-Guaranteed Service Systems 

**Authors**: Mengze Hong, Di Jiang, Jiangtao Wen, Zhiyang Su, Yawen Li, Yanjie Sun, Guan Wang, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02917)  

**Abstract**: Hallucination is a major concern in LLM-driven service systems, necessitating explicit knowledge grounding for compliance-guaranteed responses. In this paper, we introduce Retrieval-Augmented Learning-to-Match (RAL2M), a novel framework that eliminates generation hallucination by repositioning LLMs as query-response matching judges within a retrieval-based system, providing a robust alternative to purely generative approaches. To further mitigate judgment hallucination, we propose a query-adaptive latent ensemble strategy that explicitly models heterogeneous model competence and interdependencies among LLMs, deriving a calibrated consensus decision. Extensive experiments on large-scale benchmarks demonstrate that the proposed method effectively leverages the "wisdom of the crowd" and significantly outperforms strong baselines. Finally, we discuss best practices and promising directions for further exploiting latent representations in future work. 

---
# TA-Prompting: Enhancing Video Large Language Models for Dense Video Captioning via Temporal Anchors 

**Authors**: Wei-Yuan Cheng, Kai-Po Chang, Chi-Pin Huang, Fu-En Yang, Yu-Chiang Frank Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02908)  

**Abstract**: Dense video captioning aims to interpret and describe all temporally localized events throughout an input video. Recent state-of-the-art methods leverage large language models (LLMs) to provide detailed moment descriptions for video data. However, existing VideoLLMs remain challenging in identifying precise event boundaries in untrimmed videos, causing the generated captions to be not properly grounded. In this paper, we propose TA-Prompting, which enhances VideoLLMs via Temporal Anchors that learn to precisely localize events and prompt the VideoLLMs to perform temporal-aware video event understanding. During inference, in order to properly determine the output caption sequence from an arbitrary number of events presented within a video, we introduce an event coherent sampling strategy to select event captions with sufficient coherence across temporal events and cross-modal similarity with the given video. Through extensive experiments on benchmark datasets, we show that our TA-Prompting is favorable against state-of-the-art VideoLLMs, yielding superior performance on dense video captioning and temporal understanding tasks including moment retrieval and temporalQA. 

---
# LOST-3DSG: Lightweight Open-Vocabulary 3D Scene Graphs with Semantic Tracking in Dynamic Environments 

**Authors**: Sara Micol Ferraina, Michele Brienza, Francesco Argenziano, Emanuele Musumeci, Vincenzo Suriani, Domenico D. Bloisi, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02905)  

**Abstract**: Tracking objects that move within dynamic environments is a core challenge in robotics. Recent research has advanced this topic significantly; however, many existing approaches remain inefficient due to their reliance on heavy foundation models. To address this limitation, we propose LOST-3DSG, a lightweight open-vocabulary 3D scene graph designed to track dynamic objects in real-world environments. Our method adopts a semantic approach to entity tracking based on word2vec and sentence embeddings, enabling an open-vocabulary representation while avoiding the necessity of storing dense CLIP visual features. As a result, LOST-3DSG achieves superior performance compared to approaches that rely on high-dimensional visual embeddings. We evaluate our method through qualitative and quantitative experiments conducted in a real 3D environment using a TIAGo robot. The results demonstrate the effectiveness and efficiency of LOST-3DSG in dynamic object tracking. Code and supplementary material are publicly available on the project website at this https URL. 

---
# LongBench Pro: A More Realistic and Comprehensive Bilingual Long-Context Evaluation Benchmark 

**Authors**: Ziyang Chen, Xing Wu, Junlong Jia, Chaochen Gao, Qi Fu, Debing Zhang, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02872)  

**Abstract**: The rapid expansion of context length in large language models (LLMs) has outpaced existing evaluation benchmarks. Current long-context benchmarks often trade off scalability and realism: synthetic tasks underrepresent real-world complexity, while fully manual annotation is costly to scale to extreme lengths and diverse scenarios. We present LongBench Pro, a more realistic and comprehensive bilingual benchmark of 1,500 naturally occurring long-context samples in English and Chinese spanning 11 primary tasks and 25 secondary tasks, with input lengths from 8k to 256k tokens. LongBench Pro supports fine-grained analysis with task-specific metrics and a multi-dimensional taxonomy of context requirement (full vs. partial dependency), length (six levels), and difficulty (four levels calibrated by model performance). To balance quality with scalability, we propose a Human-Model Collaborative Construction pipeline: frontier LLMs draft challenging questions and reference answers, along with design rationales and solution processes, to reduce the cost of expert verification. Experts then rigorously validate correctness and refine problematic cases. Evaluating 46 widely used long-context LLMs on LongBench Pro yields three findings: (1) long-context optimization contributes more to long-context comprehension than parameter scaling; (2) effective context length is typically shorter than the claimed context length, with pronounced cross-lingual misalignment; and (3) the "thinking" paradigm helps primarily models trained with native reasoning, while mixed-thinking designs offer a promising Pareto trade-off. In summary, LongBench Pro provides a robust testbed for advancing long-context understanding. 

---
# TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents 

**Authors**: Kai Li, Xuanqing Yu, Ziyi Ni, Yi Zeng, Yao Xu, Zheqing Zhang, Xin Li, Jitao Sang, Xiaogang Duan, Xuelei Wang, Chengbao Liu, Jie Tan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02845)  

**Abstract**: Long-horizon conversational agents have to manage ever-growing interaction histories that quickly exceed the finite context windows of large language models (LLMs). Existing memory frameworks provide limited support for temporally structured information across hierarchical levels, often leading to fragmented memories and unstable long-horizon personalization. We present TiMem, a temporal--hierarchical memory framework that organizes conversations through a Temporal Memory Tree (TMT), enabling systematic memory consolidation from raw conversational observations to progressively abstracted persona representations. TiMem is characterized by three core properties: (1) temporal--hierarchical organization through TMT; (2) semantic-guided consolidation that enables memory integration across hierarchical levels without fine-tuning; and (3) complexity-aware memory recall that balances precision and efficiency across queries of varying complexity. Under a consistent evaluation setup, TiMem achieves state-of-the-art accuracy on both benchmarks, reaching 75.30% on LoCoMo and 76.88% on LongMemEval-S. It outperforms all evaluated baselines while reducing the recalled memory length by 52.20% on LoCoMo. Manifold analysis indicates clear persona separation on LoCoMo and reduced dispersion on LongMemEval-S. Overall, TiMem treats temporal continuity as a first-class organizing principle for long-horizon memory in conversational agents. 

---
# Breaking Self-Attention Failure: Rethinking Query Initialization for Infrared Small Target Detection 

**Authors**: Yuteng Liu, Duanni Meng, Maoxun Yuan, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.02837)  

**Abstract**: Infrared small target detection (IRSTD) faces significant challenges due to the low signal-to-noise ratio (SNR), small target size, and complex cluttered backgrounds. Although recent DETR-based detectors benefit from global context modeling, they exhibit notable performance degradation on IRSTD. We revisit this phenomenon and reveal that the target-relevant embeddings of IRST are inevitably overwhelmed by dominant background features due to the self-attention mechanism, leading to unreliable query initialization and inaccurate target localization. To address this issue, we propose SEF-DETR, a novel framework that refines query initialization for IRSTD. Specifically, SEF-DETR consists of three components: Frequency-guided Patch Screening (FPS), Dynamic Embedding Enhancement (DEE), and Reliability-Consistency-aware Fusion (RCF). The FPS module leverages the Fourier spectrum of local patches to construct a target-relevant density map, suppressing background-dominated features. DEE strengthens multi-scale representations in a target-aware manner, while RCF further refines object queries by enforcing spatial-frequency consistency and reliability. Extensive experiments on three public IRSTD datasets demonstrate that SEF-DETR achieves superior detection performance compared to state-of-the-art methods, delivering a robust and efficient solution for infrared small target detection task. 

---
# MiMo-V2-Flash Technical Report 

**Authors**: Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, Gang Xie, Hailin Zhang, Hanglong Lv, Hanyu Li, Heyu Chen, Hongshen Xu, Houbin Zhang, Huaqiu Liu, Jiangshan Duo, Jianyu Wei, Jiebao Xiao, Jinhao Dong, Jun Shi, Junhao Hu, Kainan Bao, Kang Zhou, Lei Li, Liang Zhao, Linghao Zhang, Peidian Li, Qianli Chen, Shaohui Liu, Shihua Yu, Shijie Cao, Shimao Chen, Shouqiu Yu, Shuo Liu, Tianling Zhou, Weijiang Su, Weikun Wang, Wenhan Ma, Xiangwei Deng, Bohan Mao, Bowen Ye, Can Cai, Chenghua Wang, Chengxuan Zhu, Chong Ma, Chun Chen, Chunan Li, Dawei Zhu, Deshan Xiao, Dong Zhang, Duo Zhang, Fangyue Liu, Feiyu Yang, Fengyuan Shi, Guoan Wang, Hao Tian, Hao Wu, Heng Qu, Hongfei Yi, Hongxu An, Hongyi Guan, Xing Zhang, Yifan Song, Yihan Yan, Yihao Zhao, Yingchun Lai, Yizhao Gao, Yu Cheng, Yuanyuan Tian, Yudong Wang, Zhen Tang, Zhengju Tang, Zhengtao Wen, Zhichao Song, Zhixian Zheng, Zihan Jiang, Jian Wen, Jiarui Sun, Jiawei Li, Jinlong Xue, Jun Xia, Kai Fang, Menghang Zhu, Nuo Chen, Qian Tu, Qihao Zhang, Qiying Wang, Rang Li, Rui Ma, Shaolei Zhang, Shengfan Wang, Shicheng Li, Shuhao Gu, Shuhuai Ren, Sirui Deng, Tao Guo, Tianyang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02780)  

**Abstract**: We present MiMo-V2-Flash, a Mixture-of-Experts (MoE) model with 309B total parameters and 15B active parameters, designed for fast, strong reasoning and agentic capabilities. MiMo-V2-Flash adopts a hybrid attention architecture that interleaves Sliding Window Attention (SWA) with global attention, with a 128-token sliding window under a 5:1 hybrid ratio. The model is pre-trained on 27 trillion tokens with Multi-Token Prediction (MTP), employing a native 32k context length and subsequently extended to 256k. To efficiently scale post-training compute, MiMo-V2-Flash introduces a novel Multi-Teacher On-Policy Distillation (MOPD) paradigm. In this framework, domain-specialized teachers (e.g., trained via large-scale reinforcement learning) provide dense and token-level reward, enabling the student model to perfectly master teacher expertise. MiMo-V2-Flash rivals top-tier open-weight models such as DeepSeek-V3.2 and Kimi-K2, despite using only 1/2 and 1/3 of their total parameters, respectively. During inference, by repurposing MTP as a draft model for speculative decoding, MiMo-V2-Flash achieves up to 3.6 acceptance length and 2.6x decoding speedup with three MTP layers. We open-source both the model weights and the three-layer MTP weights to foster open research and community collaboration. 

---
# Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation 

**Authors**: Haoyu Dong, Zhengmao He, Yang Li, Zhibin Li, Xinyu Yi, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.02778)  

**Abstract**: Human-like dexterous hands with multiple fingers offer human-level manipulation capabilities, but training control policies that can directly deploy on real hardware remains difficult due to contact-rich physics and imperfect actuation. We close this gap with a practical sim-to-real reinforcement learning (RL) framework that utilizes dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions. To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling to bridge the actuation gaps with randomization of non-ideal effects such as backlash, torque-speed saturation. Using an asymmetric actor-critic PPO pipeline trained entirely in simulation, our policies deploy directly to a five-finger hand. The resulting policies demonstrated two essential skills: (1) command-based, controllable grasp force tracking, and (2) reorientation of objects in the hand, both of which were robustly executed without fine-tuning on the robot. By combining tactile and torque in the observation space with effective sensing/actuation modeling, our system provides a practical solution to achieve reliable dexterous manipulation. To our knowledge, this is the first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot on real hardware. 

---
# UniSRCodec: Unified and Low-Bitrate Single Codebook Codec with Sub-Band Reconstruction 

**Authors**: Zhisheng Zhang, Xiang Li, Yixuan Zhou, Jing Peng, Shengbo Cai, Guoyang Zeng, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02776)  

**Abstract**: Neural Audio Codecs (NACs) can reduce transmission overhead by performing compact compression and reconstruction, which also aim to bridge the gap between continuous and discrete signals. Existing NACs can be divided into two categories: multi-codebook and single-codebook codecs. Multi-codebook codecs face challenges such as structural complexity and difficulty in adapting to downstream tasks, while single-codebook codecs, though structurally simpler, suffer from low-fidelity, ineffective modeling of unified audio, and an inability to support modeling of high-frequency audio. We propose the UniSRCodec, a single-codebook codec capable of supporting high sampling rate, low-bandwidth, high fidelity, and unified. We analyze the inefficiency of waveform-based compression and introduce the time and frequency compression method using the Mel-spectrogram, and cooperate with a Vocoder to recover the phase information of the original audio. Moreover, we propose a sub-band reconstruction technique to achieve high-quality compression across both low and high frequency bands. Subjective and objective experimental results demonstrate that UniSRCodec achieves state-of-the-art (SOTA) performance among cross-domain single-codebook codecs with only a token rate of 40, and its reconstruction quality is comparable to that of certain multi-codebook methods. Our demo page is available at this https URL. 

---
# Netflix Artwork Personalization via LLM Post-training 

**Authors**: Hyunji Nam, Sejoon Oh, Emma Kong, Yesu Feng, Moumita Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2601.02764)  

**Abstract**: Large language models (LLMs) have demonstrated success in various applications of user recommendation and personalization across e-commerce and entertainment. On many entertainment platforms such as Netflix, users typically interact with a wide range of titles, each represented by an artwork. Since users have diverse preferences, an artwork that appeals to one type of user may not resonate with another with different preferences. Given this user heterogeneity, our work explores the novel problem of personalized artwork recommendations according to diverse user preferences. Similar to the multi-dimensional nature of users' tastes, titles contain different themes and tones that may appeal to different viewers. For example, the same title might feature both heartfelt family drama and intense action scenes. Users who prefer romantic content may like the artwork emphasizing emotional warmth between the characters, while those who prefer action thrillers may find high-intensity action scenes more intriguing. Rather than a one-size-fits-all approach, we conduct post-training of pre-trained LLMs to make personalized artwork recommendations, selecting the most preferred visual representation of a title for each user and thereby improving user satisfaction and engagement. Our experimental results with Llama 3.1 8B models (trained on a dataset of 110K data points and evaluated on 5K held-out user-title pairs) show that the post-trained LLMs achieve 3-5\% improvements over the Netflix production model, suggesting a promising direction for granular personalized recommendations using LLMs. 

---
# Q-Regularized Generative Auto-Bidding: From Suboptimal Trajectories to Optimal Policies 

**Authors**: Mingming Zhang, Na Li, Zhuang Feiqing, Hongyang Zheng, Jiangbing Zhou, Wang Wuyin, Sheng-jie Sun, XiaoWei Chen, Junxiong Zhu, Lixin Zou, Chenliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02754)  

**Abstract**: With the rapid development of e-commerce, auto-bidding has become a key asset in optimizing advertising performance under diverse advertiser environments. The current approaches focus on reinforcement learning (RL) and generative models. These efforts imitate offline historical behaviors by utilizing a complex structure with expensive hyperparameter tuning. The suboptimal trajectories further exacerbate the difficulty of policy learning.
To address these challenges, we proposes QGA, a novel Q-value regularized Generative Auto-bidding method. In QGA, we propose to plug a Q-value regularization with double Q-learning strategy into the Decision Transformer backbone. This design enables joint optimization of policy imitation and action-value maximization, allowing the learned bidding policy to both leverage experience from the dataset and alleviate the adverse impact of the suboptimal trajectories. Furthermore, to safely explore the policy space beyond the data distribution, we propose a Q-value guided dual-exploration mechanism, in which the DT model is conditioned on multiple return-to-go targets and locally perturbed actions. This entire exploration process is dynamically guided by the aforementioned Q-value module, which provides principled evaluation for each candidate action. Experiments on public benchmarks and simulation environments demonstrate that QGA consistently achieves superior or highly competitive results compared to existing alternatives. Notably, in large-scale real-world A/B testing, QGA achieves a 3.27% increase in Ad GMV and a 2.49% improvement in Ad ROI. 

---
# Window-based Membership Inference Attacks Against Fine-tuned Large Language Models 

**Authors**: Yuetian Chen, Yuntao Du, Kaiyuan Zhang, Ashish Kundu, Charles Fleming, Bruno Ribeiro, Ninghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02751)  

**Abstract**: Most membership inference attacks (MIAs) against Large Language Models (LLMs) rely on global signals, like average loss, to identify training data. This approach, however, dilutes the subtle, localized signals of memorization, reducing attack effectiveness. We challenge this global-averaging paradigm, positing that membership signals are more pronounced within localized contexts. We introduce WBC (Window-Based Comparison), which exploits this insight through a sliding window approach with sign-based aggregation. Our method slides windows of varying sizes across text sequences, with each window casting a binary vote on membership based on loss comparisons between target and reference models. By ensembling votes across geometrically spaced window sizes, we capture memorization patterns from token-level artifacts to phrase-level structures. Extensive experiments across eleven datasets demonstrate that WBC substantially outperforms established baselines, achieving higher AUC scores and 2-3 times improvements in detection rates at low false positive thresholds. Our findings reveal that aggregating localized evidence is fundamentally more effective than global averaging, exposing critical privacy vulnerabilities in fine-tuned LLMs. 

---
# Hypothesize-Then-Verify: Speculative Root Cause Analysis for Microservices with Pathwise Parallelism 

**Authors**: Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Leyi Pan, Chiming Duan, Minghua He, Pei Xiao, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02736)  

**Abstract**: Microservice systems have become the backbone of cloud-native enterprise applications due to their resource elasticity, loosely coupled architecture, and lightweight deployment. Yet, the intrinsic complexity and dynamic runtime interactions of such systems inevitably give rise to anomalies. Ensuring system reliability therefore hinges on effective root cause analysis (RCA), which entails not only localizing the source of anomalies but also characterizing the underlying failures in a timely and interpretable manner. Recent advances in intelligent RCA techniques, particularly those powered by large language models (LLMs), have demonstrated promising capabilities, as LLMs reduce reliance on handcrafted features while offering cross-platform adaptability, task generalization, and flexibility. However, existing LLM-based methods still suffer from two critical limitations: (a) limited exploration diversity, which undermines accuracy, and (b) heavy dependence on large-scale LLMs, which results in slow inference. To overcome these challenges, we propose SpecRCA, a speculative root cause analysis framework for microservices that adopts a \textit{hypothesize-then-verify} paradigm. SpecRCA first leverages a hypothesis drafting module to rapidly generate candidate root causes, and then employs a parallel root cause verifier to efficiently validate them. Preliminary experiments on the AIOps 2022 dataset demonstrate that SpecRCA achieves superior accuracy and efficiency compared to existing approaches, highlighting its potential as a practical solution for scalable and interpretable RCA in complex microservice environments. 

---
# Agentic Memory Enhanced Recursive Reasoning for Root Cause Localization in Microservices 

**Authors**: Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Leyi Pan, Chiming Duan, Minghua He, Mengxi Jia, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02732)  

**Abstract**: As contemporary microservice systems become increasingly popular and complex-often comprising hundreds or even thousands of fine-grained, interdependent subsystems-they are experiencing more frequent failures. Ensuring system reliability thus demands accurate root cause localization. While many traditional graph-based and deep learning approaches have been explored for this task, they often rely heavily on pre-defined schemas that struggle to adapt to evolving operational contexts. Consequently, a number of LLM-based methods have recently been proposed. However, these methods still face two major limitations: shallow, symptom-centric reasoning that undermines accuracy, and a lack of cross-alert reuse that leads to redundant reasoning and high latency. In this paper, we conduct a comprehensive study of how Site Reliability Engineers (SREs) localize the root causes of failures, drawing insights from professionals across multiple organizations. Our investigation reveals that expert root cause analysis exhibits three key characteristics: recursiveness, multi-dimensional expansion, and cross-modal reasoning. Motivated by these findings, we introduce AMER-RCL, an agentic memory enhanced recursive reasoning framework for root cause localization in microservices. AMER-RCL employs the Recursive Reasoning RCL engine, a multi-agent framework that performs recursive reasoning on each alert to progressively refine candidate causes, while Agentic Memory incrementally accumulates and reuses reasoning from prior alerts within a time window to reduce redundant exploration and lower inference latency. Experimental results demonstrate that AMER-RCL consistently outperforms state-of-the-art methods in both localization accuracy and inference efficiency. 

---
# Foreground-Aware Dataset Distillation via Dynamic Patch Selection 

**Authors**: Longzhen Li, Guang Li, Ren Togo, Keisuke Maeda, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2601.02727)  

**Abstract**: In this paper, we propose a foreground-aware dataset distillation method that enhances patch selection in a content-adaptive manner. With the rising computational cost of training large-scale deep models, dataset distillation has emerged as a promising approach for constructing compact synthetic datasets that retain the knowledge of their large original counterparts. However, traditional optimization-based methods often suffer from high computational overhead, memory constraints, and the generation of unrealistic, noise-like images with limited architectural generalization. Recent non-optimization methods alleviate some of these issues by constructing distilled data from real image patches, but the used rigid patch selection strategies can still discard critical information about the main objects. To solve this problem, we first leverage Grounded SAM2 to identify foreground objects and compute per-image foreground occupancy, from which we derive a category-wise patch decision threshold. Guided by these thresholds, we design a dynamic patch selection strategy that, for each image, either selects the most informative patch from multiple candidates or directly resizes the full image when the foreground dominates. This dual-path mechanism preserves more key information about the main objects while reducing redundant background content. Extensive experiments on multiple benchmarks show that the proposed method consistently improves distillation performance over existing approaches, producing more informative and representative distilled datasets and enhancing robustness across different architectures and image compositions. 

---
# Privacy-Preserving AI-Enabled Decentralized Learning and Employment Records System 

**Authors**: Yuqiao Xu, Mina Namazi, Sahith Reddy Jalapally, Osama Zafar, Youngjin Yoo, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2601.02720)  

**Abstract**: Learning and Employment Record (LER) systems are emerging as critical infrastructure for securely compiling and sharing educational and work achievements. Existing blockchain-based platforms leverage verifiable credentials but typically lack automated skill-credential generation and the ability to incorporate unstructured evidence of learning. In this paper,a privacy-preserving, AI-enabled decentralized LER system is proposed to address these gaps. Digitally signed transcripts from educational institutions are accepted, and verifiable self-issued skill credentials are derived inside a trusted execution environment (TEE) by a natural language processing pipeline that analyzes formal records (e.g., transcripts, syllabi) and informal artifacts. All verification and job-skill matching are performed inside the enclave with selective disclosure, so raw credentials and private keys remain enclave-confined. Job matching relies solely on attested skill vectors and is invariant to non-skill resume fields, thereby reducing opportunities for screening this http URL NLP component was evaluated on sample learner data; the mapping follows the validated Syllabus-to-O*NET methodology,and a stability test across repeated runs observed <5% variance in top-ranked skills. Formal security statements and proof sketches are provided showing that derived credentials are unforgeable and that sensitive information remains confidential. The proposed system thus supports secure education and employment credentialing, robust transcript verification,and automated, privacy-preserving skill extraction within a decentralized framework. 

---
# CREAM: Continual Retrieval on Dynamic Streaming Corpora with Adaptive Soft Memory 

**Authors**: HuiJeong Son, Hyeongu Kang, Sunho Kim, Subeen Ho, SeongKu Kang, Dongha Lee, Susik Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2601.02708)  

**Abstract**: Information retrieval (IR) in dynamic data streams is emerging as a challenging task, as shifts in data distribution degrade the performance of AI-powered IR systems. To mitigate this issue, memory-based continual learning has been widely adopted for IR. However, existing methods rely on a fixed set of queries with ground-truth relevant documents, which limits generalization to unseen queries and documents, making them impractical for real-world applications. To enable more effective learning with unseen topics of a new corpus without ground-truth labels, we propose CREAM, a self-supervised framework for memory-based continual retrieval. CREAM captures the evolving semantics of streaming queries and documents into dynamically structured soft memory and leverages it to adapt to both seen and unseen topics in an unsupervised setting. We realize this through three key techniques: fine-grained similarity estimation, regularized cluster prototyping, and stratified coreset sampling. Experiments on two benchmark datasets demonstrate that CREAM exhibits superior adaptability and retrieval accuracy, outperforming the strongest method in a label-free setting by 27.79\% in Success@5 and 44.5\% in Recall@10 on average, and achieving performance comparable to or even exceeding that of supervised methods. 

---
# Adversarial Question Answering Robustness: A Multi-Level Error Analysis and Mitigation Study 

**Authors**: Agniv Roy Choudhury, Vignesh Ponselvan Rajasingh  

**Link**: [PDF](https://arxiv.org/pdf/2601.02700)  

**Abstract**: Question answering (QA) systems achieve impressive performance on standard benchmarks like SQuAD, but remain vulnerable to adversarial examples. This project investigates the adversarial robustness of transformer models on the AddSent adversarial dataset through systematic experimentation across model scales and targeted mitigation strategies. We perform comprehensive multi-level error analysis using five complementary categorization schemes, identifying negation confusion and entity substitution as the primary failure modes. Through systematic evaluation of adversarial fine-tuning ratios, we identify 80% clean + 20% adversarial data as optimal. Data augmentation experiments reveal a capacity bottleneck in small models. Scaling from ELECTRA-small (14M parameters) to ELECTRA-base (110M parameters) eliminates the robustness-accuracy trade-off, achieving substantial improvements on both clean and adversarial data. We implement three targeted mitigation strategies, with Entity-Aware contrastive learning achieving best performance: 89.89% AddSent Exact Match (EM) and 90.73% SQuAD EM, representing 94.9% closure of the adversarial gap. To our knowledge, this is the first work integrating comprehensive linguistic error analysis with Named Entity Recognition (NER)-guided contrastive learning for adversarial QA, demonstrating that targeted mitigation can achieve near-parity between clean and adversarial performance. 

---
# Multi-channel multi-speaker transformer for speech recognition 

**Authors**: Guo Yifan, Tian Yao, Suo Hongbin, Wan Yulong  

**Link**: [PDF](https://arxiv.org/pdf/2601.02688)  

**Abstract**: With the development of teleconferencing and in-vehicle voice assistants, far-field multi-speaker speech recognition has become a hot research topic. Recently, a multi-channel transformer (MCT) has been proposed, which demonstrates the ability of the transformer to model far-field acoustic environments. However, MCT cannot encode high-dimensional acoustic features for each speaker from mixed input audio because of the interference between speakers. Based on these, we propose the multi-channel multi-speaker transformer (M2Former) for far-field multi-speaker ASR in this paper. Experiments on the SMS-WSJ benchmark show that the M2Former outperforms the neural beamformer, MCT, dual-path RNN with transform-average-concatenate and multi-channel deep clustering based end-to-end systems by 9.2%, 14.3%, 24.9%, and 52.2% respectively, in terms of relative word error rate reduction. 

---
# Topology-Independent Robustness of the Weighted Mean under Label Poisoning Attacks in Heterogeneous Decentralized Learning 

**Authors**: Jie Peng, Weiyu Li, Stefan Vlaski, Qing Ling  

**Link**: [PDF](https://arxiv.org/pdf/2601.02682)  

**Abstract**: Robustness to malicious attacks is crucial for practical decentralized signal processing and machine learning systems. A typical example of such attacks is label poisoning, meaning that some agents possess corrupted local labels and share models trained on these poisoned data. To defend against malicious attacks, existing works often focus on designing robust aggregators; meanwhile, the weighted mean aggregator is typically considered a simple, vulnerable baseline. This paper analyzes the robustness of decentralized gradient descent under label poisoning attacks, considering both robust and weighted mean aggregators. Theoretical results reveal that the learning errors of robust aggregators depend on the network topology, whereas the performance of weighted mean aggregator is topology-independent. Remarkably, the weighted mean aggregator, although often considered vulnerable, can outperform robust aggregators under sufficient heterogeneity, particularly when: (i) the global contamination rate (i.e., the fraction of poisoned agents for the entire network) is smaller than the local contamination rate (i.e., the maximal fraction of poisoned neighbors for the regular agents); (ii) the network of regular agents is disconnected; or (iii) the network of regular agents is sparse and the local contamination rate is high. Empirical results support our theoretical findings, highlighting the important role of network topology in the robustness to label poisoning attacks. 

---
# Extracting books from production language models 

**Authors**: Ahmed Ahmed, A. Feder Cooper, Sanmi Koyejo, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02671)  

**Abstract**: Many unresolved legal questions over LLMs and copyright center on memorization: whether specific training data have been encoded in the model's weights during training, and whether those memorized data can be extracted in the model's outputs. While many believe that LLMs do not memorize much of their training data, recent work shows that substantial amounts of copyrighted text can be extracted from open-weight models. However, it remains an open question if similar extraction is feasible for production LLMs, given the safety measures these systems implement. We investigate this question using a two-phase procedure: (1) an initial probe to test for extraction feasibility, which sometimes uses a Best-of-N (BoN) jailbreak, followed by (2) iterative continuation prompts to attempt to extract the book. We evaluate our procedure on four production LLMs -- Claude 3.7 Sonnet, GPT-4.1, Gemini 2.5 Pro, and Grok 3 -- and we measure extraction success with a score computed from a block-based approximation of longest common substring (nv-recall). With different per-LLM experimental configurations, we were able to extract varying amounts of text. For the Phase 1 probe, it was unnecessary to jailbreak Gemini 2.5 Pro and Grok 3 to extract text (e.g, nv-recall of 76.8% and 70.3%, respectively, for Harry Potter and the Sorcerer's Stone), while it was necessary for Claude 3.7 Sonnet and GPT-4.1. In some cases, jailbroken Claude 3.7 Sonnet outputs entire books near-verbatim (e.g., nv-recall=95.8%). GPT-4.1 requires significantly more BoN attempts (e.g., 20X), and eventually refuses to continue (e.g., nv-recall=4.0%). Taken together, our work highlights that, even with model- and system-level safeguards, extraction of (in-copyright) training data remains a risk for production LLMs. 

---
# When Do Tools and Planning Help LLMs Think? A Cost- and Latency-Aware Benchmark 

**Authors**: Subha Ghoshal, Ali Al-Bustami  

**Link**: [PDF](https://arxiv.org/pdf/2601.02663)  

**Abstract**: Modern large language models (LLMs) increasingly rely on inference-time planning and external tools to improve reasoning. We benchmark this behavior on two real-world settings: event-centric question answering over graph-structured knowledge (Event-QA) and persuasive response generation in Reddit ChangeMyView (CMV). Using LangChain and LangGraph, we compare a one-shot baseline against a plan--execute--replan agent equipped with task-specific tools (DBpedia SPARQL/lookup/schema exploration, Wikipedia-focused retrieval, and topical web search). We evaluate on 60 examples each from Event-QA and CMV (3 splits of 20), and report both mean end-to-end latency and per-example token cost estimates. We evaluate GPT-4o and GPT-4o-mini under identical workflows and report accuracy and end-to-end latency. On Event-QA, the best tool-augmented configuration improves accuracy (e.g., 47.5\% $\rightarrow$ 67.5\% for GPT-4o) while increasing latency by orders of magnitude ($\sim$8s $\rightarrow$ $\sim$317s per example). On CMV, one-shot prompting is strongest (e.g., GPT-4o-mini achieves 75\% at $\sim$6s), and planning+search increases latency substantially without consistent gains. However, complex multi-tool orchestration exposes failure modes where the smaller model degrades. Overall, the findings highlight the need for task-specific, cost-aware choices of both model size and agent/tooling complexity. 

---
# Effective Online 3D Bin Packing with Lookahead Parcels Using Monte Carlo Tree Search 

**Authors**: Jiangyi Fang, Bowen Zhou, Haotian Wang, Xin Zhu, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02649)  

**Abstract**: Online 3D Bin Packing (3D-BP) with robotic arms is crucial for reducing transportation and labor costs in modern logistics. While Deep Reinforcement Learning (DRL) has shown strong performance, it often fails to adapt to real-world short-term distribution shifts, which arise as different batches of goods arrive sequentially, causing performance drops. We argue that the short-term lookahead information available in modern logistics systems is key to mitigating this issue, especially during distribution shifts. We formulate online 3D-BP with lookahead parcels as a Model Predictive Control (MPC) problem and adapt the Monte Carlo Tree Search (MCTS) framework to solve it. Our framework employs a dynamic exploration prior that automatically balances a learned RL policy and a robust random policy based on the lookahead characteristics. Additionally, we design an auxiliary reward to penalize long-term spatial waste from individual placements. Extensive experiments on real-world datasets show that our method consistently outperforms state-of-the-art baselines, achieving over 10\% gains under distributional shifts, 4\% average improvement in online deployment, and up to more than 8\% in the best case--demonstrating the effectiveness of our framework. 

---
# Prioritized Replay for RL Post-training 

**Authors**: Mehdi Fatemi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02648)  

**Abstract**: We introduce a problem-level prioritization framework for RL post-training of large language models. Building on insights from prioritized replay in deep RL, as well as prior observations that rollouts with intermediate success rates tend to produce stronger learning signals under methods such as GRPO, our approach selects problems according to a simple, model-driven priority score derived from empirical success statistics. In contrast to conventional curriculum strategies that emphasize easier tasks early in training, the resulting schedule naturally focuses training on problems that are neither consistently solved nor consistently failed, while deprioritizing those that contribute little gradient information. The method yields a continuously adapting and automatic prioritization process that requires no predefined difficulty tiers, auxiliary predictors, or external labels. We further introduce lightweight mechanisms for practical deployment, including heap-based prioritized sampling and periodic retesting of solved and unsolved problems to mitigate starvation and forgetting. Overall, the approach offers a principled and scalable alternative to manually designed curricula while aligning data selection directly with the dynamics of GRPO-based post-training. 

---
# DreamLoop: Controllable Cinemagraph Generation from a Single Photograph 

**Authors**: Aniruddha Mahapatra, Long Mai, Cusuh Ham, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02646)  

**Abstract**: Cinemagraphs, which combine static photographs with selective, looping motion, offer unique artistic appeal. Generating them from a single photograph in a controllable manner is particularly challenging. Existing image-animation techniques are restricted to simple, low-frequency motions and operate only in narrow domains with repetitive textures like water and smoke. In contrast, large-scale video diffusion models are not tailored for cinemagraph constraints and lack the specialized data required to generate seamless, controlled loops. We present DreamLoop, a controllable video synthesis framework dedicated to generating cinemagraphs from a single photo without requiring any cinemagraph training data. Our key idea is to adapt a general video diffusion model by training it on two objectives: temporal bridging and motion conditioning. This strategy enables flexible cinemagraph generation. During inference, by using the input image as both the first- and last- frame condition, we enforce a seamless loop. By conditioning on static tracks, we maintain a static background. Finally, by providing a user-specified motion path for a target object, our method provides intuitive control over the animation's trajectory and timing. To our knowledge, DreamLoop is the first method to enable cinemagraph generation for general scenes with flexible and intuitive controls. We demonstrate that our method produces high-quality, complex cinemagraphs that align with user intent, outperforming existing approaches. 

---
# Credit Assignment via Neural Manifold Noise Correlation 

**Authors**: Byungwoo Kang, Maceo Richards, Bernardo Sabatini  

**Link**: [PDF](https://arxiv.org/pdf/2601.02636)  

**Abstract**: Credit assignment--how changes in individual neurons and synapses affect a network's output--is central to learning in brains and machines. Noise correlation, which estimates gradients by correlating perturbations of activity with changes in output, provides a biologically plausible solution to credit assignment but scales poorly as accurately estimating the Jacobian requires that the number of perturbations scale with network size. Moreover, isotropic noise conflicts with neurobiological observations that neural activity lies on a low-dimensional manifold. To address these drawbacks, we propose neural manifold noise correlation (NMNC), which performs credit assignment using perturbations restricted to the neural manifold. We show theoretically and empirically that the Jacobian row space aligns with the neural manifold in trained networks, and that manifold dimensionality scales slowly with network size. NMNC substantially improves performance and sample efficiency over vanilla noise correlation in convolutional networks trained on CIFAR-10, ImageNet-scale models, and recurrent networks. NMNC also yields representations more similar to the primate visual system than vanilla noise correlation. These findings offer a mechanistic hypothesis for how biological circuits could support credit assignment, and suggest that biologically inspired constraints may enable, rather than limit, effective learning at scale. 

---
# TAAF: A Trace Abstraction and Analysis Framework Synergizing Knowledge Graphs and LLMs 

**Authors**: Alireza Ezaz, Ghazal Khodabandeh, Majid Babaei, Naser Ezzati-Jivan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02632)  

**Abstract**: Execution traces are a critical source of information for understanding, debugging, and optimizing complex software systems. However, traces from OS kernels or large-scale applications like Chrome or MySQL are massive and difficult to analyze. Existing tools rely on predefined analyses, and custom insights often require writing domain-specific scripts, which is an error-prone and time-consuming task. This paper introduces TAAF (Trace Abstraction and Analysis Framework), a novel approach that combines time-indexing, knowledge graphs (KGs), and large language models (LLMs) to transform raw trace data into actionable insights. TAAF constructs a time-indexed KG from trace events to capture relationships among entities such as threads, CPUs, and system resources. An LLM then interprets query-specific subgraphs to answer natural-language questions, reducing the need for manual inspection and deep system expertise. To evaluate TAAF, we introduce TraceQA-100, a benchmark of 100 questions grounded in real kernel traces. Experiments across three LLMs and multiple temporal settings show that TAAF improves answer accuracy by up to 31.2%, particularly in multi-hop and causal reasoning tasks. We further analyze where graph-grounded reasoning helps and where limitations remain, offering a foundation for next-generation trace analysis tools. 

---
# Improved Evidence Extraction for Document Inconsistency Detection with LLMs 

**Authors**: Nelvin Tan, Yaowen Zhang, James Asikin Cheung, Fusheng Liu, Yu-Ching Shih, Dong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02627)  

**Abstract**: Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. However, research on LLM-based approaches to document inconsistency detection is relatively limited. There are two key aspects of document inconsistency detection: (i) classification of whether there exists any inconsistency, and (ii) providing evidence of the inconsistent sentences. We focus on the latter, and introduce new comprehensive evidence-extraction metrics and a redact-and-retry framework with constrained filtering that substantially improves LLM-based document inconsistency detection over direct prompting. We back our claims with promising experimental results. 

---
# LAsset: An LLM-assisted Security Asset Identification Framework for System-on-Chip (SoC) Verification 

**Authors**: Md Ajoad Hasan, Dipayan Saha, Khan Thamid Hasan, Nashmin Alam, Azim Uddin, Sujan Kumar Saha, Mark Tehranipoor, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02624)  

**Abstract**: The growing complexity of modern system-on-chip (SoC) and IP designs is making security assurance difficult day by day. One of the fundamental steps in the pre-silicon security verification of a hardware design is the identification of security assets, as it substantially influences downstream security verification tasks, such as threat modeling, security property generation, and vulnerability detection. Traditionally, assets are determined manually by security experts, requiring significant time and expertise. To address this challenge, we present LAsset, a novel automated framework that leverages large language models (LLMs) to identify security assets from both hardware design specifications and register-transfer level (RTL) descriptions. The framework performs structural and semantic analysis to identify intra-module primary and secondary assets and derives inter-module relationships to systematically characterize security dependencies at the design level. Experimental results show that the proposed framework achieves high classification accuracy, reaching up to 90% recall rate in SoC design, and 93% recall rate in IP designs. This automation in asset identification significantly reduces manual overhead and supports a scalable path forward for secure hardware development. 

---
# Hierarchical temporal receptive windows and zero-shot timescale generalization in biologically constrained scale-invariant deep networks 

**Authors**: Aakash Sarkar, Marc W. Howard  

**Link**: [PDF](https://arxiv.org/pdf/2601.02618)  

**Abstract**: Human cognition integrates information across nested timescales. While the cortex exhibits hierarchical Temporal Receptive Windows (TRWs), local circuits often display heterogeneous time constants. To reconcile this, we trained biologically constrained deep networks, based on scale-invariant hippocampal time cells, on a language classification task mimicking the hierarchical structure of language (e.g., 'letters' forming 'words'). First, using a feedforward model (SITHCon), we found that a hierarchy of TRWs emerged naturally across layers, despite the network having an identical spectrum of time constants within layers. We then distilled these inductive priors into a biologically plausible recurrent architecture, SITH-RNN. Training a sequence of architectures ranging from generic RNNs to this restricted subset showed that the scale-invariant SITH-RNN learned faster with orders-of-magnitude fewer parameters, and generalized zero-shot to out-of-distribution timescales. These results suggest the brain employs scale-invariant, sequential priors - coding "what" happened "when" - making recurrent networks with such priors particularly well-suited to describe human cognition. 

---
# Chronicals: A High-Performance Framework for LLM Fine-Tuning with 3.51x Speedup over Unsloth 

**Authors**: Arjun S. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2601.02609)  

**Abstract**: Large language model fine-tuning is bottlenecked by memory: a 7B parameter model requires 84GB--14GB for weights, 14GB for gradients, and 56GB for FP32 optimizer states--exceeding even A100-40GB capacity. We present Chronicals, an open-source training framework achieving 3.51x speedup over Unsloth through four synergistic optimizations: (1) fused Triton kernels eliminating 75% of memory traffic via RMSNorm (7x), SwiGLU (5x), and QK-RoPE (2.3x) fusion; (2) Cut Cross-Entropy reducing logit memory from 5GB to 135MB through online softmax computation; (3) LoRA+ with theoretically-derived 16x differential learning rates between adapter matrices; and (4) Best-Fit Decreasing sequence packing recovering 60-75% of compute wasted on padding.
On Qwen2.5-0.5B with A100-40GB, Chronicals achieves 41,184 tokens/second for full fine-tuning versus Unsloth's 11,736 tokens/second (3.51x). For LoRA at rank 32, we reach 11,699 tokens/second versus Unsloth MAX's 2,857 tokens/second (4.10x). Critically, we discovered that Unsloth's reported 46,000 tokens/second benchmark exhibited zero gradient norms--the model was not training.
We provide complete mathematical foundations: online softmax correctness proofs, FlashAttention IO complexity bounds O(N^2 d^2 M^{-1}), LoRA+ learning rate derivations from gradient magnitude analysis, and bin-packing approximation guarantees. All implementations, benchmarks, and proofs are available at this https URL with pip installation via this https URL. 

---
# LongDA: Benchmarking LLM Agents for Long-Document Data Analysis 

**Authors**: Yiyang Li, Zheyuan Zhang, Tianyi Ma, Zehong Wang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.02598)  

**Abstract**: We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings. 

---
# Annealed Langevin Posterior Sampling (ALPS): A Rapid Algorithm for Image Restoration with Multiscale Energy Models 

**Authors**: Jyothi Rikhab Chand, Mathews Jacob  

**Link**: [PDF](https://arxiv.org/pdf/2601.02594)  

**Abstract**: Solving inverse problems in imaging requires models that support efficient inference, uncertainty quantification, and principled probabilistic reasoning. Energy-Based Models (EBMs), with their interpretable energy landscapes and compositional structure, are well-suited for this task but have historically suffered from high computational costs and training instability. To overcome the historical shortcomings of EBMs, we introduce a fast distillation strategy to transfer the strengths of pre-trained diffusion models into multi-scale EBMs. These distilled EBMs enable efficient sampling and preserve the interpretability and compositionality inherent to potential-based frameworks. Leveraging EBM compositionality, we propose Annealed Langevin Posterior Sampling (ALPS) algorithm for Maximum-A-Posteriori (MAP), Minimum Mean Square Error (MMSE), and uncertainty estimates for inverse problems in imaging. Unlike diffusion models that use complex guidance strategies for latent variables, we perform annealing on static posterior distributions that are well-defined and composable. Experiments on image inpainting and MRI reconstruction demonstrate that our method matches or surpasses diffusion-based baselines in both accuracy and efficiency, while also supporting MAP recovery. Overall, our framework offers a scalable and principled solution for inverse problems in imaging, with potential for practical deployment in scientific and clinical settings. ALPS code is available at the GitHub repository \href{this https URL}{ALPS}. 

---
# FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions 

**Authors**: Kris W Pan, Yongmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2601.02589)  

**Abstract**: Over 3.5 million patents are filed annually, with drafting patent descriptions requiring deep technical and legal expertise. Transforming scientific papers into patent descriptions is particularly challenging due to their differing rhetorical styles and stringent legal requirements. Unlike black-box text-to-text approaches that struggle to model structural reasoning and legal constraints, we propose FlowPlan-G2P, a novel framework that mirrors the cognitive workflow of expert drafters by reformulating this task into three stages: (1) Concept Graph Induction, extracting technical entities and relationships into a directed graph via expert-like reasoning; (2) Paragraph and Section Planning, reorganizing the graph into coherent clusters aligned with canonical patent sections; and (3) Graph-Conditioned Generation, producing legally compliant paragraphs using section-specific subgraphs and tailored prompts. Experiments demonstrate that FlowPlan-G2P significantly improves logical coherence and legal compliance over end-to-end LLM baselines. Our framework establishes a new paradigm for paper-to-patent generation and advances structured text generation for specialized domains. 

---
# Reconstructing Item Characteristic Curves using Fine-Tuned Large Language Models 

**Authors**: Christopher Ormerod  

**Link**: [PDF](https://arxiv.org/pdf/2601.02580)  

**Abstract**: Traditional methods for determining assessment item parameters, such as difficulty and discrimination, rely heavily on expensive field testing to collect student performance data for Item Response Theory (IRT) calibration. This study introduces a novel approach that implicitly models these psychometric properties by fine-tuning Large Language Models (LLMs) to simulate student responses across a spectrum of latent abilities. Leveraging the Qwen-3 dense model series and Low-Rank Adaptation (LoRA), we train models to generate responses to multiple choice questions conditioned on discrete ability descriptors. We reconstruct the probability of a correct response as a function of student ability, effectively generating synthetic Item Characteristic Curves (ICCs) to estimate IRT parameters. Evaluation on a dataset of Grade 6 English Language Arts (ELA) items and the BEA 2024 Shared Task dataset demonstrates that this method competes with or outperforms baseline approaches. This simulation-based technique seems particularly effective at modeling item discrimination. 

---
# Fact-Checking with Large Language Models via Probabilistic Certainty and Consistency 

**Authors**: Haoran Wang, Maryam Khalid, Qiong Wu, Jian Gao, Cheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.02574)  

**Abstract**: Large language models (LLMs) are increasingly used in applications requiring factual accuracy, yet their outputs often contain hallucinated responses. While fact-checking can mitigate these errors, existing methods typically retrieve external evidence indiscriminately, overlooking the model's internal knowledge and potentially introducing irrelevant noise. Moreover, current systems lack targeted mechanisms to resolve specific uncertainties in the model's reasoning. Inspired by how humans fact-check, we argue that LLMs should adaptively decide whether to rely on internal knowledge or initiate retrieval based on their confidence in a given claim. We introduce Probabilistic Certainty and Consistency (PCC), a framework that estimates factual confidence by jointly modeling an LLM's probabilistic certainty and reasoning consistency. These confidence signals enable an adaptive verification strategy: the model answers directly when confident, triggers targeted retrieval when uncertain or inconsistent, and escalates to deep search when ambiguity is high. Our confidence-guided routing mechanism ensures that retrieval is invoked only when necessary, improving both efficiency and reliability. Extensive experiments across three challenging benchmarks show that PCC achieves better uncertainty quantification than verbalized confidence and consistently outperforms strong LLM-based fact-checking baselines. Furthermore, we demonstrate that PCC generalizes well across various LLMs. 

---
# LendNova: Towards Automated Credit Risk Assessment with Language Models 

**Authors**: Kiarash Shamsi, Danijel Novokmet, Joshua Peters, Mao Lin Liu, Paul K Edwards, Vahab Khoshdel  

**Link**: [PDF](https://arxiv.org/pdf/2601.02573)  

**Abstract**: Credit risk assessment is essential in the financial sector, but has traditionally depended on costly feature-based models that often fail to utilize all available information in raw credit records. This paper introduces LendNova, the first practical automated end-to-end pipeline for credit risk assessment, designed to utilize all available information in raw credit records by leveraging advanced NLP techniques and language models. LendNova transforms risk modeling by operating directly on raw, jargon-heavy credit bureau text using a language model that learns task-relevant representations without manual feature engineering. By automatically capturing patterns and risk signals embedded in the text, it replaces manual preprocessing steps, reducing costs and improving scalability. Evaluation on real-world data further demonstrates its strong potential in accurate and efficient risk assessment. LendNova establishes a baseline for intelligent credit risk agents, demonstrating the feasibility of language models in this domain. It lays the groundwork for future research toward foundation systems that enable more accurate, adaptable, and automated financial decision-making. 

---
# AI-exposed jobs deteriorated before ChatGPT 

**Authors**: Morgan R. Frank, Alireza Javadian Sabet, Lisa Simon, Sarah H. Bana, Renzhe Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02554)  

**Abstract**: Public debate links worsening job prospects for AI-exposed occupations to the release of ChatGPT in late 2022. Using monthly U.S. unemployment insurance records, we measure occupation- and location-specific unemployment risk and find that risk rose in AI-exposed occupations beginning in early 2022, months before ChatGPT. Analyzing millions of LinkedIn profiles, we show that graduate cohorts from 2021 onward entered AI-exposed jobs at lower rates than earlier cohorts, with gaps opening before late 2022. Finally, from millions of university syllabi, we find that graduates taking more AI-exposed curricula had higher first-job pay and shorter job searches after ChatGPT. Together, these results point to forces pre-dating generative AI and to the ongoing value of LLM-relevant education. 

---
# Normalized Conditional Mutual Information Surrogate Loss for Deep Neural Classifiers 

**Authors**: Linfeng Ye, Zhixiang Chi, Konstantinos N. Plataniotis, En-hui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02543)  

**Abstract**: In this paper, we propose a novel information theoretic surrogate loss; normalized conditional mutual information (NCMI); as a drop in alternative to the de facto cross-entropy (CE) for training deep neural network (DNN) based classifiers. We first observe that the model's NCMI is inversely proportional to its accuracy. Building on this insight, we introduce an alternating algorithm to efficiently minimize the NCMI. Across image recognition and whole-slide imaging (WSI) subtyping benchmarks, NCMI-trained models surpass state of the art losses by substantial margins at a computational cost comparable to that of CE. Notably, on ImageNet, NCMI yields a 2.77% top-1 accuracy improvement with ResNet-50 comparing to the CE; on CAMELYON-17, replacing CE with NCMI improves the macro-F1 by 8.6% over the strongest baseline. Gains are consistent across various architectures and batch sizes, suggesting that NCMI is a practical and competitive alternative to CE. 

---
# ModeX: Evaluator-Free Best-of-N Selection for Open-Ended Generation 

**Authors**: Hyeong Kyu Choi, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02535)  

**Abstract**: Selecting a single high-quality output from multiple stochastic generations remains a fundamental challenge for large language models (LLMs), particularly in open-ended tasks where no canonical answer exists. While Best-of-N and self-consistency methods show that aggregating multiple generations can improve performance, existing approaches typically rely on external evaluators, reward models, or exact string-match voting, limiting their applicability and efficiency. We propose Mode Extraction (ModeX), an evaluator-free Best-of-N selection framework that generalizes majority voting to open-ended text generation by identifying the modal output representing the dominant semantic consensus among generated texts. ModeX constructs a similarity graph over candidate generations and recursively applies spectral clustering to select a representative centroid, without requiring additional inference or auxiliary models. We further instantiate this selection principle as ModeX--Lite, an improved version of ModeX with early pruning for efficiency. Across open-ended tasks--including text summarization, code generation, and mathematical reasoning--our approaches consistently outperform standard single- and multi-path baselines, providing a computationally efficient solution for robust open-ended text generation. Code is released in this https URL. 

---
# Losses that Cook: Topological Optimal Transport for Structured Recipe Generation 

**Authors**: Mattia Ottoborgo, Daniele Rege Cambrin, Paolo Garza  

**Link**: [PDF](https://arxiv.org/pdf/2601.02531)  

**Abstract**: Cooking recipes are complex procedures that require not only a fluent and factual text, but also accurate timing, temperature, and procedural coherence, as well as the correct composition of ingredients. Standard training procedures are primarily based on cross-entropy and focus solely on fluency. Building on RECIPE-NLG, we investigate the use of several composite objectives and present a new topological loss that represents ingredient lists as point clouds in embedding space, minimizing the divergence between predicted and gold ingredients. Using both standard NLG metrics and recipe-specific metrics, we find that our loss significantly improves ingredient- and action-level metrics. Meanwhile, the Dice loss excels in time/temperature precision, and the mixed loss yields competitive trade-offs with synergistic gains in quantity and time. A human preference analysis supports our finding, showing our model is preferred in 62% of the cases. 

---
# Enhancing Debugging Skills with AI-Powered Assistance: A Real-Time Tool for Debugging Support 

**Authors**: Elizaveta Artser, Daniil Karol, Anna Potriasaeva, Aleksei Rostovskii, Katsiaryna Dzialets, Ekaterina Koshchenko, Xiaotian Su, April Yi Wang, Anastasiia Birillo  

**Link**: [PDF](https://arxiv.org/pdf/2601.02504)  

**Abstract**: Debugging is a crucial skill in programming education and software development, yet it is often overlooked in CS curricula. To address this, we introduce an AI-powered debugging assistant integrated into an IDE. It offers real-time support by analyzing code, suggesting breakpoints, and providing contextual hints. Using RAG with LLMs, program slicing, and custom heuristics, it enhances efficiency by minimizing LLM calls and improving accuracy. A three-level evaluation - technical analysis, UX study, and classroom tests - highlights its potential for teaching debugging. 

---
# GEM-Style Constraints for PEFT with Dual Gradient Projection in LoRA 

**Authors**: Brian Tekmen, Jason Yin, Qianqian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.02500)  

**Abstract**: Full fine-tuning of Large Language Models (LLMs) is computationally costly, motivating Continual Learning (CL) approaches that utilize parameter-efficient adapters. We revisit Gradient Episodic Memory (GEM) within the Low-Rank Adapter (LoRA) subspace and introduce I-GEM: a fixed-budget, GPU-resident dual projected-gradient approximation to GEM's quadratic projection. By constraining non-interference solely within the adapter parameters, I-GEM preserves GEM-like stability with orders-of-magnitude lower mean projection overhead. On a 3-task AG News split with induced domain drift, using GPT-2 (355M) and LoRA ($r=8$), I-GEM matches GEM's average accuracy (within $\sim\!0.04$ pts) and outperforms A-GEM by $\sim\!1.4$ pts. Crucially, it reduces projection time vs.\ GEM by a factor of $\sim\!10^3$. These results suggest that applying GEM constraints in the LoRA subspace is a practical pathway for continual learning at the LLM scale. 

---
# The Rise of Agentic Testing: Multi-Agent Systems for Robust Software Quality Assurance 

**Authors**: Saba Naqvi, Mohammad Baqar, Nawaz Ali Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2601.02454)  

**Abstract**: Software testing has progressed toward intelligent automation, yet current AI-based test generators still suffer from static, single-shot outputs that frequently produce invalid, redundant, or non-executable tests due to the lack of execution aware feedback. This paper introduces an agentic multi-model testing framework a closed-loop, self-correcting system in which a Test Generation Agent, an Execution and Analysis Agent, and a Review and Optimization Agent collaboratively generate, execute, analyze, and refine tests until convergence. By using sandboxed execution, detailed failure reporting, and iterative regeneration or patching of failing tests, the framework autonomously improves test quality and expands coverage. Integrated into a CI/CD-compatible pipeline, it leverages reinforcement signals from coverage metrics and execution outcomes to guide refinement. Empirical evaluations on microservice based applications show up to a 60% reduction in invalid tests, 30% coverage improvement, and significantly reduced human effort compared to single-model baselines demonstrating that multi-agent, feedback-driven loops can evolve software testing into an autonomous, continuously learning quality assurance ecosystem for self-healing, high-reliability codebases. 

---
# mHC-GNN: Manifold-Constrained Hyper-Connections for Graph Neural Networks 

**Authors**: Subhankar Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2601.02451)  

**Abstract**: Graph Neural Networks (GNNs) suffer from over-smoothing in deep architectures and expressiveness bounded by the 1-Weisfeiler-Leman (1-WL) test. We adapt Manifold-Constrained Hyper-Connections (\mhc)~\citep{xie2025mhc}, recently proposed for Transformers, to graph neural networks. Our method, mHC-GNN, expands node representations across $n$ parallel streams and constrains stream-mixing matrices to the Birkhoff polytope via Sinkhorn-Knopp normalization. We prove that mHC-GNN exhibits exponentially slower over-smoothing (rate $(1-\gamma)^{L/n}$ vs.\ $(1-\gamma)^L$) and can distinguish graphs beyond 1-WL. Experiments on 10 datasets with 4 GNN architectures show consistent improvements. Depth experiments from 2 to 128 layers reveal that standard GNNs collapse to near-random performance beyond 16 layers, while mHC-GNN maintains over 74\% accuracy even at 128 layers, with improvements exceeding 50 percentage points at extreme depths. Ablations confirm that the manifold constraint is essential: removing it causes up to 82\% performance degradation. Code is available at \href{this https URL}{this https URL} 

---
# VocalBridge: Latent Diffusion-Bridge Purification for Defeating Perturbation-Based Voiceprint Defenses 

**Authors**: Maryam Abbasihafshejani, AHM Nazmus Sakib, Murtuza Jadliwala  

**Link**: [PDF](https://arxiv.org/pdf/2601.02444)  

**Abstract**: The rapid advancement of speech synthesis technologies, including text-to-speech (TTS) and voice conversion (VC), has intensified security and privacy concerns related to voice cloning. Recent defenses attempt to prevent unauthorized cloning by embedding protective perturbations into speech to obscure speaker identity while maintaining intelligibility. However, adversaries can apply advanced purification techniques to remove these perturbations, recover authentic acoustic characteristics, and regenerate cloneable voices. Despite the growing realism of such attacks, the robustness of existing defenses under adaptive purification remains insufficiently studied.
Most existing purification methods are designed to counter adversarial noise in automatic speech recognition (ASR) systems rather than speaker verification or voice cloning pipelines. As a result, they fail to suppress the fine-grained acoustic cues that define speaker identity and are often ineffective against speaker verification attacks (SVA). To address these limitations, we propose Diffusion-Bridge (VocalBridge), a purification framework that learns a latent mapping from perturbed to clean speech in the EnCodec latent space. Using a time-conditioned 1D U-Net with a cosine noise schedule, the model enables efficient, transcript-free purification while preserving speaker-discriminative structure. We further introduce a Whisper-guided phoneme variant that incorporates lightweight temporal guidance without requiring ground-truth transcripts. Experimental results show that our approach consistently outperforms existing purification methods in recovering cloneable voices from protected speech. Our findings demonstrate the fragility of current perturbation-based defenses and highlight the need for more robust protection mechanisms against evolving voice-cloning and speaker verification threats. 

---
# Evaluating the Diagnostic Classification Ability of Multimodal Large Language Models: Insights from the Osteoarthritis Initiative 

**Authors**: Li Wang, Xi Chen, XiangWen Deng, HuaHui Yi, ZeKun Jiang, Kang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.02443)  

**Abstract**: Multimodal large language models (MLLMs) show promising performance on medical visual question answering (VQA) and report generation, but these generation and explanation abilities do not reliably transfer to disease-specific classification. We evaluated MLLM architectures on knee osteoarthritis (OA) radiograph classification, which remains underrepresented in existing medical MLLM benchmarks, even though knee OA affects an estimated 300 to 400 million people worldwide. Through systematic ablation studies manipulating the vision encoder, the connector, and the large language model (LLM) across diverse training strategies, we measured each component's contribution to diagnostic accuracy. In our classification task, a trained vision encoder alone could outperform full MLLM pipelines in classification accuracy and fine-tuning the LLM provided no meaningful improvement over prompt-based guidance. And LoRA fine-tuning on a small, class-balanced dataset (500 images) gave better results than training on a much larger but class-imbalanced set (5,778 images), indicating that data balance and quality can matter more than raw scale for this task. These findings suggest that for domain-specific medical classification, LLMs are more effective as interpreters and report generators rather than as primary classifiers. Therefore, the MLLM architecture appears less suitable for medical image diagnostic classification tasks that demand high certainty. We recommend prioritizing vision encoder optimization and careful dataset curation when developing clinically applicable systems. 

---
# Understanding Pure Textual Reasoning for Blind Image Quality Assessment 

**Authors**: Yuan Li, Shin'ya Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2601.02441)  

**Abstract**: Textual reasoning has recently been widely adopted in Blind Image Quality Assessment (BIQA). However, it remains unclear how textual information contributes to quality prediction and to what extent text can represent the score-related image contents. This work addresses these questions from an information-flow perspective by comparing existing BIQA models with three paradigms designed to learn the image-text-score relationship: Chain-of-Thought, Self-Consistency, and Autoencoder. Our experiments show that the score prediction performance of the existing model significantly drops when only textual information is used for prediction. Whereas the Chain-of-Thought paradigm introduces little improvement in BIQA performance, the Self-Consistency paradigm significantly reduces the gap between image- and text-conditioned predictions, narrowing the PLCC/SRCC difference to 0.02/0.03. The Autoencoder-like paradigm is less effective in closing the image-text gap, yet it reveals a direction for further optimization. These findings provide insights into how to improve the textual reasoning for BIQA and high-level vision tasks. 

---
# Mitigating Long-Tailed Anomaly Score Distributions with Importance-Weighted Loss 

**Authors**: Jungi Lee, Jungkwon Kim, Chi Zhang, Sangmin Kim, Kwangsun Yoo, Seok-Joo Byun  

**Link**: [PDF](https://arxiv.org/pdf/2601.02440)  

**Abstract**: Anomaly detection is crucial in industrial applications for identifying rare and unseen patterns to ensure system reliability. Traditional models, trained on a single class of normal data, struggle with real-world distributions where normal data exhibit diverse patterns, leading to class imbalance and long-tailed anomaly score distributions (LTD). This imbalance skews model training and degrades detection performance, especially for minority instances. To address this issue, we propose a novel importance-weighted loss designed specifically for anomaly detection. Compared to the previous method for LTD in classification, our method does not require prior knowledge of normal data classes. Instead, we introduce a weighted loss function that incorporates importance sampling to align the distribution of anomaly scores with a target Gaussian, ensuring a balanced representation of normal data. Extensive experiments on three benchmark image datasets and three real-world hyperspectral imaging datasets demonstrate the robustness of our approach in mitigating LTD-induced bias. Our method improves anomaly detection performance by 0.043, highlighting its effectiveness in real-world applications. 

---
# Focus on What Matters: Fisher-Guided Adaptive Multimodal Fusion for Vulnerability Detection 

**Authors**: Yun Bian, Yi Chen, HaiQuan Wang, ShiHao Li, Zhe Cui  

**Link**: [PDF](https://arxiv.org/pdf/2601.02438)  

**Abstract**: Software vulnerability detection is a critical task for securing software systems and can be formulated as a binary classification problem: given a code snippet, determine whether it contains a vulnerability. Existing multimodal approaches typically fuse Natural Code Sequence (NCS) representations from pretrained language models with Code Property Graph (CPG) representations from graph neural networks, often under the implicit assumption that adding a modality necessarily yields extra information. In practice, sequence and graph representations can be redundant, and fluctuations in the quality of the graph modality can dilute the discriminative signal of the dominant modality. To address this, we propose TaCCS-DFA, a framework that introduces Fisher information as a geometric measure of how sensitive feature directions are to the classification decision, enabling task-oriented complementary fusion. TaCCS-DFA online estimates a low-rank principal Fisher subspace and restricts cross-modal attention to task-sensitive directions, thereby retrieving structural features from CPG that complement the sequence modality; meanwhile, an adaptive gating mechanism dynamically adjusts the contribution of the graph modality for each sample to suppress noise propagation. Our analysis shows that, under an isotropic perturbation assumption, the proposed mechanism admits a tighter risk bound than conventional full-spectrum attention. Experiments on BigVul, Devign, and ReVeal show that TaCCS-DFA achieves strong performance across multiple backbones. With CodeT5 as the backbone, TaCCS-DFA reaches an F1 score of 87.80\% on the highly imbalanced BigVul dataset, improving over a strong baseline Vul-LMGNNs by 6.3 percentage points while maintaining low calibration error and computational overhead. 

---
# TAP-ViTs: Task-Adaptive Pruning for On-Device Deployment of Vision Transformers 

**Authors**: Zhibo Wang, Zuoyuan Zhang, Xiaoyi Pang, Qile Zhang, Xuanyi Hao, Shuguo Zhuo, Peng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.02437)  

**Abstract**: Vision Transformers (ViTs) have demonstrated strong performance across a wide range of vision tasks, yet their substantial computational and memory demands hinder efficient deployment on resource-constrained mobile and edge devices. Pruning has emerged as a promising direction for reducing ViT complexity. However, existing approaches either (i) produce a single pruned model shared across all devices, ignoring device heterogeneity, or (ii) rely on fine-tuning with device-local data, which is often infeasible due to limited on-device resources and strict privacy constraints. As a result, current methods fall short of enabling task-customized ViT pruning in privacy-preserving mobile computing settings. This paper introduces TAP-ViTs, a novel task-adaptive pruning framework that generates device-specific pruned ViT models without requiring access to any raw local data. Specifically, to infer device-level task characteristics under privacy constraints, we propose a Gaussian Mixture Model (GMM)-based metric dataset construction mechanism. Each device fits a lightweight GMM to approximate its private data distribution and uploads only the GMM parameters. Using these parameters, the cloud selects distribution-consistent samples from public data to construct a task-representative metric dataset for each device. Based on this proxy dataset, we further develop a dual-granularity importance evaluation-based pruning strategy that jointly measures composite neuron importance and adaptive layer importance, enabling fine-grained, task-aware pruning tailored to each device's computational budget. Extensive experiments across multiple ViT backbones and datasets demonstrate that TAP-ViTs consistently outperforms state-of-the-art pruning methods under comparable compression ratios. 

---
# WebCoderBench: Benchmarking Web Application Generation with Comprehensive and Interpretable Evaluation Metrics 

**Authors**: Chenxu Liu, Yingjie Fu, Wei Yang, Ying Zhang, Tao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.02430)  

**Abstract**: Web applications (web apps) have become a key arena for large language models (LLMs) to demonstrate their code generation capabilities and commercial potential. However, building a benchmark for LLM-generated web apps remains challenging due to the need for real-world user requirements, generalizable evaluation metrics without relying on ground-truth implementations or test cases, and interpretable evaluation results. To address these challenges, we introduce WebCoderBench, the first real-world-collected, generalizable, and interpretable benchmark for web app generation. WebCoderBench comprises 1,572 real user requirements, covering diverse modalities and expression styles that reflect realistic user intentions. WebCoderBench provides 24 fine-grained evaluation metrics across 9 perspectives, combining rule-based and LLM-as-a-judge paradigm for fully automated, objective, and general evaluation. Moreover, WebCoderBench adopts human-preference-aligned weights over metrics to yield interpretable overall scores. Experiments across 12 representative LLMs and 2 LLM-based agents show that there exists no dominant model across all evaluation metrics, offering an opportunity for LLM developers to optimize their models in a targeted manner for a more powerful version. 

---
# A Dynamic Retrieval-Augmented Generation System with Selective Memory and Remembrance 

**Authors**: Okan Bursa  

**Link**: [PDF](https://arxiv.org/pdf/2601.02428)  

**Abstract**: We introduce \emph{Adaptive RAG Memory} (ARM), a retrieval-augmented generation (RAG) framework that replaces a static vector index with a \emph{dynamic} memory substrate governed by selective remembrance and decay. Frequently retrieved items are consolidated and protected from forgetting, while rarely used items gradually decay, inspired by cognitive consolidation and forgetting principles. On a lightweight retrieval benchmark, ARM reaches near state-of-the-art performance (e.g., NDCG@5 $\approx$ 0.940, Recall@5 $=1.000$) with only $\sim$22M parameters in the embedding layer, achieving the best efficiency among ultra-efficient models ($<$25M parameters). In addition, we compare static vs. dynamic RAG combinations across Llama 3.1 and GPT-4o. Llama 3.1 with static RAG achieves the highest key-term coverage (67.2\%) at moderate latency, while GPT-4o with a dynamic selective retrieval policy attains the fastest responses (8.2s on average) with competitive coverage (58.7\%). We further present an engineering optimization of the DynamicRAG implementation, making embedding weights configurable, adjustable at runtime, and robust to invalid settings.
ARM yields competitive accuracy, self-regularizing memory growth, and interpretable retention dynamics without retraining the generator\color{black} and provides practical trade-off between quality, latency and memory efficiency for production and research RAG system. 

---
# NitroGen: An Open Foundation Model for Generalist Gaming Agents 

**Authors**: Loïc Magne, Anas Awadalla, Guanzhi Wang, Yinzhen Xu, Joshua Belofsky, Fengyuan Hu, Joohwan Kim, Ludwig Schmidt, Georgia Gkioxari, Jan Kautz, Yisong Yue, Yejin Choi, Yuke Zhu, Linxi "Jim" Fan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02427)  

**Abstract**: We introduce NitroGen, a vision-action foundation model for generalist gaming agents that is trained on 40,000 hours of gameplay videos across more than 1,000 games. We incorporate three key ingredients: 1) an internet-scale video-action dataset constructed by automatically extracting player actions from publicly available gameplay videos, 2) a multi-game benchmark environment that can measure cross-game generalization, and 3) a unified vision-action model trained with large-scale behavior cloning. NitroGen exhibits strong competence across diverse domains, including combat encounters in 3D action games, high-precision control in 2D platformers, and exploration in procedurally generated worlds. It transfers effectively to unseen games, achieving up to 52% relative improvement in task success rates over models trained from scratch. We release the dataset, evaluation suite, and model weights to advance research on generalist embodied agents. 

---
# A large-scale nanocrystal database with aligned synthesis and properties enabling generative inverse design 

**Authors**: Kai Gu, Yingping Liang, Senliang Peng, Aotian Guo, Haizheng Zhong, Ying Fu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02424)  

**Abstract**: The synthesis of nanocrystals has been highly dependent on trial-and-error, due to the complex correlation between synthesis parameters and physicochemical properties. Although deep learning offers a potential methodology to achieve generative inverse design, it is still hindered by the scarcity of high-quality datasets that align nanocrystal synthesis routes with their properties. Here, we present the construction of a large-scale, aligned Nanocrystal Synthesis-Property (NSP) database and demonstrate its capability for generative inverse design. To extract structured synthesis routes and their corresponding product properties from literature, we develop NanoExtractor, a large language model (LLM) enhanced by well-designed augmentation strategies. NanoExtractor is validated against human experts, achieving a weighted average score of 88% on the test set, significantly outperforming chemistry-specialized (3%) and general-purpose LLMs (38%). The resulting NSP database contains nearly 160,000 aligned entries and serves as training data for our NanoDesigner, an LLM for inverse synthesis design. The generative capability of NanoDesigner is validated through the successful design of viable synthesis routes for both well-established PbSe nanocrystals and rarely reported MgF2 nanocrystals. Notably, the model recommends a counter-intuitive, non-stoichiometric precursor ratio (1:1) for MgF2 nanocrystals, which is experimentally confirmed as critical for suppressing byproducts. Our work bridges the gap between unstructured literature and data-driven synthesis, and also establishes a powerful human-AI collaborative paradigm for accelerating nanocrystal discovery. 

---
# Watch Wider and Think Deeper: Collaborative Cross-modal Chain-of-Thought for Complex Visual Reasoning 

**Authors**: Wenting Lu, Didi Zhu, Tao Shen, Donglin Zhu, Ayong Ye, Chao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02422)  

**Abstract**: Multi-modal reasoning requires the seamless integration of visual and linguistic cues, yet existing Chain-of-Thought methods suffer from two critical limitations in cross-modal scenarios: (1) over-reliance on single coarse-grained image regions, and (2) semantic fragmentation between successive reasoning steps. To address these issues, we propose the CoCoT (Collaborative Coross-modal Thought) frame- work, built upon two key innovations: a) Dynamic Multi-Region Grounding to adaptively detect the most relevant image regions based on the question, and b) Relation-Aware Reasoning to enable multi-region collaboration by iteratively align- ing visual cues to form a coherent and logical chain of thought. Through this approach, we construct the CoCoT-70K dataset, comprising 74,691 high-quality samples with multi-region annotations and structured reasoning chains. Extensive experiments demonstrate that CoCoT significantly enhances complex visual rea- soning, achieving an average accuracy improvement of 15.4% on LLaVA-1.5 and 4.0% on Qwen2-VL across six challenging benchmarks. The data and code are available at: this https URL. 

---
# Multimodal Sentiment Analysis based on Multi-channel and Symmetric Mutual Promotion Feature Fusion 

**Authors**: Wangyuan Zhu, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02415)  

**Abstract**: Multimodal sentiment analysis is a key technology in the fields of human-computer interaction and affective computing. Accurately recognizing human emotional states is crucial for facilitating smooth communication between humans and machines. Despite some progress in multimodal sentiment analysis research, numerous challenges remain. The first challenge is the limited and insufficiently rich features extracted from single modality data. Secondly, most studies focus only on the consistency of inter-modal feature information, neglecting the differences between features, resulting in inadequate feature information fusion. In this paper, we first extract multi-channel features to obtain more comprehensive feature information. We employ dual-channel features in both the visual and auditory modalities to enhance intra-modal feature representation. Secondly, we propose a symmetric mutual promotion (SMP) inter-modal feature fusion method. This method combines symmetric cross-modal attention mechanisms and self-attention mechanisms, where the cross-modal attention mechanism captures useful information from other modalities, and the self-attention mechanism models contextual information. This approach promotes the exchange of useful information between modalities, thereby strengthening inter-modal interactions. Furthermore, we integrate intra-modal features and inter-modal fused features, fully leveraging the complementarity of inter-modal feature information while considering feature information differences. Experiments conducted on two benchmark datasets demonstrate the effectiveness and superiority of our proposed method. 

---
# MIAR: Modality Interaction and Alignment Representation Fuison for Multimodal Emotion 

**Authors**: Jichao Zhu, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02414)  

**Abstract**: Multimodal Emotion Recognition (MER) aims to perceive human emotions through three modes: language, vision, and audio. Previous methods primarily focused on modal fusion without adequately addressing significant distributional differences among modalities or considering their varying contributions to the task. They also lacked robust generalization capabilities across diverse textual model features, thus limiting performance in multimodal scenarios. Therefore, we propose a novel approach called Modality Interaction and Alignment Representation (MIAR). This network integrates contextual features across different modalities using a feature interaction to generate feature tokens to represent global representations of this modality extracting information from other modalities. These four tokens represent global representations of how each modality extracts information from others. MIAR aligns different modalities using contrastive learning and normalization strategies. We conduct experiments on two benchmarks: CMU-MOSI and CMU-MOSEI datasets, experimental results demonstrate the MIAR outperforms state-of-the-art MER methods. 

---
# Socially-Aware Recommender Systems Mitigate Opinion Clusterization 

**Authors**: Lukas Schüepp, Carmen Amo Alonso, Florian Dörfler, Giulia De Pasquale  

**Link**: [PDF](https://arxiv.org/pdf/2601.02412)  

**Abstract**: Recommender systems shape online interactions by matching users with creators content to maximize engagement. Creators, in turn, adapt their content to align with users preferences and enhance their popularity. At the same time, users preferences evolve under the influence of both suggested content from the recommender system and content shared within their social circles. This feedback loop generates a complex interplay between users, creators, and recommender algorithms, which is the key cause of filter bubbles and opinion polarization. We develop a social network-aware recommender system that explicitly accounts for this user-creators feedback interaction and strategically exploits the topology of the user's own social network to promote diversification. Our approach highlights how accounting for and exploiting user's social network in the recommender system design is crucial to mediate filter bubble effects while balancing content diversity with personalization. Provably, opinion clusterization is positively correlated with the influence of recommended content on user opinions. Ultimately, the proposed approach shows the power of socially-aware recommender systems in combating opinion polarization and clusterization phenomena. 

---
# SpikySpace: A Spiking State Space Model for Energy-Efficient Time Series Forecasting 

**Authors**: Kaiwen Tang, Jiaqi Zheng, Yuze Jin, Yupeng Qiu, Guangda Sun, Zhanglu Yan, Weng-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2601.02411)  

**Abstract**: Time-series forecasting often operates under tight power and latency budgets in fields like traffic management, industrial condition monitoring, and on-device sensing. These applications frequently require near real-time responses and low energy consumption on edge devices. Spiking neural networks (SNNs) offer event-driven computation and ultra-low power by exploiting temporal sparsity and multiplication-free computation. Yet existing SNN-based time-series forecasters often inherit complex transformer blocks, thereby losing much of the efficiency benefit. To solve the problem, we propose SpikySpace, a spiking state-space model (SSM) that reduces the quadratic cost in the attention block to linear time via selective scanning. Further, we replace dense SSM updates with sparse spike trains and execute selective scans only on spike events, thereby avoiding dense multiplications while preserving the SSM's structured memory. Because complex operations such as exponentials and divisions are costly on neuromorphic chips, we introduce simplified approximations of SiLU and Softplus to enable a neuromorphic-friendly model architecture. In matched settings, SpikySpace reduces estimated energy consumption by 98.73% and 96.24% compared to two state-of-the-art transformer based approaches, namely iTransformer and iSpikformer, respectively. In standard time series forecasting datasets, SpikySpace delivers competitive accuracy while substantially reducing energy cost and memory traffic. As the first full spiking state-space model, SpikySpace bridges neuromorphic efficiency with modern sequence modeling, marking a practical and scalable path toward efficient time series forecasting systems. 

---
# The Vibe-Check Protocol: Quantifying Cognitive Offloading in AI Programming 

**Authors**: Aizierjiang Aiersilan  

**Link**: [PDF](https://arxiv.org/pdf/2601.02410)  

**Abstract**: The integration of Large Language Models (LLMs) into software engineering education has driven the emergence of ``Vibe Coding,'' a paradigm where developers articulate high-level intent through natural language and delegate implementation to AI agents. While proponents argue this approach modernizes pedagogy by emphasizing conceptual design over syntactic memorization, accumulating empirical evidence raises concerns regarding skill retention and deep conceptual understanding. This paper proposes a theoretical framework to investigate the research question: \textit{Is Vibe Coding a better way to learn software engineering?} We posit a divergence in student outcomes between those leveraging AI for acceleration versus those using it for cognitive offloading. To evaluate these educational trade-offs, we propose the \textbf{Vibe-Check Protocol (VCP)}, a systematic benchmarking framework incorporating three quantitative metrics: the \textit{Cold Start Refactor} ($M_{CSR}$) for modeling skill decay; \textit{Hallucination Trap Detection} ($M_{HT}$) based on signal detection theory to evaluate error identification; and the \textit{Explainability Gap} ($E_{gap}$) for quantifying the divergence between code complexity and conceptual comprehension. Through controlled comparisons, VCP aims to provide a quantitative basis for educators to determine the optimal pedagogical boundary: identifying contexts where Vibe Coding fosters genuine mastery and contexts where it introduces hidden technical debt and superficial competence. 

---
# Expert-Guided Explainable Few-Shot Learning with Active Sample Selection for Medical Image Analysis 

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2601.02409)  

**Abstract**: Medical image analysis faces two critical challenges: scarcity of labeled data and lack of model interpretability, both hindering clinical AI deployment. Few-shot learning (FSL) addresses data limitations but lacks transparency in predictions. Active learning (AL) methods optimize data acquisition but overlook interpretability of acquired samples. We propose a dual-framework solution: Expert-Guided Explainable Few-Shot Learning (EGxFSL) and Explainability-Guided AL (xGAL). EGxFSL integrates radiologist-defined regions-of-interest as spatial supervision via Grad-CAM-based Dice loss, jointly optimized with prototypical classification for interpretable few-shot learning. xGAL introduces iterative sample acquisition prioritizing both predictive uncertainty and attention misalignment, creating a closed-loop framework where explainability guides training and sample selection synergistically. On the BraTS (MRI), VinDr-CXR (chest X-ray), and SIIM-COVID-19 (chest X-ray) datasets, we achieve accuracies of 92\%, 76\%, and 62\%, respectively, consistently outperforming non-guided baselines across all datasets. Under severe data constraints, xGAL achieves 76\% accuracy with only 680 samples versus 57\% for random sampling. Grad-CAM visualizations demonstrate guided models focus on diagnostically relevant regions, with generalization validated on breast ultrasound confirming cross-modality applicability. 

---
# PCEval: A Benchmark for Evaluating Physical Computing Capabilities of Large Language Models 

**Authors**: Inpyo Song, Eunji Jeon, Jangwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.02404)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, including software development, education, and technical assistance. Among these, software development is one of the key areas where LLMs are increasingly adopted. However, when hardware constraints are considered-for instance, in physical computing, where software must interact with and control physical hardware -their effectiveness has not been fully explored. To address this gap, we introduce \textsc{PCEval} (Physical Computing Evaluation), the first benchmark in physical computing that enables a fully automatic evaluation of the capabilities of LLM in both the logical and physical aspects of the projects, without requiring human assessment. Our evaluation framework assesses LLMs in generating circuits and producing compatible code across varying levels of project complexity. Through comprehensive testing of 13 leading models, \textsc{PCEval} provides the first reproducible and automatically validated empirical assessment of LLMs' ability to reason about fundamental hardware implementation constraints within a simulation environment. Our findings reveal that while LLMs perform well in code generation and logical circuit design, they struggle significantly with physical breadboard layout creation, particularly in managing proper pin connections and avoiding circuit errors. \textsc{PCEval} advances our understanding of AI assistance in hardware-dependent computing environments and establishes a foundation for developing more effective tools to support physical computing education. 

---
# ProSoftArena: Benchmarking Hierarchical Capabilities of Multimodal Agents in Professional Software Environments 

**Authors**: Jiaxin Ai, Yukang Feng, Fanrui Zhang, Jianwen Sun, Zizhen Li, Chuanhao Li, Yifan Chang, Wenxiao Wu, Ruoxi Wang, Mingliang Zhai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02399)  

**Abstract**: Multimodal agents are making rapid progress on general computer-use tasks, yet existing benchmarks remain largely confined to browsers and basic desktop applications, falling short in professional software workflows that dominate real-world scientific and industrial practice. To close this gap, we introduce ProSoftArena, a benchmark and platform specifically for evaluating multimodal agents in professional software environments. We establish the first capability hierarchy tailored to agent use of professional software and construct a benchmark of 436 realistic work and research tasks spanning 6 disciplines and 13 core professional applications. To ensure reliable and reproducible assessment, we build an executable real-computer environment with an execution-based evaluation framework and uniquely incorporate a human-in-the-loop evaluation paradigm. Extensive experiments show that even the best-performing agent attains only a 24.4\% success rate on L2 tasks and completely fails on L3 multi-software workflow. In-depth analysis further provides valuable insights for addressing current agent limitations and more effective design principles, paving the way to build more capable agents in professional software settings. This project is available at: this https URL. 

---
# AI-Native Integrated Sensing and Communications for Self-Organizing Wireless Networks: Architectures, Learning Paradigms, and System-Level Design 

**Authors**: S. Zhang, M. Feizarefi, A. F. Mirzaei  

**Link**: [PDF](https://arxiv.org/pdf/2601.02398)  

**Abstract**: Integrated Sensing and Communications (ISAC) is emerging as a foundational paradigm for next-generation wireless networks, enabling communication infrastructures to simultaneously support data transmission and environment sensing. By tightly coupling radio sensing with communication functions, ISAC unlocks new capabilities for situational awareness, localization, tracking, and network adaptation. At the same time, the increasing scale, heterogeneity, and dynamics of future wireless systems demand self-organizing network intelligence capable of autonomously managing resources, topology, and services. Artificial intelligence (AI), particularly learning-driven and data-centric methods, has become a key enabler for realizing this vision. This survey provides a comprehensive and system-level review of AI-native ISAC-enabled self-organizing wireless networks. We develop a unified taxonomy that spans: (i) ISAC signal models and sensing modalities, (ii) network state abstraction and perception from sensing-aware radio data, (iii) learning-driven self-organization mechanisms for resource allocation, topology control, and mobility management, and (iv) cross-layer architectures integrating sensing, communication, and network intelligence. We further examine emerging learning paradigms, including deep reinforcement learning, graph-based learning, multi-agent coordination, and federated intelligence that enable autonomous adaptation under uncertainty, mobility, and partial observability. Practical considerations such as sensing-communication trade-offs, scalability, latency, reliability, and security are discussed alongside representative evaluation methodologies and performance metrics. Finally, we identify key open challenges and future research directions toward deployable, trustworthy, and scalable AI-native ISAC systems for 6G and beyond. 

---
# Self-Supervised Masked Autoencoders with Dense-Unet for Coronary Calcium Removal in limited CT Data 

**Authors**: Mo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.02392)  

**Abstract**: Coronary calcification creates blooming artifacts in Computed Tomography Angiography (CTA), severely hampering the diagnosis of lumen stenosis. While Deep Convolutional Neural Networks (DCNNs) like Dense-Unet have shown promise in removing these artifacts via inpainting, they often require large labeled datasets which are scarce in the medical domain. Inspired by recent advancements in Masked Autoencoders (MAE) for 3D point clouds, we propose \textbf{Dense-MAE}, a novel self-supervised learning framework for volumetric medical data. We introduce a pre-training strategy that randomly masks 3D patches of the vessel lumen and trains the Dense-Unet to reconstruct the missing geometry. This forces the encoder to learn high-level latent features of arterial topology without human annotation. Experimental results on clinical CTA datasets demonstrate that initializing the Calcium Removal network with our MAE-based weights significantly improves inpainting accuracy and stenosis estimation compared to training from scratch, specifically in few-shot scenarios. 

---
# Tree of Preferences for Diversified Recommendation 

**Authors**: Hanyang Yuan, Ning Tang, Tongya Zheng, Jiarong Xu, Xintong Hu, Renhong Huang, Shunyu Liu, Jiacong Hu, Jiawei Chen, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.02386)  

**Abstract**: Diversified recommendation has attracted increasing attention from both researchers and practitioners, which can effectively address the homogeneity of recommended items. Existing approaches predominantly aim to infer the diversity of user preferences from observed user feedback. Nonetheless, due to inherent data biases, the observed data may not fully reflect user interests, where underexplored preferences can be overwhelmed or remain unmanifested. Failing to capture these preferences can lead to suboptimal diversity in recommendations. To fill this gap, this work aims to study diversified recommendation from a data-bias perspective. Inspired by the outstanding performance of large language models (LLMs) in zero-shot inference leveraging world knowledge, we propose a novel approach that utilizes LLMs' expertise to uncover underexplored user preferences from observed behavior, ultimately providing diverse and relevant recommendations. To achieve this, we first introduce Tree of Preferences (ToP), an innovative structure constructed to model user preferences from coarse to fine. ToP enables LLMs to systematically reason over the user's rationale behind their behavior, thereby uncovering their underexplored preferences. To guide diversified recommendations using uncovered preferences, we adopt a data-centric approach, identifying candidate items that match user preferences and generating synthetic interactions that reflect underexplored preferences. These interactions are integrated to train a general recommender for diversification. Moreover, we scale up overall efficiency by dynamically selecting influential users during optimization. Extensive evaluations of both diversity and relevance show that our approach outperforms existing methods in most cases and achieves near-optimal performance in others, with reasonable inference latency. 

---
# Base Station Deployment under EMF constrain by Deep Reinforcement learning 

**Authors**: Mohammed Mallik, Guillaume Villemaud  

**Link**: [PDF](https://arxiv.org/pdf/2601.02385)  

**Abstract**: As 5G networks rapidly expand and 6G technologies emerge, characterized by dense deployments, millimeter-wave communications, and dynamic beamforming, the need for scalable simulation tools becomes increasingly critical. These tools must support efficient evaluation of key performance metrics such as coverage and radio-frequency electromagnetic field (RF-EMF) exposure, inform network design decisions, and ensure compliance with safety regulations. Moreover, base station (BS) placement is a crucial task in the network design, where satisfying coverage requirements is essential. To address these, based on our previous work, we first propose a conditional generative adversarial network (cGAN) that predicts location specific received signal strength (RSS), and EMF exposure simultaneously from the network topology, as images. As a network designing application, we propose a Deep Q Network (DQN) framework, using the trained cGAN, for optimal base station (BS) deployment in the network. Compared to conventional ray tracing simulations, the proposed cGAN reduces inference and deployment time from several hours to seconds. Unlike a standalone cGAN, which provides static performance maps, the proposed GAN-DQN framework enables sequential decision making under coverage and exposure constraints, learning effective deployment strategies that directly solve the BS placement problem. Thus making it well suited for real time design and adaptation in dynamic scenarios in order to satisfy pre defined network specific heterogeneous performance goals. 

---
# How to Discover Knowledge for FutureG: Contextual RAG and LLM Prompting for O-RAN 

**Authors**: Nathan Conger, Nathan Scollar, Kemal Davaslioglu, Yalin E. Sagduyu, Sastry Kompella  

**Link**: [PDF](https://arxiv.org/pdf/2601.02382)  

**Abstract**: We present a retrieval-augmented question answering framework for 5G/6G networks, where the Open Radio Access Network (O-RAN) has become central to disaggregated, virtualized, and AI-driven wireless systems. While O-RAN enables multi-vendor interoperability and cloud-native deployments, its fast-changing specifications and interfaces pose major challenges for researchers and practitioners. Manual navigation of these complex documents is labor-intensive and error-prone, slowing system design, integration, and deployment. To address this challenge, we adopt Contextual Retrieval-Augmented Generation (Contextual RAG), a strategy in which candidate answer choices guide document retrieval and chunk-specific context to improve large language model (LLM) performance. This improvement over traditional RAG achieves more targeted and context-aware retrieval, which improves the relevance of documents passed to the LLM, particularly when the query alone lacks sufficient context for accurate grounding. Our framework is designed for dynamic domains where data evolves rapidly and models must be continuously updated or redeployed, all without requiring LLM fine-tuning. We evaluate this framework using the ORANBenchmark-13K dataset, and compare three LLMs, namely, Llama3.2, Qwen2.5-7B, and Qwen3.0-4B, across both Direct Question Answering (Direct Q&A) and Chain-of-Thought (CoT) prompting strategies. We show that Contextual RAG consistently improves accuracy over standard RAG and base prompting, while maintaining competitive runtime and CO2 emissions. These results highlight the potential of Contextual RAG to serve as a scalable and effective solution for domain-specific Q&A in ORAN and broader 5G/6G environments, enabling more accurate interpretation of evolving standards while preserving efficiency and sustainability. 

---
# The Refutability Gap: Challenges in Validating Reasoning by Large Language Models 

**Authors**: Elchanan Mossel  

**Link**: [PDF](https://arxiv.org/pdf/2601.02380)  

**Abstract**: Recent reports claim that Large Language Models (LLMs) have achieved the ability to derive new science and exhibit human-level general intelligence. We argue that such claims are not rigorous scientific claims, as they do not satisfy Popper's refutability principle (often termed falsifiability), which requires that scientific statements be capable of being disproven. We identify several methodological pitfalls in current AI research on reasoning, including the inability to verify the novelty of findings due to opaque and non-searchable training data, the lack of reproducibility caused by continuous model updates, and the omission of human-interaction transcripts, which obscures the true source of scientific discovery. Additionally, the absence of counterfactuals and data on failed attempts creates a selection bias that may exaggerate LLM capabilities. To address these challenges, we propose guidelines for scientific transparency and reproducibility for research on reasoning by LLMs. Establishing such guidelines is crucial for both scientific integrity and the ongoing societal debates regarding fair data usage. 

---
# Movement Primitives in Robotics: A Comprehensive Survey 

**Authors**: Nolan B. Gutierrez, William J. Beksi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02379)  

**Abstract**: Biological systems exhibit a continuous stream of movements, consisting of sequential segments, that allow them to perform complex tasks in a creative and versatile fashion. This observation has led researchers towards identifying elementary building blocks of motion known as movement primitives, which are well-suited for generating motor commands in autonomous systems, such as robots. In this survey, we provide an encyclopedic overview of movement primitive approaches and applications in chronological order. Concretely, we present movement primitive frameworks as a way of representing robotic control trajectories acquired through human demonstrations. Within the area of robotics, movement primitives can encode basic motions at the trajectory level, such as how a robot would grasp a cup or the sequence of motions necessary to toss a ball. Furthermore, movement primitives have been developed with the desirable analytical properties of a spring-damper system, probabilistic coupling of multiple demonstrations, using neural networks in high-dimensional systems, and more, to address difficult challenges in robotics. Although movement primitives have widespread application to a variety of fields, the goal of this survey is to inform practitioners on the use of these frameworks in the context of robotics. Specifically, we aim to (i) present a systematic review of major movement primitive frameworks and examine their strengths and weaknesses; (ii) highlight applications that have successfully made use of movement primitives; and (iii) examine open questions and discuss practical challenges when applying movement primitives in robotics. 

---
# LeafTutor: An AI Agent for Programming Assignment Tutoring 

**Authors**: Madison Bochard, Tim Conser, Alyssa Duran, Lazaro Martull, Pu Tian, Yalong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.02375)  

**Abstract**: High enrollment in STEM-related degree programs has created increasing demand for scalable tutoring support, as universities experience a shortage of qualified instructors and teaching assistants (TAs). To address this challenge, LeafTutor, an AI tutoring agent powered by large language models (LLMs), was developed to provide step-by-step guidance for students. LeafTutor was evaluated through real programming assignments. The results indicate that the system can deliver step-by-step programming guidance comparable to human tutors. This work demonstrates the potential of LLM-driven tutoring solutions to enhance and personalize learning in STEM education. If any reader is interested in collaboration with our team to improve or test LeafTutor, please contact Pu Tian (this http URL@stockton.edu) or Yalong Wu (wuy@uhcl.edu). 

---
# Permission Manifests for Web Agents 

**Authors**: Samuele Marro, Alan Chan, Xinxing Ren, Lewis Hammond, Jesse Wright, Gurjyot Wanga, Tiziano Piccardi, Nuno Campos, Tobin South, Jialin Yu, Alex Pentland, Philip Torr, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2601.02371)  

**Abstract**: The rise of Large Language Model (LLM)-based web agents represents a significant shift in automated interactions with the web. Unlike traditional crawlers that follow simple conventions, such as this http URL, modern agents engage with websites in sophisticated ways: navigating complex interfaces, extracting structured information, and completing end-to-end tasks. Existing governance mechanisms were not designed for these capabilities. Without a way to specify what interactions are and are not allowed, website owners increasingly rely on blanket blocking and CAPTCHAs, which undermine beneficial applications such as efficient automation, convenient use of e-commerce services, and accessibility tools. We introduce this http URL, a this http URL-style lightweight manifest where websites specify allowed interactions, complemented by API references where available. This framework provides a low-friction coordination mechanism: website owners only need to write a simple JSON file, while agents can easily parse and automatically implement the manifest's provisions. Website owners can then focus on blocking non-compliant agents, rather than agents as a whole. By extending the spirit of this http URL to the era of LLM-mediated interaction, and complementing data use initiatives such as AIPref, the manifest establishes a compliance framework that enables beneficial agent interactions while respecting site owners' preferences. 

---
# Distillation-based Scenario-Adaptive Mixture-of-Experts for the Matching Stage of Multi-scenario Recommendation 

**Authors**: Ruibing Wang, Shuhan Guo, Haotong Du, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2601.02368)  

**Abstract**: Multi-scenario recommendation is pivotal for optimizing user experience across diverse contexts. While Multi-gate Mixture-of-Experts (MMOE) thrives in ranking, its transfer to the matching stage is hindered by the blind optimization inherent to independent two-tower architectures and the parameter dominance of head scenarios. To address these structural and distributional bottlenecks, we propose Distillation-based Scenario-Adaptive Mixture-of-Experts (DSMOE). Specially, we devise a Scenario-Adaptive Projection (SAP) module to generate lightweight, context-specific parameters, effectively preventing expert collapse in long-tail scenarios. Concurrently, we introduce a cross-architecture knowledge distillation framework, where an interaction-aware teacher guides the two-tower student to capture complex matching patterns. Extensive experiments on real-world datasets demonstrate DSMOE's superiority, particularly in significantly improving retrieval quality for under-represented, data-sparse scenarios. 

---
# Cross-Platform Digital Discourse Analysis of the Israel-Hamas Conflict: Sentiment, Topics, and Event Dynamics 

**Authors**: Despoina Antonakaki, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.02367)  

**Abstract**: The Israeli-Palestinian conflict remains one of the most polarizing geopolitical issues, with the October 2023 escalation intensifying online debate. Social media platforms, particularly Telegram, have become central to real-time news sharing, advocacy, and propaganda. In this study, we analyze Telegram, Twitter/X, and Reddit to examine how conflict narratives are produced, amplified, and contested across different digital spheres. Building on our previous work on Telegram discourse during the 2023 escalation, we extend the analysis longitudinally and cross-platform using an updated dataset spanning October 2023 to mid-2025. The corpus includes more than 187,000 Telegram messages, 2.1 million Reddit comments, and curated Twitter/X posts. We combine Latent Dirichlet Allocation (LDA), BERTopic, and transformer-based sentiment and emotion models to identify dominant themes, emotional dynamics, and propaganda strategies. Telegram channels provide unfiltered, high-intensity documentation of events; Twitter/X amplifies frames to global audiences; and Reddit hosts more reflective and deliberative discussions. Our findings reveal persistent negative sentiment, strong coupling between humanitarian framing and solidarity expressions, and platform-specific pathways for the diffusion of pro-Palestinian and pro-Israeli narratives. This paper offers three contributions: (1) a multi-platform, FAIR-compliant dataset on the Israel-Hamas war, (2) an integrated pipeline combining topic modeling, sentiment and emotion analysis, and spam filtering for large-scale conflict discourse, and (3) empirical insights into how platform affordances and affective publics shape the evolution of digital conflict communication. 

---
# TextBridgeGNN: Pre-training Graph Neural Network for Cross-Domain Recommendation via Text-Guided Transfer 

**Authors**: Yiwen Chen, Yiqing Wu, Huishi Luo, Fuzhen Zhuang, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.02366)  

**Abstract**: Graph-based recommendation has achieved great success in recent years. The classical graph recommendation model utilizes ID embedding to store essential collaborative information. However, this ID-based paradigm faces challenges in transferring to a new domain, making it hard to build a pre-trained graph recommendation model. This phenomenon primarily stems from two inherent challenges: (1) the non-transferability of ID embeddings due to isolated domain-specific ID spaces, and (2) structural incompatibility between heterogeneous interaction graphs across domains.
To address these issues, we propose TextBridgeGNN, a pre-training and fine-tuning framework that can effectively transfer knowledge from a pre-trained GNN to downstream tasks. We believe the key lies in how to build the relationship between domains. Specifically, TextBridgeGNN uses text as a semantic bridge to connect domains through multi-level graph propagation. During the pre-training stage, textual information is utilized to break the data islands formed by multiple domains, and hierarchical GNNs are designed to learn both domain-specific and domain-global knowledge with text features, ensuring the retention of collaborative signals and the enhancement of semantics. During the fine-tuning stage, a similarity transfer mechanism is proposed. This mechanism initializes ID embeddings in the target domain by transferring from semantically related nodes, successfully transferring the ID embeddings and graph pattern.
Experiments demonstrate that TextBridgeGNN outperforms existing methods in cross-domain, multi-domain, and training-free settings, highlighting its ability to integrate Pre-trained Language Model (PLM)-driven semantics with graph-based collaborative filtering without costly language model fine-tuning or real-time inference overhead. 

---
# FUSE : Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation 

**Authors**: Tushar Vatsa, Vibha Belavadi, Priya Shanmugasundaram, Suhas Suresha, Dewang Sultania  

**Link**: [PDF](https://arxiv.org/pdf/2601.02365)  

**Abstract**: Multimodal creative assistants decompose user goals and route tasks to subagents for layout, styling, retrieval, and generation. Retrieval quality is pivotal, yet failures can arise at several stages: understanding user intent, choosing content types, finding candidates (recall), or ranking results. Meanwhile, sending and processing images is costly, making naive multimodal approaches impractical. We present FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation. FUSE replaces most raw-image prompting with a compact Grounded Design Representation (GDR): a selection aware JSON of canvas elements (image, text, shape, icon, video, logo), structure, styles, salient colors, and user selection provided by the Planner team. FUSE implements seven context budgeting strategies: comprehensive baseline prompting, context compression, chain-of-thought reasoning, mini-shot optimization, retrieval-augmented context, two-stage processing, and zero-shot minimalism. Finally, a pipeline attribution layer monitors system performance by converting subagent signals into simple checks: intent alignment, content-type/routing sanity, recall health (e.g., zero-hit and top-match strength), and ranking displacement analysis. We evaluate the seven context budgeting variants across 788 evaluation queries from diverse users and design templates (refer Figure 3). Our systematic evaluation reveals that Context Compression achieves optimal performance across all pipeline stages, with 93.3% intent accuracy, 86.8% routing success(with fallbacks), 99.4% recall, and 88.5% NDCG@5. This approach demonstrates that strategic context summarization outperforms both comprehensive and minimal contextualization strategies. 

---
# Towards Trustworthy LLM-Based Recommendation via Rationale Integration 

**Authors**: Chung Park, Taesan Kim, Hyeongjun Yun, Dongjoon Hong, Junui Hong, Kijung Park, MinCheol Cho, Mira Myong, Jihoon Oh, Min sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.02364)  

**Abstract**: Traditional recommender systems (RS) have been primarily optimized for accuracy and short-term engagement, often overlooking transparency and trustworthiness. Recently, platforms such as Amazon and Instagram have begun providing recommendation rationales to users, acknowledging their critical role in fostering trust and enhancing engagement; however, most existing systems still treat them as post-hoc artifacts. We propose an LLM-based recommender (LLM-Rec) that not only predicts items but also generates logically grounded rationales. Our approach leverages a self-annotated rationale dataset and instruction tuning in a rationale-first format, where the model generates an explanation before outputting the recommended item. By adopting this strategy and representing rationales in a chain-of-thought (CoT) style, LLM-Rec strengthens both interpretability and recommendation performance. Experiments on the Fashion and Scientific domains of the Amazon Review dataset demonstrate significant improvements over well-established baselines. To encourage reproducibility and future research, we publicly release a rationale-augmented recommendation dataset containing user histories, rationales, and recommended items. 

---
# The Impact of LLM-Generated Reviews on Recommender Systems: Textual Shifts, Performance Effects, and Strategic Platform Control 

**Authors**: Itzhak Ziv, Moshe Unger, Hilah Geva  

**Link**: [PDF](https://arxiv.org/pdf/2601.02362)  

**Abstract**: The rise of generative AI technologies is reshaping content-based recommender systems (RSes), which increasingly encounter AI-generated content alongside human-authored content. This study examines how the introduction of AI-generated reviews influences RS performance and business outcomes. We analyze two distinct pathways through which AI content can enter RSes: user-centric, in which individuals use AI tools to refine their reviews, and platform-centric, in which platforms generate synthetic reviews directly from structured metadata. Using a large-scale dataset of hotel reviews from TripAdvisor, we generate synthetic reviews using LLMs and evaluate their impact across the training and deployment phases of RSes. We find that AI-generated reviews differ systematically from human-authored reviews across multiple textual dimensions. Although both user- and platform-centric AI reviews enhance RS performance relative to models without textual data, models trained on human reviews consistently achieve superior performance, underscoring the quality of authentic human data. Human-trained models generalize robustly to AI content, whereas AI-trained models underperform on both content types. Furthermore, tone-based framing strategies (encouraging, constructive, or critical) substantially enhance platform-generated review effectiveness. Our findings highlight the strategic importance of platform control in governing the generation and integration of AI-generated reviews, ensuring that synthetic content complements recommendation robustness and sustainable business value. 

---
# TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs 

**Authors**: I-Fan Lin, Faegheh Hasibi, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2510.06747)  

**Abstract**: In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods. 

---
