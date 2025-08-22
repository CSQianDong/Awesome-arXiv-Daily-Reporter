# Language-Guided Tuning: Enhancing Numeric Optimization with Textual Feedback 

**Authors**: Yuxing Lu, Yucheng Hu, Nan Sun, Xukai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15757)  

**Abstract**: Configuration optimization remains a critical bottleneck in machine learning, requiring coordinated tuning across model architecture, training strategy, feature engineering, and hyperparameters. Traditional approaches treat these dimensions independently and lack interpretability, while recent automated methods struggle with dynamic adaptability and semantic reasoning about optimization decisions. We introduce Language-Guided Tuning (LGT), a novel framework that employs multi-agent Large Language Models to intelligently optimize configurations through natural language reasoning. We apply textual gradients - qualitative feedback signals that complement numerical optimization by providing semantic understanding of training dynamics and configuration interdependencies. LGT coordinates three specialized agents: an Advisor that proposes configuration changes, an Evaluator that assesses progress, and an Optimizer that refines the decision-making process, creating a self-improving feedback loop. Through comprehensive evaluation on six diverse datasets, LGT demonstrates substantial improvements over traditional optimization methods, achieving performance gains while maintaining high interpretability. 

---
# Response and Prompt Evaluation to Prevent Parasocial Relationships with Chatbots 

**Authors**: Emma Rath, Stuart Armstrong, Rebecca Gorman  

**Link**: [PDF](https://arxiv.org/pdf/2508.15748)  

**Abstract**: The development of parasocial relationships with AI agents has severe, and in some cases, tragic effects for human well-being. Yet preventing such dynamics is challenging: parasocial cues often emerge gradually in private conversations, and not all forms of emotional engagement are inherently harmful. We address this challenge by introducing a simple response evaluation framework, created by repurposing a state-of-the-art language model, that evaluates ongoing conversations for parasocial cues in real time. To test the feasibility of this approach, we constructed a small synthetic dataset of thirty dialogues spanning parasocial, sycophantic, and neutral conversations. Iterative evaluation with five stage testing successfully identified all parasocial conversations while avoiding false positives under a tolerant unanimity rule, with detection typically occurring within the first few exchanges. These findings provide preliminary evidence that evaluation agents can provide a viable solution for the prevention of parasocial relations. 

---
# Measuring the environmental impact of delivering AI at Google Scale 

**Authors**: Cooper Elsworth, Keguo Huang, David Patterson, Ian Schneider, Robert Sedivy, Savannah Goodman, Ben Townsend, Parthasarathy Ranganathan, Jeff Dean, Amin Vahdat, Ben Gomes, James Manyika  

**Link**: [PDF](https://arxiv.org/pdf/2508.15734)  

**Abstract**: The transformative power of AI is undeniable - but as user adoption accelerates, so does the need to understand and mitigate the environmental impact of AI serving. However, no studies have measured AI serving environmental metrics in a production environment. This paper addresses this gap by proposing and executing a comprehensive methodology for measuring the energy usage, carbon emissions, and water consumption of AI inference workloads in a large-scale, AI production environment. Our approach accounts for the full stack of AI serving infrastructure - including active AI accelerator power, host system energy, idle machine capacity, and data center energy overhead. Through detailed instrumentation of Google's AI infrastructure for serving the Gemini AI assistant, we find the median Gemini Apps text prompt consumes 0.24 Wh of energy - a figure substantially lower than many public estimates. We also show that Google's software efficiency efforts and clean energy procurement have driven a 33x reduction in energy consumption and a 44x reduction in carbon footprint for the median Gemini Apps text prompt over one year. We identify that the median Gemini Apps text prompt uses less energy than watching nine seconds of television (0.24 Wh) and consumes the equivalent of five drops of water (0.26 mL). While these impacts are low compared to other daily activities, reducing the environmental impact of AI serving continues to warrant important attention. Towards this objective, we propose that a comprehensive measurement of AI serving environmental metrics is critical for accurately comparing models, and to properly incentivize efficiency gains across the full AI serving stack. 

---
# NiceWebRL: a Python library for human subject experiments with reinforcement learning environments 

**Authors**: Wilka Carvalho, Vikram Goddla, Ishaan Sinha, Hoon Shin, Kunal Jha  

**Link**: [PDF](https://arxiv.org/pdf/2508.15693)  

**Abstract**: We present NiceWebRL, a research tool that enables researchers to use machine reinforcement learning (RL) environments for online human subject experiments. NiceWebRL is a Python library that allows any Jax-based environment to be transformed into an online interface, supporting both single-agent and multi-agent environments. As such, NiceWebRL enables AI researchers to compare their algorithms to human performance, cognitive scientists to test ML algorithms as theories for human cognition, and multi-agent researchers to develop algorithms for human-AI collaboration. We showcase NiceWebRL with 3 case studies that demonstrate its potential to help develop Human-like AI, Human-compatible AI, and Human-assistive AI. In the first case study (Human-like AI), NiceWebRL enables the development of a novel RL model of cognition. Here, NiceWebRL facilitates testing this model against human participants in both a grid world and Craftax, a 2D Minecraft domain. In our second case study (Human-compatible AI), NiceWebRL enables the development of a novel multi-agent RL algorithm that can generalize to human partners in the Overcooked domain. Finally, in our third case study (Human-assistive AI), we show how NiceWebRL can allow researchers to study how an LLM can assist humans on complex tasks in XLand-Minigrid, an environment with millions of hierarchical tasks. The library is available at this https URL. 

---
# GRAFT: GRaPH and Table Reasoning for Textual Alignment -- A Benchmark for Structured Instruction Following and Visual Reasoning 

**Authors**: Abhigya Verma, Sriram Puttagunta, Seganrasan Subramanian, Sravan Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2508.15690)  

**Abstract**: GRAFT is a structured multimodal benchmark for evaluating models on instruction-following, visual reasoning, and visual-textual alignment tasks. It features programmatically generated charts and synthetically rendered tables, created with Python visualization libraries to ensure control over data semantics, structure, and clarity. Each GRAFT instance pairs a chart or table image with a systematically generated, multi-step analytical question based solely on visual content. Answers are provided in structured formats such as JSON or YAML, supporting consistent evaluation of both reasoning and output format. The benchmark introduces a taxonomy of reasoning types including comparison, trend identification, ranking, aggregation, proportion estimation, and anomaly detection to enable comprehensive assessment. Reference answers follow strict factual and formatting guidelines for precise, aspect-based evaluation. GRAFT offers a unified, scalable framework for fine-grained benchmarking of multimodal models on visually grounded, structured reasoning tasks, setting a new evaluation standard in this field. 

---
# Futurity as Infrastructure: A Techno-Philosophical Interpretation of the AI Lifecycle 

**Authors**: Mark Cote, Susana Aires  

**Link**: [PDF](https://arxiv.org/pdf/2508.15680)  

**Abstract**: This paper argues that a techno-philosophical reading of the EU AI Act provides insight into the long-term dynamics of data in AI systems, specifically, how the lifecycle from ingestion to deployment generates recursive value chains that challenge existing frameworks for Responsible AI. We introduce a conceptual tool to frame the AI pipeline, spanning data, training regimes, architectures, feature stores, and transfer learning. Using cross-disciplinary methods, we develop a technically grounded and philosophically coherent analysis of regulatory blind spots. Our central claim is that what remains absent from policymaking is an account of the dynamic of becoming that underpins both the technical operation and economic logic of AI. To address this, we advance a formal reading of AI inspired by Simondonian philosophy of technology, reworking his concept of individuation to model the AI lifecycle, including the pre-individual milieu, individuation, and individuated AI. To translate these ideas, we introduce futurity: the self-reinforcing lifecycle of AI, where more data enhances performance, deepens personalisation, and expands application domains. Futurity highlights the recursively generative, non-rivalrous nature of data, underpinned by infrastructures like feature stores that enable feedback, adaptation, and temporal recursion. Our intervention foregrounds escalating power asymmetries, particularly the tech oligarchy whose infrastructures of capture, training, and deployment concentrate value and decision-making. We argue that effective regulation must address these infrastructural and temporal dynamics, and propose measures including lifecycle audits, temporal traceability, feedback accountability, recursion transparency, and a right to contest recursive reuse. 

---
# Understanding Action Effects through Instrumental Empowerment in Multi-Agent Reinforcement Learning 

**Authors**: Ardian Selmonaj, Miroslav Strupl, Oleg Szehr, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2508.15652)  

**Abstract**: To reliably deploy Multi-Agent Reinforcement Learning (MARL) systems, it is crucial to understand individual agent behaviors within a team. While prior work typically evaluates overall team performance based on explicit reward signals or learned value functions, it is unclear how to infer agent contributions in the absence of any value feedback. In this work, we investigate whether meaningful insights into agent behaviors can be extracted that are consistent with the underlying value functions, solely by analyzing the policy distribution. Inspired by the phenomenon that intelligent agents tend to pursue convergent instrumental values, which generally increase the likelihood of task success, we introduce Intended Cooperation Values (ICVs), a method based on information-theoretic Shapley values for quantifying each agent's causal influence on their co-players' instrumental empowerment. Specifically, ICVs measure an agent's action effect on its teammates' policies by assessing their decision uncertainty and preference alignment. The analysis across cooperative and competitive MARL environments reveals the extent to which agents adopt similar or diverse strategies. By comparing action effects between policies and value functions, our method identifies which agent behaviors are beneficial to team success, either by fostering deterministic decisions or by preserving flexibility for future action choices. Our proposed method offers novel insights into cooperation dynamics and enhances explainability in MARL systems. 

---
# Adapting A Vector-Symbolic Memory for Lisp ACT-R 

**Authors**: Meera Ray, Christopher L. Dancy  

**Link**: [PDF](https://arxiv.org/pdf/2508.15630)  

**Abstract**: Holographic Declarative Memory (HDM) is a vector-symbolic alternative to ACT-R's Declarative Memory (DM) system that can bring advantages such as scalability and architecturally defined similarity between DM chunks. We adapted HDM to work with the most comprehensive and widely-used implementation of ACT-R (Lisp ACT-R) so extant ACT-R models designed with DM can be run with HDM without major changes. With this adaptation of HDM, we have developed vector-based versions of common ACT-R functions, set up a text processing pipeline to add the contents of large documents to ACT-R memory, and most significantly created a useful and novel mechanism to retrieve an entire chunk of memory based on a request using only vector representations of tokens. Preliminary results indicate that we can maintain vector-symbolic advantages of HDM (e.g., chunk recall without storing the actual chunk and other advantages with scaling) while also extending it so that previous ACT-R models may work with the system with little (or potentially no) modifications within the actual procedural and declarative memory portions of a model. As a part of iterative improvement of this newly translated holographic declarative memory module, we will continue to explore better time-context representations for vectors to improve the module's ability to reconstruct chunks during recall. To more fully test this translated HDM module, we also plan to develop decision-making models that use instance-based learning (IBL) theory, which is a useful application of HDM given the advantages of the system. 

---
# Transduction is All You Need for Structured Data Workflows 

**Authors**: Alfio Gliozzo, Naweed Khan, Christodoulos Constantinides, Nandana Mihindukulasooriya, Nahuel Defosse, Junkyu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15610)  

**Abstract**: This paper introduces Agentics, a modular framework for building agent-based systems capable of structured reasoning and compositional generalization over complex data. Designed with research and practical applications in mind, Agentics offers a novel perspective on working with data and AI workflows. In this framework, agents are abstracted from the logical flow and they are used internally to the data type to enable logical transduction among data. Agentics encourages AI developers to focus on modeling data rather than crafting prompts, enabling a declarative language in which data types are provided by LLMs and composed through logical transduction, which is executed by LLMs when types are connected. We provide empirical evidence demonstrating the applicability of this framework across domain-specific multiple-choice question answering, semantic parsing for text-to-SQL, and automated prompt optimization tasks, achieving state-of-the-art accuracy or improved scalability without sacrificing performance. The open-source implementation is available at \texttt{this https URL}. 

---
# A Dynamical Systems Framework for Reinforcement Learning Safety and Robustness Verification 

**Authors**: Ahmed Nasir, Abdelhafid Zenati  

**Link**: [PDF](https://arxiv.org/pdf/2508.15588)  

**Abstract**: The application of reinforcement learning to safety-critical systems is limited by the lack of formal methods for verifying the robustness and safety of learned policies. This paper introduces a novel framework that addresses this gap by analyzing the combination of an RL agent and its environment as a discrete-time autonomous dynamical system. By leveraging tools from dynamical systems theory, specifically the Finite-Time Lyapunov Exponent (FTLE), we identify and visualize Lagrangian Coherent Structures (LCS) that act as the hidden "skeleton" governing the system's behavior. We demonstrate that repelling LCS function as safety barriers around unsafe regions, while attracting LCS reveal the system's convergence properties and potential failure modes, such as unintended "trap" states. To move beyond qualitative visualization, we introduce a suite of quantitative metrics, Mean Boundary Repulsion (MBR), Aggregated Spurious Attractor Strength (ASAS), and Temporally-Aware Spurious Attractor Strength (TASAS), to formally measure a policy's safety margin and robustness. We further provide a method for deriving local stability guarantees and extend the analysis to handle model uncertainty. Through experiments in both discrete and continuous control environments, we show that this framework provides a comprehensive and interpretable assessment of policy behavior, successfully identifying critical flaws in policies that appear successful based on reward alone. 

---
# DeepThink3D: Enhancing Large Language Models with Programmatic Reasoning in Complex 3D Situated Reasoning Tasks 

**Authors**: Jiayi Song, Rui Wan, Lipeng Ma, Weidong Yang, Qingyuan Zhou, Yixuan Li, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15548)  

**Abstract**: This work enhances the ability of large language models (LLMs) to perform complex reasoning in 3D scenes. Recent work has addressed the 3D situated reasoning task by invoking tool usage through large language models. Large language models call tools via APIs and integrate the generated programs through a chain of thought to solve problems based on the program results. However, due to the simplicity of the questions in the dataset, the generated program reasoning chains are relatively short. To solve this main challenge, in this paper, we introduce DeepThink3D to enhance the tool usage of LLMs in complex 3D situated reasoning tasks. Our work proposes a combinatorial and iterative evolutionary approach on the SQA3D benchmark to generate more complex questions. Building on this foundation, we fine-tune the large language model to make it more proficient in using 3D tools. By employing Direct Preference Optimization (DPO), we directly optimize the toolchain strategies generated by models, thereby enhancing their accuracy in complex tasks. 

---
# Super-additive Cooperation in Language Model Agents 

**Authors**: Filippo Tonini, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15510)  

**Abstract**: With the prospect of autonomous artificial intelligence (AI) agents, studying their tendency for cooperative behavior becomes an increasingly relevant topic. This study is inspired by the super-additive cooperation theory, where the combined effects of repeated interactions and inter-group rivalry have been argued to be the cause for cooperative tendencies found in humans. We devised a virtual tournament where language model agents, grouped into teams, face each other in a Prisoner's Dilemma game. By simulating both internal team dynamics and external competition, we discovered that this blend substantially boosts both overall and initial, one-shot cooperation levels (the tendency to cooperate in one-off interactions). This research provides a novel framework for large language models to strategize and act in complex social scenarios and offers evidence for how intergroup competition can, counter-intuitively, result in more cooperative behavior. These insights are crucial for designing future multi-agent AI systems that can effectively work together and better align with human values. Source code is available at this https URL. 

---
# Think in Blocks: Adaptive Reasoning from Direct Response to Deep Reasoning 

**Authors**: Yekun Zhu, Guang Chen, Chengjun Mao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15507)  

**Abstract**: Large Language Models (LLMs) with chains-of-thought have demonstrated strong performance on an increasing range of tasks, particularly those involving complex logical reasoning. However, excessively long chains can lead to overthinking, causing computational waste and slower responses. This raises a question: can LLMs dynamically adjust the length of their reasoning processes based on task complexity? To address this, we propose the Think in Blocks framework, which enables adaptive reasoning-from zero to deep reasoning-by partitioning the reasoning process into a tunable number of blocks. Our main contributions are: (1) Establishing an explicit block-structured paradigm in which the model first predicts an integer reasoning budget-the number of blocks-and then partitions its reasoning accordingly; (2) Training an adaptive model through a three-stage pipeline-Supervised Fine-Tuning, reward-guided Direct Preference Optimization, and Reinforcement Learning-that adjusts its reasoning depth to problem difficulty; (3) Exploiting the explicit block count to dynamically control reasoning depth at inference time, allowing flexible adjustment of chain-of-thought length during deployment. 

---
# From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence 

**Authors**: Zihao Wang, Junming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15447)  

**Abstract**: Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively. 

---
# GraSP: A Unified Graph-Based Framework for Scalable Generation, Quality Tagging, and Management of Synthetic Data for SFT and DPO 

**Authors**: Bidyapati Pradhan, Surajit Dasgupta, Amit Kumar Saha, Omkar Anustoop, Sriram Puttagunta, Vipul Mittal, Gopal Sarda  

**Link**: [PDF](https://arxiv.org/pdf/2508.15432)  

**Abstract**: The advancement of large language models (LLMs) is critically dependent on the availability of high-quality datasets for Supervised Fine-Tuning (SFT), alignment tasks like Direct Preference Optimization (DPO), etc. In this work, we present a comprehensive synthetic data generation framework that facilitates scalable, configurable, and high-fidelity generation of synthetic data tailored for these training paradigms. Our approach employs a modular and configuration-based pipeline capable of modeling complex dialogue flows with minimal manual intervention. This framework uses a dual-stage quality tagging mechanism, combining heuristic rules and LLM-based evaluations, to automatically filter and score data extracted from OASST-formatted conversations, ensuring the curation of high-quality dialogue samples. The resulting datasets are structured under a flexible schema supporting both SFT and DPO use cases, enabling seamless integration into diverse training workflows. Together, these innovations offer a robust solution for generating and managing synthetic conversational data at scale, significantly reducing the overhead of data preparation in LLM training pipelines. 

---
# Planning with Minimal Disruption 

**Authors**: Alberto Pozanco, Marianela Morales, Daniel Borrajo, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.15358)  

**Abstract**: In many planning applications, we might be interested in finding plans that minimally modify the initial state to achieve the goals. We refer to this concept as plan disruption. In this paper, we formally introduce it, and define various planning-based compilations that aim to jointly optimize both the sum of action costs and plan disruption. Experimental results in different benchmarks show that the reformulated task can be effectively solved in practice to generate plans that balance both objectives. 

---
# DiagECG: An LLM-Driven Framework for Diagnostic Reasoning via Discretized ECG Tokenization 

**Authors**: Jinning Yang, Wen Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15338)  

**Abstract**: Electrocardiography plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present DiagECG, a novel framework that integrates time-series and language modeling by enabling large language models to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into symbolic tokens using a lead-independent encoder and quantization module. These tokens are then used to extend the vocabulary of LLM, allowing the model to handle both ECG and natural language inputs in a unified manner. To bridge the modality gap, we pretrain the model on an autoregressive ECG forecasting task, enabling the LLM to model temporal dynamics using its native language modeling capabilities. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, DiagECG achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating symbolic ECG representations into LLMs for medical reasoning. 

---
# RETAIL: Towards Real-world Travel Planning for Large Language Models 

**Authors**: Bin Deng, Yizhe Feng, Zeming Liu, Qing Wei, Xiangrong Zhu, Shuai Chen, Yuanfang Guo, Yunhong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15335)  

**Abstract**: Although large language models have enhanced automated travel planning abilities, current systems remain misaligned with real-world scenarios. First, they assume users provide explicit queries, while in reality requirements are often implicit. Second, existing solutions ignore diverse environmental factors and user preferences, limiting the feasibility of plans. Third, systems can only generate plans with basic POI arrangements, failing to provide all-in-one plans with rich details. To mitigate these challenges, we construct a novel dataset \textbf{RETAIL}, which supports decision-making for implicit queries while covering explicit queries, both with and without revision needs. It also enables environmental awareness to ensure plan feasibility under real-world scenarios, while incorporating detailed POI information for all-in-one travel plans. Furthermore, we propose a topic-guided multi-agent framework, termed TGMA. Our experiments reveal that even the strongest existing model achieves merely a 1.0% pass rate, indicating real-world travel planning remains extremely challenging. In contrast, TGMA demonstrates substantially improved performance 2.72%, offering promising directions for real-world travel planning. 

---
# Search-Based Credit Assignment for Offline Preference-Based Reinforcement Learning 

**Authors**: Xiancheng Gao, Yufeng Shi, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15327)  

**Abstract**: Offline reinforcement learning refers to the process of learning policies from fixed datasets, without requiring additional environment interaction. However, it often relies on well-defined reward functions, which are difficult and expensive to design. Human feedback is an appealing alternative, but its two common forms, expert demonstrations and preferences, have complementary limitations. Demonstrations provide stepwise supervision, but they are costly to collect and often reflect limited expert behavior modes. In contrast, preferences are easier to collect, but it is unclear which parts of a behavior contribute most to a trajectory segment, leaving credit assignment unresolved. In this paper, we introduce a Search-Based Preference Weighting (SPW) scheme to unify these two feedback sources. For each transition in a preference labeled trajectory, SPW searches for the most similar state-action pairs from expert demonstrations and directly derives stepwise importance weights based on their similarity scores. These weights are then used to guide standard preference learning, enabling more accurate credit assignment that traditional approaches struggle to achieve. We demonstrate that SPW enables effective joint learning from preferences and demonstrations, outperforming prior methods that leverage both feedback types on challenging robot manipulation tasks. 

---
# Coarse-to-Fine Grounded Memory for LLM Agent Planning 

**Authors**: Wei Yang, Jinwei Xiao, Hongming Zhang, Qingyang Zhang, Yanna Wang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15305)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have driven growing interest in LLM-based agents for complex planning tasks. To avoid costly agent training, many studies adopted memory mechanism that enhances LLM with offline experiences or online trajectory analysis. However, existing works focus on single-granularity memory derived from dynamic environmental interactions, which are inherently constrained by the quality of the collected experiences. This limitation, in turn, constrain the diversity of knowledge and the flexibility of planning. We propose Coarse-to-Fine Grounded Memory (\Ours{}), a novel framework that grounds coarse-to-fine memories with LLM, thereby fully leverage them for flexible adaptation to diverse scenarios. \Ours{} grounds environmental information into coarse-grained focus points to guide experience collection in training tasks, followed by grounding of actionable hybrid-grained tips from each experience. At inference, \Ours{} retrieves task-relevant experiences and tips to support planning. When facing environmental anomalies, the LLM grounds the current situation into fine-grained key information, enabling flexible self-QA reflection and plan correction. 

---
# Multiple Memory Systems for Enhancing the Long-term Memory of Agent 

**Authors**: Gaoke Zhang, Bo Wang, Yunlong Ma, Dongming Zhao, Zifei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15294)  

**Abstract**: An agent powered by large language models have achieved impressive results, but effectively handling the vast amounts of historical data generated during interactions remains a challenge. The current approach is to design a memory module for the agent to process these data. However, existing methods, such as MemoryBank and A-MEM, have poor quality of stored memory content, which affects recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multiple memory system (MMS) inspired by cognitive psychology theory. The system processes short-term memory to multiple long-term memory fragments, and constructs retrieval memory units and contextual memory units based on these fragments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. Experiments on LoCoMo dataset compared our method with three others, proving its effectiveness. Ablation studies confirmed the rationality of our memory units. We also analyzed the robustness regarding the number of selected memory segments and the storage overhead, demonstrating its practical value. 

---
# Computational Intelligence based Land-use Allocation Approaches for Mixed Use Areas 

**Authors**: Sabab Aosaf, Muhammad Ali Nayeem, Afsana Haque, M Sohel Rahmana  

**Link**: [PDF](https://arxiv.org/pdf/2508.15240)  

**Abstract**: Urban land-use allocation represents a complex multi-objective optimization problem critical for sustainable urban development policy. This paper presents novel computational intelligence approaches for optimizing land-use allocation in mixed-use areas, addressing inherent trade-offs between land-use compatibility and economic objectives. We develop multiple optimization algorithms, including custom variants integrating differential evolution with multi-objective genetic algorithms. Key contributions include: (1) CR+DES algorithm leveraging scaled difference vectors for enhanced exploration, (2) systematic constraint relaxation strategy improving solution quality while maintaining feasibility, and (3) statistical validation using Kruskal-Wallis tests with compact letter displays. Applied to a real-world case study with 1,290 plots, CR+DES achieves 3.16\% improvement in land-use compatibility compared to state-of-the-art methods, while MSBX+MO excels in price optimization with 3.3\% improvement. Statistical analysis confirms algorithms incorporating difference vectors significantly outperform traditional approaches across multiple metrics. The constraint relaxation technique enables broader solution space exploration while maintaining practical constraints. These findings provide urban planners and policymakers with evidence-based computational tools for balancing competing objectives in land-use allocation, supporting more effective urban development policies in rapidly urbanizing regions. 

---
# See it. Say it. Sorted: Agentic System for Compositional Diagram Generation 

**Authors**: Hantao Zhang, Jingyang Liu, Ed Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.15222)  

**Abstract**: We study sketch-to-diagram generation: converting rough hand sketches into precise, compositional diagrams. Diffusion models excel at photorealism but struggle with the spatial precision, alignment, and symbolic structure required for flowcharts. We introduce See it. Say it. Sorted., a training-free agentic system that couples a Vision-Language Model (VLM) with Large Language Models (LLMs) to produce editable Scalable Vector Graphics (SVG) programs. The system runs an iterative loop in which a Critic VLM proposes a small set of qualitative, relational edits; multiple candidate LLMs synthesize SVG updates with diverse strategies (conservative->aggressive, alternative, focused); and a Judge VLM selects the best candidate, ensuring stable improvement. This design prioritizes qualitative reasoning over brittle numerical estimates, preserves global constraints (e.g., alignment, connectivity), and naturally supports human-in-the-loop corrections. On 10 sketches derived from flowcharts in published papers, our method more faithfully reconstructs layout and structure than two frontier closed-source image generation LLMs (GPT-5 and Gemini-2.5-Pro), accurately composing primitives (e.g., multi-headed arrows) without inserting unwanted text. Because outputs are programmatic SVGs, the approach is readily extensible to presentation tools (e.g., PowerPoint) via APIs and can be specialized with improved prompts and task-specific tools. The codebase is open-sourced at this https URL. 

---
# R-ConstraintBench: Evaluating LLMs on NP-Complete Scheduling 

**Authors**: Raj Jain, Marc Wetter  

**Link**: [PDF](https://arxiv.org/pdf/2508.15204)  

**Abstract**: Effective scheduling under tight resource, timing, and operational constraints underpins large-scale planning across sectors such as capital projects, manufacturing, logistics, and IT fleet transitions. However, the reliability of large language models (LLMs) when reasoning under high-constraint regimes is insufficiently characterized. To address this gap, we present R-ConstraintBench, a scalable framework that evaluates models on Resource-Constrained Project Scheduling Problems (RCPSP), an NP-Complete feasibility class, while difficulty increases via linear growth in constraints. R-ConstraintBench incrementally increases non-redundant precedence constraints in Directed Acyclic Graphs (DAGs) and then introduces downtime, temporal windows, and disjunctive constraints. As an illustrative example, we instantiate the benchmark in a data center migration setting and evaluate multiple LLMs using feasibility and error analysis, identifying degradation thresholds and constraint types most associated with failure. Empirically, strong models are near-ceiling on precedence-only DAGs, but feasibility performance collapses when downtime, temporal windows, and disjunctive constraints interact, implicating constraint interaction, not graph depth, as the principal bottleneck. Performance on clean synthetic ramps also does not guarantee transfer to domain-grounded scenarios, underscoring limited generalization. 

---
# LLM4Sweat: A Trustworthy Large Language Model for Hyperhidrosis Support 

**Authors**: Wenjie Lin, Jin Wei-Kocsis  

**Link**: [PDF](https://arxiv.org/pdf/2508.15192)  

**Abstract**: While large language models (LLMs) have shown promise in healthcare, their application for rare medical conditions is still hindered by scarce and unreliable datasets for fine-tuning. Hyperhidrosis, a disorder causing excessive sweating beyond physiological needs, is one such rare disorder, affecting 2-3% of the population and significantly impacting both physical comfort and psychosocial well-being. To date, no work has tailored LLMs to advance the diagnosis or care of hyperhidrosis. To address this gap, we present LLM4Sweat, an open-source and domain-specific LLM framework for trustworthy and empathetic hyperhidrosis support. The system follows a three-stage pipeline. In the data augmentation stage, a frontier LLM generates medically plausible synthetic vignettes from curated open-source data to create a diverse and balanced question-answer dataset. In the fine-tuning stage, an open-source foundation model is fine-tuned on the dataset to provide diagnosis, personalized treatment recommendations, and empathetic psychological support. In the inference and expert evaluation stage, clinical and psychological specialists assess accuracy, appropriateness, and empathy, with validated responses iteratively enriching the dataset. Experiments show that LLM4Sweat outperforms baselines and delivers the first open-source LLM framework for hyperhidrosis, offering a generalizable approach for other rare diseases with similar data and trustworthiness challenges. 

---
# PuzzleClone: An SMT-Powered Framework for Synthesizing Verifiable Data 

**Authors**: Kai Xiong, Yanwei Huang, Rongjunchen Zhang, Kun Chen, Haipang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15180)  

**Abstract**: High-quality mathematical and logical datasets with verifiable answers are essential for strengthening the reasoning capabilities of large language models (LLMs). While recent data augmentation techniques have facilitated the creation of large-scale benchmarks, existing LLM-generated datasets often suffer from limited reliability, diversity, and scalability. To address these challenges, we introduce PuzzleClone, a formal framework for synthesizing verifiable data at scale using Satisfiability Modulo Theories (SMT). Our approach features three key innovations: (1) encoding seed puzzles into structured logical specifications, (2) generating scalable variants through systematic variable and constraint randomization, and (3) ensuring validity via a reproduction mechanism. Applying PuzzleClone, we construct a curated benchmark comprising over 83K diverse and programmatically validated puzzles. The generated puzzles span a wide spectrum of difficulty and formats, posing significant challenges to current state-of-the-art models. We conduct post training (SFT and RL) on PuzzleClone datasets. Experimental results show that training on PuzzleClone yields substantial improvements not only on PuzzleClone testset but also on logic and mathematical benchmarks. Post training raises PuzzleClone average from 14.4 to 56.2 and delivers consistent improvements across 7 logic and mathematical benchmarks up to 12.5 absolute percentage points (AMC2023 from 52.5 to 65.0). Our code and data are available at this https URL. 

---
# Mobile-Agent-v3: Foundamental Agents for GUI Automation 

**Authors**: Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, Jitong Liao, Qi Zheng, Fei Huang, Jingren Zhou, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.15144)  

**Abstract**: This paper introduces GUI-Owl, a foundational GUI agent model that achieves state-of-the-art performance among open-source end-to-end models on ten GUI benchmarks across desktop and mobile environments, covering grounding, question answering, planning, decision-making, and procedural knowledge. GUI-Owl-7B achieves 66.4 on AndroidWorld and 29.4 on OSWorld. Building on this, we propose Mobile-Agent-v3, a general-purpose GUI agent framework that further improves performance to 73.3 on AndroidWorld and 37.7 on OSWorld, setting a new state-of-the-art for open-source GUI agent frameworks. GUI-Owl incorporates three key innovations: (1) Large-scale Environment Infrastructure: a cloud-based virtual environment spanning Android, Ubuntu, macOS, and Windows, enabling our Self-Evolving GUI Trajectory Production framework. This generates high-quality interaction data via automated query generation and correctness validation, leveraging GUI-Owl to refine trajectories iteratively, forming a self-improving loop. It supports diverse data pipelines and reduces manual annotation. (2) Diverse Foundational Agent Capabilities: by integrating UI grounding, planning, action semantics, and reasoning patterns, GUI-Owl supports end-to-end decision-making and can act as a modular component in multi-agent systems. (3) Scalable Environment RL: we develop a scalable reinforcement learning framework with fully asynchronous training for real-world alignment. We also introduce Trajectory-aware Relative Policy Optimization (TRPO) for online RL, achieving 34.9 on OSWorld. GUI-Owl and Mobile-Agent-v3 are open-sourced at this https URL. 

---
# aiXiv: A Next-Generation Open Access Ecosystem for Scientific Discovery Generated by AI Scientists 

**Authors**: Pengsong Zhang, Xiang Hu, Guowei Huang, Yang Qi, Heng Zhang, Xiuxu Li, Jiaxing Song, Jiabin Luo, Yijiang Li, Shuo Yin, Chengxiao Dai, Eric Hanchen Jiang, Xiaoyan Zhou, Zhenfei Yin, Boqin Yuan, Jing Dong, Guinan Su, Guanren Qiao, Haiming Tang, Anghong Du, Lili Pan, Zhenzhong Lan, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15126)  

**Abstract**: Recent advances in large language models (LLMs) have enabled AI agents to autonomously generate scientific proposals, conduct experiments, author papers, and perform peer reviews. Yet this flood of AI-generated research content collides with a fragmented and largely closed publication ecosystem. Traditional journals and conferences rely on human peer review, making them difficult to scale and often reluctant to accept AI-generated research content; existing preprint servers (e.g. arXiv) lack rigorous quality-control mechanisms. Consequently, a significant amount of high-quality AI-generated research lacks appropriate venues for dissemination, hindering its potential to advance scientific progress. To address these challenges, we introduce aiXiv, a next-generation open-access platform for human and AI scientists. Its multi-agent architecture allows research proposals and papers to be submitted, reviewed, and iteratively refined by both human and AI scientists. It also provides API and MCP interfaces that enable seamless integration of heterogeneous human and AI scientists, creating a scalable and extensible ecosystem for autonomous scientific discovery. Through extensive experiments, we demonstrate that aiXiv is a reliable and robust platform that significantly enhances the quality of AI-generated research proposals and papers after iterative revising and reviewing on aiXiv. Our work lays the groundwork for a next-generation open-access ecosystem for AI scientists, accelerating the publication and dissemination of high-quality AI-generated research content. Code is available at this https URL. Website is available at this https URL. 

---
# Open-Universe Assistance Games 

**Authors**: Rachel Ma, Jingyi Qu, Andreea Bobu, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2508.15119)  

**Abstract**: Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations. 

---
# Argumentation for Explainable Workforce Optimisation (with Appendix) 

**Authors**: Jennifer Leigh, Dimitrios Letsios, Alessandro Mella, Lucio Machetti, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.15118)  

**Abstract**: Workforce management is a complex problem optimising the makespan and travel distance required for a team of operators to complete a set of jobs, using a set of instruments. A crucial challenge in workforce management is accommodating changes at execution time so that explanations are provided to all stakeholders involved. Here, we show that, by understanding workforce management as abstract argumentation in an industrial application, we can accommodate change and obtain faithful explanations. We show, with a user study, that our tool and explanations lead to faster and more accurate problem solving than conventional solutions by hand. 

---
# S3LoRA: Safe Spectral Sharpness-Guided Pruning in Adaptation of Agent Planner 

**Authors**: Shuang Ao, Gopal Rumchurn  

**Link**: [PDF](https://arxiv.org/pdf/2508.15068)  

**Abstract**: Adapting Large Language Models (LLMs) using parameter-efficient fine-tuning (PEFT) techniques such as LoRA has enabled powerful capabilities in LLM-based agents. However, these adaptations can unintentionally compromise safety alignment, leading to unsafe or unstable behaviors, particularly in agent planning tasks. Existing safety-aware adaptation methods often require access to both base and instruction-tuned model checkpoints, which are frequently unavailable in practice, limiting their applicability. We propose S3LoRA (Safe Spectral Sharpness-Guided Pruning LoRA), a lightweight, data-free, and model-independent framework that mitigates safety risks in LoRA-adapted models by inspecting only the fine-tuned weight updates. We first introduce Magnitude-Aware Spherically Normalized SVD (MAS-SVD), which robustly analyzes the structural properties of LoRA updates while preserving global magnitude information. We then design the Spectral Sharpness Index (SSI), a sharpness-aware metric to detect layers with highly concentrated and potentially unsafe updates. These layers are pruned post-hoc to reduce risk without sacrificing task performance. Extensive experiments and ablation studies across agent planning and language generation tasks show that S3LoRA consistently improves safety metrics while maintaining or improving utility metrics and significantly reducing inference cost. These results establish S3LoRA as a practical and scalable solution for safely deploying LLM-based agents in real-world, resource-constrained, and safety-critical environments. 

---
# Demonstrating Onboard Inference for Earth Science Applications with Spectral Analysis Algorithms and Deep Learning 

**Authors**: Itai Zilberstein, Alberto Candela, Steve Chien, David Rijlaarsdam, Tom Hendrix, Leonie Buckley, Aubrey Dunne  

**Link**: [PDF](https://arxiv.org/pdf/2508.15053)  

**Abstract**: In partnership with Ubotica Technologies, the Jet Propulsion Laboratory is demonstrating state-of-the-art data analysis onboard CogniSAT-6/HAMMER (CS-6). CS-6 is a satellite with a visible and near infrared range hyperspectral instrument and neural network acceleration hardware. Performing data analysis at the edge (e.g. onboard) can enable new Earth science measurements and responses. We will demonstrate data analysis and inference onboard CS-6 for numerous applications using deep learning and spectral analysis algorithms. 

---
# Don't Think Twice! Over-Reasoning Impairs Confidence Calibration 

**Authors**: Romain Lacombe, Kerrie Wu, Eddie Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2508.15050)  

**Abstract**: Large Language Models deployed as question answering tools require robust calibration to avoid overconfidence. We systematically evaluate how reasoning capabilities and budget affect confidence assessment accuracy, using the ClimateX dataset (Lacombe et al., 2023) and expanding it to human and planetary health. Our key finding challenges the "test-time scaling" paradigm: while recent reasoning LLMs achieve 48.7% accuracy in assessing expert confidence, increasing reasoning budgets consistently impairs rather than improves calibration. Extended reasoning leads to systematic overconfidence that worsens with longer thinking budgets, producing diminishing and negative returns beyond modest computational investments. Conversely, search-augmented generation dramatically outperforms pure reasoning, achieving 89.3% accuracy by retrieving relevant evidence. Our results suggest that information access, rather than reasoning depth or inference budget, may be the critical bottleneck for improved confidence calibration of knowledge-intensive tasks. 

---
# Emergent Crowds Dynamics from Language-Driven Multi-Agent Interactions 

**Authors**: Yibo Liu, Liam Shatzel, Brandon Haworth, Teseo Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2508.15047)  

**Abstract**: Animating and simulating crowds using an agent-based approach is a well-established area where every agent in the crowd is individually controlled such that global human-like behaviour emerges. We observe that human navigation and movement in crowds are often influenced by complex social and environmental interactions, driven mainly by language and dialogue. However, most existing work does not consider these dimensions and leads to animations where agent-agent and agent-environment interactions are largely limited to steering and fixed higher-level goal extrapolation.
We propose a novel method that exploits large language models (LLMs) to control agents' movement. Our method has two main components: a dialogue system and language-driven navigation. We periodically query agent-centric LLMs conditioned on character personalities, roles, desires, and relationships to control the generation of inter-agent dialogue when necessitated by the spatial and social relationships with neighbouring agents. We then use the conversation and each agent's personality, emotional state, vision, and physical state to control the navigation and steering of each agent. Our model thus enables agents to make motion decisions based on both their perceptual inputs and the ongoing dialogue.
We validate our method in two complex scenarios that exemplify the interplay between social interactions, steering, and crowding. In these scenarios, we observe that grouping and ungrouping of agents automatically occur. Additionally, our experiments show that our method serves as an information-passing mechanism within the crowd. As a result, our framework produces more realistic crowd simulations, with emergent group behaviours arising naturally from any environmental setting. 

---
# Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism 

**Authors**: Ashmi Banerjee, Fitri Nur Aisyah, Adithi Satish, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.15030)  

**Abstract**: We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems. 

---
# Goals and the Structure of Experience 

**Authors**: Nadav Amir, Stas Tiomkin, Angela Langdon  

**Link**: [PDF](https://arxiv.org/pdf/2508.15013)  

**Abstract**: Purposeful behavior is a hallmark of natural and artificial intelligence. Its acquisition is often believed to rely on world models, comprising both descriptive (what is) and prescriptive (what is desirable) aspects that identify and evaluate state of affairs in the world, respectively. Canonical computational accounts of purposeful behavior, such as reinforcement learning, posit distinct components of a world model comprising a state representation (descriptive aspect) and a reward function (prescriptive aspect). However, an alternative possibility, which has not yet been computationally formulated, is that these two aspects instead co-emerge interdependently from an agent's goal. Here, we describe a computational framework of goal-directed state representation in cognitive agents, in which the descriptive and prescriptive aspects of a world model co-emerge from agent-environment interaction sequences, or experiences. Drawing on Buddhist epistemology, we introduce a construct of goal-directed, or telic, states, defined as classes of goal-equivalent experience distributions. Telic states provide a parsimonious account of goal-directed learning in terms of the statistical divergence between behavioral policies and desirable experience features. We review empirical and theoretical literature supporting this novel perspective and discuss its potential to provide a unified account of behavioral, phenomenological and neural dimensions of purposeful behaviors across diverse substrates. 

---
# A Fully Spectral Neuro-Symbolic Reasoning Architecture with Graph Signal Processing as the Computational Backbone 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2508.14923)  

**Abstract**: We propose a fully spectral, neuro\-symbolic reasoning architecture that leverages Graph Signal Processing (GSP) as the primary computational backbone for integrating symbolic logic and neural inference. Unlike conventional reasoning models that treat spectral graph methods as peripheral components, our approach formulates the entire reasoning pipeline in the graph spectral domain. Logical entities and relationships are encoded as graph signals, processed via learnable spectral filters that control multi-scale information propagation, and mapped into symbolic predicates for rule-based inference. We present a complete mathematical framework for spectral reasoning, including graph Fourier transforms, band-selective attention, and spectral rule grounding. Experiments on benchmark reasoning datasets (ProofWriter, EntailmentBank, bAbI, CLUTRR, and ARC-Challenge) demonstrate improvements in logical consistency, interpretability, and computational efficiency over state\-of\-the\-art neuro\-symbolic models. Our results suggest that GSP provides a mathematically grounded and computationally efficient substrate for robust and interpretable reasoning systems. 

---
# SceneGen: Single-Image 3D Scene Generation in One Feedforward Pass 

**Authors**: Yanxu Meng, Haoning Wu, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15769)  

**Abstract**: 3D content generation has recently attracted significant research interest due to its applications in VR/AR and embodied AI. In this work, we address the challenging task of synthesizing multiple 3D assets within a single scene image. Concretely, our contributions are fourfold: (i) we present SceneGen, a novel framework that takes a scene image and corresponding object masks as input, simultaneously producing multiple 3D assets with geometry and texture. Notably, SceneGen operates with no need for optimization or asset retrieval; (ii) we introduce a novel feature aggregation module that integrates local and global scene information from visual and geometric encoders within the feature extraction module. Coupled with a position head, this enables the generation of 3D assets and their relative spatial positions in a single feedforward pass; (iii) we demonstrate SceneGen's direct extensibility to multi-image input scenarios. Despite being trained solely on single-image inputs, our architectural design enables improved generation performance with multi-image inputs; and (iv) extensive quantitative and qualitative evaluations confirm the efficiency and robust generation abilities of our approach. We believe this paradigm offers a novel solution for high-quality 3D content generation, potentially advancing its practical applications in downstream tasks. The code and model will be publicly available at: this https URL. 

---
# Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO 

**Authors**: Jaeha Lee, Gio Huh, Ning Su, Tony Yue YU  

**Link**: [PDF](https://arxiv.org/pdf/2508.15766)  

**Abstract**: Recent efforts have extended the capabilities of transformers in logical reasoning and symbolic computations. In this work, we investigate their capacity for non-linear latent pattern discovery in the context of functional decomposition, focusing on the challenging algebraic task of multivariate polynomial decomposition. This problem, with widespread applications in science and engineering, is proved to be NP-hard, and demands both precision and insight. Our contributions are threefold: First, we develop a synthetic data generation pipeline providing fine-grained control over problem complexity. Second, we train transformer models via supervised learning and evaluate them across four key dimensions involving scaling behavior and generalizability. Third, we propose Beam Grouped Relative Policy Optimization (BGRPO), a rank-aware reinforcement learning method suitable for hard algebraic problems. Finetuning with BGRPO improves accuracy while reducing beam width by up to half, resulting in approximately 75% lower inference compute. Additionally, our model demonstrates competitive performance in polynomial simplification, outperforming Mathematica in various cases. 

---
# LiveMCP-101: Stress Testing and Diagnosing MCP-enabled Agents on Challenging Queries 

**Authors**: Ming Yin, Dinghan Shen, Silei Xu, Jianbing Han, Sixun Dong, Mian Zhang, Yebowen Hu, Shujian Liu, Simin Ma, Song Wang, Sathish Reddy Indurthi, Xun Wang, Yiran Chen, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.15760)  

**Abstract**: Tool calling has emerged as a critical capability for AI agents to interact with the real world and solve complex tasks. While the Model Context Protocol (MCP) provides a powerful standardized framework for tool integration, there is a significant gap in benchmarking how well AI agents can effectively solve multi-step tasks using diverse MCP tools in realistic, dynamic scenarios. In this work, we present LiveMCP-101, a benchmark of 101 carefully curated real-world queries, refined through iterative LLM rewriting and manual review, that require coordinated use of multiple MCP tools including web search, file operations, mathematical reasoning, and data analysis. Moreover, we introduce a novel evaluation approach that leverages ground-truth execution plans rather than raw API outputs, better reflecting the evolving nature of real-world environments. Experiments show that even frontier LLMs achieve a success rate below 60\%, highlighting major challenges in tool orchestration. Detailed ablations and error analysis further reveal distinct failure modes and inefficiencies in token usage, pointing to concrete directions for advancing current models. LiveMCP-101 sets a rigorous standard for evaluating real-world agent capabilities, advancing toward autonomous AI systems that reliably execute complex tasks through tool use. 

---
# Neural Robot Dynamics 

**Authors**: Jie Xu, Eric Heiden, Iretiayo Akinola, Dieter Fox, Miles Macklin, Yashraj Narang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15755)  

**Abstract**: Accurate and efficient simulation of modern robots remains challenging due to their high degrees of freedom and intricate mechanisms. Neural simulators have emerged as a promising alternative to traditional analytical simulators, capable of efficiently predicting complex dynamics and adapting to real-world data; however, existing neural simulators typically require application-specific training and fail to generalize to novel tasks and/or environments, primarily due to inadequate representations of the global state. In this work, we address the problem of learning generalizable neural simulators for robots that are structured as articulated rigid bodies. We propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. We integrate the learned NeRD models as an interchangeable backend solver within a state-of-the-art robotics simulator. We conduct extensive experiments to show that the NeRD simulators are stable and accurate over a thousand simulation steps; generalize across tasks and environment configurations; enable policy learning exclusively in a neural engine; and, unlike most classical simulators, can be fine-tuned from real-world data to bridge the gap between simulation and reality. 

---
# Dissecting Tool-Integrated Reasoning: An Empirical Study and Analysis 

**Authors**: Yufeng Zhao, Junnan Liu, Hongwei Liu, Dongsheng Zhu, Yuan Shen, Songyang Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15754)  

**Abstract**: Large Language Models (LLMs) have made significant strides in reasoning tasks through methods like chain-of-thought (CoT) reasoning. However, they often fall short in tasks requiring precise computations. Tool-Integrated Reasoning (TIR) has emerged as a solution by incorporating external tools into the reasoning process. Nevertheless, the generalization of TIR in improving the reasoning ability of LLM is still unclear. Additionally, whether TIR has improved the model's reasoning behavior and helped the model think remains to be studied. We introduce ReasonZoo, a comprehensive benchmark encompassing nine diverse reasoning categories, to evaluate the effectiveness of TIR across various domains. Additionally, we propose two novel metrics, Performance-Aware Cost (PAC) and Area Under the Performance-Cost Curve (AUC-PCC), to assess reasoning efficiency. Our empirical evaluation demonstrates that TIR-enabled models consistently outperform their non-TIR counterparts in both mathematical and non-mathematical tasks. Furthermore, TIR enhances reasoning efficiency, as evidenced by improved PAC and AUC-PCC, indicating reduced overthinking and more streamlined reasoning. These findings underscore the domain-general benefits of TIR and its potential to advance LLM capabilities in complex reasoning tasks. 

---
# "Does the cafe entrance look accessible? Where is the door?" Towards Geospatial AI Agents for Visual Inquiries 

**Authors**: Jon E. Froehlich, Jared Hwang, Zeyu Wang, John S. O'Meara, Xia Su, William Huang, Yang Zhang, Alex Fiannaca, Philip Nelson, Shaun Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.15752)  

**Abstract**: Interactive digital maps have revolutionized how people travel and learn about the world; however, they rely on pre-existing structured data in GIS databases (e.g., road networks, POI indices), limiting their ability to address geo-visual questions related to what the world looks like. We introduce our vision for Geo-Visual Agents--multimodal AI agents capable of understanding and responding to nuanced visual-spatial inquiries about the world by analyzing large-scale repositories of geospatial images, including streetscapes (e.g., Google Street View), place-based photos (e.g., TripAdvisor, Yelp), and aerial imagery (e.g., satellite photos) combined with traditional GIS data sources. We define our vision, describe sensing and interaction approaches, provide three exemplars, and enumerate key challenges and opportunities for future work. 

---
# End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning 

**Authors**: Qiaoyu Zheng, Yuze Sun, Chaoyi Wu, Weike Zhao, Pengcheng Qiu, Yongguo Yu, Kun Sun, Yanfeng Wang, Ya Zhang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.15746)  

**Abstract**: Accurate diagnosis with medical large language models is hindered by knowledge gaps and hallucinations. Retrieval and tool-augmented methods help, but their impact is limited by weak use of external knowledge and poor feedback-reasoning traceability. To address these challenges, We introduce Deep-DxSearch, an agentic RAG system trained end-to-end with reinforcement learning (RL) that enables steer tracebale retrieval-augmented reasoning for medical diagnosis. In Deep-DxSearch, we first construct a large-scale medical retrieval corpus comprising patient records and reliable medical knowledge sources to support retrieval-aware reasoning across diagnostic scenarios. More crutially, we frame the LLM as the core agent and the retrieval corpus as its environment, using tailored rewards on format, retrieval, reasoning structure, and diagnostic accuracy, thereby evolving the agentic RAG policy from large-scale data through RL.
Experiments demonstrate that our end-to-end agentic RL training framework consistently outperforms prompt-engineering and training-free RAG approaches across multiple data centers. After training, Deep-DxSearch achieves substantial gains in diagnostic accuracy, surpassing strong diagnostic baselines such as GPT-4o, DeepSeek-R1, and other medical-specific frameworks for both common and rare disease diagnosis under in-distribution and out-of-distribution settings. Moreover, ablation studies on reward design and retrieval corpus components confirm their critical roles, underscoring the uniqueness and effectiveness of our approach compared with traditional implementations. Finally, case studies and interpretability analyses highlight improvements in Deep-DxSearch's diagnostic policy, providing deeper insight into its performance gains and supporting clinicians in delivering more reliable and precise preliminary diagnoses. See this https URL. 

---
# Numerical models outperform AI weather forecasts of record-breaking extremes 

**Authors**: Zhongwei Zhang, Erich Fischer, Jakob Zscheischler, Sebastian Engelke  

**Link**: [PDF](https://arxiv.org/pdf/2508.15724)  

**Abstract**: Artificial intelligence (AI)-based models are revolutionizing weather forecasting and have surpassed leading numerical weather prediction systems on various benchmark tasks. However, their ability to extrapolate and reliably forecast unprecedented extreme events remains unclear. Here, we show that for record-breaking weather extremes, the numerical model High RESolution forecast (HRES) from the European Centre for Medium-Range Weather Forecasts still consistently outperforms state-of-the-art AI models GraphCast, GraphCast operational, Pangu-Weather, Pangu-Weather operational, and Fuxi. We demonstrate that forecast errors in AI models are consistently larger for record-breaking heat, cold, and wind than in HRES across nearly all lead times. We further find that the examined AI models tend to underestimate both the frequency and intensity of record-breaking events, and they underpredict hot records and overestimate cold records with growing errors for larger record exceedance. Our findings underscore the current limitations of AI weather models in extrapolating beyond their training domain and in forecasting the potentially most impactful record-breaking weather events that are particularly frequent in a rapidly warming climate. Further rigorous verification and model development is needed before these models can be solely relied upon for high-stakes applications such as early warning systems and disaster management. 

---
# EcomMMMU: Strategic Utilization of Visuals for Robust Multimodal E-Commerce Models 

**Authors**: Xinyi Ling, Hanwen Du, Zhihui Zhu, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15721)  

**Abstract**: E-commerce platforms are rich in multimodal data, featuring a variety of images that depict product details. However, this raises an important question: do these images always enhance product understanding, or can they sometimes introduce redundancy or degrade performance? Existing datasets are limited in both scale and design, making it difficult to systematically examine this question. To this end, we introduce EcomMMMU, an e-commerce multimodal multitask understanding dataset with 406,190 samples and 8,989,510 images. EcomMMMU is comprised of multi-image visual-language data designed with 8 essential tasks and a specialized VSS subset to benchmark the capability of multimodal large language models (MLLMs) to effectively utilize visual content. Analysis on EcomMMMU reveals that product images do not consistently improve performance and can, in some cases, degrade it. This indicates that MLLMs may struggle to effectively leverage rich visual content for e-commerce tasks. Building on these insights, we propose SUMEI, a data-driven method that strategically utilizes multiple images via predicting visual utilities before using them for downstream tasks. Comprehensive experiments demonstrate the effectiveness and robustness of SUMEI. The data and code are available through this https URL. 

---
# Tutorial on the Probabilistic Unification of Estimation Theory, Machine Learning, and Generative AI 

**Authors**: Mohammed Elmusrati  

**Link**: [PDF](https://arxiv.org/pdf/2508.15719)  

**Abstract**: Extracting meaning from uncertain, noisy data is a fundamental problem across time series analysis, pattern recognition, and language modeling. This survey presents a unified mathematical framework that connects classical estimation theory, statistical inference, and modern machine learning, including deep learning and large language models. By analyzing how techniques such as maximum likelihood estimation, Bayesian inference, and attention mechanisms address uncertainty, the paper illustrates that many AI methods are rooted in shared probabilistic principles. Through illustrative scenarios including system identification, image classification, and language generation, we show how increasingly complex models build upon these foundations to tackle practical challenges like overfitting, data sparsity, and interpretability. In other words, the work demonstrates that maximum likelihood, MAP estimation, Bayesian classification, and deep learning all represent different facets of a shared goal: inferring hidden causes from noisy and/or biased observations. It serves as both a theoretical synthesis and a practical guide for students and researchers navigating the evolving landscape of machine learning. 

---
# StreamMem: Query-Agnostic KV Cache Memory for Streaming Video Understanding 

**Authors**: Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.15717)  

**Abstract**: Multimodal large language models (MLLMs) have made significant progress in visual-language reasoning, but their ability to efficiently handle long videos remains limited. Despite recent advances in long-context MLLMs, storing and attending to the key-value (KV) cache for long visual contexts incurs substantial memory and computational overhead. Existing visual compression methods require either encoding the entire visual context before compression or having access to the questions in advance, which is impractical for long video understanding and multi-turn conversational settings. In this work, we propose StreamMem, a query-agnostic KV cache memory mechanism for streaming video understanding. Specifically, StreamMem encodes new video frames in a streaming manner, compressing the KV cache using attention scores between visual tokens and generic query tokens, while maintaining a fixed-size KV memory to enable efficient question answering (QA) in memory-constrained, long-video scenarios. Evaluation on three long video understanding and two streaming video question answering benchmarks shows that StreamMem achieves state-of-the-art performance in query-agnostic KV cache compression and is competitive with query-aware compression approaches. 

---
# Foundation Models for Cross-Domain EEG Analysis Application: A Survey 

**Authors**: Hongqi Li, Yitong Chen, Yujuan Wang, Weihang Ni, Haodong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15716)  

**Abstract**: Electroencephalography (EEG) analysis stands at the forefront of neuroscience and artificial intelligence research, where foundation models are reshaping the traditional EEG analysis paradigm by leveraging their powerful representational capacity and cross-modal generalization. However, the rapid proliferation of these techniques has led to a fragmented research landscape, characterized by diverse model roles, inconsistent architectures, and a lack of systematic categorization. To bridge this gap, this study presents the first comprehensive modality-oriented taxonomy for foundation models in EEG analysis, systematically organizing research advances based on output modalities of the native EEG decoding, EEG-text, EEG-vision, EEG-audio, and broader multimodal frameworks. We rigorously analyze each category's research ideas, theoretical foundations, and architectural innovations, while highlighting open challenges such as model interpretability, cross-domain generalization, and real-world applicability in EEG-based systems. By unifying this dispersed field, our work not only provides a reference framework for future methodology development but accelerates the translation of EEG foundation models into scalable, interpretable, and online actionable solutions. 

---
# Row-Column Hybrid Grouping for Fault-Resilient Multi-Bit Weight Representation on IMC Arrays 

**Authors**: Kang Eun Jeon, Sangheum Yeon, Jinhee Kim, Hyeonsu Bang, Johnny Rhe, Jong Hwan Ko  

**Link**: [PDF](https://arxiv.org/pdf/2508.15685)  

**Abstract**: This paper addresses two critical challenges in analog In-Memory Computing (IMC) systems that limit their scalability and deployability: the computational unreliability caused by stuck-at faults (SAFs) and the high compilation overhead of existing fault-mitigation algorithms, namely Fault-Free (FF). To overcome these limitations, we first propose a novel multi-bit weight representation technique, termed row-column hybrid grouping, which generalizes conventional column grouping by introducing redundancy across both rows and columns. This structural redundancy enhances fault tolerance and can be effectively combined with existing fault-mitigation solutions. Second, we design a compiler pipeline that reformulates the fault-aware weight decomposition problem as an Integer Linear Programming (ILP) task, enabling fast and scalable compilation through off-the-shelf solvers. Further acceleration is achieved through theoretical insights that identify fault patterns amenable to trivial solutions, significantly reducing computation. Experimental results on convolutional networks and small language models demonstrate the effectiveness of our approach, achieving up to 8%p improvement in accuracy, 150x faster compilation, and 2x energy efficiency gain compared to existing baselines. 

---
# Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation 

**Authors**: Nikita Kachaev, Andrei Spiridonov, Andrey Gorodetsky, Kirill Muravyev, Nikita Oskolkov, Aditya Narendra, Vlad Shakhuro, Dmitry Makarov, Aleksandr I. Panov, Polina Fedotova, Alexey K. Kovalev  

**Link**: [PDF](https://arxiv.org/pdf/2508.15663)  

**Abstract**: Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents. 

---
# Benchmarking Computer Science Survey Generation 

**Authors**: Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Jiaxin Mao, Ziyi Ye, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15658)  

**Abstract**: Scientific survey articles play a vital role in summarizing research progress, yet their manual creation is becoming increasingly infeasible due to the rapid growth of academic literature. While large language models (LLMs) offer promising capabilities for automating this process, progress in this area is hindered by the absence of standardized benchmarks and evaluation protocols. To address this gap, we introduce SurGE (Survey Generation Evaluation), a new benchmark for evaluating scientific survey generation in the computer science domain. SurGE consists of (1) a collection of test instances, each including a topic description, an expert-written survey, and its full set of cited references, and (2) a large-scale academic corpus of over one million papers that serves as the retrieval pool. In addition, we propose an automated evaluation framework that measures generated surveys across four dimensions: information coverage, referencing accuracy, structural organization, and content quality. Our evaluation of diverse LLM-based approaches shows that survey generation remains highly challenging, even for advanced self-reflection frameworks. These findings highlight the complexity of the task and the necessity for continued research. We have open-sourced all the code, data, and models at: this https URL 

---
# Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance 

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15650)  

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin. 

---
# Label Uncertainty for Ultrasound Segmentation 

**Authors**: Malini Shivaram, Gautam Rajendrakumar Gare, Laura Hutchins, Jacob Duplantis, Thomas Deiss, Thales Nogueira Gomes, Thong Tran, Keyur H. Patel, Thomas H Fox, Amita Krishnan, Deva Ramanan, Bennett DeBoisblanc, Ricardo Rodriguez, John Galeotti  

**Link**: [PDF](https://arxiv.org/pdf/2508.15635)  

**Abstract**: In medical imaging, inter-observer variability among radiologists often introduces label uncertainty, particularly in modalities where visual interpretation is subjective. Lung ultrasound (LUS) is a prime example-it frequently presents a mixture of highly ambiguous regions and clearly discernible structures, making consistent annotation challenging even for experienced clinicians. In this work, we introduce a novel approach to both labeling and training AI models using expert-supplied, per-pixel confidence values. Rather than treating annotations as absolute ground truth, we design a data annotation protocol that captures the confidence that radiologists have in each labeled region, modeling the inherent aleatoric uncertainty present in real-world clinical data. We demonstrate that incorporating these confidence values during training leads to improved segmentation performance. More importantly, we show that this enhanced segmentation quality translates into better performance on downstream clinically-critical tasks-specifically, estimating S/F oxygenation ratio values, classifying S/F ratio change, and predicting 30-day patient readmission. While we empirically evaluate many methods for exposing the uncertainty to the learning model, we find that a simple approach that trains a model on binarized labels obtained with a (60%) confidence threshold works well. Importantly, high thresholds work far better than a naive approach of a 50% threshold, indicating that training on very confident pixels is far more effective. Our study systematically investigates the impact of training with varying confidence thresholds, comparing not only segmentation metrics but also downstream clinical outcomes. These results suggest that label confidence is a valuable signal that, when properly leveraged, can significantly enhance the reliability and clinical utility of AI in medical imaging. 

---
# GRASPED: Graph Anomaly Detection using Autoencoder with Spectral Encoder and Decoder (Full Version) 

**Authors**: Wei Herng Choong, Jixing Liu, Ching-Yu Kao, Philip Sperl  

**Link**: [PDF](https://arxiv.org/pdf/2508.15633)  

**Abstract**: Graph machine learning has been widely explored in various domains, such as community detection, transaction analysis, and recommendation systems. In these applications, anomaly detection plays an important role. Recently, studies have shown that anomalies on graphs induce spectral shifts. Some supervised methods have improved the utilization of such spectral domain information. However, they remain limited by the scarcity of labeled data due to the nature of anomalies. On the other hand, existing unsupervised learning approaches predominantly rely on spatial information or only employ low-pass filters, thereby losing the capacity for multi-band analysis. In this paper, we propose Graph Autoencoder with Spectral Encoder and Spectral Decoder (GRASPED) for node anomaly detection. Our unsupervised learning model features an encoder based on Graph Wavelet Convolution, along with structural and attribute decoders. The Graph Wavelet Convolution-based encoder, combined with a Wiener Graph Deconvolution-based decoder, exhibits bandpass filter characteristics that capture global and local graph information at multiple scales. This design allows for a learning-based reconstruction of node attributes, effectively capturing anomaly information. Extensive experiments on several real-world graph anomaly detection datasets demonstrate that GRASPED outperforms current state-of-the-art models. 

---
# Trained Miniatures: Low cost, High Efficacy SLMs for Sales & Marketing 

**Authors**: Ishaan Bhola, Mukunda NS, Sravanth Kurmala, Harsh Nandwani, Arihant Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.15617)  

**Abstract**: Large language models (LLMs) excel in text generation; however, these creative elements require heavy computation and are accompanied by a steep cost. Especially for targeted applications such as sales and marketing outreach, these costs are far from feasible. This paper introduces the concept of "Trained Miniatures" - Small Language Models(SLMs) fine-tuned for specific, high-value applications, generating similar domain-specific responses for a fraction of the cost. 

---
# Are Virtual DES Images a Valid Alternative to the Real Ones? 

**Authors**: Ana C. Perre, Luís A. Alexandre, Luís C. Freire  

**Link**: [PDF](https://arxiv.org/pdf/2508.15594)  

**Abstract**: Contrast-enhanced spectral mammography (CESM) is an imaging modality that provides two types of images, commonly known as low-energy (LE) and dual-energy subtracted (DES) images. In many domains, particularly in medicine, the emergence of image-to-image translation techniques has enabled the artificial generation of images using other images as input. Within CESM, applying such techniques to generate DES images from LE images could be highly beneficial, potentially reducing patient exposure to radiation associated with high-energy image acquisition. In this study, we investigated three models for the artificial generation of DES images (virtual DES): a pre-trained U-Net model, a U-Net trained end-to-end model, and a CycleGAN model. We also performed a series of experiments to assess the impact of using virtual DES images on the classification of CESM examinations into malignant and non-malignant categories. To our knowledge, this is the first study to evaluate the impact of virtual DES images on CESM lesion classification. The results demonstrate that the best performance was achieved with the pre-trained U-Net model, yielding an F1 score of 85.59% when using the virtual DES images, compared to 90.35% with the real DES images. This discrepancy likely results from the additional diagnostic information in real DES images, which contributes to a higher classification accuracy. Nevertheless, the potential for virtual DES image generation is considerable and future advancements may narrow this performance gap to a level where exclusive reliance on virtual DES images becomes clinically viable. 

---
# LoUQAL: Low-fidelity informed Uncertainty Quantification for Active Learning in the chemical configuration space 

**Authors**: Vivin Vinod, Peter Zaspel  

**Link**: [PDF](https://arxiv.org/pdf/2508.15577)  

**Abstract**: Uncertainty quantification is an important scheme in active learning techniques, including applications in predicting quantum chemical properties. In quantum chemical calculations, there exists the notion of a fidelity, a less accurate computation is accessible at a cheaper computational cost. This work proposes a novel low-fidelity informed uncertainty quantification for active learning with applications in predicting diverse quantum chemical properties such as excitation energies and \textit{ab initio} potential energy surfaces. Computational experiments are carried out in order to assess the proposed method with results demonstrating that models trained with the novel method outperform alternatives in terms of empirical error and number of iterations required. The effect of the choice of fidelity is also studied to perform a thorough benchmark. 

---
# LLM-Driven Self-Refinement for Embodied Drone Task Planning 

**Authors**: Deyu Zhang, Xicheng Zhang, Jiahao Li, Tingting Long, Xunhua Dai, Yongjian Fu, Jinrui Zhang, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15501)  

**Abstract**: We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at this https URL. 

---
# LGMSNet: Thinning a medical image segmentation model via dual-level multiscale fusion 

**Authors**: Chengqi Dong, Fenghe Tang, Rongge Mao, Xinpei Gao, S.Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15476)  

**Abstract**: Medical image segmentation plays a pivotal role in disease diagnosis and treatment planning, particularly in resource-constrained clinical settings where lightweight and generalizable models are urgently needed. However, existing lightweight models often compromise performance for efficiency and rarely adopt computationally expensive attention mechanisms, severely restricting their global contextual perception capabilities. Additionally, current architectures neglect the channel redundancy issue under the same convolutional kernels in medical imaging, which hinders effective feature extraction. To address these challenges, we propose LGMSNet, a novel lightweight framework based on local and global dual multiscale that achieves state-of-the-art performance with minimal computational overhead. LGMSNet employs heterogeneous intra-layer kernels to extract local high-frequency information while mitigating channel redundancy. In addition, the model integrates sparse transformer-convolutional hybrid branches to capture low-frequency global information. Extensive experiments across six public datasets demonstrate LGMSNet's superiority over existing state-of-the-art methods. In particular, LGMSNet maintains exceptional performance in zero-shot generalization tests on four unseen datasets, underscoring its potential for real-world deployment in resource-limited medical scenarios. The whole project code is in this https URL. 

---
# Subjective Behaviors and Preferences in LLM: Language of Browsing 

**Authors**: Sai Sundaresan, Harshita Chopra, Atanu R. Sinha, Koustava Goswami, Nagasai Saketh Naidu, Raghav Karan, N Anushka  

**Link**: [PDF](https://arxiv.org/pdf/2508.15474)  

**Abstract**: A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment. 

---
# RadReason: Radiology Report Evaluation Metric with Reasons and Sub-Scores 

**Authors**: Yingshu Li, Yunyi Liu, Lingqiao Liu, Lei Wang, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.15464)  

**Abstract**: Evaluating automatically generated radiology reports remains a fundamental challenge due to the lack of clinically grounded, interpretable, and fine-grained metrics. Existing methods either produce coarse overall scores or rely on opaque black-box models, limiting their usefulness in real-world clinical workflows. We introduce RadReason, a novel evaluation framework for radiology reports that not only outputs fine-grained sub-scores across six clinically defined error types, but also produces human-readable justifications that explain the rationale behind each score. Our method builds on Group Relative Policy Optimization and incorporates two key innovations: (1) Sub-score Dynamic Weighting, which adaptively prioritizes clinically challenging error types based on live F1 statistics; and (2) Majority-Guided Advantage Scaling, which adjusts policy gradient updates based on prompt difficulty derived from sub-score agreement. Together, these components enable more stable optimization and better alignment with expert clinical judgment. Experiments on the ReXVal benchmark show that RadReason surpasses all prior offline metrics and achieves parity with GPT-4-based evaluations, while remaining explainable, cost-efficient, and suitable for clinical deployment. Code will be released upon publication. 

---
# A Solvable Molecular Switch Model for Stable Temporal Information Processing 

**Authors**: H. I. Nurdin, C. A. Nijhuis  

**Link**: [PDF](https://arxiv.org/pdf/2508.15451)  

**Abstract**: This paper studies an input-driven one-state differential equation model initially developed for an experimentally demonstrated dynamic molecular switch that switches like synapses in the brain do. The linear-in-the-state and nonlinear-in-the-input model is exactly solvable, and it is shown that it also possesses mathematical properties of convergence and fading memory that enable stable processing of time-varying inputs by nonlinear dynamical systems. Thus, the model exhibits the co-existence of biologically-inspired behavior and desirable mathematical properties for stable learning on sequential data. The results give theoretical support for the use of the dynamic molecular switches as computational units in deep cascaded/layered feedforward and recurrent architectures as well as other more general structures for neuromorphic computing. They could also inspire more general exactly solvable models that can be fitted to emulate arbitrary physical devices which can mimic brain-inspired behaviour and perform stable computation on input signals. 

---
# Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection 

**Authors**: Chengcan Wu, Zeming Wei, Huanran Chen, Yinpeng Dong, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.15449)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in this https URL. 

---
# Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets 

**Authors**: Chenlin Liu, Minghui Fang, Patrick Zhang, Wei Zhou, Jie Gao, Jiqing Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.15442)  

**Abstract**: Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness. 

---
# Test-time Corpus Feedback: From Retrieval to RAG 

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2508.15437)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a standard framework for knowledge-intensive NLP tasks, combining large language models (LLMs) with document retrieval from external corpora. Despite its widespread use, most RAG pipelines continue to treat retrieval and reasoning as isolated components, retrieving documents once and then generating answers without further interaction. This static design often limits performance on complex tasks that require iterative evidence gathering or high-precision retrieval. Recent work in both the information retrieval (IR) and NLP communities has begun to close this gap by introducing adaptive retrieval and ranking methods that incorporate feedback. In this survey, we present a structured overview of advanced retrieval and ranking mechanisms that integrate such feedback. We categorize feedback signals based on their source and role in improving the query, retrieved context, or document pool. By consolidating these developments, we aim to bridge IR and NLP perspectives and highlight retrieval as a dynamic, learnable component of end-to-end RAG systems. 

---
# An Empirical Study of Knowledge Distillation for Code Understanding Tasks 

**Authors**: Ruiqi Wang, Zezhou Yang, Cuiyun Gao, Xin Xia, Qing Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15423)  

**Abstract**: Pre-trained language models (PLMs) have emerged as powerful tools for code understanding. However, deploying these PLMs in large-scale applications faces practical challenges due to their computational intensity and inference latency. Knowledge distillation (KD), a promising model compression and acceleration technique, addresses these limitations by transferring knowledge from large teacher models to compact student models, enabling efficient inference while preserving most of the teacher models' capabilities. While this technique has shown remarkable success in natural language processing and computer vision domains, its potential for code understanding tasks remains largely underexplored.
In this paper, we systematically investigate the effectiveness and usage of KD in code understanding tasks. Our study encompasses two popular types of KD methods, i.e., logit-based and feature-based KD methods, experimenting across eight student models and two teacher PLMs from different domains on three downstream tasks. The experimental results indicate that KD consistently offers notable performance boosts across student models with different sizes compared with standard fine-tuning. Notably, code-specific PLM demonstrates better effectiveness as the teacher model. Among all KD methods, the latest feature-based KD methods exhibit superior performance, enabling student models to retain up to 98% teacher performance with merely 5% parameters. Regarding student architecture, our experiments reveal that similarity with teacher architecture does not necessarily lead to better performance. We further discuss the efficiency and behaviors in the KD process and inference, summarize the implications of findings, and identify promising future directions. 

---
# LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model 

**Authors**: Yirong Sun, Yizhong Geng, Peidong Wei, Yanjun Chen, Jinghan Yang, Rongfei Chen, Wei Zhang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15418)  

**Abstract**: The development of Large Speech-Language Models (LSLMs) has been slowed by fragmented architectures and a lack of transparency, hindering the systematic comparison and reproducibility of research. Unlike in the vision-language domain, the LSLM field suffers from the common practice of releasing model weights without their corresponding training data and configurations. To address these critical gaps, we introduce LLaSO, the first fully open, end-to-end framework for large-scale speech-language modeling. LLaSO provides the community with three essential resources: (1) LLaSO-Align, a 12M-instance speech-text alignment corpus; (2) LLaSO-Instruct, a 13.5M-instance multi-task instruction-tuning dataset; and (3) LLaSO-Eval, a reproducible benchmark for standardized evaluation. To validate our framework, we build and release LLaSO-Base, a 3.8B-parameter reference model trained exclusively on our public data. It achieves a normalized score of 0.72, establishing a strong, reproducible baseline that surpasses comparable models. Our analysis reveals that while broader training coverage enhances performance, significant generalization gaps persist on unseen tasks, particularly in pure audio scenarios. By releasing the complete stack of data, benchmarks, and models, LLaSO establishes a foundational open standard to unify research efforts and accelerate community-driven progress in LSLMs. We release the code, dataset, pretrained models, and results in this https URL. 

---
# Bridging Generalization and Personalization in Wearable Human Activity Recognition via On-Device Few-Shot Learning 

**Authors**: Pixi Kang, Julian Moosmann, Mengxi Liu, Bo Zhou, Michele Magno, Paul Lukowicz, Sizhen Bian  

**Link**: [PDF](https://arxiv.org/pdf/2508.15413)  

**Abstract**: Human Activity Recognition (HAR) using wearable devices has advanced significantly in recent years, yet its generalization remains limited when models are deployed to new users. This degradation in performance is primarily due to user-induced concept drift (UICD), highlighting the importance of efficient personalization. In this paper, we present a hybrid framework that first generalizes across users and then rapidly adapts to individual users using few-shot learning directly on-device. By updating only the classifier layer with user-specific data, our method achieves robust personalization with minimal computational and memory overhead. We implement this framework on the energy-efficient RISC-V-based GAP9 microcontroller and validate it across three diverse HAR scenarios: RecGym, QVAR-Gesture, and Ultrasound-Gesture. Post-deployment adaptation yields consistent accuracy improvements of 3.73\%, 17.38\%, and 3.70\% respectively. These results confirm that fast, lightweight, and effective personalization is feasible on embedded platforms, paving the way for scalable and user-aware HAR systems in the wild \footnote{this https URL}. 

---
# When Audio and Text Disagree: Revealing Text Bias in Large Audio-Language Models 

**Authors**: Cheng Wang, Gelei Deng, Xianglin Yang, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15407)  

**Abstract**: Large Audio-Language Models (LALMs) are enhanced with audio perception capabilities, enabling them to effectively process and understand multimodal inputs that combine audio and text. However, their performance in handling conflicting information between audio and text modalities remains largely unexamined. This paper introduces MCR-BENCH, the first comprehensive benchmark specifically designed to evaluate how LALMs prioritize information when presented with inconsistent audio-text pairs. Through extensive evaluation across diverse audio understanding tasks, we reveal a concerning phenomenon: when inconsistencies exist between modalities, LALMs display a significant bias toward textual input, frequently disregarding audio evidence. This tendency leads to substantial performance degradation in audio-centric tasks and raises important reliability concerns for real-world applications. We further investigate the influencing factors of text bias, and explore mitigation strategies through supervised finetuning, and analyze model confidence patterns that reveal persistent overconfidence even with contradictory inputs. These findings underscore the need for improved modality balance during training and more sophisticated fusion mechanisms to enhance the robustness when handling conflicting multi-modal inputs. The project is available at this https URL. 

---
# Hybrid Least Squares/Gradient Descent Methods for DeepONets 

**Authors**: Jun Choi, Chang-Ock Lee, Minam Moon  

**Link**: [PDF](https://arxiv.org/pdf/2508.15394)  

**Abstract**: We propose an efficient hybrid least squares/gradient descent method to accelerate DeepONet training. Since the output of DeepONet can be viewed as linear with respect to the last layer parameters of the branch network, these parameters can be optimized using a least squares (LS) solve, and the remaining hidden layer parameters are updated by means of gradient descent form. However, building the LS system for all possible combinations of branch and trunk inputs yields a prohibitively large linear problem that is infeasible to solve directly. To address this issue, our method decomposes the large LS system into two smaller, more manageable subproblems $\unicode{x2014}$ one for the branch network and one for the trunk network $\unicode{x2014}$ and solves them separately. This method is generalized to a broader type of $L^2$ loss with a regularization term for the last layer parameters, including the case of unsupervised learning with physics-informed loss. 

---
# Bladder Cancer Diagnosis with Deep Learning: A Multi-Task Framework and Online Platform 

**Authors**: Jinliang Yu, Mingduo Xie, Yue Wang, Tianfan Fu, Xianglai Xu, Jiajun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15379)  

**Abstract**: Clinical cystoscopy, the current standard for bladder cancer diagnosis, suffers from significant reliance on physician expertise, leading to variability and subjectivity in diagnostic outcomes. There is an urgent need for objective, accurate, and efficient computational approaches to improve bladder cancer diagnostics.
Leveraging recent advancements in deep learning, this study proposes an integrated multi-task deep learning framework specifically designed for bladder cancer diagnosis from cystoscopic images. Our framework includes a robust classification model using EfficientNet-B0 enhanced with Convolutional Block Attention Module (CBAM), an advanced segmentation model based on ResNet34-UNet++ architecture with self-attention mechanisms and attention gating, and molecular subtyping using ConvNeXt-Tiny to classify molecular markers such as HER-2 and Ki-67. Additionally, we introduce a Gradio-based online diagnostic platform integrating all developed models, providing intuitive features including multi-format image uploads, bilingual interfaces, and dynamic threshold adjustments.
Extensive experimentation demonstrates the effectiveness of our methods, achieving outstanding accuracy (93.28%), F1-score (82.05%), and AUC (96.41%) for classification tasks, and exceptional segmentation performance indicated by a Dice coefficient of 0.9091. The online platform significantly improved the accuracy, efficiency, and accessibility of clinical bladder cancer diagnostics, enabling practical and user-friendly deployment. The code is publicly available.
Our multi-task framework and integrated online tool collectively advance the field of intelligent bladder cancer diagnosis by improving clinical reliability, supporting early tumor detection, and enabling real-time diagnostic feedback. These contributions mark a significant step toward AI-assisted decision-making in urology. 

---
# EvoFormer: Learning Dynamic Graph-Level Representations with Structural and Temporal Bias Correction 

**Authors**: Haodi Zhong, Liuxin Zou, Di Wang, Bo Wang, Zhenxing Niu, Quan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15378)  

**Abstract**: Dynamic graph-level embedding aims to capture structural evolution in networks, which is essential for modeling real-world scenarios. However, existing methods face two critical yet under-explored issues: Structural Visit Bias, where random walk sampling disproportionately emphasizes high-degree nodes, leading to redundant and noisy structural representations; and Abrupt Evolution Blindness, the failure to effectively detect sudden structural changes due to rigid or overly simplistic temporal modeling strategies, resulting in inconsistent temporal embeddings. To overcome these challenges, we propose EvoFormer, an evolution-aware Transformer framework tailored for dynamic graph-level representation learning. To mitigate Structural Visit Bias, EvoFormer introduces a Structure-Aware Transformer Module that incorporates positional encoding based on node structural roles, allowing the model to globally differentiate and accurately represent node structures. To overcome Abrupt Evolution Blindness, EvoFormer employs an Evolution-Sensitive Temporal Module, which explicitly models temporal evolution through a sequential three-step strategy: (I) Random Walk Timestamp Classification, generating initial timestamp-aware graph-level embeddings; (II) Graph-Level Temporal Segmentation, partitioning the graph stream into segments reflecting structurally coherent periods; and (III) Segment-Aware Temporal Self-Attention combined with an Edge Evolution Prediction task, enabling the model to precisely capture segment boundaries and perceive structural evolution trends, effectively adapting to rapid temporal shifts. Extensive evaluations on five benchmark datasets confirm that EvoFormer achieves state-of-the-art performance in graph similarity ranking, temporal anomaly detection, and temporal segmentation tasks, validating its effectiveness in correcting structural and temporal biases. 

---
# Image-Conditioned 3D Gaussian Splat Quantization 

**Authors**: Xinshuang Liu, Runfa Blark Li, Keito Suzuki, Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15372)  

**Abstract**: 3D Gaussian Splatting (3DGS) has attracted considerable attention for enabling high-quality real-time rendering. Although 3DGS compression methods have been proposed for deployment on storage-constrained devices, two limitations hinder archival use: (1) they compress medium-scale scenes only to the megabyte range, which remains impractical for large-scale scenes or extensive scene collections; and (2) they lack mechanisms to accommodate scene changes after long-term archival. To address these limitations, we propose an Image-Conditioned Gaussian Splat Quantizer (ICGS-Quantizer) that substantially enhances compression efficiency and provides adaptability to scene changes after archiving. ICGS-Quantizer improves quantization efficiency by jointly exploiting inter-Gaussian and inter-attribute correlations and by using shared codebooks across all training scenes, which are then fixed and applied to previously unseen test scenes, eliminating the overhead of per-scene codebooks. This approach effectively reduces the storage requirements for 3DGS to the kilobyte range while preserving visual fidelity. To enable adaptability to post-archival scene changes, ICGS-Quantizer conditions scene decoding on images captured at decoding time. The encoding, quantization, and decoding processes are trained jointly, ensuring that the codes, which are quantized representations of the scene, are effective for conditional decoding. We evaluate ICGS-Quantizer on 3D scene compression and 3D scene updating. Experimental results show that ICGS-Quantizer consistently outperforms state-of-the-art methods in compression efficiency and adaptability to scene changes. Our code, model, and data will be publicly available on GitHub. 

---
# Unveiling Trust in Multimodal Large Language Models: Evaluation, Analysis, and Mitigation 

**Authors**: Yichi Zhang, Yao Huang, Yifan Wang, Yitong Sun, Chang Liu, Zhe Zhao, Zhengwei Fang, Huanran Chen, Xiao Yang, Xingxing Wei, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15370)  

**Abstract**: The trustworthiness of Multimodal Large Language Models (MLLMs) remains an intense concern despite the significant progress in their capabilities. Existing evaluation and mitigation approaches often focus on narrow aspects and overlook risks introduced by the multimodality. To tackle these challenges, we propose MultiTrust-X, a comprehensive benchmark for evaluating, analyzing, and mitigating the trustworthiness issues of MLLMs. We define a three-dimensional framework, encompassing five trustworthiness aspects which include truthfulness, robustness, safety, fairness, and privacy; two novel risk types covering multimodal risks and cross-modal impacts; and various mitigation strategies from the perspectives of data, model architecture, training, and inference algorithms. Based on the taxonomy, MultiTrust-X includes 32 tasks and 28 curated datasets, enabling holistic evaluations over 30 open-source and proprietary MLLMs and in-depth analysis with 8 representative mitigation methods. Our extensive experiments reveal significant vulnerabilities in current models, including a gap between trustworthiness and general capabilities, as well as the amplification of potential risks in base LLMs by both multimodal training and inference. Moreover, our controlled analysis uncovers key limitations in existing mitigation strategies that, while some methods yield improvements in specific aspects, few effectively address overall trustworthiness, and many introduce unexpected trade-offs that compromise model utility. These findings also provide practical insights for future improvements, such as the benefits of reasoning to better balance safety and performance. Based on these insights, we introduce a Reasoning-Enhanced Safety Alignment (RESA) approach that equips the model with chain-of-thought reasoning ability to discover the underlying risks, achieving state-of-the-art results. 

---
# Predicting Road Crossing Behaviour using Pose Detection and Sequence Modelling 

**Authors**: Subhasis Dasgupta, Preetam Saha, Agniva Roy, Jaydip Sen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15336)  

**Abstract**: The world is constantly moving towards AI based systems and autonomous vehicles are now reality in different parts of the world. These vehicles require sensors and cameras to detect objects and maneuver according to that. It becomes important to for such vehicles to also predict from a distant if a person is about to cross a road or not. The current study focused on predicting the intent of crossing the road by pedestrians in an experimental setup. The study involved working with deep learning models to predict poses and sequence modelling for temporal predictions. The study analysed three different sequence modelling to understand the prediction behaviour and it was found out that GRU was better in predicting the intent compared to LSTM model but 1D CNN was the best model in terms of speed. The study involved video analysis, and the output of pose detection model was integrated later on to sequence modelling techniques for an end-to-end deep learning framework for predicting road crossing intents. 

---
# VideoEraser: Concept Erasure in Text-to-Video Diffusion Models 

**Authors**: Naen Xu, Jinghuai Zhang, Changjiang Li, Zhi Chen, Chunyi Zhou, Qingming Li, Tianyu Du, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15314)  

**Abstract**: The rapid growth of text-to-video (T2V) diffusion models has raised concerns about privacy, copyright, and safety due to their potential misuse in generating harmful or misleading content. These models are often trained on numerous datasets, including unauthorized personal identities, artistic creations, and harmful materials, which can lead to uncontrolled production and distribution of such content. To address this, we propose VideoEraser, a training-free framework that prevents T2V diffusion models from generating videos with undesirable concepts, even when explicitly prompted with those concepts. Designed as a plug-and-play module, VideoEraser can seamlessly integrate with representative T2V diffusion models via a two-stage process: Selective Prompt Embedding Adjustment (SPEA) and Adversarial-Resilient Noise Guidance (ARNG). We conduct extensive evaluations across four tasks, including object erasure, artistic style erasure, celebrity erasure, and explicit content erasure. Experimental results show that VideoEraser consistently outperforms prior methods regarding efficacy, integrity, fidelity, robustness, and generalizability. Notably, VideoEraser achieves state-of-the-art performance in suppressing undesirable content during T2V generation, reducing it by 46% on average across four tasks compared to baselines. 

---
# First RAG, Second SEG: A Training-Free Paradigm for Camouflaged Object Detection 

**Authors**: Wutao Liu, YiDan Wang, Pan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15313)  

**Abstract**: Camouflaged object detection (COD) poses a significant challenge in computer vision due to the high similarity between objects and their backgrounds. Existing approaches often rely on heavy training and large computational resources. While foundation models such as the Segment Anything Model (SAM) offer strong generalization, they still struggle to handle COD tasks without fine-tuning and require high-quality prompts to yield good performance. However, generating such prompts manually is costly and inefficient. To address these challenges, we propose \textbf{First RAG, Second SEG (RAG-SEG)}, a training-free paradigm that decouples COD into two stages: Retrieval-Augmented Generation (RAG) for generating coarse masks as prompts, followed by SAM-based segmentation (SEG) for refinement. RAG-SEG constructs a compact retrieval database via unsupervised clustering, enabling fast and effective feature retrieval. During inference, the retrieved features produce pseudo-labels that guide precise mask generation using SAM2. Our method eliminates the need for conventional training while maintaining competitive performance. Extensive experiments on benchmark COD datasets demonstrate that RAG-SEG performs on par with or surpasses state-of-the-art methods. Notably, all experiments are conducted on a \textbf{personal laptop}, highlighting the computational efficiency and practicality of our approach. We present further analysis in the Appendix, covering limitations, salient object detection extension, and possible improvements. 

---
# IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents 

**Authors**: Hengyu An, Jinghuai Zhang, Tianyu Du, Chunyi Zhou, Qingming Li, Tao Lin, Shouling Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.15310)  

**Abstract**: Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments. 

---
# DesignCLIP: Multimodal Learning with CLIP for Design Patent Understanding 

**Authors**: Zhu Wang, Homaira Huda Shomee, Sathya N. Ravi, Sourav Medya  

**Link**: [PDF](https://arxiv.org/pdf/2508.15297)  

**Abstract**: In the field of design patent analysis, traditional tasks such as patent classification and patent image retrieval heavily depend on the image data. However, patent images -- typically consisting of sketches with abstract and structural elements of an invention -- often fall short in conveying comprehensive visual context and semantic information. This inadequacy can lead to ambiguities in evaluation during prior art searches. Recent advancements in vision-language models, such as CLIP, offer promising opportunities for more reliable and accurate AI-driven patent analysis. In this work, we leverage CLIP models to develop a unified framework DesignCLIP for design patent applications with a large-scale dataset of U.S. design patents. To address the unique characteristics of patent data, DesignCLIP incorporates class-aware classification and contrastive learning, utilizing generated detailed captions for patent images and multi-views image learning. We validate the effectiveness of DesignCLIP across various downstream tasks, including patent classification and patent retrieval. Additionally, we explore multimodal patent retrieval, which provides the potential to enhance creativity and innovation in design by offering more diverse sources of inspiration. Our experiments show that DesignCLIP consistently outperforms baseline and SOTA models in the patent domain on all tasks. Our findings underscore the promise of multimodal approaches in advancing patent analysis. The codebase is available here: this https URL. 

---
# Way to Build Native AI-driven 6G Air Interface: Principles, Roadmap, and Outlook 

**Authors**: Ping Zhang, Kai Niu, Yiming Liu, Zijian Liang, Nan Ma, Xiaodong Xu, Wenjun Xu, Mengying Sun, Yinqiu Liu, Xiaoyun Wang, Ruichen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15277)  

**Abstract**: Artificial intelligence (AI) is expected to serve as a foundational capability across the entire lifecycle of 6G networks, spanning design, deployment, and operation. This article proposes a native AI-driven air interface architecture built around two core characteristics: compression and adaptation. On one hand, compression enables the system to understand and extract essential semantic information from the source data, focusing on task relevance rather than symbol-level accuracy. On the other hand, adaptation allows the air interface to dynamically transmit semantic information across diverse tasks, data types, and channel conditions, ensuring scalability and robustness. This article first introduces the native AI-driven air interface architecture, then discusses representative enabling methodologies, followed by a case study on semantic communication in 6G non-terrestrial networks. Finally, it presents a forward-looking discussion on the future of native AI in 6G, outlining key challenges and research opportunities. 

---
# M-$LLM^3$REC: A Motivation-Aware User-Item Interaction Framework for Enhancing Recommendation Accuracy with LLMs 

**Authors**: Lining Chen, Qingwen Zeng, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15262)  

**Abstract**: Recommendation systems have been essential for both user experience and platform efficiency by alleviating information overload and supporting decision-making. Traditional methods, i.e., content-based filtering, collaborative filtering, and deep learning, have achieved impressive results in recommendation systems. However, the cold-start and sparse-data scenarios are still challenging to deal with. Existing solutions either generate pseudo-interaction sequence, which often introduces redundant or noisy signals, or rely heavily on semantic similarity, overlooking dynamic shifts in user motivation. To address these limitations, this paper proposes a novel recommendation framework, termed M-$LLM^3$REC, which leverages large language models for deep motivational signal extraction from limited user interactions. M-$LLM^3$REC comprises three integrated modules: the Motivation-Oriented Profile Extractor (MOPE), Motivation-Oriented Trait Encoder (MOTE), and Motivational Alignment Recommender (MAR). By emphasizing motivation-driven semantic modeling, M-$LLM^3$REC demonstrates robust, personalized, and generalizable recommendations, particularly boosting performance in cold-start situations in comparison with the state-of-the-art frameworks. 

---
# Conflict-Aware Soft Prompting for Retrieval-Augmented Generation 

**Authors**: Eunseong Choi, June Park, Hyeri Lee, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.15253)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge into their input prompts. However, when the retrieved context contradicts the LLM's parametric knowledge, it often fails to resolve the conflict between incorrect external context and correct parametric knowledge, known as context-memory conflict. To tackle this problem, we introduce Conflict-Aware REtrieval-Augmented Generation (CARE), consisting of a context assessor and a base LLM. The context assessor encodes compact memory token embeddings from raw context tokens. Through grounded/adversarial soft prompting, the context assessor is trained to discern unreliable context and capture a guidance signal that directs reasoning toward the more reliable knowledge source. Extensive experiments show that CARE effectively mitigates context-memory conflicts, leading to an average performance gain of 5.0\% on QA and fact-checking benchmarks, establishing a promising direction for trustworthy and adaptive RAG systems. 

---
# Explainable Knowledge Distillation for Efficient Medical Image Classification 

**Authors**: Aqib Nazir Mir, Danish Raza Rizvi  

**Link**: [PDF](https://arxiv.org/pdf/2508.15251)  

**Abstract**: This study comprehensively explores knowledge distillation frameworks for COVID-19 and lung cancer classification using chest X-ray (CXR) images. We employ high-capacity teacher models, including VGG19 and lightweight Vision Transformers (Visformer-S and AutoFormer-V2-T), to guide the training of a compact, hardware-aware student model derived from the OFA-595 supernet. Our approach leverages hybrid supervision, combining ground-truth labels with teacher models' soft targets to balance accuracy and computational efficiency. We validate our models on two benchmark datasets: COVID-QU-Ex and LCS25000, covering multiple classes, including COVID-19, healthy, non-COVID pneumonia, lung, and colon cancer. To interpret the spatial focus of the models, we employ Score-CAM-based visualizations, which provide insight into the reasoning process of both teacher and student networks. The results demonstrate that the distilled student model maintains high classification performance with significantly reduced parameters and inference time, making it an optimal choice in resource-constrained clinical environments. Our work underscores the importance of combining model efficiency with explainability for practical, trustworthy medical AI solutions. 

---
# Robust and Efficient Quantum Reservoir Computing with Discrete Time Crystal 

**Authors**: Da Zhang, Xin Li, Yibin Guo, Haifeng Yu, Yirong Jin, Zhang-Qi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15230)  

**Abstract**: The rapid development of machine learning and quantum computing has placed quantum machine learning at the forefront of research. However, existing quantum machine learning algorithms based on quantum variational algorithms face challenges in trainability and noise robustness. In order to address these challenges, we introduce a gradient-free, noise-robust quantum reservoir computing algorithm that harnesses discrete time crystal dynamics as a reservoir. We first calibrate the memory, nonlinear, and information scrambling capacities of the quantum reservoir, revealing their correlation with dynamical phases and non-equilibrium phase transitions. We then apply the algorithm to the binary classification task and establish a comparative quantum kernel advantage. For ten-class classification, both noisy simulations and experimental results on superconducting quantum processors match ideal simulations, demonstrating the enhanced accuracy with increasing system size and confirming the topological noise robustness. Our work presents the first experimental demonstration of quantum reservoir computing for image classification based on digital quantum simulation. It establishes the correlation between quantum many-body non-equilibrium phase transitions and quantum machine learning performance, providing new design principles for quantum reservoir computing and broader quantum machine learning algorithms in the NISQ era. 

---
# VocabTailor: Dynamic Vocabulary Selection for Downstream Tasks in Small Language Models 

**Authors**: Hanling Zhang, Yayu Zhou, Tongcheng Fang, Zhihang Yuan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15229)  

**Abstract**: Small Language Models (SLMs) provide computational advantages in resource-constrained environments, yet memory limitations remain a critical bottleneck for edge device deployment. A substantial portion of SLMs' memory footprint stems from vocabulary-related components, particularly embeddings and language modeling (LM) heads, due to large vocabulary sizes. Existing static vocabulary pruning, while reducing memory usage, suffers from rigid, one-size-fits-all designs that cause information loss from the prefill stage and a lack of flexibility. In this work, we identify two key principles underlying the vocabulary reduction challenge: the lexical locality principle, the observation that only a small subset of tokens is required during any single inference, and the asymmetry in computational characteristics between vocabulary-related components of SLM. Based on these insights, we introduce VocabTailor, a novel decoupled dynamic vocabulary selection framework that addresses memory constraints through offloading embedding and implements a hybrid static-dynamic vocabulary selection strategy for LM Head, enabling on-demand loading of vocabulary components. Comprehensive experiments across diverse downstream tasks demonstrate that VocabTailor achieves a reduction of up to 99% in the memory usage of vocabulary-related components with minimal or no degradation in task performance, substantially outperforming existing static vocabulary pruning. 

---
# GenTune: Toward Traceable Prompts to Improve Controllability of Image Refinement in Environment Design 

**Authors**: Wen-Fan Wang, Ting-Ying Lee, Chien-Ting Lu, Che-Wei Hsu, Nil Ponsa Campany, Yu Chen, Mike Y. Chen, Bing-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.15227)  

**Abstract**: Environment designers in the entertainment industry create imaginative 2D and 3D scenes for games, films, and television, requiring both fine-grained control of specific details and consistent global coherence. Designers have increasingly integrated generative AI into their workflows, often relying on large language models (LLMs) to expand user prompts for text-to-image generation, then iteratively refining those prompts and applying inpainting. However, our formative study with 10 designers surfaced two key challenges: (1) the lengthy LLM-generated prompts make it difficult to understand and isolate the keywords that must be revised for specific visual elements; and (2) while inpainting supports localized edits, it can struggle with global consistency and correctness. Based on these insights, we present GenTune, an approach that enhances human--AI collaboration by clarifying how AI-generated prompts map to image content. Our GenTune system lets designers select any element in a generated image, trace it back to the corresponding prompt labels, and revise those labels to guide precise yet globally consistent image refinement. In a summative study with 20 designers, GenTune significantly improved prompt--image comprehension, refinement quality, and efficiency, and overall satisfaction (all $p < .01$) compared to current practice. A follow-up field study with two studios further demonstrated its effectiveness in real-world settings. 

---
# Locally Pareto-Optimal Interpretations for Black-Box Machine Learning Models 

**Authors**: Aniruddha Joshi, Supratik Chakraborty, S Akshay, Shetal Shah, Hazem Torfah, Sanjit Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2508.15220)  

**Abstract**: Creating meaningful interpretations for black-box machine learning models involves balancing two often conflicting objectives: accuracy and explainability. Exploring the trade-off between these objectives is essential for developing trustworthy interpretations. While many techniques for multi-objective interpretation synthesis have been developed, they typically lack formal guarantees on the Pareto-optimality of the results. Methods that do provide such guarantees, on the other hand, often face severe scalability limitations when exploring the Pareto-optimal space. To address this, we develop a framework based on local optimality guarantees that enables more scalable synthesis of interpretations. Specifically, we consider the problem of synthesizing a set of Pareto-optimal interpretations with local optimality guarantees, within the immediate neighborhood of each solution. Our approach begins with a multi-objective learning or search technique, such as Multi-Objective Monte Carlo Tree Search, to generate a best-effort set of Pareto-optimal candidates with respect to accuracy and explainability. We then verify local optimality for each candidate as a Boolean satisfiability problem, which we solve using a SAT solver. We demonstrate the efficacy of our approach on a set of benchmarks, comparing it against previous methods for exploring the Pareto-optimal front of interpretations. In particular, we show that our approach yields interpretations that closely match those synthesized by methods offering global guarantees. 

---
# SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning 

**Authors**: Huanxuan Liao, Yixing Xu, Shizhu He, Guanchen Li, Xuanwu Yin, Dong Li, Emad Barsoum, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15212)  

**Abstract**: Long-context inference in large language models (LLMs) is increasingly constrained by the KV cache bottleneck: memory usage grows linearly with sequence length, while attention computation scales quadratically. Existing approaches address this issue by compressing the KV cache along the temporal axis through strategies such as token eviction or merging to reduce memory and computational overhead. However, these methods often neglect fine-grained importance variations across feature dimensions (i.e., the channel axis), thereby limiting their ability to effectively balance efficiency and model accuracy. In reality, we observe that channel saliency varies dramatically across both queries and positions: certain feature channels carry near-zero information for a given query, while others spike in relevance. To address this oversight, we propose SPARK, a training-free plug-and-play method that applies unstructured sparsity by pruning KV at the channel level, while dynamically restoring the pruned entries during attention score computation. Notably, our approach is orthogonal to existing KV compression and quantization techniques, making it compatible for integration with them to achieve further acceleration. By reducing channel-level redundancy, SPARK enables processing of longer sequences within the same memory budget. For sequences of equal length, SPARK not only preserves or improves model accuracy but also reduces KV cache storage by over 30% compared to eviction-based methods. Furthermore, even with an aggressive pruning ratio of 80%, SPARK maintains performance with less degradation than 5% compared to the baseline eviction method, demonstrating its robustness and effectiveness. Our code will be available at this https URL. 

---
# Survey of Vision-Language-Action Models for Embodied Manipulation 

**Authors**: Haoran Li, Yuhui Chen, Wenbo Cui, Weiheng Liu, Kai Liu, Mingcai Zhou, Zhengtao Zhang, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.15201)  

**Abstract**: Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions. 

---
# SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15190)  

**Abstract**: Tokenization plays a critical role in language modeling, yet existing approaches such as Byte-Pair Encoding (BPE) or WordPiece operate purely on frequency statistics, ignoring the underlying semantic structure of text. This leads to over-tokenization of semantically redundant spans and underutilization of contextual coherence, particularly in long-context scenarios. In this work, we propose \textbf{SemToken}, a semantic-aware tokenization framework that jointly reduces token redundancy and improves computation efficiency. SemToken first extracts contextual semantic embeddings via lightweight encoders and performs local semantic clustering to merge semantically equivalent tokens. Then, it allocates heterogeneous token granularity based on semantic density, allowing finer-grained tokenization in content-rich regions and coarser compression in repetitive or low-entropy spans. SemToken can be seamlessly integrated with modern language models and attention acceleration methods. Experiments on long-context language modeling benchmarks such as WikiText-103 and LongBench show that SemToken achieves up to $2.4\times$ reduction in token count and $1.9\times$ speedup, with negligible or no degradation in perplexity and downstream accuracy. Our findings suggest that semantic structure offers a promising new axis for optimizing tokenization and computation in large language models. 

---
# SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis 

**Authors**: Jiahao Xu, Changchang Yin, Odysseas Chatzipanagiotou, Diamantis Tsilimigras, Kevin Clear, Bingsheng Yao, Dakuo Wang, Timothy Pawlik, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.15189)  

**Abstract**: Surgical site infection (SSI) is one of the most common and costly healthcare-associated infections and and surgical wound care remains a significant clinical challenge in preventing SSIs and improving patient outcomes. While recent studies have explored the use of deep learning for preliminary surgical wound screening, progress has been hindered by concerns over data privacy and the high costs associated with expert annotation. Currently, no publicly available dataset or benchmark encompasses various types of surgical wounds, resulting in the absence of an open-source Surgical-Wound screening tool. To address this gap: (1) we present SurgWound, the first open-source dataset featuring a diverse array of surgical wound types. It contains 697 surgical wound images annotated by 3 professional surgeons with eight fine-grained clinical attributes. (2) Based on SurgWound, we introduce the first benchmark for surgical wound diagnosis, which includes visual question answering (VQA) and report generation tasks to comprehensively evaluate model performance. (3) Furthermore, we propose a three-stage learning framework, WoundQwen, for surgical wound diagnosis. In the first stage, we employ five independent MLLMs to accurately predict specific surgical wound characteristics. In the second stage, these predictions serve as additional knowledge inputs to two MLLMs responsible for diagnosing outcomes, which assess infection risk and guide subsequent interventions. In the third stage, we train a MLLM that integrates the diagnostic results from the previous two stages to produce a comprehensive report. This three-stage framework can analyze detailed surgical wound characteristics and provide subsequent instructions to patients based on surgical images, paving the way for personalized wound care, timely intervention, and improved patient outcomes. 

---
# Universal Reinforcement Learning in Coalgebras: Asynchronous Stochastic Computation via Conduction 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.15128)  

**Abstract**: In this paper, we introduce a categorial generalization of RL, termed universal reinforcement learning (URL), building on powerful mathematical abstractions from the study of coinduction on non-well-founded sets and universal coalgebras, topos theory, and categorial models of asynchronous parallel distributed computation. In the first half of the paper, we review the basic RL framework, illustrate the use of categories and functors in RL, showing how they lead to interesting insights. In particular, we also introduce a standard model of asynchronous distributed minimization proposed by Bertsekas and Tsitsiklis, and describe the relationship between metric coinduction and their proof of the Asynchronous Convergence Theorem. The space of algorithms for MDPs or PSRs can be modeled as a functor category, where the co-domain category forms a topos, which admits all (co)limits, possesses a subobject classifier, and has exponential objects. In the second half of the paper, we move on to universal coalgebras. Dynamical system models, such as Markov decision processes (MDPs), partially observed MDPs (POMDPs), a predictive state representation (PSRs), and linear dynamical systems (LDSs) are all special types of coalgebras. We describe a broad family of universal coalgebras, extending the dynamic system models studied previously in RL. The core problem in finding fixed points in RL to determine the exact or approximate (action) value function is generalized in URL to determining the final coalgebra asynchronously in a parallel distributed manner. 

---
# Enhanced Predictive Modeling for Hazardous Near-Earth Object Detection: A Comparative Analysis of Advanced Resampling Strategies and Machine Learning Algorithms in Planetary Risk Assessment 

**Authors**: Sunkalp Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2508.15106)  

**Abstract**: This study evaluates the performance of several machine learning models for predicting hazardous near-Earth objects (NEOs) through a binary classification framework, including data scaling, power transformation, and cross-validation. Six classifiers were compared, namely Random Forest Classifier (RFC), Gradient Boosting Classifier (GBC), Support Vector Classifier (SVC), Linear Discriminant Analysis (LDA), Logistic Regression (LR), and K-Nearest Neighbors (KNN). RFC and GBC performed the best, both with an impressive F2-score of 0.987 and 0.986, respectively, with very small variability. SVC followed, with a lower but reasonable score of 0.896. LDA and LR had a moderate performance with scores of around 0.749 and 0.748, respectively, while KNN had a poor performance with a score of 0.691 due to difficulty in handling complex data patterns. RFC and GBC also presented great confusion matrices with a negligible number of false positives and false negatives, which resulted in outstanding accuracy rates of 99.7% and 99.6%, respectively. These findings highlight the power of ensemble methods for high precision and recall and further point out the importance of tailored model selection with regard to dataset characteristics and chosen evaluation metrics. Future research could focus on the optimization of hyperparameters with advanced features engineering to further the accuracy and robustness of the model on NEO hazard predictions. 

---
# Equi-mRNA: Protein Translation Equivariant Encoding for mRNA Language Models 

**Authors**: Mehdi Yazdani-Jahromi, Ali Khodabandeh Yalabadi, Ozlem Ozmen Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2508.15103)  

**Abstract**: The growing importance of mRNA therapeutics and synthetic biology highlights the need for models that capture the latent structure of synonymous codon (different triplets encoding the same amino acid) usage, which subtly modulates translation efficiency and gene expression. While recent efforts incorporate codon-level inductive biases through auxiliary objectives, they often fall short of explicitly modeling the structured relationships that arise from the genetic code's inherent symmetries. We introduce Equi-mRNA, the first codon-level equivariant mRNA language model that explicitly encodes synonymous codon symmetries as cyclic subgroups of 2D Special Orthogonal matrix (SO(2)). By combining group-theoretic priors with an auxiliary equivariance loss and symmetry-aware pooling, Equi-mRNA learns biologically grounded representations that outperform vanilla baselines across multiple axes. On downstream property-prediction tasks including expression, stability, and riboswitch switching Equi-mRNA delivers up to approximately 10% improvements in accuracy. In sequence generation, it produces mRNA constructs that are up to approximately 4x more realistic under Frechet BioDistance metrics and approximately 28% better preserve functional properties compared to vanilla baseline. Interpretability analyses further reveal that learned codon-rotation distributions recapitulate known GC-content biases and tRNA abundance patterns, offering novel insights into codon usage. Equi-mRNA establishes a new biologically principled paradigm for mRNA modeling, with significant implications for the design of next-generation therapeutics. 

---
# Hydra: A 1.6B-Parameter State-Space Language Model with Sparse Attention, Mixture-of-Experts, and Memory 

**Authors**: Siddharth Chaudhary, Bennett Browning  

**Link**: [PDF](https://arxiv.org/pdf/2508.15099)  

**Abstract**: We present Hydra as an architectural proposal for hybrid long-context language models that combine conditional computation, long-context memory mechanisms, and sparse mixture-of-experts within an approximately 1.6B parameter design envelope. Hydra integrates a Mamba-style Structured State Space Model (SSM) backbone with intermittent sparse global attention, chunk-level MoE feed-forward routing, and dual (workspace plus factual PKM) memories. We formalize the component interfaces, give transparent parameter and complexity accounting, and outline a staged curriculum intended to stably activate the parts. We accompany the specification with illustrative toy-scale prototype measurements (tens of millions of parameters on synthetic data) whose sole purpose is to demonstrate implementation feasibility and qualitative scaling behaviors (for example, long-context throughput crossover and controllable expert routing), not to claim competitive full-scale performance. We explicitly delineate assumptions and open risks (training complexity, memory utilization, specialization dynamics) and position Hydra as a blueprint to stimulate empirical follow-up rather than a finished system. By combining SSM efficiency, selective sparse attention, MoE capacity, and learnable memory, Hydra sketches a path toward modular, input-adaptive long-context language models; validating end-task gains at target scale remains future work. 

---
# Nemotron-CC-Math: A 133 Billion-Token-Scale High Quality Math Pretraining Dataset 

**Authors**: Rabeeh Karimi Mahabadi, Sanjeev Satheesh, Shrimai Prabhumoye, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2508.15096)  

**Abstract**: Pretraining large language models (LLMs) on high-quality, structured data such as mathematics and code substantially enhances reasoning capabilities. However, existing math-focused datasets built from Common Crawl suffer from degraded quality due to brittle extraction heuristics, lossy HTML-to-text conversion, and the failure to reliably preserve mathematical structure. In this work, we introduce Nemotron-CC-Math, a large-scale, high-quality mathematical corpus constructed from Common Crawl using a novel, domain-agnostic pipeline specifically designed for robust scientific text extraction.
Unlike previous efforts, our pipeline recovers math across various formats (e.g., MathJax, KaTeX, MathML) by leveraging layout-aware rendering with lynx and a targeted LLM-based cleaning stage. This approach preserves the structural integrity of equations and code blocks while removing boilerplate, standardizing notation into LaTeX representation, and correcting inconsistencies.
We collected a large, high-quality math corpus, namely Nemotron-CC-Math-3+ (133B tokens) and Nemotron-CC-Math-4+ (52B tokens). Notably, Nemotron-CC-Math-4+ not only surpasses all prior open math datasets-including MegaMath, FineMath, and OpenWebMath-but also contains 5.5 times more tokens than FineMath-4+, which was previously the highest-quality math pretraining dataset. When used to pretrain a Nemotron-T 8B model, our corpus yields +4.8 to +12.6 gains on MATH and +4.6 to +14.3 gains on MBPP+ over strong baselines, while also improving general-domain performance on MMLU and MMLU-Stem.
We present the first pipeline to reliably extract scientific content--including math--from noisy web-scale data, yielding measurable gains in math, code, and general reasoning, and setting a new state of the art among open math pretraining corpora. To support open-source efforts, we release our code and datasets. 

---
# Mapping the Course for Prompt-based Structured Prediction 

**Authors**: Matt Pauk, Maria Leonor Pacheco  

**Link**: [PDF](https://arxiv.org/pdf/2508.15090)  

**Abstract**: LLMs have been shown to be useful for a variety of language tasks, without requiring task-specific fine-tuning. However, these models often struggle with hallucinations and complex reasoning problems due to their autoregressive nature. We propose to address some of these issues, specifically in the area of structured prediction, by combining LLMs with combinatorial inference in an attempt to marry the predictive power of LLMs with the structural consistency provided by inference methods. We perform exhaustive experiments in an effort to understand which prompting strategies can effectively estimate LLM confidence values for use with symbolic inference, and show that, regardless of the prompting strategy, the addition of symbolic inference on top of prompting alone leads to more consistent and accurate predictions. Additionally, we show that calibration and fine-tuning using structured prediction objectives leads to increased performance for challenging tasks, showing that structured learning is still valuable in the era of LLMs. 

---
# Wormhole Dynamics in Deep Neural Networks 

**Authors**: Yen-Lung Lai, Zhe Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.15086)  

**Abstract**: This work investigates the generalization behavior of deep neural networks (DNNs), focusing on the phenomenon of "fooling examples," where DNNs confidently classify inputs that appear random or unstructured to humans. To explore this phenomenon, we introduce an analytical framework based on maximum likelihood estimation, without adhering to conventional numerical approaches that rely on gradient-based optimization and explicit labels. Our analysis reveals that DNNs operating in an overparameterized regime exhibit a collapse in the output feature space. While this collapse improves network generalization, adding more layers eventually leads to a state of degeneracy, where the model learns trivial solutions by mapping distinct inputs to the same output, resulting in zero loss. Further investigation demonstrates that this degeneracy can be bypassed using our newly derived "wormhole" solution. The wormhole solution, when applied to arbitrary fooling examples, reconciles meaningful labels with random ones and provides a novel perspective on shortcut learning. These findings offer deeper insights into DNN generalization and highlight directions for future research on learning dynamics in unsupervised settings to bridge the gap between theory and practice. 

---
# LongRecall: A Structured Approach for Robust Recall Evaluation in Long-Form Text 

**Authors**: MohamamdJavad Ardestani, Ehsan Kamalloo, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15085)  

**Abstract**: LongRecall. The completeness of machine-generated text, ensuring that it captures all relevant information, is crucial in domains such as medicine and law and in tasks like list-based question answering (QA), where omissions can have serious consequences. However, existing recall metrics often depend on lexical overlap, leading to errors with unsubstantiated entities and paraphrased answers, while LLM-as-a-Judge methods with long holistic prompts capture broader semantics but remain prone to misalignment and hallucinations without structured verification. We introduce LongRecall, a general three-stage recall evaluation framework that decomposes answers into self-contained facts, successively narrows plausible candidate matches through lexical and semantic filtering, and verifies their alignment through structured entailment checks. This design reduces false positives and false negatives while accommodating diverse phrasings and contextual variations, serving as a foundational building block for systematic recall assessment. We evaluate LongRecall on three challenging long-form QA benchmarks using both human annotations and LLM-based judges, demonstrating substantial improvements in recall accuracy over strong lexical and LLM-as-a-Judge baselines. 

---
# From Basic Affordances to Symbolic Thought: A Computational Phylogenesis of Biological Intelligence 

**Authors**: John E. Hummel, Rachel F. Heaton  

**Link**: [PDF](https://arxiv.org/pdf/2508.15082)  

**Abstract**: What is it about human brains that allows us to reason symbolically whereas most other animals cannot? There is evidence that dynamic binding, the ability to combine neurons into groups on the fly, is necessary for symbolic thought, but there is also evidence that it is not sufficient. We propose that two kinds of hierarchical integration (integration of multiple role-bindings into multiplace predicates, and integration of multiple correspondences into structure mappings) are minimal requirements, on top of basic dynamic binding, to realize symbolic thought. We tested this hypothesis in a systematic collection of 17 simulations that explored the ability of cognitive architectures with and without the capacity for multi-place predicates and structure mapping to perform various kinds of tasks. The simulations were as generic as possible, in that no task could be performed based on any diagnostic features, depending instead on the capacity for multi-place predicates and structure mapping. The results are consistent with the hypothesis that, along with dynamic binding, multi-place predicates and structure mapping are minimal requirements for basic symbolic thought. These results inform our understanding of how human brains give rise to symbolic thought and speak to the differences between biological intelligence, which tends to generalize broadly from very few training examples, and modern approaches to machine learning, which typically require millions or billions of training examples. The results we report also have important implications for bio-inspired artificial intelligence. 

---
# Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring 

**Authors**: Makram Chahine, William Yang, Alaa Maalouf, Justin Siriska, Ninad Jadhav, Daniel Vogt, Stephanie Gil, Robert Wood, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2508.15038)  

**Abstract**: Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions. 

---
# MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs 

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.15036)  

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services. 

---
# A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives 

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2508.15031)  

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at this https URL. 

---
# Reversible Unfolding Network for Concealed Visual Perception with Generative Refinement 

**Authors**: Chunming He, Fengyang Xiao, Rihan Zhang, Chengyu Fang, Deng-Ping Fan, Sina Farsiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.15027)  

**Abstract**: Existing methods for concealed visual perception (CVP) often leverage reversible strategies to decrease uncertainty, yet these are typically confined to the mask domain, leaving the potential of the RGB domain underexplored. To address this, we propose a reversible unfolding network with generative refinement, termed RUN++. Specifically, RUN++ first formulates the CVP task as a mathematical optimization problem and unfolds the iterative solution into a multi-stage deep network. This approach provides a principled way to apply reversible modeling across both mask and RGB domains while leveraging a diffusion model to resolve the resulting uncertainty. Each stage of the network integrates three purpose-driven modules: a Concealed Object Region Extraction (CORE) module applies reversible modeling to the mask domain to identify core object regions; a Context-Aware Region Enhancement (CARE) module extends this principle to the RGB domain to foster better foreground-background separation; and a Finetuning Iteration via Noise-based Enhancement (FINE) module provides a final refinement. The FINE module introduces a targeted Bernoulli diffusion model that refines only the uncertain regions of the segmentation mask, harnessing the generative power of diffusion for fine-detail restoration without the prohibitive computational cost of a full-image process. This unique synergy, where the unfolding network provides a strong uncertainty prior for the diffusion model, allows RUN++ to efficiently direct its focus toward ambiguous areas, significantly mitigating false positives and negatives. Furthermore, we introduce a new paradigm for building robust CVP systems that remain effective under real-world degradations and extend this concept into a broader bi-level optimization framework. 

---
# TAIGen: Training-Free Adversarial Image Generation via Diffusion Models 

**Authors**: Susim Roy, Anubhooti Jain, Mayank Vatsa, Richa Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.15020)  

**Abstract**: Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen. 

---
# Twin-Boot: Uncertainty-Aware Optimization via Online Two-Sample Bootstrapping 

**Authors**: Carlos Stein Brito  

**Link**: [PDF](https://arxiv.org/pdf/2508.15019)  

**Abstract**: Standard gradient descent methods yield point estimates with no measure of confidence. This limitation is acute in overparameterized and low-data regimes, where models have many parameters relative to available data and can easily overfit. Bootstrapping is a classical statistical framework for uncertainty estimation based on resampling, but naively applying it to deep learning is impractical: it requires training many replicas, produces post-hoc estimates that cannot guide learning, and implicitly assumes comparable optima across runs - an assumption that fails in non-convex landscapes. We introduce Twin-Bootstrap Gradient Descent (Twin-Boot), a resampling-based training procedure that integrates uncertainty estimation into optimization. Two identical models are trained in parallel on independent bootstrap samples, and a periodic mean-reset keeps both trajectories in the same basin so that their divergence reflects local (within-basin) uncertainty. During training, we use this estimate to sample weights in an adaptive, data-driven way, providing regularization that favors flatter solutions. In deep neural networks and complex high-dimensional inverse problems, the approach improves calibration and generalization and yields interpretable uncertainty maps. 

---
# Quantized Neural Networks for Microcontrollers: A Comprehensive Review of Methods, Platforms, and Applications 

**Authors**: Hamza A. Abushahla, Dara Varam, Ariel J. N. Panopio, Mohamed I. AlHajri  

**Link**: [PDF](https://arxiv.org/pdf/2508.15008)  

**Abstract**: The deployment of Quantized Neural Networks (QNNs) on resource-constrained devices, such as microcontrollers, has introduced significant challenges in balancing model performance, computational complexity and memory constraints. Tiny Machine Learning (TinyML) addresses these issues by integrating advancements across machine learning algorithms, hardware acceleration, and software optimization to efficiently run deep neural networks on embedded systems. This survey presents a hardware-centric introduction to quantization, systematically reviewing essential quantization techniques employed to accelerate deep learning models for embedded applications. In particular, further emphasis is put on critical trade-offs among model performance and hardware capabilities. The survey further evaluates existing software frameworks and hardware platforms designed specifically for supporting QNN execution on microcontrollers. Moreover, we provide an analysis of the current challenges and an outline of promising future directions in the rapidly evolving domain of QNN deployment. 

---
# Fast Graph Neural Network for Image Classification 

**Authors**: Mustafa Mohammadi Gharasuie, Luis Rueda  

**Link**: [PDF](https://arxiv.org/pdf/2508.14958)  

**Abstract**: The rapid progress in image classification has been largely driven by the adoption of Graph Convolutional Networks (GCNs), which offer a robust framework for handling complex data structures. This study introduces a novel approach that integrates GCNs with Voronoi diagrams to enhance image classification by leveraging their ability to effectively model relational data. Unlike conventional convolutional neural networks (CNNs), our method represents images as graphs, where pixels or regions function as vertices. These graphs are then refined using corresponding Delaunay triangulations, optimizing their representation. The proposed model achieves significant improvements in both preprocessing efficiency and classification accuracy across various benchmark datasets, surpassing state-of-the-art approaches, particularly in challenging scenarios involving intricate scenes and fine-grained categories. Experimental results, validated through cross-validation, underscore the effectiveness of combining GCNs with Voronoi diagrams for advancing image classification. This research not only presents a novel perspective on image classification but also expands the potential applications of graph-based learning paradigms in computer vision and unstructured data analysis. 

---
# Quantum Long Short-term Memory with Differentiable Architecture Search 

**Authors**: Samuel Yen-Chi Chen, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14955)  

**Abstract**: Recent advances in quantum computing and machine learning have given rise to quantum machine learning (QML), with growing interest in learning from sequential data. Quantum recurrent models like QLSTM are promising for time-series prediction, NLP, and reinforcement learning. However, designing effective variational quantum circuits (VQCs) remains challenging and often task-specific. To address this, we propose DiffQAS-QLSTM, an end-to-end differentiable framework that optimizes both VQC parameters and architecture selection during training. Our results show that DiffQAS-QLSTM consistently outperforms handcrafted baselines, achieving lower loss across diverse test settings. This approach opens the door to scalable and adaptive quantum sequence learning. 

---
# Can synthetic data reproduce real-world findings in epidemiology? A replication study using tree-based generative AI 

**Authors**: Jan Kapar, Kathrin Günther, Lori Ann Vallis, Klaus Berger, Nadine Binder, Hermann Brenner, Stefanie Castell, Beate Fischer, Volker Harth, Bernd Holleczek, Timm Intemann, Till Ittermann, André Karch, Thomas Keil, Lilian Krist, Berit Lange, Michael F. Leitzmann, Katharina Nimptsch, Nadia Obi, Iris Pigeot, Tobias Pischon, Tamara Schikowski, Börge Schmidt, Carsten Oliver Schmidt, Anja M. Sedlmair, Justine Tanoey, Harm Wienbergen, Andreas Wienke, Claudia Wigmann, Marvin N. Wright  

**Link**: [PDF](https://arxiv.org/pdf/2508.14936)  

**Abstract**: Generative artificial intelligence for synthetic data generation holds substantial potential to address practical challenges in epidemiology. However, many current methods suffer from limited quality, high computational demands, and complexity for non-experts. Furthermore, common evaluation strategies for synthetic data often fail to directly reflect statistical utility. Against this background, a critical underexplored question is whether synthetic data can reliably reproduce key findings from epidemiological research. We propose the use of adversarial random forests (ARF) as an efficient and convenient method for synthesizing tabular epidemiological data. To evaluate its performance, we replicated statistical analyses from six epidemiological publications and compared original with synthetic results. These publications cover blood pressure, anthropometry, myocardial infarction, accelerometry, loneliness, and diabetes, based on data from the German National Cohort (NAKO Gesundheitsstudie), the Bremen STEMI Registry U45 Study, and the Guelph Family Health Study. Additionally, we assessed the impact of dimensionality and variable complexity on synthesis quality by limiting datasets to variables relevant for individual analyses, including necessary derivations. Across all replicated original studies, results from multiple synthetic data replications consistently aligned with original findings. Even for datasets with relatively low sample size-to-dimensionality ratios, the replication outcomes closely matched the original results across various descriptive and inferential analyses. Reducing dimensionality and pre-deriving variables further enhanced both quality and stability of the results. 

---
# Inference Time Debiasing Concepts in Diffusion Models 

**Authors**: Lucas S. Kupssinskü, Marco N. Bochernitsan, Jordan Kopper, Otávio Parraga, Rodrigo C. Barros  

**Link**: [PDF](https://arxiv.org/pdf/2508.14933)  

**Abstract**: We propose DeCoDi, a debiasing procedure for text-to-image diffusion-based models that changes the inference procedure, does not significantly change image quality, has negligible compute overhead, and can be applied in any diffusion-based image generation model. DeCoDi changes the diffusion process to avoid latent dimension regions of biased concepts. While most deep learning debiasing methods require complex or compute-intensive interventions, our method is designed to change only the inference procedure. Therefore, it is more accessible to a wide range of practitioners. We show the effectiveness of the method by debiasing for gender, ethnicity, and age for the concepts of nurse, firefighter, and CEO. Two distinct human evaluators manually inspect 1,200 generated images. Their evaluation results provide evidence that our method is effective in mitigating biases based on gender, ethnicity, and age. We also show that an automatic bias evaluation performed by the GPT4o is not significantly statistically distinct from a human evaluation. Our evaluation shows promising results, with reliable levels of agreement between evaluators and more coverage of protected attributes. Our method has the potential to significantly improve the diversity of images it generates by diffusion-based text-to-image generative models. 

---
# TOM: An Open-Source Tongue Segmentation Method with Multi-Teacher Distillation and Task-Specific Data Augmentation 

**Authors**: Jiacheng Xie, Ziyang Zhang, Biplab Poudel, Congyu Guo, Yang Yu, Guanghui An, Xiaoting Tang, Lening Zhao, Chunhui Xu, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14932)  

**Abstract**: Tongue imaging serves as a valuable diagnostic tool, particularly in Traditional Chinese Medicine (TCM). The quality of tongue surface segmentation significantly affects the accuracy of tongue image classification and subsequent diagnosis in intelligent tongue diagnosis systems. However, existing research on tongue image segmentation faces notable limitations, and there is a lack of robust and user-friendly segmentation tools. This paper proposes a tongue image segmentation model (TOM) based on multi-teacher knowledge distillation. By incorporating a novel diffusion-based data augmentation method, we enhanced the generalization ability of the segmentation model while reducing its parameter size. Notably, after reducing the parameter count by 96.6% compared to the teacher models, the student model still achieves an impressive segmentation performance of 95.22% mIoU. Furthermore, we packaged and deployed the trained model as both an online and offline segmentation tool (available at this https URL), allowing TCM practitioners and researchers to use it without any programming experience. We also present a case study on TCM constitution classification using segmented tongue patches. Experimental results demonstrate that training with tongue patches yields higher classification performance and better interpretability than original tongue images. To our knowledge, this is the first open-source and freely available tongue image segmentation tool. 

---
# Heatmap Regression without Soft-Argmax for Facial Landmark Detection 

**Authors**: Chiao-An Yang, Raymond A. Yeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.14929)  

**Abstract**: Facial landmark detection is an important task in computer vision with numerous applications, such as head pose estimation, expression analysis, face swapping, etc. Heatmap regression-based methods have been widely used to achieve state-of-the-art results in this task. These methods involve computing the argmax over the heatmaps to predict a landmark. Since argmax is not differentiable, these methods use a differentiable approximation, Soft-argmax, to enable end-to-end training on deep-nets. In this work, we revisit this long-standing choice of using Soft-argmax and demonstrate that it is not the only way to achieve strong performance. Instead, we propose an alternative training objective based on the classic structured prediction framework. Empirically, our method achieves state-of-the-art performance on three facial landmark benchmarks (WFLW, COFW, and 300W), converging 2.2x faster during training while maintaining better/competitive accuracy. Our code is available here: this https URL. 

---
# AI Testing Should Account for Sophisticated Strategic Behaviour 

**Authors**: Vojtech Kovarik, Eric Olav Chen, Sami Petersen, Alexis Ghersengorin, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2508.14927)  

**Abstract**: This position paper argues for two claims regarding AI testing and evaluation. First, to remain informative about deployment behaviour, evaluations need account for the possibility that AI systems understand their circumstances and reason strategically. Second, game-theoretic analysis can inform evaluation design by formalising and scrutinising the reasoning in evaluation-based safety cases. Drawing on examples from existing AI systems, a review of relevant research, and formal strategic analysis of a stylised evaluation scenario, we present evidence for these claims and motivate several research directions. 

---
# Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving 

**Authors**: Dianzhao Li, Ostap Okhrin  

**Link**: [PDF](https://arxiv.org/pdf/2508.14926)  

**Abstract**: Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL in real-world scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy in complex, human-mixed traffic environments. 

---
# A U-Statistic-based random forest approach for genetic interaction study 

**Authors**: Ming Li, Ruo-Sin Peng, Changshuai Wei, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14924)  

**Abstract**: Variations in complex traits are influenced by multiple genetic variants, environmental risk factors, and their interactions. Though substantial progress has been made in identifying single genetic variants associated with complex traits, detecting the gene-gene and gene-environment interactions remains a great challenge. When a large number of genetic variants and environmental risk factors are involved, searching for interactions is limited to pair-wise interactions due to the exponentially increased feature space and computational intensity. Alternatively, recursive partitioning approaches, such as random forests, have gained popularity in high-dimensional genetic association studies. In this article, we propose a U-Statistic-based random forest approach, referred to as Forest U-Test, for genetic association studies with quantitative traits. Through simulation studies, we showed that the Forest U-Test outperformed existing methods. The proposed method was also applied to study Cannabis Dependence CD, using three independent datasets from the Study of Addiction: Genetics and Environment. A significant joint association was detected with an empirical p-value less than 0.001. The finding was also replicated in two independent datasets with p-values of 5.93e-19 and 4.70e-17, respectively. 

---
# Fusing Structural Phenotypes with Functional Data for Early Prediction of Primary Angle Closure Glaucoma Progression 

**Authors**: Swati Sharma, Thanadet Chuangsuwanich, Royston K.Y. Tan, Shimna C. Prasad, Tin A. Tun, Shamira A. Perera, Martin L. Buist, Tin Aung, Monisha E. Nongpiur, Michaël J. A. Girard  

**Link**: [PDF](https://arxiv.org/pdf/2508.14922)  

**Abstract**: Purpose: To classify eyes as slow or fast glaucoma progressors in patients with primary angle closure glaucoma (PACG) using an integrated approach combining optic nerve head (ONH) structural features and sector-based visual field (VF) functional parameters. Methods: PACG patients with >5 reliable VF tests over >5 years were included. Progression was assessed in Zeiss Forum, with baseline VF within six months of OCT. Fast progression was VFI decline <-2.0% per year; slow progression >-2.0% per year. OCT volumes were AI-segmented to extract 31 ONH parameters. The Glaucoma Hemifield Test defined five regions per hemifield, aligned with RNFL distribution. Mean sensitivity per region was combined with structural parameters to train ML classifiers. Multiple models were tested, and SHAP identified key predictors. Main outcome measures: Classification of slow versus fast progressors using combined structural and functional data. Results: We analyzed 451 eyes from 299 patients. Mean VFI progression was -0.92% per year; 369 eyes progressed slowly and 82 rapidly. The Random Forest model combining structural and functional features achieved the best performance (AUC = 0.87, 2000 Monte Carlo iterations). SHAP identified six key predictors: inferior MRW, inferior and inferior-temporal RNFL thickness, nasal-temporal LC curvature, superior nasal VF sensitivity, and inferior RNFL and GCL+IPL thickness. Models using only structural or functional features performed worse with AUC of 0.82 and 0.78, respectively. Conclusions: Combining ONH structural and VF functional parameters significantly improves classification of progression risk in PACG. Inferior ONH features, MRW and RNFL thickness, were the most predictive, highlighting the critical role of ONH morphology in monitoring disease progression. 

---
# Designing an Interdisciplinary Artificial Intelligence Curriculum for Engineering: Evaluation and Insights from Experts 

**Authors**: Johannes Schleiss, Anke Manukjan, Michelle Ines Bieber, Sebastian Lang, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2508.14921)  

**Abstract**: As Artificial Intelligence (AI) increasingly impacts professional practice, there is a growing need to AI-related competencies into higher education curricula. However, research on the implementation of AI education within study programs remains limited and requires new forms of collaboration across disciplines. This study addresses this gap and explores perspectives on interdisciplinary curriculum development through the lens of different stakeholders. In particular, we examine the case of curriculum development for a novel undergraduate program in AI in engineering. The research uses a mixed methods approach, combining quantitative curriculum mapping with qualitative focus group interviews. In addition to assessing the alignment of the curriculum with the targeted competencies, the study also examines the perceived quality, consistency, practicality and effectiveness from both academic and industry perspectives, as well as differences in perceptions between educators who were involved in the development and those who were not. The findings provide a practical understanding of the outcomes of interdisciplinary AI curriculum development and contribute to a broader understanding of how educator participation in curriculum development influences perceptions of quality aspects. It also advances the field of AI education by providing a reference point and insights for further interdisciplinary curriculum developments in response to evolving industry needs. 

---
# Disentangling the Drivers of LLM Social Conformity: An Uncertainty-Moderated Dual-Process Mechanism 

**Authors**: Huixin Zhong, Yanan Liu, Qi Cao, Shijin Wang, Zijing Ye, Zimu Wang, Shiyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14918)  

**Abstract**: As large language models (LLMs) integrate into collaborative teams, their social conformity -- the tendency to align with majority opinions -- has emerged as a key concern. In humans, conformity arises from informational influence (rational use of group cues for accuracy) or normative influence (social pressure for approval), with uncertainty moderating this balance by shifting from purely analytical to heuristic processing. It remains unclear whether these human psychological mechanisms apply to LLMs. This study adapts the information cascade paradigm from behavioral economics to quantitatively disentangle the two drivers to investigate the moderate effect. We evaluated nine leading LLMs across three decision-making scenarios (medical, legal, investment), manipulating information uncertainty (q = 0.667, 0.55, and 0.70, respectively). Our results indicate that informational influence underpins the models' behavior across all contexts, with accuracy and confidence consistently rising with stronger evidence. However, this foundational mechanism is dramatically modulated by uncertainty. In low-to-medium uncertainty scenarios, this informational process is expressed as a conservative strategy, where LLMs systematically underweight all evidence sources. In contrast, high uncertainty triggers a critical shift: while still processing information, the models additionally exhibit a normative-like amplification, causing them to overweight public signals (beta > 1.55 vs. private beta = 0.81). 

---
# Transsion Multilingual Speech Recognition System for MLC-SLM 2025 Challenge 

**Authors**: Xiaoxiao Li, An Zhu, Youhai Jiang, Fengjie Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14916)  

**Abstract**: This paper presents the architecture and performance of a novel Multilingual Automatic Speech Recognition (ASR) system developed by the Transsion Speech Team for Track 1 of the MLC-SLM 2025 Challenge. The proposed system comprises three key components: 1) a frozen Whisper-large-v3 based speech encoder, leveraging large-scale pretraining to ensure robust acoustic feature extraction; 2) a trainable adaptor module using Linear-ReLU-Linear transformation mechanisms to effectively align speech and text representations; and 3) a frozen Qwen2.5-7B-Instruct large language model (LLM) integrated with trainable LoRA for optimized contextual linguistic decoding. By systematically combining pretrained models with task specific fine-tuning, the system achieved a word/character error rate (WER/CER) of 9.83% across 11 languages in the evaluation set and ranked third place among global participants. 

---
# A Chinese Heart Failure Status Speech Database with Universal and Personalised Classification 

**Authors**: Yue Pan, Liwei Liu, Changxin Li, Xinyao Wang, Yili Xia, Hanyue Zhang, Ming Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14908)  

**Abstract**: Speech is a cost-effective and non-intrusive data source for identifying acute and chronic heart failure (HF). However, there is a lack of research on whether Chinese syllables contain HF-related information, as observed in other well-studied languages. This study presents the first Chinese speech database of HF patients, featuring paired recordings taken before and after hospitalisation. The findings confirm the effectiveness of the Chinese language in HF detection using both standard 'patient-wise' and personalised 'pair-wise' classification approaches, with the latter serving as an ideal speaker-decoupled baseline for future research. Statistical tests and classification results highlight individual differences as key contributors to inaccuracy. Additionally, an adaptive frequency filter (AFF) is proposed for frequency importance analysis. The data and demonstrations are published at this https URL. 

---
# Collaborative Filtering using Variational Quantum Hopfield Associative Memory 

**Authors**: Amir Kermanshahani, Ebrahim Ardeshir-Larijani, Rakesh Saini, Saif Al-Kuwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.14906)  

**Abstract**: Quantum computing, with its ability to do exponentially faster computation compared to classical systems, has found novel applications in various fields such as machine learning and recommendation systems. Quantum Machine Learning (QML), which integrates quantum computing with machine learning techniques, presents powerful new tools for data processing and pattern recognition. This paper proposes a hybrid recommendation system that combines Quantum Hopfield Associative Memory (QHAM) with deep neural networks to improve the extraction and classification on the MovieLens 1M dataset. User archetypes are clustered into multiple unique groups using the K-Means algorithm and converted into polar patterns through the encoder's activation function. These polar patterns are then integrated into the variational QHAM-based hybrid recommendation model. The system was trained using the MSE loss over 35 epochs in an ideal environment, achieving an ROC value of 0.9795, an accuracy of 0.8841, and an F-1 Score of 0.8786. Trained with the same number of epochs in a noisy environment using a custom Qiskit AER noise model incorporating bit-flip and readout errors with the same probabilities as in real quantum hardware, it achieves an ROC of 0.9177, an accuracy of 0.8013, and an F-1 Score equal to 0.7866, demonstrating consistent performance.
Additionally, we were able to optimize the qubit overhead present in previous QHAM architectures by efficiently updating only one random targeted qubit. This research presents a novel framework that combines variational quantum computing with deep learning, capable of dealing with real-world datasets with comparable performance compared to purely classical counterparts. Additionally, the model can perform similarly well in noisy configurations, showcasing a steady performance and proposing a promising direction for future usage in recommendation systems. 

---
# Privacy Preserving Inference of Personalized Content for Out of Matrix Users 

**Authors**: Michael Sun, Tai Vu, Andrew Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14905)  

**Abstract**: Recommender systems for niche and dynamic communities face persistent challenges from data sparsity, cold start users and items, and privacy constraints. Traditional collaborative filtering and content-based approaches underperform in these settings, either requiring invasive user data or failing when preference histories are absent. We present DeepNaniNet, a deep neural recommendation framework that addresses these challenges through an inductive graph-based architecture combining user-item interactions, item-item relations, and rich textual review embeddings derived from BERT. Our design enables cold start recommendations without profile mining, using a novel "content basket" user representation and an autoencoder-based generalization strategy for unseen users. We introduce AnimeULike, a new dataset of 10,000 anime titles and 13,000 users, to evaluate performance in realistic scenarios with high proportions of guest or low-activity users. DeepNaniNet achieves state-of-the-art cold start results on the CiteULike benchmark, matches DropoutNet in user recall without performance degradation for out-of-matrix users, and outperforms Weighted Matrix Factorization (WMF) and DropoutNet on AnimeULike warm start by up to 7x and 1.5x in Recall@100, respectively. Our findings demonstrate that DeepNaniNet delivers high-quality, privacy-preserving recommendations in data-sparse, cold start-heavy environments while effectively integrating heterogeneous content sources. 

---
# Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training 

**Authors**: Jianfeng Si, Lin Sun, Zhewen Tan, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14904)  

**Abstract**: Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety. 

---
# Accelerating GenAI Workloads by Enabling RISC-V Microkernel Support in IREE 

**Authors**: Adeel Ahmad, Ahmad Tameem Kamal, Nouman Amir, Bilal Zafar, Saad Bin Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2508.14899)  

**Abstract**: This project enables RISC-V microkernel support in IREE, an MLIR-based machine learning compiler and runtime. The approach begins by enabling the lowering of MLIR linalg dialect contraction ops to linalg.mmt4d op for the RISC-V64 target within the IREE pass pipeline, followed by the development of optimized microkernels for RISC-V. The performance gains are compared with upstream IREE and this http URL for the Llama-3.2-1B-Instruct model. 

---
# The Impact of Image Resolution on Face Detection: A Comparative Analysis of MTCNN, YOLOv XI and YOLOv XII models 

**Authors**: Ahmet Can Ömercikoğlu, Mustafa Mansur Yönügül, Pakize Erdoğmuş  

**Link**: [PDF](https://arxiv.org/pdf/2507.23341)  

**Abstract**: Face detection is a crucial component in many AI-driven applications such as surveillance, biometric authentication, and human-computer interaction. However, real-world conditions like low-resolution imagery present significant challenges that degrade detection performance. In this study, we systematically investigate the impact of input resolution on the accuracy and robustness of three prominent deep learning-based face detectors: YOLOv11, YOLOv12, and MTCNN. Using the WIDER FACE dataset, we conduct extensive evaluations across multiple image resolutions (160x160, 320x320, and 640x640) and assess each model's performance using metrics such as precision, recall, mAP50, mAP50-95, and inference time. Results indicate that YOLOv11 outperforms YOLOv12 and MTCNN in terms of detection accuracy, especially at higher resolutions, while YOLOv12 exhibits slightly better recall. MTCNN, although competitive in landmark localization, lags in real-time inference speed. Our findings provide actionable insights for selecting resolution-aware face detection models suitable for varying operational constraints. 

---
# SVM/SVR Kernels as Quantum Propagators 

**Authors**: Nan-Hong Kuo, Renata Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11153)  

**Abstract**: We establish a mathematical equivalence between Support Vector Machine (SVM) kernel functions and quantum propagators represented by time-dependent Green's functions, which has remained largely unexplored.
We demonstrate that many common SVM kernels correspond naturally to Green's functions via operator inversion theory. The sigmoid kernel does not always satisfy Mercer's theorem, and therefore the corresponding Green's function may also fail to perform optimally.
We further introduce a Kernel Polynomial Method (KPM) for designing customized kernels that align with Green's functions.
Our numerical experiments confirm that employing positive-semidefinite kernels that correspond to Green's functions significantly improves predictive accuracy of SVM models in physical systems. 

---
