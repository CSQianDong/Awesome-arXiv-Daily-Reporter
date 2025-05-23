# X-MAS: Towards Building Multi-Agent Systems with Heterogeneous LLMs 

**Authors**: Rui Ye, Xiangrui Liu, Qimin Wu, Xianghe Pang, Zhenfei Yin, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16997)  

**Abstract**: LLM-based multi-agent systems (MAS) extend the capabilities of single LLMs by enabling cooperation among multiple specialized agents. However, most existing MAS frameworks rely on a single LLM to drive all agents, constraining the system's intelligence to the limit of that model. This paper explores the paradigm of heterogeneous LLM-driven MAS (X-MAS), where agents are powered by diverse LLMs, elevating the system's potential to the collective intelligence of diverse LLMs. We introduce X-MAS-Bench, a comprehensive testbed designed to evaluate the performance of various LLMs across different domains and MAS-related functions. As an extensive empirical study, we assess 27 LLMs across 5 domains (encompassing 21 test sets) and 5 functions, conducting over 1.7 million evaluations to identify optimal model selections for each domain-function combination. Building on these findings, we demonstrate that transitioning from homogeneous to heterogeneous LLM-driven MAS can significantly enhance system performance without requiring structural redesign. Specifically, in a chatbot-only MAS scenario, the heterogeneous configuration yields up to 8.4\% performance improvement on the MATH dataset. In a mixed chatbot-reasoner scenario, the heterogeneous MAS could achieve a remarkable 47\% performance boost on the AIME dataset. Our results underscore the transformative potential of heterogeneous LLMs in MAS, highlighting a promising avenue for advancing scalable, collaborative AI systems. 

---
# Beyond Correlation: Towards Causal Large Language Model Agents in Biomedicine 

**Authors**: Adib Bazgir, Amir Habibdoust Lafmajani, Yuwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16982)  

**Abstract**: Large Language Models (LLMs) show promise in biomedicine but lack true causal understanding, relying instead on correlations. This paper envisions causal LLM agents that integrate multimodal data (text, images, genomics, etc.) and perform intervention-based reasoning to infer cause-and-effect. Addressing this requires overcoming key challenges: designing safe, controllable agentic frameworks; developing rigorous benchmarks for causal evaluation; integrating heterogeneous data sources; and synergistically combining LLMs with structured knowledge (KGs) and formal causal inference tools. Such agents could unlock transformative opportunities, including accelerating drug discovery through automated hypothesis generation and simulation, enabling personalized medicine through patient-specific causal models. This research agenda aims to foster interdisciplinary efforts, bridging causal concepts and foundation models to develop reliable AI partners for biomedical progress. 

---
# Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design 

**Authors**: Zhenkun Li, Lingyao Li, Shuhang Lin, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16979)  

**Abstract**: Single-agent LLMs hit hard limits--finite context, role overload, and brittle domain transfer. Conventional multi-agent fixes soften those edges yet expose fresh pains: ill-posed decompositions, fuzzy contracts, and verification overhead that blunts the gains. We therefore present Know-The-Ropes (KtR), a framework that converts domain priors into an algorithmic blueprint hierarchy, in which tasks are recursively split into typed, controller-mediated subtasks, each solved zero-shot or with the lightest viable boost (e.g., chain-of-thought, micro-tune, self-check). Grounded in the No-Free-Lunch theorem, KtR trades the chase for a universal prompt for disciplined decomposition. On the Knapsack problem (3-8 items), three GPT-4o-mini agents raise accuracy from 3% zero-shot to 95% on size-5 instances after patching a single bottleneck agent. On the tougher Task-Assignment problem (6-15 jobs), a six-agent o3-mini blueprint hits 100% up to size 10 and 84% on sizes 13-15, versus 11% zero-shot. Algorithm-aware decomposition plus targeted augmentation thus turns modest models into reliable collaborators--no ever-larger monoliths required. 

---
# HyGenar: An LLM-Driven Hybrid Genetic Algorithm for Few-Shot Grammar Generation 

**Authors**: Weizhi Tang, Yixuan Li, Chris Sypherd, Elizabeth Polgreen, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2505.16978)  

**Abstract**: Grammar plays a critical role in natural language processing and text/code generation by enabling the definition of syntax, the creation of parsers, and guiding structured outputs. Although large language models (LLMs) demonstrate impressive capabilities across domains, their ability to infer and generate grammars has not yet been thoroughly explored. In this paper, we aim to study and improve the ability of LLMs for few-shot grammar generation, where grammars are inferred from sets of a small number of positive and negative examples and generated in Backus-Naur Form. To explore this, we introduced a novel dataset comprising 540 structured grammar generation challenges, devised 6 metrics, and evaluated 8 various LLMs against it. Our findings reveal that existing LLMs perform sub-optimally in grammar generation. To address this, we propose an LLM-driven hybrid genetic algorithm, namely HyGenar, to optimize grammar generation. HyGenar achieves substantial improvements in both the syntactic and semantic correctness of generated grammars across LLMs. 

---
# AGENTIF: Benchmarking Instruction Following of Large Language Models in Agentic Scenarios 

**Authors**: Yunjia Qi, Hao Peng, Xiaozhi Wang, Amy Xin, Youfeng Liu, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16944)  

**Abstract**: Large Language Models (LLMs) have demonstrated advanced capabilities in real-world agentic applications. Growing research efforts aim to develop LLM-based agents to address practical demands, introducing a new challenge: agentic scenarios often involve lengthy instructions with complex constraints, such as extended system prompts and detailed tool specifications. While adherence to such instructions is crucial for agentic applications, whether LLMs can reliably follow them remains underexplored. In this paper, we introduce AgentIF, the first benchmark for systematically evaluating LLM instruction following ability in agentic scenarios. AgentIF features three key characteristics: (1) Realistic, constructed from 50 real-world agentic applications. (2) Long, averaging 1,723 words with a maximum of 15,630 words. (3) Complex, averaging 11.9 constraints per instruction, covering diverse constraint types, such as tool specifications and condition constraints. To construct AgentIF, we collect 707 human-annotated instructions across 50 agentic tasks from industrial application agents and open-source agentic systems. For each instruction, we annotate the associated constraints and corresponding evaluation metrics, including code-based evaluation, LLM-based evaluation, and hybrid code-LLM evaluation. We use AgentIF to systematically evaluate existing advanced LLMs. We observe that current models generally perform poorly, especially in handling complex constraint structures and tool specifications. We further conduct error analysis and analytical experiments on instruction length and meta constraints, providing some findings about the failure modes of existing LLMs. We have released the code and data to facilitate future research. 

---
# NovelSeek: When Agent Becomes the Scientist -- Building Closed-Loop System from Hypothesis to Verification 

**Authors**: NovelSeek Team, Bo Zhang, Shiyang Feng, Xiangchao Yan, Jiakang Yuan, Zhiyin Yu, Xiaohan He, Songtao Huang, Shaowei Hou, Zheng Nie, Zhilong Wang, Jinyao Liu, Runmin Ma, Tianshuo Peng, Peng Ye, Dongzhan Zhou, Shufei Zhang, Xiaosong Wang, Yilan Zhang, Meng Li, Zhongying Tu, Xiangyu Yue, Wangli Ouyang, Bowen Zhou, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.16938)  

**Abstract**: Artificial Intelligence (AI) is accelerating the transformation of scientific research paradigms, not only enhancing research efficiency but also driving innovation. We introduce NovelSeek, a unified closed-loop multi-agent framework to conduct Autonomous Scientific Research (ASR) across various scientific research fields, enabling researchers to tackle complicated problems in these fields with unprecedented speed and precision. NovelSeek highlights three key advantages: 1) Scalability: NovelSeek has demonstrated its versatility across 12 scientific research tasks, capable of generating innovative ideas to enhance the performance of baseline code. 2) Interactivity: NovelSeek provides an interface for human expert feedback and multi-agent interaction in automated end-to-end processes, allowing for the seamless integration of domain expert knowledge. 3) Efficiency: NovelSeek has achieved promising performance gains in several scientific fields with significantly less time cost compared to human efforts. For instance, in reaction yield prediction, it increased from 27.6% to 35.4% in just 12 hours; in enhancer activity prediction, accuracy rose from 0.52 to 0.79 with only 4 hours of processing; and in 2D semantic segmentation, precision advanced from 78.8% to 81.0% in a mere 30 hours. 

---
# Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning 

**Authors**: Bosung Kim, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16928)  

**Abstract**: We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning. 

---
# Identifying, Evaluating, and Mitigating Risks of AI Thought Partnerships 

**Authors**: Kerem Oktar, Katherine M. Collins, Jose Hernandez-Orallo, Diane Coyle, Stephen Cave, Adrian Weller, Ilia Sucholutsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.16899)  

**Abstract**: Artificial Intelligence (AI) systems have historically been used as tools that execute narrowly defined tasks. Yet recent advances in AI have unlocked possibilities for a new class of models that genuinely collaborate with humans in complex reasoning, from conceptualizing problems to brainstorming solutions. Such AI thought partners enable novel forms of collaboration and extended cognition, yet they also pose major risks-including and beyond risks of typical AI tools and agents. In this commentary, we systematically identify risks of AI thought partners through a novel framework that identifies risks at multiple levels of analysis, including Real-time, Individual, and Societal risks arising from collaborative cognition (RISc). We leverage this framework to propose concrete metrics for risk evaluation, and finally suggest specific mitigation strategies for developers and policymakers. As AI thought partners continue to proliferate, these strategies can help prevent major harms and ensure that humans actively benefit from productive thought partnerships. 

---
# Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings 

**Authors**: Yuqicheng Zhu, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2505.16877)  

**Abstract**: Uncertainty quantification in Knowledge Graph Embedding (KGE) methods is crucial for ensuring the reliability of downstream applications. A recent work applies conformal prediction to KGE methods, providing uncertainty estimates by generating a set of answers that is guaranteed to include the true answer with a predefined confidence level. However, existing methods provide probabilistic guarantees averaged over a reference set of queries and answers (marginal coverage guarantee). In high-stakes applications such as medical diagnosis, a stronger guarantee is often required: the predicted sets must provide consistent coverage per query (conditional coverage guarantee). We propose CondKGCP, a novel method that approximates predicate-conditional coverage guarantees while maintaining compact prediction sets. CondKGCP merges predicates with similar vector representations and augments calibration with rank information. We prove the theoretical guarantees and demonstrate empirical effectiveness of CondKGCP by comprehensive evaluations. 

---
# Think or Not? Selective Reasoning via Reinforcement Learning for Vision-Language Models 

**Authors**: Jiaqi Wang, Kevin Qinghong Lin, James Cheng, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16854)  

**Abstract**: Reinforcement Learning (RL) has proven to be an effective post-training strategy for enhancing reasoning in vision-language models (VLMs). Group Relative Policy Optimization (GRPO) is a recent prominent method that encourages models to generate complete reasoning traces before answering, leading to increased token usage and computational cost. Inspired by the human-like thinking process-where people skip reasoning for easy questions but think carefully when needed-we explore how to enable VLMs to first decide when reasoning is necessary. To realize this, we propose TON, a two-stage training strategy: (i) a supervised fine-tuning (SFT) stage with a simple yet effective 'thought dropout' operation, where reasoning traces are randomly replaced with empty thoughts. This introduces a think-or-not format that serves as a cold start for selective reasoning; (ii) a GRPO stage that enables the model to freely explore when to think or not, while maximizing task-aware outcome rewards. Experimental results show that TON can reduce the completion length by up to 90% compared to vanilla GRPO, without sacrificing performance or even improving it. Further evaluations across diverse vision-language tasks-covering a range of reasoning difficulties under both 3B and 7B models-consistently reveal that the model progressively learns to bypass unnecessary reasoning steps as training advances. These findings shed light on the path toward human-like reasoning patterns in reinforcement learning approaches. Our code is available at this https URL. 

---
# From EduVisBench to EduVisAgent: A Benchmark and Multi-Agent Framework for Pedagogical Visualization 

**Authors**: Haonian Ji, Shi Qiu, Siyang Xin, Siwei Han, Zhaorun Chen, Hongyi Wang, Dake Zhang, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16832)  

**Abstract**: While foundation models (FMs), such as diffusion models and large vision-language models (LVLMs), have been widely applied in educational contexts, their ability to generate pedagogically effective visual explanations remains limited. Most existing approaches focus primarily on textual reasoning, overlooking the critical role of structured and interpretable visualizations in supporting conceptual understanding. To better assess the visual reasoning capabilities of FMs in educational settings, we introduce EduVisBench, a multi-domain, multi-level benchmark. EduVisBench features diverse STEM problem sets requiring visually grounded solutions, along with a fine-grained evaluation rubric informed by pedagogical theory. Our empirical analysis reveals that existing models frequently struggle with the inherent challenge of decomposing complex reasoning and translating it into visual representations aligned with human cognitive processes. To address these limitations, we propose EduVisAgent, a multi-agent collaborative framework that coordinates specialized agents for instructional planning, reasoning decomposition, metacognitive prompting, and visualization design. Experimental results show that EduVisAgent substantially outperforms all baselines, achieving a 40.2% improvement and delivering more educationally aligned visualizations. EduVisBench and EduVisAgent are available at this https URL and this https URL. 

---
# GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent 

**Authors**: Bin Xie, Rui Shao, Gongwei Chen, Kaiwen Zhou, Yinchuan Li, Jie Liu, Min Zhang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16827)  

**Abstract**: GUI automation faces critical challenges in dynamic environments. MLLMs suffer from two key issues: misinterpreting UI components and outdated knowledge. Traditional fine-tuning methods are costly for app-specific knowledge updates. We propose GUI-explorer, a training-free GUI agent that incorporates two fundamental mechanisms: (1) Autonomous Exploration of Function-aware Trajectory. To comprehensively cover all application functionalities, we design a Function-aware Task Goal Generator that automatically constructs exploration goals by analyzing GUI structural information (e.g., screenshots and activity hierarchies). This enables systematic exploration to collect diverse trajectories. (2) Unsupervised Mining of Transition-aware Knowledge. To establish precise screen-operation logic, we develop a Transition-aware Knowledge Extractor that extracts effective screen-operation logic through unsupervised analysis the state transition of structured interaction triples (observation, action, outcome). This eliminates the need for human involvement in knowledge extraction. With a task success rate of 53.7% on SPA-Bench and 47.4% on AndroidWorld, GUI-explorer shows significant improvements over SOTA agents. It requires no parameter updates for new apps. GUI-explorer is open-sourced and publicly available at this https URL. 

---
# KTAE: A Model-Free Algorithm to Key-Tokens Advantage Estimation in Mathematical Reasoning 

**Authors**: Wei Sun, Wen Yang, Pu Jian, Qianlong Du, Fuwei Cui, Shuo Ren, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16826)  

**Abstract**: Recent advances have demonstrated that integrating reinforcement learning with rule-based rewards can significantly enhance the reasoning capabilities of large language models, even without supervised fine-tuning. However, prevalent reinforcement learning algorithms such as GRPO and its variants like DAPO, suffer from a coarse granularity issue when computing the advantage. Specifically, they compute rollout-level advantages that assign identical values to every token within a sequence, failing to capture token-specific contributions and hindering effective learning. To address this limitation, we propose Key-token Advantage Estimation (KTAE) - a novel algorithm that estimates fine-grained, token-level advantages without introducing additional models. KTAE leverages the correctness of sampled rollouts and applies statistical analysis to quantify the importance of individual tokens within a sequence to the final outcome. This quantified token-level importance is then combined with the rollout-level advantage to obtain a more fine-grained token-level advantage estimation. Empirical results show that models trained with GRPO+KTAE and DAPO+KTAE outperform baseline methods across five mathematical reasoning benchmarks. Notably, they achieve higher accuracy with shorter responses and even surpass R1-Distill-Qwen-1.5B using the same base model. 

---
# Gaze Into the Abyss -- Planning to Seek Entropy When Reward is Scarce 

**Authors**: Ashish Sundar, Chunbo Luo, Xiaoyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16787)  

**Abstract**: Model-based reinforcement learning (MBRL) offers an intuitive way to increase the sample efficiency of model-free RL methods by simultaneously training a world model that learns to predict the future. MBRL methods have progressed by largely prioritising the actor; optimising the world model learning has been neglected meanwhile. Improving the fidelity of the world model and reducing its time to convergence can yield significant downstream benefits, one of which is improving the ensuing performance of any actor it may train. We propose a novel approach that anticipates and actively seeks out high-entropy states using short-horizon latent predictions generated by the world model, offering a principled alternative to traditional curiosity-driven methods that chase once-novel states well after they were stumbled into. While many model predictive control (MPC) based methods offer similar alternatives, they typically lack commitment, synthesising multi step plans after every step. To mitigate this, we present a hierarchical planner that dynamically decides when to replan, planning horizon length, and the weighting between reward and entropy. While our method can theoretically be applied to any model that trains its own actors with solely model generated data, we have applied it to just Dreamer as a proof of concept. Our method finishes the Miniworld procedurally generated mazes 50% faster than base Dreamer at convergence and the policy trained in imagination converges in only 60% of the environment steps that base Dreamer needs. 

---
# Fuzzy Information Evolution with Three-Way Decision in Social Network Group Decision-Making 

**Authors**: Qianlei Jia, Xinliang Zhou, Ondrej Krejcar, Enrique Herrera-Viedma  

**Link**: [PDF](https://arxiv.org/pdf/2505.16781)  

**Abstract**: In group decision-making (GDM) scenarios, uncertainty, dynamic social structures, and vague information present major challenges for traditional opinion dynamics models. To address these issues, this study proposes a novel social network group decision-making (SNGDM) framework that integrates three-way decision (3WD) theory, dynamic network reconstruction, and linguistic opinion representation. First, the 3WD mechanism is introduced to explicitly model hesitation and ambiguity in agent judgments, thereby preventing irrational decisions. Second, a connection adjustment rule based on opinion similarity is developed, enabling agents to adaptively update their communication links and better reflect the evolving nature of social relationships. Third, linguistic terms are used to describe agent opinions, allowing the model to handle subjective, vague, or incomplete information more effectively. Finally, an integrated multi-agent decision-making framework is constructed, which simultaneously considers individual uncertainty, opinion evolution, and network dynamics. The proposed model is applied to a multi-UAV cooperative decision-making scenario, where simulation results and consensus analysis demonstrate its effectiveness. Experimental comparisons further verify the advantages of the algorithm in enhancing system stability and representing realistic decision-making behaviors. 

---
# Data-Driven Breakthroughs and Future Directions in AI Infrastructure: A Comprehensive Review 

**Authors**: Beyazit Bestami Yuksel, Ayse Yilmazer Metin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16771)  

**Abstract**: This paper presents a comprehensive synthesis of major breakthroughs in artificial intelligence (AI) over the past fifteen years, integrating historical, theoretical, and technological perspectives. It identifies key inflection points in AI' s evolution by tracing the convergence of computational resources, data access, and algorithmic innovation. The analysis highlights how researchers enabled GPU based model training, triggered a data centric shift with ImageNet, simplified architectures through the Transformer, and expanded modeling capabilities with the GPT series. Rather than treating these advances as isolated milestones, the paper frames them as indicators of deeper paradigm shifts. By applying concepts from statistical learning theory such as sample complexity and data efficiency, the paper explains how researchers translated breakthroughs into scalable solutions and why the field must now embrace data centric approaches. In response to rising privacy concerns and tightening regulations, the paper evaluates emerging solutions like federated learning, privacy enhancing technologies (PETs), and the data site paradigm, which reframe data access and security. In cases where real world data remains inaccessible, the paper also assesses the utility and constraints of mock and synthetic data generation. By aligning technical insights with evolving data infrastructure, this study offers strategic guidance for future AI research and policy development. 

---
# MCP-RADAR: A Multi-Dimensional Benchmark for Evaluating Tool Use Capabilities in Large Language Models 

**Authors**: Xuanqi Gao, Siyi Xie, Juan Zhai, Shqing Ma, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16700)  

**Abstract**: As Large Language Models (LLMs) evolve from passive text generators to active reasoning agents capable of tool interaction, the Model Context Protocol (MCP) has emerged as a standardized framework for dynamic tool discovery and orchestration. Despite widespread industry adoption, existing evaluation methodologies fail to adequately assess tool utilization capabilities within this new paradigm. This paper introduces MCP-RADAR, the first comprehensive benchmark specifically designed to evaluate LLM performance in the MCP framework through a novel five-dimensional approach measuring: answer accuracy, tool selection efficiency, computational resource efficiency, parameter construction accuracy, and execution speed. Unlike conventional benchmarks that rely on subjective human evaluations or binary success metrics, MCP-RADAR employs objective, quantifiable measurements across multiple task domains including software engineering, mathematical reasoning, and general problem-solving. Our evaluations of leading commercial and open-source LLMs reveal distinctive capability profiles with significant trade-offs between accuracy, efficiency, and speed, challenging traditional single-metric performance rankings. Besides, we provide valuable guidance for developers to optimize their tools for maximum model compatibility and effectiveness. While focused on MCP due to its standardized approach, our methodology remains applicable across all LLM agent tool integration frameworks, providing valuable insights for both LLM developers and tool creators to optimize the entire LLM-tool interaction ecosystem. The implementation, configurations, and datasets used in our evaluation are publicly available at this https URL. 

---
# SPaRC: A Spatial Pathfinding Reasoning Challenge 

**Authors**: Lars Benedikt Kaesberg, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2505.16686)  

**Abstract**: Existing reasoning datasets saturate and fail to test abstract, multi-step problems, especially pathfinding and complex rule constraint satisfaction. We introduce SPaRC (Spatial Pathfinding Reasoning Challenge), a dataset of 1,000 2D grid pathfinding puzzles to evaluate spatial and symbolic reasoning, requiring step-by-step planning with arithmetic and geometric rules. Humans achieve near-perfect accuracy (98.0%; 94.5% on hard puzzles), while the best reasoning models, such as o4-mini, struggle (15.8%; 1.1% on hard puzzles). Models often generate invalid paths (>50% of puzzles for o4-mini), and reasoning tokens reveal they make errors in navigation and spatial logic. Unlike humans, who take longer on hard puzzles, models fail to scale test-time compute with difficulty. Allowing models to make multiple solution attempts improves accuracy, suggesting potential for better spatial reasoning with improved training and efficient test-time scaling methods. SPaRC can be used as a window into models' spatial reasoning limitations and drive research toward new methods that excel in abstract, multi-step problem-solving. 

---
# ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming 

**Authors**: Xinwei Yang, Zhaofeng Liu, Chen Huang, Jiashuai Zhang, Tong Zhang, Yifan Zhang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16667)  

**Abstract**: While recent research increasingly emphasizes the value of human-LLM collaboration in competitive programming and proposes numerous empirical methods, a comprehensive understanding remains elusive due to the fragmented nature of existing studies and their use of diverse, application-specific human feedback. Thus, our work serves a three-fold purpose: First, we present the first taxonomy of human feedback consolidating the entire programming process, which promotes fine-grained evaluation. Second, we introduce ELABORATIONSET, a novel programming dataset specifically designed for human-LLM collaboration, meticulously annotated to enable large-scale simulated human feedback and facilitate costeffective real human interaction studies. Third, we introduce ELABORATION, a novel benchmark to facilitate a thorough assessment of human-LLM competitive programming. With ELABORATION, we pinpoint strengthes and weaknesses of existing methods, thereby setting the foundation for future improvement. Our code and dataset are available at this https URL 

---
# SMART: Self-Generating and Self-Validating Multi-Dimensional Assessment for LLMs' Mathematical Problem Solving 

**Authors**: Yujie Hou, Ting Zhang, Mei Wang, Xuetao Ma, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16646)  

**Abstract**: Large Language Models have achieved remarkable results on a variety of mathematical benchmarks. However, concerns remain as to whether these successes reflect genuine mathematical reasoning or superficial pattern recognition. Common evaluation metrics, such as final answer accuracy, fail to disentangle the underlying competencies involved, offering limited diagnostic value. To address these limitations, we introduce SMART: a Self-Generating and Self-Validating Multi-Dimensional Assessment Framework. SMART decomposes mathematical problem solving into four distinct dimensions: understanding, reasoning, arithmetic, and reflection \& refinement. Each dimension is evaluated independently through tailored tasks, enabling interpretable and fine-grained analysis of LLM behavior. Crucially, SMART integrates an automated self-generating and self-validating mechanism to produce and verify benchmark data, ensuring both scalability and reliability. We apply SMART to 21 state-of-the-art open- and closed-source LLMs, uncovering significant discrepancies in their abilities across different dimensions. Our findings demonstrate the inadequacy of final answer accuracy as a sole metric and motivate a new holistic metric to better capture true problem-solving capabilities. Code and benchmarks will be released upon acceptance. 

---
# Open and Sustainable AI: challenges, opportunities and the road ahead in the life sciences 

**Authors**: Gavin Farrell, Eleni Adamidi, Rafael Andrade Buono, Mihail Anton, Omar Abdelghani Attafi, Salvador Capella Gutierrez, Emidio Capriotti, Leyla Jael Castro, Davide Cirillo, Lisa Crossman, Christophe Dessimoz, Alexandros Dimopoulos, Raul Fernandez-Diaz, Styliani-Christina Fragkouli, Carole Goble, Wei Gu, John M. Hancock, Alireza Khanteymoori, Tom Lenaerts, Fabio G. Liberante, Peter Maccallum, Alexander Miguel Monzon, Magnus Palmblad, Lucy Poveda, Ovidiu Radulescu, Denis C. Shields, Shoaib Sufi, Thanasis Vergoulis, Fotis Psomopoulos, Silvio C.E. Tosatto  

**Link**: [PDF](https://arxiv.org/pdf/2505.16619)  

**Abstract**: Artificial intelligence (AI) has recently seen transformative breakthroughs in the life sciences, expanding possibilities for researchers to interpret biological information at an unprecedented capacity, with novel applications and advances being made almost daily. In order to maximise return on the growing investments in AI-based life science research and accelerate this progress, it has become urgent to address the exacerbation of long-standing research challenges arising from the rapid adoption of AI methods. We review the increased erosion of trust in AI research outputs, driven by the issues of poor reusability and reproducibility, and highlight their consequent impact on environmental sustainability. Furthermore, we discuss the fragmented components of the AI ecosystem and lack of guiding pathways to best support Open and Sustainable AI (OSAI) model development. In response, this perspective introduces a practical set of OSAI recommendations directly mapped to over 300 components of the AI ecosystem. Our work connects researchers with relevant AI resources, facilitating the implementation of sustainable, reusable and transparent AI. Built upon life science community consensus and aligned to existing efforts, the outputs of this perspective are designed to aid the future development of policy and structured pathways for guiding AI implementation. 

---
# Bridging the Dynamic Perception Gap: Training-Free Draft Chain-of-Thought for Dynamic Multimodal Spatial Reasoning 

**Authors**: Siqu Ou, Hongcheng Liu, Pingjie Wang, Yusheng Liao, Chuan Xuan, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16579)  

**Abstract**: While chains-of-thought (CoT) have advanced complex reasoning in multimodal large language models (MLLMs), existing methods remain confined to text or static visual domains, often faltering in dynamic spatial reasoning tasks. To bridge this gap, we present GRASSLAND, a novel maze navigation benchmark designed to evaluate dynamic spatial reasoning. Our experiments show that augmenting textual reasoning chains with dynamic visual drafts, overlaid on input images, significantly outperforms conventional approaches, offering new insights into spatial reasoning in evolving environments. To generalize this capability, we propose D2R (Dynamic Draft-Augmented Reasoning), a training-free framework that seamlessly integrates textual CoT with corresponding visual drafts into MLLMs. Extensive evaluations demonstrate that D2R consistently enhances performance across diverse tasks, establishing a robust baseline for dynamic spatial reasoning without requiring model fine-tuning. Project is open at this https URL. 

---
# Relevance for Stability of Verification Status of a Set of Arguments in Incomplete Argumentation Frameworks (with Proofs) 

**Authors**: Anshu Xiong, Songmao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16507)  

**Abstract**: The notion of relevance was proposed for stability of justification status of a single argument in incomplete argumentation frameworks (IAFs) in 2024 by Odekerken et al. To extend the notion, we study the relevance for stability of verification status of a set of arguments in this paper, i.e., the uncertainties in an IAF that have to be resolved in some situations so that answering whether a given set of arguments is an extension obtains the same result in every completion of the IAF. Further we propose the notion of strong relevance for describing the necessity of resolution in all situations reaching stability. An analysis of complexity reveals that detecting the (strong) relevance for stability of sets of arguments can be accomplished in P time under the most semantics discussed in the paper. We also discuss the difficulty in finding tractable methods for relevance detection under grounded semantics. 

---
# Minimizing the energy depletion in wireless rechargeable sensor networks using bi-level metaheuristic charging schemes 

**Authors**: Huynh Thi Thanh Binh, Le Van Cuong, Dang Hai Dang, Le Trong Vinh  

**Link**: [PDF](https://arxiv.org/pdf/2505.16482)  

**Abstract**: Recently, Wireless Rechargeable Sensor Networks (WRSNs) that leveraged the advantage of wireless energy transfer technology have opened a promising opportunity in solving the limited energy issue. However, an ineffective charging strategy may reduce the charging performance. Although many practical charging algorithms have been introduced, these studies mainly focus on optimizing the charging path with a fully charging approach. This approach may lead to the death of a series of sensors due to their extended charging latency. This paper introduces a novel partial charging approach that follows a bi-level optimized scheme to minimize energy depletion in WRSNs. We aim at optimizing simultaneously two factors: the charging path and time. To accomplish this, we first formulate a mathematical model of the investigated problem. We then propose two approximate algorithms in which the optimization of the charging path and the charging time are considered as the upper and lower level, respectively. The first algorithm combines a Multi-start Local Search method and a Genetic Algorithm to find a solution. The second algorithm adopts a nested approach that utilizes the advantages of the Multitasking and Covariance Matrix Adaptation Evolutionary Strategies. Experimental validations on various network scenarios demonstrate that our proposed algorithms outperform the existing works. 

---
# Advancing the Scientific Method with Large Language Models: From Hypothesis to Discovery 

**Authors**: Yanbo Zhang, Sumeer A. Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared Dunnmon, James Evans, Alan Bundy, Saso Dzeroski, Jesper Tegner, Hector Zenil  

**Link**: [PDF](https://arxiv.org/pdf/2505.16477)  

**Abstract**: With recent Nobel Prizes recognising AI contributions to science, Large Language Models (LLMs) are transforming scientific research by enhancing productivity and reshaping the scientific method. LLMs are now involved in experimental design, data analysis, and workflows, particularly in chemistry and biology. However, challenges such as hallucinations and reliability persist. In this contribution, we review how Large Language Models (LLMs) are redefining the scientific method and explore their potential applications across different stages of the scientific cycle, from hypothesis testing to discovery. We conclude that, for LLMs to serve as relevant and effective creative engines and productivity enhancers, their deep integration into all steps of the scientific process should be pursued in collaboration and alignment with human scientific goals, with clear evaluation metrics. The transition to AI-driven science raises ethical questions about creativity, oversight, and responsibility. With careful guidance, LLMs could evolve into creative engines, driving transformative breakthroughs across scientific disciplines responsibly and effectively. However, the scientific community must also decide how much it leaves to LLMs to drive science, even when associations with 'reasoning', mostly currently undeserved, are made in exchange for the potential to explore hypothesis and solution regions that might otherwise remain unexplored by human exploration alone. 

---
# ReflectEvo: Improving Meta Introspection of Small LLMs by Learning Self-Reflection 

**Authors**: Jiaqi Li, Xinyi Dong, Yang Liu, Zhizhuo Yang, Quansen Wang, Xiaobo Wang, SongChun Zhu, Zixia Jia, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.16475)  

**Abstract**: We present a novel pipeline, ReflectEvo, to demonstrate that small language models (SLMs) can enhance meta introspection through reflection learning. This process iteratively generates self-reflection for self-training, fostering a continuous and self-evolving process. Leveraging this pipeline, we construct ReflectEvo-460k, a large-scale, comprehensive, self-generated reflection dataset with broadened instructions and diverse multi-domain tasks. Building upon this dataset, we demonstrate the effectiveness of reflection learning to improve SLMs' reasoning abilities using SFT and DPO with remarkable performance, substantially boosting Llama-3 from 52.4% to 71.2% and Mistral from 44.4% to 71.1%. It validates that ReflectEvo can rival or even surpass the reasoning capability of the three prominent open-sourced models on BIG-bench without distillation from superior models or fine-grained human annotation. We further conduct a deeper analysis of the high quality of self-generated reflections and their impact on error localization and correction. Our work highlights the potential of continuously enhancing the reasoning performance of SLMs through iterative reflection learning in the long run. 

---
# MMMR: Benchmarking Massive Multi-Modal Reasoning Tasks 

**Authors**: Guiyao Tie, Xueyang Zhou, Tianhe Gu, Ruihang Zhang, Chaoran Hu, Sizhe Zhang, Mengqu Sun, Yan Zhang, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16459)  

**Abstract**: Recent advances in Multi-Modal Large Language Models (MLLMs) have enabled unified processing of language, vision, and structured inputs, opening the door to complex tasks such as logical deduction, spatial reasoning, and scientific analysis. Despite their promise, the reasoning capabilities of MLLMs, particularly those augmented with intermediate thinking traces (MLLMs-T), remain poorly understood and lack standardized evaluation benchmarks. Existing work focuses primarily on perception or final answer correctness, offering limited insight into how models reason or fail across modalities. To address this gap, we introduce the MMMR, a new benchmark designed to rigorously evaluate multi-modal reasoning with explicit thinking. The MMMR comprises 1) a high-difficulty dataset of 1,083 questions spanning six diverse reasoning types with symbolic depth and multi-hop demands and 2) a modular Reasoning Trace Evaluation Pipeline (RTEP) for assessing reasoning quality beyond accuracy through metrics like relevance, consistency, and structured error annotations. Empirical results show that MLLMs-T overall outperform non-thinking counterparts, but even top models like Claude-3.7-Sonnet and Gemini-2.5 Pro suffer from reasoning pathologies such as inconsistency and overthinking. This benchmark reveals persistent gaps between accuracy and reasoning quality and provides an actionable evaluation pipeline for future model development. Overall, the MMMR offers a scalable foundation for evaluating, comparing, and improving the next generation of multi-modal reasoning systems. 

---
# Psychology-driven LLM Agents for Explainable Panic Prediction on Social Media during Sudden Disaster Events 

**Authors**: Mengzhu Liu, Zhengqiu Zhu, Chuan Ai, Chen Gao, Xinghong Li, Lingnan He, Kaisheng Lai, Yingfeng Chen, Xin Lu, Yong Li, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16455)  

**Abstract**: During sudden disaster events, accurately predicting public panic sentiment on social media is crucial for proactive governance and crisis management. Current efforts on this problem face three main challenges: lack of finely annotated data hinders emotion prediction studies, unmodeled risk perception causes prediction inaccuracies, and insufficient interpretability of panic formation mechanisms. We address these issues by proposing a Psychology-driven generative Agent framework (PsychoAgent) for explainable panic prediction based on emotion arousal theory. Specifically, we first construct a fine-grained open panic emotion dataset (namely COPE) via human-large language models (LLMs) collaboration to mitigate semantic bias. Then, we develop a framework integrating cross-domain heterogeneous data grounded in psychological mechanisms to model risk perception and cognitive differences in emotion generation. To enhance interpretability, we design an LLM-based role-playing agent that simulates individual psychological chains through dedicatedly designed prompts. Experimental results on our annotated dataset show that PsychoAgent improves panic emotion prediction performance by 12.6% to 21.7% compared to baseline models. Furthermore, the explainability and generalization of our approach is validated. Crucially, this represents a paradigm shift from opaque "data-driven fitting" to transparent "role-based simulation with mechanistic interpretation" for panic emotion prediction during emergencies. Our implementation is publicly available at: this https URL. 

---
# Internal Bias in Reasoning Models leads to Overthinking 

**Authors**: Renfei Dang, Shujian Huang, Jiajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16448)  

**Abstract**: While current reasoning models possess strong exploratory capabilities, they are often criticized for overthinking due to redundant and unnecessary reflections. In this work, we reveal for the first time that overthinking in reasoning models may stem from their internal bias towards input texts. Upon encountering a reasoning problem, the model immediately forms a preliminary guess about the answer, which we term as an internal bias since it is not derived through actual reasoning. When this guess conflicts with its reasoning result, the model tends to engage in reflection, leading to the waste of computational resources. Through further interpretability experiments, we find that this behavior is largely driven by the model's excessive attention to the input section, which amplifies the influence of internal bias on its decision-making process. Additionally, by masking out the original input section, the affect of internal bias can be effectively alleviated and the reasoning length could be reduced by 31%-53% across different complex reasoning tasks. Notably, in most cases, this approach also leads to improvements in accuracy. These findings demonstrate a causal relationship between internal bias and overthinking. 

---
# FREESON: Retriever-Free Retrieval-Augmented Reasoning via Corpus-Traversing MCTS 

**Authors**: Chaeeun Kim, Seungone Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16409)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in multi-step reasoning and calling search engines at appropriate steps. However, existing retrieval-augmented reasoning approaches rely on separate retrieval models, limiting the LRM's role in retrieval to deciding when to retrieve and how to query. This separation not only increases hardware and operational costs but also leads to errors in the retrieval process due to the representation bottleneck, a phenomenon where the retriever's embedding space is not expressive enough to meet the generator's requirements. To address this, we shift our perspective from sequence-to-sequence matching to locating the answer-containing paths within the corpus, and propose a novel framework called FREESON (Retriever-FREE Retrieval-Augmented ReaSONing). This framework enables LRMs to retrieve relevant knowledge on their own by acting as both a generator and retriever. To achieve this, we introduce a variant of the MCTS algorithm specialized for the retrieval task, which we call CT-MCTS (Corpus-Traversing Monte Carlo Tree Search). In this algorithm, LRMs traverse through the corpus toward answer-containing regions. Our results on five open-domain QA benchmarks, including single-hop and multi-hop questions, show that FREESON achieves an average improvement of 14.4% in EM and F1 over four multi-step reasoning models with a separate retriever, and it also performs comparably to the strongest baseline, surpassing it by 3% on PopQA and 2WikiMultihopQA. 

---
# Serious Games: Human-AI Interaction, Evolution, and Coevolution 

**Authors**: Nandini Doreswamy, Louise Horstmanshof  

**Link**: [PDF](https://arxiv.org/pdf/2505.16388)  

**Abstract**: The serious games between humans and AI have only just begun. Evolutionary Game Theory (EGT) models the competitive and cooperative strategies of biological entities. EGT could help predict the potential evolutionary equilibrium of humans and AI. The objective of this work was to examine some of the EGT models relevant to human-AI interaction, evolution, and coevolution. Of thirteen EGT models considered, three were examined: the Hawk-Dove Game, Iterated Prisoner's Dilemma, and the War of Attrition. This selection was based on the widespread acceptance and clear relevance of these models to potential human-AI evolutionary dynamics and coevolutionary trajectories. The Hawk-Dove Game predicts balanced mixed-strategy equilibria based on the costs of conflict. It also shows the potential for balanced coevolution rather than dominance. Iterated Prisoner's Dilemma suggests that repeated interaction may lead to cognitive coevolution. It demonstrates how memory and reciprocity can lead to cooperation. The War of Attrition suggests that competition for resources may result in strategic coevolution, asymmetric equilibria, and conventions on sharing resources. Therefore, EGT may provide a suitable framework to understand and predict the human-AI evolutionary dynamic. However, future research could extend beyond EGT and explore additional frameworks, empirical validation methods, and interdisciplinary perspectives. AI is being shaped by human input and is evolving in response to it. So too, neuroplasticity allows the human brain to grow and evolve in response to stimuli. If humans and AI converge in future, what might be the result of human neuroplasticity combined with an ever-evolving AI? Future research should be mindful of the ethical and cognitive implications of human-AI interaction, evolution, and coevolution. 

---
# Incentivizing Dual Process Thinking for Efficient Large Language Model Reasoning 

**Authors**: Xiaoxue Cheng, Junyi Li, Zhenduo Zhang, Xinyu Tang, Wayne Xin Zhao, Xinyu Kong, Zhiqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16315)  

**Abstract**: Large reasoning models (LRMs) have demonstrated strong performance on complex reasoning tasks, but often suffer from overthinking, generating redundant content regardless of task difficulty. Inspired by the dual process theory in cognitive science, we propose Adaptive Cognition Policy Optimization (ACPO), a reinforcement learning framework that enables LRMs to achieve efficient reasoning through adaptive cognitive allocation and dynamic system switch. ACPO incorporates two key components: (1) introducing system-aware reasoning tokens to explicitly represent the thinking modes thereby making the model's cognitive process transparent, and (2) integrating online difficulty estimation and token length budget to guide adaptive system switch and reasoning during reinforcement learning. To this end, we propose a two-stage training strategy. The first stage begins with supervised fine-tuning to cold start the model, enabling it to generate reasoning paths with explicit thinking modes. In the second stage, we apply ACPO to further enhance adaptive system switch for difficulty-aware reasoning. Experimental results demonstrate that ACPO effectively reduces redundant reasoning while adaptively adjusting cognitive allocation based on task complexity, achieving efficient hybrid reasoning. 

---
# EquivPruner: Boosting Efficiency and Quality in LLM-Based Search via Action Pruning 

**Authors**: Jiawei Liu, Qisi Chen, Jianshu Zhang, Quan Liu, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.16312)  

**Abstract**: Large Language Models (LLMs) excel at complex reasoning through search algorithms, yet current strategies often suffer from massive token consumption due to redundant exploration of semantically equivalent steps. Existing semantic similarity methods struggle to accurately identify such equivalence in domain-specific contexts like mathematical reasoning. To address this, we propose EquivPruner, a simple yet effective approach that identifies and prunes semantically equivalent actions during LLM reasoning search. We also introduce MathEquiv, the first dataset we created for mathematical statement equivalence, which enables the training of a lightweight equivalence detector. Extensive experiments across various models and tasks demonstrate that EquivPruner significantly reduces token consumption, improving searching efficiency and often bolstering reasoning accuracy. For instance, when applied to Qwen2.5-Math-7B-Instruct on GSM8K, EquivPruner reduced token consumption by 48.1\% while also improving accuracy. Our code is available at this https URL. 

---
# No Black Boxes: Interpretable and Interactable Predictive Healthcare with Knowledge-Enhanced Agentic Causal Discovery 

**Authors**: Xiaoxue Han, Pengfei Hu, Jun-En Ding, Chang Lu, Feng Liu, Yue Ning  

**Link**: [PDF](https://arxiv.org/pdf/2505.16288)  

**Abstract**: Deep learning models trained on extensive Electronic Health Records (EHR) data have achieved high accuracy in diagnosis prediction, offering the potential to assist clinicians in decision-making and treatment planning. However, these models lack two crucial features that clinicians highly value: interpretability and interactivity. The ``black-box'' nature of these models makes it difficult for clinicians to understand the reasoning behind predictions, limiting their ability to make informed decisions. Additionally, the absence of interactive mechanisms prevents clinicians from incorporating their own knowledge and experience into the decision-making process. To address these limitations, we propose II-KEA, a knowledge-enhanced agent-driven causal discovery framework that integrates personalized knowledge databases and agentic LLMs. II-KEA enhances interpretability through explicit reasoning and causal analysis, while also improving interactivity by allowing clinicians to inject their knowledge and experience through customized knowledge bases and prompts. II-KEA is evaluated on both MIMIC-III and MIMIC-IV, demonstrating superior performance along with enhanced interpretability and interactivity, as evidenced by its strong results from extensive case studies. 

---
# How do Scaling Laws Apply to Knowledge Graph Engineering Tasks? The Impact of Model Size on Large Language Model Performance 

**Authors**: Desiree Heim, Lars-Peter Meyer, Markus Schröder, Johannes Frey, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.16276)  

**Abstract**: When using Large Language Models (LLMs) to support Knowledge Graph Engineering (KGE), one of the first indications when searching for an appropriate model is its size. According to the scaling laws, larger models typically show higher capabilities. However, in practice, resource costs are also an important factor and thus it makes sense to consider the ratio between model performance and costs. The LLM-KG-Bench framework enables the comparison of LLMs in the context of KGE tasks and assesses their capabilities of understanding and producing KGs and KG queries. Based on a dataset created in an LLM-KG-Bench run covering 26 open state-of-the-art LLMs, we explore the model size scaling laws specific to KGE tasks. In our analyses, we assess how benchmark scores evolve between different model size categories. Additionally, we inspect how the general score development of single models and families of models correlates to their size. Our analyses revealed that, with a few exceptions, the model size scaling laws generally also apply to the selected KGE tasks. However, in some cases, plateau or ceiling effects occurred, i.e., the task performance did not change much between a model and the next larger model. In these cases, smaller models could be considered to achieve high cost-effectiveness. Regarding models of the same family, sometimes larger models performed worse than smaller models of the same family. These effects occurred only locally. Hence it is advisable to additionally test the next smallest and largest model of the same family. 

---
# MAPLE: Many-Shot Adaptive Pseudo-Labeling for In-Context Learning 

**Authors**: Zihan Chen, Song Wang, Zhen Tan, Jundong Li, Cong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16225)  

**Abstract**: In-Context Learning (ICL) empowers Large Language Models (LLMs) to tackle diverse tasks by incorporating multiple input-output examples, known as demonstrations, into the input of LLMs. More recently, advancements in the expanded context windows of LLMs have led to many-shot ICL, which uses hundreds of demonstrations and outperforms few-shot ICL, which relies on fewer examples. However, this approach is often hindered by the high cost of obtaining large amounts of labeled data. To address this challenge, we propose Many-Shot Adaptive Pseudo-LabEling, namely MAPLE, a novel influence-based many-shot ICL framework that utilizes pseudo-labeled samples to compensate for the lack of label information. We first identify a subset of impactful unlabeled samples and perform pseudo-labeling on them by querying LLMs. These pseudo-labeled samples are then adaptively selected and tailored to each test query as input to improve the performance of many-shot ICL, without significant labeling costs. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework, showcasing its ability to enhance LLM adaptability and performance with limited labeled data. 

---
# MADCluster: Model-agnostic Anomaly Detection with Self-supervised Clustering Network 

**Authors**: Sangyong Lee, Subo Hwang, Dohoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16223)  

**Abstract**: In this paper, we propose MADCluster, a novel model-agnostic anomaly detection framework utilizing self-supervised clustering. MADCluster is applicable to various deep learning architectures and addresses the 'hypersphere collapse' problem inherent in existing deep learning-based anomaly detection methods. The core idea is to cluster normal pattern data into a 'single cluster' while simultaneously learning the cluster center and mapping data close to this center. Also, to improve expressiveness and enable effective single clustering, we propose a new 'One-directed Adaptive loss'. The optimization of this loss is mathematically proven. MADCluster consists of three main components: Base Embedder capturing high-dimensional temporal dynamics, Cluster Distance Mapping, and Sequence-wise Clustering for continuous center updates. Its model-agnostic characteristics are achieved by applying various architectures to the Base Embedder. Experiments on four time series benchmark datasets demonstrate that applying MADCluster improves the overall performance of comparative models. In conclusion, the compatibility of MADCluster shows potential for enhancing model performance across various architectures. 

---
# LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead 

**Authors**: Yifan Zhang, Xinkui Zhao, Zuxin Wang, Guanjie Cheng, Yueshen Xu, Shuiguang Deng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16221)  

**Abstract**: The rapid advancement of large language models has unlocked remarkable capabilities across a diverse array of natural language processing tasks. However, the considerable differences among available LLMs-in terms of cost, performance, and computational demands-pose significant challenges for users aiming to identify the most suitable model for specific tasks. In this work, we present LightRouter, a novel framework designed to systematically select and integrate a small subset of LLMs from a larger pool, with the objective of jointly optimizing both task performance and cost efficiency. LightRouter leverages an adaptive selection mechanism to identify models that require only a minimal number of boot tokens, thereby reducing costs, and further employs an effective integration strategy to combine their outputs. Extensive experiments across multiple benchmarks demonstrate that LightRouter matches or outperforms widely-used ensemble baselines, achieving up to a 25% improvement in accuracy. Compared with leading high-performing models, LightRouter achieves comparable performance while reducing inference costs by up to 27%. Importantly, our framework operates without any prior knowledge of individual models and relies exclusively on inexpensive, lightweight models. This work introduces a practical approach for efficient LLM selection and provides valuable insights into optimal strategies for model combination. 

---
# Velocity Completion Task and Method for Event-based Player Positional Data in Soccer 

**Authors**: Rikuhei Umemoto, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2505.16199)  

**Abstract**: In many real-world complex systems, the behavior can be observed as a collection of discrete events generated by multiple interacting agents. Analyzing the dynamics of these multi-agent systems, especially team sports, often relies on understanding the movement and interactions of individual agents. However, while providing valuable snapshots, event-based positional data typically lacks the continuous temporal information needed to directly calculate crucial properties such as velocity. This absence severely limits the depth of dynamic analysis, preventing a comprehensive understanding of individual agent behaviors and emergent team strategies. To address this challenge, we propose a new method to simultaneously complete the velocity of all agents using only the event-based positional data from team sports. Based on this completed velocity information, we investigate the applicability of existing team sports analysis and evaluation methods. Experiments using soccer event data demonstrate that neural network-based approaches outperformed rule-based methods regarding velocity completion error, considering the underlying temporal dependencies and graph structure of player-to-player or player-to-ball interaction. Moreover, the space evaluation results obtained using the completed velocity are closer to those derived from complete tracking data, highlighting our method's potential for enhanced team sports system analysis. 

---
# SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning 

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16186)  

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations. 

---
# Dynamic Sampling that Adapts: Iterative DPO for Self-Aware Mathematical Reasoning 

**Authors**: Jun Rao, Xuebo Liu, Hexuan Deng, Zepeng Lin, Zixiong Yu, Jiansheng Wei, Xiaojun Meng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16176)  

**Abstract**: In the realm of data selection for reasoning tasks, existing approaches predominantly rely on externally predefined static metrics such as difficulty and diversity, which are often designed for supervised fine-tuning (SFT) and lack adaptability to continuous training processes. A critical limitation of these methods is their inability to dynamically align with the evolving capabilities of models during online training, a gap that becomes increasingly pronounced with the rise of dynamic training paradigms and online reinforcement learning (RL) frameworks (e.g., R1 models). To address this, we introduce SAI-DPO, an algorithm that dynamically selects training data by continuously assessing a model's stage-specific reasoning abilities across different training phases. By integrating real-time model performance feedback, SAI-DPO adaptively adapts data selection to the evolving strengths and weaknesses of the model, thus enhancing both data utilization efficiency and final task performance. Extensive experiments on three state-of-the-art models and eight mathematical reasoning benchmarks, including challenging competition-level datasets (e.g., AIME24 and AMC23), demonstrate that SAI-DPO achieves an average performance boost of up to 21.3 percentage points, with particularly notable improvements of 10 and 15 points on AIME24 and AMC23, respectively. These results highlight the superiority of dynamic, model-adaptive data selection over static, externally defined strategies in advancing reasoning. 

---
# Losing is for Cherishing: Data Valuation Based on Machine Unlearning and Shapley Value 

**Authors**: Le Ma, Shirao Yang, Zihao Wang, Yinggui Wang, Lei Wang, Tao Wei, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16147)  

**Abstract**: The proliferation of large models has intensified the need for efficient data valuation methods to quantify the contribution of individual data providers. Traditional approaches, such as game-theory-based Shapley value and influence-function-based techniques, face prohibitive computational costs or require access to full data and model training details, making them hardly achieve partial data valuation. To address this, we propose Unlearning Shapley, a novel framework that leverages machine unlearning to estimate data values efficiently. By unlearning target data from a pretrained model and measuring performance shifts on a reachable test set, our method computes Shapley values via Monte Carlo sampling, avoiding retraining and eliminating dependence on full data. Crucially, Unlearning Shapley supports both full and partial data valuation, making it scalable for large models (e.g., LLMs) and practical for data markets. Experiments on benchmark datasets and large-scale text corpora demonstrate that our approach matches the accuracy of state-of-the-art methods while reducing computational overhead by orders of magnitude. Further analysis confirms a strong correlation between estimated values and the true impact of data subsets, validating its reliability in real-world scenarios. This work bridges the gap between data valuation theory and practical deployment, offering a scalable, privacy-compliant solution for modern AI ecosystems. 

---
# Sudoku-Bench: Evaluating creative reasoning with Sudoku variants 

**Authors**: Jeffrey Seely, Yuki Imajuku, Tianyu Zhao, Edoardo Cetin, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.16135)  

**Abstract**: Existing reasoning benchmarks for large language models (LLMs) frequently fail to capture authentic creativity, often rewarding memorization of previously observed patterns. We address this shortcoming with Sudoku-Bench, a curated benchmark of challenging and unconventional Sudoku variants specifically selected to evaluate creative, multi-step logical reasoning. Sudoku variants form an unusually effective domain for reasoning research: each puzzle introduces unique or subtly interacting constraints, making memorization infeasible and requiring solvers to identify novel logical breakthroughs (``break-ins''). Despite their diversity, Sudoku variants maintain a common and compact structure, enabling clear and consistent evaluation. Sudoku-Bench includes a carefully chosen puzzle set, a standardized text-based puzzle representation, and flexible tools compatible with thousands of publicly available puzzles -- making it easy to extend into a general research environment. Baseline experiments show that state-of-the-art LLMs solve fewer than 15\% of puzzles unaided, highlighting significant opportunities to advance long-horizon, strategic reasoning capabilities. 

---
# LLM-Powered AI Agent Systems and Their Applications in Industry 

**Authors**: Guannan Liang, Qianqian Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16120)  

**Abstract**: The emergence of Large Language Models (LLMs) has reshaped agent systems. Unlike traditional rule-based agents with limited task scope, LLM-powered agents offer greater flexibility, cross-domain reasoning, and natural language interaction. Moreover, with the integration of multi-modal LLMs, current agent systems are highly capable of processing diverse data modalities, including text, images, audio, and structured tabular data, enabling richer and more adaptive real-world behavior. This paper comprehensively examines the evolution of agent systems from the pre-LLM era to current LLM-powered architectures. We categorize agent systems into software-based, physical, and adaptive hybrid systems, highlighting applications across customer service, software development, manufacturing automation, personalized education, financial trading, and healthcare. We further discuss the primary challenges posed by LLM-powered agents, including high inference latency, output uncertainty, lack of evaluation metrics, and security vulnerabilities, and propose potential solutions to mitigate these concerns. 

---
# Logic-of-Thought: Empowering Large Language Models with Logic Programs for Solving Puzzles in Natural Language 

**Authors**: Naiqi Li, Peiyuan Liu, Zheng Liu, Tao Dai, Yong Jiang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16114)  

**Abstract**: Solving puzzles in natural language poses a long-standing challenge in AI. While large language models (LLMs) have recently shown impressive capabilities in a variety of tasks, they continue to struggle with complex puzzles that demand precise reasoning and exhaustive search. In this paper, we propose Logic-of-Thought (Logot), a novel framework that bridges LLMs with logic programming to address this problem. Our method leverages LLMs to translate puzzle rules and states into answer set programs (ASPs), the solution of which are then accurately and efficiently inferred by an ASP interpreter. This hybrid approach combines the natural language understanding of LLMs with the precise reasoning capabilities of logic programs. We evaluate our method on various grid puzzles and dynamic puzzles involving actions, demonstrating near-perfect accuracy across all tasks. Our code and data are available at: this https URL. 

---
# BioDSA-1K: Benchmarking Data Science Agents for Biomedical Research 

**Authors**: Zifeng Wang, Benjamin Danek, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16100)  

**Abstract**: Validating scientific hypotheses is a central challenge in biomedical research, and remains difficult for artificial intelligence (AI) agents due to the complexity of real-world data analysis and evidence interpretation. In this work, we present BioDSA-1K, a benchmark designed to evaluate AI agents on realistic, data-driven biomedical hypothesis validation tasks. BioDSA-1K consists of 1,029 hypothesis-centric tasks paired with 1,177 analysis plans, curated from over 300 published biomedical studies to reflect the structure and reasoning found in authentic research workflows. Each task includes a structured hypothesis derived from the original study's conclusions, expressed in the affirmative to reflect the language of scientific reporting, and one or more pieces of supporting evidence grounded in empirical data tables. While these hypotheses mirror published claims, they remain testable using standard statistical or machine learning methods. The benchmark enables evaluation along four axes: (1) hypothesis decision accuracy, (2) alignment between evidence and conclusion, (3) correctness of the reasoning process, and (4) executability of the AI-generated analysis code. Importantly, BioDSA-1K includes non-verifiable hypotheses: cases where the available data are insufficient to support or refute a claim, reflecting a common yet underexplored scenario in real-world science. We propose BioDSA-1K as a foundation for building and evaluating generalizable, trustworthy AI agents for biomedical discovery. 

---
# TrialPanorama: Database and Benchmark for Systematic Review and Design of Clinical Trials 

**Authors**: Zifeng Wang, Qiao Jin, Jiacheng Lin, Junyi Gao, Jathurshan Pradeepkumar, Pengcheng Jiang, Benjamin Danek, Zhiyong Lu, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16097)  

**Abstract**: Developing artificial intelligence (AI) for vertical domains requires a solid data foundation for both training and evaluation. In this work, we introduce TrialPanorama, a large-scale, structured database comprising 1,657,476 clinical trial records aggregated from 15 global sources. The database captures key aspects of trial design and execution, including trial setups, interventions, conditions, biomarkers, and outcomes, and links them to standard biomedical ontologies such as DrugBank and MedDRA. This structured and ontology-grounded design enables TrialPanorama to serve as a unified, extensible resource for a wide range of clinical trial tasks, including trial planning, design, and summarization. To demonstrate its utility, we derive a suite of benchmark tasks directly from the TrialPanorama database. The benchmark spans eight tasks across two categories: three for systematic review (study search, study screening, and evidence summarization) and five for trial design (arm design, eligibility criteria, endpoint selection, sample size estimation, and trial completion assessment). The experiments using five state-of-the-art large language models (LLMs) show that while general-purpose LLMs exhibit some zero-shot capability, their performance is still inadequate for high-stakes clinical trial workflows. We release TrialPanorama database and the benchmark to facilitate further research on AI for clinical trials. 

---
# Can AI Read Between The Lines? Benchmarking LLMs On Financial Nuance 

**Authors**: Dominick Kubica, Dylan T. Gordon, Nanami Emura, Derleen Saini, Charlie Goldenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.16090)  

**Abstract**: As of 2025, Generative Artificial Intelligence (GenAI) has become a central tool for productivity across industries. Beyond text generation, GenAI now plays a critical role in coding, data analysis, and research workflows. As large language models (LLMs) continue to evolve, it is essential to assess the reliability and accuracy of their outputs, especially in specialized, high-stakes domains like finance. Most modern LLMs transform text into numerical vectors, which are used in operations such as cosine similarity searches to generate responses. However, this abstraction process can lead to misinterpretation of emotional tone, particularly in nuanced financial contexts. While LLMs generally excel at identifying sentiment in everyday language, these models often struggle with the nuanced, strategically ambiguous language found in earnings call transcripts. Financial disclosures frequently embed sentiment in hedged statements, forward-looking language, and industry-specific jargon, making it difficult even for human analysts to interpret consistently, let alone AI models. This paper presents findings from the Santa Clara Microsoft Practicum Project, led by Professor Charlie Goldenberg, which benchmarks the performance of Microsoft's Copilot, OpenAI's ChatGPT, Google's Gemini, and traditional machine learning models for sentiment analysis of financial text. Using Microsoft earnings call transcripts, the analysis assesses how well LLM-derived sentiment correlates with market sentiment and stock movements and evaluates the accuracy of model outputs. Prompt engineering techniques are also examined to improve sentiment analysis results. Visualizations of sentiment consistency are developed to evaluate alignment between tone and stock performance, with sentiment trends analyzed across Microsoft's lines of business to determine which segments exert the greatest influence. 

---
# Optimizing LLM-Based Multi-Agent System with Textual Feedback: A Case Study on Software Development 

**Authors**: Ming Shen, Raphael Shu, Anurag Pratik, James Gung, Yubin Ge, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16086)  

**Abstract**: We have seen remarkable progress in large language models (LLMs) empowered multi-agent systems solving complex tasks necessitating cooperation among experts with diverse skills. However, optimizing LLM-based multi-agent systems remains challenging. In this work, we perform an empirical case study on group optimization of role-based multi-agent systems utilizing natural language feedback for challenging software development tasks under various evaluation dimensions. We propose a two-step agent prompts optimization pipeline: identifying underperforming agents with their failure explanations utilizing textual feedback and then optimizing system prompts of identified agents utilizing failure explanations. We then study the impact of various optimization settings on system performance with two comparison groups: online against offline optimization and individual against group optimization. For group optimization, we study two prompting strategies: one-pass and multi-pass prompting optimizations. Overall, we demonstrate the effectiveness of our optimization method for role-based multi-agent systems tackling software development tasks evaluated on diverse evaluation dimensions, and we investigate the impact of diverse optimization settings on group behaviors of the multi-agent systems to provide practical insights for future development. 

---
# SynEVO: A neuro-inspired spatiotemporal evolutional framework for cross-domain adaptation 

**Authors**: Jiayue Liu, Zhongchao Yi, Zhengyang Zhou, Qihe Huang, Kuo Yang, Xu Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16080)  

**Abstract**: Discovering regularities from spatiotemporal systems can benefit various scientific and social planning. Current spatiotemporal learners usually train an independent model from a specific source data that leads to limited transferability among sources, where even correlated tasks requires new design and training. The key towards increasing cross-domain knowledge is to enable collective intelligence and model evolution. In this paper, inspired by neuroscience theories, we theoretically derive the increased information boundary via learning cross-domain collective intelligence and propose a Synaptic EVOlutional spatiotemporal network, SynEVO, where SynEVO breaks the model independence and enables cross-domain knowledge to be shared and aggregated. Specifically, we first re-order the sample groups to imitate the human curriculum learning, and devise two complementary learners, elastic common container and task-independent extractor to allow model growth and task-wise commonality and personality disentanglement. Then an adaptive dynamic coupler with a new difference metric determines whether the new sample group should be incorporated into common container to achieve model evolution under various domains. Experiments show that SynEVO improves the generalization capacity by at most 42% under cross-domain scenarios and SynEVO provides a paradigm of NeuroAI for knowledge transfer and adaptation. 

---
# How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior 

**Authors**: Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16067)  

**Abstract**: Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory operations that are widely used by many agent frameworks-addition, which incorporates new experiences into the memory base, and deletion, which selectively removes past experiences-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where outdated or irrelevant experiences negatively influence current tasks. Through controlled experiments, we show that combining selective addition and deletion strategies can help mitigate these negative effects, yielding an average absolute performance gain of 10% compared to naive memory growth. Furthermore, we highlight how memory management choices affect agents' behavior under challenging conditions such as task distribution shifts and constrained memory resources. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance. We also release our code to facilitate further study. 

---
# SPhyR: Spatial-Physical Reasoning Benchmark on Material Distribution 

**Authors**: Philipp D. Siedler  

**Link**: [PDF](https://arxiv.org/pdf/2505.16048)  

**Abstract**: We introduce a novel dataset designed to benchmark the physical and spatial reasoning capabilities of Large Language Models (LLM) based on topology optimization, a method for computing optimal material distributions within a design space under prescribed loads and supports. In this dataset, LLMs are provided with conditions such as 2D boundary, applied forces and supports, and must reason about the resulting optimal material distribution. The dataset includes a variety of tasks, ranging from filling in masked regions within partial structures to predicting complete material distributions. Solving these tasks requires understanding the flow of forces and the required material distribution under given constraints, without access to simulation tools or explicit physical models, challenging models to reason about structural stability and spatial organization. Our dataset targets the evaluation of spatial and physical reasoning abilities in 2D settings, offering a complementary perspective to traditional language and logic benchmarks. 

---
# Causal LLM Routing: End-to-End Regret Minimization from Observational Data 

**Authors**: Asterios Tsiourvas, Wei Sun, Georgia Perakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.16037)  

**Abstract**: LLM routing aims to select the most appropriate model for each query, balancing competing performance metrics such as accuracy and cost across a pool of language models. Prior approaches typically adopt a decoupled strategy, where the metrics are first predicted and the model is then selected based on these estimates. This setup is prone to compounding errors and often relies on full-feedback data, where each query is evaluated by all candidate models, which is costly to obtain and maintain in practice. In contrast, we learn from observational data, which records only the outcome of the model actually deployed. We propose a causal end-to-end framework that learns routing policies by minimizing decision-making regret from observational data. To enable efficient optimization, we introduce two theoretically grounded surrogate objectives: a classification-based upper bound, and a softmax-weighted regret approximation shown to recover the optimal policy at convergence. We further extend our framework to handle heterogeneous cost preferences via an interval-conditioned architecture. Experiments on public benchmarks show that our method outperforms existing baselines, achieving state-of-the-art performance across different embedding models. 

---
# Children's Mental Models of AI Reasoning: Implications for AI Literacy Education 

**Authors**: Aayushi Dangol, Robert Wolfe, Runhua Zhao, JaeWon Kim, Trushaa Ramanan, Katie Davis, Julie A. Kientz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16031)  

**Abstract**: As artificial intelligence (AI) advances in reasoning capabilities, most recently with the emergence of Large Reasoning Models (LRMs), understanding how children conceptualize AI's reasoning processes becomes critical for fostering AI literacy. While one of the "Five Big Ideas" in AI education highlights reasoning algorithms as central to AI decision-making, less is known about children's mental models in this area. Through a two-phase approach, consisting of a co-design session with 8 children followed by a field study with 106 children (grades 3-8), we identified three models of AI reasoning: Deductive, Inductive, and Inherent. Our findings reveal that younger children (grades 3-5) often attribute AI's reasoning to inherent intelligence, while older children (grades 6-8) recognize AI as a pattern recognizer. We highlight three tensions that surfaced in children's understanding of AI reasoning and conclude with implications for scaffolding AI curricula and designing explainable AI tools. 

---
# Exploring Flow-Lenia Universes with a Curiosity-driven AI Scientist: Discovering Diverse Ecosystem Dynamics 

**Authors**: Thomas Michel, Marko Cvjetko, Gautier Hamon, Pierre-Yves Oudeyer, Clément Moulin-Frier  

**Link**: [PDF](https://arxiv.org/pdf/2505.15998)  

**Abstract**: We present a method for the automated discovery of system-level dynamics in Flow-Lenia$-$a continuous cellular automaton (CA) with mass conservation and parameter localization$-$using a curiosity-driven AI scientist. This method aims to uncover processes leading to self-organization of evolutionary and ecosystemic dynamics in CAs. We build on previous work which uses diversity search algorithms in Lenia to find self-organized individual patterns, and extend it to large environments that support distinct interacting patterns. We adapt Intrinsically Motivated Goal Exploration Processes (IMGEPs) to drive exploration of diverse Flow-Lenia environments using simulation-wide metrics, such as evolutionary activity, compression-based complexity, and multi-scale entropy. We test our method in two experiments, showcasing its ability to illuminate significantly more diverse dynamics compared to random search. We show qualitative results illustrating how ecosystemic simulations enable self-organization of complex collective behaviors not captured by previous individual pattern search and analysis. We complement automated discovery with an interactive exploration tool, creating an effective human-AI collaborative workflow for scientific investigation. Though demonstrated specifically with Flow-Lenia, this methodology provides a framework potentially applicable to other parameterizable complex systems where understanding emergent collective properties is of interest. 

---
# PhyX: Does Your Model Have the "Wits" for Physical Reasoning? 

**Authors**: Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, Zhongwei Wan, Kai Zhang, Wendong Xu, Jing Xiong, Ping Luo, Wenhu Chen, Chaofan Tao, Zhuoqing Mao, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15929)  

**Abstract**: Existing benchmarks fail to capture a crucial aspect of intelligence: physical reasoning, the integrated ability to combine domain knowledge, symbolic reasoning, and understanding of real-world constraints. To address this gap, we introduce PhyX: the first large-scale benchmark designed to assess models capacity for physics-grounded reasoning in visual scenarios. PhyX includes 3K meticulously curated multimodal questions spanning 6 reasoning types across 25 sub-domains and 6 core physics domains: thermodynamics, electromagnetism, mechanics, modern physics, optics, and wave\&acoustics. In our comprehensive evaluation, even state-of-the-art models struggle significantly with physical reasoning. GPT-4o, Claude3.7-Sonnet, and GPT-o4-mini achieve only 32.5\%, 42.2\%, and 45.8\% accuracy respectively-performance gaps exceeding 29\% compared to human experts. Our analysis exposes critical limitations in current models: over-reliance on memorized disciplinary knowledge, excessive dependence on mathematical formulations, and surface-level visual pattern matching rather than genuine physical understanding. We provide in-depth analysis through fine-grained statistics, detailed case studies, and multiple evaluation paradigms to thoroughly examine physical reasoning capabilities. To ensure reproducibility, we implement a compatible evaluation protocol based on widely-used toolkits such as VLMEvalKit, enabling one-click evaluation. 

---
# Bandit based Dynamic Candidate Edge Selection in Solving Traveling Salesman Problems 

**Authors**: Long Wanga, Jiongzhi Zheng, Zhengda Xiong, ChuMin Li, Kun He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15862)  

**Abstract**: Algorithms designed for routing problems typically rely on high-quality candidate edges to guide their search, aiming to reduce the search space and enhance the search efficiency. However, many existing algorithms, like the classical Lin-Kernighan-Helsgaun (LKH) algorithm for the Traveling Salesman Problem (TSP), often use predetermined candidate edges that remain static throughout local searches. This rigidity could cause the algorithm to get trapped in local optima, limiting its potential to find better solutions. To address this issue, we propose expanding the candidate sets to include other promising edges, providing them an opportunity for selection. Specifically, we incorporate multi-armed bandit models to dynamically select the most suitable candidate edges in each iteration, enabling LKH to make smarter choices and lead to improved solutions. Extensive experiments on multiple TSP benchmarks show the excellent performance of our method. Moreover, we employ this bandit-based method to LKH-3, an extension of LKH tailored for solving various TSP variant problems, and our method also significantly enhances LKH-3's performance across typical TSP variants. 

---
# GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning 

**Authors**: Chengqi Duan, Rongyao Fang, Yuqing Wang, Kun Wang, Linjiang Huang, Xingyu Zeng, Hongsheng Li, Xihui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17022)  

**Abstract**: Visual generation models have made remarkable progress in creating realistic images from text prompts, yet struggle with complex prompts that specify multiple objects with precise spatial relationships and attributes. Effective handling of such prompts requires explicit reasoning about the semantic content and spatial layout. We present GoT-R1, a framework that applies reinforcement learning to enhance semantic-spatial reasoning in visual generation. Building upon the Generation Chain-of-Thought approach, GoT-R1 enables models to autonomously discover effective reasoning strategies beyond predefined templates through carefully designed reinforcement learning. To achieve this, we propose a dual-stage multi-dimensional reward framework that leverages MLLMs to evaluate both the reasoning process and final output, enabling effective supervision across the entire generation pipeline. The reward system assesses semantic alignment, spatial accuracy, and visual quality in a unified approach. Experimental results demonstrate significant improvements on T2I-CompBench benchmark, particularly in compositional tasks involving precise spatial relationships and attribute binding. GoT-R1 advances the state-of-the-art in image generation by successfully transferring sophisticated reasoning capabilities to the visual generation domain. To facilitate future research, we make our code and pretrained models publicly available at this https URL. 

---
# Let Androids Dream of Electric Sheep: A Human-like Image Implication Understanding and Reasoning Framework 

**Authors**: Chenhao Zhang, Yazhe Niu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17019)  

**Abstract**: Metaphorical comprehension in images remains a critical challenge for AI systems, as existing models struggle to grasp the nuanced cultural, emotional, and contextual implications embedded in visual content. While multimodal large language models (MLLMs) excel in basic Visual Question Answer (VQA) tasks, they struggle with a fundamental limitation on image implication tasks: contextual gaps that obscure the relationships between different visual elements and their abstract meanings. Inspired by the human cognitive process, we propose Let Androids Dream (LAD), a novel framework for image implication understanding and reasoning. LAD addresses contextual missing through the three-stage framework: (1) Perception: converting visual information into rich and multi-level textual representations, (2) Search: iteratively searching and integrating cross-domain knowledge to resolve ambiguity, and (3) Reasoning: generating context-alignment image implication via explicit reasoning. Our framework with the lightweight GPT-4o-mini model achieves SOTA performance compared to 15+ MLLMs on English image implication benchmark and a huge improvement on Chinese benchmark, performing comparable with the GPT-4o model on Multiple-Choice Question (MCQ) and outperforms 36.7% on Open-Style Question (OSQ). Additionally, our work provides new insights into how AI can more effectively interpret image implications, advancing the field of vision-language reasoning and human-AI interaction. Our project is publicly available at this https URL. 

---
# Delving into RL for Image Generation with CoT: A Study on DPO vs. GRPO 

**Authors**: Chengzhuo Tong, Ziyu Guo, Renrui Zhang, Wenyu Shan, Xinyu Wei, Zhenghao Xing, Hongsheng Li, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.17017)  

**Abstract**: Recent advancements underscore the significant role of Reinforcement Learning (RL) in enhancing the Chain-of-Thought (CoT) reasoning capabilities of large language models (LLMs). Two prominent RL algorithms, Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO), are central to these developments, showcasing different pros and cons. Autoregressive image generation, also interpretable as a sequential CoT reasoning process, presents unique challenges distinct from LLM-based CoT reasoning. These encompass ensuring text-image consistency, improving image aesthetic quality, and designing sophisticated reward models, rather than relying on simpler rule-based rewards. While recent efforts have extended RL to this domain, these explorations typically lack an in-depth analysis of the domain-specific challenges and the characteristics of different RL strategies. To bridge this gap, we provide the first comprehensive investigation of the GRPO and DPO algorithms in autoregressive image generation, evaluating their in-domain performance and out-of-domain generalization, while scrutinizing the impact of different reward models on their respective capabilities. Our findings reveal that GRPO and DPO exhibit distinct advantages, and crucially, that reward models possessing stronger intrinsic generalization capabilities potentially enhance the generalization potential of the applied RL algorithms. Furthermore, we systematically explore three prevalent scaling strategies to enhance both their in-domain and out-of-domain proficiency, deriving unique insights into efficiently scaling performance for each paradigm. We hope our study paves a new path for inspiring future work on developing more effective RL algorithms to achieve robust CoT reasoning in the realm of autoregressive image generation. Code is released at this https URL 

---
# Interactive Post-Training for Vision-Language-Action Models 

**Authors**: Shuhan Tan, Kairan Dou, Yue Zhao, Philipp Krähenbühl  

**Link**: [PDF](https://arxiv.org/pdf/2505.17016)  

**Abstract**: We introduce RIPT-VLA, a simple and scalable reinforcement-learning-based interactive post-training paradigm that fine-tunes pretrained Vision-Language-Action (VLA) models using only sparse binary success rewards. Existing VLA training pipelines rely heavily on offline expert demonstration data and supervised imitation, limiting their ability to adapt to new tasks and environments under low-data regimes. RIPT-VLA addresses this by enabling interactive post-training with a stable policy optimization algorithm based on dynamic rollout sampling and leave-one-out advantage estimation.
RIPT-VLA has the following characteristics. First, it applies to various VLA models, resulting in an improvement on the lightweight QueST model by 21.2%, and the 7B OpenVLA-OFT model to an unprecedented 97.5% success rate. Second, it is computationally efficient and data-efficient: with only one demonstration, RIPT-VLA enables an unworkable SFT model (4%) to succeed with a 97% success rate within 15 iterations. Furthermore, we demonstrate that the policy learned by RIPT-VLA generalizes across different tasks and scenarios and is robust to the initial state context. These results highlight RIPT-VLA as a practical and effective paradigm for post-training VLA models through minimal supervision. 

---
# SpatialScore: Towards Unified Evaluation for Multimodal Spatial Understanding 

**Authors**: Haoning Wu, Xiao Huang, Yaohui Chen, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.17012)  

**Abstract**: Multimodal large language models (MLLMs) have achieved impressive success in question-answering tasks, yet their capabilities for spatial understanding are less explored. This work investigates a critical question: do existing MLLMs possess 3D spatial perception and understanding abilities? Concretely, we make the following contributions in this paper: (i) we introduce VGBench, a benchmark specifically designed to assess MLLMs for visual geometry perception, e.g., camera pose and motion estimation; (ii) we propose SpatialScore, the most comprehensive and diverse multimodal spatial understanding benchmark to date, integrating VGBench with relevant data from the other 11 existing datasets. This benchmark comprises 28K samples across various spatial understanding tasks, modalities, and QA formats, along with a carefully curated challenging subset, SpatialScore-Hard; (iii) we develop SpatialAgent, a novel multi-agent system incorporating 9 specialized tools for spatial understanding, supporting both Plan-Execute and ReAct reasoning paradigms; (iv) we conduct extensive evaluations to reveal persistent challenges in spatial reasoning while demonstrating the effectiveness of SpatialAgent. We believe SpatialScore will offer valuable insights and serve as a rigorous benchmark for the next evolution of MLLMs. 

---
# Understanding Prompt Tuning and In-Context Learning via Meta-Learning 

**Authors**: Tim Genewein, Kevin Wenliang Li, Jordi Grau-Moya, Anian Ruoss, Laurent Orseau, Marcus Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.17010)  

**Abstract**: Prompting is one of the main ways to adapt a pretrained model to target tasks. Besides manually constructing prompts, many prompt optimization methods have been proposed in the literature. Method development is mainly empirically driven, with less emphasis on a conceptual understanding of prompting. In this paper we discuss how optimal prompting can be understood through a Bayesian view, which also implies some fundamental limitations of prompting that can only be overcome by tuning weights. The paper explains in detail how meta-trained neural networks behave as Bayesian predictors over the pretraining distribution, whose hallmark feature is rapid in-context adaptation. Optimal prompting can be studied formally as conditioning these Bayesian predictors, yielding criteria for target tasks where optimal prompting is and is not possible. We support the theory with educational experiments on LSTMs and Transformers, where we compare different versions of prefix-tuning and different weight-tuning methods. We also confirm that soft prefixes, which are sequences of real-valued vectors outside the token alphabet, can lead to very effective prompts for trained and even untrained networks by manipulating activations in ways that are not achievable by hard tokens. This adds an important mechanistic aspect beyond the conceptual Bayesian theory. 

---
# R1-Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via Reinforcement Learning 

**Authors**: Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17005)  

**Abstract**: Large Language Models (LLMs) are powerful but prone to hallucinations due to static knowledge. Retrieval-Augmented Generation (RAG) helps by injecting external information, but current methods often are costly, generalize poorly, or ignore the internal knowledge of the model. In this paper, we introduce R1-Searcher++, a novel framework designed to train LLMs to adaptively leverage both internal and external knowledge sources. R1-Searcher++ employs a two-stage training strategy: an initial SFT Cold-start phase for preliminary format learning, followed by RL for Dynamic Knowledge Acquisition. The RL stage uses outcome-supervision to encourage exploration, incorporates a reward mechanism for internal knowledge utilization, and integrates a memorization mechanism to continuously assimilate retrieved information, thereby enriching the model's internal knowledge. By leveraging internal knowledge and external search engine, the model continuously improves its capabilities, enabling efficient retrieval-augmented reasoning. Our experiments demonstrate that R1-Searcher++ outperforms previous RAG and reasoning methods and achieves efficient retrieval. The code is available at this https URL. 

---
# Guided Diffusion Sampling on Function Spaces with Applications to PDEs 

**Authors**: Jiachen Yao, Abbas Mammadov, Julius Berner, Gavin Kerrigan, Jong Chul Ye, Kamyar Azizzadenesheli, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.17004)  

**Abstract**: We propose a general framework for conditional sampling in PDE-based inverse problems, targeting the recovery of whole solutions from extremely sparse or noisy measurements. This is accomplished by a function-space diffusion model and plug-and-play guidance for conditioning. Our method first trains an unconditional discretization-agnostic denoising model using neural operator architectures. At inference, we refine the samples to satisfy sparse observation data via a gradient-based guidance mechanism. Through rigorous mathematical analysis, we extend Tweedie's formula to infinite-dimensional Hilbert spaces, providing the theoretical foundation for our posterior sampling approach. Our method (FunDPS) accurately captures posterior distributions in function spaces under minimal supervision and severe data scarcity. Across five PDE tasks with only 3% observation, our method achieves an average 32% accuracy improvement over state-of-the-art fixed-resolution diffusion baselines while reducing sampling steps by 4x. Furthermore, multi-resolution fine-tuning ensures strong cross-resolution generalizability. To the best of our knowledge, this is the first diffusion-based framework to operate independently of discretization, offering a practical and flexible solution for forward and inverse problems in the context of PDEs. Code is available at this https URL 

---
# PAEFF: Precise Alignment and Enhanced Gated Feature Fusion for Face-Voice Association 

**Authors**: Abdul Hannan, Muhammad Arslan Manzoor, Shah Nawaz, Muhammad Irzam Liaqat, Markus Schedl, Mubashir Noman  

**Link**: [PDF](https://arxiv.org/pdf/2505.17002)  

**Abstract**: We study the task of learning association between faces and voices, which is gaining interest in the multimodal community lately. These methods suffer from the deliberate crafting of negative mining procedures as well as the reliance on the distant margin parameter. These issues are addressed by learning a joint embedding space in which orthogonality constraints are applied to the fused embeddings of faces and voices. However, embedding spaces of faces and voices possess different characteristics and require spaces to be aligned before fusing them. To this end, we propose a method that accurately aligns the embedding spaces and fuses them with an enhanced gated fusion thereby improving the performance of face-voice association. Extensive experiments on the VoxCeleb dataset reveals the merits of the proposed approach. 

---
# Do Large Language Models Excel in Complex Logical Reasoning with Formal Language? 

**Authors**: Jin Jiang, Jianing Wang, Yuchen Yan, Yang Liu, Jianhua Zhu, Mengdi Zhang, Xunliang Cai, Liangcai Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16998)  

**Abstract**: Large Language Models (LLMs) have been shown to achieve breakthrough performance on complex logical reasoning tasks. Nevertheless, most existing research focuses on employing formal language to guide LLMs to derive reliable reasoning paths, while systematic evaluations of these capabilities are still limited. In this paper, we aim to conduct a comprehensive evaluation of LLMs across various logical reasoning problems utilizing formal languages. From the perspective of three dimensions, i.e., spectrum of LLMs, taxonomy of tasks, and format of trajectories, our key findings are: 1) Thinking models significantly outperform Instruct models, especially when formal language is employed; 2) All LLMs exhibit limitations in inductive reasoning capability, irrespective of whether they use a formal language; 3) Data with PoT format achieves the best generalization performance across other languages. Additionally, we also curate the formal-relative training data to further enhance the small language models, and the experimental results indicate that a simple rejected fine-tuning method can better enable LLMs to generalize across formal languages and achieve the best overall performance. Our codes and reports are available at this https URL. 

---
# $\text{R}^2\text{ec}$: Towards Large Recommender Models with Reasoning 

**Authors**: Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16994)  

**Abstract**: Large recommender models have extended LLMs as powerful recommenders via encoding or item generation, and recent breakthroughs in LLM reasoning synchronously motivate the exploration of reasoning in recommendation. Current studies usually position LLMs as external reasoning modules to yield auxiliary thought for augmenting conventional recommendation pipelines. However, such decoupled designs are limited in significant resource cost and suboptimal joint optimization. To address these issues, we propose \name, a unified large recommender model with intrinsic reasoning capabilities. Initially, we reconceptualize the model architecture to facilitate interleaved reasoning and recommendation in the autoregressive process. Subsequently, we propose RecPO, a corresponding reinforcement learning framework that optimizes \name\ both the reasoning and recommendation capabilities simultaneously in a single policy update; RecPO introduces a fused reward scheme that solely leverages recommendation labels to simulate the reasoning capability, eliminating dependency on specialized reasoning annotations. Experiments on three datasets with various baselines verify the effectiveness of \name, showing relative improvements of 68.67\% in Hit@5 and 45.21\% in NDCG@20. Code available at this https URL. 

---
# MASLab: A Unified and Comprehensive Codebase for LLM-based Multi-Agent Systems 

**Authors**: Rui Ye, Keduan Huang, Qimin Wu, Yuzhu Cai, Tian Jin, Xianghe Pang, Xiangrui Liu, Jiaqi Su, Chen Qian, Bohan Tang, Kaiqu Liang, Jiaao Chen, Yue Hu, Zhenfei Yin, Rongye Shi, Bo An, Yang Gao, Wenjun Wu, Lei Bai, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16988)  

**Abstract**: LLM-based multi-agent systems (MAS) have demonstrated significant potential in enhancing single LLMs to address complex and diverse tasks in practical applications. Despite considerable advancements, the field lacks a unified codebase that consolidates existing methods, resulting in redundant re-implementation efforts, unfair comparisons, and high entry barriers for researchers. To address these challenges, we introduce MASLab, a unified, comprehensive, and research-friendly codebase for LLM-based MAS. (1) MASLab integrates over 20 established methods across multiple domains, each rigorously validated by comparing step-by-step outputs with its official implementation. (2) MASLab provides a unified environment with various benchmarks for fair comparisons among methods, ensuring consistent inputs and standardized evaluation protocols. (3) MASLab implements methods within a shared streamlined structure, lowering the barriers for understanding and extension. Building on MASLab, we conduct extensive experiments covering 10+ benchmarks and 8 models, offering researchers a clear and comprehensive view of the current landscape of MAS methods. MASLab will continue to evolve, tracking the latest developments in the field, and invite contributions from the broader open-source community. 

---
# T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning 

**Authors**: Amartya Chakraborty, Paresh Dashore, Nadia Bathaee, Anmol Jain, Anirban Das, Shi-Xiong Zhang, Sambit Sahu, Milind Naphade, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2505.16986)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities as intelligent agents capable of solving complex problems. However, effective planning in scenarios involving dependencies between API or tool calls-particularly in multi-turn conversations-remains a significant challenge. To address this, we introduce T1, a tool-augmented, multi-domain, multi-turn conversational dataset specifically designed to capture and manage inter-tool dependencies across diverse domains. T1 enables rigorous evaluation of agents' ability to coordinate tool use across nine distinct domains (4 single domain and 5 multi-domain) with the help of an integrated caching mechanism for both short- and long-term memory, while supporting dynamic replanning-such as deciding whether to recompute or reuse cached results. Beyond facilitating research on tool use and planning, T1 also serves as a benchmark for evaluating the performance of open-source language models. We present results powered by T1-Agent, highlighting their ability to plan and reason in complex, tool-dependent scenarios. 

---
# Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation 

**Authors**: Moru Liu, Hao Dong, Jessica Kelly, Olga Fink, Mario Trapp  

**Link**: [PDF](https://arxiv.org/pdf/2505.16985)  

**Abstract**: Out-of-distribution (OOD) detection and segmentation are crucial for deploying machine learning models in safety-critical applications such as autonomous driving and robot-assisted surgery. While prior research has primarily focused on unimodal image data, real-world applications are inherently multimodal, requiring the integration of multiple modalities for improved OOD detection. A key challenge is the lack of supervision signals from unknown data, leading to overconfident predictions on OOD samples. To address this challenge, we propose Feature Mixing, an extremely simple and fast method for multimodal outlier synthesis with theoretical support, which can be further optimized to help the model better distinguish between in-distribution (ID) and OOD data. Feature Mixing is modality-agnostic and applicable to various modality combinations. Additionally, we introduce CARLA-OOD, a novel multimodal dataset for OOD segmentation, featuring synthetic OOD objects across diverse scenes and weather conditions. Extensive experiments on SemanticKITTI, nuScenes, CARLA-OOD datasets, and the MultiOOD benchmark demonstrate that Feature Mixing achieves state-of-the-art performance with a $10 \times$ to $370 \times$ speedup. Our source code and dataset will be available at this https URL. 

---
# CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark 

**Authors**: Ahmed Heakl, Sarim Hashmi, Gustavo Bertolo Stahl, Seung Hun Eddie Han, Salman Khan, Abdulrahman Mahmoud  

**Link**: [PDF](https://arxiv.org/pdf/2505.16968)  

**Abstract**: We introduce \texttt{CASS}, the first large-scale dataset and model suite for cross-architecture GPU code transpilation, targeting both source-level (CUDA~$\leftrightarrow$~HIP) and assembly-level (Nvidia SASS~$\leftrightarrow$~AMD RDNA3) translation. The dataset comprises 70k verified code pairs across host and device, addressing a critical gap in low-level GPU code portability. Leveraging this resource, we train the \texttt{CASS} family of domain-specific language models, achieving 95\% source translation accuracy and 37.5\% assembly translation accuracy, substantially outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our generated code matches native performance in over 85\% of test cases, preserving runtime and memory behavior. To support rigorous evaluation, we introduce \texttt{CASS-Bench}, a curated benchmark spanning 16 GPU domains with ground-truth execution. All data, models, and evaluation tools are released as open source to foster progress in GPU compiler tooling, binary compatibility, and LLM-guided hardware translation. Dataset and benchmark are on \href{this https URL}{\textcolor{blue}{HuggingFace}}, with code at \href{this https URL}{\textcolor{blue}{GitHub}}. 

---
# Fixing Data That Hurts Performance: Cascading LLMs to Relabel Hard Negatives for Robust Information Retrieval 

**Authors**: Nandan Thakur, Crystina Zhang, Xueguang Ma, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16967)  

**Abstract**: Training robust retrieval and reranker models typically relies on large-scale retrieval datasets; for example, the BGE collection contains 1.6 million query-passage pairs sourced from various data sources. However, we find that certain datasets can negatively impact model effectiveness -- pruning 8 out of 15 datasets from the BGE collection reduces the training set size by 2.35$\times$ and increases nDCG@10 on BEIR by 1.0 point. This motivates a deeper examination of training data quality, with a particular focus on "false negatives", where relevant passages are incorrectly labeled as irrelevant. We propose a simple, cost-effective approach using cascading LLM prompts to identify and relabel hard negatives. Experimental results show that relabeling false negatives with true positives improves both E5 (base) and Qwen2.5-7B retrieval models by 0.7-1.4 nDCG@10 on BEIR and by 1.7-1.8 nDCG@10 on zero-shot AIR-Bench evaluation. Similar gains are observed for rerankers fine-tuned on the relabeled data, such as Qwen2.5-3B on BEIR. The reliability of the cascading design is further supported by human annotation results, where we find judgment by GPT-4o shows much higher agreement with humans than GPT-4o-mini. 

---
# BP-Seg: A graphical model approach to unsupervised and non-contiguous text segmentation using belief propagation 

**Authors**: Fengyi Li, Kayhan Behdin, Natesh Pillai, Xiaofeng Wang, Zhipeng Wang, Ercan Yildiz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16965)  

**Abstract**: Text segmentation based on the semantic meaning of sentences is a fundamental task with broad utility in many downstream applications. In this paper, we propose a graphical model-based unsupervised learning approach, named BP-Seg for efficient text segmentation. Our method not only considers local coherence, capturing the intuition that adjacent sentences are often more related, but also effectively groups sentences that are distant in the text yet semantically similar. This is achieved through belief propagation on the carefully constructed graphical models. Experimental results on both an illustrative example and a dataset with long-form documents demonstrate that our method performs favorably compared to competing approaches. 

---
# Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models 

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16957)  

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content. 

---
# Bottlenecked Transformers: Periodic KV Cache Abstraction for Generalised Reasoning 

**Authors**: Adnan Oomerjee, Zafeirios Fountas, Zhongwei Yu, Haitham Bou-Ammar, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16950)  

**Abstract**: Despite their impressive capabilities, Large Language Models struggle with generalisation beyond their training distribution, often exhibiting sophisticated pattern interpolation rather than true abstract reasoning (extrapolation). In this work, we approach this limitation through the lens of Information Bottleneck (IB) theory, which posits that model generalisation emerges from an optimal balance between input compression and retention of predictive information in latent representations. We prove using IB theory that decoder-only Transformers are inherently constrained in their ability to form task-optimal sequence representations. We then use this result to demonstrate that periodic global transformation of the internal sequence-level representations (KV cache) is a necessary computational step for improving Transformer generalisation in reasoning tasks. Based on these theoretical insights, we propose a modification to the Transformer architecture, in the form of an additional module that globally rewrites the KV cache at periodic intervals, shifting its capacity away from memorising input prefixes and toward encoding features most useful for predicting future tokens. Our model delivers substantial gains on mathematical reasoning benchmarks, outperforming both vanilla Transformers with up to 3.5x more parameters, as well as heuristic-driven pruning mechanisms for cache compression. Our approach can be seen as a principled generalisation of existing KV-cache compression methods; whereas such methods focus solely on compressing input representations, they often do so at the expense of retaining predictive information, and thus their capabilities are inherently bounded by those of an unconstrained model. This establishes a principled framework to manipulate Transformer memory using information theory, addressing fundamental reasoning limitations that scaling alone cannot overcome. 

---
# MixAT: Combining Continuous and Discrete Adversarial Training for LLMs 

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16947)  

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at this https URL. 

---
# FoMoH: A clinically meaningful foundation model evaluation for structured electronic health records 

**Authors**: Chao Pang, Vincent Jeanselme, Young Sang Choi, Xinzhuo Jiang, Zilin Jing, Aparajita Kashyap, Yuta Kobayashi, Yanwei Li, Florent Pollet, Karthik Natarajan, Shalmali Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16941)  

**Abstract**: Foundation models hold significant promise in healthcare, given their capacity to extract meaningful representations independent of downstream tasks. This property has enabled state-of-the-art performance across several clinical applications trained on structured electronic health record (EHR) data, even in settings with limited labeled data, a prevalent challenge in healthcare. However, there is little consensus on these models' potential for clinical utility due to the lack of desiderata of comprehensive and meaningful tasks and sufficiently diverse evaluations to characterize the benefit over conventional supervised learning. To address this gap, we propose a suite of clinically meaningful tasks spanning patient outcomes, early prediction of acute and chronic conditions, including desiderata for robust evaluations. We evaluate state-of-the-art foundation models on EHR data consisting of 5 million patients from Columbia University Irving Medical Center (CUMC), a large urban academic medical center in New York City, across 14 clinically relevant tasks. We measure overall accuracy, calibration, and subpopulation performance to surface tradeoffs based on the choice of pre-training, tokenization, and data representation strategies. Our study aims to advance the empirical evaluation of structured EHR foundation models and guide the development of future healthcare foundation models. 

---
# The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm 

**Authors**: Noah Amsel, David Persson, Christopher Musco, Robert Gower  

**Link**: [PDF](https://arxiv.org/pdf/2505.16932)  

**Abstract**: Computing the polar decomposition and the related matrix sign function, has been a well-studied problem in numerical analysis for decades. More recently, it has emerged as an important subroutine in deep learning, particularly within the Muon optimization framework. However, the requirements in this setting differ significantly from those of traditional numerical analysis. In deep learning, methods must be highly efficient and GPU-compatible, but high accuracy is often unnecessary. As a result, classical algorithms like Newton-Schulz (which suffers from slow initial convergence) and methods based on rational functions (which rely on QR decompositions or matrix inverses) are poorly suited to this context. In this work, we introduce Polar Express, a GPU-friendly algorithm for computing the polar decomposition. Like classical polynomial methods such as Newton-Schulz, our approach uses only matrix-matrix multiplications, making it GPU-compatible. Motivated by earlier work of Chen & Chow and Nakatsukasa & Freund, Polar Express adapts the polynomial update rule at each iteration by solving a minimax optimization problem, and we prove that it enjoys a strong worst-case optimality guarantee. This property ensures both rapid early convergence and fast asymptotic convergence. We also address finite-precision issues, making it stable in bfloat16 in practice. We apply Polar Express within the Muon optimization framework and show consistent improvements in validation loss on large-scale models such as GPT-2, outperforming recent alternatives across a range of learning rates. 

---
# Latent Principle Discovery for Language Model Self-Improvement 

**Authors**: Keshav Ramji, Tahira Naseem, Ramón Fernandez Astudillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16927)  

**Abstract**: When language model (LM) users aim to improve the quality of its generations, it is crucial to specify concrete behavioral attributes that the model should strive to reflect. However, curating such principles across many domains, even non-exhaustively, requires a labor-intensive annotation process. To automate this process, we propose eliciting these latent attributes guiding model reasoning towards human-preferred responses by explicitly modeling them in a self-correction setting. Our approach mines new principles from the LM itself and compresses the discovered elements to an interpretable set via clustering. Specifically, we employ an approximation of posterior-regularized Monte Carlo Expectation-Maximization to both identify a condensed set of the most effective latent principles and teach the LM to strategically invoke them in order to intrinsically refine its responses. We demonstrate that bootstrapping our algorithm over multiple iterations enables smaller language models (7-8B parameters) to self-improve, achieving +8-10% in AlpacaEval win-rate, an average of +0.3 on MT-Bench, and +19-23% in principle-following win-rate on IFEval. We also show that clustering the principles yields interpretable and diverse model-generated constitutions while retaining model performance. The gains our method achieves highlight the potential of automated, principle-driven post-training recipes toward continual self-improvement. 

---
# DetailMaster: Can Your Text-to-Image Model Handle Long Prompts? 

**Authors**: Qirui Jiao, Daoyuan Chen, Yilun Huang, Xika Lin, Ying Shen, Yaliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16915)  

**Abstract**: While recent text-to-image (T2I) models show impressive capabilities in synthesizing images from brief descriptions, their performance significantly degrades when confronted with long, detail-intensive prompts required in professional applications. We present DetailMaster, the first comprehensive benchmark specifically designed to evaluate T2I models' systematical abilities to handle extended textual inputs that contain complex compositional requirements. Our benchmark introduces four critical evaluation dimensions: Character Attributes, Structured Character Locations, Multi-Dimensional Scene Attributes, and Explicit Spatial/Interactive Relationships. The benchmark comprises long and detail-rich prompts averaging 284.89 tokens, with high quality validated by expert annotators. Evaluation on 7 general-purpose and 5 long-prompt-optimized T2I models reveals critical performance limitations: state-of-the-art models achieve merely ~50% accuracy in key dimensions like attribute binding and spatial reasoning, while all models showing progressive performance degradation as prompt length increases. Our analysis highlights systemic failures in structural comprehension and detail overload handling, motivating future research into architectures with enhanced compositional reasoning. We open-source the dataset, data curation code, and evaluation tools to advance detail-rich T2I generation and enable broad applications that would otherwise be infeasible due to the lack of a dedicated benchmark. 

---
# Active Speech Enhancement: Active Speech Denoising Decliping and Deveraberation 

**Authors**: Ofir Yaish, Yehuda Mishaly, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2505.16911)  

**Abstract**: We introduce a new paradigm for active sound modification: Active Speech Enhancement (ASE). While Active Noise Cancellation (ANC) algorithms focus on suppressing external interference, ASE goes further by actively shaping the speech signal -- both attenuating unwanted noise components and amplifying speech-relevant frequencies -- to improve intelligibility and perceptual quality. To enable this, we propose a novel Transformer-Mamba-based architecture, along with a task-specific loss function designed to jointly optimize interference suppression and signal enrichment. Our method outperforms existing baselines across multiple speech processing tasks -- including denoising, dereverberation, and declipping -- demonstrating the effectiveness of active, targeted modulation in challenging acoustic environments. 

---
# Structure-Aligned Protein Language Model 

**Authors**: Can Chen, David Heurtel-Depeiges, Robert M. Vernon, Christopher James Langmead, Yoshua Bengio, Quentin Fournier  

**Link**: [PDF](https://arxiv.org/pdf/2505.16896)  

**Abstract**: Protein language models (pLMs) pre-trained on vast protein sequence databases excel at various downstream tasks but lack the structural knowledge essential for many biological applications. To address this, we integrate structural insights from pre-trained protein graph neural networks (pGNNs) into pLMs through a latent-level contrastive learning task. This task aligns residue representations from pLMs with those from pGNNs across multiple proteins, enriching pLMs with inter-protein structural knowledge. Additionally, we incorporate a physical-level task that infuses intra-protein structural knowledge by optimizing pLMs to predict structural tokens. The proposed dual-task framework effectively incorporates both inter-protein and intra-protein structural knowledge into pLMs. Given the variability in the quality of protein structures in PDB, we further introduce a residue loss selection module, which uses a small model trained on high-quality structures to select reliable yet challenging residue losses for the pLM to learn. Applying our structure alignment method to the state-of-the-art ESM2 and AMPLIFY results in notable performance gains across a wide range of tasks, including a 12.7% increase in ESM2 contact prediction. The data, code, and resulting SaESM2 and SaAMPLIFY models will be released on Hugging Face. 

---
# CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework 

**Authors**: Viet Pham, Thai Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.16888)  

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available. 

---
# Don't "Overthink" Passage Reranking: Is Reasoning Truly Necessary? 

**Authors**: Nour Jedidi, Yung-Sung Chuang, James Glass, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16886)  

**Abstract**: With the growing success of reasoning models across complex natural language tasks, researchers in the Information Retrieval (IR) community have begun exploring how similar reasoning capabilities can be integrated into passage rerankers built on Large Language Models (LLMs). These methods typically employ an LLM to produce an explicit, step-by-step reasoning process before arriving at a final relevance prediction. But, does reasoning actually improve reranking accuracy? In this paper, we dive deeper into this question, studying the impact of the reasoning process by comparing reasoning-based pointwise rerankers (ReasonRR) to standard, non-reasoning pointwise rerankers (StandardRR) under identical training conditions, and observe that StandardRR generally outperforms ReasonRR. Building on this observation, we then study the importance of reasoning to ReasonRR by disabling its reasoning process (ReasonRR-NoReason), and find that ReasonRR-NoReason is surprisingly more effective than ReasonRR. Examining the cause of this result, our findings reveal that reasoning-based rerankers are limited by the LLM's reasoning process, which pushes it toward polarized relevance scores and thus fails to consider the partial relevance of passages, a key factor for the accuracy of pointwise rerankers. 

---
# CASTILLO: Characterizing Response Length Distributions of Large Language Models 

**Authors**: Daniel F. Perez-Ramirez, Dejan Kostic, Magnus Boman  

**Link**: [PDF](https://arxiv.org/pdf/2505.16881)  

**Abstract**: Efficiently managing compute resources for Large Language Model (LLM) inference remains challenging due to the inherently stochastic and variable lengths of autoregressive text generation. Accurately estimating response lengths in advance enables proactive resource allocation, yet existing approaches either bias text generation towards certain lengths or rely on assumptions that ignore model- and prompt-specific variability. We introduce CASTILLO, a dataset characterizing response length distributions across 13 widely-used open-source LLMs evaluated on seven distinct instruction-following corpora. For each $\langle$prompt, model$\rangle$ sample pair, we generate 10 independent completions using fixed decoding hyper-parameters, record the token length of each response, and publish summary statistics (mean, std-dev, percentiles), along with the shortest and longest completions, and the exact generation settings. Our analysis reveals significant inter- and intra-model variability in response lengths (even under identical generation settings), as well as model-specific behaviors and occurrences of partial text degeneration in only subsets of responses. CASTILLO enables the development of predictive models for proactive scheduling and provides a systematic framework for analyzing model-specific generation behaviors. We publicly release the dataset and code to foster research at the intersection of generative language modeling and systems. 

---
# T2I-ConBench: Text-to-Image Benchmark for Continual Post-training 

**Authors**: Zhehao Huang, Yuhang Liu, Yixin Lou, Zhengbao He, Mingzhen He, Wenxing Zhou, Tao Li, Kehan Li, Zeyi Huang, Xiaolin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16875)  

**Abstract**: Continual post-training adapts a single text-to-image diffusion model to learn new tasks without incurring the cost of separate models, but naive post-training causes forgetting of pretrained knowledge and undermines zero-shot compositionality. We observe that the absence of a standardized evaluation protocol hampers related research for continual post-training. To address this, we introduce T2I-ConBench, a unified benchmark for continual post-training of text-to-image models. T2I-ConBench focuses on two practical scenarios, item customization and domain enhancement, and analyzes four dimensions: (1) retention of generality, (2) target-task performance, (3) catastrophic forgetting, and (4) cross-task generalization. It combines automated metrics, human-preference modeling, and vision-language QA for comprehensive assessment. We benchmark ten representative methods across three realistic task sequences and find that no approach excels on all fronts. Even joint "oracle" training does not succeed for every task, and cross-task generalization remains unsolved. We release all datasets, code, and evaluation tools to accelerate research in continual post-training for text-to-image models. 

---
# GCAL: Adapting Graph Models to Evolving Domain Shifts 

**Authors**: Ziyue Qiao, Qianyi Cai, Hao Dong, Jiawei Gu, Pengyang Wang, Meng Xiao, Xiao Luo, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.16860)  

**Abstract**: This paper addresses the challenge of graph domain adaptation on evolving, multiple out-of-distribution (OOD) graphs. Conventional graph domain adaptation methods are confined to single-step adaptation, making them ineffective in handling continuous domain shifts and prone to catastrophic forgetting. This paper introduces the Graph Continual Adaptive Learning (GCAL) method, designed to enhance model sustainability and adaptability across various graph domains. GCAL employs a bilevel optimization strategy. The "adapt" phase uses an information maximization approach to fine-tune the model with new graph domains while re-adapting past memories to mitigate forgetting. Concurrently, the "generate memory" phase, guided by a theoretical lower bound derived from information bottleneck theory, involves a variational memory graph generation module to condense original graphs into memories. Extensive experimental evaluations demonstrate that GCAL substantially outperforms existing methods in terms of adaptability and knowledge retention. 

---
# Efficient Online RL Fine Tuning with Offline Pre-trained Policy Only 

**Authors**: Wei Xiao, Jiacheng Liu, Zifeng Zhuang, Runze Suo, Shangke Lyu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16856)  

**Abstract**: Improving the performance of pre-trained policies through online reinforcement learning (RL) is a critical yet challenging topic. Existing online RL fine-tuning methods require continued training with offline pretrained Q-functions for stability and performance. However, these offline pretrained Q-functions commonly underestimate state-action pairs beyond the offline dataset due to the conservatism in most offline RL methods, which hinders further exploration when transitioning from the offline to the online setting. Additionally, this requirement limits their applicability in scenarios where only pre-trained policies are available but pre-trained Q-functions are absent, such as in imitation learning (IL) pre-training. To address these challenges, we propose a method for efficient online RL fine-tuning using solely the offline pre-trained policy, eliminating reliance on pre-trained Q-functions. We introduce PORL (Policy-Only Reinforcement Learning Fine-Tuning), which rapidly initializes the Q-function from scratch during the online phase to avoid detrimental pessimism. Our method not only achieves competitive performance with advanced offline-to-online RL algorithms and online RL approaches that leverage data or policies prior, but also pioneers a new path for directly fine-tuning behavior cloning (BC) policies. 

---
# Unlocking Temporal Flexibility: Neural Speech Codec with Variable Frame Rate 

**Authors**: Hanglei Zhang, Yiwei Guo, Zhihan Li, Xiang Hao, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16845)  

**Abstract**: Most neural speech codecs achieve bitrate adjustment through intra-frame mechanisms, such as codebook dropout, at a Constant Frame Rate (CFR). However, speech segments inherently have time-varying information density (e.g., silent intervals versus voiced regions). This property makes CFR not optimal in terms of bitrate and token sequence length, hindering efficiency in real-time applications. In this work, we propose a Temporally Flexible Coding (TFC) technique, introducing variable frame rate (VFR) into neural speech codecs for the first time. TFC enables seamlessly tunable average frame rates and dynamically allocates frame rates based on temporal entropy. Experimental results show that a codec with TFC achieves optimal reconstruction quality with high flexibility, and maintains competitive performance even at lower frame rates. Our approach is promising for the integration with other efforts to develop low-frame-rate neural speech codecs for more efficient downstream tasks. 

---
# Fact-R1: Towards Explainable Video Misinformation Detection with Deep Reasoning 

**Authors**: Fanrui Zhang, Dian Li, Qiang Zhang, Chenjun, sinbadliu, Junxiong Lin, Jiahong Yan, Jiawei Liu, Zheng-Jun Zha  

**Link**: [PDF](https://arxiv.org/pdf/2505.16836)  

**Abstract**: The rapid spread of multimodal misinformation on social media has raised growing concerns, while research on video misinformation detection remains limited due to the lack of large-scale, diverse datasets. Existing methods often overfit to rigid templates and lack deep reasoning over deceptive content. To address these challenges, we introduce FakeVV, a large-scale benchmark comprising over 100,000 video-text pairs with fine-grained, interpretable annotations. In addition, we further propose Fact-R1, a novel framework that integrates deep reasoning with collaborative rule-based reinforcement learning. Fact-R1 is trained through a three-stage process: (1) misinformation long-Chain-of-Thought (CoT) instruction tuning, (2) preference alignment via Direct Preference Optimization (DPO), and (3) Group Relative Policy Optimization (GRPO) using a novel verifiable reward function. This enables Fact-R1 to exhibit emergent reasoning behaviors comparable to those observed in advanced text-based reinforcement learning systems, but in the more complex multimodal misinformation setting. Our work establishes a new paradigm for misinformation detection, bridging large-scale video understanding, reasoning-guided alignment, and interpretable verification. 

---
# SimpleDeepSearcher: Deep Information Seeking via Web-Powered Reasoning Trajectory Synthesis 

**Authors**: Shuang Sun, Huatong Song, Yuhao Wang, Ruiyang Ren, Jinhao Jiang, Junjie Zhang, Fei Bai, Jia Deng, Wayne Xin Zhao, Zheng Liu, Lei Fang, Zhongyuan Wang, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16834)  

**Abstract**: Retrieval-augmented generation (RAG) systems have advanced large language models (LLMs) in complex deep search scenarios requiring multi-step reasoning and iterative information retrieval. However, existing approaches face critical limitations that lack high-quality training trajectories or suffer from the distributional mismatches in simulated environments and prohibitive computational costs for real-world deployment. This paper introduces SimpleDeepSearcher, a lightweight yet effective framework that bridges this gap through strategic data engineering rather than complex training paradigms. Our approach synthesizes high-quality training data by simulating realistic user interactions in live web search environments, coupled with a multi-criteria curation strategy that optimizes the diversity and quality of input and output side. Experiments on five benchmarks across diverse domains demonstrate that SFT on only 871 curated samples yields significant improvements over RL-based baselines. Our work establishes SFT as a viable pathway by systematically addressing the data-scarce bottleneck, offering practical insights for efficient deep search systems. Our code is available at this https URL. 

---
# Unlearning Isn't Deletion: Investigating Reversibility of Machine Unlearning in LLMs 

**Authors**: Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Haibo Hu, Minxin Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.16831)  

**Abstract**: Unlearning in large language models (LLMs) is intended to remove the influence of specific data, yet current evaluations rely heavily on token-level metrics such as accuracy and perplexity. We show that these metrics can be misleading: models often appear to forget, but their original behavior can be rapidly restored with minimal fine-tuning, revealing that unlearning may obscure information rather than erase it. To diagnose this phenomenon, we introduce a representation-level evaluation framework using PCA-based similarity and shift, centered kernel alignment, and Fisher information. Applying this toolkit across six unlearning methods, three domains (text, code, math), and two open-source LLMs, we uncover a critical distinction between reversible and irreversible forgetting. In reversible cases, models suffer token-level collapse yet retain latent features; in irreversible cases, deeper representational damage occurs. We further provide a theoretical account linking shallow weight perturbations near output layers to misleading unlearning signals, and show that reversibility is modulated by task type and hyperparameters. Our findings reveal a fundamental gap in current evaluation practices and establish a new diagnostic foundation for trustworthy unlearning in LLMs. We provide a unified toolkit for analyzing LLM representation changes under unlearning and relearning: this https URL. 

---
# Dynamic Reservoir Computing with Physical Neuromorphic Networks 

**Authors**: Yinhao Xu, Georg A. Gottwald, Zdenka Kuncic  

**Link**: [PDF](https://arxiv.org/pdf/2505.16813)  

**Abstract**: Reservoir Computing (RC) with physical systems requires an understanding of the underlying structure and internal dynamics of the specific physical reservoir. In this study, physical nano-electronic networks with neuromorphic dynamics are investigated for their use as physical reservoirs in an RC framework. These neuromorphic networks operate as dynamic reservoirs, with node activities in general coupled to the edge dynamics through nonlinear nano-electronic circuit elements, and the reservoir outputs influenced by the underlying network connectivity structure. This study finds that networks with varying degrees of sparsity generate more useful nonlinear temporal outputs for dynamic RC compared to dense networks. Dynamic RC is also tested on an autonomous multivariate chaotic time series prediction task with networks of varying densities, which revealed the importance of network sparsity in maintaining network activity and overall dynamics, that in turn enabled the learning of the chaotic Lorenz63 system's attractor behavior. 

---
# A modular framework for automated evaluation of procedural content generation in serious games with deep reinforcement learning agents 

**Authors**: Eleftherios Kalafatis, Konstantinos Mitsis, Konstantia Zarkogianni, Maria Athanasiou, Konstantina Nikita  

**Link**: [PDF](https://arxiv.org/pdf/2505.16801)  

**Abstract**: Serious Games (SGs) are nowadays shifting focus to include procedural content generation (PCG) in the development process as a means of offering personalized and enhanced player experience. However, the development of a framework to assess the impact of PCG techniques when integrated into SGs remains particularly challenging. This study proposes a methodology for automated evaluation of PCG integration in SGs, incorporating deep reinforcement learning (DRL) game testing agents. To validate the proposed framework, a previously introduced SG featuring card game mechanics and incorporating three different versions of PCG for nonplayer character (NPC) creation has been deployed. Version 1 features random NPC creation, while versions 2 and 3 utilize a genetic algorithm approach. These versions are used to test the impact of different dynamic SG environments on the proposed framework's agents. The obtained results highlight the superiority of the DRL game testing agents trained on Versions 2 and 3 over those trained on Version 1 in terms of win rate (i.e. number of wins per played games) and training time. More specifically, within the execution of a test emulating regular gameplay, both Versions 2 and 3 peaked at a 97% win rate and achieved statistically significant higher (p=0009) win rates compared to those achieved in Version 1 that peaked at 94%. Overall, results advocate towards the proposed framework's capability to produce meaningful data for the evaluation of procedurally generated content in SGs. 

---
# SEED: Speaker Embedding Enhancement Diffusion Model 

**Authors**: KiHyun Nam, Jungwoo Heo, Jee-weon Jung, Gangin Park, Chaeyoung Jung, Ha-Jin Yu, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2505.16798)  

**Abstract**: A primary challenge when deploying speaker recognition systems in real-world applications is performance degradation caused by environmental mismatch. We propose a diffusion-based method that takes speaker embeddings extracted from a pre-trained speaker recognition model and generates refined embeddings. For training, our approach progressively adds Gaussian noise to both clean and noisy speaker embeddings extracted from clean and noisy speech, respectively, via forward process of a diffusion model, and then reconstructs them to clean embeddings in the reverse process. While inferencing, all embeddings are regenerated via diffusion process. Our method needs neither speaker label nor any modification to the existing speaker recognition pipeline. Experiments on evaluation sets simulating environment mismatch scenarios show that our method can improve recognition accuracy by up to 19.6% over baseline models while retaining performance on conventional scenarios. We publish our code here this https URL 

---
# REPA Works Until It Doesn't: Early-Stopped, Holistic Alignment Supercharges Diffusion Training 

**Authors**: Ziqiao Wang, Wangbo Zhao, Yuhao Zhou, Zekai Li, Zhiyuan Liang, Mingjia Shi, Xuanlei Zhao, Pengfei Zhou, Kaipeng Zhang, Zhangyang Wang, Kai Wang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2505.16792)  

**Abstract**: Diffusion Transformers (DiTs) deliver state-of-the-art image quality, yet their training remains notoriously slow. A recent remedy -- representation alignment (REPA) that matches DiT hidden features to those of a non-generative teacher (e.g. DINO) -- dramatically accelerates the early epochs but plateaus or even degrades performance later. We trace this failure to a capacity mismatch: once the generative student begins modelling the joint data distribution, the teacher's lower-dimensional embeddings and attention patterns become a straitjacket rather than a guide. We then introduce HASTE (Holistic Alignment with Stage-wise Termination for Efficient training), a two-phase schedule that keeps the help and drops the hindrance. Phase I applies a holistic alignment loss that simultaneously distills attention maps (relational priors) and feature projections (semantic anchors) from the teacher into mid-level layers of the DiT, yielding rapid convergence. Phase II then performs one-shot termination that deactivates the alignment loss, once a simple trigger such as a fixed iteration is hit, freeing the DiT to focus on denoising and exploit its generative capacity. HASTE speeds up training of diverse DiTs without architecture changes. On ImageNet 256X256, it reaches the vanilla SiT-XL/2 baseline FID in 50 epochs and matches REPA's best FID in 500 epochs, amounting to a 28X reduction in optimization steps. HASTE also improves text-to-image DiTs on MS-COCO, demonstrating to be a simple yet principled recipe for efficient diffusion training across various tasks. Our code is available at this https URL . 

---
# Cohort-Based Active Modality Acquisition 

**Authors**: Tillmann Rheude, Roland Eils, Benjamin Wild  

**Link**: [PDF](https://arxiv.org/pdf/2505.16791)  

**Abstract**: Real-world machine learning applications often involve data from multiple modalities that must be integrated effectively to make robust predictions. However, in many practical settings, not all modalities are available for every sample, and acquiring additional modalities can be costly. This raises the question: which samples should be prioritized for additional modality acquisition when resources are limited? While prior work has explored individual-level acquisition strategies and training-time active learning paradigms, test-time and cohort-based acquisition remain underexplored despite their importance in many real-world settings. We introduce Cohort-based Active Modality Acquisition (CAMA), a novel test-time setting to formalize the challenge of selecting which samples should receive additional modalities. We derive acquisition strategies that leverage a combination of generative imputation and discriminative modeling to estimate the expected benefit of acquiring missing modalities based on common evaluation metrics. We also introduce upper-bound heuristics that provide performance ceilings to benchmark acquisition strategies. Experiments on common multimodal datasets demonstrate that our proposed imputation-based strategies can more effectively guide the acquisition of new samples in comparison to those relying solely on unimodal information, entropy guidance, and random selections. Our work provides an effective solution for optimizing modality acquisition at the cohort level, enabling better utilization of resources in constrained settings. 

---
# Learning Flexible Forward Trajectories for Masked Molecular Diffusion 

**Authors**: Hyunjin Seo, Taewon Kim, Sihyun Yu, SungSoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.16790)  

**Abstract**: Masked diffusion models (MDMs) have achieved notable progress in modeling discrete data, while their potential in molecular generation remains underexplored. In this work, we explore their potential and introduce the surprising result that naively applying standards MDMs severely degrades the performance. We identify the critical cause of this issue as a state-clashing problem-where the forward diffusion of distinct molecules collapse into a common state, resulting in a mixture of reconstruction targets that cannot be learned using typical reverse diffusion process with unimodal predictions. To mitigate this, we propose Masked Element-wise Learnable Diffusion (MELD) that orchestrates per-element corruption trajectories to avoid collision between distinct molecular graphs. This is achieved through a parameterized noise scheduling network that assigns distinct corruption rates to individual graph elements, i.e., atoms and bonds. Extensive experiments on diverse molecular benchmarks reveal that MELD markedly enhances overall generation quality compared to element-agnostic noise scheduling, increasing the chemical validity of vanilla MDMs on ZINC250K from 15% to 93%, Furthermore, it achieves state-of-the-art property alignment in conditional generation tasks. 

---
# Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability 

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.16789)  

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at this https URL. 

---
# CoTSRF: Utilize Chain of Thought as Stealthy and Robust Fingerprint of Large Language Models 

**Authors**: Zhenzhen Ren, GuoBiao Li, Sheng Li, Zhenxing Qian, Xinpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16785)  

**Abstract**: Despite providing superior performance, open-source large language models (LLMs) are vulnerable to abusive usage. To address this issue, recent works propose LLM fingerprinting methods to identify the specific source LLMs behind suspect applications. However, these methods fail to provide stealthy and robust fingerprint verification. In this paper, we propose a novel LLM fingerprinting scheme, namely CoTSRF, which utilizes the Chain of Thought (CoT) as the fingerprint of an LLM. CoTSRF first collects the responses from the source LLM by querying it with crafted CoT queries. Then, it applies contrastive learning to train a CoT extractor that extracts the CoT feature (i.e., fingerprint) from the responses. Finally, CoTSRF conducts fingerprint verification by comparing the Kullback-Leibler divergence between the CoT features of the source and suspect LLMs against an empirical threshold. Various experiments have been conducted to demonstrate the advantage of our proposed CoTSRF for fingerprinting LLMs, particularly in stealthy and robust fingerprint verification. 

---
# Mitigating Overfitting in Medical Imaging: Self-Supervised Pretraining vs. ImageNet Transfer Learning for Dermatological Diagnosis 

**Authors**: Iván Matas, Carmen Serrano, Miguel Nogales, David Moreno, Lara Ferrándiz, Teresa Ojeda, Begoña Acha  

**Link**: [PDF](https://arxiv.org/pdf/2505.16773)  

**Abstract**: Deep learning has transformed computer vision but relies heavily on large labeled datasets and computational resources. Transfer learning, particularly fine-tuning pretrained models, offers a practical alternative; however, models pretrained on natural image datasets such as ImageNet may fail to capture domain-specific characteristics in medical imaging. This study introduces an unsupervised learning framework that extracts high-value dermatological features instead of relying solely on ImageNet-based pretraining. We employ a Variational Autoencoder (VAE) trained from scratch on a proprietary dermatological dataset, allowing the model to learn a structured and clinically relevant latent space. This self-supervised feature extractor is then compared to an ImageNet-pretrained backbone under identical classification conditions, highlighting the trade-offs between general-purpose and domain-specific pretraining. Our results reveal distinct learning patterns. The self-supervised model achieves a final validation loss of 0.110 (-33.33%), while the ImageNet-pretrained model stagnates at 0.100 (-16.67%), indicating overfitting. Accuracy trends confirm this: the self-supervised model improves from 45% to 65% (+44.44%) with a near-zero overfitting gap, whereas the ImageNet-pretrained model reaches 87% (+50.00%) but plateaus at 75% (+19.05%), with its overfitting gap increasing to +0.060. These findings suggest that while ImageNet pretraining accelerates convergence, it also amplifies overfitting on non-clinically relevant features. In contrast, self-supervised learning achieves steady improvements, stronger generalization, and superior adaptability, underscoring the importance of domain-specific feature extraction in medical imaging. 

---
# When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques 

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16765)  

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at this https URL 

---
# Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation 

**Authors**: Hao Guo, Erpeng Xue, Lei Huang, Shichao Wang, Xiaolei Wang, Lei Wang, Jinpeng Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16752)  

**Abstract**: We introduce the Dual-Flow Generative Ranking Network (DFGR), a two-stream architecture designed for recommendation systems. DFGR integrates innovative interaction patterns between real and fake flows within the QKV modules of the self-attention mechanism, enhancing both training and inference efficiency. This approach effectively addresses a key limitation observed in Meta's proposed HSTU generative recommendation approach, where heterogeneous information volumes are mapped into identical vector spaces, leading to training instability. Unlike traditional recommendation models, DFGR only relies on user history behavior sequences and minimal attribute information, eliminating the need for extensive manual feature engineering. Comprehensive evaluations on open-source and industrial datasets reveal DFGR's superior performance compared to established baselines such as DIN, DCN, DIEN, and DeepFM. We also investigate optimal parameter allocation strategies under computational constraints, establishing DFGR as an efficient and effective next-generation generate ranking paradigm. 

---
# TRIM: Achieving Extreme Sparsity with Targeted Row-wise Iterative Metric-driven Pruning 

**Authors**: Florentin Beck, William Rudman, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.16743)  

**Abstract**: Large Language Models (LLMs) present significant computational and memory challenges due to their extensive size, making pruning essential for their efficient deployment. Existing one-shot pruning methods often apply uniform sparsity constraints across layers or within each layer, resulting in suboptimal performance, especially at high sparsity ratios. This work introduces TRIM (Targeted Row-wise Iterative Metric-driven pruning), a novel approach that applies varying sparsity ratios to individual output dimensions (rows) within each layer. TRIM employs an iterative adjustment process guided by quality metrics to optimize dimension-wise sparsity allocation, focusing on reducing variance in quality retention across outputs to preserve critical information. TRIM can be seamlessly integrated with existing layer-wise pruning strategies. Our evaluations on perplexity and zero-shot tasks across diverse LLM families (Qwen2.5, LLaMA-2, and OPT) and sparsity levels demonstrate that TRIM achieves new state-of-the-art results and enhances stability. For instance, at 80% sparsity, TRIM reduces perplexity by 48% for Qwen2.5-14B and over 90% for OPT-13B compared to baseline methods. We conclude that fine-grained, dimension-wise sparsity adaptation is crucial for pushing the limits of extreme LLM compression. Code available at: this https URL 

---
# Robust Vision-Based Runway Detection through Conformal Prediction and Conformal mAP 

**Authors**: Alya Zouzou, Léo andéol, Mélanie Ducoffe, Ryma Boumazouza  

**Link**: [PDF](https://arxiv.org/pdf/2505.16740)  

**Abstract**: We explore the use of conformal prediction to provide statistical uncertainty guarantees for runway detection in vision-based landing systems (VLS). Using fine-tuned YOLOv5 and YOLOv6 models on aerial imagery, we apply conformal prediction to quantify localization reliability under user-defined risk levels. We also introduce Conformal mean Average Precision (C-mAP), a novel metric aligning object detection performance with conformal guarantees. Our results show that conformal prediction can improve the reliability of runway detection by quantifying uncertainty in a statistically sound way, increasing safety on-board and paving the way for certification of ML system in the aerospace domain. 

---
# Mitigating Fine-tuning Risks in LLMs via Safety-Aware Probing Optimization 

**Authors**: Chengcan Wu, Zhixin Zhang, Zeming Wei, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16737)  

**Abstract**: The significant progress of large language models (LLMs) has led to remarkable achievements across numerous applications. However, their ability to generate harmful content has sparked substantial safety concerns. Despite the implementation of safety alignment techniques during the pre-training phase, recent research indicates that fine-tuning LLMs on adversarial or even benign data can inadvertently compromise their safety. In this paper, we re-examine the fundamental issue of why fine-tuning on non-harmful data still results in safety degradation. We introduce a safety-aware probing (SAP) optimization framework designed to mitigate the safety risks of fine-tuning LLMs. Specifically, SAP incorporates a safety-aware probe into the gradient propagation process, mitigating the model's risk of safety degradation by identifying potential pitfalls in gradient directions, thereby enhancing task-specific performance while successfully preserving model safety. Our extensive experimental results demonstrate that SAP effectively reduces harmfulness below the original fine-tuned model and achieves comparable test loss to standard fine-tuning methods. Our code is available at this https URL. 

---
# Adversarial Deep Metric Learning for Cross-Modal Audio-Text Alignment in Open-Vocabulary Keyword Spotting 

**Authors**: Youngmoon Jung, Yong-Hyeok Lee, Myunghun Jung, Jaeyoung Roh, Chang Woo Han, Hoon-Young Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.16735)  

**Abstract**: For text enrollment-based open-vocabulary keyword spotting (KWS), acoustic and text embeddings are typically compared at either the phoneme or utterance level. To facilitate this, we optimize acoustic and text encoders using deep metric learning (DML), enabling direct comparison of multi-modal embeddings in a shared embedding space. However, the inherent heterogeneity between audio and text modalities presents a significant challenge. To address this, we propose Modality Adversarial Learning (MAL), which reduces the domain gap in heterogeneous modality representations. Specifically, we train a modality classifier adversarially to encourage both encoders to generate modality-invariant embeddings. Additionally, we apply DML to achieve phoneme-level alignment between audio and text, and conduct comprehensive comparisons across various DML objectives. Experiments on the Wall Street Journal (WSJ) and LibriPhrase datasets demonstrate the effectiveness of the proposed approach. 

---
# Sequential Monte Carlo for Policy Optimization in Continuous POMDPs 

**Authors**: Hany Abdulsamad, Sahel Iqbal, Simo Särkkä  

**Link**: [PDF](https://arxiv.org/pdf/2505.16732)  

**Abstract**: Optimal decision-making under partial observability requires agents to balance reducing uncertainty (exploration) against pursuing immediate objectives (exploitation). In this paper, we introduce a novel policy optimization framework for continuous partially observable Markov decision processes (POMDPs) that explicitly addresses this challenge. Our method casts policy learning as probabilistic inference in a non-Markovian Feynman--Kac model that inherently captures the value of information gathering by anticipating future observations, without requiring extrinsic exploration bonuses or handcrafted heuristics. To optimize policies under this model, we develop a nested sequential Monte Carlo~(SMC) algorithm that efficiently estimates a history-dependent policy gradient under samples from the optimal trajectory distribution induced by the POMDP. We demonstrate the effectiveness of our algorithm across standard continuous POMDP benchmarks, where existing methods struggle to act under uncertainty. 

---
# Advancing Brainwave Modeling with a Codebook-Based Foundation Model 

**Authors**: Konstantinos Barmpas, Na Lee, Yannis Panagakis, Dimitrios A. Adamos, Nikolaos Laskaris, Stefanos Zafeiriou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16724)  

**Abstract**: Recent advances in large-scale pre-trained Electroencephalogram (EEG) models have shown great promise, driving progress in Brain-Computer Interfaces (BCIs) and healthcare applications. However, despite their success, many existing pre-trained models have struggled to fully capture the rich information content of neural oscillations, a limitation that fundamentally constrains their performance and generalizability across diverse BCI tasks. This limitation is frequently rooted in suboptimal architectural design choices which constrain their representational capacity. In this work, we introduce LaBraM++, an enhanced Large Brainwave Foundation Model (LBM) that incorporates principled improvements grounded in robust signal processing foundations. LaBraM++ demonstrates substantial gains across a variety of tasks, consistently outperforming its originally-based architecture and achieving competitive results when compared to other open-source LBMs. Its superior performance and training efficiency highlight its potential as a strong foundation for future advancements in LBMs. 

---
# Breaking mBad! Supervised Fine-tuning for Cross-Lingual Detoxification 

**Authors**: Himanshu Beniwal, Youngwoo Kim, Maarten Sap, Soham Dan, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16722)  

**Abstract**: As large language models (LLMs) become increasingly prevalent in global applications, ensuring that they are toxicity-free across diverse linguistic contexts remains a critical challenge. We explore "Cross-lingual Detoxification", a cross-lingual paradigm that mitigates toxicity, enabling detoxification capabilities to transfer between high and low-resource languages across different script families. We analyze cross-lingual detoxification's effectiveness through 504 extensive settings to evaluate toxicity reduction in cross-distribution settings with limited data and investigate how mitigation impacts model performance on non-toxic tasks, revealing trade-offs between safety and knowledge preservation. Our code and dataset are publicly available at this https URL. 

---
# Training Long-Context LLMs Efficiently via Chunk-wise Optimization 

**Authors**: Wenhao Li, Yuxin Zhang, Gen Luo, Daohai Yu, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.16710)  

**Abstract**: While long-context large language models (LLMs) exhibit remarkable document processing capabilities, their prohibitively high training costs often hinder customized applications. To mitigate this issue, we propose \textit{Sequential Chunk-wise Optimization} (SeCO), a memory-efficient training paradigm that partitions lengthy inputs into manageable chunks. Each chunk independently constructs its computational graph and performs localized backpropagation, ensuring that only one chunk's forward activations are stored in memory. Building on SeCO, we further introduce \textit{Sparse Chunk-wise Optimization} (SpaCO), which reduces computational overhead by selectively propagating gradients to specific chunks and incorporates a carefully designed compensation factor to ensure unbiased gradient estimation. SpaCO decouples the computational cost of backpropagation from the context length, enabling training time to gradually converge to inference time as sequences become longer. Implemented as lightweight training wrappers, both SeCO and SpaCO offer substantial practical benefits. For example, when fine-tuning an 8B model with LoRA on a single RTX 3090 GPU, SeCO expands maximum sequence length from 1K to 16K tokens, while SpaCO demonstrates accelerated training speed -- achieving up to 3x faster than SeCO under the same experimental setup. These innovations provide new insights into optimizing long-context models, making them more accessible for practical applications. We have open-sourced the code at \href{this https URL}{here}. 

---
# An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations 

**Authors**: Seonghwan Park, Jueun Mun, Donghyun Oh, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16705)  

**Abstract**: Concept bottleneck models (CBMs) ensure interpretability by decomposing predictions into human interpretable concepts. Yet the annotations used for training CBMs that enable this transparency are often noisy, and the impact of such corruption is not well understood. In this study, we present the first systematic study of noise in CBMs and show that even moderate corruption simultaneously impairs prediction performance, interpretability, and the intervention effectiveness. Our analysis identifies a susceptible subset of concepts whose accuracy declines far more than the average gap between noisy and clean supervision and whose corruption accounts for most performance loss. To mitigate this vulnerability we propose a two-stage framework. During training, sharpness-aware minimization stabilizes the learning of noise-sensitive concepts. During inference, where clean labels are unavailable, we rank concepts by predictive entropy and correct only the most uncertain ones, using uncertainty as a proxy for susceptibility. Theoretical analysis and extensive ablations elucidate why sharpness-aware training confers robustness and why uncertainty reliably identifies susceptible concepts, providing a principled basis that preserves both interpretability and resilience in the presence of noise. 

---
# Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence 

**Authors**: Gouki Minegishi, Hiroki Furuta, Shohei Taniguchi, Yusuke Iwasawa, Yutaka Matsuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16694)  

**Abstract**: Transformer-based language models exhibit In-Context Learning (ICL), where predictions are made adaptively based on context. While prior work links induction heads to ICL through a sudden jump in accuracy, this can only account for ICL when the answer is included within the context. However, an important property of practical ICL in large language models is the ability to meta-learn how to solve tasks from context, rather than just copying answers from context; how such an ability is obtained during training is largely unexplored. In this paper, we experimentally clarify how such meta-learning ability is acquired by analyzing the dynamics of the model's circuit during training. Specifically, we extend the copy task from previous research into an In-Context Meta Learning setting, where models must infer a task from examples to answer queries. Interestingly, in this setting, we find that there are multiple phases in the process of acquiring such abilities, and that a unique circuit emerges in each phase, contrasting with the single-phases change in induction heads. The emergence of such circuits can be related to several phenomena known in large language models, and our analysis lead to a deeper understanding of the source of the transformer's ICL ability. 

---
# EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion 

**Authors**: Advait Joglekar, Divyanshu Singh, Rooshil Rohit Bhatia, S. Umesh  

**Link**: [PDF](https://arxiv.org/pdf/2505.16691)  

**Abstract**: Voice Conversion research in recent times has increasingly focused on improving the zero-shot capabilities of existing methods. Despite remarkable advancements, current architectures still tend to struggle in zero-shot cross-lingual settings. They are also often unable to generalize for speakers of unseen languages and accents. In this paper, we adopt a simple yet effective approach that combines discrete speech representations from self-supervised models with a non-autoregressive Diffusion-Transformer based conditional flow matching speech decoder. We show that this architecture allows us to train a voice-conversion model in a purely textless, self-supervised fashion. Our technique works without requiring multiple encoders to disentangle speech features. Our model also manages to excel in zero-shot cross-lingual settings even for unseen languages. 

---
# Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator 

**Authors**: Beier Luo, Shuoyuan Wang, Yixuan Li, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16690)  

**Abstract**: Post-training of large language models is essential for adapting pre-trained language models (PLMs) to align with human preferences and downstream tasks. While PLMs typically exhibit well-calibrated confidence, post-trained language models (PoLMs) often suffer from over-confidence, assigning high confidence to both correct and incorrect outputs, which can undermine reliability in critical applications. A major obstacle in calibrating PoLMs is the scarcity of labeled data for individual downstream tasks. To address this, we propose Disagreement-Aware Confidence Alignment (DACA), a novel unsupervised method to optimize the parameters (e.g., temperature $\tau$) in post-hoc confidence calibration. Our method is motivated by the under-confidence issue caused by prediction disagreement between the PLM and PoLM while aligning their confidence via temperature scaling. Theoretically, the PLM's confidence underestimates PoLM's prediction accuracy on disagreement examples, causing a larger $\tau$ and producing under-confident predictions. DACA mitigates this by selectively using only agreement examples for calibration, effectively decoupling the influence of disagreement. In this manner, our method avoids an overly large $\tau$ in temperature scaling caused by disagreement examples, improving calibration performance. Extensive experiments demonstrate the effectiveness of our method, improving the average ECE of open-sourced and API-based LLMs (e.g. GPT-4o) by up to 15.08$\%$ on common benchmarks. 

---
# Semantic Compression of 3D Objects for Open and Collaborative Virtual Worlds 

**Authors**: Jordan Dotzel, Tony Montes, Mohamed S. Abdelfattah, Zhiru Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16679)  

**Abstract**: Traditional methods for 3D object compression operate only on structural information within the object vertices, polygons, and textures. These methods are effective at compression rates up to 10x for standard object sizes but quickly deteriorate at higher compression rates with texture artifacts, low-polygon counts, and mesh gaps. In contrast, semantic compression ignores structural information and operates directly on the core concepts to push to extreme levels of compression. In addition, it uses natural language as its storage format, which makes it natively human-readable and a natural fit for emerging applications built around large-scale, collaborative projects within augmented and virtual reality. It deprioritizes structural information like location, size, and orientation and predicts the missing information with state-of-the-art deep generative models. In this work, we construct a pipeline for 3D semantic compression from public generative models and explore the quality-compression frontier for 3D object compression. We apply this pipeline to achieve rates as high as 105x for 3D objects taken from the Objaverse dataset and show that semantic compression can outperform traditional methods in the important quality-preserving region around 100x compression. 

---
# R1-ShareVL: Incentivizing Reasoning Capability of Multimodal Large Language Models via Share-GRPO 

**Authors**: Huanjin Yao, Qixiang Yin, Jingyi Zhang, Min Yang, Yibo Wang, Wenhao Wu, Fei Su, Li Shen, Minghui Qiu, Dacheng Tao, Jiaxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16673)  

**Abstract**: In this work, we aim to incentivize the reasoning ability of Multimodal Large Language Models (MLLMs) via reinforcement learning (RL) and develop an effective approach that mitigates the sparse reward and advantage vanishing issues during RL. To this end, we propose Share-GRPO, a novel RL approach that tackle these issues by exploring and sharing diverse reasoning trajectories over expanded question space. Specifically, Share-GRPO first expands the question space for a given question via data transformation techniques, and then encourages MLLM to effectively explore diverse reasoning trajectories over the expanded question space and shares the discovered reasoning trajectories across the expanded questions during RL. In addition, Share-GRPO also shares reward information during advantage computation, which estimates solution advantages hierarchically across and within question variants, allowing more accurate estimation of relative advantages and improving the stability of policy training. Extensive evaluations over six widely-used reasoning benchmarks showcase the superior performance of our method. Code will be available at this https URL. 

---
# BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models 

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16670)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs. 

---
# End-to-End Framework for Predicting the Remaining Useful Life of Lithium-Ion Batteries 

**Authors**: Khoa Tran, Tri Le, Bao Huynh, Hung-Cuong Trinh, Vy-Rin Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16664)  

**Abstract**: Accurate prediction of the Remaining Useful Life (RUL) is essential for enabling timely maintenance of lithium-ion batteries, impacting the operational efficiency of electric applications that rely on them. This paper proposes a RUL prediction approach that leverages data from recent charge-discharge cycles to estimate the number of remaining usable cycles. The approach introduces both a novel signal processing pipeline and a deep learning prediction model. In the signal preprocessing pipeline, a derived capacity feature is computed based on current and capacity signals. Alongside original capacity, voltage and current, these features are denoised and enhanced using statistical metrics and a delta-based method to capture differences between the current and previous cycles. In the prediction model, the processed features are then fed into a hybrid deep learning architecture composed of 1D Convolutional Neural Networks (CNN), Attentional Long Short-Term Memory (A-LSTM), and Ordinary Differential Equation-based LSTM (ODE-LSTM) modules. This architecture is designed to capture both local signal characteristics and long-range temporal dependencies while modeling the continuous-time dynamics of battery degradation. The model is further evaluated using transfer learning across different learning strategies and target data partitioning scenarios. Results indicate that the model maintains robust performance, even when fine-tuned on limited target data. Experimental results on two publicly available large-scale datasets demonstrate that the proposed method outperforms a baseline deep learning approach and machine learning techniques, achieving an RMSE of 101.59, highlighting its strong potential for real-world RUL prediction applications. 

---
# Can reasoning models comprehend mathematical problems in Chinese ancient texts? An empirical study based on data from Suanjing Shishu 

**Authors**: Liu Chang, Wang Dongbo, Liu liu, Zhao Zhixiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16660)  

**Abstract**: This study addresses the challenges in intelligent processing of Chinese ancient mathematical classics by constructing Guji_MATH, a benchmark for evaluating classical texts based on Suanjing Shishu. It systematically assesses the mathematical problem-solving capabilities of mainstream reasoning models under the unique linguistic constraints of classical Chinese. Through machine-assisted annotation and manual verification, 538 mathematical problems were extracted from 8 canonical texts, forming a structured dataset centered on the "Question-Answer-Solution" framework, supplemented by problem types and difficulty levels. Dual evaluation modes--closed-book (autonomous problem-solving) and open-book (reproducing classical solution methods)--were designed to evaluate the performance of six reasoning models on ancient Chinese mathematical problems. Results indicate that reasoning models can partially comprehend and solve these problems, yet their overall performance remains inferior to benchmarks on modern mathematical tasks. Enhancing models' classical Chinese comprehension and cultural knowledge should be prioritized for optimization. This study provides methodological support for mining mathematical knowledge from ancient texts and disseminating traditional culture, while offering new perspectives for evaluating cross-linguistic and cross-cultural capabilities of reasoning models. 

---
# Collaboration among Multiple Large Language Models for Medical Question Answering 

**Authors**: Kexin Shang, Chia-Hsuan Chang, Christopher C. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16648)  

**Abstract**: Empowered by vast internal knowledge reservoir, the new generation of large language models (LLMs) demonstrate untapped potential to tackle medical tasks. However, there is insufficient effort made towards summoning up a synergic effect from multiple LLMs' expertise and background. In this study, we propose a multi-LLM collaboration framework tailored on a medical multiple-choice questions dataset. Through post-hoc analysis on 3 pre-trained LLM participants, our framework is proved to boost all LLMs reasoning ability as well as alleviate their divergence among questions. We also measure an LLM's confidence when it confronts with adversary opinions from other LLMs and observe a concurrence between LLM's confidence and prediction accuracy. 

---
# Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models 

**Authors**: Sushant Gautam, Michael A. Riegler, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16647)  

**Abstract**: We investigate fine-tuning Vision-Language Models (VLMs) for multi-task medical image understanding, focusing on detection, localization, and counting of findings in medical images. Our objective is to evaluate whether instruction-tuned VLMs can simultaneously improve these tasks, with the goal of enhancing diagnostic accuracy and efficiency. Using MedMultiPoints, a multimodal dataset with annotations from endoscopy (polyps and instruments) and microscopy (sperm cells), we reformulate each task into instruction-based prompts suitable for vision-language reasoning. We fine-tune Qwen2.5-VL-7B-Instruct using Low-Rank Adaptation (LoRA) across multiple task combinations. Results show that multi-task training improves robustness and accuracy. For example, it reduces the Count Mean Absolute Error (MAE) and increases Matching Accuracy in the Counting + Pointing task. However, trade-offs emerge, such as more zero-case point predictions, indicating reduced reliability in edge cases despite overall performance gains. Our study highlights the potential of adapting general-purpose VLMs to specialized medical tasks via prompt-driven fine-tuning. This approach mirrors clinical workflows, where radiologists simultaneously localize, count, and describe findings - demonstrating how VLMs can learn composite diagnostic reasoning patterns. The model produces interpretable, structured outputs, offering a promising step toward explainable and versatile medical AI. Code, model weights, and scripts will be released for reproducibility at this https URL. 

---
# From Evaluation to Defense: Advancing Safety in Video Large Language Models 

**Authors**: Yiwei Sun, Peiqi Jiang, Chuanbin Liu, Luohao Lin, Zhiying Lu, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16643)  

**Abstract**: While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.} 

---
# BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization 

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16640)  

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at this https URL. 

---
# SSR-Zero: Simple Self-Rewarding Reinforcement Learning for Machine Translation 

**Authors**: Wenjie Yang, Mao Zheng, Mingyang Song, Zheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16637)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable capabilities in machine translation (MT). However, most advanced MT-specific LLMs heavily rely on external supervision signals during training, such as human-annotated reference data or trained reward models (RMs), which are often expensive to obtain and challenging to scale. To overcome this limitation, we propose a Simple Self-Rewarding (SSR) Reinforcement Learning (RL) framework for MT that is reference-free, fully online, and relies solely on self-judging rewards. Training with SSR using 13K monolingual examples and Qwen-2.5-7B as the backbone, our model SSR-Zero-7B outperforms existing MT-specific LLMs, e.g., TowerInstruct-13B and GemmaX-28-9B, as well as larger general LLMs like Qwen2.5-32B-Instruct in English $\leftrightarrow$ Chinese translation tasks from WMT23, WMT24, and Flores200 benchmarks. Furthermore, by augmenting SSR with external supervision from COMET, our strongest model, SSR-X-Zero-7B, achieves state-of-the-art performance in English $\leftrightarrow$ Chinese translation, surpassing all existing open-source models under 72B parameters and even outperforming closed-source models, e.g., GPT-4o and Gemini 1.5 Pro. Our analysis highlights the effectiveness of the self-rewarding mechanism compared to the external LLM-as-a-judge approach in MT and demonstrates its complementary benefits when combined with trained RMs. Our findings provide valuable insight into the potential of self-improving RL methods. We have publicly released our code, data and models. 

---
# SoccerChat: Integrating Multimodal Data for Enhanced Soccer Game Understanding 

**Authors**: Sushant Gautam, Cise Midoglu, Vajira Thambawita, Michael A. Riegler, Pål Halvorsen, Mubarak Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.16630)  

**Abstract**: The integration of artificial intelligence in sports analytics has transformed soccer video understanding, enabling real-time, automated insights into complex game dynamics. Traditional approaches rely on isolated data streams, limiting their effectiveness in capturing the full context of a match. To address this, we introduce SoccerChat, a multimodal conversational AI framework that integrates visual and textual data for enhanced soccer video comprehension. Leveraging the extensive SoccerNet dataset, enriched with jersey color annotations and automatic speech recognition (ASR) transcripts, SoccerChat is fine-tuned on a structured video instruction dataset to facilitate accurate game understanding, event classification, and referee decision making. We benchmark SoccerChat on action classification and referee decision-making tasks, demonstrating its performance in general soccer event comprehension while maintaining competitive accuracy in referee decision making. Our findings highlight the importance of multimodal integration in advancing soccer analytics, paving the way for more interactive and explainable AI-driven sports analysis. this https URL 

---
# Steering Large Language Models for Machine Translation Personalization 

**Authors**: Daniel Scalena, Gabriele Sarti, Arianna Bisazza, Elisabetta Fersini, Malvina Nissim  

**Link**: [PDF](https://arxiv.org/pdf/2505.16612)  

**Abstract**: High-quality machine translation systems based on large language models (LLMs) have simplified the production of personalized translations reflecting specific stylistic constraints. However, these systems still struggle in settings where stylistic requirements are less explicit and might be harder to convey via prompting. We explore various strategies for personalizing LLM-generated translations in low-resource settings, focusing on the challenging literary translation domain. We explore prompting strategies and inference-time interventions for steering model generations towards a personalized style, and propose a contrastive framework exploiting latent concepts extracted from sparse autoencoders to identify salient personalization properties. Our results show that steering achieves strong personalization while preserving translation quality. We further examine the impact of steering on LLM representations, finding model layers with a relevant impact for personalization are impacted similarly by multi-shot prompting and our steering method, suggesting similar mechanism at play. 

---
# Safe Uncertainty-Aware Learning of Robotic Suturing 

**Authors**: Wilbert Peter Empleo, Yitaek Kim, Hansoul Kim, Thiusius Rajeeth Savarimuthu, Iñigo Iturrate  

**Link**: [PDF](https://arxiv.org/pdf/2505.16596)  

**Abstract**: Robot-Assisted Minimally Invasive Surgery is currently fully manually controlled by a trained surgeon. Automating this has great potential for alleviating issues, e.g., physical strain, highly repetitive tasks, and shortages of trained surgeons. For these reasons, recent works have utilized Artificial Intelligence methods, which show promising adaptability. Despite these advances, there is skepticism of these methods because they lack explainability and robust safety guarantees. This paper presents a framework for a safe, uncertainty-aware learning method. We train an Ensemble Model of Diffusion Policies using expert demonstrations of needle insertion. Using an Ensemble model, we can quantify the policy's epistemic uncertainty, which is used to determine Out-Of-Distribution scenarios. This allows the system to release control back to the surgeon in the event of an unsafe scenario. Additionally, we implement a model-free Control Barrier Function to place formal safety guarantees on the predicted action. We experimentally evaluate our proposed framework using a state-of-the-art robotic suturing simulator. We evaluate multiple scenarios, such as dropping the needle, moving the camera, and moving the phantom. The learned policy is robust to these perturbations, showing corrective behaviors and generalization, and it is possible to detect Out-Of-Distribution scenarios. We further demonstrate that the Control Barrier Function successfully limits the action to remain within our specified safety set in the case of unsafe predictions. 

---
# O$^2$-Searcher: A Searching-based Agent Model for Open-Domain Open-Ended Question Answering 

**Authors**: Jianbiao Mei, Tao Hu, Daocheng Fu, Licheng Wen, Xuemeng Yang, Rong Wu, Pinlong Cai, Xing Gao, Yu Yang, Chengjun Xie, Botian Shi, Yong Liu, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16582)  

**Abstract**: Large Language Models (LLMs), despite their advancements, are fundamentally limited by their static parametric knowledge, hindering performance on tasks requiring open-domain up-to-date information. While enabling LLMs to interact with external knowledge environments is a promising solution, current efforts primarily address closed-end problems. Open-ended questions, which characterized by lacking a standard answer or providing non-unique and diverse answers, remain underexplored. To bridge this gap, we present O$^2$-Searcher, a novel search agent leveraging reinforcement learning to effectively tackle both open-ended and closed-ended questions in the open domain. O$^2$-Searcher leverages an efficient, locally simulated search environment for dynamic knowledge acquisition, effectively decoupling the external world knowledge from model's sophisticated reasoning processes. It employs a unified training mechanism with meticulously designed reward functions, enabling the agent to identify problem types and adapt different answer generation strategies. Furthermore, to evaluate performance on complex open-ended tasks, we construct O$^2$-QA, a high-quality benchmark featuring 300 manually curated, multi-domain open-ended questions with associated web page caches. Extensive experiments show that O$^2$-Searcher, using only a 3B model, significantly surpasses leading LLM agents on O$^2$-QA. It also achieves SOTA results on various closed-ended QA benchmarks against similarly-sized models, while performing on par with much larger ones. 

---
# How Ensembles of Distilled Policies Improve Generalisation in Reinforcement Learning 

**Authors**: Max Weltevrede, Moritz A. Zanger, Matthijs T.J. Spaan, Wendelin Böhmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.16581)  

**Abstract**: In the zero-shot policy transfer setting in reinforcement learning, the goal is to train an agent on a fixed set of training environments so that it can generalise to similar, but unseen, testing environments. Previous work has shown that policy distillation after training can sometimes produce a policy that outperforms the original in the testing environments. However, it is not yet entirely clear why that is, or what data should be used to distil the policy. In this paper, we prove, under certain assumptions, a generalisation bound for policy distillation after training. The theory provides two practical insights: for improved generalisation, you should 1) train an ensemble of distilled policies, and 2) distil it on as much data from the training environments as possible. We empirically verify that these insights hold in more general settings, when the assumptions required for the theory no longer hold. Finally, we demonstrate that an ensemble of policies distilled on a diverse dataset can generalise significantly better than the original agent. 

---
# From Local Patterns to Global Understanding: Cross-Stock Trend Integration for Enhanced Predictive Modeling 

**Authors**: Yi Hu, Hanchi Ren, Jingjing Deng, Xianghua Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.16573)  

**Abstract**: Stock price prediction is a critical area of financial forecasting, traditionally approached by training models using the historical price data of individual stocks. While these models effectively capture single-stock patterns, they fail to leverage potential correlations among stock trends, which could improve predictive performance. Current single-stock learning methods are thus limited in their ability to provide a broader understanding of price dynamics across multiple stocks. To address this, we propose a novel method that merges local patterns into a global understanding through cross-stock pattern integration. Our strategy is inspired by Federated Learning (FL), a paradigm designed for decentralized model training. FL enables collaborative learning across distributed datasets without sharing raw data, facilitating the aggregation of global insights while preserving data privacy. In our adaptation, we train models on individual stock data and iteratively merge them to create a unified global model. This global model is subsequently fine-tuned on specific stock data to retain local relevance. The proposed strategy enables parallel training of individual stock models, facilitating efficient utilization of computational resources and reducing overall training time. We conducted extensive experiments to evaluate the proposed method, demonstrating that it outperforms benchmark models and enhances the predictive capabilities of state-of-the-art approaches. Our results highlight the efficacy of Cross-Stock Trend Integration (CSTI) in advancing stock price prediction, offering a robust alternative to traditional single-stock learning methodologies. 

---
# Finetuning-Activated Backdoors in LLMs 

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16567)  

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs. 

---
# Auto-nnU-Net: Towards Automated Medical Image Segmentation 

**Authors**: Jannis Becktepe, Leona Hennig, Steffen Oeltze-Jafra, Marius Lindauer  

**Link**: [PDF](https://arxiv.org/pdf/2505.16561)  

**Abstract**: Medical Image Segmentation (MIS) includes diverse tasks, from bone to organ segmentation, each with its own challenges in finding the best segmentation model. The state-of-the-art AutoML-related MIS-framework nnU-Net automates many aspects of model configuration but remains constrained by fixed hyperparameters and heuristic design choices. As a full-AutoML framework for MIS, we propose Auto-nnU-Net, a novel nnU-Net variant enabling hyperparameter optimization (HPO), neural architecture search (NAS), and hierarchical NAS (HNAS). Additionally, we propose Regularized PriorBand to balance model accuracy with the computational resources required for training, addressing the resource constraints often faced in real-world medical settings that limit the feasibility of extensive training procedures. We evaluate our approach across diverse MIS datasets from the well-established Medical Segmentation Decathlon, analyzing the impact of AutoML techniques on segmentation performance, computational efficiency, and model design choices. The results demonstrate that our AutoML approach substantially improves the segmentation performance of nnU-Net on 6 out of 10 datasets and is on par on the other datasets while maintaining practical resource requirements. Our code is available at this https URL. 

---
# Find the Fruit: Designing a Zero-Shot Sim2Real Deep RL Planner for Occlusion Aware Plant Manipulation 

**Authors**: Nitesh Subedi, Hsin-Jung Yang, Devesh K. Jha, Soumik Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.16547)  

**Abstract**: This paper presents an end-to-end deep reinforcement learning (RL) framework for occlusion-aware robotic manipulation in cluttered plant environments. Our approach enables a robot to interact with a deformable plant to reveal hidden objects of interest, such as fruits, using multimodal observations. We decouple the kinematic planning problem from robot control to simplify zero-shot sim2real transfer for the trained policy. Our results demonstrate that the trained policy, deployed using our framework, achieves up to 86.7% success in real-world trials across diverse initial conditions. Our findings pave the way toward autonomous, perception-driven agricultural robots that intelligently interact with complex foliage plants to "find the fruit" in challenging occluded scenarios, without the need for explicitly designed geometric and dynamic models of every plant scenario. 

---
# TextureSAM: Towards a Texture Aware Foundation Model for Segmentation 

**Authors**: Inbal Cohen, Boaz Meivar, Peihan Tu, Shai Avidan, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2505.16540)  

**Abstract**: Segment Anything Models (SAM) have achieved remarkable success in object segmentation tasks across diverse datasets. However, these models are predominantly trained on large-scale semantic segmentation datasets, which introduce a bias toward object shape rather than texture cues in the image. This limitation is critical in domains such as medical imaging, material classification, and remote sensing, where texture changes define object boundaries. In this study, we investigate SAM's bias toward semantics over textures and introduce a new texture-aware foundation model, TextureSAM, which performs superior segmentation in texture-dominant scenarios. To achieve this, we employ a novel fine-tuning approach that incorporates texture augmentation techniques, incrementally modifying training images to emphasize texture features. By leveraging a novel texture-alternation of the ADE20K dataset, we guide TextureSAM to prioritize texture-defined regions, thereby mitigating the inherent shape bias present in the original SAM model. Our extensive experiments demonstrate that TextureSAM significantly outperforms SAM-2 on both natural (+0.2 mIoU) and synthetic (+0.18 mIoU) texture-based segmentation datasets. The code and texture-augmented dataset will be publicly available. 

---
# DuFFin: A Dual-Level Fingerprinting Framework for LLMs IP Protection 

**Authors**: Yuliang Yan, Haochun Tang, Shuo Yan, Enyan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2505.16530)  

**Abstract**: Large language models (LLMs) are considered valuable Intellectual Properties (IP) for legitimate owners due to the enormous computational cost of training. It is crucial to protect the IP of LLMs from malicious stealing or unauthorized deployment. Despite existing efforts in watermarking and fingerprinting LLMs, these methods either impact the text generation process or are limited in white-box access to the suspect model, making them impractical. Hence, we propose DuFFin, a novel $\textbf{Du}$al-Level $\textbf{Fin}$gerprinting $\textbf{F}$ramework for black-box setting ownership verification. DuFFin extracts the trigger pattern and the knowledge-level fingerprints to identify the source of a suspect model. We conduct experiments on a variety of models collected from the open-source website, including four popular base models as protected LLMs and their fine-tuning, quantization, and safety alignment versions, which are released by large companies, start-ups, and individual users. Results show that our method can accurately verify the copyright of the base protected LLM on their model variants, achieving the IP-ROC metric greater than 0.95. Our code is available at this https URL. 

---
# Benchmarking and Pushing the Multi-Bias Elimination Boundary of LLMs via Causal Effect Estimation-guided Debiasing 

**Authors**: Zhouhao Sun, Zhiyuan Kan, Xiao Ding, Li Du, Yang Zhao, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16522)  

**Abstract**: Despite significant progress, recent studies have indicated that current large language models (LLMs) may still utilize bias during inference, leading to the poor generalizability of LLMs. Some benchmarks are proposed to investigate the generalizability of LLMs, with each piece of data typically containing one type of controlled bias. However, a single piece of data may contain multiple types of biases in practical applications. To bridge this gap, we propose a multi-bias benchmark where each piece of data contains five types of biases. The evaluations conducted on this benchmark reveal that the performance of existing LLMs and debiasing methods is unsatisfying, highlighting the challenge of eliminating multiple types of biases simultaneously. To overcome this challenge, we propose a causal effect estimation-guided multi-bias elimination method (CMBE). This method first estimates the causal effect of multiple types of biases simultaneously. Subsequently, we eliminate the causal effect of biases from the total causal effect exerted by both the semantic information and biases during inference. Experimental results show that CMBE can effectively eliminate multiple types of bias simultaneously to enhance the generalizability of LLMs. 

---
# Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs 

**Authors**: Giovanni Servedio, Alessandro De Bellis, Dario Di Palma, Vito Walter Anelli, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16520)  

**Abstract**: Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation. 

---
# CUB: Benchmarking Context Utilisation Techniques for Language Models 

**Authors**: Lovisa Hagström, Youna Kim, Haeun Yu, Sang-goo Lee, Richard Johansson, Hyunsoo Cho, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.16518)  

**Abstract**: Incorporating external knowledge is crucial for knowledge-intensive tasks, such as question answering and fact checking. However, language models (LMs) may ignore relevant information that contradicts outdated parametric memory or be distracted by irrelevant contexts. While many context utilisation manipulation techniques (CMTs) that encourage or suppress context utilisation have recently been proposed to alleviate these issues, few have seen systematic comparison. In this paper, we develop CUB (Context Utilisation Benchmark) to help practitioners within retrieval-augmented generation (RAG) identify the best CMT for their needs. CUB allows for rigorous testing on three distinct context types, observed to capture key challenges in realistic context utilisation scenarios. With this benchmark, we evaluate seven state-of-the-art methods, representative of the main categories of CMTs, across three diverse datasets and tasks, applied to nine LMs. Our results show that most of the existing CMTs struggle to handle the full set of types of contexts that may be encountered in real-world retrieval-augmented scenarios. Moreover, we find that many CMTs display an inflated performance on simple synthesised datasets, compared to more realistic datasets with naturally occurring samples. Altogether, our results show the need for holistic tests of CMTs and the development of CMTs that can handle multiple context types. 

---
# Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods 

**Authors**: Majid Mohammadi, Siu Lun Chau, Krikamol Muandet  

**Link**: [PDF](https://arxiv.org/pdf/2505.16516)  

**Abstract**: Kernel methods are widely used in machine learning due to their flexibility and expressive power. However, their black-box nature poses significant challenges to interpretability, limiting their adoption in high-stakes applications. Shapley value-based feature attribution techniques, such as SHAP and kernel-specific variants like RKHS-SHAP, offer a promising path toward explainability. Yet, computing exact Shapley values remains computationally intractable in general, motivating the development of various approximation schemes. In this work, we introduce PKeX-Shapley, a novel algorithm that utilizes the multiplicative structure of product kernels to enable the exact computation of Shapley values in polynomial time. We show that product-kernel models admit a functional decomposition that allows for a recursive formulation of Shapley values. This decomposition not only yields computational efficiency but also enhances interpretability in kernel-based learning. We also demonstrate how our framework can be generalized to explain kernel-based statistical discrepancies such as the Maximum Mean Discrepancy (MMD) and the Hilbert-Schmidt Independence Criterion (HSIC), thus offering new tools for interpretable statistical inference. 

---
# Beyond Face Swapping: A Diffusion-Based Digital Human Benchmark for Multimodal Deepfake Detection 

**Authors**: Jiaxin Liu, Jia Wang, Saihui Hou, Min Ren, Huijia Wu, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16512)  

**Abstract**: In recent years, the rapid development of deepfake technology has given rise to an emerging and serious threat to public security: diffusion model-based digital human generation. Unlike traditional face manipulation methods, such models can generate highly realistic videos with consistency through multimodal control signals. Their flexibility and covertness pose severe challenges to existing detection strategies. To bridge this gap, we introduce DigiFakeAV, the first large-scale multimodal digital human forgery dataset based on diffusion models. Employing five latest digital human generation methods (Sonic, Hallo, etc.) and voice cloning method, we systematically produce a dataset comprising 60,000 videos (8.4 million frames), covering multiple nationalities, skin tones, genders, and real-world scenarios, significantly enhancing data diversity and realism. User studies show that the confusion rate between forged and real videos reaches 68%, and existing state-of-the-art (SOTA) detection models exhibit large drops in AUC values on DigiFakeAV, highlighting the challenge of the dataset. To address this problem, we further propose DigiShield, a detection baseline based on spatiotemporal and cross-modal fusion. By jointly modeling the 3D spatiotemporal features of videos and the semantic-acoustic features of audio, DigiShield achieves SOTA performance on both the DigiFakeAV and DF-TIMIT datasets. Experiments show that this method effectively identifies covert artifacts through fine-grained analysis of the temporal evolution of facial features in synthetic videos. 

---
# Edge-First Language Model Inference: Models, Metrics, and Tradeoffs 

**Authors**: SiYoung Jang, Roberto Morabito  

**Link**: [PDF](https://arxiv.org/pdf/2505.16508)  

**Abstract**: The widespread adoption of Language Models (LMs) across industries is driving interest in deploying these services across the computing continuum, from the cloud to the network edge. This shift aims to reduce costs, lower latency, and improve reliability and privacy. Small Language Models (SLMs), enabled by advances in model compression, are central to this shift, offering a path to on-device inference on resource-constrained edge platforms. This work examines the interplay between edge and cloud deployments, starting from detailed benchmarking of SLM capabilities on single edge devices, and extending to distributed edge clusters. We identify scenarios where edge inference offers comparable performance with lower costs, and others where cloud fallback becomes essential due to limits in scalability or model capacity. Rather than proposing a one-size-fits-all solution, we present platform-level comparisons and design insights for building efficient, adaptive LM inference systems across heterogeneous environments. 

---
# Sparse Activation Editing for Reliable Instruction Following in Narratives 

**Authors**: Runcong Zhao, Chengyu Cao, Qinglin Zhu, Xiucheng Lv, Shun Shao, Lin Gui, Ruifeng Xu, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16505)  

**Abstract**: Complex narrative contexts often challenge language models' ability to follow instructions, and existing benchmarks fail to capture these difficulties. To address this, we propose Concise-SAE, a training-free framework that improves instruction following by identifying and editing instruction-relevant neurons using only natural language instructions, without requiring labelled data. To thoroughly evaluate our method, we introduce FreeInstruct, a diverse and realistic benchmark of 1,212 examples that highlights the challenges of instruction following in narrative-rich settings. While initially motivated by complex narratives, Concise-SAE demonstrates state-of-the-art instruction adherence across varied tasks without compromising generation quality. 

---
# Smaller, Smarter, Closer: The Edge of Collaborative Generative AI 

**Authors**: Roberto Morabito, SiYoung Jang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16499)  

**Abstract**: The rapid adoption of generative AI (GenAI), particularly Large Language Models (LLMs), has exposed critical limitations of cloud-centric deployments, including latency, cost, and privacy concerns. Meanwhile, Small Language Models (SLMs) are emerging as viable alternatives for resource-constrained edge environments, though they often lack the capabilities of their larger counterparts. This article explores the potential of collaborative inference systems that leverage both edge and cloud resources to address these challenges. By presenting distinct cooperation strategies alongside practical design principles and experimental insights, we offer actionable guidance for deploying GenAI across the computing continuum. 

---
# Human-like Semantic Navigation for Autonomous Driving using Knowledge Representation and Large Language Models 

**Authors**: Augusto Luis Ballardini, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16498)  

**Abstract**: Achieving full automation in self-driving vehicles remains a challenge, especially in dynamic urban environments where navigation requires real-time adaptability. Existing systems struggle to handle navigation plans when faced with unpredictable changes in road layouts, spontaneous detours, or missing map data, due to their heavy reliance on predefined cartographic information. In this work, we explore the use of Large Language Models to generate Answer Set Programming rules by translating informal navigation instructions into structured, logic-based reasoning. ASP provides non-monotonic reasoning, allowing autonomous vehicles to adapt to evolving scenarios without relying on predefined maps. We present an experimental evaluation in which LLMs generate ASP constraints that encode real-world urban driving logic into a formal knowledge representation. By automating the translation of informal navigation instructions into logical rules, our method improves adaptability and explainability in autonomous navigation. Results show that LLM-driven ASP rule generation supports semantic-based decision-making, offering an explainable framework for dynamic navigation planning that aligns closely with how humans communicate navigational intent. 

---
# LLaMAs Have Feelings Too: Unveiling Sentiment and Emotion Representations in LLaMA Models Through Probing 

**Authors**: Dario Di Palma, Alessandro De Bellis, Giovanni Servedio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.16491)  

**Abstract**: Large Language Models (LLMs) have rapidly become central to NLP, demonstrating their ability to adapt to various tasks through prompting techniques, including sentiment analysis. However, we still have a limited understanding of how these models capture sentiment-related information. This study probes the hidden layers of Llama models to pinpoint where sentiment features are most represented and to assess how this affects sentiment analysis.
Using probe classifiers, we analyze sentiment encoding across layers and scales, identifying the layers and pooling methods that best capture sentiment signals. Our results show that sentiment information is most concentrated in mid-layers for binary polarity tasks, with detection accuracy increasing up to 14% over prompting techniques. Additionally, we find that in decoder-only models, the last token is not consistently the most informative for sentiment encoding. Finally, this approach enables sentiment tasks to be performed with memory requirements reduced by an average of 57%.
These insights contribute to a broader understanding of sentiment in LLMs, suggesting layer-specific probing as an effective approach for sentiment tasks beyond prompting, with potential to enhance model utility and reduce memory requirements. 

---
# Teaching Large Language Models to Maintain Contextual Faithfulness via Synthetic Tasks and Reinforcement Learning 

**Authors**: Shuzheng Si, Haozhe Zhao, Cheng Gao, Yuzhuo Bai, Zhitong Wang, Bofei Gao, Kangyang Luo, Wenhao Li, Yufei Huang, Gang Chen, Fanchao Qi, Minjia Zhang, Baobao Chang, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16483)  

**Abstract**: Teaching large language models (LLMs) to be faithful in the provided context is crucial for building reliable information-seeking systems. Therefore, we propose a systematic framework, CANOE, to improve the faithfulness of LLMs in both short-form and long-form generation tasks without human annotations. Specifically, we first synthesize short-form question-answering (QA) data with four diverse tasks to construct high-quality and easily verifiable training data without human annotation. Also, we propose Dual-GRPO, a rule-based reinforcement learning method that includes three tailored rule-based rewards derived from synthesized short-form QA data, while simultaneously optimizing both short-form and long-form response generation. Notably, Dual-GRPO eliminates the need to manually label preference data to train reward models and avoids over-optimizing short-form generation when relying only on the synthesized short-form QA data. Experimental results show that CANOE greatly improves the faithfulness of LLMs across 11 different downstream tasks, even outperforming the most advanced LLMs, e.g., GPT-4o and OpenAI o1. 

---
# Conf-GNNRec: Quantifying and Calibrating the Prediction Confidence for GNN-based Recommendation Methods 

**Authors**: Meng Yan, Cai Xu, Xujing Wang, Ziyu Guan, Wei Zhao, Yuhang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.16466)  

**Abstract**: Recommender systems based on graph neural networks perform well in tasks such as rating and ranking. However, in real-world recommendation scenarios, noise such as user misuse and malicious advertisement gradually accumulates through the message propagation mechanism. Even if existing studies mitigate their effects by reducing the noise propagation weights, the severe sparsity of the recommender system still leads to the low-weighted noisy neighbors being mistaken as meaningful information, and the prediction result obtained based on the polluted nodes is not entirely trustworthy. Therefore, it is crucial to measure the confidence of the prediction results in this highly noisy framework. Furthermore, our evaluation of the existing representative GNN-based recommendation shows that it suffers from overconfidence. Based on the above considerations, we propose a new method to quantify and calibrate the prediction confidence of GNN-based recommendations (Conf-GNNRec). Specifically, we propose a rating calibration method that dynamically adjusts excessive ratings to mitigate overconfidence based on user personalization. We also design a confidence loss function to reduce the overconfidence of negative samples and effectively improve recommendation performance. Experiments on public datasets demonstrate the validity of Conf-GNNRec in prediction confidence and recommendation performance. 

---
# University of Indonesia at SemEval-2025 Task 11: Evaluating State-of-the-Art Encoders for Multi-Label Emotion Detection 

**Authors**: Ikhlasul Akmal Hanif, Eryawan Presma Yulianrifat, Jaycent Gunawan Ongris, Eduardus Tjitrahardja, Muhammad Falensi Azmi, Rahmat Bryan Naufal, Alfan Farizki Wicaksono  

**Link**: [PDF](https://arxiv.org/pdf/2505.16460)  

**Abstract**: This paper presents our approach for SemEval 2025 Task 11 Track A, focusing on multilabel emotion classification across 28 languages. We explore two main strategies: fully fine-tuning transformer models and classifier-only training, evaluating different settings such as fine-tuning strategies, model architectures, loss functions, encoders, and classifiers. Our findings suggest that training a classifier on top of prompt-based encoders such as mE5 and BGE yields significantly better results than fully fine-tuning XLMR and mBERT. Our best-performing model on the final leaderboard is an ensemble combining multiple BGE models, where CatBoost serves as the classifier, with different configurations. This ensemble achieves an average F1-macro score of 56.58 across all languages. 

---
# CMRINet: Joint Groupwise Registration and Segmentation for Cardiac Function Quantification from Cine-MRI 

**Authors**: Mohamed S. Elmahdy, Marius Staring, Patrick J. H. de Koning, Samer Alabed, Mahan Salehi, Faisal Alandejani, Michael Sharkey, Ziad Aldabbagh, Andrew J. Swift, Rob J. van der Geest  

**Link**: [PDF](https://arxiv.org/pdf/2505.16452)  

**Abstract**: Accurate and efficient quantification of cardiac function is essential for the estimation of prognosis of cardiovascular diseases (CVDs). One of the most commonly used metrics for evaluating cardiac pumping performance is left ventricular ejection fraction (LVEF). However, LVEF can be affected by factors such as inter-observer variability and varying pre-load and after-load conditions, which can reduce its reproducibility. Additionally, cardiac dysfunction may not always manifest as alterations in LVEF, such as in heart failure and cardiotoxicity diseases. An alternative measure that can provide a relatively load-independent quantitative assessment of myocardial contractility is myocardial strain and strain rate. By using LVEF in combination with myocardial strain, it is possible to obtain a thorough description of cardiac function. Automated estimation of LVEF and other volumetric measures from cine-MRI sequences can be achieved through segmentation models, while strain calculation requires the estimation of tissue displacement between sequential frames, which can be accomplished using registration models. These tasks are often performed separately, potentially limiting the assessment of cardiac function. To address this issue, in this study we propose an end-to-end deep learning (DL) model that jointly estimates groupwise (GW) registration and segmentation for cardiac cine-MRI images. The proposed anatomically-guided Deep GW network was trained and validated on a large dataset of 4-chamber view cine-MRI image series of 374 subjects. A quantitative comparison with conventional GW registration using elastix and two DL-based methods showed that the proposed model improved performance and substantially reduced computation time. 

---
# AutoMCQ -- Automatically Generate Code Comprehension Questions using GenAI 

**Authors**: Martin Goodfellow, Robbie Booth, Andrew Fagan, Alasdair Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2505.16430)  

**Abstract**: Students often do not fully understand the code they have written. This sometimes does not become evident until later in their education, which can mean it is harder to fix their incorrect knowledge or misunderstandings. In addition, being able to fully understand code is increasingly important in a world where students have access to generative artificial intelligence (GenAI) tools, such as GitHub Copilot. One effective solution is to utilise code comprehension questions, where a marker asks questions about a submission to gauge understanding, this can also have the side effect of helping to detect plagiarism. However, this approach is time consuming and can be difficult and/or expensive to scale. This paper introduces AutoMCQ, which uses GenAI for the automatic generation of multiple-choice code comprehension questions. This is integrated with the CodeRunner automated assessment platform. 

---
# Beyond Static Testbeds: An Interaction-Centric Agent Simulation Platform for Dynamic Recommender Systems 

**Authors**: Song Jin, Juntian Zhang, Yuhan Liu, Xun Zhang, Yufei Zhang, Guojun Yin, Fei Jiang, Wei Lin, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16429)  

**Abstract**: Evaluating and iterating upon recommender systems is crucial, yet traditional A/B testing is resource-intensive, and offline methods struggle with dynamic user-platform interactions. While agent-based simulation is promising, existing platforms often lack a mechanism for user actions to dynamically reshape the environment. To bridge this gap, we introduce RecInter, a novel agent-based simulation platform for recommender systems featuring a robust interaction mechanism. In RecInter platform, simulated user actions (e.g., likes, reviews, purchases) dynamically update item attributes in real-time, and introduced Merchant Agents can reply, fostering a more realistic and evolving ecosystem. High-fidelity simulation is ensured through Multidimensional User Profiling module, Advanced Agent Architecture, and LLM fine-tuned on Chain-of-Thought (CoT) enriched interaction data. Our platform achieves significantly improved simulation credibility and successfully replicates emergent phenomena like Brand Loyalty and the Matthew Effect. Experiments demonstrate that this interaction mechanism is pivotal for simulating realistic system evolution, establishing our platform as a credible testbed for recommender systems research. 

---
# $I^2G$: Generating Instructional Illustrations via Text-Conditioned Diffusion 

**Authors**: Jing Bi, Pinxin Liu, Ali Vosoughi, Jiarui Wu, Jinxi He, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16425)  

**Abstract**: The effective communication of procedural knowledge remains a significant challenge in natural language processing (NLP), as purely textual instructions often fail to convey complex physical actions and spatial relationships. We address this limitation by proposing a language-driven framework that translates procedural text into coherent visual instructions. Our approach models the linguistic structure of instructional content by decomposing it into goal statements and sequential steps, then conditioning visual generation on these linguistic elements. We introduce three key innovations: (1) a constituency parser-based text encoding mechanism that preserves semantic completeness even with lengthy instructions, (2) a pairwise discourse coherence model that maintains consistency across instruction sequences, and (3) a novel evaluation protocol specifically designed for procedural language-to-image alignment. Our experiments across three instructional datasets (HTStep, CaptainCook4D, and WikiAll) demonstrate that our method significantly outperforms existing baselines in generating visuals that accurately reflect the linguistic content and sequential nature of instructions. This work contributes to the growing body of research on grounding procedural language in visual content, with applications spanning education, task guidance, and multimodal language understanding. 

---
# Investigating Fine- and Coarse-grained Structural Correspondences Between Deep Neural Networks and Human Object Image Similarity Judgments Using Unsupervised Alignment 

**Authors**: Soh Takahashi, Masaru Sasaki, Ken Takeda, Masafumi Oizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16419)  

**Abstract**: The learning mechanisms by which humans acquire internal representations of objects are not fully understood. Deep neural networks (DNNs) have emerged as a useful tool for investigating this question, as they have internal representations similar to those of humans as a byproduct of optimizing their objective functions. While previous studies have shown that models trained with various learning paradigms - such as supervised, self-supervised, and CLIP - acquire human-like representations, it remains unclear whether their similarity to human representations is primarily at a coarse category level or extends to finer details. Here, we employ an unsupervised alignment method based on Gromov-Wasserstein Optimal Transport to compare human and model object representations at both fine-grained and coarse-grained levels. The unique feature of this method compared to conventional representational similarity analysis is that it estimates optimal fine-grained mappings between the representation of each object in human and model representations. We used this unsupervised alignment method to assess the extent to which the representation of each object in humans is correctly mapped to the corresponding representation of the same object in models. Using human similarity judgments of 1,854 objects from the THINGS dataset, we find that models trained with CLIP consistently achieve strong fine- and coarse-grained matching with human object representations. In contrast, self-supervised models showed limited matching at both fine- and coarse-grained levels, but still formed object clusters that reflected human coarse category structure. Our results offer new insights into the role of linguistic information in acquiring precise object representations and the potential of self-supervised learning to capture coarse categorical structures. 

---
# Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models 

**Authors**: Chengcheng Wang, Jianyuan Guo, Hongguang Li, Yuchuan Tian, Ying Nie, Chang Xu, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.16416)  

**Abstract**: Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs). However, when extended to large vision-language models (LVLMs), its variants introduce unintended cross-modal positional biases. Specifically, they enforce relative positional dependencies between text token indices and image tokens, causing spurious alignments. This issue arises because image tokens representing the same content but located at different spatial positions are assigned distinct positional biases, leading to inconsistent cross-modal associations. To address this, we propose Per-Token Distance (PTD) - a simple yet effective metric for quantifying the independence of positional encodings across modalities. Informed by this analysis, we introduce Circle-RoPE, a novel encoding scheme that maps image token indices onto a circular trajectory orthogonal to the linear path of text token indices, forming a cone-like structure. This configuration ensures that each text token maintains an equal distance to all image tokens, reducing artificial cross-modal biases while preserving intra-image spatial information. To further enhance performance, we propose a staggered layer strategy that applies different RoPE variants across layers. This design leverages the complementary strengths of each RoPE variant, thereby enhancing the model's overall performance. Our experimental results demonstrate that our method effectively preserves spatial information from images while reducing relative positional bias, offering a more robust and flexible positional encoding framework for LVLMs. The code is available at [this https URL](this https URL). 

---
# Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation 

**Authors**: Ruizhe Li, Chen Chen, Yuchen Hu, Yanjun Gao, Xi Wang, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.16415)  

**Abstract**: Retrieval-Augmented Generation (RAG) leverages large language models (LLMs) combined with external contexts to enhance the accuracy and reliability of generated responses. However, reliably attributing generated content to specific context segments, context attribution, remains challenging due to the computationally intensive nature of current methods, which often require extensive fine-tuning or human annotation. In this work, we introduce a novel Jensen-Shannon Divergence driven method to Attribute Response to Context (ARC-JSD), enabling efficient and accurate identification of essential context sentences without additional fine-tuning or surrogate modelling. Evaluations on a wide range of RAG benchmarks, such as TyDi QA, Hotpot QA, and Musique, using instruction-tuned LLMs in different scales demonstrate superior accuracy and significant computational efficiency improvements compared to the previous surrogate-based method. Furthermore, our mechanistic analysis reveals specific attention heads and multilayer perceptron (MLP) layers responsible for context attribution, providing valuable insights into the internal workings of RAG models. 

---
# Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning 

**Authors**: Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu Mao, Guorui Zhou, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16410)  

**Abstract**: Recently, large language models (LLMs) have shown remarkable reasoning capabilities via large-scale reinforcement learning (RL). However, leveraging the RL algorithm to empower effective multi-tool collaborative reasoning in LLMs remains an open challenge. In this paper, we introduce Tool-Star, an RL-based framework designed to empower LLMs to autonomously invoke multiple external tools during stepwise reasoning. Tool-Star integrates six types of tools and incorporates systematic designs in both data synthesis and training. To address the scarcity of tool-use data, we propose a general tool-integrated reasoning data synthesis pipeline, which combines tool-integrated prompting with hint-based sampling to automatically and scalably generate tool-use trajectories. A subsequent quality normalization and difficulty-aware classification process filters out low-quality samples and organizes the dataset from easy to hard. Furthermore, we propose a two-stage training framework to enhance multi-tool collaborative reasoning by: (1) cold-start fine-tuning, which guides LLMs to explore reasoning patterns via tool-invocation feedback; and (2) a multi-tool self-critic RL algorithm with hierarchical reward design, which reinforces reward understanding and promotes effective tool collaboration. Experimental analyses on over 10 challenging reasoning benchmarks highlight the effectiveness and efficiency of Tool-Star. The code is available at this https URL. 

---
# AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning 

**Authors**: Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2505.16400)  

**Abstract**: Despite recent progress in large-scale reinforcement learning (RL) for reasoning, the training recipe for building high-performing reasoning models remains elusive. Key implementation details of frontier models, such as DeepSeek-R1, including data curation strategies and RL training recipe, are often omitted. Moreover, recent research indicates distillation remains more effective than RL for smaller models. In this work, we demonstrate that large-scale RL can significantly enhance the reasoning capabilities of strong, small- and mid-sized models, achieving results that surpass those of state-of-the-art distillation-based models. We systematically study the RL training process through extensive ablations and propose a simple yet effective approach: first training on math-only prompts, then on code-only prompts. Notably, we find that math-only RL not only significantly enhances the performance of strong distilled models on math benchmarks (e.g., +14.6% / +17.2% on AIME 2025 for the 7B / 14B models), but also code reasoning tasks (e.g., +6.8% / +5.8% on LiveCodeBench for the 7B / 14B models). In addition, extended code-only RL iterations further improve performance on code benchmarks with minimal or no degradation in math results. We develop a robust data curation pipeline to collect challenging prompts with high-quality, verifiable answers and test cases to enable verification-based RL across both domains. Finally, we identify key experimental insights, including curriculum learning with progressively increasing response lengths and the stabilizing effect of on-policy parameter updates. We find that RL not only elicits the foundational reasoning capabilities acquired during pretraining and supervised fine-tuning (e.g., distillation), but also pushes the limits of the model's reasoning ability, enabling it to solve problems that were previously unsolvable. 

---
# Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2) 

**Authors**: Zhenjie Yang, Xiaosong Jia, Qifeng Li, Xue Yang, Maoqing Yao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16394)  

**Abstract**: Reinforcement Learning (RL) can mitigate the causal confusion and distribution shift inherent to imitation learning (IL). However, applying RL to end-to-end autonomous driving (E2E-AD) remains an open problem for its training difficulty, and IL is still the mainstream paradigm in both academia and industry. Recently Model-based Reinforcement Learning (MBRL) have demonstrated promising results in neural planning; however, these methods typically require privileged information as input rather than raw sensor data. We fill this gap by designing Raw2Drive, a dual-stream MBRL approach. Initially, we efficiently train an auxiliary privileged world model paired with a neural planner that uses privileged information as input. Subsequently, we introduce a raw sensor world model trained via our proposed Guidance Mechanism, which ensures consistency between the raw sensor world model and the privileged world model during rollouts. Finally, the raw sensor world model combines the prior knowledge embedded in the heads of the privileged world model to effectively guide the training of the raw sensor policy. Raw2Drive is so far the only RL based end-to-end method on CARLA Leaderboard 2.0, and Bench2Drive and it achieves state-of-the-art performance. 

---
# Resource for Error Analysis in Text Simplification: New Taxonomy and Test Collection 

**Authors**: Benjamin Vendeville, Liana Ermakova, Pierre De Loor  

**Link**: [PDF](https://arxiv.org/pdf/2505.16392)  

**Abstract**: The general public often encounters complex texts but does not have the time or expertise to fully understand them, leading to the spread of misinformation. Automatic Text Simplification (ATS) helps make information more accessible, but its evaluation methods have not kept up with advances in text generation, especially with Large Language Models (LLMs). In particular, recent studies have shown that current ATS metrics do not correlate with the presence of errors. Manual inspections have further revealed a variety of errors, underscoring the need for a more nuanced evaluation framework, which is currently lacking. This resource paper addresses this gap by introducing a test collection for detecting and classifying errors in simplified texts. First, we propose a taxonomy of errors, with a formal focus on information distortion. Next, we introduce a parallel dataset of automatically simplified scientific texts. This dataset has been human-annotated with labels based on our proposed taxonomy. Finally, we analyze the quality of the dataset, and we study the performance of existing models to detect and classify errors from that taxonomy. These contributions give researchers the tools to better evaluate errors in ATS, develop more reliable models, and ultimately improve the quality of automatically simplified texts. 

---
# Materials Generation in the Era of Artificial Intelligence: A Comprehensive Survey 

**Authors**: Zhixun Li, Bin Cao, Rui Jiao, Liang Wang, Ding Wang, Yang Liu, Dingshuo Chen, Jia Li, Qiang Liu, Yu Rong, Liang Wang, Tong-yi Zhang, Jeffrey Xu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16379)  

**Abstract**: Materials are the foundation of modern society, underpinning advancements in energy, electronics, healthcare, transportation, and infrastructure. The ability to discover and design new materials with tailored properties is critical to solving some of the most pressing global challenges. In recent years, the growing availability of high-quality materials data combined with rapid advances in Artificial Intelligence (AI) has opened new opportunities for accelerating materials discovery. Data-driven generative models provide a powerful tool for materials design by directly create novel materials that satisfy predefined property requirements. Despite the proliferation of related work, there remains a notable lack of up-to-date and systematic surveys in this area. To fill this gap, this paper provides a comprehensive overview of recent progress in AI-driven materials generation. We first organize various types of materials and illustrate multiple representations of crystalline materials. We then provide a detailed summary and taxonomy of current AI-driven materials generation approaches. Furthermore, we discuss the common evaluation metrics and summarize open-source codes and benchmark datasets. Finally, we conclude with potential future directions and challenges in this fast-growing field. The related sources can be found at this https URL. 

---
# VL-SAFE: Vision-Language Guided Safety-Aware Reinforcement Learning with World Models for Autonomous Driving 

**Authors**: Yansong Qu, Zilin Huang, Zihao Sheng, Jiancong Chen, Sikai Chen, Samuel Labi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16377)  

**Abstract**: Reinforcement learning (RL)-based autonomous driving policy learning faces critical limitations such as low sample efficiency and poor generalization; its reliance on online interactions and trial-and-error learning is especially unacceptable in safety-critical scenarios. Existing methods including safe RL often fail to capture the true semantic meaning of "safety" in complex driving contexts, leading to either overly conservative driving behavior or constraint violations. To address these challenges, we propose VL-SAFE, a world model-based safe RL framework with Vision-Language model (VLM)-as-safety-guidance paradigm, designed for offline safe policy learning. Specifically, we construct offline datasets containing data collected by expert agents and labeled with safety scores derived from VLMs. A world model is trained to generate imagined rollouts together with safety estimations, allowing the agent to perform safe planning without interacting with the real environment. Based on these imagined trajectories and safety evaluations, actor-critic learning is conducted under VLM-based safety guidance to optimize the driving policy more safely and efficiently. Extensive evaluations demonstrate that VL-SAFE achieves superior sample efficiency, generalization, safety, and overall performance compared to existing baselines. To the best of our knowledge, this is the first work that introduces a VLM-guided world model-based approach for safe autonomous driving. The demo video and code can be accessed at: this https URL 

---
# DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos 

**Authors**: Zijia Lu, A S M Iftekhar, Gaurav Mittal, Tianjian Meng, Xiawei Wang, Cheng Zhao, Rohith Kukkala, Ehsan Elhamifar, Mei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16376)  

**Abstract**: Long Video Temporal Grounding (LVTG) aims at identifying specific moments within lengthy videos based on user-provided text queries for effective content retrieval. The approach taken by existing methods of dividing video into clips and processing each clip via a full-scale expert encoder is challenging to scale due to prohibitive computational costs of processing a large number of clips in long videos. To address this issue, we introduce DeCafNet, an approach employing ``delegate-and-conquer'' strategy to achieve computation efficiency without sacrificing grounding performance. DeCafNet introduces a sidekick encoder that performs dense feature extraction over all video clips in a resource-efficient manner, while generating a saliency map to identify the most relevant clips for full processing by the expert encoder. To effectively leverage features from sidekick and expert encoders that exist at different temporal resolutions, we introduce DeCaf-Grounder, which unifies and refines them via query-aware temporal aggregation and multi-scale temporal refinement for accurate grounding. Experiments on two LTVG benchmark datasets demonstrate that DeCafNet reduces computation by up to 47\% while still outperforming existing methods, establishing a new state-of-the-art for LTVG in terms of both efficiency and performance. Our code is available at this https URL. 

---
# Temporal and Spatial Feature Fusion Framework for Dynamic Micro Expression Recognition 

**Authors**: Feng Liu, Bingyu Nan, Xuezhong Qian, Xiaolan Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16372)  

**Abstract**: When emotions are repressed, an individual's true feelings may be revealed through micro-expressions. Consequently, micro-expressions are regarded as a genuine source of insight into an individual's authentic emotions. However, the transient and highly localised nature of micro-expressions poses a significant challenge to their accurate recognition, with the accuracy rate of micro-expression recognition being as low as 50%, even for professionals. In order to address these challenges, it is necessary to explore the field of dynamic micro expression recognition (DMER) using multimodal fusion techniques, with special attention to the diverse fusion of temporal and spatial modal features. In this paper, we propose a novel Temporal and Spatial feature Fusion framework for DMER (TSFmicro). This framework integrates a Retention Network (RetNet) and a transformer-based DMER network, with the objective of efficient micro-expression recognition through the capture and fusion of temporal and spatial relations. Meanwhile, we propose a novel parallel time-space fusion method from the perspective of modal fusion, which fuses spatio-temporal information in high-dimensional feature space, resulting in complementary "where-how" relationships at the semantic level and providing richer semantic information for the model. The experimental results demonstrate the superior performance of the TSFmicro method in comparison to other contemporary state-of-the-art methods. This is evidenced by its effectiveness on three well-recognised micro-expression datasets. 

---
# SATURN: SAT-based Reinforcement Learning to Unleash Language Model Reasoning 

**Authors**: Huanyu Liu, Jia Li, Hao Zhu, Kechi Zhang, Yihong Dong, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16368)  

**Abstract**: How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard.
To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLM reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions.
We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research. 

---
# A collaborative constrained graph diffusion model for the generation of realistic synthetic molecules 

**Authors**: Manuel Ruiz-Botella, Marta Sales-Pardo, Roger Guimerà  

**Link**: [PDF](https://arxiv.org/pdf/2505.16365)  

**Abstract**: Developing new molecular compounds is crucial to address pressing challenges, from health to environmental sustainability. However, exploring the molecular space to discover new molecules is difficult due to the vastness of the space. Here we introduce CoCoGraph, a collaborative and constrained graph diffusion model capable of generating molecules that are guaranteed to be chemically valid. Thanks to the constraints built into the model and to the collaborative mechanism, CoCoGraph outperforms state-of-the-art approaches on standard benchmarks while requiring up to an order of magnitude fewer parameters. Analysis of 36 chemical properties also demonstrates that CoCoGraph generates molecules with distributions more closely matching real molecules than current models. Leveraging the model's efficiency, we created a database of 8.2M million synthetically generated molecules and conducted a Turing-like test with organic chemistry experts to further assess the plausibility of the generated molecules, and potential biases and limitations of CoCoGraph. 

---
# AdamS: Momentum Itself Can Be A Normalizer for LLM Pretraining and Post-training 

**Authors**: Huishuai Zhang, Bohan Wang, Luoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16363)  

**Abstract**: We introduce AdamS, a simple yet effective alternative to Adam for large language model (LLM) pretraining and post-training. By leveraging a novel denominator, i.e., the root of weighted sum of squares of the momentum and the current gradient, AdamS eliminates the need for second-moment estimates. Hence, AdamS is efficient, matching the memory and compute footprint of SGD with momentum while delivering superior optimization performance. Moreover, AdamS is easy to adopt: it can directly inherit hyperparameters of AdamW, and is entirely model-agnostic, integrating seamlessly into existing pipelines without modifications to optimizer APIs or architectures. The motivation behind AdamS stems from the observed $(L_0, L_1)$ smoothness properties in transformer objectives, where local smoothness is governed by gradient magnitudes that can be further approximated by momentum magnitudes. We establish rigorous theoretical convergence guarantees and provide practical guidelines for hyperparameter selection. Empirically, AdamS demonstrates strong performance in various tasks, including pre-training runs on GPT-2 and Llama2 (up to 13B parameters) and reinforcement learning in post-training regimes. With its efficiency, simplicity, and theoretical grounding, AdamS stands as a compelling alternative to existing optimizers. 

---
# Neuromorphic-based metaheuristics: A new generation of low power, low latency and small footprint optimization algorithms 

**Authors**: El-ghazali Talbi  

**Link**: [PDF](https://arxiv.org/pdf/2505.16362)  

**Abstract**: Neuromorphic computing (NC) introduces a novel algorithmic paradigm representing a major shift from traditional digital computing of Von Neumann architectures. NC emulates or simulates the neural dynamics of brains in the form of Spiking Neural Networks (SNNs). Much of the research in NC has concentrated on machine learning applications and neuroscience simulations. This paper investigates the modelling and implementation of optimization algorithms and particularly metaheuristics using the NC paradigm as an alternative to Von Neumann architectures, leading to breakthroughs in solving optimization problems.
Neuromorphic-based metaheuristics (Nheuristics) are supposed to be characterized by low power, low latency and small footprint. Since NC systems are fundamentally different from conventional Von Neumann computers, several challenges are posed to the design and implementation of Nheuristics. A guideline based on a classification and critical analysis is conducted on the different families of metaheuristics and optimization problems they address. We also discuss future directions that need to be addressed to expand both the development and application of Nheuristics. 

---
# Dysfluent WFST: A Framework for Zero-Shot Speech Dysfluency Transcription and Detection 

**Authors**: Chenxu Guo, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Shuhe Li, Zongli Ye, Hwi Joo Park, Anaisha Das, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.16351)  

**Abstract**: Automatic detection of speech dysfluency aids speech-language pathologists in efficient transcription of disordered speech, enhancing diagnostics and treatment planning. Traditional methods, often limited to classification, provide insufficient clinical insight, and text-independent models misclassify dysfluency, especially in context-dependent cases. This work introduces Dysfluent-WFST, a zero-shot decoder that simultaneously transcribes phonemes and detects dysfluency. Unlike previous models, Dysfluent-WFST operates with upstream encoders like WavLM and requires no additional training. It achieves state-of-the-art performance in both phonetic error rate and dysfluency detection on simulated and real speech data. Our approach is lightweight, interpretable, and effective, demonstrating that explicit modeling of pronunciation behavior in decoding, rather than complex architectures, is key to improving dysfluency processing systems. 

---
# FPQVAR: Floating Point Quantization for Visual Autoregressive Model with FPGA Hardware Co-design 

**Authors**: Renjie Wei, Songqiang Xu, Qingyu Guo, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16335)  

**Abstract**: Visual autoregressive (VAR) modeling has marked a paradigm shift in image generation from next-token prediction to next-scale prediction. VAR predicts a set of tokens at each step from coarse to fine scale, leading to better image quality and faster inference speed compared to existing diffusion models. However, the large parameter size and computation cost hinder its deployment on edge devices. To reduce the memory and computation cost, we propose FPQVAR, an efficient post-training floating-point (FP) quantization framework for VAR featuring algorithm and hardware co-design. At the algorithm level, we first identify the challenges of quantizing VAR. To address them, we propose Dual Format Quantization for the highly imbalanced input activation. We further propose Group-wise Hadamard Transformation and GHT-Aware Learnable Transformation to address the time-varying outlier channels. At the hardware level, we design the first low-bit FP quantizer and multiplier with lookup tables on FPGA and propose the first FPGA-based VAR accelerator featuring low-bit FP computation and an elaborate two-level pipeline. Extensive experiments show that compared to the state-of-the-art quantization method, our proposed FPQVAR significantly improves Fréchet Inception Distance (FID) from 10.83 to 3.58, Inception Score (IS) from 175.9 to 241.5 under 4-bit quantization. FPQVAR also significantly improves the performance of 6-bit quantized VAR, bringing it on par with the FP16 model. Our accelerator on AMD-Xilinx VCK190 FPGA achieves a throughput of 1.1 image/s, which is 3.1x higher than the integer-based accelerator. It also demonstrates 3.6x and 2.8x higher energy efficiency compared to the integer-based accelerator and GPU baseline, respectively. 

---
# Is Quantum Optimization Ready? An Effort Towards Neural Network Compression using Adiabatic Quantum Computing 

**Authors**: Zhehui Wanga, Benjamin Chen Ming Choonga, Tian Huang, Daniel Gerlinghoffa, Rick Siow Mong Goh, Cheng Liu, Tao Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16332)  

**Abstract**: Quantum optimization is the most mature quantum computing technology to date, providing a promising approach towards efficiently solving complex combinatorial problems. Methods such as adiabatic quantum computing (AQC) have been employed in recent years on important optimization problems across various domains. In deep learning, deep neural networks (DNN) have reached immense sizes to support new predictive capabilities. Optimization of large-scale models is critical for sustainable deployment, but becomes increasingly challenging with ever-growing model sizes and complexity. While quantum optimization is suitable for solving complex problems, its application to DNN optimization is not straightforward, requiring thorough reformulation for compatibility with commercially available quantum devices. In this work, we explore the potential of adopting AQC for fine-grained pruning-quantization of convolutional neural networks. We rework established heuristics to formulate model compression as a quadratic unconstrained binary optimization (QUBO) problem, and assess the solution space offered by commercial quantum annealing devices. Through our exploratory efforts of reformulation, we demonstrate that AQC can achieve effective compression of practical DNN models. Experiments demonstrate that adiabatic quantum computing (AQC) not only outperforms classical algorithms like genetic algorithms and reinforcement learning in terms of time efficiency but also excels at identifying global optima. 

---
# SC4ANM: Identifying Optimal Section Combinations for Automated Novelty Prediction in Academic Papers 

**Authors**: Wenqing Wu, Chengzhi Zhang, Tong Bao, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16330)  

**Abstract**: Novelty is a core component of academic papers, and there are multiple perspectives on the assessment of novelty. Existing methods often focus on word or entity combinations, which provide limited insights. The content related to a paper's novelty is typically distributed across different core sections, e.g., Introduction, Methodology and Results. Therefore, exploring the optimal combination of sections for evaluating the novelty of a paper is important for advancing automated novelty assessment. In this paper, we utilize different combinations of sections from academic papers as inputs to drive language models to predict novelty scores. We then analyze the results to determine the optimal section combinations for novelty score prediction. We first employ natural language processing techniques to identify the sectional structure of academic papers, categorizing them into introduction, methods, results, and discussion (IMRaD). Subsequently, we used different combinations of these sections (e.g., introduction and methods) as inputs for pretrained language models (PLMs) and large language models (LLMs), employing novelty scores provided by human expert reviewers as ground truth labels to obtain prediction results. The results indicate that using introduction, results and discussion is most appropriate for assessing the novelty of a paper, while the use of the entire text does not yield significant results. Furthermore, based on the results of the PLMs and LLMs, the introduction and results appear to be the most important section for the task of novelty score prediction. The code and dataset for this paper can be accessed at this https URL. 

---
# CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation 

**Authors**: Yuyang Jiang, Chacha Chen, Shengyuan Wang, Feng Li, Zecong Tang, Benjamin M. Mervak, Lydia Chelala, Christopher M Straus, Reve Chahine, Samuel G. Armato III, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16325)  

**Abstract**: Existing metrics often lack the granularity and interpretability to capture nuanced clinical differences between candidate and ground-truth radiology reports, resulting in suboptimal evaluation. We introduce a Clinically-grounded tabular framework with Expert-curated labels and Attribute-level comparison for Radiology report evaluation (CLEAR). CLEAR not only examines whether a report can accurately identify the presence or absence of medical conditions, but also assesses whether it can precisely describe each positively identified condition across five key attributes: first occurrence, change, severity, descriptive location, and recommendation. Compared to prior works, CLEAR's multi-dimensional, attribute-level outputs enable a more comprehensive and clinically interpretable evaluation of report quality. Additionally, to measure the clinical alignment of CLEAR, we collaborate with five board-certified radiologists to develop CLEAR-Bench, a dataset of 100 chest X-ray reports from MIMIC-CXR, annotated across 6 curated attributes and 13 CheXpert conditions. Our experiments show that CLEAR achieves high accuracy in extracting clinical attributes and provides automated metrics that are strongly aligned with clinical judgment. 

---
# AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners 

**Authors**: Woosung Koh, Wonbeen Oh, Jaein Jang, MinHyung Lee, Hyeongjin Kim, Ah Yeon Kim, Joonkee Kim, Junghyun Lee, Taehyeon Kim, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16322)  

**Abstract**: Self-Taught Reasoners (STaR), synonymously known as Rejection sampling Fine-Tuning (RFT), is an integral part of the training pipeline of self-improving reasoning Language Models (LMs). The self-improving mechanism often employs random observation (data) sampling. However, this results in trained observation imbalance; inefficiently over-training on solved examples while under-training on challenging ones. In response, we introduce Adaptive STaR (AdaSTaR), a novel algorithm that rectifies this by integrating two adaptive sampling principles: (1) Adaptive Sampling for Diversity: promoting balanced training across observations, and (2) Adaptive Sampling for Curriculum: dynamically adjusting data difficulty to match the model's evolving strength. Across six benchmarks, AdaSTaR achieves best test accuracy in all instances (6/6) and reduces training FLOPs by an average of 58.6% against an extensive list of baselines. These improvements in performance and efficiency generalize to different pre-trained LMs and larger models, paving the way for more efficient and effective self-improving LMs. 

---
# NTIRE 2025 challenge on Text to Image Generation Model Quality Assessment 

**Authors**: Shuhao Han, Haotian Fan, Fangyuan Kong, Wenjie Liao, Chunle Guo, Chongyi Li, Radu Timofte, Liang Li, Tao Li, Junhui Cui, Yunqiu Wang, Yang Tai, Jingwei Sun, Jianhui Sun, Xinli Yue, Tianyi Wang, Huan Hou, Junda Lu, Xinyang Huang, Zitang Zhou, Zijian Zhang, Xuhui Zheng, Xuecheng Wu, Chong Peng, Xuezhi Cao, Trong-Hieu Nguyen-Mau, Minh-Hoang Le, Minh-Khoa Le-Phan, Duy-Nam Ly, Hai-Dang Nguyen, Minh-Triet Tran, Yukang Lin, Yan Hong, Chuanbiao Song, Siyuan Li, Jun Lan, Zhichao Zhang, Xinyue Li, Wei Sun, Zicheng Zhang, Yunhao Li, Xiaohong Liu, Guangtao Zhai, Zitong Xu, Huiyu Duan, Jiarui Wang, Guangji Ma, Liu Yang, Lu Liu, Qiang Hu, Xiongkuo Min, Zichuan Wang, Zhenchen Tang, Bo Peng, Jing Dong, Fengbin Guan, Zihao Yu, Yiting Lu, Wei Luo, Xin Li, Minhao Lin, Haofeng Chen, Xuanxuan He, Kele Xu, Qisheng Xu, Zijian Gao, Tianjiao Wan, Bo-Cheng Qiu, Chih-Chung Hsu, Chia-ming Lee, Yu-Fan Lin, Bo Yu, Zehao Wang, Da Mu, Mingxiu Chen, Junkang Fang, Huamei Sun, Wending Zhao, Zhiyu Wang, Wang Liu, Weikang Yu, Puhong Duan, Bin Sun, Xudong Kang, Shutao Li, Shuai He, Lingzhi Fu, Heng Cong, Rongyu Zhang, Jiarong He, Zhishan Qiao, Yongqing Huang, Zewen Chen, Zhe Pang, Juan Wang, Jian Guo, Zhizhuo Shao, Ziyu Feng, Bing Li, Weiming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.16314)  

**Abstract**: This paper reports on the NTIRE 2025 challenge on Text to Image (T2I) generation model quality assessment, which will be held in conjunction with the New Trends in Image Restoration and Enhancement Workshop (NTIRE) at CVPR 2025. The aim of this challenge is to address the fine-grained quality assessment of text-to-image generation models. This challenge evaluates text-to-image models from two aspects: image-text alignment and image structural distortion detection, and is divided into the alignment track and the structural track. The alignment track uses the EvalMuse-40K, which contains around 40K AI-Generated Images (AIGIs) generated by 20 popular generative models. The alignment track has a total of 371 registered participants. A total of 1,883 submissions are received in the development phase, and 507 submissions are received in the test phase. Finally, 12 participating teams submitted their models and fact sheets. The structure track uses the EvalMuse-Structure, which contains 10,000 AI-Generated Images (AIGIs) with corresponding structural distortion mask. A total of 211 participants have registered in the structure track. A total of 1155 submissions are received in the development phase, and 487 submissions are received in the test phase. Finally, 8 participating teams submitted their models and fact sheets. Almost all methods have achieved better results than baseline methods, and the winning methods in both tracks have demonstrated superior prediction performance on T2I model quality assessment. 

---
# PMPO: Probabilistic Metric Prompt Optimization for Small and Large Language Models 

**Authors**: Chenzhuo Zhao, Ziqian Liu, Xingda Wang, Junting Lu, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16307)  

**Abstract**: Prompt optimization offers a practical and broadly applicable alternative to fine-tuning for improving large language model (LLM) performance. However, existing methods often rely on costly output generation, self-critiquing abilities, or human-annotated preferences, which limit their scalability, especially for smaller or non-instruction-tuned models. We introduce PMPO (Probabilistic Metric Prompt Optimization), a unified framework that refines prompts using token-level cross-entropy loss as a direct, lightweight evaluation signal. PMPO identifies low-quality prompt segments by masking and measuring their impact on loss, then rewrites and selects improved variants by minimizing loss over positive and negative examples. Unlike prior methods, it requires no output sampling or human evaluation during optimization, relying only on forward passes and log-likelihoods. PMPO supports both supervised and preference-based tasks through a closely aligned loss-based evaluation strategy. Experiments show that PMPO consistently outperforms prior methods across model sizes and tasks: it achieves the highest average accuracy on BBH, performs strongly on GSM8K and AQUA-RAT, and improves AlpacaEval 2.0 win rates by over 19 points. These results highlight PMPO's effectiveness, efficiency, and broad applicability. 

---
# Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models 

**Authors**: Yizhi Zhou, Haina Zhu, Hangting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16306)  

**Abstract**: Recently, pre-trained models for music information retrieval based on self-supervised learning (SSL) are becoming popular, showing success in various downstream tasks. However, there is limited research on the specific meanings of the encoded information and their applicability. Exploring these aspects can help us better understand their capabilities and limitations, leading to more effective use in downstream tasks.
In this study, we analyze the advanced music representation model MusicFM and the newly emerged SSL model MuQ. We focus on three main aspects: (i) validating the advantages of SSL models across multiple downstream tasks, (ii) exploring the specialization of layer-wise information for different tasks, and (iii) comparing performance differences when selecting specific layers. Through this analysis, we reveal insights into the structure and potential applications of SSL models in music information retrieval. 

---
# Artificial Intelligence for Direct Prediction of Molecular Dynamics Across Chemical Space 

**Authors**: Fuchun Ge, Pavlo O. Dral  

**Link**: [PDF](https://arxiv.org/pdf/2505.16301)  

**Abstract**: Molecular dynamics (MD) is a powerful tool for exploring the behavior of atomistic systems, but its reliance on sequential numerical integration limits simulation efficiency. We present MDtrajNet-1, a foundational AI model that directly generates MD trajectories across chemical space, bypassing force calculations and integration. This approach accelerates simulations by up to two orders of magnitude compared to traditional MD, even those enhanced by machine-learning interatomic potentials. MDtrajNet-1 combines equivariant neural networks with a Transformer-based architecture to achieve strong accuracy and transferability in predicting long-time trajectories for both known and unseen systems. Remarkably, the errors of the trajectories generated by MDtrajNet-1 for various molecular systems are close to those of the conventional ab initio MD. The model's flexible design supports diverse application scenarios, including different statistical ensembles, boundary conditions, and interaction types. By overcoming the intrinsic speed barrier of conventional MD, MDtrajNet-1 opens new frontiers in efficient and scalable atomistic simulations. 

---
# Multimodal Generative AI for Story Point Estimation in Software Development 

**Authors**: Mohammad Rubyet Islam, Peter Sandborn  

**Link**: [PDF](https://arxiv.org/pdf/2505.16290)  

**Abstract**: This research explores the application of Multimodal Generative AI to enhance story point estimation in Agile software development. By integrating text, image, and categorical data using advanced models like BERT, CNN, and XGBoost, our approach surpasses the limitations of traditional single-modal estimation methods. The results demonstrate strong accuracy for simpler story points, while also highlighting challenges in more complex categories due to data imbalance. This study further explores the impact of categorical data, particularly severity, on the estimation process, emphasizing its influence on model performance. Our findings emphasize the transformative potential of multimodal data integration in refining AI-driven project management, paving the way for more precise, adaptable, and domain-specific AI capabilities. Additionally, this work outlines future directions for addressing data variability and enhancing the robustness of AI in Agile methodologies. 

---
# DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving 

**Authors**: Zhenjie Yang, Yilin Chai, Xiaosong Jia, Qifeng Li, Yuqian Shao, Xuekai Zhu, Haisheng Su, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.16278)  

**Abstract**: End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE to Drive-$\pi_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$\pi_0$. 

---
# Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning 

**Authors**: Jiaru Zou, Yikun Ban, Zihao Li, Yunzhe Qi, Ruizhong Qiu, Ling Yang, Jingrui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16270)  

**Abstract**: Large language models are typically adapted to downstream tasks through supervised fine-tuning on domain-specific data. While standard fine-tuning focuses on minimizing generation loss to optimize model parameters, we take a deeper step by retaining and leveraging the model's own learning signals, analogous to how human learners reflect on past mistakes to improve future performance. We first introduce the concept of Mistake Log to systematically track the model's learning behavior and recurring errors throughout fine-tuning. Treating the original transformer-based model as the Pilot, we correspondingly design a Copilot model to refine the Pilot's inference performance via logits rectification. We name the overall Pilot-Copilot framework the Transformer Copilot, which introduces (i) a novel Copilot model design, (ii) a joint training paradigm where the Copilot continuously learns from the evolving Mistake Log alongside the Pilot, and (iii) a fused inference paradigm where the Copilot rectifies the Pilot's logits for enhanced generation. We provide both theoretical and empirical analyses on our new learning framework. Experiments on 12 benchmarks spanning commonsense, arithmetic, and recommendation tasks demonstrate that Transformer Copilot consistently improves performance by up to 34.5%, while introducing marginal computational overhead to Pilot models and exhibiting strong scalability and transferability. 

---
# Dialogue in Resonance: An Interactive Music Piece for Piano and Real-Time Automatic Transcription System 

**Authors**: Hayeon Bang, Taegyun Kwon, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2505.16259)  

**Abstract**: This paper presents <Dialogue in Resonance>, an interactive music piece for a human pianist and a computer-controlled piano that integrates real-time automatic music transcription into a score-driven framework. Unlike previous approaches that primarily focus on improvisation-based interactions, our work establishes a balanced framework that combines composed structure with dynamic interaction. Through real-time automatic transcription as its core mechanism, the computer interprets and responds to the human performer's input in real time, creating a musical dialogue that balances compositional intent with live interaction while incorporating elements of unpredictability. In this paper, we present the development process from composition to premiere performance, including technical implementation, rehearsal process, and performance considerations. 

---
# IRONIC: Coherence-Aware Reasoning Chains for Multi-Modal Sarcasm Detection 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.16258)  

**Abstract**: Interpreting figurative language such as sarcasm across multi-modal inputs presents unique challenges, often requiring task-specific fine-tuning and extensive reasoning steps. However, current Chain-of-Thought approaches do not efficiently leverage the same cognitive processes that enable humans to identify sarcasm. We present IRONIC, an in-context learning framework that leverages Multi-modal Coherence Relations to analyze referential, analogical and pragmatic image-text linkages. Our experiments show that IRONIC achieves state-of-the-art performance on zero-shot Multi-modal Sarcasm Detection across different baselines. This demonstrates the need for incorporating linguistic and cognitive insights into the design of multi-modal reasoning strategies. Our code is available at: this https URL 

---
# DualComp: End-to-End Learning of a Unified Dual-Modality Lossless Compressor 

**Authors**: Yan Zhao, Zhengxue Cheng, Junxuan Zhang, Qunshan Gu, Qi Wang, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.16256)  

**Abstract**: Most learning-based lossless compressors are designed for a single modality, requiring separate models for multi-modal data and lacking flexibility. However, different modalities vary significantly in format and statistical properties, making it ineffective to use compressors that lack modality-specific adaptations. While multi-modal large language models (MLLMs) offer a potential solution for modality-unified compression, their excessive complexity hinders practical deployment. To address these challenges, we focus on the two most common modalities, image and text, and propose DualComp, the first unified and lightweight learning-based dual-modality lossless compressor. Built on a lightweight backbone, DualComp incorporates three key structural enhancements to handle modality heterogeneity: modality-unified tokenization, modality-switching contextual learning, and modality-routing mixture-of-experts. A reparameterization training strategy is also used to boost compression performance. DualComp integrates both modality-specific and shared parameters for efficient parameter utilization, enabling near real-time inference (200KB/s) on desktop CPUs. With much fewer parameters, DualComp achieves compression performance on par with the SOTA LLM-based methods for both text and image datasets. Its simplified single-modality variant surpasses the previous best image compressor on the Kodak dataset by about 9% using just 1.2% of the model size. 

---
# Manipulating Elasto-Plastic Objects With 3D Occupancy and Learning-Based Predictive Control 

**Authors**: Zhen Zhang, Xiangyu Chu, Yunxi Tang, Lulu Zhao, Jing Huang, Zhongliang Jiang, K. W. Samuel Au  

**Link**: [PDF](https://arxiv.org/pdf/2505.16249)  

**Abstract**: Manipulating elasto-plastic objects remains a significant challenge due to severe self-occlusion, difficulties of representation, and complicated dynamics. This work proposes a novel framework for elasto-plastic object manipulation with a quasi-static assumption for motions, leveraging 3D occupancy to represent such objects, a learned dynamics model trained with 3D occupancy, and a learning-based predictive control algorithm to address these challenges effectively. We build a novel data collection platform to collect full spatial information and propose a pipeline for generating a 3D occupancy dataset. To infer the 3D occupancy during manipulation, an occupancy prediction network is trained with multiple RGB images supervised by the generated dataset. We design a deep neural network empowered by a 3D convolution neural network (CNN) and a graph neural network (GNN) to predict the complex deformation with the inferred 3D occupancy results. A learning-based predictive control algorithm is introduced to plan the robot actions, incorporating a novel shape-based action initialization module specifically designed to improve the planner efficiency. The proposed framework in this paper can successfully shape the elasto-plastic objects into a given goal shape and has been verified in various experiments both in simulation and the real world. 

---
# LIFEBench: Evaluating Length Instruction Following in Large Language Models 

**Authors**: Wei Zhang, Zhenhong Zhou, Junfeng Fang, Rongwu Xu, Kun Wang, Yuanhe Zhang, Rui Wang, Ge Zhang, Xinfeng Li, Li Sun, Lingjuan Lyu, Yang Liu, Sen Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.16234)  

**Abstract**: While large language models (LLMs) can solve PhD-level reasoning problems over long context inputs, they still struggle with a seemingly simpler task: following explicit length instructions-e.g., write a 10,000-word novel. Additionally, models often generate far too short outputs, terminate prematurely, or even refuse the request. Existing benchmarks focus primarily on evaluating generations quality, but often overlook whether the generations meet length constraints. To this end, we introduce Length Instruction Following Evaluation Benchmark (LIFEBench) to comprehensively evaluate LLMs' ability to follow length instructions across diverse tasks and a wide range of specified lengths. LIFEBench consists of 10,800 instances across 4 task categories in both English and Chinese, covering length constraints ranging from 16 to 8192 words. We evaluate 26 widely-used LLMs and find that most models reasonably follow short-length instructions but deteriorate sharply beyond a certain threshold. Surprisingly, almost all models fail to reach the vendor-claimed maximum output lengths in practice, as further confirmed by our evaluations extending up to 32K words. Even long-context LLMs, despite their extended input-output windows, counterintuitively fail to improve length-instructions following. Notably, Reasoning LLMs outperform even specialized long-text generation models, achieving state-of-the-art length following. Overall, LIFEBench uncovers fundamental limitations in current LLMs' length instructions following ability, offering critical insights for future progress. 

---
# Explain Less, Understand More: Jargon Detection via Personalized Parameter-Efficient Fine-tuning 

**Authors**: Bohao Wu, Qingyun Wang, Yue Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.16227)  

**Abstract**: Personalizing jargon detection and explanation is essential for making technical documents accessible to readers with diverse disciplinary backgrounds. However, tailoring models to individual users typically requires substantial annotation efforts and computational resources due to user-specific finetuning. To address this, we present a systematic study of personalized jargon detection, focusing on methods that are both efficient and scalable for real-world deployment. We explore two personalization strategies: (1) lightweight fine-tuning using Low-Rank Adaptation (LoRA) on open-source models, and (2) personalized prompting, which tailors model behavior at inference time without retaining. To reflect realistic constraints, we also investigate hybrid approaches that combine limited annotated data with unsupervised user background signals. Our personalized LoRA model outperforms GPT-4 by 21.4% in F1 score and exceeds the best performing oracle baseline by 8.3%. Remarkably, our method achieves comparable performance using only 10% of the annotated training data, demonstrating its practicality for resource-constrained settings. Our study offers the first work to systematically explore efficient, low-resource personalization of jargon detection using open-source language models, offering a practical path toward scalable, user-adaptive NLP system. 

---
# AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models 

**Authors**: Kai Li, Can Shen, Yile Liu, Jirui Han, Kelong Zheng, Xuechao Zou, Zhe Wang, Xingjian Du, Shun Zhang, Hanjun Luo, Yingbin Jin, Xinxin Xing, Ziyang Ma, Yue Liu, Xiaojun Jia, Yifan Zhang, Junfeng Fang, Kun Wang, Yibo Yan, Haoyang Li, Yiming Li, Xiaobin Zhuang, Yang Liu, Haibo Hu, Zhuo Chen, Zhizheng Wu, Xiaolin Hu, Eng-Siong Chng, XiaoFeng Wang, Wenyuan Xu, Wei Dong, Xinfeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16211)  

**Abstract**: The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at this https URL. 

---
# NQKV: A KV Cache Quantization Scheme Based on Normal Distribution Characteristics 

**Authors**: Zhihang Cai, Xingjun Zhang, Zhendong Tan, Zheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16210)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency across a wide range of tasks. However, LLMs often require larger batch sizes to enhance throughput or longer context lengths to meet task demands, which significantly increases the memory resource consumption of the Key-Value (KV) cache during inference, becoming a major bottleneck in LLM deployment. To address this issue, quantization is a common and straightforward approach. Currently, quantization methods for activations are limited to 8-bit, and quantization to even lower bits can lead to substantial accuracy drops. To further save space by quantizing the KV cache to even lower bits, we analyzed the element distribution of the KV cache and designed the NQKV algorithm. Since the elements within each block of the KV cache follow a normal distribution, NQKV employs per-block quantile quantization to achieve information-theoretically optimal quantization error. Without significantly compromising model output quality, NQKV enables the OPT model to perform inference with an 2x larger batch size or a 4x longer context length, and it improves throughput by 9.3x compared to when the KV cache is not used. 

---
# Using Echo-State Networks to Reproduce Rare Events in Chaotic Systems 

**Authors**: Anton Erofeev, Balasubramanya T. Nadiga, Ilya Timofeyev  

**Link**: [PDF](https://arxiv.org/pdf/2505.16208)  

**Abstract**: We apply the Echo-State Networks to predict the time series and statistical properties of the competitive Lotka-Volterra model in the chaotic regime. In particular, we demonstrate that Echo-State Networks successfully learn the chaotic attractor of the competitive Lotka-Volterra model and reproduce histograms of dependent variables, including tails and rare events. We use the Generalized Extreme Value distribution to quantify the tail behavior. 

---
# SEM: Enhancing Spatial Understanding for Robust Robot Manipulation 

**Authors**: Xuewu Lin, Tianwei Lin, Lichao Huang, Hongyu Xie, Yiwei Jin, Keyu Li, Zhizhong Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.16196)  

**Abstract**: A key challenge in robot manipulation lies in developing policy models with strong spatial understanding, the ability to reason about 3D geometry, object relations, and robot embodiment. Existing methods often fall short: 3D point cloud models lack semantic abstraction, while 2D image encoders struggle with spatial reasoning. To address this, we propose SEM (Spatial Enhanced Manipulation model), a novel diffusion-based policy framework that explicitly enhances spatial understanding from two complementary perspectives. A spatial enhancer augments visual representations with 3D geometric context, while a robot state encoder captures embodiment-aware structure through graphbased modeling of joint dependencies. By integrating these modules, SEM significantly improves spatial understanding, leading to robust and generalizable manipulation across diverse tasks that outperform existing baselines. 

---
# SpecMaskFoley: Steering Pretrained Spectral Masked Generative Transformer Toward Synchronized Video-to-audio Synthesis via ControlNet 

**Authors**: Zhi Zhong, Akira Takahashi, Shuyang Cui, Keisuke Toyama, Shusuke Takahashi, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2505.16195)  

**Abstract**: Foley synthesis aims to synthesize high-quality audio that is both semantically and temporally aligned with video frames. Given its broad application in creative industries, the task has gained increasing attention in the research community. To avoid the non-trivial task of training audio generative models from scratch, adapting pretrained audio generative models for video-synchronized foley synthesis presents an attractive direction. ControlNet, a method for adding fine-grained controls to pretrained generative models, has been applied to foley synthesis, but its use has been limited to handcrafted human-readable temporal conditions. In contrast, from-scratch models achieved success by leveraging high-dimensional deep features extracted using pretrained video encoders. We have observed a performance gap between ControlNet-based and from-scratch foley models. To narrow this gap, we propose SpecMaskFoley, a method that steers the pretrained SpecMaskGIT model toward video-synchronized foley synthesis via ControlNet. To unlock the potential of a single ControlNet branch, we resolve the discrepancy between the temporal video features and the time-frequency nature of the pretrained SpecMaskGIT via a frequency-aware temporal feature aligner, eliminating the need for complicated conditioning mechanisms widely used in prior arts. Evaluations on a common foley synthesis benchmark demonstrate that SpecMaskFoley could even outperform strong from-scratch baselines, substantially advancing the development of ControlNet-based foley synthesis models. Demo page: this https URL 

---
# VLM-R$^3$: Region Recognition, Reasoning, and Refinement for Enhanced Multimodal Chain-of-Thought 

**Authors**: Chaoya Jiang, Yongrui Heng, Wei Ye, Han Yang, Haiyang Xu, Ming Yan, Ji Zhang, Fei Huang, Shikun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16192)  

**Abstract**: Recently, reasoning-based MLLMs have achieved a degree of success in generating long-form textual reasoning chains. However, they still struggle with complex tasks that necessitate dynamic and iterative focusing on and revisiting of visual regions to achieve precise grounding of textual reasoning in visual evidence. We introduce \textbf{VLM-R$^3$} (\textbf{V}isual \textbf{L}anguage \textbf{M}odel with \textbf{R}egion \textbf{R}ecognition and \textbf{R}easoning), a framework that equips an MLLM with the ability to (i) decide \emph{when} additional visual evidence is needed, (ii) determine \emph{where} to ground within the image, and (iii) seamlessly weave the relevant sub-image content back into an interleaved chain-of-thought. The core of our method is \textbf{Region-Conditioned Reinforcement Policy Optimization (R-GRPO)}, a training paradigm that rewards the model for selecting informative regions, formulating appropriate transformations (e.g.\ crop, zoom), and integrating the resulting visual context into subsequent reasoning steps. To bootstrap this policy, we compile a modest but carefully curated Visuo-Lingual Interleaved Rationale (VLIR) corpus that provides step-level supervision on region selection and textual justification. Extensive experiments on MathVista, ScienceQA, and other benchmarks show that VLM-R$^3$ sets a new state of the art in zero-shot and few-shot settings, with the largest gains appearing on questions demanding subtle spatial reasoning or fine-grained visual cue extraction. 

---
# EasyInsert: A Data-Efficient and Generalizable Insertion Policy 

**Authors**: Guanghe Li, Junming Zhao, Shengjie Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16187)  

**Abstract**: Insertion task is highly challenging that requires robots to operate with exceptional precision in cluttered environments. Existing methods often have poor generalization capabilities. They typically function in restricted and structured environments, and frequently fail when the plug and socket are far apart, when the scene is densely cluttered, or when handling novel objects. They also rely on strong assumptions such as access to CAD models or a digital twin in simulation. To address this, we propose EasyInsert, a framework which leverages the human intuition that relative pose (delta pose) between plug and socket is sufficient for successful insertion, and employs efficient and automated real-world data collection with minimal human labor to train a generalizable model for relative pose prediction. During execution, EasyInsert follows a coarse-to-fine execution procedure based on predicted delta pose, and successfully performs various insertion tasks. EasyInsert demonstrates strong zero-shot generalization capability for unseen objects in cluttered environments, handling cases with significant initial pose deviations while maintaining high sample efficiency and requiring little human effort. In real-world experiments, with just 5 hours of training data, EasyInsert achieves over 90% success in zero-shot insertion for 13 out of 15 unseen novel objects, including challenging objects like Type-C cables, HDMI cables, and Ethernet cables. Furthermore, with only one human demonstration and 4 minutes of automatically collected data for fine-tuning, it reaches over 90% success rate for all 15 objects. 

---
# Understanding Generative AI Capabilities in Everyday Image Editing Tasks 

**Authors**: Mohammad Reza Taesiri, Brandon Collins, Logan Bolton, Viet Dac Lai, Franck Dernoncourt, Trung Bui, Anh Totti Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16181)  

**Abstract**: Generative AI (GenAI) holds significant promise for automating everyday image editing tasks, especially following the recent release of GPT-4o on March 25, 2025. However, what subjects do people most often want edited? What kinds of editing actions do they want to perform (e.g., removing or stylizing the subject)? Do people prefer precise edits with predictable outcomes or highly creative ones? By understanding the characteristics of real-world requests and the corresponding edits made by freelance photo-editing wizards, can we draw lessons for improving AI-based editors and determine which types of requests can currently be handled successfully by AI editors? In this paper, we present a unique study addressing these questions by analyzing 83k requests from the past 12 years (2013-2025) on the Reddit community, which collected 305k PSR-wizard edits. According to human ratings, approximately only 33% of requests can be fulfilled by the best AI editors (including GPT-4o, Gemini-2.0-Flash, SeedEdit). Interestingly, AI editors perform worse on low-creativity requests that require precise editing than on more open-ended tasks. They often struggle to preserve the identity of people and animals, and frequently make non-requested touch-ups. On the other side of the table, VLM judges (e.g., o1) perform differently from human judges and may prefer AI edits more than human edits. Code and qualitative examples are available at: this https URL 

---
# QuickVideo: Real-Time Long Video Understanding with System Algorithm Co-Design 

**Authors**: Benjamin Schneider, Dongfu Jiang, Chao Du, Tianyu Pang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.16175)  

**Abstract**: Long-video understanding has emerged as a crucial capability in real-world applications such as video surveillance, meeting summarization, educational lecture analysis, and sports broadcasting. However, it remains computationally prohibitive for VideoLLMs, primarily due to two bottlenecks: 1) sequential video decoding, the process of converting the raw bit stream to RGB frames can take up to a minute for hour-long video inputs, and 2) costly prefilling of up to several million tokens for LLM inference, resulting in high latency and memory use. To address these challenges, we propose QuickVideo, a system-algorithm co-design that substantially accelerates long-video understanding to support real-time downstream applications. It comprises three key innovations: QuickDecoder, a parallelized CPU-based video decoder that achieves 2-3 times speedup by splitting videos into keyframe-aligned intervals processed concurrently; QuickPrefill, a memory-efficient prefilling method using KV-cache pruning to support more frames with less GPU memory; and an overlapping scheme that overlaps CPU video decoding with GPU inference. Together, these components infernece time reduce by a minute on long video inputs, enabling scalable, high-quality video understanding even on limited hardware. Experiments show that QuickVideo generalizes across durations and sampling rates, making long video processing feasible in practice. 

---
# Automated Feedback Loops to Protect Text Simplification with Generative AI from Information Loss 

**Authors**: Abhay Kumara Sri Krishna Nandiraju, Gondy Leroy, David Kauchak, Arif Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.16172)  

**Abstract**: Understanding health information is essential in achieving and maintaining a healthy life. We focus on simplifying health information for better understanding. With the availability of generative AI, the simplification process has become efficient and of reasonable quality, however, the algorithms remove information that may be crucial for comprehension. In this study, we compare generative AI to detect missing information in simplified text, evaluate its importance, and fix the text with the missing information. We collected 50 health information texts and simplified them using gpt-4-0613. We compare five approaches to identify missing elements and regenerate the text by inserting the missing elements. These five approaches involve adding missing entities and missing words in various ways: 1) adding all the missing entities, 2) adding all missing words, 3) adding the top-3 entities ranked by gpt-4-0613, and 4, 5) serving as controls for comparison, adding randomly chosen entities. We use cosine similarity and ROUGE scores to evaluate the semantic similarity and content overlap between the original, simplified, and reconstructed simplified text. We do this for both summaries and full text. Overall, we find that adding missing entities improves the text. Adding all the missing entities resulted in better text regeneration, which was better than adding the top-ranked entities or words, or random words. Current tools can identify these entities, but are not valuable in ranking them. 

---
# When VLMs Meet Image Classification: Test Sets Renovation via Missing Label Identification 

**Authors**: Zirui Pang, Haosheng Tan, Yuhan Pu, Zhijie Deng, Zhouan Shen, Keyu Hu, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16149)  

**Abstract**: Image classification benchmark datasets such as CIFAR, MNIST, and ImageNet serve as critical tools for model evaluation. However, despite the cleaning efforts, these datasets still suffer from pervasive noisy labels and often contain missing labels due to the co-existing image pattern where multiple classes appear in an image sample. This results in misleading model comparisons and unfair evaluations. Existing label cleaning methods focus primarily on noisy labels, but the issue of missing labels remains largely overlooked. Motivated by these challenges, we present a comprehensive framework named REVEAL, integrating state-of-the-art pre-trained vision-language models (e.g., LLaVA, BLIP, Janus, Qwen) with advanced machine/human label curation methods (e.g., Docta, Cleanlab, MTurk), to systematically address both noisy labels and missing label detection in widely-used image classification test sets. REVEAL detects potential noisy labels and omissions, aggregates predictions from various methods, and refines label accuracy through confidence-informed predictions and consensus-based filtering. Additionally, we provide a thorough analysis of state-of-the-art vision-language models and pre-trained image classifiers, highlighting their strengths and limitations within the context of dataset renovation by revealing 10 observations. Our method effectively reveals missing labels from public datasets and provides soft-labeled results with likelihoods. Through human verifications, REVEAL significantly improves the quality of 6 benchmark test sets, highly aligning to human judgments and enabling more accurate and meaningful comparisons in image classification. 

---
# Steering LVLMs via Sparse Autoencoder for Hallucination Mitigation 

**Authors**: Zhenglin Hua, Jinghan He, Zijun Yao, Tianxu Han, Haiyun Guo, Yuheng Jia, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16146)  

**Abstract**: Large vision-language models (LVLMs) have achieved remarkable performance on multimodal tasks such as visual question answering (VQA) and image captioning. However, they still suffer from hallucinations, generating text inconsistent with visual input, posing significant risks in real-world applications. Existing approaches to address this issue focus on incorporating external knowledge bases, alignment training, or decoding strategies, all of which require substantial computational cost and time. Recent works try to explore more efficient alternatives by adjusting LVLMs' internal representations. Although promising, these methods may cause hallucinations to be insufficiently suppressed or lead to excessive interventions that negatively affect normal semantics. In this work, we leverage sparse autoencoders (SAEs) to identify semantic directions closely associated with either hallucinations or actuality, realizing more precise and direct hallucination-related representations. Our analysis demonstrates that interventions along the faithful direction we identified can mitigate hallucinations, while those along the hallucinatory direction can exacerbate them. Building on these insights, we propose Steering LVLMs via SAE Latent Directions (SSL), a training-free method based on SAE-derived latent directions to mitigate hallucinations in LVLMs. Extensive experiments demonstrate that SSL significantly outperforms existing decoding approaches in mitigating hallucinations, while maintaining transferability across different model architectures with negligible additional time overhead. 

---
# Interpretable Machine Learning for Macro Alpha: A News Sentiment Case Study 

**Authors**: Yuke Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.16136)  

**Abstract**: This study introduces an interpretable machine learning (ML) framework to extract macroeconomic alpha from global news sentiment. We process the Global Database of Events, Language, and Tone (GDELT) Project's worldwide news feed using FinBERT -- a Bidirectional Encoder Representations from Transformers (BERT) based model pretrained on finance-specific language -- to construct daily sentiment indices incorporating mean tone, dispersion, and event impact. These indices drive an XGBoost classifier, benchmarked against logistic regression, to predict next-day returns for EUR/USD, USD/JPY, and 10-year U.S. Treasury futures (ZN). Rigorous out-of-sample (OOS) backtesting (5-fold expanding-window cross-validation, OOS period: c. 2017-April 2025) demonstrates exceptional, cost-adjusted performance for the XGBoost strategy: Sharpe ratios achieve 5.87 (EUR/USD), 4.65 (USD/JPY), and 4.65 (Treasuries), with respective compound annual growth rates (CAGRs) exceeding 50% in Foreign Exchange (FX) and 22% in bonds. Shapley Additive Explanations (SHAP) affirm that sentiment dispersion and article impact are key predictive features. Our findings establish that integrating domain-specific Natural Language Processing (NLP) with interpretable ML offers a potent and explainable source of macro alpha. 

---
# Scalable Graph Generative Modeling via Substructure Sequences 

**Authors**: Zehong Wang, Zheyuan Zhang, Tianyi Ma, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.16130)  

**Abstract**: Graph neural networks (GNNs) has been predominantly driven by message-passing, where node representations are iteratively updated via local neighborhood aggregation. Despite their success, message-passing suffers from fundamental limitations -- including constrained expressiveness, over-smoothing, over-squashing, and limited capacity to model long-range dependencies. These issues hinder scalability: increasing data size or model size often fails to yield improved performance, limiting the viability of GNNs as backbones for graph foundation models. In this work, we explore pathways beyond message-passing and introduce Generative Graph Pattern Machine (G$^2$PM), a generative Transformer pre-training framework for graphs. G$^2$PM represents graph instances (nodes, edges, or entire graphs) as sequences of substructures, and employs generative pre-training over the sequences to learn generalizable, transferable representations. Empirically, G$^2$PM demonstrates strong scalability: on the ogbn-arxiv benchmark, it continues to improve with model sizes up to 60M parameters, outperforming prior generative approaches that plateau at significantly smaller scales (e.g., 3M). In addition, we systematically analyze the model design space, highlighting key architectural choices that contribute to its scalability and generalization. Across diverse tasks -- including node classification, graph classification, and transfer learning -- G$^2$PM consistently outperforms strong baselines, establishing a compelling foundation for scalable graph learning. The code and dataset are available at this https URL. 

---
# Towards Trustworthy Keylogger detection: A Comprehensive Analysis of Ensemble Techniques and Feature Selections through Explainable AI 

**Authors**: Monirul Islam Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2505.16103)  

**Abstract**: Keylogger detection involves monitoring for unusual system behaviors such as delays between typing and character display, analyzing network traffic patterns for data exfiltration. In this study, we provide a comprehensive analysis for keylogger detection with traditional machine learning models - SVC, Random Forest, Decision Tree, XGBoost, AdaBoost, Logistic Regression and Naive Bayes and advanced ensemble methods including Stacking, Blending and Voting. Moreover, feature selection approaches such as Information gain, Lasso L1 and Fisher Score are thoroughly assessed to improve predictive performance and lower computational complexity. The Keylogger Detection dataset from publicly available Kaggle website is used in this project. In addition to accuracy-based classification, this study implements the approach for model interpretation using Explainable AI (XAI) techniques namely SHAP (Global) and LIME (Local) to deliver finer explanations for how much each feature contributes in assisting or hindering the detection process. To evaluate the models result, we have used AUC score, sensitivity, Specificity, Accuracy and F1 score. The best performance was achieved by AdaBoost with 99.76% accuracy, F1 score of 0.99, 100% precision, 98.6% recall, 1.0 specificity and 0.99 of AUC that is near-perfect classification with Fisher Score. 

---
# Date Fragments: A Hidden Bottleneck of Tokenization for Temporal Reasoning 

**Authors**: Gagan Bhatia, Maxime Peyrard, Wei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.16088)  

**Abstract**: Modern BPE tokenizers often split calendar dates into meaningless fragments, e.g., 20250312 $\rightarrow$ 202, 503, 12, inflating token counts and obscuring the inherent structure needed for robust temporal reasoning. In this work, we (1) introduce a simple yet interpretable metric, termed date fragmentation ratio, that measures how faithfully a tokenizer preserves multi-digit date components; (2) release DateAugBench, a suite of 6500 examples spanning three temporal reasoning tasks: context-based date resolution, format-invariance puzzles, and date arithmetic across historical, contemporary, and future regimes; and (3) through layer-wise probing and causal attention-hop analyses, uncover an emergent date-abstraction mechanism whereby large language models stitch together the fragments of month, day, and year components for temporal reasoning. Our experiments show that excessive fragmentation correlates with accuracy drops of up to 10 points on uncommon dates like historical and futuristic dates. Further, we find that the larger the model, the faster the emergent date abstraction that heals date fragments is accomplished. Lastly, we observe a reasoning path that LLMs follow to assemble date fragments, typically differing from human interpretation (year $\rightarrow$ month $\rightarrow$ day). 

---
# Bidirectional Variational Autoencoders 

**Authors**: Bart Kosko, Olaoluwa Adigun  

**Link**: [PDF](https://arxiv.org/pdf/2505.16074)  

**Abstract**: We present the new bidirectional variational autoencoder (BVAE) network architecture. The BVAE uses a single neural network both to encode and decode instead of an encoder-decoder network pair. The network encodes in the forward direction and decodes in the backward direction through the same synaptic web. Simulations compared BVAEs and ordinary VAEs on the four image tasks of image reconstruction, classification, interpolation, and generation. The image datasets included MNIST handwritten digits, Fashion-MNIST, CIFAR-10, and CelebA-64 face images. The bidirectional structure of BVAEs cut the parameter count by almost 50% and still slightly outperformed the unidirectional VAEs. 

---
# Merge to Mix: Mixing Datasets via Model Merging 

**Authors**: Zhixu Silvia Tao, Kasper Vinken, Hao-Wei Yeh, Avi Cooper, Xavier Boix  

**Link**: [PDF](https://arxiv.org/pdf/2505.16066)  

**Abstract**: Mixing datasets for fine-tuning large models (LMs) has become critical for maximizing performance on downstream tasks. However, composing effective dataset mixtures typically relies on heuristics and trial-and-error, often requiring multiple fine-tuning runs to achieve the desired outcome. We propose a novel method, $\textit{Merge to Mix}$, that accelerates composing dataset mixtures through model merging. Model merging is a recent technique that combines the abilities of multiple individually fine-tuned LMs into a single LM by using a few simple arithmetic operations. Our key insight is that merging models individually fine-tuned on each dataset in a mixture can effectively serve as a surrogate for a model fine-tuned on the entire mixture. Merge to Mix leverages this insight to accelerate selecting dataset mixtures without requiring full fine-tuning on each candidate mixture. Our experiments demonstrate that Merge to Mix surpasses state-of-the-art methods in dataset selection for fine-tuning LMs. 

---
# Mesh-free sparse identification of nonlinear dynamics 

**Authors**: Mars Liyao Gao, J. Nathan Kutz, Bernat Font  

**Link**: [PDF](https://arxiv.org/pdf/2505.16058)  

**Abstract**: Identifying the governing equations of a dynamical system is one of the most important tasks for scientific modeling. However, this procedure often requires high-quality spatio-temporal data uniformly sampled on structured grids. In this paper, we propose mesh-free SINDy, a novel algorithm which leverages the power of neural network approximation as well as auto-differentiation to identify governing equations from arbitrary sensor placements and non-uniform temporal data sampling. We show that mesh-free SINDy is robust to high noise levels and limited data while remaining computationally efficient. In our implementation, the training procedure is straight-forward and nearly free of hyperparameter tuning, making mesh-free SINDy widely applicable to many scientific and engineering problems. In the experiments, we demonstrate its effectiveness on a series of PDEs including the Burgers' equation, the heat equation, the Korteweg-De Vries equation and the 2D advection-diffusion equation. We conduct detailed numerical experiments on all datasets, varying the noise levels and number of samples, and we also compare our approach to previous state-of-the-art methods. It is noteworthy that, even in high-noise and low-data scenarios, mesh-free SINDy demonstrates robust PDE discovery, achieving successful identification with up to 75% noise for the Burgers' equation using 5,000 samples and with as few as 100 samples and 1% noise. All of this is achieved within a training time of under one minute. 

---
# Signals of Provenance: Practices & Challenges of Navigating Indicators in AI-Generated Media for Sighted and Blind Individuals 

**Authors**: Ayae Ide, Tory Park, Jaron Mink, Tanusree Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2505.16057)  

**Abstract**: AI-Generated (AIG) content has become increasingly widespread by recent advances in generative models and the easy-to-use tools that have significantly lowered the technical barriers for producing highly realistic audio, images, and videos through simple natural language prompts. In response, platforms are adopting provable provenance with platforms recommending AIG to be self-disclosed and signaled to users. However, these indicators may be often missed, especially when they rely solely on visual cues and make them ineffective to users with different sensory abilities. To address the gap, we conducted semi-structured interviews (N=28) with 15 sighted and 13 BLV participants to examine their interaction with AIG content through self-disclosed AI indicators. Our findings reveal diverse mental models and practices, highlighting different strengths and weaknesses of content-based (e.g., title, description) and menu-aided (e.g., AI labels) indicators. While sighted participants leveraged visual and audio cues, BLV participants primarily relied on audio and existing assistive tools, limiting their ability to identify AIG. Across both groups, they frequently overlooked menu-aided indicators deployed by platforms and rather interacted with content-based indicators such as title and comments. We uncovered usability challenges stemming from inconsistent indicator placement, unclear metadata, and cognitive overload. These issues were especially critical for BLV individuals due to the insufficient accessibility of interface elements. We provide practical recommendations and design implications for future AIG indicators across several dimensions. 

---
# Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models 

**Authors**: Jingcong Liang, Siyuan Wang, Miren Tian, Yitong Li, Duyu Tang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.16056)  

**Abstract**: Mixture-of-Experts (MoE) enables efficient scaling of large language models (LLMs) with sparsely activated experts during inference. To effectively deploy large MoE models on memory-constrained devices, many systems introduce *expert offloading* that caches a subset of experts in fast memory, leaving others on slow memory to run on CPU or load on demand. While some research has exploited the locality of expert activations, where consecutive tokens activate similar experts, the degree of this **local routing consistency** varies across models and remains understudied. In this paper, we propose two metrics to measure local routing consistency of MoE models: (1) **Segment Routing Best Performance (SRP)**, which evaluates how well a fixed group of experts can cover the needs of a segment of tokens, and (2) **Segment Cache Best Hit Rate (SCH)**, which measures the optimal segment-level cache hit rate under a given cache size limit. We analyzed 20 MoE LLMs with diverse sizes and architectures and found that models that apply MoE on every layer and do not use shared experts exhibit the highest local routing consistency. We further showed that domain-specialized experts contribute more to routing consistency than vocabulary-specialized ones, and that most models can balance between cache effectiveness and efficiency with cache sizes approximately 2x the active experts. These findings pave the way for memory-efficient MoE design and deployment without compromising inference speed. We publish the code for replicating experiments at this https URL . 

---
# Equivariant Eikonal Neural Networks: Grid-Free, Scalable Travel-Time Prediction on Homogeneous Spaces 

**Authors**: Alejandro García-Castellanos, David R. Wessels, Nicky J. van den Berg, Remco Duits, Daniël M. Pelt, Erik J. Bekkers  

**Link**: [PDF](https://arxiv.org/pdf/2505.16035)  

**Abstract**: We introduce Equivariant Neural Eikonal Solvers, a novel framework that integrates Equivariant Neural Fields (ENFs) with Neural Eikonal Solvers. Our approach employs a single neural field where a unified shared backbone is conditioned on signal-specific latent variables - represented as point clouds in a Lie group - to model diverse Eikonal solutions. The ENF integration ensures equivariant mapping from these latent representations to the solution field, delivering three key benefits: enhanced representation efficiency through weight-sharing, robust geometric grounding, and solution steerability. This steerability allows transformations applied to the latent point cloud to induce predictable, geometrically meaningful modifications in the resulting Eikonal solution. By coupling these steerable representations with Physics-Informed Neural Networks (PINNs), our framework accurately models Eikonal travel-time solutions while generalizing to arbitrary Riemannian manifolds with regular group actions. This includes homogeneous spaces such as Euclidean, position-orientation, spherical, and hyperbolic manifolds. We validate our approach through applications in seismic travel-time modeling of 2D and 3D benchmark datasets. Experimental results demonstrate superior performance, scalability, adaptability, and user controllability compared to existing Neural Operator-based Eikonal solver methods. 

---
# Benchmarking Chest X-ray Diagnosis Models Across Multinational Datasets 

**Authors**: Qinmei Xu, Yiheng Li, Xianghao Zhan, Ahmet Gorkem Er, Brittany Dashevsky, Chuanjun Xu, Mohammed Alawad, Mengya Yang, Liu Ya, Changsheng Zhou, Xiao Li, Haruka Itakura, Olivier Gevaert  

**Link**: [PDF](https://arxiv.org/pdf/2505.16027)  

**Abstract**: Foundation models leveraging vision-language pretraining have shown promise in chest X-ray (CXR) interpretation, yet their real-world performance across diverse populations and diagnostic tasks remains insufficiently evaluated. This study benchmarks the diagnostic performance and generalizability of foundation models versus traditional convolutional neural networks (CNNs) on multinational CXR datasets. We evaluated eight CXR diagnostic models - five vision-language foundation models and three CNN-based architectures - across 37 standardized classification tasks using six public datasets from the USA, Spain, India, and Vietnam, and three private datasets from hospitals in China. Performance was assessed using AUROC, AUPRC, and other metrics across both shared and dataset-specific tasks. Foundation models outperformed CNNs in both accuracy and task coverage. MAVL, a model incorporating knowledge-enhanced prompts and structured supervision, achieved the highest performance on public (mean AUROC: 0.82; AUPRC: 0.32) and private (mean AUROC: 0.95; AUPRC: 0.89) datasets, ranking first in 14 of 37 public and 3 of 4 private tasks. All models showed reduced performance on pediatric cases, with average AUROC dropping from 0.88 +/- 0.18 in adults to 0.57 +/- 0.29 in children (p = 0.0202). These findings highlight the value of structured supervision and prompt design in radiologic AI and suggest future directions including geographic expansion and ensemble modeling for clinical deployment. Code for all evaluated models is available at this https URL 

---
# Toward Theoretical Insights into Diffusion Trajectory Distillation via Operator Merging 

**Authors**: Weiguo Gao, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16024)  

**Abstract**: Diffusion trajectory distillation methods aim to accelerate sampling in diffusion models, which produce high-quality outputs but suffer from slow sampling speeds. These methods train a student model to approximate the multi-step denoising process of a pretrained teacher model in a single step, enabling one-shot generation. However, theoretical insights into the trade-off between different distillation strategies and generative quality remain limited, complicating their optimization and selection. In this work, we take a first step toward addressing this gap. Specifically, we reinterpret trajectory distillation as an operator merging problem in the linear regime, where each step of the teacher model is represented as a linear operator acting on noisy data. These operators admit a clear geometric interpretation as projections and rescalings corresponding to the noise schedule. During merging, signal shrinkage occurs as a convex combination of operators, arising from both discretization and limited optimization time of the student model. We propose a dynamic programming algorithm to compute the optimal merging strategy that maximally preserves signal fidelity. Additionally, we demonstrate the existence of a sharp phase transition in the optimal strategy, governed by data covariance structures. Our findings enhance the theoretical understanding of diffusion trajectory distillation and offer practical insights for improving distillation strategies. 

---
# NOVER: Incentive Training for Language Models via Verifier-Free Reinforcement Learning 

**Authors**: Wei Liu, Siya Qi, Xinyu Wang, Chen Qian, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.16022)  

**Abstract**: Recent advances such as DeepSeek R1-Zero highlight the effectiveness of incentive training, a reinforcement learning paradigm that computes rewards solely based on the final answer part of a language model's output, thereby encouraging the generation of intermediate reasoning steps. However, these methods fundamentally rely on external verifiers, which limits their applicability to domains like mathematics and coding where such verifiers are readily available. Although reward models can serve as verifiers, they require high-quality annotated data and are costly to train. In this work, we propose NOVER, NO-VERifier Reinforcement Learning, a general reinforcement learning framework that requires only standard supervised fine-tuning data with no need for an external verifier. NOVER enables incentive training across a wide range of text-to-text tasks and outperforms the model of the same size distilled from large reasoning models such as DeepSeek R1 671B by 7.7 percent. Moreover, the flexibility of NOVER enables new possibilities for optimizing large language models, such as inverse incentive training. 

---
# LAGO: Few-shot Crosslingual Embedding Inversion Attacks via Language Similarity-Aware Graph Optimization 

**Authors**: Wenrui Yu, Yiyi Chen, Johannes Bjerva, Sokol Kosta, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.16008)  

**Abstract**: We propose LAGO - Language Similarity-Aware Graph Optimization - a novel approach for few-shot cross-lingual embedding inversion attacks, addressing critical privacy vulnerabilities in multilingual NLP systems. Unlike prior work in embedding inversion attacks that treat languages independently, LAGO explicitly models linguistic relationships through a graph-based constrained distributed optimization framework. By integrating syntactic and lexical similarity as edge constraints, our method enables collaborative parameter learning across related languages. Theoretically, we show this formulation generalizes prior approaches, such as ALGEN, which emerges as a special case when similarity constraints are relaxed. Our framework uniquely combines Frobenius-norm regularization with linear inequality or total variation constraints, ensuring robust alignment of cross-lingual embedding spaces even with extremely limited data (as few as 10 samples per language). Extensive experiments across multiple languages and embedding models demonstrate that LAGO substantially improves the transferability of attacks with 10-20% increase in Rouge-L score over baselines. This work establishes language similarity as a critical factor in inversion attack transferability, urging renewed focus on language-aware privacy-preserving multilingual embeddings. 

---
# Interpretability Illusions with Sparse Autoencoders: Evaluating Robustness of Concept Representations 

**Authors**: Aaron J. Li, Suraj Srinivas, Usha Bhalla, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.16004)  

**Abstract**: Sparse autoencoders (SAEs) are commonly used to interpret the internal activations of large language models (LLMs) by mapping them to human-interpretable concept representations. While existing evaluations of SAEs focus on metrics such as the reconstruction-sparsity tradeoff, human (auto-)interpretability, and feature disentanglement, they overlook a critical aspect: the robustness of concept representations to input perturbations. We argue that robustness must be a fundamental consideration for concept representations, reflecting the fidelity of concept labeling. To this end, we formulate robustness quantification as input-space optimization problems and develop a comprehensive evaluation framework featuring realistic scenarios in which adversarial perturbations are crafted to manipulate SAE representations. Empirically, we find that tiny adversarial input perturbations can effectively manipulate concept-based interpretations in most scenarios without notably affecting the outputs of the base LLMs themselves. Overall, our results suggest that SAE concept representations are fragile and may be ill-suited for applications in model monitoring and oversight. 

---
# SLMEval: Entropy-Based Calibration for Human-Aligned Evaluation of Large Language Models 

**Authors**: Roland Daynauth, Christopher Clarke, Krisztian Flautner, Lingjia Tang, Jason Mars  

**Link**: [PDF](https://arxiv.org/pdf/2505.16003)  

**Abstract**: The LLM-as-a-Judge paradigm offers a scalable, reference-free approach for evaluating language models. Although several calibration techniques have been proposed to better align these evaluators with human judgment, prior studies focus primarily on narrow, well-structured benchmarks. As a result, it remains unclear whether such calibrations generalize to real-world, open-ended tasks.
In this work, we show that SOTA calibrated evaluators often fail in these settings, exhibiting weak or even negative correlation with human judgments. To address this, we propose SLMEval, a novel and efficient calibration method based on entropy maximization over a small amount of human preference data. By estimating a latent distribution over model quality and reweighting evaluator scores accordingly, SLMEval achieves strong correlation with human evaluations across two real-world production use cases and the public benchmark. For example, on one such task, SLMEval achieves a Spearman correlation of 0.57 with human judgments, while G-Eval yields a negative correlation. In addition, SLMEval reduces evaluation costs by 5-30x compared to GPT-4-based calibrated evaluators such as G-eval. 

---
# Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions 

**Authors**: Sasha Boguraev, Christopher Potts, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2505.16002)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful sources of evidence for linguists seeking to develop theories of syntax. In this paper, we argue that causal interpretability methods, applied to LLMs, can greatly enhance the value of such evidence by helping us characterize the abstract mechanisms that LLMs learn to use. Our empirical focus is a set of English filler-gap dependency constructions (e.g., questions, relative clauses). Linguistic theories largely agree that these constructions share many properties. Using experiments based in Distributed Interchange Interventions, we show that LLMs converge on similar abstract analyses of these constructions. These analyses also reveal previously overlooked factors -- relating to frequency, filler type, and surrounding context -- that could motivate changes to standard linguistic theory. Overall, these results suggest that mechanistic, internal analyses of LLMs can push linguistic theory forward. 

---
# Leveraging Online Data to Enhance Medical Knowledge in a Small Persian Language Model 

**Authors**: Mehrdad ghassabi, Pedram Rostami, Hamidreza Baradaran Kashani, Amirhossein Poursina, Zahra Kazemi, Milad Tavakoli  

**Link**: [PDF](https://arxiv.org/pdf/2505.16000)  

**Abstract**: The rapid advancement of language models has demonstrated the potential of artificial intelligence in the healthcare industry. However, small language models struggle with specialized domains in low-resource languages like Persian. While numerous medical-domain websites exist in Persian, no curated dataset or corpus has been available making ours the first of its kind. This study explores the enhancement of medical knowledge in a small language model by leveraging accessible online data, including a crawled corpus from medical magazines and a dataset of real doctor-patient QA pairs. We fine-tuned a baseline model using our curated data to improve its medical knowledge. Benchmark evaluations demonstrate that the fine-tuned model achieves improved accuracy in medical question answering and provides better responses compared to its baseline. This work highlights the potential of leveraging open-access online data to enrich small language models in medical fields, providing a novel solution for Persian medical AI applications suitable for resource-constrained environments. 

---
# Domain Adaptive Skin Lesion Classification via Conformal Ensemble of Vision Transformers 

**Authors**: Mehran Zoravar, Shadi Alijani, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.15997)  

**Abstract**: Exploring the trustworthiness of deep learning models is crucial, especially in critical domains such as medical imaging decision support systems. Conformal prediction has emerged as a rigorous means of providing deep learning models with reliable uncertainty estimates and safety guarantees. However, conformal prediction results face challenges due to the backbone model's struggles in domain-shifted scenarios, such as variations in different sources. To aim this challenge, this paper proposes a novel framework termed Conformal Ensemble of Vision Transformers (CE-ViTs) designed to enhance image classification performance by prioritizing domain adaptation and model robustness, while accounting for uncertainty. The proposed method leverages an ensemble of vision transformer models in the backbone, trained on diverse datasets including HAM10000, Dermofit, and Skin Cancer ISIC datasets. This ensemble learning approach, calibrated through the combined mentioned datasets, aims to enhance domain adaptation through conformal learning. Experimental results underscore that the framework achieves a high coverage rate of 90.38\%, representing an improvement of 9.95\% compared to the HAM10000 model. This indicates a strong likelihood that the prediction set includes the true label compared to singular models. Ensemble learning in CE-ViTs significantly improves conformal prediction performance, increasing the average prediction set size for challenging misclassified samples from 1.86 to 3.075. 

---
# Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning 

**Authors**: Alex Su, Haozhe Wang, Weimin Ren, Fangzhen Lin, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15966)  

**Abstract**: Chain-of-thought reasoning has significantly improved the performance of Large Language Models (LLMs) across various domains. However, this reasoning process has been confined exclusively to textual space, limiting its effectiveness in visually intensive tasks. To address this limitation, we introduce the concept of reasoning in the pixel-space. Within this novel framework, Vision-Language Models (VLMs) are equipped with a suite of visual reasoning operations, such as zoom-in and select-frame. These operations enable VLMs to directly inspect, interrogate, and infer from visual evidences, thereby enhancing reasoning fidelity for visual tasks. Cultivating such pixel-space reasoning capabilities in VLMs presents notable challenges, including the model's initially imbalanced competence and its reluctance to adopt the newly introduced pixel-space operations. We address these challenges through a two-phase training approach. The first phase employs instruction tuning on synthesized reasoning traces to familiarize the model with the novel visual operations. Following this, a reinforcement learning (RL) phase leverages a curiosity-driven reward scheme to balance exploration between pixel-space reasoning and textual reasoning. With these visual operations, VLMs can interact with complex visual inputs, such as information-rich images or videos to proactively gather necessary information. We demonstrate that this approach significantly improves VLM performance across diverse visual reasoning benchmarks. Our 7B model, \model, achieves 84\% on V* bench, 74\% on TallyQA-Complex, and 84\% on InfographicsVQA, marking the highest accuracy achieved by any open-source model to date. These results highlight the importance of pixel-space reasoning and the effectiveness of our framework. 

---
# Pre-training Large Memory Language Models with Internal and External Knowledge 

**Authors**: Linxi Zhao, Sofian Zalouk, Christian K. Belardi, Justin Lovelace, Jin Peng Zhou, Kilian Q. Weinberger, Yoav Artzi, Jennifer J. Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.15962)  

**Abstract**: Neural language models are black-boxes -- both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We propose a new class of language models, Large Memory Language Models (LMLM) with a pre-training recipe that stores factual knowledge in both internal weights and an external database. Our approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger, knowledge-dense LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases. This work represents a fundamental shift in how language models interact with and manage factual knowledge. 

---
# Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey 

**Authors**: Chih-Kai Yang, Neo S. Ho, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15957)  

**Abstract**: With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field. 

---
# VideoGameQA-Bench: Evaluating Vision-Language Models for Video Game Quality Assurance 

**Authors**: Mohammad Reza Taesiri, Abhijay Ghildyal, Saman Zadtootaghaj, Nabajeet Barman, Cor-Paul Bezemer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15952)  

**Abstract**: With video games now generating the highest revenues in the entertainment industry, optimizing game development workflows has become essential for the sector's sustained growth. Recent advancements in Vision-Language Models (VLMs) offer considerable potential to automate and enhance various aspects of game development, particularly Quality Assurance (QA), which remains one of the industry's most labor-intensive processes with limited automation options. To accurately evaluate the performance of VLMs in video game QA tasks and determine their effectiveness in handling real-world scenarios, there is a clear need for standardized benchmarks, as existing benchmarks are insufficient to address the specific requirements of this domain. To bridge this gap, we introduce VideoGameQA-Bench, a comprehensive benchmark that covers a wide array of game QA activities, including visual unit testing, visual regression testing, needle-in-a-haystack tasks, glitch detection, and bug report generation for both images and videos of various games. Code and data are available at: this https URL 

---
# MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding 

**Authors**: Yuxiang Wei, Yanteng Zhang, Xi Xiao, Tianyang Wang, Xiao Wang, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2505.15946)  

**Abstract**: Decoding visual experiences from fMRI offers a powerful avenue to understand human perception and develop advanced brain-computer interfaces. However, current progress often prioritizes maximizing reconstruction fidelity while overlooking interpretability, an essential aspect for deriving neuroscientific insight. To address this gap, we propose MoRE-Brain, a neuro-inspired framework designed for high-fidelity, adaptable, and interpretable visual reconstruction. MoRE-Brain uniquely employs a hierarchical Mixture-of-Experts architecture where distinct experts process fMRI signals from functionally related voxel groups, mimicking specialized brain networks. The experts are first trained to encode fMRI into the frozen CLIP space. A finetuned diffusion model then synthesizes images, guided by expert outputs through a novel dual-stage routing mechanism that dynamically weighs expert contributions across the diffusion process. MoRE-Brain offers three main advancements: First, it introduces a novel Mixture-of-Experts architecture grounded in brain network principles for neuro-decoding. Second, it achieves efficient cross-subject generalization by sharing core expert networks while adapting only subject-specific routers. Third, it provides enhanced mechanistic insight, as the explicit routing reveals precisely how different modeled brain regions shape the semantic and spatial attributes of the reconstructed image. Extensive experiments validate MoRE-Brain's high reconstruction fidelity, with bottleneck analyses further demonstrating its effective utilization of fMRI signals, distinguishing genuine neural decoding from over-reliance on generative priors. Consequently, MoRE-Brain marks a substantial advance towards more generalizable and interpretable fMRI-based visual decoding. Code will be publicly available soon: this https URL. 

---
# VERDI: VLM-Embedded Reasoning for Autonomous Driving 

**Authors**: Bowen Feng, Zhiting Mei, Baiang Li, Julian Ost, Roger Girgis, Anirudha Majumdar, Felix Heide  

**Link**: [PDF](https://arxiv.org/pdf/2505.15925)  

**Abstract**: While autonomous driving (AD) stacks struggle with decision making under partial observability and real-world complexity, human drivers are capable of commonsense reasoning to make near-optimal decisions with limited information. Recent work has attempted to leverage finetuned Vision-Language Models (VLMs) for trajectory planning at inference time to emulate human behavior. Despite their success in benchmark evaluations, these methods are often impractical to deploy (a 70B parameter VLM inference at merely 8 tokens per second requires more than 160G of memory), and their monolithic network structure prohibits safety decomposition. To bridge this gap, we propose VLM-Embedded Reasoning for autonomous Driving (VERDI), a training-time framework that distills the reasoning process and commonsense knowledge of VLMs into the AD stack. VERDI augments modular differentiable end-to-end (e2e) AD models by aligning intermediate module outputs at the perception, prediction, and planning stages with text features explaining the driving reasoning process produced by VLMs. By encouraging alignment in latent space, \textsc{VERDI} enables the modular AD stack to internalize structured reasoning, without incurring the inference-time costs of large VLMs. We demonstrate the effectiveness of our method on the NuScenes dataset and find that VERDI outperforms existing e2e methods that do not embed reasoning by 10% in $\ell_{2}$ distance, while maintaining high inference speed. 

---
# Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization 

**Authors**: Aliakbar Nafar, Kristen Brent Venable, Zijun Cui, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15918)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential as factual knowledge bases; however, their capability to generate probabilistic knowledge about real-world events remains understudied. This paper investigates using probabilistic knowledge inherent in LLMs to derive probability estimates for statements concerning events and their interrelationships captured via a Bayesian Network (BN). Using LLMs in this context allows for the parameterization of BNs, enabling probabilistic modeling within specific domains. Experiments on eighty publicly available Bayesian Networks, from healthcare to finance, demonstrate that querying LLMs about the conditional probabilities of events provides meaningful results when compared to baselines, including random and uniform distributions, as well as approaches based on next-token generation probabilities. We explore how these LLM-derived distributions can serve as expert priors to refine distributions extracted from minimal data, significantly reducing systematic biases. Overall, this work introduces a promising strategy for automatically constructing Bayesian Networks by combining probabilistic knowledge extracted from LLMs with small amounts of real-world data. Additionally, we evaluate several prompting strategies for eliciting probabilistic knowledge from LLMs and establish the first comprehensive baseline for assessing LLM performance in extracting probabilistic knowledge. 

---
# BR-TaxQA-R: A Dataset for Question Answering with References for Brazilian Personal Income Tax Law, including case law 

**Authors**: Juvenal Domingos Júnior, Augusto Faria, E. Seiti de Oliveira, Erick de Brito, Matheus Teotonio, Andre Assumpção, Diedre Carmo, Roberto Lotufo, Jayr Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.15916)  

**Abstract**: This paper presents BR-TaxQA-R, a novel dataset designed to support question answering with references in the context of Brazilian personal income tax law. The dataset contains 715 questions from the 2024 official Q\&A document published by Brazil's Internal Revenue Service, enriched with statutory norms and administrative rulings from the Conselho Administrativo de Recursos Fiscais (CARF). We implement a Retrieval-Augmented Generation (RAG) pipeline using OpenAI embeddings for searching and GPT-4o-mini for answer generation. We compare different text segmentation strategies and benchmark our system against commercial tools such as ChatGPT and this http URL using RAGAS-based metrics. Results show that our custom RAG pipeline outperforms commercial systems in Response Relevancy, indicating stronger alignment with user queries, while commercial models achieve higher scores in Factual Correctness and fluency. These findings highlight a trade-off between legally grounded generation and linguistic fluency. Crucially, we argue that human expert evaluation remains essential to ensure the legal validity of AI-generated answers in high-stakes domains such as taxation. BR-TaxQA-R is publicly available at this https URL. 

---
# Last Layer Empirical Bayes 

**Authors**: Valentin Villecroze, Yixin Wang, Gabriel Loaiza-Ganem  

**Link**: [PDF](https://arxiv.org/pdf/2505.15888)  

**Abstract**: The task of quantifying the inherent uncertainty associated with neural network predictions is a key challenge in artificial intelligence. Bayesian neural networks (BNNs) and deep ensembles are among the most prominent approaches to tackle this task. Both approaches produce predictions by computing an expectation of neural network outputs over some distribution on the corresponding weights; this distribution is given by the posterior in the case of BNNs, and by a mixture of point masses for ensembles. Inspired by recent work showing that the distribution used by ensembles can be understood as a posterior corresponding to a learned data-dependent prior, we propose last layer empirical Bayes (LLEB). LLEB instantiates a learnable prior as a normalizing flow, which is then trained to maximize the evidence lower bound; to retain tractability we use the flow only on the last layer. We show why LLEB is well motivated, and how it interpolates between standard BNNs and ensembles in terms of the strength of the prior that they use. LLEB performs on par with existing approaches, highlighting that empirical Bayes is a promising direction for future research in uncertainty quantification. 

---
# GRIT: Teaching MLLMs to Think with Images 

**Authors**: Yue Fan, Xuehai He, Diji Yang, Kaizhi Zheng, Ching-Chen Kuo, Yuting Zheng, Sravana Jyothi Narayanaraju, Xinze Guan, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15879)  

**Abstract**: Recent studies have demonstrated the efficacy of using Reinforcement Learning (RL) in building reasoning models that articulate chains of thoughts prior to producing final answers. However, despite ongoing advances that aim at enabling reasoning for vision-language tasks, existing open-source visual reasoning models typically generate reasoning content with pure natural language, lacking explicit integration of visual information. This limits their ability to produce clearly articulated and visually grounded reasoning chains. To this end, we propose Grounded Reasoning with Images and Texts (GRIT), a novel method for training MLLMs to think with images. GRIT introduces a grounded reasoning paradigm, in which models generate reasoning chains that interleave natural language and explicit bounding box coordinates. These coordinates point to regions of the input image that the model consults during its reasoning process. Additionally, GRIT is equipped with a reinforcement learning approach, GRPO-GR, built upon the GRPO algorithm. GRPO-GR employs robust rewards focused on the final answer accuracy and format of the grounded reasoning output, which eliminates the need for data with reasoning chain annotations or explicit bounding box labels. As a result, GRIT achieves exceptional data efficiency, requiring as few as 20 image-question-answer triplets from existing datasets. Comprehensive evaluations demonstrate that GRIT effectively trains MLLMs to produce coherent and visually grounded reasoning chains, showing a successful unification of reasoning and grounding abilities. 

---
# An Inclusive Foundation Model for Generalizable Cytogenetics in Precision Oncology 

**Authors**: Changchun Yang, Weiqian Dai, Yilan Zhang, Siyuan Chen, Jingdong Hu, Junkai Su, Yuxuan Chen, Ao Xu, Na Li, Xin Gao, Yongguo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15868)  

**Abstract**: Chromosome analysis is vital for diagnosing genetic disorders and guiding cancer therapy decisions through the identification of somatic clonal aberrations. However, developing an AI model are hindered by the overwhelming complexity and diversity of chromosomal abnormalities, requiring extensive annotation efforts, while automated methods remain task-specific and lack generalizability due to the scarcity of comprehensive datasets spanning diverse resource conditions. Here, we introduce CHROMA, a foundation model for cytogenomics, designed to overcome these challenges by learning generalizable representations of chromosomal abnormalities. Pre-trained on over 84,000 specimens (~4 million chromosomal images) via self-supervised learning, CHROMA outperforms other methods across all types of abnormalities, even when trained on fewer labelled data and more imbalanced datasets. By facilitating comprehensive mapping of instability and clonal leisons across various aberration types, CHROMA offers a scalable and generalizable solution for reliable and automated clinical analysis, reducing the annotation workload for experts and advancing precision oncology through the early detection of rare genomic abnormalities, enabling broad clinical AI applications and making advanced genomic analysis more accessible. 

---
# AutoData: A Multi-Agent System for Open Web Data Collection 

**Authors**: Tianyi Ma, Yiyue Qian, Zheyuan Zhang, Zehong Wang, Xiaoye Qian, Feifan Bai, Yifan Ding, Xuwei Luo, Shinan Zhang, Keerthiram Murugesan, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15859)  

**Abstract**: The exponential growth of data-driven systems and AI technologies has intensified the demand for high-quality web-sourced datasets. While existing datasets have proven valuable, conventional web data collection approaches face significant limitations in terms of human effort and scalability. Current data-collecting solutions fall into two categories: wrapper-based methods that struggle with adaptability and reproducibility, and large language model (LLM)-based approaches that incur substantial computational and financial costs. To address these challenges, we propose AutoData, a novel multi-agent system for Automated web Data collection, that requires minimal human intervention, i.e., only necessitating a natural language instruction specifying the desired dataset. In addition, AutoData is designed with a robust multi-agent architecture, featuring a novel oriented message hypergraph coordinated by a central task manager, to efficiently organize agents across research and development squads. Besides, we introduce a novel hypergraph cache system to advance the multi-agent collaboration process that enables efficient automated data collection and mitigates the token cost issues prevalent in existing LLM-based systems. Moreover, we introduce Instruct2DS, a new benchmark dataset supporting live data collection from web sources across three domains: academic, finance, and sports. Comprehensive evaluations over Instruct2DS and three existing benchmark datasets demonstrate AutoData's superior performance compared to baseline methods. Case studies on challenging tasks such as picture book collection and paper extraction from surveys further validate its applicability. Our source code and dataset are available at this https URL. 

---
# DisastIR: A Comprehensive Information Retrieval Benchmark for Disaster Management 

**Authors**: Kai Yin, Xiangjue Dong, Chengkai Liu, Lipai Huang, Yiming Xiao, Zhewei Liu, Ali Mostafavi, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15856)  

**Abstract**: Effective disaster management requires timely access to accurate and contextually relevant information. Existing Information Retrieval (IR) benchmarks, however, focus primarily on general or specialized domains, such as medicine or finance, neglecting the unique linguistic complexity and diverse information needs encountered in disaster management scenarios. To bridge this gap, we introduce DisastIR, the first comprehensive IR evaluation benchmark specifically tailored for disaster management. DisastIR comprises 9,600 diverse user queries and more than 1.3 million labeled query-passage pairs, covering 48 distinct retrieval tasks derived from six search intents and eight general disaster categories that include 301 specific event types. Our evaluations of 30 state-of-the-art retrieval models demonstrate significant performance variances across tasks, with no single model excelling universally. Furthermore, comparative analyses reveal significant performance gaps between general-domain and disaster management-specific tasks, highlighting the necessity of disaster management-specific benchmarks for guiding IR model selection to support effective decision-making in disaster management scenarios. All source codes and DisastIR are available at this https URL. 

---
# Integration of TinyML and LargeML: A Survey of 6G and Beyond 

**Authors**: Thai-Hoc Vu, Ngo Hoang Tu, Thien Huynh-The, Kyungchun Lee, Sunghwan Kim, Miroslav Voznak, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2505.15854)  

**Abstract**: The transition from 5G networks to 6G highlights a significant demand for machine learning (ML). Deep learning models, in particular, have seen wide application in mobile networking and communications to support advanced services in emerging wireless environments, such as smart healthcare, smart grids, autonomous vehicles, aerial platforms, digital twins, and the metaverse. The rapid expansion of Internet-of-Things (IoT) devices, many with limited computational capabilities, has accelerated the development of tiny machine learning (TinyML) and resource-efficient ML approaches for cost-effective services. However, the deployment of large-scale machine learning (LargeML) solutions require major computing resources and complex management strategies to support extensive IoT services and ML-generated content applications. Consequently, the integration of TinyML and LargeML is projected as a promising approach for future seamless connectivity and efficient resource management.
Although the integration of TinyML and LargeML shows abundant potential, several challenges persist, including performance optimization, practical deployment strategies, effective resource management, and security considerations. In this survey, we review and analyze the latest research aimed at enabling the integration of TinyML and LargeML models for the realization of smart services and applications in future 6G networks and beyond. The paper concludes by outlining critical challenges and identifying future research directions for the holistic integration of TinyML and LargeML in next-generation wireless networks. 

---
# Exploring Moral Exercises for Human Oversight of AI systems: Insights from Three Pilot Studies 

**Authors**: Silvia Crafa, Teresa Scantamburlo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15851)  

**Abstract**: This paper elaborates on the concept of moral exercises as a means to help AI actors cultivate virtues that enable effective human oversight of AI systems. We explore the conceptual framework and significance of moral exercises, situating them within the contexts of philosophical discourse, ancient practices, and contemporary AI ethics scholarship. We outline the core pillars of the moral exercises methodology - eliciting an engaged personal disposition, fostering relational understanding, and cultivating technomoral wisdom - and emphasize their relevance to key activities and competencies essential for human oversight of AI systems. Our argument is supported by findings from three pilot studies involving a company, a multidisciplinary team of AI researchers, and higher education students. These studies allow us to explore both the potential and the limitations of moral exercises. Based on the collected data, we offer insights into how moral exercises can foster a responsible AI culture within organizations, and suggest directions for future research. 

---
# What Lives? A meta-analysis of diverse opinions on the definition of life 

**Authors**: Reed Bender, Karina Kofman, Blaise Agüera y Arcas, Michael Levin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15849)  

**Abstract**: The question of "what is life?" has challenged scientists and philosophers for centuries, producing an array of definitions that reflect both the mystery of its emergence and the diversity of disciplinary perspectives brought to bear on the question. Despite significant progress in our understanding of biological systems, psychology, computation, and information theory, no single definition for life has yet achieved universal acceptance. This challenge becomes increasingly urgent as advances in synthetic biology, artificial intelligence, and astrobiology challenge our traditional conceptions of what it means to be alive. We undertook a methodological approach that leverages large language models (LLMs) to analyze a set of definitions of life provided by a curated set of cross-disciplinary experts. We used a novel pairwise correlation analysis to map the definitions into distinct feature vectors, followed by agglomerative clustering, intra-cluster semantic analysis, and t-SNE projection to reveal underlying conceptual archetypes. This methodology revealed a continuous landscape of the themes relating to the definition of life, suggesting that what has historically been approached as a binary taxonomic problem should be instead conceived as differentiated perspectives within a unified conceptual latent space. We offer a new methodological bridge between reductionist and holistic approaches to fundamental questions in science and philosophy, demonstrating how computational semantic analysis can reveal conceptual patterns across disciplinary boundaries, and opening similar pathways for addressing other contested definitional territories across the sciences. 

---
# TDFormer: A Top-Down Attention-Controlled Spiking Transformer 

**Authors**: Zizheng Zhu, Yingchao Yu, Zeqi Zheng, Zhaofei Yu, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15840)  

**Abstract**: Traditional spiking neural networks (SNNs) can be viewed as a combination of multiple subnetworks with each running for one time step, where the parameters are shared, and the membrane potential serves as the only information link between them. However, the implicit nature of the membrane potential limits its ability to effectively represent temporal information. As a result, each time step cannot fully leverage information from previous time steps, seriously limiting the model's performance. Inspired by the top-down mechanism in the brain, we introduce TDFormer, a novel model with a top-down feedback structure that functions hierarchically and leverages high-order representations from earlier time steps to modulate the processing of low-order information at later stages. The feedback structure plays a role from two perspectives: 1) During forward propagation, our model increases the mutual information across time steps, indicating that richer temporal information is being transmitted and integrated in different time steps. 2) During backward propagation, we theoretically prove that the feedback structure alleviates the problem of vanishing gradients along the time dimension. We find that these mechanisms together significantly and consistently improve the model performance on multiple datasets. In particular, our model achieves state-of-the-art performance on ImageNet with an accuracy of 86.83%. 

---
# Quantum-Evolutionary Neural Networks for Multi-Agent Federated Learning 

**Authors**: Aarav Lala, Kalyan Cherukuri  

**Link**: [PDF](https://arxiv.org/pdf/2505.15836)  

**Abstract**: As artificial intelligence continues to drive innovation in complex, decentralized environments, the need for scalable, adaptive, and privacy-preserving decision-making systems has become critical. This paper introduces a novel framework combining quantum-inspired neural networks with evolutionary algorithms to optimize real-time decision-making in multi-agent systems (MAS). The proposed Quantum-Evolutionary Neural Network (QE-NN) leverages quantum computing principles -- such as quantum superposition and entanglement -- to enhance learning speed and decision accuracy, while integrating evolutionary optimization to continually refine agent behaviors in dynamic, uncertain environments. By utilizing federated learning, QE-NN ensures privacy preservation, enabling decentralized agents to collaborate without sharing sensitive data. The framework is designed to allow agents to adapt in real-time to their environments, optimizing decision-making processes for applications in areas such as autonomous systems, smart cities, and healthcare. This research represents a breakthrough in merging quantum computing, evolutionary optimization, and privacy-preserving techniques to solve complex problems in multi-agent decision-making systems, pushing the boundaries of AI in real-world, privacy-sensitive applications. 

---
# Transforming Decoder-Only Transformers for Accurate WiFi-Telemetry Based Indoor Localization 

**Authors**: Nayan Sanjay Bhatia, Katia Obraczka  

**Link**: [PDF](https://arxiv.org/pdf/2505.15835)  

**Abstract**: Wireless Fidelity (WiFi) based indoor positioning is a widely researched area for determining the position of devices within a wireless network. Accurate indoor location has numerous applications, such as asset tracking and indoor navigation. Despite advances in WiFi localization techniques -- in particular approaches that leverage WiFi telemetry -- their adoption in practice remains limited due to several factors including environmental changes that cause signal fading, multipath effects, interference, which, in turn, impact positioning accuracy. In addition, telemetry data differs depending on the WiFi device vendor, offering distinct features and formats; use case requirements can also vary widely. Currently, there is no unified model to handle all these variations effectively. In this paper, we present WiFiGPT, a Generative Pretrained Transformer (GPT) based system that is able to handle these variations while achieving high localization accuracy. Our experiments with WiFiGPT demonstrate that GPTs, in particular Large Language Models (LLMs), can effectively capture subtle spatial patterns in noisy wireless telemetry, making them reliable regressors. Compared to existing state-of-the-art methods, our method matches and often surpasses conventional approaches for multiple types of telemetry. Achieving sub-meter accuracy for RSSI and FTM and centimeter-level precision for CSI demonstrates the potential of LLM-based localisation to outperform specialized techniques, all without handcrafted signal processing or calibration. 

---
# MPPFND: A Dataset and Analysis of Detecting Fake News with Multi-Platform Propagation 

**Authors**: Congyuan Zhao, Lingwei Wei, Ziming Qin, Wei Zhou, Yunya Song, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15834)  

**Abstract**: Fake news spreads widely on social media, leading to numerous negative effects. Most existing detection algorithms focus on analyzing news content and social context to detect fake news. However, these approaches typically detect fake news based on specific platforms, ignoring differences in propagation characteristics across platforms. In this paper, we introduce the MPPFND dataset, which captures propagation structures across multiple platforms. We also describe the commenting and propagation characteristics of different platforms to show that their social contexts have distinct features. We propose a multi-platform fake news detection model (APSL) that uses graph neural networks to extract social context features from various platforms. Experiments show that accounting for cross-platform propagation differences improves fake news detection performance. 

---
# From Hand-Crafted Metrics to Evolved Training-Free Performance Predictors for Neural Architecture Search via Genetic Programming 

**Authors**: Quan Minh Phan, Ngoc Hoang Luong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15832)  

**Abstract**: Estimating the network performance using zero-cost (ZC) metrics has proven both its efficiency and efficacy in Neural Architecture Search (NAS). However, a notable limitation of most ZC proxies is their inconsistency, as reflected by the substantial variation in their performance across different problems. Furthermore, the design of existing ZC metrics is manual, involving a time-consuming trial-and-error process that requires substantial domain expertise. These challenges raise two critical questions: (1) Can we automate the design of ZC metrics? and (2) Can we utilize the existing hand-crafted ZC metrics to synthesize a more generalizable one? In this study, we propose a framework based on Symbolic Regression via Genetic Programming to automate the design of ZC metrics. Our framework is not only highly extensible but also capable of quickly producing a ZC metric with a strong positive rank correlation to true network performance across diverse NAS search spaces and tasks. Extensive experiments on 13 problems from NAS-Bench-Suite-Zero demonstrate that our automatically generated proxies consistently outperform hand-crafted alternatives. Using our evolved proxy metric as the search objective in an evolutionary algorithm, we could identify network architectures with competitive performance within 15 minutes using a single consumer GPU. 

---
# Generative AI-Aided QoE Maximization for RIS-Assisted Digital Twin Interaction 

**Authors**: Jiayuan Chen, Yuxiang Li, Changyan Yi, Shimin Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15828)  

**Abstract**: In this paper, we investigate a quality of experience (QoE)-aware resource allocation problem for reconfigurable intelligent surface (RIS)-assisted digital twin (DT) interaction with uncertain evolution. In the considered system, mobile users are expected to interact with a DT model maintained on a DT server that is deployed on a base station, via effective uplink and downlink channels assisted by an RIS. Our goal is to maximize the sum of all mobile users' joint subjective and objective QoE in DT interactions across various DT scenes, by jointly optimizing phase shift matrix, receive/transmit beamforming matrix, rendering resolution configuration and computing resource allocation. While solving this problem is challenging mainly due to the uncertain evolution of the DT model, which leads to multiple scene-specific problems, and require us to constantly re-solve each of them whenever DT model evolves.
To this end, leveraging the dynamic optimization capabilities of decision transformers and the generalization strengths of generative artificial intelligence (GAI), we propose a novel GAI-aided approach, called the prompt-guided decision transformer integrated with zero-forcing optimization (PG-ZFO). Simulations are conducted to evaluate the proposed PG-ZFO, demonstrating its effectiveness and superiority over counterparts. 

---
# Multilinear subspace learning for person re-identification based fusion of high order tensor features 

**Authors**: Ammar Chouchane, Mohcene Bessaoudi, Hamza Kheddar, Abdelmalik Ouamane, Tiago Vieira, Mahmoud Hassaballah  

**Link**: [PDF](https://arxiv.org/pdf/2505.15825)  

**Abstract**: Video surveillance image analysis and processing is a challenging field in computer vision, with one of its most difficult tasks being Person Re-Identification (PRe-ID). PRe-ID aims to identify and track target individuals who have already been detected in a network of cameras, using a robust description of their pedestrian images. The success of recent research in person PRe-ID is largely due to effective feature extraction and representation, as well as the powerful learning of these features to reliably discriminate between pedestrian images. To this end, two powerful features, Convolutional Neural Networks (CNN) and Local Maximal Occurrence (LOMO), are modeled on multidimensional data using the proposed method, High-Dimensional Feature Fusion (HDFF). Specifically, a new tensor fusion scheme is introduced to leverage and combine these two types of features in a single tensor, even though their dimensions are not identical. To enhance the system's accuracy, we employ Tensor Cross-View Quadratic Analysis (TXQDA) for multilinear subspace learning, followed by cosine similarity for matching. TXQDA efficiently facilitates learning while reducing the high dimensionality inherent in high-order tensor data. The effectiveness of our approach is verified through experiments on three widely-used PRe-ID datasets: VIPeR, GRID, and PRID450S. Extensive experiments demonstrate that our approach outperforms recent state-of-the-art methods. 

---
# A Novel Compound AI Model for 6G Networks in 3D Continuum 

**Authors**: Milos Gravara, Andrija Stanisic, Stefan Nastic  

**Link**: [PDF](https://arxiv.org/pdf/2505.15821)  

**Abstract**: The 3D continuum presents a complex environment that spans the terrestrial, aerial and space domains, with 6Gnetworks serving as a key enabling technology. Current AI approaches for network management rely on monolithic models that fail to capture cross-domain interactions, lack adaptability,and demand prohibitive computational resources. This paper presents a formal model of Compound AI systems, introducing a novel tripartite framework that decomposes complex tasks into specialized, interoperable modules. The proposed modular architecture provides essential capabilities to address the unique challenges of 6G networks in the 3D continuum, where heterogeneous components require coordinated, yet distributed, intelligence. This approach introduces a fundamental trade-off between model and system performance, which must be carefully addressed. Furthermore, we identify key challenges faced by Compound AI systems within 6G networks operating in the 3D continuum, including cross-domain resource orchestration, adaptation to dynamic topologies, and the maintenance of consistent AI service quality across heterogeneous environments. 

---
# Common Data Format (CDF): A Standardized Format for Match-Data in Football (Soccer) 

**Authors**: Gabriel Anzer, Kilian Arnsmeyer, Pascal Bauer, Joris Bekkers, Ulf Brefeld, Jesse Davis, Nicolas Evans, Matthias Kempe, Samuel J Robertson, Joshua Wyatt Smith, Jan Van Haaren  

**Link**: [PDF](https://arxiv.org/pdf/2505.15820)  

**Abstract**: During football matches, a variety of different parties (e.g., companies) each collect (possibly overlapping) data about the match ranging from basic information (e.g., starting players) to detailed positional data. This data is provided to clubs, federations, and other organizations who are increasingly interested in leveraging this data to inform their decision making. Unfortunately, analyzing such data pose significant barriers because each provider may (1) collect different data, (2) use different specifications even within the same category of data, (3) represent the data differently, and (4) delivers the data in a different manner (e.g., file format, protocol). Consequently, working with these data requires a significant investment of time and money. The goal of this work is to propose a uniform and standardized format for football data called the Common Data Format (CDF). The CDF specifies a minimal schema for five types of match data: match sheet data, video footage, event data, tracking data, and match meta data. It aims to ensure that the provided data is clear, sufficiently contextualized (e.g., its provenance is clear), and complete such that it enables common downstream analysis tasks. Concretely, this paper will detail the technical specifications of the CDF, the representational choices that were made to help ensure the clarity of the provided data, and a concrete approach for delivering data in the CDF. 

---
# UltraEdit: Training-, Subject-, and Memory-Free Lifelong Editing in Large Language Models 

**Authors**: Xiaojie Gu, Guangxu Chen, Jungang Li, Jia-Chen Gu, Xuming Hu, Kai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14679)  

**Abstract**: Lifelong learning enables large language models (LLMs) to adapt to evolving information by continually updating their internal knowledge. An ideal system should support efficient, wide-ranging updates while preserving existing capabilities and ensuring reliable deployment. Model editing stands out as a promising solution for this goal, offering a focused and efficient way to revise a model's internal knowledge. Although recent paradigms have made notable progress, they often struggle to meet the demands of practical lifelong adaptation at scale. To bridge this gap, we propose ULTRAEDIT-a fundamentally new editing solution that is training-, subject- and memory-free, making it particularly well-suited for ultra-scalable, real-world lifelong model editing. ULTRAEDIT performs editing through a self-contained process that relies solely on lightweight linear algebra operations to compute parameter shifts, enabling fast and consistent parameter modifications with minimal overhead. To improve scalability in lifelong settings, ULTRAEDIT employs a lifelong normalization strategy that continuously updates feature statistics across turns, allowing it to adapt to distributional shifts and maintain consistency over time. ULTRAEDIT achieves editing speeds over 7x faster than the previous state-of-the-art method-which was also the fastest known approach-while consuming less than 1/3 the VRAM, making it the only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU. Furthermore, we construct ULTRAEDITBENCH-the largest dataset in the field to date, with over 2M editing pairs-and demonstrate that our method supports up to 1M edits while maintaining high accuracy. Comprehensive experiments on four datasets and six models show that ULTRAEDIT consistently achieves superior performance across diverse model editing scenarios. Our code is available at: this https URL. 

---
# Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning 

**Authors**: Xuetao Ma, Wenbin Jiang, Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15401)  

**Abstract**: In-context learning (ICL) can significantly enhance the complex reasoning capabilities of large language models (LLMs), with the key lying in the selection and ordering of demonstration examples. Previous methods typically relied on simple features to measure the relevance between examples. We argue that these features are not sufficient to reflect the intrinsic connections between examples. In this study, we propose a curriculum ICL strategy guided by problem-solving logic. We select demonstration examples by analyzing the problem-solving logic and order them based on curriculum learning. Specifically, we constructed a problem-solving logic instruction set based on the BREAK dataset and fine-tuned a language model to analyze the problem-solving logic of examples. Subsequently, we selected appropriate demonstration examples based on problem-solving logic and assessed their difficulty according to the number of problem-solving steps. In accordance with the principles of curriculum learning, we ordered the examples from easy to hard to serve as contextual prompts. Experimental results on multiple benchmarks indicate that our method outperforms previous ICL approaches in terms of performance and efficiency, effectively enhancing the complex reasoning capabilities of LLMs. Our project will be publicly available subsequently. 

---
