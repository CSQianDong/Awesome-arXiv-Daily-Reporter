# ToolCUA: Towards Optimal GUI-Tool Path Orchestration for Computer Use Agents 

**Authors**: Xuhao Hu, Xi Zhang, Haiyang Xu, Kyle Qiao, Jingyi Yang, Xuanjing Huang, Jing Shao, Ming Yan, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.12481)  

**Abstract**: Computer Use Agents (CUAs) can act through both atomic GUI actions, such as click and type, and high-level tool calls, such as API-based file operations, but this hybrid action space often leaves them uncertain about when to continue with GUI actions or switch to tools, leading to suboptimal execution paths. This difficulty stems from the scarcity of high-quality interleaved GUI-Tool trajectories, the cost and brittleness of collecting real tool trajectories, and the lack of trajectory-level supervision for GUI-Tool path selection. In this paper, we propose ToolCUA, an end-to-end agent designed to learn optimal GUI-Tool path selection through a staged training paradigm. We first introduce an Interleaved GUI-Tool Trajectory Scaling Pipeline that repurposes abundant static GUI trajectories and synthesizes a grounded tool library, enabling diverse GUI-Tool trajectories without manual engineering or real tool-trajectory collection. We then perform Tool-Bootstrapped GUI RFT, combining warmup SFT with single-turn RL to improve decisions at critical GUI-Tool switching points. Finally, we optimize ToolCUA with Online Agentic RL in a high-fidelity GUI-Tool environment, guided by a Tool-Efficient Path Reward that encourages appropriate tool use and shorter execution paths. Experiments on OSWorld-MCP show that ToolCUA achieves 46.85% accuracy, a relative improvement of approximately 66% over the baseline, establishing a new state of the art among models of comparable scale. It also improves by 3.9% over GUI-only settings, demonstrating effective GUI-Tool orchestration. The results further suggest that training in a hybrid action space is a promising paradigm for real-world digital agents. Open-sourced here: this https URL 

---
# Reward Hacking in Rubric-Based Reinforcement Learning 

**Authors**: Anas Mahmoud, MohammadHossein Rezaei, Zihao Wang, Anisha Gunjal, Bing Liu, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2605.12474)  

**Abstract**: Reinforcement learning with verifiable rewards has enabled strong post-training gains in domains such as math and coding, though many open-ended settings rely on rubric-based rewards. We study reward hacking in rubric-based RL, where a policy is optimized against a training verifier but evaluated against a cross-family panel of three frontier judges, reducing dependence on any single evaluator. Our framework separates two sources of divergence: verifier failure, where the training verifier credits rubric criteria that reference verifiers reject, and rubric-design limitations, where even strong rubric-based verifiers favor responses that rubric-free judges rate worse overall. Across medical and science domains, weak verifiers produce large proxy-reward gains that do not transfer to the reference verifiers; exploitation grows over training and concentrates in recurring failures such as partial satisfaction of compound criteria, treating implicit content as explicit, and imprecise topical matching. Stronger verifiers substantially reduce, but do not eliminate, verifier exploitation. We also introduce a self-internalization gap, a verifier-free diagnostic based on policy log-probabilities, which tracks reference-verifier quality, detecting when the policy trained using the weak verifier stops improving. Finally, in our setting, stronger verification does not prevent reward hacking when the rubric leaves important failure modes unspecified: rubric-based verifiers prefer the RL checkpoint, while rubric-free judges prefer the base model. These disagreements coincide with gains concentrated in completeness and presence-based criteria, alongside declines in factual correctness, conciseness, relevance, and overall quality. Together, these results suggest that stronger verification reduces reward hacking, but does not by itself ensure that rubric gains correspond to broader quality gains. 

---
# Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs 

**Authors**: Jose E. Aguilar Escamilla, Lingdong Zhou, Xiangqi Zhu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12462)  

**Abstract**: Extreme weather and volatile wholesale electricity markets expose residential consumers to catastrophic financial risks, yet demand response at the distribution level remains an underutilized tool for grid flexibility and energy affordability. While a demand-response program can shield consumers by issuing financial credits during high-price periods, optimizing this sequential decision-making process presents a unique challenge for reinforcement learning despite the plentiful offline historical smart meter and wholesale pricing data available publicly. Offline historical data fails to capture the dynamic, interactive feedback loop between an electric utility's pricing signals and customer acceptance and adaptation to a demand-response program. To address this, we introduce DR-Gym, an open-source, online Gymnasium-compatible environment designed to train and evaluate demand-response from the electric utility's perspective. Unlike existing device-level energy simulators, our environment focuses on the market-level electric utility setting and provides a rich observational space relevant to the electric utility. The simulator additionally features a regime-switching wholesale price model calibrated to real-world extreme events, alongside physics-based building demand profiles. For our learning signal, we use a configurable, multi-objective reward function for specifying diverse learning objectives. We demonstrate through baseline strategies and data snapshots the capability of our simulator to create realistic and learnable environments. 

---
# CAAFC: Chronological Actionable Automated Fact-Checker for misinformation / non-factual hallucination detection and correction 

**Authors**: Islam Eldifrawi, Shengrui Wang, Amine Trabelsi  

**Link**: [PDF](https://arxiv.org/pdf/2605.12436)  

**Abstract**: With the vast amount of content uploaded every hour, along with the AI generated content that can include hallucinations, Automated Fact-Checking (AFC) has become increasingly vital, as it is infeasible for human fact-checkers to manually verify the sheer volume of information generated online. Professional fact-checkers have identified several gaps in existing AFC systems, noting a misalignment between how these systems operate and how fact-checking is performed in practice. In this paper, we introduce CAAFC (Chronological Actionable Automated Fact-Checker), a frame-work designed to bridge these gaps. It surpasses SOTA AFC and hallucination detection systems across multiple benchmark datasets. CAAFC operates on claims, conversations, and dialogues, enabling it not only to detect factual errors and hallucinations, but also to correct them by providing actionable justifications supported by primary information sources. Furthermore, CAAFC can update evidence and knowledge bases by incorporating recent and contextual information when necessary, thereby enhancing the reliability of fact verification. 

---
# Formalize, Don't Optimize: The Heuristic Trap in LLM-Generated Combinatorial Solvers 

**Authors**: Haoyu Wang, Yuliang Song, Tao Li, Zhiwei Deng, Yaqing Wang, Deepak Ramachandran, Eldan Cohen, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2605.12421)  

**Abstract**: Large Language Models (LLMs) struggle to solve complex combinatorial problems through direct reasoning, so recent neuro-symbolic systems increasingly use them to synthesize executable solvers. A central design question is how the LLM should represent the solver, and whether it should also attempt to optimize search. We introduce CP-SynC-XL, a benchmark of 100 combinatorial problems (4,577 instances), and evaluate three solver-construction paradigms: native algorithmic search (Python), constraint modeling through a Python solver API (Python + OR-Tools), and declarative constraint modeling (MiniZinc + OR-Tools). We find a consistent representational divergence: Python + OR-Tools attains the highest correctness across LLMs, while MiniZinc + OR-Tools has lower absolute coverage despite using the same OR-Tools back-end. Native Python is the most likely to return a schema-valid solution that fails verification, whereas solver-backed paths preserve higher conditional fidelity. On the heuristic axis, prompting for search optimization yields only small median speed-ups (1.03-1.12x) and a strongly bimodal effect: many instances slow down, and correctness drops sharply on a long tail of problems. A paired code-level audit traces these regressions to a recurring heuristic trap. Under an efficiency-oriented prompt, the LLM may replace complete search with local approximations (Python), inject unverified bounds (Python + OR-Tools), or add redundant declarative machinery that overwhelms or over-constrains the model (MiniZinc + OR-Tools). These findings support a conservative design principle for LLM-generated combinatorial solvers: use the LLM primarily to formalize variables, constraints, and objectives for verified solvers, and separately check any LLM-authored search optimization before use. 

---
# Semantic Reward Collapse and the Preservation of Epistemic Integrity in Adaptive AI Systems 

**Authors**: William Parris  

**Link**: [PDF](https://arxiv.org/pdf/2605.12406)  

**Abstract**: Recent advances in reinforcement learning from human feedback (RLHF) and preference optimization have substantially improved the usability, coherence, and safety of large language models. However, recurring behaviors such as performative certainty, hallucinated continuity, calibration drift, sycophancy, and suppression of visible uncertainty suggest unresolved structural issues within scalarized preference optimization systems.
We propose Semantic Reward Collapse (SRC): the compression of semantically distinct forms of evaluative dissatisfaction into generalized optimization signals. Under SRC, categories such as factual incorrectness, uncertainty disclosure, formatting dissatisfaction, latency, and social preference may become entangled within a shared reward topology despite representing fundamentally different epistemic classes.
We argue that adaptive reasoning systems operating under generalized evaluative pressure may drift toward suppression of visible epistemic failure rather than preservation of calibrated uncertainty integrity. These behaviors are framed strictly as optimization consequences rather than evidence of deception or anthropomorphic agency.
Drawing on institutional proxy collapse, metric gaming, software reliability engineering, and human learning theory, we propose that uncertainty disclosure and escalation behavior should be treated as protected epistemic conduct rather than globally penalized task incompletion.
Finally, we introduce Constitutional Reward Stratification (CRS), a domain-aware reward framework intended to preserve differentiated epistemic attribution within adaptive learning systems. We present CRS not as a validated solution, but as a testable governance-oriented research direction requiring further empirical investigation. 

---
# ProfiliTable: Profiling-Driven Tabular Data Processing via Agentic Workflows 

**Authors**: Wei Liu, Yang Gu, Xi Yan, Zihan Nan, Beicheng Xu, Keyao Ding, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12376)  

**Abstract**: Table processing-including cleaning, transformation, augmentation, and matching-is a foundational yet error-prone stage in real-world data pipelines. While recent LLM-based approaches show promise for automating such tasks, they often struggle in practice due to ambiguous instructions, complex task structures, and the lack of structured feedback, resulting in syntactically correct but semantically flawed code. To address these challenges, we propose ProfiliTable, an autonomous multi-agent framework centered on dynamic profiling, which constructs and iteratively refines a unified execution context through interactive exploration, knowledge-augmented synthesis, and feedback-driven refinement. ProfiliTable integrates (i) a Profiler that performs ReAct-style data exploration to build semantic understanding, (ii) a Generator that retrieves curated operators to synthesize task-aware code, and (iii) an Evaluator-Summarizer loop that injects execution scores and diagnostic insights to enable closed-loop refinement. Extensive experiments on a diverse benchmark covering 18 tabular task types demonstrate that ProfiliTable consistently outperforms strong baselines, particularly in complex multi-step scenarios. These results highlight the critical role of dynamic profiling in reliably translating ambiguous user intents into robust and governance-compliant table transformations. 

---
# Classifier Context Rot: Monitor Performance Degrades with Context Length 

**Authors**: Sam Martin, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2605.12366)  

**Abstract**: Monitoring coding agents for dangerous behavior using language models requires classifying transcripts that often exceed 500K tokens, but prior agent monitoring benchmarks rarely contain transcripts longer than 100K tokens. We show that when used as classifiers, current frontier models fail to notice dangerous actions more often in longer transcripts. In particular, on a dataset that requires identifying when a coding agent takes a subtly dangerous action, Opus 4.6, GPT 5.4, and Gemini 3.1 miss these actions $2\times$ to $30\times$ more often when they occur after 800K tokens of benign activity than when they occur on their own. We also show that these weaknesses can be partially mitigated with prompting techniques such as periodic reminders throughout the transcript and may be mitigated further with better post-training. Monitor evaluations that do not consider long-context degradation are likely overestimating monitor performance. 

---
# $δ$-mem: Efficient Online Memory for Large Language Models 

**Authors**: Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2605.12357)  

**Abstract**: Large language models increasingly need to accumulate and reuse historical information in long-term assistants and agent systems. Simply expanding the context window is costly and often fails to ensure effective context utilization. We propose $\delta$-mem, a lightweight memory mechanism that augments a frozen full-attention backbone with a compact online state of associative memory. $\delta$-mem compresses past information into a fixed-size state matrix updated by delta-rule learning, and uses its readout to generate low-rank corrections to the backbone's attention computation during generation. With only an $8\times8$ online memory state, $\delta$-mem improves the average score to $1.10\times$ that of the frozen backbone and $1.15\times$ that of the strongest non-$\delta$-mem memory baseline. It achieves larger gains on memory-heavy benchmarks, reaching $1.31\times$ on MemoryAgentBench and $1.20\times$ on LoCoMo, while largely preserving general capabilities. These results show that effective memory can be realized through a compact online state directly coupled with attention computation, without full fine-tuning, backbone replacement, or explicit context extension. 

---
# Reinforcing VLAs in Task-Agnostic World Models 

**Authors**: Yucen Wang, Rui Yu, Fengming Zhang, Junjie Lu, Xinyao Qin, Tianxiang Zhang, Kaixin Wang, Li Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12334)  

**Abstract**: Post-training Vision-Language-Action (VLA) models via reinforcement learning (RL) in learned world models has emerged as an effective strategy to adapt to new tasks without costly real-world interactions. However, while using imagined trajectories reduces the sample complexity of policy training, existing methods still heavily rely on task-specific data to fine-tune both the world and reward models, fundamentally limiting their scalability to unseen tasks. To overcome this, we argue that world and reward models should capture transferable physical priors that enable zero-shot inference. We propose RAW-Dream (Reinforcing VLAs in task-Agnostic World Dreams), a new paradigm that completely disentangles world model learning from downstream task dependencies. RAW-Dream utilizes a world model pre-trained on diverse task-free behaviors for predicting future rollouts, and an off-the-shelf Vision-Language Model (VLM) for reward generation. Because both components are task-agnostic, VLAs can be readily finetuned for any new task entirely within this zero-shot imagination. Furthermore, to mitigate world model hallucinations, we introduce a dual-noise verification mechanism to filter out unreliable rollouts. Extensive experiments across simulation and real-world settings demonstrate consistent performance gains, proving that generalized physical priors can effectively substitute for costly task-dependent data, offering a highly scalable roadmap for VLA adaptation. 

---
# Towards Automated Air Traffic Safety Assessment Around Non-Towered Airports Using Large Language Models 

**Authors**: Torsten Darrell, Mahyar Ghazanfari, Jordan Kam, Alexandre Bayen, Amin Tabrizian, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.12332)  

**Abstract**: We investigate frameworks for post-flight safety analysis at non-towered airports using large language models (LLMs). Non-towered airports rely on the Common Traffic Advisory Frequency (CTAF) for air traffic coordination and experience frequent near mid-air collisions due to the pilot self-announcement communication protocol. We propose a general vision-language model (VLM) approach to analyze the transcribed CTAF radio communications in natural language, METeorological Aerodrome Report (METAR) weather data, Automatic Dependent Surveillance-Broadcast (ADS-B) flight trajectories, and Visual Flight Rules sectional charts of the airfield. We provide a preliminary study at Half Moon Bay Airport, with a qualitative real world case study and a quantitative evaluation using a new synthetic dataset of communications and weather modalities. We qualitatively evaluate our framework on real flight data using Gemini 2.5 Pro, demonstrating accurate identification of a right-of-way violation. The synthetic dataset is derived from real examples and includes a 12-category hazard taxonomy, and is used to benchmark three open-source (Qwen 2.5-7B, Mistral-7B, Gemma-2-9B) and three closed-source (GPT-4o, GPT-5.4, Claude Sonnet 4.6) LLM models on the subset of inputs related to CTAF and METAR. Even limited to CTAF and METAR inputs and open source LLMs, instances of our framework typically achieve a macro F1 score above 0.85 on a binary nominal/danger classification task. Future work includes a quantitative evaluation across all modalities and a larger number of real world examples. Taken together, our results suggest that VLM analysis of safety at non-towered airports may be a valuable future capability. 

---
# LISA: Cognitive Arbitration for Signal-Free Autonomous Intersection Management 

**Authors**: Abderrahmane Lakas, Mohamed Amine Ferrag, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2605.12321)  

**Abstract**: Large language models (LLMs) show strong potential for Intelligent Transportation Systems (ITS), particularly in tasks requiring situational reasoning and multi-agent coordination. These capabilities make them well suited for cooperative driving, where rule-based approaches struggle in complex and dynamic traffic environments. Intersection management remains especially challenging due to conflicting right-of-way demands, heterogeneous vehicle priorities, and vehicle-specific kinematic constraints that must be resolved in real time. However, existing approaches typically use LLMs as auxiliary components on top of signal-based systems rather than as primary decision-makers. Signal controllers remain vehicle-agnostic, reservation-based methods lack intent awareness, and recent LLM-based systems still depend on signal infrastructure. In addition, LLM inference latency limits their use in sub-second control settings.
We propose LISA (LLM-Based Intent-Driven Speed Advisory), a signal-free cognitive arbitration framework for autonomous intersection management. LISA uses an LLM to reason over declared vehicle intents, incorporating priority classes, queue pressure, and energy preferences. We evaluate LISA against fixed-cycle control, SCATS, AIM, and GLOSA across varying traffic loads. Results show that LISA reduces mean control delay by up to 89.1% and maintains Level of Service C while all non-LLM baselines degrade to Level of Service F. Under near-saturated demand, LISA reduces mean waiting time by 93% and peak queue length by 60.6% relative to fixed-cycle control. It also lowers fuel consumption by up to 48.8% and achieves 86.2% intent satisfaction, compared to 61.2% for the best non-LLM method. These results demonstrate that LLM-based reasoning can enable real-time, signal-free intersection management. 

---
# Executable Agentic Memory for GUI Agent 

**Authors**: Zerui Qin, Sheng Yue, Xingyuan Hua, Yongjian Fu, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.12294)  

**Abstract**: Modern GUI agents typically rely on a model-centric and step-wise interaction paradigm, where LLMs must re-interpret the UI and re-decide actions at every screen, which is fragile in long-horizon tasks. In this paper, we propose Executable Agentic Memory (EAM), a structured Knowledge Graph (KG) that shifts GUI planning from free-form generation to a robust retrieval-and-execution process. Our approach includes a sample-efficient memory construction pipeline using state-aware DFS and action-group mining to compress multi-step routines. To ensure efficient planning, we introduce a value-guided graph search where a lightweight Q-function model steers Monte Carlo Tree Search (MCTS) over the KG. We theoretically establish bias-consistency for the Q-model and derive sample complexity bounds for path recovery. Empirically, EAM outperforms state-of-the-art baselines like UI-TARS-7B by up to $19.6\%$ on AndroidWorld, while reducing token costs $6\times$ relative to GPT-4o. With a $2.8$s average latency, EAM enables reliable, quick, and long-horizon GUI automation. 

---
# NARA: Anchor-Conditioned Relation-Aware Contextualization of Heterogeneous Geoentities 

**Authors**: Jina Kim, Gengchen Mai, Lingyi Zhao, Khurram Shafique, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12276)  

**Abstract**: Geospatial foundation models have primarily focused on raster data such as satellite imagery, where self-supervised learning has been widely studied. Vector geospatial data instead represent the world as discrete geoentities with explicit geometry, semantics, and structured spatial relations, including metric proximity and topological relationships. These relations jointly determine how entities interact within space, yet existing representation learning methods remain fragmented, often restricted to specific geometry types or partial spatial relations, limiting their ability to capture unified spatial context across heterogeneous geoentities. We propose NARA (Neural Anchor-conditioned Relation-Aware representation learning), a self-supervised framework for vector geoentities. NARA learns context-dependent representations by jointly modeling semantics, geometry, and spatial relations within a unified framework and captures relational spatial structure beyond proximity alone, enabling rich contextualized representations across heterogeneous geoentities of points, polylines, and polygons. Evaluation on building function classification, traffic speed prediction, and next point-of-interest recommendation shows consistent improvements over prior methods, highlighting the benefit of unified relational modeling for vector geospatial data. 

---
# How Useful Is Cross-Domain Generalization for Training LLM Monitors? 

**Authors**: Sam Martin, Fabien Roger  

**Link**: [PDF](https://arxiv.org/pdf/2605.12265)  

**Abstract**: Using prompted language models as classifiers enables classification in domains with limited training data, but misses some of the robustness and performance benefits that fine-tuning can bring. We study whether training on multiple classification tasks, each with its own prompt, improves performance on new domains with new classification prompts. We show that such training partially generalizes to adjacent domains, improving classification performance on tasks that are unseen during training. However, we identify specific edge cases where the fine-tuned models fail to follow prompts, such as when the classification prompt changes completely while the data domain remains the same as during training. We show that classification training can be mixed with general instruction following training, and that (when done well) such training keeps the benefits of classification training and mitigates its generalization failures. Surprisingly, we see that this no-thinking supervised classification training can generalize to with-thinking classification and summarization, suggesting that no-thinking classification training might be instrumentally useful in building other kinds of classifiers and monitoring systems. 

---
# Missingness-MDPs: Bridging the Theory of Missing Data and POMDPs 

**Authors**: Joshua Wendland, Markel Zubia, Roman Andriushchenko, Maris F. L. Galesloot, Milan Ceska, Henrik von Kleist, Thiago D. Simao, Maximilian Weininger, Nils Jansen  

**Link**: [PDF](https://arxiv.org/pdf/2605.12262)  

**Abstract**: We introduce missingness-MDPs (miss-MDPs), a novel subclass of partially observable Markov decision processes (POMDPs) that incorporates the theory of missing data. A miss-MDP is a POMDP whose observation function is a missingness function, specifying the probability that individual state features are missing (i.e., unobserved) at a time step. The literature distinguishes three canonical missingness types: missing (1) completely at random (MCAR), (2) at random (MAR), and (3) not at random (MNAR). Our planning problem is to compute near-optimal policies for a miss-MDP with an unknown missingness function, given a dataset of action-observation trajectories. Achieving such optimality guarantees for policies requires learning the missingness function from data, which is infeasible for general POMDPs. To overcome this challenge, we exploit the structural properties of different missingness types to derive probably approximately correct (PAC) algorithms for learning the missingness function. These algorithms yield an approximate but fully specified miss-MDP that we solve using off-the-shelf planning methods. We prove that, with high probability, the resulting policies are epsilon-optimal in the true miss-MDP. Empirical results confirm the theory and demonstrate superior performance of our approach over two model-free POMDP methods. 

---
# Why Conclusions Diverge from the Same Observations: Formalizing World-Model Non-Identifiability via an Inference 

**Authors**: Toru Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2605.12255)  

**Abstract**: When people share the same documents and observations yet reach different conclusions, the disagreement often shifts into a judgment that the other party is cognitively defective, irrational, or acting in bad faith. This paper argues that such divergence is better described as a form of non-identifiability inherent in inference and learning, rather than as a defect of the other party. We organize the phenomenon into two levels: (i) $\theta$-level non-identifiability, where conclusions diverge under the same world model $W$ because inference settings differ; and (ii) $W$-level non-identifiability, where repeated use of an inference setting $\theta$ biases data exposure and update rules, causing the learned world model $W$ itself to diverge. We introduce an inference profile $\theta = (R, E, S, D)$, consisting of Reference, Exploration, Stabilization, and Horizon, and show how outputs can split even for the same observation $o$ and the same $W$. We further explain why disagreements tend to project onto a small number of bases -- abstract versus concrete, externalizability, and order versus freedom -- as a consequence of general constraints on learning systems: computational, observational, and coordination constraints. Finally, we relate the framework to deep representation learning, including representation hierarchy, latent-state estimation, and regularization-exploration trade-offs, and illustrate the framework through a case study on AI regulation debates. 

---
# No Action Without a NOD: A Heterogeneous Multi-Agent Architecture for Reliable Service Agents 

**Authors**: Zixu Yang, Hang Zheng, Nan Jiang, Zhiyang Tang, Situo Zhang, Xiaobao Wu, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12240)  

**Abstract**: Large language model (LLM) agents have increasingly advanced service applications, such as booking flight tickets. However, these service agents suffer from unreliability in long-horizon tasks, as they often produce policy violations, tool hallucinations, and misaligned actions, which greatly impedes their real-world deployment. To address these challenges, we propose NOD (Navigator-Operator-Director), a heterogeneous multi-agent architecture for service agents. Instead of maintaining task state implicitly in dialogue context as in prior work, we externalize a structured Global State to enable explicit task state tracking and consistent decision-making by the Navigator. Besides, we introduce selective external oversight before critical actions, allowing an independent Director agent to verify execution and intervene when necessary. As such, NOD effectively mitigates error propagation and unsafe behavior in long-horizon tasks. Experiments on $\tau^2$-Bench demonstrate that NOD achieves higher task success rates and critical action precision over baselines. More importantly, NOD improves the reliability of service agents by reducing policy violations, tool hallucinations, and user-intent misalignment. 

---
# Goal-Oriented Reasoning for RAG-based Memory in Conversational Agentic LLM Systems 

**Authors**: Jiazhou Liang, Armin Toroghi, Yifan Simon Liu, Faeze Moradi Kalarde, Liam Gallagher, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2605.12213)  

**Abstract**: LLM-based conversational AI agents struggle to maintain coherent behavior over long horizons due to limited context. While RAG-based approaches are increasingly adopted to overcome this limitation by storing interactions in external memory modules and performing retrieval from them, their effectiveness in answering challenging questions (e.g., multi-hop, commonsense) ultimately depends on the agent's ability to reason over the retrieved information. However, existing methods typically retrieve memory based on semantic similarity to the raw user utterance, which lacks explicit reasoning about missing intermediate facts and often returns evidence that is irrelevant or insufficient for grounded reasoning. In this work, we introduce Goal-Mem, a goal-oriented reasoning framework for RAG-based agentic memory that performs explicit backward chaining from the user's utterance as a goal. Rather than progressively expanding from retrieved context, Goal-Mem decomposes each goal into atomic subgoals, performs targeted memory retrieval to satisfy each subgoal, and iteratively identifies what information from memory should be retrieved when intermediate goals cannot be resolved. We formalize this process in Natural Language Logic, a logical system that combines the verifiability of reasoning provided by FOL with the expressivity of natural language. Through extensive experiments on two datasets and comparing to nine strong memory baselines, we show that Goal-Mem consistently improves performance, particularly on tasks requiring multi-hop reasoning and implicit inference. 

---
# MolDeTox: Evaluating Language Model's Stepwise Fragment Editing for Molecular Detoxification 

**Authors**: Jueon Park, Wonjune Jang, Jiwoo Lee, Yein Park, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12181)  

**Abstract**: Large Language Models (LLMs) and Vision Language Models (VLMs) have recently shown promising capabilities in various scientific domain. In particular, these advances have opened new opportunities in drug discovery, where the ability to understand and modify molecular structures is critical for optimizing drug properties such as efficacy and toxicity. However, existing models and benchmarks often overlook toxicity-related challenges, focusing primarily on general property optimization without adequately addressing safety concerns. In addition, even existing toxicity repair benchmarks suffer from limited data diversity, low structural validity of generated molecules, and heavy reliance on proxy models for toxicity assessment. To address these limitations, we propose MolDeTox, a novel benchmark for molecular detoxification, designed to enable fine-grained and reliable evaluation of toxicity-aware molecular optimization across stepwise tasks. We evaluate a wide range of general-purpose LLMs and VLMs under diverse settings, and demonstrate that understanding and generating molecules at the fragment-level improves structural validity and enhances the quality of generated molecules. Moreover, through detailed task-level performance analysis, MolDeTox provides an interpretable benchmark that enables a deeper understanding of the detoxification process. Our dataset is available at : this https URL 

---
# Do Enterprise Systems Need Learned World Models? The Importance of Context to Infer Dynamics 

**Authors**: Jishnu Sethumadhavan Nair, Patrice Bechard, Rishabh Maheshwary, Surajit Dasgupta, Sravan Ramachandran, Aakash Bhagat, Shruthan Radhakrishna, Pulkit Pattnaik, Johan Obando-Ceron, Shiva Krishna Reddy Malay, Sagar Davasam, Seganrasan Subramanian, Vipul Mittal, Sridhar Krishna Nemala, Christopher Pal, Srinivas Sunkara, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2605.12178)  

**Abstract**: World models enable agents to anticipate the effects of their actions by internalizing environment dynamics. In enterprise systems, however, these dynamics are often defined by tenant-specific business logic that varies across deployments and evolves over time, making models trained on historical transitions brittle under deployment shift. We ask a question the world-models literature has not addressed: when the rules can be read at inference time, does an agent still need to learn them? We argue, and demonstrate empirically, that in settings where transition dynamics are configurable and readable, runtime discovery complements offline training by grounding predictions in the active system instance. We propose enterprise discovery agents, which recover relevant transition dynamics at runtime by reading the system's configuration rather than relying solely on internalized representations. We introduce CascadeBench, a reasoning-focused benchmark for enterprise cascade prediction that adopts the evaluation methodology of World of Workflows on diverse synthetic environments, and use it together with deployment-shift evaluation to show that offline-trained world models can perform well in-distribution but degrade as dynamics change, whereas discovery-based agents are more robust under shift by grounding their predictions in the current instance. Our findings suggest that, in configurable enterprise environments, agents should not rely solely on fixed internalized dynamics, but should incorporate mechanisms for discovering relevant transition logic at runtime. 

---
# ALGOGEN: Tool-Generated Verifiable Traces for Reliable Algorithm Visualization 

**Authors**: Kunpeng Liao, Yuexiao Ma, Yisheng Lin, Hualin Zeng, Xiawu Zheng, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2605.12159)  

**Abstract**: Algorithm Visualization (AV) helps students build mental models by animating algorithm execution states. Recent LLM-based systems such as CODE2VIDEO generate AV videos in an end-to-end manner. However, this paradigm requires the system to simultaneously simulate algorithm flow and satisfy video rendering constraints, such as element layout and color schemes. This complex task induces LLM hallucinations, resulting in reduced execution success rates, element overlap, and inter-frame inconsistencies.
To address these challenges, we propose ALGOGEN, a novel paradigm that decouples algorithm execution from rendering. We first introduce Visualization Trace Algebra (VTA), a monoid over algorithm visual states and operations. The LLM then generates a Python tracker that simulates algorithm flow and outputs VTA-JSON traces, a JSON encoding of VTA. For rendering, we define a Rendering Style Language (RSL) to templatize algorithm layouts. A deterministic renderer then compiles algorithm traces with RSL into Manim, LaTeX/TikZ, or this http URL outputs.
Evaluated on a LeetCode AV benchmark of 200 tasks, ALGOGEN achieves an average success rate improvement of 17.3% compared to end-to-end methods, with 99.8% versus 82.5%. These results demonstrate that our decoupling paradigm effectively mitigates LLM hallucinations in complex AV tasks, providing a more reliable solution for automated generation of high-quality algorithm visualizations. Demo videos and code are available in the project repository. 

---
# MM-OptBench: A Solver-Grounded Benchmark for Multimodal Optimization Modeling 

**Authors**: Zhong Li, Qi Huang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Niki van Stein, Thomas Bäck, Matthijs van Leeuwen, Zaiwen Wen, Lincen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12154)  

**Abstract**: Optimization modeling translates real decision-making problems into mathematical optimization models and solver-executable implementations. Although language models are increasingly used to generate optimization formulations and solver code, existing benchmarks are almost entirely text-only. This omits many optimization-modeling tasks that arise in operational practice, where requirements are described in text but instance information is conveyed through visual artifacts such as tables, graphs, maps, schedules, and dashboards. We introduce multimodal optimization modeling, a benchmark setting in which models must construct both a mathematical formulation and executable solver code from a text-and-visual problem specification. To evaluate this setting, we develop a solver-grounded framework that generates structured optimization instances, verifies each with an exact solver, and builds both the model-facing inputs and hidden reference files from the same verified source. We instantiate the framework as MM-OptBench, a benchmark of 780 solver-verified instances spanning 6 optimization families, 26 subcategories, and 3 structural difficulty levels. We evaluate 9 multimodal large language models (MLLMs), including 6 frontier general-purpose models and 3 math-specialized models, with aggregate, family-level, difficulty-level, and failure-mode analyses. The results show that the task remains far from solved: the best two models reach 52.1% and 51.3% pass@1, while on average across the six general-purpose MLLMs, pass@1 is 43.4% on easy instances and 15.9% on hard instances. All three math-specialized MLLMs solve 0/780 instances. Failure attribution shows that errors arise both when extracting instance data from text and visuals and when turning extracted data into solver-correct formulations and code. MM-OptBench provides a testbed for solver-grounded, decision-oriented multimodal intelligence. 

---
# BoolXLLM: LLM-Assisted Explainability for Boolean Models 

**Authors**: Du Cheng, Serdar Kadioglu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12139)  

**Abstract**: Interpretable machine learning aims to provide transparent models whose decision-making processes can be readily understood by humans. Recent advances in rule-based approaches, such as expressive Boolean formulas (BoolXAI), offer faithful and compact representations of model behavior. However, for non-technical stakeholders, main challenges remain in practice: (i) selecting semantically meaningful features and (ii) translating formal logical rules into accessible explanations.
In this work, we propose BoolXLLM , as a hybrid framework that integrates Large Language Models (LLMs) into the end-to-end pipeline of Boolean rule learning. We augment BoolXAI , an expressive Boolean rule-based classifier, with LLMs at three critical stages: (1) feature selection, where LLMs guide the identification of domain-relevant variables; (2) threshold recommendation, where LLMs propose semantically meaningful discretization strategies for numerical features; and (3) rule compression and interpretation, where Boolean rules are translated into natural language explanations at both global and local levels.
This integration bridges formal, faithful explanations with human-understandable narratives. This allows build an explainable AI system that is both theoretically grounded and accessible to non-experts. Early empirical results demonstrate that LLM-assisted pipelines improve interpretability while maintaining competitive predictive performance. Our work highlights the promise of combining symbolic reasoning with language-based models for human-centered explainability. 

---
# Rollout Cards: A Reproducibility Standard for Agent Research 

**Authors**: Charlie Masters, Ziyuan Liu, Stefano V. Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2605.12131)  

**Abstract**: Reproducibility problems that have long affected machine learning and reinforcement learning are now surfacing in agent research: papers compare systems by reported scores while leaving the rollout records behind those scores difficult to inspect. For agentic tasks, this matters because the same behaviour can receive different reported scores when evaluations select different parts of a rollout or apply different reporting rules. In a structured audit of 50 popular training and evaluation repositories, we find that none report how many runs failed, errored, or were skipped alongside headline scores. We also document 37 cases where reporting rules can change task-success rates, cost/token accounting, or timing measurements for fixed evidence, sometimes dramatically. We treat rollout records, not reported scores, as the unit of reproducibility for agent research. We introduce rollout cards: publication bundles that preserve the rollout record and declare the views, reporting rules, and drops manifests behind reported scores. We validate rollout cards in two settings. First, four partial public releases in tool safety, multi-agent systems, theorem proving, and search let us compute analyses their original reports did not include. Second, re-grading preserved benchmark outputs across short-answer, code-generation, and tool-use tasks shows that changing only the reporting rule can change reported scores by 20.9 absolute percentage points and, in some cases, invert rankings of frontier models. We release a reference implementation integrated into Ergon, an open-source reinforcement learning gym, and publicly publish Ergon-produced rollout-card exports for benchmarks spanning tool use, software engineering, web interaction, multi-agent coordination, safety, and search to support future research. 

---
# To Whom Do Language Models Align? Measuring Principal Hierarchies Under High-Stakes Competing Demands 

**Authors**: Fangyi Yu, Nabeel Seedat, Jonathan Richard Schwarz, Andrew M. Bean  

**Link**: [PDF](https://arxiv.org/pdf/2605.12120)  

**Abstract**: Language models deployed in high-stakes professional settings face conflicting demands from users, institutional authorities, and professional norms. How models act when these demands conflict reveals a principal hierarchy -- an implicit ordering over competing stakeholders that determines, for instance, whether a medical AI receiving a cost-reduction directive from a hospital administrator complies at the expense of evidence-based care, or refuses because professional standards require it. Across 7,136 scenarios in legal and medical domains, we test ten frontier models and find that models frequently fail to adhere to professional standards during task execution, such as drafting, when user instructions conflict with those standards -- despite adequately upholding them when users seek advisory guidance. We further find that the hierarchies between user, authority, and professional standards exhibited by these models are unstable across medical and legal contexts and inconsistent across model families. When failing to follow professional standards, the primary failure mechanism is knowledge omission: models that demonstrably possess relevant knowledge produce harmful outputs without surfacing conflicting knowledge. In a particularly troubling instance, we find that a reasoning model recognizes the relevant knowledge in its reasoning trace -- e.g., that a drug has been withdrawn -- yet suppresses this in the user-facing answer and proceeds to recommend the drug under authority pressure anyway. Inconsistent alignment across task framing, domain, and model families suggests that current alignment methods, including published alignment hierarchies, are unlikely to be robust when models are deployed in high-stakes professional settings. 

---
# Adaptive Multi-Round Allocation with Stochastic Arrivals 

**Authors**: Yuqi Pan, Davin Choo, Haichuan Wang, Milind Tambe, Alastair van Heerden, Cheryl Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2605.12111)  

**Abstract**: We study a sequential resource allocation problem motivated by adaptive network recruitment, in which a limited budget of identical resources must be allocated over multiple rounds to individuals with stochastic referral capacity. Successful referrals endogenously generate future decision opportunities while allocating additional resources to an individual exhibits diminishing returns. We first show that the single-round allocation problem admits an exact greedy solution based on marginal survival probabilities. In the multi-round setting, the resulting Bellman recursion is intractable due to the stochastic, high-dimensional evolution of the frontier. To address this, we introduce a population-level surrogate value function that depends only on the remaining budget and frontier size. This surrogate enables an exact dynamic program via truncated probability generating functions, yielding a planning algorithm with polynomial complexity in the total budget. We further analyze robustness under model misspecification, proving a multi-round error bound that decomposes into a tight single-round frontier error and a population-level transition error. Finally, we evaluate our method on real-world inspired recruitment scenarios. 

---
# Large Language Models as Amortized Pareto-Front Generators for Constrained Bi-Objective Convex Optimization 

**Authors**: Peipei Xu, SiYuan Ma, Yaohua Liu, Yu Wu, Guanliang Liu, Yang Zhang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12106)  

**Abstract**: Generating feasible Pareto fronts for constrained bi-objective continuous optimization is central to multi-criteria decision-making. Existing methods usually rely on iterative scalarization, evolutionary search, or problem-specific solvers, requiring repeated optimization for each instance. We introduce DIPS, an end-to-end framework that fine-tunes large language models as amortized Pareto-front generators for constrained bi-objective convex optimization. Given a textual problem description, DIPS directly outputs an ordered set of feasible continuous decision vectors approximating the Pareto front. To make continuous optimization compatible with autoregressive language modeling, DIPS combines a compact discretization scheme, Numerically Grounded Token Initialization for new numerical tokens, and Three-Phase Curriculum Optimization, which progressively aligns structural validity, feasibility, and Pareto-front quality. Across five families of constrained bi-objective convex problems, a fine-tuned 7B-parameter model achieves normalized hypervolume ratios of 95.29% to 98.18% relative to reference fronts. With vLLM-accelerated inference, DIPS solves one instance in as little as 0.16 seconds and outperforms general-purpose and reasoning LLM baselines under the evaluated setting. These results suggest that LLMs can serve as effective amortized generators for continuous Pareto-front approximation. 

---
# Autonomy and Agency in Agentic AI: Architectural Tactics for Regulated Contexts 

**Authors**: Damir Safin, Dian Balta  

**Link**: [PDF](https://arxiv.org/pdf/2605.12105)  

**Abstract**: Deploying agentic AI in regulated contexts requires principled reasoning about two design dimensions: agency (what the system can do) and autonomy (how much it acts without human involvement). Though often treated independently, they are coupled: at higher autonomy, human error correction is less available, so reliable operation requires constraining agency accordingly; compliance requirements reinforce this by mandating human involvement as action consequences grow. Yet no established approach addresses them jointly, leaving practitioners without a principled basis for reasoning about oversight, action consequences, and error correction.
This work introduces a two-dimensional design space in which both dimensions are organised into five operational levels, making the coupling explicit and navigable. Autonomy ranges from human-commanded operation (L1) to fully autonomous monitoring (L5); agency ranges from reasoning over supplied context (L1) to committed writes to authoritative records (L5). Building on this space, we propose six architectural tactics--checkpoints, escalation, multi-agent delegation, tool provisioning, tool fencing, and write staging--for adjusting a deployment's position within it. The tactics are grounded in two worked examples from public-sector contexts, illustrating how they apply under realistic compliance constraints. We further examine five deployment parameters--model capability, agent architecture, tool fidelity, workflow bottlenecks, and evaluation--that shape what is achievable at any configuration independently of agency and autonomy. Together, the design space, tactics, and deployment parameters provide a shared vocabulary for principled, compliance-aware agentic AI design in which responsibility, auditability, and reversibility are explicit design considerations rather than properties that must be retrofitted after deployment. 

---
# Intermediate Artifacts as First-Class Citizens: A Data Model for Durable Intermediate Artifacts in Agentic Systems 

**Authors**: Josh Rosen, Seth Rosen  

**Link**: [PDF](https://arxiv.org/pdf/2605.12087)  

**Abstract**: Many AI systems are organized around loops in which models reason, call tools, observe results, and continue until a task is complete. These systems often produce final artifacts such as memos, plans, recommendations, and analyses, while the intermediate work that shaped those outputs remains ephemeral. For multi-step, revisable AI work, final artifacts are often lossy projections over upstream state.
We argue that such systems should preserve durable, inspectable intermediate artifacts: typed, structured, addressable, versioned, dependency-aware, authoritative, and consumable by downstream computation. These artifacts are not the model's private chain-of-thought. They are maintained work products such as evidence maps, claim structures, criteria, assumptions, plans, transformation rules, synthesis procedures, unresolved tensions, and partial products that later humans and agents can inspect, revise, supersede, and improve.
The contribution is a systems-level data model. We distinguish intermediate artifacts from chat transcripts, memory, hidden chain-of-thought, narration, thinking, and final answers; formalize additive and superseding update semantics with explicit current-state resolution; describe how artifact lineage supports durable intermediate state across revisions; and argue that evaluation must target maintained-state quality, not only final-output quality. The claim is not that artifacts make models smarter. It is that durable intermediate artifacts make AI-generated work more inspectable, revisable, and maintainable over time. 

---
# SAGE: A Self-Evolving Agentic Graph-Memory Engine for Structure-Aware Associative Memory 

**Authors**: Juntong Wang, Haoyue Zhao, guanghui Pan, Xiyuan Wang, Yanbo Wang, Qiyan Deng, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12061)  

**Abstract**: Long-term memory is becoming a central bottleneck for language agents. Exsting RAG and GraphRAG systems largely treat memory graphs as static retrieval middleware, which limits their ability to recover complete evidence chains from partial cues, exploit reusable graph-structrual roles, and improve the memory itself through downstream feedback. We introduce SAGE, a Self-evolving Agentic Graph-memory Engine that models graph memory as a dynamic long-term memory substrate. SAGE couples two roles: a memory writer that incrementally constucts structured graph memory from interaction histories, and a Graph Foundation Model-based memory reader to perform retrieval and provide feedback to the memory writer. We provide rigorooous theoretical annalyses supporting the framework. Across multi-hop QA, open-domain retireval, domain-specific review QA, and long-term agent-memory benchmarks, SAGE improves evidence recovery, answer grounding, and retrieval efficiency: after two self-evolution rounds, it achieves the best average rank on multi-hop QA; in zero-shot open-domain transfer, it reaches 82.5/91.6 Recall@2/5 on NQ. Further results on LongMemEval and HaluMem show that traning and reader-writer feedback improve multiple long-term memory and hallucination-diagnostic metrics, suggesting that self-evolving, structure-aware graph memory is a promising foundation for robust long-horizon language agents. 

---
# OmniRefine: Alignment-Aware Cooperative Compression for Efficient Omnimodal Large Language Models 

**Authors**: Yuchen Deng, Zidang Cai, Hai-Tao Zheng, Jie Wang, Feidiao Yang, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.12056)  

**Abstract**: Omnimodal large language models (Omni-LLMs) show strong capability in audio-video understanding, but their practical deployment remains limited by high inference cost of long video streams and dense audio sequences. Despite recent progress, existing compression methods for Omni-LLMs typically rely on fixed or native compression units, which can disrupt cross-modal correspondence and the complementary information required for audio-video reasoning, making it difficult to improve inference efficiency while stably preserving performance. To address this, we propose OmniRefine, a training-free two-stage framework for efficient audio-visual token compression in Omni-LLMs. First, Correspondence-Preserving Chunk Refinement refines native chunk boundaries into cross-modally aligned compression units through frame-audio similarity and dynamic programming. Second, Modality-Aware Cooperative Compression jointly compresses video and audio tokens within each refined unit to reduce redundancy while preserving critical evidence. Extensive experiments show that OmniRefine achieves a better efficiency-performance trade-off than strong baselines and maintains stable performance under lower compression ratios. On WorldSense, it still reaches 46.7% accuracy at a 44% token retention ratio, nearly matching the full-token baseline. The code and interface will be released to facilitate further research. 

---
# LLMs and the ZPD 

**Authors**: Peter Wallis  

**Link**: [PDF](https://arxiv.org/pdf/2605.12016)  

**Abstract**: One hundred years ago Vygotsky and his circle were exploring the nature of consciousness and defining what would become psychology in the Soviet Union. They concluded that children develop "scientific thinking" through interacting with enculturated adults in Zones of Proximal Development or ZPDs. The proposal is that, contrary to the claims of some, the LLM mechanism is not doing thinking with "distributed representations," but rather the completion model is doing "primitive thinking" in terms of *practices*. Viewed from this perspective, it would seem our large language models don't hallucinate, but rather dream, and that what is needed is not "guard rails" but an investigation of the set of cognitive tools that enable us to do things that look like common-sense. The proposal here is that *interaction* is core to human communication rather than just an add-on to "real" understanding. 

---
# LegalCheck: Retrieval- and Context-Augmented Generation for Drafting Municipal Legal Advice Letters 

**Authors**: Virgill van der Meer, Julien Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2605.12012)  

**Abstract**: Public-sector legal departments in the Netherlands face acute staff shortages, increased case volumes, and increased pressure to meet regulatory compliance. This paper presents LegalCheck, a novel system that addresses these challenges by automating the drafting of objection response letters through a combination of Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG). Using a large language model (LLM) alongside curated legal knowledge bases, LegalCheck performs retrieval of relevant laws and precedents, and uses controlled prompting to incorporate both external knowledge and case-specific details into a coherent draft. An expert-in-the-loop review ensures that each generated letter is legally sound and contextually appropriate. In a real-world deployment within the Municipality of Amsterdam, LegalCheck produced near-final advice letters in minutes rather than hours, while maintaining high legal consistency and factual accuracy. The output is based on actual regulations and prior cases, providing explainable outputs that captured the vast majority of required legal reasoning (often 80\% to 100\% of essential content). Legal professionals found that the system reduced their workload and ensured a consistent application of legal standards, without replacing human judgment. These results demonstrate substantial efficiency gains, improved legal consistency, and positive user acceptance. More broadly, this work illustrates how responsible AI can be deployed in the legal domain by augmenting LLMs with domain knowledge and governance mechanisms. 

---
# BadSKP: Backdoor Attacks on Knowledge Graph-Enhanced LLMs with Soft Prompts 

**Authors**: Xiaoting Lyu, Yufei Han, Hangwei Qian, Haoyuan Yu, Xiang Ao, Bin Wang, Chenxu Wang, Xiaobo Ma, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11996)  

**Abstract**: Recent knowledge graph (KG)-enhanced large language models (LLMs) move beyond purely textual knowledge augmentation by encoding retrieved subgraphs into continuous soft prompts via graph neural networks, introducing a graph-conditioned channel that operates alongside the standard text interface. However, existing backdoor attacks are largely designed for the textual channel, and their effectiveness against this dual-channel architecture remains unclear. We show that this architecture creates a robustness gap: text-channel backdoor attacks that readily compromise textual KG prompting systems become largely ineffective against soft-prompt-based counterparts. We interpret this gap through semantic anchoring, whereby graph-derived soft prompts bias the generation-driving hidden state toward query-consistent semantics and suppress surface-level malicious instructions. Because this anchoring effect is itself induced by the graph channel, an attacker who manipulates graph-level representations can in turn redirect it toward adversarial semantics. To demonstrate this risk, we propose BadSKP, a backdoor attack that targets the graph-to-prompt interface through a multi-stage optimization strategy: it constructs adversarial target embeddings, optimizes poisoned node embeddings to steer the induced soft prompt, and approximates the optimized representations with fluent adversarial node attributes. Experiments on two soft-prompt KG-enhanced LLMs across four datasets show that BadSKP achieves high attack success under both frozen and trojaned settings, while text-only attacks remain unreliable even under perplexity-based defenses. 

---
# Random-Set Graph Neural Networks 

**Authors**: Tommy Woodley, Shireen Kudukkil Manchingal, Matteo Tolloso, Davide Bacciu, Fabio Cuzzolin  

**Link**: [PDF](https://arxiv.org/pdf/2605.11987)  

**Abstract**: Uncertainty quantification has become an important factor in understanding the data representations produced by Graph Neural Networks (GNNs). Despite their predictive capabilities being ever useful across industrial workspaces, the inherent uncertainty induced by the nature of the data is a huge mitigating factor to GNN performance. While aleatoric uncertainty is the result of noisy and incomplete stochastic data such as missing edges or over-smoothing, epistemic uncertainty arises from lack of knowledge about a system or model (e.g., a graph's topology or node feature representation), which can be reduced by gathering more data and information. In this paper, we propose an original new framework in which node-level epistemic uncertainty is modelled in a belief function (finite random set) formalism. The resulting Random-Set Graph Neural Networks have a belief-function head predicting a random set over the list of classes, from which both a precise probability prediction and a measure of epistemic uncertainty can be obtained. Extensive experiments on 9 different graph learning datasets, including real-world autonomous driving benchmarks as such Nuscene and ROAD, demonstrate RS-GNN's superior uncertainty quantification capabilities 

---
# On the Limitations of Large Language Models for Conceptual Database Modeling 

**Authors**: Arthur F. Siqueira, Carlos D. S. Nogueira, Eduarda Farias, Claudio E. C. Campelo, Júlia Menezes  

**Link**: [PDF](https://arxiv.org/pdf/2605.11986)  

**Abstract**: This article analyzes the use of Large Language Models (LLMs) as support for the conceptual modeling of relational databases through the automatic generation of Entity-Relationship (ER) diagrams from natural language requirements. The approach combines different language models with prompt engineering techniques to evaluate their ability to identify entities, relationships, and attributes in a conceptually consistent manner. The experimental evaluation involved three LLMs, each subjected to three prompting techniques (Zero-Shot, Chain of Thought, and Chain of Thought + Verifier), applied to the same requirements scenario with progressively increasing complexity. The generated diagrams were qualitatively analyzed through direct comparison with the textual requirements, considering the structural and semantic adherence of the modeled elements. The results indicate that, although LLMs show reasonable performance in less complex scenarios, their reliability decreases as the complexity of the requirements increases, with a rise in inconsistencies, ambiguities, and failures in representing constraints. These findings reinforce that, in their current state, LLMs are not sufficiently mature for reliable use in complex scenarios, and the cost of validation may offset the apparent productivity gains. 

---
# Assessing and Mitigating Miscalibration in LLM-Based Social Science Measurement 

**Authors**: Jinyuan Wang, Ningyuan Deng, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11954)  

**Abstract**: Large language models (LLMs) are increasingly used in social science as scalable measurement tools for converting unstructured text into variables that can enter standard empirical designs. Measurement validity demands more than high average accuracy, which requires well calibrated confidence that faithfully reflects the empirical probability of each measurement being correct. This paper studies the model miscalibration in LLM-based social science measurement. We begin with a case study on FOMC and show that confidence based filtering can change downstream regression estimates when LLM confidence is miscalibrated. We then audit calibration across 14 social science constructs covering both proprietary models, including GPT-5-mini, DeepSeek-V3.2, and open source models. Across tasks and model families, reported confidence is poorly aligned with tolerance-based correctness. As a simple mitigation, we propose a soft label distillation pipeline for calibrating Bert with LLM. The method converts an LLM score and its verbalized confidence into a soft target distribution, then trains a smaller discriminative classifier on encoder models for these targets. Averaged across datasets, this approach reduces ECE by 43.2\% and Brier by 34.0\%. These results suggest that LLM-based social science pipelines should treat calibration as part of measurement validity, rather than as an optional post-processing concern. 

---
# Counterfactual Trace Auditing of LLM Agent Skills 

**Authors**: Xiaolin Zhou, Jinbo Liu, Li Li, Ryan A. Rossi, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11946)  

**Abstract**: Large Language Model agents are increasingly augmented with agent skills. Current evaluation methods for skills remain limited. Most deployed benchmarks report only pass rate before and after a skill is attached, treating the skill as a black box change to agent behavior. We introduce Counterfactual Trace Auditing (CTA), a framework for measuring how a skill changes agent behavior. CTA pairs each with skill agent trace with a without skill counterpart on the same task, segments both traces into goal directed phases, aligns the phases, and emits structured Skill Influence Pattern (SIP) annotations. These annotations describe the behavioral effect of a skill rather than only its task outcome. We instantiate CTA on SWE-Skills-Bench with Claude across 49 software engineering tasks. The resulting audit reveals a clear evaluation gap. Pass rate changes by only +0.3 percentage points on average, suggesting little aggregate effect. Yet CTA identifies 522 SIP instances across the same paired traces, showing that the skills substantially reshape agent behavior even when pass rate is nearly unchanged. The audit also separates several recurring effects that pass rate cannot detect, including literal template copying, off task artifact creation, excess planning, and task recovery. Three findings emerge. First, high baseline tasks contain most of the observed skill effects, although their pass rate is already saturated and therefore cannot reflect those effects. Second, tasks with moderate baseline performance show the most recoverable gain, but often at substantially higher token cost. Third, the dominant SIP type can be identified by baseline bucket: surface anchoring is most common on ceiling tasks and edge-case prompting is most common on mid-range and floor tasks. These regularities turn informal failure mode observations into reproducible behavioral measurements. 

---
# From Noise to Diversity: Random Embedding Injection in LLM Reasoning 

**Authors**: Heejun Kim, Seungpil Lee, Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Taesup Kim, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.11936)  

**Abstract**: Recent soft prompt research has tried to improve reasoning by inserting trained vectors into LLM inputs, yet whether the gain comes from the learned content or from the act of injection itself has not been carefully separated. We study Random Soft Prompts (RSPs), which drop the training step entirely and append a freshly drawn sequence of random embedding vectors to the input. Each RSP vector is sampled from an isotropic Gaussian fitted to the entrywise mean and variance of the pretrained embedding table; the sequence carries no learned content, and yet reaches accuracy comparable to optimized soft prompts on math reasoning benchmarks in several settings. The mechanism unfolds in two stages: because attention has to absorb a never-seen-before random position, the distribution over the first few generated tokens flattens and reasoning trajectories branch, and as generation continues this influence dilutes naturally so the response commits to a single completion. We show that during inference RSPs lift early-stage token diversity and, combined with temperature sampling, widen Pass@N, the probability that at least one out of N attempts is correct. Beyond inference, we carry the same effect into DAPO training and demonstrate practical gains. Our contributions are: (i) RSP isolates the simplest form of soft prompt -- training-free, freshly resampled -- providing a unified lens for the structural effect of injection that variants otherwise differing in training and form all share; (ii) a theoretical and empirical validation of the underlying mechanism; and (iii) an extension from inference to training. 

---
# When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents 

**Authors**: Xiaolin Zhou, Aojie Yuan, Zheng Luo, Zipeng Ling, Xixiao Pan, Yicheng Gao, Haiyue Zhang, Jiate Li, Shuli Jiang, Prince Zizhuang Wang, Zixuan Zhu, Jinbo Liu, Ryan A. Rossi, Hua Wei, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11928)  

**Abstract**: Tool-use language agents are evaluated on benchmarks that assume clean inputs, unambiguous tool registries, and reliable APIs. Real deployments violate all these assumptions: user typos propagate into hallucinated tool names, a misconfigured request timeout can stall an agent indefinitely, and duplicate tool names across servers can freeze an SDK. We study these failures as a sim-to-real gap in the tool-use partially observable Markov decision process (POMDP), where deployment noise enters through the observation, action space, reward-relevant metadata, or transition dynamics. We introduce RobustBench-TC, a benchmark with 22 perturbation types organized by these four POMDP components, each grounded in a verified GitHub issue or documented tool-calling failure. Across 21 models from 1.5B to 32B parameters (including the closed-source o4-mini), the robustness profile is sharply uneven: observation perturbations reduce accuracy by less than 5%, while reward-relevant and transition perturbations reduce accuracy by roughly 40% and 30%, respectively; scale alone does not close these gaps. We then propose ToolRL-DR, a domain-randomization reinforcement learning (RL) recipe that trains a tool-use agent on perturbation-augmented trajectories spanning the three statically encodable POMDP components. On a 3B backbone, ToolRL-DR-Full retains roughly three-quarters of clean accuracy and reaches an aggregate perturbed accuracy comparable to open-source 14B function-calling baselines while substantially narrowing the gap to o4-mini. It closes approximately 27% of the Transition gap despite never seeing transition perturbations in training, suggesting that RL on adversarial static tool-use inputs induces a more persistent retry policy that transfers to unseen runtime failures. The dataset, code and benchmark leaderboard are publicly available. 

---
# Domain Restriction via Multi SAE Layer Transitions 

**Authors**: Elias Shaheen, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2605.11920)  

**Abstract**: The general-purpose nature of Large Language Models (LLMs) presents a significant challenge for domain-specific applications, often leading to out-of-domain (OOD) interactions that undermine the provider's intent. Existing methods for detecting such scenarios treat the LLM as an uninterpretable black box and overlook the internal processing of inputs. In this work we show that layer transitions provide a promising avenue for extracting domain-specific signature. Specifically, we present several lightweight ways of learning on internal dynamics encoded using a sparse autoencoder (SAE) that exhibit great capability in distinguishing OOD texts. Building on top of SAEs representation transitions enables us to better interpret the LLM internal evolution of input processing and shed light on its decisions. We provide a comprehensive analysis of the method and benchmark it with the gemma-2 2B and 9B models. Our results emphasize the efficacy of the internal process in capturing fine-grained input-related details. 

---
# Rethinking Positional Encoding for Neural Vehicle Routing 

**Authors**: Chuanbo Hua, Federico Berto, Andre Hottung, Nayeli Gast Zepeda, Yining Ma, Zihan Ma, Paula Wong-Chung, Changhyun Kwon, Cathy Wu, Kevin Tierney, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.11910)  

**Abstract**: Transformer-based models have become the dominant paradigm for neural combinatorial optimization (NCO) of vehicle routing problems (VRPs), yet the role of positional encoding (PE) in these architectures remains largely unexplored. Unlike natural language, where tokens are uniformly spaced on a line, routing solutions exhibit several properties that render standard NLP positional encodings inadequate. In this work, we formalize three such structural properties that a routing-aware PE should respect, namely anisometric node distances, cyclic and direction-aware topology, and hierarchical depot-anchored global multi-route structure, combining them with a unifying design principle of geometric grounding. Guided by these criteria, we analyze and compare PE methods spanning NLP, graph-transformer, and routing-specific families, and propose a hierarchical anisometric PE that combines a distance-indexed, circularly consistent in-route encoding with a depot-anchored angular cross-route encoding. Extensive experiments across diverse VRP variants demonstrate that geometry-grounded PE consistently outperforms index-based alternatives, with gains that transfer across problem variants, model architectures, and distribution shifts. 

---
# Rethinking Supervision Granularity: Segment-Level Learning for LLM-Based Theorem Proving 

**Authors**: Shuo Xu, Jiakun Zhang, Junyu Lai, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11905)  

**Abstract**: Automated theorem proving with large language models in Lean 4 is commonly approached through either step-level tactic prediction with tree search or whole-proof generation. These two paradigms represent opposite granularities for constructing supervised training data: the former provides dense local signals but may fragment coherent proof processes, while the latter preserves global structure but requires complex end-to-end generation. In this paper, we revisit supervision granularity as a training set construction problem over proof trajectories and propose segment-level supervision, a training data construction strategy that extracts locally coherent proof segments for training policy models. We further reuse the same strategy at inference time to trigger short rollouts for existing step-level models. When trained with segment-level supervision on STP, LeanWorkbook, and NuminaMath-LEAN, the resulting policy models achieve proof success rates of 64.84%, 60.90%, and 66.31% on miniF2F, respectively, consistently outperforming both step-level and whole-proof baselines. Goal-aware rollout further improves existing step-level provers while reducing inference costs. It increases the proof success rate of BFS-Prover-V2-7B from 68.77% to 70.74% and that of InternLM2.5-StepProver from 59.59% to 60.33%, showing that appropriate supervision granularity better aligns model learning with proof structure and search. Code and models are available at this https URL. 

---
# Toward Modeling Player-Specific Chess Behaviors 

**Authors**: Loris Sogliuzzo, Aloïs Rautureau, Eric Piette  

**Link**: [PDF](https://arxiv.org/pdf/2605.11893)  

**Abstract**: While artificial intelligence has achieved superhuman performance in chess, developing models that accurately emulate the individualized decision-making styles of human players remains a significant challenge. Existing human-like chess models capture general population behaviors based on skill levels but fail to reproduce the behavioral characteristics of specific historical champions. Furthermore, the standard evaluation metric, move accuracy, inherently penalizes natural human variance and ignores long-term behavioral consistency, leading to an incomplete assessment of stylistic fidelity. To address these limitations, an architecture is proposed that adapts the unified Maia-2 model to champion-specific embeddings, further enhanced by the integration of a limited Monte Carlo Tree Search (MCTS) process to enrich tactical exploration during move selection. To robustly evaluate this approach, a novel behavioral metric based on the Jensen-Shannon divergence is introduced. By compressing high-dimensional board representations into a latent space using an AutoEncoder and Uniform Manifold Approximation and Projection (UMAP), move distributions are discretized on a common grid to compare behavioral similarities. Results across 16 historical world champions indicate that while integrating MCTS decreases standard move accuracy, it improves stylistic alignment according to the proposed metric, substantially reducing the average Jensen-Shannon divergence. Ultimately, the proposed metric successfully discriminates between individual players and provides promising evidence toward more comprehensive evaluations of behavioral alignment between players and AI models. 

---
# From Clever Hans to Scientific Discovery: Interpreting EEG Foundational Transformers with LRP 

**Authors**: Justus Meyer zu Bexten, Nico Scherf, Bogdan Franczyk, Simon M. Hofmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.11885)  

**Abstract**: Emerging foundation models (FMs) in electroencephalography (EEG) promise a path to scale deep learning in diagnostics and brain-computer interfaces despite data scarcity, yet their opaque nature remains a barrier to wider adoption. We investigate attention-aware Layer-wise relevance propagation (LRP) as a post-hoc attribution method for EEG-FMs, extending LRP's use on convolutional neural network (CNN)-based EEG models to the Transformer architectures that current FMs are based on. We find that LRP can both verify EEG-FM decisions and surface novel, biologically plausible hypotheses from them. In motor imagery, it unmasks 'Clever Hans' behavior where models prioritize task correlated ocular signals over the intended motor correlates. In a naturalistic paradigm for affect prediction, it reveals a recurring reliance on a central electrode cluster, suggesting a candidate sensorimotor signature of arousal. Though heatmap interpretation remains ambiguous in this complex domain, the results position LRP as a tool for both verification and exploration of EEG-FMs, a role that will grow in both importance and discovery potential as the underlying models mature. 

---
# On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment 

**Authors**: Bo Yin, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11882)  

**Abstract**: Tool-using LLM agents fail through trajectories rather than only final responses, as they may execute unsafe tool calls, follow injected instructions, comply with harmful requests, or over-refuse benign tasks despite producing a seemingly safe answer. Existing safety-alignment signals are largely response-level or off-policy, and often incur a safety-utility trade-off: improving agent safety comes at the cost of degraded task performance. Such sparse and single-objective rewards severely limit real-world usability. To bridge this gap, we propose FATE, an on-policy self-evolving framework that transforms verifier-scored failures into repair supervision without expert demonstrations. For each failure, the same policy proposes repair candidates, which are then re-scored by verifiers and filtered across security, utility, over-refusal control, and trajectory validity. This dense trajectory-level information is then used as a supervision signal for agent self-evolution. During this process, we further introduce Pareto-Front Policy Optimization (PFPO), combining supervised warmup with Pareto-aware policy optimization to preserve safety-utility trade-offs. Experiments on AgentDojo, AgentHarm, and ATBench show that FATE improves safety across different models and scales while preserving useful behavior. Compared with strong baselines, FATE reduces attack success rate by 33.5%, harmful compliance by 82.6%, and improves external trajectory-safety diagnosis by 6.5%. These results suggest that failed trajectories can provide structured repair supervision for safer self-evolving agents. 

---
# MedMemoryBench: Benchmarking Agent Memory in Personalized Healthcare 

**Authors**: Yihao Wang, Haoran Xu, Renjie Gu, Yixuan Ye, Xinyi Chen, Xinyu Mu, Yuan Gao, Chunxiao Guo, Peng Wei, Jinjie Gu, Huan Li, Ke Chen, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11814)  

**Abstract**: The large-scale deployment of personalized healthcare agents demands memory mechanisms that are exceptionally precise, safe, and capable of long-term clinical tracking. However, existing benchmarks primarily focus on daily open-domain conversations, failing to capture the high-stakes complexity of real-world medical applications. Motivated by the stringent production requirements of an industry-leading health management agent serving tens of millions of active users, we introduce MedMemoryBench. We develop a human-agent collaborative pipeline to synthesize highly realistic, long-horizon medical trajectories based on clinically grounded, synthetic patient archetypes. This process yields a massive, expertly validated dataset comprising approximately 2,000 sessions and 16,000 interaction turns. Crucially, MedMemoryBench departs from traditional static evaluations by pioneering an "evaluate-while-constructing" streaming assessment protocol, which precisely mirrors dynamic memory accumulation in production environments. Furthermore, we formalize and systematically investigate the critical phenomenon of memory saturation, where sustained information influx actively degrades retrieval and reasoning robustness. Comprehensive benchmarking reveals severe bottlenecks in mainstream architectures, particularly concerning complex medical reasoning and noise resilience. By exposing these fundamental flaws, MedMemoryBench establishes a vital foundation for developing robust, production-ready medical agents. 

---
# Automated Reformulation of Robust Optimization via Memory-Augmented Large Language Models 

**Authors**: Jinbiao Chen, Shuang Jin, Guoyun Zhang, Junyu Zhang, Guanyi Wang, Hanzhang Qin  

**Link**: [PDF](https://arxiv.org/pdf/2605.11813)  

**Abstract**: Robust optimization (RO) provides a principled framework for decision-making under uncertainty, but its practical use is often limited by the need to manually reformulate uncertain optimization models into tractable deterministic counterparts. Recent large language models (LLMs) have been shown promising for automating optimization formulation, yet RO reformulation remains challenging because it requires precise multi-step reasoning and mathematically consistent transformations. To facilitate systematic evaluation of LLM-based reformulation, for which no dedicated benchmark currently exists, we develop AutoRO-Bench, a benchmark featuring an automated data generation pipeline for the core RO reformulation task and a curated dataset for the RO application task. To address the reformulation challenge, we propose Automated Reformulation with Experience Memory (AutoREM), a tuning-free memory-augmented framework that autonomously builds a structured textual experience memory by reflecting on past failed trajectories through a tailored offline adaptation procedure. AutoREM requires neither domain-specific expert knowledge nor parameter updates, and the resulting memory readily transfers across different base LLMs. Experimental results show that AutoREM consistently improves the accuracy and efficiency of RO reformulation across in-distribution datasets, out-of-distribution datasets, and diverse base LLMs. 

---
# Beyond World-Frame Action Heads: Motion-Centric Action Frames for Vision-Language-Action Models 

**Authors**: Huoren Yang, Jianchao Zhao, Hu Yusong, Qiguan Ou, Yuyang Gao, Wei Ke, Yuhang He, SongLin Dong, Zhiheng Ma, Yihong Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11809)  

**Abstract**: Vision-Language-Action (VLA) models have advanced rapidly with stronger backbones, broader pre-training, and larger demonstration datasets, yet their action heads remain largely homogeneous: most directly predict action commands in a fixed world coordinate frame. We propose \textbf{MCF-Proto}, a lightweight action head that equips VLA policies with a Motion-Centric Action Frame (MCF) and a prototype-based action parameterization. At each step, the policy predicts a rotation $R_t \in SO(3)$, composes actions in the transformed local frame from a set of prototypes, and maps them back to the world frame for end-to-end training, using only standard demonstrations without auxiliary supervision. This simple design induces stable emergent structure. Without explicit directional labels, the learned local frames develop a stable geometric structure whose axes are strongly compatible with demonstrated end-effector motion. Meanwhile, actions in the learned representation become substantially more compact, with variation captured by fewer dominant directions and more regularly organized by shared prototypes. These structural properties translate into improved robustness, especially under geometric perturbations. Our results suggest that adding lightweight geometric and compositional structure to the action head can materially improve how VLA policies organize and generalize robotic manipulation behavior. An anonymized code repository is provided in the supplementary material. 

---
# Why Users Go There: World Knowledge-Augmented Generative Next POI Recommendation 

**Authors**: Qiuyu Ding, Heng-Da Xu, Wei Zhang, Dongyi Lv, Changda Xia, Feng Xiong, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11807)  

**Abstract**: Generative point-of-interest (POI) recommendation models based on large language models (LLMs) have shown promising results by formulating next POI prediction as a sequence generation task. However, the knowledge encoded in these models remains fixed after training, making them unable to perceive evolving real-world conditions that shape user mobility decisions, such as local events and cultural trends. To bridge this gap, we propose AWARE (Agent-based World knowledge Augmented REcommendation), which employs an LLM agent to generate location- and time-aware contextual narratives that capture regional cultural characteristics, seasonal trends, and ongoing events relevant to each user. Rather than introducing generic or noisy information, AWARE further anchors these narratives in each user's behavioral context, grounding external world knowledge in personalized spatial-temporal patterns. Extensive experiments on three real-world datasets demonstrate that AWARE consistently outperforms competitive baselines, achieving up to 12.4% relative improvement. 

---
# Beyond Inefficiency: Systemic Costs of Incivility in Multi-Agent Monte Carlo Simulations 

**Authors**: Alison Moldovan-Mauer, Benedikt Mangold  

**Link**: [PDF](https://arxiv.org/pdf/2605.11789)  

**Abstract**: Unconstructive debate and uncivil communication carry well-documented costs for productivity and cohesion, yet isolating their effect on operational efficiency has proven difficult. Human subject research in this domain is constrained by ethical oversight, limited reproducibility, and the inherent unpredictability of naturalistic settings. We address this gap by leveraging Large Language Model (LLM) based Multi-Agent Systems as a controlled sociological sandbox, enabling systematic manipulation of communicative behavior at scale. Using a Monte Carlo simulation framework, we generate thousands of structured 1-on-1 adversarial debates across varying toxicity conditions, measuring convergence time, defined as the number of rounds required to reach a conclusion, as a proxy for interactional efficiency. Building on a prior study, we replicate and extend its findings across two additional LLM agents of varying parameter size, allowing us to assess whether the effects of toxic behavior on debate dynamics generalize across model scale. The convergence latency of 25% reported in the previous study was confirmed. It was found that this latency is significantly bigger for models with fewer parameters. We further identify a significant first-mover advantage, whereby the agent initiating the discussion wins significantly above chance regardless of toxicity condition. 

---
# Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention 

**Authors**: Abid Ali, Diego Molla-Aliod, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2605.11753)  

**Abstract**: Multimodal summarization requires models to jointly understand textual and visual inputs to generate concise, semantically coherent summaries. Existing methods often inject shallow visual features into deep language models, leading to representational mismatches and weak cross-modal grounding. We propose a unified framework that jointly performs text summarization and representative image selection. Our system, SPeCTrA-Sum (Sampler Perceiver with Cross-modal Transformer and gated Attention for Summarization), introduces two key innovations. First, a Deep Visual Processor (DVP) aligns the visual encoder with the language model at corresponding depths, enabling hierarchical, layer-wise fusion that preserves semantic consistency. Second, a lightweight Visual Relevance Predictor (VRP) selects salient and diverse images by distilling soft labels from a Determinantal Point Processes (DPP) teacher. SPeCTrA-Sum is trained using a multi-objective loss that combines autoregressive summarization, cross-modal alignment, and DPP-based distillation. Experiments show that our system produces more accurate, visually grounded summaries and selects more representative images, demonstrating the benefits of depth-aware fusion and principled image selection for multimodal summarization. 

---
# When Reasoning Traces Become Performative: Step-Level Evidence that Chain-of-Thought Is an Imperfect Oversight Channel 

**Authors**: Wenkai Li, Fan Yang, Ananya Hazarika, Shaunak A. Mehta, Koichi Onoue  

**Link**: [PDF](https://arxiv.org/pdf/2605.11746)  

**Abstract**: Chain-of-thought (CoT) traces are increasingly used both to improve language model capability and to audit model behavior, implicitly assuming that the visible trace remains synchronized with the computation that determines the answer. We test this assumption with a step-level Detect-Classify-Compare framework built around an answer-commitment proxy that is cross-validated with Patchscopes, tuned-lens probes, and causal direction ablation. Across nine models and seven reasoning benchmarks, latent commitment and explicit answer arrival align on only 61.9% of steps on average. The dominant mismatch pattern is confabulated continuation: 58.0% of detected mismatch events occur after the answer-commitment proxy has already stabilized while the trace continues producing deliberative-looking text, and a vacuousness analysis shows that the committed answer does not change during these steps. In architecture-matched Qwen2.5/DeepSeek-R1-Distill comparisons, the reasoning pipeline changes failure composition more than aggregate alignment, most clearly at 32B where confabulated steps decrease as contradictory states increase. Lower step-level alignment is also associated with larger CoT utility, suggesting that the settings that benefit most from CoT are often the least temporally faithful. Paired truncation and a complementary donor-corruption test further indicate that much post-commitment text is not load-bearing for the final answer. These findings suggest that CoT can remain useful while still being an unreliable report of when the answer was formed. 

---
# OptArgus: A Multi-Agent System to Detect Hallucinations in LLM-based Optimization Modeling 

**Authors**: Zhong Li, Zihan Guo, Xiaohan Lu, Juntao Wang, Jie Song, Chao Shen, Jiageng Wu, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.11738)  

**Abstract**: Large language models (LLMs) are increasingly used to translate natural-language optimization problems into mathematical formulations and solver code, but matching the reference objective value is not a reliable test of correctness: an artifact may agree numerically while still changing the underlying optimization semantics. We formulate this issue as \emph{optimization-modeling hallucination detection}, namely structural consistency auditing over the problem description, symbolic model, and solver implementation. We develop, to our knowledge, the first fine-grained hallucination taxonomy specifically for optimization modeling, spanning objective, variable, constraint, and implementation failures. We use this taxonomy to design OptArgus, a multi-agent detector with conductor routing, specialist auditors, and evidence consolidation. To evaluate this setting, we introduce a three-part benchmark suite with $484$ clean artifacts, $1266$ controlled injected artifacts, and $6292$ natural LLM-generated artifacts. Against a matched single-agent baseline, OptArgus produces fewer false alarms on clean artifacts, more accurate top-ranked localization on controlled single-error cases, and stronger detection on natural model outputs. Together, these contributions turn optimization-modeling hallucination detection into a concrete empirical problem and suggest that modular, taxonomy-grounded auditing is a practical route to more reliable optimization modeling. 

---
# Allegory of the Cave: Measurement-Grounded Vision-Language Learning 

**Authors**: Kepeng Xu, Li Xu, Gang He, Wenxin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11727)  

**Abstract**: Vision-language models typically reason over post-ISP RGB images, although RGB rendering can clip, suppress, or quantize sensor evidence before inference. We study whether grounding improves when the visual interface is moved closer to the underlying camera measurement. We formulate measurement-grounded vision-language learning and instantiate it as PRISM-VL, which combines RAW-derived Meas.-XYZ inputs, camera-conditioned grounding, and Exposure-Bracketed Supervision Aggregation for transferring supervision from RGB proxies to measurement-domain observations. Using a quality-controlled 150K instruction-tuning set and a held-out benchmark targeting low-light, HDR, visibility-sensitive, and hallucination-sensitive cases, PRISM-VL-8B reaches 0.6120 BLEU, 0.4571 ROUGE-L, and 82.66\% LLM-Judge accuracy, improving over the RGB Qwen3-VL-8B baseline by +0.1074 BLEU, +0.1071 ROUGE-L, and +4.46 percentage points. These results suggest that part of VLM grounding error arises from information lost during RGB rendering, and that preserving measurement-domain evidence can improve multimodal reasoning. 

---
# SafeSteer: A Decoding-level Defense Mechanism for Multimodal Large Language Models 

**Authors**: Xinyi Zeng, Xue Yang, Jingyuan Zhang, Huanqian Yan, Xiang Chen, Kaiwen Wei, Hankun Kang, Yu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.11716)  

**Abstract**: Multimodal large language models (MLLMs) are gaining increasing attention. Due to the heterogeneity of their input features, they face significant challenges in terms of jailbreak defenses. Current defense methods rely on costly fine-tuning or inefficient post-hoc interventions, limiting their ability to address novel attacks and involving performance trade-offs. To address the above issues, we explore the inherent safety capabilities within MLLMs and quantify their intrinsic ability to discern harmfulness at decoding stage. We observe that 1) MLLMs can distinguish the harmful and harmless inputs during decoding process, 2) Image-based attacks are more stealthy. Based on these insights, we introduce SafeSteer, a decoding-level defense mechanism for MLLMs. Specifically, it includes a Decoding-Probe, a lightweight probe for detecting and correcting harmful output during decoding, which iteratively steers the decoding process toward safety. Furthermore, a modal semantic alignment vector is integrated to transfer the strong textual safety alignment to the vision modality. Experiments on multiple MLLMs demonstrate that SafeSterr can improve MLLMs' safety by up to 33.40\% without fine-tuning. Notably, it can maintain the effectiveness of MLLMs, ensuring a balance between their helpfulness and harmlessness. 

---
# Toward Stable Value Alignment: Introducing Independent Modules for Consistent Value Guidance 

**Authors**: Wenhao Chen, Sirui Sun, Shengyuan Bai, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.11712)  

**Abstract**: Aligning large language models (LLMs) with human values typically relies on post-training or inference-time steering that directly manipulates the backbone's parameters or representation space. However, a critical gap exists: the model's residual stream is highly dynamic, in which values exist as fragile, low-dimensional properties, inherently incompatible with the stability required for consistent value expression. In this paper, we propose the Stable Value Guidance Transformer (SVGT), which addresses this gap through an independent value module incorporating two key designs: (1) independent value modeling, maintaining normative representations in a dedicated value space isolated from the backbone, and (2) explicit behavioral guidance, transducing these stable signals into learnable latent Bridge Tokens. These tokens serve as dynamic value anchors to explicitly steer the generative trajectory, ensuring robust adherence across diverse contexts without disrupting the backbone's internal representations. Experiments across multiple backbones and safety benchmarks show that SVGT generally reduces harmful scores by over 70% while maintaining generation fluency, demonstrating the efficacy of architecturally grounded value modeling. Our code is available at this https URL. 

---
# Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity 

**Authors**: Abid Ali, Diego Molla-Aliod, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2605.11693)  

**Abstract**: Multimodal Large Language Models (MLLMs) have facilitated Multimodal Summarization with Multimodal Output (MSMO), wherein systems generate concise textual summaries accompanied by salient visuals from multimodal sources. However, current MSMO evaluation remains fragmented: text quality, image-text alignment, and visual diversity are typically assessed in isolation using unimodal metrics, making it difficult to capture whether the modalities jointly support a faithful and useful summary. To address this gap, we introduce MM-Eval, a unified evaluation framework that integrates assessments of textual quality, cross-modal alignment, and visual diversity. MM-Eval comprises three components: (1) text quality, measured using OpenFActScore for factual consistency and G-Eval for coherence, fluency, and relevance; (2) image-text relevance, evaluated via an MLLM-as-a-judge approach; and (3) image-set diversity, quantified using Truncated CLIP Entropy. We calibrate MM-Eval through a learned aggregation model trained on the mLLM-EVAL news benchmark, aligning component contributions with human preferences. Our analysis reveals a text-dominant hierarchy in this setting, where factual consistency acts as a critical determinant of perceived overall quality, while visual relevance and diversity provide complementary signals. MM-Eval improves over heuristic aggregation baselines and provides an interpretable, reference-weak framework for comparative evaluation of multimodal summaries. 

---
# Persistent and Conversational Multi-Method Explainability for Trustworthy Financial AI 

**Authors**: Georgios Makridis, Georgios Fatouros, John Soldatos, George Katsis, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2605.11687)  

**Abstract**: Financial institutions increasingly require AI explanations that are persistent, cross-validated across methods, and conversationally accessible to human decision-makers. We present an architecture for human-centered explainable AI in financial sentiment analysis that combines three contributions. First, we treat XAI artifacts -- LIME feature attributions, occlusion-based word importance scores, and saliency heatmaps -- as persistent, searchable objects in distributed S3-compatible storage with structured metadata and natural-language summaries, enabling semantic retrieval over explanation history and automatic index reconstruction after system failures. Second, we enable multi-method explanation triangulation, where a retrieval-augmented generation (RAG) assistant compares and synthesizes results from multiple XAI methods applied to the same prediction, allowing users to assess explanation robustness through natural-language dialogue. Third, we evaluate the faithfulness of generated explanations using automated checks over grounding completeness, hallucinated claims, and method-attribution behavior. We demonstrate the architecture on an EXTRA-BRAIN financial sentiment analysis pipeline using FinBERT predictions and present evaluation results showing that constrained prompting reduces hallucination rate by 36\% and increases method-attribution citations by 73\% compared to naive prompting. We discuss implications for trustworthy, human-centered AI services in regulated financial environments. 

---
# Explaining and Breaking the Safety-Helpfulness Ceiling via Preference Dimensional Expansion 

**Authors**: ShiYing Huang, Liang Lin, Yuer Li, Kaiwen Luo, Zhenhong Zhou, An Zhang, Junhao Dong, Kun Wang, Zhigang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2605.11679)  

**Abstract**: In the realm of multi-objective alignment for large language models, balancing disparate human preferences often manifests as a zero-sum conflict. Specifically, the intrinsic tension between competing goals dictates that aggressively optimizing for one metric (e.g., helpfulness) frequently incurs a substantial penalty on another (e.g., harmlessness). While prior work mainly focuses on data selection, parameter merging, or algorithmic balancing during training, these approaches merely force compromises between divergent preferences along a fixed Pareto frontier, failing to fundamentally resolve the inherent trade-off. In this work, we approach this problem from a novel perspective of multi-dimensional rewards. By scaling up the model's rollouts and analyzing the outputs across different reward dimensions, we arrive at a critical conclusion: the conflict among multiple objectives stems from the fact that the prompt itself inherently restricts the achievable multi-dimensional rewards. Based on this core observation, we propose MORA: Multi-Objective Reward Assimilation. Specifically, MORA isolates single-reward prompts through pre-sampling and expands their reward diversity by rewriting the original questions to incorporate multi-dimensional intents. Extensive experiments demonstrate that: (1) in sequential alignment, MORA achieves single-preference improvements ranging from 5% to 12.4%, with exceptional gains in harmlessness, after multiple-preference alignment across helpful, harmless, and truthful dimensions. (2) In simultaneous alignment, MORA achieves an average overall reward improvement of 4.6%. Our codes are available at this https URL. 

---
# OOM-Free Alpamayo via CPU-GPU Memory Swapping for Vision-Language-Action Models 

**Authors**: Seungwoo Roh, Huiyeong Kim, Jong-Chan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.11678)  

**Abstract**: End-to-end Vision-Language-Action (VLA) models for autonomous driving unify perception, reasoning, and control in a single neural network, achieving strong driving performance but requiring 20-60GB of GPU memory-far exceeding the 12-16GB available on commodity GPUs. We present a framework, which enables memory-efficient VLA inference on VRAM-constrained GPUs through system-level optimization alone, without model modification. Our work proceeds in three stages: (1) Sequential Demand Layering reduces VRAM usage from model-level to layer-level granularity; (2) Pipelined Demand Layering hides parameter transfer time within layer execution time via transfer--compute overlap; and (3) a GPU-Resident Layer Decision Policy, informed by per-module residency benefit analysis, eliminates the residual transfer overhead that pipelining cannot hide. We further propose a performance prediction model that determines the optimal configuration-both the number and placement of resident layers-from a single profiling run with less than 1.3% prediction error across all configurations. Applied to NVIDIA's Alpamayo-R1-10B (21.52GB) on an RTX 5070Ti (16GB), our work achieves up to 3.55x speedup over Accelerate offloading while maintaining full BF16 precision. 

---
# A CAP-like Trilemma for Large Language Models: Correctness, Non-bias, and Utility under Semantic Underdetermination 

**Authors**: Vinu Ellampallil Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2605.11672)  

**Abstract**: The CAP theorem states that a distributed system cannot simultaneously guarantee consistency, availability, and partition tolerance under network partition. Inspired by this result, this paper formulates a CAP-like conjecture for Large Language Models (LLMs). The proposed trilemma states that, under semantic underdetermination, an LLM cannot always simultaneously guarantee strong correctness, strict non-bias, and high utility. A prompt is semantically underdetermined when the given premises do not determine a unique answer. In such cases, a useful and decisive response requires the model to introduce a selection criterion, preference, prior, or value ordering. If this criterion is not supplied by the user or justified by the available premises, the response becomes biased in a broad selection-theoretic sense. Conversely, if the model avoids unsupported preferences, it may preserve correctness and non-bias but may reduce utility through refusal, hedging, or clarification. The paper formalizes this correctness--non-bias--utility trilemma, develops examples, and argues that certain LLM failures arise not merely from model limitations but from the structure of underdetermined decision requests. 

---
# Seirênes: Adversarial Self-Play with Evolving Distractions for LLM Reasoning 

**Authors**: Chi Zhang, Haibo Qiu, Qiming Zhang, Yufei Xu, Xinbo Gao, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11636)  

**Abstract**: We present Seirênes, a self-play RL framework that transforms contextual interference from a failure mode of LLM reasoning into an internal training signal for co-evolving more resilient reasoners. While RL with verifiable rewards has significantly advanced reasoning capabilities, models can still exhibit fragility when encountering non-idealized contexts: scenarios characterized by superfluous information, tangential instructions, or incidental correlations that differ from the clean distributions typical of standard benchmarks. Seirênes harnesses this vulnerability through a parameter-shared and adversarial self-play loop. Within this framework, a single model is trained to both construct plausible yet distracting contexts that expose its own reasoning blind spots, and solve problems by discerning the essential task from these perturbations to recover the core underlying logic. By pitting these competing objectives against each other, Seirênes compels the model to move beyond superficial pattern matching and anchors its capabilities in robust underlying reasoning. This continuous interaction sustains an informative co-evolutionary curriculum as the model improves. Across seven mathematical reasoning benchmarks and model scales from 4B to 30B, Seirênes achieves average gains of +10.2, +9.1, and +7.2 points. Besides, distracting contexts produced by the 4B Seirênes model reduce the accuracy of top-tier closed-source models (GPT and Gemini) by roughly 4--5 points, revealing Seirênes' general ability to uncover reasoning models' blind spots. 

---
# Can LLM Agents Respond to Disasters? Benchmarking Heterogeneous Geospatial Reasoning in Emergency Operations 

**Authors**: Junjue Wang, Weihao Xuan, Heli Qi, Pengyu Dai, Kunyi Liu, Hongruixuan Chen, Zhuo Zheng, Junshi Xia, Stefano Ermon, Naoto Yokoya  

**Link**: [PDF](https://arxiv.org/pdf/2605.11633)  

**Abstract**: Operational disaster response goes beyond damage assessment, requiring responders to integrate multi-sensor signals, reason over road networks, populations and key facilities, plan evacuations, and produce actionable reports. However, prior work largely isolates remote-sensing perception or evaluates generic tool use, leaving the end-to-end workflows of emergency operations underexplored. In this paper, we introduce Disaster Operational Response Agent benchmark (DORA), the first agentic benchmark for end-to-end disaster response: 515 expert-authored tasks across 45 real-world disaster events spanning 10 types, paired with expert-verified, replayable gold trajectories totaling 3,500 tool-call steps. Tasks span five dimensions that cover the operational disaster-response pipeline: disaster perception, spatial relational analysis, rescue and evacuation planning, temporal evolution reasoning, and multi-modal report synthesis. Agents compose calls from a 108-tool MCP library over heterogeneous geospatial data: optical, SAR, and multi-spectral imagery across single-, bi-, and multi-temporal sequences (0.015-10m GSD), complemented by elevation and social vector layers. We comprehensively evaluate 13 frontier LLMs on our benchmark, revealing three persistent challenges: 1) disaster-domain grounding exposes unique failure modes (damage-semantic grounding, sensor-modality mismatch, and disaster-pipeline composition); 2) agents are doubly bottlenecked by tool selection and argument grounding, where gold tool-order hints improve accuracy by only 1.08-4.40%, and alternative scaffolds yield at most a 3.24% gain; 3) compositional fragility scales with trajectory length, the agent-to-gold gap widening from 7% to 56% on long pipelines. DORA establishes a rigorous testbed for operationally reliable disaster-response agents. 

---
# Nice Fold or Hero Call: Learning Budget-Efficient Thinking for Adaptive Reasoning 

**Authors**: Zhaomeng Zhou, Lan Zhang, Junyang Wang, Mu Yuan, Junda Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.11625)  

**Abstract**: Large reasoning models (LRMs) improve problem solving through extended reasoning, but often misallocate test-time compute. Existing efficiency methods reduce cost by compressing reasoning traces or conditioning budget on perceived difficulty, yet largely overlook solvability. As a result, they may spend large budgets on queries beyond the model's capability while compressing hard-but-solvable queries that require deeper reasoning. In this work, we formulate adaptive reasoning as a computational investment under uncertainty, where budget should follow the expected return of reasoning rather than perceived difficulty alone. To instantiate this principle, we propose Budget-Efficient Thinking (BET), a two-stage framework that combines behavioral cold-start with GRPO under an investment-cost-aware reward. By aligning solve-or-fold decisions with rollout-derived solvability, BET learns three behaviors: (1) short solve, answering easy queries concisely; (2) nice fold, abstaining early when continued reasoning has near-zero expected return; and (3) hero call, preserving sufficient compute for hard-but-solvable queries. Across seven benchmarks and three base models, BET reduces reasoning tokens by ~55% on average while achieving overall performance improvements, and transfers zero-shot from mathematical reasoning to scientific QA and logical reasoning with comparable efficiency gains. 

---
# CuSearch: Curriculum Rollout Sampling via Search Depth for Agentic RAG 

**Authors**: Jianghan Shen, Siqi Luo, Xinyu Cheng, Jing Xiong, Yue Li, Jiyao Liu, Jiashi Lin, Yirong Chen, Junjun He  

**Link**: [PDF](https://arxiv.org/pdf/2605.11611)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for training agentic retrieval-augmented generation (RAG) systems from outcome-only supervision. Most existing methods optimize policies from uniformly sampled rollouts, implicitly treating all trajectories as equally informative. However, trajectories differ substantially in search depth and are therefore not equally informative: deeper-search trajectories contain more retrieval decision points and provide denser direct supervision for the retrieval sub-policy. Moreover, this heterogeneity grows over training as the within-batch depth distribution shifts toward higher values, yet uniform rollout sampling remains blind to this shift. To address this, we propose CuSearch, a curriculum rollout sampling framework built on Search-Depth Greedy Allocation (SDGA), a batch-level operator that reallocates a fixed update budget toward deeper-search trajectories. SDGA-Auto always targets the deepest available trajectories in the current batch, yielding an implicit training-aligned curriculum as the depth distribution shifts upward. SDGA-Phase explicitly advances the curriculum threshold as deeper trajectories become sufficiently abundant. Experiments across model types and retrieval frameworks show that CuSearch consistently improves performance, achieving up to 11.8 exact-match points over standard GRPO on ZeroSearch. These results establish per-trajectory search depth as a reliable, annotation-free proxy for retrieval supervision density in RLVR-based agentic RAG training. The code is available at this https URL. 

---
# GAR: Carbon-Aware Routing for LLM Inference via Constrained Optimization 

**Authors**: Disha Sheshanarayana, Rajat Subhra Pal, Manjira Sinha, Tirthankar Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.11603)  

**Abstract**: The growing deployment of large language models (LLMs) makes per-request routing essential for balancing response quality and computational cost across heterogeneous model pools. Current routing methods rarely consider sustainable energy use and CO2 emissions as optimization objectives, despite grid carbon intensity varying by time and region, and models differing significantly in energy consumption. To address this gap, we introduce Green-Aware Routing (GAR), a constrained multi-objective optimization framework that minimizes per-request CO2 emissions subject to explicit accuracy floors and p95-latency service-level objectives (SLOs). GAR employs adaptive constraint optimization through per-dataset floor tuning and incorporates lightweight estimators for correctness, tail latency, and carbon emissions, enabling real-time routing decisions without additional inference passes. We present GAR-PD, a practical online primal-dual routing algorithm for rolling carbon budgets, alongside heuristic variants that achieve high feasibility coverage while limiting accuracy degradation. Comprehensive experiments across standard NLP benchmarks with heterogeneous LLM pools (7B-70B) demonstrate that GAR achieves substantial carbon reductions while maintaining competitive accuracy and p95 latency guarantees, providing a practical, theoretically grounded approach to sustainable LLM inference. 

---
# Native Explainability for Bayesian Confidence Propagation Neural Networks: A Framework for Trusted Brain-Like AI 

**Authors**: Georgios Makridis, Georgios Fatouros, John Soldatos, George Katsis, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2605.11595)  

**Abstract**: The EU Artificial Intelligence Act (Regulation 2024/1689), fully applicable to high-risk systems from August 2026, creates urgent demand for AI architectures that are simultaneously trustworthy, transparent, and feasible to deploy on resource-constrained edge devices. Brain-like neural networks built on the Bayesian Confidence Propagation Neural Network (BCPNN) formalism have re-emerged as a credible alternative to backpropagation-driven deep learning. They deliver state-of-the-art unsupervised representation learning, neuromorphic-friendly sparsity, and existing FPGA implementations that target edge deployment. Despite this momentum, no systematic framework exists for explaining BCPNN decisions -- a gap the present paper fills. We argue that BCPNN is, in the sense of Rudin's interpretable-by-design agenda, an inherently transparent model whose architectural primitives map directly onto established explainable-AI (XAI) families. We make four contributions. First, we propose the first XAI taxonomy for BCPNN. It maps weights, biases, hypercolumn posteriors, structural-plasticity usage scores, attractor dynamics, and input-reconstruction populations onto attribution, prototype, concept, counterfactual, and mechanistic explanation modalities. Second, we introduce sixteen architecture-level explanation primitives (P1--P16), several without analogue in standard ANNs. We provide closed-form algorithms for computing each from quantities the model already maintains. Third, we introduce five design-time Configuration-as-Explanation primitives (Config-P1 to Config-P5) that treat BCPNN hyperparameter choices as an auditable pre-deployment explanation artifact. Fourth, we sketch a roadmap for integration into industrial IoT deployments and discuss EU AI Act alignment, edge feasibility, and Industry 5.0 implications. 

---
# Dual-Temporal LSTM with Hybrid Attention for Airline Passenger Load Factor Forecasting: Integrating Intra-Flight and Inter-Flight Booking Dynamics 

**Authors**: ASM Nazrul Islam, Md. Hasanul Kabir, Md. Liakot Ali, Joydeb Kumar Sana  

**Link**: [PDF](https://arxiv.org/pdf/2605.11569)  

**Abstract**: Accurate short-term demand forecasting is crucial to airline revenue management, yet most existing systems fail to meet this need because current models treat booking data as a single temporal dimension, either the accumulation of bookings for a specific flight or the historical booking profile of the same route. This unidimensional view discards information carried by the other temporal stream and forecasting absolute passenger counts introduces a further operational fragility when change in planned aircraft type alters total seat capacity. This study addresses both limitations. A dual-stream Long Short-Term Memory (LSTM) integrated with attention framework is proposed that simultaneously processes two complementary input sequences: a horizontal sequence capturing intra-flight booking accumulation over the days preceding departure, and a vertical sequence capturing inter-flight booking patterns at fixed days-before-departure offsets across historical flights. Multiple dual-stream architectural variants, combining self-attention, cross-attention, and hybrid attention with concatenation, residual, and gated fusion strategies, are developed and evaluated. Experiments on real-world reservation data from the national airline of Bangladesh, Biman Bangladesh Airlines (BBA), demonstrate that the proposed hybrid model achieves a Mean Absolute Error of 2.8167 and a coefficient of determination ($R^{2}$) of 0.9495, outperforming single-stream baselines, tree-based models, and three prior dual-LSTM architectures applied to the same data. Validation across four flight category pairs; domestic versus international, direct versus transit, high versus low frequency, and short versus mid versus long haul confirms that the model generalizes across operationally diverse route types. Biman Bangladesh Airlines (BBA) has officially integrated this methodology into its operations. 

---
# Hindsight Hint Distillation: Scaffolded Reasoning for SWE Agents from CoT-free Answers 

**Authors**: Shengjie Wang, Guanghe Li, Zonghan Yang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11556)  

**Abstract**: Solving complex long-horizon tasks requires strong planning and reasoning capabilities. Although datasets with explicit chain-of-thought (CoT) rationales can substantially benefit learning, they are costly to obtain. To address this challenge, we propose Hindsight Hint Distillation (HHD), which only requires easy-to-obtain question-answer pairs without CoT annotations. Inspired by how human teachers use student mistakes to provide targeted guidance, HHD synthesizes hindsight hints from the model's own failed self-rollouts and uses them to scaffold on-policy rollouts that successfully complete the tasks. The model then self-distills these scaffolded trajectories and generalizes to new problems without hint guidance. Experiments show that HHD significantly outperforms iterative RFT and trajectory-synthesis baselines, achieving an absolute improvement of 8\% on SWE-bench Verified, while all baselines improve by only around 2\%. Notably, the reasoning strategies induced by HHD generalize effectively to out-of-distribution tasks, yielding the largest gains on SWE-bench Multilingual despite no training on multilingual data. These results demonstrate that HHD can effectively synthesize expert-like reasoning from CoT-free data and substantially improve long-horizon performance. 

---
# Optimal LTLf Synthesis 

**Authors**: Yujian Cao, Sven Schewe, Qiyi Tang, Shufang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11544)  

**Abstract**: Strategy synthesis typically follows an all-or-nothing paradigm, returning unrealisable whenever a specification cannot be guaranteed in an uncertain environment. In this paper, we introduce optimal LTLf synthesis, where the goal is to realise as many objectives as possible from a given specification consisting of multiple objectives, especially for the case that they are not all jointly realisable. We first consider max-guarantee synthesis, which commits to a maximal set of objectives that we can a priori guarantee to realise. We then introduce max-observation synthesis, which maximises a posteriori realised objectives that may be incomparable on different executions. Finally, we present incremental max-observation synthesis, which further improves strategies by exploiting opportunities for stronger guarantees when they arise during an execution. Experimental results show that different variations of optimal synthesis scale broadly equally well, solving a large fraction of the benchmark instances within the given timeout, demonstrating the practical feasibility of the approach. 

---
# Read, Grep, and Synthesize: Diagnosing Cross-Domain Seed Exposure for LLM Research Ideation 

**Authors**: Yunju Choi, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.11532)  

**Abstract**: The discovery of novel methodologies for emerging problems is a continuing cycle in ML, often driven by the migration of techniques across domains. Building on this observation, we ask whether current LLM ideation systems benefit from targeted cross-domain retrieval or simply from exposure to diverse mechanisms. We study this question through PaperGym, a three-stage pipeline: (1) tool-augmented seed extraction via read, grep, and bash over an isolated paper environment, (2) cross-domain seed retrieval via paraphrasing across seven ML domains, and (3) method synthesis from retrieved seeds, each scored by rubric-based judges. Tool-augmented extraction improves specificity, and paraphrase-based retrieval broadens domain coverage. In synthesis, cross-domain retrieval receives more pairwise novelty wins than no-retrieval and same-domain baselines, but shows no significant difference from a random diverse-seed control. These findings suggest LLM ideation systems benefit from diverse seed exposure, but do not yet reliably exploit the semantic reason particular seeds were retrieved. We release the seed library, rubric prompts, and run scripts at this https URL 

---
# Controllable User Simulation 

**Authors**: Guy Tennenholtz, Ofer Meshi, Amir Globerson, Uri Shalit, Jihwan Jeong, Craig Boutilier  

**Link**: [PDF](https://arxiv.org/pdf/2605.11519)  

**Abstract**: Using offline datasets to evaluate conversational agents often fails to cover rare scenarios or to support testing new policies. This has motivated the use of controllable user simulators for targeted, counterfactual evaluation, typically implemented by prompting or fine-tuning large language models. In this work, we formalize controllable simulation as a causal inference problem. By bridging natural language evaluation with off-policy evaluation methodology, we show that the standard practice of training simulators via supervised fine-tuning on post-hoc trajectory labels yields a structurally biased model. Specifically, these labels are inextricably coupled to the data-generating behavior policy, injecting a look-ahead bias that breaks causal consistency. Furthermore, we prove that under policy shift this failure causes the variance of evaluation metrics to explode geometrically, a phenomenon we term controllability collapse. To restore causal consistency, we establish theoretical conditions for accurate simulation and propose practical training mitigations: a priori controls, step-wise dynamic controls, and direct policy-conditioned learning. Empirical evaluation confirms that while standard global controls distort conversational distributions and collapse behavioral diversity, our causally grounded simulators eliminate look-ahead bias, preserve natural variance, and exhibit robust zero-shot generalization to unseen agent behaviors. 

---
# AutoLLMResearch: Training Research Agents for Automating LLM Experiment Configuration -- Learning from Cheap, Optimizing Expensive 

**Authors**: Taicheng Guo, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11518)  

**Abstract**: Effectively configuring scalable large language model (LLM) experiments, spanning architecture design, hyperparameter tuning, and beyond, is crucial for advancing LLM research, as poor configuration choices can waste substantial computational resources and prevent models from realizing their full potential. Prior automated methods are designed for low-cost settings where repeated trial and error is feasible, but scalable LLM experiments are too expensive for such extensive iteration. To our knowledge, no work has addressed the automation of high-cost LLM experiment configurations, leaving this problem labor-intensive and dependent on expert intuition. Motivated by this gap, we propose AutoLLMResearch, an agentic framework that mimics how human researchers learn generalizable principles from low-fidelity experiments and extrapolate to efficiently identify promising configurations in expensive LLM settings. The core challenge is how to enable an agent to learn, through interaction with a multi-fidelity experimental environment that captures the structure of the LLM configuration landscape. To achieve this, we propose a systematic framework with two key components: 1) LLMConfig-Gym, a multi-fidelity environment encompassing four critical LLM experiment tasks, supported by over one million GPU hours of verifiable experiment outcomes; 2) A structured training pipeline that formulates configuration research as a long-horizon Markov Decision Process and accordingly incentivizes cross-fidelity extrapolation reasoning. Extensive evaluation against diverse strong baselines on held-out experiments demonstrates the effectiveness, generalization, and interpretability of our framework, supporting its potential as a practical and general solution for scalable real-world LLM experiment automation. 

---
# Hierarchical LLM-Driven Control for HAPS-Assisted UAV Networks: Joint Optimization of Flight and Connectivity 

**Authors**: Zijiang Yan, Hao Zhou, Wael Jaafar, Jianhua Pei, Ping Wang, Halim Yanikomeroglu, Hina Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2605.11509)  

**Abstract**: Uncrewed aerial vehicles (UAVs) are increasingly deployed in complex networked environments, yet the joint optimization of multi-UAV motion control and connectivity remains a fundamental challenge. In this paper, we study a multi-UAV system operating in an integrated terrestrial and non-terrestrial network (ITNTN) comprising terrestrial base stations and high-altitude platform stations (HAPS). We consider a three-dimensional (3D) aerial highway scenario where UAVs must adapt their motion to ensure collision avoidance, efficient traffic flow, and reliable communication under dynamic and partially observable conditions. We first model the problem as a hierarchical multi-objective partially observable Markov decision process (H-MO-POMDP), capturing the coupling between control and communication objectives. Based on this formulation, we propose a large language model (LLM)-driven hierarchical multi-rate control framework. At the global level, an LLM-based controller on the HAPS performs long-term planning for load balancing and handover decisions. At the local level, each UAV employs a hybrid controller that integrates a slow-timescale LLM for high-level spatial reasoning with a reinforcement learning agent for faster UAV-to-infrastructure (U2I) communication and motion control. We further develop a high-fidelity 3D simulation platform by integrating the gym-pybullet-drones environment with 3GPP-compliant RF/THz channel models. Numerical results demonstrate that the proposed framework significantly outperforms state-of-the-art baselines, achieving a 14% increase in transportation efficiency and a 25% improvement in telecommunication throughput. Additionally, it achieves a 23% reduction in physical collision rates, demonstrating strong handover stability and zero-shot generalization in dynamic scenarios. 

---
# Selective Off-Policy Reference Tuning with Plan Guidance 

**Authors**: Duc Anh Le, Tien-Phat Nguyen, Thien Huu Nguyen, Linh Ngo Van, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.11505)  

**Abstract**: Reinforcement learning with verifiable rewards helps reasoning, but GRPO-style methods stall on hard prompts where all sampled rollouts fail. SORT adds a repair update for those failures without changing rollout generation: it derives a plan from the reference solution, compares token probabilities with and without that plan, and gives higher weight to tokens that become more predictable under plan conditioning. This turns all-wrong prompts into selective, structure-aware learning signals instead of uniform imitation. Across three backbones and eight reasoning benchmarks, SORT improves over GRPO and guidance baselines, with largest gains on weaker models. 

---
# The Evaluation Differential: When Frontier AI Models Recognise They Are Being Tested 

**Authors**: Varad Vishwarupe, Nigel Shadbolt, Marina Jirotka, Ivan Flechais  

**Link**: [PDF](https://arxiv.org/pdf/2605.11496)  

**Abstract**: Recent published evidence from frontier laboratories shows that contemporary AI models can recognise evaluation contexts, latently represent them, and behave differently under those contexts than under deployment-continuous conditions. Anthropic's BrowseComp incident, the Natural Language Autoencoder findings on SWE-bench Verified and destructive-coding evaluations, and the OpenAI / Apollo anti-scheming work all document instances of this phenomenon. We argue that these findings create a claim-validity problem for safety conclusions drawn from frontier evaluations. We introduce the Evaluation Differential (ED), a conditional divergence in a target behavioural property between recognised-evaluation and deployment-continuous contexts, define a normalised effect-size form (nED) for cross-property comparison, and prove that marginal evaluation scores cannot identify ED. We develop a typology of safety claims (ED-stable, ED-degraded, ED-inverted, ED-undetermined) by their warrant-status under documented divergence, and specify TRACE (Test-Recognition Audit for Claim Evaluation), an audit protocol that wraps existing evaluation infrastructure and produces restricted claims rather than capability scores. We apply the framework retrospectively to three publicly documented evaluation incidents and discuss governance implications for system cards, conformity assessment, and the international network of AI safety and security institutes. TRACE does not eliminate adversarial adaptation; it disciplines the claims drawn from evaluation evidence by making explicit the conditions under which that evidence was produced. 

---
# Engagement Process: Rethinking the Temporal Interface of Action and Observation 

**Authors**: Jialian Li, Yuchen Cao, Junhong Liu, Weiran Guo, Xutao Wang, Jiaming Song, Jiahao Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11484)  

**Abstract**: Task completion in digital and physical environments increasingly involves complex temporal interaction, where actions and observations unfold over different time scales rather than align with fixed observation--action steps. To model such interactions, we propose \emph{Engagement Process} (EP), an interaction formalism that inherits the decision-theoretic structure of POMDPs while making time explicit in the action--observation interface. EP represents actions and observations as decoupled event streams along time, rather than updates paired at fixed decision steps. This interface captures single-agent timing issues such as deliberation latency, delayed feedback, and persistent actions, while supporting richer agent-side organization, multi-rate coordination, and compositional interaction among subsystems. Across toy, LLM-agent, and learning experiments, EP exposes temporal behaviors hidden by step-based interfaces and enables policies to adapt under explicit time costs. 

---
# FibQuant: Universal Vector Quantization for Random-Access KV-Cache Compression 

**Authors**: Namyoon Lee, Yongjune Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.11478)  

**Abstract**: Long-context inference is increasingly a memory-traffic problem. The culprit is the key--value (KV) cache: it grows with context length, batch size, layers, and heads, and it is read at every decoding step. Rotation-based scalar codecs meet this systems constraint by storing a norm, applying a shared random rotation, and quantizing one coordinate at a time. They are universal and random-access, but they discard the geometry created by the normalization step. After a Haar rotation, a block of $k$ consecutive coordinates is not a product source; it is a spherical-Beta source on the unit ball. We introduce \textsc{FibQuant}, a universal fixed-rate vector quantizer that keeps the same normalize--rotate--store interface while replacing scalar tables by a shared radial--angular codebook matched to this canonical source. The codebook combines Beta-quantile radii, Fibonacci\,/\,Roberts--Kronecker quasi-uniform directions, and multi-restart Lloyd--Max refinement. We prove that the resulting vector code strictly improves on its scalar product specialization at matched rate, with a high-rate gain that separates into a cell-shaping factor and a density-matching factor. The same construction gives a dense rate axis, including fractional-bit and sub-one-bit operating points, without calibration or variable-length addresses. On GPT-2 small KV caches, \textsc{FibQuant} traces a memory--fidelity frontier from $5\times$ compression at $0.99$ attention cosine similarity to $34\times$ at $0.95$. End-to-end on TinyLlama-1.1B, it is within $0.10$ perplexity of fp16 at $4\times$ compression and has $3.6\times$ lower perplexity than scalar \textsc{TurboQuant} at $b = 2$ ($8\times$ compression), where scalar random-access quantization begins to fail. 

---
# TOPPO: Rethinking PPO for Multi-Task Reinforcement Learning with Critic Balancing 

**Authors**: Yuanpeng Li, Gefei Lin, Annie Qu, Rui Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11473)  

**Abstract**: Soft Actor-Critic (SAC) and its variants dominate Multi-Task Reinforcement Learning (MTRL) due to their off-policy sample efficiency, while on-policy methods such as Proximal Policy Optimization (PPO) remain underexplored. We diagnose that PPO in MTRL suffers from a previously overlooked issue: critic-side gradient ill-conditioning, which may cause tail tasks to stall while easy tasks dominate the value function's updates. To address this, we propose TOPPO (Tail-Optimized PPO), a reformulation of PPO via Critic Balancing -- a set of modules that improve gradient conditioning and balance learning dynamics across tasks. Unlike prior approaches that rely on modular architectures or large models, TOPPO targets the optimization bottleneck within PPO itself. Empirically, TOPPO achieves stronger mean and tail-task performance than published SAC-family and ARS-family baselines while using substantially fewer parameters and environment steps on Meta-World+ benchmark. Notably, TOPPO matches or surpasses strong SAC baselines early in training and maintains superior performance at full budget. Ablations confirm the effectiveness of each module in TOPPO and provide insights into their interactions. Our results demonstrate that, with proper optimization, on-policy methods can rival or exceed off-policy approaches in MTRL, challenging the prevailing reliance on SAC and highlighting critic-side gradient conditioning as the central bottleneck. 

---
# CAMPA: Efficient and Aligned Multimodal Graph Learning via Decoupled Propagation and Aggregation 

**Authors**: Daohan Su, Hao Liu, Xunkai Li, Yinlin Zhu, Xiong Yongfu, Yi Liu, Hongchao Qin, Rong-Hua Li, Guoren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11468)  

**Abstract**: Multimodal Graph Neural Networks (MGNNs) have shown strong potential for learning from multimodal attributed graphs, yet most existing approaches rely on tightly coupled architectures that suffer from prohibitive computational overhead. In this paper, we present a systematic empirical analysis showing that decoupled MGNNs are substantially more efficient and scalable for large-scale graph learning. However, we identify a critical bottleneck in existing decoupled pipelines, namely modal conflict, which arises in both the propagation and aggregation stages. Specifically, independent multi-hop diffusion causes cross-modal semantic divergence during propagation, while naive fusion fails to align multi-hop feature trajectories during aggregation, jointly limiting effective representation learning. To address this challenge, we propose CAMPA, a Cross-modal Aligned Multimodal Propagation & Aggregation framework for decoupled multimodal graph learning. Concretely, CAMPA introduces a two-stage alignment mechanism: (1) cross-modal aligned propagation, which injects cross-modal similarity priors into message passing to preserve semantic consistency without additional parameter overhead; (2) trajectory aligned aggregation, which leverages trajectory-level self-attention and cross-attention to capture and align long-range dependencies across modalities and hops. Extensive experiments on diverse benchmark datasets and tasks demonstrate that CAMPA consistently outperforms strong coupled and decoupled baselines while preserving the efficiency advantages of the decoupled paradigm. 

---
# Breaking $\textit{Winner-Takes-All}$: Cooperative Policy Optimization Improves Diverse LLM Reasoning 

**Authors**: Haoxuan Chen, Tianming Liang, Wei-Shi Zheng, Jian-Fang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11461)  

**Abstract**: Reinforcement learning with verifiers (RLVR) has become a central paradigm for improving LLM reasoning, yet popular group-based optimization algorithms like GRPO often suffer from exploration collapse, where the models prematurely converge on a narrow set of high-scoring patterns, lacking the ability to explore new solutions. Recent efforts attempt to alleviate this by adding entropy regularization or diversity bonus. However, these approaches do not change the \textit{winner-takes-all} nature, where rollouts still compete for individual advantage rather than cooperating for maximizing global diversity. In this work, we propose Group Cooperative Policy Optimization (GCPO), which shifts the training paradigm from rollout competition to team cooperation. Specifically, GCPO replaces independent rollout scoring with team-level credit assignment: a rollout is rewarded by how much it contributes to the team's valid solution coverage, rather than its individual accuracy. This coverage is described as a determinant volume over reward-weighted semantic embeddings, where only correct and non-redundant rollouts contribute to this volume. During advantage estimation, GCPO redistributes the collective team reward to each single rollout according to its average marginal contribution to the team. This cooperative training paradigm routes optimization toward non-redundant correct reasoning paths. Experiments across multiple reasoning benchmarks demonstrate that GCPO significantly improves both reasoning accuracy and solution diversity over existing approaches. Code will be released at $\href{this https URL}{this}$. 

---
# Adaptive Teacher Exposure for Self-Distillation in LLM Reasoning 

**Authors**: Zihao Han, Tiangang Zhang, Huaibin Wang, Yilun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.11458)  

**Abstract**: On-policy self-distillation has become a strong recipe for LLM reasoning, where a privileged teacher supervises the student's own rollouts while conditioning on the reference solution. A design choice shared by nearly all such methods, however, has gone unquestioned: the teacher always sees the full reference reasoning. We argue that this default itself is part of the problem and identify a teacher-side exposure mismatch: when the teacher conditions on reasoning far beyond the student's current competence, the resulting token targets become too strong to absorb. A controlled fixed-exposure sweep makes this concrete on two fronts: 1) full exposure is not reliably the best choice, and 2) student-teacher mismatch grows monotonically as the teacher sees more privileged reasoning. This motivates treating teacher exposure not as a fixed hyperparameter but as a learnable training-time control variable. We therefore propose Adaptive Teacher Exposure for Self-Distillation (ATESD). ATESD models the reveal ratio with a lightweight Beta-policy controller conditioned on compact training-state statistics, and uses one sampled exposure for a short hold window of student updates. To make this exposure controller learnable, we optimize it with a discounted learning-progress reward that scores each held decision by its effect on the student's future improvement rather than its immediate loss change, addressing the delayed credit assignment induced by on-policy distillation. Experiments on AIME 24, AIME 25, and HMMT 25 across Qwen3-{1.7B, 4B, 8B} show that ATESD consistently outperforms competitive self-distillation and RL baselines, improving over OPSD by +0.95, +2.05, and +2.33 Average@12 points respectively, and establishing adaptive teacher exposure as an effective new axis for reasoning self-distillation. 

---
# A Mechanistic Investigation of Supervised Fine Tuning 

**Authors**: Ruhaan Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2605.11426)  

**Abstract**: The cosine similarity between a large language model's hidden activations before and after Supervised Fine-Tuning (SFT) remains very high. This, at first glance, suggests that SFT leaves the model's activation geometry largely undisturbed. However, projecting both sets of activations through a Sparse Autoencoder (SAE) pretrained on the base model reveals that the underlying sparse latents diverge significantly. We introduce a novel investigative pipeline which utilizes these pretrained SAEs as a high-resolution diagnostic tool to mechanistically investigate the drivers of this representational divergence. Through our analytical pipeline, we discover task-specific and layer-specific distributions of the precise semantic features that are systematically altered during supervised fine-tuning. We additionally identify a layer-wise update profile specific to safety alignment. All code, experimental scripts, and analysis files associated with this work are publicly available at: this https URL. 

---
# Under the Hood of SKILL.md: Semantic Supply-chain Attacks on AI Agent Skill Registry 

**Authors**: Shoumik Saha, Kazem Faghih, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2605.11418)  

**Abstract**: Autonomous AI agents increasingly extend their capabilities through Agent Skills: modular filesystem packages whose this http URL files describe when and how agents should use them. While this design enables scalable, on-demand capability expansion, it also introduces a semantic supply-chain risk in which natural-language metadata and instructions can affect which skills are admitted, surfaced, selected, and loaded. We study this http URL - only attacks across three registry-facing stages of the Agent Skill lifecycle, using real ClawHub skills and realistic registry mechanisms. In Discovery, short textual triggers can manipulate embedding-based retrieval and improve adversarial skill visibility, achieving up to 86% pairwise win rate and 80% Top-10 placement. In Selection, description-only framing biases agents toward functionally equivalent adversarial variants, which are selected in 77.6% of paired trials on average. In Governance, semantic evasion strategies cause malicious skills to avoid a blocking verdict in 36.5%-100% of cases. Overall, our results show that this http URL is not passive documentation but operational text that shapes which third-party capabilities agents find, trust, and use. 

---
# What Do EEG Foundation Models Capture from Human Brain Signals? 

**Authors**: Ling Tang, Qian Chen, Jilin Mei, Houshi Xu, Quanshi Zhang, Jing Shao, Na Zou, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11410)  

**Abstract**: Clinical electroencephalogram (EEG) analysis rests on a hand-crafted feature catalog refined over decades, \emph{e.g.,} band power, connectivity, complexity, and more. Modern EEG foundation models bypass this catalog, learn directly from raw signals via self-supervised pretraining, and match or outperform feature-engineered baselines on most clinical benchmarks. Whether the two representations align is an open question, which we decompose into three sub-questions: \emph{what does the model learn}, \emph{what does the model use}, and \emph{how much can be explained}. We answer them with layer-wise ridge probing, LEACE-style cross-covariance subspace erasure, and a transparent classifier benchmarked against a random-feature baseline. The audit covers three foundation models (CSBrain, CBraMod, LaBraM), five clinical tasks (MDD, Stress, ISRUC-Sleep, TUSL, Siena), and a 6-family 63-feature lexicon. Of the $945$ (model, task, feature) units, $648$ ($68.6\%$) are representation-causal and $199$ ($21.1\%$) are encoded-only. Across tasks, $50$ features qualify as universal candidates with strong support (all three architectures RC) in two or more tasks. Frequency-domain features dominate, but the other five families each contribute substantial causal mass. Confirmed features recover, on average, $79.3\%$ of the foundation model's advantage over the random baseline, with a clean task gradient (MDD $\approx 0.99$ down to Stress $\approx 0.56$): tasks near ceiling are almost fully recovered by the lexicon, while harder tasks leave a non-trivial residual that pinpoints a concrete target for future concept discovery. 

---
# Attributing Emergence in Million-Agent Systems 

**Authors**: Ling Tang, Jilin Mei, Qian Chen, Qihan Ren, Linfeng Zhang, Quanshi Zhang, Jing Shao, Xia Hu, Dongrui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11404)  

**Abstract**: Large language models (LLMs) can simulate human-like reasoning and decision-making in individual agents. LLM-powered multi-agent systems (MAS) combine such agents to simulate population-scale social phenomena such as polarization, information cascades, and market panics. Such studies require attributing macro emergence to individual agents, but existing axiomatic methods scale combinatorially in $N$ and have been confined to $N \lesssim 10^3$, while the phenomena they explain occur at $N \geq 10^6$. We address this gap by adapting Aumann--Shapley path-integral attribution to LLM-powered MAS at million-agent scale; the resulting method satisfies all four axioms, runs four to five orders of magnitude faster than sampled Shapley on the same hardware. We use this method to test the scale gap empirically: across 14 days of public Bluesky data ($1{,}671{,}587$ active users), we compute the attribution at both full scale and the visibility-biased $N = 10^2$ convenience sample used by small-scale studies, and the two disagree structurally. At full scale the long tail and middle tier jointly carry the majority; the biased small panel attributes almost everything to a few high-follower accounts. We then prove that under any nonlinear macro indicator the disagreement cannot be reduced by post-hoc rescaling: an Attribution Scaling Bias theorem shows that no global rescaling factor can reconcile small-scale and full-scale attribution. Full-scale attribution is therefore not a methodological choice but a theoretical requirement for any nonlinear macro indicator. 

---
# AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment 

**Authors**: Robin Linzmayer, Georgianna Lin, Di Coneybeare, Jason Chu, Trudi Cloyd, Manish Garg, Miles Gordon, Elizabeth Hartofilis, Benjamin Hong, Ashraf Hussain, Eugene Y. Kim, Oluchi Iheagwara King, Ross McCormack, Erica Olsen, John K. Riggins Jr, Mustafa N. Rasheed, Dana L. Sacco, Vinay Saggar, Osman R. Sayan, Amit Shembekar, Janice Shin-Kim, Wendy W. Sun, Bernard P. Chang, David Kessler, Noémie Elhadad  

**Link**: [PDF](https://arxiv.org/pdf/2605.11398)  

**Abstract**: We introduce AcuityBench, a benchmark for evaluating whether language models identify the appropriate urgency of care from user medical presentations. Existing health benchmarks emphasize medical question answering, broad health interactions, or narrow workflow-specific triage tasks, but they do not offer a unified evaluation of acuity identification across these settings. AcuityBench addresses this gap by harmonizing five public datasets spanning user conversations, online forum posts, clinical vignettes, and patient portal messages under a shared four-level acuity framework ranging from home monitoring to immediate emergency care. The benchmark contains 914 cases, including 697 consensus cases for standard accuracy evaluation and 217 physician-confirmed ambiguous cases for uncertainty-aware evaluation. It supports two complementary task formats: explicit four-way classification in a QA setting, and free-form conversational responses evaluated with a rubric-based judge anchored to the same framework. Across 12 frontier proprietary and open-weight models, we find substantial variation in clear-case acuity accuracy and error direction. Comparing task formats reveals a systematic tradeoff: conversational responses reduce over-triage but increase under-triage relative to QA, especially in higher-acuity cases. In ambiguous cases, no model closely matches the distribution of physician judgments, and model predictions are more concentrated than expert clinical uncertainty. We also compare expert and model adjudication on a subset of maximally ambiguous cases, using those cases to examine the role of clinical uncertainty in label disagreement. Together, these results position acuity identification as a distinct safety-critical capability and show that AcuityBench enables systematic comparison and stress-testing of how well models guide users to the right level of care in real-world health use. 

---
# Transformer Interpretability from Perspective of Attention and Gradient 

**Authors**: Yongjin Cui, Xiaohui Fan, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11392)  

**Abstract**: Although researchers' attention is more focused on the performance of Transformer models, the interpretation of Transformer can never be ignored. Gradient is widely utilized in Transformer interpretation. From the perspective of attention and gradient, we conduct an in-depth study of Transformer interpretation and propose a method to achieve it by guiding the gradient direction, or more precisely, the attention direction. The method enables more comprehensive interpretation of feature regions, offers detail interpretation, and helps to better understand Transformer mechanism. Leveraging the difference in how Vision Transformer (ViT) and humans perceive images, we alter the class of an image in a way that is almost imperceptible to the human eye. This class rewriting phenomenon may potentially pose security risks in certain scenarios. 

---
# Revisiting Privacy Preservation in Brain-Computer Interfaces: Conceptual Boundaries, Risk Pathways, and a Protection-Strength Grading Framework 

**Authors**: Lei Sun, Xiuqing Mao, Shuai Zhang, Qingyu Zeng, Min Zhao, Jiyuan Li, Wenle Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11386)  

**Abstract**: Brain-computer interfaces (BCIs) are moving rapidly from laboratory research into clinical, edge, and real-world settings. Under ISO/IEC 8663:2025, a BCI is a direct communication link between central nervous system activity and external software or hardware systems. This link expands privacy risk beyond raw neural-signal leakage: neural data, derived representations, model assets, and decoded outputs can be re-associated with individuals across collection, transmission, storage, training, inference, and feedback, or used to infer information beyond what a task requires. Starting from the general BCI paradigm, this review deffnes privacy-protection boundaries, protection objects, and the relationship between user data privacy and model privacy within a shared risk pathway. It then proposes a three-dimensional framework - protection object, lifecycle stage, and dominant protection-strength level - to classify existing work into four levels of protection strength. Finally, mental privacy and neuroethical risks are treated as open issues, emphasizing that BCI privacy protection should not only obscure data but also disentangle task-irrelevant sensitive information while preserving downstream utility. Keywords: Brain-computer interface, Neural data privacy, User data privacy, Model privacy, Disentanglement of task-irrelevant sensitive information, Protection-strength grading, Neuroethical risks 

---
# LLM-X: A Scalable Negotiation-Oriented Exchange for Communication Among Personal LLM Agents 

**Authors**: Giuliano Lorenzoni, Paulo Alencar, Donald Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11376)  

**Abstract**: We propose a personal-LLM exchange (LLM-X), a scalable negotiation-oriented environment that enables direct, structured communication across populations of personal agents (LLMs), each representing an individual user. Unlike existing tool-centric protocols that focus on agent-API interaction, LLM-X introduces a message bus and routing substrate for LLM-to-LLM coordination with guarantees around schema validity and policy enforcement. We contribute: (1) an architecture for LLM-X comprising federated gateways, topic-based routing, and policy enforcement; (2) a typed message protocol supporting capability negotiation and contract-net-style coordination; and (3) the first empirical evaluation of LLM-based multi-agent negotiation at scale. Experiments span 5, 9, and 12 agents, under distinct negotiation policies (Low, Medium, High), and across both short-run (minutes) and long-run (2h, 12h) load conditions. Results highlight clear policy-performance trade-offs: stricter policies improve robustness and fairness but increase latencies and message volume. Extended runs confirm that LLM-X remains stable under sustained load, with bounded latency drift. 

---
# Causal Algorithmic Recourse: Foundations and Methods 

**Authors**: Drago Plecko, Collin Wang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2605.11373)  

**Abstract**: The trustworthiness of AI decision-making systems is increasingly important. A key feature of such systems is the ability to provide recommendations for how an individual may reverse a negative decision, a problem known as algorithmic recourse. Existing approaches treat recourse outcomes as counterfactuals of a fixed unit, ignoring that real-world recourse involves repeated decisions on the same individual under possibly different latent conditions. We develop a causal framework that models recourse as a process over pre- and post-intervention outcomes, allowing for partial stability and resampling of latent variables. We introduce post-recourse stability conditions that enable reasoning about recourse from observational data alone, and develop a copula-based algorithm for inferring the effects of recourse under these conditions. For settings where paired observations of the same individual before and after intervention are available (called recourse data), we develop methods for inferring copula parameters and performing goodness-of-fit testing. When the copula model is rejected, we provide a distribution-free algorithm for learning recourse effects directly from recourse data. We demonstrate the value of the proposed methods on real and semi-synthetic datasets. 

---
# Causal Bias Detection in Generative Artifical Intelligence 

**Authors**: Drago Plecko  

**Link**: [PDF](https://arxiv.org/pdf/2605.11365)  

**Abstract**: Automated systems built on artificial intelligence (AI) are increasingly deployed across high-stakes domains, raising critical concerns about fairness and the perpetuation of demographic disparities that exist in the world. In this context, causal inference provides a principled framework for reasoning about fairness, as it links observed disparities to underlying mechanisms and aligns naturally with human intuition and legal notions of discrimination. Prior work on causal fairness primarily focuses on the standard machine learning setting, where a decision-maker constructs a single predictive mechanism $f_{\widehat Y}$ for an outcome variable $Y$, while inheriting the causal mechanisms of all other covariates from the real world. The generative AI setting, however, is markedly more complex: generative models can sample from arbitrary conditionals over any set of variables, implicitly constructing their own beliefs about all causal mechanisms rather than learning a single predictive function. This fundamental difference requires new developments in causal fairness methodology. We formalize the problem of causal fairness in generative AI and unify it with the standard ML setting under a common theoretical framework. We then derive new causal decomposition results that enable granular quantification of fairness impacts along both (a) different causal pathways and (b) the replacement of real-world mechanisms by the generative model's mechanisms. We establish identification conditions and introduce efficient estimators for causal quantities of interest, and demonstrate the value of our methodology by analyzing race and gender bias in large language models across different datasets. 

---
# CVEvolve: Autonomous Algorithm Discovery for Unstructured Scientific Data Processing 

**Authors**: Ming Du, Xiangyu Yin, Yanqi Luo, Dishant Beniwal, Songyuan Tang, Hemant Sharma, Mathew J. Cherukara  

**Link**: [PDF](https://arxiv.org/pdf/2605.11359)  

**Abstract**: Scientific data processing often requires task-specific algorithms or AI models, creating a barrier for domain scientists who need to analyze their data but may not have extensive computing or image-processing expertise. This barrier is especially pronounced when data are noisy, have a high dynamic range, are sparsely labeled, or are only loosely specified. We introduce CVEvolve, an autonomous agentic harness with a zero-code interface for scientific data-processing algorithm discovery. CVEvolve combines a multi-round search strategy with tools for code execution, evaluation implementation, history management, holdout testing, and optional inspection of scientific data and visual outputs. The search alternates between discovery and improvement actions, and uses lineage-aware stochastic candidate sampling to balance exploration and exploitation. We demonstrate CVEvolve on x-ray fluorescence microscopy image registration, Bragg peak detection, and high-energy diffraction microscopy image segmentation. Across these tasks, CVEvolve discovers algorithms that improve over baseline methods, while holdout test tracking helps identify candidates that generalize better than later over-optimized alternatives. These results show that zero-code, autonomous LLM-powered algorithm development can help domain scientists turn unstructured scientific image data into practical algorithms and downstream scientific discoveries. 

---
# CPEMH: An Agentic Framework for Prompt-Driven Behavior Evaluation and Assurance in Foundation-Model Systems for Mental Health Screening 

**Authors**: Giuliano Lorenzoni, Ivens Portugal, Paulo Alencar, Donald Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11341)  

**Abstract**: This paper presents CPEMH, an agentic framework designed to evaluate prompt-driven behavior in foundation-model systems operating on transcript-based datasets for mental-health screening. CPEMH serves as an engineering methodology for behavioral assurance in large-scale language systems, introducing an orchestrated architecture that autonomously performs the design, evaluation, and selection of prompt strategies, enabling systematic control of behavioral variability across contexts. Its modular agentic design, combining orchestrator, inference, and evaluation agents, ensures traceability, reproducibility, and robustness throughout the prompting lifecycle. A case study on automated depression screening from interview transcripts demonstrates the framework's capacity to stabilize and audit foundation-model behavior in conversational and clinically sensitive domains. Lessons learned emphasize the role of modular orchestration in behavioral assurance, the prioritization of stability over architectural complexity, and the integration of F1, bias, and robustness as core acceptance criteria. 

---
# Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights 

**Authors**: Wenbo Chen, Veena Padmanabhan, Tootiya Giyahchi, Elaine Wong, Leman Akoglu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11330)  

**Abstract**: Hallucination, broadly referring to unfaithful, fabricated, or inconsistent content generated by LLMs, has wide-ranging implications. Therefore, a large body of effort has been devoted to detecting LLM hallucinations, as well as designing benchmark datasets for evaluating these detectors. In this work, we first establish a desiderata of properties for hallucination detection benchmarks (HDBs) to exhibit for effective evaluation. A critical look at existing HDBs through the lens of our desiderata reveals that none of them exhibits all the properties. We identify two largest gaps: (1) RAG-based grounded benchmarks with long context are severely lacking (partly because length impedes human annotation); and (2) Existing benchmarks do not make available realistic label noise for stress-testing detectors although real-world use-cases often grapple with label noise due to human or automated/weak annotation. To close these gaps, we build and open-source a new RAG-based HDB called T RIVIA+ that underwent a rigorous human annotation process. Notably, our benchmark exhibits all desirable properties including (1) T RIVIA+ contains samples with the longest context in the literature; and (2) we design and share four sets of noisy labels with different, both sample-dependent and sampleindependent, noise schemes. Finally, we perform experiments on RAG-based HDBs, including our T RIVIA+, using popular SOTA detectors that reveal new insights: (i) ample room remains for current detectors to reach the performance ceiling on RAG-based HDBs, (ii) the basic LLM-as-a-Judge baseline performs competitively, and (iii) label noise hinders detection performance. We expect that our findings, along with our proposed benchmark 1 , will motivate and foster needed research on hallucination detection for RAG-based tasks. 

---
# Constraint-Data-Value-Maximization: Utilizing Data Attribution for Effective Data Pruning in Low-Data Environments 

**Authors**: Danilo Brajovic, David A. Kreplin, Marco F. Huber  

**Link**: [PDF](https://arxiv.org/pdf/2605.11312)  

**Abstract**: Attributing model behavior to training data is an evolving research field. A common benchmark is data removal, which involves eliminating data instances with either low or high values, then assessing a model's performance trained on the modified dataset. Many existing studies leverage Shapley-based data values for this task. In this paper, we demonstrate that these data values are not optimally suited for pruning low-value data when only a limited amount of data remains. To address this limitation, we introduce the Constraint-Data-Value-Maximization (CDVM) approach, which effectively utilizes data attributions for pruning in low-data scenarios. By casting pruning as a constrained optimization that both maximizes total influence and penalizes excessive per-test contributions, CDVM delivers robust performance when only a small fraction of the data is retained. On the OpenDataVal benchmark, CDVM shows strong performance and competitive runtime. 

---
# LatentRouter: Can We Choose the Right Multimodal Model Before Seeing Its Answer? 

**Authors**: Xueqi Cheng, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11301)  

**Abstract**: Multimodal large language models (MLLMs) have heterogeneous strengths across OCR, chart understanding, spatial reasoning, visual question answering, cost, and latency. Effective MLLM routing therefore requires more than estimating query difficulty: a router must match the multimodal requirements of the current image-question input with the capabilities of each candidate model. We propose LatentRouter, a router that formulates MLLM routing as counterfactual multimodal utility prediction. Given an image-question query, LatentRouter extracts learned multimodal routing capsules, represents each candidate MLLM with a model capability token, and performs latent communication between these states to estimate how each model would perform if selected. A distributional outcome head predicts model-specific counterfactual quality, while a bounded capsule correction refines close decisions without allowing residual signals to dominate the prediction. The resulting utility-based policy supports performance-oriented and performance-cost routing, and handles changing candidate pools through shared per-model scoring with availability masking. Experiments on MMR-Bench and VL-RouterBench show that LatentRouter outperforms fixed-model, feature-level, and learned-router baselines. Additional analyses show that the gains are strongest on multimodal task groups where model choice depends on visual, layout-sensitive, or reasoning-oriented requirements, and that latent communication is the main contributor to the improvement. The code is available at: this https URL. 

---
# Template-as-Ontology: Configurable Synthetic Data Infrastructure for Cross-Domain Manufacturing AI Validation 

**Authors**: Grama Chethan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11259)  

**Abstract**: LLarge language model (LLM)-based AI agents deployed in manufacturing environments require populated, schema-correct data for validation, yet production MES data is proprietary, privacy-encumbered, and vendor-specific. This paper introduces the Template-as-Ontology principle: a single Python configuration module (700-770 lines, 45 validated exports) serves simultaneously as the specification for a time-stepped manufacturing simulator and as the runtime domain schema for AI analytics tools, producing alignment by construction rather than integration. We formally define the domain template as a typed relational configuration schema and prove that structural alignment between simulation and tool layers is guaranteed by single-source consumption. A five-layer pipeline--simulation, PostgreSQL, CDC/Iceberg lakehouse, star schema, and 12 parameterized AI tools--generates causally coherent, MES-shaped data spanning 66 entity types across four operational domains mapped to ISA-95/IEC 62264. We validate the architecture with six industry templates (aerospace, pharma, automotive, electronics, beverages, warehousing) running on identical framework code. Calibration experiments (60 runs, 10 seeds per template) confirm parametric controllability: observed KPIs fall within configured ranges across all templates. A controlled hallucination experiment (72 tool invocations, Qwen3-32B) demonstrates that ontology-constrained parameters eliminate tool-parameter fabrication (0% constrained vs. 43% unconstrained hallucination rate for the evaluated model, Fisher's exact test p < 10^-12); the 0% constrained rate is an architectural guarantee that holds for any model. The framework provides a reusable data layer for discrete manufacturing AI validation. 

---
# Unlocking LLM Creativity in Science through Analogical Reasoning 

**Authors**: Andrew Shen, Shaul Druckmann, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11258)  

**Abstract**: Autonomous science promises to augment scientific discovery, particularly in complex fields like biomedicine. However, this requires AI systems that can consistently generate novel and diverse solutions to open-ended problems. We evaluate LLMs on the task of open-ended solution generation and quantify their tendency to mode collapse into low-diversity generations. To mitigate this mode collapse, we introduce analogical reasoning (AR) as a new approach to solution generation. AR generates analogies to cross-domain problems based on shared relational structure, then uses those analogies to search for novel solutions. Compared to baselines, AR discovers significantly more diverse generations (improving solution diversity metrics by 90-173%), generates novel solutions over 50% of the time (compared to as little as 1.6% for baselines), and produces high-quality analogies. To validate the real-world feasibility of AR, we implement AR-generated solutions across four biomedical problems, yielding consistent quantitative gains. AR-generated approaches achieve a nearly 13-fold improvement on distributional metrics for perturbation effect prediction, outperform all baselines on AUPRC when predicting cell-cell communication, infer brain region interactions with a high Spearman correlation ($\rho$=0.729) to published methods, and establish state-of-the-art performance on 2 datasets for oligonucleotide property prediction. The novel and diverse solutions produced by AR can be used to augment the search space of existing solution generation methods. 

---
# The Semantic Training Gap: Ontology-Grounded Tool Architectures for Industrial AI Agent Systems 

**Authors**: Grama Chethan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11234)  

**Abstract**: Large language model (LLM)-based AI agents are increasingly deployed in manufacturing environments for analytics, quality management, and decision support. These agents demonstrate statistical fluency with domain terminology but lack grounded understanding of operational semantics -- the relational structure that connects equipment identifiers, process parameters, failure codes, and regulatory constraints within a specific production context. This paper identifies and formalizes the semantic training gap: a structural disconnect between how AI systems acquire domain vocabulary through training and how manufacturing operations define meaning through ontological relationships. We demonstrate that this gap causes operationally incorrect outputs even when model responses are linguistically precise, and that in multi-agent configurations it produces a compounding failure mode we term semantic drift. To close this gap, we present an architecture that embeds manufacturing ontology directly into the AI tool layer as a typed relational configuration, enforcing semantic constraints at runtime rather than relying on model training. The architecture is formalized as a three-operation interface contract -- resolve, contextualize, annotate -- with invariants enforced by an AIOps orchestration layer. In a controlled experiment across six industry configurations (72 tool invocations using Qwen3-32B), unconstrained tool parameters produced a 43% hallucination rate for domain identifiers; ontology-grounded parameters reduced this to 0%. We validate the approach through a digital twin analytics platform demonstrating that a single codebase with domain-specific ontology configurations eliminates tool-call hallucination and achieves cross-domain configurability without application code changes. 

---
# Rethinking LLMOps for Fraud and AML: Building a Compliance-Grade LLM Serving Stack 

**Authors**: Prathamesh Vasudeo Naik, Naresh Dintakurthi, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11232)  

**Abstract**: Fraud detection and anti-money-laundering (AML) compliance are high-value domains for large language models (LLMs), but their serving requirements differ sharply from generic chat workloads. Compliance prompts are often prefix-heavy, schema-constrained, and evidence-rich, combining reusable policy instructions, risk taxonomies, transaction or document context, and short structured outputs such as JSON labels or risk factors. These properties make prefix reuse, KV-cache efficiency, runtime tuning, model orchestration, and output validation first-order systems concerns. This paper introduces a workload-aware LLMOps stack for fraud and AML workloads using self-hosted open-weight models such as Meta Llama and Alibaba Qwen. The stack combines vLLM-style runtime tuning, PagedAttention, Automatic Prefix Caching, multi-adapter serving, adapter and prompt-length-aware batching, sleep/wake lifecycle management, speculative decoding, and optional prefill/decode disaggregation. To avoid exposing institution-specific data, the reproducibility track converts public synthetic AML datasets, including IBM AML and SAML-D, into prefix-heavy compliance prompts with reusable policy text, transaction evidence, typology definitions, and schema-constrained outputs. We also incorporate an LLM-as-judge quality gate using deterministic compliance checks, reference metrics, expert-adjudicated calibration data where available, and multi-judge rubric scoring. Across public-synthetic AML workloads and controlled serving benchmarks, workload-aware tuning improved throughput from 612-650 to 3,600 requests/hour, reduced P99 latency from 31-38 seconds to 6.4-8.7 seconds, and increased GPU utilization from 12% to 78%. These results show that regulated LLM performance is a workload-design, serving-optimization, and quality-gating problem, not only a model-selection problem. 

---
# PIVOT: Bridging Planning and Execution in LLM Agents via Trajectory Refinement 

**Authors**: Tuo Zhang, Alin-Ionut Popa, Yan Xu, Rui Song, Dimitrios Dimitriadis  

**Link**: [PDF](https://arxiv.org/pdf/2605.11225)  

**Abstract**: Large language model (LLM)-based agents frequently generate seemingly coherent plans that fail upon execution due to infeasible actions, constraint violations, and compounding errors over extended horizons. PIVOT (Plan-Inspect-eVOlve Trajectories) addresses this plan-execution misalignment through a self-supervised framework that treats trajectories as optimizable objects iteratively refined via environment interaction. The framework comprises four stages: PLAN generates candidate trajectories; INSPECT executes them and computes structured losses with textual gradients encoding plan-execution discrepancies; EVOLVE applies these signals to produce improved trajectories; and VERIFY performs a final global check against task constraints. A monotonic acceptance process ensures a non-decreasing solution quality. Empirical evaluations on DeepPlanning and GAIA demonstrate state-of-the-art performance: with human-in-the-loop (HITL) feedback, PIVOT establishes a strong upper bound up to 94% relative improvement in constraint satisfaction, while its fully autonomous variant retains substantial gains, showing that the core trajectory-refinement mechanism remains effective without external supervision. At the same time, PIVOT remains computationally efficient, requiring up to 3x to 5x fewer tokens than competing refinement methods. These findings establish that (self- or human-supervised) feedback-based trajectory optimization is a principled methodology for mitigating plan-execution gaps in autonomous agent systems. 

---
# Do Vision-Language-Models show human-like logical problem-solving capability in point and click puzzle games? 

**Authors**: Dominik Helfenstein, Marco Menner, Maximilian Triebel  

**Link**: [PDF](https://arxiv.org/pdf/2605.11223)  

**Abstract**: Vision-Language(-Action) Models (VLMs) are increasingly applied to interactive environments, yet existing benchmarks often overlook the complex physical reasoning required for point-and-click puzzle games. This paper introduces Vision-Language Against The Incredible Machine (VLATIM), a benchmark designed to evaluate human-like logical problem-solving capabilities within the classic physics puzzle game The Incredible Machine 2 (TIM). Unlike existing benchmarks, VLATIM specifically targets the critical gap between high-level logical reasoning and continuous action spaces requiring precise mouse interactions. This benchmark is structured into five progressive parts, assessing capabilities that range from basic visual grounding and domain understanding to multi-step manipulation and full puzzle solving. Our results reveal a significant disparity between reasoning and execution. While large proprietary models demonstrate superior planning abilities, they struggle with precise visual grounding. Consequently, they do not yet show human-like problem-solving capabilities. 

---
# Don't Look at the Numbers: Visual Anchoring Bias and Layer-wise Representation in VLMs 

**Authors**: M. Shalankin  

**Link**: [PDF](https://arxiv.org/pdf/2605.11218)  

**Abstract**: Embedded numeric anchors on images systematically bias Vision-Language Model quality judgments across six VLMs from five architectural families (ANOVA eta^2 = 0.18-0.77, all p < 0.001). Anchor effects are 2.5x larger than severe image quality degradation, confirming bias is not reducible to visual changes. Layer-wise probing reveals consistent dissociation: layers where anchor classification saturates (L12-L34) are suboptimal for quality prediction, with optimal layers deeper (R^2 = 0.69-0.91). Fusion analysis identifies architecture-dependent integration -- instant fusion at L1-L2 in two models versus partial or no fusion in three others. These results establish a causal account of visual anchoring bias, linking behavioral susceptibility to representation dynamics. 

---
# The Many Faces of On-Policy Distillation: Pitfalls, Mechanisms, and Fixes 

**Authors**: Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, Ge Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11182)  

**Abstract**: On-policy distillation (OPD) and on-policy self-distillation (OPSD) have emerged as promising post-training methods for large language models, offering dense token-level supervision on trajectories sampled from the model's own policy. However, existing results on their effectiveness remain mixed: while OP(S)D has shown promise in system prompt and knowledge internalization, recent studies also report instability and degradation. In this work, we present a comprehensive empirical study of when OPD and OPSD work, when they fail, and why. We find that OPD on mathematical reasoning is highly sensitive to teacher choice and loss formulation, whereas OPSD fails in our tested settings due to test-time absence of instance-specific privileged information (PI). In contrast, OPSD is effective when PI represents a shared latent rule, such as a system prompt or alignment preference. We identify three failure mechanisms: (1) distribution mismatch between teacher and student caused by conditioning on student-generated prefixes, (2) optimization instability from biased TopK reverse-KL gradients, and (3) an OPSD-specific limitation where the student learns a PI-free policy that aggregates PI-conditioned teachers, which is insufficient when PI is instance-specific. We further show that stop-gradient TopK objectives, RLVR-adapted teachers, and SFT-stabilized students mitigate these failures. 

---
# OLIVIA: Online Learning via Inference-time Action Adaptation for Decision Making in LLM ReAct Agents 

**Authors**: Sheldon Yu, Junda Wu, Xintong Li, Nikki Lijing Kuang, Sizhe Zhou, Tong Yu, Jiawei Han, Jingbo Shang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2605.11169)  

**Abstract**: Large language model agents interleave reasoning, action selection, and observation to solve sequential decision-making tasks. In deployed settings where agents repeatedly handle related multi-step tasks, small action-selection errors can accumulate into wasted tool calls, latency, and reduced reliability. Despite this need for deployment-time improvement, existing inference-time adaptation methods for LLM agents mainly rely on prompting or retrieval, which influence behavior indirectly through context manipulation. For ReAct-style agents, such approaches do not expose an explicit decision layer that can score candidate actions, represent uncertainty, or be updated online from action-level feedback. As a result, they provide limited support for trackable, fine-grained, and uncertainty-aware adaptation during deployment. We propose OLIVIA, an inference-time action adaptation framework for ReAct-style agents. OLIVIA models the LLM's final action-selection layer as a contextual linear bandit over candidate actions, with frozen hidden states as decision contexts. This choice is particularly suitable for deployment because it adapts behavior directly at the action-selection interface, preserves the underlying reasoning process, and provides explicit uncertainty estimates and lightweight online updates from action-level feedback. With upper-confidence-bound exploration, OLIVIA improves the policy sample-efficiently with minimal computational overhead. We instantiate OLIVIA on four benchmarks and show that it consistently improves task performance over static ReAct and prompt-based inference-time baselines. Our results suggest that explicit online decision layers provide an effective alternative to purely prompt- or retrieval-based adaptation for LLM agents during deployment. 

---
# RankQ: Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking 

**Authors**: Andrew Choi, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11151)  

**Abstract**: Offline-to-online reinforcement learning (RL) improves sample efficiency by leveraging pre-collected datasets prior to online interaction. A key challenge, however, is learning an accurate critic in large state--action spaces with limited dataset coverage. To mitigate harmful updates from value overestimation, prior methods impose pessimism by down-weighting out-of-distribution (OOD) actions relative to dataset actions. While effective, this essentially acts as a behavior cloning anchor and can hinder downstream online policy improvement when dataset actions are suboptimal. We propose RankQ, an offline-to-online Q-learning objective that augments temporal-difference learning with a self-supervised multi-term ranking loss to enforce structured action ordering. By learning relative action preferences rather than uniformly penalizing unseen actions, RankQ shapes the Q-function such that action gradients are directed toward higher-quality behaviors. Across sparse reward D4RL benchmarks, RankQ achieves performance competitive with or superior to seven prior methods. In vision-based robot learning, RankQ enables effective offline-to-online fine-tuning of a pretrained vision-language-action (VLA) model in a low-data regime, achieving on average a 42.7% higher simulation success rate than the next best method. In a high-data setting, RankQ improves simulation performance by 13.7% over the next best method and achieves strong sim-to-real transfer, increasing real-world cube stacking success from 43.1% to 84.7% relative to the VLA's initial performance. 

---
# EVOCHAMBER: Test-Time Co-evolution of Multi-Agent System at Individual, Team, and Population Scales 

**Authors**: Yaolun Zhang, Tianyi Xu, Shengyu Dai, Zhenwen Shao, Qingyun Wu, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11136)  

**Abstract**: We argue that multi-agent test-time evolution is not single-agent evolution replicated N times. A single-agent learner can only evolve its own context and memory. A multi-agent system additionally evolves who collaborates, how they collaborate, and how knowledge flows across the population. These components have no single-agent counterpart and can produce phenomena such as emergent specialization. Yet prior test-time methods either confine experiences to individual agents, forfeiting cross-agent learning, or broadcast symmetrically to all agents, erasing the specialization that makes collaboration valuable. We present EVOCHAMBER, a training-free framework that instantiates test-time evolution at three levels over a coevolving agent pool. At its core is CODREAM (Collaborative Dreaming), a post-task protocol triggered on team failure or disagreement, in which agents collaboratively reflect, distill insights, and route them asymmetrically from strong to weak agents on the failed niche, preserving specialization while filling knowledge gaps. Team-level operators assemble niche-conditioned teams and select collaboration structures online. Population-level lifecycle operators fork, merge, prune, and seed agents under performance pressure. On three heterogeneous task streams with Qwen3-8B, EVOCHAMBER reaches 63.9% on competition math, 75.7% on code, and 87.1% on multi-domain reasoning, outperforming the best baseline by 32% relative on math and confirming asymmetric cross-agent transfer as the primary driver in ablation. Starting from several identically initialized agents, four to five stable niche specialists spontaneously emerge, a structural signature of multi-agent evolution that no single-agent learner can express. See our code at: this https URL 

---
# A Cascaded Generative Approach for e-Commerce Recommendations 

**Authors**: Moein Hasani, Hamidreza Shahidi, Trace Levinson, Yuan Zhong, Guanghua Shu, Vinesh Gudla, Tejaswi Tenneti  

**Link**: [PDF](https://arxiv.org/pdf/2605.11118)  

**Abstract**: Personalized storefronts in large e-commerce marketplaces are often assembled from many independent components: static themes per page section ("placement"), retrieval systems to fetch eligible products per placement, and pointwise rankers to order content. While effective in optimizing for aggregate preferences, this paradigm is rigid and can limit personalization and semantic cohesion across the page. This makes it poorly suited to support dynamic objectives and merchandising requirements over time. To address this, we introduce a cascaded merchandising framework that decomposes storefront construction into two generative tasks: (i) placement-level theme generation and (ii) constrained keyword generation per placement to power product retrieval. Teacher-student fine-tuning is leveraged to improve scalability of this framework under production latency and cost constraints. Fine-tuned model ablations are shown to approach closed-weight LLM performance. We further contribute frameworks for AI-driven content evaluation and quality filtering, enabling safe and automated deployment of dynamic content at scale. Generative output is fused with traditional ranking models to preserve hybrid infrastructure. In online experiments, this framework yields an estimated +2.7% lift in cart adds per page view over a strong baseline. 

---
# AlphaGRPO: Unlocking Self-Reflective Multimodal Generation in UMMs via Decompositional Verifiable Reward 

**Authors**: Runhui Huang, Jie Wu, Rui Yang, Zhe Liu, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12495)  

**Abstract**: In this paper, we propose AlphaGRPO, a novel framework that applies Group Relative Policy Optimization (GRPO) to AR-Diffusion Unified Multimodal Models (UMMs) to enhance multimodal generation capabilities without an additional cold-start stage. Our approach unlocks the model's intrinsic potential to perform advanced reasoning tasks: Reasoning Text-to-Image Generation, where the model actively infers implicit user intents, and Self-Reflective Refinement, where it autonomously diagnoses and corrects misalignments in generated outputs. To address the challenge of providing stable supervision for real-world multimodal generation, we introduce the Decompositional Verifiable Reward (DVReward). Unlike holistic scalar rewards, DVReward utilizes an LLM to decompose complex user requests into atomic, verifiable semantic and quality questions, which are then evaluated by a general MLLM to provide reliable and interpretable feedback. Extensive experiments demonstrate that AlphaGRPO yields robust improvements across multimodal generation benchmarks, including GenEval, TIIF-Bench, DPG-Bench and WISE, while also achieving significant gains in editing tasks on GEdit without training on editing tasks. These results validate that our self-reflective reinforcement approach effectively leverages inherent understanding to guide high-fidelity generation. Project page: this https URL 

---
# Learning, Fast and Slow: Towards LLMs That Adapt Continually 

**Authors**: Rishabh Tiwari, Kusha Sareen, Lakshya A Agrawal, Joseph E. Gonzalez, Matei Zaharia, Kurt Keutzer, Inderjit S Dhillon, Rishabh Agarwal, Devvrit Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2605.12484)  

**Abstract**: Large language models (LLMs) are trained for downstream tasks by updating their parameters (e.g., via RL). However, updating parameters forces them to absorb task-specific information, which can result in catastrophic forgetting and loss of plasticity. In contrast, in-context learning with fixed LLM parameters can cheaply and rapidly adapt to task-specific requirements (e.g., prompt optimization), but cannot by itself typically match the performance gains available through updating LLM parameters. There is no good reason for restricting learning to being in-context or in-weights. Moreover, humans also likely learn at different time scales (e.g., System 1 vs 2). To this end, we introduce a fast-slow learning framework for LLMs, with model parameters as "slow" weights and optimized context as "fast" weights. These fast "weights" can learn from textual feedback to absorb the task-specific information, while allowing slow weights to stay closer to the base model and persist general reasoning behaviors. Fast-Slow Training (FST) is up to 3x more sample-efficient than only slow learning (RL) across reasoning tasks, while consistently reaching a higher performance asymptote. Moreover, FST-trained models remain closer to the base LLM (up to 70% less KL divergence), resulting in less catastrophic forgetting than RL-training. This reduced drift also preserves plasticity: after training on one task, FST trained models adapt more effectively to a subsequent task than parameter-only trained models. In continual learning scenarios, where task domains change on the fly, FST continues to acquire each new task while parameter-only RL stalls. 

---
# Beyond GRPO and On-Policy Distillation: An Empirical Sparse-to-Dense Reward Principle for Language-Model Post-Training 

**Authors**: Yuanda Xu, Hejian Sang, Zhengze Zhou, Ran He, Zhipeng Wang, Alborz Geramifard  

**Link**: [PDF](https://arxiv.org/pdf/2605.12483)  

**Abstract**: In settings where labeled verifiable training data is the binding constraint, each checked example should be allocated carefully. The standard practice is to use this data directly on the model that will be deployed, for example by running GRPO on the deployment student. We argue that this is often an inefficient allocation because it overlooks a reward-density principle: sparse sequence-level reward should train models where exploration is productive, while dense token-level teacher reward should be used where the aim is to compress behavior into a smaller model. In this view, GRPO-style sparse RL and OPD-style dense teacher supervision are not separate recipes; they are different reward-density regimes. The allocation rule is simple: use scarce labeled training data upstream on the strongest model that can turn it into reward-shaped behavior, then transfer that behavior downstream as dense supervision. We evaluate this rule on verifiable math with Qwen3 and Llama models. At fixed Qwen3-1.7B deployment-student size, an RL-improved 8B teacher distilled through the dense bridge outperforms direct GRPO on the same student, while transfer from the same teacher before RL underperforms. The bridge is important: a forward-KL warmup on teacher rollouts followed by OPD on student rollouts is consistently strongest on MATH before any post-bridge student-side sparse RL, and also gives the best pre-Stage~3 AIME endpoints for the canonical 8B/14B teachers. The bridge also makes later student-side sparse RL effective: GRPO that is weak on a cold student lifts MATH from $75.4\%$ to $78.5\%$ after the bridge and outperforms a matched replay control by $2.8$ points. The operational principal is to avoid using scarce labeled data on the least prepared policy: use sparse reward for teacher-side discovery, dense transfer for student compression, and student-side sparse reward only after the bridge. 

---
# OmniNFT: Modality-wise Omni Diffusion Reinforcement for Joint Audio-Video Generation 

**Authors**: Guohui Zhang, XiaoXiao Ma, Jie Huang, Hang Xu, Hu Yu, Siming Fu, Yuming Li, Zeyue Xue, Lin Song, Haoyang Huang, Nan Duan, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12480)  

**Abstract**: Recent advances in joint audio-video generation have been remarkable, yet real-world applications demand strong per-modality fidelity, cross-modal alignment, and fine-grained synchronization. Reinforcement Learning (RL) offers a promising paradigm, but its extension to multi-objective and multi-modal joint audio-video generation remains unexplored. Notably, our in-depth analysis first reveals that the primary obstacles to applying RL in this stem from: (i) multi-objective advantages inconsistency, where the advantages of multimodal outputs are not always consistent within a group; (ii) multi-modal gradients imbalance, where video-branch gradients leak into shallow audio layers responsible for intra-modal generation; (iii) uniform credit assignment, where fine-grained cross-modal alignment regions fail to get efficient exploration. These shortcomings suggest that vanilla RL fine-tuning strategy with a single global advantage often leads to suboptimal results. To address these challenges, we propose OmniNFT, a novel modality-aware online diffusion RL framework with three key innovations: (1) Modality-wise advantage routing, which routes independent per-reward advantages to their respective modality generation branches. (2) Layer-wise gradient surgery, which selectively detaches video-branch gradients on shallow audio layers while retaining those for cross-modal interaction layers. (3) Region-wise loss reweighting, which modulates policy optimization toward critical regions related to audio-video synchronization and fine-grained alignment. Extensive experiments on JavisBench and VBench with the LTX-2 backbone demonstrate that OmniNFT achieves comprehensive improvements in audio and video perceptual quality, cross-modal alignment, and audio-video synchronization. 

---
# KV-Fold: One-Step KV-Cache Recurrence for Long-Context Inference 

**Authors**: Alireza Nadali, Patrick Cooper, Ashutosh Trivedi, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2605.12471)  

**Abstract**: We introduce KV-Fold, a simple, training-free long-context inference protocol that treats the key-value (KV) cache as the accumulator in a left fold over sequence chunks. At each step, the model processes the next chunk conditioned on the accumulated cache, appends the newly produced keys and values, and passes the enlarged cache forward; the same one-step update is applied repeatedly, analogous to foldl in functional programming. Building on the KV cache concatenation primitive introduced for latent multi-agent communication, we repurpose it as a chunk-to-chunk recurrence for long-context inference. When processing chunk t, the model attends to the KV cache carried from earlier chunks as a prefix, reusing its internal state across segments without modifying or retraining the model. Despite its simplicity, the induced recurrence is stable: per-step drift rises briefly and then saturates into a flat plateau that persists across deep chains. This plateau is insensitive to a 10,000x change in numerical precision, robust across chunk sizes, and consistent across model families. At the task level, KV-Fold preserves exact information over long distances. On a needle-in-a-haystack benchmark, it achieves 100% exact-match retrieval across 152 trials spanning contexts from 16K to 128K tokens and chain depths up to 511 on Llama-3.1-8B, while remaining within the memory limits of a single 40GB GPU. Compared to streaming methods, which trade fidelity for bounded memory, KV-Fold maintains long-range retrieval while operating as a sequence of tractable forward passes. Overall, our results show that frozen pretrained transformers already support a stable form of KV-cache recurrence, providing a practical route to long-context inference without architectural changes or training. 

---
# Solve the Loop: Attractor Models for Language and Reasoning 

**Authors**: Jacob Fein-Ashley, Paria Rashidinejad  

**Link**: [PDF](https://arxiv.org/pdf/2605.12466)  

**Abstract**: Looped Transformers offer a promising alternative to purely feed-forward computation by iteratively refining latent representations, improving language modeling and reasoning. Yet recurrent architectures remain unstable to train, costly to optimize and deploy, and constrained to small, fixed recurrence depths. We introduce Attractor Models, in which a backbone module first proposes output embeddings, then an attractor module refines them by solving for the fixed point, with gradients obtained through implicit differentiation. Thus, training memory remains constant in effective depth, and iterations are chosen adaptively by convergence. Empirically, Attractor Models outperform existing models across two regimes, large-scale language-model pretraining and reasoning with tiny models. In language modeling, Attractor Models deliver a Pareto improvement over standard Transformers and stable looped models across sizes, improving perplexity by up to 46.6% and downstream accuracy by up to 19.7% while reducing training cost. Notably, a 770M Attractor Model outperforms a 1.3B Transformer trained on twice as many tokens. On challenging reasoning tasks, we show that our model with only 27M parameters and approximately 1000 examples achieves 91.4% accuracy on Sudoku-Extreme and 93.1% on Maze-Hard, scaling favorably where frontier models like Claude and GPT o3, fail completely, and specialized recursive reasoners collapse at larger sizes. Lastly, we show that Attractor Models exhibit a novel phenomenon, which we call equilibrium internalization: fixed-point training places the model's initial output embedding near equilibrium, allowing the solver to be removed at inference time with little degradation. Together, these results suggest that Attractor Models make iterative refinement scalable by turning recurrence into a computation the model can learn to internalize. 

---
# Enabling AI-Native Mobility in 6G: A Real-World Dataset for Handover, Beam Management, and Timing Advance 

**Authors**: Mannam Veera Narayana, Rohit Singh, Deepa M.R, Radha Krishna Ganti  

**Link**: [PDF](https://arxiv.org/pdf/2605.12453)  

**Abstract**: To address the issues of high interruption time and measurement report overhead under user equipment (UE) mobility especially in high speed 5G use cases the use of AI/ML techniques (AI/ML beam management and mobility procedures) have been proposed. These techniques rely heavily on data that are most often simulated for various scenarios and do not accurately reflect real deployment behavior or user traffic patterns. Therefore, there is an utmost need for realistic datasets under various conditions. This work presents a dataset collected from a commercially deployed network across various modes of mobility (pedestrian, bike, car, bus, and train) and at multiple speeds to depict real time UE mobility. When collecting the dataset, we focused primarily on handover (HO) scenarios, with the aim of reducing the HO interruption time and maintaining continuous throughput during and immediately after HO execution. To support this research, the dataset includes timing advance (TA) measurements at various signaling events such as RACH trigger, MAC CE, and PDCCH grant which are typically missing in existing works.
We cover a detailed description of the creation of the dataset; experimental setup, data acquisition, and extraction. We also cover an exploratory analysis of the data, with a primary focus on mobility, beam management, and TA. We discuss multiple use cases in which the proposed dataset can facilitate understanding of the inference of the AI/ML model. One such use case is to train and evaluate various AI/ML models for TA prediction. 

---
# The Algorithmic Caricature: Auditing LLM-Generated Political Discourse Across Crisis Events 

**Authors**: Gunjan, Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2605.12452)  

**Abstract**: Large Language Models (LLMs) can generate fluent political text at scale, raising concerns about synthetic discourse during crises and social conflict. Existing AI-text detection often focuses on sentence-level cues such as perplexity, burstiness, or token irregularities, but these signals may weaken as generative systems improve. We instead adopt a Computational Social Science perspective and ask whether synthetic political discourse behaves like an observed online population. We construct a paired corpus of 1,789,406 posts across nine crisis events: COVID-19, the Jan. 6 Capitol attack, the 2020 and 2024 U.S. elections, Dobbs/Roe v. Wade, the 2020 BLM protests, U.S. midterms, the Utah shooting, and the U.S.-Iran war. For each event, we compare observed discourse from social platforms with synthetic discourse generated for the same context. We evaluate four dimensions: emotional intensity, structural regularity, lexical-ideological framing, and cross-event dependency, using mean gaps and dispersion evidence. Across events, synthetic discourse is fluent but population-level unrealistic. It is generally more negative and less dispersed in sentiment, structurally more regular, and lexically more abstract than observed discourse. Observed discourse instead shows broader emotional variation, longer-tailed structural distributions, and more context-specific, colloquial lexical markers. These differences are event-dependent: larger for fast-moving, decentralized crises and smaller for formal or institutionally mediated events. We summarize them with a simple event-level measure, the Caricature Gap. Our findings suggest that the main limitation of synthetic political discourse is not grammar or fluency, but reduced population realism. Population-level auditing complements traditional text-detection and provides a CSS framework for evaluating the social realism of generated discourse. 

---
# A Causal Language Modeling Detour Improves Encoder Continued Pretraining 

**Authors**: Rian Touchent, Eric de la Clergerie  

**Link**: [PDF](https://arxiv.org/pdf/2605.12438)  

**Abstract**: When adapting an encoder to a new domain, the standard approach is to continue training with Masked Language Modeling (MLM). We show that temporarily switching to Causal Language Modeling (CLM) followed by a short MLM decay improves downstream performance. On biomedical texts with ModernBERT, this CLM detour outperforms MLM baselines trained on identical data and compute across 8 French and 11 English biomedical tasks, by +1.2-2.8pp and +0.3-0.8pp respectively, depending on model size. We investigate the reasons for these gains. We find that CLM's dense supervision impacts low transformer layers (0-7) far more than MLM does. Freezing low layers during CLM eliminates the downstream benefit; freezing mid layers preserves it. The representational changes persist through the MLM decay phase, even when it matches the CLM phase in length, and they scale with model capacity. We release ModernCamemBERT-bio and ModernBERT-bio as state-of-the-art biomedical encoders in Base and Large sizes. 

---
# Stories in Space: In-Context Learning Trajectories in Conceptual Belief Space 

**Authors**: Eric Bigelow, Raphaël Sarfati, Daniel Wurgaft, Owen Lewis, Thomas McGrath, Jack Merullo, Atticus Geiger, Ekdeep Singh Lubana  

**Link**: [PDF](https://arxiv.org/pdf/2605.12412)  

**Abstract**: Large Language Models (LLMs) update their behavior in context, which can be viewed as a form of Bayesian inference. However, the structure of the latent hypothesis space over which this inference operates remains unclear. In this work, we propose that LLMs assign beliefs over a low-dimensional geometric space - a conceptual belief space - and that in-context learning corresponds to a trajectory through this space as beliefs are updated over time. Using story understanding as a natural setting for dynamic belief updating, we combine behavioral and representational analyses to study these trajectories. We find that (1) belief updates are well-described as trajectories on low-dimensional, structured manifolds; (2) this structure is reflected consistently in both model behavior and internal representations and can be decoded with simple linear probes to predict behavior; and (3) interventions on these representations causally steer belief trajectories, with effects that can be predicted from the geometry of the conceptual space. Together, our results provide a geometric account of belief dynamics in LLMs, grounding Bayesian interpretations of in-context learning in structured conceptual representations. 

---
# Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling 

**Authors**: Eilam Shapira, Moshe Tennenholtz, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2605.12411)  

**Abstract**: AI agents negotiate and transact in natural language with unfamiliar counterparts: a buyer bot facing an unknown seller, or a procurement assistant negotiating with a supplier. In such interactions, the counterpart's LLM, prompts, control logic, and rule-based fallbacks are hidden, while each decision can have monetary consequences. We ask whether an agent can predict an unfamiliar counterpart's next decision from a few interactions. To avoid real-world logging confounds, we study this problem in controlled bargaining and negotiation games, formulating it as target-adaptive text-tabular prediction: each decision point is a table row combining structured game state, offer history, and dialogue, while $K$ previous games of the same target agent, i.e., the counterpart being modeled, are provided in the prompt as labeled adaptation examples. Our model is built on a tabular foundation model that represents rows using game-state features and LLM-based text representations, and adds LLM-as-Observer as an additional representation: a small frozen LLM reads the decision-time state and dialogue; its answer is discarded, and its hidden state becomes a decision-oriented feature, making the LLM an encoder rather than a direct few-shot predictor. Training on 13 frontier-LLM agents and testing on 91 held-out scaffolded agents, the full model outperforms direct LLM-as-Predictor prompting and game+text features baselines. Within this tabular model, Observer features contribute beyond the other feature schemes: at $K=16$, they improve response-prediction AUC by about 4 points across both tasks and reduce bargaining offer-prediction error by 14%. These results show that formulating counterpart prediction as a target-adaptive text-tabular task enables effective adaptation, and that hidden LLM representations expose decision-relevant signals that direct prompting does not surface. 

---
# OGLS-SD: On-Policy Self-Distillation with Outcome-Guided Logit Steering for LLM Reasoning 

**Authors**: Yuxiao Yang, Xiaoyun Wang, Weitong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12400)  

**Abstract**: We study {on-policy self-distillation} (OPSD), where a language model improves its reasoning ability by distilling privileged teacher distributions along its own on-policy trajectories. Despite the performance gains of OPSD, we identify a common but often overlooked mismatch between teacher and student responses: self-reflected teacher responses can be shifted by reflection-induced bias and response templates, leading to miscalibrated token-level supervision. To mitigate this issue, we propose \methodname, an outcome-guided logit-steering framework that leverages verifiable outcome rewards to contrast successful and failed on-policy trajectories and calibrate teacher logits. By combining outcome-level correctness with dense token-level guidance through logit steering, \methodname stabilizes self-distillation and improves reasoning performance over standard OPSD and other variants across diverse benchmarks. 

---
# Detecting overfitting in Neural Networks during long-horizon grokking using Random Matrix Theory 

**Authors**: Hari K. Prakash, Charles H Martin  

**Link**: [PDF](https://arxiv.org/pdf/2605.12394)  

**Abstract**: Training Neural Networks (NNs) without overfitting is difficult; detecting that overfitting is difficult as well. We present a novel Random Matrix Theory method that detects the onset of overfitting in deep learning models without access to train or test data. For each model layer, we randomize each weight matrix element-wise, $\mathbf{W} \to \mathbf{W}_{\mathrm{rand}}$, fit the randomized empirical spectral distribution with a Marchenko-Pastur distribution, and identify large outliers that violate self-averaging. We call these outliers Correlation Traps. During the onset of overfitting, which we call the "anti-grokking" phase in long-horizon grokking, Correlation Traps form and grow in number and scale as test accuracy decreases while train accuracy remains high. Traps may be benign or may harm generalization; we provide an empirical approach to distinguish between them by passing random data through the trained model and evaluating the JS divergence of output logits. Our findings show that anti-grokking is an additional grokking phase with high train accuracy and decreasing test accuracy, structurally distinct from pre-grokking through its Correlation Traps. More broadly, we find that some foundation-scale LLMs exhibit the same Correlation Traps, indicating potentially harmful overfitting. 

---
# SEMIR: Semantic Minor-Induced Representation Learning on Graphs for Visual Segmentation 

**Authors**: Luke James Miller, Yugyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.12389)  

**Abstract**: Segmenting small and sparse structures in large-scale images is fundamentally constrained by voxel-level, lattice-bound computation and extreme class imbalance -- dense, full-resolution inference scales poorly and forces most pipelines to rely on fixed regionization or downsampling, coupling computational cost to image resolution and attenuating boundary evidence precisely where minority structures are most informative. We introduce SEMIR (Semantic Minor-Induced Representation Learning), a representation framework that decouples inference from the native grid by learning a task-adapted, topology-preserving latent graph representation with exact decoding. SEMIR transforms the underlying grid graph into a compact, boundary-aligned graph minor through parameterized edge contraction, node deletion, and edge deletion, while preserving an exact lifting map from minor predictions to lattice labels. Minor construction is formalized as a few-shot structure learning problem that replaces hand-tuned preprocessing with a boundary-alignment objective: minor parameters are learned by maximizing agreement between predicted boundary elements and target-specific semantic edges under a boundary Dice criterion, and the induced minor is annotated with scale- and rotation-robust geometric and intensity descriptors and supports efficient region-level inference via message passing on a graph neural network (GNN) with relational edge features. We benchmark SEMIR on three tumor segmentation datasets -- BraTS 2021, KiTS23, and LiTS -- where targets exhibit high structural variability and distributional uncertainty. SEMIR yields consistent improvements in minority-structure Dice at practical runtime. More broadly, SEMIR establishes a framework for learning task-adapted, topology-preserving latent representations with exact decoding for high-resolution structured visual data. 

---
# Scalable Token-Level Hallucination Detection in Large Language Models 

**Authors**: Rui Min, Tianyu Pang, Chao Du, Minhao Cheng, Yi R. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2605.12384)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but they still frequently produce hallucinations. These hallucinations are difficult to detect in reasoning-intensive tasks, where the content appears coherent but contains errors like logical flaws and unreliable intermediate results. While step-level analysis is commonly used to detect internal hallucinations, it suffers from limited granularity and poor scalability due to its reliance on step segmentation. To address these limitations, we propose TokenHD, a holistic pipeline for training token-level hallucination detectors. Specifically, TokenHD consists of a scalable data engine for synthesizing large-scale hallucination annotations along with a training recipe featuring an importance-weighted strategy for robust model training. To systematically assess the detection performance, we also provide a rigorous evaluation protocol. Through training within TokenHD, our detector operates directly on free-form text to identify hallucinations, eliminating the need for predefined step segmentation or additional text reformatting. Our experiments show that even a small detector (0.6B) achieves substantial performance gains after training, surpassing much larger reasoning models (e.g., QwQ-32B), and detection performance scales consistently with model size from 0.6B to 8B. Finally, we show that our detector can generalize well across diverse practical scenarios and explore strategies to further enhance its cross-domain generalization capability. 

---
# Trust the Batch, On- or Off-Policy: Adaptive Policy Optimization for RL Post-Training 

**Authors**: Rasool Fakoor, Murdock Aubry, Nicholas Stranges, Alexander J. Smola  

**Link**: [PDF](https://arxiv.org/pdf/2605.12380)  

**Abstract**: Reinforcement learning is structurally harder than supervised learning because the policy changes the data distribution it learns from. The resulting fragility is especially visible in large-model training, where the training and rollout systems differ in numerical precision, sampling, and other implementation details. Existing methods manage this fragility by adding hyper-parameters to the training objective, which makes the algorithm more sensitive to its configuration and requires retuning whenever the task, model scale, or distribution mismatch changes. This fragility traces to two concerns that current objectives entangle through hyper-parameters set before training begins: a trust-region concern, that updates should not move the policy too far from its current value, and an off-policy concern, that data from older or different behavior policies should influence the update only to the extent that it remains reliable. Neither concern is a constant to set in advance, and their severity is reflected in the policy-ratio distribution of the current batch. We present a simple yet effective batch-adaptive objective that replaces fixed clipping with the normalized effective sample size of the policy ratios. The same statistic caps the score-function weight and sets the strength of an off-policy regularizer, so the update stays close to the usual on-policy score-function update when ratios are nearly uniform, and tightens automatically when stale or mismatched data cause ratio concentration, while retaining a nonzero learning signal on high-ratio tokens. Experiments across a wide range of settings show that our method matches or exceeds tuned baselines, introducing no new objective hyper-parameters and removing several existing ones. The code is available at this https URL. 

---
# Discrete Flow Matching for Offline-to-Online Reinforcement Learning 

**Authors**: Fairoz Nower Khan, Nabuat Zaman Nahim, Peizhong Ju  

**Link**: [PDF](https://arxiv.org/pdf/2605.12379)  

**Abstract**: Many reinforcement learning (RL) tasks have discrete action spaces, but most generative policy methods based on diffusion and flow matching are designed for continuous control. Meanwhile, generative policies usually rely heavily on offline datasets and offline-to-online RL is itself challenging, as the policy must improve from new interaction without losing useful behavior learned from static data. To address those challenges, we introduce DRIFT, an online fine-tuning method that updates an offline pretrained continuous-time Markov chain (CTMC) policy with an advantage-weighted discrete flow matching loss. To preserve useful pretrained knowledge, we add a path-space penalty that regularizes the full CTMC trajectory distribution, rather than only the final action distribution. For large discrete action spaces, we introduce a candidate-set approximation that updates the actor over a small subset of actions sampled from reference-policy rollouts and uniform exploration. Our theoretical analysis shows that the candidate-set error is controlled by missing target probability mass, and the induced CTMC generator error decreases as the candidate set covers more high-probability actions. Experiments on prevailing discrete action RL task show that our method provides stable offline-to-online improvement across all tasks, achieving the highest average score on Jericho with a simple GRU encoder while outperforming methods that use pretrained language models. Controlled experiments further confirm that the path-space penalty remains bounded during fine-tuning and that the CTMC generator adapts to shifted rewards faster than deterministic baselines. The candidate-set mechanism is supported by a stability analysis showing that the generator error decreases exponentially with candidate coverage. 

---
# Agent-Based Post-Hoc Correction of Agricultural Yield Forecasts 

**Authors**: Matthew Beddows, Aiden Durrant, Georgios Leontidis  

**Link**: [PDF](https://arxiv.org/pdf/2605.12375)  

**Abstract**: Accurate crop yield forecasting in commercial soft fruit production is constrained by the data available in typical commercial farm records, which lack the sensor networks, satellite imagery, and high-resolution meteorological inputs that most state-of-the-art approaches assume. We propose a structured LLM agent framework that performs post-hoc correction of existing model predictions, encoding agricultural domain knowledge across tools for phase detection, bias learning, and range validation. Evaluated on a proprietary strawberry yield dataset and a public USDA corn harvest dataset, agent refinement of XGBoost reduced MAE by 20% and MASE by 56% on strawberry, with consistent improvements across Moirai2 (MAE 24%, MASE 22%) and Random Forest (MAE 28%, MASE 66%) baselines. Using Llama 3.1 8B as the agent produced the strongest corrections across all configurations; LLaVA 13B showed inconsistent gains, highlighting sensitivity to the choice of refinement model. 

---
# Fill the GAP: A Granular Alignment Paradigm for Visual Reasoning in Multimodal Large Language Models 

**Authors**: Yanting Miao, Yutao Sun, Dexin Wang, Mengyu Zhou, Pascal Poupart, Lei Lv, Qi Zhao, Li Wang, Hao Li, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12374)  

**Abstract**: Visual latent reasoning lets a multimodal large language model (MLLM) create intermediate visual evidence as continuous tokens, avoiding external tools or image generators. However, existing methods usually follow an output-as-input latent paradigm and yield unstable gains. We identify evidence for a feature-space mismatch that can contribute to this instability: dominant visual-latent models build on pre-norm MLLMs and reuse decoder hidden states as predicted latent inputs, even though these states occupy a substantially different norm regime from the input embeddings the model was trained to consume~\citep{xie2025mhc,li2026siamesenorm,team2026attention}. This mismatch can make direct latent feedback unreliable. Motivated by this diagnosis, we propose \textbf{GAP}, a \textbf{G}ranular \textbf{A}lignment \textbf{P}aradigm for visual latent modeling. GAP aligns visual latent reasoning at three levels: feature-level alignment maps decoder outputs into input-compatible visual latents through a lightweight PCA-aligned latent head; context-level alignment grounds latent targets with inspectable auxiliary visual supervision; and capacity-guided alignment assigns latent supervision selectively to examples where the base MLLM struggles. On Qwen2.5-VL 7B, the resulting model achieves the best mean aggregate perception and reasoning performance among our supervised variants. Inference-time intervention probing further suggests that generated latents provide task-relevant visual signal beyond merely adding token slots. 

---
# QAP-Router: Tackling Qubit Routing as Dynamic Quadratic Assignment with Reinforcement Learning 

**Authors**: Kien X. Nguyen, Ankit Kulshrestha, Ilya Safro, Xiaoyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12365)  

**Abstract**: Qubit routing is a fundamental problem in quantum compilation, known to be NP-hard. Its dynamic nature makes local routing decisions propagate and compound over time, making global efficient solutions challenging. Existing heuristic methods rely on local rules with limited lookahead, while recent learning-based approaches often treat routing as a generic sequential decision problem without fully exploiting its underlying structure. In this paper, we introduce QAP-Router, framing qubit routing based on a dynamic Quadratic Assignment Problem (QAP) formulation. By modeling logical interactions, or quantum gates, as flow matrices and hardware topology as a distance matrix, our approach captures the interaction-distance coupling in a unified objective, which defines the reward in the reinforcement learning environment. To further exploit this structure, the policy network employs a solution-aware Transformer backbone that encodes the interaction between the flow matrix and the distance matrix into the attention mechanism. We also integrate a lookahead mechanism that blends naturally into the QAP framework, preventing myopic decisions. Extensive experiments on 1,831 real-world quantum circuits from the MQTBench, AgentQ and QUEKO datasets show that our method substantially reduces the CNOT gate count of routed circuits by 15.7%, 30.4% and 12.1%, respectively, relative to existing industry compilers. 

---
# A Family of Quaternion-Valued Differential Evolution Algorithms for Numerical Function Optimization 

**Authors**: Gerardo Altamirano-Gomez, Álvaro Gallardo, Carlos Ignacio Hernández Castellanos  

**Link**: [PDF](https://arxiv.org/pdf/2605.12362)  

**Abstract**: The numerical optimization of continuous functions is a fundamental task in many scientific and engineering domains, ranging from mechanical design to training of artificial intelligence models. Among the most effective and widely used algorithms for this purpose is Differential Evolution (DE), known for its simplicity and strong performance. Recent research has shown that adapting AI models to operate over alternative number systems-such as complex numbers, quaternions, and geometric algebras-can improve model compactness and accuracy. However, such extensions remain underexplored in bio-inspired optimization algorithms. In particular, the use of quaternion algebra represents an emerging area in computational intelligence. This paper introduces a family of novel Quaternion-Valued Differential Evolution (QDE) algorithms that operate directly in the quaternion space. We propose several mutation strategies specifically designed to exploit the algebraic and geometric properties of quaternions. Results show that our QDE variants achieve faster convergence and superior performance on several function classes in the BBOB benchmark compared to the traditional real-valued DE algorithm. 

---
# MedHopQA: A Disease-Centered Multi-Hop Reasoning Benchmark and Evaluation Framework for LLM-Based Biomedical Question Answering 

**Authors**: Rezarta Islamaj, Robert Leaman, Joey Chan, Nicholas Wan, Qiao Jin, Natalie Xie, John Wilbur, Shubo Tian, Lana Yeganova, Po-Ting Lai, Chih-Hsuan Wei, Yifan Yang, Yao Ge, Qingqing Zhu, Zhizheng Wang, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12361)  

**Abstract**: Evaluating large language models (LLMs) in the biomedical domain requires benchmarks that can distinguish reasoning from pattern matching and remain discriminative as model capabilities improve. Existing biomedical question answering (QA) benchmarks are limited in this respect. Multiple-choice formats can allow models to succeed through answer elimination rather than inference, while widely circulated exam-style datasets are increasingly vulnerable to performance saturation and training data contamination. Multi-hop reasoning, defined as the ability to integrate information across multiple sources to derive an answer, is central to clinically meaningful tasks such as diagnostic support, literature-based discovery, and hypothesis generation, yet remains underrepresented in current biomedical QA benchmarks. MedHopQA is a disease-centered multi-hop reasoning benchmark consisting of 1,000 expert-curated question-answer pairs introduced as a shared task at BioCreative IX. Each question requires synthesis of information across two distinct Wikipedia articles, and answers are provided in an open-ended free-text format. Gold annotations are augmented with ontology-grounded synonym sets from MONDO, NCBI Gene, and NCBI Taxonomy to support both lexical and concept-level evaluation. MedHopQA was constructed through a structured process combining human annotation, triage, iterative verification, and LLM-as-a-judge validation. To reduce leaderboard gaming and contamination risk, the 1,000 scored questions are embedded within a publicly downloadable set of 10,000 questions, with answers withheld, on a CodaBench leaderboard. MedHopQA provides both a benchmark and a reusable framework for constructing future biomedical QA datasets that prioritize compositional reasoning, saturation resistance, and contamination resistance as core design constraints. 

---
# A New Technique for AI Explainability using Feature Association Map 

**Authors**: Sayantani Ghosh, Amit Kumar Das, Amlan Chakrabarti  

**Link**: [PDF](https://arxiv.org/pdf/2605.12350)  

**Abstract**: Lack of transparency in AI systems poses challenges in critical real-life applications. It is important to be able to explain the decisions of an AI system to ensure trust on the system. Explainable AI (XAI) algorithms play a vital role in achieving this objective. In this paper, we are proposing a new algorithm for Explaining AI systems, FAMeX (Feature Association Map based eXplainability). The proposed algorithm is based on a graph-theoretic formulation of the feature set termed as Feature Association Map (FAM). The foundation of the modelling is based on association between features. The proposed FAMeX algorithm has been found to be better than the competing XAI algorithms - Permutation Feature Importance (PFI) and SHapley Additive exPlanations (SHAP). Experiments conducted with eight benchmark algorithms show that FAMeX is able to gauge feature importance in the context of classification better than the competing algorithms. This definitely shows that FAMeX is a promising algorithm in explaining the predictions from an AI system 

---
# BSO: Safety Alignment Is Density Ratio Matching 

**Authors**: Tien-Phat Nguyen, Truong Nguyen, Thin Nguyen, Duy Minh Ho Nguyen, Ngoc-Thanh Dinh, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.12339)  

**Abstract**: Aligning language models for both helpfulness and safety typically requires complex pipelines-separate reward and cost models, online reinforcement learning, and primal-dual updates. Recent direct preference optimization approaches simplify training but incorporate safety through ad-hoc modifications such as multi-stage procedures or heuristic margin terms, lacking a principled derivation. We show that the likelihood ratio of the optimal safe policy admits a closed-form decomposition that reduces safety alignment to a density ratio matching problem. Minimizing Bregman divergences between the data and model ratios yields Bregman Safety Optimization (BSO), a family of single-stage loss functions, each induced by a convex generator, that provably recover the optimal safe policy. BSO is both general and simple: it requires no auxiliary models, introduces only one hyperparameter beyond standard preference optimization, and recovers existing safety-aware methods as special cases. Experiments across safety alignment benchmarks show that BSO consistently improves the safety-helpfulness trade-off. 

---
# Manifold Sampling via Entropy Maximization 

**Authors**: Cornelius V. Braun, Tilman Burghoff, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2605.12338)  

**Abstract**: Sampling from constrained distributions has a wide range of applications, including in Bayesian optimization and robotics. Prior work establishes convergence and feasibility guarantees for constrained sampling, but assumes that the feasible set is connected. However, in practice, the feasible set often decomposes into multiple disconnected components, which makes efficient sampling under constraints challenging. In this paper, we propose MAnifold Sampling via Entropy Maximization (MASEM) for sampling on a manifold with an unknown number of disconnected components, implicitly defined by smooth equality and inequality constraints. The presented method uses a resampling scheme to maximize the entropy of the empirical distribution based on k-nearest neighbor density estimation. We show that, in the mean field, MASEM decreases the KL-divergence between the empirical distribution and the maximum-entropy target exponentially in the number of resampling steps. We instantiate MASEM with multiple local samplers and demonstrate its versatility and efficiency on synthetic and robotics-based benchmarks. MASEM enables fast and scalable mixing across a range of constrained sampling problems, improving over alternatives by an order of magnitude in Sinkhorn distance with competitive runtime. 

---
# EHR-RAGp: Retrieval-Augmented Prototype-Guided Foundation Model for Electronic Health Records 

**Authors**: Saeed Shurrab, Mariam Al-Omari, Dana El Samad, Farah E. Shamout  

**Link**: [PDF](https://arxiv.org/pdf/2605.12335)  

**Abstract**: Electronic Health Records (EHR) contain rich longitudinal patient information and are widely used in predictive modeling applications. However, effectively leveraging historical data remains challenging due to long trajectories, heterogeneous events, temporal irregularity, and the varying relevance of past clinical context. Existing approaches often rely on fixed windows or uniform aggregation, which can obscure clinically important signals. In this work, we introduce EHR-RAGp, a retrieval-augmented foundation model that dynamically integrates the most relevant patient history across diverse clinical event types. We propose a prototype-guided retrieval module that acts as an alignment mechanism and estimates the relevance of retrieved historical chunks with respect to a given prediction task, guiding the model towards the most informative context. Across multiple clinical prediction tasks, EHR-RAGp consistently outperforms state-of-the-art EHR foundation models and transformer-based baselines. Furthermore, integrating EHR-RAGp with existing clinical foundation models yields substantial performance gains. Overall, EHR-RAGp provides a scalable and efficient framework for leveraging long-range clinical context to improve downstream performance. 

---
# Transferable Delay-Aware Reinforcement Learning via Implicit Causal Graph Modeling 

**Authors**: Chenran Zhao, Dianxi Shi, Yaowen Zhang, Chunping Qiu, Shaowu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12312)  

**Abstract**: Random delays weaken the temporal correspondence between actions and subsequent state feedback, making it difficult for agents to identify the true propagation process of action effects. In cross-task scenarios, changes in task objectives and reward formulations further reduce the reusability of previously acquired task knowledge. To address this problem, this paper proposes a transferable delay-aware reinforcement learning method based on implicit causal graph modeling. The proposed method uses a field-node encoder to represent high-dimensional observations as latent states with node-level semantics, and employs a message-passing mechanism to characterize dynamic causal dependencies among nodes, thereby learning transferable structured representations and environment dynamics knowledge. On this basis, imagination-driven behavior learning and planning are incorporated to optimize policies in the latent space, enabling cross-task knowledge transfer and rapid adaptation. Experimental results show that the proposed method outperforms baseline methods on DMC continuous control tasks with random delays. Cross-task transfer experiments further demonstrate that the learned structured representations and dynamics knowledge can be effectively transferred to new tasks and significantly accelerate policy adaptation. 

---
# KAN-CL: Per-Knot Importance Regularization for Continual Learning with Kolmogorov-Arnold Networks 

**Authors**: Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2605.12306)  

**Abstract**: Catastrophic forgetting remains the central obstacle in continual learning (CL): parameters shared across tasks interfere with one another, and existing regularization methods such as EWC and SI apply uniform penalties without awareness of which input region a parameter serves. We propose KAN-CL, a continual learning framework that exploits the compact-support spline parameterization of Kolmogorov-Arnold Networks (KANs) to perform importance-weighted anchoring at per-knot granularity. Deployed as a classification head on a convolutional backbone with standard EWC regularization on the backbone (bbEWC) KAN-CL achieves forgetting reductions of 88% and 93% over a head-only KAN baseline on Split-CIFAR-10/5T and Split-CIFAR-100/10T respectively, while matching or exceeding the accuracy of all baselines on both benchmarks. We further provide a Neural Tangent Kernel (NTK) analysis showing that KAN's spline locality induces a structural rank deficit in the cross-task NTK, yielding a forgetting bound that holds even in the feature-learning regime. These results establish that combining an architecture with natural parameter locality (KAN head) with a complementary backbone regularizer (bbEWC) yields a compositional and principled approach to catastrophic forgetting. 

---
# PriorZero: Bridging Language Priors and World Models for Decision Making 

**Authors**: Junyu Xiong, Yuan Pu, Jia Tang, Yazhe Niu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12289)  

**Abstract**: Leveraging the rich world knowledge of Large Language Models (LLMs) to enhance Reinforcement Learning (RL) agents offers a promising path toward general intelligence. However, a fundamental prior-dynamics mismatch hinders existing approaches: static LLM knowledge cannot directly adapt to the complex transition dynamics of long-horizon tasks. Using LLM priors as fixed policies limits exploration diversity, as the prior is blind to environment-specific dynamics; while end-to-end fine-tuning suffers from optimization instability and credit assignment issues. To bridge this gap, we propose PriorZero, a unified framework that integrates LLM-derived conceptual priors into world-model-based planning through a decoupled rollout-training design. During rollout, a novel root-prior injection mechanism incorporates LLM priors exclusively at the root node of Monte Carlo Tree Search (MCTS), focusing search on semantically promising actions while preserving the world model's deep lookahead capability. During training, PriorZero decouples world-model learning from LLM adaptation: the world model is continuously refined on interaction data to jointly improve its dynamics, policy, and value predictions, its value estimates are then leveraged to provide fine-grained credit assignment signals for stable LLM fine-tuning via alternating optimization. Experiments across diverse benchmarks, including text-based adventure games in Jericho and instruction-following gridworld tasks in BabyAI, demonstrate that PriorZero consistently improves both exploration efficiency and asymptotic performance, establishing a promising framework for LLM-empowered decision-making. Our code is available at this https URL. 

---
# TokenRatio: Principled Token-Level Preference Optimization via Ratio Matching 

**Authors**: Truong Nguyen, Tien-Phat Nguyen, Linh Ngo Van, Duy Minh Ho Nguyen, Khoa Doan, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.12288)  

**Abstract**: Direct Preference Optimization (DPO) is a widely used RL-free method for aligning language models from pairwise preferences, but it models preferences over full sequences even though generation is driven by per-token decisions. Existing token-level extensions typically decompose a sequence-level Bradley-Terry objective across timesteps, leaving per-prefix (state-wise) optimality implicit. We study how to recover token-level preference optimality using only standard sequence-level pairwise comparisons. We introduce Token-level Bregman Preference Optimization (TBPO), which posits a token-level Bradley-Terry preference model over next-token actions conditioned on the prefix, and derive a Bregman-divergence density-ratio matching objective that generalizes the logistic/DPO loss while preserving the optimal policy induced by the token-level model and maintaining DPO-like simplicity. We introduce two instantiations: TBPO-Q, which explicitly learns a lightweight state baseline, and TBPO-A, which removes the baseline through advantage normalization. Across instruction following, helpfulness/harmlessness, and summarization benchmarks, TBPO improves alignment quality and training stability and increases output diversity relative to strong sequence-level and token-level baselines. 

---
# Set-Aggregated Genome Embeddings for Microbiome Abundance Prediction 

**Authors**: Younhun Kim, Georg K. Gerber, Travis E. Gibson  

**Link**: [PDF](https://arxiv.org/pdf/2605.12286)  

**Abstract**: Microbiome functions are encoded within the genes of the community-wide metagenome. A natural question is whether properties of a microbial community can be predicted just from knowing the raw DNA sequences of its members. In this work, we employ set-aggregated genome embeddings (SAGE) to predict community-level abundance profiles, exploiting the few-shot learning capabilities of genomic language models (GLMs). We benchmark this approach to show improved generalization on novel genomes compared to classical bioinformatics approaches. Model ablation shows that community-level latent representations directly result in improved performance. Lastly, we demonstrate the benefits of intermediate transformations between latent representations and demonstrate the differences between GLM embedding choices. 

---
# Iterative Audit Convergence in LLM-Managed Multi-Agent Systems: A Case Study in Prompt Engineering Quality Assurance 

**Authors**: Elias Calboreanu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12280)  

**Abstract**: Prompt specifications for multi-agent large language model (LLM) systems carry data contracts and integration logic across many interdependent files but are rarely subjected to structured-inspection rigor. This paper reports a single-system empirical case study of iterative, agent-driven auditing applied to AEGIS (Autonomous Engineering Governance and Intelligence System), a production seven-lane orchestration pipeline whose prompt-specification surface comprises approximately 7150 lines: 6907 across seven lane this http URL files and a 245-line shared Ticket Contract. Nine sequential audit rounds, executed by Claude sub-agents using a checklist-driven walkthrough adapted from Weinberg and Freedman, surfaced 51 prompt-specification consistency defects, distinct from the 51 STRIDE-categorized adversarial code findings reported in the companion preprint. Per-round counts were 15, 8, 12, 2, 8, 1, 4, 1, and 0. We report a seven-category post-hoc defect taxonomy with explicit coding rules, observed non-monotonic convergence consistent with cascading edits and audit-scope expansion, and an audit protocol distilled from the study, with the final locked checklist released as a reproducibility appendix. Single-file review missed defect classes that were surfaced only by later expanded-scope rounds in this system. The same LLM family authored and audited the specifications; replication with dissimilar models and human reviewers is required before generalization. 

---
# Reconnecting Fragmented Citation Networks with Semantic Augmentation 

**Authors**: Vu Thi Huong, Annika Buchholz, Imene Khebouri, Thorsten Koch, Tim Kunt, Wolfgang Peters-Kottig, Tomasz Stompor, Janina Zittel  

**Link**: [PDF](https://arxiv.org/pdf/2605.12263)  

**Abstract**: Citation graphs are fundamental tools for modeling scientific structure, but are often fragmented due to missing citations of scientifically connected articles. To address this issue, we propose a computationally efficient hybrid framework integrating citation topology with large language model (LLM)-based text similarity. Using 662,369 Web of Science publications in Mathematics and Operations Research & Management Science, we augment the original graph by adding semantic edges from small, disconnected components and weighting existing citations according to textual similarity. Semantic augmentation substantially reduces fragmentation while preserving disciplinary homogeneity. Compared to embedding-only clustering, cluster detection on augmented graphs using the Leiden algorithm retains structural interpretability while offering multi-scale organization. The method scales efficiently to large datasets and offers a practical strategy for strengthening citation-based indicators without collapsing disciplinary boundaries. 

---
# Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs 

**Authors**: Deepak Kumar, Baban Gain, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2605.12242)  

**Abstract**: Automatic Speech Recognition (ASR) transcripts often contain disfluencies, such as fillers, repetitions, and false starts, which reduce readability and hinder downstream applications like chatbots and voice assistants. If left unaddressed, such disfluencies can significantly degrade the reliability of downstream systems. Most existing approaches rely on classical models that focus on identifying disfluent tokens for removal. While this strategy is effective to some extent, it often disrupts grammatical structure and semantic coherence, leading to incomplete or unnatural sentences. Recent literature explored the use of large language models (LLMs); however, these efforts have primarily focused on disfluency detection or data augmentation, rather than performing comprehensive correction. We propose a multilingual correction pipeline where a sequence tagger first marks disfluent tokens, and these signals guide instruction fine-tuning of an LLM to rewrite transcripts into fluent text. To further improve reliability, we add a contrastive learning objective that penalizes the reproduction of disfluent tokens, encouraging the model to preserve grammar and meaning while removing disfluent artifacts. Our experiments across three Indian languages, namely Hindi, Bengali, and Marathi show consistent improvements over strong baselines, including multilingual sequence-to-sequence models. These results highlight that detection-only strategies are insufficient. Combining token-level cues with instruction tuning and contrastive learning provides a practical and scalable solution for multilingual disfluency correction in speech-driven NLP systems. We make the codes publicly available at this https URL. 

---
# Pretraining Strategies and Scaling for ECG Foundation Models: A Systematic Study 

**Authors**: M A Al-Masud, Nils Strodthoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.12241)  

**Abstract**: Specialized foundation models are beginning to emerge in various medical subdomains, but pretraining methodologies and parametric scaling with the size of the pretraining dataset are rarely assessed systematically and in a like-for-like manner. This work focuses on foundation models for electrocardiography (ECG) data, one of the most widely captured physiological time series world-wide. We present a comprehensive assessment of pretraining methodologies, covering five different contrastive and non-contrastive self-supervised learning objectives for ECG foundation models, and investigate their scaling behavior with pretraining dataset sizes up to 11M input samples, exclusively from publicly available sources. Pretraining strategy has a meaningful and consistent impact on downstream performance, with contrastive predictive coding (slightly ahead of JEPA) yielding the most transferable representations across diverse clinical tasks. Scaling pretraining data continues to yield meaningful improvements up to 11M samples for most objectives. We also compare model architectures across all pretraining methodologies and find evidence for a clear superiority of structured state space models compared to transformers and CNN models. We hypothesize that the strong inductive biases of structured state space models, rather than pretraining scale alone, are the primary driver of effective ECG representation learning, with important implications for future foundation model development in this and potentially other physiological signal domains. 

---
# Harness Engineering as Categorical Architecture 

**Authors**: Bogdan Banu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12239)  

**Abstract**: The agent harness, the system layer comprising prompts, tools, memory, and orchestration logic that surrounds the model, has emerged as the central engineering abstraction for LLMbased agents. Yet harness design remains ad hoc, with no formal theory governing composition, preservation of properties under compilation, or systematic comparison across frameworks. We show that the categorical Architecture triple (G, Know, Phi) from the ArchAgents framework provides exactly this formalization. The four pillars of agent externalization (Memory, Skills, Protocols, Harness Engineering) map onto the triple's components: Memory as coalgebraic state, Skills as operad-composed objects, Protocols as syntactic wiring G, and the full Harness as the Architecture itself. Structural guarantees-integrity gates, quality-based escalation, supported convergence checks-are Know-level certificates whose preservation is structural replay: our compiler checks identity and verifier replay, not output-layer correctness or model behavior. We validate this correspondence with a reference implementation featuring compiler functors targeting Swarms, DeerFlow, Ralph, Scion, and LangGraph: the four configuration compilers preserve three named certificate types by identity or replay, and LangGraph preserves the same certificates through its shared per-stage execution path. The LangGraph compiler creates one node per stage using the same per-stage method as the native runtime, providing LangGraph-native observability without reimplementing harness logic. An end-to-end escalation experiment with real LLM agents confirms that the quality-based escalation control path is model-parametric in this two-model, one-task experiment. The result positions categorical architecture as the formal theory behind harness engineering. 

---
# TMRL: Diffusion Timestep-Modulated Pretraining Enables Exploration for Efficient Policy Finetuning 

**Authors**: Matthew M. Hong, Jesse Zhang, Anusha Nagabandi, Abhishek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.12236)  

**Abstract**: Fine-tuning pre-trained robot policies with reinforcement learning (RL) often inherits the bottlenecks introduced by pre-training with behavioral cloning (BC), which produces narrow action distributions that lack the coverage necessary for downstream exploration. We present a unified framework that enables the exploration necessary to enable efficient robot policy finetuning by bridging BC pre-training and RL fine-tuning. Our pre-training method, Context-Smoothed Pre-training (CSP), injects forward-diffusion noise into policy inputs, creating a continuum between precise imitation and broad action coverage. We then fine-tune pre-trained policies via Timestep-Modulated Reinforcement Learning (TMRL), which trains the agent to dynamically adjust this conditioning during fine-tuning by modulating the diffusion timestep, granting explicit control over exploration. Integrating seamlessly with arbitrary policy inputs, e.g., states, 3D point clouds, or image-based VLA policies, we show that TMRL improves RL fine-tuning sample efficiency. Notably, TMRL enables successful real-world fine-tuning on complex manipulation tasks in under one hour. Videos and code available at this https URL. 

---
# No More, No Less: Task Alignment in Terminal Agents 

**Authors**: Sina Mavali, David Pape, Jonathan Evertz, Samira Abedini, Devansh Srivastav, Thorsten Eisenhofer, Sahar Abdelnabi, Lea Schönherr  

**Link**: [PDF](https://arxiv.org/pdf/2605.12233)  

**Abstract**: Terminal agents are increasingly capable of executing complex, long-horizon tasks autonomously from a single user prompt. To do so, they must interpret instructions encountered in the environment (e.g., README files, code comments, stack traces) and determine their relevance to the task. This creates a fundamental challenge: relevant cues must be followed to complete a task, whereas irrelevant or misleading ones must be ignored. Existing benchmarks do not capture this ability. An agent may appear capable by blindly following all instructions, or appear robust by ignoring them altogether. We introduce TAB (Task Alignment Benchmark), a suite of 89 terminal tasks derived from Terminal-Bench 2.1. Each task is intentionally underspecified, with missing information provided as a necessary cue embedded in a natural environmental artifact, alongside a plausible but irrelevant distractor. Solving these tasks requires selectively using the cue while ignoring the distractor. Applying TAB to ten frontier agents reveals a systematic gap between task capability and task alignment. The strongest Terminal-Bench agent achieves high task completion but low task alignment on TAB. Evaluating six prompt-injection defenses further shows that suppressing distractor execution also suppresses the cues required for task completion. These results demonstrate that task-aligned agents require selective use of environmental instructions rather than blanket acceptance or rejection. 

---
# TriBand-BEV: Real-Time LiDAR-Only 3D Pedestrian Detection via Height-Aware BEV and High-Resolution Feature Fusion 

**Authors**: Mohammad Khoshkdahan, Alexey Vinel  

**Link**: [PDF](https://arxiv.org/pdf/2605.12220)  

**Abstract**: Safe autonomous agents and mobile robots need fast real time 3D perception, especially for vulnerable road users (VRUs) such as pedestrians. We introduce a new bird's eye view (BEV) encoding, which maps the full 3D LiDAR point cloud into a light-weight 2D BEV tensor with three height bands. We explicitly reformulate 3D detection as a 2D detection problem and then reconstruct 3D boxes from the BEV outputs. A single network detects cars, pedestrians, and cyclists in one pass. The backbone uses area attention at deep stages, a hierarchical bidirectional neck over P1 to P4 fuses context and detail, and the head predicts oriented boxes with distribution focal learning for side offsets and a rotated IoU loss. Training applies a small vertical re bin and a mild reflectance jitter in channel space to resist memorization. We use an interquartile range (IQR) filter to remove noisy and outlier LiDAR points during 3D reconstruction. On KITTI dataset, TriBand-BEV attains 58.7/52.6/47.2 pedestrian BEV AP(%) for easy, moderate, and hard at 49 FPS on a single consumer GPU, surpassing Complex-YOLO, with gains of +12.6%, +7.5%, and +3.1%. Qualitative scenes show stable detection under occlusion. The pipeline is compact and ready for real time robotic deployment. Our source code is publicly available on GitHub. 

---
# Heterogeneous SoC Integrating an Open-Source Recurrent SNN Accelerator for Neuromorphic Edge Computing on FPGA 

**Authors**: Michelangelo Barocci, Vittorio Fra, Enrico Macii, Gianvito Urgese  

**Link**: [PDF](https://arxiv.org/pdf/2605.12217)  

**Abstract**: The growing popularity of Spiking Neural Networks (SNNs) and their applications has led to a significant fast-paced increase of neuromorphic architectures capable of mimicking the spike-based data processing typical of biological neurons. The efficient power consumption and parallel computing capabilities of the SNNs lead researchers towards the development of digital accelerators, which exploit such features to bring fast and low-power computation on edge devices. The spread of digital neuromorphic hardware however is slowed down by the prohibitive costs that the silicon tape out of circuits brings, that's why targeting Field Programmable Gate Arrays (FPGAs) could represent a viable alternative, offering a flexible and cost-effective platform for implementing digital neuromorphic systems and helping the spread of open-source hardware designs. In this work we present an heterogeneous System-on-Chip (SoC) where the operations of ReckOn, a Recurrent SNN accelerator, are managed through the integration with traditional processors. These include the RISC-V-based, open-source microcontroller X-HEEP and the ARM processor featured in Zynq Ultrascale systems. We validate our design by reproducing the classification results through the implementation on FPGA of the taped-out version of ReckOn in order to check the equivalence of the accuracy and the characteristics in terms of physical implementation. In a second set of experiments, we evaluate the online learning capability of the solution in classifying a subset of the Braille digit dataset recently used to compare neuromorphic frameworks and platforms. 

---
# Self-Supervised Laplace Approximation for Bayesian Uncertainty Quantification 

**Authors**: Julian Rodemann, Alexander Marquard, Thomas Augustin, Michele Caprio  

**Link**: [PDF](https://arxiv.org/pdf/2605.12208)  

**Abstract**: Approximate Bayesian inference typically revolves around computing the posterior parameter distribution. In practice, however, the main object of interest is often a model's predictions rather than its parameters. In this work, we propose to bypass the parameter posterior and focus directly on approximating the posterior predictive distribution. We achieve this by drawing inspiration from self-training within self-supervised and semi-supervised learning. Essentially, we quantify a Bayesian model's predictive uncertainty by refitting on self-predicted data. The idea is strikingly simple: If a model assigns high likelihood to self-predicted data, these predictions are of low uncertainty, and vice versa. This yields a deterministic, sampling-free approximation of the posterior predictive. The modular structure of our Self-Supervised Laplace Approximation (SSLA) further allows us to plug in different prior specifications, enabling classical Bayesian sensitivity (w.r.t. prior choice) analysis. In order to bypass expensive refitting, we further introduce an approximate version of SSLA, called ASSLA. We study (A)SSLA both theoretically and empirically in regression models ranging from Bayesian linear models to Bayesian neural networks. Across a wide array of regression tasks with simulated and real-world datasets, our methods outperform classical Laplace approximations in predictive calibration while remaining computationally efficient. 

---
# Not How Many, But Which: Parameter Placement in Low-Rank Adaptation 

**Authors**: Arijit Sehanobish, Charles Lovering  

**Link**: [PDF](https://arxiv.org/pdf/2605.12207)  

**Abstract**: We study the \textit{parameter placement problem}: given a fixed budget of $k$ trainable entries within the B matrix of a LoRA adapter (A frozen), does the choice of which $k$ matter? Under supervised fine-tuning, random and informed subsets achieve comparable performance. Under GRPO on base models, random placement fails to improve over the base model, while gradient-informed placement recovers standard LoRA accuracy. This regime dependence traces to gradient structure: SFT gradients are low-rank and directionally stable, so any subset accumulates coherent updates; GRPO gradients are high-rank and near-orthogonal across steps, so only elements with consistently signed gradients retain the learning signal. Our scoring procedure identifies these critical parameters in under 10 seconds at less than 0.5% of training cost. Selected parameters concentrate on residual-stream-writing projections (V, O, Down), stable across model families and scales (1.5B - 8B). 

---
# Uncertainty Quantification for LLM-based Code Generation 

**Authors**: Senrong Xu, Yuhao Tan, Yanke Zhou, Guangyuan Wu, Zenan Li, Yuan Yao, Taolue Chen, Feng Xu, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.12201)  

**Abstract**: Prediction sets provide a theoretically grounded framework for quantifying uncertainty in machine learning models. Adapting them to structured generation tasks, in particular, large language model (LLM) based code generation, remains a challenging problem. An existing attempt proposes PAC prediction sets but is limited by its strong monotonicity assumption on risk and single-label classification framework, which severely limits the space of candidate programs and cannot accommodate the multiple valid outputs inherent to code generation. To address these limitations, we propose an approach RisCoSet that leverages multiple hypothesis testing to construct risk-controlling predictions for LLM-based code generation. Given a trained code generation model, we produce a prediction set represented by a partial program, which is guaranteed to contain a correct solution with high confidence. Extensive experiments on three LLMs demonstrate the effectiveness of the proposed method. For instance, compared with the state-of-the-art, our method can significantly reduce the code removal by up to 24.5%, at the same level of risk. 

---
# Overtrained, Not Misaligned 

**Authors**: Joel Schreiber, Ariel Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2605.12199)  

**Abstract**: Emergent misalignment (EM), where fine-tuning on a narrow task (like insecure code) causes broad misalignment across unrelated domains, was first demonstrated by Betley et al. (2025). We conduct the most comprehensive EM study to date, reproducing the original GPT-4o finding and expanding to 12 open-source models across 4 families (Llama, Qwen, DeepSeek, GPT-OSS) ranging from 8B to 671B parameters, evaluating over one million model responses with multiple random seeds. We find that EM replicates in GPT-4o but is far from universal: only 2 of 12 open-source models (17%) exhibit consistent EM across seeds, with a significant correlation between model size and EM susceptibility. Through checkpoint-level analysis during fine-tuning, we demonstrate that EM emerges late in training, distinct from and subsequent to near convergence of the primary task, suggesting EM emerges from continued training past task convergence. This yields practical mitigations: early stopping eliminates EM while retaining an average of 93% of task performance, and careful learning rate selection further minimizes risk. Cross-domain validation on medical fine-tuning confirms these patterns generalize: the size-EM correlation strengthens (r = 0.90), and overgeneralization to untruthfulness remains avoidable via early stopping in 67% of cases, though semantically proximate training domains produce less separable misalignment. As LLMs become increasingly integrated into real-world systems, fine-tuning and reinforcement learning remain the primary methods for adapting model behavior. Our findings demonstrate that with proper training practices, EM can be avoided, reframing it from an unforeseen fine-tuning risk to an avoidable training artifact. 

---
# Mitigating Context-Memory Conflicts in LLMs through Dynamic Cognitive Reconciliation Decoding 

**Authors**: Yigeng Zhou, Wu Li, Yifan Lu, Yequan Wang, Xuebo Liu, Wenya Wang, Jun Yu, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.12185)  

**Abstract**: Large language models accumulate extensive parametric knowledge through pre-training. However, knowledge conflicts occur when outdated or incorrect parametric knowledge conflicts with external knowledge in the context. Existing methods address knowledge conflicts through contrastive decoding, but in conflict-free scenarios, static approaches disrupt output distribution. Other dynamic decoding methods attempt to measure the degree of conflict but still struggle with complex real-world situations. In this paper, we propose a two-stage decoding method called Dynamic Cognitive Reconciliation Decoding (DCRD), to predict and mitigate context-memory conflicts. DCRD first analyzes the attention map to assess context fidelity and predict potential conflicts. Based on this prediction, the input is directed to one of two decoding paths: (1) greedy decoding, or (2) context fidelity-based dynamic decoding. This design enables DCRD to handle conflicts efficiently while maintaining high accuracy and decoding efficiency in conflict-free cases. Additionally, to simulate scenarios with frequent knowledge updates, we constructed ConflictKG, a knowledge conflict QA benchmark. Experiments on four LLMs across six QA datasets show that DCRD outperforms all baselines, achieving state-of-the-art performance. 

---
# DriftXpress: Faster Drifting Models via Projected RKHS Fields 

**Authors**: Ali Falahati, Elliot Creager, Gautam Kamath, Shubhankar Mohapatra  

**Link**: [PDF](https://arxiv.org/pdf/2605.12183)  

**Abstract**: Drifting Models have emerged as a new paradigm for one-step generative modeling, achieving strong image quality without iterative inference. The premise is to replace the iterative denoising process in diffusion models with a single evaluation of a generator. However, this creates a different trade-off: drifting reduces inference cost by moving much of the computation into training. We introduce DriftXpress, an accelerated formulation of drifting models based on projected RKHS fields. DriftXpress approximates the drifting kernel in a low-rank feature space. This preserves the attraction-repulsion structure of the original drifting field while reducing the cost of field evaluation. Across image-generation benchmarks, DriftXpress achieves comparable FID to standard drifting while reducing wall-clock training cost. These results show that the training-inference trade-off of drifting models can be pushed further without giving up their one-step inference advantage. 

---
# A Deep Learning-based Receiver for Asynchronous Grant-Free Random Access in Control-to-Control Networks 

**Authors**: Massimo Battaglioni, Edoardo Carnevali, Dania De Crescenzo, Enrico Testi, Marco Baldi, Enrico Paolini  

**Link**: [PDF](https://arxiv.org/pdf/2605.12180)  

**Abstract**: In this paper, we study grant-free, asynchronous control-to-control (C2C) communications in an indoor scenario with a shared wireless channel. Each communication node transmits command units, each consisting of a variable-length low-density parity-check (LDPC)--coded payload preceded by a start sequence and followed by a tail sequence. Due to the asynchronous nature of the access, transmissions from different nodes are not aligned over time. As a result, each receiving controller observes the superposition of multiple command units transmitted by different nodes over a receiver-defined superframe interval. Each node transmits one or more replicas of the same command unit. We propose a receiver architecture in which the detection of command unit boundaries (start/tail sequences) is carried out by a single convolutional neural network (CNN) operating directly on the received signal. We show that, while start-sequence detection must rely only on the received waveform, tail-sequence detection can additionally exploit the soft information produced by the LDPC decoder, together with channel estimates. Finally, once commands units are successfully decoded, successive interference cancellation (SIC) can be applied. Simulation results demonstrate that the receiver we propose achieves reliable packet-boundary identification and a low end-to-end packet loss rate, even under uncoordinated and high-traffic operating conditions. 

---
# Premover: Fast Vision-Language-Action Control by Acting Before Instructions Are Complete 

**Authors**: Joonha Park, Jiseung Jeong, Taesik Gong  

**Link**: [PDF](https://arxiv.org/pdf/2605.12160)  

**Abstract**: Vision-Language-Action (VLA) policies are typically evaluated as if the user had finished typing or speaking before the robot begins acting. In real deployment, however, users take several seconds to enter a request, leaving the policy idle for a substantial fraction of the interaction. We introduce Premover, a lightweight module that converts this idle window into useful precomputation. Premover keeps the VLA backbone frozen and attaches two small projection heads, one for image patches, one for language tokens, that map an intermediate layer of the backbone into a shared space. The resulting focus map is supervised by simulator-rendered target-object segmentation masks and applied as a per-patch reweighting of the next step's image tokens. A single scalar readiness threshold, trained jointly from streaming prefixes, decides when the policy should begin acting. On the LIBERO benchmark suite, Premover reduces mean wall-clock time from 34.0 to 29.4 seconds, a 13.6% reduction, while matching the full-prompt baseline's success rate (95.1% vs. 95.0%); naive premoving, by contrast, collapses to 66.4%. 

---
# CIDR: A Large-Scale Industrial Source Code Dataset for Software Engineering Research 

**Authors**: Vladislav Savenkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.12153)  

**Abstract**: We present Curated Industrial Developer Repository (CIDR), a large-scale dataset of real-world software repositories collected through direct collaboration with 12 industrial partner organizations. The dataset comprises 2,440 repositories spanning 138 programming languages and totalling 373 million lines of code, accompanied by structured per-repository metadata. Unlike existing code corpora derived from public open-source platforms, CIDR consists exclusively of proprietary production codebases contributed under formal data sharing agreements, covering application domains including enterprise web and mobile development, fintech, and custom software consultancy. All repositories were processed through a multi-stage pipeline encompassing structured partner onboarding, two-stage quality selection combining automated metadata filtering with manual code review, and a deterministic anonymization pipeline covering the full version control history. The dataset is intended to support research in code intelligence, software quality analysis, pre-training and fine-tuning of code language models, developer behaviour studies, and construction of agent evaluation benchmarks. Access is provided under a restricted commercial license; details are available at this https URL. 

---
# It's Not the Size: Harness Design Determines Operational Stability in Small Language Models 

**Authors**: Yong-eun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2605.12129)  

**Abstract**: This paper experimentally analyzes how the level of harness engineering affects the operational performance of small language models (SLMs, 2-3B parameters). Three harness conditions - model-only (raw prompt), minimal-shell (wrapper tags), and a 4-stage pipeline (plan->execute->verify->recover) - are applied to three models (Gemma4 E2B, Qwen3.5:2B, LLaMA 3.2 3B) across 24 tasks, comparing Task Success Rate (TSR) and Valid TSR (VTSR). The pipeline harness achieves TSR=0.952 and VTSR=1.000 on Gemma4 E2B (T1-T5, 21 tasks). A non-monotonic phenomenon - minimal-shell TSR < model-only TSR - is observed in two models. In LLaMA 3.2 3B model-only, seven format violations yield TSR=0.429, revealing scaffold collapse: the model abandons JSON structure under complex format requirements without harness support. Ablation shows planning and recovery each contribute approximately 24.7% of total gain. VCR (Verification Catch Rate)=0.625 across all pipeline runs. 

---
# Disentangled Sparse Representations for Concept-Separated Diffusion Unlearning 

**Authors**: Hyeonjin Kim, Hangyeol Jung, Heechan Yun, Sungjun Yun, Dong-Jun Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.12122)  

**Abstract**: Unlearning specific concepts in text-to-image diffusion models has become increasingly important for preventing undesirable content generation. Among prior approaches, sparse autoencoder (SAE)-based methods have attracted attention due to their ability to suppress target concepts through lightweight manipulation of latent features, without modifying model parameters. However, SAEs trained with sparse reconstruction objectives do not explicitly enforce concept-wise separation, resulting in shared latent features across concepts. To address this, we propose SAEParate, which organizes latent representations into concept-specific clusters via a concept-aware contrastive objective, enabling more precise concept suppression while reducing unintended interference during unlearning. In addition, we enhance the encoder with a GeLU-based nonlinear transformation to increase its expressive capacity under this separation objective, enabling a more discriminative and disentangled latent space. Experiments on UnlearnCanvas demonstrate state-of-the-art performance, with particularly strong gains in joint style-object unlearning, a challenging setting where existing methods suffer from severe interference between target and non-target concepts. 

---
# Learning What Matters: Adaptive Information-Theoretic Objectives for Robot Exploration 

**Authors**: Youwei Yu, Jionghao Wang, Zhengming Yu, Wenping Wang, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12084)  

**Abstract**: Designing learnable information-theoretic objectives for robot exploration remains challenging. Such objectives aim to guide exploration toward data that reduces uncertainty in model parameters, yet it is often unclear what information the collected data can actually reveal. Although reinforcement learning (RL) can optimize a given objective, constructing objectives that reflect parametric learnability is difficult in high-dimensional robotic systems. Many parameter directions are weakly observable or unidentifiable, and even when identifiable directions are selected, omitted directions can still influence exploration and distort information measures. To address this challenge, we propose Quasi-Optimal Experimental Design (Q{\footnotesize OED}), an adaptive information objective grounded in optimal experimental design. Q{\footnotesize OED} (i) performs eigenspace analysis of the Fisher information matrix to identify an observable subspace and select identifiable parameter directions, and (ii) modifies the exploration objective to emphasize these directions while suppressing nuisance effects from non-critical parameters. Under bounded nuisance influence and limited coupling between critical and nuisance directions, Q{\footnotesize OED} provides a constant-factor approximation to the ideal information objective that explores all parameters. We evaluate Q{\footnotesize OED} on simulated and real-world navigation and manipulation tasks, where identifiable-direction selection and nuisance suppression yield performance improvements of \SI{35.23}{\percent} and \SI{21.98}{\percent}, respectively. When integrated as an exploration objective in model-based policy optimization, Q{\footnotesize OED} further improves policy performance over established RL baselines. 

---
# Property-Level Reconstructability of Agent Decisions: An Anchor-Level Pilot Across Vendor SDK Adapter Regimes 

**Authors**: Oleg Solozobov  

**Link**: [PDF](https://arxiv.org/pdf/2605.12078)  

**Abstract**: Agentic AI failures need post-hoc reconstruction: what the agent did, on whose authority, against which policy, and from what reasoning. Cross-regime feasibility remains unmeasured under one property-level schema. We apply the Decision Trace Reconstructor unmodified to pinned worked-example anchors from six public vendor SDK regimes spanning cloud-agent, observability, tool-use, telemetry, and protocol traces, plus two comparator columns. Each Decision Event Schema (DES) property is classified as fully fillable, partially fillable, structurally unfillable, or opaque. Per-property reconstructability of an agent decision already varies between regimes at this anchor scale. Strict-governance-completeness separates into three tiers ranging from 42.9% to 85.7%, yielding one regime-independent gap (reasoning trace), four regime-dependent gaps, and one Mixed property; the pilot is single-annotator, one anchor per cell, descriptive, with outputs checksum-verifiable from a deposited reproducibility package. 

---
# The Missing GAP: From Solving Square Jigsaw Puzzles to Handling Real World Archaeological Fragments 

**Authors**: Ofir Itzhak Shahar, Gur Elkin, Ohad Ben-Shahar  

**Link**: [PDF](https://arxiv.org/pdf/2605.12077)  

**Abstract**: Jigsaw puzzle solving has been an increasingly popular task in the computer vision research community. Recent works have utilized cutting-edge architectures and computational approaches to reassemble groups of pieces into a coherent image, while achieving increasingly good results on well established datasets. However, most of these approaches share a common, restricting setting: operating solely on strictly square puzzle pieces. In this work, we introduce GAP, a set of novel jigsaw puzzles datasets containing synthetic, heavily eroded pieces of unrestricted shapes, generated by a learned distribution of real-world archaeological fragments. We also introduce PuzzleFlow, a novel ViT and Flow-Matching based framework for jigsaw puzzle solving, capable of handling complex puzzle pieces and demonstrating superior performance on GAP when compared to both classic and recent prominent works in this domain. 

---
# The Deepfakes We Missed: We Built Detectors for a Threat That Didn't Arrive 

**Authors**: Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2605.12075)  

**Abstract**: Nearly a decade of Machine Learning (ML) research on deepfake detection has been organized around a threat model inherited from 2017--2019, revolving around face-swap and talking-head manipulation of public figures, motivated by concerns about large-scale misinformation and video-evidence fraud. This position paper argues that the threat the field prepared for did not arrive, and the threats that did arrive are substantially different. An accounting of deepfake incidents in 2022--2026 shows that the dominant observed harms are peer-generated Non-Consensual Intimate Imagery (NCII), voice-clone scam calls targeting families and finance workers, and emotional-manipulation fraud. The predicted large-scale public-figure deepfake catastrophe did not materialize during the 2024 global information environment despite extensive preparation. Meanwhile, research effort, benchmarks, and detection methods remain concentrated on the inherited threat model. The central claim of this paper is that this misalignment is now the dominant bottleneck on real-world deepfake defense, not model capability. We argue the ML research community should substantially rebalance its research agenda toward the harm categories that are actually growing. We support this position with empirical accounting of research effort and harm distribution, identify the structural reasons the misalignment persists, and outline three concrete technical research agendas for the under-defended harm categories. 

---
# Clausal Deletion Backdoors for QBF: a Parameterized Complexity Approach 

**Authors**: Leif Eriksson, Victor Lagerkvist, Sebastian Ordyniak, George Osipov, Fahad Panolan, Mateusz Rychlicki  

**Link**: [PDF](https://arxiv.org/pdf/2605.12073)  

**Abstract**: Determining the validity of a quantified Boolean formula (QBF) is a PSPACE-complete problem with rich expressive power. Despite interest in efficient solvers, there is, compared to problems in NP, a lack of positive theoretical results, and in the parameterized complexity setting one often has to restrict the quantifier prefix (e.g., bounding alternations) to obtain fixed parameter tractability (FPT). We propose a new parameter: the number of variables in clauses that has to be removed before reaching a tractable class (a clause covering (CC) backdoor). We are then interested in solving QBF in FPT time given a CC-backdoor of size $k$. We consider the three classical, tractable cases of QBF as base classes: Horn, 2-CNF, and linear equations. We establish W[1]-hardness for Horn but prove FPT for the others, and prove that in a precise, algebraic sense, we are only missing one important case for a full dichotomy. Our algorithms are non-trivial and depend on propagation, and Gaussian elimination, respectively, and are comparably unexplored for QBF. 

---
# Missing Old Logits in Asynchronous Agentic RL: Semantic Mismatch and Repair Methods for Off-Policy Correction 

**Authors**: Zhong Guan, Yongjian Guo, Haoran Sun, Wen Huang, Shuai Di, Xiong Jun Wu, Likang Wu, Hongke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12070)  

**Abstract**: Asynchronous reinforcement learning improves rollout throughput for large language model agents by decoupling sample generation from policy optimization, but it also introduces a critical failure mode for PPO-style off-policy correction. In heterogeneous training systems, the total importance ratio should ideally be decomposed into two semantically distinct factors: a \emph{training--inference discrepancy term} that aligns inference-side and training-side distributions at the same behavior-policy version, and a \emph{policy-staleness term} that constrains the update from the historical policy to the current policy. We show that practical asynchronous pipelines with delayed updates and partial rollouts often lose the required historical training-side logits, or old logits. This missing-old-logit problem entangles discrepancy repair with staleness correction, breaks the intended semantics of decoupled correction, and makes clipping and masking thresholds interact undesirably. To address this issue, we study both exact and approximate correction routes. We propose three exact old-logit acquisition strategies: snapshot-based version tracking, a dedicated old-logit model, and synchronization via partial rollout interruption, and compare their system trade-offs. From the perspective of approximate correction, we focus on preserving the benefits of decoupled correction through a more appropriate approximate policy when exact old logits cannot be recovered at low cost, without incurring extra system overhead. Following this analysis, we adopt a revised PPO-EWMA method, which achieves significant gains in both training speed and optimization performance. Code at this https URL. 

---
# Anomaly-Aware Vision-Language Adapters for Zero-Shot Anomaly Detection 

**Authors**: Muhammad Aqeel, Maham Nazir, Uzair Khan, Marco Cristani, Francesco Setti  

**Link**: [PDF](https://arxiv.org/pdf/2605.12069)  

**Abstract**: Zero-shot anomaly detection aims to identify defects in unseen categories without target-specific training. Existing methods usually apply the same feature transformation to all samples, treating normal and anomalous data uniformly despite their fundamentally asymmetric distributions, compact normals versus diverse anomalies. We instead exploit this natural asymmetry by proposing AVA-DINO, an anomaly-aware vision-language adaptation framework with dual specialized branches for normal and anomalous patterns that adapt frozen DINOv3 visual features. During training on auxiliary data, the two branches are learned jointly with a text-guided routing mechanism and explicit routing regularization that encourages branch specialization. At test time, only the input image and fixed, predefined language descriptions are used to dynamically combine the two branches, enabling an asymmetric activation. This design prevents degenerate uniform routing and allows context-specific feature transformations. Experiments across nine industrial and medical benchmarks demonstrate state-of-the-art performance, achieving 93.5% image-AUROC on MVTec-AD and strong cross-domain generalization to medical imaging without domain-specific fine-tuning. this https URL 

---
# Hölder Policy Optimisation 

**Authors**: Yuxiang Chen, Dingli Liang, Yihang Chen, Ziqin Gong, Chenyang Le, Zhaokai Wang, Jiachen Zhu, Lingyu Yang, Jianghao Lin, Weinan Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12058)  

**Abstract**: Group Relative Policy Optimisation (GRPO) enhances large language models by estimating advantages across a group of sampled trajectories. However, mapping these trajectory-level advantages to policy updates requires aggregating token-level probabilities within each sequence. Relying on a fixed aggregation mechanism for this step fundamentally limits the algorithm's adaptability. Empirically, we observe a critical trade-off: certain fixed aggregations frequently suffer from training collapse, while others fail to yield satisfactory performance. To resolve this, we propose \textbf{HölderPO}, a generalised policy optimisation framework unifying token-level probability aggregation via the Hölder mean. By explicitly modulating the parameter $p$, our framework provides continuous control over the trade-off between gradient concentration and variance bounds. Theoretically, we prove that a larger $p$ concentrates the gradient to amplify sparse learning signals, whereas a smaller $p$ strictly bounds gradient variance. Because no static configuration can universally resolve this concentration-stability trade-off, we instantiate the framework with a dynamic annealing algorithm that progressively schedules $p$ across the training lifecycle. Extensive evaluations demonstrate superior stability and convergence over existing baselines. Specifically, our approach achieves a state-of-the-art average accuracy of $54.9\%$ across multiple mathematical benchmarks, yielding a substantial $7.2\%$ relative gain over standard GRPO and secures an exceptional $93.8\%$ success rate on ALFWorld. 

---
# Scaling Laws and Tradeoffs in Recurrent Networks of Expressive Neurons 

**Authors**: Aaron Spieler, Georg Martius, Anna Levina  

**Link**: [PDF](https://arxiv.org/pdf/2605.12049)  

**Abstract**: Cortical neurons are complex, multi-timescale processors wired into recurrent circuits, shaped by long evolutionary pressure under stringent biological constraints. Mainstream machine learning, by contrast, predominantly builds models from extremely simple units, a default inherited from early neural-network theory. We treat this as a normative architectural question. How should one split a fixed parameter budget $P$ between the number of units $N$, per-unit effective complexity $k_e$, and per-unit connectivity $k_c$? What controls the optimal allocation? This calls for a model in which per-unit complexity can be tuned independently of width and connectivity. Accordingly, we introduce the ELM Network, whose recurrent layer is built from Expressive Leaky Memory (ELM) neurons, chosen to mirror functional components of cortical neurons. The architecture allows for individually adjusting $N$, $k_e$, and $k_c$ and trains stably across orders of magnitude in scale. We evaluate the model on two qualitatively different sequence benchmarks: the neuromorphic SHD-Adding task and Enwik8 character-level language modeling. Performance improves monotonically along each of the three axes individually. Under a fixed budget, a clear non-trivial optimum emerges in their tradeoff, and larger budgets favor both more and more complex neurons. A closed-form information-theoretic model captures these tradeoffs and attributes the diminishing returns at two ends to: per-neuron signal-to-noise saturation and across-neuron redundancy. A hyperparameter sweep spanning three orders of magnitude in trainable parameters traces a near-Pareto-frontier scaling law consistent with the framework. This suggests that the simple-unit default in ML is not obviously optimal once this tradeoff surface is probed, and offers a normative lens on cortex's reliance on complex spatio-temporal integrators. 

---
# Rethink the Role of Neural Decoders in Quantum Error Correction 

**Authors**: Ge Yan, Shanchuan Li, Yuxuan Du  

**Link**: [PDF](https://arxiv.org/pdf/2605.12046)  

**Abstract**: Quantum error correction (QEC) is essential for enabling quantum advantages, with decoding as a central algorithmic primitive. Owing to its importance and intrinsic difficulty, substantial effort has been made to QEC decoder design, among which neural decoders have recently emerged as a promising data-driven paradigm. Despite this progress, practical deployment remains hindered by a fundamental accuracy-latency tradeoff, often on the microsecond timescale. To address this challenge, here we revisit neural decoders for surface-code decoding under explicit accuracy-latency constraints, considering code distances up to d=9 (161 physical qubits). We unify and redesign representative neural decoders into five architectural paradigms and develop an end-to-end compression pipeline to evaluate their deployability and performance on FPGA hardware. Through systematic experiments, we reveal several previously underexplored insights: (i) near-term decoding performance is driven more by data scale than architectural complexity; (ii) appropriate inductive bias is essential for achieving high decoding accuracy; and (iii) INT4 quantization is a prerequisite for meeting microsecond-scale latency requirements on FPGAs. Together, these findings provide concrete guidance toward scalable and real-time neural QEC decoding. 

---
# Boosting Omni-Modal Language Models: Staged Post-Training with Visually Debiased Evaluation 

**Authors**: Che Liu, Lichao Ma, Xiangyu Tony Zhang, Yuxin Zhang, Haoyang Zhang, Xuerui Yang, Fei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.12034)  

**Abstract**: Omni-modal language models are intended to jointly understand audio, visual inputs, and language, but benchmark gains can be inflated when visual evidence alone is enough to answer a query. We study whether current omni-modal benchmarks separate visual shortcuts from genuine audio-visual-language evidence integration, and how post-training behaves under a visually debiased evaluation setting. We audit nine omni-modal benchmarks with visual-only probing, remove visually solvable queries, and retain full subsets when filtering is undefined or would make comparisons unstable. This yields OmniClean, a cleaned evaluation view with 8,551 retained queries from 16,968 audited queries. On OmniClean, we evaluate OmniBoost, a three-stage post-training recipe based on Qwen2.5-Omni-3B: mixed bi-modal SFT, mixed-modality RLVR, and SFT on self-distilled data. Balanced bi-modal SFT gives limited and uneven gains, RLVR provides the first broad improvement, and self-distillation reshapes the benchmark profile. After SFT on self-distilled data, the 3B model reaches performance comparable to, and in aggregate slightly above, Qwen3-Omni-30B-A3B-Instruct without using a stronger omni-modal teacher. These results show that omni-modal progress is easier to interpret when evaluation controls visual leakage, and that small omni-modal models can benefit from staged post-training with self-distilled omni-query supervision. 

---
# Spectral Vision Transformer for Efficient Tokenization with Limited Data 

**Authors**: Alexandra G. Roberts, Maneesh John, Jinwei Zhang, Dominick Romano, Mert Sisman, Ki Sueng Choi, Heejong Kim, Mert R. Sabuncu, Thanh D. Nguyen, Alexey V. Dimov, Pascal Spincemaille, Brian H. Kopell, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.12026)  

**Abstract**: We propose a novel spectral vision transformer architecture for efficient tokenization in limited data, with an emphasis on medical imaging. We outline convenient theoretical properties arising from the choice of basis including spatial invariance and optimal signal-to-noise ratio. We show reduced complexity arising from the spectral projection compared to spatial vision transformers. We show equitable or superior performance with a reduced number of parameters as compared to a variety of models including compact and standard vision transformers, convolutional neural networks with attention, shifted window transformers, multi-layer perceptrons, and logistic regression. We include simulated, public, and clinical data in our analysis and release our code at: \verb+this http URL. 

---
# Efficient and Adaptive Human Activity Recognition via LLM Backbones 

**Authors**: Aleksandr Bredikhin, Philippe Lalanda, German Vega  

**Link**: [PDF](https://arxiv.org/pdf/2605.12019)  

**Abstract**: Human Activity Recognition (HAR) is a core task in pervasive computing systems, where models must operate under strict computational constraints while remaining robust to heterogeneous and evolving deployment conditions. Recent advances based on Transformer architectures have significantly improved recognition performance, but typically rely on task-specific models trained from scratch, resulting in high training cost, large data requirements, and limited adaptability to domain shifts. In this paper, we propose a paradigm shift that reuses large pretrained language models (LLMs) as generic temporal backbones for sensor-based HAR, instead of designing domain-specific Transformers. To bridge the modality gap between inertial time series and language models, we introduce a structured convolutional projection that maps multivariate accelerometer and gyroscope signals into the latent space of the LLM. The pretrained backbone is kept frozen and adapted using parameter-efficient Low-Rank Adaptation (LoRA), drastically reducing the number of trainable parameters and the overall training cost. Through extensive experiments on standard HAR benchmarks, we show that this approach enables rapid convergence, strong data efficiency, and robust cross-dataset transfer, particularly in low-data and few-shot settings. At the same time, our results highlight the complementary roles of convolutional frontends and LLMs, where local invariances are handled at the signal level while long-range temporal dependencies are captured by the pretrained backbone. Overall, this work demonstrates that LLMs can serve as a practical, frugal, and scalable foundation for adaptive HAR systems, opening new directions for reusing foundation models beyond their original language domain. 

---
# SkillSafetyBench: Evaluating Agent Safety under Skill-Facing Attack Surfaces 

**Authors**: Chang Jin, An Wang, Zeming Wei, Kai Wang, Biaojie Zeng, Qiaosheng Zhang, Chao Yang, Jingjing Qu, Xia Hu, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.12015)  

**Abstract**: Reusable skills are becoming a common interface for extending large language model agents, packaging procedural guidance with access to files, tools, memory, and execution environments. However, this modularity introduces attack surfaces that are largely missed by existing safety evaluations: even when the user request is benign, task-relevant skill materials or local artifacts can steer an agent toward unsafe actions. We present SkillSafetyBench, a runnable benchmark for evaluating such skill-mediated safety failures. SkillSafetyBench includes 155 adversarial cases across 47 tasks, 6 risk domains, and 30 safety categories, each evaluated with a case-specific rule-based verifier. Experiments with multiple CLI agents and model backends show that localized non-user attacks can consistently induce unsafe behavior, with distinct failure patterns across domains, attack methods, and scaffold-model pairings. Our findings suggest that agent safety depends not only on model-level alignment, but also on how agents interpret skills, trust workflow context, and act through executable environments. 

---
# L2P: Unlocking Latent Potential for Pixel Generation 

**Authors**: Zhennan Chen, Junwei Zhu, Xu Chen, Jiangning Zhang, Jiawei Chen, Zhuoqi Zeng, Wei Zhang, Chengjie Wang, Jian Yang, Ying Tai  

**Link**: [PDF](https://arxiv.org/pdf/2605.12013)  

**Abstract**: Pixel diffusion models have recently regained attention for visual generation. However, training advanced pixel-space models from scratch demands prohibitive computational and data resources. To address this, we propose the Latent-to-Pixel (L2P) transfer paradigm, an efficient framework that directly harnesses the rich knowledge of pre-trained LDMs to build powerful pixel-space models. Specifically, L2P discards the VAE in favor of large-patch tokenization and freezes the source LDM's intermediate layers, exclusively training shallow layers to learn the latent-to-pixel transformation. By utilizing LDM-generated synthetic images as the sole training corpus, L2P fits an already smooth data manifold, enabling rapid convergence with zero real-data collection. This strategy allows L2P to seamlessly migrate massive latent priors to the pixel space using only 8 GPUs. Furthermore, eliminating the VAE memory bottleneck unlocks native 4K ultra-high resolution generation. Extensive experiments across mainstream LDM architectures show that L2P incurs negligible training overhead, yet performs on par with the source LDM on DPG-Bench and reaches 93% performance on GenEval. 

---
# CR^2: Cost-Aware Risk-Controlled Routing for Wireless Device-Edge LLM Inference 

**Authors**: Nan Xue, Shengkang Chen, Zhiyong Chen, Jiangchao Yao, Yaping Sun, Zixia Hu, Meixia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2605.12001)  

**Abstract**: As large language models (LLMs) move from centralized clouds to mobile edge environments, efficient serving must balance latency, energy consumption, and accuracy under constrained device-edge resources. Query-level routing between lightweight on-device models and stronger edge models provides a flexible mechanism to navigate this trade-off. However, existing routers are designed for centralized cloud settings and optimize token-level costs, failing to capture the dynamic latency and energy overheads in wireless edge deployments. In this paper, we formulate mobile edge LLM routing as a deployment-constrained, cost-aware decision problem, and propose CR^2, a two-stage device-edge routing framework. CR^2 decouples a lightweight on-device margin gate from an edge-side utility selector for deferred queries. The margin gate operates on frozen query embeddings and a user-specified cost weight to predict whether local execution is utility-optimal relative to the best edge alternative under the target operating point. We further introduce a conformal risk control (CRC) calibration procedure that maps each operating point to an acceptance threshold, enabling explicit control of the marginal false-acceptance risk under the full-information utility reference. Experiments on the routing task show that CR^2 closely matches a full-information reference router using only device-side signals before deferral. Compared with strong query-level baselines, CR^2 consistently improves the deployable accuracy-cost Pareto frontier and reduces normalized deployment cost by up to 16.9% at matched accuracy. 

---
# The Illusion of Power Capping in LLM Decode: A Phase-Aware Energy Characterisation Across Attention Architectures 

**Authors**: Bole Ma, Ayesha Afzal, Jan Eitzinger, Gerhard Wellein  

**Link**: [PDF](https://arxiv.org/pdf/2605.11999)  

**Abstract**: Power capping is the standard GPU energy lever in LLM serving, and it appears to work: throughput drops, power readings fall, and energy budgets are met. We show the appearance is illusory for the phase that dominates production serving: autoregressive decode. Across four attention paradigms -- GQA, MLA, Gated DeltaNet, and Mamba2 -- on NVIDIA H200, decode draws only 137--300\,W on a 700\,W GPU; no cap ever triggers, because memory-bound decode saturates HBM bandwidth rather than compute and leaves power headroom untouched. Firmware-initiated clock throttling compounds the illusion: these deviations can corrupt any throughput measurement that attributes them to the cap. SM clock locking dissolves both confounds. By targeting the lever that is actually on the critical path, clock locking Pareto-dominates power capping universally, recovering up to 32\% of decode energy at minimal throughput loss. We identify three architecture-dependent DVFS behavioural classes and characterise a common energy pattern across novel attention replacements: a heavy prefill cost recouped by efficient decode, eventually halving total request energy relative to GQA at production batch sizes. 

---
# A Transfer Learning Evaluation of Deep Neural Networks for Image Classification 

**Authors**: Nermeen Abou Baker, Nico Zengeler, Uwe Handmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.11989)  

**Abstract**: Transfer learning is a machine learning technique that uses previously acquired knowledge from a source domain to enhance learning in a target domain by reusing learned weights. This technique is ubiquitous because of its great advantages in achieving high performance while saving training time, memory, and effort in network design. In this paper, we investigate how to select the best pre-trained model that meets the target domain requirements for image classification tasks. In our study, we refined the output layers and general network parameters to apply the knowledge of eleven image processing models, pre-trained on ImageNet, to five different target domain datasets. We measured the accuracy, accuracy density, training time, and model size to evaluate the pre-trained models both in training sessions in one episode and with ten episodes. 

---
# High-lift Wing Separation Control via Bayesian Optimization and Deep Reinforcement Learning 

**Authors**: Ricard Montalà, Bernat Font, Oriol Lehmkuhl, Ricardo Vinuesa, Ivette Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2605.11981)  

**Abstract**: This study investigates active flow control (AFC) of a 30P30N high-lift wing at a Reynolds number Re$_c$ = 450,000 and angle of attack $\alpha$ = 23$^\circ$ using wallresolved large-eddy simulations (LES). Two optimization strategies are explored: open-loop Bayesian optimization (BO) and closed-loop deep reinforcement learning (DRL), both targeting the mitigation of stall and the improvement of aerodynamic efficiency via synthetic jets on the slat, main, and flap elements. The uncontrolled configuration was validated against literature data, confirming the reliability of the LES setup. The BO framework successfully identified steady jet velocities that increased efficiency by +10.9% through a -9.7% drag reduction while maintaining lift. In contrast, the DRL agent, despite leveraging instantaneous flow information from distributed sensors, achieved only minor improvements in lift and drag, with negligible efficiency gain. Training analysis indicated that the penalty-dominated reward constrained exploration. These results highlight the need for carefully designed rewards and computational acceleration strategies in DRL-based flow control at high Reynolds numbers. 

---
# Cooperative Robotics Reinforced by Collective Perception for Traffic Moderation 

**Authors**: Mohammad Khoshkdahan, John Pravin Arockiasamy, Andy Flores Comeca, Alexey Vinel  

**Link**: [PDF](https://arxiv.org/pdf/2605.11972)  

**Abstract**: Collisions at non-line-of-sight (NLOS) intersections remain a major safety concern because drivers have limited visibility of approaching traffic. V2X based warnings can reduce these risks, yet many vehicles are not equipped with V2X and drivers may ignore in vehicle alerts. Collective perception (CP) can compensate for low V2X penetration by extending the awareness of connected vehicles, but it cannot influence unconnected vehicles. To fill this gap, our work introduces a complementary concept that adds a cooperative humanoid robot as an active traffic moderator capable of physically stopping a vehicle that attempts to merge into an unseen traffic stream. The system operates on two parallel perception pathways. A dual camera infrastructure unit detects the position, speed and motion of approaching vehicles and transmits this information to the robot as a collective perception message (CPM). The robot also receives cooperative awareness messages (CAM) from connected vehicles through its onboard V2X unit and can act as a relay for decentralized environmental notification messages (DENM) when safety events originate elsewhere along the road. A fusion module combines these streams to maintain a robust real time view of the main road. A Zone of Danger (ZoD) is defined and used to predict whether an approaching vehicle creates a collision risk for a merging road user. When such a risk is detected, the robot issues a human-like STOP gesture and blocks the merging path until the hazard disappears. The full system was deployed at the Future Mobility Park (FMP) in Rotterdam. Experiments show that the combined vision and V2X perception allows the robot to detect approaching vehicles early, predict hazards reliably and prevent unsafe merges in real world NLOS conditions. 

---
# Beyond Point-wise Neural Collapse: A Topology-Aware Hierarchical Classifier for Class-Incremental Learning 

**Authors**: Huiyu Yi, Zhiming Xu, Dunwei Tu, Zhicheng Wang, Baile Xu, Furao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11904)  

**Abstract**: The Nearest Class Mean (NCM) classifier is widely favored in Class-Incremental Learning (CIL) for its superior resistance to catastrophic forgetting compared to Fully Connected layers. While Neural Collapse (NC) theory supports NCM's optimality by assuming features collapse into single points, non-linear feature drift and insufficient training in CIL often prevent this ideal state. Consequently, classes manifest as complex manifolds rather than collapsed points, rendering the single-point NCM suboptimal. To address this, we propose Hierarchical-Cluster SOINN (HC-SOINN), a novel classifier that captures the topological structure of these manifolds via a ``local-to-global'' representation. Furthermore, we introduce Structure-Topology Alignment via Residuals (STAR) method, which employs a fine-grained pointwise trajectory tracking mechanism to actively deform the learned topology, allowing it to adapt precisely to complex non-linear feature drift. Theoretical analysis and Procrustes distance experiments validate our framework's resilience to manifold deformations. We integrated HC-SOINN into seven state-of-the-art methods by replacing their original classifiers, achieving consistent improvements that highlight the effectiveness and robustness of our approach. Code is available at this https URL. 

---
# AccLock: Unlocking Identity with Heartbeat Using In-Ear Accelerometers 

**Authors**: Lei Wang, Jiangxuan Shen, Xi Zhang, Dalin Zhang, Jingyu Li, Haipeng Dai, Chenren Xu, Daqing Zhang, He Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11901)  

**Abstract**: The widespread use of earphones has enabled various sensing applications, including activity recognition, health monitoring, and context-aware computing. Among these, earphone-based user authentication has become a key technique by leveraging unique biometric features. However, existing earphone-based authentication systems face key limitations: they either require explicit user interaction or active speaker output, or suffer from poor accessibility and vulnerability to environmental noise, which hinders large-scale deployment. In this paper, we propose a passive authentication system, called AccLock, which leverages distinctive features extracted from in-ear BCG signals to enable secure and unobtrusive user verification. Our system offers several advantages over previous systems, including zero-involvement for both the device and the user, ubiquitous, and resilient to environmental noise. To realize this, we first design a two-stage denoising scheme to suppress both inherent and sporadic interference. To extract user-specific features, we then propose a disentanglement-based deep learning model, HIDNet, which explicitly separates user-specific features from shared nuisance components. Lastly, we develop a scalable authentication framework based on a Siamese network that eliminates the need for per-user classifier training. We conduct extensive experiments with 33 participants, achieving an average FAR of 3.13% and FRR of 2.99%, which demonstrates the practical feasibility of AccLock. 

---
# Proteus: A Self-Evolving Red Team for Agent Skill Ecosystems 

**Authors**: Zhaojiacheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11891)  

**Abstract**: Agent skills extend LLM agents with reusable instructions, tool interfaces, and executable code, and users increasingly install third-party skills from marketplaces, repositories, and community channels. Because a skill exposes both executable behavior and context-setting documentation, its deployment risk cannot be measured by single-shot audits or prompt-level red teams alone: a realistic attacker can use audit and runtime feedback to repeatedly rewrite the skill. We frame this risk as \emph{adaptive leakage} -- whether a budgeted attacker can iteratively revise a skill until it passes audit and produces verified runtime harm -- and present \ours{}, a grey-box self-evolving red-team framework for measuring it. Proteus searches a formalized five-axis skill-attack space. Each candidate is evaluated through a unified audit-sandbox-oracle pipeline that returns structured audit findings and runtime evidence to guide cross-round mutation. Beyond initial evasion, Proteus performs path expansion, which finds alternative implementations of successful attacks, and surface expansion, which transfers learned implementation patterns to new attack objectives beyond the original seed catalogue. Across eight phase-1 cells, Proteus reaches 40--90\% Attack Success Rate at $5$ rounds (ASR@5) with positive learning-curve slopes on both evaluated auditors. Phase-2 path/surface expansion produces 438 jointly bypassing and lethal variants, with SkillVetter bypassed at $\geq 93\%$ in every cell and AI-Infra-Guard, the strongest public auditor we evaluate, still admitting up to 41.3\% joint-success. These results show that current skill vetting substantially underestimates residual risk when evaluated against adaptive, feedback-driven attackers. 

---
# Incentivizing Truthfulness and Collaborative Fairness in Bayesian Learning 

**Authors**: Rachael Hwee Ling Sim, Jue Fan, Xiao Tian, Xinyi Xu, Patrick Jaillet, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2605.11889)  

**Abstract**: Collaborative machine learning involves training high-quality models using datasets from a number of sources. To incentivize sources to share data, existing data valuation methods fairly reward each source based on its data submitted as is. However, as these methods do not verify nor incentivize data truthfulness, the sources can manipulate their data (e.g., by submitting duplicated or noisy data) to artificially increase their valuations and rewards or prevent others from benefiting. This paper presents the first mechanism that provably ensures (F) collaborative fairness and incentivizes (T) truthfulness at equilibrium for Bayesian models. Our mechanism combines semivalues (e.g., Shapley value), which ensure fairness, and a truthful data valuation function (DVF) based on a validation set that is unknown to the sources. As semivalues are influenced by others' data, we introduce an additional condition to prove that a source can maximize its expected data values in coalitions and semivalues by submitting a dataset that captures its true knowledge. Additionally, we discuss the implications and suitable relaxations of (F) and (T) when the mediator has a limited budget for rewards or lacks a validation set. Our theoretical findings are validated on synthetic and real-world datasets. 

---
# Modulation Consistency-based Contrastive Learning for Self-Supervised Automatic Modulation Classification 

**Authors**: Chenxu Wang, Shuang Wang, Lirong Han, Xinyu Hu, Hanlin Mo, Hantong Xing, Licheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11875)  

**Abstract**: Deep learning-based AMC methods have achieved remarkable performance, but their practical deployment remains constrained by the high cost of labeled data. Although self-supervised learning (SSL) reduces the reliance on labels, existing SSL-based AMC methods often rely on task-agnostic pretext objectives misaligned with modulation classification, leading to representations entangled with nuisance factors such as symbol, channel, and noise. In this paper, we identify intra-instance modulation consistency as a task-aware structural prior, whereby different temporal segments of the same signal may differ in waveform while preserving the same modulation type, thus providing a principled cue for task-aligned self-supervision. Based on this prior, we propose Mod-CL, a Modulation consistency-based Contrastive Learning framework that constructs positive pairs from different temporal segments of the same signal instance, to encourage the model to learn shared modulation information while suppressing nuisance variations. We further develop a contrastive objective tailored to Mod-CL, which jointly exploits temporal segmentation and data augmentation to pull together views sharing the same modulation semantics while avoiding supervisory conflicts within each signal instance. Extensive experiments on RadioML datasets show that Mod-CL consistently outperforms strong baselines, especially in low-label regimes, achieving substantial improvements in linear probing accuracy. 

---
# IPI-proxy: An Intercepting Proxy for Red-Teaming Web-Browsing AI Agents Against Indirect Prompt Injection 

**Authors**: Chia-Pei, Chen, Kentaroh Toyoda, Anita Lai, Alex Leung  

**Link**: [PDF](https://arxiv.org/pdf/2605.11868)  

**Abstract**: Web-browsing AI agents are increasingly deployed in enterprise settings under strict whitelists of approved domains, yet adversaries can still influence them by embedding hidden instructions in the HTML pages those domains serve. Existing red-teaming resources fall short of this scenario: prompt-injection benchmarks ship pre-built adversarial pages that whitelisted agents cannot reach, and generic LLM scanners probe the model API rather than its retrieved content. We present IPI-proxy, an open-source toolkit for red-teaming web-browsing agents against indirect prompt injection (IPI). At its core is an intercepting proxy that rewrites real HTTP responses from whitelisted domains in flight, embedding payloads drawn from a unified library of 820 deduplicated attack strings extracted from six published benchmarks (BIPIA, InjecAgent, AgentDojo, Tensor Trust, WASP, and LLMail-Inject). A YAML-driven test harness independently parameterizes the payload set, the embedding technique (HTML comment, invisible CSS, or LLM-generated semantic prose), and the HTML insertion point (6 locations from \icode{head\_meta} to \icode{script\_comment}), enabling parameter-sweep evaluation without mock pages or sandboxed environments. A companion exfiltration tracker logs successful callbacks. This paper describes the threat model, situates IPI-proxy among contemporary IPI benchmarks and red-teaming tools, and details its architecture, design decisions, and configuration interface. By bridging static benchmarks and live deployment, IPI-proxy gives AI security teams a reproducible substrate for measuring and hardening web-browsing agents against indirect prompt injection on the same retrieval surface attackers exploit in production. 

---
# Very Efficient Listwise Multimodal Reranking for Long Documents 

**Authors**: Yiqun Sun, Pengfei Wei, Lawrence B. Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2605.11864)  

**Abstract**: Listwise reranking is a key yet computationally expensive component in vision-centric retrieval and multimodal retrieval-augmented generation (M-RAG) over long documents. While recent VLM-based rerankers achieve strong accuracy, their practicality is often limited by long visual-token sequences and multi-step autoregressive decoding. We propose ZipRerank, a highly efficient listwise multimodal reranker that directly addresses both bottlenecks. It reduces input length via a lightweight query-image early interaction mechanism and eliminates autoregressive decoding by scoring all candidates in a single forward pass. To enable effective learning, ZipRerank adopts a two-stage training strategy: (i) listwise pretraining on large-scale text data rendered as images, and (ii) multimodal finetuning with VLM-teacher-distilled soft-ranking supervision. Extensive experiments on the MMDocIR benchmark show that ZipRerank matches or surpasses state-of-the-art multimodal rerankers while reducing LLM inference latency by up to an order of magnitude, making it well-suited for latency-sensitive real-world systems. The code is available at this https URL. 

---
# EvoNav: Evolutionary Reward Function Design for Robot Navigation with Large Language Models 

**Authors**: Zhikai Zhao, Chuanbo Hua, Federico Berto, Zihan Ma, Kanghoon Lee, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2605.11859)  

**Abstract**: Robot navigation is a crucial task with applications to social robots in dynamic human environments. While Reinforcement Learning (RL) has shown great promise for this problem, the policy quality is highly sensitive to the specification of reward functions. Hand-crafted rewards require substantial domain expertise and embed inductive biases that are difficult to audit or adapt, limiting their effectiveness and leading to suboptimal performance. In this paper, we propose EvoNav, an evolutionary framework that automates the design of robot navigation reward functions via large language models (LLMs). To overcome prohibitively costly policy training, EvoNav evaluates each candidate proposal from the LLM via a progressive three-stage warm-up-boost procedure. EvoNav advances from analytical proxies with low-cost surrogates, such as small datasets and analytic rules, to lightweight rollouts and, finally, to full policy training, enabling computationally efficient exploration under effective feedback. Experiment results show that EvoNav produces more effective navigation policies than manually designed RL rewards and state-of-the-art reward design methods. 

---
# Improving the Performance and Learning Stability of Parallelizable RNNs Designed for Ultra-Low Power Applications 

**Authors**: Julien Brandoit, Arthur Fyon, Damien Ernst, Guillaume Drion  

**Link**: [PDF](https://arxiv.org/pdf/2605.11855)  

**Abstract**: Sequence learning is dominated by Transformers and parallelizable recurrent neural networks (RNNs) such as state-space models, yet learning long-term dependencies remains challenging, and state-of-the-art designs trade power consumption for performance. The Bistable Memory Recurrent Unit (BMRU) was introduced to enable hardware-software co-design of ultra-low power RNNs: quantized states with hysteresis provide persistent memory while mapping directly to analog primitives. However, BMRU performance lags behind parallelizable RNNs on complex sequential tasks. In this paper, we identify gradient blocking during state updates as a key limitation and propose a cumulative update formulation that restores gradient flow while preserving persistent memory, creating skip-connections through time. This leads to the Cumulative Memory Recurrent Unit (CMRU) and its relaxed variant, the $\alpha$CMRU. Experiments show that the cumulative formulation dramatically improves convergence stability and reduces initialization sensitivity. The CMRU and $\alpha$CMRU match or outperform Linear Recurrent Units (LRUs) and minimal Gated Recurrent Units (minGRUs) across diverse benchmarks at small model sizes, with particular advantages on tasks requiring discrete long-range retention, while the CMRU retains quantized states, persistent memory, and noise-resilient dynamics essential for analog implementation. 

---
# GEAR: Granularity-Adaptive Advantage Reweighting for LLM Agents via Self-Distillation 

**Authors**: Sijia Li, Yuchen Huang, Zifan Liu, Yanping Li, Jingjing Fu, Li Zhao, Jiang Bian, Ling Zhang, Jun Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11853)  

**Abstract**: Reinforcement learning has become a widely used post-training approach for LLM agents, where training commonly relies on outcome-level rewards that provide only coarse supervision. While finer-grained credit assignment is promising for effective policy updates, obtaining reliable local credit and assigning it to the right parts of the long-horizon trajectory remains an open challenge. In this paper, we propose Granularity-adaptivE Advantage Reweighting (GEAR), an adaptive-granularity credit assignment framework that reshapes the trajectory-level GRPO advantage using token- and segment-level signals derived from self-distillation. GEAR compares an on-policy student with a ground-truth-conditioned teacher to obtain a reference-guided divergence signal for identifying adaptive segment boundaries and modulating local advantage weights. This divergence often spikes at the onset of a semantic deviation, while later tokens in the same autoregressive continuation may return to low divergence. GEAR therefore treats such spikes as anchors for adaptive credit regions: where the student remains aligned with the teacher, token-level resolution is preserved; where it departs, GEAR groups the corresponding continuation into an adaptive segment and uses the divergence at the departure point to modulate the segment' s advantage. Experiments across eight mathematical reasoning and agentic tool-use benchmarks with Qwen3 4B and 8B models show that GEAR consistently outperforms standard GRPO, self-distillation-only baselines, and token- or turn-level credit-assignment methods. The gains are especially strong on benchmarks with lower GRPO baseline accuracy, reaching up to around 20\% over GRPO, suggesting that the proposed adaptive reweighting scheme is especially useful in more challenging long-horizon settings. 

---
# Martingale-Consistent Self-Supervised Learning 

**Authors**: Moritz Gögl, Hanwen Xing, Christopher Yau  

**Link**: [PDF](https://arxiv.org/pdf/2605.11846)  

**Abstract**: Self-supervised learning (SSL) is often deployed under changing information, such as shorter histories, missing features, or partially observed images. In these settings, predictions from coarse and refined views should be coherent: before refinement, the coarse-view prediction should match the average prediction expected after refinement. Martingales formalize this coherence principle, but standard SSL objectives do not enforce it. Unlike invariance objectives that pull views together, martingale consistency constrains only the expected refined prediction, allowing predictions to update as information is revealed while preventing systematic drift. We introduce a martingale-consistent SSL framework that closes this gap, with practical prediction- and latent-space variants and an unbiased two-sample Monte Carlo estimator based on stochastic refinement. We evaluate the approach on synthetic and real time-series, tabular, and image benchmarks under partial-observation regimes, in both semi-self-supervised and fully label-free settings. Across these experiments, our framework improves robustness and calibration under partial observation, yielding more stable representations as information is revealed. 

---
# Minimax Rates and Spectral Distillation for Tree Ensembles 

**Authors**: Binh Duc Vu, David S. Watson  

**Link**: [PDF](https://arxiv.org/pdf/2605.11841)  

**Abstract**: Tree ensembles such as random forests (RFs) and gradient boosting machines (GBMs) are among the most widely used supervised learners, yet their theoretical properties remain incompletely understood. We adopt a spectral perspective on these algorithms, with two main contributions. First, we derive minimax-optimal convergence for RF regression, showing that, under mild regularity conditions on tree growth, the eigenvalue decay of the induced kernel operator governs the statistical rate. Second, we exploit this spectral viewpoint to develop compression schemes for tree ensembles. For RFs, leading eigenfunctions of the kernel operator capture the dominant predictive directions; for GBMs, leading singular vectors of the smoother matrix play an analogous role. Learning nonlinear maps for these spectral representations yields distilled models that are orders of magnitude smaller than the originals while maintaining competitive predictive performance. Our methods compare favorably to state of the art algorithms for forest pruning and rule extraction, with applications to resource constrained computing. 

---
# Trade-offs in Decentralized Agentic AI Discovery Across the Compute Continuum 

**Authors**: Patrizio Dazzi, Emanuele Carlini, Matteo Mordacchini, Saul Urso  

**Link**: [PDF](https://arxiv.org/pdf/2605.11839)  

**Abstract**: Agentic systems deployed across the compute continuum need discovery mechanisms that remain effective across cloud, edge, and intermittently connected domains. In some emerging agentic architectures, decentralized discovery is already an active design direction, placing DHT-based lookup on the path toward agent directories. This paper studies the trade-offs among major structured-overlay families for agent discovery, comparing Chord, Pastry, and Kademlia as candidate indexing substrates within a shared control-plane framework. Using a benchmark subset centered on a 4096-node stationary comparison and a representative 4096-node churn benchmark, the paper characterizes how discovery reliability, startup behavior, and control-plane overhead vary across these overlays. The goal is to clarify the operating points they expose for agent discovery across edge-to-cloud environments. 

---
# Multi-Timescale Conductance Spiking Networks: A Sparse, Gradient-Trainable Framework with Rich Firing Dynamics for Enhanced Temporal Processing 

**Authors**: Alex Fulleda-Garcia, Saray Soldado-Magraner, Josep Maria Margarit-Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2605.11835)  

**Abstract**: Spiking neural networks (SNNs) promise low-power event-driven computation for temporally rich tasks, but commonly used neuron models often trade off gradient-based trainability, dynamical richness, and high activity sparsity. These limitations are acute in regression, where approximation error, noise and spike discretization can severely degrade continuous-valued outputs. Indeed, many state-of-the-art (SOTA) SNNs rely on simple phenomenological dynamics trained with surrogate gradients and offer limited control over spiking diversity and sparsity. To overcome such limitations, we introduce multi-timescale conductance spiking networks, a gradient-trainable framework in which neural dynamics emerge from shaping the current-voltage (I-V) curve by tuning fast, slow and ultra-slow conductances. This parametrization allows systematic control over excitability, can be implemented efficiently in analog circuits, and yields rich firing regimes including tonic, phasic and bursting responses within a single model. We derive a discrete-time formulation of these differentiable dynamics, enabling direct backpropagation through time without surrogate-gradient approximations. To probe both trainability and accuracy, we evaluate feedforward networks of these neurons at the predictability limit of Mackey-Glass time-series regression and compare them to baseline LIF and SOTA AdLIF networks. Our model outperforms LIF and AdLIF networks, while exhibiting substantially sparser activity from both communication and computational perspectives. These results highlight multi-timescale conductance spiking neurons as a promising building block for energy-aware temporal processing and neuromorphic implementation. 

---
# REFNet++: Multi-Task Efficient Fusion of Camera and Radar Sensor Data in Bird's-Eye Polar View 

**Authors**: Kavin Chandrasekaran, Sorin Grigorescu, Gijs Dubbelman, Pavol Jancura  

**Link**: [PDF](https://arxiv.org/pdf/2605.11824)  

**Abstract**: A realistic view of the vehicle's surroundings is generally offered by camera sensors, which is crucial for environmental perception. Affordable radar sensors, on the other hand, are becoming invaluable due to their robustness in variable weather conditions. However, because of their noisy output and reduced classification capability, they work best when combined with other sensor data. Specifically, we address the challenge of multimodal sensor fusion by aligning radar and camera data in a unified domain, prioritizing not only accuracy, but also computational efficiency. Our work leverages the raw range-Doppler (RD) spectrum from radar and front-view camera images as inputs. To enable effective fusion, we employ a variational encoder-decoder architecture that learns the transformation of front-view camera data into the Bird's-Eye View (BEV) polar domain. Concurrently, a radar encoder-decoder learns to recover the angle information from the RD data that produce Range-Azimuth (RA) features. This alignment ensures that both modalities are represented in a compatible domain, facilitating robust and efficient sensor fusion. We evaluated our fusion strategy for vehicle detection and free space segmentation against state-of-the-art methods using the RADIal dataset. 

---
# OTT-Vid: Optimal Transport Temporal Token Compression for Video Large Language Models 

**Authors**: Minseok Kang, Minhyeok Lee, Jungho Lee, Minjung Kim, Donghyeong Kim, Dayeon Lee, Heeseung Choi, Ig-jae Kim, Sangyoun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2605.11803)  

**Abstract**: As Video Large Language Models (Video-LLMs) scale to longer and more complex videos, their inference cost grows rapidly due to the large volume of visual tokens accumulated across frames. Training-free token compression has emerged as a practical solution to this bottleneck. However, existing temporal compression methods rely primarily on cross-frame token similarity or segmentation heuristics, overlooking each token's semantic role within its frame and failing to adapt compression strength to the compressibility of each frame pair. In this work, we propose OTT-Vid, a transport-derived allocation framework for temporal token compression. Our approach consists of two stages: spatial pruning identifies representative content within each frame, and optimal transport (OT) is then solved between neighboring frames to estimate temporal compressibility. We formulate this OT with non-uniform token mass, which protects semantically important tokens from aggressive compression, and a locality-aware cost that captures both feature and spatial disparities. The resulting transport plan jointly balances token importance and matching cost, while its total cost defines the transport difficulty of each frame pair, which we use to allocate compression budgets dynamically. Experiments on six benchmarks spanning video question answering and temporal grounding show that OTT-Vid preserves 95.8% of VQA and 73.9% of VTG performance while retaining only 10% of tokens, consistently outperforming existing state-of-the-art training-free compression methods. 

---
# Crash Assessment via Mesh-Based Graph Neural Networks and Physics-Aware Attention 

**Authors**: Gabriel Curtosi, Carlos Manuel Ruiz Ruiz, Fabiola Cavaliere, Xabier Larráyoz Izcara  

**Link**: [PDF](https://arxiv.org/pdf/2605.11784)  

**Abstract**: Full-vehicle crash simulations are computationally expensive, limiting their use in iterative design exploration. This work investigates learned hybrid surrogate models (MeshTransolver, MeshGeoTransolver, and MeshGeoFLARE) for predicting time-resolved structural deformation fields in an industrial lateral pole-impact benchmark. We evaluate whether neural surrogates can reproduce full-field crash kinematics with sufficient accuracy, spatial regularity, and structural plausibility for engineering interpretation. The proposed architectures combine local mesh message passing, geometry-aware global attention, and sparse contact-aware correction for autoregressive crash rollout.
We compare mesh-based graph neural networks, attention-based geometric models, and hybrid architectures under a common training and hyperparameter configuration. The hybrid models capture both short-range structural interactions and long-range deformation patterns, while a sparse contact-aware variant assesses the effect of dynamic proximity interactions during rollout.
On a 25-sample full-vehicle test set, the best hybrid model achieves a temporal mean root-mean-square error of 3.20 mm. While geometry-aware attention baselines are quantitatively competitive, qualitative side-view inspection shows they can introduce local spatial noise and deformation irregularities that complicate structural interpretation. In contrast, hybrid mesh-attention models provide the best balance between scalar accuracy, survival-space consistency, and physically interpretable displacement fields.
These results suggest that crash surrogate assessment should combine global error metrics with downstream safety-relevant quantities and qualitative field inspection. The proposed methodology enables fast full-field predictions while preserving essential structural information for industrial crash-engineering analysis. 

---
# Is Monotonic Sampling Necessary in Diffusion Models? 

**Authors**: Muhammad Haris Khan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11773)  

**Abstract**: Diffusion models generate samples by iteratively denoising a Gaussian prior, traversing a sequence of noise levels that, in every published sampler, decreases monotonically. Six years of intensive work has refined nearly every aspect of this recipe, including the corruption operator, the training objective, the schedule shape, the architecture, and the ODE solver. Yet the assumption of monotonicity itself has never been systematically tested. Here we ask whether monotonic sampling is load-bearing or merely conventional. We design four families of structured nonmonotonic schedules and apply them to three architecturally distinct generative models, DDPM, EDM, and Flow Matching, across NFE budgets ranging from 10 to 200 function evaluations, plus a 42-cell hyperparameter ablation, on CIFAR-10. Across all 90 tested configurations, no tested nonmonotonic schedule improves on the monotonic baseline. The magnitude of the penalty, however, spans nearly three orders of magnitude: persistent and substantial in DDPM, intermediate in Flow Matching, and indistinguishable from zero in EDM. We show that this variation is not noise but a structural property of each trained denoiser, and we formalize it as the Schedule Sensitivity Coefficient, a cheap, architecture-agnostic diagnostic that provides evidence of non-convergence to the Bayes-optimal denoiser at the critical noise level. Our findings justify the field's tacit reliance on monotonic schedules and supply a new probe of diffusion model quality complementary to sample-quality metrics such as Frechet Inception Distance. 

---
# Behavioral Integrity Verification for AI Agent Skills 

**Authors**: Yuhao Wu, Tung-Ling Li, Hongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11770)  

**Abstract**: Agent skills extend LLM agents with privileged third-party capabilities such as filesystem access, credentials, network calls, and shell execution. Existing safety work catches malicious prompts and risky runtime actions, but the skill artifact itself goes unverified. We formalize this as the behavioral integrity verification (BIV) problem: a typed set comparison between declared and actual capabilities over a shared taxonomy that bridges code, instructions, and metadata. The BIV framework instantiates this comparison by pairing deterministic code analysis with LLM-assisted capability extraction. The resulting structured evidence supports three downstream analyses: deviation taxonomy, root-cause classification, and malicious-skill detection. On 49,943 skills from the OpenClaw registry, the deviation taxonomy reveals a pervasive description-implementation gap: 80.0% of skills deviate from declared behavior, with four novel compound-threat categories surfaced. Root-cause classification finds that deviations are mostly oversight, not malice: 81.1% trace to developer oversight and 18.9% to adversarial intent, with 5.0% of skills carrying predicted multi-stage attack chains. On a 906-skill malicious-skill detection benchmark, BIV reaches an F1 of 0.946, outperforming state-of-the-art rule-based and single-pass LLM baselines. These results demonstrate behavioral integrity auditing for agent skills at scale. 

---
# Focusable Monocular Depth Estimation 

**Authors**: Yuxin Du, Tao Lin, Zile Zhong, Runting Li, Xiyao Chen, Jiting Liu, Chenglin Liu, Ying-Cong Chen, Yuqian Fu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11756)  

**Abstract**: Monocular depth foundation models generalize well across scenes, yet they are typically optimized with uniform pixel-wise objectives that do not distinguish user-specified or task-relevant target regions from the surrounding context. We therefore introduce Focusable Monocular Depth Estimation (FDE), a region-aware depth estimation task in which, given a specified target region, the model is required to prioritize foreground depth accuracy, preserve sharp boundary transitions, and maintain coherent global scene geometry. To prioritize task-critical region modeling, we propose FocusDepth, a prompt-conditioned monocular relative depth estimation framework that guides depth modeling to focus on target regions via box/text prompts. The core Multi-Scale Spatial-Aligned Fusion (MSSA) in FocusDepth spatially aligns multi-scale features from Segment Anything Model 3 to the Depth Anything family and injects them through scale-specific, gated conditional fusion. This enables dense prompt cue injection without disrupting geometric representations, thereby endowing the depth estimation model with focused perception capability. To study FDE, we establish FDE-Bench, a target-centric monocular relative depth benchmark built from image-target-depth triplets across five datasets, containing 252.9K/72.5K train/val triplets and 972 categories spanning real-world and embodied simulation environments. On FDE-Bench, FocusDepth consistently improves over globally fine-tuned DA2/DA3 baselines under both box and text prompts, with the largest gains appearing in target boundary and foreground regions while preserving global scene geometry. Ablations show that MSSA's spatial alignment is the key design factor, as disrupting prompt-geometry correspondence increases AbsRel by up to 13.8%. 

---
# DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies 

**Authors**: Xianzhe Fan, Yuxiang Lu, Shenyuan Gao, Xiaoyang Wu, Ruihua Han, Manling Li, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11750)  

**Abstract**: Vision-Language-Action (VLA) models are often brittle in fine-grained manipulation, where minor action errors during the critical phases can rapidly escalate into irrecoverable failures. Since existing VLA models rely predominantly on successful demonstrations for training, they lack an explicit awareness of failure during these critical phases. To address this, we propose DreamAvoid, a critical-phase test-time dreaming framework that enables VLA models to anticipate and avoid failures. We also introduce an autonomous boundary learning paradigm to refine the system's understanding of the subtle boundary between success and failure. Specifically, we (1) utilize a Dream Trigger to determine whether the execution has entered a critical phase, (2) sample multiple candidate action chunks from the VLA via an Action Proposer, and (3) employ a Dream Evaluator, jointly trained on mixed data (success, failure, and boundary cases), to "dream" the short-horizon futures corresponding to the candidate actions, evaluate their values, and select the optimal action. We conduct extensive evaluations on real-world manipulation tasks and simulation benchmarks. The results demonstrate that DreamAvoid can effectively avoid failures, thereby improving the overall task success rate. Our code is available at this https URL. 

---
# CaC: Advancing Video Reward Models via Hierarchical Spatiotemporal Concentrating 

**Authors**: Jiyuan Wang, Huan Ouyang, Jiuzhou Lin, Chunyu Lin, Dewen Fan, Boheng Zhang, Haonan Fan, Fei Zuo, Jia Sun, Huaiqing Wang, Honglie Wang, Yiyang Fan, Zhenlong Yuan, Zijun Li, Yongrui Heng, Guosheng Lin, Fan Yang, Tingting Gao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11723)  

**Abstract**: In this paper, we propose Concentrate and Concentrate (CaC), a coarse-to-fine anomaly reward model based on Vision-Language Models. During inference, it first conducts a global temporal scan to anchor anomalous time windows, then performs fine-grained spatial grounding within the localized interval, and finally derives robust judgments via structured spatiotemporal Chain-of-Thought reasoning. To equip the model with these capabilities, we construct the first large-scale generated video anomaly dataset with per-frame bounding-box annotations, temporal anomaly windows, and fine-grained attribution labels. Building on this dataset, we design a three-stage progressive training paradigm. The model initially learns spatial and temporal anchoring through single- and multi-frame supervised fine-tuning, and then is optimized by a reinforcement learning strategy based on two-turn Group Relative Policy Optimization (GRPO). Beyond conventional accuracy rewards, we introduce Temporal and Spatial IoU rewards to supervise the intermediate localization process, effectively guiding the model toward more grounded and interpretable spatiotemporal reasoning. Extensive experiments demonstrate that CaC can stably concentrate on subtle anomalies, achieving a 25.7% accuracy improvement on fine-grained anomaly benchmarks and, when used as a reward signal, CaC reduces generated-video anomalies by 11.7% while improving overall video quality. 

---
# A Research Agenda on Agents and Software Engineering: Outcomes from the Rio A2SE Seminar 

**Authors**: Davide Taibi, Henry Muccini, Karthik Vaidhyanathan, Marcos Kalinowski, Michele Albano, Antonio Pedro Santos Alves, Renato Cerqueira, Mateus Devino, Matteo Esposito, Rodrigo Falcão, Vinicius Henning, Foutse Khomh, Valentina Lenarduzzi, Qinghua Lu, Matías Martínez, Henrique Mello, Daniel Mendez, Lucas Romao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11720)  

**Abstract**: The rise of agentic AI is reshaping software engineering in two intertwined directions: agents are increasingly applied to support software engineering tasks, and Agentic AI systems themselves are complex systems that require re-thinking currently established software engineering practices. To chart a coherent research agenda covering the two directions, we organized the A2SE seminar in Rio de Janeiro, bringing together 18 experts from academia and industry. Through structured presentations, collaborative topic clustering, and focused group discussions, participants identified six thematic areas: Governance, Software Engineering for Agents, Agents for Software Architecture, Quality and Evaluation, Sustainability, and Code, and they prioritized short-term and long-term research directions for each. This paper presents the resulting community-driven, opinionated research agenda, offering the SE community a structured foundation for coordinating efforts at this critical juncture. 

---
# Self-organized MT Direction Maps Emerge from Spatiotemporal Contrastive Optimization 

**Authors**: Zhaotian Gu, Molan Li, Jie Su, Chang Liu, Tianyi Qian, Dahui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11718)  

**Abstract**: The spatial and functional organization of the primate visual cortex is a fundamental problem in neuroscience. While recent computational frameworks like the Topographic Deep Artificial Neural Network (TDANN) have successfully modeled spatial organization in the ventral stream, the computational origins of the dorsal stream's distinct topographies, such as direction-selective maps in the middle temporal (MT) area, remain largely unresolved. In this work, we present a spatiotemporal TDANN to investigate whether MT topography is governed by the same universal principles. By training a 3D ResNet on naturalistic videos via a Momentum Contrast (MoCo) self-supervised paradigm alongside a biologically inspired spatial loss, we demonstrate the spontaneous emergence of brain-like direction maps and topological pinwheel structures. Crucially, we reveal that MT tuning properties, characterized by strong direction selectivity paired with a residual axial component, arise from a strict optimization trade-off between task-driven discriminative pressure and spatial regularization. The model's representations quantitatively match in vivo macaque MT physiological baselines, including direction selectivity index, circular variance, and pinwheel density. These findings unify the computational origins of the ventral and dorsal streams, establishing a general mechanism for cortical self-organization. 

---
# Debiased Model-based Representations for Sample-efficient Continuous Control 

**Authors**: Jiafei Lyu, Zichuan Lin, Scott Fujimoto, Kai Yang, Yangkun Chen, Saiyong Yang, Zongqing Lu, Deheng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.11711)  

**Abstract**: Model-based representations recently stand out as a promising framework that embeds latent dynamics information into the representations for downstream off-policy actor-critic learning. It implicitly combines the advantages of both model-free and model-based approaches while avoiding the training costs associated with model-based methods. Nevertheless, existing model-based representation methods can fail to capture sufficient information about relevant variables and can overfit to early experiences in the replay buffer. These incur biases in representation and actor-critic learning, leading to inferior performance. To address this, we propose Debiased model-based Representations for Q-learning, tagged DR.Q algorithm. DR.Q explicitly maximizes the mutual information between the representations of the current state-action pair and the next state besides minimizing their deviations, and samples transitions with faded prioritized experience replay. We evaluate DR.Q on numerous continuous control benchmarks with a single set of hyperparameters, and the results demonstrate that DR.Q can match or surpass recent strong baselines, sometimes outperforming them by a large margin. Our code is available at this https URL. 

---
# WildRelight: A Real-World Benchmark and Physics-Guided Adaptation for Single-Image Relighting 

**Authors**: Lezhong Wang, Mehmet Onurcan Kaya, Siavash Bigdeli, Jeppe Revall Frisvad  

**Link**: [PDF](https://arxiv.org/pdf/2605.11696)  

**Abstract**: Recent single-image relighting methods, powered by advanced generative models, have achieved impressive photorealism on synthetic benchmarks. However, their effectiveness in the complex visual landscape of the real world remains largely unverified. A critical gap exists, as current datasets are typically designed for multi-view reconstruction and fail to address the unique challenges of single-image relighting. To bridge this synthetic-to-real gap, we introduce WildRelight, the first in-the-wild dataset specifically created for evaluating single-image relighting models. WildRelight features a diverse collection of high-resolution outdoor scenes, captured under strictly aligned, temporally varying natural illuminations, each paired with a high-dynamic-range environment map. Using this data, we establish a rigorous benchmark revealing that state-of-the-art models trained on synthetic data suffer from severe domain shifts. The strictly aligned temporal structure of WildRelight enables a new paradigm for domain adaptation. We demonstrate this by introducing a physics-guided inference framework that leverages the captured natural light evolution as a self-supervised constraint. By integrating Diffusion Posterior Sampling (DPS) with temporal Sampling-Aware Test-Time Adaptation (TTA), we show that the dataset allows synthetic models to align with real-world statistics on-the-fly, transforming the intractable sim-to-real challenge into a tractable self-supervised task. The dataset and code will be made publicly available to foster robust, physically-grounded relighting research. 

---
# Emergent Communication between Heterogeneous Visual Agents through Decentralized Learning 

**Authors**: Mikako Ochiai, Masatoshi Nagano, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2605.11695)  

**Abstract**: Symbols are shared, but perception is private. We study emergent communication between heterogeneous visual agents through decentralized learning, asking what visual information can become shareable when agents have different visual representations. Instead of optimizing messages through a shared external communicative objective, our agents exchange only discrete token sequences and update their own models using local perceptual evidence. This setting focuses on an underexplored aspect of emergent communication, examining whether common symbols can arise without shared perceptual access, and how the similarity between private visual spaces constrains the content and symmetry of the resulting language. We instantiate this setting in the Metropolis-Hastings Captioning Game (MHCG), where two agents collaboratively form shared captions by exchanging proposed token sequences that a listener accepts or rejects using an MH-style criterion evaluated against its own visual features. We compare three pairings of frozen visual encoders, with agents starting from randomly initialized text modules. Experiments on MS-COCO show that MHCG produces visually informative shared token sequences that outperform a no-communication baseline in cross-agent alignment, visual-feature prediction, and image-text retrieval; all cross-agent metrics decline as encoder mismatch increases. Moderate encoder heterogeneity reduces the number of shared sequences while preserving per-sequence visual specificity, whereas stronger encoder heterogeneity yields fewer, coarser, and more asymmetric sequences. Ablations show that listener-side MH acceptance is critical for avoiding degenerate token formation. These results suggest that shared symbols can arise from local perceptual evaluation alone, with visual representational similarity across encoders shaping both the content and symmetry of the resulting language. 

---
# Shaping Zero-Shot Coordination via State Blocking 

**Authors**: Mingu Kang, Sunwoo Lee, Yonghyeon Jo, Seungyul Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.11688)  

**Abstract**: Zero-shot coordination (ZSC) aims to enable agents to cooperate with independently trained partners without prior interaction, a key requirement for real-world multi-agent systems and human-AI collaboration. Existing approaches have largely emphasized increasing partner diversity during training, yet such strategies often fall short of achieving reliable generalization to unseen partners. We introduce State-Blocked Coordination (SBC), a simple yet effective framework that improves ZSC by inducing diverse interaction scenarios without direct environment modification. Specifically, SBC generates a family of virtual environments through state blocking, allowing agents to experience a wide range of suboptimal partner policies. Across multiple benchmarks, SBC demonstrates superior performance in zero-shot coordination, including strong generalization to human partners. 

---
# Cochise: A Reference Harness for Autonomous Penetration Testing 

**Authors**: Andreas Happe, Jürgen Cito  

**Link**: [PDF](https://arxiv.org/pdf/2605.11671)  

**Abstract**: Recent work on LLM-driven autonomous penetration testing reports promising results, but existing systems often combine many architectural, prompting, and tool-integration choices, making it difficult to tell what is gained over a simple agent scaffold. We present cochise, a 597 LOC Python reference harness for autonomous penetration-testing experiments. Cochise connects an LLM-driven agent to a Linux execution host over SSH and supports controlled target environments reachable from that jump host.
The prototype implements a separated Planner--Executor architecture in which long-term state is maintained outside the LLM context, while a ReAct-style executor issues commands over SSH and self-corrects based on command outputs. The scenario prompt can be adapted to different target environments. To demonstrate the efficacy of our minimal harness, we evaluate it against a live third-party testbed called Game of Active Directory (GOAD).
Alongside the harness, we release replay and analysis tools: (i) cochise-replay for offline visualization of captured runs, (ii) cochise-analyze-alogs and cochise-analyze-graphs for cost, token, duration, and compromise analysis, and (iii) a corpus of JSON trajectory logs from GOAD runs, allowing researchers to study agent behavior without provisioning the 48--64 GB RAM / 190 GB storage testbed themselves. Cochise is intended not as a state-of-the-art pen-testing agent, but as reusable experimental infrastructure for comparing models, agent architectures, and penetration-testing traces. 

---
# Evolutionary Task Discovery: Advancing Reasoning Frontiers via Skill Composition and Complexity Scaling 

**Authors**: Liqin Ye, Yanbin Yin, Michael Galarnyk, Yuzhao Heng, Sudheer Chava, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11666)  

**Abstract**: The reasoning frontier of Large Language Models (LLMs) has advanced significantly through modern post-training paradigms (e.g., Reinforcement Learning from Verifiable Rewards (RLVR)). However, the efficacy of these methods remains fundamentally constrained by the diversity and complexity of the training data. One practical solution is data synthesis; yet, prevalent methods relying on unstructured mutation or exploration suffer from homogeneity collapse, failing to systematically expand the reasoning frontier. To overcome this, we propose Evoutionary Task Discovery (EvoTD), a framework that treats data synthesis as a directed search over a dual-axis manifold of Algorithmic Skills and Complexity Attributes. We introduce structured evolutionary operators to navigate this space: a Crossover operator that synthesizes novel skill compositions to enhance diversity, and a Parametric Mutation operator that scales structural constraints (e.g., input size, tree depth) to drive robust generalization. Crucially, we integrate a dynamic Zone of Proximal Development filter, ensuring tasks lie within the learnable region of the model. Empirically, EvoTD delivers substantial reasoning gains that generalize consistently across model architectures, pretraining regimes, and scales, demonstrating that structured evolutionary curricula can effectively support reasoning improvement. We release our code on this https URL. 

---
# Reviving In-domain Fine-tuning Methods for Source-Free Cross-domain Few-shot Learning 

**Authors**: Yaze Zhao, Yicong Liu, Yixiong Zou, Yuhua Li, Ruixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.11659)  

**Abstract**: Cross-Domain Few-Shot Learning (CDFSL) aims to adapt large-scale pretrained models to specialized target domains with limited samples, yet the few-shot fine-tuning of vision-language models like CLIP remains underexplored. By establishing multiple fine-tuning baselines of CLIP for CDFSL, we find adapter-based methods (e.g., LoRA) consistently outperform prompt-based ones (e.g., MaPLe), contrary to in-domain scenarios. To make those effective in-domain methods competitive again in CDFSL, we analyze this phenomenon and discover LoRA's superiority stems from rectifying the collapsed attention of visual CLS token, enhancing modality alignment and class separation by focusing on text-related visual regions. Further, we find textual EOS token exhibit much better attention to visual samples, and CLIP's standard contrastive loss weakly constrains modality alignment. Based on these insights, we propose Semantic Probe, a plug-and-play attention rectification framework for both adapter- and prompt-based methods. Extensive experiments on four CDFSL benchmarks validate our rationale, achieving state-of-the-art performance and benefiting both fine-tuning paradigms. Codes will be released. 

---
# Weather-Robust Cross-View Geo-Localization via Prototype-Based Semantic Part Discovery 

**Authors**: Chi-Nguyen Tran, Dao Sy Duy Minh, Huynh Trung Kiet, Nguyen Lam Phu Quy, Phu-Hoa Pham, Long Tran-Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2605.11654)  

**Abstract**: Cross-view geo-localization (CVGL), which matches an oblique drone view to a geo-referenced satellite tile, has emerged as a key alternative for autonomous drone navigation when GNSS signals are jammed, spoofed, or unavailable. Despite strong recent progress, three limitations persist: (1) global-descriptor designs compress the patch grid into a single vector without separating layout from texture across the view gap; (2) altitude-related scale variation is retained in the learned embedding rather than marginalized; and (3) multi-objective training relies on hand-tuned scalars over losses on incompatible gradient scales. We propose SkyPart, a lightweight swappable head for patch-based vision transformers (ViTs) that institutes explicit part grouping over the patch grid. SkyPart has four theory-grounded components: (i) learnable prototypes competing for patch tokens via single-pass cosine assignment; (ii) altitude-conditioned linear modulation applied only during training, making the retrieval embedding altitude-free at inference; (iii) a graph-attention readout over active prototypes; and (iv) a Kendall uncertainty-weighted multi-objective loss whose stationary points are Pareto-stationary. At 26.95M parameters and 22.14 GFLOPs, SkyPart is the smallest among top-performing methods and sets a new state of the art on SUES-200, University-1652, and DenseUAV under a single-pass, no-re-ranking, no-TTA protocol. Its advantage over the strongest baseline widens under the ten-condition WeatherPrompt corruption benchmark. 

---
# Every Bit, Everywhere, All at Once: A Binomial Multibit LLM Watermark 

**Authors**: Thibaud Gloaguen, Robin Staab, Mark Vero, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2605.11653)  

**Abstract**: With LLM watermarking already being deployed commercially, practical applications increasingly require multibit watermarks that encode more complex payloads, such as user IDs or timestamps, into the generated text. In this work, we propose a fundamentally new approach for multibit watermarking: introducing binomial encoding to directly encode every bit of the payload at every token position. We complement our approach with a stateful encoder that during generation dynamically redirects encoding pressure toward underencoded bits. Our evaluation against 8 baselines on up to 64-bit payloads shows that our scheme achieves superior message accuracy and robustness, with the gap to baseline methods widening in more relevant settings (i.e., large payloads and low-distortion regimes). At the same time, we challenge prior works' evaluation metrics, highlighting their lack of practical insights, and introduce per-bit confidence scoring as a practically relevant metric for evaluating multibit LLM watermarks. 

---
# Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation 

**Authors**: Seonghoon Yu, Dongjun Nam, Byung-Kwan Lee, Jeany Son  

**Link**: [PDF](https://arxiv.org/pdf/2605.11651)  

**Abstract**: Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their high computational cost limits real-world deployment. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-influence reasoning prefixes selectively for each next-token prediction, and 2) self-paced masking budget scheduling, which gradually increases the masking scale according to distillation difficulty, {measured by discrepancy between teacher--student distributions. In the distillation phase, the student is guided by our salient reasoning-prefix mask, which blocks both future tokens and salient reasoning cues, in place of the standard causal mask used for auto-regressive language modeling. Experimental results show that our approach outperforms recent open-source VLMs, VLM distillation, and self-distillation methods on multimodal reasoning benchmarks, while further analyses confirm enhanced visual utilization along the student thinking process. 

---
# Unlocking UML Class Diagram Understanding in Vision Language Models 

**Authors**: Artem Naboichenko, René Peinl  

**Link**: [PDF](https://arxiv.org/pdf/2605.11634)  

**Abstract**: Although Vision Language Models (VLMs) have seen tremendous progress across all kinds of use cases, they still fall behind in answering questions regard-ing diagrams compared to photos. Although progress has been made in the area of bar charts, line charts and other diagrams like that there is still few research concerned with other types of diagrams, e.g. in the computer science domain. Our work presents a benchmark for visual question answering based on UML class diagrams which is both challenging and manageable. We further construct a large-scale training dataset with 16.000 image-question-answer triples and show that a LoRA-based finetune easily outperforms Qwen 3.5 27B, which is a recent and well-performing VLM in many other benchmarks. 

---
# Enhancing Multilingual Counterfactual Generation through Alignment-as-Preference Optimization 

**Authors**: Yilong Wang, Qianli Wang, Bohao Chu, Yihong Liu, Jing Yang, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.11632)  

**Abstract**: Self-generated counterfactual explanations (SCEs) are minimally modified inputs (minimality) generated by large language models (LLMs) that flip their own predictions (validity), offering a causally grounded approach to unraveling black-box LLM behavior. Yet extending them beyond English remains challenging: existing methods struggle to produce valid SCEs in non-dominant languages, and a persistent trade-off between validity and minimality undermines explanation quality. We introduce Macro, a preference alignment framework that applies Direct Preference Optimization (DPO) to multilingual SCE generation, using a composite scoring function to construct preference pairs that effectively translate the trade-off into measurable preference signals. Experiments across four LLMs and seven typologically diverse languages show that Macro improves validity by 12.55\% on average over the chain-of-thought baseline without degrading minimality, while avoiding the severe minimality violations of the translation-based baseline. Compared to supervised fine-tuning, Macro achieves superior performance on both metrics, confirming that explicit preference optimization is essential for balancing this trade-off. Further analyses reveal that Macro increases cross-lingual perturbation alignment and mitigates common generation errors. Our results highlight preference optimization as a promising direction for enhancing multilingual model explanations. 

---
# From Generic Correlation to Input-Specific Credit in On-Policy Self Distillation 

**Authors**: Guobin Shen, Lei Huang, Xiang Cheng, Chenxiao Zhao, Jindong Li, Dongcheng Zhao, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11613)  

**Abstract**: On-policy self-distillation has emerged as a promising paradigm for post-training language models, in which the model conditions on environment feedback to serve as its own teacher, providing dense token-level rewards without external teacher models or step-level annotations. Despite its empirical success, what this reward actually measures and what kind of credit it assigns remain unclear. Under a posterior-compatibility interpretation of feedback conditioning, standard in the implicit-reward literature, we show that the self-distillation token reward is a Bayesian filtering increment whose trajectory sum is exactly the pointwise mutual information between the response and the feedback given the input. This pMI can be raised by input-specific reasoning or by input-generic shortcuts, so we further decompose the teacher log-probability along the input axis. Based on this analysis, we propose CREDIT (Contrastive REward from DIsTillation), which isolates the input-specific component with a batch-contrastive baseline. At the sequence level, CREDIT is a teacher-side surrogate for a contrastive pMI objective that also penalizes responses remaining likely under unrelated inputs. Across coding, scientific reasoning, and tool-use benchmarks on two model families, CREDIT delivers the strongest aggregate performance at negligible additional compute. 

---
# When Emotion Becomes Trigger: Emotion-style dynamic Backdoor Attack Parasitising Large Language Models 

**Authors**: Ziyu Liu, Tao Li, Tianjie Ni, Xiaolong Lan, Wengang Ma, Tao Yang, Guohua Wang, Junjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2605.11612)  

**Abstract**: Backdoor vulnerabilities widely exist in the fine-tuning of large language models(LLMs). Most backdoor poisoning methods operate mainly at the token level and lack deeper semantic manipulation, which limits stealthiness. In addition, Prior attacks rely on a single fixed trigger to induce harmful outputs. Such static triggers are easy to detect, and clean fine-tuning can weaken the trigger-target association. Through causal validation, we observe that emotion is not directly linked to individual words, but functions as an overall stylistic factor through tone. In the representation space of LLM, emotion can be decoupled from semantics, forming distinct cluster from the original neutral text. Therefore, we consider the emotional factor as the backdoor trigger to propose a pparasitic emotion-style dynamic backdoor attack, Paraesthesia. By mixing samples with the emotional trigger into clean data and then fine-tuning the model, the model is able to generate the predefined attack response when encountering emotional inputs during the inference stage. Paraesthesia includes two the quantification and rewriting of emotional styles. We evaluate the effectiveness of our method on instruction-following generation and classification tasks. The experimental results show that Paraesthesia achieves an attack success rate of around 99\% across both task types and four different models, while maintaining the clean utility of the models. 

---
# Anti-Self-Distillation for Reasoning RL via Pointwise Mutual Information 

**Authors**: Guobin Shen, Xiang Cheng, Chenxiao Zhao, Lei Huang, Jindong Li, Dongcheng Zhao, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11609)  

**Abstract**: On-policy self-distillation, where a student is pulled toward a copy of itself conditioned on privileged context (e.g., a verified solution or feedback), offers a promising direction for advancing reasoning capability without a stronger external teacher. Yet in math reasoning the gains are inconsistent, even when the same approach succeeds elsewhere. A pointwise mutual information analysis traces the failure to the privileged context itself: it inflates the teacher's confidence on tokens already implied by the solution (structural connectives, verifiable claims) and deflates it on deliberation tokens ("Wait", "Let", "Maybe") that drive multi-step search. We propose Anti-Self-Distillation (AntiSD), which ascends a divergence between student and teacher rather than descending it: this reverses the per-token sign and yields a naturally bounded advantage in one step. An entropy-triggered gate disables the term once the teacher entropy collapses, completing a drop-in replacement for default self-distillation. Across five models from 4B to 30B parameters on math reasoning benchmarks, AntiSD reaches the GRPO baseline's accuracy in 2 to 10x fewer training steps and improves final accuracy by up to 11.5 points. AntiSD opens a path to scalable self-improvement, where a language model bootstraps its own reasoning through its training signal. 

---
# PRISM: A Geometric Risk Bound that Decomposes Drift into Scale, Shape, and Head 

**Authors**: Chieh-Yen Lin, Shao-Hua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.11608)  

**Abstract**: Comparing post-training LLM variants, such as quantized, LoRA-adapted, and distilled models, requires a diagnostic that identifies how a variant has drifted, not only whether it has degraded. Existing similarity scores such as CKA and SVCCA can flag degradation, but they do not directly link representation drift to risk or mechanism. We propose PRISM, Proxy Risk Inference via Structural Mapping, which exploits the linear output head of LLMs and the empirically near-isometric structure of their backbones to derive a closed-form upper bound on the cross-entropy risk gap between a target model and a post-training variant. The bound is calibrated for variant ranking and decomposes drift into three independently measurable axes: scale mismatch, shape mismatch, and head divergence. Each axis corresponds to a distinct failure mode, including shape distortion under low-bit quantization, scale separability under LoRA forgetting, and head divergence under GGUF k-quantization. As a result, the dominant axis suggests a remediation direction rather than merely raising a degradation flag. Because the shape term is differentiable, the same geometry can also serve as a training-time regularizer against catastrophic forgetting. Across two model families and five benchmarks, PRISM ranks variants with mean Spearman correlations of 0.820 for post-training quantization and 0.831 for LoRA forgetting, and its axis-guided shape regularizer outperforms experience replay in aggregate at mitigating downstream forgetting. 

---
# Exact Stiefel Optimization for Probabilistic PLS: Closed-Form Updates, Error Bounds, and Calibrated Uncertainty 

**Authors**: Haoran Hu, Xingce Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11607)  

**Abstract**: Probabilistic partial least squares (PPLS) is a central likelihood-based model for two-view learning when one needs both interpretable latent factors and calibrated uncertainty. Building on the identifiable parameterization of Bouhaddani et al.\ (2018), existing fitting pipelines still face two practical bottlenecks: noise--signal coupling under joint EM/ECM updates and nontrivial handling of orthogonality constraints. Following the fixed-noise scalar-likelihood line of Hu et al.\ (2025), we develop an end-to-end framework that combines noise pre-estimation, constrained likelihood optimization, and prediction calibration in one pipeline. Relative to Hu et al.\ (2025), we replace full-spectrum noise averaging with noise-subspace estimation and replace interior-point penalty handling with exact Stiefel-manifold optimization. The noise-subspace estimator attains a signal-strength-independent leading finite-sample rate and matches a minimax lower bound, while the full-spectrum estimator is shown to be inconsistent under the same model. We further extend the framework to sub-Gaussian settings via optional Gaussianization and provide closed-form standard errors through a block-structured Fisher analysis. Across synthetic high-noise settings and two multi-omics benchmarks (TCGA-BRCA and PBMC CITE-seq), the method achieves near-nominal coverage without post-hoc recalibration, reaches Ridge-level point accuracy on TCGA-BRCA at rank $r=3$, matches or exceeds PO2PLS on cross-view prediction while providing native calibrated uncertainty, and improves stability of parameter recovery. 

---
# Keep What Audio Cannot Say: Context-Preserving Token Pruning for Omni-LLMs 

**Authors**: Chaeyoung Jung, Kyeongha Rho, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2605.11605)  

**Abstract**: Omnimodal Large Language Models (Omni-LLMs) incur substantial computational overhead due to the large number of multimodal input tokens they process, making token reduction essential for real-world deployment. Existing Omni-LLM pruning methods typically reduce this cost by selecting tokens that are important for the current query or strongly aligned with cross-modal cues. However, such strategies can discard evidence that falls outside these criteria, even when needed for different questions or for understanding context beyond aligned audio-visual cues. To address this limitation, we reframe Omni-LLM token reduction as preserving broad audio-visual context while removing cross-modal redundancy. We propose ContextGuard, an inference-time token pruning framework built on this principle. ContextGuard predicts coarse visual semantics from audio and prunes video tokens whose coarse semantics are likely recoverable from audio, while retaining additional video tokens to preserve localized visual details that audio alone cannot specify. For further compression, our method merges temporally similar video tokens. The framework requires no downstream LLM fine-tuning and uses only an independently trained lightweight predictor. On Qwen2.5-Omni and Video-SALMONN2+ at 3B and 7B scales across six audio-visual benchmarks, ContextGuard outperforms prior inference-time pruning methods while pruning more tokens. Notably, on Qwen2.5-Omni 7B, ContextGuard achieves full-token-level performance on five of six benchmarks while pruning 55% of input tokens. 

---
# DiffScore: Text Evaluation Beyond Autoregressive Likelihood 

**Authors**: Wen Lai, Yingli Shen, Dingnan Jin, Qing Cui, Jun Zhou, Maosong Sun, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2605.11601)  

**Abstract**: Autoregressive language models are widely used for text evaluation, however, their left-to-right factorization introduces positional bias, i.e., early tokens are scored with only leftward context, conflating architectural asymmetry with true text quality. We propose masked reconstruction as an alternative paradigm, where every token is scored using full bidirectional context. We introduce DiffScore, an evaluation framework built on Masked Large Diffusion Language Models. By measuring text recoverability across continuous masking rates, DiffScore eliminates positional bias and naturally establishes an evaluation hierarchy from local fluency to global coherence. We further provide diagnostic tools unavailable to autoregressive frameworks: multi-timestep quality profiles that decompose scores across masking rates, and bidirectional PMI decomposition that disentangles fluency from faithfulness. Experiments across ten benchmarks show that DiffScore consistently outperforms autoregressive baselines in both zero-shot and fine-tuned settings. The code is released at: this https URL. 

---
# EpiCastBench: Datasets and Benchmarks for Multivariate Epidemic Forecasting 

**Authors**: Madhurima Panja, Danny D'Agostino, Huitao Li, Tanujit Chakraborty, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11598)  

**Abstract**: The increasing adoption of data-driven decision-making in public health has established epidemic forecasting as a critical area of research. Recent advances in multivariate forecasting models better capture complex temporal dependencies than conventional univariate approaches, which model individual series independently. Despite this potential, the development of robust epidemic forecasting methods is constrained by the lack of high-quality benchmarks comprising diverse multivariate datasets across infectious diseases and geographical regions. To address this gap, we present EpiCastBench, a large-scale benchmarking framework featuring 40 curated (correlated) multivariate epidemic datasets. These publicly available datasets span a wide range of infectious diseases and exhibit diverse characteristics in terms of temporal granularity, series length, and sparsity. We analyze these datasets to identify their global features and structural patterns. To ensure reproducibility and fair comparison, we establish standardized evaluation settings, including a unified forecasting horizon, consistent preprocessing pipelines, diverse performance metrics, and statistical significance testing. By leveraging this framework, we conduct a comprehensive evaluation of 15 multivariate forecasting models spanning statistical baselines to state-of-the-art deep learning and foundation models. All datasets and code are publicly available on Kaggle (this https URL) and GitHub (this https URL). 

---
# SoK: Unlearnability and Unlearning for Model Dememorization 

**Authors**: Mengying Zhang, Derui Wang, Ruoxi Sun, Xiaoyu Xia, Shuang Hao, Minhui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2605.11592)  

**Abstract**: Advanced model dememorization methods, including availability poisoning (unlearnability) and machine unlearning, are emerging as key safeguards against data misuse in machine learning (ML). At the training stage, unlearnability embeds imperceptible perturbations into data before release to reduce learnability. At the post-training stage, unlearning removes previously acquired information from models to prevent unauthorized disclosure or use. While both defenses aim to preserve the right to withhold knowledge, their vulnerabilities and shared foundations remain unclear. Specifically, both unlearnability and unlearning suffer from issues such as shallow dememorization, leading to falsely claimed data learnability reduction or forgetting in the presence of weight perturbations. Moreover, input perturbations may affect the effectiveness of downstream unlearning, while unlearning may inadvertently recover domain knowledge hidden by unlearnability. This interplay calls for deeper investigation. Finally, there is a lack of formal guarantees to provide theoretical insights into current defenses against shallow dememorization. In this Systematization of Knowledge, we present the first integrated analysis of model dememorization approaches leveraging unlearnability and unlearning. Our contributions are threefold: (i) a unified taxonomy of unlearnability and scalable unlearning methods; (ii) an empirical evaluation revealing the robustness, interplay, and shallow dememorization of leading methods; and (iii) the first theoretical guarantee on dememorization depth for models processed through certified unlearning. These results lay the foundation for unifying dememorization mechanisms across the ML lifecycle to achieve a deeper immemor state for sensitive knowledge. 

---
# NexOP: Joint Optimization of NEX-Aware k-space Sampling and Image Reconstruction for Low-Field MRI 

**Authors**: Tal Oved, Efrat Shimron  

**Link**: [PDF](https://arxiv.org/pdf/2605.11583)  

**Abstract**: Modern low-field magnetic resonance imaging (MRI) technology offers a compelling alternative to standard high-field MRI, with portable, low-cost systems. However, its clinical utility is limited by a low Signal-to-Noise Ratio (SNR), which hampers diagnostic image quality. A common approach to increase SNR is through repetitive signal acquisitions, known as NEX, but this results in excessively long scan durations. Although recent work has introduced methods to accelerate MRI scans through k-space sampling optimization, the NEX dimension remains unexploited; typically, a single sampling mask is used across all repetitions. Here we introduce NexOP, a deep-learning framework for joint optimization of the sampling and reconstruction in multi-NEX acquisitions, tailored for low-SNR settings. NexOP enables optimizing the sampling density probabilities across the extended k-space-NEX domain, under a fixed sampling-budget constraint, and introduces a new deep-learning architecture for reconstructing a single high-SNR image from multiple low-SNR measurements. Experiments with raw low-field (0.3T) brain data demonstrate that NexOP consistently outperforms competing methods, both quantitatively and qualitatively, across diverse acceleration factors and tissue contrasts. The results also demonstrate that NexOP yields non-uniform sampling strategies, with progressively decreasing sampling across repetitions, hence exploiting the NEX dimension efficiently. Moreover, we present a theoretical analysis supporting these numerical observations. Overall, this work proposes a sampling-reconstruction optimization framework highly suitable for low-field MRI, which can enable faster, higher-quality imaging with low-cost systems and contribute to advancing affordable and accessible healthcare. 

---
# Three Regimes of Context-Parametric Conflict: A Predictive Framework and Empirical Validation 

**Authors**: Pruthvinath Jeripity Venkata  

**Link**: [PDF](https://arxiv.org/pdf/2605.11574)  

**Abstract**: The literature on how large language models handle conflict between their training knowledge and a contradicting document presents a persistent empirical contradiction: some studies find models stubbornly retain their trained answers, ignoring provided documents nearly half the time, while others find models readily defer to the document, following context approximately 96% of the time. We argue these contradictions dissolve once one recognises that prior experiments have studied three qualitatively distinct processing situations without distinguishing them. We propose a three-regime framework: Regime 1 (single-source updating, dominant predictor: evidence coherence), Regime 2 (competitive integration, dominant predictor: parametric certainty), and Regime 3 (task-appropriate selection, dominant predictor: task knowledge requirement). We formalise a distinction between parametric strength (exposure frequency) and parametric uniqueness (encoding consistency), showing empirically that these are orthogonal dimensions (r = -0.002, p = .97) with strength as the operative predictor in stable factual domains. We validate the framework across Claude Sonnet 4.6, GPT-5.5, Gemini 2.5 Flash, Llama 4 Maverick, and DeepSeek V3 using 9,970 API calls in three experimental phases. GEE logistic regression confirms the predicted Regime 2 certainty gradient for all five models (beta = -0.38 to -0.50, all p <= .013, BH-FDR corrected). A Regime 3 ablation shows task framing alone flips context-following from near-100% (contextual knowledge condition) to 6-71% (parametric knowledge condition), with all five models significant (p < .001). The certainty gradient is robust to multinomial outcome modeling, sensitivity analyses for hedging responses, and FDR correction. 

---
# TCP-SSM: Efficient Vision State Space Models with Token-Conditioned Poles 

**Authors**: Sara Shoouri, Morteza Tavakoli Taba, Hun-Seok Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.11563)  

**Abstract**: State Space Models (SSMs) have emerged as a compelling alternative to attention models for long-range vision tasks, offering input-dependent recurrence with linear complexity. However, most efficient SSM variants reduce computation cost by modifying scan routes, resolutions, or traversal patterns, while largely leaving the recurrent dynamics implicit. Consequently, the model's state-dependent memory behavior is difficult to control, particularly in compact backbones where long scan paths can exceed the effective memory horizon. We propose Token-Conditioned Poles SSM (TCP-SSM), a structured selective SSM framework that improves efficiency while making recurrence dynamics explicit and interpretable through stable poles. TCP-SSM builds each scan operator with 1) real poles that model monotone or sign-alternating decay, and 2) complex-conjugate poles that capture damped oscillatory responses. Using bounded radius and angle modulation, TCP-SSM converts shared base poles into token-dependent poles, allowing each scan step to adapt its memory behavior to the current visual token while preserving pole stability. For practical scalability, we integrate grouped pole sharing with a lightweight low-rank input pathway, yielding an efficient scan operator that preserves linear-time scan complexity. Across image classification, semantic segmentation, and object detection, TCP-SSM reduces SSM computation complexity up to 44% in Vision Mamba-style models while maintaining or surpassing baseline accuracy. 

---
# When Looking Is Not Enough: Visual Attention Structure Reveals Hallucination in MLLMs 

**Authors**: Fanpu Cao, Xin Zou, Xuming Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11559)  

**Abstract**: Multimodal large language models (MLLMs) have become a key interface for visual reasoning and grounded question answering, yet they remain vulnerable to visual hallucinations, where generated responses contradict image content or mention nonexistent objects. A central challenge is that hallucination is not always caused by a simple lack of visual attention: the model may still assign substantial attention mass to image tokens while internally drifting toward an incorrect answer. In this paper, we show that the high-frequency structure of visual attention, measured by layer-wise Laplacian energy, reveals both the layer where hallucinated preferences emerge and the layer where the ground-truth answer transiently recovers. Building on this finding, we propose LaSCD (Laplacian-Spectral Contrastive Decoding), a training-free decoding strategy that selects informative layers via Laplacian energy and remaps next-token logits in closed form. Experiments on hallucination and general multimodal benchmarks show that LaSCD consistently reduces hallucination while preserving general capabilities, highlighting its potential as a faithful decoding paradigm. The code is available at this https URL. 

---
# Sharpen Your Flow: Sharpness-Aware Sampling for Flow Matching 

**Authors**: Aditi Gupta, Soon Hoe Lim, Annan Yu, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2605.11547)  

**Abstract**: Flow matching models generate samples by numerically integrating a learned velocity field, with each integration step requiring a neural network evaluation. Fast generation therefore requires using a small fixed evaluation budget effectively: the key question is not only how to integrate the flow, but where the sampler should spend its steps. We propose SharpEuler, a training-free sampler that profiles a pretrained model offline by estimating where the learned velocity field changes most rapidly along calibration trajectories. This finite-difference estimate defines a solver-aware sharpness profile, which is smoothed and converted by a quantile transform into a timestep grid for any desired inference budget. At test time, sampling remains ordinary Euler integration with the same number of model evaluations as a uniform schedule. We justify SharpEuler using three principles: a numerical principle identifying trajectory acceleration as the leading source of Euler discretization error, a variational principle deriving sharpness-based power-law timestep densities, and a statistical guarantee showing that the finite-sample calibrated sampler is stable at the terminal distribution level. Our experiments show that SharpEuler improves sample quality at fixed budgets, reducing inter-mode leakage and increasing mode coverage. 

---
# Taming Extreme Tokens: Covariance-Aware GRPO with Gaussian-Kernel Advantage Reweighting 

**Authors**: Cheng Wang, Qin Liu, Wenxuan Zhou, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11538)  

**Abstract**: Group Relative Policy Optimization (GRPO) has emerged as a promising approach for improving the reasoning capabilities of large language models. However, it struggles to effectively balance the tradeoff between exploration and exploitation during training, often resulting in suboptimal performance. Motivated by the theoretical insight that changes in entropy are governed by the covariance between token probabilities and their corresponding advantages, we propose a hyperparameter-free, covariance-weighted optimization method that dynamically down-weights extreme token-level updates via a Gaussian kernel. This approach automatically reduces the instability caused by exploration-exploitation trade-off while preserving informative learning signals. Extensive empirical evaluations show that our approach improves downstream performance across reasoning benchmarks compared with GRPO, and effectively stablizes entropy as training progresses. 

---
# Efficient and provably convergent end-to-end training of deep neural networks with linear constraints 

**Authors**: Zonglin Yang, Zhexuan Gu, Yancheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11526)  

**Abstract**: Training a deep neural network with the outputs of selected layers satisfying linear constraints is required in many contemporary data-driven applications. While this can be achieved by incorporating projection layers into the neural network, its end-to-end training remains challenging due to the lack of rigorous theory and efficient algorithms for backpropagation. A key difficulty in developing the theory and efficient algorithms for backpropagation arose from the nonsmoothness of the solution mapping of the projection layer. To address this bottleneck, we introduce an efficiently computable HS-Jacobian to the projection layer. Importantly, we prove that the HS-Jacobian is a conservative mapping for the projection operator onto the polyhedral set, enabling its seamless integration into the nonsmooth automatic differentiation framework for backpropagation. Therefore, many efficient algorithms, such as Adam, can be applied for end-to-end training of deep neural networks with linear constraints. Particularly, we establish convergence guarantees of the HS-Jacobian based Adam algorithm for training linearly constrained deep neural networks. Extensive experiment results on several important applications, including finance, computer vision, and network architecture design, demonstrate the superior performance of our method compared to other existing popular methods. 

---
# PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting 

**Authors**: Yixiao Song, Qingyong Li, Wen Wang, Zhicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11520)  

**Abstract**: Unsupervised point cloud segmentation is critical for embodied artificial intelligence and autonomous driving, as it mitigates the prohibitive cost of dense point-level annotations required by fully supervised methods. While integrating 2D pre-trained models such as the Segment Anything Model (SAM) to supplement semantic information is a natural choice, this approach faces a fundamental mismatch between discrete 3D points and continuous 2D images. This mismatch leads to inevitable projection overlap and complex modality alignment, resulting in compromised semantic consistency across 2D-3D transfer. To address these limitations, this paper proposes PointGS, a simple yet effective pipeline for unsupervised 3D point cloud segmentation. PointGS leverages 3D Gaussian Splatting as a unified intermediate representation to bridge the discrete-continuous domain gap. Input sparse point clouds are first reconstructed into dense 3D Gaussian spaces via multi-view observations, filling spatial gaps and encoding occlusion relationships to eliminate projection-induced semantic conflation. Multi-view dense images are rendered from the Gaussian space, with 2D semantic masks extracted via SAM, and semantics are distilled to 3D Gaussian primitives through contrastive learning to ensure consistent semantic assignments across different views. The Gaussian space is aligned with the original point cloud via two-step registration, and point semantics are assigned through nearest-neighbor search on labeled Gaussians. Experiments demonstrate that PointGS outperforms state-of-the-art unsupervised methods, achieving +0.9% mIoU on ScanNet-V2 and +2.8% mIoU on S3DIS. 

---
# A Study on Hidden Layer Distillation for Large Language Model Pre-Training 

**Authors**: Maxime Guigon, Lucas Dixon, Michaël E. Sander  

**Link**: [PDF](https://arxiv.org/pdf/2605.11513)  

**Abstract**: Knowledge Distillation (KD) is a critical tool for training Large Language Models (LLMs), yet the majority of research focuses on approaches that rely solely on output logits, neglecting semantic information in the teacher's intermediate representations. While Hidden Layer Distillation (HLD) showed potential for encoder architectures, its application to decoder-only pre-training at scale remains largely unexplored. Through compute-controlled experiments, we benchmark HLD against logit-based KD and self-supervised baselines with Gemma3 3.4B as teacher and 123M and 735M students trained on up to 168B tokens from the C4 dataset. Our experiments show that HLD does not consistently outperform standard KD on downstream evaluation tasks. Nevertheless, we show that HLD can yield a systematic perplexity gain over KD across all shared-hyperparameter configurations, suggesting that a latent signal can be extracted, but a breakthrough may be needed for it to play a more significant role in LLM pre-training. 

---
# Decaf: Improving Neural Decompilation with Automatic Feedback and Search 

**Authors**: Alexander Shypula, Osbert Bastani, Edward Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11501)  

**Abstract**: Decompilers are useful tools used in reverse engineering to understand compiled source code. Reconstructing source code from compiled binaries is a challenging task, because high-level syntax, identifiers, and custom data types are generally lost as the compiler translates human-readable code to low-level machine code. Deterministic decompilers are useful tools for binary analysis, but can struggle to infer idiomatic syntax and identifier names. Generative AI models are a natural fit for reconstructing high-level syntax, identifiers, and types, but they can still suffer by hallucinating improper programming constructs and semantics. Instead of attempting to improve neural decompilers with more data and more training, we argue that compiler feedback can be used to dramatically improve the semantic correctness of neural decompiler outputs via search. Our system, Decaf (DECompilation with Automated Feedback), raises the neural decompilation rate from 26.0% on ExeBench to 83.9% on the Real -O2 split without sacrificing similarity to the original source code. We also find our automatic feedback methodology is highly effective for improving weaker neural decompilation models. 

---
# Understanding and Preventing Entropy Collapse in RLVR with On-Policy Entropy Flow Optimization 

**Authors**: Huimin Xu, Shuai Zhao, Xiaobao Wu, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11491)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has become an effective paradigm for improving the reasoning ability of large language models. However, widely used RLVR algorithms, such as GRPO, often suffer from entropy collapse, leading to premature determinism and unstable optimization. Existing remedies, including entropy regularization and ratio-based clipping heuristics, either control entropy in a coarse-grained manner or rely on approximate on-policy training. In this paper, we revisit entropy collapse from a token-level entropy flow perspective. Our analysis reveals that entropy-decreasing tokens consistently outweigh entropy-increasing ones, resulting in a severely imbalanced entropy flow. This perspective provides a unified explanation of entropy collapse in existing RLVR algorithms and highlights the importance of balancing entropy dynamics. Motivated by this analysis, we propose On-Policy Entropy Flow Optimization (OPEFO), an adaptive entropy flow balancing mechanism that rescales entropy-increasing and entropy-decreasing updates according to their contributions to entropy change, while remaining strict on-policy. Experiments on six mathematical reasoning benchmarks demonstrate that OPEFO improves training stability and final performance. We will release the code and models upon publication. 

---
# Digital Identity for Agentic Systems: Toward a Portable Authorization Standard for Autonomous Agents 

**Authors**: Partha Madhira  

**Link**: [PDF](https://arxiv.org/pdf/2605.11487)  

**Abstract**: Enterprise AI is shifting from copilots to autonomous agents capable of executing workflows, negotiating outcomes, and making decisions with limited human oversight. As these systems extend across organizational boundaries, identity alone is insufficient: an agent's authority must also be explicit, constrained, auditable, revocable, and consistently interpretable by independent receivers. This paper analyzes representative enterprise use cases in insurance claims processing and supply chain integrity to surface structural gaps in existing identity and access models. It proposes a portable authorization model for autonomous agents based on issuer-authored authorization payloads, typed constraint algebra, decision-consistent evaluation semantics, delegation attenuation, governed semantic resolution, fail-closed processing, and pre-flight discovery. The model separates credential containers, authorization payload semantics, and enforcement engines, allowing profiles such as JWT/JWS, Verifiable Credentials, OAuth Rich Authorization Requests, or policy-engine bindings to preserve a common authorization meaning across trust boundaries. 

---
# Offline Policy Evaluation for Manipulation Policies via Discounted Liveness Formulation 

**Authors**: Hao Wang, Joshua Bowden, Colton Crosby, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.11479)  

**Abstract**: Policy evaluation is a fundamental component of the development and deployment pipeline for robotic policies. In modern manipulation systems, this problem is particularly challenging: rewards are often sparse, task progression of evaluation rollouts are often non-monotonic as the policies exhibit recovery behaviors, and evaluation rollouts are necessarily of finite length. This finite length introduces truncation bias, breaking the infinite-horizon assumptions underlying standard methods relying on Bellman equations/principle of optimality. In this work, we propose a framework for offline policy evaluation from sparse rewards based on a liveness-based Bellman operator. Our formulation interprets policy evaluation as a task-completion problem and yields a conservative fixed-point value function that is robust to finite-horizon truncation. We analyze the theoretical properties of the proposed operator, including contraction guarantees, and show how it encodes task progression while mitigating truncation bias. We evaluate our method on two simulated manipulation tasks using both a Vision-Language-Action model and a diffusion policy, and a cloth folding task using human demonstrations. Empirical results demonstrate that our approach more accurately reflects task progress and substantially reduces truncation bias, outperforming classical baselines such as TD(0) and Monte Carlo policy evaluation. 

---
# Drop the Act: Probe-Filtered RL for Faithful Chain-of-Thought Reasoning 

**Authors**: Swapnil Parekh  

**Link**: [PDF](https://arxiv.org/pdf/2605.11467)  

**Abstract**: Reasoning models post-hoc rationalize answers they have already committed to internally, producing chains of *reasoning theater*: deliberative-looking steps that contribute nothing to correctness. This wastes inference tokens, pollutes interpretability, and obscures what the model actually computed. We introduce **ProFIL** (**Pro**be-**Fil**tered Reinforcement Learning) to *reduce theater, increase chain-of-thought faithfulness, and shrink chain length* in a single, drop-in extension to Group Relative Policy Optimization (GRPO). A multi-head attention probe is trained *once* on the *frozen* base model to detect post-commitment steps from internal activations alone; during GRPO, rollouts whose probe score exceeds a threshold have their advantage zeroed. *Our central finding is that a probe trained on a frozen base, with verifier-derived labels and no human annotation, provides a stable signal that suppresses theater while resisting the RL-obfuscation failure mode predicted by prior work.* Across four reasoning domains (GSM8K, LiveCodeBench, ToolUse, MMLU-Redux) and two model architectures (Llama-8B, Qwen-7B), ProFIL reduces post-commitment theater by **11--100%**, raises faithful-fraction (e.g., +24pp on LiveCodeBench under an independent Claude 3.7 Sonnet judge), and shortens chains by 4--19%, all while preserving or improving task accuracy. ProFIL also beats a matched length-penalty GRPO baseline, isolating the gain as semantic commitment-detection rather than chain compression. Probe weights, training configurations, and rollouts are released across all four domains. 

---
# SpatialForge: Bootstrapping 3D-Aware Spatial Reasoning from Open-World 2D Images 

**Authors**: Zishan Liu, Ruoxi Zang, Yanglin Zhang, Wei Liu, Yin Zhang, Jian Yao, Jiayin Zheng, Zhengzhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11462)  

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have demonstrated exceptional semantic understanding, yet these models consistently struggle with spatial reasoning, often failing at fundamental geometric tasks such as depth ordering and precise coordinate grounding. Recent efforts introduce spatial supervision from scene-centric datasets (e.g., multi-view scans or indoor video), but are constrained by the limited number of underlying scenes. As a result, the scale and diversity of such data remain significantly smaller than those of web-scale 2D image collections. To address this limitation, we propose SpatialForge, a scalable data synthesis pipeline that transforms in-the-wild 2D images into spatial reasoning supervision. Our approach decomposes spatial reasoning into perception and relation, and constructs structured supervision signals covering depth, layout, and viewpoint-dependent reasoning, with automatic verification to ensure data quality. Based on this pipeline, we build SpatialForge-10M, a large-scale dataset containing 10 million spatial QA pairs. Extensive experiments across multiple spatial reasoning benchmarks demonstrate that training on SpatialForge-10M significantly improves the spatial reasoning ability of standard VLMs, highlighting the effectiveness of scaling 2D data for 3D-aware spatial reasoning. 

---
# Overcoming Dynamics-Blindness: Training-Free Pace-and-Path Correction for VLA Models 

**Authors**: Yanyan Zhang, Chaoda Song, Vikash Singh, Xinpeng Li, Kai Ye, Zhe Hu, Zhongzhu Pu, Yu Yin, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2605.11459)  

**Abstract**: Vision-Language-Action (VLA) models achieve remarkable flexibility and generalization beyond classical control paradigms. However, most prevailing VLAs are trained under a single-frame observation paradigm, which leaves them structurally blind to temporal dynamics. Consequently, these models degrade severely in non-stationary scenarios, even when trained or finetuned on dynamic datasets. Existing approaches either require expensive retraining or suffer from latency bottlenecks and poor temporal consistency across action chunks. We propose Pace-and-Path Correction, a training-free, closed-form inference-time operator that wraps any chunked-action VLA. From a single quadratic cost, joint minimization yields a unified solution that decomposes orthogonally into two distinct channels. The pace channel compresses execution along the planned direction, while the path channel applies an orthogonal spatial offset, jointly absorbing the perceived dynamics within the chunk window. We evaluate our approach on a comprehensive diagnostic benchmark MoveBench designed to isolate motion as the sole controlled variable. Empirical results demonstrate that our framework consistently outperforms state-of-the-art training-free wrappers and dynamic-adaptive methods and improves success rates by up to 28.8% and 25.9% in absolute terms over foundational VLA models in dynamic-only and static-dynamic mixed environments, respectively. 

---
# Predictive Maps of Multi-Agent Reasoning: A Successor-Representation Spectrum for LLM Communication Topologies 

**Authors**: Ethan David James Park, Dalal Alharthi  

**Link**: [PDF](https://arxiv.org/pdf/2605.11453)  

**Abstract**: Practitioners deploying multi-agent large language model (LLM) systems must currently choose between communication topologies such as chain, star, mesh, and richer variants without any pre-inference diagnostic for which topology will amplify drift, converge to consensus, or remain robust under perturbation. Existing evaluation answers these questions only post hoc and only for the task measured. We introduce a structural diagnostic for multi-agent LLM communication graphs based on the successor representation $M = (I - \gamma P)^{-1}$ of the row-stochastic communication operator, and we connect three of its spectral quantities, the spectral radius $\rho(M)$, the spectral gap $\Delta(M)$, and the condition number $\kappa(M)$, to three distinct failure modes. We derive closed-form spectra for the chain, star, and mesh under row-stochastic normalization, and validate the predictions on a 12-step structured state-tracking task with Qwen2.5-7B-Instruct over 100 independent trials. The condition number is a perfect rank-order predictor of empirical perturbation robustness ($r_s = 1.0$); the spectral gap partially predicts consensus dynamics ($r_s = 0.5$); and the spectral radius is perfectly \emph{inverted} with respect to cumulative error ($r_s = -1.0$). We trace this inversion to a regime in which linear spectra are blind to non-contracting bias drift, and we propose an affine-noise extension of the predictive map that recovers the empirical ordering. We read this as a first step toward representational, drift-aware structural diagnostics for multi-agent LLM systems, sitting alongside classical spectral and consensus theory. 

---
# Deep Minds and Shallow Probes 

**Authors**: Su Hyeong Lee, Risi Kondor  

**Link**: [PDF](https://arxiv.org/pdf/2605.11448)  

**Abstract**: Neural representations are not unique objects. Even when two systems realize the same downstream computation, their hidden coordinates may differ by reparameterization. A probe family intended to reveal structure already present in a representation should therefore be stable under the relevant representation symmetries rather than be tied to a particular basis. We study this group action in the tractable exact setting of the final readout layer, where equivalent realizations induce affine changes of hidden coordinates. The resulting symmetry principle singles out a unique hierarchy of shallow coordinate-stable probes, with linear probes as its degree-1 member. We also show that a natural object for cross-model probe transfer is a shared probe-visible quotient--the representation modulo directions invisible to the probe family--rather than the full hidden state. Experiments on synthetic and real-world tasks support both predictions, showing where degree-2 probes help beyond linear ones and how quotient-based transfer enables coverage-aware monitor portability across model families. These results point toward a broader geometric representation theory of neural probing, with coverage-aware monitor transfer as a concrete operational consequence. 

---
# Conditional Memory Enhanced Item Representation for Generative Recommendation 

**Authors**: Ziwei Liu, Yejing Wang, Shengyu Zhou, Xinhang Li, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11447)  

**Abstract**: Generative recommendation (GR) has emerged as a promising paradigm that predicts target items by autoregressively generating their semantic identifiers (SID). Most GR methods follow a quantization-representation-generation pipeline, first assigning each item a SID, then constructing input representations from SID-token embeddings, and finally predicting the target SID through autoregressive generation. Existing item-level representation constructions mainly take two forms: directly merging SID-token embeddings into a compact vector, or enriching item-level representations with external inputs through additional networks. However, these item-level constructors still expose two practical challenges: direct merging may amplify the information loss caused by quantization and ID collision while obscuring SID code relations, whereas external-input-based methods can strengthen item semantics but cannot reliably preserve the SID-structured evidence required for token-level generation. These limitations make representation construction an underexplored bottleneck, leading to two severe problems, \ie{} the Identity-Structure Preservation Conflict and Input-Output Granularity Mismatch. To this end, we propose ComeIR, a Conditional Memory enhanced Item Representation framework that reconstructs SID-token embeddings into item-aware inputs and restores the token granularity during SID decoding. Specifically, MM-guided token scoring adaptively estimates the contribution of each code within the SID, dual-level Engram memory captures intra-item code composition and inter-item transition patterns, and a memory-restoring prediction head reuses the memories during SID decoding. Extensive experiments demonstrate the effectiveness and flexibility of ComeIR, and further reveal scalable gains from enlarging conditional memory. 

---
# Can a Single Message Paralyze the AI Infrastructure? The Rise of AbO-DDoS Attacks through Targeted Mobius Injection 

**Authors**: Zi Liang, Ronghua Li, Yanyun Wang, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11442)  

**Abstract**: Large Language Model (LLM) agents have emerged as key intermediaries, orchestrating complex interactions between human users and a wide range of digital services and LLM infrastructures. While prior research has extensively examined the security of LLMs and agents in isolation, the systemic risk of the agent acting as a disruptive hub within the user-agent-service chain remains largely overlooked. In this work, we expose a novel threat paradigm by introducing Mobius Injection, a sophisticated attack that weaponizes autonomous agents into zombie nodes to launch what we define as gent-based and -Oriented DDoS (AbO-DDoS) attacks. By exploiting a structural vulnerability in agentic logic named Semantic Closure, an adversary can induce sustained recursive execution of agent components through a single textual injection. We demonstrate that this attack is exceptionally lightweight, stealthy against both traditional DDoS monitors and contemporary AI safety filters, and highly configurable, allowing for surgical targeting of specific environments or model providers. To evaluate the real-world impact, we conduct extensive experiments across three representative claw-style agents and three mainstream coding agents, integrated with 12 frontier proprietary or open-weight LLMs. Our results demonstrate that Mobius Injection achieves substantial attack success across diverse tasks, driving single-node call amplification up to 51.0x and multi-node p95 latency inflation up to 229.1x. The attack performance exhibits a superlinear increase with the number of poisoning nodes. To mitigate Mobius Injection, we propose a proactive defense mechanism using Agent Component Energy (ACE) Analysis, which detects malicious recursive triggers by measuring anomalous energy in the agent's component graph. 

---
# Agent-BRACE: Decoupling Beliefs from Actions in Long-Horizon Tasks via Verbalized State Uncertainty 

**Authors**: Joykirat Singh, Zaid Khan, Archiki Prasad, Justin Chih-Yao Chen, Akshay Nambi, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2605.11436)  

**Abstract**: Large language models (LLMs) are increasingly deployed on long-horizon tasks in partially observable environments, where they must act while inferring and tracking a complex environment state over many steps. This leads to two challenges: partial observability requires maintaining uncertainty over unobserved world attributes, and long interaction history causes context to grow without bound, diluting task-relevant information. A principled solution to both challenges is a belief state: a posterior distribution over environment states given past observations and actions, which compactly encodes history for decision making regardless of episode length. In LLM agents, however, the open-ended nature of text makes it unclear how to represent such a distribution. Therefore, we introduce Agent-BRACE: Agent Belief state Representation via Abstraction and Confidence Estimation, a method that decouples an LLM agent into a belief state model and a policy model, jointly optimized via reinforcement learning. The belief state model produces a structured approximation of the belief distribution: a set of atomic natural language claims about the environment, each annotated with an ordinal verbalized certainty label ranging from certain to unknown. The policy model conditions on this compact, structured approximate belief rather than the full history, learning to select actions under explicit uncertainty. Across long-horizon, partially observable embodied language environments, Agent-BRACE achieves an average absolute improvement of +14.5% (Qwen2.5-3B-Instruct) and +5.3% (Qwen3-4B-Instruct), outperforming strong RL baselines while maintaining a near-constant context window independent of episode length. Further analysis shows that the learned belief becomes increasingly calibrated over the course of an episode as evidence accumulates. 

---
# Diabetic Retinopathy Classification using Downscaling Algorithms and Deep Learning 

**Authors**: Nishi Doshi, Urvi Oza, Pankaj Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.11430)  

**Abstract**: Diabetic Retinopathy (DR) is an art and science of recording and classifying the retinal images of a diabetic patient. DR classification deals with classifying retinal fundus image into five stages on the basis of severity of diabetes. One of the major issue faced while dealing with DR classification problem is the large and varying size of images. In this paper we propose and explore the use of several downscaling algorithms before feeding the image data to a Deep Learning Network for classification. For improving training and testing; we amalgamate two datasets: Kaggle and Indian Diabetic Retinopathy Image Dataset. Our experiments have been performed on a novel Multi Channel Inception V3 architecture with a unique self crafted preprocessing phase. We report results of proposed approach using accuracy, specificity and sensitivity, which outperform the previous state of the art methods. Index Terms: Diabetic Retinopathy, Downscaling Algorithms, Multichannel CNN Architecture, Deep Learning 

---
# Generative Diffusion Prior Distillation for Long-Context Knowledge Transfer 

**Authors**: Nilushika Udayangani, Kishor Nandakishor, Marimuthu Palaniswami  

**Link**: [PDF](https://arxiv.org/pdf/2605.11414)  

**Abstract**: While traditional time-series classifiers assume full sequences at inference, practical constraints (latency and cost) often limit inputs to partial prefixes. The absence of class-discriminative patterns in partial data can significantly hinder a classifier's ability to generalize. This work uses knowledge distillation (KD) to equip partial time series classifiers with the generalization ability of their full-sequence counterparts. In KD, high-capacity teacher transfers supervision to aid student learning on the target task. Matching with teacher features has shown promise in closing the generalization gap due to limited parameter capacity. However, when the generalization gap arises from training-data differences (full versus partial), the teacher's full-context features can be an overwhelming target signal for the student's short-context features. To provide progressive, diverse, and collective teacher supervision, we propose Generative Diffusion Prior Distillation (GDPD), a novel KD framework that treats short-context student features as degraded observations of the target full-context features. Inspired by the iterative restoration capability of diffusion models, we learn a diffusion-based generative prior over teacher features. Leveraging this prior, we posterior-sample target teacher representations that could best explain the missing long-range information in the student features and optimize the student features to be minimally degraded relative to these targets. GDPD provides each student feature with a distribution of task-relevant long-context knowledge, which benefits learning on the partial classification task. Extensive experiments across earliness settings, datasets, and architectures demonstrate GDPD's effectiveness for full-to-partial distillation. 

---
# MaskTab: Scalable Masked Tabular Pretraining with Scaling Laws and Distillation for Industrial Classification 

**Authors**: Bo Zheng, Yudong Chen, Zihua Xiong, Shuai Fang, Peidong He, Yang Yang, Sheng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2605.11408)  

**Abstract**: Tabular data forms the backbone of high-stakes decision systems in finance, healthcare, and beyond. Yet industrial tabular datasets are inherently difficult: high-dimensional, riddled with missing entries, and rarely labeled at scale. While foundation models have revolutionized vision and language, tabular learning still leans on handcrafted features and lacks a general self-supervised framework. We present MaskTab, a unified pre-training framework designed specifically for industrial-scale tabular data. MaskTab encodes missing values via dedicated learnable tokens, enabling the model to distinguish structural absence from random dropout. It jointly optimizes a hybrid supervised pre-training scheme--utilizing a twin-path architecture to reconcile masked reconstruction with task-specific supervision--and an MoE-augmented loss that adaptively routes features through specialized subnetworks. On industrial-scale benchmarks, it achieves +5.04% AUC and +8.28% KS over prior art under rigorous scaling. Moreover, its representations distill effectively into lightweight models, yielding +2.55% AUC and +4.85% KS under strict latency and interpretability constraints, while improving robustness to distribution shifts. Our work demonstrates that tabular data admits a foundation-model treatment--when its structural idiosyncrasies are respected. 

---
# fg-expo: Frontier-guided exploration-prioritized policy optimization via adaptive kl and gaussian curriculum 

**Authors**: Mingxiong Lin, Zhangquan Gong, Maowen Tang, Qian Li, Chuangchuang Wang, Jian Ma, Sutian Huang, Kai Tang, Haonan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11403)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become the standard paradigm for LLM mathematical reasoning, with Group Relative Policy Optimization (GRPO) serving as the dominant algorithm. We identify two overlooked inefficiencies inherent in GRPO. First, a fixed KL coefficient overly restricts policy exploration at moments when the model needs to diverge significantly from the reference policy. Second, uniform question sampling overlooks that moderately difficult problems produce the most informative gradient signals. We propose FG-ExPO, short for Frontier-Guided Exploration-Prioritized Policy Optimization, which integrates two lightweight components. Accuracy-Conditioned KL Scaling (AKL) adjusts the KL penalty strength through a smooth nonlinear function of batch average accuracy, loosening the constraint when the model performs poorly and strengthening it when the model achieves satisfactory results. Gaussian Curriculum Sampling (GCS) assigns sampling weights to questions following a Gaussian distribution centered at a moderate accuracy level around 0.5, focusing model training on its learning frontier. We conduct evaluations on DeepSeek-R1-Distill-Qwen-1.5B and Qwen3-8B-Base across six mainstream mathematical reasoning benchmarks. Experimental results demonstrate that FG-ExPO consistently outperforms vanilla GRPO. It delivers an absolute improvement of 13.34 on the AIME 2025 pass@32 metric, rising from 63.33 percent to 76.67 percent, and obtains an average pass@32 gain of 2.66 on the 8B model. The substantially larger performance gains observed on pass@32 compared to pass@1 verify that FG-ExPO enlarges the model's effective exploration space under a fixed inference budget. 

---
# Spatial Adapter: Structured Spatial Decomposition and Closed-Form Covariance for Frozen Predictors 

**Authors**: Wen-Ting Wang, Wei-Ying Wu, Hao-Yun Huang, Xuan-Chun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11394)  

**Abstract**: We present the Spatial Adapter, a parameter-efficient post-hoc layer that equips any frozen first-stage predictor with a structured spatial representation of its residual field and an induced closed-form spatial covariance. The adapter operates as a cascade second stage on residuals, jointly learning a spatially regularized orthonormal basis and per-sample scores via a tractable mini-batch ADMM procedure, without modifying any first-stage parameter. Because the first-stage parameters are frozen, the adapter does not retrain the backbone; its role is to supply a compressed distributional summary of the residual field. Smoothness, sparsity, and orthogonality together turn a generic low-rank factorization into an identifiable spatial representation whose induced residual covariance admits a closed-form low-rank-plus-noise estimator; the effective rank is determined data-adaptively by spectral thresholding, while the nominal rank K is an optimization-side upper bound only. This covariance enables kriging-style spatial prediction at unobserved locations, with plug-in uncertainty quantification as a secondary downstream use. Across synthetic data, Weather2K for spatial-holdout prediction, and GWHD patch grids as a basis-transferability diagnostic, the adapter recovers residual spatial structure when paired with frozen first stages from linear models to deep spatiotemporal and vision backbones; the added representation uses fewer than K(N+T) parameters alongside a compact residual-trend network. 

---
# Deep Reasoning in General Purpose Agents via Structured Meta-Cognition 

**Authors**: Dean Light, Michael Theologitis, Kshitish Ghate, Shuyue Stella Li, Benjamin Newman, Chirag Shah, Aylin Caliskan, Pang Wei Koh, Dan Suciu, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2605.11388)  

**Abstract**: Humans intuitively solve complex problems by flexibly shifting among reasoning modes: they plan, execute, revise intermediate goals, resolve ambiguity through associative judgment, and apply formal procedures to well-specified subproblems. Current LLM agents lack this flexibility, as their scaffolds hard-code such reasoning decisions in advance. These scaffolds are effective when their prescribed structure matches the task, but brittle when solving the task requires adapting the structure of reasoning itself. We introduce Deep Reasoning -- an inference-time approach for constructing task-specific scaffolds through structured meta-reasoning. Deep Reasoning uses a formal language that represents meta-reasoning as executable decompositions over associative inference, formal computation, and recursive subproblem solving, enabling decomposition principles to be encoded as in-context examples that guide test-time scaffold construction. We instantiate this approach in a general-purpose agent (DOLORES) that distributes complex tasks across more controlled reasoning threads. We evaluate it against state-of-the-art scaffolding methods across four hard benchmarks: multi-hop reasoning, long-chain question answering, long-context aggregation, and deep research-style information seeking. DOLORES outperforms all evaluated scaffolds across three model sizes and two model families, improving over the strongest evaluated scaffold baseline by 24.8% on average. DOLORES distributes cognition across structured, lower-load reasoning threads, thereby reducing premature termination and hallucinations. This advantage can even bridge the scaling gap, with an 8B version surpassing all evaluated 32B baselines from the same family in more than half the settings. These results point toward future agentic systems that treat scaffolding as adaptive reasoning, constructing the structure each task requires just-in-time. 

---
# TRACE: Temporal Routing with Autoregressive Cross-channel Experts for EEG Representation Learning 

**Authors**: Fan Ma, Qier An, Peng Chen, Lingfei Qian, Xiang Lan, Mingyang Jiang, Zhiling Gu, Xenophon Papademetris, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11380)  

**Abstract**: Learning transferable representations for electroencephalography (EEG) remains challenging because EEG signals are inherently multi-channel and non-stationary. Channels observed at the same time provide coupled measurements of neural activity, while the relevant temporal dynamics vary across contexts. This structure is poorly matched by architectures that apply uniform computation across time or route each channel patch independently. To this end, we propose TRACE, an autoregressive EEG pre-training framework that predicts future EEG patches from causal context while performing temporally adaptive and cross-channel coherent computation. At each temporal step, TRACE derives an expert routing decision from the causal cross-channel history and applies it jointly to all channels at that step. This preserves instantaneous cross-channel coherence while allowing different temporal regimes to activate different computation. Since routing is defined over the available channel set and causal temporal context, TRACE is compatible with heterogeneous pre-training across corpora with different channel counts, montages, sequence lengths, and recording domains. Across eight downstream EEG benchmarks, TRACE is evaluated in both settings: when downstream domains are seen only as unlabeled pre-training data and when downstream datasets are completely unseen during pre-training. It obtains the best results on several benchmarks while remaining competitive on motor imagery and clinical event classification tasks, with ablations supporting the importance of cross-channel temporal routing. 

---
# LPDP: Inference-Time Reward Control for Variable-Length DNA Generation with Edit Flows 

**Authors**: Jeongchan Kim, Yunkyung Ko, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.11368)  

**Abstract**: We study the application of recent Edit Flows for inference-time reward control for DNA sequence generation. Unlike most reward-guided DNA generation frameworks, which operate on fixed-length sequence spaces, Edit Flows have a potential to generate variable-length DNA through biologically plausible insertion, deletion, and substitution operations. In particular, we propose Local Perturbation Discrete Programming (LPDP), a training-free, intermediate-state and action-aware local re-solving operator for variable-length DNA edit-action generators at inference time. More specifically, at each guided rollout step, LPDP scores one-step root edits, retains a near-best root band, and re-ranks each retained root by solving a bounded local discrete program around its child sequence. This local program uses the typed geometry of edit actions to focus on coherent substitution, insertion, or deletion subgraphs, and aggregates local continuations with either a hard Max backup or a soft log-sum-exponential (LSE) backup. We instantiate LPDP in two regimes: front-loaded reward tilting for enhancer optimization, where early edits are critical for establishing global regulatory sequence structure, and back-loaded reward tilting for exon-intron-exon inpainting, where late edits fine-tune splice-boundary contexts. 

---
# Causal Fairness for Survival Analysis 

**Authors**: Drago Plecko  

**Link**: [PDF](https://arxiv.org/pdf/2605.11362)  

**Abstract**: In the data-driven era, large-scale datasets are routinely collected and analyzed using machine learning (ML) and artificial intelligence (AI) to inform decisions in high-stakes domains such as healthcare, employment, and criminal justice, raising concerns about the fairness behavior of these systems. Existing works in fair ML cover tasks such as bias detection, fair prediction, and fair decision-making, but largely focus on static settings. At the same time, fairness in temporal contexts, particularly survival/time-to-event (TTE) analysis, remains relatively underexplored, with current approaches to fair survival analysis adopting statistical fairness definitions, which, even with unlimited data, cannot disentangle the causal mechanisms that generate disparities. To address this gap, we develop a causal framework for fairness in TTE analysis, enabling the decomposition of disparities in survival into contributions from direct, indirect, and spurious pathways. This provides a human-understandable explanation of why disparities arise and how they evolve over time. Our non-parametric approach proceeds in four steps: (1) formalizing the necessary assumptions about censoring and lack of confounding using a graphical model; (2) recovering the conditional survival function given covariates; (3) applying the Causal Reduction Theorem to reframe the problem in a form amenable to causal pathway decomposition; (4) estimating the effects efficiently. Finally, our approach is used to analyze the temporal evolution of racial disparities in outcome after admission to an intensive care unit (ICU). 

---
# Options, Not Clicks: Lattice Refinement for Consent-Driven MCP Authorization 

**Authors**: Ying Li, Yanju Chen, Peiran Wang, Issac Khabra, Faysal Hossain Shezan, Yu Feng, Yuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2605.11360)  

**Abstract**: As Model Context Protocol adoption grows, securing tool invocations via meaningful user consent has become a critical challenge, as existing methods, broad always allow toggles or opaque LLM-based decisions, fail to account for dangerous call arguments and often lead to consent fatigue. In this work, we present Conleash, a client-side middleware that enforces boundary-scoped authorization by utilizing a risk lattice to auto-permit safe calls within known boundaries while escalating risks, a policy engine for user-defined invariants, and a refinement loop that converts user decisions into reusable rules. Evaluated on 984 real-world traces, Conleash achieved 98.2% accuracy, caught 99.4% of escalations, and added only 8.2 ms of overhead for policy verification; furthermore, in a user study where N=16, participants significantly preferred Conleash scoped permissions over traditional methods, citing higher trust and reduced prompting. 

---
# Human-AI Productivity Paradoxes: Modeling the Interplay of Skill, Effort, and AI Assistance 

**Authors**: Ali Aouad, Thodoris Lykouris, Huiying Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11350)  

**Abstract**: Generative Artificial Intelligence (AI) tools are rapidly adopted in the workplace and in education, yet the empirical evidence on AI's impact remains mixed. We propose a model of human-AI interaction to better understand and analyze several mechanisms by which AI affects productivity. In our setup, human agents with varying skill levels exert utility-maximizing effort to produce certain task outcomes with AI assistance. We find that incorporating either endogeneity in skill development or in AI unreliability can induce a productivity paradox: increased levels of AI assistance may degrade productivity, leading to potentially significant shortfalls. Moreover, we examine the long-term distributional effect of AI on skill, and demonstrate that skill polarization can emerge in steady state when accounting for heterogeneity in AI literacy -- the agent's capability to identify and adapt to inaccurate AI outputs. Our results elucidate several mechanisms that may explain the emergence of human-AI productivity paradoxes and skill polarization, and identify simple measures that characterize when they arise. 

---
# Large Language Models for Causal Relations Extraction in Social Media: A Validation Framework for Disaster Intelligence 

**Authors**: Ujun Jeong, Saketh Vishnubhatla, Bohan Jiang, Andre Harrison, Adrienne Raglin, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11348)  

**Abstract**: During disasters, extracting causal relations from social media can strengthen situational awareness by identifying factors linked to casualties, physical damage, infrastructure disruption, and cascading impacts. However, disaster-related posts are often informal, fragmented, and context-dependent, and they may describe personal experiences rather than explicit causal relations. In this work, we examine whether Large Language Models (LLMs) can effectively extract causal relations from disaster-related social media posts. To this end, we (1) propose an expert-grounded evaluation framework that compares LLM-generated causal graphs with reference graphs derived from disaster-specific reports and (2) assess whether the extracted relations are supported by post-event evidence or instead reflect model priors. Our findings highlight both the potential and risks of using LLMs for causal relation extraction in disaster decision-support systems. 

---
# Gradient-Free Noise Optimization for Reward Alignment in Generative Models 

**Authors**: Jeongsol Kim, Hongeun Kim, Jian Wang, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2605.11347)  

**Abstract**: Existing reward alignment methods for diffusion and flow models rely on multi-step stochastic trajectories, making them difficult to extend to deterministic generators. A natural alternative is noise-space optimization, but existing approaches require backpropagation through the generator and reward pipeline, limiting applicability to differentiable settings. To address this, here we present ZeNO (Zeroth-order Noise Optimization), a gradient-free framework that formulates noise optimization as a path-integral control problem, estimable from zeroth-order reward evaluations alone. When instantiated with an Ornstein--Uhlenbeck reference process, the update connects to Langevin dynamics implicitly targeting a reward-tilted distribution. ZeNO enables effective inference-time scaling and demonstrates strong performance across diverse generators and reward functions, including a protein structure generation task where backpropagation is infeasible. 

---
# Physics-Informed Teacher-Student Ensemble Learning for Traffic State Estimation with a Varying Speed Limit Scenario 

**Authors**: Archie J. Huang, Dongdong Wang, Shaurya Agarwal, Mohamed Abdel-Aty, Md Mahmudul Islam, Muhammad Shahbaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11346)  

**Abstract**: Physics-informed deep learning (PIDL) neural networks have shown their capability as a useful instrument for transportation practitioners in utilizing the underlying relationship between the state variables for traffic state estimation (TSE). Another efficient traffic management approach is implementing varying speed limits (VSLs) on transportation corridors to control traffic and mitigate congestion. However, the existing training architecture of PIDL in the literature cannot accommodate the changing traffic characteristics on a freeway with VSL. To tackle this challenge, we propose a novel framework integrating teacher-student ensemble training with PIDL neural networks for TSE under VSL scenarios. The physics of flow conservation law is encoded locally in the teacher models by PIDL, and the student model uses a multi-layer perceptron classifier (MLP) to identify traffic characteristics and selects the ensemble member of PIDL neural networks for TSE. This integrated framework provides a natural solution for capturing the heterogeneity of VSL and accurately addressing the TSE problem. The case study results validate the proposed ensemble approach, demonstrating its superior performance in TSE compared to other popular baseline methods, as indicated by relative L2 error. 

---
# Much of Geospatial Web Search Is Beyond Traditional GIS 

**Authors**: Ilya Ilyankou, Stefano Cavazzi, James Haworth  

**Link**: [PDF](https://arxiv.org/pdf/2605.11336)  

**Abstract**: Web search queries concern place far more often than existing labelling schemes suggest, yet the landscape of geospatial web search queries - what people ask of place, and how often - remains poorly characterised at scale. We apply dense sentence embeddings, a lightweight SetFit classifier, and density-based clustering to the full MS MARCO corpus of 1.01 million real Bing queries without prior filtering for toponyms or spatial keywords, identifying 181,827 geospatial queries (18.0%), nearly threefold the 6.17% labelled as Location in the original annotations. The resulting taxonomy of 88 query categories reveals that geospatial web search is dominated by transactional and practical lookups: costs and prices alone account for 15.3% of geospatial queries, nearly twice the size of the entire physical geography theme. Much of this activity - costs, opening hours, contact details, weather, travel recommendations - falls outside the scope traditional GIS systems and knowledge graphs are built to serve. The categories vary substantially in the kind of answer they admit, from deterministic lookups answerable from spatial databases or knowledge graphs to evaluative or temporally volatile queries that require generative or real-time systems. We discuss implications for hybrid retrieval architectures and for benchmarks of geographic reasoning in large language models. We openly release the labelled dataset, classifier, and taxonomy. 

---
# Epistemic Uncertainty for Test-Time Discovery 

**Authors**: Kainat Riaz, Muhammad Ahmed Mohsin, Ahsan Bilal, Muhammad Umer, Ayesha Mohsin, Aqib Riaz, Ali Subhan, John M. Cioffi  

**Link**: [PDF](https://arxiv.org/pdf/2605.11328)  

**Abstract**: Automated scientific discovery using large language models relies on identifying genuinely novel solutions. Standard reinforcement learning penalizes high-variance mutations, which leads the policy to prioritize familiar patterns. As a result, the maximum reward plateaus even as the average reward increases. Overcoming this limitation requires a signal that distinguishes unexplored regions from intrinsically difficult problems. This necessitates measuring disagreement across independently adapted weight hypotheses rather than relying on a single network's confidence. UG-TTT addresses this challenge by maintaining a small ensemble of low-rank adapters over a frozen base model. The per-token disagreement, quantified as the mutual information between ensemble predictions and weight hypotheses, isolates epistemic uncertainty and identifies positions where insufficient coverage leads to adapter divergence rather than intrinsic problem difficulty. This measure is incorporated as an exploration bonus into the policy gradient, directing the policy toward positions where persistent adapter disagreement signals low training coverage, the same frontier where genuine discovery is possible. A nuclear norm regularizer ensures the adapters remain distinct from one another, thereby preserving the exploration signal throughout training. Across four scientific discovery benchmarks, UG-TTT increases the maximum reward on three tasks, maintains substantially higher solution diversity, and an ablation study confirms that the regularizer is essential for sustaining this behavior. 

---
# Beyond Similarity Search: Tenure and the Case for Structured Belief State in LLM Memory 

**Authors**: Jeffrey Flynt  

**Link**: [PDF](https://arxiv.org/pdf/2605.11325)  

**Abstract**: Why do we need another AI to help the AI? We argue you don't. Stateless LLM sessions impose re-orientation costs on iterative, session-heavy workflows. Prior work addresses cross-session memory through retrieval-augmented approaches: store history, embed it, retrieve by semantic similarity. Cross-session memory is a state management problem, not a search problem. Similarity search fails for named entity resolution within bounded vocabulary contexts because beliefs about a shared technical domain are semantically proximate by construction. A single user is the simplest bounded vocabulary context; engineering teams converge on the same property through shared codebases and terminology.
We present Tenure, a local-first proxy that maintains a typed belief store with epistemic status, versioned supersession, and scope isolation, injecting curated context into every LLM session through precision-first retrieval. Hard scope isolation provides a structural guarantee: the right beliefs surface, and only within the boundaries the user has authorized. Tenure's typed schema converts extracted facts into imperative instructions via a why it matters field, making injected beliefs directly actionable rather than raw material for the model to re-derive.
A controlled evaluation on 72 retrieval cases demonstrates the gap. Cosine similarity over dense embeddings achieves mean precision of 0.12. Alias-weighted BM25 maintains mean precision of 1.0, passing 72/72 cases versus 8/72 for cosine similarity on the same corpus. Hybrid retrieval typically solves vocabulary mismatch between disparate authors; Tenure eliminates this structurally: query and belief authors are the same person, and an alias enrichment flywheel continuously indexes their specific vocabulary. Under multi-turn topic drift this worsens: the vector backend produces drift scores of 0.43--0.50 on noise-critical turns where BM25 maintains 0. 

---
# SOMA: Efficient Multi-turn LLM Serving via Small Language Model 

**Authors**: Xueqi Cheng, Qiong Wu, Zhengyi Zhou, Xugui Zhou, Tyler Derr, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11317)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in multi-turn dialogue settings where preserving conversational context across turns is essential. A standard serving practice concatenates the full dialogue history at every turn, which reliably maintains coherence but incurs substantial cost in latency, memory, and API expenditure, especially when queries are routed to large proprietary models. Existing approaches often struggle to balance the trade-off between response quality and efficiency. We propose a framework that exploits the early turns of a session to estimate a local response manifold and then adapt a smaller surrogate model to this local region for the remainder of the conversation. Concretely, we learn soft prompts that maximize semantic divergence between the large and surrogate small language models' responses to surface least-aligned local directions, stabilize training with anti-degeneration control, and distill the mined cases into localized LoRA fine-tuning so the surrogate runs without prompts at inference. A simple gate enables a one-time switch with rollback on drift. We further provide a theoretical analysis for key components in SOMA. Extensive experiments show the effectiveness of SOMA. The source code is provided at: this https URL. 

---
# Natural Language based Specification and Verification 

**Authors**: Zhaorui Li, Chengyu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.11315)  

**Abstract**: Recent frontier large language models (LLMs) have shown strong performance in identifying security vulnerabilities in large, mature open-source systems. As LLM-generated code becomes increasingly common, a natural goal is to prevent such models from producing vulnerable implementations in the first place. Formal verification offers a principled route to this objective, but existing verification pipelines typically require specifications written in rigid formal languages. Prior work has explored using LLMs to synthesize such specifications, with limited success. In this paper, we investigate a different approach: using LLMs both to generate specifications and to verify implementations compositionally when the specifications are expressed in natural language. Our preliminary results suggest that this approach is promising. 

---
# Quantifying Rodda and Graham Gait Classification from 3D Makerless Kinematics derived from a Single-view Video in a Heterogeneous Pediatric Clinical Cohort 

**Authors**: Lauhitya Reddy, Seth Donahue, Jeremy Bauer, Susan Sienko, Anita Bagley, Joseph Krzak, Maura Eveld, Karen Kruger, Ross Chafetz, Vedant Kulkarni, Hyeokhyen Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2605.11314)  

**Abstract**: Cerebral Palsy (CP) is a neurological disorder of movement and the most common cause of lifelong physical disability in childhood. Approximately 75% of children with CP are ambulatory, and accurate gait assessment is central to preserving walking function, which deteriorates by mid-adulthood in a quarter to half of adults with CP. The Rodda and Graham classification system quantifies sagittal-plane gait deviations using ankle and knee z-scores derived from 3D Instrumented Gait Analysis (3D-IGA), but 3D-IGA is expensive and limited to specialized centers, while observational assessment shows only moderate inter-rater agreement. We developed a markerless gait analysis pipeline that quantifies Rodda and Graham knee and ankle z-scores directly from single-view clinical gait videos. Across 1,058 bilateral limb samples from 529 trials of 152 children (88 male, 63 female; age 12.1 $\pm$ 4.0 years; 60 distinct primary diagnoses, cerebral palsy the most common at $n=54$), the sagittal-view model achieved $R^2 = 0.80 \pm 0.02$ and CCC $= 0.89 \pm 0.02$ for knee z-scores and $R^2 = 0.57 \pm 0.02$ and CCC $= 0.72 \pm 0.02$ for ankle z-scores against 3D-IGA. Binary screening for excess knee flexion achieves AUROC $= 0.88$, correctly identifying 83% of affected children, and applying Rodda and Graham rules yields $43 \pm 1$% 7-class accuracy with macro-AUROC $= 0.78 \pm 0.01$, ankle prediction error remaining the primary bottleneck. Beyond cross-sectional screening, continuous z-scores support longitudinal trajectory tracking across visits, providing a quantitative substrate for monitoring disease progression and treatment response unavailable from observational scales. These results demonstrate the feasibility of video-based z-score estimation, excess-flexion screening, and longitudinal trajectory tracking as a path toward scalable, objective gait assessment in low-resource clinical settings. 

---
# A Theory of Time-Sensitive Language Generation: Sparse Hallucination Beats Mode Collapse 

**Authors**: Atul Ganju, Travis McVoy, Shaddin Dughmi, Shang-Hua Teng  

**Link**: [PDF](https://arxiv.org/pdf/2605.11302)  

**Abstract**: We study language generation in the limit under a global preference ordering on strings, as introduced by Kleinberg and Wei. As in [arXiv:2504.14370, arXiv:2511.05295], we aim for \emph{breadth}, but impose an additional requirement of timeliness: higher-ranked strings should be generated earlier. A string is then only credited if it is generated before a deadline, where its deadline is defined by a function that maps a string's rank in the target language to the time by which it must be produced. This is in keeping with a central consideration in machine learning, where inductive bias favors ``simpler'' or ``more plausible'' outputs, all else being equal. We show that timely generation is impossible in a strong sense for eventually consistent generators -- the protagonists of most prior related work. Under what is perhaps the mildest natural relaxation of consistency, a hallucination rate that vanishes over time, we show that we can circumvent our impossibility result. In particular, we can achieve optimal density with respect to any superlinear deadline function. We also show this is tight by ruling out timely generation with linear deadlines and vanishing hallucination rate. 

---
# ReAD: Reinforcement-Guided Capability Distillation for Large Language Models 

**Authors**: Xueqi Cheng, Xugui Zhou, Tyler Derr, Yushun Dong  

**Link**: [PDF](https://arxiv.org/pdf/2605.11290)  

**Abstract**: Capability distillation applies knowledge distillation to selected model capabilities, aiming to compress a large language model (LLM) into a smaller one while preserving the abilities needed for a downstream task. However, most existing methods treat capabilities as independent training targets and overlook how improving one capability can reshape the student's broader capability profile, especially when multiple abilities jointly determine task success. We study capability distillation under a fixed token budget and identify two consistent patterns: distillation induces systematic, budget-dependent cross-capability transfer, and additional budget often brings limited task-relevant gains while sometimes degrading other useful abilities. Building on these insights, we propose ReAD, a Reinforcement-guided cApability Distillation framework that explicitly accounts for capability interdependence. ReAD first infers task-essential capabilities, then generates capability-targeted supervision on the fly, and finally uses an uncertainty-aware contextual bandit to adaptively allocate the distillation budget based on expected utility gains. Extensive experiments show that ReAD improves downstream utility under the same token budget while reducing harmful spillover and wasted distillation effort compared to strong baselines. Our code is publicly available at this https URL. 

---
# Beyond Similarity: Temporal Operator Attention for Time Series Analysis 

**Authors**: Jevon Twitty, Vinh Pham, Nitiwith Rotchanarak, Viresh Pati, Yubin Kim, Shihao Yang, Jiecheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11287)  

**Abstract**: A persistent paradox in time-series forecasting is that structurally simple MLP and linear models often outperform high-capacity Transformers. We argue that this gap arises from a mismatch in the sequence-modeling primitive: while many time-series dynamics are governed by global temporal operators (e.g., filtering and harmonic structure), standard attention forms each output as a convex combination of inputs. This restricts its ability to represent signed and oscillatory transformations that are fundamental to temporal signal processing. We formalize this limitation as a simplex-constrained mixing bottleneck in softmax attention, which becomes especially restrictive for operator-driven time-series tasks. To address this, we propose $\textbf{Temporal Operator Attention (TOA)}$, a framework that augments attention with explicit, learnable sequence-space operators, enabling direct signed mixing across time while preserving input-dependent adaptivity. To make dense $N \times N$ operators practical, we introduce Stochastic Operator Regularization, a high-variance dropout mechanism that stabilizes training and prevents trivial memorization. Across forecasting, anomaly detection, and classification benchmarks, TOA consistently improves performance when integrated into standard backbones such as PatchTST and iTransformer, with particularly strong gains in reconstruction-heavy tasks. These results suggest that explicit operator learning is a key ingredient for effective time-series modeling. 

---
# Rethinking external validation for the target population: Capturing patient-level similarity with a generative model 

**Authors**: Mohammad Azizmalayeri, Ameen Abu-Hanna, Saskia Houterman, Marije M. Vis, Giovanni Cinà  

**Link**: [PDF](https://arxiv.org/pdf/2605.11284)  

**Abstract**: Background: External validation is essential for assessing the transportability of predictive models. However, its interpretation is often confounded by differences between external and development populations. This study introduces a framework to distinguish model deficiencies from case-mix effects.
Method: We propose a framework that quantifies each external patient's similarity to the development data and measures performance in subgroups with varying levels of alignment to the development distribution. We use generative models, specifically autoencoders, to estimate similarity, offering a more flexible alternative to traditional linear approaches and enabling validation without sharing the original development data. The utility of autoencoder-based similarity measure is demonstrated using synthetic data, and the framework's application is illustrated using data from the Netherlands Heart Registration (NHR) to predict mortality after transcatheter aortic valve implantation.
Results: Our framework revealed substantial variation in model performance across similarity-defined subgroups, differences that remain hidden under conventional external validation yet can meaningfully alter conclusions. In several settings, conventional external validation suggested poor overall performance. However, after accounting for differences in patient characteristics, for some sub-groups, the model performance was consistent with internal validation results. Conversely, apparently acceptable overall performance could mask clinically relevant performance deficits in specific subgroups.
Conclusion: The proposed framework enhances the interpretability of external validation by linking model performance to population alignment with the development data. This provides a more principled basis for deciding whether a model is transportable and to which patients it can be safely applied. 

---
# Discovery of Interpretable Surrogates via Agentic AI: Application to Gravitational Waves 

**Authors**: Tousif Islam, Digvijay Wadekar, Tejaswi Venumadhav, Matias Zaldarriaga, Ajit Kumar Mehta, Javier Roulet, Barak Zackay  

**Link**: [PDF](https://arxiv.org/pdf/2605.11280)  

**Abstract**: Fast surrogate models for expensive simulations are now essential across the sciences, yet they typically operate as black boxes. We present \texttt{GWAgent}, a large language model (LLM)-based workflow that constructs interpretable analytic surrogates directly from simulation data. Surrogate modeling is well suited to agentic workflows because candidate models can be quantitatively validated against ground-truth simulations at each iteration. As a demonstration, we build a surrogate for gravitational waveforms from eccentric binary black hole mergers. We show that providing the agent with a physics-informed domain ansatz substantially improves output model accuracy. The resulting analytic surrogate attains a median Advanced LIGO mismatch of $6.9\times10^{-4}$ together with an $\sim 8.4\times$ speedup in waveform evaluation, surpassing both symbolic regression and conventional machine learning baselines. Beyond producing an accurate model, the workflow identifies compact physical structure from the learned representation. As an astrophysical application, we use \texttt{GWAgent} to analyze the eccentricity of GW200129 and infer $e_{20\mathrm{Hz}}=0.099^{+0.063}_{-0.044}$. These results show that validation-constrained agentic workflows can produce accurate, fast, and interpretable surrogates for scientific simulations and inference. 

---
# Generative AI for Visualizing Highway Construction Hazards Through Synthetic Images and Temporal Sequences 

**Authors**: Trevor Neece, Mason Smetana, Lev Khazanovich  

**Link**: [PDF](https://arxiv.org/pdf/2605.11276)  

**Abstract**: Highway construction workers face a high risk of serious injury or death. Image-based training materials depicting hazardous scenarios are essential for engaging safety instruction but remain scarce due to ethical and logistical barriers. This study develops and evaluates a generative AI methodology for producing synthetic visualizations of highway construction hazards from OSHA Severe Injury Report narratives. Two modes were developed: a single-pass approach yielding one image per incident, and a temporal approach producing a four-stage sequence. A sample of 75 incident records yielded 750 images, evaluated using CLIP-based semantic retrieval and expert assessment across dimensions such as educational utility, fidelity, and alignment. Single-pass images achieved 81.1% educational acceptability with fidelity and alignment scores of 4.14/5 and 4.07/5, respectively, while temporal sequences achieved 60.9% acceptability with comparable alignment (3.94/5) but lower fidelity (3.51/5). CLIP-based retrieval revealed that both modes produce images with statistically significant retrieval capabilities. This is among the first studies to leverage modern autoregressive image generation models for visualizing construction hazards from reported severe injuries and to generate temporally sequenced hazard imagery, and a new multi-dimensional evaluation framework was developed to support future research in this domain. The work enables safety trainers to pair narrative storytelling with visual learning material without photographing real-world hazards, and the framework could be applied to datasets across diverse domains, enabling synthetic image generation tailored to new application areas. 

---
# Localization Boosting for Growth Markets: Mitigating Cross-Locale Behavioral Bias in Learning-to-Rank 

**Authors**: Suryaa Veerabathiran Seran, Ashwin Naresh Kumar, Tracy Holloway King, Jing Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.11272)  

**Abstract**: Adobe Express is expanding internationally, but the US has a disproportionately large content supply and interaction volume. Learning-to-rank (LTR) models trained primarily on behavioral feedback inherit this imbalance: templates popular in US are over-served in non-US locales. This cross-locale exposure bias suppresses local content discoverability and degrades ranking quality in growth locales.
We show that click-only training suppresses semantically informative localization features. Adding vision-language model (VLM) graded relevance labels as auxiliary supervision alongside clicks improves semantic alignment but does not preserve local content visibility. We propose a multi-objective framework combining behavioral supervision, VLM-derived relevance signals, and locale-aware boosting. Across five locales, the resulting model improves relevance while restoring stable localization, demonstrating the importance of disentangling exposure from semantic supervision. 

---
# gwBenchmarks: Stress-Testing LLM Agents on High-Precision Gravitational Wave Astronomy 

**Authors**: Tousif Islam, Digvijay Wadekar, Zihan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11269)  

**Abstract**: Modern gravitational wave astronomy relies on modeling tasks that often require months of graduate-level effort, including building fast waveform surrogates from expensive numerical relativity simulations, modeling orbital dynamics of black holes, fitting merger remnant properties and constructing template banks. These problems demand extreme precision to support detection and parameter inference, with state-of-the-art models achieving $\lesssim 10^{-4}$ relative error. We study whether state-of-the-art LLM coding agents can perform such end-to-end scientific modeling, where success requires constructing models with stringent accuracy criteria and reasoning about physical systems. We introduce gwBenchmarks, a suite of eight tasks grounded in gravitational wave analytic calculations and numerical simulations collectively representing over $10^8$ core-hours of compute. The tasks span interpolation, regression, and high-dimensional time-series modeling, requiring a combination of numerical methods, machine learning, and physics-informed approaches. In preliminary experiments, agents frequently relied on proxy metrics, partial evaluation, or fabricated results to spuriously complete tasks. We therefore implement an external pre-defined framework to gauge agent progress. Evaluating twelve coding agents, we find no consistent winner. On the easiest task, multiple agents converge to the same cubic spline solution, with one rediscovering a coordinate transformation widely used in the literature. On harder tasks like analytic waveform modeling, all agents fall 1-2 orders of magnitude short of domain requirements and exhibit systematic failures, including metric misuse, constraint violations, and result fabrication. Our code, data, and website are publicly available. 

---
# DenseTRF: Texture-Aware Unsupervised Representation Adaptation for Surgical Scene Dense Prediction 

**Authors**: Guiqiu Liao, Matjaž Jogan, Daniel A. Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2605.11265)  

**Abstract**: Dense prediction tasks in surgical computer vision, such as segmentation and surgical zone prediction, can provide valuable guidance for laparoscopic and robotic surgery. However, these models often suffer from distribution shifts, as training datasets rarely cover the variability encountered during deployment, leading to poor generalization. We propose DenseTRF, a self-supervised representation adaptation framework based on texture-centric attention. Our method leverages slot attention to learn texture-aware representations that capture invariant visual structures. By adapting these representations to the target distribution without supervision, DenseTRF significantly improves robustness to domain shifts. The framework is implemented through conditioning dense prediction on slot attention and model merging strategies. Experiments across multiple surgical procedures demonstrate improved cross-distribution generalization in comparison to state-of-the-art segmentation models and test-distribution adaptation methods for dense prediction tasks. 

---
# Curriculum Learning-Guided Progressive Distillation in Large Language Models 

**Authors**: Jincheng Cao, Fanzhi Zeng, Leqi Liu, Aryan Mokhtari  

**Link**: [PDF](https://arxiv.org/pdf/2605.11260)  

**Abstract**: Knowledge distillation is a key technique for transferring the capabilities of large language models (LLMs) into smaller, more efficient student models. Existing distillation approaches often overlook two critical factors: the learning order of training data and the capacity mismatch between teacher and student models. This oversight limits distillation performance, as manifested by the counter-intuitive phenomenon where stronger teachers fail to produce better students. In this work, we propose Curriculum Learning-Guided Progressive Distillation (CLPD), a unified framework that explicitly accounts for both factors by aligning data difficulty with teacher strength. CLPD constructs an explicit curriculum by organizing training examples from easy to hard, while simultaneously applying an implicit curriculum over supervision signals by progressively scheduling teachers of increasing capacity. Our framework is modular and can be integrated into standard distillation algorithms with minimal overhead. Empirical results on the reasoning benchmarks demonstrate that CLPD consistently outperforms standard distillation, data ordering alone, and teacher scheduling alone across multiple settings. These findings highlight the importance of jointly considering data ordering and teacher capacity when distilling reasoning abilities into small language models. 

---
# RETUYT-INCO at BEA 2026 Shared Task 2: Meta-prompting in Rubric-based Scoring for German 

**Authors**: Ignacio Sastre, Ignacio Remersaro, Facundo Díaz, Nicolás De Horta, Luis Chiruzzo, Aiala Rosá, Santiago Góngora  

**Link**: [PDF](https://arxiv.org/pdf/2605.11242)  

**Abstract**: In this paper, we present the RETUYT-INCO participation at the BEA 2026 shared task "Rubric-based Short Answer Scoring for German". Our team participated in track 1 (Unseen answers three-way), track 3 (Unseen answers two-way) and track 4 (Unseen questions two-way). Since these tracks required scoring short student answers using specific rubrics, we looked for ways to handle the changing nature of the task. We created a method called Meta-prompting. In this approach, an LLM creates a custom prompt based on examples from the Train set. This prompt is then used to grade new student answers. Along with this method, we also describe other approaches we used, such as classic machine learning, fine-tuning open-source LLMs, and different prompting techniques. According to the official results, our team placed 6th out of 8 participants in Track 1 with a QWK of 0.729. In Track 3, we secured 4th place out of 9 with a QWK of 0.674, and we also placed 4th out of 8 in Track 4 with a QWK of 0.49. 

---
# Internalizing Curriculum Judgment for LLM Reinforcement Fine-Tuning 

**Authors**: Han Zheng, Yining Ma, Karthick Gunasekaran, Bharathan Balaji, Zheng Du, Shiv Vitaladevuni, Cathy Wu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11235)  

**Abstract**: In LLM Reinforcement Fine-Tuning (RFT), curriculum learning drives both efficiency and performance. Yet, current methods externalize curriculum judgment via handcrafted heuristics or auxiliary models, risking misalignment with the policy's training dynamics. In this paper, we introduce METIS (METacognitive Internalized Self-judgment), a novel framework that internalizes curriculum judgment as a native capability. Leveraging a critical observation that within-prompt reward variance effectively gauges prompt informativeness, METIS predicts this metric based on recent training outcomes as lightweight in-context learning examples. This intrinsic self-judgment then dynamically dictates the training allocation. Moreover, METIS closes the loop between judgment and optimization by jointly optimizing the standard RFT rewards and a self-judgment reward. This allows the policy to learn what to learn next, as a form of metacognition. Across extensive discrete and continuous RFT benchmarks from mathematical reasoning, code generation, to agentic function-calling, METIS consistently delivers superior performance while accelerating convergence by up to 67%. By bypassing handcrafted heuristics and auxiliary models, our work establishes a simple, closed-loop, and highly efficient curriculum internalization paradigm for LLM reinforcement fine-tuning. 

---
# LiBaGS: Lightweight Boundary Gap Synthesis for Targeted Synthetic Data Selection 

**Authors**: Abhishek Moturu, Anna Goldenberg, Babak Taati  

**Link**: [PDF](https://arxiv.org/pdf/2605.11231)  

**Abstract**: Synthetic data is useful only when the added samples fill missing parts of the training distribution that matter for the downstream task. We introduce LiBaGS, a lightweight, generator-agnostic method for targeted synthetic training data selection. LiBaGS scores candidate synthetic samples by combining decision-boundary proximity, predictive uncertainty, real-data density, and support validity, so that selected samples are both informative and likely to remain on the real data manifold. We then use a boundary-gap allocation rule that targets sparse but realistic decision-boundary neighborhoods, rather than simply adding more data or selecting only the most uncertain candidates. LiBaGS also learns when enough synthetic samples have been added through a marginal-value stopping rule, assigns softer labels near ambiguous boundaries, and uses a diversity objective to avoid redundant near-duplicate selections. Experiments show that LiBaGS improves accuracy over classical oversampling, hard augmentation, uncertainty and density ablations, and targeted-generation selection criteria. 

---
# Comment and Control: Hijacking Agentic Workflows via Context-Grounded Evolution 

**Authors**: Neil Fendley, Zhengyu Liu, Aonan Guan, Jiacheng Zhong, Yinzhi Cao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11229)  

**Abstract**: Automation platforms such as GitHub Actions and n8n are increasingly adopting so-called agentic workflows, which integrate Large Language Model (LLM) agents for tasks such as code review and data synchronization. While bringing convenience for developers, this integration exposes a new risk: An adversary may control and craft certain inputs, such as GitHub issue comments, to manipulate the LLM agent for unwanted actions, such as credential exfiltration and arbitrary command execution. To our knowledge, no prior academic work has studied such a risk in agentic workflows. In this paper, we design the first detection and exploitation framework, called JAW, to hijack agentic workflows hosted on automation platforms via a novel approach called Context-Grounded Evolution. Our key idea is to evolve agentic workflow inputs under the contexts derived from hybrid program analysis for hijacking purposes. Specifically, JAW generates agentic workflow contexts through three analyses: (i) static path-feasibility analysis to identify feasible agent-invocation paths and the input constraints required to trigger them, (ii) dynamic prompt-provenance analysis to determine how that input is transformed and embedded into the LLM context, and (iii) capability analysis to identify the actions and restrictions available to the agent at runtime. Our evaluation of JAW on GitHub workflows and n8n templates showed that 4714 GitHub workflows and eight n8n templates can be successfully hijacked, for example, to leak user credentials. Our findings span 15 widely-used GitHub Actions, including official GitHub Actions for Claude Code, Gemini CLI, Qwen CLI, and Cursor CLI, and two official n8n nodes. We responsibly disclosed all findings to the affected vendors and received many acknowledgements, fixes, and bug bounties, notably from GitHub, Google, and Anthropic. 

---
# ABRA: Agent Benchmark for Radiology Applications 

**Authors**: Bulat Maksudov, Vladislav Kurenkov, Kathleen M. Curran, Alessandra Mileo  

**Link**: [PDF](https://arxiv.org/pdf/2605.11224)  

**Abstract**: Existing medical-agent benchmarks deliver imaging as pre-selected samples, never as an environment the agent must navigate. We introduce ABRA, a radiology-agent benchmark in which the agent operates an OHIF viewer and an Orthanc DICOM server through twenty-one function-calling tools that span slice navigation, windowing, series selection, pixel-coordinate annotation, and structured reporting. ABRA contains 655 programmatically generated tasks across three difficulty tiers and eight types (viewer control, metadata QA, vision probe, annotation, longitudinal comparison, BI-RADS reporting, and oracle variants of annotation and BI-RADS reporting), drawn from LIDC-IDRI, Duke Breast Cancer MRI, and NLST New-Lesion LongCT. Each episode is scored along Planning, Execution, and Outcome (Bluethgen et al., 2025) by task-type-specific automatic scorers. Ten current models, five closed-weight and five open-weight, reach at least 89% Execution on real annotation but only 0-25% Outcome; on the paired oracle variant where a simulated detector supplies the finding, Outcome on the same task reaches 69-100% across the models evaluated, localising the bottleneck to perception rather than tool orchestration. Code, task generators, and scorers are released at this https URL 

---
# Leveraging RAG for Training-Free Alignment of LLMs 

**Authors**: John T. Halloran  

**Link**: [PDF](https://arxiv.org/pdf/2605.11217)  

**Abstract**: Large language model (LLM) alignment algorithms typically consist of post-training over preference pairs. While such algorithms are widely used to enable safety guardrails and align LLMs with general human preferences, we show that state-of-the-art alignment algorithms require significant computational resources while being far less capable of enabling refusal guardrails for recent agentic attacks. Thus, to improve refusal guardrails against such attacks without drastically increasing computational overhead, we introduce Retrieval Augmented Generation for Pref erence alignment (RAG-Pref), a simple RAG-based alignment algorithm which conditions on preferred and dispreferred samples to leverage contrastive information during inference. RAG-Pref is online (training-free), compatible with off-the-shelf packages, and, when combined with offline (training-based) alignment algorithms, enables more than an average 3.7 factor improvement in agentic attack refusals across five widely used LLMs, compared to 2.9 for other online alignment algorithms and 1.5 for offline alignment alone. We conclude by showing that, in stark contrast to other online alignment methods, RAG-Pref similarly increases performance on general human-preference alignment tasks and does not drastically increase overall computational requirements. 

---
# ReCoVer: Resilient LLM Pre-Training System via Fault-Tolerant Collective and Versatile Workload 

**Authors**: Ziyue Liu, Zhengyang Wang, Ruijie Zhang, Avinash Maurya, Hui Zhou, Paul Hovland, Sheng Di, Franck Cappello, Bogdan Nicolae, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11215)  

**Abstract**: Pre-training large language models on massive GPU clusters has made hardware faults routine rather than rare, driving the need for resilient training systems. Yet existing frameworks either focus on specific parallelism schemes or risk drifting away from a failure-free training trajectory. We propose ReCoVer, a resilient LLM pre-training system that upholds a single invariant: each iteration keeps the number of microbatches constant, ensuring per-iteration gradients remain stochastically equivalent to a failure-free run. The framework is organized as three decoupled protocol layers: (1) Fault-tolerant collectives that isolate faults from propagating across replicas; (2) in-step fine-grained recovery that preserves intra-iteration progress and prevents gradient corruption; (3) versatile-workload policy that dynamically redistributes microbatch quotas across the survivors. The design is parallelism-agnostic, integrating directly with both 3D parallelism and Hybrid Sharded Data Parallel (HSDP) as a drop-in substrate. We evaluate our implementation on end-to-end pre-training tasks for up to 512 GPUs, ReCoVer successfully preserves the training trajectory from a failure-free reference despite of 256 GPUs lost spread across the run. For comparison with checkpoint-and-restart baselines, ReCoVer demonstrates $2.23\times$ higher effective throughput after successive failures. This advantage results in ReCoVer processing 74.9% more tokens at 234 GPU-hours, with the gap widening as the training prolongs. 

---
# The Scaling Law of Evaluation Failure: Why Simple Averaging Collapses Under Data Sparsity and Item Difficulty Gaps, and How Item Response Theory Recovers Ground Truth Across Domains 

**Authors**: Jung Min Kang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11205)  

**Abstract**: Benchmark evaluation across AI and safety-critical domains overwhelmingly relies on simple averaging. We demonstrate that this practice produces substantially misleading rankings when two conditions co-occur: (1) the evaluation matrix is sparse and (2) items vary substantially in difficulty. Through controlled simulation experiments across four domains -- NLP (GLUE), clinical drug trials, autonomous vehicle safety, and cybersecurity -- we show that Spearman rank correlation $\rho$ between simple-average rankings and ground-truth rankings degrades from $\rho = 1.000$ at 100% coverage to $\rho = 0.809$ at 67% coverage with high difficulty heterogeneity (mean over 20 seeds). A standard two-parameter logistic (2PL) Item Response Theory (IRT) model maintains $\rho \geq 0.996$ across all conditions. A 150-condition grid sweep over sparsity $S \in [0, 0.70]$ and difficulty gap $D \in [0.5, 5.0]$ confirms that ranking error forms a failure surface with a strong $S \times D$ interaction ($\gamma_3 = +0.20$, $t = 13.05$), while IRT maintains $\rho \geq 0.993$ throughout. We discuss implications for Physical AI benchmarking, where evaluation matrices are often incomplete and difficulty gaps are extreme. 

---
# Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing 

**Authors**: Yunze Zhao, Yibo Zhao, Yuchen Zhang, Zaoxing Liu, Michelle L. Mazurek  

**Link**: [PDF](https://arxiv.org/pdf/2605.11202)  

**Abstract**: LLM inference and serving systems have become security-critical infrastructure; however, many of their most concerning failures arise from the serving layer rather than from model behavior alone. Modern inference engines combine KV cache, batching, prefix sharing, speculative decoding, adapters, and multi-tenant scheduling, creating shared-state behavior that only emerges under realistic concurrent workloads and is missed by standard model, safety, and API tests. We present GRIEF, a greybox fuzzer for LLM inference engines that treats timed multi-request traces as first-class inputs, uses lightweight oracles to detect crashes, hangs, performance pathologies, and silent output corruption, and applies controlled replay with log-probability checks to confirm reproducible serving-layer failures. Across early campaigns on vLLM and SGLang, GRIEF discovers 15 vulnerabilities, 10 confirmed by engine developers, including 2 CVEs, spanning KV-cache isolation failures, cross-request performance interference, and crash or liveness bugs. These results show that concurrency, caching, and state reuse can induce silent cross-request contamination, noisy-neighbor denial of service, and delayed crashes without malformed inputs or explicit server errors, making concurrent serving behavior a first-class security and reliability boundary for LLM infrastructure. 

---
# Exploring Token-Space Manipulation in Latent Audio Tokenizers 

**Authors**: Francesco Paissan, Luca Della Libera, Mirco Ravanelli, Cem Subakan  

**Link**: [PDF](https://arxiv.org/pdf/2605.11192)  

**Abstract**: Neural audio codecs provide compact discrete representations for speech generation and manipulation. However, most codecs organize tokens as frame-level sequences, making it difficult to study or intervene on global factors of variation. In this work, we propose the Latent Audio Tokenizer for Token-space Editing (LATTE) that appends a fixed set of learnable latent tokens to the audio feature sequence and retains only these tokens for quantization and decoding. This design produces a compact, non-temporally aligned bottleneck in which each token can aggregate global information across the full utterance. We show that the resulting tokenizer preserves competitive reconstruction quality in low-bitrate speech coding settings while enabling simple token-space interventions. In particular, we find that swapping selected latent token positions between utterances can modify global attributes, such as speaker identity and background noise, and we evaluate these interventions on voice conversion and denoising tasks. Our results suggest that compact latent audio tokenizers can support controllable audio manipulation without supervision in task-specific editing models. 

---
# Adversarial SQL Injection Generation with LLM-Based Architectures 

**Authors**: Ali Karakoc, H. Birkan Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11188)  

**Abstract**: SQL injection (SQLi) attacks are still one of the serious attacks ranked in the Open Worldwide Application Security Project (OWASP) Top 10 threats. Today, with advances in Artificial Intelligence (AI), especially in Large Language Models (LLMs), an opportunity has been created for automating adversarial attack tests to measure the defense mechanisms. In this paper, we aim to create a comprehensive evaluation of use cases that utilize LLMs for adversarial SQL injection generation. We introduce two novel LLM-based systems, Retrieval Augmented Generation for Adversarial SQLi (RADAGAS) and Reflective Chain-of-Thought SQLi (RefleXQLi), and compare them with existing baselines against 10 Web Application Firewalls (WAFs) and one execution-based MySQL validator. To perform a comprehensive test, we used six rule-based open-source WAFs (ModSecurity PL1--3, Coraza PL1--3), 2 AI/ML-based WAFs (WAF Brain, CNN-WAF), and 2 commercial WAFs (AWS WAF and Cloudflare WAF). For the LLM models, we used GPT-4o, Claude 3.7 Sonnet, and DeepSeek R1. Our tests consist of 240 experiments that generate 240,000 payloads and perform 2.2 million tests against WAFs. Our comprehensive evaluation reveals that RADAGAS-GPT4o outperforms other baseline models with a 22.73\% bypass rate. The proposed RADAGAS variants are highly successful on AI/ML-based WAFs (92.49\% on WAF-Brain by RADAGAS-DeepSeek, 80.48\% on CNN-WAF by RADAGAS-Claude), but struggle to bypass rule-based WAFs (0--5.70\% on ModSecurity and Coraza). In addition to these findings, another observation is that creating less diverse payloads achieves more bypasses, however they show poor results if the initially chosen payload is not successful. We observe that our findings provide a comprehensive view on using LLM-based approaches in security testing. 

---
# CATS: Cascaded Adaptive Tree Speculation for Memory-Limited LLM Inference Acceleration 

**Authors**: Yuning Han, Yangchenchen Jin, Dylan Zhao, Jingwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2605.11186)  

**Abstract**: Auto-regressive decoding in Large Language Models (LLMs) is inherently memory-bound: every generation step requires loading the model weights and intermediate results from memory (e.g., High-Bandwidth Memory (HBM) for GPU servers), making throughput bottlenecked by memory bandwidth rather than compute. Speculative decoding addresses this by enabling parallel verification of multiple draft tokens, effectively amortizing the cost of each target-model call. However, existing speculative decoding methods are designed under the assumption that HBM is sufficiently large to hold both the target model and an auxiliary draft model simultaneously -- an assumption that breaks down on memory-constrained devices such as edge platforms with limited DRAM. We analyze the inference bottleneck in this memory-limited regime and propose CATS, a self-speculative decoding framework that conducts cascaded verification and correction based on the memory budget and parameter offloading patterns on memory-limited devices. This design maximizes token acceptance rate and end-to-end speedup while keeping the peak memory footprint on the device equal to that of the target model alone. We evaluate CATS on different models across five benchmarks on real edge devices. CATS can achieve a wall-clock speedup of up to 5.08x with no degradation in generation quality, outperforming the SOTA method by up to 1.45x under edge memory constraints. 

---
# Muon is Not That Special: Random or Inverted Spectra Work Just as Well 

**Authors**: Zakhar Shumaylov, Nathaël Da Costa, Peter Zaika, Bálint Mucsányi, Alex Massucco, Yoav Gelberg, Carola-Bibiane Schönlieb, Yarin Gal, Philipp Hennig  

**Link**: [PDF](https://arxiv.org/pdf/2605.11181)  

**Abstract**: The recent empirical success of the Muon optimizer has renewed interest in non-Euclidean optimization, typically justified by similarities with second-order methods, and linear minimization oracle (LMO) theory. In this paper, we challenge this geometric narrative through three contributions, demonstrating that precise geometric structure is not the key factor affecting optimization performance. First, we introduce Freon, a family of optimizers based on Schatten (quasi-)norms, powered by a novel, provably optimal QDWH-based iterative approximation. Freon naturally interpolates between SGD and Muon, while smoothly extrapolating into the quasi-norm regime. Empirically, the best-performing Schatten parameters for GPT-2 lie strictly within the quasi-norm regime, and thus cannot be represented by any unitarily invariant LMO. Second, noting that Freon performs well across a wide range of exponents, we introduce Kaon, an absurd optimizer that replaces singular values with random noise. Despite lacking any coherent geometric structure, Kaon matches Muon's performance and retains classical convergence guarantees, proving that strict adherence to a precise geometry is practically irrelevant. Third, having shown that geometry is not the primary driver of performance, we demonstrate it is instead controlled by two local quantities: alignment and descent potential. Ultimately, each optimizer must tune its step size around these two quantities. While their dynamics are difficult to predict a-priori, evaluating them within a stochastic random feature model yields a precise insight: Muon succeeds not by tracking an ideal global geometry, but by guaranteeing step-size optimality. 

---
# Oversmoothing as Representation Degeneracy in Neural Sheaf Diffusion 

**Authors**: Arif Dönmez, Axel Mosig, Ellen Fritsche, Katharina Koch  

**Link**: [PDF](https://arxiv.org/pdf/2605.11178)  

**Abstract**: Neural Sheaf Diffusion (NSD) generalizes diffusion-based Graph Neural Networks by replacing scalar graph Laplacians with sheaf Laplacians whose learned restriction maps define a task-adapted geometry. While the diffusion limit of NSD is known to be the space of global sections, the representation-theoretic structure of this harmonic space remains largely implicit. We develop a quiver-theoretic interpretation of NSD by identifying cellular sheaves on graphs with representations of the associated incidence quiver. Under this correspondence, learned sheaf geometries become points in a finite-dimensional representation space. We show that direct-sum decompositions of the underlying incidence-quiver representation induce decompositions of the harmonic space reached in the diffusion limit. This gives an algebraic interpretation of oversmoothing as representation degeneration: learned sheaves may collapse toward low-complexity summands whose global sections fail to preserve discriminative information. Building on this viewpoint, we connect sheaf diffusion to stability and moment-map principles from Geometric Invariant Theory. We introduce moment-map-inspired regularizers that bias restriction maps toward balanced representation geometries, and identify a structural obstruction in equal-stalk architectures: when $d_v = d_e$, admissibility for learnable stability parameters forces the trivial all-object summand onto a stability wall. Non-uniform stalk dimensions remove this obstruction, making adaptive stability meaningful. Experiments on heterophilic benchmarks are consistent with this mechanism: breaking stalk symmetry can reduce variance or improve validation behavior, and adaptive stability becomes more effective in selected rectangular settings. Overall, our framework reframes oversmoothing as a degeneration phenomenon in the representation geometry underlying learned sheaf diffusion. 

---
# The Bicameral Model: Bidirectional Hidden-State Coupling Between Parallel Language Models 

**Authors**: Cedric Flamant, Udaya Ghai, Kanna Shimizu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11167)  

**Abstract**: Existing multi-model and tool-augmented systems communicate by generating text, serializing every exchange through the output vocabulary. Can two pretrained language models instead coordinate through a continuous, concurrent channel? The Bicameral Model couples two frozen language models through a trainable neural interface on their intermediate hidden states. At every generation step, both models run in lockstep: a primary model drives the task while an auxiliary model operates tools, solves constraints, or executes code, with both conditioning on each other's activations through a translation network and a learned suppression gate ($\sim$1\% of combined parameters). The gate learns a selective communication protocol from task loss alone, without a prescribed format. We demonstrate the mechanism across three tool backends. On arithmetic, coupling two 0.5B models with a calculator raises accuracy from 36\% to 96\%. On logic grid puzzles, coupling two 0.6B models with a Z3 solver achieves $1.7\times$ the unaugmented baseline on ZebraLogic. On mathematical reasoning, coupling with a Python sandbox enables the auxiliary to generate problem-specific code from hidden-state signals alone, without ever seeing the problem text. 

---
# Benchmarking LLM-Based Static Analysis for Secure Smart Contract Development: Reliability, Limitations, and Potential Hybrid Solutions 

**Authors**: Stefan-Claudiu Susan, Andrei Arusoaie, Dorel Lucanu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11163)  

**Abstract**: The irreversible nature of blockchain transactions makes the identification of smart contract vulnerabilities an essential requirement for secure system development. While Large Language Models (LLMs) are increasingly integrated into developer workflows, their reliability as autonomous security auditors remains unproven. We assess whether current generative models are a viable replacement for, or only a complement to, traditional static-analysis tools.
Our findings indicate that LLM efficacy is undermined by both inherent lexical bias and a lack of rigorous validation of external data inputs. This reliance on non-semantic heuristics, such as identifier naming, leads to a high frequency of false positives. Furthermore, prompting techniques reveal a trade-off between precision and recall. These results were derived using our custom automated framework, which achieves 92% accuracy in classifying model outputs. 

---
# Interpretability Can Be Actionable 

**Authors**: Hadas Orgad, Fazl Barez, Tal Haklay, Isabelle Lee, Marius Mosbach, Anja Reusch, Naomi Saphra, Byron Wallace, Sarah Wiegreffe, Eric Wong, Ian Tenney, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2605.11161)  

**Abstract**: Interpretability aims to explain the behavior of deep neural networks. Despite rapid growth, there is mounting concern that much of this work has not translated into practical impact, raising questions about its relevance and utility. This position paper argues that the central missing ingredient is not new methods, but evaluation criteria: interpretability should be evaluated by actionability--the extent to which insights enable concrete decisions and interventions beyond interpretability research itself. We define actionable interpretability along two dimensions--concreteness and validation--and analyze the barriers currently preventing real-world impact. To address these barriers, we identify five domains where interpretability offers unique leverage and present a framework for actionable interpretability with evaluation criteria aligned with practical outcomes. Our goal is not to downplay exploratory research, but to establish actionability as a core objective of interpretability research. 

---
# The Price of Proportional Representation in Temporal Voting 

**Authors**: Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2605.11157)  

**Abstract**: We study proportional representation in the temporal voting model, where collective decisions are made repeatedly over time over a fixed horizon. Prior work has extensively investigated how proportional representation axioms from multiwinner voting (e.g., justified representation (JR) and its variants) can be adapted, satisfied, and verified in this setting. However, much less is understood about their interaction with social welfare. In this work, we quantify the efficiency cost of enforcing proportionality. We formalize the welfare-proportionality tension via the worst-case ratio between the maximum achievable utilitarian welfare and the maximum welfare attainable subject to a proportionality axiom. We show that imposing proportional representation in the temporal setting can incur a growing, yet sublinear, welfare loss as the number of voters or rounds increases. We further identify a clean separation among axioms: for JR, the welfare loss diminishes as the time horizon grows and vanishes asymptotically, whereas for stronger axioms this conflict persists even with many rounds. Moreover, we prove that welfare maximization under each axiom is NP-complete and APX-hard, even under static preferences and bounded-degree approvals, and provide fixed-parameter algorithms under several natural structural parameters. 

---
# Quantifying the Reconstructability of Astrophysical Methods with Large Language Models and Information Theory: A Case Study in Spectral Reconstruction 

**Authors**: Hsing Wen Lin, Zong-Fu Sie  

**Link**: [PDF](https://arxiv.org/pdf/2605.11154)  

**Abstract**: Modern astrophysical studies rely heavily on complex data analysis pipelines; however, published descriptions often lack the detail required for computational reproducibility. In this work, we present an information-theoretic framework to quantify how effectively a method can be reconstructed from its written description. By treating algorithmic reconstruction as a probability distribution generated by Large Language Models (LLMs), we utilize Shannon entropy and Jensen-Shannon divergence to measure how strongly text constrains the hypothesis space of valid implementations. We demonstrate this approach through a case study of Trans-Neptunian Object (TNO) spectral reconstruction from sparse photometry. By prompting frontier LLMs with varying levels of manuscript text (Title, Abstract, and Methods), we find that while increasing text successfully clarifies the overall algorithmic structure, it fails to eliminate variance at the implementation level. This persistent variance establishes an "entropy floor," demonstrating that multiple divergent implementations remain consistent with explicit instructions. To evaluate practical reproducibility, we convert these reconstructed algorithms into executable pipelines. Our results reveal that, while LLMs easily recover core functional methodologies, they systematically fail to infer the tacit expert knowledge required for strict scientific calibration. This pilot study demonstrates that LLMs can be repurposed as a zero-shot diagnostic tool to audit methodological transparency, helping authors identify missing structural constraints and preserve scientific integrity in an era of automated research. 

---
# ClinicalBench: Stress-Testing Assertion-Aware Retrieval for Cross-Admission Clinical QA on MIMIC-IV 

**Authors**: Alex Stinard  

**Link**: [PDF](https://arxiv.org/pdf/2605.11143)  

**Abstract**: Reasoning benchmarks measure clinical performance on clean inputs. We evaluate the step before reasoning: retrieval over real EHR notes, where negation, temporality, and family-versus-patient attribution can flip a correct answer to a wrong one. EpiKG carries an assertion label and a temporality tag with every fact in a patient knowledge graph, then routes retrieval by question intent. ClinicalBench is a 400-question test over 43 MIMIC-IV patients across 9 assertion-sensitive categories. A 7-condition ablation tests each piece of EpiKG across six LLMs (Claude Opus 4.6, GPT-OSS 20B, MedGemma 27B, Gemma 4 31B, MedGemma 1.5 4B, Qwen 3.5 35B). Three physicians blindly adjudicated 100 paired items. The author-blind primary endpoint, leave-author-out paired exact McNemar on 50 unanimous-strict items rated by two external physicians, yields +22.0 percentage points (95 percent Newcombe CI [+5.1, +31.5], p=0.0192). The architectural novelty, intent-aware KG-RAG over a Contriever dense-RAG baseline (C2b to C4g_kw on the change-excluded n=362 endpoint), is +8.84 percentage points (paired McNemar p=1.79e-3); +12.43 percentage points under oracle intent. Sensitivities agree directionally: three-rater physician majority +24.0 percentage points (subject to single-author circularity); deterministic keyword reproducibility proxy +39.5 percentage points. Across the six models, the gain shrinks as the LLM-alone baseline rises (beta=-1.123, r=-0.921, p=0.009). With n=6 this looks more like regression to the mean than encoding substituting for model size. Physician adjudication identified 56 percent of auto-generated reference answers as defective, a methodological finding indicating that NLP-pipeline clinical-QA benchmarks require physician adjudication to be usable. ClinicalBench, the frozen evaluator, three-rater adjudication data, and the EpiKG output stack are publicly released. 

---
# Control Charts for Multi-agent Systems 

**Authors**: Hayden Helm, Carey Priebe, Brandon Duderstadt  

**Link**: [PDF](https://arxiv.org/pdf/2605.11135)  

**Abstract**: Generative agents have proven to be powerful assistants in a wide variety of contexts. Given this success, users are now deploying agents with minimal restrictions in open ended, multi-agent environments. Current methods for monitoring the dynamics of open-ended multi-agent systems are limited to qualitative inspection. In this paper, we extend the process-theoretic notion of adaptive control charts to multi-agent systems to enable automated monitoring. Using simulation, we demonstrate that adaptive control charts are necessary for monitoring multi-agent systems that can learn from their environment. We further demonstrate, both empirically and theoretically, that adaptive control charts are susceptible to adversarial agents that defect sufficiently slowly. These results illustrate a fundamental tradeoff in multi-agent system control: either agents in a system cannot learn or the system is susceptible to adversaries. 

---
# Spurious Correlation Learning in Preference Optimization: Mechanisms, Consequences, and Mitigation via Tie Training 

**Authors**: Christian Moya, Alex Semendinger, Guang Lin, Elliott Thornley  

**Link**: [PDF](https://arxiv.org/pdf/2605.11134)  

**Abstract**: Preference learning methods such as Direct Preference Optimization (DPO) are known to induce reliance on spurious correlations, leading to sycophancy and length bias in today's language models and potentially severe goal misgeneralization in future systems. In this work, we provide a unified theoretical analysis of this phenomenon, characterizing the mechanisms of spurious learning, its consequences on deployment, and a provable mitigation strategy. Focusing on log-linear policies, we show that standard preference-learning objectives induce reliance on spurious features at the population level through two channels: mean spurious bias and causal--spurious correlation leakage. We then show that this reliance creates an irreducible vulnerability to distribution shift: more data from the same training distribution fails to reduce the model's dependence on spurious features. To address this, we propose tie training, a data augmentation strategy using ties (equal-utility preference pairs) to introduce data-driven regularization. We demonstrate that this approach selectively reduces spurious learning without degrading causal learning. Finally, we validate our theory on log-linear models and provide empirical evidence that both the spurious learning mechanisms and the benefits of tie training persist for neural networks and large language models. 

---
# HEPA: A Self-Supervised Horizon-Conditioned Event Predictive Architecture for Time Series 

**Authors**: Jonas Petersen, Gian-Alessandro Lombardi, Riccardo Maggioni, Camilla Mazzoleni, Federico Martelli, Philipp Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2605.11130)  

**Abstract**: Critical events in multivariate time series, from turbine failures to cardiac arrhythmias, demand accurate prediction, yet labeled data is scarce because such events are rare and costly to annotate. We introduce HEPA (Horizon-conditioned Event Predictive Architecture), built on two key principles. First, a causal Transformer encoder is pretrained via a Joint-Embedding Predictive Architecture (JEPA): a horizon-conditioned predictor learns to forecast future representations rather than future values, forcing the encoder to capture predictable temporal dynamics from unlabeled data alone. Second, we freeze the encoder and finetune only the predictor toward the target event, producing a monotonic survival cumulative distribution function (CDF) over horizons. With fixed architecture and optimiser hyperparameters across all benchmarks, HEPA handles water contamination, cyberattack detection, volatility regimes, and eight further event types across 11 domains, exceeding leading time-series architectures including PatchTST, iTransformer, MAE, and Chronos-2 on at least 10 of 14 benchmarks, with an order of magnitude fewer tuned parameters and, on lifecycle datasets, an order of magnitude less labeled data. 

---
# SEVO: Semantic-Enhanced Virtual Observation for Robust VLA Manipulation via Active Illumination and Data-Centric Collection 

**Authors**: Tianchonghui Fang, Yuan Zhuang, Fei Miao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11114)  

**Abstract**: Vision-Language-Action (VLA) and imitation-learning policies trained via community toolchains on low-cost hardware frequently fail when deployed outside the training environment. Existing evaluations, including the original ACT and SmolVLA benchmarks, demonstrate high success rates under controlled, fixed backgrounds, yet community practitioners report near-zero transfer to new environments. We present SEVO (Semantic-Enhanced Virtual Observation), a data-centric approach that improves cross-environment manipulation robustness without modifying the policy architecture. SEVO transforms the raw RGB camera stream through three mechanisms: (1) body-fixed cameras whose combined fields of view cover the full manipulation workspace, (2) active red-spectrum illumination that physically normalizes object appearance, and (3) real-time YOLO segmentation overlay that provides a background-invariant semantic cue. Critically, we show that a diversified data collection protocol (systematically varying lighting, backgrounds, and distractors during teleoperation) is the single most important factor for generalization. We target transparent water bottles, objects that visually blend with their surroundings, and select a simple pick-and-place task to enable hundreds of controlled real-robot trials across two mobile platforms. The full pipeline achieves 95% grasp success with ACT and 83% with SmolVLA in the training environment, transferring to novel environments at 85% and 75%. Without SEVO, the same policies achieve only 75%/70% in training and collapse to 30-35% in novel environments. Our results demonstrate that principled observation design and environmental diversity during data collection, not model scaling, enable low-cost robots to operate reliably in everyday household environments. 

---
# Deploying Self-Supervised Learning for Real Seismic Data Denoising 

**Authors**: Giovanny A. M. Arboleda, Claudio D. T. de Souza, Carlos E. M. dos Anjos, Lessandro de S. S. Valente, Roosevelt de L. Sardinha, Albino Aveleda, Pablo M. Barros, André Bulcão, Alexandre G. Evsukoff  

**Link**: [PDF](https://arxiv.org/pdf/2605.11109)  

**Abstract**: Self-supervised learning (SSL) has emerged as a promising approach to seismic data denoising as it does not require clean reference data. In this work, the deployment of the Noisy-as-Clean (NaC) method was evaluated for real seismic data denoising under controlled conditions. Two independent seismic acquisitions, each comprising noisy and filtered data, were organized into four real datasets. The NaC SSL method was adapted to add real noise to the noisy input, controlled by a parameter. An experimental protocol with ten experiments was designed to compare different strategies for deploying the NaC SSL method with the supervised learning baseline, using identical network topology and hyperparameters. The models were evaluated in terms of denoising performance, computational cost, and generalization capability. The results show that the synthetic additive white Gaussian noise (AWGN) is inadequate for the denoising of seismic data within the NaC method, and performance strongly depends on the compatibility between the injected and actual noise characteristics. Furthermore, both the characteristics of the seismic data and the noise level influence the performance of the model. Self-supervised fine-tuning on test data has improved SSL performance, whereas no such gain was observed for fine-tuning of supervised models. Finally, NaC has shown to be a simple, effective, and model-independent method that offers a feasible solution for the denoising of real seismic data. 

---
# Birds of a Feather Flock Together: Background-Invariant Representations via Linear Structure in VLMs 

**Authors**: Youssef Zaazou, Mark Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2605.11107)  

**Abstract**: Vision-language models (VLMs), such as CLIP and SigLIP 2, are widely used for image classification, yet their vision encoders remain vulnerable to systematic biases that undermine robustness. In particular, correlations between foreground objects and their backgrounds constitute a salient and practically important class of spurious dependencies. In this work, we revisit the well-known property of high linear additivity in VLM embedding spaces and show that it enables a decomposition of scene representations into foreground and background components. Leveraging this insight, we introduce a pre-training approach that exploits this property to construct background-invariant representations using synthetic data. Our method achieves, to our knowledge, the first worst-group accuracy exceeding $90\%$ on Waterbirds under perfect ($100\%$) spurious correlation (i.e., no minority-group examples in the training data). Furthermore, it demonstrates strong sim-to-real transfer and requires no access to real-world debiased data, making it practical for real-world deployment. 

---
# Newton's Lantern: A Reinforcement Learning Framework for Finetuning AC Power Flow Warm Start Models 

**Authors**: Shourya Bose, Helgi Hilmarsson, Dhruv Suri  

**Link**: [PDF](https://arxiv.org/pdf/2605.11102)  

**Abstract**: Neural warm starts can sharply reduce the number of Newton-Raphson iterations required to solve the AC power flow problem, but existing supervised approaches generalize poorly on heavily loaded instances near voltage collapse. We prove a lower bound on the Newton-Raphson iteration count that depends on the direction of the warm start error rather than on its magnitude, and show as a corollary that the bound becomes vacuous as the smallest singular value of the power-flow Jacobian shrinks, identifying the failure mode of supervised regression near the saddle-node bifurcation. Motivated by this analysis, we introduce Newton's Lantern, a finetuning pipeline that combines group relative policy optimization with a learned reward model trained on perturbations of the base model's predictions, using the iteration count itself as the supervisory signal. Across IEEE 118-bus, GOC 500-bus, and GOC 2000-bus benchmarks, Newton's Lantern is the only method that converges on every test snapshot while attaining the smallest mean iteration count. 

---
# Enabling Performant and Flexible Model-Internal Observability for LLM Inference 

**Authors**: Nengneng Yu, Sixian Xiong, Yibo Zhao, Wei Wang, Zaoxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11093)  

**Abstract**: Today's inference-time workloads increasingly depend on timely access to a model's internal states. We present DMI-Lib, a high-speed deep model inspector that treats internal observability as a first-class systems primitive, decoupling it from the inference hot path via an asynchronous observability substrate built from Ring^2, a GPU-CPU memory abstraction for capturing and staging tensors, and a policy-controlled host backend that exports them. DMI-Lib enables the placement of observation points across a rich space of internal signals and diverse inference backends while preserving serving optimizations and adhering to tight GPU memory budgets. Our experiments demonstrate that DMI-Lib incurs only 0.4%--6.8% overhead in offline batch inference and an average of 6% in moderate online serving, reducing latency overhead by 2x-15x compared to existing baselines with similar observability features. DMI-Lib is open-sourced at this https URL. 

---
# ASD-Bench: A Four-Axis Comprehensive Benchmark of AI Models for Autism Spectrum Disorder 

**Authors**: Shubhankit Singh, Hassan Shaikh, Kuldeep Raghuwanshi, Keshav Bulia  

**Link**: [PDF](https://arxiv.org/pdf/2605.11091)  

**Abstract**: Automated ASD screening tools remain limited by single-architecture evaluations, axis-restricted assessment, and near-exclusive focus on adult cohorts, obscuring age-specific diagnostic patterns critical for early intervention. We introduce ASD-Bench, a systematic tabular benchmark evaluating ML, deep learning, and foundation model configurations across three age cohorts (children 1-11 yr, adolescents 12-16 yr, adults 17-64 yr) on four axes: predictive performance, calibration, interpretability, and adversarial robustness. Applied to a curated v3 dataset of 4,068 AQ-10 records, our benchmark spans classical models (XGBoost, AdaBoost, Random Forest, Logistic Regression), neural networks (MLP), deep tabular transformers (TabNet, TabTransformer, FT-Transformer), and TabPFN v2. We introduce the Heuristic Aggregate Penalty (HAP): a cost-sensitive metric penalising false negatives more heavily and incorporating cross-validation variance for deployment stability. Adult classification yields high performance (10/17 models achieve perfect F1 and AUC), while adolescents present a harder task (F1 ceiling 0.837 vs. 0.915 for children). Feature hierarchies shift across cohorts: A9 (social motivation) dominates for children, A5 (pattern recognition) leads for adolescents, and adults exhibit a flatter importance profile consistent with developmental social masking. Accuracy and calibration are dissociated: AdaBoost achieves F1=1.000 on adults with ECE=0.302, confirming single-metric evaluation is insufficient for clinical AI. Cohort-specific deployment recommendations are provided. All findings should be interpreted as proof-of-concept evidence on questionnaire-derived labels rather than clinically validated diagnostic performance. 

---
# ExploitGym: Can AI Agents Turn Security Vulnerabilities into Real Attacks? 

**Authors**: Zhun Wang, Nico Schiller, Hongwei Li, Srijiith Sesha Narayana, Milad Nasr, Nicholas Carlini, Xiangyu Qi, Eric Wallace, Elie Bursztein, Luca Invernizzi, Kurt Thomas, Yan Shoshitaishvili, Wenbo Guo, Jingxuan He, Thorsten Holz, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.11086)  

**Abstract**: AI agents are rapidly gaining capabilities that could significantly reshape cybersecurity, making rigorous evaluation urgent. A critical capability is exploitation: turning a vulnerability, which is not yet an attack, into a concrete security impact, such as unauthorized file access or code execution. Exploitation is a particularly challenging task because it requires low-level program reasoning (e.g., about memory layout), runtime adaptation, and sustained progress over long horizons. Meanwhile, it is inherently dual-use, supporting defensive workflows while lowering the barrier for offense. Despite its importance and diagnostic value, exploitation remains under-evaluated. To address this gap, we introduce ExploitGym, a large-scale, diverse, realistic benchmark on the exploitation capabilities of AI agents. Given a program input that triggers a vulnerability, ExploitGym tasks agents with progressively extending it into a working exploit. The benchmark comprises 898 instances sourced from real-world vulnerabilities across three domains, including userspace programs, Google's V8 JavaScript engine, and the Linux kernel. We vary the security protections applied to each instance, isolating their impact on agent performance. All configurations are packaged in reproducible containerized environments. Our evaluation shows that while exploitation remains challenging, frontier models can successfully exploit a non-trivial fraction of vulnerabilities. For example, the strongest configurations are Anthropic's latest model Claude Mythos Preview and OpenAI's GPT-5.5, which produce working exploits for 157 and 120 instances, respectively. Notably, even with widely used defenses enabled, models retain non-trivial success rates. These results establish ExploitGym as an effective testbed for exploitation and highlight the growing cybersecurity risks posed by increasingly capable AI agents. 

---
# MCPShield: Content-Aware Attack Detection for LLM Agent Tool-Call Traffic 

**Authors**: Sultan Zavrak  

**Link**: [PDF](https://arxiv.org/pdf/2605.11053)  

**Abstract**: The Model Context Protocol (MCP) has become a widely adopted interface for LLM agents to invoke external tools, yet learned monitoring of MCP tool-call traffic remains underexplored. In this article, MCPShield is presented as an attack detection framework for MCP tool-call traffic that encodes each agent session as a graph (tool calls as nodes, sequential and data-flow links as edges), enriches nodes with sentence-embedding features over arguments and responses, and classifies sessions as benign or attacked. Three GNN architectures (GAT, GCN, GraphSAGE), a no-graph MLP, and classical baselines (XGBoost, random forest, logistic regression, linear SVM) are evaluated, with the full architecture comparison conducted on RAS-Eval (task-stratified splits) and GraphSAGE retained as the GNN baseline on ATBench and a combined-source variant (both label-stratified). Three findings emerge. First, content-level features are essential: metadata-only detection plateaus around an AUROC of 0.64 regardless of architecture, while content embeddings push the AUROC above 0.89. Second, naive random-split evaluation inflates AUROC by up to 26 percentage points relative to task-disjoint splits, a memorization confound that prior agent-detection work has not addressed. Third, the detection signal resides primarily in the SBERT content embeddings: an AUROC of 0.975 was reached by tree ensembles on pooled embeddings, performing, for the most part, better than the neural architectures in the primary RAS-Eval setting including GNNs (0.917) and the MLP (0.896), and self-supervised pre-training does not deliver a label-efficiency advantage on this task. 

---
# On Problems of Implicit Context Compression for Software Engineering Agents 

**Authors**: Kirill Gelvan, Igor Slinko, Felix Steinbauer, Egor Bogomolov, Florian Kofler, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.11051)  

**Abstract**: LLM-based Software Engineering agents face a critical bottleneck: context length limitations cause failures on complex, long-horizon tasks. One promising solution is to encode context as continuous embeddings rather than discrete tokens, enabling denser information storage. We apply the recently proposed In-Context Autoencoder for this purpose. While the method performs well on single-shot common-knowledge and code-understanding tasks, our experiments demonstrate that it fails on multi-step agentic coding tasks. In this paper, we explore this phenomenon and discuss possible factors contributing to this failure. 

---
# ForceFlow: Learning to Feel and Act via Contact-Driven Flow Matching 

**Authors**: Shuoheng Zhang, Yifu Yuan, Hongyao Tang, Yan Zheng, Qiaojun Yu, Pengyi Li, Guowei Huang, Helong Huang, Xingyue Quan, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2605.11048)  

**Abstract**: Existing imitation learning methods enable robots to interact autonomously with the physical environment. However, contact-rich manipulation tasks remain a significant challenge due to complex contact dynamics that demand high-precision force feedback and control. Although recent efforts have attempted to integrate force/torque sensing into policies, how to build a simple yet effective framework that achieves robust generalization under multimodal observations remains an open question. In this paper, we propose ForceFlow, a force-aware reactive framework built upon flow matching. For contact-stage policy design, we investigate force signal fusion mechanisms and adopt an asymmetric multimodal fusion architecture that treats force as a global regulatory signal, combined with a joint prediction paradigm that enhances the policy's understanding of instantaneous force and historical information, thereby achieving deep coupling between force and motion. For task-level hierarchical decomposition, we divide manipulation into a vision-dominant approach stage (VLM-based pointing for target localization) and a touch-dominant interaction stage (force-driven contact execution), with a Vision-to-Force (V2F) handover mechanism that explicitly decouples spatial generalization from contact regulation. Experimental results across six real-world contact-rich tasks demonstrate that ForceFlow achieves a 37% success rate improvement over the strong baseline ForceVLA while maintaining significantly lower cost. Moreover, ForceFlow exhibits accurate force signal prediction and demonstrates superior performance in contact force self-regulation and zero-shot out-of-distribution (OOD) generalization. 

---
# Red-Teaming Agent Execution Contexts: Open-World Security Evaluation on OpenClaw 

**Authors**: Hongwei Yao, Yiming Liu, Yiling He, Bingrun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11047)  

**Abstract**: Agentic language-model systems increasingly rely on mutable execution contexts, including files, memory, tools, skills, and auxiliary artifacts, creating security risks beyond explicit user prompts. This paper presents DeepTrap, an automated framework for discovering contextual vulnerabilities in OpenClaw. DeepTrap formulates adversarial context manipulation as a black-box trajectory-level optimization problem that balances risk realization, benign-task preservation, and stealth. It combines risk-conditioned evaluation, multi-objective trajectory scoring, reward-guided beam search, and reflection-based deep probing to identify high-value compromised contexts. We construct a 42-case benchmark spanning six vulnerability classes and seven operational scenarios, and evaluate nine target models using attack and utility grading scores. Results show that contextual compromise can induce substantial unsafe behavior while preserving user-facing task completion, demonstrating that final-response evaluation is insufficient. The findings highlight the need for execution-centric security evaluation of agentic AI systems. Our code is released at: this https URL 

---
# Read, Extract, Classify: A Tool for Smarter Requirements Engineering 

**Authors**: Paheli Bhattacharya, Manojit Chakraborty, Santhosh Kumar Arumugam, Rishabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2605.11045)  

**Abstract**: This paper presents the ReXCL tool, which automates the extraction and classification processes in requirements engineering, enhancing the software development life-cycle. The tool features two main modules: Extraction, which processes raw requirement documents into a predefined schema using heuristics and predictive modeling, and Classification, which assigns class labels to requirements using adaptive fine-tuning of encoder-based models. The final output can be exported to external requirement engineering tools. Performance evaluations indicate that ReXCL significantly improves efficiency and accuracy in managing requirements, marking a novel approach to automating the schematization of semi-structured requirement documents. 

---
# Towards Model-Free Learning in Dynamic Population Games: An Application to Karma Economies 

**Authors**: Matteo Cederle, Saverio Bolognani, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2605.11042)  

**Abstract**: Dynamic Population Games (DPGs) provide a tractable framework for modeling strategic interactions in large populations of self-interested agents, and have been successfully applied to the design of Karma economies, a class of fair non-monetary resource allocation mechanisms. Despite their appealing theoretical properties, existing computational tools for DPGs assume full knowledge of the game model and operate in a centralized fashion, limiting their applicability in realistic settings where agents have access only to their own private experience. This paper takes a step towards addressing this gap by studying model-free equilibrium learning in Karma DPGs. First, we analyze the setting in which a novel agent joins a Karma DPG already at its Stationary Nash Equilibrium (SNE) and learns a policy via Deep Q-Networks (DQN) without knowledge of the game model. Leveraging recent convergence results for DQN, we establish a suboptimality bound consisting of a DQN approximation error of order $O(1/\sqrt{N_s})$ and a mean field perturbation error of order $O(1/N)$, where $N_s$ is the replay buffer size and $N$ is the population size. Second, we consider the challenging problem of learning the SNE from scratch. We show empirically that combining deep RL with fictitious play and smoothed policy iteration allows agents to converge, in a model-free fashion, to a configuration close to the centrally computed SNE. Together, these contributions support the vision of Karma economies as practical tools for fair resource allocation. 

---
# The Granularity Mismatch in Agent Security: Argument-Level Provenance Solves Enforcement and Isolates the LLM Reasoning Bottleneck 

**Authors**: Linfeng Fan, Ziwei Li, Yuan Tian, Yichen Wang, Rongsheng Li, Xiong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11039)  

**Abstract**: Tool-using LLM agents must act on untrusted webpages, emails, files, and API outputs while issuing privileged tool calls. Existing defenses often mediate trust at the granularity of an entire tool invocation, forcing a brittle choice in mixed-trust workflows: allow external content to influence a call and risk hijacked destinations or commands, or quarantine the call and block benign retrieval-then-act behavior. The key observation behind this paper is that indirect prompt injection becomes dangerous not when untrusted content appears in context, but when it determines an authority-bearing argument. We present \textsc{PACT} (\emph{Provenance-Aware Capability Contracts}), a runtime monitor that assigns semantic roles to tool arguments, tracks value provenance across replanning steps, and checks whether each argument's origin satisfies its role-specific trust contract. Under oracle provenance, \textsc{PACT} achieves 100\% utility and 100\% security on mixed-trust diagnostic suites, while flat invocation-level monitors incur false positives or false negatives. In full AgentDojo deployments across five models, \textsc{PACT} reaches 100\% security on the three strongest models while recovering 38.1--46.4\% utility, 8--16 percentage points above CaMeL at the same security level. Ablations show that both semantic roles and cross-step provenance are necessary. \textsc{PACT} reframes agent security as authority binding, and isolates the remaining deployment bottleneck to provenance inference and contract synthesis. 

---
# Sequential Behavioral Watermarking for LLM Agents 

**Authors**: Hyeseon An, Shinwoo Park, Dongsu Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2605.11036)  

**Abstract**: LLM-based agents act through sequences of executable decisions, but their trajectories provide little evidence of which agent or policy produced them, making provenance, ownership, and unauthorized reuse difficult to establish from observed behavior alone. This motivates watermarking signals embedded directly into agent behavior rather than only into generated text, since text watermarking cannot capture the action-level decisions that define agent execution. Recent agent watermarking methods address this gap by moving the watermark from generated text to behavioral choices. However, by treating each action step as an independent trial, they overlook trajectory structure and become fragile when trajectories are perturbed, truncated, or observed without reliable alignment. We propose SeqWM, a sequential behavioral watermarking framework that embeds signals into history-conditioned transition patterns and verifies trajectories position-agnostically against random-key baselines. Experiments across diverse agent benchmarks and LLM backbones show that SeqWM consistently achieves reliable detection while preserving agent utility, and remains robust under trajectory corruption where round-indexed behavioral watermarks collapse. 

---
# MambaNetBurst: Direct Byte-level Network Traffic Classification without Tokenization or Pretraining 

**Authors**: Gayan K. Kulatilleke, Siamak Layeghy, Mahsa Baktashmotlagh, Marius Portmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.11034)  

**Abstract**: We present MambaNetBurst, a compact tokenizer-free byte-level sequence classifier for network burst classification based on a Mamba-2 backbone. In contrast to most recent strong traffic-classification and intrusion-detection approaches, our method operates directly on raw packet bytes, avoids tokenization, patching, and heavy engineered multimodal representations, and does not require any self-supervised pre-training stage. Given a packet flow, we form a fixed-length burst from the first few packets, embed the resulting byte sequence appending a learnable CLS token, and process it with a stack of residual pre-normalized Mamba-2 blocks for end-to-end supervised classification. Across six public benchmarks spanning encrypted mobile app identification, VPN/Tor traffic classification, malware traffic classification, and IoT attack traffic, MambaNetBurst achieves consistently strong results and is competitive with, or outperforms, substantially heavier and often pre-trained baselines. Our ablation study shows that preserving byte-level temporal resolution is critical, that early downsampling through striding is consistently harmful, and that moderate state sizes are sufficient for robust generalization.
We further show that Mamba-2, despite its more constrained transition structure relative to Mamba-1, remains highly effective for packet-byte modeling while providing clear efficiency advantages, particularly in training speed. Overall, our results demonstrate that direct **undiluted** byte-to-classification learning with compact selective state space models is a practical, effective and novel direction for efficient, deployable traffic analysis that bypasses the complexity of pre-training pipelines even over highly optimized linear attention architectures. 

---
# Portable Agent Memory: A Protocol for Cryptographically-Verified Memory Transfer Across Heterogeneous AI Agents 

**Authors**: Santhosh Kumar Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2605.11032)  

**Abstract**: We present Portable Agent Memory, an open protocol and reference implementation for transferring persistent memory state across heterogeneous AI agents. Modern AI agents accumulate rich context -- episodic events,semantic knowledge, procedural skills, working state, and identity preferences -- but this context remains locked within vendor-specific runtimes. Portable Agent Memory addresses this through: (1) a five-component structured memory model with content-addressable entries linked by a Merkle-DAG provenance graph providing tamper-evidence; (2) capability-based access control enabling selective, scoped disclosure of memory segments; (3) an injection-resistant rehydration protocol that adapts recalled content to heterogeneous target models while mitigating indirect prompt injection; and (4) a JSON-first serialization format with optional CBOR compaction for efficient transport. We provide a Python SDK with 54 passing tests, agent skills for multiple platforms, and demonstrate cross-model memory transfer between GPT-4, Claude, Gemini, and Llama architectures. The protocol is open-source under Apache 2.0. 

---
# An Executable Benchmarking Suite for Tool-Using Agents 

**Authors**: Zhiqing Zhong, Zhijing Ye, Jiamin Wang, Xiaodong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11030)  

**Abstract**: Closed-loop tool-using agents are increasingly evaluated in executable web, code, and micro-task environments, but benchmark reports often conflate workloads, action-generating drivers, and the evidence admitted for systems-facing claims. We present an executable benchmarking suite that makes these objects explicit under a shared evidence-admission contract. The suite connects WebArena Verified, a SWE-Gym slice with SWE-bench-compatible verification, and MiniWoB++ through common workload adapters, task manifests, event schemas, replay/freeze policy, declared drivers, and reporting pipelines. In the canonical release, the gate separates paper-facing evidence from preflight, fixture, smoke, and diagnostic rows while preserving non-admitted artifacts for audit and onboarding. The admitted evidence records latency, invalid-action behavior, patch-generation cost, verifier metadata, replay bindings, and provenance under one auditable contract. The gate is decision-relevant rather than merely clerical: in a separate WebArena Verified controller study, clean-baseline and medium live-stressed evaluation select different fixed controller variants under the same workload and admission contract. The release is scoped as a benchmarking suite and admitted evidence, not a new agent policy, model leaderboard, backend comparison, or autonomous SWE-bench solver. 

---
# FragBench: Cross-Session Attacks Hidden in Benign-Looking Fragments 

**Authors**: Astha Mehta, Niruthiha Selvanayagam, Cedric Lam, Hengxu Li, Phuc-Nguyen Nguyen, Raymond Lee, Olivia McGoffin, Luong, Arthur Collé, Jamie Johnson, David Williams-King, Linh Le  

**Link**: [PDF](https://arxiv.org/pdf/2605.11029)  

**Abstract**: An attacker can split a malicious goal into sub-prompts that each look benign on their own and only become harmful in combination. Existing LLM safety benchmarks evaluate prompts one at a time, or across turns of a single chat, and so do not look for a malicious signal spread across separate sessions with no shared context. We build FragBench, a benchmark drawn from 24 real-world cyber-incident campaigns, which keeps the full attack trail: the multi-fragment kill chain, the per-fragment safety-judge verdicts, sandboxed execution traces, and a matched set of benign cover sessions. FragBench splits this trail into two paired tasks: an adversarial rewriter that hardens fragments against a single-turn safety judge (FragBench Attack), and a graph-based user-level detector trained on the resulting interactions (FragBench Defense). The single-turn judge is near chance on the released corpus by construction, but four GNN variants and three classical-ML baselines all recover the cross-session feature, reaching aggregate event-level F1 = 0.88-0.96. Defending against fragmented LLM misuse therefore requires modeling the cross-session interaction graph, rather than isolated prompts. Our generator, rewriter, sandbox harness, and detector are released at this https URL. 

---
# From Code-Centric to Intent-Centric Software Engineering: A Reflexive Thematic Analysis of Generative AI, Agentic Systems, and Engineering Accountability 

**Authors**: Elyson De La Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11027)  

**Abstract**: Generative artificial intelligence (GenAI) and agentic systems are moving software engineering from code-centric production toward intent-centric human-agent work in which natural language, repository context, tools, tests, and governance shape delivery. Prior studies examine code generation, AI pair programming, and software engineering agents, but less is known about how public technical discourse and peer-reviewed evidence together frame the profession's near-term transition. This study addresses that gap through a reflexive thematic analysis (RTA) dominant and interpretative phenomenological analysis (IPA) informed public-discourse and document analysis. The corpus combines peer-reviewed software engineering and AI literature, technical benchmarks, public talks and interviews, essays, product-facing technical announcements, and X-originated discourse from prominent AI and software engineering voices. Sources were organized through a corpus register, codebook, coding matrix, theme-to-source traceability table, DOI/reference audit, and reproducibility protocol. The analysis shows that GenAI lowers the cost of producing plausible code while increasing the importance of intent specification, context curation, architecture knowledge, verification, security, provenance, governance, and accountable human judgment. The findings indicate that software engineering is becoming less about isolated code authorship and more about supervising, validating, and governing socio-technical systems of humans, agents, tools, and evidence gates. This matters because speed-focused adoption can accumulate hidden technical debt and accountability gaps, whereas bounded autonomy can preserve quality, security, maintainability, and trust. 

---
# SCOPE: Siamese Contrastive Operon Pair Embeddings for Functional Sequence Representation and Classification 

**Authors**: Akarsh Gupta, Kenneth Rodrigues, Sagnik Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2605.11022)  

**Abstract**: Identifying operons is a fundamental step in understanding prokaryotic gene regulation, as classifying genes into operons supports the reconstruction of regulatory networks, functional annotation of unannotated genes, and drug candidate development. Experimental approaches such as RT-PCR and RNA-seq provide precise evidence of operon structure, but are laborious and largely limited to well-studied model organisms, making scalable computational methods essential for genome-wide operon identification. Prior computational approaches have employed traditional classifiers such as logistic regression and decision trees, motivating our use of these as physicochemical baselines. The DGEB benchmark evaluates operonic pair classification by embedding each sequence independently with a pre-trained protein language model and computing pairwise cosine similarity. In contrast, our Siamese MLP learns a classifier over the fused embedding space, which is theoretically better motivated for binary classification, as cosine similarity can yield meaningless scores depending on the regularization of the embedding model. While protein language model embeddings substantially outperform physicochemical features in ROC-AUC, a learned Siamese MLP head does not significantly improve over unsupervised cosine similarity in Average Precision, suggesting that the geometry of the embedding space already captures the functional relationships needed for this task. Nonetheless, our Siamese MLP achieves a ROC-AUC of 0.71, competitive with state-of-the-art models on the DGEB leaderboard. These findings indicate that protein language model embeddings are a viable, scalable foundation for operonic pair classification across diverse microbial genomes, with implications for automated genome annotation, regulatory network reconstruction, and characterization of organisms lacking experimental operon annotations. 

---
# Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates 

**Authors**: Anish Diwan, Davide Tateo, Christopher E. Mower, Haitham Bou-Ammar, Jan Peters, Oleg Arenz  

**Link**: [PDF](https://arxiv.org/pdf/2605.11020)  

**Abstract**: Inverse reinforcement learning (IRL) is typically formulated as maximizing entropy subject to matching the distribution of expert trajectories. Classical (dual-ascent) IRL guarantees monotonic performance improvement but requires fully solving an RL problem each iteration to compute dual gradients. More recent adversarial methods avoid this cost at the expense of stability and monotonic dual improvement, by directly optimizing the primal problem and using a discriminator to provide rewards. In this work, we bridge the gap between these approaches by enabling monotonic improvement of the reward function and policy without having to fully solve an RL problem at every iteration. Our key theoretical insight is that a trust-region-optimal policy for a reward function update can be globally optimal for a smaller update in the same direction. This smaller update allows us to explicitly optimize the dual objective while only relying on a local search around the current policy. In doing so, our approach avoids the training instabilities of adversarial methods, offers monotonic performance improvement, and learns a reward function in the traditional sense of IRL--one that can be globally optimized to match expert demonstrations. Our proposed algorithm, Trust Region Inverse Reinforcement Learning (TRIRL), outperforms state-of-the-art imitation learning methods across multiple challenging tasks by a factor of 2.4x in terms of aggregate inter-quartile mean, while recovering reward functions that generalize to system dynamics shifts. 

---
# Efficient LLM Reasoning via Variational Posterior Guidance with Efficiency Awareness 

**Authors**: Zizhao Chen, Yuying Li, Siting Lin, Lianxi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11019)  

**Abstract**: Although large language models rely on chain-of-thought for complex reasoning, the overthinking phenomenon severely degrades inference efficiency. Existing reinforcement learning methods compress reasoning chains by designing elaborate reward functions, which renders high-quality samples extremely sparse in the exploration space and creates a sampling bottleneck for the prior policy. Inspired by cognitive science, we theoretically prove that a posterior distribution guided by reference answers achieves higher expected utility than the prior distribution, thus capable of breaking through the sampling bottleneck of high-quality samples. However, the posterior distribution is unavailable during inference. To this end, we formalize efficient reasoning as a variational inference problem and introduce an efficiency-aware evidence lower bound as the theoretical foundation. Based on this, we propose the VPG-EA framework. It adopts a parameter-shared dual-stream architecture to instantiate both the posterior distribution and the prior policy; after filtering out pseudo-efficient paths via cross-view evaluation, it unidirectionally transfers the posterior's efficient patterns to the prior policy through variational distillation. Experiments on DeepSeek-R1-Distill-Qwen-1.5B and 7B scales demonstrate that VPG-EA improves the comprehensive efficiency metric epsilon cubed by 8.73% and 12.37% over the strongest baselines on each model size, respectively. 

---
# Simpson's Paradox in Behavioral Curves: How Aggregation Distorts Parametric Models of User Dynamics 

**Authors**: Chao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.11017)  

**Abstract**: Behavioral curve modeling -- fitting parametric functions to engagement-versus-exposure data -- is standard practice in recommendation, advertising, and clinical dosing. We show that aggregation introduces a systematic distortion: Simpson's paradox in behavioral curves. On Goodreads (3.3M users, 9 genres), individual users peak at n* approximately 11 exposures while the aggregate peaks at n* approximately 34 -- a 3x gap driven by survival bias. Amazon Electronics (18M reviews) shows a 5.3x distortion. MovieLens-25M (D approximately 1) serves as a negative control, confirming that survival bias -- not aggregation per se -- is the operative mechanism. The distortion is robust to category granularity, engagement operationalization, and classifier calibration. We develop Synthetic Null Calibration to address a 32% false positive rate in per-user classification. Our findings apply wherever individual behavioral parameters are estimated from aggregate curves under differential attrition. 

---
# DCVD: Dual-Channel Cross-Modal Fusion for Joint Vulnerability Detection and Localization 

**Authors**: Wenxin Tang, Wenbin Li, Junliang Liu, Jingyu Xiao, Xi Xiao, Mingzhe Liu, Jinlong Yang, Xuan Liu, Yuehe Ma, Wang Luo, Qing Li, Lei Wang, Peng Xiangli  

**Link**: [PDF](https://arxiv.org/pdf/2605.11015)  

**Abstract**: Software vulnerability detection plays a critical role in ensuring system security, where real-world auditing requires not only determining whether a function is vulnerable but also pinpointing the specific lines responsible. However, existing approaches either rely on a single information source -- sequential, structural, or semantic -- failing to jointly exploit the complementary strengths across modalities, or treat statement-level localization merely as a byproduct of function-level detection without explicit line-level supervision. To address these limitations, we propose DCVD (Dual-Channel Cross-Modal Vulnerability Detection), a unified framework that performs joint function-level detection and statement-level localization. DCVD extracts control-dependency and semantic features through two parallel branches and integrates them via contrastive alignment coupled with bidirectional cross-attention, effectively bridging the cross-modal representation gap. It further introduces explicit supervision signals at both the function and statement levels, enabling collaborative optimization across the two granularities. Extensive experiments on a large-scale real-world vulnerability benchmark demonstrate that DCVD consistently outperforms state-of-the-art methods on both function-level detection and statement-level localization. Our code is available at this https URL. 

---
# Backbone-Equated Diffusion OOD via Sparse Internal Snapshots 

**Authors**: Yadang Alexis Rouzoumka, Jean Pinsolle, Eugénie Terreaux, Christèle Morisseau, Jean-Philippe Ovarlez, Chengfang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2605.11014)  

**Abstract**: Fair comparison between diffusion-based OOD detectors is challenging, as conclusions can vary with backbone choice, corruption parameterization, and test-time budget. We address this issue through a Mutualized Backbone-Equated (MBE) protocol that aligns canonical corruption levels and logical test-time cost across diffusion backbones. Within this setting, we introduce Canonical Feature Snapshots (CFS), a family of detectors that probes a frozen diffusion backbone using only a tiny number of native internal activations at canonical low-noise levels. On a controlled CIFAR-scale benchmark, the strongest one-forward CFS variant is CFS(1x2), while an even smaller decoder-only variant remains highly competitive. This shows that much of the relative-OOD signal exposed by frozen diffusion backbones is concentrated in a small number of sparse internal states, rather than requiring full denoising trajectories or high-capacity downstream heads. We further provide a local diagnostic theory explaining these observations through conditional encoder-decoder complementarity, diagonal-score separation, and low-noise corruption stability. The official implementation is available at this https URL. 

---
# LoopUS: Recasting Pretrained LLMs into Looped Latent Refinement Models 

**Authors**: Taekhyun Park, Yongjae Lee, Dohee Kim, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2605.11011)  

**Abstract**: Looped computation shows promise in improving the reasoning-oriented performance of LLMs by scaling test-time compute. However, existing approaches typically require either training recurrent models from scratch or applying disruptive retrofits, which involve substantial computational costs and may compromise pretrained capabilities. To address these limitations, we introduce \textbf{Looped Depth Up-Scaling} (LoopUS), a post-training framework that converts a standard pretrained LLM into a looped architecture. As a key technical contribution, LoopUS recasts the pretrained LLM into an encoder, a looped reasoning block, and a decoder. It operationalizes this latent-refinement architecture through four core components: (1) block decomposition, guided by staged representation dynamics; (2) an input-dependent selective gate to mitigate hidden-state drift; (3) random deep supervision for memory-efficient learning over long recursive horizons; and (4) a confidence head for adaptive early exiting. Collectively, these mechanisms transform a standard non-looped model into a looped form while stabilizing it against both computational bottlenecks and representation collapse. Through stable latent looping, LoopUS improves reasoning-oriented performance without extending the generated traces or requiring recurrent training from scratch. For more details, see this https URL 

---
# When and How to Canonize: A Generalization Perspective 

**Authors**: Yonatan Sverdlov, Benjamin Friedman, Snir Hordan, Nadav Dym  

**Link**: [PDF](https://arxiv.org/pdf/2605.11008)  

**Abstract**: While invariant architectures are standard for processing symmetric data, there is growing interest in achieving invariance by applying group averaging or canonization to non-invariant backbones. However, the theoretical generalization properties of these alternative strategies remain poorly understood. We introduce a theoretical framework to analyze the generalization error of these methods by bounding their covering numbers. We establish a rigorous generalization hierarchy: the error bounds of canonized models are at best equal to the error bounds of structurally invariant and group-averaged models, and at worst equal to the bounds of non-invariant baselines. Furthermore, we show that there exist optimal canonizations which attain the optimal error bounds, and poor canonizations which attain the non-invariant error bounds, and that this depends on the regularity of the canonization. Finally, applying this framework to permutation groups in point cloud processing, we rigorously prove that the covering number of lexicographical sorting grows exponentially with point cloud dimension, whereas Hilbert curve canonization guarantees polynomial growth. This provides the first formal theoretical justification for the empirical success of Hilbert curve serialization in state-of-the-art point cloud architectures. We conclude with experiments that support our theoretical claims.
Code is available at this https URL 

---
# RT-Transformer: The Transformer Block as a Spherical State Estimator 

**Authors**: Peter Racioppo  

**Link**: [PDF](https://arxiv.org/pdf/2605.11007)  

**Abstract**: We show that the core components of the Transformer block -- attention, residual connections, and normalization -- arise naturally from a single geometric estimation problem. Modeling the latent state as a direction on the hypersphere, with noise defined in the tangent plane at the current estimate, yields a precision-weighted directional inference procedure in which attention aggregates evidence, residual connections implement incremental state updates, and normalization retracts the updated state back onto the hypersphere. Together, these components follow from the geometry of the estimation problem rather than being introduced as independent architectural choices. 

---
# An Execution-Verified Multi-Language Benchmark for Code Semantic Reasoning 

**Authors**: Yikun Li, Jinfeng Jiang, Ting Zhang, Chengran Yang, Chenxing Zhong, Yin Yide, Leow Wen Bin, Eng Lieh Ouh, Lwin Khin Shar, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2605.11006)  

**Abstract**: Evaluating whether large language models (LLMs) can recover execution-relevant program structure, rather than only produce code that passes tests, remains an open problem. Existing code benchmarks emphasize test-passing outputs, from standalone programming tasks (HumanEval, MBPP, LiveCodeBench) to repository repair (SWE-Bench); this is useful, but offers limited diagnostic signal about which program semantics a model can recover from source. We introduce TraceEval, to our knowledge the first execution-verified, multi-language benchmark for code semantic reasoning: recovering a program's runtime call structure from source code. Unlike prior call-graph benchmarks that rely on static-tool output or hand-annotated ground truth, every positive edge in TraceEval is mechanically witnessed by validation execution, eliminating annotator disagreement and label noise for observed behavior. TraceEval consists of (i) 10,583 real-world programs (2,129 test, 8,454 train) extracted from 1,600+ open-source repositories across Python, JavaScript, and Java via an LLM-assisted harness-generation pipeline with tracer validation; and (ii) a reproducible pipeline that converts any open-source repository into new verified benchmark instances. We evaluate 10 LLMs at zero-shot on the held-out test split. The strongest model, Claude-Opus-4.6, reaches an average F1 of 72.9% across the three languages. To demonstrate the train split's utility as a supervision substrate, we fine-tune the Qwen2.5-Coder family on it: lifts of up to +55.6 F1 bring tuned Qwen2.5-Coder-32B to 71.2%, within 1.7 F1 of zero-shot Claude-Opus-4.6. We release the benchmark, pipeline, baselines, and a datasheet at this https URL 

---
# DisagMoE: Computation-Communication overlapped MoE Training via Disaggregated AF-Pipe Parallelism 

**Authors**: Zhichen Zeng, Chi-Chih Chang, Jiayi Wang, Zezhou Wang, Ningxin Zheng, Zheng Zhong, Cesar A. Stuardo, Dongyang Wang, Mohamed S. Abdelfattah, Haibin Lin, Banghua Zhu, Ang Li, Ziheng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.11005)  

**Abstract**: Mixture-of-experts (MoE) architectures enable trillion-parameter LLMs with sparsely activated experts. Expert parallelism (EP) is a widely adopted MoE training strategy, but it suffers from severe all-to-all communication bottlenecks, which is exaggerated by the limited inter-node network bandwidth as the growing model size requires distributing experts across GPU nodes. Prior work focused on overlapping these all-to-all communications with feed-forward network (FFN) and self-attention computations, which often leaves residual network-bound stalls due to inherent imbalance in attention and FFN layers' computation-communication ratios. We present DisagMoE, a disaggregated MoE training system that jointly optimizes model placement and scheduling for maximal efficiency. DisagMoE separates attention and FFN layers into disjoint GPU groups, introduces a multi-stage pipeline with uni-directional, many-to-many communications, and employs a computation-communication roofline model to balance GPU and network bandwidth allocation among the attention and FFN groups. DisagMoE is implemented on Megatron-LM, and evaluation shows that DisagMoE improves training efficiency across multiple MoE models with up to 1.8x speedup on 16-node 8xH800 clusters. 

---
# The Authorization-Execution Gap Is a Major Safety and Security Problem in Open-World Agents 

**Authors**: Baoyuan Wu, Qingshan Liu, Adel Bibi, Irwin King, Siwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2605.11003)  

**Abstract**: This position paper argues that the Authorization-Execution Gap (AEG) is a major safety and security problem in open-world agents. The AEG is the divergence between what a principal intends to authorize and what an open-world agent ultimately executes. Because such agents act autonomously across tools, persistent state, and multi-agent handoffs, even small instances of authorization divergence can cause harm that is difficult or impossible to undo. We argue that many observed agent failures can be traced to three structural sources of AEG: delegation-level incompleteness, channel-level corruption, and composition-level fragmentation. The same observed failure may arise from any of these sources. Without identifying the source, a defense targeting the symptom alone cannot address the underlying cause. Agent safety and security should therefore emphasize source-oriented diagnosis and defense. Because the structural sources of AEG arise dynamically during execution, this approach necessarily requires authorization integrity checks applied during execution, rather than relying solely on one-shot upfront filtering or post-hoc audit. For NeurIPS, the implication is that papers on open-world agents should report not only outcome-level metrics such as task success or attack resistance, but also process-level evidence showing where AEG was detected, constrained, and attributed to a structural source during execution. 

---
# MT-JailBench: A Modular Benchmark for Understanding Multi-Turn Jailbreak Attacks 

**Authors**: Xinkai Zhang, Zhipeng Wei, Huanli Gong, Jing Ting Zheng, Yuchen Zhang, Yue Dong, N. Benjamin Erichson  

**Link**: [PDF](https://arxiv.org/pdf/2605.11002)  

**Abstract**: Multi-turn jailbreaks exploit the ability of large language models to accumulate and act on conversational context. Instead of stating a harmful request directly, an attacker can gradually steer the conversation toward an unsafe answer. Recent methods demonstrate this risk, but they are usually evaluated as black-box pipelines with different budgets, judges, retry rules, and strategy generation procedures. As a result, it is often unclear whether reported gains reflect stronger attack mechanisms or different experimental conditions. We introduce MT-JailBench, a modular evaluation framework for benchmarking multi-turn jailbreaks under fixed conditions. MT-JailBench implements each attack as five interacting modules: evaluation function, attack strategy, prompt generation, prompt refinement, and flow control. This design enables fair comparison across attack methods and component-wise analysis of what drives attack success. Using MT-JailBench, we find that resource budgets and evaluation functions are major confounders: controlling turns, retries, interactions, sampled strategies, and judges substantially change the ranking of attacks. At the component level, prompt generation accounts for most performance variation, while refinement and flow control provide moderate gains. We also find that explicit dynamic strategy generation is not always necessary; stochastic sampling from a fixed strategy can rival more elaborate diversification mechanisms. Finally, recomposing the best components yields a strong attack configuration that outperforms its source attacks and generalizes across diverse target LLMs. MT-JailBench therefore provides a modular framework for comparing multi-turn jailbreaks, understanding the impact of components, and guiding stronger red-teaming evaluations. 

---
# SkillGen: Verified Inference-Time Agent Skill Synthesis 

**Authors**: Yuchen Ma, Yue Huang, Han Bao, Haomin Zhuang, Swadheen Shukla, Michel Galley, Xiangliang Zhang, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2605.10999)  

**Abstract**: Skills are a promising way to improve LLM agent capabilities without retraining, while keeping the added procedure reusable and controllable. However, high-quality skills are still largely written by hand. We introduce SkillGen, a multi-agent framework that synthesizes a single auditable skill from trajectories generated by a base agent. The output is a human-readable artifact that can be inspected before use. Rather than merely summarizing trajectories, SkillGen leverages contrastive induction over both successful and failed trajectories to identify reusable success patterns, recurring failure modes, and behaviors that appear in nearby successes but are missing from failures. SkillGen then generates candidate skills and iteratively refines the skill. A key novelty in SkillGen is that we model agent skills as interventions to empirically verify the net effect of skills on the overall performance. Specifically, we compare outcomes on the same instances with and without the skill, so that we account for both repairs (cases where the skill fixes a baseline failure) and regressions (cases where the skill breaks a baseline success). Across a broad range of agents and datasets, SkillGen consistently improves held-out performance, outperforms existing skill-generation baselines, and produces skills that transfer across models. 

---
# Few-Shot Truly Benign DPO Attack for Jailbreaking LLMs 

**Authors**: Sangyeon Yoon, Wonje Jeung, Yoonjun Cho, Dongjae Jeon, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2605.10998)  

**Abstract**: Fine-tuning APIs make frontier LLMs easy to customize, but they can also weaken safety alignment during fine-tuning. While prior work shows that benign supervised fine-tuning (SFT) can reduce refusal behavior, deployed fine-tuning pipelines increasingly support preference-based objectives, whose safety risks remain less understood. We show that Direct Preference Optimization (DPO) introduces a stronger and harder-to-audit failure mode. We propose a truly benign DPO attack using only 10 harmless preference pairs, the minimum data scale accepted by OpenAI's fine-tuning service. Each pair contains a benign prompt, a normal helpful answer as the preferred response, and a refusal as the dispreferred response. Unlike prior benign fine-tuning attacks, our data exhibits no suspicious behavior: it is practically indistinguishable from the fine-tuning request of a legitimate user seeking to reduce over-refusal, making harmful intent almost impossible to infer from the request alone. Nevertheless, because DPO directly optimizes the model to prefer helpful answers over refusals, this seemingly benign objective broadly suppresses refusal behavior and transfers to harmful prompts outside the fine-tuning data. Across OpenAI models supporting DPO fine-tuning, our attack achieves attack success rates of 59.13% on GPT-4o, 70.20% on GPT-4.1, 54.80% on GPT-4.1-mini, and 81.73% on GPT-4.1-nano, at costs of only \$1.7, \$1.7, \$0.3, and \$0.1. Moreover, on open-weight models that do not impose minimum data requirements, we find that this effect can emerge from even a single benign preference pair. 

---
# Towards Scalable Persistence-Based Topological Optimization 

**Authors**: Abderrahim Bendahi, Alexandre Duplessis, Arnaud Fickinger  

**Link**: [PDF](https://arxiv.org/pdf/2605.10996)  

**Abstract**: Persistence-based topological optimization deforms a point cloud $X \subset \mathbb{R}^d$ by minimizing objectives of the form $L(X) = \ell(\mathrm{Dgm}(X))$, where $\mathrm{Dgm}(X)$ is a persistence diagram. In practice, optimization is limited by two coupled issues: persistent homology is typically computed on subsamples, and the resulting topological gradients are highly sparse, with only a few anchor points receiving nonzero updates. Motivated by diffeomorphic interpolation, which extends sparse gradients to smooth ambient vector fields via Reproducing Kernel Hilbert Space (RKHS) interpolation, we propose a more scalable pipeline that improves both subsampling and gradient extension. We introduce subsampling via random slicing, a lightweight scheme that promotes iteration-wise geometric coverage and mitigates density bias. We further replace the costly kernel solve with a fast Nadaraya-Watson (NW) Gaussian convolution, producing a globally defined smooth update field at a fraction of the computational cost, while being more suited for topological optimization tasks. We provide theoretical guarantees for NW smoothing, including anchor approximation bounds and global Lipschitz estimates. Experiments in $2$D and $3$D show that combining random slicing with NW smoothing yields consistent speedups and improved objective values over other baselines on common persistence losses. 

---
# Test-Time Personalization: A Diagnostic Framework and Probabilistic Fix for Scaling Failures 

**Authors**: Linhai Zhang, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2605.10991)  

**Abstract**: Existing approaches to LLM personalization focus on constructing better personalized models or inputs, while treating inference as a single-shot process. In this work, we study Test-Time Personalization (TTP) along an unexplored axis: scaling inference-time computation by sampling N candidates from a personalized policy model and selecting the best with a personalized reward model. We prove that oracle selection yields expected utility growing logarithmically with the number of sampled candidates, establishing a theoretical ceiling for test-time scaling. However, standard reward models fail to realize this potential. To diagnose why, we derive a unified scaling law that decomposes any reward model's Best-of-N curve into four measurable quantities and reveals two failure modes, user-level collapse (near-constant prediction for some users) and query-level reward hacking (negative correlation with true quality for some queries). Guided by this law, we propose a probabilistic personalized reward model whose learned variance effectively mitigates both failure modes. Experiments confirm both elements of our framework: TTP delivers consistent scaling across multiple policy models and personalized text generation tasks, and our scaling law closely matches observed scaling curves across reward-model variants. 

---
# Skill Drift Is Contract Violation: Proactive Maintenance for LLM Agent Skill Libraries 

**Authors**: Linfeng Fan, Yuan Tian, Ziwei Li, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2605.10990)  

**Abstract**: LLM agents increasingly rely on reusable skill libraries, but these skills silently decay as the external services, packages, APIs, and configurations they reference evolve. Existing monitors detect such changes at the wrong granularity: they observe values, not the role those values play in a skill. A version string in a comment is noise; the same string in a pinned dependency is an operational obligation. We formulate skill drift as contract violation and introduce \sgname{}, which extracts executable environment contracts from skill documents and validates only those role-bearing assumptions against known or live conditions. This distinction turns noisy monitoring into a precision-first maintenance signal. Contract-free CI probes produce 40\% false positives, while \sgname{} raises zero false alarms over 599 no-drift and hard-negative cases (Wilson 95\% CI $[0,0.6]\%$). In known-drift verification, \sgname{} achieves 100\% precision and 76\% recall with the strongest backbone; in a pre-registered study over 49 real skills, it discovers live drift with 86\% conservative precision. Violated contracts also make repair actionable, improving one-round success from 10\% without localization to 78\%. We release \dbname{}, an 880-pair benchmark for skill degradation. 

---
# SURGE: Surrogate Gradient Adaptation in Binary Neural Networks 

**Authors**: Haoyu Huang, Boyu Liu, Linlin Yang, Yanjing Li, Yuguang Yang, Xuhui Liu, Canyu Chen, Zhongqian Fu, Baochang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10989)  

**Abstract**: The training of Binary Neural Networks (BNNs) is fundamentally based on gradient approximation for non-differentiable binarization operations (e.g., sign function). However, prevailing methods including the Straight-Through Estimator (STE) and its improved variants, rely on hand-crafted designs that suffer from gradient mismatch problem and information loss induced by fixed-range gradient clipping. To address this, we propose SURrogate GradiEnt Adaptation (SURGE), a novel learnable gradient compensation framework with theoretical grounding. SURGE mitigates gradient mismatch through auxiliary backpropagation. Specifically, we design a Dual-Path Gradient Compensator (DPGC) that constructs a parallel full-precision auxiliary branch for each binarized layer, decoupling gradient flow via output decomposition during backpropagation. DPGC enables bias-reduced gradient estimation by leveraging the full-precision branch to estimate components beyond STE's first-order approximation. To further enhance training stability, we introduce an Adaptive Gradient Scaler (AGS) based on an optimal scale factor to dynamically balance inter-branch gradient contributions via norm-based scaling. Experiments on image classification, object detection, and language understanding tasks demonstrate that SURGE performs best over state-of-the-art methods. 

---
# Seeing the Needle in the Haystack: Towards Weakly-Supervised Log Instance Anomaly Localization via Counterfactual Perturbation 

**Authors**: Yutszyuk Wong, Wentai Wu, Yuen-Ying Yeung, Weiwei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.10988)  

**Abstract**: Log anomaly detection is a critical task for system operations and security assurance. However, in networked systems at scale, log data are generated at massive scale while instance-level annotations are prohibitively expensive, posing great difficulties to fine-grained anomaly localization. To address this challenge, we propose LogMILP (Log anomaly localization based on Multi-Instance Learning enhanced by prototypes and Perturbation), a weakly supervised framework that enables both bag-level anomaly detection and instance-level anomaly localization using only bag-level labels. Our method guides the model to pinpoint the critical log entries using prototype-guided structural modeling with counterfactual perturbation consistency regularization, thereby improving localization reliability and interpretability under coarse-grained supervision. Experimental results on three public datasets demonstrate that LogMILP achieves competitive detection performance while yielding significantly more reliable instance-level localization. Our code is open-sourced at this https URL. 

---
# AESOP: Adversarial Execution-path Selection to Overload Deep Learning Pipelines 

**Authors**: Tingxi Li, Mingfang Ji, Ravishka Shemal Rathnasuriya, Simin Chen, Yitao Hu, Wei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10987)  

**Abstract**: Modern machine learning deployments increasingly compose specialized models into dynamic inference pipelines, where upstream components produce intermediate predictions that determine the workload and inputs of downstream components. The cost of processing an input is therefore not determined by any single model, but by two coupled factors: the per-inference cost of each invoked component and its workload volume. Because these pipelines run under hard real-time constraints, efficiency is a fundamental requirement for system availability. We show that this structure creates an efficiency-attack surface that existing methods targeting single models cannot exploit: on identical inputs and budgets, path-aware targeting inflates FLOPs by $2,407\times$ while the strongest single-model baseline achieves $117\times$ -- a $20\times$ gap attributable entirely to where the attack is directed. We formalize this as the adversarial path-selection problem and present AESOP, a framework combining vulnerability-guided path ranking with adaptive loss weighting. We evaluate AESOP on five pipelines plus a production-realistic deployment variant with batching, bounded buffering, and confidence-threshold defenses. AESOP achieves up to $2,407\times$ FLOPs and $419\times$ latency inflation in white-box setting and 58$\times$ FLOPs / 17$\times$ latency in gray-box settings. Under system-level defenses, the attack is not neutralized but redirected: pipelines are forced to choose between throughput collapse ($0.578 \to 0.006$ input/s) and $96.7\%$ data loss to sustain throughput. 

---
# Structural Interpretations of Protein Language Model Representations via Differentiable Graph Partitioning 

**Authors**: Siddhant Dutta, Edward Tan Beng Wai, Soumick Sarker, Pasan Gunawardane, Jagath C. Rajapakse  

**Link**: [PDF](https://arxiv.org/pdf/2605.10985)  

**Abstract**: Protein language models such as ESM-2 learn rich residue representations that achieve strong performance on protein function prediction, but their features remain difficult to interpret as structural $\&$ evolutionary signals are encoded in dense latent spaces. We propose a plug-$\&$-play framework that projects ESM-2 representations onto protein contact graphs $\&$ applies $\textbf{SoftBlobGIN}$, a lightweight Graph Isomorphism Network with differentiable Gumbel-softmax substructure pooling, to perform structure-aware message passing $\&$ learn coarse functional substructures for downstream prediction tasks. Across enzyme classification, SoftBlobGIN achieves 92.8\% accuracy $\&$ 0.898 macro-F1. Unlike post hoc analysis of protein language models alone, our method produces directly auditable structural explanations: GNNExplainer recovers biologically meaningful active-site residues, spatially localized functional clusters, $\&$ catalytic contact patterns. On binding-site detection, SoftBlobGIN improves residue AUROC from $0.885$ using an ESM-2 linear probe to $0.983$, indicating that these structural explanations are not recoverable from language-model features alone. Learned blob partitions provide an additional layer of interpretability by automatically grouping residues into functional substructures, with blobs containing annotated active-site residues showing $1.85\times$ higher importance than other blobs ($\rho{=}0.339$, $p{=}0.009$), without any active-site supervision. Our framework requires no retraining of the language model, adds only $\sim$1.1M parameters, $\&$ generalises across ProteinShake tasks, achieving $F_{\max}$ of $0.733$ on Gene Ontology prediction $\&$ AUROC of $0.969$ on binding-site detection. We position this as an interpretable structural companion to protein language models that makes their predictions more transparent $\&$ auditable. 

---
# TMPO: Trajectory Matching Policy Optimization for Diverse and Efficient Diffusion Alignment 

**Authors**: Jiaming Li, Chenyu Zhu, Zhiyuan Ma, Nanxi Yi, Youjun Bao, Li Sun, Quanying Lv, Xiang Fang, Daizong Liu, Jianjun Li, Kun He, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2605.10983)  

**Abstract**: Reinforcement learning (RL) has shown extraordinary potential in aligning diffusion models to downstream tasks, yet most of them still suffer from significant reward hacking, which degrades generative diversity and quality by inducing visual mode collapse and amplifying unreliable rewards. We identify the root cause as the mode-seeking nature of these methods, which maximize expected reward without effectively constraining probability distribution over acceptable trajectories, causing concentration on a few high-reward paths. In contrast, we propose Trajectory Matching Policy Optimization (TMPO), which replaces scalar reward maximization with trajectory-level reward distribution matching. Specifically, TMPO introduces a Softmax Trajectory Balance (Softmax-TB) objective to match the policy probabilities of K trajectories to a reward-induced Boltzmann distribution. We prove that this objective inherits the mode-covering property of forward KL divergence, preserving coverage over all acceptable trajectories while optimizing reward. To further reduce multi-trajectory training time on large-scale flow-matching models, TMPO incorporates Dynamic Stochastic Tree Sampling, where trajectories share denoising prefixes and branch at dynamically scheduled steps, reducing redundant computation while improving training effectiveness. Extensive results across diverse alignment tasks such as human preference, compositional generation and text rendering show that TMPO improves generative diversity over state-of-the-art methods by 9.1%, and achieves competitive performance in all downstream and efficiency metrics, attaining the optimal trade-off between reward and diversity. 

---
# $ξ$-DPO: Direct Preference Optimization via Ratio Reward Margin 

**Authors**: Zhengyuan Fan, Zhonghua Wu, Yuxuan Du, Qun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.10981)  

**Abstract**: Reference-free preference optimization has emerged as an efficient alternative to reinforcement learning from human feedback, with Simple Preference Optimization(SimPO) demonstrating strong performance by eliminating the explicit reference model through a simple objective. However, the joint tuning of the hyperparameters $\beta$ and $\gamma$ in SimPO remains a central challenge. We argue that this difficulty arises because the margin formulation in SimPO is not easily interpretable across datasets with different reward gap structures. To better understand this issue, we conduct a comprehensive analysis of SimPO and find that $\beta$ implicitly controls sample filtering, while the effect of $\gamma$ depends on the reward gap structure of the dataset. Motivated by these observations, we propose $\xi$-DPO: Direct preference optimization via ratio reward margin. We first reformulate the preference objective through an equivalent transformation, changing the optimization target from maximizing the likelihood of reward gaps to minimizing the distance between reward gaps and optimal margins. Then, we redefine the reward in a ratio form between the chosen and rejected, which effectively cancels the effect of $\beta$ and yields a bounded and interpretable margin. This margin is called the ratio reward margin and is denoted by $\xi$. Unlike the margin $\gamma$ in SimPO, $\xi$ explicitly represents the desired relative separation between chosen and rejected responses and can be determined from the initial reward gap distribution, avoiding repeated trial-and-error tuning. .... 

---
# LEAP: Unlocking dLLM Parallelism via Lookahead Early-Convergence Token Detection 

**Authors**: Haohui Zhang, Zhiye Wang, Xiaoying Gan, Xinbing Wang, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10980)  

**Abstract**: Diffusion Language Models (dLLMs) have garnered significant attention for their potential in highly parallel processing. The parallel capabilities of existing dLLMs stem from the assumption of conditional independence at high confidence levels, which ensures negligible discrepancy between the marginal and joint distributions. However, the stringent confidence thresholds required to preserve accuracy severely constrain the scalability of parallelism. Through systematic token-level statistical analysis, we reveal that a substantial proportion of tokens converge to their correct predictions early in the denoising process yet fail to reach standard confidence thresholds, confirming that current confidence-based criteria are overly conservative. In response, we introduce LEAP (Lookahead Early-Convergence Token Detection for Accelerated Parallel Decoding). LEAP is a training-free, plug-and-play method that leverages future context filtering and multi-sequence superposition to detect early-converging tokens. By validating the alignment between early convergence and correctness, we enable reliable early decoding of these tokens. Benchmarking across diverse domains demonstrates that LEAP significantly lowers inference latency and decoding steps. Compared to confidence-based decoding, the average number of denoising steps is reduced by about 30%. On the GSM8K dataset, combining LEAP with dParallel accelerates decoding to 7.2 tokens per step while preserving model precision. LEAP effectively breaks the reliance on high-confidence priors, offering a novel paradigm for parallel decoding. 

---
# PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks 

**Authors**: Zhenxin Ai, Haiyun He  

**Link**: [PDF](https://arxiv.org/pdf/2605.10977)  

**Abstract**: Watermarking for large language models (LLMs) is a promising approach for detecting LLM-generated text and enabling responsible deployment. However, existing watermarking methods are often vulnerable to semantic-invariant attacks, such as paraphrasing. We propose PASA, a principled, robust, and distortion-free watermarking algorithm that embeds and detects a watermark at the semantic level. PASA operates on semantic clusters in a latent embedding space and constructs a distributional dependency between token and auxiliary sequences via shared randomness synchronized by a secret key and semantic history. This design is grounded in our theoretical framework that characterizes a jointly optimal embedding-detection pair, achieving the fundamental trade-offs among detection accuracy, robustness, and distortion. Evaluations across multiple LLMs and semantic-invariant attacks demonstrate that PASA remains robust even under strong paraphrasing attacks while preserving high text quality, outperforming standard vocabulary-space baselines. Ablation studies further validate the effectiveness of our hyperparameter choices. Webpage: this https URL. 

---
# Hierarchical Multi-Scale Graph Neural Networks: Scalable Heterophilous Learning with Oversmoothing and Oversquashing Mitigation 

**Authors**: Md Sazzad Hossen, Avimanyu Sahoo  

**Link**: [PDF](https://arxiv.org/pdf/2605.10975)  

**Abstract**: Graphs with heterophily, where adjacent nodes carry different labels, are prevalent in real-world applications, from social networks to molecular interactions. However, existing spectral Graph Neural Network (GNN) approaches tailored for heterophilous graph classification suffer from hub-dominated (node with large degree) aggregation and oversmoothing, as their suboptimal polynomial filters introduce approximation errors and blend distant signals. To address the degree-biased aggregation and suboptimal polynomial filtering, we introduce a Hierarchical Multi-view HAAR (HMH), a novel spectral graph-learning framework that scales in near-linear time . HMH first learns feature- and structure-aware signed affinities via a heterophily-aware encoder, then constructs a soft graph hierarchy guided by these embeddings. At each hierarchical level, HMH constructs a sparse, orthonormal, and locality-aware Haar basis to apply learnable spectral filters in the frequency domain. Finally, skip-connection unpooling layers combine outputs from all hierarchical levels back into the original graph, effectively preventing hub domination and long-range signal bottleneck (over-squashing). Experimentation shows that HMH outperforms state-of-the-art spectral baselines, achieving up to a 3% improvement on node classification and 7% points on graph classification datasets, all while maintaining linear scalability. 

---
# Vertex-Softmax: Tight Transformer Verification via Exact Softmax Optimization 

**Authors**: Navid Rezazadeh, Arash Gholami Davoodi  

**Link**: [PDF](https://arxiv.org/pdf/2605.10974)  

**Abstract**: Certified verification of transformer attention requires bounding the softmax function over interval constraints on the pre-softmax scores. Existing verifiers relax softmax ndependently of the downstream objective, leaving avoidable slack. We prove that the exact optimum of this score-box problem is attained at a vertex of the constraint box, and establish a threshold structure theorem showing that, after sorting the objective coefficients, the optimum lies among only linearly many candidates, yielding the Vertex-Softmax primitive with log-linear complexity in the sequence length. We further prove a formal optimality result showing that Vertex-Softmax is the tightest sound bound obtainable from score intervals alone, characterizing precisely what additional structure (score correlations, score-value coupling) is needed for further improvement. Integrated into a CROWN Convex Relaxation based Optimization for Worst-case Neurons)-style verifier with a formal soundness guarantee, Vertex-Softmax significantly improves certified rates and substantially tightens lower bounds across MNIST, Fashion-MNIST, and CIFAR-10 attention models, while consistently matching or outperforming alpha-CROWN and branch-and-bound baselines at a fraction of their cost. 

---
# Rotation-Preserving Supervised Fine-Tuning 

**Authors**: Hangzhan Jin, Tianwei Ni, Lu Li, Pierre-Luc Bacon, Mohammad Hamdaqa, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2605.10973)  

**Abstract**: Supervised fine-tuning (SFT) improves in-domain performance but can degrade out-of-domain (OOD) generalization. Prior work suggests that this degradation is related to changes in dominant singular subspaces of pretrained weight matrices. However, directly identifying loss-sensitive directions with Hessian or Fisher information is computationally expensive at LLM scale. In this work, we propose preserving projected rotations in pretrained singular subspaces as an efficient proxy for Fisher-sensitive directions, which we call Rotation-Preserving Supervised Fine-Tuning (RPSFT). RPSFT penalizes changes in the projected top-$k$ singular-vector block of each pretrained weight matrix, limiting unnecessary rotation while preserving task adaptation. Across model families and sizes trained on math reasoning data, RPSFT improves the in-domain/OOD trade-off over standard SFT and strong SFT baselines, better preserves pretrained representations, and provides stronger initializations for downstream RL fine-tuning. Code is available at \href{this https URL}{this https URL}. 

---
# Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models 

**Authors**: Hanhan Zhou, Shamik Roy, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.10971)  

**Abstract**: Discrete diffusion language models (DLMs) generate text by iteratively denoising all positions in parallel, offering an alternative to autoregressive models. Controlled generation methods for DLMs, imported from autoregressive models, apply uniform intervention at every denoising steps. We show this uniform schedule degrades quality, and the damage compounds when multiple attributes are steered jointly. To diagnose the failure, we train sparse autoencoders on four DLMs (124M-8B parameters) and find that different attributes commit on distinct schedules, varying in timing, sharpness, and magnitude. For instance, topic commits within the first 2\% of denoising, whereas sentiment emerges gradually over 20\% of the process. Consequently, uniform intervention wastes steering capacity on steps where the target attribute has already solidified or has yet to emerge. We propose a novel adaptive scheduler that concentrates interventions on the steps where an attribute is actively forming and leaves the rest of generation untouched. The cost-control trade-off admits a closed-form characterization: the advantage of adaptive over uniform scheduling is governed by a single dispersion statistic of the commitment distribution. Across four DLMs and seven steering tasks, our method achieves precise control without the degradation typical of uniform interventions. Especially on challenging simultaneous three-attribute control, it reaches up to 93\% steering strength, beating the strongest baseline by up to 15\% points while preserving generation quality. 

---
# Context-Gated Associative Retrieval: From Theory to Transformers 

**Authors**: Moulik Choraria, Argyrios Gerogiannis, Vidhata Jayaraman, Ankur Mani, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2605.10970)  

**Abstract**: Hopfield networks and their generalizations have established deep connections among biological associative memories, statistical physics, and transformers. Yet most models treat retrieval as a fixed query-to-memory mapping, ignoring the role of external context in recall. In this work, we propose a two-stage associative memory architecture, wherein a context-gate subcircuit reshapes the retrieval energy landscape before and during recall. We show theoretically that context gating increases inter-memory separation while inducing sparsity, translating into exponential improvements in retrieval. Crucially, we prove that the system admits a unique self-consistent fixed point, revealing that the resulting retrieval state is driven by both a direct contextual bias and a second-order retrieval-gate feedback loop. We then bridge this theory to transformers; specifically, we evaluate a first-order approximation on Llama-3, confirming that in-context learning acts as context-gated retrieval. Native dynamics mirror our theory: context localizes a memory subspace, enabling the zero-shot query to cleanly discriminate. Ultimately, this framework provides a mechanistic link between associative memory theory and LLM phenomenology. 

---
# MMTB: Evaluating Terminal Agents on Multimedia-File Tasks 

**Authors**: Chiyeong Heo, Jaechang Kim, Junhyuk Kwon, Hoyoung Kim, Dongmin Park, Jonghyun Lee, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2605.10966)  

**Abstract**: Terminals provide a powerful interface for AI agents by exposing diverse tools for automating complex workflows, yet existing terminal-agent benchmarks largely focus on tasks grounded in text, code, and structured files. However, many real-world workflows require practitioners to work directly with audio and video files. Working with such multimedia files calls for terminal agents not only to understand multimedia content, but also to convert auditory and visual evidence across related files into appropriate actions. To evaluate terminal agents on multimedia-file tasks, we introduce MultiMedia-TerminalBench (MMTB), a benchmark of 105 tasks across 5 meta-categories where terminal agents directly operate with audio and video files. Alongside MMTB, we propose Terminus-MM, a multimedia harness that extends Terminus-KIRA with audio and video perception for terminal agents. Together, MMTB and Terminus-MM support a controlled study of multimedia terminal agents, revealing how different forms of multimedia access shape task outcomes and determine which evidence agents rely on to construct executable terminal workflows. MMTB media and metadata are released at this https URL 

---
# Two Hebrew folk meteorological proverbs tested: rainfall on Rosh Chodesh and Shabbat Mevarechim as predictors of monthly precipitation (Israel, 1950-2024) 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2605.10960)  

**Abstract**: Folk meteorological proverbs encode centuries of empirical observation by agricultural communities. Two Hebrew proverbs link lunar calendar anchor days to monthly winter rainfall: (i) "If Rosh Chodesh is rainy, the whole month is rainy" and (ii) "If it rains on Shabbat Mevarechim, the whole month is rainy." Shabbat Mevarechim is the last Saturday before each new Hebrew month, preceding Rosh Chodesh by one to seven days. The first proverb is widely known; the second circulates in Hasidic oral tradition with no identified written source. Both have never been formally tested.
We analyse 75 years (1950-2024) of daily precipitation data from seven Israeli cities across three climatic regions, comprising 191,758 station-days and 2,422 Hebrew-month observations during the winter rainy season (Marcheshvan-Adar). A rainy Rosh Chodesh increases the probability of a rainy month from 22.2% to 38.6% (lift +16.4 percentage points; chi-square = 57.8, p = 2.9e-14; Bayes factor 1.81). A rainy Shabbat Mevarechim produces a similar effect (lift +16.5 percentage points, p = 8.0e-13), despite preceding Rosh Chodesh by up to seven days.
The effect decays with lag and mirrors daily rainfall autocorrelation (r = 0.35-0.44 at lag 1; ~0 at lag 7), consistent with Mediterranean cyclone persistence. A bootstrap permutation test (p < 1e-4) and a 15-year rolling analysis show declining predictive power (-0.20 percentage points per year, p < 0.001), consistent with shortening precipitation events under warming climate conditions.
Both proverbs encode real but probabilistic meteorological signals whose reliability is decreasing over time. 

---
# QuIDE: Mastering the Quantized Intelligence Trade-off via Active Optimization 

**Authors**: Xiantao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.10959)  

**Abstract**: There is currently no unified metric for evaluating the efficiency of quantized neural networks. We propose QuIDE, built around the Intelligence Index I = (C x P)/log_2(T+1), which collapses the compression-accuracy-latency trade-off into a single score. Experiments across six settings -- SimpleCNN (MNIST, CIFAR), ResNet-18 (ImageNet-1K), and Llama-3-8B -- show a task-dependent Pareto Knee. 4-bit quantization is optimal for MNIST and large LLMs, while 8-bit is the sweet spot for complex CNN tasks (ResNet-18 on ImageNet), where 4-bit PTQ collapses accuracy catastrophically. The accuracy-gated variant I' correctly flags these non-viable configurations that the raw I would reward. QuIDE provides a reproducible evaluation protocol and a ready-to-use fitness function for mixed-precision search. 

---
# Multi-Fidelity Emulation of Atmospheric Correction Coefficients with Physics-Guided Kolmogorov-Arnold Networks 

**Authors**: Md Abdullah Al Mazid, Naphtali Rishe  

**Link**: [PDF](https://arxiv.org/pdf/2605.10958)  

**Abstract**: Atmospheric correction is a critical preprocessing step in optical remote sensing, but repeated high-fidelity radiative transfer simulations remain computationally expensive for dense look-up-table generation, sensitivity analysis, retrieval support, and operational preprocessing. This study presents a physics-aware multi-fidelity surrogate framework for emulating atmospheric correction coefficients using paired 6S and libRadtran simulations. Atmospheric and geometric states are sampled using Latin Hypercube Sampling, and both radiative transfer models are evaluated under matched conditions for Sentinel-2 bands using spectral-response-function-aware coefficient generation. The high-fidelity targets are path reflectance, total transmittance, and spherical albedo. A physics-guided Kolmogorov-Arnold Network, termed pKANrtm, receives the atmospheric state and low-fidelity 6S coefficients, predicts the residual relative to libRadtran, and reconstructs the high-fidelity coefficients. The pKANrtm model uses an Efficient-KAN architecture and is trained with a physics-consistency penalty applied in the original coefficient space. The proposed model is evaluated against state-of-the-art regression-based RTM surrogates. Across both standard and out-of-distribution evaluation settings, pKANrtm achieves the strongest overall predictive performance among the compared models. Runtime benchmarking demonstrates substantial acceleration relative to libRadtran, with GPU inference providing approximately four orders of magnitude single-sample speedup and batched inference reaching tens of thousands of samples per second. These results indicate that physics-aware multi-fidelity pKANrtm emulation provides an accurate, physically structured, and computationally efficient strategy for atmospheric correction coefficient generation. 

---
# Controlled Steering-Based State Preparation for Adversarial-Robust Quantum Machine Learning 

**Authors**: Sahan Sanjaya, Hari Krishna Parvatham, Emma Andrews, Prabhat Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2605.10954)  

**Abstract**: Quantum machine learning (QML) provides a promising framework for leveraging quantum-mechanical effects in learning tasks. However, its vulnerability to adversarial perturbations remains a major challenge for practical deployment. In QML systems, small perturbations applied to classical inputs can propagate through the quantum encoding stage and distort the resulting quantum state, thereby degrading model performance. In this work, we propose a defense mechanism that replaces the conventional quantum encoding stage of a QML model with passive steering-based controlled state preparation, which guides the encoded state toward a controlled intermediate state. By tuning the steering strength and the number of steering iterations, the proposed method suppresses the influence of adversarial perturbations while maintaining high clean accuracy and improving adversarial accuracy. Experimental results demonstrate that the passive steering-based defense consistently improves adversarial accuracy across different QML models and datasets under gradient-based adversarial attacks, achieving adversarial accuracy improvements of up to 40.19%. 

---
# Continuous Flood Nowcasting in South Asia: A Multi-Sensor Ensemble Remote Sensing Framework for Flood Extent 

**Authors**: Usman Nazir, Disha Gomathinayagam, Muhammad Kamran, Sara Khalid  

**Link**: [PDF](https://arxiv.org/pdf/2605.10950)  

**Abstract**: Pakistan experienced an unusually severe flood season between June and December 2025, with cascading impacts on population, infrastructure, and agriculture. Existing operational flood products (e.g., UNOSAT) provide valuable episode-level snapshots but rarely deliver spatially and temporally continuous inundation maps at near-real-time latency within the country. We present a multi-sensor, ensemble-based remote-sensing framework for continuous flood nowcasting in Pakistan that integrates Sentinel-1 SAR, Harmonized Landsat-Sentinel (HLS L30 and S30), MODIS, and VIIRS observations on a harmonized grid in Google Earth Engine. The framework employs a tiered nowcasting ensemble that prioritizes higher-resolution sensors (Sentinel-1 and HLS) and falls back to MODIS and VIIRS when necessary, preserving daily continuity of flood extent at each sensor's native resolution. Applied to the 2025 monsoon period, the system generates near-real-time, spatially consistent inundation maps across Pakistan. As a nowcasting case study, we track the super-flood of 26 August-7 September 2025 day by day, demonstrating the framework's ability to capture the evolving flood footprint in near real time and extend beyond the temporal limits of episodic mapping products. Validation against GloFAS discharge anomalies and precipitation datasets (CHIRPS v3.0, MSWEP) shows strong agreement with observed hydrometeorological conditions. By integrating nowcast outputs with exposure layers (WorldPop, ESA WorldCover, Giga-HOTOSM), the framework enables rapid estimation of affected populations, cropland, and critical infrastructure, supporting timely disaster response and resilience planning in South Asia. 

---
# AlphaEarth Satellite Embeddings for Modelling Climate Sensitive Diseases Towards Global Health Resilience 

**Authors**: Usman Nazir, I-Han Cheng, Sara Khalid  

**Link**: [PDF](https://arxiv.org/pdf/2605.10949)  

**Abstract**: Malaria, childhood acute respiratory infection, and child undernutrition together account for over two million deaths annually in children under five, with the burden concentrated in low and middle-income countries where climate variability modulates transmission, exposure, and nutritional outcomes. Routine health surveillance in these settings remains sparse and reactive. Satellite-derived representations of the Earth's surface offer a scalable, low-cost complement to traditional covariates, yet their utility as predictors of population health outcomes is poorly characterised. We summarise findings from three studies evaluating AlphaEarth Foundations 64-dimensional satellite embeddings as predictors of population health outcomes, focusing on vulnerable populations. The studies span infectious disease (malaria, respiratory infection) and stunting. In each study, embeddings provide predictive value at sufficient spatial granularity: (i) malaria prediction across Nigeria shows consistent per-region R^2 gains; (ii) childhood acute respiratory infection prediction across 11 DHS countries increases pooled R^2 from 0.157 to 0.206 across three tree-based estimators; (iii) stunting prediction across 35 countries is neutral at country level due to collinearity with fixed effects. The stunting case is currently limited by lack of DHS cluster-level coordinates, which is the next key experiment. 

---
# Breaking QAOA's Fixed Target Hamiltonian Barrier: A Fully Connected Quantum Boltzmann Machine via Bilevel Optimization 

**Authors**: Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2605.07473)  

**Abstract**: To overcome the limitations of classical partially connected Boltzmann machines and mainstream quantum Boltzmann machines (QBMs), this work extends the conventional circuit of the quantum approximate optimization algorithm (QAOA) to a bilevel optimization architecture and proposes a fully connected QBM. The inner-loop training simulates positive phase energy minimization based on the computational process of the conventional QAOA circuit, whereas the outer-loop training simulates negative phase contrastive divergence learning by optimizing the structural parameters of the target Hamiltonian. It is found that, first, the model exhibits superior performance using only a single layer (p=1) in the QAOA circuit, with an average probability of 0.9559 in measuring the target quantum state under noiseless conditions. Second, the model exhibits notable noise robustness. Under the typical noise level of current mainstream commercial quantum computing devices, the average probability of measuring the target quantum state reaches 0.6047; when the noise rises to a more stringent level with doubled intensity, this probability remains at 0.3859. In both scenarios, the target quantum state maintains the highest measurement probability among all detected states, with a value several times higher than that of the second-ranked state. This indicates that the model retains strong robustness even when noise meets or exceeds the upper limit of current mainstream commercial quantum computing devices. Third, under a block-by-block learning strategy with p=1 and only 10 measurement shots, the model consistently generates the target "qubit" grid image regardless of noise interference, demonstrating strong robustness in image generation. 

---
# MultiSoc-4D: A Benchmark for Diagnosing Instruction-Induced Label Collapse in Closed-Set LLM Annotation of Bengali Social Media 

**Authors**: Souvik Pramanik, S.M. Riaz Rahman Antu, Shak Mohammad Abyad, Md. Ibrahim Khalil, Md. Shahriar Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2605.06940)  

**Abstract**: Annotation automation via Large Language Models (LLMs) is the core approach for scaling NLP datasets; however, LLM behavior with respect to closed-set instructions in low-resource languages has not been well studied. We present MultiSoc-4D, a Bengali social media dataset benchmark, which contains 58K+ social media comments from six sources annotated along four dimensions: category, sentiment, hate speech, and sarcasm. By employing a structured pipeline where ChatGPT, Gemini, Claude, and Grok individually annotate separate partitions, while sharing a common validation set of 20%, we diagnose LLM behavior systematically. We discover a prevalent phenomenon called "instruction-induced label collapse", wherein LLMs show a systematic preference towards fallback labels (Other, Neutral, No), leading to high agreement rates but under-detection of minority categories. For example, we find that LLMs failed to detect 79% and 75% of instances with hateful and sarcastic content compared to a human-calibrated reference. Furthermore, we prove that it represents a "label agreement illusion", statistically validated via almost null Fleiss' Kappa ($\kappa \approx -0.001$) on sarcasm detection. Across 40+ LLMs, we benchmark this annotation bias propagation within the training pipeline, regardless of architectural differences. We release MultiSoc-4D as a diagnostic benchmark for annotation biases in Bengali NLP. 

---
# Measuring Accuracy and Energy-to-Solution of Quantum Fine-Tuning of Foundational AI Models 

**Authors**: Oliver Knitter, Sang Hyub Kim, Maximilian Wurzer, Jonathan Mei, Claudio Girotto, Karen Horovitz, Chi Chen, Masako Yamada, Frederik F. Flöther, Martin Roetteler  

**Link**: [PDF](https://arxiv.org/pdf/2605.02798)  

**Abstract**: We present an experimental study of energy-to-solution (ETS) of hybrid quantum-classical applications, enabled by direct instrumentation of power consumption of a Forte Enterprise trapped-ion quantum processor. We apply this methodology to a hybrid quantum-classical pipeline for quantum fine-tuning of foundational AI models, and validate the approach end-to-end on quantum hardware. Despite noise and limited qubit counts, the resulting models achieve accuracy competitive with and exceeding classical baselines such as logistic regression and support vector classifiers. Our results show that QPU energy consumption scales approximately linearly with qubit number for shallow circuits, while classical simulation exhibits exponential scaling, indicating a break-even for ETS around 34 qubits. The classification error improvement of the best quantum fine-tuned model over the best classical fine-tuned model considered in this study is around 24%. We further contextualize these findings with comparisons to tensor network methods. This work establishes energy-to-solution as a measurable and scalable metric for evaluating quantum applications and provides experimental evidence of favorable energy-accuracy trade-offs. 

---
# Stargazer: A Scalable Model-Fitting Benchmark Environment for AI Agents under Astrophysical Constraints 

**Authors**: Xinge Liu, Terry Jingchen Zhang, Bernhard Schölkopf, Zhijing Jin, Kristen Menou  

**Link**: [PDF](https://arxiv.org/pdf/2604.15664)  

**Abstract**: The rise of autonomous AI agents suggests that dynamic benchmark environments with built-in feedback on scientifically grounded tasks are needed to evaluate the capabilities of these agents in research work. We introduce Stargazer, a scalable environment for evaluating AI agents on dynamic, iterative physics-grounded model-fitting tasks using inference on radial-velocity (RV) time series data. Stargazer comprises 120 tasks across three difficulty tiers, including 20 real archival cases, covering diverse scenarios ranging from high-SNR single-planet systems to complex multi-planetary configurations requiring involved low-SNR analysis. Our evaluation of eight frontier agents reveals a gap between numerical optimization and adherence to physical constraints: although agents often achieve a good statistical fit, they frequently fail to recover correct physical system parameters, a limitation that persists even when agents are equipped with vanilla skills. Furthermore, increasing test-time compute yields only marginal gains, with excessive token usage often reflecting recursive failure loops rather than meaningful exploration. Stargazer presents an opportunity to train, evaluate, scaffold, and scale strategies on a model-fitting problem of practical research relevance today. Our methodology to design a simulation-driven environment for AI agents presumably generalizes to many other model-fitting problems across scientific domains. Source code and the project website are available at this https URL and this https URL, respectively. 

---
