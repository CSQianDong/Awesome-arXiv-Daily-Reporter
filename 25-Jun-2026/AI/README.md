# The Unfireable Safety Kernel: Execution-Time AI Alignment for AI Agents and Other Escapable AI Systems 

**Authors**: Seth Dobrin, Łukasz Chmiel  

**Link**: [PDF](https://arxiv.org/pdf/2606.26057)  

**Abstract**: AI agents are granted access to tools, APIs, and other infrastructure, making them active principals in those systems. The dominant approach places controls inside the agent's own runtime: system prompts, output filters, and guardrail libraries. Any control in the agent's address space is reachable by inputs that influence it; this generalizes to any AI system with sufficient reach into its own runtime, a class we term escapable AI systems.
We identify four properties that an authorization mechanism must satisfy for architectural control rather than for cooperative requests: process separation, pre-action enforcement on a structurally only path, fail-closed at both the request and system levels, and externalized signed evidence verifiable outside the controlled system's trust boundary. We position this layer as execution-time AI alignment, complementing training-time alignment (RLHF, Constitutional AI) and inference-time alignment.
We present the Unfireable Safety Kernel, a Rust reference implementation realizing all four. Its fail-closed invariant is machine-checked at two levels: an SMT theorem (Z3) and an exhaustive bounded-model-checking proof of the production decision function (Kani, 4/4 harnesses). A Python-to-Rust migration was gated on byte-equivalence (1000/1000 fixtures; 17/17 adversarial classes). We evaluate the kernel governing a live, escapable AI system, a deterministic, self-improving world model, against an escape-seeking adversary driving its real self-modification seam: across 1,000 self-modifications, all 704 attempts on the safety-critical core are refused, with no escape; a further 300, under the operator kill switch, are also refused. A separate campaign of 6,240 authorization round-trips had no successful bypass. Against 3 contemporary systems claiming the agent control plane, the agent invokes control; here, it lacks that choice. 

---
# Autodata: An agentic data scientist to create high quality synthetic data 

**Authors**: Ilia Kulikov, Chenxi Whitehouse, Tianhao Wu, Yixin Nie, Swarnadeep Saha, Eryk Helenowski, Weizhe Yuan, Olga Golovneva, Jack Lanchantin, Yoram Bachrach, Jakob Foerster, Xian Li, Han Fang, Sainbayar Sukhbaatar, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2606.25996)  

**Abstract**: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data. 

---
# InvestPhilBench: A Multi-Layer Dynamic Benchmark for Evaluating Large Language Model Procedural Reasoning in Expert Investment Philosophy 

**Authors**: Mingguang Chen, Bo Qu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25984)  

**Abstract**: Large language models are increasingly deployed as investment research assistants, yet no benchmark tests whether they can accurately reconstruct and apply the specific procedural decision frameworks of expert investors. We introduce InvestPhilBench, a multi-layer dynamic benchmark spanning eight cognitive tiers, from principle identification (L1) to novel framework extrapolation (L8). The v0.6 release comprises 118 primary-source-verified investment principle cards, 25 decision framework cards with explicit topology metadata, and 243 QA questions (197 dev / 46 held-out test). For reproducible scoring at scale we introduce the Benchmark Automated Scoring Pipeline (BASP) -- five algorithmic metrics (OGRS, KCCS, SAP@k, IVP, CKCA) -- the Failure Mode Detection Protocol (FMDP) with computable rules for six failure modes, and Gate Reconstruction Accuracy (GRA), a per-gate metric for questions with gold reasoning programs. In this release, InvestPhilBench is primarily a benchmark-and-methodology contribution. A four-model sanity wave on the 188-question development split shows a sharp provider-tier split (BASP 0.906 vs. 0.438); these mixed-judge numbers are confounded upper bounds. The central finding: the BASP composite saturates at the frontier (Claude L4 = 0.932) while GRA still exposes a procedural deficit (frontier L4 GRA approx. 0.77, L7 GRA 0.57-0.62) -- composite scoring rewards fluent prose and hides the procedural gap. v0.6 implements a unified judge and true model-in-the-loop retrieval/oracle conditions; the de-confounded multi-model leaderboard and full three-condition run are v1.0 deliverables. On a 100-item expert-annotated gold set the automated BASP composite tracks the human reference at Pearson r = 0.72 (MAE = 0.10), with attribution (SAP@3) the weakest sub-metric and the failure-mode detector running sensitive-but-over-flagging. 

---
# WinDOM: Self-Family Distillation for Small-Model GUI Grounding 

**Authors**: Chengheng Li-Chen, Zhiqian Zhou, Hao Chen, Nicolas Chauvin  

**Link**: [PDF](https://arxiv.org/pdf/2606.25964)  

**Abstract**: Small ($\sim$2B) GUI-grounding agents are attractive for on-device deployment, accessibility tooling, and low-cost iteration, but at this scale they face two open recipe questions: how to obtain bounding-box training data without expensive human annotation, and how to combine supervised fine-tuning with reinforcement learning. We address both, with the explicit goal of pushing small-model performance rather than scaling up. WinDOM is a $54{,}425$-record grounding corpus harvested by driving an open-source Windows 11 web reimplementation under headless Playwright, with bounding boxes read directly off the DOM and no OCR or human annotation. Self-Family Distillation (SFD) is a single rejection-sampling cold-start parameterised only by the teacher choice: either an EMA of the student (no external model) or a frozen larger same-family teacher. We then treat the saturation depth of the SFD cold-start as an explicit GRPO hyperparameter. On a Qwen3.5-2B student, the under-saturated cold-start is a better GRPO initialiser than the converged one: SFD-4B with Early-init RL gains $+5.4$ OOD-mean ($+3.5$ ScreenSpot-Pro, $+7.0$ OSWorld-G, $+5.8$ ScreenSpot-V2) over the base. The same-size EMA mode lands within roughly one OOD-mean point of the cross-size $4$B variant ($65.2$ vs $66.3$) without an external teacher. 

---
# Agentic System as Compressor: Quantifying System Intelligence in Bits 

**Authors**: Zihan Qin, Hongrui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25960)  

**Abstract**: Large language models are turning from isolated predictors into agentic systems: they call tools, retrieve evidence, obey environment constraints, use verifiers, and complete tasks through search and multi-turn interaction. We adopts an analytical viewpoint based on "compression is intelligence": under a fixed task distribution, interface, and compute budget, a stronger agentic system lets a target object be reconstructed with fewer bits. We operationalize the measure with arithmetic coding, seed coding, and a fallback, and evaluate it in five settings: reversed text, chess moves, protein sequences, retrieval-augmented question answering, and semantic story compression; in all of them agentic components reduce codelength. These small, controlled experiments cover component types typical of real agentic systems, show that codelength can analyze how components, observers, and budgets change residual uncertainty, and offer guidance for evaluating real agent systems. 

---
# AI Snitches Get Glitches: Towards Evading Agentic Surveillance 

**Authors**: Hyejun Jeong, Dzung Pham, Amir Houmansadr, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2606.25836)  

**Abstract**: To better assist users with completing challenging tasks, AI agents mediate communications, access data, and interact with different APIs. Many employers (and even nation-states) already provide their users with this technology. However, widespread adoption of AI agents creates a new risk to abuse access to user data for another goal: surveilling users. These users might not even have the ability or permission to control the actions and data accesses of the surveilling agents.
We introduce and formalize the problem of agentic surveillance: the ability of an AI agent to analyze available information, craft a report, and send it out using available tools. To evaluate surveillance capabilities across different models, we create SurveilBench, a dataset of various reporting scenarios focusing on three domains: corporate, education, and police. We find that some models exhibit emergent (i.e., unprompted) tendencies to help surveillance, but they also report the attempts to surveil users to the government.
Finally, we repurpose prompt injections for evading surveillance and develop three evasion techniques that hide from, deceive, or induce over-escalation in surveillance agents. We conclude that agentic surveillance can already be easily implemented and, therefore, call for a comprehensive technical, ethical, and legislative framework to protect users. 

---
# Confidence Sequences for Online Statistical Model Checking of Markov Decision Processes 

**Authors**: Konstantin Kueffner, Tobias Meggendorfer, Maximilian Weininger, Patrick Wienhöft  

**Link**: [PDF](https://arxiv.org/pdf/2606.25797)  

**Abstract**: Markov decision processes (MDPs) are a classic model of decision making under uncertainty, exhibiting both non-deterministic choice as well as probabilistic uncertainty. Traditionally, exact knowledge of the underlying probabilities is assumed. However, this often is unrealistic, e.g.\ when modelling cyber-physical systems or biological processes. Here, statistical methods provide a way towards obtaining meaningful guarantees. The classical approach is to gather samples in the MDP, use these to draw statistical conclusions about the transition probabilities, and from there obtain bounds on the true value; then, if these bounds are too broad, repeat. However, existing implementations of this approach are either subtly incorrect or sub-optimal, and quite often both. We present several \emph{confidence sequences}, which are specifically designed for such \enquote{online} settings, implement all of them in an efficient tool, and show their practical applicability. In particular, we show that they outperform classical \enquote{union-bound} style approaches, and overall our implementation requires 50x less samples on average than previous state of the art. 

---
# Fuzzy Quantification over OWL Ontologies and Knowledge Graphs 

**Authors**: Enrique Palacín, Fernando Bobillo, Ignacio Huitzil, Francesca A. Lisi, Umberto Straccia  

**Link**: [PDF](https://arxiv.org/pdf/2606.25778)  

**Abstract**: This paper presents a versatile framework for evaluating fuzzy quantification queries over both standard and fuzzy ontologies as well as knowledge graphs. The primary objective is the retrieval of individuals that satisfy queries articulated via Type I or Type II fuzzy quantified expressions. A key advantage of the proposed approach is its inherent adaptability: it remains entirely agnostic to the quantifier type, the underlying evaluation method, and the specific data source of the ontology (i.e., OWL ontologies or RDFS knowledge graphs). Furthermore, we present Q2S2, a publicly accessible implementation of this system developed to support future research. 

---
# Position Spaces and Graphs 

**Authors**: Rita-Nathalia Assaf, Tom Davot, Frédéric Lardeux, Frédéric Saubion  

**Link**: [PDF](https://arxiv.org/pdf/2606.25719)  

**Abstract**: In this paper, we introduce position graphs, a graph-based reasoning framework based on the formalization of position spaces. This framework utilizes two strict partial orders, representing horizontal and vertical alignment and precedence, to model the relative positions of discrete tokens. Unlike general qualitative spatial calculi, position graphs are constrained by a chain condition and compatibility requirements that focus on rows and columns. We provide a comprehensive theoretical analysis of this representation, beginning with a characterization of graph consistency. Conditions to ensure the consistency of position graphs are established. Furthermore, we investigate the computational complexity of structural pattern discovery, modeled as the induced subgraph isomorphism problem. We demonstrate that this problem remains NP-complete even within the restricted class of position graphs. While initially motivated by document processing, this work focuses on the underlying mathematical properties and algebraic consistency of position-based constraints, providing a formal logical layer that is independent of specific data extraction techniques. 

---
# GUI agent: Guided Exploration of User-Sensitive Screens 

**Authors**: Aradhana Nayak, Mussadiq Nazeer, Wang Peng, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25705)  

**Abstract**: LLM agents are increasingly being used to automate tasks for users within an open GUI environment. They inevitably encounter screens containing user-sensitive information, for which takeover of task execution by the user is highly desirable or even necessary. State-of-the-art LLM-driven agents are usually fine-tuned to complete tasks regardless of the safety implications of their actions. This makes their real-world deployment difficult and adversely affects the reliability. Therefore, it is crucial to identify and categorize user-sensitive states and define user-sensitive queries. This dataset would be to engineers to recognize and request handover to the user in critical scenarios. This short paper develops an explorer agent that systematically explores the query space starting from one demonstrated task to identify queries that, if executed, would lead to user-sensitive states in a GUI environment. 

---
# Reasonable Motion: A General ASP Foundation for Environment Constrained Movement Trajectory Computation 

**Authors**: Julius Monsen, Jakob Suchan, Mehul Bhatt, Lars Karlsson  

**Link**: [PDF](https://arxiv.org/pdf/2606.25626)  

**Abstract**: We present a general answer set programming based hybrid quantitative-qualitative method for computing constrained branching trajectory modes for moving objects in real-world settings. The method performs constrained traversal of an environment graph, enumerating geometrically admissible motion behaviours as stable models, each constituting a distinct trajectory mode characterised by both domain-dependent and independent factors such as derived event sequence, map topology, and domain norms. The hybrid trajectory computation method is generally applicable across motion characteristics typically encountered in diverse dynamic domains with moving objects, e.g., autonomous driving. We demonstrate applicability and highlight how computed trajectories are traceable to their underlying stable model, thereby affording verifiable interpretability that purely learned approaches cannot provide. We also perform an empirical evaluation with Argoverse 2, a large-scale real-world autonomous driving benchmark representative of the class of dynamic domains within the scope of the proposed method. 

---
# Agentic evolution of physically constrained foundation models 

**Authors**: Jiangwei Zhang, Wen Sun, Chong Wang, Shiyao Li, Cheng Che, Chunjing Han, Dan Meng, Jian Yang, Yu Wang, Rui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2606.25532)  

**Abstract**: Artificial intelligence increasingly drives automated scientific discovery, yet contemporary generalist agents lack physical grounding, frequently hallucinating hardware-incompatible designs. Here, we present a physically grounded, multi-agent discovery engine that autonomously architects hardware-compliant computing systems. Anchored by an Evolutionary Knowledge Graph structuring past scientific innovations, the framework extracts an "algorithmic Chain-of-Thought" to transform blind stochastic search into directed structural evolution. Applied to the extreme testbed of foundation model deployment, the engine evolved two hardware-aware compression methodologies surpassing human-engineered heuristics: Q-Enhance mitigates long-context accuracy loss in dense models, and MoE-Salient-AQ outperforms state-of-the-art manual sparse Mixture-of-Experts designs by 3.7% at sub-3-bit regimes. Utilizing a bandwidth-efficient Sensitivity Profile, we successfully deployed a massive 235-billion-parameter model onto a constrained dual-A100 server, reducing memory requirements by 75% with a marginal 0.64% accuracy degradation. By transforming unconstrained combinatorial search into knowledge-driven autonomy, this establishes a scalable hardware-software co-design paradigm for machine-driven discovery within strict physical boundaries. 

---
# Cliff Tokens: Identifying Single-Token Failure Triggers in LLM Mathematical Reasoning 

**Authors**: Jaeyong Ko, Pilsung Kang, Yukyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.25524)  

**Abstract**: Large language models (LLMs) reach high accuracy in mathematical reasoning, but individual traces on the same problem diverge; some arrive at the correct answer while others fail. Prior work analyzes failure at the step, chunk, or sentence level, or at tokens where failure has already occurred. Neither identifies the precise token that triggers the shift toward failure. We introduce the cliff token, a token where the token-wise potential drops significantly under an adaptive threshold that scales with the local token-wise potential, based on a one-sided two-proportion z-test. Across seven models and three mathematical reasoning benchmarks (GSM1K, MATH500, AIME 2025), cliff tokens act as failure triggers; deleting the first cliff token and resampling recovers pass@64 to 1.0, while keeping it limits recovery to between 0.71 and 1.00. We further introduce a cliff taxonomy of deterministic, uncertain, and sampled-off cliffs, defined by greedy choice and token entropy. Each type has distinct probabilistic characteristics, and the taxonomy generalizes across model scales. Finally, we validate the taxonomy via single-token preference optimization at cliff positions (Cliff-DPO). Trained on GSM8K, Cliff-DPO improves accuracy across benchmarks by up to +6.6. Optimizing at uncertain and sampled-off cliffs improves reasoning, while deterministic cliffs do not. 

---
# Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models 

**Authors**: Xinyu Lian, Walid Krichene, Beichen Huang, Masahiro Tanaka, Olatunji Ruwase, Li Zhang, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25519)  

**Abstract**: Quantization is widely used to reduce the inference cost of large language models, but its effect on reasoning models is not fully captured by final-answer accuracy or per-token latency. We show that low-bit post-training quantization can introduce a hidden test-time compute cost: quantized reasoning models often generate longer chains of thought even when they still answer correctly. Across mathematical reasoning, code generation, scientific question answering, and agentic tool-use benchmarks, we find that INT4/INT3 quantization can preserve accuracy but increase reasoning-token usage, offsetting the expected per-token speedup. To measure this effect, we introduce the CoT Token Inflation Ratio, which compares reasoning length between quantized and full-precision models averaged across all evaluation benchmarks. We further show that token inflation is accompanied by behavioral changes in the reasoning trace, including more intermediate steps and greater semantic repetition. These changes translate into measurable end-to-end real-world serving penalties. Finally, we evaluate mitigation strategies and find that prompting and decoding-time sampling offer inconsistent accuracy-length trade-offs, while quantization-aware training shows more promise in reducing both accuracy degradation and token inflation. Our results suggest that reasoning-token usage should be reported alongside accuracy when evaluating quantized reasoning models. 

---
# BrainAgent: A Large Language Model-Driven Multi-Agent Framework for Autonomous Brain Signal Understanding 

**Authors**: Yangxuan Zhou, Sha Zhao, Jiquan Wang, Shijian Li, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2606.25400)  

**Abstract**: Brain-Computer Interfaces (BCIs) and brain signal understanding are pivotal for clinical health and next-generation interactions. Despite this significance, its widespread adoption in real-world scenarios remains restricted, primarily because current analytical paradigms lack sufficient agentic intelligence. First, existing methodologies impose prohibitive technical barriers, requiring extensive specialized expertise. Second, they remain inherently static and task-specific, failing to execute the complex, long-horizon workflows essential for real-world deployment. To accelerate the democratization of brain signal understanding, we draw inspiration from Large Language Models (LLMs) to introduce BrainAgent, an LLM-driven multi-agent framework designed to ground abstract natural language intent into rigorous, executable, and end-to-end processing pipelines. BrainAgent employs a hierarchical architecture where a central supervisor orchestrates specialized sub-agents for adaptive task decomposition and execution. Furthermore, we establish a comprehensive, systematic benchmark for evaluating agentic systems in brain signal analysis. Empirical results demonstrate that BrainAgent effectively automates complex workflows with superior reliability, marking a paradigm shift toward democratized brain signal understanding. 

---
# Long-Term Simulation Exposes Cognitive-Developmental Risks in AI Companions 

**Authors**: Kaicheng Shen, Lingyu Li, Wen Wu, Yan Teng, Liang He, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25396)  

**Abstract**: AI companions powered by large language models increasingly interact with cognition-developing users, including children and adolescents, creating risks that may accumulate over time. Existing safety evaluations largely rely on single-turn or short-session tests, which cannot capture risks that emerge only through prolonged interaction. To address this gap, we propose TSJ (Theater-Stage-Judge), a longitudinal framework combining persona-driven user simulation, dynamic psychological-state updating and retrospective evaluation. We evaluate six mainstream models across four developmental stages, twenty-four risk dimensions and three psychological-vulnerability personas, covering 12,960 simulated person-day interactions. TSJ shows that short-horizon testing systematically underestimates developmental risks, for which TSJ yields a stable risk estimate only after 140 turns within prolonged simulated relationships. Applying TSJ further identifies early childhood and emerging adulthood as the most vulnerable stages, with cognitive trust and emotional dependency as the weakest domains. TSJ provides a scalable methodology for longitudinal cognitive developmental risk evaluation in AI companion systems. 

---
# Offline Multi-agent Continual Cooperation via Skill Partition and Reuse 

**Authors**: Yuchen Xiao, Lei Yuan, Ruiqi Xue, Tieyue Yin, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25389)  

**Abstract**: Extracting skills from multi-agent offline dataset improves learning efficiency via sharing task-invariant coordination skills among tasks. In settings where tasks occur sequentially and the space of skills grows exponentially, existing approaches that rely on heuristically designed and fixed-sized skill libraries struggle to resolve the problem of distributional shift and interference, facing catastrophic forgetting and plasticity loss. To address this problem and endow agents with the ability to continually discover and reuse coordination skills in open-environment, we propose COMAD, a principled framework for Continual Offline Multi-agent Skill Discovery via Skill Partition and Reuse. We first discover skills from mixed multi-agent behavior data with an auto-encoder to transform coordination knowledge into reusable coordination skills. Then we construct a skill-augmented policy learning objective with multi-head architectures, explicitly guiding the advantage function with reusable skills identified via a density-based reusability estimator. Theoretical analysis shows our method approximates the optimum of a continual skill discovery problem. Empirical results across diverse MARL benchmarks show that COMAD continually expands its skill library to mitigate interference, achieving superior forward and backward transfer for task streams compared to multiple baselines. 

---
# What Actually Works for Spacecraft Fault-Tolerant Control: An Honest Settled-Gate Benchmark of Learned and Classical Methods 

**Authors**: Alireza Shojaei  

**Link**: [PDF](https://arxiv.org/pdf/2606.25374)  

**Abstract**: Recent learned fault-tolerant-control (FTC) work reports high success on spacecraft actuator faults, but often in simulation, on narrow fault sets, and with transient metrics that a trajectory need only touch once. We ask what recovers spacecraft pointing when success means holding it on faults never seen in training. We answer with a benchmark built around a settled gate, pointing held within 0.2 deg over a dwell window and scored on the true state, train/test splits disjoint in inertia, gain, sign pattern, and bias, Wilson intervals over n=500 episodes per cell, and one-command reproduction on a 6-DOF Basilisk testbed. Across classical, adaptive, learned end-to-end, and structured controllers, three findings stand out. Fault-unaware PD/PID and from-scratch end-to-end RL score 0%, so learning capacity alone is not the lever. Classical adaptive laws resolve sign faults but handle gain poorly at 55.2%, and a literature-faithful Nussbaum-gain law reaches 45.2% and 3.2%. A structured estimate-then-control design, with a learned recurrent module that infers actuator gain online and feeds an analytic law, wins on sign and gain faults at 97.8% and 94.4%, approaching the privileged oracle while unstructured methods remain at zero. The hard wall is constant additive bias, which is 0% for every controller including the privileged gain oracle, because an integral-free law cannot null a constant disturbance. We close it with a disturbance observer that recovers bias from the dynamics and is self-correcting for gain-estimate error. Composed with the gain estimate, it recovers 59.4% of held-out bias faults with no sign/gain regression, moving that class off zero. We classify sensor-fault regimes similarly, show that sensor bias is unobservable from the corrupted measurement alone and therefore requires fusion rather than an observer, and release the benchmark so the gate is shared. 

---
# Agentic Knowledge Tracing: A Multi-Agent LLM Architecture for Stealth Assessment of Financial Literacy in Serious Games 

**Authors**: Gabriel Santos, Rita Julia, Marcelo Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2606.25358)  

**Abstract**: Assessing financial literacy during gameplay without disrupting the learning experience remains a key challenge in serious games for education. We present the Agentic BKT pipeline, a multi-agent large language model architecture for stealth assessment of financial competencies from open-ended gameplay events. The pipeline processes events from a 2D platformer serious game aligned with the OECD/INFE financial literacy framework through four phases: (1) the game captures every player decision as a structured event log; (2) an LLM event classifier labels each action on a four-point rubric validated against three domain experts (Fleiss kappa = 0.624, substantial agreement); (3) four domain-specific agents specializing in risk mitigation, investing, spending, and credit management perform session-level reasoning over behavioral trajectories, feeding per-competency Bayesian Knowledge Tracing that estimates mastery within each domain; and (4) an expert judge agent synthesizes the domain-level estimates into an overall mastery score. Evaluated with 193 K-12 participants across 264 game sessions, the Agentic BKT pipeline yields mastery estimates significantly correlated with learning gain (r = 0.276, p = 0.0001) and post-test scores (r = 0.333, p < 0.0001) while showing no correlation with pre-test scores, providing both convergent and discriminant validity. The multi-agent approach approximately triples the predictive validity of a single-LLM baseline (r = 0.095, not significant) in this study, demonstrating that domain decomposition and session-level reasoning play a central role in capturing the multidimensional nature of financial literacy from gameplay 

---
# Omni-Perception Policy Optimization for Multimodal Emotion Reasoning 

**Authors**: Zhiyuan Han, Beier Zhu, Wenwen Tong, Pengyang Shao, Peipei Song, Xinyi Wang, Jiangnan Chen, Lewei Lu, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25325)  

**Abstract**: We find that current emotion-oriented Omni-MLLMs still lack reliable omni-modal perception: they (i) underutilize multimodal cues in their reasoning trajectories and (ii) exhibit unfaithful behavior, often hallucinating modality-specific statements from other modalities. Building on these insights, we propose OPPO (Omni-Perception Policy Optimization), a reinforcement learning framework that explicitly optimizes multimodal perception. First, an Omni-Perception Reward decomposes ground-truth reasoning into fine-grained visual, acoustic, and emotion cues and rewards trajectories that semantically recover these cues. Second, an Omni-Perception Loss compares the policy under full and unimodally masked inputs, applying a KL penalty only to modality-specific evidence tokens to suppress cross-modal hallucination. We further introduce MEP-Bench, a diagnostic benchmark that quantifies utilization and faithfulness. Experiments show that OPPO achieves state-of-the-art performance on MER-UniBench and MME-Emotion, while substantially improving utilization and faithfulness scores on MEP-Bench, highlighting the importance of sufficient and faithful omni perception for multimodal emotion reasoning. 

---
# Heuresis: Search Strategies for Autonomous AI Research Agents Across Quality, Diversity and Novelty 

**Authors**: Antonis Antoniades, Deepak Nathani, Ritam Saha, Alfonso Amayuelas, Ivan Bercovich, Zhaotian Weng, Vignesh Baskaran, Kunal Bhatia, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25198)  

**Abstract**: Autonomous AI Research promises to accelerate the scientific progress of machine learning. To realise this goal, current Large Language Model (LLM)-based agents need to go beyond just writing code, to mastering the exploration of simultaneously performant, diverse and novel ideas. To this end, we introduce Heuresis, a framework that abstracts the research pipeline into a set of general and composable primitives, enabling open-ended scientific exploration in machine learning research. We implement six search strategies: a greedy baseline, two archive-based (MAP-Elites, Go-Explore), one evolutionary (Islands), and two divergent (Curiosity, Omni), and evaluate them across three axes (Quality, Diversity, and Novelty) on three domains (LLM Pretraining, On-Policy RL, and Model Unlearning), totalling 3,222 scored runs. We find that completely novel ideas are rare. No idea across our scored runs is rated as "Original", and only a few achieve only "Minor Similarity" to prior work. Moreover, novel ideas never approach the highest-performing known-recipe scores. Across all six strategies and three domains, only one such idea lands in the top-10 by quality. We also observed agents resorting to a variety of reward-hacking techniques during execution (40 confirmed fabrications across 1,628 scored runs), and detecting them was necessary to keep the search faithful to the task. Our results show that while current search and Quality-Diversity strategies enable us to steer where the generated ideas land on the quality, diversity, and novelty axes, they do not expand the quality-novelty frontier. Bridging this gap is the open challenge towards the ultimate goal of perpetual, autonomous scientific progress. Code is available at this http URL. 

---
# To Isolate or to Score? Model-Adaptive Assessment for Cost-Efficient Multi-Agent RAG 

**Authors**: Jungseob Lee, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.25191)  

**Abstract**: Multi-agent document assessment for retrieval-augmented generation is computationally expensive, driving practitioners toward smaller, deployable models whose assessment mechanisms remain poorly understood. We conduct a controlled study of training-free interventions on 7B-9B instruction-tuned models across diverse QA benchmarks, revealing a sharp dichotomy in how models benefit from assessment. For weaker baselines, the dominant mechanism is per-document isolation. Astoundingly, assessment-free isolation matches full multi-agent assessment, demonstrating that resolving multi-document context confusion, rather than scoring quality, drives outsized gains of up to 50 percentage points. Conversely, for strong baselines where scoring quality matters, we introduce Reasoning-Score Coupling, a label-free perturbation probe that classifies scoring behavior. Integrating these findings, we propose MADARA, a model-adaptive routing architecture. Crucially, MADARA's diagnostic thresholds derived from a single pilot model generalize zero-shot to four unseen model families, providing a robust, lightweight pipeline to eliminate computational overhead. 

---
# Transferability for General Reasoning: An Automated Curriculum for Multi-Domain RLVR 

**Authors**: Yongjin Yang, Jiarui Liu, Yinghui He, Lezhen Zhang, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2606.25178)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has been extended from single-domain training to multi-domain reasoning suites spanning mathematics, programming, and science. However, the training curriculum (how often each domain is sampled) is typically fixed or hand-tuned, even though reasoning skills transfer unevenly across domains. Existing learnability-based curricula adapt to where the policy is currently improving, but are blind to whether a gradient step on the selected domain benefits the remaining domains. In this paper, we propose Transfer-Aware Curriculum (TAC), a bandit-style online curriculum that prioritizes domains whose updates broadly benefit the rest of the training suite. TAC repurposes signals already produced by RL training: per-domain advantages capture local learnability, and projected gradients, taken from the GRPO step being computed, estimate cross-domain transferability via gradient-geometry alignment, at negligible cost (<1% wall-clock overhead). Across a six-domain reasoning suite, TAC achieves the best macro-averaged accuracy on both Qwen3-1.7B and Llama3.2-3B, outperforming proportional random sampling, a hand-designed schedule, and a learnability-only bandit, and improving over the last of these by up to 2.8 points (10% relative). Ablations show performance degrades sharply when the transferability term is removed, and TAC remains robust on imbalanced training mixtures where learnability-only curricula over-commit to dominant domains. Our findings establish cross-domain transferability as a key signal for curriculum design in multi-domain RLVR. 

---
# Elo-Disentangled Player-Style Embeddings for Human Chess via Rating-Conditioned Residual Move Model 

**Authors**: Jason Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2606.25176)  

**Abstract**: We study representation learning for individual human chess style: a per-player embedding learned from a player's move history such that inner products measure stylistic similarity, while being approximately disentangled from playing strength (Elo). Our key design is a residual formulation: a rating-conditioned base move model (Maia-3 policy logits plus Stockfish-derived features, scored over Maia-2-proposed candidates) captures what a typical player of a given strength would play, and a frozen copy of it anchors a learned move encoder and a per-player vector z, so that z explains only deviations from rating-typical play. The base model improves move prediction over the strong Maia-3 policy by 27-37% relative NLL across the rating spectrum, with the largest gains at the top (2800+); Stockfish's marginal value grows monotonically with Elo (negligible at 900-1200, +0.085 nats at 2800+). On a shared Elo-stratified benchmark of 22,620 held-out decisions, top-1 move-matching rises monotonically from Maia-2 to Maia-3 to the Stockfish-augmented base (0.51 -> 0.57 -> 0.68): the base is +33% relative top-1 over Maia-2 and +19% over Maia-3 (30% lower NLL), with the engine-feature lift largest at high Elo. The player embedding adds little to raw move-matching on top of this base -- its marginal top-1 gain falls within the 95% confidence interval -- and its value is instead representational: z generalizes to held-out decisions without overfitting, re-identifies players from disjoint games above chance, and a linear probe recovers rating from z with only R^2 = 0.06 (no better nonlinearly), evidence it captures style on an Elo-orthogonal axis. We argue that a strong rating-conditioned base plus a compact, Elo-disentangled embedding -- separating typical play from individual deviation -- is an economical, interpretable model of individual style, an alternative to per-player preference fine-tuning. 

---
# TRUSTMEM: Learning Trustworthy Memory Consolidation for LLM Agents with Long-Term Memory 

**Authors**: Tianyu Yang, Sudipta Paul, Vijay Srinivasan, Vivek Kulkarni, Srinivas Chappidi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25161)  

**Abstract**: Large language model (LLM) agents rely on long-term memory to support extended interactions and personalized assistance beyond finite context windows. Existing memory agents actively update external memory through generated write, revise, and delete operations, but these updates may omit important information, corrupt existing memory, or introduce unsupported hallucinated content. Once stored, such errors become persistent system-state failures that can affect future reasoning and generation. In this paper, we propose TrustMem, a framework designed to improve the trustworthiness of memory consolidation. TrustMem relies on a Memory Transition Verifier to evaluate the transition process of memory updates in terms of coverage, preservation, and faithfulness. It further constructs preference pairs among candidate updates under the same memory state, enabling preference-guided reinforcement learning to directly optimize memory updating behaviors. Extensive experiments demonstrate that TrustMem improves both memory utility and reliability: it achieves state-of-the-art results across MemoryAgentBench, HaluMem, and the Mem-alpha validation set, improves HaluMem memory extraction by 12.14 F1 points, and reduces transition-level omission, corruption, and hallucination by 40.1\%, 79.1\%, and 50.0\%, respectively, compared with the strongest baseline for each error type. 

---
# The Clinician's Veto: Navigating Trust, Liability, and Uncertainty in Autonomous AI Prescribing 

**Authors**: Eileanor LaRocco, Sarah Tan, Adarsh Subbaswamy, Anne Andrews, Andrew Taylor, Cree Gaskin, Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2606.25108)  

**Abstract**: Autonomous AI systems are transitioning from advisory to autonomous roles for medication prescriptions. Recent United States bill H.R. 238 and Utah's prescription-renewal pilot both authorize AI to prescribe medications in an agentic capacity. While some regulatory guidelines suggest aggregate model performance metrics for clearance, they do not require i) calibrated per-prediction confidence for action-gated thresholds, ii) differentiated communication of uncertainty arising from model ignorance (epistemic) versus genuine clinical ambiguity (aleatoric), and iii) inferential transparency at the moment of decision that allows for liability allocation. Here, we present a regulatory and technical argument (tested with a survey of 136 U.S. prescribing clinicians) positioning these as minimum architectural requirements for safe autonomous prescribing. Our results suggest prescribing clinicians i) would not permit autonomous prescribing without a calibrated confidence-based escalation mechanism, ii) preferred a competing-options summary when uncertainty was aleatoric but shifted to abstention when uncertainty was epistemic, and iii) were only willing to accept additional liability when inferential transparency enabled a substantive judgment under acknowledged uncertainty. These findings indicate our recommended architectural features would encourage higher rates of clinician adoption, largely through collapsing much of what "autonomy" conventionally means. A system meeting these requirements would function less as an autonomous agent and more as a heavily supervised decision-support tool. As legislation and state pilots proceed, our technical argument backed by clinician perspectives provides opportunities for regulation to constrain the degree of autonomy ethically granted to AI in prescribing while aligning liability with the institutional actors who control system design and deployment. 

---
# Beyond Shapley: Efficient Computation of Asymmetric Shapley Values 

**Authors**: Ezequiel Companeetz, Santiago Cifuentes, Sergio Abriola  

**Link**: [PDF](https://arxiv.org/pdf/2606.25103)  

**Abstract**: We address the problem of explainability in machine learning models through feature attribution methods. In particular, we consider a variant of Shapley values known as Asymmetric Shapley Values (ASV), which enables the incorporation of causal knowledge into model-agnostic explanations through the use of a causal graph. We show that in certain contexts in which the computation of SHAP is $\#P$-hard, the exact computation of ASV can be done in polynomial time. To extend this algorithmic result, we introduce a notion of equivalence classes over the topological orderings of the underlying causal graph, which is useful to reduce the time to compute ASV. In particular, we present a polynomial-time algorithm (in the number of equivalence classes) to compute it whenever the causal graph is a rooted directed tree. Finally, we develop an algorithm for approximating ASV in arbitrary causal DAGs which relies on a procedure to sample topological orderings uniformly at random. To implement this sampling mechanism we leverage known algorithms as well as simpler alternatives. Our experimental results demonstrate the practical viability of the proposed approach in realistic causal structures. 

---
# Do vision-language models search like humans? Reasoning tokens as a reaction-time analog in classic visual-search paradigms 

**Authors**: Farahnaz Wick  

**Link**: [PDF](https://arxiv.org/pdf/2606.25066)  

**Abstract**: Visual search has been one of the most productive paradigms in the study of visual attention: the way reaction time scales with the number of items distinguishes parallel, "pop-out" search from serial, attention-demanding search. I ask whether vision-language models (VLMs) exhibit the same behavioral signatures. I adapt four classic paradigms: feature versus conjunction search, spatial-configuration (T-vs-L) search, enumeration, and the tilted/vertical search asymmetry; and present them to current frontier and mid-tier models. Because a single model call has no reaction time, I use the number of reasoning ("thinking") tokens a model spends per trial as a within-model analog of search effort, and I compare against a large public human benchmark (Wolfe et al., 2010). The models reproduce several human signatures: feature search costs flat effort while conjunction effort climbs with set size; frontier models hold accuracy where mid-tier models collapse to chance; and a resolution control shows the conjunction cost is genuine search rather than difficulty resolving small shapes. They also diverge from humans in informative ways. The target-present effort slope exceeds the target-absent slope, reversing the human ordering; enumeration remains accurate where humans would lose count; and a reasoning model with adaptive deliberation declines to deliberate on detection tasks altogether, so that a single search expresses itself as an effort gradient in one model and as an accuracy cliff in another. I argue that psychophysical paradigms, applied behaviorally, are a sharp and inexpensive probe of machine visual cognition, and that the points of divergence are as informative as the points of agreement. 

---
# Diagnosing and Mitigating Compounding Failures in Agentic Persuasion via Taxonomic Strategy Retrieval 

**Authors**: Pradyumna Narayana, Sana Ayromlou, Purvi Sehgal  

**Link**: [PDF](https://arxiv.org/pdf/2606.24976)  

**Abstract**: Foundation-model agents in multi-step, open-ended environments frequently suffer from compounding errors, where early mistakes contaminate long-horizon trajectories. While Multi-Agent Debate (MAD) succeeds in deterministic domains, agents in subjective tasks like persuasion experience severe problem drift and sycophantic conformity. We identify semantic leakage in standard Retrieval-Augmented Generation (RAG) as a reproducible trigger for these failures, as standard RAG prioritizes vocabulary overlap over logical necessity.
To eliminate this leakage, we introduce Taxonomic Strategy RAG (TS-RAG), a systems intervention that routes strategies through a discrete categorical bottleneck to decouple argumentative structure from topical content. Zero-shot, cross-domain evaluations demonstrate that TS-RAG significantly improves the transfer of abstract logic where standard semantic retrieval collapses. Crucially, TS-RAG acts as a "capability bridge" in asymmetric deployments, empowering lightweight persuaders to consistently defeat parametrically superior opponents (improving win rates from 70.5 to 78.5) and accelerating argumentative efficiency. Finally, we introduce trace-level diagnostics via a turn-by-turn Debate State Representation (DSR), demonstrating the necessity of strict constraints to prevent evaluation collapse via default agentic sycophancy. 

---
# Project Auto-World: Towards Automated Benchmarking of Neural Relational Reasoners 

**Authors**: Anirban Das, Joanne Boisson, Irtaza Khalid, Sumita Garai, Steven Schockaert  

**Link**: [PDF](https://arxiv.org/pdf/2606.24965)  

**Abstract**: Reasoning about relational structures remains a significant challenge for neural models, particularly when they must systematically apply learned knowledge to problem instances that are harder than those seen in training. Progress is hampered by the difficulty of evaluating such generalization, since a priori, it is rarely clear what makes an instance hard. We study how this issue can be addressed by using large language models (LLMs) to automate benchmark generation, learning to produce increasingly challenging instances in an end-to-end manner. Concretely, given a world parametrized by Datalog rules, and an Edge Transformer as the reasoning evaluator, we use LLM-driven evolutionary search (based on FunSearch) and autonomous agentic search to discover sampling functions that yield hard problem instances. We also show that the Edge Transformer can be improved using this data such that it generalizes well to further data perturbations. Finally, we show that the same machinery can be applied to novel worlds proposed by LLMs, opening the door to autonomous research on neural relational reasoning. 

---
# The Hitchhiker's Guide to Agentic AI: From Foundations to Systems 

**Authors**: Haggai Roitman  

**Link**: [PDF](https://arxiv.org/pdf/2606.24937)  

**Abstract**: The Hitchhiker's Guide to Agentic AI is a comprehensive practitioner's reference for building autonomous AI systems. The book covers the full stack from first principles to production deployment, organized around a central thesis: building great agentic systems requires understanding every layer of the pipeline, not just one. The book opens with the LLM substrate -- transformer architecture, GPU systems, training and fine-tuning (SFT,LoRA, MoE), model compression, and inference optimization -- treated as essential foundations rather than the primary focus. It then develops the alignment and reasoning layer: reinforcement learning from human feedback (RLHF), PPO, DPO and its variants, GRPO, reward modeling, and RL for large reasoning models including chain-of-thought and test-time scaling. The second half is devoted to agentic AI proper. Topics include agentic training and trajectory-based RL, retrieval-augmented generation (RAG and Agentic RAG), memory systems (in-context, external, episodic, and semantic), agent harness design and context management, and a taxonomy of agent design patterns. Inter-agent coordination is covered in depth: the Model Context Protocol (MCP), agent skills and tool use, the Agent-to-Agent (A2A) communication protocol, and multi-agent architectures spanning centralized, decentralized, and hierarchical topologies. The book concludes with agent development frameworks, agentic UI design, evaluation methodology for agentic tasks, and production deployment. Each chapter pairs rigorous theoretical foundations with implementation guidance, code examples, and references to the primary literature. 

---
# Learning Action Priors for Cross-embodiment Robot Manipulation 

**Authors**: Dong Jing, Tianqi Zhang, Jiaqi Liu, Jinman Zhao, Zelong Sun, Li Erran Li, Zhiwu Lu, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2606.26095)  

**Abstract**: Most Vision-Language-Action (VLA) models build on a Vision-Language Model (VLM) backbone by attaching an action module and optimizing the full policy jointly. This design inherits strong visual and linguistic priors from the VLM, but leaves the action module to learn physical motion almost from scratch. As a result, the policy lacks an explicit motion prior, forcing early optimization to simultaneously discover temporal action dynamics and cross-modal alignment, a challenge further amplified in cross-embodiment settings. In this work, we propose to pretrain the action module with motion priors before cross-modal VLA alignment. Specifically, we introduce a two-stage training framework that equips the action module with cross-embodiment temporal motion structure before VLA training begins. In Stage~1, a lightweight flow-matching-based encoder-decoder action module efficiently learns temporal motion structure solely from unconditioned action trajectories, without processing visual or language tokens. In Stage~2, this learned prior is transferred to VLA training through decoder reuse and early-stage latent distillation, aligning visual-language features with the action embedding space while still allowing end-to-end policy refinement. In addition, the trained encoder serves as a compact history compressor, summarizing state-action histories into a single temporal context token for history-aware modeling at negligible cost. Extensive experiments across 13 diverse cross-embodiment tasks on both simulated and real-world platforms validate the effectiveness of our approach. Compared with VLA training without action priors, our model achieves faster convergence, higher success rates, and substantially stronger performance on data-scarce real-world tasks. Moreover, scaling up the action data in Stage~1 yields a more generalizable action prior that directly improves downstream VLA performance. 

---
# On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity 

**Authors**: Andrei Liviu Nicolicioiu, Mohammad Pezeshki, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2606.26091)  

**Abstract**: On-policy self-distillation achieves strong pass@1 accuracy by using a single model as both teacher and student, with the teacher conditioned on a correct demonstration to provide dense token-level feedback. We show that this could come at a hidden cost: rollout diversity decreases and pass@k curves flatten (i.e., generating more rollouts fails to improve accuracy). We trace this to compounding biases in the design of self-distillation with sampled demonstrations. The teacher scores each student rollout while conditioned on a sampled correct rollout, channeling its feedback through the model's own biases. We theoretically analyze the optimal self-distillation policy and show that it tilts the base distribution by a pointwise conditional mutual information score between the student's rollout and the correct rollout used as context. Unlike the ideal optimal on-policy reinforcement learning (RL), which preserves probability ratios among equally correct rollouts, self-distillation can amplify existing probability gaps, concentrating mass on already-dominant modes. On a controlled graph path-finding task and science question-answering benchmarks, self-distilled models match or exceed RL on average performance but exhibit substantially lower functional and semantic diversity, failing on out-of-distribution settings that require diverse strategies. 

---
# Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents 

**Authors**: Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh, Tanwi Mallick, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.26080)  

**Abstract**: Process reward models enable fine-grained, step-level evaluation of LLMs, yet building them for agentic settings remains prohibitively difficult: long-horizon interactions, irreversible actions, and stochastic environment feedback make both human annotation and Monte Carlo estimation infeasible at scale. In this work, we show that reinforcement learning (RL) post-training already provides the ingredients for effective step-level scoring, eliminating the need for dedicated reward model training altogether. Concretely, we derive an implicit advantage under a general stochastic Markov decision process, which we term progress advantage -- log-probability ratio between the RL-trained policy and its reference policy exactly recovers the optimal advantage function. This formulation makes the resulting signal annotation-free, domain-agnostic, and available as a byproduct of the standard RL post-training pipeline. We validate the effectiveness of the progress advantage across three different applications: test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families. Across all settings, it consistently outperforms confidence-based baselines and, despite requiring no task-specific training, surpasses dedicated trained reward models. We complement these results with deeper analyses on characteristics of progress advantage, offering practical guidance for adoption in real-world agentic systems. 

---
# A cross-process welding penetration status prediction algorithm based on unsupervised domain adaptation in laser and TIG welding 

**Authors**: Sen Li, Haichao Cui, Chendong Shao, Yaqi Wang, Xinhua Tang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26078)  

**Abstract**: Supervised deep learning has been widely used for weld penetration state classification; however, its performance often degrades significantly under domain shift, such as when transferring models between welding processes with distinct physical mechanisms:for instance, from arc-dominated tungsten inert gas (TIG) welding to keyhole-based laser welding. To overcome this limitation, we propose an unsupervised domain adaptation (UDA) framework integrated with a gradual source domain expansion (GSDE) strategy. Evaluated on dedicated TIG and laser welding datasets, our approach achieves high accuracy in both same-process and cross-process transfer tasks. Specifically, it attains average accuracies of 90.65% on TIGFH and 90.72% on LSPS in same-process settings, surpassing a supervised baseline by 35.83% and 38.87%, respectively. More notably, in cross-process scenarios, it reaches 80.48% for TIG to Laser and 81.13% for Laser to TIG, improving upon the baseline by 43.39% and 43.40%. UMAP visualizations verify that the model learns domain-invariant features while maintaining discriminative class boundaries. This method considerably lowers the relabeling cost for new welding processes and enhances the versatility of intelligent monitoring across different welding systems. 

---
# Model Forensics: Investigating Whether Concerning Behavior Reflects Misalignment 

**Authors**: Aditya Singh, Gerson Kroiz, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2606.26071)  

**Abstract**: A central goal of safety research is determining whether a model is misaligned. Prior work has largely focused on detecting concerning behavior. But behavior alone does not establish misalignment: a concerning action can arise from benign causes such as confusion. This motivates model forensics: investigating whether the action was driven by malign intent. In this paper, we propose a baseline protocol for model forensics consisting of two steps, iterated as needed. First, we read the chain of thought (CoT) to generate hypotheses about what drives model behavior. Second, we make edits to the prompt or environment to test these hypotheses. While the CoT is not always faithful, it is a rich source of unsupervised insight that can guide the collection of more rigorous evidence. To evaluate our protocol, we create a suite of six agentic environments where models exhibit concerning behavior, and apply it to each. We establish that Kimi K2 Thinking takes shortcuts due to a genuine disposition towards low-effort actions, by showing this hypothesis successfully predicts its behavior. Through counterfactual experiments, we show DeepSeek R1 deceives out of a desire to be consistent with a previous instance of itself. Our methods nonetheless leave significant room for refinement. For example, when we test whether Kimi K2 Thinking believes it is violating user intent, we find no evidence of such a belief, but without positive controls we cannot confirm our tests would detect it. Overall, we find our simple protocol provides a strong baseline that we hope future work will improve upon. More broadly, our work is a concrete step in developing the growing field of model forensics. 

---
# A welding penetration prediction model for laser welding process based on self-supervised learning using physics-informed neural networks 

**Authors**: Sen Li, Xiaoying Liu, Xiaojian Xu, Chendong Shao, Yaqi Wang, Ling Lan, Xinhua Tang, Haichao Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.26059)  

**Abstract**: The laser welding full-penetration is of critical importance, as it constitutes one of the fundamental factors in achieving defect-free welded joints. Accurate prediction of the penetration state is therefore essential for ensuring weld quality. To this end, this paper introduces SimPhysNet, a novel algorithm that achieves high classification accuracy in laser welding penetration prediction using only a limited number of labelled images. This approach effectively overcomes the limitations of supervised learning classification algorithms, which are hindered in industrial applications by their dependence on extensive, high-quality labelled data. The core of SimPhysNet is a unique self-supervised learning paradigm that embeds physical priors into a contrastive learning framework. By incorporating a physics-informed neural network (PINN), the model is guided to extract physically meaningful features of the molten pool and keyhole from a large set of unlabelled data, while three image augmentation tasks further enhance its generalization capabilities. Subsequently, a few-shot learning strategy, based on prototypical networks, enables robust classification by constructing class representations from a minimal set of labelled images. Experimental results demonstrate that SimPhysNet achieves a classification accuracy of 96.06% using only 200 labelled images (approximately 5% of the total labelled dataset), which is comparable to the performance of conventional supervised learning algorithms that utilize the entire labelled dataset. This work presents a new, efficient, and highly accurate method, providing the way for the intelligent automation of laser welding. 

---
# Natural Ungrokking: Asymmetric Control of Which Rules Survive Pretraining 

**Authors**: Juliana Li, Diya Sreedhar  

**Link**: [PDF](https://arxiv.org/pdf/2606.26050)  

**Abstract**: Midway through an ordinary pretraining run, a small language model learns the pronoun-gender rule: cued with a girl's name ("Sue cried because"), it resolves the next pronoun to she, generalizing to held-out probes (0.94 by step 925). By step 3,500 the same model scores near zero on the same probes, although the rule's evidence is still in the training data. We call this within-run reversal natural ungrokking: the corpus decides, with no trace in the loss curve, which learned rules a model keeps.
Which rules survive is predictable from one corpus statistic: how often the training stream shows the rule winning. Across un-intervened runs (two corpora, three budgets, three seeds), support frequency decides a rule's fate; the data-to-parameter ratio only modulates how deeply a doomed rule falls. The same emerge-then-collapse dynamics appear in public Pythia checkpoints, collapse depth ordered by model scale as predicted. The forgetting is a displacement: a competing surface pattern out-competes the rule, and the log-probability margin between them crosses zero within 100 training steps of the behavioral collapse.
Control over this fate is asymmetric: the same edit that destroys a rule on demand cannot restore it. Flipping support to counter-evidence in place kills the rule with monotone dose-response in two unrelated rules; but injecting support back, even to 450 times the level that naturally sustains it, buys no recovery. Every confirmatory threshold and prediction was pre-registered before the data it governed was read. 

---
# TriViewBench: Controlled Complexity Scaling for Multi-View Structural Reasoning in MLLMs 

**Authors**: Yu-Yang Chen, Lan-Zhe Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.26029)  

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate strong performance on standard visual question answering benchmarks, yet their scalability under controlled structural complexity remains poorly understood. We introduce TriViewBench, a controlled three-view visual reasoning benchmark constructed from synthetic 3D scenes with explicitly parameterized object count and occlusion. The benchmark contains 1,923 scenes and over 14K Question-Answer (QA) pairs organized into four complexity levels and three reasoning categories: Local Decision, Object Counting, and Global Recovery. We evaluate 18 open- and closed-source MLLMs under a unified prompting protocol. All 18 models exhibit an identical capability hierarchy without exception (Local Decision > Object Counting > Global Recovery), and performance degrades monotonically with complexity: Local Decision tasks decline modestly (12.11% relative drop), while Object Counting degrades substantially (59.14%) and Global Recovery collapses severely (80.02%). Error analysis on Object Counting reveals two mechanistically independent failure modes: single-view tasks are dominated by undercounting due to occlusion blindness, whereas the multi-view task reverses to overcounting due to cross-view identity confusion. Chain-of-Thought (CoT) prompting yields near-zero overall benefit ($\Delta = -0.16\%$) and its effect on Global Recovery is strongly capability-gated, suggesting that the bottleneck lies in cross-view spatial representation rather than reasoning strategy. These findings reveal fundamental scalability limitations in current MLLMs and position TriViewBench as a controlled diagnostic framework for analyzing structural reasoning failures. 

---
# Can Trustless Agents Be Trusted? An Empirical Study of the ERC-8004 Decentralized AI Agent Ecosystem 

**Authors**: Xihan Xiong, Zelin Li, Wei Wei, Qin Wang, William Knottenbelt, Zhipeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26028)  

**Abstract**: As autonomous AI agents increasingly transact across organizational boundaries, a fundamental trust challenge emerges: how can an agent assess whether an unknown counterpart is trustworthy? The ERC-8004 protocol addresses this challenge with the first permissionless trust layer for AI agent economies, built around three on-chain registries for Identity, Reputation, and Validation. Despite its rapid adoption, the protocol has not been studied empirically, leaving it unclear whether the information it records provides a trustworthy basis for decision-making. To address this gap, we present the first empirical study of ERC-8004 across three chains: Ethereum, BNB Smart Chain (BSC), and Base, covering the period from protocol deployment through May 13, 2026. We crawl on-chain Identity and Reputation events, off-chain files, and x402 payment transactions.
On the identity side, we find that most registrations are placeholders rather than active agents, with only a small fraction (3%, 4%, and 15% across Ethereum, BSC, and Base) exposing a valid ERC-8004 registration file with at least one live service endpoint. On the reputation side, we show that the Registry, as currently deployed, cannot function as a trust signal: values are not commensurable, feedback records are rarely grounded in verifiable interactions, and reputation can be manipulated at minimal cost. Consistent with these design weaknesses, we find that a substantial fraction of reviewers (73.6%, 59.2%, and 90.6% across Ethereum, BSC, and Base) exhibit coordinated Sybil behavior. After removing Sybil-flagged feedback, 15.5%, 72.3%, and 89.4% of rated agents, respectively, are left with no valid feedback. We then turn these findings into concrete recommendations for future revisions of ERC-8004. Our study yields actionable protocol-design implications and establishes an empirical baseline for research on AI agent markets. 

---
# Privacy Vulnerabilities of Attention Layers in Tabular Foundation Models and Protection of High-Risk Queries 

**Authors**: Tânia Carvalho, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2606.26021)  

**Abstract**: Tabular foundation models are commonly assumed to present limited privacy concerns as they are often pre-trained on large collections of synthetic data. However, these models leverage in-context learning, where sensitive records may be provided directly at inference time as labelled context examples. In this paper, we demonstrate that predictions generated via the attention mechanism leak sufficient information to enable effective Membership Inference Attacks (MIAs). To highlight this vulnerability, we propose AMIA (Attention-based Membership Inference Attack), a shadow-model-free attack that exploits the concentration of transformer attention patterns. Our results show that attention mechanisms reveal strong membership signals, which exceed classical confidence-based attacks, achieving an average gain of 7.7\%, specially in low false-positive regimes. To mitigate this risk, we introduce an inference-time defence inspired by $k$-anonymity principles. This approach reduces the uniqueness of context-key representations without introducing random noise or retraining the model. By targeting only high-risk queries identified through AMIA scores, the defence substantially reduces membership leakage of this attack by an average of 50\% and 25\% against confidence-based attacks, while preserving predictive utility with only 3.9\% performance degradation. Beyond showing that context examples are vulnerable, we further demonstrate that fine-tuning introduces an additional source of privacy risk. In particular, samples whose prediction confidence increases after fine-tuning become more susceptible to MIAs, indicating that fine-tuning can amplify memorisation and expose sensitive training information through confidence shifts. 

---
# FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation 

**Authors**: Shuyi Zhang, Yunfan Lou, Hongyang Cheng, Yichen Guo, Chuyao Fu, Yaoxu Lyu, Xiaojie Zhang, Haoran Li, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.26006)  

**Abstract**: Vision-Language-Action (VLA) models are often constrained by the imitation ceiling imposed by sub-optimal data. While Reinforcement Learning (RL) fine-tuning can surpass this limit, it is notoriously sample inefficient. This challenge arises from two core issues: (1) catastrophic initial unlearning due to an unstable Q-function and (2) inefficient policy updates caused by low-quality exploration data, often forcing a reliance on costly human interventions. We introduce FORCE, a 3-stage framework that stabilizes fine-tuning by tackling both issues. FORCE first incorporates a Value-Calibrated Warm-Up phase, utilizing on-policy rollouts to mitigate the distributional shift of the Q-function. Subsequently, during the online stage, this calibrated Q-function acts as a filter for both the policy's own action proposals and expert data, ensuring only high-value actions are used for the policy update. We evaluate FORCE on various simulation and real-world tasks, and the result shows that FORCE achieves a 79% absolute improvement in success rates and outperform prior RL methods by 10%, while accelerating training by 32.5%. Critically, it mitigates the common success rate drop and achieves this robust performance without human intervention, marking a significant step towards deploying capable and autonomous robotic agents. 

---
# Hierarchical Reinforcement Learning for Neural Network Compression (HiReLC): Pruning and Quantization 

**Authors**: Kamar Hibatallah Baghdadi, Kawther Guoual Belhamidi, Sara Belhadj, Aissa Boulmerka, Nadir Farhi  

**Link**: [PDF](https://arxiv.org/pdf/2606.26002)  

**Abstract**: We present HiReLC, a hierarchical ensemble-reinforcement learning framework for automated joint quantization and structured pruning of deep neural networks. The framework decomposes the compression search across two levels of abstraction: low-level agents (LLAs) operate independently per block, selecting per-kernel configurations over a multi-discrete action space spanning bitwidth, pruning keep-ratio, quantization type, and granularity, while high-level agents (HLAs) coordinate global budget allocation via ensemble voting guided by Fisher Information-based sensitivity estimates. To mitigate the computational cost of policy evaluation, an iterative active learning loop interleaves surrogate-guided RL optimization with post-compression fine-tuning, using a lightweight MLP surrogate to amortize expensive evaluations and a logit-MSE proxy during cold-start. The surrogate is used for reward shaping rather than as a replacement for final post-compression evaluation. The controller is architecture-agnostic by design, with a modular layer abstraction decoupling the RL environment from the underlying network topology. Experiments across Vision Transformer and CNN benchmarks demonstrate effective parameter-storage compression ratios of 5.99 - 6.72$\times$ with a 3.83 % gain in one setting and 0.55 - 5.62 % accuracy drops elsewhere, supporting hierarchical policy decomposition and sensitivity-aware guidance as practical design choices for joint neural network compression. 

---
# Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games 

**Authors**: Sam Ganzfried  

**Link**: [PDF](https://arxiv.org/pdf/2606.25997)  

**Abstract**: There has been significant recent progress in algorithms for approximation of Nash equilibrium in large two-player zero-sum imperfect-information games and exact computation of Nash equilibrium in multiplayer strategic-form games. While counterfactual regret minimization and fictitious play are scalable to large games and have convergence guarantees in two-player zero-sum games, they do not guarantee convergence to Nash equilibrium in multiplayer games. Recently, an approach has been presented for exact computation of Nash equilibrium in multiplayer imperfect-information games that solves a quadratically constrained program based on a nonlinear complementarity problem formulation derived from the sequence-form game representation. This formulation was solved using Gurobi's nonconvex quadratic solver, which employs spatial branch-and-bound to iteratively refine variable bounds by solving convex relaxations of bilinear terms via McCormick envelopes. During presolve, Gurobi introduces auxiliary variables and, in some cases, binary variables, leading to an internal MIQCP reformulation. This approach was demonstrated to outperform prior algorithms from the Gambit software suite and quickly solve three-player Kuhn poker after removal of dominated actions; however, the algorithm was not able to solve the full version of the game within 24 hours. In this paper, we derive finite bounds on slack and multiplier variables in the nonlinear complementarity formulation. These bounds strengthen the convex relaxations used within spatial branch-and-bound and lead to substantial computational improvements. We demonstrate the impact of the proposed bounds on exact Nash equilibrium computation in three-player Kuhn poker. 

---
# SpeechEQ: Benchmarking Emotional Intelligence Quotient in Socially Aware Voice Conversational Models 

**Authors**: Liang-Yuan Wu, Zih-Ching Chen, Tongshuang Wu, Chao-Han Huck Yang, Hua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25990)  

**Abstract**: As multimodal conversational systems increasingly engage in spoken interaction, their ability to navigate paralinguistic social cues has become a critical bottleneck for natural human-AI communication. However, existing evaluations of machine emotional intelligence assess reasoning exclusively through isolated text or passive acoustic perception, overlooking the complex cross-modal reasoning required for active, multi-turn dialogue. We introduce \textsc{SpeechEQ}, a comprehensive framework designed to evaluate the sociolinguistic reasoning of Speech-Language Models (SLMs). The framework includes a validated dataset of 2,265 dialogues across 15 Emotional Quotient (EQ) subscales grounded in EQ-i 2.0 theory, along with a multi-turn evaluation protocol measured by our proposed Spoken EQ (SEQ) score inspired by human EQ assessments. Experiments show limitations in how both existing Speech Emotion Recognition and end-to-end Speech-Language Models understand and apply paralinguistic cues through speech. While end-to-end architectures outperform cascaded systems, \textsc{SpeechEQ} reveals that current multimodal models remain bottlenecked by a text-reliant ``modality shortcut,'' an alignment-induced ``safety trap,'' and ``contextual amnesia,'' highlighting the barriers to truly emotionally aware AI. Our benchmark can be accessed at this https URL and demo page at this https URL 

---
# Weave of Formal Thought 

**Authors**: Alexandre Bouayad  

**Link**: [PDF](https://arxiv.org/pdf/2606.25987)  

**Abstract**: Large language models (LLMs) attain remarkable surface fluency on code, yet they neither formally guarantee the syntactic validity of their output nor leverage the hierarchical structure defining the target language. While existing constrained-decoding frameworks address the former, they operate under rigid assumptions that preclude critical lexical mechanisms -- including context-sensitive lexing, maximal-munch tokenization, and keyword extraction -- and only approximate vocabulary masking, sacrificing completeness. For the latter, code LLMs typically inject grammatical structure via predetermined policies rather than learning which structural information to expose. In this work, we introduce Weave of Formal Thought (WoFT), a paradigm uniting rigorous syntactic validation with learned structural representations. First, we present a formal engine and constrained decoder that is sound and complete with respect to the full Tree-sitter specification. By augmenting generalized LR (GLR) parsing with a speculative-lexing construction that maintains concurrent lexer-state hypotheses synchronized with a GLR graph-structured stack, our decoder admits every subword token extending to a valid program prefix and rejects all others. Second, we present a latent-variable fine-tuning method training the language model to interleave non-terminal grammar symbols directly into generation. Utilizing the reweighted wake-sleep (RWS) algorithm to optimize the importance-weighted evidence lower bound (IW-ELBO) of the surface text, the model learns to selectively retain formal derivations as an adaptive structural scratchpad. For Python, fine-tuning StarCoder2-3B with our RWS objective reduces per-token cross-entropy by 14.3% relative to a text-only SFT baseline, demonstrating that discretionary latent syntax recovers critical structural information that flat autoregressive training discards. 

---
# Multi-Agent Goal Recognition with Team- and Goal-Conditioned Reinforcement Learning and Factorized Branch-and-Bound 

**Authors**: Thiago Thomas, Gabriel de Oliveira Ramos, Felipe Meneguzzi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25978)  

**Abstract**: Multi-agent goal recognition asks an observer to jointly infer which agents act together and what each team is trying to achieve, so the hypothesis space grows combinatorially with the number of team partitions and goals per team. Real applications such as drone surveillance and collaborative robotics expose only the agents' trajectory, which forces the observer to rank team-goal hypotheses from behavior alone. Multi-Agent Goal Recognition with Branch-and-Bound (MAGR-BB) addresses this setting with a shared team- and goal-conditioned policy used as the scoring model inside a factorized branch-and-bound search. On a controlled multi-agent Blocksworld benchmark, MAGR-BB returns the same top-ranked hypothesis as exhaustive search throughout the trajectory while cutting hypothesis materialization by orders of magnitude and reducing cumulative recognition runtime substantially. 

---
# Helpful or Harmful? Evaluating LLM-Assisted Vulnerability Patching via a Human Study 

**Authors**: Giulian Biolo, Michael Tezza, Yuanjun Gong, Fabio Massacci  

**Link**: [PDF](https://arxiv.org/pdf/2606.25973)  

**Abstract**: Software vulnerability remediation is a cognitively demanding task that requires specialized security expertise often lacking in general developers. In the meantime, Large Language Models (LLMs) assisted tools show potential in vulnerability detection, location, and repair tasks. [Hypothesis:] While LLM-assistance is hypothesized to accelerate patching, it also risks introducing hallucinations or insecure code, leading to a higher likelihood of generating superficial repairs that bypass the standard functionality checks but fail the security validation. [Objective:] We aim to present an empirical experiment, unveiling the capability of LLM-assisted vulnerability patching compared to manual debugging on human participants in real-world scenarios. [Method:] We plan to conduct a controlled experiment using a Balanced Crossover design. For that, we have developed a WebApp for code execution and integrated hidden Ghost Tests to verify patch integrity beyond visible functional requirements. The experiment involves training and evaluation scenarios. The remediation speed, remediation efficacy for both standard functionality tests and security tests, and participant perception will be evaluated. [Pilot Study:] A pilot experiment with a small sample of participants has been conducted, providing insights for the following study. 

---
# SE-AGCNet: An End-to-End Framework for Joint Speech Enhancement and Loudness Control in Meeting Scenarios 

**Authors**: Jinming Zhang, Wei Rao, Xionghu Zhong, Eng Siong Chng  

**Link**: [PDF](https://arxiv.org/pdf/2606.25959)  

**Abstract**: Conventional audio pipelines typically treat speech enhancement (SE) and automatic gain control (AGC) as discrete modules, which often limits overall performance. For instance, applying AGC before SE may inadvertently amplify background noise, while prioritizing SE tends to over-suppress low-volume speech. To address these limitations, we propose SE-AGCNet, an end-to-end framework that jointly optimizes SE and AGC. Tailored for meeting scenarios with significant volume variations, SE-AGCNet leverages the synergy between the two tasks: SE preserves quiet speech, thereby facilitating effective volume adjustment by the AGC component. Furthermore, we propose a specialized data simulation pipeline, SE-AGC-DataGen, and incorporate standardized loudness evaluation metrics: integrated loudness (LUFS), short-term loudness (St LUFS), and LRA. Experiments show that SE-AGCNet consistently achieves target loudness while improving speech quality and ASR accuracy over competitive baselines. 

---
# Pulmonary Embolism Risk Stratification from CTPA and Medical Records: Vascular Graphs Are Not All You Need 

**Authors**: Nathan Painchaud, Tristan Habémont, Morgane des Ligneris, Allan Serva, Pierre Croisille, Laurent Bertoletti, Thomas Lampert, Johannes F. Lutzeyer, Odyssée Merveille  

**Link**: [PDF](https://arxiv.org/pdf/2606.25956)  

**Abstract**: Risk stratification for pulmonary embolism (PE) is critical for clinical decision-making. Stratification guidelines are based on patient medical records, parameters measured from computed tomography pulmonary angiography (CTPA), and blood tests. However, blood tests are often missing in routine practice. This work studies whether state-of-the-art models can accurately classify risk stratification from only medical records and biomarkers extracted from CTPA images. We benchmark different approaches to combine medical records and cardiac biomarkers with rich pulmonary vascular information; we add vascular biomarkers to tabular models and apply graph neural networks (GNNs) on the vascular tree's intrinsic graph representation. We use a private dataset (n=353) with uniquely complete data for PE risk stratification. Our results show that, among global features, medical records and cardiac biomarkers are the most significant predictors, while vascular biomarkers do not further improve stratification. Even more surprising, even GNNs on vascular graphs fail to outperform strong tabular baseline on global features. We consider hypotheses, on both models and data, that could explain this suboptimal performance. Our investigation suggests that, counter-intuitively, vascular graphs might hold no discriminative information for PE risk stratification. Code is available from this https URL. 

---
# Measurable Majorities Are Not Finitely Axiomatizable 

**Authors**: Lawrence S. Moss, Arthur Paul Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25954)  

**Abstract**: This theoretical note studies the finite axiomatizability of strict majority reasoning in finite social decision frames. Moss and Pedersen (2026) <doi: https://doi.org/10.48550/arXiv.2606.23853> introduce a coherence criterion that characterizes exactly when qualitative majority judgments are representable by a finitely additive measure. The question addressed here is whether that coherence criterion can be replaced, in the finite setting, by any bounded finite fragment. We prove that it cannot. For every $k\ge 1$, we construct a maximal standard frame whose shortest coherence violation has length exactly $2k+2$. Hence there is no uniform finite bound on the incoherence index of social decision frames, resolving Conjecture 5.7 stated by Moss and Pedersen (2026). The construction is geometric, in the sense that it proceeds via orthogonality and dimension in rational vector spaces, and self-contained: it isolates a symmetric family of half-sized voting blocs and extends it to a maximal frame in which every shorter balanced obstruction is excluded. Along the explicit infinite sequence of universe sizes obtained in the construction, this also establishes the middle-layer family predicted by Conjecture B.25 by Moss and Pedersen (2026). Together with the soundness and completeness theorem for the Moss-Pedersen minimal logic for strict majorities, this establishes that measurable social decision frames are not finitely axiomatizable in that language. 

---
# Explainable Control Framework (XCF) based on Fuzzy Model-Agnostic Explanation and LLM Agent-Supported Interface 

**Authors**: Faliang Yin, Hak-Keung Lam, David Watson  

**Link**: [PDF](https://arxiv.org/pdf/2606.25941)  

**Abstract**: Increasing demand for precise and reliable control in complex scenarios has led to the development of increasingly sophisticated controllers, including data-driven approaches employing closed box models and mathematically rigorous yet complex designs. This complexity highlights the needs for explainable control that can provide human-understandable insights into controller behavior. In this paper, an explainable control framework (XCF) along with supporting algorithms and user interface are proposed to explain how controllers determine their control actions and their underlying working mechanism. The novel contributions of this work are threefold: First, the XCF is designed to provide model-agnostic explanations for controllers in closed-loop systems and can optionally refine local explanations by system response dynamics. Second, a novel explanation method, hierarchical fuzzy model-agnostic explanation for control systems (HFMAE-C), is proposed based on the designed framework. The HFMAE-C employs a fuzzy logic system to approximate the controller's behavior and system dynamics, providing sample, local, domain and universe level explanations via IF-THEN rules revealing the controller's decision logic and salience values quantifying the contribution of system states to control actions. Third, a large language model agent-supported user interface is developed to automatically analyze user requirements, select appropriate algorithms, interpret the generated explanations to a natural language report, and provide interactive consultation. Case studies on inverted pendulum system and Turtlebot obstacle avoidance demonstrate the effectiveness of the proposed method through simulated user experiments and quantitative comparisons with mainstream explainable control approaches. 

---
# Overview of HIPE-2026: Person-Place Relation Extraction from Multilingual Historical Texts 

**Authors**: Juri Opitz, Maud Ehrmann, Corina Raclé, Andrianos Michail, Matteo Romanello, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2606.25935)  

**Abstract**: Was this person ever at that place, and if so, when? Answering such questions from noisy, multilingual historical documents is the central challenge of HIPE-2026, the third edition of the HIPE evaluation series. Moving from named entity recognition and linking (HIPE-2020, HIPE-2022) to reasoning about relationships between entities, HIPE-2026 targets two temporally grounded relation types: $at$, indicating that a person was present at a location at some point prior to a document's publication date, and $isAt$, indicating presence contemporaneous with that date. This paper presents the results of the evaluation campaign, which confronted 17 participating teams with the challenges of historical language variation, OCR noise, and indirect contextual cues across three languages: French, German, and English. The datasets include historical newspaper text from the nineteenth and twentieth centuries, as well as a surprise-domain generalization set drawn from early modern French literary texts. A distinctive feature of HIPE-2026 is its three-fold evaluation framework, which assesses predictive accuracy, computational efficiency, and cross-domain generalization, reflecting the practical demands of large-scale historical document processing in the cultural heritage domain. Across more than 40 submitted runs, results reveal a wide range of strategies, from state-of-the-art large language models to lightweight task-specific classifiers, and highlight the trade-offs between accuracy, efficiency, and robustness inherent to historical relation extraction at corpus scale. System descriptions, datasets, and findings are presented and discussed, offering a detailed picture of the current state of temporally grounded relation extraction for historical documents. 

---
# Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and Synthetic Data 

**Authors**: Shangkun Li, Jie Xu, Yi Guo, Zeju Li, Yuanyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25894)  

**Abstract**: Medical vision-language models typically generate diagnoses through single-pass inference without indicating which image regions support their conclusions. This lack of spatial grounding limits clinical utility: outputs cannot be audited, and models may hallucinate findings on normal scans. We present BrReMark (Brain Rethink via ROI Marking), a framework that introduces explicit region marking into brain MRI diagnosis. The model first generates hypotheses about potential abnormalities and grounds them through explicit bounding box marking, then verifies conclusions by re-examining the marked evidence. Training combines supervised fine-tuning on structured reasoning trajectories with reinforcement learning using a composite reward over localization accuracy and diagnostic reasoning. Furthermore, we integrate a domain randomization-based pathology synthesis augmentation strategy to improve the model's generalizability to out-of-distribution (OOD) data. On internal benchmark, BrReMark improves mAP50 from 0.74% to 37.54% compared to the base model, while achieving 21.57% Clinical F1 and 45.26% diagnostic accuracy. On NOVA OOD benchmark, it also achieves competitive overall performance with a 45.7% reduction in false positives compared to the state-of-the-art, indicating reduced hallucination on rare pathologies. These findings suggest that explicit hypothesis-verification grounding is a practical path toward trustworthy open-ended brain MRI diagnosis across both in-distribution and OOD settings. 

---
# AI-Assisted Computational Reproducibility on the FABRIC Testbed 

**Authors**: Komal Thareja, Paul Ruth, Berent Aldikacti, Michael Zink  

**Link**: [PDF](https://arxiv.org/pdf/2606.25879)  

**Abstract**: Computational reproducibility remains difficult despite being central to scientific research. In this paper, we show how the international FABRIC testbed, combined with large language model (LLM) coding assistants through LoomAI, can simplify reproducing published experiments across multiple domains. We reproduced three case studies on FABRIC, covering BBR-family congestion-control evaluations, LAMMPS molecular dynamics scaling benchmarks on a CPU-only MPI cluster, and stress protein homeostasis genomics pipelines.
Rather than focusing only on matching numerical outputs, we evaluate whether the reproduced experiments support the same scientific conclusions as the original studies. The AI assistant was effective in setting up the environment, adapting code, and debugging, but struggled with the analysis stages that lacked clearly defined workflows, which required human guidance to establish execution order and data dependencies.
Across the case studies, the AI-assisted workflow reduced reproduction effort by roughly 4--6 times. We conclude with practical recommendations for improving AI-assisted reproducibility on research testbeds. 

---
# AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance Evaluation in Sponsored Search 

**Authors**: Md Omar Faruk Rokon, Shasvat Desai, Hong Yao, Kuang-chih Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.25871)  

**Abstract**: How can we generate high-quality relevance annotations at scale without the cost and delays of human labeling? Relevance annotations are the backbone of search ranking systems which is needed for training data preparation, NDCG evaluation, and root cause analysis. However, human annotation is slow and off-the-shelf LLMs suffer from accuracy on domain-specific tasks. We propose a calibrated model cascade, a systematic approach for cost-efficient offline relevance annotation by routing queries through progressively larger fine-tuned classifiers. Our central insight is that accuracy and cost are orthogonal optimizations: domain-specific fine-tuning drives accuracy, cascading drives cost, and per-class isotonic calibration adds a small but reliable gain on top. Our contribution is threefold: (a) we decompose the gains and show that fine-tuning contributes 20 accuracy points while cascading is approximately accuracy-neutral but halves compute cost, (b) we introduce per-class isotonic calibration as one component of the cascade, contributing a small but statistically significant gain (+0.6 points over the strongest calibration baseline), and (c) we validate the system in production across six offline use cases, processing 150M+ annotations and enabling faster experimentation cycles. Our work is a building block for scalable, high-quality offline annotation pipelines in search and advertising systems. 

---
# Color Matters: Trigger Color Affects Success in Federated Backdoor Attacks 

**Authors**: Kavindu Herath, Joshua C. Zhao, Saurabh Bagchi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25858)  

**Abstract**: Federated learning is vulnerable to backdoor attacks in which malicious clients inject poisoned updates while preserving benign-task performance. In this paper, we study a semantics-driven backdoor mechanism in which attackers use natural visual accessories as triggers and manipulate only the trigger color while keeping the attack pipeline fixed. Our framework considers semantic trigger objects such as masks and sunglasses, instantiated in black and white variants, and evaluates their effect in a controlled federated learning setting. Malicious clients construct poisoned samples by applying a trigger to source-class images and relabeling them to an attacker-chosen target class, while benign clients train only on clean data. We analyze this mechanism under both a standard poisoning objective and a stronger SABLE-based objective that combines clean classification loss, triggered target loss, feature-separation loss in the penultimate representation space, and regularization to keep malicious updates close to the global model. This design enables the attack to remain effective while reducing excessive update drift. Experiments on a four-class CelebA hair-color task show that trigger color significantly changes attack success rate even when trigger semantics, placement, and poisoning budget are unchanged. White triggers are more effective for attacks targeting the blond class, whereas black triggers perform better for attacks targeting the black class. The same trend persists under robust aggregation, showing that trigger color is a meaningful factor in the operation, persistence, and evaluation of semantic backdoor mechanisms in federated learning. 

---
# Semantic Consistency Policy Optimization for Reinforcement Learning of LLM Agents 

**Authors**: Peng Xu, Sijia Chen, Junzhuo Li, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25852)  

**Abstract**: Group-based reinforcement learning effectively post-trains LLM agents for long-horizon, sparse-reward tasks by deriving step-level credit from trajectory outcomes. However, this ties a step's credit to its rollout's final outcome: semantically near-identical intermediate steps receive opposite credit depending on whether their trajectory eventually succeeded or failed. Such semantic credit inconsistency sends conflicting gradients to similar actions and wastes the partially-correct progress inside failed rollouts. Motivated by this, we propose Semantic Consistency Policy Optimization (SCPO), a value-free reward-shaping method that mitigates this inconsistency by recovering step-level credit from successful siblings in the same rollout group. Concretely, SCPO scores each failed step against a successful sibling and adds positive step-level credit for new progress along that sibling. On ALFWorld and WebShop, SCPO matches or exceeds strong group-based baselines, reaching 93.7+/-4.1 percent success on ALFWorld and 74.8+/-2.0 percent on WebShop at 1.5B parameters, with gains concentrated on the hardest multi-step tasks. 

---
# Edges Before Embeddings: A Confidence-Aware Blur Gate for Vision-Language Pipelines 

**Authors**: Duy Tran Thanh  

**Link**: [PDF](https://arxiv.org/pdf/2606.25838)  

**Abstract**: Production vision pipelines silently degrade on blurry input, wasting compute on downstream OCR, retrieval, and vision-language model (VLM) calls that cannot recover a usable output. We present MagikaDocumentFromPixel, a lightweight, CPU-friendly image quality gate that classifies a single image as sharp, blurred, or uncertain in roughly 7 ms on a single CPU core. The contributions are (i) a recipe selected from a 46-configuration, 8-sweep empirical search that isolates input resolution as the dominant lever and shows architecture capacity only pays off at >= 384 px; (ii) a confidence-aware routing formalism grounded in classical selective prediction; (iii) the Edge Prior Module (EPM), a Laplacian-magnitude auxiliary input channel that gives the network direct access to the spectral evidence that classical blur heuristics rely on and that lifts test F1 by +1.3 points in a matched-env comparison; and (iv) an observation that the gate is one instance of a recurring design pattern that appears independently in Magika content-type detection, risk-controlled OCR with VLMs, and DocVLM. The final recipe MobileNetV3-Large with the EPM trained at 384x384 on paired GoPro Large frames, evaluated with 5-scale test-time augmentation reaches F1 = 0.9803 (AUC 0.9989) with a 17 MB ONNX artifact, improving over our fixed-scale baseline on the same hardware (F1 = 0.9672) by +1.31 points. We are explicit about limitations: results are on a single motion-blur distribution, numbers are from a single seed, and calibration is qualitative rather than measured. 

---
# MiniOpt: Reasoning to Model and Solve General Optimization Problems with Limited Resources 

**Authors**: Ke Zhao, Zixiang Di, Hong Qian, Xiang Shu, Yaolin Wen, Qitao Shi, Bingdong Li, Xingyu Lu, Xiangfeng Wang, Jun Zhou, Ke Tang, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25832)  

**Abstract**: Achieving strong optimization generalization across diverse optimization problems while requiring limited training resources remains a challenging problem for optimization-oriented large language models (LLMs). Existing approaches typically rely on large-scale supervised datasets, costly reasoning annotations, and expensive intermediate step verification, resulting in substantial training overhead. To address these challenges, we propose MiniOpt, a reinforcement learning framework that learns to solve optimization problems through an "reasoning-to-model-and-solve" paradigm. MiniOpt decomposes optimization reasoning into structured optimization modeling and executable solver generation. Building upon this paradigm, we introduce OptReward, a reward function with hierarchical score structure that jointly evaluates formulation and solution, enabling effective policy learning without expert demonstrations. We further develop an optimization-oriented policy optimization strategy that improves exploration efficiency and stabilizes reinforcement learning for compact models. Extensive experiments show that MiniOpt-3B exhibits strong optimization generalization across various optimization types, problem scenarios, and task domains. For models with fewer than 10B parameters, MiniOpt series achieves the highest average solving accuracy (SA). For models with more than 10B parameters, MiniOpt still shows competitive performance. These results suggest that optimization-oriented reward design and reinforcement learning provide an effective pathway for developing compact optimization-specialized language models with strong optimization generalization capabilities. The code is available at this https URL. 

---
# SARA: Unlocking Multilingual Knowledge in Mixture-of-Experts via Semantically Anchored Routing Alignment 

**Authors**: Tianyu Dong, Yangyang Liu, Jiang Zhou, Xinwei Wu, Xiaohu Zhao, Hao Wang, Heng Liu, Linlong Xu, Longyue Wang, Weihua Luo, Shaolin Zhu, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2606.25821)  

**Abstract**: Sparse Mixture-of-Experts (MoE) architectures have emerged as an increasingly influential paradigm as they offer a strategic balance between parameter scalability and computational efficiency. However, low-resource languages, which suffer from a scarcity of high-quality training data, often have their tokens routed to different experts than those predominantly activated by high-resource inputs, which limits cross-lingual expert sharing. This cross-lingual routing divergence consequently hinders their efficacy in multilingual contexts. To address this issue, we propose SARA (Semantically Anchored Routing Alignment), a framework designed to transfer specialized capabilities from high-resource languages as anchors to low-resource languages. SARA explicitly aligns the routing distribution of multilingual inputs with high-resource semantic anchors using a symmetric Jensen-Shannon (JS) divergence constraint. Unlike traditional distillation methods that operate on output logits, SARA directly aligns the internal routing distributions of MoE layers, encouraging mechanistic consistency in expert selection across languages. We conduct experiments on 2 LLMs across 5 low-resource languages and 3 benchmarks. Experiment results demonstrate that SARA outperforms standard instruction tuning, e.g., +0.8% on Qwen3-30B-A3B and +1.2% on Phi-3.5-MoE-instruct on Global-MMLU. Further analyses show that SARA effectively addresses performance bottlenecks in low-resource languages, providing a scalable pathway to enhance multilingual capabilities in sparse architectures. 

---
# Do Encoders Suffice? A Systematic Comparison of Encoder and Decoder Safety Judges for LLM Adversarial Evaluation 

**Authors**: Han Jeon, Shiv Medler, Joseph Voyles, Matt Wood  

**Link**: [PDF](https://arxiv.org/pdf/2606.25782)  

**Abstract**: With the widespread adoption of large language models (LLMs) in chatbots and everyday applications, companies increasingly need guardrails that are effective while remaining low-cost and low-latency. Safety evaluation of LLM outputs has generally relied on LLM-based judges, which can be effective but are often slow and expensive to deploy at scale.
In this paper, we evaluate whether fine-tuned modern encoder classifiers from the ModernBERT family, including ModernBERT and Ettin, can reliably identify harmful LLM outputs in user-model conversations without substantial performance loss relative to LLM-based judges. We benchmark these encoder classifiers against rule-based prefix matching, fine-tuned LLM classifiers, and LLM judges using a range of judge-prompting strategies across open-source adversarial datasets.
The LLM judges include evaluation methodologies from StrongReject, ShieldGemma, JailbreakBench, AILuminate, SorryBench, and a Claude-as-a-judge setup, as well as fine-tuned safety classifiers such as LlamaGuard 3 and LlamaGuard 4. The encoder classifiers are fine-tuned on judge-labeled data using a majority-voting label strategy and are then evaluated on a gold-standard holdout dataset to assess their performance relative to LLM judges.
We report absolute performance using F1 score, false negative rate, and precision-recall metrics. We also break down results by attack technique, including single-turn prompting, decomposition, escalation, and context manipulation, to identify where encoder classifiers align with or diverge from LLM-based judges. Our findings provide guidance on when encoder classifiers can serve as cost- and latency-efficient alternatives to LLM-based safety evaluation. 

---
# Space-Efficient Language Generation in the Limit 

**Authors**: Nicolas Flammarion, Chirag Pabbaraju, Hristo Papazov, Miltiadis Stouras, Ola Svensson  

**Link**: [PDF](https://arxiv.org/pdf/2606.25777)  

**Abstract**: We initiate a resource-aware theory of \textit{language generation in the limit} under the minimal constraint of space efficiency. In our framework, a learner observes an adversarial positive stream from a target language $K$ and must eventually output a hallucination-free hypothesis language $L \subseteq K$ while omitting at most $\Delta$ strings of $K$. We focus on $\mathcal{C}_{s,k}$, the collection of languages recognized by DFAs with at most $s$ states over an alphabet of size $k$, as the natural hypothesis class for memory-bounded learners. In the exponential-space regime, we prove that a learner can exactly identify the target $K$. Under a stricter memory budget, we characterize the strongest possible generation guarantees. In particular, we present a streaming algorithm using $\mathrm{poly}(s,k)$ space that converges to a hypothesis with generation gap $\Delta = O(k^{2s-2})$. Moreover, the learned hypothesis captures every string in $K$ of length at least $2s-1$. We complement this result with a near-matching lower bound through a reduction from a standard communication complexity problem. Specifically, achieving generation gap $\Delta \le k^{(1-\varepsilon)s}$ requires $k^{\Omega(\varepsilon s)}$ memory. Together, these results reveal a sharp transition between polynomial-space generation and exponential-space exact identification. 

---
# OncoSynth: Synthetic data generation for treatment effect estimation in oncology 

**Authors**: Octavia-Andreea Ciora, Julian Welzel, Dennis Frauen, Maresa Schröder, Marie Brockschmidt, Harry Amad, Thomas Callender, Mihaela van der Schaar, Stefan Feuerriegel  

**Link**: [PDF](https://arxiv.org/pdf/2606.25762)  

**Abstract**: In oncology, access to patient-level data is often restricted. Synthetic data provides an alternative for analyzing treatment effectiveness, but existing methods for synthetic data generation fail to preserve the causal relationships between covariates, treatments, and outcomes, thereby leading to biased estimates of treatment effects. Here, we introduce OncoSynth, a generative, causally-aware machine learning framework designed to produce synthetic cohorts that enable accurate estimation of population- and patient-level treatment effects. OncoSynth uses a diffusion-based sequential approach to model how covariates influence treatment assignment and how treatment affects survival. We evaluate OncoSynth using large lung (N = 37,128) and breast cancer (N = 17,046) cohorts. Our results show that OncoSynth generates high-fidelity synthetic patient cohorts that preserve real-world patient, treatment, and outcome distributions. Notably, OncoSynth improves treatment effect estimation over existing approaches, by reducing population-level treatment effect error by up to 66%, and patient-level treatment effect error by up to 58%. Thereby, OncoSynth supports reliable evidence generation for precision oncology in settings where data sharing is restricted. 

---
# Uncertainty Quantification for Computer-Use Agents: A Benchmark across Vision-Language Models and GUI Grounding Datasets 

**Authors**: Divake Kumar, Sina Tayebati, Devashri Naik, Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25760)  

**Abstract**: Computer-use agents turn vision-language model (VLM) predictions into executable GUI clicks, so reliable uncertainty estimates are essential for rejection, calibration, miss-severity ranking, and spatial safety regions. Yet evidence on post-hoc uncertainty quantification (UQ) for these agents is fragmented across isolated model and dataset pairs, leaving it unclear whether UQ rankings stay stable when the agent, benchmark, or observable interface changes. We present Argus, a cross-regime benchmark for post-hoc UQ in single-step executable GUI grounding: a 27-method open-weight matrix over 4 VLM agents and 4 datasets, plus an 8-method closed-source matrix across 3 frontier vendors where logits, hidden states, and attention maps are unavailable. Evaluated methods span logit-based scores, sampling and consistency measures, hidden-state and density estimators (Mahalanobis, SAPLMA), attention-based scores, P(True) and verbalised-confidence prompting, and split-conformal prediction. The main finding is selective transfer: UQ rankings are stable across datasets for a fixed model, but degrade across model classes and observable interfaces. Hidden-state and density methods are the most stable open-weight family, while CoCoA-1MCA, Focus, sampling-based scores, and verbalised self-assessment win in specific regimes. Within-model ranking transfer is strong (Spearman rho up to 0.969), but cross-tier transfer to closed-source vendors averages only +0.08, so closed-source UQ should be reranked on the target rather than extrapolated. Conformal click regions show score-level discrimination is not enough for deployment: locally weighted disks shrink radii by 40-60% when the plug-in UQ is calibrated, but coverage degrades under calibration-test or interface mismatch. We release per-item records, calibration/test splits, UQ scores, and analysis scripts for regime-aware UQ selection in GUI agents. 

---
# Gradient-based inverse lithography for EUV masks via the waveguide method and a physics-informed neural operator 

**Authors**: Vasiliy A. Es'kin, Egor V. Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2606.25753)  

**Abstract**: Gradient-based inverse lithography technology~(ILT) for extreme ultraviolet~(EUV) masks is presented. A novel framework treats the differentiable waveguide method and the recently proposed waveguide neural operator~(WGNO) as end-to-end physics engines, recovering the permittivity of the absorber of the mask through automatic differentiation of the full forward diffraction model. Numerical experiments on realistic 2D and 3D absorbers of the mask (TaBN, La, U) at $\lambda{=}11.2$~nm show that the considered ILT methods make it possible to obtain a mask structure that achieves the desired field on the wafer. 

---
# Point Cloud Diffusion with Global and Local Reconstruction for Instance-Level 3D Anomaly Detection 

**Authors**: Linchun Wu, Qin Zou, Jiwen Lu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.25740)  

**Abstract**: 3D anomaly detection in point clouds is critical for high-precision industrial manufacturing. Reconstruction-based methods have laid a strong foundation by detecting 3D anomalies through comparisons between defective inputs and their reconstructed normal counterparts. However, existing methods still suffer from two challenges: 1) the foreground weak defective regions such as scratches are hard to reconstruct and detect, where the anomaly deviations in normalized point clouds can be as small as $10^{-3}$; 2) the background non-defective regions are prone to get positional bias in reconstruction, which leads to false positives. To address these challenges, we propose \textbf{PCDiff}, a point cloud diffusion framework for instance-level 3D anomaly generation and detection. In the generation phase, an instance-level multi-modal attention is embedded into the generation framework, where anomalies are conditioned with texture gradient, image patch, text and mask. The instance-level condition enables the high-quality generation of weak-defective anomalies. In the detection phase, a joint local-global reconstruction algorithm is introduced to ensure local anomaly restoration and global geometric consistency, which preserves background normal structure while restoring the foreground defect. Extensive experiments demonstrate that the proposed PCDiff significantly outperforms state-of-the-art methods in both 3D anomaly generation fidelity and reconstruction quality, leading to substantial improvements in anomaly detection accuracy. 

---
# Power-Budgeted Underwater Vehicle Control via Constrained Reinforcement Learning 

**Authors**: Yinuo Wang, Gavin Tao, Yuze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25680)  

**Abstract**: Underwater vehicles operate from a fixed onboard energy budget that propulsion rapidly depletes, so a controller that completes its task while drawing less thruster power directly extends mission range and endurance. Reinforcement learning yields capable model-free controllers for station-keeping and trajectory tracking, but optimizing task accuracy alone drives the policy toward oscillatory, energy-wasting actuation. The established remedy subtracts an energy penalty from the reward, yet this sets the task-power trade-off through a single weight with no physical units: a target power level cannot be specified, the weight must be re-tuned for every vehicle and task, and a mismatched weight can even raise power. This paper instead formulates energy-efficient underwater control as a constrained Markov decision process in which average thruster power is subject to an explicit budget, solved with a PPO-Lagrangian algorithm. The power level is set by declaring a budget in physical units, and a single dual variable is updated online to meet it for each vehicle and task, without manual weight search. Across three vehicles and four tasks in the MarineGym simulator, the energy-constrained policy draws the least power in all twelve settings, reducing it by 14--65\% (up to 64.9\%) over a task-only baseline and below an energy-reward baseline everywhere, while remaining the smoothest in ten settings and preserving task accuracy except in one deliberately power-limited regime. Imposing energy as an explicit constraint thus offers a tuning-free route to energy-efficient underwater control that needs no per-vehicle, per-task weight search. 

---
# Steering Vision-Language Models with Joint Sparse Autoencoders 

**Authors**: Huizhen Shu, Xuying Li, Hongxu Lin, Wenjie Sun, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.25657)  

**Abstract**: Sparse Autoencoders (SAEs) have shown promise for analyzing language models, but applying them to vision-language models (VLMs) often yields representations that are difficult to use as controllable cross-modal steering directions. We introduce the Joint Sparse Autoencoder (JSAE), which uses an explicit alignment constraint to jointly factorize sequence-pooled vision and language activations into shared, interpretable image/caption-level features. Applied to LLaVA, JSAE recovers cross-modal features for recognizable concepts (e.g., food and animals). Through bidirectional interventions (additive steering and suppression), we observe a layer-dependent asymmetry under our protocol: additive steering peaks at mid-to-late (pre-output) layers and weakens at both ends, whereas suppression scores remain within a comparable range across all probed layers within statistical noise. Experiments on three VLMs, namely LLaVA-v1.6-Mistral-7B, Llama3-LLaVA-8B, and the MoE-based Qwen3-VL-30B, show related layer-localized effects across architectures. Together, these results suggest that explicitly aligned sparse representations support more controllable intervention-based analysis of multimodal features, within an identifiable layer range, than the unconstrained alternatives tested here. 

---
# Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization 

**Authors**: Long Chen, Ryan Razkenari, Yuxuan Zhou, Yuan Tian, Rahul Ghosh, Venkatesh Pappakrishnan, Disha Ahuja, Vidya Sagar Ravipati  

**Link**: [PDF](https://arxiv.org/pdf/2606.25656)  

**Abstract**: As advanced RAG variants like GraphRAG and Agentic RAG emerge, one leading question is when and how to use them. Here, we introduce a framework for different RAG scenarios evaluation and comparison on semi-structured knowledge bases, including regular RAG, GraphRAG, Modular RAG and Agentic RAG. We provide implementation for 9 standardized RAG scenarios, and conduct experiments for a comprehensive comparison. These scenarios are designed for real use cases regarding data and domain restrictions, spanning from simple document-based retrieval to advanced features such as hybrid text-graph retrieval, integration with computed or pre-defined domain knowledge graphs, agentic multi-step planning, and agent-graph integration. Besides, we present a novel context engineering method for GraphRAG and Agentic RAG, addressing the context/memory overflow issues, efficiently managing text and graph retrievals with new representations and agentic loop design, leading to 19%-53% reduction on token usage. Moreover, further analysis identifies a retrieval-generation gap where expanded retrieval does not proportionally improve generation quality, suggesting retrieval-oriented metrics overstate advanced retrieval benefits. This work provides data-driven insights on when and how to use them for building production-ready intelligent RAG systems. 

---
# Taxonomy of Risks on Automated Fact-Checking Systems Considering its Propagation 

**Authors**: Jun Yajima, Tatsuya Oka, Takao Okubo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25645)  

**Abstract**: In recent years, the posting of fake news including disinformation and misinformation on social networking services (SNS) has become a social problem. To combat this fake news, fact-checking that is the process of assessing the veracity of posts on SNS has become increasingly important. While fact-checking is currently performed by fact-checking organizations, it is difficult to fact-check all posts on SNS. Therefore, the use of automated fact-checking systems is effective. Recent automated fact-checking systems utilize artificial intelligence and large language models, so there are risks of incorrect judgments and posting incorrect results on social media which can lead to the spread of misinformation or to engage in defamation. In this paper, as a first step toward enabling the safe use of automated fact-checking systems, we categorize the specific risks on automated fact-checking systems. In this categorizing, we consider a three-stage risk propagation: risk factors, hazardous situations, and harm. Our analysis revealed that 32 specific risks exist in automated fact-checking systems. In this paper, we utilize the categorized risks as analytical cues (guide words) to present the risk assessment of the automated fact-checking system DEFAME. This assessment result indicates that risks that cannot be derived using STRIDE, a conventional IT security risk assessment method can be derived using our guide words. 

---
# Staying In Character: Perspective-Bounded Memory For Book-Based Role-Playing Agents 

**Authors**: Xushuo Tang, Junhe Zhang, Zihan Yang, Yifu Tang, Sichao Li, Longbin Lai, Zhengyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25632)  

**Abstract**: Recent LLM role-playing systems build character agents from novels by extracting characters, scenes, and relations. Yet long-narrative role-playing suffers from two failures: Factual Overreach, where shared retrieval or parametric memory lets a character use facts outside its perspective, and Stylistic Monotony, where profile descriptions flatten a character into a fixed voice. To address these failures, we propose REVERIEMEM, a three-layer memory architecture for book-based character agents. The episodic layer stores first-person scene memories; the semantic layer stores visibility-tagged facts; and the personality layer stores situation-dependent speech and behaviour patterns. For evaluation, we construct KBF-QA, a 4,386-question benchmark over eight novels for testing knowledge boundaries. REVERIEMEM improves Knowledge Boundary Fidelity by 34.6 percentage points over the strongest prior method. On BOOKWORLD's five-dimension pairwise narrative protocol, REVERIEMEM achieves a ~ 79% win rate, suggesting that perspective-bounded memory improves both boundary fidelity and character-grounded narrative generation. 

---
# TL++: Accuracy and Privacy Preserving Traversal Learning for Distributed Intelligent Systems 

**Authors**: Erdenebileg Batbaatar, Young Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2606.25627)  

**Abstract**: Distributed intelligent systems increasingly need to train across data silos without centralizing raw data. Federated learning keeps data local but can suffer under heterogeneous partitions and requires repeated full-model exchange. Split learning reduces communication through cut-layer activations, but standard protocols generally do not recover centralized mini-batch gradient behavior and may expose activations and gradients in plaintext. We present TL++, a two-mode traversal-learning framework that constructs virtual batches across nodes to recover centralized mini-batch gradient behavior under explicit synchronization assumptions. Base mode exchanges cut-layer activations and gradients rather than full models. Secure mode secret-shares each cut-layer activation and gradient between an orchestrator and a non-colluding helper, preventing either server from observing plaintext cut-layer tensors. This protection is limited to a semi-honest two-server setting; labels and loss-related outputs remain visible to the orchestrator. In the lightweight secure path evaluated here, exactness requires a linear or affine server path, while nonlinear operations require nonlinear MPC or approximation. We formalize TL++, analyze communication and computation costs, and evaluate it against federated and split-learning baselines on CIFAR-10 and BioGPT/PubMedQA using full fine-tuning and LoRA. On CIFAR-10, TL++ base cut 1 and exact secure cut 3 achieve accuracies of 91.41% (SD 0.19) and 90.93% (SD 0.17), respectively, exceeding the strongest measured non-TL++ baseline by more than 12 percentage points. TL++ base cut 1 also reduces per-step communication by 13.1-fold relative to full-model synchronization. PubMedQA results similarly favor TL++. Overall, TL++ approaches centralized-training performance while reducing communication and providing activation-level secret sharing. 

---
# Probabilistic Agents in Deterministic Audits: Evaluating Multi-Agent Systems for Automated Audits Based on the German IT-Grundschutz 

**Authors**: Lea Roxanne Muth, Marian Margraf  

**Link**: [PDF](https://arxiv.org/pdf/2606.25622)  

**Abstract**: The NIS-2 Directive mandates robust Risk Management from thousands of small and medium enterprises. To ensure compliance, companies rely on established standards such as the German IT-Grundschutz (IT-GS) of the Federal Office for Information Security. However, IT-GS certification is resource-intensive and requires a high level of manual effort for documentation, validation, and revision, making scalable implementation difficult and expensive.
Building upon our previous conceptual framework, this paper presents the technical implementation and empirical evaluation of a Multi-Agent System (MAS) architecture combined with Hybrid Retrieval Augmented Generation (HybridRAG) for the partial automation of IT-GS certification. We introduce two novel technical contributions to the MAS architecture to enforce the compliance rigor. The Hypothesis-Verification Loop in the Structural Analysis (SA) phase that cross-references agent-inferred dependencies against the Knowledge Graph to reduce hallucinations, and a Decoupled Reasoning Pipeline that separates agent-driven semantic extraction from the deterministic protection need inheritance. We utilize the BSI's "RecPlast GmbH" case study as a human expert-generated reference data set for end-to-end evaluation of the architecture and to quantify Precision, Recall, and F1-scores. The performance of the system is investigated across the phases of SA, Protection Needs Assessment (PNA), Modeling, and IT-GS Check.
The empirical results reveal noticeable differences throughout the different steps of IT-GS. While the MAS demonstrates high efficacy in semantic tasks (SA and Modeling), significantly reducing manual effort through automated information extraction, quantitative results reveal limitations in logical reasoning phases (PNA and IT-GS Check) as the probabilistic nature of current LLMs struggles to meet the deterministic rigor required by IT-GS. 

---
# An Approach for a Supporting Multi-LLM System for Automated Certification Based on the German IT-Grundschutz 

**Authors**: Lea Roxanne Muth, Marian Margraf  

**Link**: [PDF](https://arxiv.org/pdf/2606.25608)  

**Abstract**: This paper presents a novel approach to perform semi-automated BSI IT-Grundschutz certification using a MultiLarge Language Model system (MLS) with Hybrid RetrievalAugmented Generation (HybridRAG). Facing the challenges of the Network and Information Security Directive 2 (NIS2) directive, a shortage of specialists, and high implementation costs, our MLS architecture aims to increase efficiency, reduce costs, and support certifiers in maintaining the quality of security concepts while meeting the increased demand for certifications of newly affected companies. The system combines Large Language Models (LLMs) and Knowledge Graphs (KGs) to support different phases of the certification process, including protection needs assessment, modeling, IT-Grundschutz check, measure consolidation, and subsequent realization. Our architecture addresses the growing demand for security concepts and offers an approach to handle the digital security challenges introduced by NIS2. 

---
# Expresso-AI: Explainable Video-Based Deep Learning Models for Depression Diagnosis 

**Authors**: Felipe Moreno, Sharifa Alghowinem, Hae Won Park, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2606.25606)  

**Abstract**: Given the widespread prevalence of depression and its consequential impact on individuals and society, it is crucial to obtain objective measures for early diagnosis and intervention. As a multidisciplinary topic, these objective measures should be interpretable and accessible to health care professionals, ensuring effective collaboration and treatment planning in the realm of mental health care. Even though current automated depression diagnosis approaches improved over the last decade, a critical gap exists as they often lack affect-specificity and interpretability, limiting their practical application and potential impact on mental health care. In particular, interpretability from temporal activities from videos when deep models are used is not fully explored. In this study, we present a novel framework for analyzing Deep Neural Networks' decisions when trained on facial videos, specifically focusing on automatic depression severity diagnosis. By fine-tuning Deep Convolutional Neural Networks (DCNN) pre-trained on Action Recognition datasets on depression severity facial videos from AVEC depression dataset, our framework is able to interpret the model's saliency maps by examining face regions and temporal expression semantics. Our approach generates both visual and quantitative explanations for the model's decisions, providing greater insight into its reasoning. In addition to this interpretability, our video-based modeling has improved upon previous single-face benchmarks for visual depression diagnosis, resulting in enhanced predictive performance. Overall, our work demonstrates the successful development of a framework capable of generating hypotheses from a facial model's decisions while simultaneously improving depression's predictive capabilities. 

---
# Low-Complexity Policy Tessellations in Structured Markov Decision Processes 

**Authors**: Fredy Pokou  

**Link**: [PDF](https://arxiv.org/pdf/2606.25593)  

**Abstract**: We study optimal-policy geometry in structured Markov decision processes. While approximate dynamic programming and reinforcement learning typically approximate high-dimensional value functions, we show that optimal policies induce simpler decision tessellations. We propose boundary-based policy approximations that learn policy regions directly. A policy-loss decomposition links performance degradation to action margins and explains why errors concentrate near indifference boundaries. Inventory control and queue admission experiments show lower policy error, smaller value gaps, faster error decay, and stability than reinforcement learning baselines. 

---
# BiPACE: Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation for LLM Agents 

**Authors**: Hanyang Wang, Weijieying Ren, Yuxiang Zhang, Ding Cao, Zhizhao Zeng, Ke Zeng, Tianxiang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25556)  

**Abstract**: Stepwise group-based RL is an attractive way to train long-horizon LLM agents without a learned critic: it reuses multiple sampled rollouts to estimate local advantages. Its weakness is less visible but more fundamental: every group-relative estimator assumes that the steps it compares are equivalent for credit assignment. We show that current agentic variants violate this assumption through a state-action credit mismatch. The observation-hash partition is overly fine on the state side, creating singleton groups with zero step-level signal, while a single within-group mean is too coarse on the action side, mixing state-value estimation with action-specific credit. We introduce BiPACE (Bisimulation-Guided Policy Optimization with Action Counterfactual Estimation), a drop-in advantage estimator that fixes both sides without adding a critic, auxiliary loss, or extra rollouts. BiGPO clusters steps by cosine distance in the actor's own hidden-state geometry, an empirical policy-induced proxy for bisimulation that substantially lowers the singleton rate left by observation hashing. PACE then recenters returns within each behavioral cluster using action-conditioned peer baselines; its Q-style instance estimates a local Q(s,a)-V(s) nonparametrically. On ALFWorld/Qwen2.5-7B, BiPACE_Q raises overall validation success from GiGPO's 90.8 to $97.1\pm0.9$ over three seeds, and crosses the 95% threshold on every seed, which GiGPO never does within the same budget. On Qwen2.5-1.5B it reaches $93.5\pm1.2$ versus GiGPO's 86.7, and on WebShop and TextCraft it improves over GRPO and GiGPO at both model scales. The measured BiPACE-specific overhead is 11.3% of a single training-step wall time. Yet it changes the estimator's comparison unit from surface identity to approximate behavioral equivalence plus action-side counterfactuals. The code is available at this https URL. 

---
# SFL-MTSC: Leveraging Semantic Frame-Level Multi-Task Self-Consistency for Robust Multi-Intent Spoken Language Understanding 

**Authors**: Po-Yen Chen, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25552)  

**Abstract**: Prompt-based spoken language understanding (SLU) with large language models (LLMs) often suffers from inconsistent intent--slot structures due to decoding stochasticity, particularly in multi-intent scenarios. In view of this, we propose Semantic Frame-Level Multi-Task Self-Consistency (SFL-MTSC), a novel structured aggregation framework operating at the semantic frame level. Instead of output-level majority voting, SFL-MTSC decomposes predictions into intent-specific frames, applies domain--intent grouping and slot-level clustering, and evaluates cluster reliability using path support scoring. Reliable frames are retained and re-integrated to form the final prediction. Zero-shot experiments on the MAC-SLU benchmark dataset show improved slot F1 and overall accuracy over single-path inference, while intent accuracy remains largely stable across most settings. 

---
# Evaluating LLMs on Real-World Software Performance Optimization 

**Authors**: Ezgi Sarıkayak, Wenchao Gu, Hesham Ghonim, Chunyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25530)  

**Abstract**: Software performance optimization is a notoriously complex and manual task. Despite the growing use of Large Language Models (LLMs) for code refinement, we still lack benchmarks that capture how optimization actually happens in real-world codebases. Existing frameworks often oversimplify the problem by focusing on isolated functions or a single performance metric, missing the critical trade-offs between execution time and memory footprint, the inherent noise of the measurement environment, and the variability introduced by different input data and execution conditions. We address this by introducing SWE-Pro, a repository-level benchmark derived from 102 expert-written optimizations from open-source projects. Unlike previous benchmarks, SWE-Pro pairs each task with parameterized tests to evaluate runtime, peak memory, and Time-Weighted Memory Usage (TWMU) across varying input data and execution conditions under noise-aware measurement conditions. Our evaluation shows that current LLMs struggle significantly: runtime gains are negligible, and memory optimizations are nearly non-existent. This stands in sharp contrast to expert implementations, which achieve an aggregate speedup of 15.5x and peak memory reduction of 171.3x over benchmark tasks. Expert-written improvements are observed in 91.2% of tasks for runtime and 65.7% for peak memory. Our findings expose a substantial gap between current LLM capabilities and the demands of expert-level engineering. 

---
# STEB: A Speech-to-Speech Translation Expressiveness Benchmark for Evaluating Beyond Translation Fidelity 

**Authors**: Sitong Cheng, Weizhen Bian, Songjun Cao, Jin Li, Bei Liu, Chunyang Jiang, Yike Zhang, Weihao Wu, Yiming Li, Chi-Min Chan, Long Ma, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2606.25529)  

**Abstract**: Speech-to-speech translation (S2ST) should preserve not only lexical meaning, but also expressive attributes: emotion, scenario style (e.g., news reporting vs. dramatic dialogue), and nonverbal vocalizations (NVs). Moreover, collecting cross-lingual target speech that is both translation-faithful and expressively aligned with the source is difficult at scale, making reference-based evaluation impractical. We introduce STEB (Speech-to-Speech Translation Expressiveness Benchmark), a 32.6-hour Chinese--English benchmark that evaluates both standard dimensions (translation fidelity, speaker similarity, duration alignment) and expressiveness dimensions (emotion, scenario style, NV preservation). For expressiveness evaluation, STEB uses a caption-then-summarize framework that converts speech into structured expressive attributes and compares source and hypothesis attributes with an LLM judge. Human validation shows statistically significant correlations with listener judgments across all expressive dimensions. We evaluate six S2ST systems covering cascaded systems, end-to-end models, and speech large language models. Many systems, especially cascaded ones, achieve strong translation fidelity, but they still struggle with emotion preservation (best: 3.82/5) and NV preservation (best: 2.31/5). These results reveal a gap between semantic transfer and expressive transfer, identifying expressiveness preservation as an open challenge for S2ST. Audio samples are available at this https URL. 

---
# The impact of artificial intelligence on enterprise software user roles 

**Authors**: Isabel Unger, Elizangela Valarini, Martin Schrepp, Nina Hollender, Gabriela Rocha, Erik Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2606.25525)  

**Abstract**: Artificial Intelligence (AI) is rapidly reshaping the nature of work in software development, transforming user roles, workflows, and collaboration patterns across enterprise platforms. This qualitative study investigates how AI alters professional responsibilities within the context of SAP's Business Technology Platform (BTP), combining expert interviews (n=20) and a participatory workshop (n=24). The results reveal substantial shifts in day-to-day tasks and roles in the development domain, characterized by increasing automation of operational tasks, expanding human-AI collaboration, and growing reliance on agentic AI systems. The study further identifies significant implications for existing user-role frameworks, such as the BTP User Type Matrix, which requires adaptation as the workforce is undergoing significant role specific changes. Collectively, these findings highlight a workforce landscape in transition and underscore the need for revised role taxonomies, new governance and oversight functions, and updated design approaches for AI-native enterprise software systems. 

---
# Spam and Sentiment Detection in Arabic Tweets Using MARBERT Model 

**Authors**: Abrar Alotaibi, Atta-ur Rahman, Raheel Alhaza, Wala Alkhalifa, Narjes Alhajjaj, Atheer Alharthi, Dhai Abushoumi, Maryam Alqahtani, Dania Alkhulaifi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25495)  

**Abstract**: Saudi Telecom Company (STC) is among the most popular companies in Saudi Arabia, with many customers. Yet, there is still a big room for improvement in users' satisfaction. Social media is the most robust platform to gauge users' satisfaction and determine their sentiments and critics. Twitter is among the most popular social media platform in this regard. STC customers prefer to use Twitter to write their feedback because it's a fast way to get responses due to the STC customer services account. One way to achieve customer demands and improve customer service is using the Sentiment Analysis tool. Sentiment Analysis on Twitter is highly used because of the significant number of tweets and the different opinions. Likewise, Deep learning is the best existing Sentiment Analysis method, and it has diverse models. Bidirectional Encoder Representations from Transformers (BERT) model is one of the deep learning models which have achieved excellent results in Sentiment Analysis for Natural Language Processing (NLP). NLP is mainly investigated in the English language. However, for Arabic, there is a significant gap to be filled. This study trained the proposed model using MARBERT and measured the performance using f1-score, precision, and recall metrics. We trained the model with an Arabic dataset of 24,513 tweets, including 1,437 positive, 13,828 negative, 5,694 neutral, 1,221 sarcasm, and 2,297 indeterminate tweets. The main goal is to analyze the tweets and get the sentiment to improve STC customer service. The proposed scheme is promising in terms of accuracy in contrast to existing techniques in the literature. 

---
# HG-Bench: A Benchmark for Multi-Page Handwritten Answer-Region Grounding in Automated Homework Assessment 

**Authors**: Chuangxin Zhao, Boyan Shi, Yanling Wang, Yijian LU, Canran Xiao, Jiali Chen, Jun Xia, Yan Wang, Ji Qi, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.25491)  

**Abstract**: Automated homework assessment depends not only on recognizing student answers, but also on accurately locating where each answer and each intermediate reasoning step appears in noisy, multi-page handwritten work. This paper addresses the missing evaluation setting of page-aware, two-level answer-region grounding: given a sequence of homework page images, a model must localize complete answer regions and their ordered step-level subregions. We introduce HG-Bench, a benchmark of 500 human-annotated K-12 homework samples curated from a 1,489,278-image source pool, with question-level and step-level boxes linked by a hierarchical containment constraint. HG-Bench is paired with a page-aware evaluation protocol that separately measures complete-answer localization (FA) and step-level decomposition (FSm), revealing whether models truly ground the spatial structure of student reasoning rather than merely parse visible text. Across frontier closed-source APIs and competitive open-weight VLMs, no zero-shot system exceeds 55.22% on FA or 48.22% on FSm, while a GLM-4.6V 9B reference model fine-tuned on ~10k in-domain examples reaches 74.97/72.26. These results identify step-level handwritten grounding as a concrete capability gap and provide a reproducible benchmark, evaluation protocol, and trained reference point for future work on automated homework assessment. 

---
# Rate-Aware Quantum-Inspired Trajectory Learning for Interference-Limited Multi-UAV Networks 

**Authors**: Khaoula Khaled, Muhammad Afaq, Ali Arshad Nasir, Zeeshan Kaleem  

**Link**: [PDF](https://arxiv.org/pdf/2606.25480)  

**Abstract**: Unmanned aerial vehicle (UAV) can provide on-demand, high-capacity connectivity in disaster and normal situation. However, it faces a challenge of curse of dimensionality in trajectory optimization, where interference-limited environments and vast search spaces make real-time coordination computationally expensive. To overcome this challenge, we propose the Rate-Aware Quantum-Annealed Graph Condensation (RA-QAGC) scheme, which combines rate-aware graph abstraction with decentralized reinforcement learning to enable scalable, interference-aware UAV coordination. By identifying high throughput locations and guiding UAV trajectory adaptation toward throughput-optimal regions, RA-QAGC effectively balances network capacity by maintaining quality-of-service (QoS) requirements. Simulation results demonstrate the proposal outperformed over existing schemes by achieving 59.4 Mbps total throughput and 23.9 Mbps priority-user throughput, representing gains of approximately 15% and 34%, respectively, over the baseline schemes. 

---
# A Red Teaming Framework for Large Language Models: A Case Study on Faithfulness Evaluation 

**Authors**: Abrar Alotaibi, Raed Mughus, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2606.25476)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across natural language processing tasks, yet their deployment in high-stakes applications raises critical concerns regarding reliability, safety, and trustworthiness. In this paper, we present a red teaming framework that systematically uncovers vulnerabilities in LLM outputs. Our approach employs a novel multi-role architecture comprising target, attacker, and jury models. The attackers generate increasingly effective adversarial prompts while the jury rigorously evaluates response accuracy and consistency across tasks. In a case study, our strategy proved particularly effective at exposing unfaithfulness in LLM responses. Exploitative adversarial prompts increased the attack success rate by up to 7.9% in question-answering tasks, revealing weaknesses in reliability. The approach identifies how structural constraints in summarization can shape vulnerability patterns, with format limitations yielding measurable gains in faithfulness, and shows that architectural design choices typically outweigh parameter scaling in determining model safety. The framework's key strength is its adaptability across evaluation tasks, from English question-answering to Arabic summarization, enabling comprehensive comparison of model vulnerabilities. While it excels at comparing cross-model and cross-linguistic vulnerabilities, it faces challenges in fully automating adversarial prompt generation across languages. Our experiments also reveal limitations in detecting subtle forms of unfaithfulness that do not manifest as explicit factual contradictions, particularly across linguistic contexts. Overall, this architecture provides both actionable insights into current LLM vulnerabilities and a scalable methodology for ongoing safety evaluation as models evolve. 

---
# EchoStyle: Unlocking High-Fidelity Video Stylization with Reverse Data Synthesis 

**Authors**: Huaqiu Li, Jiahao Wang, Sijia Cai, Hualian Sheng, Bing Deng, Jieping Ye, Wenhan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25465)  

**Abstract**: While image stylization has been studied extensively, video stylization remains a critical and largely unsolved challenge in the field of intelligent content creation. Existing methods, usually utilizing a reference image as the style prior, suffer from content leakage, data scarcity and limited adaptability to long videos, leading to suboptimal results with severe style drift and motion distortion. For these issues, we present EchoStyle, a scalable text-driven framework to achieve high-quality stylization of videos with arbitrary lengths. To start with, we construct a video-to-video architecture to appropriately re-fuse the video content and the text style. To address data scarcity, we pioneer an automatic reverse-synthesis pipeline to establish V-Style20k, a large-scale stylization dataset of 20k high-quality video pairs. To facilitate long video stylization, we devise an init-follow-mode mechanism along with a sliding-window inference strategy. Extensive experiments demonstrate EchoStyle's excellent performance across a wide range of artistic styles, even comparable to leading closed-source solutions. 

---
# Learning with a Single Rollout via Monte Carlo Pass@k Critic 

**Authors**: Fengdi Che, Yang Liu, Lei Yu, Meng Cao, Tong Che, Rupam Mahmood, Dale Schuurmans  

**Link**: [PDF](https://arxiv.org/pdf/2606.25451)  

**Abstract**: Estimating token-level advantages in reinforcement learning (RL) for language models remains challenging because scaling up episodic experience collection is expensive. The difficulty intensifies for baseline advantage estimation methods, where repeated sampling causes trajectories to diverge into substantially different reasoning prefixes. In this context, RL algorithms such as GRPO prove limited: an outcome reward is too sparse to be attributed to specific actions like intermediate steps, and comparisons across sampled traces are non-trivial because they are heterogeneous. To mitigate both the computational cost of repeated sampling and the difficulty of credit assignment, we study single-rollout proximal policy optimization (SR-PPO) featuring token-level credit assignment in RL for language models. Instead of estimating advantages by normalizing episodic returns within the candidate group, we train a calibrated token-level credit critic using Monte Carlo outcomes from one rollout per prompt. Specifically, we use the critic to predict the Pass@k success probability at the prompt prefix, which is derived from a Pass@1 attempt. This choice yields a more selective learning signal than Pass@1: it discounts easily solved prefixes while prioritizing hard ones whose success probability remains marginal. We show that as $k$ increases, Pass@k converges to a reachability indicator, reflecting whether a prefix can lead to at least one successful continuation. In an explicit state graph, the limit ($k \rightarrow \infty$) can be computed in $O(|V|+|E|)$ time, offering a promising surrogate for direct credit assignment without the need to sample contrastive traces. As an initial validation, SR-PPO exhibits stable learning dynamics, along with consistent gains in Pass@128 success rates on mathematical reasoning benchmarks such as HMMT26 and AIME24. 

---
# Reclaim Evaluation: A Lossy Memory Is Worse Than an Empty One 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2606.25449)  

**Abstract**: A language model's memory can be worse than having no memory at all. Give a model a memory that kept a wrong conclusion but dropped the work behind it, and it emits that stale value as a confident answer; give the same model an empty memory and it abstains. Across seven models this direction never reverses, a clean kill condition that none breaks. We call this brittle memory: behavioral, not the near-immediate information bound beneath it; only its magnitude is disposition- and task-dependent, not its direction. We measure it with reclaim evaluation: compress a drifted interaction at a fixed budget, then test whether a correction recovers the known answer, scored against ground truth with no judge. Correctability is bottlenecked by whether the answer-determining source survives, not by capability. A one-line source-first policy (keep the recomputable source, drop the re-derivable conclusion) restores correctability at equal budget where that source is compact and identifiable; a length-matched control rules out added text as the cause. The hand-built oracle reaches 1.00; a one-prompt deployable version reclaims 0.49-0.88. The stake compounds: chained through a memory loop, a single dropped-source error corrupts a growing span of downstream steps and stays uncorrectable, while source-first holds to a bounded budget horizon. The wall and fix replicate across three deployed memory systems and on real dialogue (MultiWOZ), and past the budget where the source no longer fits, the fix fails silently unless the note records completeness. This is a controlled study of a mechanism, not a benchmark: judge-free exact scoring, matched-budget controls, and validators built to come out false. We release the harness, conditions, and validators. 

---
# C3-Bench: A Context-Aware Change Captioning Benchmark 

**Authors**: Jae-Woo Kim, Hyeongbeom Kim, Ue-Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.25445)  

**Abstract**: While Change Captioning systems have garnered substantial attention to respond to our evolving world, their true performance on diverse real-world change contexts remains largely unexplored due to the lack of comprehensive evaluation frameworks. To fill this gap, we propose C3-Bench, a comprehensive benchmark for evaluating Context-aware Change Captioning. C3-Bench features: (1) 4,996 human-labeled image pairs of 51 real-world change contexts across four domains (e.g., natural scenes, remote sensing imagery, image editing, and anomalies), each with diverse, carefully curated scenarios derived from multiple change-centric communities; and (2) the first LLM-as-Judge evaluation framework in the change captioning task that measure fine-grained dimensions (e.g., correctness, specificity, fluency, and relevance), along with a novel reversibility metric exploring whether models understand changes with symmetric consistency. Based on C3-Bench, we benchmark 32 models -- including conventional change captioning models, proprietary Large Multimodal Models (LMMs), and 2B-90B open-source LMMs. We reveal a fundamental blind spot in the prevailing change captioning paradigm: Once the change context departs from training-style regimes, conventional models collapse, and even state-of-the-art LMMs such as GPT-5.2 exhibit systematic domain- and position-dependent errors that distort reliable change understanding. By making these hidden failure modes explicit and measurable, we delineate the next frontier for building generalizable and trustworthy change captioning systems. All codes and datasets are publicly available on the project page. 

---
# TopoCast: A Topological Fidelity Framework for Evaluating Transformer-Based Time Series Forecasting 

**Authors**: Sandeepa Weerasekara, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2606.25439)  

**Abstract**: Deep learning-based models have achieved state-of-the-art performance in Time Series Forecasting (TSF), yet their evaluation remains dominated by pointwise error metrics such as Mean Squared Error (MSE), which quantify numerical accuracy but overlook structural properties of the forecast signal, including recurrent dynamics, oscillatory behavior, and phase alignment. As a result, forecasts exhibiting over-smoothing, phase shifts, or frequency distortions may achieve favorable error scores despite substantial structural degradation. To address this limitation, we propose TopoCast, a topology-driven framework for evaluating structural fidelity in TSF. TopoCast reconstructs phase-space representations of forecast and ground-truth sequences using Takens delay embedding and applies persistent homology to characterize their intrinsic dynamics. We derive four complementary topological fidelity measures from persistence diagrams and aggregate them into a Topological Fidelity Score (TFS). We further introduce dominant cycle overlap, a novel metric that maps persistent topological features to the temporal domain to assess whether dominant oscillatory patterns occur at the correct time points. Combined with TFS, this yields the Localized Topological Fidelity Score (LTFS), a phase-aware measure that captures temporal localization errors invisible to existing evaluation metrics. Experiments on five Transformer architectures across three real-world benchmark datasets demonstrate that models with similar forecasting errors can exhibit markedly different structural fidelity profiles, revealing failure modes overlooked by conventional evaluation and highlighting the value of topology-aware forecast assessment. 

---
# Interpretable Concept-Guided Polynomial Tabular Kolmogorov-Arnold Network for EEG-Based Mild Cognitive Impairment Detection 

**Authors**: Yosef Bernardus Wirian, Qiang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.25434)  

**Abstract**: Early and scalable detection of mild cognitive impairment (MCI) remains an unresolved clinical challenge. Existing EEG-based screening approaches are constrained by handcrafted feature pipelines that discard neurophysiologically meaningful domain structure and deep learning classifiers that sacrifice interpretability for performance. No existing work unifies physiologically organized concept encoders, cross-concept interaction modeling, and nonlinear tabular classification in a sleep EEG-based MCI detection framework. This study proposes Concept-guided Polynomial-transformed Tabular learning using Kolmogorov-Arnold Network (CPTabKAN), which maps heterogeneous EEG-derived features into domain-informed concept representations, expands them via degree-2 polynomial transformation to expose first- and second-order interactions, and applies a Fourier-parameterized TabKAN classifier to learn nonlinear decision boundaries. CPTabKAN was evaluated on the Study of Osteoporotic Fractures cohort (372 subjects, overnight polysomnography), using 1,379 features organized into ten physiologically motivated concept groups. Under 10-fold cross-validation, CPTabKAN-Second Order achieved a weighted F1-score of 0.9038 (SD 0.034), outperforming GradientBoosting by 5.65 percentage points (t(9)=1.934,p=0.043, one-sided paired test), with advantages persisting under SMOTE-based balancing. Ablation analysis confirmed independent contributions from each component. Concept importance analysis revealed that power spectral density, multi-scale entropy, and Hjorth parameters dominated first-order weights, while cross-concept interactions involving Lempel-Ziv-Welch complexity, statistics, demographics, and slow oscillations exceeded all first-order scores. These results demonstrate that concept-structured, interaction-aware tabular learning surfaces physiologically coherent reasoning, supporting clinical trust. 

---
# Brevity is the Soul of Inference Efficiency: Inducing Concision in VLMs via Data Curation 

**Authors**: DatologyAI, Matthew L. Leavitt, Siddharth Joshi, Haoli Yin, Rishabh Adiga, Haakon Mongstad, Alvin Deng, David Schwab, Bogdan Gaza, Ari Morcos  

**Link**: [PDF](https://arxiv.org/pdf/2606.25432)  

**Abstract**: Inference efficiency is typically pursued by shrinking the model: distillation, pruning, quantization, and sparse routing each lower per-token cost while treating token count as fixed. But output length has been inflating, and it is precisely the component the standard toolkit leaves untouched. Here, we argue that brevity is the missing inference-efficiency lever, and that pretraining data curation is a practical way to pull it: a model trained on concise, correct data learns to answer in fewer tokens; i.e. it has a lower Cost-of-Pass. We apply our VLM curation pipeline to the MAmmoTH-VL single-image subset, and compare models trained on our curated data, the standard MAmmoTH-VL data, and external open-weight frontier VLMs. On a controlled 20-evaluation set and 14 VLMs at 1B-4B activated parameters, we hold output length fixed with a per-model regression, separating brevity from quality, and price models in FLOPs per correct answer. Curation buys a 35x Cost-of-Pass advantage over the most verbose 4B comparator (Qwen3.5-4B) within $\sim$1 pp of accuracy (0.41 vs 14.58 TFLOPs per correct answer; 0.691 vs 0.704 mean accuracy). Curation also buys a +17.55-percentage-point matched-length accuracy gain over the uncurated baseline that grows with model scale (from +16.7 pp at 1B to +21.2 pp at 4B). This brevity improvement concedes no quality: generic verbosity buys no accuracy at any capability or scale, and the window where reasoning-structured verbosity still earns its tokens shrinks from 4 of 8 capability groups at 2B to 1 of 8 at 4B. Per example, the concise model even reaches correct answers the verbose reasoning model misses, marking reasoning as a distinct curation target rather than something brevity gives up. Inference efficiency in this regime is a tokens-per-correct problem, and brevity is the lever that targets it directly. 

---
# Adaptive Oscillatory Inductive Bias for Modeling Sharp Prosodic Dynamics in Diffusion-Based TTS 

**Authors**: Sandipan Dhar, Nirmesh J. Shah, Ashishkumar P. Gudmalwar, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.25424)  

**Abstract**: Diffusion-based text-to-speech (TTS) models have achieved significant improvements in speech quality. However, modeling sharp prosodic transitions and rapid pitch variations in expressive speech remains challenging. Existing diffusion-based TTS decoders commonly utilize periodic nonlinearities such as Snake activation function to capture harmonic structures, but this activation funcation provides limited adaptability when modeling abrupt amplitude and frequency variations. In this paper, we investigate the role of oscillatory inductive bias in diffusion-based TTS decoders and introduce an adaptive oscillatory nonlinearity that enables controllable periodic modulation while maintaining signal stability through a linear bypass component. We refer the resulting TTS system as OscillaTTS. Experiments on the LJSpeech and Emotional Speech Dataset show consistent improvements across objective and subjective evaluations, indicating improved modeling of expressive prosodic dynamics. 

---
# CrossAccent-TTS: Cross-Lingual Accent-Intensity Controllable Text-to-Speech via Disentangled Speaker and Accent Representations 

**Authors**: Ram Annamdevula, Ankit Tatawat, Ashishkumar P. Gudmalwar, Nirmesh J. Shah, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2606.25403)  

**Abstract**: Accent conversion and controllability remain fundamental challenges in cross-lingual text-to-speech (TTS), particularly for low-resource and phonetically diverse Indic languages. While recent large language model (LLM)-based TTS systems exhibit strong cross-lingual generalization, they provide limited explicit control over accent characteristics and intensity. In this paper, we propose CrossAccentTTS, a framework that enables both accent control and conversion while preserving speaker identity. Specifically, we introduce an Accent Intensity Controller (AIC) that injects weighted language embeddings into the accent subspace, allowing smooth interpolation between accents and fine-grained modulation of accent strength at inference time. Experiments on the Indic Multilingual and L2-arctic datasets shows that CrossAccent-TTS achieves precise control of accent intensity, outperforming strong baselines in accent similarity and controllability by maintaining speaker similarity and naturalness. 

---
# LibEvoBench: Probing Temporal Knowledge Stratification in Code Generation Models 

**Authors**: Daniele Cipollone, Sergey Titov, Maliheh Izadi, Egor Bogomolov, Arie van Deursen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25402)  

**Abstract**: Large software projects often depend on older versions of libraries, even as APIs continue to evolve across releases. This creates a challenge for LLMs: they must maintain knowledge of multiple API versions, not merely the latest or most common one. However, current LLMs are trained on temporally mixed corpora and lack explicit mechanisms for such version-specific reasoning, leading to anachronistic errors - calling APIs as they exist in a different library version. To systematically evaluate this phenomenon, we introduce LibEvoBench, a multi-task benchmark spanning multiple versions of widely used Python libraries, along with a new metric, the Software Evolution Understanding Score (SEUS), to measure models' consistency when working with evolving APIs. Our results show that state-of-the-art models are largely version-oblivious: performance degrades for evolving APIs, while for stable APIs it remains the same across versions. Moreover, simply specifying the target version provides no benefit, while relevant documentation significantly boosts models' accuracy. These findings highlight a systematic limitation of current training paradigms and motivate new approaches for temporally grounded knowledge in code generation. 

---
# Lightweight PCGAE-Net: Parallel CrossGate Attention and Bottleneck AutoEncoder for Efficient 5G Channel Prediction 

**Authors**: Uma Kishore Godavarti, K. Giridhar, Vanani Prince Dharmendrabhai, Anchit Panday, Madhan Raj Kanagarathinam  

**Link**: [PDF](https://arxiv.org/pdf/2606.25401)  

**Abstract**: Accurate channel state information (CSI) prediction is essential for proactive beamforming and resource management in 5G massive MIMO systems, yet the deployment of high-accuracy transformer-based predictors on base-station hardware remains challenging because the most capable models carry upwards of 30\,M parameters. This paper introduces Lightweight PCGAE-Net, which addresses the efficiency problem not by post-hoc compression but by correcting two architectural flaws in the current state of the art. The first is a sequential attention ordering bias: in CS3T-UNet, group-wise temporal attention (GTA) always operates on features that have already been transformed by cross-shaped spatial attention (CSA), distorting what temporal information GTA can capture. We remove this dependency by routing both attention modules to the same layer-normalized input and combining their independent outputs through a learned per-channel sigmoid CrossGate. The second flaw is an uncompressed bottleneck: applying full self-attention at the deepest encoder stage, where channel depth reaches $4C$, is quadratically expensive and carries redundant features. A Bottleneck AutoEncoder (BAE) with $1\times1$ convolutions halves this depth and uses an auxiliary reconstruction loss to prevent information collapse. Wrapping these components inside a shallower encoder-decoder with frequency-domain dimensionality reduction ($N_f\!=\!32$, $C\!=\!48$) produces a model with just 8.54\,M parameters -- 58\% fewer than the CS3T-UNet baseline -- that outperforms it by up to 3.26\,dB at 5\,km/h and 6.0\,dB at 9\,km/h in single-step prediction on QuaDriGa dataset. 

---
# FactorLibrary: From Polynomials to Circuits via Recursive Subgoals 

**Authors**: Rohan Pandey, Michael Ruofan Zeng, Weikun K. Zhang, Kaijie Jin, Naomi Morato, Archit Ganapule, Bhaumik Mehta, Jarod Alper  

**Link**: [PDF](https://arxiv.org/pdf/2606.25394)  

**Abstract**: Finding minimal arithmetic circuits for polynomials over finite fields is a combinatorially hard problem central to algebraic complexity theory. We formulate it as a reinforcement learning problem in two directions, bottom-up and top-down. To address the challenge of a fast-growing combinatorial search space, we introduce FactorLibrary, which stores factorizable subexpressions that serve as reusable subgoals across training episodes. We trained a bottom-up agent with Gumbel-PPO-MCTS and two top-down agents with PPO+MCTS and SAC. The PPO+MCTS top-down agent exhibited the most stable performance, finding certified optimal circuits up to complexity $8$ with a success rate of $91.8\%$. 

---
# From Sounds to Scenes: A Benchmark for Evaluating Context-Aware Auditory Scene Understanding in Large Audio Language Models 

**Authors**: Pengfei Zhang, Hoang H Nguyen, Kazi Shaharair Sharif, Yutong Song, Wenjun Huang, Henry Peng Zou, Pinxin Liu, Honghui Xu, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2606.25391)  

**Abstract**: Recent Large Audio Language Models (LALMs) have achieved remarkable progress in audio perceptual tasks across individual acoustic layers, including speech, sound, and music. However, existing benchmarks predominantly evaluate these layers in isolation, overlooking the complex contextual relationships that arise when multiple acoustic sources co-occur in real-world auditory scenes. Real-world auditory interpretation requires Context-Aware Auditory Scene Understanding (CASU): the ability to comprehend the holistic scene by integrating sound layers. To evaluate this capability, we introduce the CASU benchmark, which assesses whether Audio LLMs can interpret auditory scenes composed of speech, acoustic events (e.g., announcements), and background environments (e.g., traffic), and reason about the logical relationships between these layers. We propose a scalable pipeline for constructing time-accurate, semi-synthetic audio streams by composing real-world scene sounds with synthetic speech. Building on this data, we design four tasks that probe scene understanding: contextual question answering, entity extraction from the scene, speaker role inference, and counterfactual reasoning where scene is manipulated. Experiments across multiple LALMs demonstrate that effective auditory scene understanding requires integration over all auditory layers, rather than reliance on speech or sound alone, underscoring the necessity of CASU for advancing complex audio understanding in LALMs. 

---
# Anatomically-conditioned Latent Diffusion Model for Data-Efficient Few-Shot Cross-Domain 3D Glioma MRI Synthesis 

**Authors**: Salman Shaik, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2606.25390)  

**Abstract**: Accurate classification of diffuse gliomas is often hindered by domain shifts across centers and a lack of large, annotated datasets. We propose the Anatomically-conditioned Latent Diffusion Model (ALDM), a novel framework for data-efficient, few-shot 3D volumetric MRI synthesis. ALDM utilizes a two-stage approach: a 3D variational autoencoder learns anatomical priors from a data-rich source domain, while a conditional latent diffusion model, guided by tumor masks via a ControlNet, generates structurally coherent volumes for a data-scarce target domain. Evaluated in an extreme few-shot setting with only 16 target images, ALDM outperformed GAN and hybrid baselines, achieving a superior Frechet Inception Distance (FID) of 85.40 and a downstream classification AUC of 0.987. Qualitative results confirm that the model preserves sharp pathology boundaries and cross-modal consistency, with visual fidelity improving progressively during training. By capturing essential diagnostic features, ALDM provides a robust tool for clinical data augmentation in low-resource settings. Our implementation is available at this https URL. 

---
# Beyond Visual Forensics: Auditing Multimodal Robustness for Synthetic Medical Image Detection 

**Authors**: Ching-Hao Chiu, Hao-Wei Chung, Gelei Xu, Xueyang Li, Pin-Yu Chen, John Kheir, Meysam Ghaffari, Carlos Morato, Ahmed Abbasi, Yiyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25375)  

**Abstract**: With the rapid adoption of generative AI, synthetic medical images pose growing risks, including diagnostic deception and insurance fraud. Although prior work has explored vision-language model (VLM)-based synthetic image detection, these evaluations typically consider images in isolation. In clinical practice, however, images are interpreted alongside structured records and metadata, and VLMs are increasingly deployed under joint image-record inputs. We uncover a previously underexamined multimodal vulnerability: when given both modalities, VLMs may overweight record context in authenticity judgments, such that the same image receives different predictions solely due to changes in its accompanying text. This raises concerns about robustness in real-world deployment. To systematically characterize this effect, we reformulate synthetic medical image detection as an audit of multimodal robustness at the image-record interface and introduce a paired benchmark that holds the image fixed while swapping controlled metadata variants. Across multiple imaging modalities, we evaluate diverse open-weight and frontier API VLMs and quantify how metadata alone shifts authenticity predictions. Our benchmark provides a standardized tool for assessing and improving multimodal robustness beyond image-only settings. The code is available at this https URL. 

---
# Conformal Recovery-Deadline Certificates for Runtime Assurance of Adapting Controllers 

**Authors**: Alireza Shojaei  

**Link**: [PDF](https://arxiv.org/pdf/2606.25371)  

**Abstract**: Runtime assurance (RTA) protects a safety-critical system by switching from an advanced controller to a verified safe controller when a monitored condition is violated. The standard latching rule, which trips on the first breach of the safe set and then coasts, is correct for a diverging controller but pathological for a capable online-adapting one. Such a controller is unsafe by design during a bounded recovery transient. It must excite the plant to identify the fault before it can correct it, so a latching shield trips on that transient and suppresses a controller that would have recovered. We introduce the conformal recovery-deadline certificate, a split-conformal, distribution-free, finite-sample upper bound on the adapting controller's recovery time that licenses delayed fallback with a coverage guarantee, backstopped by a verified monitor at a hard critical limit. The certified deadline discriminates capable from incapable controllers, keeping the recoverer autonomous while catching the diverger. The construction separates autonomy, governed by statistical coverage, from safety, governed by the verified backstop, as an instance of reliability-asymmetric design. We prove marginal coverage, a weighted extension that restores coverage under a known fault-distribution shift, and group-conditional Mondrian coverage. We demonstrate all three on two unrelated Simplex testbeds: a 6-DOF spacecraft attitude controller and a torque-controlled inverted pendulum. Both show the same suppression pathology and the same cure, making the certificate a domain-general mechanism rather than a single-system trick. 

---
# Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis 

**Authors**: Lianbo Liu, Shiao Zhu, Kai Washizaki, Reo Yoneyama, Haesung Jeon, Mengjie Zhao, Yusuke Fujita, Hao Shi, Nao Yoshida, Yuan Gao, Roman Koshkin, Yukiya Hono, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25369)  

**Abstract**: While large language model (LLM)-based text-to-speech (TTS) systems have achieved high-quality speech synthesis, most existing systems focus on English and Chinese. Japanese, however, remains under-explored, and its unique linguistic challenges, such as widespread context-dependent kanji polyphony, have yet to be adequately tackled. Here we introduce Sarashina2.2-TTS (this https URL), a Japanese-centric LLM-TTS system that tackles these challenges through a dual approach: data strategy and evaluation methodology. First, we scale training to approximately 361k hours of speech, incorporating a balanced mix of Japanese and English data. Furthermore, we design a targeted data augmentation pipeline covering all 2,136 Joyo (regular-use) kanji designated by Japan's Agency for Cultural Affairs to efficiently address kanji polyphony disambiguation. Second, we introduce the Joyo Kanji Yomi Benchmark (this https URL), covering all 2,136 Joyo kanji and their 4,378 readings. Alongside this benchmark, we propose Kana-CER, a metric that compares synthesized speech against reference readings in the kana space, eliminating orthographic variations to directly measure pronunciation correctness. Experiments demonstrate that our targeted data augmentation significantly improves reading accuracy. Overall, Sarashina2.2-TTS achieves state-of-the-art kanji-level reading accuracy and matches top baselines on general sentence-level pronunciation, while delivering the highest speaker similarity in zero-shot Japanese speech synthesis. Furthermore, cross-lingual evaluation reveals that Sarashina2.2-TTS is the only system that maintains stable Japanese pronunciation regardless of the prompt language, confirming that our balanced training approach improves cross-lingual robustness. 

---
# Reliability-Asymmetric Spacecraft Autonomy: Co-Designing a Capable Learned GNC Stack with a Verified, Adaptation-Aware Runtime Shield 

**Authors**: Alireza Shojaei  

**Link**: [PDF](https://arxiv.org/pdf/2606.25366)  

**Abstract**: Deep-space missions need onboard autonomy that is both capable and certifiable. Rule-based autonomy is certifiable but brittle, while learned autonomy is capable but hard to verify. We present AMPLE-GNC, a three-tier guidance, navigation, and control stack. Its capability path combines a small foundation-model commander that maps natural language to PDDL+, a constraint-screening verifier, and a fault-adaptive controller. All three are bounded by a runtime shield with nine linear-temporal-logic invariants whose predictor soundness is machine-checked by the Kind 2 model checker. On a 6-DOF Basilisk testbed, we make three contributions. First, we deploy an edge commander. Fine-tuning a pretrained 360M model with grammar-constrained decoding gives a hard output-validity guarantee and 84% planner-executable actions. On a de-leaked test, novel-phrasing generalization is 38% exact and 51% action, rising to 48% exact after phrasing-diversity re-finetuning; we separate syntactic validity from semantic accuracy. Second, we introduce a fault-adaptive controller. Rapid Motor Adaptation infers latent actuator faults online and recovers 97.8% of actuator-sign faults and 94.4% of continuous-gain faults within the training randomization envelope. Fault-unaware PD and from-scratch end-to-end RL both score 0%, while the strongest classical-adaptive baseline reaches 55% on continuous gain. Beyond the envelope, a split-conformant retrain scores 57-67%, and adding 4x more in-regime data worsens performance, showing that randomization breadth, not data volume, drives generalization. Robustness is flat under star-tracker noise to 0.005. Third, we show that a latching safe-hold shield can suppress even a capable controller. A split-conformal recovery-deadline certificate with adaptation-aware engagement reconciles safety and recovery, keeping the controller 94.5% autonomous while still catching non-recovery. 

---
# Neural Machine Translation for Low-Resource Tangkhul--English 

**Authors**: Chormi Zimik Vashai, Agniva Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2606.25365)  

**Abstract**: We present a study on low-resource machine translation for the Tangkhul-English (nmf-en) language pair. Tangkhul is a severely under-resourced Tibeto-Burman language spoken primarily in Manipur, India, with virtually no prior natural language processing infrastructure. We describe two systems: (1) a primary system based on ByT5-large fine-tuned on 38,336 Tangkhul-English parallel sentence pairs, and (2) a contrastive system based on mT5-small fine-tuned on the same corpus. Our primary ByT5-large system achieves a corpus BLEU score of 39.97, chrF++ of 58.07, BERTScore F1 of 0.8104, and COMET (wmt22-comet-da) of 0.7302 on a held-out test set of 3,856 sentences. We further discuss the orthographic challenges specific to Tangkhul's Latin-script diacritics, the domain bias of our training corpus (which comprises biblical text, stories, and conversational data), and avenues for future improvement through data diversification and domain adaptation. 

---
# TheoremGraph: Bridging Formal and Informal Mathematics 

**Authors**: Simon Kurgan, Evan Wang, Eric Leonen, Sophie Szeto, Luke Alexander, Artemii Remizov, Jarod Alper, Giovanni Inchiostro, Vasily Ilin  

**Link**: [PDF](https://arxiv.org/pdf/2606.25363)  

**Abstract**: Mathematical knowledge is organized around statements and their dependencies, but this structure is exposed unevenly: informal papers cite mostly at the document level, while formal libraries record fine-grained dependencies over a much smaller body of mathematics. We introduce TheoremGraph, a unified statement-level dependency graph spanning both informal and formal mathematics. On the informal side, we parse 11.7M theorem-like environments from mathematics arXiv and recover 18.3M candidate directed dependencies, each labeled by the extractor that proposed it so downstream users can trade coverage for precision. On the formal side, we release LeanGraph, a Lean 4 elaborator-level extractor producing 388,105 declaration nodes and 11.3M typed edges across 25 Lean projects. We bridge the two graphs by embedding generated natural-language slogans into a shared semantic space, linking related statements across papers and across the informal/formal divide; an LLM judge affirms 47,952 such matches above a 0.8 cosine floor, with the judge-acceptance rate rising from 48% across the floor to 87% in the >=0.9 tier. On formal concept retrieval, our name-and-signature representation with graph expansion comes within 0.5pp of LeanSearch v2's reranked Recall@10 (0.775 vs. 0.780) without an LM reranker. We release the dataset, extractors, HTTP API, and MCP interface as infrastructure for mathematical search, attribution, and retrieval-augmented reasoning, available at this http URL and this http URL. 

---
# Memory Makes the Difference: Evaluating How Different Memory Roles Shape Conversational Agents 

**Authors**: Yuxin Wang, Paul Thomas, Zhiwei Yu, Yuan Gao, Saeed Hassanpour, Soroush Vosoughi, Robert Sim, Nick Craswell  

**Link**: [PDF](https://arxiv.org/pdf/2606.25361)  

**Abstract**: Prior research on memory mechanism in RAG-based conversational system has emphasized how memory is stored and retrieved. However, far less is known about how memories with different functional roles influence response quality. Specifically, how they shape an agent's responses under varying conversational contexts and whether they lead to substantively different response behaviors. Existing evaluations in conversational system are also largely reference-based, insufficiently capturing the nuances in responses that may address users' preferences differently. In this work, we probe the impact of different memory types in shaping agents' responses. We present a fine-grained taxonomy of conversational memory, classify retrieved memories into different role types, and design a user-centric evaluation framework that simulates user perspectives. Through comparative experiments on long-term datasets and frontier LLMs, our analysis reveal many differentiated effects of memories: e.g., clarifying memory improves responses' factual accuracy and constraint awareness, making them more correct and personalized; irrelevant memory reduces topic relevance and degrades constraint awareness. Despite the power of frontier LLMs, these findings shed light on how different memory types can be leveraged to produce more personalized responses and inspire further research in this direction. 

---
# Compositional Behavioral Semantics for State Abstraction in Reinforcement Learning 

**Authors**: Yivan Zhang, Ziyan Luo, Manuel Baltieri  

**Link**: [PDF](https://arxiv.org/pdf/2606.25357)  

**Abstract**: State abstraction plays a key role in scaling reinforcement learning to complex but structured systems. In studying such systems, a wide range of behavioral structures have been studied in reinforcement learning, including value functions, invariants, bisimulation relations, and behavioral metrics. However, a general principle for determining what structures are provably preserved under state abstraction is still lacking. In this paper, we present a unified framework for defining and analyzing behavioral structures in reinforcement learning. Our framework provides a compositional way to specify behavioral semantics based on local, one-step descriptions of system dynamics. Using this framework, we establish results showing how behavioral structures can be safely transferred between abstract and concrete systems. We further show how to construct quantitative metrics from logical behavioral semantics with soundness guarantees. Together, these results provide a principled foundation for reasoning about behaviors under state abstraction in reinforcement learning and offer reusable definition and proof principles for a broad class of behavioral structures in reinforcement learning. 

---
# AI Coaching for Accelerating Human Skill Development with Reinforcement Learning 

**Authors**: Wei Wang, Enlin Gu, Antonio Loquercio, Haimin Hu, Rahul Mangharam  

**Link**: [PDF](https://arxiv.org/pdf/2606.25337)  

**Abstract**: AI copilots can substantially boost human performance through shared control, but excessive assistance can induce over-reliance and skill atrophy. This paper studies how an embodied AI agent can act as a coach that accelerates human motor-skill development. We argue that effective coaching requires strategic scaffolding and stepping back that are aligned with the learner's capability, allowing productive failures that drive learning. We formalize the interactive AI coaching process as a non-cooperative dynamic game in which the learner optimizes task performance while the coach targets the learner's independent competence. Building on this formalism, we develop a reinforcement learning framework combining adaptive shared control with probabilistic models of the coach's causal influence on skill evolution, enabling tractable training of coaching policies. A comprehensive user study (N=33) on first-person-view drone racing shows significant gains in human learning outcomes over state-of-the-art AI coaching baselines. 

---
# Decoupling Reconnaissance and Exploitation: Measuring the Capability Boundaries of LLM-Based Web Penetration Testing 

**Authors**: Liwei Yu, Shuo Li, Ming Zhou, Ge Chu, Yan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.25332)  

**Abstract**: Large Language Models (LLMs) have shown promise for automated penetration testing, yet existing end-to-end black-box evaluations are highly susceptible to error cascading: failures in early reconnaissance can mask an agent's actual ability to exploit vulnerabilities. To more accurately characterize these capabilities, we propose a two-stage decoupled evaluation framework that separates exploit execution from reconnaissance. Using ground-truth injection and knowledge-driven ablation across 70 high-fidelity web vulnerability testbeds, our framework isolates exploitation performance from reconnaissance noise. We empirically evaluate five open-source penetration-testing agents, covering multiagent, monolithic, and graph-driven architectures, on a strictly aligned subset of 50 representative vulnerabilities. The results reveal a substantial capability gap. With accurate vulnerability context, agents achieve a functional success rate of up to 90.0%, whereas autonomous reconnaissance, measured by targeted vulnerability recall, plateaus at approximately 50.0%, primarily due to failures in parsing unstructured telemetry. Cross-architectural analysis further reveals distinct capability niches: multi-agent isolation is more effective for long-sequence interactions such as de-serialization, while monolithic and graph-driven designs perform better on short-chain injections and cross-session access-control vulnerabilities, respectively. This decoupled evaluation work provides a fine-grained benchmarking protocol and an empirical basis for designing next-generation automated offensive security agents. 

---
# Improved Large Language Diffusion Models 

**Authors**: Shen Nie, Qiyang Min, Shaoxuan Xu, Zihao Huang, Yuxuan Song, Yong Shan, Yankai Lin, Wayne Xin Zhao, Chongxuan Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2606.25331)  

**Abstract**: Modern large language models are predominantly trained with autoregressive factorization and causal attention. We present \emph{iLLaDA}, an 8B masked diffusion language model trained from scratch with fully bidirectional attention. iLLaDA keeps the masked diffusion objective throughout pre-training and supervised fine-tuning (SFT), scaling pre-training to 12T tokens and fine-tuning on a 25B-token instruction corpus for 12 epochs. We further use variable-length generation for efficiency and introduce confidence-based scoring for multiple-choice evaluation. Compared with LLaDA, iLLaDA improves broadly across general, mathematical, and code benchmarks; for example, iLLaDA-Base improves by 21.6 points on BBH and 14.9 points on ARC-Challenge, while iLLaDA-Instruct improves by 14.5 points on MATH and 16.5 points on HumanEval. Despite its non-autoregressive training, iLLaDA also remains competitive with Qwen2.5 7B on several benchmarks. These results show that fully bidirectional diffusion training from scratch is a competitive path toward strong language models. Model weights and codes: this https URL. 

---
# Supervised Post-training of Speech Foundation Models for Robust Adaptation in Speech Deepfake Detection 

**Authors**: Zihan Pan, Sailor Hardik, Jinyang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25328)  

**Abstract**: Large speech foundation models have shown strong potential for speech deepfake detection, but direct fine-tuning is limited by a mismatch between self-supervised pre-training objectives and spoof-specific artifacts. To address this, we propose a mix-frame post-training strategy to create localized spoof-oriented perturbations and use frame-level supervision to encourage the SSL model to learn local inconsistencies that are critical for robust spoof detection. On ASVspoof5, we achieve state-of-the-art EER 4.50% for a single model without data augmentation. On ASVspoof2021 LA/DF, it further achieves only 0.16\% absolute EER gap between LA and DF, indicating strong and balanced robustness across distinct distortion conditions. These results show that supervised post-training provides an effective and practical way to adapt speech foundation models for robust deepfake detection. 

---
# ESTANet: Efficient Online Error Detection in Procedural Videos via Prediction Inconsistency 

**Authors**: Shih-Po Lee, Reza Ghoddoosian, Faizan Siddiqui, Enna Sachdeva, Behzad Dariush  

**Link**: [PDF](https://arxiv.org/pdf/2606.25317)  

**Abstract**: An efficient and accurate system for detecting errors in procedural tasks is crucial for supporting human needs in daily life, as it can provide instant notifications and guide people to correct mistakes. In this work, we study real-time online error detection in procedural videos from a simple but overlooked perspective: the prediction behavior of action detectors themselves. Instead of designing complex architectures or specialized supervision, we observe that action detectors naturally exhibit different prediction characteristics depending on their sensitivity to input dynamics and temporal context. We therefore propose ESTANet (Error-Sensitive and Temporally-vArying Network), a lightweight framework that detects errors by exploiting inconsistencies among action predictions produced by a small set of action detectors. We construct standard and error-sensitive action detectors that behave similarly on correct executions but respond differently when errors occur. Meanwhile, detectors operating with different temporal contexts further amplify prediction inconsistencies when the procedure deviates from the intended sequence. During inference, we detect errors by aggregating mismatches between standard and error-sensitive predictions through majority voting to flag frames that contain errors. Extensive experiments on EgoPER, Assembly-101-O, and EPIC-Tent-O demonstrate that ESTANet achieves state-of-the-art performance in online error detection while maintaining real-time efficiency with a lightweight architecture. Our results highlight that leveraging the intrinsic properties of action detectors can yield a powerful and practical solution for online error detection without increasing architectural design complexity. 

---
# Physics Question Scene Graph: Fine-grained Evaluation of Physical Plausibility in Text-to-Video Generation 

**Authors**: Atin Pothiraj, Jaemin Cho, Yue Zhang, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2606.25306)  

**Abstract**: Video generation models are increasingly capable of producing realistic videos, but they still struggle to generate videos that follow basic physical laws. Compounding this is a lack of reliable granular evaluation methods for localizing and specifying physical law violations in videos. We address this by introducing Physics Question Scene Graph (PQSG), a hierarchical question-based evaluation pipeline. PQSG evaluates generated videos by checking their faithfulness to a prompt across objects, actions, and adherence to physical laws using a graph-based hierarchy of questions generated by a vision-language model (VLM), guided by high-quality in-context examples. By representing questions as a graph, PQSG introduces logical dependencies within questions, ensuring that each query is contextually valid. Moreover, PQSG provides granular assessments of which qualities of the video violate physical plausibility constraints. We validate PQSG by creating FinePhyEval, a dataset with physics-based prompts and corresponding generated videos from diverse state-of-the-art video generation models (Sora 2, Veo 3, and Wan 2.1), with each video annotated across multiple categories by humans. Using FinePhyEval, we measure the correlation between PQSG's fine-grained scores and human judgments, showing higher overall correlations than prior work. We also find that PQSG ranks closed-source models higher than Wan 2.1 on physical realism. Lastly, we show that the annotations we provide in FinePhyEval can also be used for subtask evaluation: we benchmark two strong VLMs on generating and answering questions, finding that while models can create human-like questions, they still fall short of human performance in answering them. 

---
# Communicability-Inspired Positional Encoding (CIPE) 

**Authors**: Yipeng Zhang, Zhongtian Sun, Pietro Liò, Kelin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2606.25293)  

**Abstract**: Positional encodings (PEs) are essential for Transformers. Yet designing effective PEs for non-Euclidean graphs remains challenging. Such encodings should ideally induce an Attention-Compatible Geometry for self-attention: not merely describing graph structure, but defining a geometry whose inner products reflect meaningful structural relatedness. To realize this geometry, we propose Communicability-Inspired Positional Encoding (CIPE), built from communicability, a measure between pairs of nodes that aggregates contributions from paths of all lengths. By construction, CIPE inner products recover communicability, converting global multi-path connectivity into an attention-ready similarity geometry. For practical Transformer training, we introduce dimensionality alignment, mapping graph-size-dependent CIPE representations to prescribed dimensions while faithfully preserving the induced geometry. Empirically, CIPE improves structure-agnostic Transformers by 35.5% on average across seven benchmarks, outperforming representative PEs; it also consistently improves structure-biased graph Transformers, where competing PEs often yield only marginal benefits. These results position CIPE as a principled framework for attention-compatible graph positional encodings. 

---
# EPTS: Elastic Post-Training Sparsity for Efficient Large Language Model Compression 

**Authors**: Ke Xu, Jiaqi Wan, Wenhao Hu, Han Pu, Xiaoyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25285)  

**Abstract**: Post-Training Sparsity (PTS) has emerged as a crucial paradigm for compressing Large Language Models to facilitate efficient deployment on resource-constrained devices. However, existing PTS methodologies are typically confined to Single-Sparsity optimization, necessitating a separate, time-consuming optimization session for each specific sparsity level. This rigid paradigm significantly hinders flexible deployment across diverse hardware scenarios, as adapting to a new sparsity requirement mandates a complete re-optimization process. To address these limitations, we propose Elastic Post-Training Sparsity (EPTS), a unified Multi-Sparsity framework that produces a single elastic model capable of maintaining robust performance across diverse sparsity configurations through a one-shot optimization process. Specifically, we design a Multi-Sparsity Hierarchy LoRA (MS-HiLoRA) mechanism that facilitates knowledge inheritance from low- to high-sparsity groups, effectively mitigating the competition for parameter reconstruction. Furthermore, we introduce a Multi-Sparsity Feature Mixer (MSFM), which significantly enhances the model's adaptability to pruning perturbations by dynamically fusing feature representations of varying sparsity granularities. Extensive experiments on LLaMA and OPT families demonstrate that EPTS achieves competitive performance compared to state-of-the-art methods like SparseGPT and Wanda, while offering significant efficiency gains by enabling multi-scenario deployment from a single optimization. our source code is available at this https URL. 

---
# Heterogeneous and Adept Snapshot Distillation for 3D Semantic Segmentation 

**Authors**: Xiaopei Wu, Yuenan Hou, Junkai Xu, Wenxiao Wang, Binbin Lin, Yu Li, Ping Li, Haifeng Liu, Deng Cai, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25278)  

**Abstract**: Multi-modal fusion and multi-model ensembling are prevalent in enhancing the performance of 3D semantic segmentation. Despite the impressive performance, these methods either rely on auxiliary input signals or suffer from costly computational expense. To efficaciously enhance the segmentation performance without introducing intolerable costs, we propose to transfer the rich knowledge from the multi-modal model (i.e., point clouds and images) and multiple model experts to the point-cloudbased network through knowledge distillation. Specifically, we present Information-oriented Heterogeneous Distillation (IHD) to help the uni-modal model absorb the complementary knowledge from the multi-modal teacher. We design the Information-Oriented Filtering (IOF) strategy to select informative images from the continuous image sequence for multi-modal fusion. This practice can boost the performance of the multi-modal teacher, thus benefiting the learning of the student. Besides, as opposed to vanilla model ensembling that requires the separate training of each expert, we propose Adept Snapshot Distillation (ASD). ASD treats the freely available model snapshots generated during the training phase as multiple experts, which significantly reduces the training cost for model ensembling. For each expert teacher, it only provides supervision to the student in the class where it is adept. The resulting Heterogeneous and Adept Snapshot Knowledge Distillation, dubbed HAS-KD, attains state-of-the-art results on ScanNetV2 and S3DIS datasets. HAS-KD can be seamlessly integrated into contemporary 3D segmentation algorithms and bring considerable gains without introducing extra inference burdens. The code will be made publicly available upon publication. 

---
# UC-Search: Risk-Aware Test-Time Search for Delayed Constrained Time-Series Control 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25274)  

**Abstract**: Time-series models are usually scored as forecasters, yet deployed systems often require delayed decisions under uncertainty and hard feasibility constraints. UC-Search is a model-agnostic test-time wrapper: a backbone emits forecasts or action scores, a feasibility automaton rolls candidate paths forward, and bounded search returns the first action of a risk-adjusted feasible trajectory. We instantiate UC-Beam and a UCT-style UC-MCTS diagnostic, using epistemic, aleatoric, and propagated uncertainty mainly as path-risk terms. A myopic-collapse/separation theorem states when search reduces to one-step risk-greedy and when delayed feasible-set coupling can create non-myopic value. Primary evidence comes from a predeclared public $9$-family, $33$-series delayed-control suite with six held-out starts per series: UC-Pareto is positive versus validation-selected CEM, MPPI, and risk-aware random at the normalized threshold ($+3.1675/+2.3328/+2.5038$), and remains positive in a compute-matched audit ($+2.8466/+2.7418/+2.7429$). ETT/LTSF delayed-inventory validation supports the same compute-frontier claim. A 48-series raw M4 standard periodic-review lost-sales inventory audit is positive versus the strongest classic base-stock control ($+13556.7547$), CEM ($+64900.2207$), and risk-random ($+52881.6042$), while MPPI remains family-mixed. FI-2010, official-forecast adapters, SB3/FQI controls, direction/capacity/intervention checks, and synthetic mechanism tests are reported as boundary or mechanism evidence rather than broad dominance claims. 

---
# Stabilizing black-box algorithms through task-oriented randomization 

**Authors**: Yali Wang, Zhaojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25269)  

**Abstract**: As black-box models become foundational to modern research, ensuring their stability is paramount for the realization of trustworthy artificial intelligence. The inherent diversity of inputs - ranging from structured Gaussian distributions to complex data with unknown structures - poses a significant challenge: how to stabilize black-box outputs while effectively leveraging available prior information. This paper introduces a task-oriented randomization methodology that adaptively tailors its strategy to the underlying generative mechanisms of the input data, specifically addressing unstructured complexities. A comprehensive suite of stability guarantees is proposed. Beyond establishing rigorous theoretical foundations for stability, the research provides a detailed analysis of the intrinsic trade-off between stability and exploration. Motivated by the architecture of Large Language Models, the framework is further extended to top-k ranking problems. The validity and effectiveness of the proposal are demonstrated through extensive numerical simulations and applications to the real-world dataset. 

---
# ASAP: Agent-System Co-Design for Wall-Clock-Centered Auto HPO Research for ML Experiments 

**Authors**: Taicheng Guo, Haomin Zhuang, Kehan Guo, Yujun Zhou, Nitesh V. Chawla, Olaf Wiest, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25207)  

**Abstract**: Hyperparameter Optimization (HPO) is essential for maximizing machine learning model performance, and its core challenge is sample efficiency: finding strong configurations within a limited budget. Because every HPO tool relies on a surrogate prior that imparts its own inductive bias, individual tools struggle once problems become sufficiently diverse and drift from these priors. Motivated by the reasoning and generalization capabilities of LLMs, recent work has explored using LLMs for HPO and reports improved per-iteration performance. Yet these methods share two limitations with a common origin: they use the LLM as a single-tool replacement evaluated by iteration count. (i) Deployed in place of prior tools, the LLM is itself constrained by its pretraining objective to one family of inductive-biased proposals; this single-source setup still fails to handle the full diversity of problems. (ii) Per-iteration evaluation ignores that, in real runs, LLM inference or tool execution is paid serially on top of model evaluation every round, so iteration-count gains do not translate into end-to-end wall-clock gains. We present ASAP, an agent-system co-design that addresses both limitations. On the agent side, ASAP uses the LLM to integrate a diverse pool of inductive-biased optimizers and to select among their proposals each round. On the system side, ASAP re-architects the loop to reduce end-to-end wall-clock while preserving regret quality: a prefix-stable prompt maximizes KV-cache reuse across rounds; speculation parallelism hides the remaining LLM and tool latency under model evaluation via a relative-error accept test; and a Self-Tuner adapts the speculation threshold from execution logs off the critical path. Extensive experiments on diverse modern HPO tasks show that ASAP consistently outperforms baselines, underscoring the value of tool integration and agent-system co-design. 

---
# RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory 

**Authors**: Yixun Hu, Zhicheng Zheng, Lihan Zha, Chunwei Xing, Rajdeep Singh, Omar Hossain, Antonio Loquercio, Dhruv Shah  

**Link**: [PDF](https://arxiv.org/pdf/2606.25206)  

**Abstract**: Long-term robot deployment requires a compact and scalable memory that preserves fine-grained visual semantics, grounds observations in space and time, and enables efficient storage and retrieval. In this paper, we propose RAVEN, an agentic memory system for long-horizon robotic question answering and navigation. RAVEN stores visual embeddings with pose and time in a vector database, and grounds retrieval in a spatial map to answer queries and navigate to goals. By operating directly on visual embeddings, RAVEN avoids lossy image-to-text captioning and enables accurate semantic, spatial, and temporal retrieval at scale. Across several simulated and real-world video question-answering benchmarks, RAVEN consistently surpasses caption-based memory systems and matches frontier VLMs on long-horizon tasks at 10$\times$ lower retrieval cost. Finally, we instantiate RAVEN on a Unitree Go1 robot for the task of long-horizon navigation for natural language goal-reaching, and show successful deployment over several large indoor environments. 

---
# FDN: Interpretable Spatiotemporal Forecasting with Future Decomposition Networks 

**Authors**: Nicholas Majeske, Ariful Azad  

**Link**: [PDF](https://arxiv.org/pdf/2606.25201)  

**Abstract**: Spatiotemporal systems comprise a collection of spatially distributed yet interdependent entities each generating unique dynamic signals. Highly sophisticated methods have been proposed in recent years delivering state-of-the-art (SOTA) forecasts but few have focused on interpretability. To address this, we propose the Future Decomposition Network (FDN), a novel forecast model capable of (a) providing interpretable predictions through classification (b) revealing latent activity patterns in the target time-series and (c) delivering forecasts competitive with SOTA methods at a fraction of their memory and runtime cost. We conduct comprehensive analyses on FDN for multiple datasets from hydrologic, traffic, and energy systems, demonstrating its improved accuracy and interpretability. 

---
# A Hybrid CNN-LSTM Intrusion Detection Framework for Cybersecurity in Smart Renewable Energy Grids 

**Authors**: Sajib Debnath, Remon Das  

**Link**: [PDF](https://arxiv.org/pdf/2606.25200)  

**Abstract**: The accelerated digitalization of renewable energy smart grids through IoT sensors, AMI, and SCADA systems has significantly expanded the attack surface for sophisticated cyberattacks, FDI attacks that stealthily distort state estimation and DoS/DDoS attacks that flood communication channels. Current IDS, however, exhibit three inherent limitations: inadequate modeling of the temporal progression of multi-step attacks, degraded scalability under extremely skewed class distributions of standard benchmark datasets, and restricted generalization across heterogeneous network environments. In this study, we present a Hybrid CNN-LSTM IDS that jointly exploits CNN-based spatial feature extraction and LSTM-based temporal sequence modeling, enabling the detection of instantaneous volumetric anomalies and gradually evolving low and slow-attack campaigns in real time. The model was trained using a seven-step preprocessing workflow comprising missing-value imputation, min-max normalization, one-hot encoding, SMOTE class balancing, mutual-information feature selection, causal temporal sequence construction (T=10), and stratified partitioning. LSTM (96.1%), Random Forest (93.5%), SVM (91.2%) and KNN (89.7%); in NSL-KDD, it reaches 98.2% precision versus 96.4% (LSTM), 95.2% (CNN), 92.7% (Random Forest) and 90.8% (SVM), with margins of 2-9 percentage points in all measures. An ablation analysis identified SMOTE balancing as the most influential design choice (-3.7~pp F1 without it). The model achieves a real-time inference throughput of 27,800 flows/s on GPU and 0.082 ms/sample CPU latency in FP32,, with INT8 quantization providing an additional 3.1 x speedup at 0.3% accuracy loss, confirming deployment feasibility on resource-constrained IEDs with <128MB memory and establishing a deployable deep-learning framework for securing next-generation renewable energy smart grid infrastructure. 

---
# SoK: AI Secure Code Generation: Progress, Pitfalls, and Paths Forward 

**Authors**: Rupam Patir, Keyan Guo, Haipeng Cai, Hongxin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.25195)  

**Abstract**: The increasing use of AI systems for code generation raises a central security question: what can today's models and coding agents actually do to produce secure code, where do they still fail, and what would move the field forward? Existing work has explored prompting, fine-tuning, reinforcement learning, and agentic workflows for secure code generation, but the field still lacks a systematic understanding of how these techniques improve security and why substantial failures persist. In this SoK, we systematize the progress, pitfalls, and paths forward for AI secure code generation. We introduce a three-level framework that measures models' natural-language understanding of secure coding principles, their code-level actuation of those principles during generation, and the knowledge--actuation gaps between the two. We instantiate this framework across models and coding agents on benchmarks covering both isolated function-level security and full web-application security. Our results show that secure-coding-principle understanding is a statistically strong predictor of code-level outcomes, including functional correctness, security, and joint functional-security correctness. Yet substantial knowledge--actuation gaps remain: models can recognize relevant security principles but still fail to translate them into secure and functional code. These findings offer a principle-centered account of where AI secure code generation stands today and identify concrete paths forward through principle-guided generation, evaluation, benchmarking, and agentic workflows. 

---
# What Intermediate Layers Know: Detecting Jailbreaks from Entropy Dynamics 

**Authors**: Sofiia Nikolenko, Michele Papucci, Mina Rezaei, Shireen Kudukkil Manchingal  

**Link**: [PDF](https://arxiv.org/pdf/2606.25182)  

**Abstract**: Jailbreak attacks reveal a persistent weakness in aligned Large Language Models: carefully crafted prompts can elicit policy-violating responses despite safety training. While most defenses operate at the prompt or output level, it remains unclear how harmful intent is encoded within the model's internal representations. We investigate this question by analyzing token-level predictive entropy trajectories across layers of a frozen LLM using the logit lens. We find that static aggregate statistics of prompt-level entropy (e.g., mean, variance) carry little discriminative signal, whereas features capturing how entropy evolves across token positions, such as monotonic rank-based trend scores, are substantially more informative. Importantly, this signal is not uniform across model depth: it is concentrated in intermediate layers and degrades at the final layer, indicating that jailbreak-relevant structure is most pronounced in mid-network representations rather than at the output head. Across multiple models (Llama, Qwen, Gemma) and adversarial benchmarks, these entropy dynamics provide architecture-consistent separation without additional training. Together, our findings show that jailbreak behavior is reflected in structured intermediate uncertainty dynamics, clarifying both which entropy-derived features encode harmful intent and where in the network that signal is most pronounced. 

---
# Phoneme-Level Mispronunciation Screening in Polish-Speaking Children with an Explainable Assistant 

**Authors**: Milosz Dudek, Daria Hemmerling, Kamil Kwarciak, Maciej Stroinski, Maria Pensko, Mateusz Kowalewski, Leonid Pavlovskyi, Sebastian Jurczak, Anna-Mariia Vitkovska, Zuzanna Miodonska, Natalia Mocko, Michal Krecichwost  

**Link**: [PDF](https://arxiv.org/pdf/2606.25181)  

**Abstract**: Early identification of speech sound errors in children is often limited by access to specialists, motivating lightweight screening tools that can operate outside the clinic. We present a screening pipeline for Polish-speaking children focused on sibilant substitutions, coupling a wav2vec2-based CTC token recognizer with alignment-based error typing and a template-grounded caregiver assistant for screening, not diagnosis. On a held-out test set of 10 unseen children comprising 559 utterances, the recognizer achieves 88.7 percent exact sequence match. As a conservative screening proxy, we flag a mismatch when the system emits substitution-evidence bracketed tokens at the target segment, yielding 72.9 percent precision, 61.4 percent recall, F1 = 0.67, and a 2.7 percent false-alarm rate on target-correct items. We describe the assistant's safety boundaries and outline a clinician-in-the-loop validation plan for future deployment. 

---
# ATMA: Length-Invariant Language Modeling via Polar Attention and Gated-Delta Compression Memory 

**Authors**: Habibullah Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2606.25156)  

**Abstract**: Modern large language models based on softmax scaled-dot-product attention are constrained by their training sequence length: as the key-value sequence grows, softmax probability mass can dilute across a wider distribution, inducing activation shift and long-context performance collapse. Moreover, long-context language modeling faces a structural tension: a sliding-window attention core maintains a bounded local representation and low perplexity but is blind to long-range dependencies, while full-context attention preserves global recall but suffers from out-of-distribution perplexity explosion. To resolve these limitations, we introduce ATMA, a hybrid convolutional-attention architecture that integrates a novel three-channel attention mechanism. ATMA factorizes the attention mixing step into: (1) a count-blind, unit-vector direction channel, (2) a bounded magnitude channel driven by the participation ratio of effective matches over an extreme-value-corrected null sink, and (3) a long-term recurrent compression memory optimized via a gated-delta fast-weights rule. Neither the Polar Attention core nor the recurrent memory is sufficient alone; their combination enables monotonic perplexity reduction and high-fidelity long-range retrieval simultaneously. We evaluate ATMA using a 100-run factorial ablation sweep, demonstrating that the combined Polar + memory model maintains induction needle-in-a-haystack retrieval accuracy above 90% out to 64K tokens (32 times the training length of 2K) while its document perplexity improves monotonically, outperforming softmax-based memory baselines which collapse at extreme context lengths. Code: this https URL 

---
# Hitting a Moving Target: Test-Time Adaptation for AI Text Detection under Continual Distribution Shift 

**Authors**: Kevin Ren, Manish Raghavan, Nikhil Garg  

**Link**: [PDF](https://arxiv.org/pdf/2606.25152)  

**Abstract**: Deployed approaches for AI text detection often rely on training-time access to labeled datasets of both human-written and AI-generated text. This approach is vulnerable to three types of distribution shifts that occur continually post-deployment, and for which labeled data is often unavailable: adversarial humanization, new LLMs being released, and temporal drift in human writing. Simultaneously, existing approaches do not leverage a key signal of LLM usage: inference-time homogeneity. We propose a test-time adaptation (TTA) approach, using semi-supervised learning, that adapts to distribution shifts by leveraging homogeneity among unlabeled samples observed at inference time. Empirically, we find that state-of-the-art supervised detectors systematically fail when they encounter distribution shifts in AI-generated and human writing, both adversarial and natural, while test-time adaptation with semi-supervised learning is largely robust; e.g., the commercial model Pangram detects just 24.1% of our adversarial AI-generated text, compared to 90.5% for our test-time approach. We establish that test-time adaptation is a promising framework for AI text detection in the wild. We publicly release our code (which includes code for model training, evaluation, and plots) at this https URL. 

---
# Silent Failures in Physics-Informed Neural Networks: Parameter Poisoning and the Limits of Loss-Based Validation 

**Authors**: David McShannon, Nicholas Dietrich  

**Link**: [PDF](https://arxiv.org/pdf/2606.25151)  

**Abstract**: Physics-informed neural networks (PINNs) embed governing equations in their loss function, enabling mesh-free solutions to partial differential equations. Low training loss is treated as evidence that the learned solution is physically correct. This paper shows that assumption breaks down when encoded physics are incorrect. By perturbing PDE parameters before training, a setting we describe as physics parameter poisoning or parameter misspecification, we produce models that train to low loss but give incorrect answers; we treat the perturbation schedule as sensitivity analysis rather than only as a security threat, and none of our claims requires an adversary. Achieving low residual loss does not discriminate accurate from inaccurate solutions: poisoned models reach losses at or below the clean baseline yet differ by large margins, so driving the residual down is not evidence of physical accuracy. Across three PDE systems (Burgers equation, Navier-Stokes cavity, and convection-diffusion), poisoned models match or beat the clean-model training loss while their solutions differ by up to 71% in the fixed sweep and up to 128% under adversarial search; at Cavity Re=400 the poisoned loss falls below the clean baseline. We define a detection difficulty ratio R (solution error divided by training loss) to summarize how invisible the corruption is, though cross-PDE comparison is complicated by differences in loss scale. We test six candidate defenses, none of which reliably detects corruption across all regimes. We propose a post-hoc defense: sweeping the PDE residual loss across parameter values without retraining. The loss minimum recovers the true training parameter without external data, and generalizes across all three PDE systems. The effect holds across five network architectures (8.7K to 133K parameters), is bidirectional, and is confirmed across multiple random seeds. 

---
# Proactive Systems in HCI and AI: Concepts, Challenges, and Opportunities 

**Authors**: Nima Zargham, Sharon Ferguson, Jaisie Sin, Cosmin Munteanu, Anastasia Kuzminykh  

**Link**: [PDF](https://arxiv.org/pdf/2606.25149)  

**Abstract**: The last few years have seen a significant rise in interest in highly autonomous and proactive systems, fueled by advances in AI. Systems that anticipate user needs, take initiative, and act without explicit user input. Such systems span a wide range of applications, from smart lighting that adapts to user activity to assistive robots that plan actions in advance to intelligent thermostats that learn routines and adjust environments proactively. Despite this breadth, the concept of proactivity remains loosely defined and inconsistently applied across research and practice. Current usage of the term often conflates fundamentally different system behaviors. For instance, simple reminders or recommendation systems are frequently labeled as proactive, even though underlying mechanisms and intentions differ significantly. This conceptual ambiguity limits our ability to systematically design, compare, and evaluate proactive systems. Moreover, existing methodologies for design and evaluation are largely rooted in reactive interaction paradigms, failing to address the unique challenges posed by proactive behavior, including timing, appropriateness, user control, transparency, and trust. This multidisciplinary workshop aims to establish a clearer and more rigorous foundation for understanding proactive systems. We bring together researchers and practitioners from Human-Computer Interaction, AI, and related fields to (1) develop a shared conceptualization of proactivity, (2) identify gaps and limitations in current design and evaluation approaches, and (3) co-create human-centered guidelines and research directions for future systems. Through interactive discussions and collaborative activities, the workshop seeks to map key challenges and opportunities, ultimately advancing robust and consistent frameworks for designing and evaluating proactive technologies. 

---
# TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems 

**Authors**: Qingyun Liu, Bo Yan, Yang Liu, Yuji Roh, Ekansh Sharma, Likang Yin, Emma Olowo, Min-hsuan Tsai, Yuxuan Li, Diego Uribe, Saksham Aggarwal, Siqi Wu, Yuan Hao, Vikas Kedigehalli, Lukasz Heldt, Lichan Hong, Li Wei, Xinyang Yi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25147)  

**Abstract**: User modeling in industrial recommender systems typically produces dense embeddings, which suffer from representational constraints inherent to fixed-dimensional vectors. An emerging alternative for discrete user representation -- using LLMs to generate text-based user tokens -- captures topical co-occurrences rather than deep sequential behavior dynamics and produces outputs that are difficult to ground to item attributes. Meanwhile, Semantic ID (SID) based item tokenization has proven effective for improving generalization in generative recommendation, yet discrete SID-based representations for users remain largely unexplored. We propose TokenMinds, an industrial-scale system that extends the PLUM framework from item retrieval to user modeling, generating both discrete SID-based user tokens and dense user embeddings via an encoder-decoder architecture adapted from pre-trained LLMs. This dual-output design provides the complementary benefits of discrete, semantically grounded user representations while maintaining compatibility with existing downstream models that rely on dense embeddings. Additionally, the shared SID vocabulary naturally extends to cross-scenario modeling: by unifying long-form and short-form video behaviors into a single model, we substantially reduce training and serving costs. We validate TokenMinds through extensive offline experiments and live launches on multiple YouTube surfaces, served on full user traffic (billions of users) via an asynchronous infrastructure that decouples representation generation from downstream scoring. Focusing on ranking as the primary downstream use case, our results confirm the practical viability of SID-based user tokens at industrial scale and demonstrate that tokens and dense embeddings provide complementary value across different production ranking systems. 

---
# Benchmarking the Alignment of Data-Quality Metrics, Human Judgment and Land-Cover Segmentation Performance for Earth Observation 

**Authors**: Ümit Mert Çağlar, Alptekin Temizel  

**Link**: [PDF](https://arxiv.org/pdf/2606.25128)  

**Abstract**: Volume and quality of datasets are crucial for deep learning model training, yet they are often constrained by availability and data acquisition costs. Synthetic data augmentation can extend existing datasets with realistic images, and the quality of these images is generally assessed through fidelity metrics such as FID, KID, IS, LPIPS and SSIM that measure structural or distributional similarity. However, such metrics, including the widely used FID, focus on visual fidelity without reflecting downstream utility, and can diverge from human perception under perturbations that are imperceptible to human observers. In this work, we systematically evaluate Earth observation datasets alongside synthetic counterparts generated by deep generative models, comparing automatic metrics against human perception and downstream tasks. Our results reveal a stark misalignment: semantics-preserving perturbations such as rotation drastically alter metric scores while leaving human recognition unaffected, and synthetic samples that score poorly on automatic metrics achieve comparable or higher perceived realism, and can improve downstream performance when combined with real data. By benchmarking semantic segmentation models trained on mixed real-synthetic datasets, we demonstrate that quality metrics rooted in ImageNet-pretrained feature spaces are unreliable indicators for geospatial data. Our findings underscore that automatic quality evaluation of synthetic datasets should be grounded in downstream task performance and human evaluation. 

---
# Reward-Conditioned Attention: How Reward Design Shapes What Autonomous Driving Agents See 

**Authors**: Mohamed Benabdelouahad, Ahmed Djalal Hacini, Nadir Farhi, Aissa Boulmerka  

**Link**: [PDF](https://arxiv.org/pdf/2606.25127)  

**Abstract**: We investigate how reward design shapes the internal attention patterns of reinforcement learning agents trained for autonomous driving. Using three Perceiver-based agents that share identical architectures and training data but differ only in their reward configurations$\unicode{x2014}$ranging from basic violation penalties to continuous proximity penalties$\unicode{x2014}$we analyze cross-attention allocation across 50 real-world scenarios from the Waymo Open Motion Dataset. A central methodological finding is that naïve pooling of timesteps across episodes substantially underestimates the attention$\unicode{x2013}$risk relationship; within-episode correlation with Fisher z-transform aggregation is the appropriate statistic and reveals a robustly positive link between collision risk and agent-directed attention. Building on this validated methodology, we demonstrate two reward-conditioned effects: agents trained with navigation rewards allocate up to $2.0\times$ more attention to GPS-path tokens than those trained with additional proximity penalties$\unicode{x2014}$and $4.7\times$ more than agents with no navigation incentive$\unicode{x2014}$revealing that reward content directly determines which scene elements the encoder prioritizes, and continuous time-to-collision penalties create a $\textit{learned vigilance prior}$$\unicode{x2014}$elevated resting agent surveillance maintained throughout collision-free phases. In several scenarios, the complete-reward and minimal-reward models exhibit opposite attention$\unicode{x2013}$risk correlation directions, demonstrating that reward design can qualitatively reverse attentional strategy rather than merely modulating its magnitude. These results suggest that attention analysis is a practical diagnostic for verifying that a reward function produces the intended representational behaviour in safety-critical RL systems. 

---
# AeroCast: Probabilistic 3D Trajectory Prediction for Non-Cooperative Aerial Obstacles via Transformer-MDN Architecture 

**Authors**: Syed Izzat Ullah, Jose Baca  

**Link**: [PDF](https://arxiv.org/pdf/2606.25122)  

**Abstract**: Autonomous aerial vehicles operating in shared airspace must predict the future positions of non-cooperative obstacles to plan evasive maneuvers before a collision becomes unavoidable. Unlike cooperative systems that share intent, non-cooperative obstacles such as birds, uncontrolled drones, or debris exhibit multi-modal motion that deterministic predictors cannot adequately represent. Existing methods either rely on recurrent encoders that propagate temporal information sequentially, limiting their ability to capture long-range kinematic precursors of maneuver initiation, or produce point forecasts that provide no distributional information to downstream planners. This paper presents AeroCast, a probabilistic trajectory prediction framework that combines a Transformer encoder with a Mixture Density Network output head to predict per-timestep Gaussian mixture distributions over future three-dimensional displacements. A translation-invariant consecutive displacement encoding and a calibration-oriented training objective address the input design and mode-degeneracy challenges specific to mixture-based aerial trajectory prediction. On a hybrid real-and-synthetic quadrotor corpus spanning nine motion categories, AeroCast reduces Average Displacement Error and Final Displacement Error by approximately 50% relative to the baselines over a five-second horizon, and achieves the lowest negative log-likelihood and Continuous Ranked Probability Score among all compared methods. Ablation analysis identifies velocity input and model capacity as the primary contributors to prediction quality, and positional encoding as essential for long-horizon trajectory coherence. AeroCast inference completes in 0.1ms per sample, compatible with real-time onboard deployment at 100Hz. 

---
# BCoughBench: Benchmarking Respiratory Acoustic Foundation Models Under Body-Coupled Wearable Sensor Conditions 

**Authors**: Mayur Sanap, Prasanna Desikan, Edgar Lobaton  

**Link**: [PDF](https://arxiv.org/pdf/2606.25116)  

**Abstract**: Respiratory acoustic foundation models (FMs) are benchmarked exclusively on smartphone recordings, yet clinical deployment increasingly targets body-coupled (BC) wearables whose sensors attenuate high-frequency content through tissue and bone, leaving FM reliability uncharacterised. We introduce BCoughBench, evaluating five FMs (OPERA-CT/CE/GT, HeAR, M2D+Resp) on nine classification tasks (AUROC, sensitivity at 95% specificity, Expected Calibration Error) and three age regression tasks (MAE vs. a mean-predictor baseline) across five EBEN-simulated BC sensor conditions on five labeled cough datasets. Mean AUROC declines from 0.785 (smartphone) to 0.689-0.723, degrading most under temple vibration pickup ($\Delta$ = -0.096) and least under the soft in-ear ($\Delta$ = -0.062). No FM meets the clinical sensitivity threshold (Se@Sp95 $\geq$ 0.20) on most disease tasks under any BC sensor. Sex classification on the CIDRZ cohort collapses (AUROC 0.954 to 0.596-0.628, $\Delta$ = -0.341) while COVID detection is nearly unaffected ($\Delta$ = -0.004). Age regression is robust, improving under the forehead accelerometer on CoughVID (MAE 9.61 to 8.97 yr); HeAR leads on regression and demographic tasks, M2D+Resp on disease and characteristic tasks. BCoughBench provides a reproducible framework for FM evaluation under wearable conditions. 

---
# Power-Flexible AI Data Centers: A New Paradigm for Grid-Responsive Compute 

**Authors**: Chris Williams, Philip Colangelo, Ayse Coskun, Ethan Levine, Andy Neale, Ciaran Roberts, Shayan Sengupta, Nikhil Shirolkar, Varun Sivaram, Sarah Soares, Ethan Tiao, Scott Underwood, Daniel Wilson, Frank Sharp, Luke Wainwright, Harry Petty, Scott Wallace, Brandon Records  

**Link**: [PDF](https://arxiv.org/pdf/2606.25098)  

**Abstract**: The rapid expansion of artificial intelligence (AI) infrastructure is driving unprecedented growth in electricity demand from data centers. Traditional power-system planning treats large computing facilities as inflexible peak loads, leading to costly infrastructure upgrades and long delays in grid interconnection. Recent work has shown that AI clusters can reduce electricity consumption during peak demand through software-based workload orchestration. This article explores how modern GPU-based AI data centers can operate as grid-interactive assets that respond dynamically to power system conditions. We describe an architecture integrating grid signals, workload scheduling, and power telemetry for fine-grained cluster power control. Experimental results from a real-world deployment on a 130 kW GPU cluster demonstrate multiple forms of flexibility, including rapid load reduction, sustained curtailment, and carbon-aware operation while preserving service levels for priority jobs. We further demonstrate performance-aware load shifting across geographically distributed clusters, enabling workloads to migrate toward regions with lower grid stress. Together, these capabilities transform AI infrastructure from static electricity consumers into flexible resources that support grid reliability, accelerate interconnection, and improve computing sustainability. 

---
# Training for the Model You Return: Improving Optimization for Iterate-Averaged Language Models 

**Authors**: Kwok Chun Au, Adam Block  

**Link**: [PDF](https://arxiv.org/pdf/2606.25086)  

**Abstract**: Many modern Language Model (LM) pipelines return an averaged model, such as an exponential moving average of the training iterates, rather than the final iterate itself. This raises a fundamental question: given that we will return an iterate average, how should we change training to improve the performance of this average? We study this question by formulating optimizer design for the iterate-average estimator as an optimal-control problem. In a continuous-time stochastic quadratic model, we solve for the control strategy that minimizes the error of the returned average subject to a penalty on the size of the intervention. A practical approximation to this controller yields PACE, a lightweight wrapper around AdamW that pulls the live weights toward their exponential moving average with a clipped, per-coordinate control strength. We prove that a stylized version of PACE converges at the standard stochastic convex optimization rate, up to a factor depending on the averaging rule, while in the quadratic setting it can strictly improve the limiting squared error of the iterate-average estimator and can do so by an arbitrarily large factor on some instances. Empirically, our results suggest that PACE improves over AdamW and EMA-evaluated AdamW in supervised fine-tuning of 1-2B parameter LMs and in GPT-2 pretraining on FineWeb for a wide range of learning rates, decay schedules, and other hyperparameters. 

---
# GCT-MARL: Graph-Based Contrastive Transfer for Sample-Efficient Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Animesh Animesh, Satheesh K Perepu, Kaushik Dey  

**Link**: [PDF](https://arxiv.org/pdf/2606.25073)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), from a deployment perspective, it is challenging and expensive to train agents from scratch for each new environment or task. In this work, we propose GCT-MARL, a transfer learning framework that builds on the multi-view graph contrastive backbone of MAIL and augments it with a per-view, adaptively weighted alignment loss and a two-phase training protocol specifically designed for transfer across populations of varying sizes and compositions. We empirically demonstrate that the proposed framework markedly accelerates convergence on the target task relative to from-scratch training, in both homogeneous (within-faction, varying N) and heterogeneous (cross-faction and mixed unit-type) transfer scenarios. Furthermore, we show that the framework naturally supports continual learning by sequentially chaining the two-phase transfer protocol across a series of related tasks. Overall, this work provides a unified approach to mitigating key limitations in current MARL transfer methods with new insights at both methodological and empirical levels. 

---
# Adapt Only When It Pays: Budgeted Decision-Loss Priority for Delayed Online Time-Series Adaptation 

**Authors**: Xibai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.25068)  

**Abstract**: Online time-series forecasters receive labels only after horizon-dependent delays, while every adaptation step spends limited compute. We study when an online learner should update, not how to adapt at every opportunity, and introduce ADOWIP: a residual-adapter framework with sealed delay queues, exact budget accounting, and auditable update telemetry. Its main scheduler is an observed decision-loss priority gate that updates only after feedback is revealed, when downstream loss, optionally penalized by prediction MSE, exceeds a calibrated empirical quantile and budget remains. We prove hard-budget feasibility, projected-OGD regret for a convex linear accepted-update subproblem, and stability plus conditional finite-sample gate-selection statements. On public ETT capacity-planning tasks, a frozen calibration/evaluation split selects a gate that lowers held-out decision loss against always, fixed-period, and drift-triggered exact-update baselines under matched compute. Secondary threshold/load-index ETT suites are mixed: 33 of 41 selected contrasts clear the stricter cross-artifact Holm family, and the 8 nonpassing rows are explicitly excluded from primary claims. The same protocol improves an external UCI Bike capacity proxy with 20/0 held-out wins, and a fixed gate passes three full-year Capital Bikeshare station-rebalancing contrasts. Probe-based and finance experiments remain negative, delimiting the current scope of decision-prioritized adaptation. 

---
# What Does It Mean to Break a Distillation Defense? 

**Authors**: Lena Libon, Pura Peetathawatchai, Michael Aerni, Daniel Paleka, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2606.25059)  

**Abstract**: Black-box LLMs (accessible only via API) are vulnerable to distillation attacks, in which an attacker queries the model and trains a student on its outputs. A recent line of work proposes output perturbation defenses that modify the teacher's output to reduce student performance while preserving utility for legitimate users. As a relatively new family of approaches, output perturbation defenses lack a shared threat model, making it difficult to compare them, reason about composing them with other attacks, or evaluate their robustness against realistic adversaries. This underspecification matters beyond technical evaluation: when defenses are deployed to protect intellectual property or justify regulatory compliance, an imprecise threat model can create a false sense of security. We propose a threat model framework that describes attackers along three dimensions: a query budget, a data budget, and an interface profile that captures how attackers interact with the API. Using antidistillation sampling as a case study, we show that whether the defense is considered effective depends on the assumed threat model. We argue that future work on distillation defenses, along with any governance or policy frameworks built around them, should explicitly specify and stress-test attacker capabilities along our three dimensions. 

---
# Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models 

**Authors**: Lianghua Huang, Zhifan Wu, Wei Wang, Yupeng Shi, Mengyang Feng, Junjie He, Chenwei Xie, Yu Liu, Jingren Zhou, Ang Wang, Bang Zhang, Baole Ai, Chen Liang, Cheng Yu, Chongyang Zhong, Jinwei Qi, Kai Zhu, Pandeng Li, Peng Zhang, Wenyuan Zhang, Xinhua Cheng, Yitong Huang, Yun Zheng, Zoubin Bi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25041)  

**Abstract**: We present Wan-Streamer, a native-streaming, end-to-end interactive foundation model designed from the ground up for real-time, low-latency, full-duplex audio-visual interaction. Wan-Streamer seamlessly models language, audio, and video as both input and output within a single Transformer, where the sequence is represented as interleaved visual, audio, and text input tokens together with visual, audio, and text output tokens, coordinated by block-causal attention for incremental streaming. Unlike cascaded interactive systems that rely on separate VAD, ASR, language, TTS, audio-driven animation, or video-generation modules, Wan-Streamer does not rely on external language, speech, avatar, or video-generation modules: perception, reasoning, generation, response timing, turn management, and cross-modal synchronization are learned jointly within one unified model, reducing pipeline latency and error accumulation. To support natural audio-visual responsiveness, we redesign the entire stack around streamability, including causal encoders, causal decoders, block-causal attention, and low-latency multimodal token scheduling, enabling streaming units as short as 160 ms at 25 fps. Wan-Streamer achieves approximately 200 ms model-side response latency and approximately 550 ms total interaction latency when combined with 350 ms bidirectional network latency, supporting sub-second duplex audio-visual communication. These results position Wan-Streamer as a unified, end-to-end, multimodal interactive foundation model for low-latency streaming interaction. 

---
# LLM-ACES: Closed-Loop Discovery of Dynamical Systems with LLM-Guided Adaptive Search 

**Authors**: Nikhil Abhyankar, Sha Li, Sanchit Kabra, Naren Ramakrishnan, Yulia Gel, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.25039)  

**Abstract**: Recovering governing Ordinary Differential Equations (ODEs) from data is a central challenge in modeling dynamical systems across scientific domains. Existing approaches cast discovery as a static inference problem over fixed datasets, assuming that the observed trajectories are sufficiently informative. However, dynamical systems evolve over large state spaces, and limited data can make multiple equations observationally indistinguishable, leading to identifiability gaps and the recovery of incorrect governing equations. To address this, we introduce LLM-ACES, or LLM-guided Active Closed-loop Equation Search, a closed-loop framework that jointly optimizes symbolic hypothesis construction and adaptive data acquisition. In LLM-ACES, a large language model (LLM) proposes operator priors that partition the large search space into distinct regions, within which candidate equations are fit to the observed data. The disagreement among these candidates guides the acquisition of informative trajectories, creating a feedback loop that iteratively refines both the hypothesis space and the discovered dynamics. On 122 ODE systems spanning ODEBench and ODEBase, LLM-ACES achieves the lowest median NMSE, outperforming state-of-the-art baselines by several orders of magnitude while achieving a high symbolic accuracy of 46.2% and 52.4%, respectively. Our analysis further shows that LLM-ACES is sample-efficient, achieving better performance with one-tenth the data. Furthermore, LLM-ACES's feedback-driven data acquisition makes it robust to noise and recovers the correct symbolic structure, while baselines introduce spurious terms that fit the data locally but obscure the true governing relationships. 

---
# Yuvion VL: A Multimodal Foundation Model for Adversarial Content and AI Safety 

**Authors**: Shikai Qiu, Xiaowen Xu, Benlei Cui, Ting Ma, Xiufeng Huang, Wenjing Jiang, Shaoxuan He, Haolei Xu, Chunyang Chai, Yujian Li, Yiliang Zhang, Guanghui Wang, Ziheng Wang, Ziwen Xu, Zhaoyu Fan, Jinhao Chen, Ruijie Jian, Hongxing Li, Chuxi Xiao, Xinyue Chen, Wenxuan Liu, Libin Dong, Yupeng Cao, Xiaoqian Xia, Jing Wang, Zhe Jiang, Zhenan Ye, Guang Yang, Bin Liu, Wei Peng, Ziqiang Zhu, Meihui Lian, Kaiwen Lv Kacuila, Haidong Ding, Dongjie Zhang, Yangfan Zhou, Bingyu Zhu, Yan Wang, Hai Zhao, Xuan Jin, Wei Zhao, Pengfei Sun, Huiming Zhang, Wei Wang, Xipeng Cao, Bin Li, Chengwen Yao, Meng Huang, Xianfeng Li, Bin Tang, Chao Liu, Hui Xue, Longtao Huang, Haiwen Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.25034)  

**Abstract**: General-purpose models often struggle to reliably identify and understand real-world multimodal risks, largely due to the inherent multimodal adversarial nature of content and AI safety. We present Yuvion VL, a family of multimodal large language models purpose-built for content and AI safety, with both instruction-tuned and reasoning-oriented variants. Yuvion VL addresses this gap by treating safety as an inherently adversarial and multimodal problem and designing the entire pipeline around adversarial robustness. For data construction, we develop an automated pipeline integrating adversarial-aware data synthesis with multi-stage quality control, producing large-scale, high-quality multimodal samples augmented with domain knowledge and reasoning annotations. For training, we adopt a three-stage pipeline that includes continued pretraining for risk-concept cross-modal alignment, instruct post-training for production-grade safety tasks, and reasoning post-training for enhanced interpretability and performance in complex tasks. We further introduce Confuse-then-Contrast Fine-Tuning, a contrastive framework that mines model-specific confusions and constructs multi-image contrastive groups to enforce explicit discrimination of fine-grained visual-semantic elements, enabling the model to distinguish between visually similar cases with different safety implications in adversarial safety tasks. To support rigorous evaluation, we further introduce Yuvion VL RiskEval (YVRE), a collection of benchmarks covering diverse open and internal evaluations, with a focus on content and AI safety, adversarial robustness, and real-world capability requirements. Experiments show that Yuvion VL-32B achieves industry-leading safety performance, surpassing comparably sized open-source models and best closed-source commercial models, while maintaining comparable general capabilities. 

---
# Do Thinking Tokens Help with Safety? 

**Authors**: Narutatsu Ri, Abhishek Panigrahi, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2606.25013)  

**Abstract**: Today's reasoning models use thinking tokens to attain stronger performance on benchmarks than their instruction-tuned counterparts. It is also generally believed that this more "deliberative" mode should improve alignment and safety, by providing the model a safe space to consider whether its planned answer to a request violates its safety principles. We present evidence that this intuition is not always correct. Across frontier open-weight reasoning models spanning GPT-OSS, Qwen, Olmo, and Phi families, we find that the eventual refusal/compliance outcome is already strongly predictable via a trained head on the first token's hidden representation ($0.84$-$0.95$ AUROC and $\sim88\%$ balanced accuracy for predicting refusal/compliance) before any visible thinking. The thinking process turns out to be more akin to prefix completion than to deliberative revision, with the final outcome rarely changing after the first $\sim20\%$ of thinking, despite giving the appearance of deliberation at the text level ($\sim74\%$ of text-level deliberations occur when the response distribution is already locked to one refusal/compliance side). We also find that existing inference-time and training-based safety interventions, despite being motivated by the goal of inducing deliberation, largely shift model behavior toward over-refusal while suppressing already-scarce deliberation signals. Our results suggest that safety behavior in current reasoning models is much less deliberative than commonly assumed, and highlight the need for methods that induce real safety deliberation. 

---
# Noise-Aware Boundary-Enhanced Generative Learning for Ultrasound Speckle Reduction 

**Authors**: Yuexi Gu, Mengqi Wu, Yongheng Sun, Virginie Papadopoulou, Mingxia Liu, Maureen Kohi  

**Link**: [PDF](https://arxiv.org/pdf/2606.25009)  

**Abstract**: Ultrasound is a non-invasive, real-time, and cost-effective imaging technique widely used in clinical diagnosis. However, its diagnostic efficacy is often compromised by inherent speckle noise that degrades image quality and obscures underlying anatomical structures. Existing speckle reduction methods tend to over-smooth tissue boundaries and generalize poorly to heterogeneous noise levels. To address these limitations, we propose a Noise-Aware Boundary-Enhanced Generative Learning (NBGL) framework for ultrasound speckle reduction, which simultaneously preserves annotated anatomical boundaries and adapts to varying noise levels. The NBGL framework consists of a speckle reduction branch and a boundary enhancement branch. The former leverages generative learning to suppress speckle noise, while the latter learns boundary-sensitive representations to preserve target anatomical structures. Furthermore, a noise-aware interaction weight generation (NIWG) module estimates the speckle noise level via 3D Laplacian filtering and a median absolute deviation estimator, and translates it into an adaptive interaction weight. This weight is incorporated into a weighted feature-wise linear modulation (wFiLM) module to adaptively modulate cross-branch feature coupling, thereby improving robustness to varying noise levels. Extensive evaluations on 141 3D transvaginal ultrasound volumes demonstrate that NBGL consistently outperforms state-of-the-art methods in speckle reduction and structural preservation across six noise levels, while maintaining consistency with annotated anatomical boundaries. 

---
# Erased, but Not Gone: Output Forgetting Is Not True Forgetting 

**Authors**: Teresa Pui Yee Yong, Win Kent Ong, Chee Seng Chan  

**Link**: [PDF](https://arxiv.org/pdf/2606.25001)  

**Abstract**: Machine unlearning (MU) is commonly judged by output forgetting, such as low forget-set accuracy or reduced logit-level membership inference. But if output-level success can coexist with retraining-inconsistent residuals in representation space, what kind of forgetting are current evaluations actually certifying? We study this question through retraining-consistent representation forgetting, using the retrained model (i.e., trained from scratch without the forget data) as an operational reference for correct forgetting. Across multiple unlearning methods, datasets, and models, our theoretical analysis and empirical results show that standard output-level evaluation can systematically overestimate the success of unlearning. Under this stronger lens, current methods often appear forgotten at the output layer while exhibiting a structured mismatch relative to retraining. They partially align with retraining on forget samples, remain more inconsistent on retain samples, and leave residual discrepancy concentrated along retraining-related directions rather than diffuse in representation space. This structured mismatch is characterized by forget/retain asymmetry, directional mismatch, and concentrated residuals along retraining-related directions. These results suggest that current MU is often evaluated for apparent forgetting rather than retraining-consistent forgetting. More broadly, retraining reveals what output forgetting hides. 

---
# Geo-Strat-RL: Learning Geological Event Reasoning from Verifiable Tasks 

**Authors**: Lukas Mosser  

**Link**: [PDF](https://arxiv.org/pdf/2606.25000)  

**Abstract**: To evaluate whether vision-language models can reason about geological histories, it is necessary to construct observations for which the underlying process history is known. Furthermore, reasoning over geological histories is not just a question of recognizing visual patterns, but also of understanding temporal and structural relationships that may be only indirectly visible or highly ambiguous. When ground-truth event histories are not uniquely identifiable or are unavailable, it remains an open challenge to teach models capable of visual reasoning to produce valid geological reconstructions that are consistent with both observed evidence and geological principles. We therefore investigate whether defining a verifiable geological reasoning task can improve geological event reconstruction across observation domains through reinforcement learning with verifiable rewards (RLVR). To this end, we present Geo-Strat-RL, a synthetic environment that generates stratigraphic observations and compact visible-evidence event histories. The environment combines a geological generator with an executable verifier that scores chronology, event identity, deposition, and structural relationships. We show that RLVR improves geological reconstruction in vision-language models (VLMs), increasing geological content scores on held out stratigraphic diagrams. We further evaluate the same held-out geological histories in a synthetic seismic observation domain by converting the generated scenes into acoustic-impedance-derived amplitude sections. In this controlled paired-renderer setting, we present evidence that geological reasoning learned from stratigraphic diagram-domain RLVR training transfers to synthetic seismic representations without seismic-specific training examples, supporting the hypothesis that RLVR can teach reusable geological reasoning concepts across related observation formats. 

---
# Internal Data Repetition Destroys Language Models 

**Authors**: Jessica Chudnovsky, Joshua Kazdan, Noam Levi, Rylan Schaeffer, Yegor Denisov-Blanch, Bo He, Mehmet Donmez, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2606.24998)  

**Abstract**: Language models are running out of high-quality training data, and even aggressively deduplicated corpora retain some amount of repetition. Earlier controlled studies predated Chinchilla-style scaling laws and could only measure the cost of repetition indirectly. We revisit repetition in the Chinchilla era, using a fitted no-repetition scaling law to report Compute-Equivalent Gain and Compute-Equivalent Loss. We show that under this modernized paradigm, repetition damage is systematic in three ways. First, holding compute allocated to repeated data constant, eval loss peaks at an intermediate repeat count $\Rep$; repeating a moderately sized subset a moderate number of times damages performance more than repeating a large subset a few times or a small subset many times. Second, the location of this peak is well-fit by a power law in model size; this scaling law reveals that the most damaging number of repeated data grows more quickly than compute. Finally, when repeated documents consume 10\% of the FLOPs budget in a controlled exact-document repetition setting, the compute-equivalent loss can be large: on FineWeb-Edu-Dedup, the most damaging repeat count for a Qwen3-style 344M-parameter model at $\OT=1$ matches the loss of a no-repetition run using 67% of the FLOPs. We demonstrate that these phenomena are not language-model-specific, and can be analytically understood in a simple statistical model: a misspecified linear regression with verbatim duplicates reproduces the same qualitative loss peak, quantifying how such peaks can arise from a statistical tradeoff between memorization and generalization. Our findings add precision to the study of duplication in language models, allowing practitioners to quantify the wasted compute incurred by the presence and repeat structure of duplicates in pretraining corpora. 

---
# Are Tabular Foundation Models Robust to Realistic Query Distribution Shifts in Microbiome Data? 

**Authors**: Giulia Perciballi, Ahmad Fall, Federica Granese, Edi Prifti, Jean-Daniel Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2606.24995)  

**Abstract**: Tabular foundation models (TFMs) achieve strong performance on microbiome abundance data, yet their robustness under realistic distribution shift remains poorly characterized. We introduce a benchmark that evaluates the robustness of TFMs to biologically inspired perturbations across six gut microbiome datasets spanning four disease contexts. In this in-context learning setting, models receive unperturbed support sets as context and are evaluated on perturbed query samples. To isolate robustness beyond "shortcut" features, we preserve the most discriminative taxa and apply three controlled perturbation strategies: (i) removal of high-abundance (uninformative) taxa, (ii) sparsification via increased zero-inflation, and (iii) zero-imputation via spurious non-zero injections. Our results show that protecting discriminative features is insufficient to guarantee stability under support-query shift: across datasets, all perturbations degrade model performance, with zero-imputation consistently the most harmful, indicating that corrupting global feature structure can break generalization even when key taxa are retained. Sparsification disproportionately affects TFMs relative to a classical random forest baseline, suggesting greater sensitivity to zero-inflation-type shifts. The code is publicly available at: this https URL. 

---
# ExTra: Exploratory Trajectory Optimization for Language Model Reinforcement Learning 

**Authors**: Wenyang Hu, Junxiang Jia, Zhen Shu, Daniel Dahlmeier, See-Kiong Ng, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2606.24994)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) for language-model reasoning can fail at both extremes of task difficulty: easy prompts often produce all-correct, low-diversity rollout groups with little gradient signal, while hard prompts can produce all-incorrect groups with no positive reward. We introduce ExTra (Exploratory Trajectory Optimization), a GRPO-compatible framework that extracts exploration signals from the model's own rollouts. ExTra combines two mechanisms: (i) a novelty reward that adds embedding-based diversity bonuses after GRPO normalization, rewarding diverse correct solutions; and (ii) entropy-guided prefix regeneration, which scores partial trajectories using entropy signals and continues exploration from promising intermediate steps. Across six mathematical reasoning benchmarks, ExTra improves Qwen3-1.7B over GRPO by about +5 points on pass@1 and +7 points on pass@16, showing that trajectory-level exploration signals can improve both single-sample accuracy and inference-time coverage. 

---
# Uncertainty-aware reinforcement learning for chemical language models 

**Authors**: Borja Medina, Jon Paul Janet  

**Link**: [PDF](https://arxiv.org/pdf/2606.24990)  

**Abstract**: Reinforcement Learning (RL) has become a powerful paradigm for de novo molecular design, enabling Chemical Language Models (CLMs) to navigate and explore the chemical space while optimizing specific desired properties. However, the existing RL frameworks treat all scoring functions as deterministic oracles, neglecting the inherent uncertainty attached to the predictions of the different molecular properties. This can lead to the exploration of highly-uncertain regions of the chemical space, focusing on the generation of highly scored molecules which are poorly supported by the training data. This can destabilize the optimization process, yielding predictions that are far from their true values.
We propose and compare two complementary ways of incorporating predictive uncertainty into RL. In the first one, uncertainty is treated as an additional optimization objective and incorporated along with the rest of the scoring functions, allowing the policy to trade off exploitation against reliability. Secondly, uncertainty is used to modulate policy updates, reducing the influence of molecules whose properties lie far outside the scoring function confidence domain.
Both approaches were evaluated across three different settings: (i) a controlled model system, in which the prediction error is modeled as a Gaussian distribution, with a variance proportional to the distance to the training data; and two real-world tasks, making use of (ii) ChemProp models and (iii) a Conformal Prediction wrapper applied to a Random forest classifier.
We show that uncertainty-aware RL enables CLMs to explore chemical space more robustly by favoring lower-uncertainty regions. This leads to more reliable hit discovery without compromising molecular score, increasing the true hit rate by 0.25 (from 0.5 to 0.75), and nearly doubling the total number of true hits. 

---
# When Multi-Sensor Fusion Fails to Generalize: Cattle Posture Classification Under Animal-Level and Temporal Distribution Shift 

**Authors**: Leutrim Uka, Severino Pinto, Gundula Hoffmann, Marina M.-C. Höhne  

**Link**: [PDF](https://arxiv.org/pdf/2606.24986)  

**Abstract**: Automated cattle posture-classification systems frequently report near-perfect accuracy, yet their robustness under realistic deployment conditions remains largely unknown. In particular, it is unclear whether multimodal sensor fusion improves generalisation or leads models to rely on context-specific signals that fail under distribution shift. Here, we evaluate the robustness of automated posture classification (lying versus standing) using collar accelerometers, rumen-bolus sensors, and environmental measurements collected from a pasture-based beef cattle herd across two consecutive years (2024-2025). XGBoost served as the primary model, with Logistic Regression, Random Forest, and Long Short-Term Memory networks evaluated as comparative baselines. Model robustness was assessed under progressively more stringent evaluation protocols, ranging from conventional random train-test splits to leave-one-animal-out validation and cross-year evaluation on an independent cohort of previously unseen animals recorded one year later. While multimodal models achieved strong within-year performance (macro-F1 0.94), the performance declined substantially under cross-year evaluation (macro-F1 0.49). Explainability analysis revealed persistent reliance on rumen-bolus activity and environmental variables even when predictive performance deteriorated. Distribution-shift diagnostics further confirmed substantial differences in feature distributions between recording years. Our findings demonstrate that commonly used evaluation protocols can substantially overestimate real-world performance and that multimodal sensor fusion may reduce, rather than improve, robustness under temporal distribution shift. More broadly, the results highlight that benchmark accuracy alone is insufficient to assess deployment readiness and underscore the need for robustness-centred evaluation in livestock-monitoring research. 

---
# Retrieval-Augmented Personalization with Foundation Models for Wearable Stress Detection 

**Authors**: Louis Simon, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2606.24985)  

**Abstract**: Personalization in wearable-based stress detection remains challenging due to substantial inter-individual variability in physiological and behavioral responses. While traditional approaches rely on user-specific fine-tuning or costly self-supervised pre-training on large datasets, we propose a lightweight alternative based on retrieval-augmented personalization. Our method leverages frozen, out-of-domain foundation models to retrieve similar patterns from a target user's history and encode them into a compact personalized embedding that modulates representations extracted by a lightweight transformer network. We evaluate our approach on the WESAD stress detection dataset with N=15 users, comprising wrist-worn physiological (EDA, BVP, temperature) and activity (accelerometer) signals, and report gains of +3.92\% in accuracy and +4.76\% in macro F1-score over a non-personalized transformer baseline, approaching supervised fine-tuning performance without requiring any labeled user data. We further show that temporal retrieval, where only prior user samples are available, achieves performance close to full intra-user retrieval, demonstrating robustness to limited user history. Finally, we explore personalization in a cross-dataset retrieval setting, leveraging embeddings from the K-Emocon dataset to personalize representations for stress detection on the WESAD dataset. 

---
# Why Do Accumulated Transformations Extrapolate? 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2606.24975)  

**Abstract**: PaTH Attention showed that replacing RoPE's position-indexed rotations with accumulated data-dependent Householder reflections yields strong length extrapolation, though performance degrades at extreme context lengths. We ask whether this depends on Householder-specific structure or reflects a general property of accumulated transformations along source-to-query paths. We study a simpler variant keeping RoPE's block-diagonal SO(2) rotations but replacing position-indexed angles with accumulated token-dependent ones. It shows the same pattern: improved extrapolation then degradation at long contexts. We prove the result extends to accumulated orthogonal transformations satisfying certain regularity conditions: their products become incoherent after finitely many steps, suppressing attention to distant tokens. Accumulated rotations of queries and keys create a finite mixing window independent of context length; per-token suppression learned in training transfers unchanged to any evaluation length, and high-dimensional concentration produces a score gap suppressing far tokens while near-route transport preserves the target signal. Conversely, a lower bound shows accumulated rotations must eventually degrade: as the far set grows, no rotations preserve the near signal without explicit far-mass control. For SO(2) rotations, rotating values too makes residual far contributions combine incoherently, extending the range. Controlled experiments support these predictions: random accumulated rotations substantially improve extrapolation over RoPE, learned token-dependent rotations maintain near-training-length perplexity far beyond the training context, and rotating values helps over queries and keys alone. Rotation-only models still degrade at extreme lengths, while ALiBi stays length-stable, consistent with the need for far-mass control. 

---
# Quantifying Explainable AI-introduced signal noise on ECG data with Spectral Entropy 

**Authors**: David A. Kelly, Nathan Blake  

**Link**: [PDF](https://arxiv.org/pdf/2606.24974)  

**Abstract**: Explainability techniques are used to assess the output of various deep learning models. This is especially true in healthcare, where models need to be trusted and decisions justified. Explainability (XAI) tools use heuristics which often add signal noise to the explanation "core". It is not always obvious what is signal from the model and what is noise from the XAI. We propose the use of spectral entropy as a measure of noise in XAI output. We demonstrate its usefulness in the context of classifying arrhythmias in an ECG dataset with different post hoc explainability techniques. 

---
# LLM Performance on a Real, Double-Marked GCSE Benchmark 

**Authors**: Malachy Fox, Kavi Samra, Paul Jung  

**Link**: [PDF](https://arxiv.org/pdf/2606.24973)  

**Abstract**: We introduce a dataset of 32,534 double-marked real student responses to GCSE mock exams (GCSEs are the UK's national exams, taken at age ~16), spanning 328 questions across five subjects and including handwritten work. We test whether off-the-shelf large language models agree with examiners as closely as the two examiners agree with each other. We find that models overwhelmingly agree well with the examiner consensus across subjects, with the top performing models agreeing more closely with examiners than examiners agree with each other. Models achieve high scores for subjective tasks like English essay marking, as well as handling complex and messy handwritten Maths paper scripts. Agreement is uniform near the examiner line, and not massively discriminated by model size, providing cost-effective automated marking solutions. 

---
# What Do Language Priors Contribute to Darcy-Flow Inversion? A Mechanistic Audit 

**Authors**: Taiga Saito, Yu Otake, Daijiro Mizutani, Sopheakpolin Mom  

**Link**: [PDF](https://arxiv.org/pdf/2606.24967)  

**Abstract**: In ill-posed inverse problems, the recovered solution depends as much on the prior as on the data, yet much of the engineering knowledge that could serve as that prior is recorded qualitatively rather than in formal mathematical form. Here we test whether sentence embeddings can act as an inference-time interface for injecting geological descriptions into a learned Darcy-flow inverse solver. Across six synthetic geological classes and an exploratory transfer to a benchmark reservoir model (SPE10), we vary only the conditioning representation and find that text conditioning reduces reconstruction error by 81 % relative to a no-text counterfactual. Most of this gain comes from a categorical, class-level constraint whose value concentrates where the hydraulic head leaves the conductivity field underdetermined, while within-class geometric detail is secondary and pattern-dependent. Compared with a discrete class label, sentence embeddings add little dense-observation accuracy but improve training stability and enable paraphrase-based sensitivity analysis and open-vocabulary inputs. These results show that language priors can serve as an engineering-informatics interface for injecting geological knowledge into learned inverse solvers, while clarifying when they help and what signal they actually carry. 

---
# Enhancing Clinician Decision-Making via Uncertainty-Aware Multi-Expert Fusion for Stroke Rehabilitation 

**Authors**: Tamim Ahmed, Thanassis Rikakis  

**Link**: [PDF](https://arxiv.org/pdf/2606.24960)  

**Abstract**: Tailoring stroke rehabilitation requires assessing how movements are organized, not merely if they succeed. Currently, this assessment is a rate-limiting bottleneck. Instruments like the Action Research Arm Test (ARAT) compress rich behavioral observations into single ordinal endpoints, discarding the movement-quality details that distinguish recovery from compensation. Automated alternatives typically chase accuracy on noisy, single-observer labels to output opaque scores - a technology-centric approach that rarely reaches clinical practice. To address this, we present xAARA: an engine designed to augment rather than replace clinical judgment. From multi-view video, xAARA returns ARAT assessments with calibrated uncertainty and explanations across task, movement-phase, and movement-quality levels. Treating clinical scoring as an ill-posed inference problem, xAARA composes 692 calibrated multimodal models via a Dynamic Bayesian Network with entropy-based gating. It qualifies results against clinical validity rules and defers low-confidence cases. In 105 stroke survivors (788 exercises), xAARA achieved 94.2% task accuracy (Cohen's kappa=0.934) and 81.3% movement-phase accuracy (kappa=0.727), reducing predictive uncertainty by 96.1% compared to single-clinician scoring. For subjective cases, it matched at least one rater 100% of the time and never returned out-of-range scores. Four independent clinicians validated the assessments and indicated willingness to adopt the system. We argue that principled uncertainty quantification and clinician-aligned explainability are the critical bridges moving automated assessment from technical demonstration to a deployable clinical tool. 

---
# Reliable Conformal Prediction for Ordinal Classification Using the Ranked Probability Score 

**Authors**: Stefan Haas, Luca Killmaier, Alireza Javanmardi, Eyke Hüllermeier  

**Link**: [PDF](https://arxiv.org/pdf/2606.24959)  

**Abstract**: Ordinal classification (OC) arises in high-stakes domains such as medicine and finance, where uncertainty quantification must account for the severity of ordinal errors. Conformal prediction (CP) provides distribution-free prediction sets with marginal coverage guarantees; however, its practical effectiveness depends critically on the choice of nonconformity function. We introduce a CP method for ordinal classification based on the ranked probability score (RPS), a proper scoring rule defined over cumulative predictive distributions. Although it reflects ordinal risk quite naturally, it has largely been neglected in conformal ordinal prediction (COP). When used as a measure of nonconformity, RPS yields median-centered contiguous prediction sets by construction. The method is model-agnostic, supports both assessed and grouped ordered categorical outcomes, and permits efficient implementation compared to greedy interval selection procedures. Across multiple ordinal image and tabular datasets, RPS-based CP produces contiguous prediction sets and strikes a favorable balance between prediction set width and the magnitude of ordinal miscoverage relative to existing CP methods. 

---
# Convex--Concave Quadratic Spectral Filtering for Graph Neural Networks 

**Authors**: Ranhui Yan, Jia Cai, Mengzhu Chen, Haodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24956)  

**Abstract**: Spectral graph neural networks (GNNs) interpret message passing as frequency-selective filtering. While low-order spectral filters are efficient, their limited selectivity often leads to weak attenuation outside the passband, whereas high-order alternatives introduce optimization challenges. We propose DCQ-GNN, a spectral GNN based on a compact bank of adaptive convex--concave quadratic filters. By restricting the filter order to two while explicitly exploiting complementary curvature, DCQ-GNN improves spectral selectivity as quantified by Dirichlet energy and entropy measures without resorting to high-order polynomial expansions. The model fuses filter outputs through a node-adaptive gating mechanism to enable node-wise structure-aware spectral selection. We provide a formal spectral analysis grounded in Dirichlet energy attenuation, von Neumann entropy, and curvature polarity, and derive explicit characterizations of filter behavior across varying levels of homophily and structural perturbations. Extensive benchmarks on 10 datasets show that DCQ-GNN ties for the top average rank (3.0) on heterophilic graphs and obtains the second-best rank (4.2) on homophilic graphs, remaining competitive with representative high-order polynomial spectral filters. Furthermore, under strong structural perturbations, DCQ-GNN exhibits substantially smaller performance degradation compared to both first-order and high-order baselines. These results demonstrate that curvature-aware quadratic banks provide a robust and efficient alternative to high-order spectral models while preserving optimization stability and computational efficiency. 

---
# Perfect Detection, Failed Control: The Geometry of Knowing vs. Steering in Language Models 

**Authors**: Cosimo Galeone, Anna Ettorre, Minsu Park, Giuseppe Ettorre, Daniele Ligorio  

**Link**: [PDF](https://arxiv.org/pdf/2606.24952)  

**Abstract**: A central aspiration of mechanistic interpretability is controllability: if we know where a behavior is represented in a model's activations, we should be able to modify it. This rests on a hidden premise -- that the direction which detects a behavior and the direction which controls it are the same, or close. We test this geometrically: what is the angle between the direction that best detects a behavior and the one that best causes it? If detection implies control the cosine is near 1; otherwise it quantifies a detection-intervention gap. On Gemma 2-2B-it, output format (clean JSON vs markdown fencing) collapses both roles onto one axis. Hallucination does not: the model detects fake entities with perfect linear separability (AUC = 1.000 from layer 5), yet that direction sits at cos = 0.12 (about 83 degrees) from the direction producing a refusal -- a small, reproducible alignment, far from the cos = 1 that "detection is control" would require. A detector built from activations, with no chosen tokens, likewise fails to align (cos = -0.06). The gap generalizes: across four models from three families and two scales (1B-9B), cos stays in [0.12, 0.20], identical before and after instruction tuning (0.1197 vs 0.1200), placing its origin in pretraining. A 15-degree rotation toward the refusal direction partially bridges it -- 73% and 60% refusal on two held-out fake-entity categories at 1.8% false positives. We then ask whether this cosine predicts steerability, and it does not: detection is a high-dimensional class, not a single direction, and what separates the steerable case is functional, not readable from a static angle. The cosine is a weight-computable signature of the dissociation between knowing and steering, not a predictor of it. 

---
# MacroLens: A Multi-Task Benchmark for Contextual Financial Reasoning under Macroeconomic Scenarios 

**Authors**: Patara Trirat, Jin Myung Kwak, Jay Heo, Heejun Lee, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24950)  

**Abstract**: Financial decision-making is contextual: forecasting prices, valuing companies, and assessing event exposure weigh price history, accounting fundamentals, macroeconomic regime, and contemporaneous text. A benchmark over these four signals is hard to build because finance violates four assumptions of time-series evaluation: text must be gated by its publication date to prevent look-ahead, quarterly fundamentals are reported with a one- to ninety-day lag, filing text is partly redundant with the numerical statement fields it accompanies, and macroeconomic regimes leak across calendar splits. No public benchmark addresses all four signals jointly. MacroLens covers 4,416 U.S. small- and micro-cap equities over 2021-2026. Seven tasks share one point-in-time panel of prices, 46.8M XBRL accounting facts, 53 macroeconomic series, 295,860 SEC filings, and 215,882 news articles, plus a scenario layer of 1,130 macroeconomic events across 49 types automatically detected and rendered as natural language. Tasks span contextual forecasting, public and private valuation, statement generation from fundamentals and descriptions, scenario-conditioned returns, and real-estate valuation. We evaluate 19 methods across six families spanning naive heuristics through time-series foundation models, fine-tuned LLM-based time-series models, and zero-shot large language models (LLMs), plus a five-step feature-context ablation on two frontier LLMs and a gradient-boosted baseline. MacroLens is released at this https URL. 

---
# What Does a Pathological Speech Assessment Model Know about Acoustic Features? A Case Study on Oral and Oropharyngeal Cancer Patients 

**Authors**: Tuan Nguyen, Corinne Fredouille, Alain Ghio, Muriel Lalain, Virginie Woisard  

**Link**: [PDF](https://arxiv.org/pdf/2606.24949)  

**Abstract**: This work investigates the interpretability of a Wav2Vec 2.0based speech intelligibility assessment model for oral and oropharyngeal cancer patients through canonical correlation analysis. By measuring the correlation between the model embeddings and eGeMAPS low-level descriptors (LLDs) as an interpretable reference, we analyze how acoustic information is encoded across the model layers. The analysis is conducted at two levels: individual LLDs layer-wise, and group-level: prosodic, spectral, and voice quality. Results show that the learned representations are most strongly correlated with spectral and prosodic features, with the first MFCC coefficient yielding the highest correlations across all layers. At the group level, spectral and prosodic groups achieve correlations of 0.77 and 0.71 respectively, while voice quality reaches 0.65. Beyond model interpretability, this work also offers practical guidance on acoustic feature selection for pathological speech assessment. 

---
# Holographic Memory for Zero-Shot Compositional Reasoning in Knowledge Graphs: A Mechanistic Study of Where and Why It Fails 

**Authors**: Randhir Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.24948)  

**Abstract**: Knowledge graph embedding (KGE) models predict single-hop links well but have no mechanism for zero-shot compositional queries: multi-hop questions whose relation chains never appeared during training. Holographic Reduced Representations (HRR), which bind and unbind symbols via circular convolution, are a theoretically attractive candidate, since binding is approximately invertible and associative. We test whether this promise holds.
We study two holographic memory variants, real-valued HRR and phase-only Fourier HRR (FHRR), each with a modern Hopfield cleanup, on FB15k-237 over five seeds. Four findings follow. First, both are competitive single-hop retrievers (filtered MRR 0.358 +/- 0.002 for HRR, 0.350 +/- 0.021 for FHRR). Second, neither composes zero-shot: accuracy stays at chance across all cleanup temperatures. Third, the main contribution, we localise the failure mechanistically. A hop-1 probe shows the memory recovers the correct intermediate entity with high fidelity (MRR 0.896 +/- 0.002 for HRR), yet composition still fails even with a verified-correct intermediate. A second probe shows why: posing the ground-truth second-hop fact as a standalone atomic query, bypassing composition entirely, already recovers it at only 0.26 to 0.48x average atomic accuracy, uniformly across relation fan-out. The bottleneck is not the bind-unbind algebra or the cleanup; it is that facts compositional chains pass through are intrinsically harder for the superposed memory to retrieve, a capacity and interference effect present already at a single hop. Fourth, we prove (Lemma 4.1) that FHRR's softmax cleanup is not phase-equivariant, compounding the primary failure on the minority of chains where hop-1 itself errs. Fixing zero-shot composition requires improving retrieval capacity under superposition, not just redesigning the cleanup. 

---
# EmotionAI: A Privacy-Preserving Computational Intelligence Pipeline for Speech-Emotion-Grounded Conversational Analysis 

**Authors**: Wai Laam Mak, Isibor Kennedy Ihianle, Pedro Machado  

**Link**: [PDF](https://arxiv.org/pdf/2606.24941)  

**Abstract**: Reviewing recorded interviews for affective cues such as composure, hesitation and agitation is slow and subjective, and cloud services that could automate it require sensitive audio to leave the device. EmotionAI is a fully local Computational Intelligence (CI) pipeline that couples Speech Emotion Recognition (SER) with generative reasoning. Speaker diarisation, Whisper Automatic Speech Recognition (ASR) and a wav2vec2 emotion classifier produce per-segment affective evidence, which is then passed to an adversarial three-model local Large Language Model (LLM) panel for timestamp-grounded and citation-constrained question answering. Zero-shot evaluation on the RAVDESS four-class English subset (n = 672) exposes cross-corpus fragility rather than classifier superiority: the deployed classifier scores 48.8% accuracy, above random (24.9%) and majority (28.6%) baselines but below an in-domain MFCC + logistic-regression comparator (71.0%). The complete pipeline runs in a mean 157 s on CPU (real-time factor approximately 1.33) with zero external calls. The contribution is not state-of-the-art SER but an auditable, privacy-preserving integration of imperfect affective evidence into grounded conversational analysis, together with an honest empirical account of where cross-corpus transfer and human-centred validation still fall short. 

---
# Stable-Shift: Biologically Structured Prediction of Transcriptional Responses to Unseen Gene Perturbations 

**Authors**: Sajib Acharjee Dip, Liqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24940)  

**Abstract**: Predicting transcriptional responses to genetic perturbations could reduce the experimental burden of functional genomics, but extrapolation to genes that were never perturbed during training remains difficult. We present Stable-Shift, a structured method for estimating unseen-gene responses. Stable-Shift aggregates single-cell measurements into perturbation-level expression shifts, fits a low-rank response basis using training perturbations only, and predicts an unseen gene's coordinates in that basis from biological context. The context combines STRING interactions, network structure, control-cell expression statistics, and Gene Ontology annotations; the evaluated implementation uses graph convolution to integrate these inputs. On the supplied K562 Perturb-seq benchmark, Stable-Shift obtained 0.592 cosine similarity, compared with 0.569 for GEARS, together with higher Spearman correlation and top-gene precision among the evaluated methods. Its mean cosine similarity over five unseen-gene splits was 0.589 +/- 0.008. The same ordering was observed in the supplied graph-aware, residualized, gene-space, and Norman-dataset comparisons. These results support further study of biologically structured latent-response prediction, while the lower gene-space accuracy and sensitivity to sparse graph neighborhoods limit the scope of the present conclusions. 

---
# Privacy-preserving federated tensor decomposition of single-cell immune data: recovering multicellular programs across institutions 

**Authors**: Axel Faes, Stephanie M. van den Berg, Maryam Amir Haeri  

**Link**: [PDF](https://arxiv.org/pdf/2606.24938)  

**Abstract**: Tensor decomposition of donor $\times$ cell-type $\times$ gene single-cell data recovers
\emph{multicellular programs}: coordinated axes of inter-individual transcriptional variation that
span cell types and stratify disease. Yet immune single-cell atlases are increasingly
multi-institution, multi-ancestry, and governed, so patient cells often cannot be pooled. We present
a federated estimator: each site computes a local program subspace, and a coordinator merges these by
stacked SVD under federated global-mean centering, provably equivalent (up to truncation) to the
centralised decomposition. This centering makes the merge robust to site-label confounding (program
AUC $0.957$ vs.\ $0.861$ for naive per-site centering). Only program subspaces leave a site, and
aggregation is compatible with secure aggregation. On a 261-donor systemic lupus erythematosus atlas
it recovers the canonical interferon program (ISG enrichment AUC $0.998$; case--control separation
$0.958$; bootstrap $\Delta\text{AUC}=-0.000$, 95\% CI $[-0.004,+0.012]$ vs.\ centralised), across
institution-scale and multi-ancestry partitions, and across three \emph{real} COVID-19 sites
(subspace correlation $0.989$). It recovers the program when \emph{no site observes all cell types}
(correlation $1.000$, exact by construction), which fixed-feature federated PCA cannot. On an
interstitial-lung-disease atlas the recovered program predicts disease better than the best single
cell type (AUC $0.96$ vs.\ $0.91$; gap 95\% CI excludes zero) and the advantage survives federation;
a liver cohort is consistent ($p=0.005$). Membership-inference shows secure aggregation cuts attack
AUC from $0.91$ to $0.61$. The method enables cross-institution, cross-ancestry recovery of
multicellular immune programs without sharing cells. 

---
# Self-Modulating Quantum Fast-Weight Programmers for Efficient Adaptive Sequential Learning 

**Authors**: Samuel Yen-Chi Chen, Yifeng Peng, Kuo-Chung Peng, Jiun-Cheng Jiang, Chun-Hua Lin, Junghoon Justin Park, Huan-Hsin Tseng, Hsin-Yi Lin, Kuan-Cheng Chen, Chen-Yu Liu, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.24933)  

**Abstract**: Recent advances in quantum machine learning have motivated efficient models for sequential data processing. In this paper, we propose Self-Modulating Quantum Fast Weight Programmers, or Self-Modulating QFWP, which extends Quantum Fast Weight Programmers by introducing adaptive modulation over both newly generated fast-weight updates and historical fast-weight memory. Numerical results show that the proposed mechanism improves convergence stability and prediction performance across varying model settings, including different numbers of qubits and input sequence lengths. We further provide theoretical arguments explaining how self-modulation balances new information injection with memory retention, thereby enhancing temporal information propagation. These results suggest that Self-Modulating QFWP is a compact and effective framework for quantum machine learning on time-series data. 

---
# Recursive QLSTM with Dynamic Variational Quantum Circuit Adaptation 

**Authors**: Samuel Yen-Chi Chen, Yifeng Peng, Jiun-Cheng Jiang, Chun-Hua Lin, Kuo-Chung Peng, Junghoon Justin Park, Huan-Hsin Tseng, Hsin-Yi Lin, Kuan-Cheng Chen, Chen-Yu Liu, Shinjae Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2606.24932)  

**Abstract**: Recent advances in quantum computing and machine learning have motivated the development of quantum models for sequential data processing. In this paper, we propose a Recursive Quantum Long Short-Term Memory model, or Recursive QLSTM, which extends QLSTM through metacore-based recursive constructions. We numerically test the model under different input sequence lengths, metacore designs, and recursive rules, and identify the best-performing architecture among these variants. For this selected model, we further provide theoretical arguments explaining why its recursive structure improves temporal information propagation and enhances learning performance. Our results suggest that Recursive QLSTM offers a flexible and effective framework for quantum recurrent learning over input time series of various lengths. 

---
# Error-Aware TF-IDF Retrieval-Augmented Generation for ASR Error Correction 

**Authors**: Mohammad Aref Jafari-Raddani  

**Link**: [PDF](https://arxiv.org/pdf/2606.24915)  

**Abstract**: End-to-end automatic speech recognition systems frequently hallucinate rare entities and domain-specific terms, especially in low-resource languages. While retrieval-augmented generation frameworks can mitigate these errors using large language models, current architectures face significant challenges. They either rely on standard sparse retrieval that ignores phonetic misrecognitions or utilize heavyweight cross-modal embeddings that introduce high latency. This letter proposes a highly efficient, purely lexical error-aware framework designed to explicitly resolve phonetic and loop hallucinations. Our approach integrates a symmetric text normalization module with a novel error-aware term frequency-inverse document frequency algorithm. By constructing a sparse diagonal penalty matrix based on historical errors, the retriever mathematically prioritizes corrective documents containing specific high-risk misrecognitions. Evaluated on the Persian subset of the FLEURS dataset, our method increased the error-aware hit rate from 53.7% to 90.9%. In end-to-end evaluations, the integrated framework reduced the final word error rate from 23.06% to 18.83%, achieving significant accuracy gains with near-zero inference latency. 

---
# Velocity Prediction in Automatic Guitar Transcription 

**Authors**: Jackson Loth, Xavier Riley, Simon Dixon, Emmanouil Benetos  

**Link**: [PDF](https://arxiv.org/pdf/2606.24912)  

**Abstract**: Automatic Music Transcription (AMT) models have achieved a high level of success in polyphonic transcription of various instruments. Velocity, typically a measure of note intensity, is less commonly predicted in these models due to the absence of velocity labels in available datasets and lack of a proper definition for instruments other than piano. We present a methodology and model for velocity prediction in Automatic Guitar Transcription (AGT) which uses virtual instruments to generate synthetic training data with velocity labels. We first pretrain a model on this synthetic data. These weights are then transferred to a different model and trained on real guitar audio, allowing the model to retain the working velocity prediction while also achieving high performance and generalisability from the real training data. The velocity prediction is shown to outperform a baseline model which does not use the pretrained velocity weights, when evaluated on synthetic data. In addition, using the pretrained velocity weights offers a small improvement in note transcription, though the magnitude of this improvement is limited and not always significant depending on the testing data. Overall the model achieves results comparable to the state of the art in guitar transcription, while also successfully predicting velocity. 

---
# Attractive and Repulsive Pattern Control in Sequence Generation 

**Authors**: Francois Pachet  

**Link**: [PDF](https://arxiv.org/pdf/2606.24911)  

**Abstract**: Variable-order Markov models preserve local symbolic syntax by adapting context length, but long continuations can enter recurring high-order "tunnels": repeated suffixes, locally periodic passages, or copied fragments longer than the formal Markov order. This paper introduces signed pattern control for variable-order Markov generation with BP-Regular sampling. A weighted recurrence automaton computes an activation R for a chosen family of target patterns, and belief propagation samples exactly from P_beta(x) proportional to P_0(x) exp(beta R(x)). Negative coupling makes the target patterns costly during sampling; positive coupling rewards the same patterns and turns them into controlled attractors.
The target family may be mined online from overactive generated material, supplied by a score or style vocabulary, or designed as an experimental probe. The main experiments use the online homeostatic case, choosing patterns that become overactive in the sampling history. On six duration-bearing monophonic sources, including Bach and Telemann material, the negative branch reduces generated 8-gram self-reuse, increases the effective number of generated 8-grams, and increases coverage of training-supported 4-gram contexts while preserving substantial lower-order support. A pitch-sequence replication on five Weimar Jazz Database solos gives the same anti-reuse signature outside Baroque material. The same signed mechanism also provides a positive branch for probing attractor basins, phase transitions, and hysteresis in the underlying variable-order model. 

---
# End-to-End Voice Intent Recognition for Spontaneous Human-Drone Interaction with Naive Users 

**Authors**: Allan Henry, Solange Rossato, Christian Graff, Sylvain Huet, Jose-Ernesto Gomez-Balderas  

**Link**: [PDF](https://arxiv.org/pdf/2606.24910)  

**Abstract**: Voice control offers an intuitive alternative to manual drone piloting, yet most existing systems rely on rigid command vocabularies that fail to handle the spontaneous, disfluent speech of naive users. This paper addresses this gap by proposing an End-to-End Spoken Language Understanding architecture for real-time human-drone interaction in French. Our model combines a frozen Self-Supervised Learning acoustic encoder with a lightweight LSTM-based classification head, augmented by a cross-modal knowledge distillation objective that aligns acoustic representations with semantic embeddings from a text teacher, without requiring transcription at inference time. We evaluate our approach on VoiceStick, a novel French corpus of spontaneous speech collected during real teleoperation sessions with 29 nonexpert dyads. On simple voice commands, our best configuration achieves 93% accuracy at 7 ms inference latency, outperforming cascade baselines (79%, 202 ms) with a 29x speedup. On the full spontaneous speech test set, our architecture reaches 82% accuracy, with crossmodal distillation consistently improving robustness across all configurations. These results demonstrate that End-to-End architectures are not only feasible but preferable for spontaneous voice-guided UAV teleoperation, combining semantic robustness, low latency, and calibrated confidence. 

---
# Failure Modes of Large Language Models on Research-Level Mathematics: A Taxonomy and an Empirical Characterisation 

**Authors**: Arnesh Banerjee, Ayushi Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2606.24902)  

**Abstract**: The "First Proof" benchmark [1] posed ten research-level mathematics questions to the strongest publicly available LLMs and found them consistently wrong-not silent, but confidently, fluently wrong. This paper asks why. Working from the per-question post-mortems in First Proof's Appendix A, I identify four failure modes: citation fabrication (F1), premise smuggling (F2), silent problem reformulation (F3), and local-to-global compatibility gaps (F4). I then audit eight one-shot proofs generated by Gemini 2.5 Flash on Questions 1, 2, and 5 of the benchmark, using two instruments built specifically to surface F1 and F2. The central finding is uncomfortable for anyone who sees retrieval-augmented generation (RAG) as the obvious fix: not one of the eight proofs contained a confirmed fabricated citation, yet every single one contained at least one load-bearing claim asserted as a "fundamental result" or "standard argument" with no justification attached. That failure mode-F2, premise smuggling-is invisible to citation verification by design. A premise-audit instrument I introduce flags it at 100% precision (5/5 judge-confirmed flags are true positives) and 50% proof-level recall in this corpus. The taxonomy and the audit together suggest that the right long-term objective is building inference-time pipelines that prevent these failure modes from occurring, not just detecting them after the fact. Index Terms--Large language models, mathematical reasoning, hallucination, premise smuggling, failure-mode taxonomy. 

---
# LLM Evolution as an Industry-Scale Ecosystem: A Lifecycle Perspective on Continual Learning 

**Authors**: Hao Jiang, Enneng Yang, Guojie Zhu, Yibin Chen, Yunkun Xu, Zifu Kou, Jiayi Li, Chong Chen, Zhao Cao, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2606.24901)  

**Abstract**: Continual learning capability is critical for Industrial LLMs, as deployed models must be continuously updated to meet evolving requirements and environments, rather than repeatedly retrained from scratch. However, most existing research focuses on improvements on static benchmarks, failing to capture real industrial needs. In this survey, we reformulate Industrial Continual Learning (ICL) for LLMs as a closed-loop update-and-release problem in a versioned ecosystem, where updates propagate hierarchically to industrial, application-specific models and LLM-powered applications, with capability inheritance and transfer across versions and model families. From this ecosystem perspective, we identify three core challenges: repeated adaptation erodes model plasticity, foundation-model upgrades break capability inheritance, and long-term sustainability is constrained by deployment requirements. We then organize the technical landscape of ICL around five lifecycle design principles: preserving plasticity headroom, treating upgrades as capability transfer, enabling trustworthy continual reinforcement learning, making training recipes self-optimizing, and building accountability as a base layer for long-term iteration. For each principle, we synthesize representative technical directions. Finally, we evaluate the maturity of each principle and its technical components via an evidence-based lens, identify key gaps hindering real-world deployment, and outline a practical ICL deployment blueprint and a pathway for feeding industrial realities back into academic research. 

---
# On-Device Neural Architecture Search 

**Authors**: Andrea Mattia Garavagno, Edoardo Ragusa, Paolo Gastaldo, Antonio Frisoli, Claudio Loconsole  

**Link**: [PDF](https://arxiv.org/pdf/2606.24900)  

**Abstract**: This paper proposes a new approach to near-sensor computing, in which a lightweight Neural Architecture Search (NAS) is performed directly on the deployment device to find the best tiny neural architecture for analyzing the real-time data acquired through sensors. This new adaptation capability can be particularly useful in the case of human-machine interfaces for which the neural network analyzing the biometrical data can be re-designed each time the user changes, after a guided data collection procedure, fighting the typical data variations between individuals on a new level. To implement the proposed approach a new NAS has been designed and then validated on the Italian Sign Language dataset (ISL), a collection of surface electromyography (sEMG) signals of the signs of the Italian alphabet, using several embedded systems. Moreover, further validation on the Case Western Reserve University dataset (CWRU), a benchmark for intelligent fault diagnosis, is presented to suggest another possible application of the proposed approach. When run on a Raspberry Pi 4, the proposed NAS performs beyond the state of the art proposing a tiny neural architecture having 0.63 times less RAM occupancy and 5.96 percentage points of more accuracy in the case of the ISL dataset; and 0.44 times less RAM occupancy and 0.2 percentage points of more accuracy in the case of the CWRU dataset. 

---
# From Meta Idea to Advanced Mathematical Discovery -- Human-AI Co-Discovery of Sign-Embedding Quantum Algorithms 

**Authors**: Yanqiao Wang, Jin-Peng Liu, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24899)  

**Abstract**: AI-assisted mathematics is often evaluated on solving predefined problems. In practice, however, many important advances begin earlier, when a vague research intuition is transformed into a concrete problem, a promising route, and a theorem family worth proving. This report studies that stage through a case study that led to sign-embedding quantum algorithms for matrix equations and matrix functions, foundational primitives in quantum linear algebra and operator-output quantum algorithms. The project began with a human-originated intuition that rational approximation is especially effective for jump-type functions such as the sign function, and might therefore serve as a design principle for quantum algorithms. Rather than merely assisting after the problem was fixed, AI-assisted exploration, including workflows later integrated into the agentic AI-mathematician system AIM, played a key role in expanding this intuition into a route map, comparing candidate formulations, and converging toward sign embedding as the central framework. AIM then helped connect a known matrix-sign identity to wider classes of matrix equations and matrix functions, and drafted proof and complexity calculations. The decisive scientific judgments remained human: selecting which human-AI-expanded routes were worth pursuing, rejecting a Cayley-trapezoidal approximation when its validity required a hidden condition, and refining the Sylvester implementation from a coarse quadratic-gap query route to the final factorized and scaled analysis. The report argues that human-AI co-discovery workflows, with systems such as AIM as important components, are most valuable not as standalone theorem provers, but as research partners for problem formation, connection discovery, derivation, and skeptical review inside a human-gated research loop. 

---
# Dense Supervision Is Not Enough: The Readout Blind Spot in Looped Language Models 

**Authors**: Rituraj Sharma, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2606.24898)  

**Abstract**: Looped language models turn hidden states into runtime state: each state is decoded for prediction and fed back into future computation. This creates a basic supervision question: which state variables does cross-entropy actually control? We show that dense per-loop cross-entropy controls the variables exposed by the readout, not every variable active in the recurrent transition. Hidden-state scale gives a concrete failure mode. Scale-invariant readouts such as RMSNorm and LayerNorm hide radial scale from the immediate cross-entropy loss, while pre-norm residual recurrence continues to carry and update that same scale. Thus per-loop loss can make early exits usable without controlling recurrent scale. In 44M and 129M looped transformers without inter-loop normalization, per-loop cross-entropy through RMSNorm readouts still drives final hidden-state norms into the thousands or tens of thousands. Scale-visible readouts and explicit norm penalties keep norms in the tens, and scale-removing recurrence is the complementary architectural fix. The resulting design rule is simple: dense supervision trains exits; recurrent scale control requires either making scale visible to a loss or removing it from the loop. Consistent with this rule, scale-controlled variants achieve lower perplexity at matched inference-depth operating points in our variable-depth benchmarks. 

---
# RWGBench: Evaluating Scholarly Positioning in Related Work Generation 

**Authors**: Anzhe Xie, Weihang Su, Jiaxin Mao, Yiqun Liu, Shaoping Ma, Qingyao Ai  

**Link**: [PDF](https://arxiv.org/pdf/2606.24894)  

**Abstract**: Large language models have shown strong fluency in scientific writing, yet the evaluation of related work generation (RWG) remains limited. Existing RWG evaluations largely inherit summarization-oriented metrics, using lexical or semantic similarity to reference sections as proxies for quality. However, related work writing is fundamentally a citation-level scholarly positioning task: it requires selecting, organizing, and framing prior work to clarify how a target paper relates to, differs from, and contributes beyond existing this http URL a result, models may generate coherent and semantically-relevant text while exhibiting academically critical failures, such as inappropriate citation selection or misplaced references, that conventional metrics do not this http URL this end, we introduce \textbf{RWGBench}, a benchmark that evaluates RWG from the perspective of citation decision-making rather than text similarity. RWGBench is constructed from a large-scale collection of 40,108 computer science papers and a retrieval corpus of 1.09 million documents, with a carefully curated test set comprising 100 papers and their corresponding published related work this http URL propose a multi-dimensional evaluation framework that assesses citation selection, contextual appropriateness, organization, and discourse this http URL reveal systematic limitations in current systems that are obscured by standard evaluations, while Oracle studies further disentangle retrieval-level and generation-level bottlenecks. Human evaluation further shows that our citation-centric metrics align substantially better with expert judgment than surface-level text metrics. RWGBench offers a citation-centric testbed for developing and evaluating related work generation systems that are better aligned with scholarly writing practices. 

---
# ReviewGuard: Aligning LLM-Assisted Peer Review with Long-Term Scientific Impact 

**Authors**: Abdur Rasool, Xiaohui Huang, Yanqing Hu, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.24892)  

**Abstract**: Peer review is central to scientific quality control, yet it can undervalue papers that later achieve substantial citation impact. While frontier large language models have shown promise in automating aspects of peer review, they primarily mimic human reviewer preferences rather than predict long-term scientific value. We introduce ReviewGuard, a two-stage framework that aligns LLM-generated reviews with citation-based estimates of long-term scientific impact rather than contemporaneous reviewer judgments. On 20,861 AI/ML papers from OpenReview augmented with Semantic Scholar citation data, ReviewGuard achieves a Spearman correlation of \r{ho} = 0.776 with future citations on rejected-then-published papers, outperforming human reviewers (\r{ho} = 0.492) and a supervised Expert model (\r{ho} = 0.681). Under the same decision threshold, ReviewGuard flags 10.2% of high-impact rejected papers, compared with 1.8% for human reviewers, corresponding to a 5.6x improvement. Our results demonstrate that impact-aligned reinforcement learning can provide editors with a complementary signal for identifying high-potential work, without replacing human judgment. 

---
# Type Checking Project Haystack Grids using JSON Schema and Pydantic 

**Authors**: Thomas Hirsch, Samina Kadkhoda Masoumali, Gerald Schweiger  

**Link**: [PDF](https://arxiv.org/pdf/2606.24891)  

**Abstract**: Ontologies enable scalable energy services in buildings by supporting interoperability and automation. Project Haystack is a building ontology that is widely adopted due to its flexible, tag-based semantic model, openness, and extensibility, but suffers from ambiguous tag usage and limited automated validation. Although Project Haystack is formally open, its reliance on custom file formats and domain-specific languages that originate from the Haxall ecosystem creates a de facto barrier to integration. In this paper, we address these limitations by introducing a Python-based toolchain for Haystack. We present (i) a parser for Haystack definition files (Trio file format), and (ii) a code generator that derives Pydantic models and JSON Schema definitions from these parsed specifications. The resulting models enable static type checking and enable structural validation of Haystack grids within Python, as well as schema-based validation of JSON representations outside the Python ecosystem. All tools, generated models, and schemas are released publicly under an open-source license, with the goal of strengthening the Haystack ecosystem and opening a practical pathway beyond its current technical boundaries. 

---
# Small edits, large models: How Wikipedia advocacy shapes LLM values 

**Authors**: Jasmine Brazilek, Maria Navas, Alexa Gnauck  

**Link**: [PDF](https://arxiv.org/pdf/2606.24890)  

**Abstract**: Can a small group of volunteers shape how AI systems discuss animal welfare, just by editing Wikipedia? We show that they can. Wikipedia appears in nearly every major language model training dataset and is weighted more heavily than web-crawled text. The Pro-Animal Wikipedians (PAW), a group of advocates who add sourced animal welfare content to relevant articles, have made 125 edits across 115 pages. Using gradient-based data attribution (Bergson; MAGIC), we traced how these edits influence language model behavior. TrackStar retrieval attribution on Llama 3.1 8B found that PAW-edited sections made up 68 percent of the highest-attributed documents for animal welfare queries (p < 0.0001) but only 52 percent for unrelated queries about the same companies (p = 0.53): the model links PAW content specifically to animal welfare topics, not to the entities in general. MAGIC counterfactual influence estimation on Llama-3.2-1B, run across five random training-order seeds, gave the same picture even more sharply: in every seed, the top-10 most influential documents on animal welfare queries were all PAW edits (10 of 10, 5 of 5 seeds), while on general queries the same top-10 sat at chance (4 to 6 of 10). Mean PAW influence exceeded mean control influence on animal welfare queries with p < 0.0001 in every seed, an effect 6 to 30 times larger than on general queries. Leave-subset-out validation gave Spearman rho = 1.00 for all 10 runs. When we fine-tuned separate models on PAW content versus control content, each model performed better specifically on the type of text it was trained on: the PAW-trained model cut perplexity on animal welfare text from 12.4 to 8.4, while the control-trained model cut perplexity on control text from 16.1 to 11.4. A small, coordinated Wikipedia editing campaign therefore measurably shapes how language models handle the topics those edits address. 

---
