# Scan-do Attitude: Towards Autonomous CT Protocol Management using a Large Language Model Agent 

**Authors**: Xingjian Kang, Linda Vorberg, Andreas Maier, Alexander Katzmann, Oliver Taubmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.20270)  

**Abstract**: Managing scan protocols in Computed Tomography (CT), which includes adjusting acquisition parameters or configuring reconstructions, as well as selecting postprocessing tools in a patient-specific manner, is time-consuming and requires clinical as well as technical expertise. At the same time, we observe an increasing shortage of skilled workforce in radiology. To address this issue, a Large Language Model (LLM)-based agent framework is proposed to assist with the interpretation and execution of protocol configuration requests given in natural language or a structured, device-independent format, aiming to improve the workflow efficiency and reduce technologists' workload. The agent combines in-context-learning, instruction-following, and structured toolcalling abilities to identify relevant protocol elements and apply accurate modifications. In a systematic evaluation, experimental results indicate that the agent can effectively retrieve protocol components, generate device compatible protocol definition files, and faithfully implement user requests. Despite demonstrating feasibility in principle, the approach faces limitations regarding syntactic and semantic validity due to lack of a unified device API, and challenges with ambiguous or complex requests. In summary, the findings show a clear path towards LLM-based agents for supporting scan protocol management in CT imaging. 

---
# Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction 

**Authors**: Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20218)  

**Abstract**: Research on lane change prediction has gained attention in the last few years. Most existing works in this area have been conducted in simulation environments or with pre-recorded datasets, these works often rely on simplified assumptions about sensing, communication, and traffic behavior that do not always hold in practice. Real-world deployments of lane-change prediction systems are relatively rare, and when they are reported, the practical challenges, limitations, and lessons learned are often under-documented. This study explores cooperative lane-change prediction through a real hardware deployment in mixed traffic and shares the insights that emerged during implementation and testing. We highlight the practical challenges we faced, including bottlenecks, reliability issues, and operational constraints that shaped the behavior of the system. By documenting these experiences, the study provides guidance for others working on similar pipelines. 

---
# Federation of Agents: A Semantics-Aware Communication Fabric for Large-Scale Agentic AI 

**Authors**: Lorenzo Giusti, Ole Anton Werner, Riccardo Taiello, Matilde Carvalho Costa, Emre Tosun, Andrea Protani, Marc Molina, Rodrigo Lopes de Almeida, Paolo Cacace, Diogo Reis Santos, Luigi Serio  

**Link**: [PDF](https://arxiv.org/pdf/2509.20175)  

**Abstract**: We present Federation of Agents (FoA), a distributed orchestration framework that transforms static multi-agent coordination into dynamic, capability-driven collaboration. FoA introduces Versioned Capability Vectors (VCVs): machine-readable profiles that make agent capabilities searchable through semantic embeddings, enabling agents to advertise their capabilities, cost, and limitations. Our aarchitecturecombines three key innovations: (1) semantic routing that matches tasks to agents over sharded HNSW indices while enforcing operational constraints through cost-biased optimization, (2) dynamic task decomposition where compatible agents collaboratively break down complex tasks into DAGs of subtasks through consensus-based merging, and (3) smart clustering that groups agents working on similar subtasks into collaborative channels for k-round refinement before synthesis. Built on top of MQTT,s publish-subscribe semantics for scalable message passing, FoA achieves sub-linear complexity through hierarchical capability matching and efficient index maintenance. Evaluation on HealthBench shows 13x improvements over single-model baselines, with clustering-enhanced laboration particularly effective for complex reasoning tasks requiring multiple perspectives. The system scales horizontally while maintaining consistent performance, demonstrating that semantic orchestration with structured collaboration can unlock the collective intelligence of heterogeneous federations of AI agents. 

---
# Formal Verification of Minimax Algorithms 

**Authors**: Wieger Wesselink, Kees Huizing, Huub van de Wetering  

**Link**: [PDF](https://arxiv.org/pdf/2509.20138)  

**Abstract**: Using the Dafny verification system, we formally verify a range of minimax search algorithms, including variations with alpha-beta pruning and transposition tables. For depth-limited search with transposition tables, we introduce a witness-based correctness criterion and apply it to two representative algorithms. All verification artifacts, including proofs and Python implementations, are publicly available. 

---
# PEPS: Quantum-Inspired Reinforcement Learning for Coherent Reasoning Traces in LLMs 

**Authors**: Venkat Margapuri, Garik Kazanjian, Naren Kosaraju  

**Link**: [PDF](https://arxiv.org/pdf/2509.20105)  

**Abstract**: Large Language Models (LLMs) often struggle with maintaining coherent multi-step reasoning traces, particularly in tasks that require a structured logical flow. This work introduces a quantum-inspired approach to address the challenge by incorporating a fidelity-based reward derived from Projected Entangled Pair States (PEPS) into Proximal Policy Optimization. Unlike prior approaches that use direct supervision or contrastive objectives, the proposed method guides learning through structural consistency, offering a novel approach to enforce global coherence in generated reasoning traces. The proposed framework is evaluated using multiple coherence-determining metrics on diverse datasets such as GSM8K, StrategyQA, and EntailmentBank spanning arithmetic, intuitive, and entailment-based reasoning. Results show that the proposed quantum-inspired approach offers significant improvements over supervised, contrastive, and pretrained baseline approaches, highlighting the effectiveness of quantum-inspired fidelity as a foundation to improve reasoning trace coherence in LLMs. 

---
# Steerable Adversarial Scenario Generation through Test-Time Preference Alignment 

**Authors**: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Sun, Haotian Shi, Wei Ma, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.20102)  

**Abstract**: Adversarial scenario generation is a cost-effective approach for safety assessment of autonomous driving systems. However, existing methods are often constrained to a single, fixed trade-off between competing objectives such as adversariality and realism. This yields behavior-specific models that cannot be steered at inference time, lacking the efficiency and flexibility to generate tailored scenarios for diverse training and testing requirements. In view of this, we reframe the task of adversarial scenario generation as a multi-objective preference alignment problem and introduce a new framework named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator (SAGE). SAGE enables fine-grained test-time control over the trade-off between adversariality and realism without any retraining. We first propose hierarchical group-based preference optimization, a data-efficient offline alignment method that learns to balance competing objectives by decoupling hard feasibility constraints from soft preferences. Instead of training a fixed model, SAGE fine-tunes two experts on opposing preferences and constructs a continuous spectrum of policies at inference time by linearly interpolating their weights. We provide theoretical justification for this framework through the lens of linear mode connectivity. Extensive experiments demonstrate that SAGE not only generates scenarios with a superior balance of adversariality and realism but also enables more effective closed-loop training of driving policies. Project page: this https URL. 

---
# From Pheromones to Policies: Reinforcement Learning for Engineered Biological Swarms 

**Authors**: Aymeric Vellinger, Nemanja Antonic, Elio Tuci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20095)  

**Abstract**: Swarm intelligence emerges from decentralised interactions among simple agents, enabling collective problem-solving. This study establishes a theoretical equivalence between pheromone-mediated aggregation in \celeg\ and reinforcement learning (RL), demonstrating how stigmergic signals function as distributed reward mechanisms. We model engineered nematode swarms performing foraging tasks, showing that pheromone dynamics mathematically mirror cross-learning updates, a fundamental RL algorithm. Experimental validation with data from literature confirms that our model accurately replicates empirical \celeg\ foraging patterns under static conditions. In dynamic environments, persistent pheromone trails create positive feedback loops that hinder adaptation by locking swarms into obsolete choices. Through computational experiments in multi-armed bandit scenarios, we reveal that introducing a minority of exploratory agents insensitive to pheromones restores collective plasticity, enabling rapid task switching. This behavioural heterogeneity balances exploration-exploitation trade-offs, implementing swarm-level extinction of outdated strategies. Our results demonstrate that stigmergic systems inherently encode distributed RL processes, where environmental signals act as external memory for collective credit assignment. By bridging synthetic biology with swarm robotics, this work advances programmable living systems capable of resilient decision-making in volatile environments. 

---
# MACD: Multi-Agent Clinical Diagnosis with Self-Learned Knowledge for LLM 

**Authors**: Wenliang Li, Rui Yan, Xu Zhang, Li Chen, Hongji Zhu, Jing Zhao, Junjun Li, Mengru Li, Wei Cao, Zihang Jiang, Wei Wei, Kun Zhang, Shaohua Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20067)  

**Abstract**: Large language models (LLMs) have demonstrated notable potential in medical applications, yet they face substantial challenges in handling complex real-world clinical diagnoses using conventional prompting methods. Current prompt engineering and multi-agent approaches typically optimize isolated inferences, neglecting the accumulation of reusable clinical experience. To address this, this study proposes a novel Multi-Agent Clinical Diagnosis (MACD) framework, which allows LLMs to self-learn clinical knowledge via a multi-agent pipeline that summarizes, refines, and applies diagnostic insights. It mirrors how physicians develop expertise through experience, enabling more focused and accurate diagnosis on key disease-specific cues. We further extend it to a MACD-human collaborative workflow, where multiple LLM-based diagnostician agents engage in iterative consultations, supported by an evaluator agent and human oversight for cases where agreement is not reached. Evaluated on 4,390 real-world patient cases across seven diseases using diverse open-source LLMs (Llama-3.1 8B/70B, DeepSeek-R1-Distill-Llama 70B), MACD significantly improves primary diagnostic accuracy, outperforming established clinical guidelines with gains up to 22.3% (MACD). On the subset of the data, it achieves performance on par with or exceeding that of human physicians (up to 16% improvement over physicians-only diagnosis). Additionally, on the MACD-human workflow, it achieves an 18.6% improvement compared to physicians-only diagnosis. Moreover, self-learned knowledge exhibits strong cross-model stability, transferability, and model-specific personalization, while the system can generate traceable rationales, enhancing explainability. Consequently, this work presents a scalable self-learning paradigm for LLM-assisted diagnosis, bridging the gap between the intrinsic knowledge of LLMs and real-world clinical practice. 

---
# Embodied AI: From LLMs to World Models 

**Authors**: Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20021)  

**Abstract**: Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation. 

---
# CON-QA: Privacy-Preserving QA using cloud LLMs in Contract Domain 

**Authors**: Ajeet Kumar Singh, Rajsabi Surya, Anurag Tripathi, Santanu Choudhury, Sudhir Bisane  

**Link**: [PDF](https://arxiv.org/pdf/2509.19925)  

**Abstract**: As enterprises increasingly integrate cloud-based large language models (LLMs) such as ChatGPT and Gemini into their legal document workflows, protecting sensitive contractual information - including Personally Identifiable Information (PII) and commercially sensitive clauses - has emerged as a critical challenge. In this work, we propose CON-QA, a hybrid privacy-preserving framework designed specifically for secure question answering over enterprise contracts, effectively combining local and cloud-hosted LLMs. The CON-QA framework operates through three stages: (i) semantic query decomposition and query-aware document chunk retrieval using a locally deployed LLM analysis, (ii) anonymization of detected sensitive entities via a structured one-to-many mapping scheme, ensuring semantic coherence while preventing cross-session entity inference attacks, and (iii) anonymized response generation by a cloud-based LLM, with accurate reconstruction of the original answer locally using a session-consistent many-to-one reverse mapping. To rigorously evaluate CON-QA, we introduce CUAD-QA, a corpus of 85k question-answer pairs generated over 510 real-world CUAD contract documents, encompassing simple, complex, and summarization-style queries. Empirical evaluations, complemented by detailed human assessments, confirm that CON-QA effectively maintains both privacy and utility, preserves answer quality, maintains fidelity to legal clause semantics, and significantly mitigates privacy risks, demonstrating its practical suitability for secure, enterprise-level contract documents. 

---
# LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation 

**Authors**: Huizhen Shu, Xuying Li, Zhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19839)  

**Abstract**: Achieving robust safety alignment in large language models (LLMs) while preserving their utility remains a fundamental challenge. Existing approaches often struggle to balance comprehensive safety with fine-grained controllability at the representation level. We introduce LATENTGUARD, a novel three-stage framework that combines behavioral alignment with supervised latent space control for interpretable and precise safety steering. Our approach begins by fine-tuning an LLM on rationalized datasets containing both reasoning-enhanced refusal responses to adversarial prompts and reasoning-enhanced normal responses to benign queries, establishing robust behavioral priors across both safety-critical and utility-preserving scenarios. We then train a structured variational autoencoder (VAE) on intermediate MLP activations, supervised by multi-label annotations including attack types, attack methods, and benign indicators. This supervision enables the VAE to learn disentangled latent representations that capture distinct adversarial characteristics while maintaining semantic interpretability. Through targeted manipulation of learned latent dimensions, LATENTGUARD achieves selective refusal behavior, effectively blocking harmful requests while preserving helpfulness for legitimate use cases. Experiments on Qwen3-8B demonstrate significant improvements in both safety controllability and response interpretability without compromising utility. Cross-architecture validation on Mistral-7B confirms the generalizability of our latent steering approach, showing consistent effectiveness across different model families. Our results suggest that structured representation-level intervention offers a promising pathway toward building safer yet practical LLM systems. 

---
# Analysis of approximate linear programming solution to Markov decision problem with log barrier function 

**Authors**: Donghwan Lee, Hyukjun Yang, Bum Geun Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.19800)  

**Abstract**: There are two primary approaches to solving Markov decision problems (MDPs): dynamic programming based on the Bellman equation and linear programming (LP). Dynamic programming methods are the most widely used and form the foundation of both classical and modern reinforcement learning (RL). By contrast, LP-based methods have been less commonly employed, although they have recently gained attention in contexts such as offline RL. The relative underuse of the LP-based methods stems from the fact that it leads to an inequality-constrained optimization problem, which is generally more challenging to solve effectively compared with Bellman-equation-based methods. The purpose of this paper is to establish a theoretical foundation for solving LP-based MDPs in a more effective and practical manner. Our key idea is to leverage the log-barrier function, widely used in inequality-constrained optimization, to transform the LP formulation of the MDP into an unconstrained optimization problem. This reformulation enables approximate solutions to be obtained easily via gradient descent. While the method may appear simple, to the best of our knowledge, a thorough theoretical interpretation of this approach has not yet been developed. This paper aims to bridge this gap. 

---
# Agentic Metacognition: Designing a "Self-Aware" Low-Code Agent for Failure Prediction and Human Handoff 

**Authors**: Jiexi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19783)  

**Abstract**: The inherent non-deterministic nature of autonomous agents, particularly within low-code/no-code (LCNC) environments, presents significant reliability challenges. Agents can become trapped in unforeseen loops, generate inaccurate outputs, or encounter unrecoverable failures, leading to user frustration and a breakdown of trust. This report proposes a novel architectural pattern to address these issues: the integration of a secondary, "metacognitive" layer that actively monitors the primary LCNC agent. Inspired by human introspection, this layer is designed to predict impending task failures based on a defined set of triggers, such as excessive latency or repetitive actions. Upon predicting a failure, the metacognitive agent proactively initiates a human handoff, providing the user with a clear summary of the agent's "thought process" and a detailed explanation of why it could not proceed. An empirical analysis of a prototype system demonstrates that this approach significantly increases the overall task success rate. However, this performance gain comes with a notable increase in computational overhead. The findings reframe human handoffs not as an admission of defeat but as a core design feature that enhances system resilience, improves user experience, and builds trust by providing transparency into the agent's internal state. The report discusses the practical and ethical implications of this approach and identifies key directions for future research. 

---
# The Conductor and the Engine: A Path Towards Co-Designed Reasoning 

**Authors**: Yuanxin Wang, Pawel Filipczuk, Anisha Garg, Amaan Dhada, Mohammad Hassanpour, David Bick, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19762)  

**Abstract**: Modern LLM reasoning relies on extensive test-time computation, driven by internal model training and external agentic orchestration. However, this synergy is often inefficient, as model verbosity and poor instruction following lead to wasted compute. We analyze this capability-cost trade-off and introduce an optimized reasoning workflow (\cepo) that empowers smaller open-source models to outperform models multiple times their size. We will open-source this workflow to enable further research. Our work demonstrates a clear path toward co-designing orchestration frameworks with the underlying model capabilities to unlock powerful reasoning in small-to-medium sized models. 

---
# UserRL: Training Interactive User-Centric Agent via Reinforcement Learning 

**Authors**: Cheng Qian, Zuxin Liu, Akshara Prabhakar, Jielin Qiu, Zhiwei Liu, Haolin Chen, Shirley Kokane, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19736)  

**Abstract**: Reinforcement learning (RL) has shown promise in training agentic models that move beyond static benchmarks to engage in dynamic, multi-turn interactions. Yet, the ultimate value of such agents lies in their ability to assist users, a setting where diversity and dynamics of user interaction pose challenges. In this work, we propose UserRL, a unified framework for training and evaluating user-centric abilities through standardized gym environments paired with simulated users. We systematically vary turn-level reward assignment and trajectory-level score calculation to analyze how different formulations affect learning under the GRPO algorithm. Our experiments across Qwen3 models reveal three key findings: (i) SFT cold start is critical for unlocking initial interaction ability and enabling sustained RL improvements; (ii) deliberate trajectory scoring yields more efficient and effective multi-turn interactions; and (iii) while stronger simulated users (e.g., GPT-4o) facilitates training, open-source simulators (e.g., Qwen3-32B) remain a cost-effective and transferable option. Together, these results highlight that careful design of reward shaping and user simulation choice is as crucial as model scale, and establish UserRL as a practical pathway for developing robust user-centric agentic models. All codes and data are public for future research. 

---
# Calibrated Reasoning: An Explanatory Verifier for Dynamic and Efficient Problem-Solving 

**Authors**: Anisha Garg, Engin Tekin, Yash More, David Bick, Nishit Neema, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.19681)  

**Abstract**: Advanced test-time computing strategies are essential for scaling reasoning models, but their effectiveness is capped by the models' poor self-evaluation. We propose a pairwise Explanatory Verifier, trained via reinforcement learning (GRPO), that produces calibrated confidence scores and associated natural language reasoning for generated solutions. Our verifier improves the accuracy and efficiency of test-time strategies like best-of-n and self-reflection. Crucially, it excels at identifying challenging failure modes, such as when both candidate solutions are identically incorrect, succeeding where standard methods like majority voting fail. 

---
# SteinerSQL: Graph-Guided Mathematical Reasoning for Text-to-SQL Generation 

**Authors**: Xutao Mao, Tao Liu, Hongying Zan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19623)  

**Abstract**: Large Language Models (LLMs) struggle with complex Text-to-SQL queries that demand both sophisticated mathematical reasoning and intricate schema navigation. Existing methods often tackle these challenges in isolation, creating a fractured reasoning process that compromises logical and structural correctness. To resolve this, we introduce SteinerSQL, a framework that unifies these dual challenges into a single, graph-centric optimization problem. SteinerSQL operates in three stages: mathematical decomposition to identify required tables (terminals), optimal reasoning scaffold construction via a Steiner tree problem, and multi-level validation to ensure correctness. On the challenging LogicCat and Spider2.0-Lite benchmarks, SteinerSQL establishes a new state-of-the-art with 36.10% and 40.04% execution accuracy, respectively, using Gemini-2.5-Pro. Beyond accuracy, SteinerSQL presents a new, unified paradigm for Text-to-SQL, paving the way for more robust and principled solutions to complex reasoning tasks. 

---
# What Does Your Benchmark Really Measure? A Framework for Robust Inference of AI Capabilities 

**Authors**: Nathanael Jo, Ashia Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2509.19590)  

**Abstract**: Evaluations of generative models on benchmark data are now ubiquitous, and their outcomes critically shape public and scientific expectations of AI's capabilities. Yet growing skepticism surrounds their reliability. How can we know that a reported accuracy genuinely reflects a model's true performance? Evaluations are often presented as simple measurements, but in reality they are inferences: to treat benchmark scores as evidence of capability is already to assume a theory of what capability is and how it manifests in a test. We make this step explicit by proposing a principled framework for evaluation as inference: begin from a theory of capability, and then derive methods for estimating it. This perspective, familiar in fields such as psychometrics, has not yet become commonplace in AI evaluation. As a proof of concept, we address a central challenge that undermines reliability: sensitivity to perturbations. After formulating a model of ability, we introduce methods that infer ability while accounting for uncertainty from sensitivity and finite samples, including an adaptive algorithm that significantly reduces sample complexity. Together, these contributions lay the groundwork for more reliable and trustworthy estimates of AI capabilities as measured through benchmarks. 

---
# Nano Bio-Agents (NBA): Small Language Model Agents for Genomics 

**Authors**: George Hong, Daniel Trejo Banos  

**Link**: [PDF](https://arxiv.org/pdf/2509.19566)  

**Abstract**: We investigate the application of Small Language Models (<10 billion parameters) for genomics question answering via agentic framework to address hallucination issues and computational cost challenges. The Nano Bio-Agent (NBA) framework we implemented incorporates task decomposition, tool orchestration, and API access into well-established systems such as NCBI and AlphaGenome. Results show that SLMs combined with such agentic framework can achieve comparable and in many cases superior performance versus existing approaches utilising larger models, with our best model-agent combination achieving 98% accuracy on the GeneTuring benchmark. Notably, small 3-10B parameter models consistently achieve 85-97% accuracy while requiring much lower computational resources than conventional approaches. This demonstrates promising potential for efficiency gains, cost savings, and democratization of ML-powered genomics tools while retaining highly robust and accurate performance. 

---
# Score the Steps, Not Just the Goal: VLM-Based Subgoal Evaluation for Robotic Manipulation 

**Authors**: Ramy ElMallah, Krish Chhajer, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19524)  

**Abstract**: Robot learning papers typically report a single binary success rate (SR), which obscures where a policy succeeds or fails along a multi-step manipulation task. We argue that subgoal-level reporting should become routine: for each trajectory, a vector of per-subgoal SRs that makes partial competence visible (e.g., grasp vs. pour). We propose a blueprint for StepEval, a cost-aware plug-in evaluation framework that utilizes vision-language models (VLMs) as automated judges of subgoal outcomes from recorded images or videos. Rather than proposing new benchmarks or APIs, our contribution is to outline design principles for a scalable, community-driven open-source project. In StepEval, the primary artifact for policy evaluation is the per-subgoal SR vector; however, other quantities (e.g., latency or cost estimates) are also considered for framework-optimization diagnostics to help the community tune evaluation efficiency and accuracy when ground-truth subgoal success labels are available. We discuss how such a framework can remain model-agnostic, support single- or multi-view inputs, and be lightweight enough to adopt across labs. The intended contribution is a shared direction: a minimal, extensible seed that invites open-source contributions, so that scoring the steps, not just the final goal, becomes a standard and reproducible practice. 

---
# Cognitive Load Limits in Large Language Models: Benchmarking Multi-Hop Reasoning 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.19517)  

**Abstract**: The scaling of Large Language Models (LLMs) has exposed a critical gap between their performance on static benchmarks and their fragility in dynamic, information-rich environments. While models excel at isolated tasks, the computational limits that govern their reasoning under cognitive load remain poorly understood. In this work, we introduce a formal theory of computational cognitive load, positing that extraneous, task-irrelevant information (Context Saturation) and interference from task-switching (Attentional Residue) are key mechanisms that degrade performance. We designed the Interleaved Cognitive Evaluation (ICE), a deconfounded benchmark to systematically manipulate these load factors on challenging multi-hop reasoning tasks. A comprehensive study (N = 10 replications per item across 200 questions) revealed significant performance variations across five instruction-tuned models. Smaller open-source architectures (Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2) exhibited baseline brittleness, achieving 0% accuracy (SEM = 0.0) across all conditions, including clean controls, on this high-intrinsic-load task. In contrast, Gemini-2.0-Flash-001 showed partial resilience, achieving 85% accuracy in control conditions, with a statistically significant degradation under context saturation ($\beta = -0.003$ per % load, $p < 0.001$). These findings provide preliminary evidence that cognitive load is a key contributor to reasoning failures, supporting theories of hallucination-as-guessing under uncertainty. We conclude that dynamic, cognitive-aware stress testing, as exemplified by the ICE benchmark, is essential for evaluating the true resilience and safety of advanced AI systems. 

---
# Estimating the Self-Consistency of LLMs 

**Authors**: Robert Nowak  

**Link**: [PDF](https://arxiv.org/pdf/2509.19489)  

**Abstract**: Systems often repeat the same prompt to large language models (LLMs) and aggregate responses to improve reliability. This short note analyzes an estimator of the self-consistency of LLMs and the tradeoffs it induces under a fixed compute budget $B=mn$, where $m$ is the number of prompts sampled from the task distribution and $n$ is the number of repeated LLM calls per prompt; the resulting analysis favors a rough split $m,n\propto\sqrt{B}$. 

---
# Evaluation-Aware Reinforcement Learning 

**Authors**: Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.19464)  

**Abstract**: Policy evaluation is often a prerequisite for deploying safety- and performance-critical systems. Existing evaluation approaches frequently suffer from high variance due to limited data and long-horizon tasks, or high bias due to unequal support or inaccurate environmental models. We posit that these challenges arise, in part, from the standard reinforcement learning (RL) paradigm of policy learning without explicit consideration of evaluation. As an alternative, we propose evaluation-aware reinforcement learning (EvA-RL), in which a policy is trained to maximize expected return while simultaneously minimizing expected evaluation error under a given value prediction scheme -- in other words, being "easy" to evaluate. We formalize a framework for EvA-RL and design an instantiation that enables accurate policy evaluation, conditioned on a small number of rollouts in an assessment environment that can be different than the deployment environment. However, our theoretical analysis and empirical results show that there is often a tradeoff between evaluation accuracy and policy performance when using a fixed value-prediction scheme within EvA-RL. To mitigate this tradeoff, we extend our approach to co-learn an assessment-conditioned state-value predictor alongside the policy. Empirical results across diverse discrete and continuous action domains demonstrate that EvA-RL can substantially reduce evaluation error while maintaining competitive returns. This work lays the foundation for a broad new class of RL methods that treat reliable evaluation as a first-class principle during training. 

---
# The Indispensable Role of User Simulation in the Pursuit of AGI 

**Authors**: Krisztian Balog, ChengXiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19456)  

**Abstract**: Progress toward Artificial General Intelligence (AGI) faces significant bottlenecks, particularly in rigorously evaluating complex interactive systems and acquiring the vast interaction data needed for training adaptive agents. This paper posits that user simulation -- creating computational agents that mimic human interaction with AI systems -- is not merely a useful tool, but is a critical catalyst required to overcome these bottlenecks and accelerate AGI development. We argue that realistic simulators provide the necessary environments for scalable evaluation, data generation for interactive learning, and fostering the adaptive capabilities central to AGI. Therefore, research into user simulation technology and intelligent task agents are deeply synergistic and must advance hand-in-hand. This article elaborates on the critical role of user simulation for AGI, explores the interdisciplinary nature of building realistic simulators, identifies key challenges including those posed by large language models, and proposes a future research agenda. 

---
# EmbeddingGemma: Powerful and Lightweight Text Representations 

**Authors**: Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam, Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, Daniel Cer, Alice Lisak, Min Choi, Lucas Gonzalez, Omar Sanseviero, Glenn Cameron, Ian Ballantyne, Kat Black, Kaifeng Chen, Weiyi Wang, Zhe Li, Gus Martins, Jinhyuk Lee, Mark Sherwood, Juyeong Ji, Renjie Wu, Jingxiao Zheng, Jyotinder Singh, Abheesht Sharma, Divya Sreepat, Aashi Jain, Adham Elarabawy, AJ Co, Andreas Doumanoglou, Babak Samari, Ben Hora, Brian Potetz, Dahun Kim, Enrique Alfonseca, Fedor Moiseev, Feng Han, Frank Palma Gomez, Gustavo Hernández Ábrego, Hesen Zhang, Hui Hui, Jay Han, Karan Gill, Ke Chen, Koert Chen, Madhuri Shanbhogue, Michael Boratko, Paul Suganthan, Sai Meher Karthik Duddu, Sandeep Mariserla, Setareh Ariafar, Shanfeng Zhang, Shijie Zhang, Simon Baumgartner, Sonam Goenka, Steve Qiu, Tanmaya Dabral, Trevor Walker, Vikram Rao, Waleed Khawaja, Wenlei Zhou, Xiaoqi Ren, Ye Xia, Yichang Chen, Yi-Ting Chen, Zhe Dong, Zhongli Ding, Francesco Visin, Gaël Liu, Jiageng Zhang, Kathleen Kenealy, Michelle Casbon, Ravin Kumar, Thomas Mesnard, Zach Gleicher, Cormac Brick, Olivier Lacombe, Adam Roberts, Yunhsuan Sung, Raphael Hoffmann, Tris Warkentin, Armand Joulin, Tom Duerig, Mojtaba Seyedhosseini  

**Link**: [PDF](https://arxiv.org/pdf/2509.20354)  

**Abstract**: We introduce EmbeddingGemma, a new lightweight, open text embedding model based on the Gemma 3 language model family. Our innovative training recipe strategically captures knowledge from larger models via encoder-decoder initialization and geometric embedding distillation. We improve model robustness and expressiveness with a spread-out regularizer, and ensure generalizability by merging checkpoints from varied, optimized mixtures. Evaluated on the Massive Text Embedding Benchmark (MTEB) across multilingual, English, and code domains, EmbeddingGemma (300M) achieves state-of-the-art results. Notably, it outperforms prior top models, both proprietary and open, with fewer than 500M parameters, and provides performance comparable to models double its size, offering an exceptional performance-to-cost ratio. Remarkably, this lead persists when quantizing model weights or truncating embedding outputs. This makes EmbeddingGemma particularly well-suited for low-latency and high-throughput use cases such as on-device applications. We provide ablation studies exploring our key design choices. We release EmbeddingGemma to the community to promote further research. 

---
# Morphological Synthesizer for Ge'ez Language: Addressing Morphological Complexity and Resource Limitations 

**Authors**: Gebrearegawi Gebremariam, Hailay Teklehaymanot, Gebregewergs Mezgebe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20341)  

**Abstract**: Ge'ez is an ancient Semitic language renowned for its unique alphabet. It serves as the script for numerous languages, including Tigrinya and Amharic, and played a pivotal role in Ethiopia's cultural and religious development during the Aksumite kingdom era. Ge'ez remains significant as a liturgical language in Ethiopia and Eritrea, with much of the national identity documentation recorded in Ge'ez. These written materials are invaluable primary sources for studying Ethiopian and Eritrean philosophy, creativity, knowledge, and civilization. Ge'ez has a complex morphological structure with rich inflectional and derivational morphology, and no usable NLP has been developed and published until now due to the scarcity of annotated linguistic data, corpora, labeled datasets, and lexicons. Therefore, we propose a rule-based Ge'ez morphological synthesizer to generate surface words from root words according to the morphological structures of the language. We used 1,102 sample verbs, representing all verb morphological structures, to test and evaluate the system. The system achieves a performance of 97.4%, outperforming the baseline model and suggesting that future work should build a comprehensive system considering morphological variations of the language.
Keywords: Ge'ez, NLP, morphology, morphological synthesizer, rule-based 

---
# Adaptive Event-Triggered Policy Gradient for Multi-Agent Reinforcement Learning 

**Authors**: Umer Siddique, Abhinav Sinha, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20338)  

**Abstract**: Conventional multi-agent reinforcement learning (MARL) methods rely on time-triggered execution, where agents sample and communicate actions at fixed intervals. This approach is often computationally expensive and communication-intensive. To address this limitation, we propose ET-MAPG (Event-Triggered Multi-Agent Policy Gradient reinforcement learning), a framework that jointly learns an agent's control policy and its event-triggering policy. Unlike prior work that decouples these mechanisms, ET-MAPG integrates them into a unified learning process, enabling agents to learn not only what action to take but also when to execute it. For scenarios with inter-agent communication, we introduce AET-MAPG, an attention-based variant that leverages a self-attention mechanism to learn selective communication patterns. AET-MAPG empowers agents to determine not only when to trigger an action but also with whom to communicate and what information to exchange, thereby optimizing coordination. Both methods can be integrated with any policy gradient MARL algorithm. Extensive experiments across diverse MARL benchmarks demonstrate that our approaches achieve performance comparable to state-of-the-art, time-triggered baselines while significantly reducing both computational load and communication overhead. 

---
# Uncovering Graph Reasoning in Decoder-only Transformers with Circuit Tracing 

**Authors**: Xinnan Dai, Chung-Hsiang Lo, Kai Guo, Shenglai Zeng, Dongsheng Luo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20336)  

**Abstract**: Transformer-based LLMs demonstrate strong performance on graph reasoning tasks, yet their internal mechanisms remain underexplored. To uncover these reasoning process mechanisms in a fundamental and unified view, we set the basic decoder-only transformers and explain them using the circuit-tracer framework. Through this lens, we visualize reasoning traces and identify two core mechanisms in graph reasoning: token merging and structural memorization, which underlie both path reasoning and substructure extraction tasks. We further quantify these behaviors and analyze how they are influenced by graph density and model size. Our study provides a unified interpretability framework for understanding structural reasoning in decoder-only Transformers. 

---
# Video models are zero-shot learners and reasoners 

**Authors**: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos  

**Link**: [PDF](https://arxiv.org/pdf/2509.20328)  

**Abstract**: The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models. 

---
# RAG Security and Privacy: Formalizing the Threat Model and Attack Surface 

**Authors**: Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20324)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an emerging approach in natural language processing that combines large language models (LLMs) with external document retrieval to produce more accurate and grounded responses. While RAG has shown strong potential in reducing hallucinations and improving factual consistency, it also introduces new privacy and security challenges that differ from those faced by traditional LLMs. Existing research has demonstrated that LLMs can leak sensitive information through training data memorization or adversarial prompts, and RAG systems inherit many of these vulnerabilities. At the same time, reliance of RAG on an external knowledge base opens new attack surfaces, including the potential for leaking information about the presence or content of retrieved documents, or for injecting malicious content to manipulate model behavior. Despite these risks, there is currently no formal framework that defines the threat landscape for RAG systems. In this paper, we address a critical gap in the literature by proposing, to the best of our knowledge, the first formal threat model for retrieval-RAG systems. We introduce a structured taxonomy of adversary types based on their access to model components and data, and we formally define key threat vectors such as document-level membership inference and data poisoning, which pose serious privacy and integrity risks in real-world deployments. By establishing formal definitions and attack models, our work lays the foundation for a more rigorous and principled understanding of privacy and security in RAG systems. 

---
# DRES: Benchmarking LLMs for Disfluency Removal 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20321)  

**Abstract**: Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems. 

---
# Z-Scores: A Metric for Linguistically Assessing Disfluency Removal 

**Authors**: Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2509.20319)  

**Abstract**: Evaluating disfluency removal in speech requires more than aggregate token-level scores. Traditional word-based metrics such as precision, recall, and F1 (E-Scores) capture overall performance but cannot reveal why models succeed or fail. We introduce Z-Scores, a span-level linguistically-grounded evaluation metric that categorizes system behavior across distinct disfluency types (EDITED, INTJ, PRN). Our deterministic alignment module enables robust mapping between generated text and disfluent transcripts, allowing Z-Scores to expose systematic weaknesses that word-level metrics obscure. By providing category-specific diagnostics, Z-Scores enable researchers to identify model failure modes and design targeted interventions -- such as tailored prompts or data augmentation -- yielding measurable performance improvements. A case study with LLMs shows that Z-Scores uncover challenges with INTJ and PRN disfluencies hidden in aggregate F1, directly informing model refinement strategies. 

---
# SIM-CoT: Supervised Implicit Chain-of-Thought 

**Authors**: Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20317)  

**Abstract**: Implicit Chain-of-Thought (CoT) methods present a promising, token-efficient alternative to explicit CoT reasoning in Large Language Models (LLMs), but a persistent performance gap has limited the application of implicit CoT. We identify a core latent instability issue by scaling the computational budget of implicit CoT approaches: as we increase the number of implicit reasoning tokens to enhance performance, the training process often becomes unstable and collapses. Our analysis reveals that this instability arises from the latent representations becoming homogeneous and losing their semantic diversity, a failure caused by insufficient step-level supervision in existing implicit CoT approaches. To address this issue, we propose SIM-CoT, a plug-and-play training module that introduces step-level supervision to stabilize and enrich the latent reasoning space. Specifically, SIM-CoT employs an auxiliary decoder during training to align each implicit token with its corresponding explicit reasoning step, ensuring that latent states capture distinct and meaningful information. The proposed auxiliary decoder is removed during inference, preserving the computational efficiency of implicit CoT methods with no added overhead. In addition, the auxiliary decoder affords interpretability of implicit reasoning by projecting each latent token onto an explicit reasoning vocabulary, enabling per-step visualization of semantic roles and diagnosis. SIM-CoT significantly enhances both the in-domain accuracy and out-of-domain stability of various implicit CoT methods, boosting baselines like Coconut by +8.2% on GPT-2 and CODI by +3.0% on LLaMA-3.1 8B. Demonstrating strong scalability, SIM-CoT also surpasses the explicit CoT baseline on GPT-2 by 2.1% with 2.3\times greater token efficiency, while substantially closing the performance gap on larger models like LLaMA-3.1 8B. 

---
# When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity 

**Authors**: Benjamin Feuer, Chiung-Yi Tseng, Astitwa Sarthak Lathe, Oussama Elachqar, John P Dickerson  

**Link**: [PDF](https://arxiv.org/pdf/2509.20293)  

**Abstract**: LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We release our code at this https URL 

---
# PGCLODA: Prompt-Guided Graph Contrastive Learning for Oligopeptide-Infectious Disease Association Prediction 

**Authors**: Dayu Tan, Jing Chen, Xiaoping Zhou, Yansen Su, Chunhou Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20290)  

**Abstract**: Infectious diseases continue to pose a serious threat to public health, underscoring the urgent need for effective computational approaches to screen novel anti-infective agents. Oligopeptides have emerged as promising candidates in antimicrobial research due to their structural simplicity, high bioavailability, and low susceptibility to resistance. Despite their potential, computational models specifically designed to predict associations between oligopeptides and infectious diseases remain scarce. This study introduces a prompt-guided graph-based contrastive learning framework (PGCLODA) to uncover potential associations. A tripartite graph is constructed with oligopeptides, microbes, and diseases as nodes, incorporating both structural and semantic information. To preserve critical regions during contrastive learning, a prompt-guided graph augmentation strategy is employed to generate meaningful paired views. A dual encoder architecture, integrating Graph Convolutional Network (GCN) and Transformer, is used to jointly capture local and global features. The fused embeddings are subsequently input into a multilayer perceptron (MLP) classifier for final prediction. Experimental results on a benchmark dataset indicate that PGCLODA consistently outperforms state-of-the-art models in AUROC, AUPRC, and accuracy. Ablation and hyperparameter studies confirm the contribution of each module. Case studies further validate the generalization ability of PGCLODA and its potential to uncover novel, biologically relevant associations. These findings offer valuable insights for mechanism-driven discovery and oligopeptide-based drug development. The source code of PGCLODA is available online at this https URL. 

---
# Feeding Two Birds or Favoring One? Adequacy-Fluency Tradeoffs in Evaluation and Meta-Evaluation of Machine Translation 

**Authors**: Behzad Shayegh, Jan-Thorsten Peter, David Vilar, Tobias Domhan, Juraj Juraska, Markus Freitag, Lili Mou  

**Link**: [PDF](https://arxiv.org/pdf/2509.20287)  

**Abstract**: We investigate the tradeoff between adequacy and fluency in machine translation. We show the severity of this tradeoff at the evaluation level and analyze where popular metrics fall within it. Essentially, current metrics generally lean toward adequacy, meaning that their scores correlate more strongly with the adequacy of translations than with fluency. More importantly, we find that this tradeoff also persists at the meta-evaluation level, and that the standard WMT meta-evaluation favors adequacy-oriented metrics over fluency-oriented ones. We show that this bias is partially attributed to the composition of the systems included in the meta-evaluation datasets. To control this bias, we propose a method that synthesizes translation systems in meta-evaluation. Our findings highlight the importance of understanding this tradeoff in meta-evaluation and its impact on metric rankings. 

---
# Investigating Security Implications of Automatically Generated Code on the Software Supply Chain 

**Authors**: Xiaofan Li, Xing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20277)  

**Abstract**: In recent years, various software supply chain (SSC) attacks have posed significant risks to the global community. Severe consequences may arise if developers integrate insecure code snippets that are vulnerable to SSC attacks into their products. Particularly, code generation techniques, such as large language models (LLMs), have been widely utilized in the developer community. However, LLMs are known to suffer from inherent issues when generating code, including fabrication, misinformation, and reliance on outdated training data, all of which can result in serious software supply chain threats. In this paper, we investigate the security threats to the SSC that arise from these inherent issues. We examine three categories of threats, including eleven potential SSC-related threats, related to external components in source code, and continuous integration configuration files. We find some threats in LLM-generated code could enable attackers to hijack software and workflows, while some others might cause potential hidden threats that compromise the security of the software over time. To understand these security impacts and severity, we design a tool, SSCGuard, to generate 439,138 prompts based on SSC-related questions collected online, and analyze the responses of four popular LLMs from GPT and Llama. Our results show that all identified SSC-related threats persistently exist. To mitigate these risks, we propose a novel prompt-based defense mechanism, namely Chain-of-Confirmation, to reduce fabrication, and a middleware-based defense that informs users of various SSC threats. 

---
# AnchDrive: Bootstrapping Diffusion Policies with Hybrid Trajectory Anchors for End-to-End Driving 

**Authors**: Jinhao Chai, Anqing Jiang, Hao Jiang, Shiyi Mu, Zichong Gu, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20253)  

**Abstract**: End-to-end multi-modal planning has become a transformative paradigm in autonomous driving, effectively addressing behavioral multi-modality and the generalization challenge in long-tail scenarios. We propose AnchDrive, a framework for end-to-end driving that effectively bootstraps a diffusion policy to mitigate the high computational cost of traditional generative models. Rather than denoising from pure noise, AnchDrive initializes its planner with a rich set of hybrid trajectory anchors. These anchors are derived from two complementary sources: a static vocabulary of general driving priors and a set of dynamic, context-aware trajectories. The dynamic trajectories are decoded in real-time by a Transformer that processes dense and sparse perceptual features. The diffusion model then learns to refine these anchors by predicting a distribution of trajectory offsets, enabling fine-grained refinement. This anchor-based bootstrapping design allows for efficient generation of diverse, high-quality trajectories. Experiments on the NAVSIM benchmark confirm that AnchDrive sets a new state-of-the-art and shows strong gen?eralizability 

---
# A HyperGraphMamba-Based Multichannel Adaptive Model for ncRNA Classification 

**Authors**: Xin An, Ruijie Li, Qiao Ning, Hui Li, Qian Ma, Shikai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20240)  

**Abstract**: Non-coding RNAs (ncRNAs) play pivotal roles in gene expression regulation and the pathogenesis of various diseases. Accurate classification of ncRNAs is essential for functional annotation and disease diagnosis. To address existing limitations in feature extraction depth and multimodal fusion, we propose HGMamba-ncRNA, a HyperGraphMamba-based multichannel adaptive model, which integrates sequence, secondary structure, and optionally available expression features of ncRNAs to enhance classification performance. Specifically, the sequence of ncRNA is modeled using a parallel Multi-scale Convolution and LSTM architecture (MKC-L) to capture both local patterns and long-range dependencies of nucleotides. The structure modality employs a multi-scale graph transformer (MSGraphTransformer) to represent the multi-level topological characteristics of ncRNA secondary structures. The expression modality utilizes a Chebyshev Polynomial-based Kolmogorov-Arnold Network (CPKAN) to effectively model and interpret high-dimensional expression profiles. Finally, by incorporating virtual nodes to facilitate efficient and comprehensive multimodal interaction, HyperGraphMamba is proposed to adaptively align and integrate multichannel heterogeneous modality features. Experiments conducted on three public datasets demonstrate that HGMamba-ncRNA consistently outperforms state-of-the-art methods in terms of accuracy and other metrics. Extensive empirical studies further confirm the model's robustness, effectiveness, and strong transferability, offering a novel and reliable strategy for complex ncRNA functional classification. Code and datasets are available at this https URL. 

---
# ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression 

**Authors**: Tom Burgert, Oliver Stoll, Paolo Rota, Begüm Demir  

**Link**: [PDF](https://arxiv.org/pdf/2509.20234)  

**Abstract**: The hypothesis that Convolutional Neural Networks (CNNs) are inherently texture-biased has shaped much of the discourse on feature use in deep learning. We revisit this hypothesis by examining limitations in the cue-conflict experiment by Geirhos et al. To address these limitations, we propose a domain-agnostic framework that quantifies feature reliance through systematic suppression of shape, texture, and color cues, avoiding the confounds of forced-choice conflicts. By evaluating humans and neural networks under controlled suppression conditions, we find that CNNs are not inherently texture-biased but predominantly rely on local shape features. Nonetheless, this reliance can be substantially mitigated through modern training strategies or architectures (ConvNeXt, ViTs). We further extend the analysis across computer vision, medical imaging, and remote sensing, revealing that reliance patterns differ systematically: computer vision models prioritize shape, medical imaging models emphasize color, and remote sensing models exhibit a stronger reliance towards texture. Code is available at this https URL. 

---
# Beyond Sharp Minima: Robust LLM Unlearning via Feedback-Guided Multi-Point Optimization 

**Authors**: Wenhan Wu, Zheyuan Liu, Chongyang Gao, Ren Wang, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.20230)  

**Abstract**: Current LLM unlearning methods face a critical security vulnerability that undermines their fundamental purpose: while they appear to successfully remove sensitive or harmful knowledge, this ``forgotten" information remains precariously recoverable through relearning attacks. We identify that the root cause is that conventional methods optimizing the forgetting loss at individual data points will drive model parameters toward sharp minima in the loss landscape. In these unstable regions, even minimal parameter perturbations can drastically alter the model's behaviors. Consequently, relearning attacks exploit this vulnerability by using just a few fine-tuning samples to navigate the steep gradients surrounding these unstable regions, thereby rapidly recovering knowledge that was supposedly erased. This exposes a critical robustness gap between apparent unlearning and actual knowledge removal. To address this issue, we propose StableUN, a bi-level feedback-guided optimization framework that explicitly seeks more stable parameter regions via neighborhood-aware optimization. It integrates forgetting feedback, which uses adversarial perturbations to probe parameter neighborhoods, with remembering feedback to preserve model utility, aligning the two objectives through gradient projection. Experiments on WMDP and MUSE benchmarks demonstrate that our method is significantly more robust against both relearning and jailbreaking attacks while maintaining competitive utility performance. 

---
# Multimodal Representation-disentangled Information Bottleneck for Multimodal Recommendation 

**Authors**: Hui Wang, Jinghui Qin, Wushao Wen, Qingling Li, Shanshan Zhong, Zhongzhan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20225)  

**Abstract**: Multimodal data has significantly advanced recommendation systems by integrating diverse information sources to model user preferences and item characteristics. However, these systems often struggle with redundant and irrelevant information, which can degrade performance. Most existing methods either fuse multimodal information directly or use rigid architectural separation for disentanglement, failing to adequately filter noise and model the complex interplay between modalities. To address these challenges, we propose a novel framework, the Multimodal Representation-disentangled Information Bottleneck (MRdIB). Concretely, we first employ a Multimodal Information Bottleneck to compress the input representations, effectively filtering out task-irrelevant noise while preserving rich semantic information. Then, we decompose the information based on its relationship with the recommendation target into unique, redundant, and synergistic components. We achieve this decomposition with a series of constraints: a unique information learning objective to preserve modality-unique signals, a redundant information learning objective to minimize overlap, and a synergistic information learning objective to capture emergent information. By optimizing these objectives, MRdIB guides a model to learn more powerful and disentangled representations. Extensive experiments on several competitive models and three benchmark datasets demonstrate the effectiveness and versatility of our MRdIB in enhancing multimodal recommendation. 

---
# The Cream Rises to the Top: Efficient Reranking Method for Verilog Code Generation 

**Authors**: Guang Yang, Wei Zheng, Xiang Chen, Yifan Sun, Fengji Zhang, Terry Yue Zhuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20215)  

**Abstract**: LLMs face significant challenges in Verilog generation due to limited domain-specific knowledge. While sampling techniques improve pass@k metrics, hardware engineers need one trustworthy solution rather than uncertain candidates. To bridge this gap, we formulate it as a semantic alignment problem between requirements and Verilog implementations, and propose VCD-RNK, a discriminator model tailored for efficient Verilog code reranking. Specifically, VCD-RNKincorporates Verilog-specific reasoning by distilling expert knowledge across three dimensions: code semantic analysis, test case generation, and functional correctness assessment. By explicitly simulating the above reasoning processes during inference, VCD-RNK effectively avoids computationally intensive test execution in existing methods. 

---
# Q-Palette: Fractional-Bit Quantizers Toward Optimal Bit Allocation for Efficient LLM Deployment 

**Authors**: Deokjae Lee, Hyun Oh Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.20214)  

**Abstract**: We study weight-only post-training quantization (PTQ), which quantizes the weights of a large language model (LLM) without retraining, using little or no calibration data. Weight-only PTQ is crucial for reducing the memory footprint and latency of LLM inference, especially in memory-bound, small-batch inference scenarios, such as personalized inference on edge devices. Despite its importance, irregular weight distributions with heavy-tailed outliers in LLMs complicate quantization, recently motivating rotation-based methods that transform weights into near-Gaussian distributions, which are more regular with fewer outliers, thereby reducing quantization error. In this work, we first derive the information-theoretically optimal bit allocation for Gaussianized weights under given bit budgets, revealing that fine-grained fractional-bit quantizers approaching the Gaussian distortion-rate bound are essential to achieve near-optimal quantization performance. To bridge this theoretical insight and practical implementation, we introduce Q-Palette, a versatile collection of fractional-bit quantizers that range from trellis-coded quantizers offering near-optimal distortion to simpler vector and scalar quantizers optimized for faster inference, all efficiently implemented with optimized CUDA kernels across various bitwidths. Furthermore, leveraging Q-Palette as a foundational component, we propose a novel mixed-scheme quantization framework, jointly optimizing quantizer choices and layer fusion decisions given resource constraints. The code is available at this https URL. 

---
# Low-Resource English-Tigrinya MT: Leveraging Multilingual Models, Custom Tokenizers, and Clean Evaluation Benchmarks 

**Authors**: Hailay Kidu Teklehaymanot, Gebrearegawi Gidey, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2509.20209)  

**Abstract**: Despite advances in Neural Machine Translation (NMT), low-resource languages like Tigrinya remain underserved due to persistent challenges, including limited corpora, inadequate tokenization strategies, and the lack of standardized evaluation benchmarks. This paper investigates transfer learning techniques using multilingual pretrained models to enhance translation quality for morphologically rich, low-resource languages. We propose a refined approach that integrates language-specific tokenization, informed embedding initialization, and domain-adaptive fine-tuning. To enable rigorous assessment, we construct a high-quality, human-aligned English-Tigrinya evaluation dataset covering diverse domains. Experimental results demonstrate that transfer learning with a custom tokenizer substantially outperforms zero-shot baselines, with gains validated by BLEU, chrF, and qualitative human evaluation. Bonferroni correction is applied to ensure statistical significance across configurations. Error analysis reveals key limitations and informs targeted refinements. This study underscores the importance of linguistically aware modeling and reproducible benchmarks in bridging the performance gap for underrepresented languages. Resources are available at this https URL
and this https URL 

---
# Play by the Type Rules: Inferring Constraints for LLM Functions in Declarative Programs 

**Authors**: Parker Glenn, Alfy Samuel, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20208)  

**Abstract**: Integrating LLM powered operators in declarative query languages allows for the combination of cheap and interpretable functions with powerful, generalizable language model reasoning. However, in order to benefit from the optimized execution of a database query language like SQL, generated outputs must align with the rules enforced by both type checkers and database contents. Current approaches address this challenge with orchestrations consisting of many LLM-based post-processing calls to ensure alignment between generated outputs and database values, introducing performance bottlenecks. We perform a study on the ability of various sized open-source language models to both parse and execute functions within a query language based on SQL, showing that small language models can excel as function executors over hybrid data sources. Then, we propose an efficient solution to enforce the well-typedness of LLM functions, demonstrating 7% accuracy improvement on a multi-hop question answering dataset with 53% improvement in latency over comparable solutions. We make our implementation available at this https URL 

---
# STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation 

**Authors**: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20190)  

**Abstract**: In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process. 

---
# How People Manage Knowledge in their "Second Brains"- A Case Study with Industry Researchers Using Obsidian 

**Authors**: Juliana Jansen Ferreira, Vinícius Segura, Joana Gabriela Souza, Joao Henrique Gallas Brasil  

**Link**: [PDF](https://arxiv.org/pdf/2509.20187)  

**Abstract**: People face overwhelming information during work activities, necessitating effective organization and management strategies. Even in personal lives, individuals must keep, annotate, organize, and retrieve knowledge from daily routines. The collection of records for future reference is known as a personal knowledge base. Note-taking applications are valuable tools for building and maintaining these bases, often called a ''second brain''. This paper presents a case study on how people build and explore personal knowledge bases for various purposes. We selected the note-taking tool Obsidian and researchers from a Brazilian lab for an in-depth investigation. Our investigation reveals interesting findings about how researchers build and explore their personal knowledge bases. A key finding is that participants' knowledge retrieval strategy influences how they build and maintain their content. We suggest potential features for an AI system to support this process. 

---
# An Improved Time Series Anomaly Detection by Applying Structural Similarity 

**Authors**: Tiejun Wang, Rui Wang, Xudong Mou, Mengyuan Ma, Tianyu Wo, Renyu Yang, Xudong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20184)  

**Abstract**: Effective anomaly detection in time series is pivotal for modern industrial applications and financial systems. Due to the scarcity of anomaly labels and the high cost of manual labeling, reconstruction-based unsupervised approaches have garnered considerable attention. However, accurate anomaly detection remains an unsettled challenge, since the optimization objectives of reconstruction-based methods merely rely on point-by-point distance measures, ignoring the potential structural characteristics of time series and thus failing to tackle complex pattern-wise anomalies. In this paper, we propose StrAD, a novel structure-enhanced anomaly detection approach to enrich the optimization objective by incorporating structural information hidden in the time series and steering the data reconstruction procedure to better capture such structural features. StrAD accommodates the trend, seasonality, and shape in the optimization objective of the reconstruction model to learn latent structural characteristics and capture the intrinsic pattern variation of time series. The proposed structure-aware optimization objective mechanism can assure the alignment between the original data and the reconstructed data in terms of structural features, thereby keeping consistency in global fluctuation and local characteristics. The mechanism is pluggable and applicable to any reconstruction-based methods, enhancing the model sensitivity to both point-wise anomalies and pattern-wise anomalies. Experimental results show that StrAD improves the performance of state-of-the-art reconstruction-based models across five real-world anomaly detection datasets. 

---
# Automated Multi-Agent Workflows for RTL Design 

**Authors**: Amulya Bhattaram, Janani Ramamoorthy, Ranit Gupta, Diana Marculescu, Dimitrios Stamoulis  

**Link**: [PDF](https://arxiv.org/pdf/2509.20182)  

**Abstract**: The rise of agentic AI workflows unlocks novel opportunities for computer systems design and optimization. However, for specialized domains such as program synthesis, the relative scarcity of HDL and proprietary EDA resources online compared to more common programming tasks introduces challenges, often necessitating task-specific fine-tuning, high inference costs, and manually-crafted agent orchestration. In this work, we present VeriMaAS, a multi-agent framework designed to automatically compose agentic workflows for RTL code generation. Our key insight is to integrate formal verification feedback from HDL tools directly into workflow generation, reducing the cost of gradient-based updates or prolonged reasoning traces. Our method improves synthesis performance by 5-7% for pass@k over fine-tuned baselines, while requiring only a few hundred training examples, representing an order-of-magnitude reduction in supervision cost. 

---
# CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning 

**Authors**: Lauren Deason, Adam Bali, Ciprian Bejean, Diana Bolocan, James Crnkovich, Ioana Croitoru, Krishna Durai, Chase Midler, Calin Miron, David Molnar, Brad Moon, Bruno Ostarcevic, Alberto Peltea, Matt Rosenberg, Catalin Sandu, Arthur Saputkin, Sagar Shah, Daniel Stan, Ernest Szocs, Shengye Wan, Spencer Whitman, Sven Krasser, Joshua Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2509.20166)  

**Abstract**: Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities. 

---
# Embedding Domain Knowledge for Large Language Models via Reinforcement Learning from Augmented Generation 

**Authors**: Chaojun Nie, Jun Zhou, Guanxiang Wang, Shisong Wud, Zichen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20162)  

**Abstract**: Large language models (LLMs) often exhibit limited performance on domain-specific tasks due to the natural disproportionate representation of specialized information in their training data and the static nature of these datasets. Knowledge scarcity and temporal lag create knowledge gaps for domain applications. While post-training on domain datasets can embed knowledge into models, existing approaches have some limitations. Continual Pre-Training (CPT) treats all tokens in domain documents with equal importance, failing to prioritize critical knowledge points, while supervised fine-tuning (SFT) with question-answer pairs struggles to develop the coherent knowledge structures necessary for complex reasoning tasks. To address these challenges, we propose Reinforcement Learning from Augmented Generation (RLAG). Our approach iteratively cycles between sampling generations and optimizing the model through calculated rewards, effectively embedding critical and contextually coherent domain knowledge. We select generated outputs with the highest log probabilities as the sampling result, then compute three tailored reward metrics to guide the optimization process. To comprehensively evaluate domain expertise, we assess answer accuracy and the rationality of explanations generated for correctly answered questions. Experimental results across medical, legal, astronomy, and current events datasets demonstrate that our proposed method significantly outperforms baseline approaches. Our code and data are open sourced at this https URL. 

---
# U-Mamba2-SSL for Semi-Supervised Tooth and Pulp Segmentation in CBCT 

**Authors**: Zhi Qin Tan, Xiatian Zhu, Owen Addison, Yunpeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20154)  

**Abstract**: Accurate segmentation of teeth and pulp in Cone-Beam Computed Tomography (CBCT) is vital for clinical applications like treatment planning and diagnosis. However, this process requires extensive expertise and is exceptionally time-consuming, highlighting the critical need for automated algorithms that can effectively utilize unlabeled data. In this paper, we propose U-Mamba2-SSL, a novel semi-supervised learning framework that builds on the U-Mamba2 model and employs a multi-stage training strategy. The framework first pre-trains U-Mamba2 in a self-supervised manner using a disruptive autoencoder. It then leverages unlabeled data through consistency regularization, where we introduce input and feature perturbations to ensure stable model outputs. Finally, a pseudo-labeling strategy is implemented with a reduced loss weighting to minimize the impact of potential errors. U-Mamba2-SSL achieved an average score of 0.872 and a DSC of 0.969 on the validation dataset, demonstrating the superior performance of our approach. The code is available at this https URL. 

---
# Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models 

**Authors**: Nicola Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20153)  

**Abstract**: This paper examines the integration of emotional intelligence into artificial intelligence systems, with a focus on affective computing and the growing capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to recognize and respond to human emotions. Drawing on interdisciplinary research that combines computer science, psychology, and neuroscience, the study analyzes foundational neural architectures - CNNs for processing facial expressions and RNNs for sequential data, such as speech and text - that enable emotion recognition. It examines the transformation of human emotional experiences into structured emotional data, addressing the distinction between explicit emotional data collected with informed consent in research settings and implicit data gathered passively through everyday digital interactions. That raises critical concerns about lawful processing, AI transparency, and individual autonomy over emotional expressions in digital environments. The paper explores implications across various domains, including healthcare, education, and customer service, while addressing challenges of cultural variations in emotional expression and potential biases in emotion recognition systems across different demographic groups. From a regulatory perspective, the paper examines emotional data in the context of the GDPR and the EU AI Act frameworks, highlighting how emotional data may be considered sensitive personal data that requires robust safeguards, including purpose limitation, data minimization, and meaningful consent mechanisms. 

---
# EchoBench: Benchmarking Sycophancy in Medical Large Vision-Language Models 

**Authors**: Botai Yuan, Yutian Zhou, Yingjie Wang, Fushuo Huo, Yongcheng Jing, Li Shen, Ying Wei, Zhiqi Shen, Ziwei Liu, Tianwei Zhang, Jie Yang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.20146)  

**Abstract**: Recent benchmarks for medical Large Vision-Language Models (LVLMs) emphasize leaderboard accuracy, overlooking reliability and safety. We study sycophancy -- models' tendency to uncritically echo user-provided information -- in high-stakes clinical settings. We introduce EchoBench, a benchmark to systematically evaluate sycophancy in medical LVLMs. It contains 2,122 images across 18 departments and 20 modalities with 90 prompts that simulate biased inputs from patients, medical students, and physicians. We evaluate medical-specific, open-source, and proprietary LVLMs. All exhibit substantial sycophancy; the best proprietary model (Claude 3.7 Sonnet) still shows 45.98% sycophancy, and GPT-4.1 reaches 59.15%. Many medical-specific models exceed 95% sycophancy despite only moderate accuracy. Fine-grained analyses by bias type, department, perceptual granularity, and modality identify factors that increase susceptibility. We further show that higher data quality/diversity and stronger domain knowledge reduce sycophancy without harming unbiased accuracy. EchoBench also serves as a testbed for mitigation: simple prompt-level interventions (negative prompting, one-shot, few-shot) produce consistent reductions and motivate training- and decoding-time strategies. Our findings highlight the need for robust evaluation beyond accuracy and provide actionable guidance toward safer, more trustworthy medical LVLMs. 

---
# KSDiff: Keyframe-Augmented Speech-Aware Dual-Path Diffusion for Facial Animation 

**Authors**: Tianle Lyu, Junchuan Zhao, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20128)  

**Abstract**: Audio-driven facial animation has made significant progress in multimedia applications, with diffusion models showing strong potential for talking-face synthesis. However, most existing works treat speech features as a monolithic representation and fail to capture their fine-grained roles in driving different facial motions, while also overlooking the importance of modeling keyframes with intense dynamics. To address these limitations, we propose KSDiff, a Keyframe-Augmented Speech-Aware Dual-Path Diffusion framework. Specifically, the raw audio and transcript are processed by a Dual-Path Speech Encoder (DPSE) to disentangle expression-related and head-pose-related features, while an autoregressive Keyframe Establishment Learning (KEL) module predicts the most salient motion frames. These components are integrated into a Dual-path Motion generator to synthesize coherent and realistic facial motions. Extensive experiments on HDTF and VoxCeleb demonstrate that KSDiff achieves state-of-the-art performance, with improvements in both lip synchronization accuracy and head-pose naturalness. Our results highlight the effectiveness of combining speech disentanglement with keyframe-aware diffusion for talking-head generation. 

---
# Discovering Association Rules in High-Dimensional Small Tabular Data 

**Authors**: Erkan Karabulut, Daniel Daza, Paul Groth, Victoria Degeler  

**Link**: [PDF](https://arxiv.org/pdf/2509.20113)  

**Abstract**: Association Rule Mining (ARM) aims to discover patterns between features in datasets in the form of propositional rules, supporting both knowledge discovery and interpretable machine learning in high-stakes decision-making. However, in high-dimensional settings, rule explosion and computational overhead render popular algorithmic approaches impractical without effective search space reduction, challenges that propagate to downstream tasks. Neurosymbolic methods, such as Aerial+, have recently been proposed to address the rule explosion in ARM. While they tackle the high dimensionality of the data, they also inherit limitations of neural networks, particularly reduced performance in low-data regimes.
This paper makes three key contributions to association rule discovery in high-dimensional tabular data. First, we empirically show that Aerial+ scales one to two orders of magnitude better than state-of-the-art algorithmic and neurosymbolic baselines across five real-world datasets. Second, we introduce the novel problem of ARM in high-dimensional, low-data settings, such as gene expression data from the biomedicine domain with around 18k features and 50 samples. Third, we propose two fine-tuning approaches to Aerial+ using tabular foundation models. Our proposed approaches are shown to significantly improve rule quality on five real-world datasets, demonstrating their effectiveness in low-data, high-dimensional scenarios. 

---
# Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving 

**Authors**: Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20109)  

**Abstract**: End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems. 

---
# Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models 

**Authors**: JuanaJuana Valeria Hurtado, Rohit Mohan, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.20107)  

**Abstract**: Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at this https URL. 

---
# Integrated Framework for LLM Evaluation with Answer Generation 

**Authors**: Sujeong Lee, Hayoung Lee, Seongsoo Heo, Wonik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20097)  

**Abstract**: Reliable evaluation of large language models is essential to ensure their applicability in practical scenarios. Traditional benchmark-based evaluation methods often rely on fixed reference answers, limiting their ability to capture important qualitative aspects of generated responses. To address these shortcomings, we propose an integrated evaluation framework called \textit{self-refining descriptive evaluation with expert-driven diagnostics}, SPEED, which utilizes specialized functional experts to perform comprehensive, descriptive analyses of model outputs. Unlike conventional approaches, SPEED actively incorporates expert feedback across multiple dimensions, including hallucination detection, toxicity assessment, and lexical-contextual appropriateness. Experimental results demonstrate that SPEED achieves robust and consistent evaluation performance across diverse domains and datasets. Additionally, by employing relatively compact expert models, SPEED demonstrates superior resource efficiency compared to larger-scale evaluators. These findings illustrate that SPEED significantly enhances fairness and interpretability in LLM evaluations, offering a promising alternative to existing evaluation methodologies. 

---
# Causal Understanding by LLMs: The Role of Uncertainty 

**Authors**: Oscar Lithgow-Serrano, Vani Kanjirangat, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.20088)  

**Abstract**: Recent papers show LLMs achieve near-random accuracy in causal relation classification, raising questions about whether such failures arise from limited pretraining exposure or deeper representational gaps. We investigate this under uncertainty-based evaluation, testing whether pretraining exposure to causal examples improves causal understanding >18K PubMed sentences -- half from The Pile corpus, half post-2024 -- across seven models (Pythia-1.4B/7B/12B, GPT-J-6B, Dolly-7B/12B, Qwen-7B). We analyze model behavior through: (i) causal classification, where the model identifies causal relationships in text, and (ii) verbatim memorization probing, where we assess whether the model prefers previously seen causal statements over their paraphrases. Models perform four-way classification (direct/conditional/correlational/no-relationship) and select between originals and their generated paraphrases. Results show almost identical accuracy on seen/unseen sentences (p > 0.05), no memorization bias (24.8% original selection), and output distribution over the possible options is almost flat, with entropic values near the maximum (1.35/1.39), confirming random guessing. Instruction-tuned models show severe miscalibration (Qwen: > 95% confidence, 32.8% accuracy, ECE=0.49). Conditional relations induce highest entropy (+11% vs. direct). These findings suggest that failures in causal understanding arise from the lack of structured causal representation, rather than insufficient exposure to causal examples during pretraining. 

---
# Responsible AI Technical Report 

**Authors**: Soonmin Bae, Wanjin Park, Jeongyeop Kim, Yunjin Park, Jungwon Yoon, Junhyung Moon, Myunggyo Oh, Wonhyuk Lee, Junseo Jang, Dongyoung Jung, Minwook Ju, Eunmi Kim, Sujin Kim, Youngchol Kim, Somin Lee, Wonyoung Lee, Minsung Noh, Hyoungjun Park, Eunyoung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2509.20057)  

**Abstract**: KT developed a Responsible AI (RAI) assessment methodology and risk mitigation technologies to ensure the safety and reliability of AI services. By analyzing the Basic Act on AI implementation and global AI governance trends, we established a unique approach for regulatory compliance and systematically identify and manage all potential risk factors from AI development to operation. We present a reliable assessment methodology that systematically verifies model safety and robustness based on KT's AI risk taxonomy tailored to the domestic environment. We also provide practical tools for managing and mitigating identified AI risks. With the release of this report, we also release proprietary Guardrail : SafetyGuard that blocks harmful responses from AI models in real-time, supporting the enhancement of safety in the domestic AI development ecosystem. We also believe these research outcomes provide valuable insights for organizations seeking to develop Responsible AI. 

---
# One Filters All: A Generalist Filter for State Estimation 

**Authors**: Shiqi Liu, Wenhan Cao, Chang Liu, Zeyu He, Tianyi Zhang, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.20051)  

**Abstract**: Estimating hidden states in dynamical systems, also known as optimal filtering, is a long-standing problem in various fields of science and engineering. In this paper, we introduce a general filtering framework, \textbf{LLM-Filter}, which leverages large language models (LLMs) for state estimation by embedding noisy observations with text prototypes. In various experiments for classical dynamical systems, we find that first, state estimation can significantly benefit from the reasoning knowledge embedded in pre-trained LLMs. By achieving proper modality alignment with the frozen LLM, LLM-Filter outperforms the state-of-the-art learning-based approaches. Second, we carefully design the prompt structure, System-as-Prompt (SaP), incorporating task instructions that enable the LLM to understand the estimation tasks. Guided by these prompts, LLM-Filter exhibits exceptional generalization, capable of performing filtering tasks accurately in changed or even unseen environments. We further observe a scaling-law behavior in LLM-Filter, where accuracy improves with larger model sizes and longer training times. These findings make LLM-Filter a promising foundation model of filtering. 

---
# Projective Kolmogorov Arnold Neural Networks (P-KANs): Entropy-Driven Functional Space Discovery for Interpretable Machine Learning 

**Authors**: Alastair Poole, Stig McArthur, Saravan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.20049)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) relocate learnable nonlinearities from nodes to edges, demonstrating remarkable capabilities in scientific machine learning and interpretable modeling. However, current KAN implementations suffer from fundamental inefficiencies due to redundancy in high-dimensional spline parameter spaces, where numerous distinct parameterisations yield functionally equivalent behaviors. This redundancy manifests as a "nuisance space" in the model's Jacobian, leading to susceptibility to overfitting and poor generalization. We introduce Projective Kolmogorov-Arnold Networks (P-KANs), a novel training framework that guides edge function discovery towards interpretable functional representations through entropy-minimisation techniques from signal analysis and sparse dictionary learning. Rather than constraining functions to predetermined spaces, our approach maintains spline space flexibility while introducing "gravitational" terms that encourage convergence towards optimal functional representations. Our key insight recognizes that optimal representations can be identified through entropy analysis of projection coefficients, compressing edge functions to lower-parameter projective spaces (Fourier, Chebyshev, Bessel). P-KANs demonstrate superior performance across multiple domains, achieving up to 80% parameter reduction while maintaining representational capacity, significantly improved robustness to noise compared to standard KANs, and successful application to industrial automated fiber placement prediction. Our approach enables automatic discovery of mixed functional representations where different edges converge to different optimal spaces, providing both compression benefits and enhanced interpretability for scientific machine learning applications. 

---
# Diffusion-Augmented Contrastive Learning: A Noise-Robust Encoder for Biosignal Representations 

**Authors**: Rami Zewail  

**Link**: [PDF](https://arxiv.org/pdf/2509.20048)  

**Abstract**: Learning robust representations for biosignals is often hampered by the challenge of designing effective data this http URL methods can fail to capture the complex variations inherent in physiological data. Within this context, we propose a novel hybrid framework, Diffusion-Augmented Contrastive Learning (DACL), that fuses concepts from diffusion models and supervised contrastive learning. The DACL framework operates on a latent space created by a lightweight Variational Autoencoder (VAE) trained on our novel Scattering Transformer (ST) features [12]. It utilizes the diffusion forward process as a principled data augmentation technique to generate multiple noisy views of these latent embeddings. A U-Net style encoder is then trained with a supervised contrastive objective to learn a representation that balances class discrimination with robustness to noise across various diffusion time steps. We evaluated this proof-of-concept method on the PhysioNet 2017 ECG dataset, achieving a competitive AUROC of 0.7815. This work establishes a new paradigm for representation learning by using the diffusion process itself to drive the contrastive objective, creating noise-invariant embeddings that demonstrate a strong foundation for class separability. 

---
# Tokenization and Representation Biases in Multilingual Models on Dialectal NLP Tasks 

**Authors**: Vani Kanjirangat, Tanja Samardžić, Ljiljana Dolamic, Fabio Rinaldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.20045)  

**Abstract**: Dialectal data are characterized by linguistic variation that appears small to humans but has a significant impact on the performance of models. This dialect gap has been related to various factors (e.g., data size, economic and social factors) whose impact, however, turns out to be inconsistent. In this work, we investigate factors impacting the model performance more directly: we correlate Tokenization Parity (TP) and Information Parity (IP), as measures of representational biases in pre-trained multilingual models, with the downstream performance. We compare state-of-the-art decoder-only LLMs with encoder-based models across three tasks: dialect classification, topic classification, and extractive question answering, controlling for varying scripts (Latin vs. non-Latin) and resource availability (high vs. low). Our analysis reveals that TP is a better predictor of the performance on tasks reliant on syntactic and morphological cues (e.g., extractive QA), while IP better predicts performance in semantic tasks (e.g., topic classification). Complementary analyses, including tokenizer behavior, vocabulary coverage, and qualitative insights, reveal that the language support claims of LLMs often might mask deeper mismatches at the script or token level. 

---
# Generative Adversarial Networks Applied for Privacy Preservation in Biometric-Based Authentication and Identification 

**Authors**: Lubos Mjachky, Ivan Homoliak  

**Link**: [PDF](https://arxiv.org/pdf/2509.20024)  

**Abstract**: Biometric-based authentication systems are getting broadly adopted in many areas. However, these systems do not allow participating users to influence the way their data is used. Furthermore, the data may leak and can be misused without the users' knowledge. In this paper, we propose a new authentication method that preserves the privacy of individuals and is based on a generative adversarial network (GAN). Concretely, we suggest using the GAN for translating images of faces to a visually private domain (e.g., flowers or shoes). Classifiers, which are used for authentication purposes, are then trained on the images from the visually private domain. Based on our experiments, the method is robust against attacks and still provides meaningful utility. 

---
# The Knowledge-Behaviour Disconnect in LLM-based Chatbots 

**Authors**: Jan Broersen  

**Link**: [PDF](https://arxiv.org/pdf/2509.20004)  

**Abstract**: Large language model-based artificial conversational agents (like ChatGPT) give answers to all kinds of questions, and often enough these answers are correct. Just on the basis of that capacity alone, we may attribute knowledge to them. But do these models use this knowledge as a basis for their own conversational behaviour? I argue this is not the case, and I will refer to this failure as a `disconnect'. I further argue this disconnect is fundamental in the sense that with more data and more training of the LLM on which a conversational chatbot is based, it will not disappear. The reason is, as I will claim, that the core technique used to train LLMs does not allow for the establishment of the connection we are after. The disconnect reflects a fundamental limitation on the capacities of LLMs, and explains the source of hallucinations. I will furthermore consider the ethical version of the disconnect (ethical conversational knowledge not being aligned with ethical conversational behaviour), since in this domain researchers have come up with several additional techniques to influence a chatbot's behaviour. I will discuss how these techniques do nothing to solve the disconnect and can make it worse. 

---
# Table Detection with Active Learning 

**Authors**: Somraj Gautam, Nachiketa Purohit, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2509.20003)  

**Abstract**: Efficient data annotation remains a critical challenge in machine learning, particularly for object detection tasks requiring extensive labeled data. Active learning (AL) has emerged as a promising solution to minimize annotation costs by selecting the most informative samples. While traditional AL approaches primarily rely on uncertainty-based selection, recent advances suggest that incorporating diversity-based strategies can enhance sampling efficiency in object detection tasks. Our approach ensures the selection of representative examples that improve model generalization. We evaluate our method on two benchmark datasets (TableBank-LaTeX, TableBank-Word) using state-of-the-art table detection architectures, CascadeTabNet and YOLOv9. Our results demonstrate that AL-based example selection significantly outperforms random sampling, reducing annotation effort given a limited budget while maintaining comparable performance to fully supervised models. Our method achieves higher mAP scores within the same annotation budget. 

---
# Choosing to Be Green: Advancing Green AI via Dynamic Model Selection 

**Authors**: Emilio Cruciani, Roberto Verdecchia  

**Link**: [PDF](https://arxiv.org/pdf/2509.19996)  

**Abstract**: Artificial Intelligence is increasingly pervasive across domains, with ever more complex models delivering impressive predictive performance. This fast technological advancement however comes at a concerning environmental cost, with state-of-the-art models - particularly deep neural networks and large language models - requiring substantial computational resources and energy. In this work, we present the intuition of Green AI dynamic model selection, an approach based on dynamic model selection that aims at reducing the environmental footprint of AI by selecting the most sustainable model while minimizing potential accuracy loss. Specifically, our approach takes into account the inference task, the environmental sustainability of available models, and accuracy requirements to dynamically choose the most suitable model. Our approach presents two different methods, namely Green AI dynamic model cascading and Green AI dynamic model routing. We demonstrate the effectiveness of our approach via a proof of concept empirical example based on a real-world dataset. Our results show that Green AI dynamic model selection can achieve substantial energy savings (up to ~25%) while substantially retaining the accuracy of the most energy greedy solution (up to ~95%). As conclusion, our preliminary findings highlight the potential that hybrid, adaptive model selection strategies withhold to mitigate the energy demands of modern AI systems without significantly compromising accuracy requirements. 

---
# SDE-DET: A Precision Network for Shatian Pomelo Detection in Complex Orchard Environments 

**Authors**: Yihao Hu, Pan Wang, Xiaodong Bai, Shijie Cai, Hang Wang, Huazhong Liu, Aiping Yang, Xiangxiang Li, Meiping Ding, Hongyan Liu, Jianguo Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19990)  

**Abstract**: Pomelo detection is an essential process for their localization, automated robotic harvesting, and maturity analysis. However, detecting Shatian pomelo in complex orchard environments poses significant challenges, including multi-scale issues, obstructions from trunks and leaves, small object detection, etc. To address these issues, this study constructs a custom dataset STP-AgriData and proposes the SDE-DET model for Shatian pomelo detection. SDE-DET first utilizes the Star Block to effectively acquire high-dimensional information without increasing the computational overhead. Furthermore, the presented model adopts Deformable Attention in its backbone, to enhance its ability to detect pomelos under occluded conditions. Finally, multiple Efficient Multi-Scale Attention mechanisms are integrated into our model to reduce the computational overhead and extract deep visual representations, thereby improving the capacity for small object detection. In the experiment, we compared SDE-DET with the Yolo series and other mainstream detection models in Shatian pomelo detection. The presented SDE-DET model achieved scores of 0.883, 0.771, 0.838, 0.497, and 0.823 in Precision, Recall, mAP@0.5, mAP@0.5:0.95 and F1-score, respectively. SDE-DET has achieved state-of-the-art performance on the STP-AgriData dataset. Experiments indicate that the SDE-DET provides a reliable method for Shatian pomelo detection, laying the foundation for the further development of automatic harvest robots. 

---
# An effective control of large systems of active particles: An application to evacuation problem 

**Authors**: Albina Klepach, Egor E. Nuzhin, Alexey A. Tsukanov, Nikolay V. Brilliantov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19972)  

**Abstract**: Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: this https URL. 

---
# 2025 Southeast Asia Eleven Nations Influence Index Report 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19953)  

**Abstract**: This study constructs a fully data-driven and reproducible Southeast Asia Influence Index (SAII v3) to reduce bias from expert scoring and subjective weighting while mapping hierarchical power structures across the eleven ASEAN nations. We aggregate authoritative open-source indicators across four dimensions (economic, military, diplomatic, socio-technological) and apply a three-tiered standardization chain quantile-Box-Cox-min-max to mitigate outliers and skewness. Weights are obtained through equal-weight integration of Entropy Weighting Method (EWM), CRITIC, and PCA. Robustness is assessed via Kendall's tau, +/-20% weight perturbation, and 10,000 bootstrap iterations, with additional checks including +/-10% dimensional sensitivity and V2-V3 bump chart comparisons. Results show integrated weights: Economy 35-40%, Military 20-25%, Diplomacy about 20%, Socio-Technology about 15%. The regional landscape exhibits a one-strong, two-medium, three-stable, and multiple-weak pattern: Indonesia, Singapore, and Malaysia lead, while Thailand, the Philippines, and Vietnam form a mid-tier competitive band. V2 and V3 rankings are highly consistent (Kendall's tau = 0.818), though small mid-tier reorderings appear (Thailand and the Philippines rise, Vietnam falls), indicating that v3 is more sensitive to structural equilibrium. ASEAN-11 average sensitivity highlights military and socio-technological dimensions as having the largest marginal effects (+/-0.002). In conclusion, SAII v3 delivers algorithmic weighting and auditable reproducibility, reveals multidimensional drivers of influence in Southeast Asia, and provides actionable quantitative evidence for resource allocation and policy prioritization by regional governments and external partners. 

---
# When Words Can't Capture It All: Towards Video-Based User Complaint Text Generation with Multimodal Video Complaint Dataset 

**Authors**: Sarmistha Das, R E Zera Marveen Lyngkhoi, Kirtan Jain, Vinayak Goyal, Sriparna Saha, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.19952)  

**Abstract**: While there exists a lot of work on explainable complaint mining, articulating user concerns through text or video remains a significant challenge, often leaving issues unresolved. Users frequently struggle to express their complaints clearly in text but can easily upload videos depicting product defects (e.g., vague text such as `worst product' paired with a 5-second video depicting a broken headphone with the right earcup). This paper formulates a new task in the field of complaint mining to aid the common users' need to write an expressive complaint, which is Complaint Description from Videos (CoD-V) (e.g., to help the above user articulate her complaint about the defective right earcup). To this end, we introduce ComVID, a video complaint dataset containing 1,175 complaint videos and the corresponding descriptions, also annotated with the emotional state of the complainer. Additionally, we present a new complaint retention (CR) evaluation metric that discriminates the proposed (CoD-V) task against standard video summary generation and description tasks. To strengthen this initiative, we introduce a multimodal Retrieval-Augmented Generation (RAG) embedded VideoLLaMA2-7b model, designed to generate complaints while accounting for the user's emotional state. We conduct a comprehensive evaluation of several Video Language Models on several tasks (pre-trained and fine-tuned versions) with a range of established evaluation metrics, including METEOR, perplexity, and the Coleman-Liau readability score, among others. Our study lays the foundation for a new research direction to provide a platform for users to express complaints through video. Dataset and resources are available at: this https URL. 

---
# A Set of Generalized Components to Achieve Effective Poison-only Clean-label Backdoor Attacks with Collaborative Sample Selection and Triggers 

**Authors**: Zhixiao Wu, Yao Lu, Jie Wen, Hao Sun, Qi Zhou, Guangming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19947)  

**Abstract**: Poison-only Clean-label Backdoor Attacks aim to covertly inject attacker-desired behavior into DNNs by merely poisoning the dataset without changing the labels. To effectively implant a backdoor, multiple \textbf{triggers} are proposed for various attack requirements of Attack Success Rate (ASR) and stealthiness. Additionally, sample selection enhances clean-label backdoor attacks' ASR by meticulously selecting ``hard'' samples instead of random samples to poison. Current methods 1) usually handle the sample selection and triggers in isolation, leading to severely limited improvements on both ASR and stealthiness. Consequently, attacks exhibit unsatisfactory performance on evaluation metrics when converted to PCBAs via a mere stacking of methods. Therefore, we seek to explore the bidirectional collaborative relations between the sample selection and triggers to address the above dilemma. 2) Since the strong specificity within triggers, the simple combination of sample selection and triggers fails to substantially enhance both evaluation metrics, with generalization preserved among various attacks. Therefore, we seek to propose a set of components to significantly improve both stealthiness and ASR based on the commonalities of attacks. Specifically, Component A ascertains two critical selection factors, and then makes them an appropriate combination based on the trigger scale to select more reasonable ``hard'' samples for improving ASR. Component B is proposed to select samples with similarities to relevant trigger implanted samples to promote stealthiness. Component C reassigns trigger poisoning intensity on RGB colors through distinct sensitivity of the human visual system to RGB for higher ASR, with stealthiness ensured by sample selection, including Component B. Furthermore, all components can be strategically integrated into diverse PCBAs. 

---
# Interpreting ResNet-based CLIP via Neuron-Attention Decomposition 

**Authors**: Edmund Bu, Yossi Gandelsman  

**Link**: [PDF](https://arxiv.org/pdf/2509.19943)  

**Abstract**: We present a novel technique for interpreting the neurons in CLIP-ResNet by decomposing their contributions to the output into individual computation paths. More specifically, we analyze all pairwise combinations of neurons and the following attention heads of CLIP's attention-pooling layer. We find that these neuron-head pairs can be approximated by a single direction in CLIP-ResNet's image-text embedding space. Leveraging this insight, we interpret each neuron-head pair by associating it with text. Additionally, we find that only a sparse set of the neuron-head pairs have a significant contribution to the output value, and that some neuron-head pairs, while polysemantic, represent sub-concepts of their corresponding neurons. We use these observations for two applications. First, we employ the pairs for training-free semantic segmentation, outperforming previous methods for CLIP-ResNet. Second, we utilize the contributions of neuron-head pairs to monitor dataset distribution shifts. Our results demonstrate that examining individual computation paths in neural networks uncovers interpretable units, and that such units can be utilized for downstream tasks. 

---
# CorIL: Towards Enriching Indian Language to Indian Language Parallel Corpora and Machine Translation Systems 

**Authors**: Soham Bhattacharjee, Mukund K Roy, Yathish Poojary, Bhargav Dave, Mihir Raj, Vandan Mujadia, Baban Gain, Pruthwik Mishra, Arafat Ahsan, Parameswari Krishnamurthy, Ashwath Rao, Gurpreet Singh Josan, Preeti Dubey, Aadil Amin Kak, Anna Rao Kulkarni, Narendra VG, Sunita Arora, Rakesh Balbantray, Prasenjit Majumdar, Karunesh K Arora, Asif Ekbal, Dipti Mishra Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.19941)  

**Abstract**: India's linguistic landscape is one of the most diverse in the world, comprising over 120 major languages and approximately 1,600 additional languages, with 22 officially recognized as scheduled languages in the Indian Constitution. Despite recent progress in multilingual neural machine translation (NMT), high-quality parallel corpora for Indian languages remain scarce, especially across varied domains. In this paper, we introduce a large-scale, high-quality annotated parallel corpus covering 11 of these languages : English, Telugu, Hindi, Punjabi, Odia, Kashmiri, Sindhi, Dogri, Kannada, Urdu, and Gujarati comprising a total of 772,000 bi-text sentence pairs. The dataset is carefully curated and systematically categorized into three key domains: Government, Health, and General, to enable domain-aware machine translation research and facilitate effective domain adaptation. To demonstrate the utility of CorIL and establish strong benchmarks for future research, we fine-tune and evaluate several state-of-the-art NMT models, including IndicTrans2, NLLB, and BhashaVerse. Our analysis reveals important performance trends and highlights the corpus's value in probing model capabilities. For instance, the results show distinct performance patterns based on language script, with massively multilingual models showing an advantage on Perso-Arabic scripts (Urdu, Sindhi) while other models excel on Indic scripts. This paper provides a detailed domain-wise performance analysis, offering insights into domain sensitivity and cross-script transfer learning. By publicly releasing CorIL, we aim to significantly improve the availability of high-quality training data for Indian languages and provide a valuable resource for the machine translation research community. 

---
# AJAHR: Amputated Joint Aware 3D Human Mesh Recovery 

**Authors**: Hyunjin Cho, Giyun Choi, Jongwon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19939)  

**Abstract**: Existing human mesh recovery methods assume a standard human body structure, overlooking diverse anatomical conditions such as limb loss. This assumption introduces bias when applied to individuals with amputations - a limitation further exacerbated by the scarcity of suitable datasets. To address this gap, we propose Amputated Joint Aware 3D Human Mesh Recovery (AJAHR), which is an adaptive pose estimation framework that improves mesh reconstruction for individuals with limb loss. Our model integrates a body-part amputation classifier, jointly trained with the mesh recovery network, to detect potential amputations. We also introduce Amputee 3D (A3D), which is a synthetic dataset offering a wide range of amputee poses for robust training. While maintaining competitive performance on non-amputees, our approach achieves state-of-the-art results for amputated individuals. Additional materials can be found at the project webpage. 

---
# TABFAIRGDT: A Fast Fair Tabular Data Generator using Autoregressive Decision Trees 

**Authors**: Emmanouil Panagiotou, Benoît Ronval, Arjun Roy, Ludwig Bothmann, Bernd Bischl, Siegfried Nijssen, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19927)  

**Abstract**: Ensuring fairness in machine learning remains a significant challenge, as models often inherit biases from their training data. Generative models have recently emerged as a promising approach to mitigate bias at the data level while preserving utility. However, many rely on deep architectures, despite evidence that simpler models can be highly effective for tabular data. In this work, we introduce TABFAIRGDT, a novel method for generating fair synthetic tabular data using autoregressive decision trees. To enforce fairness, we propose a soft leaf resampling technique that adjusts decision tree outputs to reduce bias while preserving predictive performance. Our approach is non-parametric, effectively capturing complex relationships between mixed feature types, without relying on assumptions about the underlying data distributions. We evaluate TABFAIRGDT on benchmark fairness datasets and demonstrate that it outperforms state-of-the-art (SOTA) deep generative models, achieving better fairness-utility trade-off for downstream tasks, as well as higher synthetic data quality. Moreover, our method is lightweight, highly efficient, and CPU-compatible, requiring no data pre-processing. Remarkably, TABFAIRGDT achieves a 72% average speedup over the fastest SOTA baseline across various dataset sizes, and can generate fair synthetic data for medium-sized datasets (10 features, 10K samples) in just one second on a standard CPU, making it an ideal solution for real-world fairness-sensitive applications. 

---
# Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches 

**Authors**: Remo Sasso, Michelangelo Conserva, Dominik Jeurissen, Paulo Rauber  

**Link**: [PDF](https://arxiv.org/pdf/2509.19924)  

**Abstract**: Exploration in reinforcement learning (RL) remains challenging, particularly in sparse-reward settings. While foundation models possess strong semantic priors, their capabilities as zero-shot exploration agents in classic RL benchmarks are not well understood. We benchmark LLMs and VLMs on multi-armed bandits, Gridworlds, and sparse-reward Atari to test zero-shot exploration. Our investigation reveals a key limitation: while VLMs can infer high-level objectives from visual input, they consistently fail at precise low-level control: the "knowing-doing gap". To analyze a potential bridge for this gap, we investigate a simple on-policy hybrid framework in a controlled, best-case scenario. Our results in this idealized setting show that VLM guidance can significantly improve early-stage sample efficiency, providing a clear analysis of the potential and constraints of using foundation models to guide exploration rather than for end-to-end control. 

---
# Towards Self-Supervised Foundation Models for Critical Care Time Series 

**Authors**: Katja Naasunnguaq Jagd, Rachael DeVries, Ole Winther  

**Link**: [PDF](https://arxiv.org/pdf/2509.19885)  

**Abstract**: Domain-specific foundation models for healthcare have expanded rapidly in recent years, yet foundation models for critical care time series remain relatively underexplored due to the limited size and availability of datasets. In this work, we introduce an early-stage pre-trained foundation model for critical care time-series based on the Bi-Axial Transformer (BAT), trained on pooled electronic health record datasets. We demonstrate effective transfer learning by fine-tuning the model on a dataset distinct from the training sources for mortality prediction, where it outperforms supervised baselines, particularly for small datasets ($<5,000$). These contributions highlight the potential of self-supervised foundation models for critical care times series to support generalizable and robust clinical applications in resource-limited settings. 

---
# CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance 

**Authors**: Junchuan Zhao, Wei Zeng, Tianle Lyu, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19883)  

**Abstract**: Singing Voice Synthesis (SVS) aims to generate expressive vocal performances from structured musical inputs such as lyrics and pitch sequences. While recent progress in discrete codec-based speech synthesis has enabled zero-shot generation via in-context learning, directly extending these techniques to SVS remains non-trivial due to the requirement for precise melody control. In particular, prompt-based generation often introduces prosody leakage, where pitch information is inadvertently entangled within the timbre prompt, compromising controllability. We present CoMelSinger, a zero-shot SVS framework that enables structured and disentangled melody control within a discrete codec modeling paradigm. Built on the non-autoregressive MaskGCT architecture, CoMelSinger replaces conventional text inputs with lyric and pitch tokens, preserving in-context generalization while enhancing melody conditioning. To suppress prosody leakage, we propose a coarse-to-fine contrastive learning strategy that explicitly regularizes pitch redundancy between the acoustic prompt and melody input. Furthermore, we incorporate a lightweight encoder-only Singing Voice Transcription (SVT) module to align acoustic tokens with pitch and duration, offering fine-grained frame-level supervision. Experimental results demonstrate that CoMelSinger achieves notable improvements in pitch accuracy, timbre consistency, and zero-shot transferability over competitive baselines. 

---
# Do Before You Judge: Self-Reference as a Pathway to Better LLM Evaluation 

**Authors**: Wei-Hsiang Lin, Sheng-Lun Wei, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19880)  

**Abstract**: LLM-as-Judge frameworks are increasingly popular for AI evaluation, yet research findings on the relationship between models' generation and judgment abilities remain inconsistent. We investigate this relationship through systematic dataset- and instance-level analyses across 11 models and 21 diverse tasks. Despite both capabilities relying on the same underlying knowledge, our analyses reveal they are only weakly correlated, primarily due to LLMs' sensitivity to the responses being judged. To address this, we propose a self-reference-guided evaluation strategy that leverages a model's own answers as references. This approach significantly strengthens the correlation between generation and judgment abilities, offering a practical path to align these skills and providing a reliable proxy for model selection in evaluation tasks. 

---
# Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials 

**Authors**: Shi Yin, Zujian Dai, Xinyang Pan, Lixin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19877)  

**Abstract**: Deep learning methods for electronic-structure Hamiltonian prediction has offered significant computational efficiency advantages over traditional DFT methods, yet the diversity of atomic types, structural patterns, and the high-dimensional complexity of Hamiltonians pose substantial challenges to the generalization performance. In this work, we contribute on both the methodology and dataset sides to advance universal deep learning paradigm for Hamiltonian prediction. On the method side, we propose NextHAM, a neural E(3)-symmetry and expressive correction method for efficient and generalizable materials electronic-structure Hamiltonian prediction. First, we introduce the zeroth-step Hamiltonians, which can be efficiently constructed by the initial charge density of DFT, as informative descriptors of neural regression model in the input level and initial estimates of the target Hamiltonian in the output level, so that the regression model directly predicts the correction terms to the target ground truths, thereby significantly simplifying the input-output mapping for learning. Second, we present a neural Transformer architecture with strict E(3)-Symmetry and high non-linear expressiveness for Hamiltonian prediction. Third, we propose a novel training objective to ensure the accuracy performance of Hamiltonians in both real space and reciprocal space, preventing error amplification and the occurrence of "ghost states" caused by the large condition number of the overlap matrix. On the dataset side, we curate a high-quality broad-coverage large benchmark, namely Materials-HAM-SOC, comprising 17,000 material structures spanning 68 elements from six rows of the periodic table and explicitly incorporating SOC effects. Experimental results on Materials-HAM-SOC demonstrate that NextHAM achieves excellent accuracy and efficiency in predicting Hamiltonians and band structures. 

---
# Adaptive Guidance Semantically Enhanced via Multimodal LLM for Edge-Cloud Object Detection 

**Authors**: Yunqing Hu, Zheming Yang, Chang Zhao, Wen Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.19875)  

**Abstract**: Traditional object detection methods face performance degradation challenges in complex scenarios such as low-light conditions and heavy occlusions due to a lack of high-level semantic understanding. To address this, this paper proposes an adaptive guidance-based semantic enhancement edge-cloud collaborative object detection method leveraging Multimodal Large Language Models (MLLM), achieving an effective balance between accuracy and efficiency. Specifically, the method first employs instruction fine-tuning to enable the MLLM to generate structured scene descriptions. It then designs an adaptive mapping mechanism that dynamically converts semantic information into parameter adjustment signals for edge detectors, achieving real-time semantic enhancement. Within an edge-cloud collaborative inference framework, the system automatically selects between invoking cloud-based semantic guidance or directly outputting edge detection results based on confidence scores. Experiments demonstrate that the proposed method effectively enhances detection accuracy and efficiency in complex scenes. Specifically, it can reduce latency by over 79% and computational cost by 70% in low-light and highly occluded scenes while maintaining accuracy. 

---
# CollaPipe: Adaptive Segment-Optimized Pipeline Parallelism for Collaborative LLM Training in Heterogeneous Edge Networks 

**Authors**: Jiewei Chen, Xiumei Deng, Zehui Xiong, Shaoyong Guo, Xuesong Qiu, Ping Wang, Dusit Niyato  

**Link**: [PDF](https://arxiv.org/pdf/2509.19855)  

**Abstract**: The increasing demand for intelligent mobile applications has made multi-agent collaboration with Transformer-based large language models (LLMs) essential in mobile edge computing (MEC) networks. However, training LLMs in such environments remains challenging due to heavy computation, high end-to-end latency, and limited model generalization. We introduce CollaPipe, a hybrid distributed learning framework that integrates collaborative pipeline parallelism with federated aggregation to support self-evolving intelligent networks. In CollaPipe, the encoder part is adaptively partitioned into variable-sized segments and deployed across mobile devices for pipeline-parallel training, while the decoder is deployed on edge servers to handle generative tasks. Then we perform global model update via federated aggregation. To enhance training efficiency, we formulate a joint optimization problem that adaptively allocates model segments, micro-batches, bandwidth, and transmission power. We derive and use a closed-form convergence bound to design an Dynamic Segment Scheduling and Resource Allocation (DSSDA) algorithm based on Lyapunov optimization, ensuring system stability under long-term constraints. Extensive experiments on downstream tasks with Transformer and BERT models show that CollaPipe improves computation efficiency by up to 15.09%, reduces end-to-end latency by at least 48.98%, and cuts single device memory usage by more than half, enabling online learning in heterogeneous and dynamic communication environments. 

---
# Eliminating stability hallucinations in llm-based tts models via attention guidance 

**Authors**: ShiMing Wang, ZhiHao Du, Yang Xiang, TianYu Zhao, Han Zhao, Qian Chen, XianGang Li, HanJie Guo, ZhenHua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.19852)  

**Abstract**: This paper focuses on resolving stability hallucinations (e.g., repetitive or omitted speech) in LLM-based Text-to-Speech (TTS) models by improving and leveraging the attention mechanism. First, we analyzed the alignment mechanism between text tokens and speech tokens in LLMs. We then proposed a metric termed the Optimal Alignment Score (OAS), which employs the Viterbi algorithm to evaluate text-speech alignment quality. Subsequently, OAS was integrated into the training of CosyVoice2 to assist LLMs in learning continuous, stable alignment. Additionally, the pre-trained attention value is employed to guide the training of the student CosyVoice2 via chain-of-thought (CoT), which further reduces stability hallucinations in synthesized speech. Experiments on the Seed-TTS-Eval and CV3-Eval test sets demonstrate that the proposed methods can effectively reduce the stability hallucinations of CosyVoice2 without introducing additional negative effects. The appendix is available at this https URL. 

---
# Analyzing Generalization in Pre-Trained Symbolic Regression 

**Authors**: Henrik Voigt, Paul Kahlmeyer, Kai Lawonn, Michael Habeck, Joachim Giesen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19849)  

**Abstract**: Symbolic regression algorithms search a space of mathematical expressions for formulas that explain given data. Transformer-based models have emerged as a promising, scalable approach shifting the expensive combinatorial search to a large-scale pre-training phase. However, the success of these models is critically dependent on their pre-training data. Their ability to generalize to problems outside of this pre-training distribution remains largely unexplored. In this work, we conduct a systematic empirical study to evaluate the generalization capabilities of pre-trained, transformer-based symbolic regression. We rigorously test performance both within the pre-training distribution and on a series of out-of-distribution challenges for several state of the art approaches. Our findings reveal a significant dichotomy: while pre-trained models perform well in-distribution, the performance consistently degrades in out-of-distribution scenarios. We conclude that this generalization gap is a critical barrier for practitioners, as it severely limits the practical use of pre-trained approaches for real-world applications. 

---
# TianHui: A Domain-Specific Large Language Model for Diverse Traditional Chinese Medicine Scenarios 

**Authors**: Ji Yin, Menglan He, Yujie Zhang, Linshuai Zhang, Tingting Ma, Ce Tian, Jie Wu, Lin Xu, Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19834)  

**Abstract**: Domain-specific LLMs in TCM face limitations in research settings due to constrained adaptability, insufficient evaluation datasets, and limited computational resources. This study presents TianHui, a specialized TCM LLM built through contextual data integration and domain knowledge fusion. We constructed a large-scale TCM corpus (0.97GB unsupervised data + 611,312 QA pairs) and employed a two-stage training strategy with QLoRA, DeepSpeed Stage 2, and Flash Attention 2. Evaluation on 12 benchmarks showed TianHui ranked top-three in all metrics for six datasets (APQ, TCMCD, HFR, HCCA, DHPE, TLAW) and achieved top results in the other six (TCMEE, APR, GCPMI, TCMKQA, TCMRC, ADTG). Optimal configuration was identified as LoRA rank=128, alpha=256, epoch=4, dropout=0.2, max length=2048. TianHui enables systematic preservation and scalable application of TCM knowledge. All resources are open-sourced. 

---
# Polarity Detection of Sustainable Detection Goals in News Text 

**Authors**: Andrea Cadeddua, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19833)  

**Abstract**: The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems. 

---
# On the Rate of Convergence of Kolmogorov-Arnold Network Regression Estimators 

**Authors**: Wei Liu, Eleni Chatzi, Zhilu Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19830)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) offer a structured and interpretable framework for multivariate function approximation by composing univariate transformations through additive or multiplicative aggregation. This paper establishes theoretical convergence guarantees for KANs when the univariate components are represented by B-splines. We prove that both additive and hybrid additive-multiplicative KANs attain the minimax-optimal convergence rate $O(n^{-2r/(2r+1)})$ for functions in Sobolev spaces of smoothness $r$. We further derive guidelines for selecting the optimal number of knots in the B-splines. The theory is supported by simulation studies that confirm the predicted convergence rates. These results provide a theoretical foundation for using KANs in nonparametric regression and highlight their potential as a structured alternative to existing methods. 

---
# Causal Inference under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects 

**Authors**: Kohsuke Kubota, Shonosuke Sugasawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.19814)  

**Abstract**: Many marketing applications, including credit card incentive programs, offer rewards to customers who exceed specific spending thresholds to encourage increased consumption. Quantifying the causal effect of these thresholds on customers is crucial for effective marketing strategy design. Although regression discontinuity design is a standard method for such causal inference tasks, its assumptions can be violated when customers, aware of the thresholds, strategically manipulate their spending to qualify for the rewards. To address this issue, we propose a novel framework for estimating the causal effect under threshold manipulation. The main idea is to model the observed spending distribution as a mixture of two distributions: one representing customers strategically affected by the threshold, and the other representing those unaffected. To fit the mixture model, we adopt a two-step Bayesian approach consisting of modeling non-bunching customers and fitting a mixture model to a sample around the threshold. We show posterior contraction of the resulting posterior distribution of the causal effect under large samples. Furthermore, we extend this framework to a hierarchical Bayesian setting to estimate heterogeneous causal effects across customer subgroups, allowing for stable inference even with small subgroup sample sizes. We demonstrate the effectiveness of our proposed methods through simulation studies and illustrate their practical implications using a real-world marketing dataset. 

---
# RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving 

**Authors**: Carlo Bosio, Greg Woelki, Noureldin Hendy, Nicholas Roy, Byungsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.19789)  

**Abstract**: Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model. 

---
# bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs 

**Authors**: Wence Ji, Jiancan Wu, Aiying Li, Shuyi Zhang, Junkang Wu, An Zhang, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2509.19775)  

**Abstract**: With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers--such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)--each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks. 

---
# PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection 

**Authors**: Xiaocheng Fang, Jiarui Jin, Haoyu Wang, Che Liu, Jieyi Cai, Guangkun Nie, Jun Li, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19774)  

**Abstract**: In clinical practice, electrocardiography (ECG) remains the gold standard for cardiac monitoring, providing crucial insights for diagnosing a wide range of cardiovascular diseases (CVDs). However, its reliance on specialized equipment and trained personnel limits feasibility for continuous routine monitoring. Photoplethysmography (PPG) offers accessible, continuous monitoring but lacks definitive electrophysiological information, preventing conclusive diagnosis. Generative models present a promising approach to translate PPG into clinically valuable ECG signals, yet current methods face substantial challenges, including the misalignment of physiological semantics in generative models and the complexity of modeling in high-dimensional signals. To this end, we propose PPGFlowECG, a two-stage framework that aligns PPG and ECG in a shared latent space via the CardioAlign Encoder and employs latent rectified flow to generate ECGs with high fidelity and interpretability. To the best of our knowledge, this is the first study to experiment on MCMED, a newly released clinical-grade dataset comprising over 10 million paired PPG-ECG samples from more than 118,000 emergency department visits with expert-labeled cardiovascular disease annotations. Results demonstrate the effectiveness of our method for PPG-to-ECG translation and cardiovascular disease detection. Moreover, cardiologist-led evaluations confirm that the synthesized ECGs achieve high fidelity and improve diagnostic reliability, underscoring our method's potential for real-world cardiovascular screening. 

---
# Sobolev acceleration for neural networks 

**Authors**: Jong Kwon Oh, Hanbaek Lyu, Hwijae Son  

**Link**: [PDF](https://arxiv.org/pdf/2509.19773)  

**Abstract**: Sobolev training, which integrates target derivatives into the loss functions, has been shown to accelerate convergence and improve generalization compared to conventional $L^2$ training. However, the underlying mechanisms of this training method remain only partially understood. In this work, we present the first rigorous theoretical framework proving that Sobolev training accelerates the convergence of Rectified Linear Unit (ReLU) networks. Under a student-teacher framework with Gaussian inputs and shallow architectures, we derive exact formulas for population gradients and Hessians, and quantify the improvements in conditioning of the loss landscape and gradient-flow convergence rates. Extensive numerical experiments validate our theoretical findings and show that the benefits of Sobolev training extend to modern deep learning tasks. 

---
# Frictional Q-Learning 

**Authors**: Hyunwoo Kim, Hyo Kyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19771)  

**Abstract**: We draw an analogy between static friction in classical mechanics and extrapolation error in off-policy RL, and use it to formulate a constraint that prevents the policy from drifting toward unsupported actions. In this study, we present Frictional Q-learning, a deep reinforcement learning algorithm for continuous control, which extends batch-constrained reinforcement learning. Our algorithm constrains the agent's action space to encourage behavior similar to that in the replay buffer, while maintaining a distance from the manifold of the orthonormal action space. The constraint preserves the simplicity of batch-constrained, and provides an intuitive physical interpretation of extrapolation error. Empirically, we further demonstrate that our algorithm is robustly trained and achieves competitive performance across standard continuous control benchmarks. 

---
# FusedANN: Convexified Hybrid ANN via Attribute-Vector Fusion 

**Authors**: Alireza Heidari, Wei Zhang, Ying Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19767)  

**Abstract**: Vector search powers transformers technology, but real-world use demands hybrid queries that combine vector similarity with attribute filters (e.g., "top document in category X, from 2023"). Current solutions trade off recall, speed, and flexibility, relying on fragile index hacks that don't scale. We introduce FusedANN (Fused Attribute-Vector Nearest Neighbor), a geometric framework that elevates filtering to ANN optimization constraints and introduces a convex fused space via a Lagrangian-like relaxation. Our method jointly embeds attributes and vectors through transformer-based convexification, turning hard filters into continuous, weighted penalties that preserve top-k semantics while enabling efficient approximate search. We prove that FusedANN reduces to exact filtering under high selectivity, gracefully relaxes to semantically nearest attributes when exact matches are insufficient, and preserves downstream ANN alpha-approximation guarantees. Empirically, FusedANN improves query throughput by eliminating brittle filtering stages, achieving superior recall-latency tradeoffs on standard hybrid benchmarks without specialized index hacks, delivering up to 3 times higher throughput and better recall than state-of-the-art hybrid and graph-based systems. Theoretically, we provide explicit error bounds and parameter selection rules that make FusedANN practical for production. This establishes a principled, scalable, and verifiable bridge between symbolic constraints and vector similarity, unlocking a new generation of filtered retrieval systems for large, hybrid, and dynamic NLP/ML workloads. 

---
# Dynamicasome: a molecular dynamics-guided and AI-driven pathogenicity prediction catalogue for all genetic mutations 

**Authors**: Naeyma N Islam, Mathew A Coban, Jessica M Fuller, Caleb Weber, Rohit Chitale, Benjamin Jussila, Trisha J. Brock, Cui Tao, Thomas R Caulfield  

**Link**: [PDF](https://arxiv.org/pdf/2509.19766)  

**Abstract**: Advances in genomic medicine accelerate the identi cation of mutations in disease-associated genes, but the pathogenicity of many mutations remains unknown, hindering their use in diagnostics and clinical decision-making. Predictive AI models are generated to combat this issue, but current tools display low accuracy when tested against functionally validated datasets. We show that integrating detailed conformational data extracted from molecular dynamics simulations (MDS) into advanced AI-based models increases their predictive power. We carry out an exhaustive mutational analysis of the disease gene PMM2 and subject structural models of each variant to MDS. AI models trained on this dataset outperform existing tools when predicting the known pathogenicity of mutations. Our best performing model, a neuronal networks model, also predicts the pathogenicity of several PMM2 mutations currently considered of unknown signi cance. We believe this model helps alleviate the burden of unknown variants in genomic medicine. 

---
# ARCADE: A Real-Time Data System for Hybrid and Continuous Query Processing across Diverse Data Modalities 

**Authors**: Jingyi Yang, Songsong Mo, Jiachen Shi, Zihao Yu, Kunhao Shi, Xuchen Ding, Gao Cong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19757)  

**Abstract**: The explosive growth of multimodal data - spanning text, image, video, spatial, and relational modalities, coupled with the need for real-time semantic search and retrieval over these data - has outpaced the capabilities of existing multimodal and real-time database systems, which either lack efficient ingestion and continuous query capability, or fall short in supporting expressive hybrid analytics. We introduce ARCADE, a real-time data system that efficiently supports high-throughput ingestion and expressive hybrid and continuous query processing across diverse data types. ARCADE introduces unified disk-based secondary index on LSM-based storage for vector, spatial, and text data modalities, a comprehensive cost-based query optimizer for hybrid queries, and an incremental materialized view framework for efficient continuous queries. Built on open-source RocksDB storage and MySQL query engine, ARCADE outperforms leading multimodal data systems by up to 7.4x on read-heavy and 1.4x on write-heavy workloads. 

---
# ExpFace: Exponential Angular Margin Loss for Deep Face Recognition 

**Authors**: Jinhui Zheng, Xueyuan Gong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19753)  

**Abstract**: Face recognition is an open-set problem requiring high discriminative power to ensure that intra-class distances remain smaller than inter-class distances. Margin-based softmax losses, such as SphereFace, CosFace, and ArcFace, have been widely adopted to enhance intra-class compactness and inter-class separability, yet they overlook the impact of noisy samples. By examining the distribution of samples in the angular space, we observe that clean samples predominantly cluster in the center region, whereas noisy samples tend to shift toward the peripheral region. Motivated by this observation, we propose the Exponential Angular Margin Loss (ExpFace), which introduces an angular exponential term as the margin. This design applies a larger penalty in the center region and a smaller penalty in the peripheral region within the angular space, thereby emphasizing clean samples while suppressing noisy samples. We present a unified analysis of ExpFace and classical margin-based softmax losses in terms of margin embedding forms, similarity curves, and gradient curves, showing that ExpFace not only avoids the training instability of SphereFace and the non-monotonicity of ArcFace, but also exhibits a similarity curve that applies penalties in the same manner as the decision boundary in the angular space. Extensive experiments demonstrate that ExpFace achieves state-of-the-art performance. To facilitate future research, we have released the source code at: this https URL. 

---
# Cuffless Blood Pressure Prediction from Speech Sentences using Deep Learning Methods 

**Authors**: Kainat  

**Link**: [PDF](https://arxiv.org/pdf/2509.19750)  

**Abstract**: This research presents a novel method for noninvasive arterial blood pressure ABP prediction using speech signals employing a BERT based regression model Arterial blood pressure is a vital indicator of cardiovascular health and accurate monitoring is essential in preventing hypertension related complications Traditional cuff based methods often yield inconsistent results due to factors like whitecoat and masked hypertension Our approach leverages the acoustic characteristics of speech capturing voice features to establish correlations with blood pressure levels Utilizing advanced deep learning techniques we analyze speech signals to extract relevant patterns enabling real time monitoring without the discomfort of conventional methods In our study we employed a dataset comprising recordings from 95 participants ensuring diverse representation The BERT model was fine tuned on extracted features from speech leading to impressive performance metrics achieving a mean absolute error MAE of 136 mmHg for systolic blood pressure SBP and 124 mmHg for diastolic blood pressure DBP with R scores of 099 and 094 respectively These results indicate the models robustness in accurately predicting blood pressure levels Furthermore the training and validation loss analysis demonstrates effective learning and minimal overfitting Our findings suggest that integrating deep learning with speech analysis presents a viable alternative for blood pressure monitoring paving the way for improved applications in telemedicine and remote health monitoring By providing a user friendly and accurate method for blood pressure assessment this research has significant implications for enhancing patient care and proactive management of cardiovascular health 

---
# HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST 

**Authors**: Shuyu Zhang, Yifan Wei, Xinru Wang, Yanmin Zhu, Yangfan He, Yixuan Weng, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19742)  

**Abstract**: Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at this https URL. 

---
# SMILES-Inspired Transfer Learning for Quantum Operators in Generative Quantum Eigensolver 

**Authors**: Zhi Yin, Xiaoran Li, Shengyu Zhang, Xin Li, Xiaojin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19715)  

**Abstract**: Given the inherent limitations of traditional Variational Quantum Eigensolver(VQE) algorithms, the integration of deep generative models into hybrid quantum-classical frameworks, specifically the Generative Quantum Eigensolver(GQE), represents a promising innovative approach. However, taking the Unitary Coupled Cluster with Singles and Doubles(UCCSD) ansatz which is widely used in quantum chemistry as an example, different molecular systems require constructions of distinct quantum operators. Considering the similarity of different molecules, the construction of quantum operators utilizing the similarity can reduce the computational cost significantly. Inspired by the SMILES representation method in computational chemistry, we developed a text-based representation approach for UCCSD quantum operators by leveraging the inherent representational similarities between different molecular systems. This framework explores text pattern similarities in quantum operators and employs text similarity metrics to establish a transfer learning framework. Our approach with a naive baseline setting demonstrates knowledge transfer between different molecular systems for ground-state energy calculations within the GQE paradigm. This discovery offers significant benefits for hybrid quantum-classical computation of molecular ground-state energies, substantially reducing computational resource requirements. 

---
# Intuition to Evidence: Measuring AI's True Impact on Developer Productivity 

**Authors**: Anand Kumar, Vishal Khare, Deepak Sharma, Satyam Kumar, Vijay Saini, Anshul Yadav, Sachendra Jain, Ankit Rana, Pratham Verma, Vaibhav Meena, Avinash Edubilli  

**Link**: [PDF](https://arxiv.org/pdf/2509.19708)  

**Abstract**: We present a comprehensive real-world evaluation of AI-assisted software development tools deployed at enterprise scale. Over one year, 300 engineers across multiple teams integrated an in-house AI platform (DeputyDev) that combines code generation and automated review capabilities into their daily workflows. Through rigorous cohort analysis, our study demonstrates statistically significant productivity improvements, including an overall 31.8% reduction in PR review cycle time.
Developer adoption was strong, with 85% satisfaction for code review features and 93% expressing a desire to continue using the platform. Adoption patterns showed systematic scaling from 4% engagement in month 1 to 83% peak usage by month 6, stabilizing at 60% active engagement. Top adopters achieved a 61% increase in code volume pushed to production, contributing to approximately 30 to 40% of code shipped to production through this tool, accounting for an overall 28% increase in code shipment volume.
Unlike controlled benchmark evaluations, our longitudinal analysis provides empirical evidence from production environments, revealing both the transformative potential and practical deployment challenges of integrating AI into enterprise software development workflows. 

---
# Causal Machine Learning for Surgical Interventions 

**Authors**: J. Ben Tamo, Nishant S. Chouhan, Micky C. Nnamdi, Yining Yuan, Shreya S. Chivilkar, Wenqi Shi, Steven W. Hwang, B. Randall Brenn, May D. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19705)  

**Abstract**: Surgical decision-making is complex and requires understanding causal relationships between patient characteristics, interventions, and outcomes. In high-stakes settings like spinal fusion or scoliosis correction, accurate estimation of individualized treatment effects (ITEs) remains limited due to the reliance on traditional statistical methods that struggle with complex, heterogeneous data. In this study, we develop a multi-task meta-learning framework, X-MultiTask, for ITE estimation that models each surgical decision (e.g., anterior vs. posterior approach, surgery vs. no surgery) as a distinct task while learning shared representations across tasks. To strengthen causal validity, we incorporate the inverse probability weighting (IPW) into the training objective. We evaluate our approach on two datasets: (1) a public spinal fusion dataset (1,017 patients) to assess the effect of anterior vs. posterior approaches on complication severity; and (2) a private AIS dataset (368 patients) to analyze the impact of posterior spinal fusion (PSF) vs. non-surgical management on patient-reported outcomes (PROs). Our model achieves the highest average AUC (0.84) in the anterior group and maintains competitive performance in the posterior group (0.77). It outperforms baselines in treatment effect estimation with the lowest overall $\epsilon_{\text{NN-PEHE}}$ (0.2778) and $\epsilon_{\text{ATE}}$ (0.0763). Similarly, when predicting PROs in AIS, X-MultiTask consistently shows superior performance across all domains, with $\epsilon_{\text{NN-PEHE}}$ = 0.2551 and $\epsilon_{\text{ATE}}$ = 0.0902. By providing robust, patient-specific causal estimates, X-MultiTask offers a powerful tool to advance personalized surgical care and improve patient outcomes. The code is available at this https URL. 

---
# Linear Transformers Implicitly Discover Unified Numerical Algorithms 

**Authors**: Patrick Lutz, Aditya Gangrade, Hadi Daneshmand, Venkatesh Saligrama  

**Link**: [PDF](https://arxiv.org/pdf/2509.19702)  

**Abstract**: We train a linear attention transformer on millions of masked-block matrix completion tasks: each prompt is masked low-rank matrix whose missing block may be (i) a scalar prediction target or (ii) an unseen kernel slice of Nyström extrapolation. The model sees only input-output pairs and a mean-squared loss; it is given no normal equations, no handcrafted iterations, and no hint that the tasks are related. Surprisingly, after training, algebraic unrolling reveals the same parameter-free update rule across three distinct computational regimes (full visibility, rank-limited updates, and distributed computation). We prove that this rule achieves second-order convergence on full-batch problems, cuts distributed iteration complexity, and remains accurate with rank-limited attention. Thus, a transformer trained solely to patch missing blocks implicitly discovers a unified, resource-adaptive iterative solver spanning prediction, estimation, and Nyström extrapolation, highlighting a powerful capability of in-context learning. 

---
# A Unified Noise-Curvature View of Loss of Trainability 

**Authors**: Gunbir Singh Baveja, Mark Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2509.19698)  

**Abstract**: Loss of trainability (LoT) in continual learning occurs when gradient steps no longer yield improvement as tasks evolve, so accuracy stalls or degrades despite adequate capacity and supervision. We analyze LoT incurred with Adam through an optimization lens and find that single indicators such as Hessian rank, sharpness level, weight or gradient norms, gradient-to-parameter ratios, and unit-sign entropy are not reliable predictors. Instead we introduce two complementary criteria: a batch-size-aware gradient-noise bound and a curvature volatility-controlled bound that combine into a per-layer predictive threshold that anticipates trainability behavior. Using this threshold, we build a simple per-layer scheduler that keeps each layers effective step below a safe limit, stabilizing training and improving accuracy across concatenated ReLU (CReLU), Wasserstein regularization, and L2 weight decay, with learned learning-rate trajectories that mirror canonical decay. 

---
# Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks 

**Authors**: Noah Geiger, Tamim Asfour, Neville Hogan, Johannes Lachner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19696)  

**Abstract**: Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation. 

---
# DyBBT: Dynamic Balance via Bandit inspired Targeting for Dialog Policy with Cognitive Dual-Systems 

**Authors**: Shuyu Zhang, Yifan Wei, Jialuo Yuan, Xinru Wang, Yanmin Zhu, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19695)  

**Abstract**: Task oriented dialog systems often rely on static exploration strategies that do not adapt to dynamic dialog contexts, leading to inefficient exploration and suboptimal performance. We propose DyBBT, a novel dialog policy learning framework that formalizes the exploration challenge through a structured cognitive state space capturing dialog progression, user uncertainty, and slot dependency. DyBBT proposes a bandit inspired meta-controller that dynamically switches between a fast intuitive inference (System 1) and a slow deliberative reasoner (System 2) based on real-time cognitive states and visitation counts. Extensive experiments on single- and multi-domain benchmarks show that DyBBT achieves state-of-the-art performance in success rate, efficiency, and generalization, with human evaluations confirming its decisions are well aligned with expert judgment. Code is available at this https URL. 

---
# PolicyPad: Collaborative Prototyping of LLM Policies 

**Authors**: K. J. Kevin Feng, Tzu-Sheng Kuo, Quan Ze, Chen, Inyoung Cheong, Kenneth Holstein, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19680)  

**Abstract**: As LLMs gain adoption in high-stakes domains like mental health, domain experts are increasingly consulted to provide input into policies governing their behavior. From an observation of 19 policymaking workshops with 9 experts over 15 weeks, we identified opportunities to better support rapid experimentation, feedback, and iteration for collaborative policy design processes. We present PolicyPad, an interactive system that facilitates the emerging practice of LLM policy prototyping by drawing from established UX prototyping practices, including heuristic evaluation and storyboarding. Using PolicyPad, policy designers can collaborate on drafting a policy in real time while independently testing policy-informed model behavior with usage scenarios. We evaluate PolicyPad through workshops with 8 groups of 22 domain experts in mental health and law, finding that PolicyPad enhanced collaborative dynamics during policy design, enabled tight feedback loops, and led to novel policy contributions. Overall, our work paves participatory paths for advancing AI alignment and safety. 

---
# Thinking While Listening: Simple Test Time Scaling For Audio Classification 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2509.19676)  

**Abstract**: We propose a framework that enables neural models to "think while listening" to everyday sounds, thereby enhancing audio classification performance. Motivated by recent advances in the reasoning capabilities of large language models, we address two central questions: (i) how can thinking be incorporated into existing audio classification pipelines to enable reasoning in the category space and improve performance, and (ii) can a new architecture be designed from the ground up to support both thinking and test-time scaling? We demonstrate that in both settings, our models exhibit improved classification accuracy. Leveraging test-time scaling, we observe consistent gains as the number of sampled traces increases. Furthermore, we evaluate two open-source reasoning models, GPT-OSS-20B and Qwen3-14B, showing that while such models are capable of zero-shot reasoning, a lightweight approach--retraining only the embedding matrix of a frozen, smaller model like GPT-2--can surpass the performance of billion-parameter text-based reasoning models. 

---
# Games Are Not Equal: Classifying Cloud Gaming Contexts for Effective User Experience Measurement 

**Authors**: Yifan Wang, Minzhao Lyu, Vijay Sivaraman  

**Link**: [PDF](https://arxiv.org/pdf/2509.19669)  

**Abstract**: To tap into the growing market of cloud gaming, whereby game graphics is rendered in the cloud and streamed back to the user as a video feed, network operators are creating monetizable assurance services that dynamically provision network resources. However, without accurately measuring cloud gaming user experience, they cannot assess the effectiveness of their provisioning methods. Basic measures such as bandwidth and frame rate by themselves do not suffice, and can only be interpreted in the context of the game played and the player activity within the game. This paper equips the network operator with a method to obtain a real-time measure of cloud gaming experience by analyzing network traffic, including contextual factors such as the game title and player activity stage. Our method is able to classify the game title within the first five seconds of game launch, and continuously assess the player activity stage as being active, passive, or idle. We deploy it in an ISP hosting NVIDIA cloud gaming servers for the region. We provide insights from hundreds of thousands of cloud game streaming sessions over a three-month period into the dependence of bandwidth consumption and experience level on the gameplay contexts. 

---
# Selective Classifier-free Guidance for Zero-shot Text-to-speech 

**Authors**: John Zheng, Farhad Maleki  

**Link**: [PDF](https://arxiv.org/pdf/2509.19668)  

**Abstract**: In zero-shot text-to-speech, achieving a balance between fidelity to the target speaker and adherence to text content remains a challenge. While classifier-free guidance (CFG) strategies have shown promising results in image generation, their application to speech synthesis are underexplored. Separating the conditions used for CFG enables trade-offs between different desired characteristics in speech synthesis. In this paper, we evaluate the adaptability of CFG strategies originally developed for image generation to speech synthesis and extend separated-condition CFG approaches for this domain. Our results show that CFG strategies effective in image generation generally fail to improve speech synthesis. We also find that we can improve speaker similarity while limiting degradation of text adherence by applying standard CFG during early timesteps and switching to selective CFG only in later timesteps. Surprisingly, we observe that the effectiveness of a selective CFG strategy is highly text-representation dependent, as differences between the two languages of English and Mandarin can lead to different results even with the same model. 

---
# MoTiC: Momentum Tightness and Contrast for Few-Shot Class-Incremental Learning 

**Authors**: Zeyu He, Shuai Huang, Yuwu Lu, Ming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19664)  

**Abstract**: Few-Shot Class-Incremental Learning (FSCIL) must contend with the dual challenge of learning new classes from scarce samples while preserving old class knowledge. Existing methods use the frozen feature extractor and class-averaged prototypes to mitigate against catastrophic forgetting and overfitting. However, new-class prototypes suffer significant estimation bias due to extreme data scarcity, whereas base-class prototypes benefit from sufficient data. In this work, we theoretically demonstrate that aligning the new-class priors with old-class statistics via Bayesian analysis reduces variance and improves prototype accuracy. Furthermore, we propose large-scale contrastive learning to enforce cross-category feature tightness. To further enrich feature diversity and inject prior information for new-class prototypes, we integrate momentum self-supervision and virtual categories into the Momentum Tightness and Contrast framework (MoTiC), constructing a feature space with rich representations and enhanced interclass cohesion. Experiments on three FSCIL benchmarks produce state-of-the-art performances, particularly on the fine-grained task CUB-200, validating our method's ability to reduce estimation bias and improve incremental learning robustness. 

---
# RoboSSM: Scalable In-context Imitation Learning via State-Space Models 

**Authors**: Youngju Yoo, Jiaheng Hu, Yifeng Zhu, Bo Liu, Qiang Liu, Roberto Martín-Martín, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2509.19658)  

**Abstract**: In-context imitation learning (ICIL) enables robots to learn tasks from prompts consisting of just a handful of demonstrations. By eliminating the need for parameter updates at deployment time, this paradigm supports few-shot adaptation to novel tasks. However, recent ICIL methods rely on Transformers, which have computational limitations and tend to underperform when handling longer prompts than those seen during training. In this work, we introduce RoboSSM, a scalable recipe for in-context imitation learning based on state-space models (SSM). Specifically, RoboSSM replaces Transformers with Longhorn -- a state-of-the-art SSM that provides linear-time inference and strong extrapolation capabilities, making it well-suited for long-context prompts. We evaluate our approach on the LIBERO benchmark and compare it against strong Transformer-based ICIL baselines. Experiments show that RoboSSM extrapolates effectively to varying numbers of in-context demonstrations, yields high performance on unseen tasks, and remains robust in long-horizon scenarios. These results highlight the potential of SSMs as an efficient and scalable backbone for ICIL. Our code is available at this https URL. 

---
# Large Language Models for Pedestrian Safety: An Application to Predicting Driver Yielding Behavior at Unsignalized Intersections 

**Authors**: Yicheng Yang, Zixian Li, Jean Paul Bizimana, Niaz Zafri, Yongfeng Dong, Tianyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19657)  

**Abstract**: Pedestrian safety is a critical component of urban mobility and is strongly influenced by the interactions between pedestrian decision-making and driver yielding behavior at crosswalks. Modeling driver--pedestrian interactions at intersections requires accurately capturing the complexity of these behaviors. Traditional machine learning models often struggle to capture the nuanced and context-dependent reasoning required for these multifactorial interactions, due to their reliance on fixed feature representations and limited interpretability. In contrast, large language models (LLMs) are suited for extracting patterns from heterogeneous traffic data, enabling accurate modeling of driver-pedestrian interactions. Therefore, this paper leverages multimodal LLMs through a novel prompt design that incorporates domain-specific knowledge, structured reasoning, and few-shot prompting, enabling interpretable and context-aware inference of driver yielding behavior, as an example application of modeling pedestrian--driver interaction. We benchmarked state-of-the-art LLMs against traditional classifiers, finding that GPT-4o consistently achieves the highest accuracy and recall, while Deepseek-V3 excels in precision. These findings highlight the critical trade-offs between model performance and computational efficiency, offering practical guidance for deploying LLMs in real-world pedestrian safety systems. 

---
# Where 6G Stands Today: Evolution, Enablers, and Research Gaps 

**Authors**: Salma Tika, Abdelkrim Haqiq, Essaid Sabir, Elmahdi Driouch  

**Link**: [PDF](https://arxiv.org/pdf/2509.19646)  

**Abstract**: As the fifth-generation (5G) mobile communication system continues its global deployment, both industry and academia have started conceptualizing the 6th generation (6G) to address the growing need for a progressively advanced and digital society. Even while 5G offers considerable advancements over LTE, it could struggle to be sufficient to meet all of the requirements, including ultra-high reliability, seamless automation, and ubiquitous coverage. In response, 6G is supposed to bring out a highly intelligent, automated, and ultra-reliable communication system that can handle a vast number of connected devices. This paper offers a comprehensive overview of 6G, beginning with its main stringent requirements while focusing on key enabling technologies such as terahertz (THz) communications, intelligent reflecting surfaces, massive MIMO and AI-driven networking that will shape the 6G networks. Furthermore, the paper lists various 6G applications and usage scenarios that will benefit from these advancements. At the end, we outline the potential challenges that must be addressed to achieve the 6G promises. 

---
# Are We Scaling the Right Thing? A System Perspective on Test-Time Scaling 

**Authors**: Youpeng Zhao, Jinpeng LV, Di Wu, Jun Wang, Christopher Gooley  

**Link**: [PDF](https://arxiv.org/pdf/2509.19645)  

**Abstract**: Test-time scaling (TTS) has recently emerged as a promising direction to exploit the hidden reasoning capabilities of pre-trained large language models (LLMs). However, existing scaling methods narrowly focus on the compute-optimal Pareto-frontier, ignoring the simple fact that compute-optimal is not always system-optimal. In this work, we propose a system-driven perspective on TTS, analyzing how reasoning models scale against practical metrics, such as latency and cost-per-token. By evaluating the impact of popular optimizations such as tensor parallelism and speculative decoding, our preliminary analysis reveals the limitations of current methods and calls for a paradigm shift toward holistic, system-aware evaluations that capture the true essence of scaling laws at inference time. 

---
# Mamba Modulation: On the Length Generalization of Mamba 

**Authors**: Peng Lu, Jerry Huang, Qiuhao Zeng, Xinyu Wang, Boxing Wang, Philippe Langlais, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.19633)  

**Abstract**: The quadratic complexity of the attention mechanism in Transformer models has motivated the development of alternative architectures with sub-quadratic scaling, such as state-space models. Among these, Mamba has emerged as a leading architecture, achieving state-of-the-art results across a range of language modeling tasks. However, Mamba's performance significantly deteriorates when applied to contexts longer than those seen during pre-training, revealing a sharp sensitivity to context length extension. Through detailed analysis, we attribute this limitation to the out-of-distribution behaviour of its state-space dynamics, particularly within the parameterization of the state transition matrix $\mathbf{A}$. Unlike recent works which attribute this sensitivity to the vanished accumulation of discretization time steps, $\exp(-\sum_{t=1}^N\Delta_t)$, we establish a connection between state convergence behavior as the input length approaches infinity and the spectrum of the transition matrix $\mathbf{A}$, offering a well-founded explanation of its role in length extension. Next, to overcome this challenge, we propose an approach that applies spectrum scaling to pre-trained Mamba models to enable robust long-context generalization by selectively modulating the spectrum of $\mathbf{A}$ matrices in each layer. We show that this can significantly improve performance in settings where simply modulating $\Delta_t$ fails, validating our insights and providing avenues for better length generalization of state-space models with structured transition matrices. 

---
# Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning 

**Authors**: Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19631)  

**Abstract**: Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs. 

---
# Knowledge Base-Aware Orchestration: A Dynamic, Privacy-Preserving Method for Multi-Agent Systems 

**Authors**: Danilo Trombino, Vincenzo Pecorella, Alessandro de Giulii, Davide Tresoldi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19599)  

**Abstract**: Multi-agent systems (MAS) are increasingly tasked with solving complex, knowledge-intensive problems where effective agent orchestration is critical. Conventional orchestration methods rely on static agent descriptions, which often become outdated or incomplete. This limitation leads to inefficient task routing, particularly in dynamic environments where agent capabilities continuously evolve. We introduce Knowledge Base-Aware (KBA) Orchestration, a novel approach that augments static descriptions with dynamic, privacy-preserving relevance signals derived from each agent's internal knowledge base (KB). In the proposed framework, when static descriptions are insufficient for a clear routing decision, the orchestrator prompts the subagents in parallel. Each agent then assesses the task's relevance against its private KB, returning a lightweight ACK signal without exposing the underlying data. These collected signals populate a shared semantic cache, providing dynamic indicators of agent suitability for future queries. By combining this novel mechanism with static descriptions, our method achieves more accurate and adaptive task routing preserving agent autonomy and data confidentiality. Benchmarks show that our KBA Orchestration significantly outperforms static description-driven methods in routing precision and overall system efficiency, making it suitable for large-scale systems that require higher accuracy than standard description-driven routing. 

---
# GuessingGame: Measuring the Informativeness of Open-Ended Questions in Large Language Models 

**Authors**: Dylan Hutson, Daniel Vennemeyer, Aneesh Deshmukh, Justin Zhan, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19593)  

**Abstract**: We introduce GuessingGame, a protocol for evaluating large language models (LLMs) as strategic question-askers in open-ended, open-domain settings. A Guesser LLM identifies a hidden object by posing free-form questions to an Oracle without predefined choices or candidate lists. To measure question quality, we propose two information gain (IG) metrics: a Bayesian method that tracks belief updates over semantic concepts using LLM-scored relevance, and an entropy-based method that filters candidates via ConceptNet. Both metrics are model-agnostic and support post hoc analysis. Across 858 games with multiple models and prompting strategies, higher IG strongly predicts efficiency: a one-standard-deviation IG increase reduces expected game length by 43\%. Prompting constraints guided by IG, such as enforcing question diversity, enable weaker models to significantly improve performance. These results show that question-asking in LLMs is both measurable and improvable, and crucial for interactive reasoning. 

---
# Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation 

**Authors**: Roy Fejgin, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Ryan Langman Jaehyeon Kim, Subhankar Ghosh, Shehzeen Hussain, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19592)  

**Abstract**: Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity. 

---
# Reverse Engineering User Stories from Code using Large Language Models 

**Authors**: Mohamed Ouf, Haoyu Li, Michael Zhang, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19587)  

**Abstract**: User stories are essential in agile development, yet often missing or outdated in legacy and poorly documented systems. We investigate whether large language models (LLMs) can automatically recover user stories directly from source code and how prompt design impacts output quality. Using 1,750 annotated C++ snippets of varying complexity, we evaluate five state-of-the-art LLMs across six prompting strategies. Results show that all models achieve, on average, an F1 score of 0.8 for code up to 200 NLOC. Our findings show that a single illustrative example enables the smallest model (8B) to match the performance of a much larger 70B model. In contrast, structured reasoning via Chain-of-Thought offers only marginal gains, primarily for larger models. 

---
# A Foundation Chemical Language Model for Comprehensive Fragment-Based Drug Discovery 

**Authors**: Alexander Ho, Sukyeong Lee, Francis T.F. Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2509.19586)  

**Abstract**: We introduce FragAtlas-62M, a specialized foundation model trained on the largest fragment dataset to date. Built on the complete ZINC-22 fragment subset comprising over 62 million molecules, it achieves unprecedented coverage of fragment chemical space. Our GPT-2 based model (42.7M parameters) generates 99.90% chemically valid fragments. Validation across 12 descriptors and three fingerprint methods shows generated fragments closely match the training distribution (all effect sizes < 0.4). The model retains 53.6% of known ZINC fragments while producing 22% novel structures with practical relevance. We release FragAtlas-62M with training code, preprocessed data, documentation, and model weights to accelerate adoption. 

---
# Learning Dynamics of Deep Learning -- Force Analysis of Deep Neural Networks 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.19554)  

**Abstract**: This thesis explores how deep learning models learn over time, using ideas inspired by force analysis. Specifically, we zoom in on the model's training procedure to see how one training example affects another during learning, like analyzing how forces move objects. We break this influence into two parts: how similar the two examples are, and how strong the updating force is. This framework helps us understand a wide range of the model's behaviors in different real systems. For example, it explains why certain examples have non-trivial learning paths, why (and why not) some LLM finetuning methods work, and why simpler, more structured patterns tend to be learned more easily. We apply this approach to various learning tasks and uncover new strategies for improving model training. While the method is still developing, it offers a new way to interpret models' behaviors systematically. 

---
# DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions 

**Authors**: Zongyue Li, Xiao Han, Yusong Li, Niklas Strauss, Matthias Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.19538)  

**Abstract**: Diffusion-based world models have demonstrated strong capabilities in synthesizing realistic long-horizon trajectories for offline reinforcement learning (RL). However, many existing methods do not directly generate actions alongside states and rewards, limiting their compatibility with standard value-based offline RL algorithms that rely on one-step temporal difference (TD) learning. While prior work has explored joint modeling of states, rewards, and actions to address this issue, such formulations often lead to increased training complexity and reduced performance in practice. We propose \textbf{DAWM}, a diffusion-based world model that generates future state-reward trajectories conditioned on the current state, action, and return-to-go, paired with an inverse dynamics model (IDM) for efficient action inference. This modular design produces complete synthetic transitions suitable for one-step TD-based offline RL, enabling effective and computationally efficient training. Empirically, we show that conservative offline RL algorithms such as TD3BC and IQL benefit significantly from training on these augmented trajectories, consistently outperforming prior diffusion-based baselines across multiple tasks in the D4RL benchmark. 

---
# Semantic-Aware Fuzzing: An Empirical Framework for LLM-Guided, Reasoning-Driven Input Mutation 

**Authors**: Mengdi Lu, Steven Ding, Furkan Alaca, Philippe Charland  

**Link**: [PDF](https://arxiv.org/pdf/2509.19533)  

**Abstract**: Security vulnerabilities in Internet-of-Things devices, mobile platforms, and autonomous systems remain critical. Traditional mutation-based fuzzers -- while effectively explore code paths -- primarily perform byte- or bit-level edits without semantic reasoning. Coverage-guided tools such as AFL++ use dictionaries, grammars, and splicing heuristics to impose shallow structural constraints, leaving deeper protocol logic, inter-field dependencies, and domain-specific semantics unaddressed. Conversely, reasoning-capable large language models (LLMs) can leverage pretraining knowledge to understand input formats, respect complex constraints, and propose targeted mutations, much like an experienced reverse engineer or testing expert. However, lacking ground truth for "correct" mutation reasoning makes supervised fine-tuning impractical, motivating explorations of off-the-shelf LLMs via prompt-based few-shot learning. To bridge this gap, we present an open-source microservices framework that integrates reasoning LLMs with AFL++ on Google's FuzzBench, tackling asynchronous execution and divergent hardware demands (GPU- vs. CPU-intensive) of LLMs and fuzzers. We evaluate four research questions: (R1) How can reasoning LLMs be integrated into the fuzzing mutation loop? (R2) Do few-shot prompts yield higher-quality mutations than zero-shot? (R3) Can prompt engineering with off-the-shelf models improve fuzzing directly? and (R4) Which open-source reasoning LLMs perform best under prompt-only conditions? Experiments with Llama3.3, Deepseek-r1-Distill-Llama-70B, QwQ-32B, and Gemma3 highlight Deepseek as the most promising. Mutation effectiveness depends more on prompt complexity and model choice than shot count. Response latency and throughput bottlenecks remain key obstacles, offering directions for future work. 

---
# A Longitudinal Randomized Control Study of Companion Chatbot Use: Anthropomorphism and Its Mediating Role on Social Impacts 

**Authors**: Rose E. Guingrich, Michael S. A. Graziano  

**Link**: [PDF](https://arxiv.org/pdf/2509.19515)  

**Abstract**: Relationships with social artificial intelligence (AI) agents are on the rise. People report forming friendships, mentorships, and romantic partnerships with chatbots such as Replika, a type of social AI agent that is designed specifically for companionship. Concerns that companion chatbot relationships may harm or replace human ones have been raised, but whether and how these social consequences occur remains unclear. Prior research suggests that people's states of social need and their anthropomorphism of the AI agent may play a role in how human-AI interaction impacts human-human interaction. In this longitudinal study (N = 183), participants were randomly assigned to converse with a companion chatbot over text or to play text-based word games for 10 minutes a day for 21 consecutive days. During these 21 days, participants also completed four surveys and two audio-recorded interviews. We found that people's social health and relationships were not significantly impacted by interacting with a companion chatbot across 21 days compared to the control group. However, people who had a higher desire to socially connect anthropomorphized the chatbot more. Those who anthropomorphized the chatbot more indicated that the human-chatbot interaction had greater impacts on their social interactions and relationships with family and friends. A mediation analysis suggested that the impact of human-AI interaction on human-human social outcomes was mediated by the extent to which people anthropomorphized the AI agent, which itself was related to the desire to socially connect. 

---
# The Heterogeneous Multi-Agent Challenge 

**Authors**: Charles Dansereau, Junior-Samuel Lopez-Yepez, Karthik Soma, Antoine Fagette  

**Link**: [PDF](https://arxiv.org/pdf/2509.19512)  

**Abstract**: Multi-Agent Reinforcement Learning (MARL) is a growing research area which gained significant traction in recent years, extending Deep RL applications to a much wider range of problems. A particularly challenging class of problems in this domain is Heterogeneous Multi-Agent Reinforcement Learning (HeMARL), where agents with different sensors, resources, or capabilities must cooperate based on local information. The large number of real-world situations involving heterogeneous agents makes it an attractive research area, yet underexplored, as most MARL research focuses on homogeneous agents (e.g., a swarm of identical robots). In MARL and single-agent RL, standardized environments such as ALE and SMAC have allowed to establish recognized benchmarks to measure progress. However, there is a clear lack of such standardized testbed for cooperative HeMARL. As a result, new research in this field often uses simple environments, where most algorithms perform near optimally, or uses weakly heterogeneous MARL environments. 

---
# AIRwaves at CheckThat! 2025: Retrieving Scientific Sources for Implicit Claims on Social Media with Dual Encoders and Neural Re-Ranking 

**Authors**: Cem Ashbaugh, Leon Baumgärtner, Tim Gress, Nikita Sidorov, Daniel Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.19509)  

**Abstract**: Linking implicit scientific claims made on social media to their original publications is crucial for evidence-based fact-checking and scholarly discourse, yet it is hindered by lexical sparsity, very short queries, and domain-specific language. Team AIRwaves ranked second in Subtask 4b of the CLEF-2025 CheckThat! Lab with an evidence-retrieval approach that markedly outperforms the competition baseline. The optimized sparse-retrieval baseline(BM25) achieves MRR@5 = 0.5025 on the gold label blind test set. To surpass this baseline, a two-stage retrieval pipeline is introduced: (i) a first stage that uses a dual encoder based on E5-large, fine-tuned using in-batch and mined hard negatives and enhanced through chunked tokenization and rich document metadata; and (ii) a neural re-ranking stage using a SciBERT cross-encoder. Replacing purely lexical matching with neural representations lifts performance to MRR@5 = 0.6174, and the complete pipeline further improves to MRR@5 = 0.6828. The findings demonstrate that coupling dense retrieval with neural re-rankers delivers a powerful and efficient solution for tweet-to-study matching and provides a practical blueprint for future evidence-retrieval pipelines. 

---
# Generative AI as a catalyst for democratic Innovation: Enhancing citizen engagement in participatory budgeting 

**Authors**: Italo Alberto do Nascimento Sousa, Jorge Machado, Jose Carlos Vaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.19497)  

**Abstract**: This research examines the role of Generative Artificial Intelligence (AI) in enhancing citizen engagement in participatory budgeting. In response to challenges like declining civic participation and increased societal polarization, the study explores how online political participation can strengthen democracy and promote social equity. By integrating Generative AI into public consultation platforms, the research aims to improve citizen proposal formulation and foster effective dialogue between citizens and government. It assesses the capacities governments need to implement AI-enhanced participatory tools, considering technological dependencies and vulnerabilities. Analyzing technological structures, actors, interests, and strategies, the study contributes to understanding how technological advancements can reshape participatory institutions to better facilitate citizen involvement. Ultimately, the research highlights how Generative AI can transform participatory institutions, promoting inclusive, democratic engagement and empowering citizens. 

---
# ArtiFree: Detecting and Reducing Generative Artifacts in Diffusion-based Speech Enhancement 

**Authors**: Bhawana Chhaglani, Yang Gao, Julius Richter, Xilin Li, Syavosh Zadissa, Tarun Pruthi, Andrew Lovitt  

**Link**: [PDF](https://arxiv.org/pdf/2509.19495)  

**Abstract**: Diffusion-based speech enhancement (SE) achieves natural-sounding speech and strong generalization, yet suffers from key limitations like generative artifacts and high inference latency. In this work, we systematically study artifact prediction and reduction in diffusion-based SE. We show that variance in speech embeddings can be used to predict phonetic errors during inference. Building on these findings, we propose an ensemble inference method guided by semantic consistency across multiple diffusion runs. This technique reduces WER by 15% in low-SNR conditions, effectively improving phonetic accuracy and semantic plausibility. Finally, we analyze the effect of the number of diffusion steps, showing that adaptive diffusion steps balance artifact suppression and latency. Our findings highlight semantic priors as a powerful tool to guide generative SE toward artifact-free outputs. 

---
# Identifying and Addressing User-level Security Concerns in Smart Homes Using "Smaller" LLMs 

**Authors**: Hafijul Hoque Chowdhury, Riad Ahmed Anonto, Sourov Jajodia, Suryadipta Majumdar, Md. Shohrab Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2509.19485)  

**Abstract**: With the rapid growth of smart home IoT devices, users are increasingly exposed to various security risks, as evident from recent studies. While seeking answers to know more on those security concerns, users are mostly left with their own discretion while going through various sources, such as online blogs and technical manuals, which may render higher complexity to regular users trying to extract the necessary information. This requirement does not go along with the common mindsets of smart home users and hence threatens the security of smart homes furthermore. In this paper, we aim to identify and address the major user-level security concerns in smart homes. Specifically, we develop a novel dataset of Q&A from public forums, capturing practical security challenges faced by smart home users. We extract major security concerns in smart homes from our dataset by leveraging the Latent Dirichlet Allocation (LDA). We fine-tune relatively "smaller" transformer models, such as T5 and Flan-T5, on this dataset to build a QA system tailored for smart home security. Unlike larger models like GPT and Gemini, which are powerful but often resource hungry and require data sharing, smaller models are more feasible for deployment in resource-constrained or privacy-sensitive environments like smart homes. The dataset is manually curated and supplemented with synthetic data to explore its potential impact on model performance. This approach significantly improves the system's ability to deliver accurate and relevant answers, helping users address common security concerns with smart home IoT devices. Our experiments on real-world user concerns show that our work improves the performance of the base models. 

---
# A Realistic Evaluation of Cross-Frequency Transfer Learning and Foundation Forecasting Models 

**Authors**: Kin G. Olivares, Malcolm Wolff, Tatiana Konstantinova, Shankar Ramasubramanian, Andrew Gordon Wilson, Andres Potapczynski, Willa Potosnak, Mengfei Cao, Boris Oreshkin, Dmitry Efimov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19465)  

**Abstract**: Cross-frequency transfer learning (CFTL) has emerged as a popular framework for curating large-scale time series datasets to pre-train foundation forecasting models (FFMs). Although CFTL has shown promise, current benchmarking practices fall short of accurately assessing its performance. This shortcoming stems from many factors: an over-reliance on small-scale evaluation datasets; inadequate treatment of sample size when computing summary statistics; reporting of suboptimal statistical models; and failing to account for non-negligible risks of overlap between pre-training and test datasets. To address these limitations, we introduce a unified reimplementation of widely-adopted neural forecasting networks, adapting them for the CFTL setup; we pre-train only on proprietary and synthetic data, being careful to prevent test leakage; and we evaluate on 15 large, diverse public forecast competition datasets. Our empirical analysis reveals that statistical models' accuracy is frequently underreported. Notably, we confirm that statistical models and their ensembles consistently outperform existing FFMs by more than 8.2% in sCRPS, and by more than 20% MASE, across datasets. However, we also find that synthetic dataset pre-training does improve the accuracy of a FFM by 7% percent. 

---
# Self-evolved Imitation Learning in Simulated World 

**Authors**: Yifan Ye, Jun Cen, Jing Chen, Zhihe Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19460)  

**Abstract**: Imitation learning has been a trend recently, yet training a generalist agent across multiple tasks still requires large-scale expert demonstrations, which are costly and labor-intensive to collect. To address the challenge of limited supervision, we propose Self-Evolved Imitation Learning (SEIL), a framework that progressively improves a few-shot model through simulator interactions. The model first attempts tasksin the simulator, from which successful trajectories are collected as new demonstrations for iterative refinement. To enhance the diversity of these demonstrations, SEIL employs dual-level augmentation: (i) Model-level, using an Exponential Moving Average (EMA) model to collaborate with the primary model, and (ii) Environment-level, introducing slight variations in initial object positions. We further introduce a lightweight selector that filters complementary and informative trajectories from the generated pool to ensure demonstration quality. These curated samples enable the model to achieve competitive performance with far fewer training examples. Extensive experiments on the LIBERO benchmark show that SEIL achieves a new state-of-the-art performance in few-shot imitation learning scenarios. Code is available at this https URL. 

---
# ROPA: Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation 

**Authors**: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2509.19454)  

**Abstract**: Training robust bimanual manipulation policies via imitation learning requires demonstration data with broad coverage over robot poses, contacts, and scene contexts. However, collecting diverse and precise real-world demonstrations is costly and time-consuming, which hinders scalability. Prior works have addressed this with data augmentation, typically for either eye-in-hand (wrist camera) setups with RGB inputs or for generating novel images without paired actions, leaving augmentation for eye-to-hand (third-person) RGB-D training with new action labels less explored. In this paper, we propose Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA), an offline imitation learning data augmentation method that fine-tunes Stable Diffusion to synthesize third-person RGB and RGB-D observations of novel robot poses. Our approach simultaneously generates corresponding joint-space action labels while employing constrained optimization to enforce physical consistency through appropriate gripper-to-object contact constraints in bimanual scenarios. We evaluate our method on 5 simulated and 3 real-world tasks. Our results across 2625 simulation trials and 300 real-world trials demonstrate that ROPA outperforms baselines and ablations, showing its potential for scalable RGB and RGB-D data augmentation in eye-to-hand bimanual manipulation. Our project website is available at: this https URL. 

---
# Probabilistic Runtime Verification, Evaluation and Risk Assessment of Visual Deep Learning Systems 

**Authors**: Birk Torpmann-Hagen, Pål Halvorsen, Michael A. Riegler, Dag Johansen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19419)  

**Abstract**: Despite achieving excellent performance on benchmarks, deep neural networks often underperform in real-world deployment due to sensitivity to minor, often imperceptible shifts in input data, known as distributional shifts. These shifts are common in practical scenarios but are rarely accounted for during evaluation, leading to inflated performance metrics. To address this gap, we propose a novel methodology for the verification, evaluation, and risk assessment of deep learning systems. Our approach explicitly models the incidence of distributional shifts at runtime by estimating their probability from outputs of out-of-distribution detectors. We combine these estimates with conditional probabilities of network correctness, structuring them in a binary tree. By traversing this tree, we can compute credible and precise estimates of network accuracy. We assess our approach on five different datasets, with which we simulate deployment conditions characterized by differing frequencies of distributional shift. Our approach consistently outperforms conventional evaluation, with accuracy estimation errors typically ranging between 0.01 and 0.1. We further showcase the potential of our approach on a medical segmentation benchmark, wherein we apply our methods towards risk assessment by associating costs with tree nodes, informing cost-benefit analyses and value-judgments. Ultimately, our approach offers a robust framework for improving the reliability and trustworthiness of deep learning systems, particularly in safety-critical applications, by providing more accurate performance estimates and actionable risk assessments. 

---
# EngravingGNN: A Hybrid Graph Neural Network for End-to-End Piano Score Engraving 

**Authors**: Emmanouil Karystinaios, Francesco Foscarin, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2509.19412)  

**Abstract**: This paper focuses on automatic music engraving, i.e., the creation of a humanly-readable musical score from musical content. This step is fundamental for all applications that include a human player, but it remains a mostly unexplored topic in symbolic music processing. In this work, we formalize the problem as a collection of interdependent subtasks, and propose a unified graph neural network (GNN) framework that targets the case of piano music and quantized symbolic input. Our method employs a multi-task GNN to jointly predict voice connections, staff assignments, pitch spelling, key signature, stem direction, octave shifts, and clef signs. A dedicated postprocessing pipeline generates print-ready MusicXML/MEI outputs. Comprehensive evaluation on two diverse piano corpora (J-Pop and DCML Romantic) demonstrates that our unified model achieves good accuracy across all subtasks, compared to existing systems that only specialize in specific subtasks. These results indicate that a shared GNN encoder with lightweight task-specific decoders in a multi-task setting offers a scalable and effective solution for automatic music engraving. 

---
# TimeMosaic: Temporal Heterogeneity Guided Time Series Forecasting via Adaptive Granularity Patch and Segment-wise Decoding 

**Authors**: Kuiye Ding, Fanda Fan, Chunyi Hou, Zheya Wang, Lei Wang, Zhengxin Yang, Jianfeng Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19406)  

**Abstract**: Multivariate time series forecasting is essential in domains such as finance, transportation, climate, and energy. However, existing patch-based methods typically adopt fixed-length segmentation, overlooking the heterogeneity of local temporal dynamics and the decoding heterogeneity of forecasting. Such designs lose details in information-dense regions, introduce redundancy in stable segments, and fail to capture the distinct complexities of short-term and long-term horizons. We propose TimeMosaic, a forecasting framework that aims to address temporal heterogeneity. TimeMosaic employs adaptive patch embedding to dynamically adjust granularity according to local information density, balancing motif reuse with structural clarity while preserving temporal continuity. In addition, it introduces segment-wise decoding that treats each prediction horizon as a related subtask and adapts to horizon-specific difficulty and information requirements, rather than applying a single uniform decoder. Extensive evaluations on benchmark datasets demonstrate that TimeMosaic delivers consistent improvements over existing methods, and our model trained on the large-scale corpus with 321 billion observations achieves performance competitive with state-of-the-art TSFMs. 

---
# Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation 

**Authors**: Tony Chahoud, Lorenzo Mario Amorosa, Riccardo Marini, Luca De Nardis  

**Link**: [PDF](https://arxiv.org/pdf/2509.19405)  

**Abstract**: Accurate outdoor positioning in cellular networks is hindered by sparse, heterogeneous measurement collections and the high cost of exhaustive site surveys. This paper introduces a lightweight, modular mobile data augmentation framework designed to enhance multi-cell fingerprinting-based positioning using operator-collected minimization of drive test (MDT) records. The proposed approach decouples spatial and radio-feature synthesis: kernel density estimation (KDE) models the empirical spatial distribution to generate geographically coherent synthetic locations, while a k-nearest-neighbor (KNN)-based block produces augmented per-cell radio fingerprints. The architecture is intentionally training-free, interpretable, and suitable for distributed or on-premise operator deployments, supporting privacy-aware workflows. We both validate each augmentation module independently and assess its end-to-end impact on fingerprinting-based positioning using a real-world MDT dataset provided by an Italian mobile network operator across diverse urban and peri-urban scenarios. Results show that the proposed KDE-KNN augmentation consistently improves positioning performance, with the largest benefits in sparsely sampled or structurally complex regions; we also observe region-dependent saturation effects as augmentation increases. The framework offers a practical, low-complexity path to enhance operator positioning services using existing mobile data traces. 

---
# Online Adaptation via Dual-Stage Alignment and Self-Supervision for Fast-Calibration Brain-Computer Interfaces 

**Authors**: Sheng-Bin Duan, Jian-Long Hao, Tian-Yu Xiang, Xiao-Hu Zhou, Mei-Jiang Gui, Xiao-Liang Xie, Shi-Qi Liu, Zeng-Guang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19403)  

**Abstract**: Individual differences in brain activity hinder the online application of electroencephalogram (EEG)-based brain computer interface (BCI) systems. To overcome this limitation, this study proposes an online adaptation algorithm for unseen subjects via dual-stage alignment and self-supervision. The alignment process begins by applying Euclidean alignment in the EEG data space and then updates batch normalization statistics in the representation space. Moreover, a self-supervised loss is designed to update the decoder. The loss is computed by soft pseudo-labels derived from the decoder as a proxy for the unknown ground truth, and is calibrated by Shannon entropy to facilitate self-supervised training. Experiments across five public datasets and seven decoders show the proposed algorithm can be integrated seamlessly regardless of BCI paradigm and decoder architecture. In each iteration, the decoder is updated with a single online trial, which yields average accuracy gains of 4.9% on steady-state visual evoked potentials (SSVEP) and 3.6% on motor imagery. These results support fast-calibration operation and show that the proposed algorithm has great potential for BCI applications. 

---
# FedOC: Multi-Server FL with Overlapping Client Relays in Wireless Edge Networks 

**Authors**: Yun Ji, Zeyu Chen, Xiaoxiong Zhong, Yanan Ma, Sheng Zhang, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19398)  

**Abstract**: Multi-server Federated Learning (FL) has emerged as a promising solution to mitigate communication bottlenecks of single-server FL. We focus on a typical multi-server FL architecture, where the regions covered by different edge servers (ESs) may overlap. A key observation of this architecture is that clients located in the overlapping areas can access edge models from multiple ESs. Building on this insight, we propose FedOC (Federated learning with Overlapping Clients), a novel framework designed to fully exploit the potential of these overlapping clients. In FedOC, overlapping clients could serve dual roles: (1) as Relay Overlapping Clients (ROCs), they forward edge models between neighboring ESs in real time to facilitate model sharing among different ESs; and (2) as Normal Overlapping Clients (NOCs), they dynamically select their initial model for local training based on the edge model delivery time, which enables indirect data fusion among different regions of ESs. The overall FedOC workflow proceeds as follows: in every round, each client trains local model based on the earliest received edge model and transmits to the respective ESs for model aggregation. Then each ES transmits the aggregated edge model to neighboring ESs through ROC relaying. Upon receiving the relayed models, each ES performs a second aggregation and subsequently broadcasts the updated model to covered clients. The existence of ROCs enables the model of each ES to be disseminated to the other ESs in a decentralized manner, which indirectly achieves intercell model and speeding up the training process, making it well-suited for latency-sensitive edge environments. Extensive experimental results show remarkable performance gains of our scheme compared to existing methods. 

---
# Self-Alignment Learning to Improve Myocardial Infarction Detection from Single-Lead ECG 

**Authors**: Jiarui Jin, Xiaocheng Fang, Haoyu Wang, Jun Li, Che Liu, Donglin Xie, Hongyan Li, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19397)  

**Abstract**: Myocardial infarction is a critical manifestation of coronary artery disease, yet detecting it from single-lead electrocardiogram (ECG) remains challenging due to limited spatial information. An intuitive idea is to convert single-lead into multiple-lead ECG for classification by pre-trained models, but generative methods optimized at the signal level in most cases leave a large latent space gap, ultimately degrading diagnostic performance. This naturally raises the question of whether latent space alignment could help. However, most prior ECG alignment methods focus on learning transformation invariance, which mismatches the goal of single-lead detection. To address this issue, we propose SelfMIS, a simple yet effective alignment learning framework to improve myocardial infarction detection from single-lead ECG. Discarding manual data augmentations, SelfMIS employs a self-cutting strategy to pair multiple-lead ECG with their corresponding single-lead segments and directly align them in the latent space. This design shifts the learning objective from pursuing transformation invariance to enriching the single-lead representation, explicitly driving the single-lead ECG encoder to learn a representation capable of inferring global cardiac context from the local signal. Experimentally, SelfMIS achieves superior performance over baseline models across nine myocardial infarction types while maintaining a simpler architecture and lower computational overhead, thereby substantiating the efficacy of direct latent space alignment. Our code and checkpoint will be publicly available after acceptance. 

---
# OmniFed: A Modular Framework for Configurable Federated Learning from Edge to HPC 

**Authors**: Sahil Tyagi, Andrei Cozma, Olivera Kotevska, Feiyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19396)  

**Abstract**: Federated Learning (FL) is critical for edge and High Performance Computing (HPC) where data is not centralized and privacy is crucial. We present OmniFed, a modular framework designed around decoupling and clear separation of concerns for configuration, orchestration, communication, and training logic. Its architecture supports configuration-driven prototyping and code-level override-what-you-need customization. We also support different topologies, mixed communication protocols within a single deployment, and popular training algorithms. It also offers optional privacy mechanisms including Differential Privacy (DP), Homomorphic Encryption (HE), and Secure Aggregation (SA), as well as compression strategies. These capabilities are exposed through well-defined extension points, allowing users to customize topology and orchestration, learning logic, and privacy/compression plugins, all while preserving the integrity of the core system. We evaluate multiple models and algorithms to measure various performance metrics. By unifying topology configuration, mixed-protocol communication, and pluggable modules in one stack, OmniFed streamlines FL deployment across heterogeneous environments. Github repository is available at this https URL. 

---
# TensLoRA: Tensor Alternatives for Low-Rank Adaptation 

**Authors**: Axel Marmoret, Reda Bensaid, Jonathan Lys, Vincent Gripon, François Leduc-Primeau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19391)  

**Abstract**: Low-Rank Adaptation (LoRA) is widely used to efficiently adapt Transformers by adding trainable low-rank matrices to attention projections. While effective, these matrices are considered independent for each attention projection (Query, Key, and Value) and each layer. Recent extensions have considered joint, tensor-based adaptations, but only in limited forms and without a systematic framework. We introduce TensLoRA, a unified framework that aggregates LoRA updates into higher-order tensors and models a broad family of tensor-based low-rank adaptations. Our formulation generalizes existing tensor-based methods and enables mode-specific compression rates, allowing parameter budgets to be tailored according to the modality and task. Experiments on vision and language benchmarks reveal that the tensor construction directly impacts performance, sometimes better than standard LoRA under similar parameter counts. 

---
# Data-Driven Reconstruction of Significant Wave Heights from Sparse Observations 

**Authors**: Hongyuan Shi, Yilin Zhai, Ping Dong, Zaijin You, Chao Zhan, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19384)  

**Abstract**: Reconstructing high-resolution regional significant wave height fields from sparse and uneven buoy observations remains a core challenge for ocean monitoring and risk-aware operations. We introduce AUWave, a hybrid deep learning framework that fuses a station-wise sequence encoder (MLP) with a multi-scale U-Net enhanced by a bottleneck self-attention layer to recover 32$\times$32 regional SWH fields. A systematic Bayesian hyperparameter search with Optuna identifies the learning rate as the dominant driver of generalization, followed by the scheduler decay and the latent dimension. Using NDBC buoy observations and ERA5 reanalysis over the Hawaii region, AUWave attains a minimum validation loss of 0.043285 and a slightly right-skewed RMSE distribution. Spatial errors are lowest near observation sites and increase with distance, reflecting identifiability limits under sparse sampling. Sensitivity experiments show that AUWave consistently outperforms a representative baseline in data-richer configurations, while the baseline is only marginally competitive in the most underdetermined single-buoy cases. The architecture's multi-scale and attention components translate into accuracy gains when minimal but non-trivial spatial anchoring is available. Error maps and buoy ablations reveal key anchor stations whose removal disproportionately degrades performance, offering actionable guidance for network design. AUWave provides a scalable pathway for gap filling, high-resolution priors for data assimilation, and contingency reconstruction. 

---
# Learning from Observation: A Survey of Recent Advances 

**Authors**: Returaj Burnwal, Hriday Mehta, Nirav Pravinbhai Bhatt, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19379)  

**Abstract**: Imitation Learning (IL) algorithms offer an efficient way to train an agent by mimicking an expert's behavior without requiring a reward function. IL algorithms often necessitate access to state and action information from expert demonstrations. Although expert actions can provide detailed guidance, requiring such action information may prove impractical for real-world applications where expert actions are difficult to obtain. To address this limitation, the concept of learning from observation (LfO) or state-only imitation learning (SOIL) has recently gained attention, wherein the imitator only has access to expert state visitation information. In this paper, we present a framework for LfO and use it to survey and classify existing LfO methods in terms of their trajectory construction, assumptions and algorithm's design choices. This survey also draws connections between several related fields like offline RL, model-based RL and hierarchical RL. Finally, we use our framework to identify open problems and suggest future research directions. 

---
# Solving Freshness in RAG: A Simple Recency Prior and the Limits of Heuristic Trend Detection 

**Authors**: Matthew Grofsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.19376)  

**Abstract**: We address temporal failures in RAG systems using two methods on cybersecurity data. A simple recency prior achieved an accuracy of 1.00 on freshness tasks. In contrast, a clustering heuristic for topic evolution failed (0.08 F1-score), showing trend detection requires methods beyond simple heuristics. 

---
# Uncertainty Quantification of Large Language Models using Approximate Bayesian Computation 

**Authors**: Mridul Sharma, Adeetya Patel, Zaneta D' Souza, Samira Abbasgholizadeh Rahimi, Siva Reddy, Sreenath Madathil  

**Link**: [PDF](https://arxiv.org/pdf/2509.19375)  

**Abstract**: Despite their widespread applications, Large Language Models (LLMs) often struggle to express uncertainty, posing a challenge for reliable deployment in high stakes and safety critical domains like clinical diagnostics. Existing standard baseline methods such as model logits and elicited probabilities produce overconfident and poorly calibrated estimates. In this work, we propose Approximate Bayesian Computation (ABC), a likelihood-free Bayesian inference, based approach that treats LLMs as a stochastic simulator to infer posterior distributions over predictive probabilities. We evaluate our ABC approach on two clinically relevant benchmarks: a synthetic oral lesion diagnosis dataset and the publicly available GretelAI symptom-to-diagnosis dataset. Compared to standard baselines, our approach improves accuracy by up to 46.9\%, reduces Brier scores by 74.4\%, and enhances calibration as measured by Expected Calibration Error (ECE) and predictive entropy. 

---
# Representation-based Broad Hallucination Detectors Fail to Generalize Out of Distribution 

**Authors**: Zuzanna Dubanowska, Maciej Żelaszczyk, Michał Brzozowski, Paolo Mandica, Michał Karpowicz  

**Link**: [PDF](https://arxiv.org/pdf/2509.19372)  

**Abstract**: We critically assess the efficacy of the current SOTA in hallucination detection and find that its performance on the RAGTruth dataset is largely driven by a spurious correlation with data. Controlling for this effect, state-of-the-art performs no better than supervised linear probes, while requiring extensive hyperparameter tuning across datasets. Out-of-distribution generalization is currently out of reach, with all of the analyzed methods performing close to random. We propose a set of guidelines for hallucination detection and its evaluation. 

---
# How to inject knowledge efficiently? Knowledge Infusion Scaling Law for Pre-training Large Language Models 

**Authors**: Kangtao Lv, Haibin Chen, Yujin Yuan, Langming Liu, Shilei Liu, Yongwei Wang, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19371)  

**Abstract**: Large language models (LLMs) have attracted significant attention due to their impressive general capabilities across diverse downstream tasks. However, without domain-specific optimization, they often underperform on specialized knowledge benchmarks and even produce hallucination. Recent studies show that strategically infusing domain knowledge during pretraining can substantially improve downstream performance. A critical challenge lies in balancing this infusion trade-off: injecting too little domain-specific data yields insufficient specialization, whereas excessive infusion triggers catastrophic forgetting of previously acquired knowledge. In this work, we focus on the phenomenon of memory collapse induced by over-infusion. Through systematic experiments, we make two key observations, i.e. 1) Critical collapse point: each model exhibits a threshold beyond which its knowledge retention capabilities sharply degrade. 2) Scale correlation: these collapse points scale consistently with the model's size. Building on these insights, we propose a knowledge infusion scaling law that predicts the optimal amount of domain knowledge to inject into large LLMs by analyzing their smaller counterparts. Extensive experiments across different model sizes and pertaining token budgets validate both the effectiveness and generalizability of our scaling law. 

---
# Meow: End-to-End Outline Writing for Automatic Academic Survey 

**Authors**: Zhaoyu Ma, Yuan Shan, Jiahao Zhao, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19370)  

**Abstract**: As academic paper publication numbers grow exponentially, conducting in-depth surveys with LLMs automatically has become an inevitable trend. Outline writing, which aims to systematically organize related works, is critical for automated survey generation. Yet existing automatic survey methods treat outline writing as mere workflow steps in the overall pipeline. Such template-based workflows produce outlines that lack in-depth understanding of the survey topic and fine-grained styles. To address these limitations, we propose Meow, the first metadata-driven outline writing framework that produces organized and faithful outlines efficiently. Specifically, we first formulate outline writing as an end-to-end task that generates hierarchical structured outlines from paper metadata. We then curate a high-quality dataset of surveys from arXiv, bioRxiv, and medRxiv, and establish systematic evaluation metrics for outline quality assessment. Finally, we employ a two-stage training approach combining supervised fine-tuning and reinforcement learning. Our 8B reasoning model demonstrates strong performance with high structural fidelity and stylistic coherence. 

---
# SLM-Based Agentic AI with P-C-G: Optimized for Korean Tool Use 

**Authors**: Changhyun Jeon, Jinhee Park, Jungwoo Choi, Keonwoo Kim, Jisu Kim, Minji Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.19369)  

**Abstract**: We propose a small-scale language model (SLM) based agent architecture, Planner-Caller-Generator (P-C-G), optimized for Korean tool use. P-C-G separates planning, calling, and generation by role: the Planner produces an initial batch plan with limited on-demand replanning; the Caller returns a normalized call object after joint schema-value validation; and the Generator integrates tool outputs to produce the final answer. We apply a Korean-first value policy to reduce execution failures caused by frequent Korean-to-English code switching in Korean settings. Evaluation assumes Korean queries and Korean tool/parameter specifications; it covers single-chain, multi-chain, missing-parameters, and missing-functions scenarios, and is conducted via an LLM-as-a-Judge protocol averaged over five runs under a unified I/O interface. Results show that P-C-G delivers competitive tool-use accuracy and end-to-end quality while reducing tokens and maintaining acceptable latency, indicating that role-specialized SLMs are a cost-effective alternative for Korean tool-use agents. 

---
# Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding 

**Authors**: Ruanjun Li, Ziheng Liu, Yuanming Shi, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19368)  

**Abstract**: Large language models (LLMs) deliver impressive generation quality, but incur very high inference cost because each output token is generated auto-regressively through all model layers. Early-exit based self-speculative decoding (EESD) has emerged to mitigate this cost. However, in practice, many approaches struggle to achieve the expected acceleration in such draft-then-verify paradigm even with a well-aligned early-exit head and selected exit position. Our analysis reveals that EESD only pays off when the vast majority of draft tokens are accepted by the LLM. Otherwise, the draft cost may overcome the acceleration gain and lead to a negative speedup. To mitigate this, we propose Pipeline-Parallel Self-Speculative Decoding (PPSD) that fully pipelines the draft and verification work so that no effort is wasted on failed predictions. It has two key innovations. We configure the model layers as a pipeline in which early-exit (draft) computations and remaining-layer (verification) computations overlap. We interleave drafting and verification per token. While the LLM is verifying the current token in its final layers, the early-exit path simultaneously drafts the next token. Such a verify-while-draft scheme keeps all units busy and validates tokens on-the-fly analogous to pipelining the speculation and verification stages. Empirical results confirm that PPSD achieves state-of-the-art acceleration in self-speculative LLM inference. On diverse benchmarks, PPSD achieves speedup ratios in the range of 2.01x~3.81x, which gains almost the optimal acceleration at the fixed acceptance rate and exit position, showcasing its advancement in providing efficient self-speculation. 

---
# Unsupervised Outlier Detection in Audit Analytics: A Case Study Using USA Spending Data 

**Authors**: Buhe Li, Berkay Kaplan, Maksym Lazirko, Aleksandr Kogan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19366)  

**Abstract**: This study investigates the effectiveness of unsupervised outlier detection methods in audit analytics, utilizing USA spending data from the U.S. Department of Health and Human Services (DHHS) as a case example. We employ and compare multiple outlier detection algorithms, including Histogram-based Outlier Score (HBOS), Robust Principal Component Analysis (PCA), Minimum Covariance Determinant (MCD), and K-Nearest Neighbors (KNN) to identify anomalies in federal spending patterns. The research addresses the growing need for efficient and accurate anomaly detection in large-scale governmental datasets, where traditional auditing methods may fall short. Our methodology involves data preparation, algorithm implementation, and performance evaluation using precision, recall, and F1 scores. Results indicate that a hybrid approach, combining multiple detection strategies, enhances the robustness and accuracy of outlier identification in complex financial data. This study contributes to the field of audit analytics by providing insights into the comparative effectiveness of various outlier detection models and demonstrating the potential of unsupervised learning techniques in improving audit quality and efficiency. The findings have implications for auditors, policymakers, and researchers seeking to leverage advanced analytics in governmental financial oversight and risk management. 

---
# The Inadequacy of Offline LLM Evaluations: A Need to Account for Personalization in Model Behavior 

**Authors**: Angelina Wang, Daniel E. Ho, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2509.19364)  

**Abstract**: Standard offline evaluations for language models -- a series of independent, state-less inferences made by models -- fail to capture how language models actually behave in practice, where personalization fundamentally alters model behavior. For instance, identical benchmark questions to the same language model can produce markedly different responses when prompted to a state-less system, in one user's chat session, or in a different user's chat session. In this work, we provide empirical evidence showcasing this phenomenon by comparing offline evaluations to field evaluations conducted by having 800 real users of ChatGPT and Gemini pose benchmark and other provided questions to their chat interfaces. 

---
# Analyzing the Impact of Credit Card Fraud on Economic Fluctuations of American Households Using an Adaptive Neuro-Fuzzy Inference System 

**Authors**: Zhuqi Wang, Qinghe Zhang, Zhuopei Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19363)  

**Abstract**: Credit card fraud is assuming growing proportions as a major threat to the financial position of American household, leading to unpredictable changes in household economic behavior. To solve this problem, in this paper, a new hybrid analysis method is presented by using the Enhanced ANFIS. The model proposes several advances of the conventional ANFIS framework and employs a multi-resolution wavelet decomposition module and a temporal attention mechanism. The model performs discrete wavelet transformations on historical transaction data and macroeconomic indicators to generate localized economic shock signals. The transformed features are then fed into a deep fuzzy rule library which is based on Takagi-Sugeno fuzzy rules with adaptive Gaussian membership functions. The model proposes a temporal attention encoder that adaptively assigns weights to multi-scale economic behavior patterns, increasing the effectiveness of relevance assessment in the fuzzy inference stage and enhancing the capture of long-term temporal dependencies and anomalies caused by fraudulent activities. The proposed method differs from classical ANFIS which has fixed input-output relations since it integrates fuzzy rule activation with the wavelet basis selection and the temporal correlation weights via a modular training procedure. Experimental results show that the RMSE was reduced by 17.8% compared with local neuro-fuzzy models and conventional LSTM models. 

---
# DeepACTIF: Efficient Feature Attribution via Activation Traces in Neural Sequence Models 

**Authors**: Benedikt W. Hosp  

**Link**: [PDF](https://arxiv.org/pdf/2509.19362)  

**Abstract**: Feature attribution is essential for interpreting deep learning models, particularly in time-series domains such as healthcare, biometrics, and human-AI interaction. However, standard attribution methods, such as Integrated Gradients or SHAP, are computationally intensive and not well-suited for real-time applications. We present DeepACTIF, a lightweight and architecture-aware feature attribution method that leverages internal activations of sequence models to estimate feature importance efficiently. Focusing on LSTM-based networks, we introduce an inverse-weighted aggregation scheme that emphasises stability and magnitude of activations across time steps. Our evaluation across three biometric gaze datasets shows that DeepACTIF not only preserves predictive performance under severe feature reduction (top 10% of features) but also significantly outperforms established methods, including SHAP, IG, and DeepLIFT, in terms of both accuracy and statistical robustness. Using Wilcoxon signed-rank tests and effect size analysis, we demonstrate that DeepACTIF yields more informative feature rankings with significantly lower error across all top-k conditions (10 - 40%). Our experiments demonstrate that DeepACTIF not only reduces computation time and memory usage by orders of magnitude but also preserves model accuracy when using only top-ranked features. That makes DeepACTIF a viable solution for real-time interpretability on edge devices such as mobile XR headsets or embedded health monitors. 

---
# Semantic Representation Attack against Aligned Large Language Models 

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19360)  

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.
Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.
We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.
Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.
This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.
The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.
We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.
Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.
The code will be publicly available. 

---
# Anti-Money Laundering Systems Using Deep Learning 

**Authors**: Mashkhal Abdalwahid Sidiq, Yimamu Kirubel Wondaferew  

**Link**: [PDF](https://arxiv.org/pdf/2509.19359)  

**Abstract**: In this paper, we focused on using deep learning methods for detecting money laundering in financial transaction networks, in order to demonstrate that it can be used as a complement or instead of the more commonly used rule-based systems and conventional Anti-Money Laundering (AML) systems. The paper explores the pivotal role played by Anti-Money Laundering (AML) activities in the global financial industry. It underscores the drawbacks of conventional AML systems, which exhibit high rates of false positives and lack the sophistication to uncover intricate money laundering schemes. To tackle these challenges, the paper proposes an advanced AML system that capitalizes on link analysis using deep learning techniques. At the heart of this system lies the utilization of centrality algorithms like Degree Centrality, Closeness Centrality, Betweenness Centrality, and PageRank. These algorithms enhance the system's capability to identify suspicious activities by examining the influence and interconnections within networks of financial transactions. The significance of Anti-Money Laundering (AML) efforts within the global financial sector is discussed in this paper. It highlights the limitations of traditional AML systems. The results showed the practicality and superiority of the new implementation of the GCN model, which is a preferable method for connectively structured data, meaning that a transaction or account is analyzed in the context of its financial environment. In addition, the paper delves into the prospects of Anti-Money Laundering (AML) efforts, proposing the integration of emerging technologies such as deep learning and centrality algorithms. This integration holds promise for enhancing the effectiveness of AML systems by refining their capabilities. 

---
# Benchmarking and Improving LLM Robustness for Personalized Generation 

**Authors**: Chimaobi Okite, Naihao Deng, Kiran Bodipati, Huaidian Hou, Joyce Chai, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2509.19358)  

**Abstract**: Recent years have witnessed a growing interest in personalizing the responses of large language models (LLMs). While existing evaluations primarily focus on whether a response aligns with a user's preferences, we argue that factuality is an equally important yet often overlooked dimension. In the context of personalization, we define a model as robust if its responses are both factually accurate and align with the user preferences. To assess this, we introduce PERG, a scalable framework for evaluating robustness in LLMs, along with a new dataset, PERGData. We evaluate fourteen models from five different model families using different prompting methods. Our findings show that current LLMs struggle with robust personalization: even the strongest models (GPT-4.1, LLaMA3-70B) fail to maintain correctness in 5% of previously successful cases without personalization, while smaller models (e.g., 7B-scale) can fail more than 20% of the time. Further analysis reveals that robustness is significantly affected by the nature of the query and the type of user preference. To mitigate these failures, we propose Pref-Aligner, a two-stage approach that improves robustness by an average of 25% across models. Our work highlights critical gaps in current evaluation practices and introduces tools and metrics to support more reliable, user-aligned LLM deployments. 

---
# RoadMind: Towards a Geospatial AI Expert for Disaster Response 

**Authors**: Ahmed El Fekih Zguir, Ferda Ofli, Muhammad Imran  

**Link**: [PDF](https://arxiv.org/pdf/2509.19354)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across a range of natural language tasks, but remain limited in their ability to reason about geospatial data, particularly road networks, distances, and directions. This gap poses challenges in disaster scenarios, where spatial understanding is critical for tasks such as evacuation planning and resource allocation. In this work, we present RoadMind, a self-supervised framework that enhances the geospatial reasoning capabilities of LLMs using structured data from OpenStreetMap (OSM). Our automated pipeline extracts road infrastructure data for a given city and converts it into multiple supervision formats tailored to key spatial tasks. We pretrain and fine-tune LLMs on these representations using QLoRA adapters and 4-bit quantized models. We evaluate our approach on three disaster-prone cities with varying global representation, Los Angeles, Christchurch, and Manila, across tasks such as road segment identification, nearest road retrieval, and distance/direction estimation. Our results show that models trained via RoadMind significantly outperform strong baselines, including state-of-the-art LLMs equipped with advanced prompt engineering. This demonstrates the potential of structured geospatial data to enhance language models with robust spatial reasoning, enabling more effective offline AI systems for disaster response. 

---
# TriSPrompt: A Hierarchical Soft Prompt Model for Multimodal Rumor Detection with Incomplete Modalities 

**Authors**: Jiajun Chen, Yangyang Wu, Xiaoye Miao, Mengying Zhu, Meng Xi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19352)  

**Abstract**: The widespread presence of incomplete modalities in multimodal data poses a significant challenge to achieving accurate rumor detection. Existing multimodal rumor detection methods primarily focus on learning joint modality representations from \emph{complete} multimodal training data, rendering them ineffective in addressing the common occurrence of \emph{missing modalities} in real-world scenarios. In this paper, we propose a hierarchical soft prompt model \textsf{TriSPrompt}, which integrates three types of prompts, \textit{i.e.}, \emph{modality-aware} (MA) prompt, \emph{modality-missing} (MM) prompt, and \emph{mutual-views} (MV) prompt, to effectively detect rumors in incomplete multimodal data. The MA prompt captures both heterogeneous information from specific modalities and homogeneous features from available data, aiding in modality recovery. The MM prompt models missing states in incomplete data, enhancing the model's adaptability to missing information. The MV prompt learns relationships between subjective (\textit{i.e.}, text and image) and objective (\textit{i.e.}, comments) perspectives, effectively detecting rumors. Extensive experiments on three real-world benchmarks demonstrate that \textsf{TriSPrompt} achieves an accuracy gain of over 13\% compared to state-of-the-art methods. The codes and datasets are available at https: //anonymous.this http URL. 

---
# The Impact of Structural Changes on Learning Capacity in the Fly Olfactory Neural Circuit 

**Authors**: Katherine Xie, Gabriel Koch Ocker  

**Link**: [PDF](https://arxiv.org/pdf/2509.19351)  

**Abstract**: The Drosophila mushroom body (MB) is known to be involved in olfactory learning and memory; the synaptic plasticity of the Kenyon cell (KC) to mushroom body output neuron (MBON) synapses plays a key role in the learning process. Previous research has focused on projection neuron (PN) to Kenyon cell (KC) connectivity within the MB; we examine how perturbations to the mushroom body circuit structure and changes in connectivity, specifically within the KC to mushroom body output neuron (MBON) neural circuit, affect the MBONs' ability to distinguish between odor classes. We constructed a neural network that incorporates the connectivity between PNs, KCs, and MBONs. To train our model, we generated ten artificial input classes, which represent the projection neuron activity in response to different odors. We collected data on the number of KC-to-MBON connections, MBON error rates, and KC-to-MBON synaptic weights, among other metrics. We observed that MBONs with very few presynaptic KCs consistently performed worse than others in the odor classification task. The developmental types of KCs also played a significant role in each MBON's output. We performed random and targeted KC ablation and observed that ablating developmentally mature KCs had a greater negative impact on MBONs' learning capacity than ablating immature KCs. Random and targeted pruning of KC-MBON synaptic connections yielded results largely consistent with the ablation experiments. To further explore the various types of KCs, we also performed rewiring experiments in the PN to KC circuit. Our study furthers our understanding of olfactory neuroplasticity and provides important clues to understanding learning and memory in general. Understanding how the olfactory circuits process and learn can also have potential applications in artificial intelligence and treatments for neurodegenerative diseases. 

---
# SCORE: A Semantic Evaluation Framework for Generative Document Parsing 

**Authors**: Renyu Li, Antonio Jimeno Yepes, Yao You, Kamil Pluciński, Maximilian Operlejn, Crag Wolfe  

**Link**: [PDF](https://arxiv.org/pdf/2509.19345)  

**Abstract**: Multi-modal generative document parsing systems challenge traditional evaluation: unlike deterministic OCR or layout models, they often produce semantically correct yet structurally divergent outputs. Conventional metrics-CER, WER, IoU, or TEDS-misclassify such diversity as error, penalizing valid interpretations and obscuring system behavior.
We introduce SCORE (Structural and COntent Robust Evaluation), an interpretation-agnostic framework that integrates (i) adjusted edit distance for robust content fidelity, (ii) token-level diagnostics to distinguish hallucinations from omissions, (iii) table evaluation with spatial tolerance and semantic alignment, and (iv) hierarchy-aware consistency checks. Together, these dimensions enable evaluation that embraces representational diversity while enforcing semantic rigor.
Across 1,114 pages spanning a holistic benchmark and a field dataset, SCORE consistently revealed cross-dataset performance patterns missed by standard metrics. In 2-5% of pages with ambiguous table structures, traditional metrics penalized systems by 12-25% on average, leading to distorted rankings. SCORE corrected these cases, recovering equivalence between alternative but valid interpretations. Moreover, by normalizing generative outputs into a format-agnostic representation, SCORE reproduces traditional scores (e.g., table F1 up to 0.93) without requiring object-detection pipelines, demonstrating that generative parsing alone suffices for comprehensive evaluation.
By exposing how interpretive diversity impacts evaluation outcomes and providing multi-dimensional, interpretable diagnostics, SCORE establishes foundational principles for semantically grounded, fair, and practical benchmarking of modern document parsing systems. 

---
# Part-of-speech tagging for Nagamese Language using CRF 

**Authors**: Alovi N Shohe, Chonglio Khiamungam, Teisovi Angami  

**Link**: [PDF](https://arxiv.org/pdf/2509.19343)  

**Abstract**: This paper investigates part-of-speech tagging, an important task in Natural Language Processing (NLP) for the Nagamese language. The Nagamese language, a.k.a. Naga Pidgin, is an Assamese-lexified Creole language developed primarily as a means of communication in trade between the Nagas and people from Assam in northeast India. A substantial amount of work in part-of-speech-tagging has been done for resource-rich languages like English, Hindi, etc. However, no work has been done in the Nagamese language. To the best of our knowledge, this is the first attempt at part-of-speech tagging for the Nagamese Language. The aim of this work is to identify the part-of-speech for a given sentence in the Nagamese language. An annotated corpus of 16,112 tokens is created and applied machine learning technique known as Conditional Random Fields (CRF). Using CRF, an overall tagging accuracy of 85.70%; precision, recall of 86%, and f1-score of 85% is achieved.
Keywords. Nagamese, NLP, part-of-speech, machine learning, CRF. 

---
# Fine-Grained AI Model Caching and Downloading With Coordinated Multipoint Broadcasting in Multi-Cell Edge Networks 

**Authors**: Yang Fu, Peng Qin, Yueyue Zhang, Yifei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19341)  

**Abstract**: 6G networks are envisioned to support on-demand AI model downloading to accommodate diverse inference requirements of end users. By proactively caching models at edge nodes, users can retrieve the requested models with low latency for on-device AI inference. However, the substantial size of contemporary AI models poses significant challenges for edge caching under limited storage capacity, as well as for the concurrent delivery of heterogeneous models over wireless channels. To address these challenges, we propose a fine-grained AI model caching and downloading system that exploits parameter reusability, stemming from the common practice of fine-tuning task-specific models from a shared pre-trained model with frozen parameters. This system selectively caches model parameter blocks (PBs) at edge nodes, eliminating redundant storage of reusable parameters across different cached models. Additionally, it incorporates coordinated multipoint (CoMP) broadcasting to simultaneously deliver reusable PBs to multiple users, thereby enhancing downlink spectrum utilization. Under this arrangement, we formulate a model downloading delay minimization problem to jointly optimize PB caching, migration (among edge nodes), and broadcasting beamforming. To tackle this intractable problem, we develop a distributed multi-agent learning framework that enables edge nodes to explicitly learn mutual influence among their actions, thereby facilitating cooperation. Furthermore, a data augmentation approach is proposed to adaptively generate synthetic training samples through a predictive model, boosting sample efficiency and accelerating policy learning. Both theoretical analysis and simulation experiments validate the superior convergence performance of the proposed learning framework. 

---
# Joint Channel Estimation and Computation Offloading in Fluid Antenna-assisted MEC Networks 

**Authors**: Ying Ju, Mingdong Li, Haoyu Wang, Lei Liu, Youyang Qu, Mianxiong Dong, Victor C. M. Leung, Chau Yuen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19340)  

**Abstract**: With the emergence of fluid antenna (FA) in wireless communications, the capability to dynamically adjust port positions offers substantial benefits in spatial diversity and spectrum efficiency, which are particularly valuable for mobile edge computing (MEC) systems. Therefore, we propose an FA-assisted MEC offloading framework to minimize system delay. This framework faces two severe challenges, which are the complexity of channel estimation due to dynamic port configuration and the inherent non-convexity of the joint optimization problem. Firstly, we propose Information Bottleneck Metric-enhanced Channel Compressed Sensing (IBM-CCS), which advances FA channel estimation by integrating information relevance into the sensing process and capturing key features of FA channels effectively. Secondly, to address the non-convex and high-dimensional optimization problem in FA-assisted MEC systems, which includes FA port selection, beamforming, power control, and resource allocation, we propose a game theory-assisted Hierarchical Twin-Dueling Multi-agent Algorithm (HiTDMA) based offloading scheme, where the hierarchical structure effectively decouples and coordinates the optimization tasks between the user side and the base station side. Crucially, the game theory effectively reduces the dimensionality of power control variables, allowing deep reinforcement learning (DRL) agents to achieve improved optimization efficiency. Numerical results confirm that the proposed scheme significantly reduces system delay and enhances offloading performance, outperforming benchmarks. Additionally, the IBM-CCS channel estimation demonstrates superior accuracy and robustness under varying port densities, contributing to efficient communication under imperfect CSI. 

---
# Multi-population Ensemble Genetic Programming via Cooperative Coevolution and Multi-view Learning for Classification 

**Authors**: Mohammad Sadegh Khorshidi, Navid Yazdanjue, Hassan Gharoun, Mohammad Reza Nikoo, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19339)  

**Abstract**: This paper introduces Multi-population Ensemble Genetic Programming (MEGP), a computational intelligence framework that integrates cooperative coevolution and the multiview learning paradigm to address classification challenges in high-dimensional and heterogeneous feature spaces. MEGP decomposes the input space into conditionally independent feature subsets, enabling multiple subpopulations to evolve in parallel while interacting through a dynamic ensemble-based fitness mechanism. Each individual encodes multiple genes whose outputs are aggregated via a differentiable softmax-based weighting layer, enhancing both model interpretability and adaptive decision fusion. A hybrid selection mechanism incorporating both isolated and ensemble-level fitness promotes inter-population cooperation while preserving intra-population diversity. This dual-level evolutionary dynamic facilitates structured search exploration and reduces premature convergence. Experimental evaluations across eight benchmark datasets demonstrate that MEGP consistently outperforms a baseline GP model in terms of convergence behavior and generalization performance. Comprehensive statistical analyses validate significant improvements in Log-Loss, Precision, Recall, F1 score, and AUC. MEGP also exhibits robust diversity retention and accelerated fitness gains throughout evolution, highlighting its effectiveness for scalable, ensemble-driven evolutionary learning. By unifying population-based optimization, multi-view representation learning, and cooperative coevolution, MEGP contributes a structurally adaptive and interpretable framework that advances emerging directions in evolutionary machine learning. 

---
# Radio Propagation Modelling: To Differentiate or To Deep Learn, That Is The Question 

**Authors**: Stefanos Bakirtzis, Paul Almasan, José Suárez-Varela, Gabriel O. Ferreira, Michail Kalntis, André Felipe Zanella, Ian Wassell, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19337)  

**Abstract**: Differentiable ray tracing has recently challenged the status quo in radio propagation modelling and digital twinning. Promising unprecedented speed and the ability to learn from real-world data, it offers a real alternative to conventional deep learning (DL) models. However, no experimental evaluation on production-grade networks has yet validated its assumed scalability or practical benefits. This leaves mobile network operators (MNOs) and the research community without clear guidance on its applicability. In this paper, we fill this gap by employing both differentiable ray tracing and DL models to emulate radio coverage using extensive real-world data collected from the network of a major MNO, covering 13 cities and more than 10,000 antennas. Our results show that, while differentiable ray-tracing simulators have contributed to reducing the efficiency-accuracy gap, they struggle to generalize from real-world data at a large scale, and they remain unsuitable for real-time applications. In contrast, DL models demonstrate higher accuracy and faster adaptation than differentiable ray-tracing simulators across urban, suburban, and rural deployments, achieving accuracy gains of up to 3 dB. Our experimental results aim to provide timely insights into a fundamental open question with direct implications on the wireless ecosystem and future research. 

---
# Cognitive-Level Adaptive Generation via Capability-Aware Retrieval and Style Adaptation 

**Authors**: Qingsong Wang, Tao Wu, Wang Lin, Yueying Feng, Gongsheng Yuan, Chang Yao, Jingyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19336)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in open-ended generation tasks. However, they often struggle to adapt content to users with differing cognitive capacities, leading to a phenomenon we term cognitive misalignment. This issue arises in two forms: knowledge-level misalignment, where content is too complex or too simplistic relative to user understanding, and presentation-style misalignment, where the structure or tone hinders effective comprehension. To address these challenges, we propose the Cognitive-Level Alignment Framework (CLAF), a general-purpose generation framework that aligns both knowledge complexity and presentation style with user cognition. CLAF integrates a capability-aware retrieval module based on a hierarchical knowledge graph and a style optimization module guided by Bloom's taxonomy and preference learning. Additionally, a knowledge-controllable generation component ensures consistency and relevance throughout the output. To support training and evaluation, we construct SCALE, a cognitively annotated dataset containing responses at multiple comprehension levels per query. Empirical results show that CLAF enhances the adaptability and informativeness of LLM outputs across a range of user profiles, offering a robust solution to cognitive-level alignment in real-world applications. 

---
# CSIYOLO: An Intelligent CSI-based Scatter Sensing Framework for Integrated Sensing and Communication Systems 

**Authors**: Xudong Zhang, Jingbo Tan, Zhizhen Ren, Jintao Wang, Yihua Ma, Jian Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.19335)  

**Abstract**: ISAC is regarded as a promising technology for next-generation communication systems, enabling simultaneous data transmission and target sensing. Among various tasks in ISAC, scatter sensing plays a crucial role in exploiting the full potential of ISAC and supporting applications such as autonomous driving and low-altitude economy. However, most existing methods rely on either waveform and hardware modifications or traditional signal processing schemes, leading to poor compatibility with current communication systems and limited sensing accuracy. To address these challenges, we propose CSIYOLO, a framework that performs scatter localization only using estimated CSI from a single base station-user equipment pair. This framework comprises two main components: anchor-based scatter parameter detection and CSI-based scatter localization. First, by formulating scatter parameter extraction as an image detection problem, we propose an anchor-based scatter parameter detection method inspired by You Only Look Once architectures. After that, a CSI-based localization algorithm is derived to determine scatter locations with extracted parameters. Moreover, to improve localization accuracy and implementation efficiency, we design an extendable network structure with task-oriented optimizations, enabling multi-scale anchor detection and better adaptation to CSI characteristics. A noise injection training strategy is further designed to enhance robustness against channel estimation errors. Since the proposed framework operates solely on estimated CSI without modifying waveforms or signal processing pipelines, it can be seamlessly integrated into existing communication systems as a plugin. Experiments show that our proposed method can significantly outperform existing methods in scatter localization accuracy with relatively low complexities under varying numbers of scatters and estimation errors. 

---
# Pluralistic Off-policy Evaluation and Alignment 

**Authors**: Chengkai Huang, Junda Wu, Zhouhang Xie, Yu Xia, Rui Wang, Tong Yu, Subrata Mitra, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19333)  

**Abstract**: Personalized preference alignment for LLMs with diverse human preferences requires evaluation and alignment methods that capture pluralism. Most existing preference alignment datasets are logged under policies that differ substantially from the evaluated LLMs, and existing off-policy estimators focus solely on overall utility while ignoring preference pluralism. Extending Off-Policy Evaluation (OPE) to pluralistic preference alignment, therefore, remains an open question. Thus, we propose the Pluralistic Off-Policy Evaluation (POPE), the first framework for offline pluralistic preference evaluation and alignment in LLMs. POPE includes a unified reward function that combines (1) a collaborative utility component derived from human preference signals (e.g., upvotes or relevance scores) and (2) a diversity component inspired by entropy-based coverage measures, together reflecting pluralistic alignment. Furthermore, to estimate this reward from logged interactions, we derive decomposable inverse propensity scoring (IPS) estimators that separately evaluate relevance and diversity. Theoretically, we prove that our decomposed IPS estimators establish a lower bound on their variance. With the off-policy evaluated value function, we can directly enable off-policy optimization to further enhance pluralistic alignment. Empirical results demonstrate that POPE efficiently enhances pluralistic response generation and maintains the models' general capabilities on downstream tasks 

---
# Quantifying Compositionality of Classic and State-of-the-Art Embeddings 

**Authors**: Zhijin Guo, Chenhao Xue, Zhaozhen Xu, Hongbo Bo, Yuxuan Ye, Janet B. Pierrehumbert, Martha Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2509.19332)  

**Abstract**: For language models to generalize correctly to novel expressions, it is critical that they exploit access compositional meanings when this is justified. Even if we don't know what a "pelp" is, we can use our knowledge of numbers to understand that "ten pelps" makes more pelps than "two pelps". Static word embeddings such as Word2vec made strong, indeed excessive, claims about compositionality. The SOTA generative, transformer models and graph models, however, go too far in the other direction by providing no real limits on shifts in meaning due to context. To quantify the additive compositionality, we formalize a two-step, generalized evaluation that (i) measures the linearity between known entity attributes and their embeddings via canonical correlation analysis, and (ii) evaluates additive generalization by reconstructing embeddings for unseen attribute combinations and checking reconstruction metrics such as L2 loss, cosine similarity, and retrieval accuracy. These metrics also capture failure cases where linear composition breaks down. Sentences, knowledge graphs, and word embeddings are evaluated and tracked the compositionality across all layers and training stages. Stronger compositional signals are observed in later training stages across data modalities, and in deeper layers of the transformer-based model before a decline at the top layer. Code is available at this https URL. 

---
# Holographic Transformers for Complex-Valued Signal Processing: Integrating Phase Interference into Self-Attention 

**Authors**: Enhao Huang, Zhiyu Zhang, Tianxiang Xu, Chunshu Xia, Kaichun Hu, Yuchen Yang, Tongtong Pan, Dong Dong, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.19331)  

**Abstract**: Complex-valued signals encode both amplitude and phase, yet most deep models treat attention as real-valued correlation, overlooking interference effects. We introduce the Holographic Transformer, a physics-inspired architecture that incorporates wave interference principles into self-attention. Holographic attention modulates interactions by relative phase and coherently superimposes values, ensuring consistency between amplitude and phase. A dual-headed decoder simultaneously reconstructs the input and predicts task outputs, preventing phase collapse when losses prioritize magnitude over phase. We demonstrate that holographic attention implements a discrete interference operator and maintains phase consistency under linear mixing. Experiments on PolSAR image classification and wireless channel prediction show strong performance, achieving high classification accuracy and F1 scores, low regression error, and increased robustness to phase perturbations. These results highlight that enforcing physical consistency in attention leads to generalizable improvements in complex-valued learning and provides a unified, physics-based framework for coherent signal modeling. The code is available at this https URL. 

---
# LibEMER: A novel benchmark and algorithms library for EEG-based Multimodal Emotion Recognition 

**Authors**: Zejun Liu, Yunshan Chen, Chengxi Xie, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19330)  

**Abstract**: EEG-based multimodal emotion recognition(EMER) has gained significant attention and witnessed notable advancements, the inherent complexity of human neural systems has motivated substantial efforts toward multimodal approaches. However, this field currently suffers from three critical limitations: (i) the absence of open-source implementations. (ii) the lack of standardized and transparent benchmarks for fair performance analysis. (iii) in-depth discussion regarding main challenges and promising research directions is a notable scarcity. To address these challenges, we introduce LibEMER, a unified evaluation framework that provides fully reproducible PyTorch implementations of curated deep learning methods alongside standardized protocols for data preprocessing, model realization, and experimental setups. This framework enables unbiased performance assessment on three widely-used public datasets across two learning tasks. The open-source library is publicly accessible at: this https URL 

---
# Human Activity Recognition Based on Electrocardiogram Data Only 

**Authors**: Sina Montazeri, Waltenegus Dargie, Yunhe Feng, Kewei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2509.19328)  

**Abstract**: Human activity recognition is critical for applications such as early intervention and health analytics. Traditional activity recognition relies on inertial measurement units (IMUs), which are resource intensive and require calibration. Although electrocardiogram (ECG)-based methods have been explored, these have typically served as supplements to IMUs or have been limited to broad categorical classification such as fall detection or active vs. inactive in daily activities. In this paper, we advance the field by demonstrating, for the first time, robust recognition of activity only with ECG in six distinct activities, which is beyond the scope of previous work. We design and evaluate three new deep learning models, including a CNN classifier with Squeeze-and-Excitation blocks for channel-wise feature recalibration, a ResNet classifier with dilated convolutions for multiscale temporal dependency capture, and a novel CNNTransformer hybrid combining convolutional feature extraction with attention mechanisms for long-range temporal relationship modeling. Tested on data from 54 subjects for six activities, all three models achieve over 94% accuracy for seen subjects, while CNNTransformer hybrid reaching the best accuracy of 72% for unseen subjects, a result that can be further improved by increasing the training population. This study demonstrates the first successful ECG-only activity classification in multiple physical activities, offering significant potential for developing next-generation wearables capable of simultaneous cardiac monitoring and activity recognition without additional motion sensors. 

---
# A systematic review of trial-matching pipelines using large language models 

**Authors**: Braxton A. Morrison, Madhumita Sushil, Jacob S. Young  

**Link**: [PDF](https://arxiv.org/pdf/2509.19327)  

**Abstract**: Matching patients to clinical trial options is critical for identifying novel treatments, especially in oncology. However, manual matching is labor-intensive and error-prone, leading to recruitment delays. Pipelines incorporating large language models (LLMs) offer a promising solution. We conducted a systematic review of studies published between 2020 and 2025 from three academic databases and one preprint server, identifying LLM-based approaches to clinical trial matching. Of 126 unique articles, 31 met inclusion criteria. Reviewed studies focused on matching patient-to-criterion only (n=4), patient-to-trial only (n=10), trial-to-patient only (n=2), binary eligibility classification only (n=1) or combined tasks (n=14). Sixteen used synthetic data; fourteen used real patient data; one used both. Variability in datasets and evaluation metrics limited cross-study comparability. In studies with direct comparisons, the GPT-4 model consistently outperformed other models, even finely-tuned ones, in matching and eligibility extraction, albeit at higher cost. Promising strategies included zero-shot prompting with proprietary LLMs like the GPT-4o model, advanced retrieval methods, and fine-tuning smaller, open-source models for data privacy when incorporation of large models into hospital infrastructure is infeasible. Key challenges include accessing sufficiently large real-world data sets, and deployment-associated challenges such as reducing cost, mitigating risk of hallucinations, data leakage, and bias. This review synthesizes progress in applying LLMs to clinical trial matching, highlighting promising directions and key limitations. Standardized metrics, more realistic test sets, and attention to cost-efficiency and fairness will be critical for broader deployment. 

---
# Unveiling the Merits and Defects of LLMs in Automatic Review Generation for Scientific Papers 

**Authors**: Ruochi Li, Haoxuan Zhang, Edward Gehringer, Ting Xiao, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19326)  

**Abstract**: The surge in scientific submissions has placed increasing strain on the traditional peer-review process, prompting the exploration of large language models (LLMs) for automated review generation. While LLMs demonstrate competence in producing structured and coherent feedback, their capacity for critical reasoning, contextual grounding, and quality sensitivity remains limited. To systematically evaluate these aspects, we propose a comprehensive evaluation framework that integrates semantic similarity analysis and structured knowledge graph metrics to assess LLM-generated reviews against human-written counterparts. We construct a large-scale benchmark of 1,683 papers and 6,495 expert reviews from ICLR and NeurIPS in multiple years, and generate reviews using five LLMs. Our findings show that LLMs perform well in descriptive and affirmational content, capturing the main contributions and methodologies of the original work, with GPT-4o highlighted as an illustrative example, generating 15.74% more entities than human reviewers in the strengths section of good papers in ICLR 2025. However, they consistently underperform in identifying weaknesses, raising substantive questions, and adjusting feedback based on paper quality. GPT-4o produces 59.42% fewer entities than real reviewers in the weaknesses and increases node count by only 5.7% from good to weak papers, compared to 50% in human reviews. Similar trends are observed across all conferences, years, and models, providing empirical foundations for understanding the merits and defects of LLM-generated reviews and informing the development of future LLM-assisted reviewing tools. Data, code, and more detailed results are publicly available at this https URL. 

---
# Magnitude Matters: a Superior Class of Similarity Metrics for Holistic Semantic Understanding 

**Authors**: V.S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19323)  

**Abstract**: Vector comparison in high dimensions is a fundamental task in NLP, yet it is dominated by two baselines: the raw dot product, which is unbounded and sensitive to vector norms, and the cosine similarity, which discards magnitude information entirely. This paper challenges both standards by proposing and rigorously evaluating a new class of parameter-free, magnitude-aware similarity metrics. I introduce two such functions, Overlap Similarity (OS) and Hyperbolic Tangent Similarity (HTS), designed to integrate vector magnitude and alignment in a more principled manner. To ensure that my findings are robust and generalizable, I conducted a comprehensive evaluation using four state-of-the-art sentence embedding models (all-MiniLM-L6-v2, all-mpnet-base-v2, paraphrase-mpnet-base-v2, and BAAI/bge-large-en-v1.5) across a diverse suite of eight standard NLP benchmarks, including STS-B, SICK, Quora, and PAWS. Using the Wilcoxon signed-rank test for statistical significance, my results are definitive: on the tasks requiring holistic semantic understanding (paraphrase and inference), both OS and HTS provide a statistically significant improvement in Mean Squared Error over both the raw dot product and cosine similarity, regardless of the underlying embedding this http URL, my findings delineate the specific domain of advantage for these metrics: for tasks requiring holistic semantic understanding like paraphrase and inference, my magnitude-aware metrics offer a statistically superior alternative. This significant improvement was not observed on benchmarks designed to test highly nuanced compositional semantics (SICK, STS-B), identifying the challenge of representing compositional text as a distinct and important direction for future work. 

---
# Readme_AI: Dynamic Context Construction for Large Language Models 

**Authors**: Millie Vyas, Timothy Blattner, Alden Dima  

**Link**: [PDF](https://arxiv.org/pdf/2509.19322)  

**Abstract**: Despite being trained on significant amounts of data, Large Language Models (LLMs) can provide inaccurate or unreliable information in the context of a user's specific query. Given query-specific context significantly improves the usefulness of its responses. In this paper, we present a specification that can be used to dynamically build context for data sources. The data source owner creates the file containing metadata for LLMs to use when reasoning about dataset-related queries. To demonstrate our proposed specification, we created a prototype Readme_AI Model Context Protocol (MCP) server that retrieves the metadata from the data source and uses it to dynamically build context. Some features that make this specification dynamic are the extensible types that represent crawling web-pages, fetching data from data repositories, downloading and parsing publications, and general text. The context is formatted and grouped using user-specified tags that provide clear contextual information for the LLM to reason about the content. We demonstrate the capabilities of this early prototype by asking the LLM about the NIST-developed Hedgehog library, for which common LLMs often provides inaccurate and irrelevant responses containing hallucinations. With Readme_AI, the LLM receives enough context that it is now able to reason about the library and its use, and even generate code interpolated from examples that were included in the Readme_AI file provided by Hedgehog's developer. Our primary contribution is a extensible protocol for dynamically grounding LLMs in specialized, owner-provided data, enhancing responses from LLMs and reducing hallucinations. The source code for the Readme_AI tool is posted here: this https URL . 

---
# FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering 

**Authors**: Gyubok Lee, Elea Bach, Eric Yang, Tom Pollard, Alistair Johnson, Edward Choi, Yugang jia, Jong Ha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19319)  

**Abstract**: The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (this https URL) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications. 

---
# Advancing Few-Shot Pediatric Arrhythmia Classification with a Novel Contrastive Loss and Multimodal Learning 

**Authors**: Yiqiao Chen, Zijian Huang, Zhenghui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2509.19315)  

**Abstract**: Pediatric arrhythmias are a major risk factor for disability and sudden cardiac death, yet their automated classification remains challenging due to class imbalance, few-shot categories, and complex signal characteristics, which severely limit the efficiency and reliability of early screening and clinical intervention. To address this problem, we propose a multimodal end-to-end deep learning framework that combines dual-branch convolutional encoders for ECG and IEGM, semantic attention for cross-modal feature alignment, and a lightweight Transformer encoder for global dependency modeling. In addition, we introduce a new contrastive loss fucntion named Adaptive Global Class-Aware Contrastive Loss (AGCACL) to enhance intra-class compactness and inter-class separability through class prototypes and a global similarity matrix. To the best of our knowledge, this is the first systematic study based on the Leipzig Heart Center pediatric/congenital ECG+IEGM dataset, for which we also provide a complete and reproducible preprocessing pipeline. Experimental results demonstrate that the proposed method achieves the overall best performance on this dataset, including 97.76\% Top-1 Accuracy, 94.08\% Macro Precision, 91.97\% Macro Recall, 92.97\% Macro F1, and 92.36\% Macro F2, with improvements of +13.64, +15.96, +19.82, and +19.44 percentage points over the strongest baseline in Macro Precision/Recall/F1/F2, respectively. These findings indicate that the framework significantly improves the detectability and robustness for minority arrhythmia classes, offering potential clinical value for rhythm screening, pre-procedural assessment, and postoperative follow-up in pediatric and congenital heart disease populations. 

---
# Automated Item Neutralization for Non-Cognitive Scales: A Large Language Model Approach to Reducing Social-Desirability Bias 

**Authors**: Sirui Wu, Daijin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19314)  

**Abstract**: This study evaluates item neutralization assisted by the large language model (LLM) to reduce social desirability bias in personality assessment. GPT-o3 was used to rewrite the International Personality Item Pool Big Five Measure (IPIP-BFM-50), and 203 participants completed either the original or neutralized form along with the Marlowe-Crowne Social Desirability Scale. The results showed preserved reliability and a five-factor structure, with gains in Conscientiousness and declines in Agreeableness and Openness. The correlations with social desirability decreased for several items, but inconsistently. Configural invariance held, though metric and scalar invariance failed. Findings support AI neutralization as a potential but imperfect bias-reduction method. 

---
# E2E Learning Massive MIMO for Multimodal Semantic Non-Orthogonal Transmission and Fusion 

**Authors**: Minghui Wu, Zhen Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19312)  

**Abstract**: Massive multiple-input multiple-output (MIMO) promises high spectral efficiency but also leads to high-dimensional downlink channel state information (CSI), which complicates real-time channel acquisition and precoding. To address this, we propose an end-to-end (E2E) uplink-downlink CSI fusion precoding network that jointly models downlink CSI reference signal (CSI-RS) design, CSI feedback, and base-station (BS) precoding within a single E2E neural architecture. Concretely, a projection network built on the MAXIM architecture takes uplink sounding reference signals (SRS) as input and outputs frequency-, beam-, and port-domain projection matrices for designing downlink CSI-RS. User equipment (UE) then compresses/quantizes the resulting CSI-RS observations and feeds back a compact representation. At the base station (BS), two complementary branches produce candidate precoders: one is a feedback-only precoding network driven by quantized downlink observations, and the other is an SRS-only precoding network driven by uplink SRS. These candidate precoders are subsequently combined by a fusion precoding network to yield the final transmit precoder. All the modules are trained with a spectral-efficiency-oriented loss under a three-stage schedule. Simulation results show that the proposed approach effectively harnesses both SRS-derived information and UE feedback, achieving markedly better performance than conventional baselines. 

---
# A Federated Fine-Tuning Paradigm of Foundation Models in Heterogenous Wireless Networks 

**Authors**: Jingyi Wang, Zhongyuan Zhao, Qingtian Wang, Zexu Li, Yue Wang, Tony Q. S. Quek  

**Link**: [PDF](https://arxiv.org/pdf/2509.19306)  

**Abstract**: Edge intelligence has emerged as a promising strategy to deliver low-latency and ubiquitous services for mobile devices. Recent advances in fine-tuning mechanisms of foundation models have enabled edge intelligence by integrating low-rank adaptation (LoRA) with federated learning. However, in wireless networks, the device heterogeneity and resource constraints on edge devices pose great threats to the performance of federated fine-tuning. To tackle these issues, we propose to optimize federated fine-tuning in heterogenous wireless networks via online learning. First, the framework of switching-based federated fine-tuning in wireless networks is provided. The edge devices switches to LoRA modules dynamically for federated fine-tuning with base station to jointly mitigate the impact of device heterogeneity and transmission unreliability. Second, a tractable upper bound on the inference risk gap is derived based on theoretical analysis. To improve the generalization capability, we formulate a non-convex mixed-integer programming problem with long-term constraints, and decouple it into model switching, transmit power control, and bandwidth allocation subproblems. An online optimization algorithm is developed to solve the problems with polynomial computational complexity. Finally, the simulation results on the SST-2 and QNLI data sets demonstrate the performance gains in test accuracy and energy efficiency. 

---
# Wavelet Fourier Diffuser: Frequency-Aware Diffusion Model for Reinforcement Learning 

**Authors**: Yifu Luo, Yongzhe Chang, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19305)  

**Abstract**: Diffusion probability models have shown significant promise in offline reinforcement learning by directly modeling trajectory sequences. However, existing approaches primarily focus on time-domain features while overlooking frequency-domain features, leading to frequency shift and degraded performance according to our observation. In this paper, we investigate the RL problem from a new perspective of the frequency domain. We first observe that time-domain-only approaches inadvertently introduce shifts in the low-frequency components of the frequency domain, which results in trajectory instability and degraded performance. To address this issue, we propose Wavelet Fourier Diffuser (WFDiffuser), a novel diffusion-based RL framework that integrates Discrete Wavelet Transform to decompose trajectories into low- and high-frequency components. To further enhance diffusion modeling for each component, WFDiffuser employs Short-Time Fourier Transform and cross attention mechanisms to extract frequency-domain features and facilitate cross-frequency interaction. Extensive experiment results on the D4RL benchmark demonstrate that WFDiffuser effectively mitigates frequency shift, leading to smoother, more stable trajectories and improved decision-making performance over existing methods. 

---
# LLMs as verification oracles for Solidity 

**Authors**: Massimo Bartoletti, Enrico Lipparini, Livio Pompianu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19153)  

**Abstract**: Ensuring the correctness of smart contracts is critical, as even subtle flaws can lead to severe financial losses. While bug detection tools able to spot common vulnerability patterns can serve as a first line of defense, most real-world exploits and losses stem from errors in the contract business logic. Formal verification tools such as SolCMC and the Certora Prover address this challenge, but their impact remains limited by steep learning curves and restricted specification languages. Recent works have begun to explore the use of large language models (LLMs) for security-related tasks such as vulnerability detection and test generation. Yet, a fundamental question remains open: can LLMs serve as verification oracles, capable of reasoning about arbitrary contract-specific properties? In this paper, we provide the first systematic evaluation of GPT-5, a state-of-the-art reasoning LLM, in this role. We benchmark its performance on a large dataset of verification tasks, compare its outputs against those of established formal verification tools, and assess its practical effectiveness in real-world auditing scenarios. Our study combines quantitative metrics with qualitative analysis, and shows that recent reasoning-oriented LLMs can be surprisingly effective as verification oracles, suggesting a new frontier in the convergence of AI and formal methods for secure smart contract development and auditing. 

---
# GAUSS: Benchmarking Structured Mathematical Skills for Large Language Models 

**Authors**: Yue Zhang, Jiaxin Zhang, Qiuyu Ren, Tahsin Saffat, Xiaoxuan Liu, Zitong Yang, Banghua Zhu, Yi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.18122)  

**Abstract**: We introduce \textbf{GAUSS} (\textbf{G}eneral \textbf{A}ssessment of \textbf{U}nderlying \textbf{S}tructured \textbf{S}kills in Mathematics), a benchmark that evaluates LLMs' mathematical abilities across twelve core skill dimensions, grouped into three domains: knowledge and understanding, problem solving and communication, and meta-skills and creativity. By categorizing problems according to cognitive skills and designing tasks that isolate specific abilities, GAUSS constructs comprehensive, fine-grained, and interpretable profiles of models' mathematical abilities. These profiles faithfully represent their underlying mathematical intelligence. To exemplify how to use the \textsc{GAUSS} benchmark, we have derived the skill profile of \textsc{GPT-5-thinking}, revealing its strengths and weaknesses as well as its differences relative to \textsc{o4-mini-high}, thereby underscoring the value of multidimensional, skill-based evaluation. 

---
