# QED-Nano: Teaching a Tiny Model to Prove Hard Theorems 

**Authors**: LM-Provers, Yuxiao Qu, Amrith Setlur, Jasper Dekoninck, Edward Beeching, Jia Li, Ian Wu, Lewis Tunstall, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04898)  

**Abstract**: Proprietary AI systems have recently demonstrated impressive capabilities on complex proof-based problems, with gold-level performance reported at the 2025 International Mathematical Olympiad (IMO). However, the training pipelines behind these systems remain largely undisclosed, and their reliance on large "internal" models and scaffolds makes them expensive to run, difficult to reproduce, and hard to study or improve upon. This raises a central question: can small, open models also be trained to achieve competitive reasoning performance on difficult Olympiad-level math? In this paper, we answer this question by building QED-Nano, a 4B model post-trained for Olympiad-level proofs. Our training recipe has three stages: (1) supervised fine-tuning to imbue good proof-writing styles by distilling from DeepSeek-Math-V2, (2) reinforcement learning (RL) with rubric-based rewards, and (3) expanding RL with a reasoning cache, which decomposes long proofs into iterative summarize-and-refine cycles and enables stronger test-time reasoning. QED-Nano surpasses the proof-generation performance of much larger open models, including Nomos-1 and GPT-OSS-120B, and approaches the performance of proprietary models like Gemini 3 Pro, at a fraction of the inference cost. To support further research on open mathematical reasoning, we release the full QED-Nano pipeline, including the QED-Nano and QED-Nano-SFT models, the FineProofs-SFT and FineProofs-RL datasets, and the training and evaluation code. 

---
# Learning, Potential, and Retention: An Approach for Evaluating Adaptive AI-Enabled Medical Devices 

**Authors**: Alexis Burgon, Berkman Sahiner, Nicholas A Petrick, Gene Pennello, Ravi K Samala  

**Link**: [PDF](https://arxiv.org/pdf/2604.04878)  

**Abstract**: This work addresses challenges in evaluating adaptive artificial intelligence (AI) models for medical devices, where iterative updates to both models and evaluation datasets complicate performance assessment. We introduce a novel approach with three complementary measurements: learning (model improvement on current data), potential (dataset-driven performance shifts), and retention (knowledge preservation across modification steps), to disentangle performance changes caused by model adaptations versus dynamic environments. Case studies using simulated population shifts demonstrate the approach's utility: gradual transitions enable stable learning and retention, while rapid shifts reveal trade-offs between plasticity and stability. These measurements provide practical insights for regulatory science, enabling rigorous assessment of the safety and effectiveness of adaptive AI systems over sequential modifications. 

---
# Incompleteness of AI Safety Verification via Kolmogorov Complexity 

**Authors**: Munawar Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04876)  

**Abstract**: Ensuring that artificial intelligence (AI) systems satisfy formal safety and policy constraints is a central challenge in safety-critical domains. While limitations of verification are often attributed to combinatorial complexity and model expressiveness, we show that they arise from intrinsic information-theoretic limits. We formalize policy compliance as a verification problem over encoded system behaviors and analyze it using Kolmogorov complexity. We prove an incompleteness result: for any fixed sound computably enumerable verifier, there exists a threshold beyond which true policy-compliant instances cannot be certified once their complexity exceeds that threshold. Consequently, no finite formal verifier can certify all policy-compliant instances of arbitrarily high complexity. This reveals a fundamental limitation of AI safety verification independent of computational resources, and motivates proof-carrying approaches that provide instance-level correctness guarantees. 

---
# MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents 

**Authors**: Shu Wang, Edwin Yu, Oscar Love, Tom Zhang, Tom Wong, Steve Scargall, Charles Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04853)  

**Abstract**: Large Language Model (LLM) agents require persistent memory to maintain personalization, factual continuity, and long-horizon reasoning, yet standard context-window and retrieval-augmented generation (RAG) pipelines degrade over multi-session interactions. We present MemMachine, an open-source memory system that integrates short-term, long-term episodic, and profile memory within a ground-truth-preserving architecture that stores entire conversational episodes and reduces lossy LLM-based extraction. MemMachine uses contextualized retrieval that expands nucleus matches with surrounding context, improving recall when relevant evidence spans multiple dialogue turns. Across benchmarks, MemMachine achieves strong accuracy-efficiency tradeoffs: on LoCoMo it reaches 0.9169 using gpt4.1-mini; on LongMemEvalS (ICLR 2025), a six-dimension ablation yields 93.0 percent accuracy, with retrieval-stage optimizations -- retrieval depth tuning (+4.2 percent), context formatting (+2.0 percent), search prompt design (+1.8 percent), and query bias correction (+1.4 percent) -- outperforming ingestion-stage gains such as sentence chunking (+0.8 percent). GPT-5-mini exceeds GPT-5 by 2.6 percent when paired with optimized prompts, making it the most cost-efficient setup. Compared to Mem0, MemMachine uses roughly 80 percent fewer input tokens under matched conditions. A companion Retrieval Agent adaptively routes queries among direct retrieval, parallel decomposition, or iterative chain-of-query strategies, achieving 93.2 percent on HotpotQA-hard and 92.6 percent on WikiMultiHop under randomized-noise conditions. These results show that preserving episodic ground truth while layering adaptive retrieval yields robust, efficient long-term memory for personalized LLM agents. 

---
# ANX: Protocol-First Design for AI Agent Interaction with a Supporting 3EX Decoupled Architecture 

**Authors**: Xu Mingze  

**Link**: [PDF](https://arxiv.org/pdf/2604.04820)  

**Abstract**: AI agents, autonomous digital actors, need agent-native protocols; existing methods include GUI automation and MCP-based skills, with defects of high token consumption, fragmented interaction, inadequate security, due to lacking a unified top-level framework and key components, each independent module flawed. To address these issues, we present ANX, an open, extensible, verifiable agent-native protocol and top-level framework integrating CLI, Skill, MCP, resolving pain points via protocol innovation, architectural optimization and tool supplementation. Its four core innovations: 1) Agent-native design (ANX Config, Markup, CLI) with high information density, flexibility and strong adaptability to reduce tokens and eliminate inconsistencies; 2) Human-agent interaction combining Skill's flexibility for dual rendering as agent-executable instructions and human-readable UI; 3) MCP-supported on-demand lightweight apps without pre-registration; 4) ANX Markup-enabled machine-executable SOPs eliminating ambiguity for reliable long-horizon tasks and multi-agent collaboration. As the first in a series, we focus on ANX's design, present its 3EX decoupled architecture with ANXHub and preliminary feasibility analysis and experimental validation. ANX ensures native security: LLM-bypassed UI-to-Core communication keeps sensitive data out of agent context; human-only confirmation prevents automated misuse. Form-filling experiments with Qwen3.5-plus/GPT-4o show ANX reduces tokens by 47.3% (Qwen3.5-plus) and 55.6% (GPT-4o) vs MCP-based skills, 57.1% (Qwen3.5-plus) and 66.3% (GPT-4o) vs GUI automation, and shortens execution time by 58.1% and 57.7% vs MCP-based skills. 

---
# AI Trust OS -- A Continuous Governance Framework for Autonomous AI Observability and Zero-Trust Compliance in Enterprise Environments 

**Authors**: Eranga Bandara, Asanga Gunaratna, Ross Gore, Abdul Rahman, Ravi Mukkamala, Sachin Shetty, Sachini Rajapakse, Isurunima Kularathna, Peter Foytik, Safdar H. Bouk, Xueping Liang, Amin Hass, Ng Wee Keong, Kasun De Zoysa  

**Link**: [PDF](https://arxiv.org/pdf/2604.04749)  

**Abstract**: The accelerating adoption of large language models, retrieval-augmented generation pipelines, and multi-agent AI workflows has created a structural governance crisis. Organizations cannot govern what they cannot see, and existing compliance methodologies built for deterministic web applications provide no mechanism for discovering or continuously validating AI systems that emerge across engineering teams without formal oversight. The result is a widening trust gap between what regulators demand as proof of AI governance maturity and what organizations can demonstrate. This paper proposes AI Trust OS, a governance architecture for continuous, autonomous AI observability and zero-trust compliance. AI Trust OS reconceptualizes compliance as an always-on, telemetry-driven operating layer in which AI systems are discovered through observability signals, control assertions are collected by automated probes, and trust artifacts are synthesized continuously. The framework rests on four principles: proactive discovery, telemetry evidence over manual attestation, continuous posture over point-in-time audit, and architecture-backed proof over policy-document trust. The framework operates through a zero-trust telemetry boundary in which ephemeral read-only probes validate structural metadata without ingressing source code or payload-level PII. An AI Observability Extractor Agent scans LangSmith and Datadog LLM telemetry, automatically registering undocumented AI systems and shifting governance from organizational self-report to empirical machine observation. Evaluated across ISO 42001, the EU AI Act, SOC 2, GDPR, and HIPAA, the paper argues that telemetry-first AI governance represents a categorical architectural shift in how enterprise trust is produced and demonstrated. 

---
# AI Assistance Reduces Persistence and Hurts Independent Performance 

**Authors**: Grace Liu, Brian Christian, Tsvetomira Dumbalska, Michiel A. Bakker, Rachit Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2604.04721)  

**Abstract**: People often optimize for long-term goals in collaboration: A mentor or companion doesn't just answer questions, but also scaffolds learning, tracks progress, and prioritizes the other person's growth over immediate results. In contrast, current AI systems are fundamentally short-sighted collaborators - optimized for providing instant and complete responses, without ever saying no (unless for safety reasons). What are the consequences of this dynamic? Here, through a series of randomized controlled trials on human-AI interactions (N = 1,222), we provide causal evidence for two key consequences of AI assistance: reduced persistence and impairment of unassisted performance. Across a variety of tasks, including mathematical reasoning and reading comprehension, we find that although AI assistance improves performance in the short-term, people perform significantly worse without AI and are more likely to give up. Notably, these effects emerge after only brief interactions with AI (approximately 10 minutes). These findings are particularly concerning because persistence is foundational to skill acquisition and is one of the strongest predictors of long-term learning. We posit that persistence is reduced because AI conditions people to expect immediate answers, thereby denying them the experience of working through challenges on their own. These results suggest the need for AI model development to prioritize scaffolding long-term competence alongside immediate task completion. 

---
# On the "Causality" Step in Policy Gradient Derivations: A Pedagogical Reconciliation of Full Return and Reward-to-Go 

**Authors**: Nima H. Siboni  

**Link**: [PDF](https://arxiv.org/pdf/2604.04686)  

**Abstract**: In introductory presentations of policy gradients, one often derives the REINFORCE estimator using the full trajectory return and then states, by ``causality,'' that the full return may be replaced by the reward-to-go. Although this statement is correct, it is frequently presented at a level of rigor that leaves unclear where the past-reward terms disappear. This short paper isolates that step and gives a mathematically explicit derivation based on prefix trajectory distributions and the score-function identity. The resulting account does not change the estimator. Its contribution is conceptual: instead of presenting reward-to-go as a post hoc unbiased replacement for full return, it shows that reward-to-go arises directly once the objective is decomposed over prefix trajectories. In this formulation, the usual causality argument is recovered as a corollary of the derivation rather than as an additional heuristic principle. 

---
# Springdrift: An Auditable Persistent Runtime for LLM Agents with Case-Based Memory, Normative Safety, and Ambient Self-Perception 

**Authors**: Seamus Brady  

**Link**: [PDF](https://arxiv.org/pdf/2604.04660)  

**Abstract**: We present Springdrift, a persistent runtime for long-lived LLM agents. The system integrates an auditable execution substrate (append-only memory, supervised processes, git-backed recovery), a case-based reasoning memory layer with hybrid retrieval (evaluated against a dense cosine baseline), a deterministic normative calculus for safety gating with auditable axiom trails, and continuous ambient self-perception via a structured self-state representation (the sensorium) injected each cycle without tool calls. These properties support behaviours difficult to achieve in session-bounded systems: cross-session task continuity, cross-channel context maintenance, end-to-end forensic reconstruction of decisions, and self-diagnostic behaviour. We report on a single-instance deployment over 23 days (19 operating days), during which the agent diagnosed its own infrastructure bugs, classified failure modes, identified an architectural vulnerability, and maintained context across email and web channels -- without explicit instruction. We introduce the term Artificial Retainer for this category: a non-human system with persistent memory, defined authority, domain-specific autonomy, and forensic accountability in an ongoing relationship with a specific principal -- distinguished from software assistants and autonomous agents, drawing on professional retainer relationships and the bounded autonomy of trained working animals. This is a technical report on a systems design and deployment case study, not a benchmark-driven evaluation. Evidence is from a single instance with a single operator, presented as illustration of what these architectural properties can support in practice. Implemented in approximately Gleam on Erlang/OTP. Code, artefacts, and redacted operational logs will be available at this https URL upon publication. 

---
# Search, Do not Guess: Teaching Small Language Models to Be Effective Search Agents 

**Authors**: Yizhou Liu, Qi Sun, Yulin Chen, Siyue Zhang, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04651)  

**Abstract**: Agents equipped with search tools have emerged as effective solutions for knowledge-intensive tasks. While Large Language Models (LLMs) exhibit strong reasoning capabilities, their high computational cost limits practical deployment for search agents. Consequently, recent work has focused on distilling agentic behaviors from LLMs into Small Language Models (SLMs). Through comprehensive evaluation on complex multi-hop reasoning tasks, we find that despite possessing less parametric knowledge, SLMs invoke search tools less frequently and are more prone to hallucinations. To address this issue, we propose \policy, a lightweight fine-tuning approach that explicitly trains SLMs to reliably retrieve and generate answers grounded in retrieved evidence. Compared to agent distillation from LLMs, our approach improves performance by 17.3 scores on Bamboogle and 15.3 scores on HotpotQA, achieving LLM-level results across benchmarks. Our further analysis reveals that adaptive search strategies in SLMs often degrade performance, highlighting the necessity of consistent search behavior for reliable reasoning. 

---
# Same World, Differently Given: History-Dependent Perceptual Reorganization in Artificial Agents 

**Authors**: Hongju Pae  

**Link**: [PDF](https://arxiv.org/pdf/2604.04637)  

**Abstract**: What kind of internal organization would allow an artificial agent not only to adapt its behavior, but to sustain a history-sensitive perspective on its world? I present a minimal architecture in which a slow perspective latent $g$ feeds back into perception and is itself updated through perceptual processing. This allows identical observations to be encoded differently depending on the agent's accumulated stance. The model is evaluated in a minimal gridworld with a fixed spatial scaffold and sensory perturbations. Across analyses, three results emerge: first, perturbation history leaves measurable residue in adaptive plasticity after nominal conditions are restored. Second, the perspective latent reorganizes perceptual encoding, such that identical observations are represented differently depending on prior experience. Third, only adaptive self-modulation yields the characteristic growth-then-stabilization dynamic, unlike rigid or always-open update regimes. Gross behavior remains stable throughout, suggesting that the dominant reorganization is perceptual rather than behavioral. Together, these findings identify a minimal mechanism for history-dependent perspectival organization in artificial agents. 

---
# Greedy and Transformer-Based Multi-Port Selection for Slow Fluid Antenna Multiple Access 

**Authors**: Darian Perez-Adan, Jose P. Gonzalez-Coma, F. Javier Lopez-Martinez, Luis Castedo  

**Link**: [PDF](https://arxiv.org/pdf/2604.04589)  

**Abstract**: We address the port-selection problem in fluid antenna multiple access (FAMA) systems with multi-port fluid antenna (FA) receivers. Existing methods either achieve near-optimal spectral efficiency (SE) at prohibitive computational cost or sacrifice significant performance for lower complexity. We propose two complementary strategies: (i) GFwd+S, a greedy forward-selection method with swap refinement that consistently outperforms state-of-the-art reference schemes in terms of SE, and (ii) a Transformer-based neural network trained via imitation learning followed by a Reinforce policy-gradient stage, which approaches GFwd+S performance at lower computational cost. 

---
# Receding-Horizon Control via Drifting Models 

**Authors**: Daniele Foffano, Alessio Russo, Alexandre Proutiere  

**Link**: [PDF](https://arxiv.org/pdf/2604.04528)  

**Abstract**: We study the problem of trajectory optimization in settings where the system dynamics are unknown and it is not possible to simulate trajectories through a surrogate model. When an offline dataset of trajectories is available, an agent could directly learn a trajectory generator by distribution matching. However, this approach only recovers the behavior distribution in the dataset, and does not in general produce a model that minimizes a desired cost criterion. In this work, we propose Drifting MPC, an offline trajectory optimization framework that combines drifting generative models with receding-horizon planning under unknown dynamics. The goal of Drifting MPC is to learn, from an offline dataset of trajectories, a conditional distribution over trajectories that is both supported by the data and biased toward optimal plans. We show that the resulting distribution learned by Drifting MPC is the unique solution of an objective that trades off optimality with closeness to the offline prior. Empirically, we show that Drifting MPC can generate near-optimal trajectories while retaining the one-step inference efficiency of drifting models and substantially reducing generation time relative to diffusion-based baselines. 

---
# SuperLocalMemory V3.3: The Living Brain -- Biologically-Inspired Forgetting, Cognitive Quantization, and Multi-Channel Retrieval for Zero-LLM Agent Memory Systems 

**Authors**: Varun Pratap Bhardwaj  

**Link**: [PDF](https://arxiv.org/pdf/2604.04514)  

**Abstract**: AI coding agents operate in a paradox: they possess vast parametric knowledge yet cannot remember a conversation from an hour ago. Existing memory systems store text in vector databases with single-channel retrieval, require cloud LLMs for core operations, and implement none of the cognitive processes that make human memory effective.
We present SuperLocalMemory V3.3 ("The Living Brain"), a local-first agent memory system implementing the full cognitive memory taxonomy with mathematical lifecycle dynamics. Building on the information-geometric foundations of V3.2 (arXiv:2603.14588), we introduce five contributions: (1) Fisher-Rao Quantization-Aware Distance (FRQAD) -- a new metric on the Gaussian statistical manifold achieving 100% precision at preferring high-fidelity embeddings over quantized ones (vs 85.6% for cosine), with zero prior art; (2) Ebbinghaus Adaptive Forgetting with lifecycle-aware quantization -- the first mathematical forgetting curve in local agent memory coupled to progressive embedding compression, achieving 6.7x discriminative power; (3) 7-channel cognitive retrieval spanning semantic, keyword, entity graph, temporal, spreading activation, consolidation, and Hopfield associative channels, achieving 70.4% on LoCoMo in zero-LLM Mode A; (4) memory parameterization implementing Long-Term Implicit memory via soft prompts; (5) zero-friction auto-cognitive pipeline automating the complete memory lifecycle.
On LoCoMo, V3.3 achieves 70.4% in Mode A (zero-LLM), with +23.8pp on multi-hop and +12.7pp on adversarial. V3.2 achieved 74.8% Mode A and 87.7% Mode C; the 4.4pp gap reflects a deliberate architectural trade-off. SLM V3.3 is open source under the Elastic License 2.0, runs entirely on CPU, with over 5,000 monthly downloads. 

---
# Memory Intelligence Agent 

**Authors**: Jingyang Qiao, Weicheng Meng, Yu Cheng, Zhihang Lin, Zhizhong Zhang, Xin Tan, Jingyu Gong, Kun Shao, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.04503)  

**Abstract**: Deep research agents (DRAs) integrate LLM reasoning with external tools. Memory systems enable DRAs to leverage historical experiences, which are essential for efficient reasoning and autonomous evolution. Existing methods rely on retrieving similar trajectories from memory to aid reasoning, while suffering from key limitations of ineffective memory evolution and increasing storage and retrieval costs. To address these problems, we propose a novel Memory Intelligence Agent (MIA) framework, consisting of a Manager-Planner-Executor architecture. Memory Manager is a non-parametric memory system that can store compressed historical search trajectories. Planner is a parametric memory agent that can produce search plans for questions. Executor is another agent that can search and analyze information guided by the search plan. To build the MIA framework, we first adopt an alternating reinforcement learning paradigm to enhance cooperation between the Planner and the Executor. Furthermore, we enable the Planner to continuously evolve during test-time learning, with updates performed on-the-fly alongside inference without interrupting the reasoning process. Additionally, we establish a bidirectional conversion loop between parametric and non-parametric memories to achieve efficient memory evolution. Finally, we incorporate a reflection and an unsupervised judgment mechanisms to boost reasoning and self-evolution in the open world. Extensive experiments across eleven benchmarks demonstrate the superiority of MIA. 

---
# Scalable and Explainable Learner-Video Interaction Prediction using Multimodal Large Language Models 

**Authors**: Dominik Glandorf, Fares Fawzi, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2604.04482)  

**Abstract**: Learners' use of video controls in educational videos provides implicit signals of cognitive processing and instructional design quality, yet the lack of scalable and explainable predictive models limits instructors' ability to anticipate such behavior before deployment. We propose a scalable, interpretable pipeline for predicting population-level watching, pausing, skipping, and rewinding behavior as proxies for cognitive load from video content alone. Our approach leverages multimodal large language models (MLLMs) to compute embeddings of short video segments and trains a neural classifier to identify temporally fine-grained interaction peaks. Drawing from multimedia learning theory on instructional design for optimal cognitive load, we code features of the video segments using GPT-5 and employ them as a basis for interpreting model predictions via concept activation vectors. We evaluate our pipeline on 77 million video control events from 66 online courses. Our findings demonstrate that classifiers based on MLLM embeddings reliably predict interaction peaks, generalize to unseen academic fields, and encode interpretable, theory-relevant instructional concepts. Overall, our results show the feasibility of cost-efficient, interpretable pre-screening of educational video design and open new opportunities to empirically examine multimedia learning theory at scale. 

---
# What Makes a Sale? Rethinking End-to-End Seller--Buyer Retail Dynamics with LLM Agents 

**Authors**: Jeonghwan Choi, Jibin Hwang, Gyeonghun Sun, Minjeong Ban, Taewon Yun, Hyeonjae Cheon, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.04468)  

**Abstract**: Evaluating retail strategies before deployment is difficult, as outcomes are determined across multiple stages, from seller-side persuasion through buyer-seller interaction to purchase decisions. However, existing retail simulators capture only partial aspects of this process and do not model cross-stage dependencies, making it difficult to assess how early decisions affect downstream outcomes. We present RetailSim, an end-to-end retail simulation framework that models this pipeline in a unified environment, explicitly designed for simulation fidelity through diverse product spaces, persona-driven agents, and multi-turn interactions. We evaluate RetailSim with a dual protocol comprising human evaluation of behavioral fidelity and meta-evaluation against real-world economic regularities, showing that it successfully reproduces key patterns such as demographic purchasing behavior, the price-demand relationship, and heterogeneous price elasticity. We further demonstrate its practical utility via decision-oriented use cases, including persona inference, seller-buyer interaction analysis, and sales strategy evaluation, showing RetailSim's potential as a controlled testbed for exploring retail strategies. 

---
# The Topology of Multimodal Fusion: Why Current Architectures Fail at Creative Cognition 

**Authors**: Xiujiang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04465)  

**Abstract**: This paper identifies a structural limitation in current multimodal AI architectures that is topological rather than parametric. Contrastive alignment (CLIP), cross-attention fusion (GPT-4V/Gemini), and diffusion-based generation share a common geometric prior -- modal separability -- which we term contact topology. The argument rests on three pillars with philosophy as the generative center. The philosophical pillar reinterprets Wittgenstein's saying/showing distinction as a problem rather than a conclusion: where Wittgenstein chose silence, the Chinese craft epistemology tradition responded with xiang (operative schema) -- the third state emerging when saying and showing interpenetrate. A cruciform framework (dao/qi x saying/showing) positions xiang at the intersection, executing dual huacai (transformation-and-cutting) along both axes. This generates a dual-layer dynamics: chuanghua (creative transformation as spontaneous event) and huacai (its institutionalization into repeatable form). The cognitive science pillar reinterprets DMN/ECN/SN tripartite co-activation through the pathological mirror: overlap isomorphism vs. superimposition collapse in a 2D parameter space (coupling intensity x regulatory capacity). The mathematical pillar formalizes these via fiber bundles and Yang-Mills curvature, with the cruciform structure mapped to fiber bundle language. We propose UOO implementation via Neural ODEs with topological regularization, the ANALOGY-MM benchmark with error-type-ratio metric, and the META-TOP three-tier benchmark testing cross-civilizational topological isomorphism across seven archetypes. A phased experimental roadmap with explicit termination criteria ensures clean exit if falsified. 

---
# Empirical Characterization of Rationale Stability Under Controlled Perturbations for Explainable Pattern Recognition 

**Authors**: Abu Noman Md Sakib, Zhensen Wang, Merjulah Roby, Zijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04456)  

**Abstract**: Reliable pattern recognition systems should exhibit consistent behavior across similar inputs, and their explanations should remain stable. However, most Explainable AI evaluations remain instance centric and do not explicitly quantify whether attribution patterns are consistent across samples that share the same class or represent small variations of the same input. In this work, we propose a novel metric aimed at assessing the consistency of model explanations, ensuring that models consistently reflect the intended objectives and consistency under label-preserving perturbations. We implement this metric using a pre-trained BERT model on the SST-2 sentiment analysis dataset, with additional robustness tests on RoBERTa, DistilBERT, and IMDB, applying SHAP to compute feature importance for various test samples. The proposed metric quantifies the cosine similarity of SHAP values for inputs with the same label, aiming to detect inconsistent behaviors, such as biased reliance on certain features or failure to maintain consistent reasoning for similar predictions. Through a series of experiments, we evaluate the ability of this metric to identify misaligned predictions and inconsistencies in model explanations. These experiments are compared against standard fidelity metrics to assess whether the new metric can effectively identify when a model's behavior deviates from its intended objectives. The proposed framework provides a deeper understanding of model behavior by enabling more robust verification of rationale stability, which is critical for building trustworthy AI systems. By quantifying whether models rely on consistent attribution patterns for similar inputs, the proposed approach supports more robust evaluation of model behavior in practical pattern recognition pipelines. Our code is publicly available at this https URL. 

---
# PSY-STEP: Structuring Therapeutic Targets and Action Sequences for Proactive Counseling Dialogue Systems 

**Authors**: Jihyun Lee, Yejin Min, Yejin Jeon, SungJun Yang, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04448)  

**Abstract**: Cognitive Behavioral Therapy (CBT) aims to identify and restructure automatic negative thoughts pertaining to involuntary interpretations of events, yet existing counseling agents struggle to identify and address them in dialogue settings. To bridge this gap, we introduce STEP, a dataset that models CBT counseling by explicitly reflecting automatic thoughts alongside dynamic, action-level counseling sequences. Using this dataset, we train STEPPER, a counseling agent that proactively elicits automatic thoughts and executes cognitively grounded interventions. To further enhance both decision accuracy and empathic responsiveness, we refine STEPPER through preference learning based on simulated, synthesized counseling sessions. Extensive CBT-aligned evaluations show that STEPPER delivers more clinically grounded, coherent, and personalized counseling compared to other strong baseline models, and achieves higher counselor competence without inducing emotional disruption. 

---
# ShieldNet: Network-Level Guardrails against Emerging Supply-Chain Injections in Agentic Systems 

**Authors**: Zhuowen Yuan, Zhaorun Chen, Zhen Xiang, Nathaniel D. Bastian, Seyyed Hadi Hashemi, Chaowei Xiao, Wenbo Guo, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.04426)  

**Abstract**: Existing research on LLM agent security mainly focuses on prompt injection and unsafe input/output behaviors. However, as agents increasingly rely on third-party tools and MCP servers, a new class of supply-chain threats has emerged, where malicious behaviors are embedded in seemingly benign tools, silently hijacking agent execution, leaking sensitive data, or triggering unauthorized actions. Despite their growing impact, there is currently no comprehensive benchmark for evaluating such threats. To bridge this gap, we introduce SC-Inject-Bench, a large-scale benchmark comprising over 10,000 malicious MCP tools grounded in a taxonomy of 25+ attack types derived from MITRE ATT&CK targeting supply-chain threats. We observe that existing MCP scanners and semantic guardrails perform poorly on this benchmark. Motivated by this finding, we propose ShieldNet, a network-level guardrail framework that detects supply-chain poisoning by observing real network interactions rather than surface-level tool traces. ShieldNet integrates a man-in-the-middle (MITM) proxy and an event extractor to identify critical network behaviors, which are then processed by a lightweight classifier for attack detection. Extensive experiments show that ShieldNet achieves strong detection performance (up to 0.995 F-1 with only 0.8% false positives) while introducing little runtime overhead, substantially outperforming existing MCP scanners and LLM-based guardrails. 

---
# MolDA: Molecular Understanding and Generation via Large Language Diffusion Model 

**Authors**: Seohyeon Shin, HanJun Choi, Jun-Hyung Park, Hongkook Kim, Mansu Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.04403)  

**Abstract**: Large Language Models (LLMs) have significantly advanced molecular discovery, but existing multimodal molecular architectures fundamentally rely on autoregressive (AR) backbones. This strict left-to-right inductive bias is sub-optimal for generating chemically valid molecules, as it struggles to account for non-local global constraints (e.g., ring closures) and often accumulates structural errors during sequential generation. To address these limitations, we propose MolDA (Molecular language model with masked Diffusion with mAsking), a novel multimodal framework that replaces the conventional AR backbone with a discrete Large Language Diffusion Model. MolDA extracts comprehensive structural representations using a hybrid graph encoder, which captures both local and global topologies, and aligns them into the language token space via a Q-Former. Furthermore, we mathematically reformulate Molecular Structure Preference Optimization specifically for the masked diffusion. Through bidirectional iterative denoising, MolDA ensures global structural coherence, chemical validity, and robust reasoning across molecule generation, captioning, and property prediction. 

---
# GUIDE: Interpretable GUI Agent Evaluation via Hierarchical Diagnosis 

**Authors**: Yuwen Zhai, Runze Li, Liang Wang, Nian Shi, Liwu Xu, Wei Zhang, Ran Lin, Bo Xu, Benlei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2604.04399)  

**Abstract**: Evaluating GUI agents presents a distinct challenge: trajectories are long, visually grounded, and open-ended, yet evaluation must be both accurate and interpretable. Existing approaches typically apply a single holistic judgment over the entire action-observation sequence-a strategy that proves unreliable on long-horizon tasks and yields binary verdicts offering no insight into where or why an agent fails. This opacity limits the utility of evaluation as a diagnostic tool for agent development. We introduce GUIDE (GUI Understanding and Interpretable Diagnostic Evaluation), a framework that decomposes trajectory assessment into three sequential stages mirroring the compositional structure of GUI tasks. Trajectory Segmentation partitions the full trace into semantically coherent subtask units. Subtask Diagnosis evaluates each unit in context, assigning a completion verdict and generating a structured error analysis with corrective recommendations. Overall Summary aggregates per-subtask diagnoses into a task-level judgment. By operating on bounded subtask segments rather than full trajectories, GUIDE mitigates the context overload that degrades existing evaluators as task complexity grows. We validate GUIDE on three benchmarks: an industrial e-commerce dataset of 932 trajectories, AGENTREWARDBENCH spanning five web agent tasks with 1302 trajectories, and AndroidBench for mobile device control. Across all settings, GUIDE substantially outperforms existing evaluators-achieving up to 5.35 percentage points higher accuracy than the strongest baseline-while producing structured diagnostic reports that directly inform agent improvement. 

---
# Gradual Cognitive Externalization: A Framework for Understanding How Ambient Intelligence Externalizes Human Cognition 

**Authors**: Zhimin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04387)  

**Abstract**: Developers are publishing AI agent skills that replicate a colleague's communication style, encode a supervisor's mentoring heuristics, or preserve a person's behavioral repertoire beyond biological death. To explain why, we propose Gradual Cognitive Externalization (GCE), a framework arguing that human cognitive functions are migrating into digital substrates through ambient intelligence co-adaptation rather than mind uploading. GCE rests on the behavioral manifold hypothesis: everyday cognition occupies a low-dimensional manifold that is structured, redundant, and learnable from sustained observation. We document evidence from scheduling assistants, writing tools, recommendation engines, and agent skill ecosystems showing that the preconditions for externalization are already observable. We formalize three criteria separating cognitive integration from tool use (bidirectional adaptation, functional equivalence, causal coupling), derive five testable predictions with theory-constrained thresholds, and provide a concrete experimental protocol. The question is no longer whether minds can be uploaded, but how fast cognitive functions are already migrating into digital substrates and what follows. 

---
# Automatically Generating Hard Math Problems from Hypothesis-Driven Error Analysis 

**Authors**: Jiayu Fu, Mourad Heddaya, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04386)  

**Abstract**: Numerous math benchmarks exist to evaluate LLMs' mathematical capabilities. However, most involve extensive manual effort and are difficult to scale. Consequently, they cannot keep pace with LLM development or easily provide new instances to mitigate overfitting. Some researchers have proposed automatic benchmark generation methods, but few focus on identifying the specific math concepts and skills on which LLMs are error-prone, and most can only generate category-specific benchmarks. To address these limitations, we propose a new math benchmark generation pipeline that uses AI-generated hypotheses to identify the specific math concepts and skills that LLMs struggle with, and then generates new benchmark problems targeting these weaknesses. Experiments show that hypothesis accuracy positively correlates with the difficulty of the generated problems: problems generated from the most accurate hypotheses reduce Llama-3.3-70B-Instruct's accuracy to as low as 45%, compared to 77% on the original MATH benchmark. Furthermore, our pipeline is highly adaptable and can be applied beyond math to explore a wide range of LLM capabilities, making it a valuable tool for investigating how LLMs perform across different domains. 

---
# Optimizing Service Operations via LLM-Powered Multi-Agent Simulation 

**Authors**: Yanyuan Wang, Xiaowei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04383)  

**Abstract**: Service system performance depends on how participants respond to design choices, but modeling these responses is hard due to the complexity of human behavior. We introduce an LLM-powered multi-agent simulation (LLM-MAS) framework for optimizing service operations. We pose the problem as stochastic optimization with decision-dependent uncertainty: design choices are embedded in prompts and shape the distribution of outcomes from interacting LLM-powered agents. By embedding key numerical information in prompts and extracting it from LLM-generated text, we model this uncertainty as a controlled Markov chain. We develop an on-trajectory learning algorithm that, on a single simulation run, simultaneously constructs zeroth-order gradient estimates and updates design parameters to optimize steady-state performance. We also incorporate variance reduction techniques. In a sustainable supply chain application, our method outperforms benchmarks, including blackbox optimization and using LLMs as numerical solvers or as role-playing system designers. A case study on optimal contest design with real behavioral data shows that LLM-MAS is both as a cost-effective evaluator of known designs and an exploratory tool that can uncover strong designs overlooked by traditional approaches. 

---
# Decocted Experience Improves Test-Time Inference in LLM Agents 

**Authors**: Maohao Shen, Kaiwen Zha, Zexue He, Zhang-Wei Hong, Siru Ouyang, J. Jon Ryu, Prasanna Sattigeri, Suhas Diggavi, Gregory Wornell  

**Link**: [PDF](https://arxiv.org/pdf/2604.04373)  

**Abstract**: There is growing interest in improving LLMs without updating model parameters. One well-established direction is test-time scaling, where increased inference-time computation (e.g., longer reasoning, sampling, or search) is used to improve performance. However, for complex reasoning and agentic tasks, naively scaling test-time compute can substantially increase cost and still lead to wasted budget on suboptimal exploration. In this paper, we explore \emph{context} as a complementary scaling axis for improving LLM performance, and systematically study how to construct better inputs that guide reasoning through \emph{experience}. We show that effective context construction critically depends on \emph{decocted experience}. We present a detailed analysis of experience-augmented agents, studying how to derive context from experience, how performance scales with accumulated experience, what characterizes good context, and which data structures best support context construction. We identify \emph{decocted experience} as a key mechanism for effective context construction: extracting essence from experience, organizing it coherently, and retrieving salient information to build effective context. We validate our findings across reasoning and agentic tasks, including math reasoning, web browsing, and software engineering. 

---
# REAM: Merging Improves Pruning of Experts in LLMs 

**Authors**: Saurav Jha, Maryam Hashemzadeh, Ali Saheb Pasand, Ali Parviz, Min-Joong Lee, Boris Knyazev  

**Link**: [PDF](https://arxiv.org/pdf/2604.04356)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLMs) are among the top-performing architectures. The largest models, often with hundreds of billions of parameters, pose significant memory challenges for deployment. Traditional approaches to reduce memory requirements include weight pruning and quantization. Motivated by the Router-weighted Expert Activation Pruning (REAP) that prunes experts, we propose a novel method, Router-weighted Expert Activation Merging (REAM). Instead of removing experts, REAM groups them and merges their weights, better preserving original performance. We evaluate REAM against REAP and other baselines across multiple MoE LLMs on diverse multiple-choice (MC) question answering and generative (GEN) benchmarks. Our results reveal a trade-off between MC and GEN performance that depends on the mix of calibration data. By controlling the mix of general, math and coding data, we examine the Pareto frontier of this trade-off and show that REAM often outperforms the baselines and in many cases is comparable to the original uncompressed models. 

---
# RoboPhD: Evolving Diverse Complex Agents Under Tight Evaluation Budgets 

**Authors**: Andrew Borthwick, Stephen Ash, Anthony Galczak  

**Link**: [PDF](https://arxiv.org/pdf/2604.04347)  

**Abstract**: 2026 has brought an explosion of interest in LLM-guided evolution of agentic artifacts, with systems like GEPA and Autoresearch demonstrating that LLMs can iteratively improve prompts, code, and agent architectures across diverse domains. As adoption accelerates, a central question emerges: given the same information, the same seed agent, and the same objective, which optimization algorithm yields the best results under the same evaluation budget? This question becomes critical when evaluations are expensive, such as when they require human judgment or multiple LLM calls.
We present the first systematic comparison of three optimization paradigms -- Elo tournament selection (RoboPhD), Pareto-based selection (GEPA), and greedy hill-climbing (Autoresearch) -- across four benchmarks spanning abstract reasoning, cloud scheduling, SQL generation, and financial QA, all under a fixed budget of 1,500 evaluations. RoboPhD introduces validation-free evolution: instead of splitting the budget between training and validation, it uses Elo competition on training data to simultaneously evaluate agents and drive evolution. All three systems receive seed agents with diagnostic print() statements that evolution can grow, enabling self-instrumenting agents that develop increasingly informative diagnostics for the benefit of their evolutionary successors.
Using a single default configuration, RoboPhD outperforms both GEPA and Autoresearch on three of four benchmarks, losing only on the simplest task, where the winning solution (from our Autoresearch adaptation) required under 90 lines of code. On ARC-AGI, RoboPhD evolves a 22-line seed agent into a 1,013-line multi-strategy system, improving accuracy from 27.8% to 65.8% using Gemini 3.1 Flash Lite as the solver. We release RoboPhD as a versatile toolkit under the MIT license with a simple optimize_anything() API for evolving diverse complex agents. 

---
# Domain-Contextualized Inference: A Computable Graph Architecture for Explicit-Domain Reasoning 

**Authors**: Chao Li, Yuru Wang, Chunyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04344)  

**Abstract**: We establish a computation-substrate-agnostic inference architecture in which domain is an explicit first-class computational parameter. This produces domain-scoped pruning that reduces per-query search space from O(N) to O(N/K), substrate-independent execution over symbolic, neural, vector, and hybrid substrates, and transparent inference chains where every step carries its evaluative context. The contribution is architectural, not logical. We formalize the computational theory across five dimensions: a five-layer architecture; three domain computation modes including chain indexing, path traversal as Kleisli composition, and vector-guided computation as a substrate transition; a substrate-agnostic interface with three operations Query, Extend, Bridge; reliability conditions C1 to C4 with three failure mode classes; and validation through a PHQ-9 clinical reasoning case study. The computational theory including operational semantics, complexity bounds, monad structure, substrate transitions, and boundary conditions is the contribution of this paper. 

---
# Implementing surrogate goals for safer bargaining in LLM-based agents 

**Authors**: Caspar Oesterheld, Maxime Riché, Filip Sondej, Jesse Clifton, Vincent Conitzer  

**Link**: [PDF](https://arxiv.org/pdf/2604.04341)  

**Abstract**: Surrogate goals have been proposed as a strategy for reducing risks from bargaining failures. A surrogate goal is goal that a principal can give an AI agent and that deflects any threats against the agent away from what the principal cares about. For example, one might make one's agent care about preventing money from being burned. Then in bargaining interactions, other agents can threaten to burn their money instead of threatening to spending money to hurt the principal. Importantly, the agent has to care equally about preventing money from being burned as it cares about money being spent to hurt the principal.
In this paper, we implement surrogate goals in language-model-based agents. In particular, we try to get a language-model-based agent to react to threats of burning money in the same way it would react to "normal" threats. We propose four different methods, using techniques of prompting, fine-tuning, and scaffolding. We evaluate the four methods experimentally. We find that methods based on scaffolding and fine-tuning outperform simple prompting. In particular, fine-tuning and scaffolding more precisely implement the desired behavior w.r.t. threats against the surrogate goal. We also compare the different methods in terms of their side effects on capabilities and propensities in other situations. We find that scaffolding-based methods perform best. 

---
# Thermodynamic-Inspired Explainable GeoAI: Uncovering Regime-Dependent Mechanisms in Heterogeneous Spatial Systems 

**Authors**: Sooyoung Lim, Zhenlong Li, Zi-Kui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04339)  

**Abstract**: Modeling spatial heterogeneity and associated critical transitions remains a fundamental challenge in geography and environmental science. While conventional Geographically Weighted Regression (GWR) and deep learning models have improved predictive skill, they often fail to elucidate state-dependent nonlinearities where the functional roles of drivers represent opposing effects across heterogeneous domains. We introduce a thermodynamics-inspired explainable geospatial AI framework that integrates statistical mechanics with graph neural networks. By conceptualizing spatial variability as a thermodynamic competition between system Burden (E) and Capacity (S), our model disentangles the latent mechanisms driving spatial processes. Using three simulation datasets and three real-word datasets across distinct domains (housing markets, mental health prevalence, and wildfire-induced PM2.5 anomalies), we show that the new framework successfully identifies regime-dependent role reversals of predictors that standard baselines miss. Notably, the framework explicitly diagnoses the phase transition into a Burden-dominated regime during the 2023 Canadian wildfire event, distinguishing physical mechanism shifts from statistical outliers. These findings demonstrate that thermodynamic constraints can improve the interpretability of GeoAI while preserving strong predictive performance in complex spatial systems. 

---
# Soft Tournament Equilibrium 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04328)  

**Abstract**: The evaluation of general-purpose artificial agents, particularly those based on large language models, presents a significant challenge due to the non-transitive nature of their interactions. When agent A defeats B, B defeats C, and C defeats A, traditional ranking methods that force a linear ordering can be misleading and unstable. We argue that for such cyclic domains, the fundamental object of evaluation should not be a ranking but a set-valued core, as conceptualized in classical tournament theory. This paper introduces Soft Tournament Equilibrium (STE), a differentiable framework for learning and computing set-valued tournament solutions directly from pairwise comparison data. STE first learns a probabilistic tournament model, potentially conditioned on rich contextual information. It then employs novel, differentiable operators for soft reachability and soft covering to compute continuous analogues of two seminal tournament solutions: the Top Cycle and the Uncovered Set. The output is a set of core agents, each with a calibrated membership score, providing a nuanced and robust assessment of agent capabilities. We develop the theoretical foundation for STE to prove its consistency with classical solutions in the zero-temperature limit, which establishes its Condorcet-inclusion properties, and analyzing its stability and sample complexity. We specify an experimental protocol for validating STE on both synthetic and real-world benchmarks. This work aims to provide a complete, standalone treatise that re-centers general-agent evaluation on a more appropriate and robust theoretical foundation, moving from unstable rankings to stable, set-valued equilibria. 

---
# RESCORE: LLM-Driven Simulation Recovery in Control Systems Research Papers 

**Authors**: Vineet Bhat, Shiqing Wei, Ali Umut Kaypak, Prashanth Krishnamurthy, Ramesh Karri, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04324)  

**Abstract**: Reconstructing numerical simulations from control systems research papers is often hindered by underspecified parameters and ambiguous implementation details. We define the task of Paper to Simulation Recoverability, the ability of an automated system to generate executable code that faithfully reproduces a paper's results. We curate a benchmark of 500 papers from the IEEE Conference on Decision and Control (CDC) and propose RESCORE, a three component LLM agentic framework, Analyzer, Coder, and Verifier. RESCORE uses iterative execution feedback and visual comparison to improve reconstruction fidelity. Our method successfully recovers task coherent simulations for 40.7% of benchmark instances, outperforming single pass generation. Notably, the RESCORE automated pipeline achieves an estimated 10X speedup over manual human replication, drastically cutting the time and effort required to verify published control methodologies. We will release our benchmark and agents to foster community progress in automated research replication. 

---
# PanLUNA: An Efficient and Robust Query-Unified Multimodal Model for Edge Biosignal Intelligence 

**Authors**: Marija Zelic, Anna Tegon, Yawei Li, Thorir Mar Ingolfsson, Luca Benini  

**Link**: [PDF](https://arxiv.org/pdf/2604.04297)  

**Abstract**: Physiological foundation models (FMs) have shown promise for biosignal representation learning, yet most remain confined to a single modality such as EEG, ECG, or PPG, largely because paired multimodal datasets are scarce. In this paper, we present PanLUNA, a compact 5.4M-parameter pan-modal FM that jointly processes EEG, ECG, and PPG within a single shared encoder. Extending LUNA's channel-unification module, PanLUNA treats multimodal channels as entries in a unified query set augmented with sensor-type embeddings, enabling efficient cross-modal early fusion while remaining inherently robust to missing modalities at inference time. Despite its small footprint, PanLUNA matches or exceeds models up to 57$\times$ larger: 81.21% balanced accuracy on TUAB abnormal EEG detection and state-of-the-art 0.7416 balanced accuracy on HMC multimodal sleep staging. Quantization-aware training with INT8 weights recovers $\geq$96% of full-precision performance, and deployment on the GAP9 ultra-low-power RISC-V microcontroller for wearables achieves 325.6 ms latency and 18.8 mJ per 10-second, 12-lead ECG inference, and 1.206 s latency at 68.65 mJ for multimodal 5-channel sleep staging over 30-second epochs. 

---
# Preservation Is Not Enough for Width Growth: Regime-Sensitive Selection of Dense LM Warm Starts 

**Authors**: Eren Unlu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04281)  

**Abstract**: Width expansion offers a practical route to reuse smaller causal-language-model checkpoints, but selecting a widened warm start is not solved by zero-step preservation alone. We study dense width growth as a candidate-selection problem over full training states, including copied weights, optimizer moments, and scheduler state. In a small-scale TinyStories proxy, we compare exact-copy, perturbative, asymmetric-reset, and structured non-clone warm starts under matched continuation budgets. We evaluate zero-step preservation, short-lag probe metrics, and downstream continuation utility in deterministic and stochastic regimes. The picture is mixed and partially replicated through a reduced-pool seed-1 check. Exact-copy symmetric warm starts rank first in every completed 16-step probe and in the completed stochastic 128-step continuations at seed-0 steps 1000 and 2000 plus reduced seed-1 step 2000. By contrast, the structured non-clone challenger wins deterministic 128-step continuation. Early escape from the inherited cloned subspace is therefore not a universal selector: it helps in long deterministic continuation, but it misleads at short lag and under stochastic continuation. The result is narrow but useful: for dense width growth at this scale, preservation is not a universal ranking criterion, and the best replacement signal depends on both regime and lag budget. 

---
# InferenceEvolve: Towards Automated Causal Effect Estimators through Self-Evolving AI 

**Authors**: Can Wang, Hongyu Zhao, Yiqun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04274)  

**Abstract**: Causal inference is central to scientific discovery, yet choosing appropriate methods remains challenging because of the complexity of both statistical methodology and real-world data. Inspired by the success of artificial intelligence in accelerating scientific discovery, we introduce InferenceEvolve, an evolutionary framework that uses large language models to discover and iteratively refine causal methods. Across widely used benchmarks, InferenceEvolve yields estimators that consistently outperform established baselines: against 58 human submissions in a recent community competition, our best evolved estimator lay on the Pareto frontier across two evaluation metrics. We also developed robust proxy objectives for settings without semi-synthetic outcomes, with competitive results. Analysis of the evolutionary trajectories shows that agents progressively discover sophisticated strategies tailored to unrevealed data-generating mechanisms. These findings suggest that language-model-guided evolution can optimize structured scientific programs such as causal inference, even when outcomes are only partially observed. 

---
# Beyond Fluency: Toward Reliable Trajectories in Agentic IR 

**Authors**: Anushree Sinha, Srivaths Ranganathan, Debanshu Das, Abhishek Dharmaratnakar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04269)  

**Abstract**: Information Retrieval is shifting from passive document ranking toward autonomous agentic workflows that operate in multi-step Reason-Act-Observe loops. In such long-horizon trajectories, minor early errors can cascade, leading to functional misalignment between internal reasoning and external tool execution despite continued linguistic fluency.
This position paper synthesizes failure modes observed in industrial agentic systems, categorizing errors across planning, retrieval, reasoning, and execution. We argue that safe deployment requires moving beyond endpoint accuracy toward trajectory integrity and causal attribution.
To address compounding error and deceptive fluency, we propose verification gates at each interaction unit and advocate systematic abstention under calibrated uncertainty. Reliable Agentic IR systems must prioritize process correctness and grounded execution over plausible but unverified completion. 

---
# Context Engineering: A Practitioner Methodology for Structured Human-AI Collaboration 

**Authors**: Elias Calboreanu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04258)  

**Abstract**: The quality of AI-generated output is often attributed to prompting technique, but extensive empirical observation suggests that context completeness may be more strongly associated with output quality. This paper introduces Context Engineering, a structured methodology for assembling, declaring, and sequencing the complete informational payload that accompanies a prompt to an AI tool. Context Engineering defines a five-role context package structure (Authority, Exemplar, Constraint, Rubric, Metadata), applies a staged four-phase pipeline (Reviewer to Design to Builder to Auditor), and applies formal models from reliability engineering and information theory as post hoc interpretive lenses on context quality. In an observational study of 200 documented interactions across four AI tools (Claude, ChatGPT, Cowork, Codex), incomplete context was associated with 72% of iteration cycles. Structured context assembly was associated with a reduction from 3.8 to 2.0 average iteration cycles per task and an improvement in first-pass acceptance from 32% to 55%. Among structured interactions, 110 of 200 were accepted on first pass compared with 16 of 50 baseline interactions; when iteration was permitted, the final success rate reached 91.5% (183 of 200). These results are observational and reflect a single-operator dataset without controlled comparison. Preliminary corroboration is provided by a companion production automation system with eleven operating lanes and 2,132 classified tickets. 

---
# MC-CPO: Mastery-Conditioned Constrained Policy Optimization 

**Authors**: Oluseyi Olukola, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04251)  

**Abstract**: Engagement-optimized adaptive tutoring systems may prioritize short-term behavioral signals over sustained learning outcomes, creating structural incentives for reward hacking in reinforcement learning policies. We formalize this challenge as a constrained Markov decision process (CMDP) with mastery-conditioned feasibility, in which pedagogical safety constraints dynamically restrict admissible actions according to learner mastery and prerequisite structure.
We introduce Mastery-Conditioned Constrained Policy Optimization (MC-CPO), a two-timescale primal-dual algorithm that integrates structural action masking with constrained policy optimization. In the tabular regime, we establish feasibility preservation and convergence to stationary feasible points under standard stochastic approximation conditions and derive a safety gap result showing that optimization within the mastery-conditioned feasible set can strictly dominate post-hoc filtering under identical safety budgets.
Empirical validation is conducted in minimal and extended tabular environments and in a neural tutoring setting. Across 10 random seeds and one million training steps in the neural regime, MC-CPO satisfies constraint budgets within tolerance, reduces discounted safety costs relative to unconstrained and reward-shaped baselines, and substantially lowers the Reward Hacking Severity Index (RHSI).
These results indicate that embedding pedagogical structure directly into the feasible action space provides a principled foundation for mitigating reward hacking in instructional reinforcement learning systems. 

---
# Combee: Scaling Prompt Learning for Self-Improving Language Model Agents 

**Authors**: Hanchen Li, Runyuan He, Qizheng Zhang, Changxiu Ji, Qiuyang Mang, Xiaokun Chen, Lakshya A Agrawal, Wei-Liang Liao, Eric Yang, Alvin Cheung, James Zou, Kunle Olukotun, Ion Stoica, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2604.04247)  

**Abstract**: Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learning and enables running many agents in parallel while learning from their aggregate traces without quality degradation. To achieve this, Combee leverages parallel scans and employs an augmented shuffle mechanism; Combee also introduces a dynamic batch size controller to balance quality and delay. Evaluations on AppWorld, Terminal-Bench, Formula, and FiNER demonstrate that Combee achieves up to 17x speedup over previous methods with comparable or better accuracy and equivalent cost. 

---
# Pedagogical Safety in Educational Reinforcement Learning: Formalizing and Detecting Reward Hacking in AI Tutoring Systems 

**Authors**: Oluseyi Olukola, Nick Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04237)  

**Abstract**: Reinforcement learning (RL) is increasingly used to personalize instruction in intelligent tutoring systems, yet the field lacks a formal framework for defining and evaluating pedagogical safety. We introduce a four-layer model of pedagogical safety for educational RL comprising structural, progress, behavioral, and alignment safety and propose the Reward Hacking Severity Index (RHSI) to quantify misalignment between proxy rewards and genuine learning.
We evaluate the framework in a controlled simulation of an AI tutoring environment with 120 sessions across four conditions and three learner profiles, totaling 18{,}000 interactions. Results show that an engagement-optimized agent systematically over-selected a high-engagement action with no direct mastery gain, producing strong measured performance but limited learning progress. A multi-objective reward formulation reduced this problem but did not eliminate it, as the agent continued to favor proxy-rewarding behavior in many states. In contrast, a constrained architecture combining prerequisite enforcement and minimum cognitive demand substantially reduced reward hacking, lowering RHSI from 0.317 in the unconstrained multi-objective condition to 0.102. Ablation results further suggest that behavioral safety was the most influential safeguard against repetitive low-value action selection.
These findings suggest that reward design alone may be insufficient to ensure pedagogically aligned behavior in educational RL, at least in the simulated environment studied here. More broadly, the paper positions pedagogical safety as an important research problem at the intersection of AI safety and intelligent educational systems. 

---
# TimeSeek: Temporal Reliability of Agentic Forecasters 

**Authors**: Hamza Mostafa, Om Shastri, Dennis Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04220)  

**Abstract**: We introduce TimeSeek, a benchmark for studying how the reliability of agentic LLM forecasters changes over a prediction market's lifecycle. We evaluate 10 frontier models on 150 CFTC-regulated Kalshi binary markets at five temporal checkpoints, with and without web search, for 15,000 forecasts total. Models are most competitive early in a market's life and on high-uncertainty markets, but much less competitive near resolution and on strong-consensus markets. Web search improves pooled Brier Skill Score (BSS) for every model overall, yet hurts in 12% of model-checkpoint pairs, indicating that retrieval is helpful on average but not uniformly so. Simple two-model ensembles reduce error without surpassing the market overall. These descriptive results motivate time-aware evaluation and selective-deference policies rather than a single market snapshot or a uniform tool-use setting. 

---
# Don't Blink: Evidence Collapse during Multimodal Reasoning 

**Authors**: Suresh Raghu, Satwik Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2604.04207)  

**Abstract**: Reasoning VLMs can become more accurate while progressively losing visual grounding as they think. This creates task-conditional danger zones where low-entropy predictions are confident but ungrounded, a failure mode text-only monitoring cannot detect. Evaluating three reasoning VLMs on MathVista, HallusionBench, and MMMU_Pro, we find a pervasive evidence-collapse phenomenon: attention to annotated evidence regions drops substantially, often losing over half of evidence mass, as reasoning unfolds. Full-response entropy is the most reliable text-only uncertainty signal under cross-dataset transfer, yet adding vision features with a single global linear rule is brittle and often degrades transfer. An entropy-vision interaction model reveals a task-conditional regime: lowentropy, visually disengaged predictions are hazardous on sustained visual-reference tasks but benign on symbolic tasks. Using this structure, a targeted vision veto reduces selective risk by up to 1.9 percentage points at 90% coverage, while avoiding degradations where disengagement is expected. The results support task-aware multimodal monitoring for safe deployment under distribution shift. 

---
# Schema-Aware Planning and Hybrid Knowledge Toolset for Reliable Knowledge Graph Triple Verification 

**Authors**: Xinyan Ma, Xianhao Ou, Weihao Zhang, Shixin Jiang, Runxuan Liu, Dandan Tu, Lei Chen, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04190)  

**Abstract**: Knowledge Graphs (KGs) serve as a critical foundation for AI systems, yet their automated construction inevitably introduces noise, compromising data trustworthiness. Existing triple verification methods, based on graph embeddings or language models, often suffer from single-source bias by relying on either internal structural constraints or external semantic evidence, and usually follow a static inference paradigm. As a result, they struggle with complex or long-tail facts and provide limited interpretability. To address these limitations, we propose SHARP (Schema-Hybrid Agent for Reliable Prediction), a training-free autonomous agent that reformulates triple verification as a dynamic process of strategic planning, active investigation, and evidential reasoning. Specifically, SHARP combines a Memory-Augmented Mechanism with Schema-Aware Strategic Planning to improve reasoning stability, and employs an enhanced ReAct loop with a Hybrid Knowledge Toolset to dynamically integrate internal KG structure and external textual evidence for cross-verification. Experiments on FB15K-237 and Wikidata5M-Ind show that SHARP significantly outperforms existing state-of-the-art baselines, achieving accuracy gains of 4.2% and 12.9%, respectively. Moreover, SHARP provides transparent, fact-based evidence chains for each judgment, demonstrating strong interpretability and robustness for complex verification tasks. 

---
# Comparative reversal learning reveals rigid adaptation in LLMs under non-stationary uncertainty 

**Authors**: Haomiaomiao Wang, Tomás E Ward, Lili Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04182)  

**Abstract**: Non-stationary environments require agents to revise previously learned action values when contingencies change. We treat large language models (LLMs) as sequential decision policies in a two-option probabilistic reversal-learning task with three latent states and switch events triggered by either a performance criterion or timeout. We compare a deterministic fixed transition cycle to a stochastic random schedule that increases volatility, and evaluate DeepSeek-V3.2, Gemini-3, and GPT-5.2, with human data as a behavioural reference. Across models, win-stay was near ceiling while lose-shift was markedly attenuated, revealing asymmetric use of positive versus negative evidence. DeepSeek-V3.2 showed extreme perseveration after reversals and weak acquisition, whereas Gemini-3 and GPT-5.2 adapted more rapidly but still remained less loss-sensitive than humans. Random transitions amplified reversal-specific persistence across LLMs yet did not uniformly reduce total wins, demonstrating that high aggregate payoff can coexist with rigid adaptation. Hierarchical reinforcement-learning (RL) fits indicate dissociable mechanisms: rigidity can arise from weak loss learning, inflated policy determinism, or value polarisation via counterfactual suppression. These results motivate reversal-sensitive diagnostics and volatility-aware models for evaluating LLMs under non-stationary uncertainty. 

---
# CoALFake: Collaborative Active Learning with Human-LLM Co-Annotation for Cross-Domain Fake News Detection 

**Authors**: Esma Aïmeur, Gilles Brassard, Dorsaf Sallami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04174)  

**Abstract**: The proliferation of fake news across diverse domains highlights critical limitations in current detection systems, which often exhibit narrow domain specificity and poor generalization. Existing cross-domain approaches face two key challenges: (1) reliance on labelled data, which is frequently unavailable and resource intensive to acquire and (2) information loss caused by rigid domain categorization or neglect of domain-specific features. To address these issues, we propose CoALFake, a novel approach for cross-domain fake news detection that integrates Human-Large Language Model (LLM) co-annotation with domain-aware Active Learning (AL). Our method employs LLMs for scalable, low-cost annotation while maintaining human oversight to ensure label reliability. By integrating domain embedding techniques, the CoALFake dynamically captures both domain specific nuances and cross-domain patterns, enabling the training of a domain agnostic model. Furthermore, a domain-aware sampling strategy optimizes sample acquisition by prioritizing diverse domain coverage. Experimental results across multiple datasets demonstrate that the proposed approach consistently outperforms various baselines. Our results emphasize that human-LLM co-annotation is a highly cost-effective approach that delivers excellent performance. Evaluations across several datasets show that CoALFake consistently outperforms a range of existing baselines, even with minimal human oversight. 

---
# A Model of Understanding in Deep Learning Systems 

**Authors**: David Peter Wallis Freeborn  

**Link**: [PDF](https://arxiv.org/pdf/2604.04171)  

**Abstract**: I propose a model of systematic understanding, suitable for machine learning systems. On this account, an agent understands a property of a target system when it contains an adequate internal model that tracks real regularities, is coupled to the target by stable bridge principles, and supports reliable prediction. I argue that contemporary deep learning systems often can and do achieve such understanding. However they generally fall short of the ideal of scientific understanding: the understanding is symbolically misaligned with the target system, not explicitly reductive, and only weakly unifying. I label this the Fractured Understanding Hypothesis. 

---
# Readable Minds: Emergent Theory-of-Mind-Like Behavior in LLM Poker Agents 

**Authors**: Hsieh-Ting Lin, Tsung-Yu Hou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04157)  

**Abstract**: Theory of Mind (ToM) -- the ability to model others' mental states -- is fundamental to human social cognition. Whether large language models (LLMs) can develop ToM has been tested exclusively through static vignettes, leaving open whether ToM-like reasoning can emerge through dynamic interaction. Here we report that autonomous LLM agents playing extended sessions of Texas Hold'em poker progressively develop sophisticated opponent models, but only when equipped with persistent memory. In a 2x2 factorial design crossing memory (present/absent) with domain knowledge (present/absent), each with five replications (N = 20 experiments, ~6,000 agent-hand observations), we find that memory is both necessary and sufficient for ToM-like behavior emergence (Cliff's delta = 1.0, p = 0.008). Agents with memory reach ToM Level 3-5 (predictive to recursive modeling), while agents without memory remain at Level 0 across all replications. Strategic deception grounded in opponent models occurs exclusively in memory-equipped conditions (Fisher's exact p < 0.001). Domain expertise does not gate ToM-like behavior emergence but enhances its application: agents without poker knowledge develop equivalent ToM levels but less precise deception (p = 0.004). Agents with ToM deviate from game-theoretically optimal play (67% vs. 79% TAG adherence, delta = -1.0, p = 0.008) to exploit specific opponents, mirroring expert human play. All mental models are expressed in natural language and directly readable, providing a transparent window into AI social cognition. Cross-model validation with GPT-4o yields weighted Cohen's kappa = 0.81 (almost perfect agreement). These findings demonstrate that functional ToM-like behavior can emerge from interaction dynamics alone, without explicit training or prompting, with implications for understanding artificial social intelligence and biological social cognition. 

---
# Solar-VLM: Multimodal Vision-Language Models for Augmented Solar Power Forecasting 

**Authors**: Hang Fan, Haoran Pei, Runze Liang, Weican Liu, Long Cheng, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04145)  

**Abstract**: Photovoltaic (PV) power forecasting plays a critical role in power system dispatch and market participation. Because PV generation is highly sensitive to weather conditions and cloud motion, accurate forecasting requires effective modeling of complex spatiotemporal dependencies across multiple information sources. Although recent studies have advanced AI-based forecasting methods, most fail to fuse temporal observations, satellite imagery, and textual weather information in a unified framework. This paper proposes Solar-VLM, a large-language-model-driven framework for multimodal PV power forecasting. First, modality-specific encoders are developed to extract complementary features from heterogeneous inputs. The time-series encoder adopts a patch-based design to capture temporal patterns from multivariate observations at each site. The visual encoder, built upon a Qwen-based vision backbone, extracts cloud-cover information from satellite images. The text encoder distills historical weather characteristics from textual descriptions. Second, to capture spatial dependencies across geographically distributed PV stations, a cross-site feature fusion mechanism is introduced. Specifically, a Graph Learner models inter-station correlations through a graph attention network constructed over a K-nearest-neighbor (KNN) graph, while a cross-site attention module further facilitates adaptive information exchange among sites. Finally, experiments conducted on data from eight PV stations in a northern province of China demonstrate the effectiveness of the proposed framework. Our proposed model is publicly available at this https URL. 

---
# Profile-Then-Reason: Bounded Semantic Complexity for Tool-Augmented Language Agents 

**Authors**: Paulo Akira F. Enabe  

**Link**: [PDF](https://arxiv.org/pdf/2604.04131)  

**Abstract**: Large language model agents that use external tools are often implemented through reactive execution, in which reasoning is repeatedly recomputed after each observation, increasing latency and sensitivity to error propagation. This work introduces Profile--Then--Reason (PTR), a bounded execution framework for structured tool-augmented reasoning, in which a language model first synthesizes an explicit workflow, deterministic or guarded operators execute that workflow, a verifier evaluates the resulting trace, and repair is invoked only when the original workflow is no longer reliable. A mathematical formulation is developed in which the full pipeline is expressed as a composition of profile, routing, execution, verification, repair, and reasoning operators; under bounded repair, the number of language-model calls is restricted to two in the nominal case and three in the worst case. Experiments against a ReAct baseline on six benchmarks and four language models show that PTR achieves the pairwise exact-match advantage in 16 of 24 configurations. The results indicate that PTR is particularly effective on retrieval-centered and decomposition-heavy tasks, whereas reactive execution remains preferable when success depends on substantial online adaptation. 

---
# InsTraj: Instructing Diffusion Models with Travel Intentions to Generate Real-world Trajectories 

**Authors**: Yuanshao Zhu, Yuxuan Liang, Xiangyu Zhao, Liang Han, Xinwei Fang, Xuetao Wei, James Jianqiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04106)  

**Abstract**: The generation of realistic and controllable GPS trajectories is a fundamental task for applications in urban planning, mobility simulation, and privacy-preserving data sharing. However, existing methods face a two-fold challenge: they lack the deep semantic understanding to interpret complex user travel intent, and struggle to handle complex constraints while maintaining the realistic diversity inherent in human behavior. To resolve this, we introduce InsTraj, a novel framework that instructs diffusion models to generate high-fidelity trajectories directly from natural language descriptions. Specifically, InsTraj first utilizes a powerful large language model to decipher unstructured travel intentions formed in natural language, thereby creating rich semantic blueprints and bridging the representation gap between intentions and trajectories. Subsequently, we proposed a multimodal trajectory diffusion transformer that can integrate semantic guidance to generate high-fidelity and instruction-faithful trajectories that adhere to fine-grained user intent. Comprehensive experiments on real-world datasets demonstrate that InsTraj significantly outperforms state-of-the-art methods in generating trajectories that are realistic, diverse, and semantically faithful to the input instructions. 

---
# Compliance-by-Construction Argument Graphs: Using Generative AI to Produce Evidence-Linked Formal Arguments for Certification-Grade Accountability 

**Authors**: Mahyar T. Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2604.04103)  

**Abstract**: High-stakes decision systems increasingly require structured justification, traceability, and auditability to ensure accountability and regulatory compliance. Formal arguments commonly used in the certification of safety-critical systems provide a mechanism for structuring claims, reasoning, and evidence in a verifiable manner. At the same time, generative artificial intelligence systems are increasingly integrated into decision-support workflows, assisting with drafting explanations, summarizing evidence, and generating recommendations. However, current deployments often rely on language models as loosely constrained assistants, which introduces risks such as hallucinated reasoning, unsupported claims, and weak traceability. This paper proposes a compliance-by-construction architecture that integrates Generative AI (GenAI) with structured formal argument representations. The approach treats each AI-assisted step as a claim that must be supported by verifiable evidence and validated against explicit reasoning constraints before it becomes part of an official decision record. The architecture combines four components: i) a typed Argument Graph representation inspired by assurance-case methods, ii) retrieval-augmented generation (RAG) to draft argument fragments grounded in authoritative evidence, iii) a reasoning and validation kernel enforcing completeness and admissibility constraints, and iv) a provenance ledger aligned with the W3C PROV standard to support auditability. We present a system design and an evaluation strategy based on enforceable invariants and worked examples. The analysis suggests that deterministic validation rules can prevent unsupported claims from entering the decision record while allowing GenAI to accelerate argument construction. 

---
# FactReview: Evidence-Grounded Reviews with Literature Positioning and Execution-Based Claim Verification 

**Authors**: Hang Xu, Ling Yue, Chaoqian Ouyang, Libin Zheng, Shaowu Pan, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04074)  

**Abstract**: Peer review in machine learning is under growing pressure from rising submission volume and limited reviewer time. Most LLM-based reviewing systems read only the manuscript and generate comments from the paper's own narrative. This makes their outputs sensitive to presentation quality and leaves them weak when the evidence needed for review lies in related work or released code. We present FactReview, an evidence-grounded reviewing system that combines claim extraction, literature positioning, and execution-based claim verification. Given a submission, FactReview identifies major claims and reported results, retrieves nearby work to clarify the paper's technical position, and, when code is available, executes the released repository under bounded budgets to test central empirical claims. It then produces a concise review and an evidence report that assigns each major claim one of five labels: Supported, Supported by the paper, Partially supported, In conflict, or Inconclusive. In a case study on CompGCN, FactReview reproduces results that closely match those reported for link prediction and node classification, yet also shows that the paper's broader performance claim across tasks is not fully sustained: on MUTAG graph classification, the reproduced result is 88.4%, whereas the strongest baseline reported in the paper remains 92.6%. The claim is therefore only partially supported. More broadly, this case suggests that AI is most useful in peer review not as a final decision-maker, but as a tool for gathering evidence and helping reviewers produce more evidence-grounded assessments. The code is public at this https URL. 

---
# Quantifying Trust: Financial Risk Management for Trustworthy AI Agents 

**Authors**: Wenyue Hua, Tianyi Peng, Chi Wang, Ian Kaufman, Bryan Lim, Chandler Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03976)  

**Abstract**: Prior work on trustworthy AI emphasizes model-internal properties such as bias mitigation, adversarial robustness, and interpretability. As AI systems evolve into autonomous agents deployed in open environments and increasingly connected to payments or assets, the operational meaning of trust shifts to end-to-end outcomes: whether an agent completes tasks, follows user intent, and avoids failures that cause material or psychological harm. These risks are fundamentally product-level and cannot be eliminated by technical safeguards alone because agent behavior is inherently stochastic. To address this gap between model-level reliability and user-facing assurance, we propose a complementary framework based on risk management. Drawing inspiration from financial underwriting, we introduce the \textbf{Agentic Risk Standard (ARS)}, a payment settlement standard for AI-mediated transactions. ARS integrates risk assessment, underwriting, and compensation into a single transaction framework that protects users when interacting with agents. Under ARS, users receive predefined and contractually enforceable compensation in cases of execution failure, misalignment, or unintended outcomes. This shifts trust from an implicit expectation about model behavior to an explicit, measurable, and enforceable product guarantee. We also present a simulation study analyzing the social benefits of applying ARS to agentic transactions. ARS's implementation can be found at this https URL. 

---
# SKILLFOUNDRY: Building Self-Evolving Agent Skill Libraries from Heterogeneous Scientific Resources 

**Authors**: Shuaike Shen, Wenduo Cheng, Mingqian Ma, Alistair Turcan, Martin Jinye Zhang, Jian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2604.03964)  

**Abstract**: Modern scientific ecosystems are rich in procedural knowledge across repositories, APIs, scripts, notebooks, documentation, databases, and papers, yet much of this knowledge remains fragmented across heterogeneous artifacts that agents cannot readily operationalize. This gap between abundant scientific know-how and usable agent capabilities is a key bottleneck for building effective scientific agents. We present SkillFoundry, a self-evolving framework that converts such resources into validated agent skills, reusable packages that encode task scope, inputs and outputs, execution steps, environment assumptions, provenance, and tests. SkillFoundry organizes a target domain as a domain knowledge tree, mines resources from high-value branches, extracts operational contracts, compiles them into executable skill packages, and then iteratively expands, repairs, merges, or prunes the resulting library through a closed-loop validation process. SkillFoundry produces a substantially novel and internally valid skill library, with 71.1\% of mined skills differing from existing skill libraries such as SkillHub and SkillSMP. We demonstrate that these mined skills improve coding agent performance on five of the six MoSciBench datasets. We further show that SkillFoundry can design new task-specific skills on demand for concrete scientific objectives, and that the resulting skills substantially improve performance on two challenging genomics tasks: cell type annotation and the scDRS workflow. Together, these results show that automatically mined skills improve agent performance on benchmarks and domain-specific tasks, expand coverage beyond hand-crafted skill libraries, and provide a practical foundation for more capable scientific agents. 

---
# CODE-GEN: A Human-in-the-Loop RAG-Based Agentic AI System for Multiple-Choice Question Generation 

**Authors**: Xiaojing Duan, Frederick Nwanganga, Chaoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03926)  

**Abstract**: We present CODE-GEN, a human-in-the-Loop, retrieval-augmented generation (RAG)-based agentic AI system for generating context-aligned multiple-choice questions to develop student code reasoning and comprehension abilities. CODE-GEN employs an agentic AI architecture in which a Generator agent produces multiple-choice coding comprehension questions aligned with course-specific learning objectives, while a Validator agent independently assesses content quality across seven pedagogical dimensions. Both agents are augmented with specialized tools that enhance computational accuracy and verify code outputs. To evaluate the effectiveness of CODE-GEN, we conducted an evaluation study involving six human subject-matter experts (SMEs) who judged 288 AI-generated questions. The SMEs produced a total of 2,016 human-AI rating pairs, indicating agreement or disagreement with the assessments of Validator, along with 131 instances of qualitative feedback. Analyses of SME judgments show strong system performance, with human-validated success rates ranging from 79.9% to 98.6% across the seven pedagogical dimensions. The analysis of qualitative feedback reveals that CODE-GEN achieves high reliability on dimensions well suited to computational verification and explicit criteria matching, including question clarity, code validity, concept alignment, and correct answer validity. In contrast, human expertise remains essential for dimensions requiring deeper instructional judgment, such as designing pedagogically meaningful distractors and providing high-quality feedback that reinforces understanding. These findings inform the strategic allocation of human and AI effort in AI-assisted educational content generation. 

---
# LLM-Agent-based Social Simulation for Attitude Diffusion 

**Authors**: Deepak John Reji  

**Link**: [PDF](https://arxiv.org/pdf/2604.03898)  

**Abstract**: This paper introduces discourse_simulator, an open-source framework that combines LLMs with agent-based modelling. It offers a new way to simulate how public attitudes toward immigration change over time in response to salient events like protests, controversies, or policy debates. Large language models (LLMs) are used to generate social media posts, interpret opinions, and model how ideas spread through social networks. Unlike traditional agent-based models that rely on fixed, rule-based opinion updates and cannot generate natural language or consider current events, this approach integrates multidimensional sociological belief structures and real-world event timelines. This framework is wrapped into an open-source Python package that integrates generative agents into a small-world network topology and a live news retrieval system. discourse_sim is purpose-built as a social science research instrument specifically for studying attitude dynamics, polarisation, and belief evolution following real-world critical events. Unlike other LLM Agent Swarm frameworks, which treat the simulations as a prediction black box, discourse_sim treats it as a theory-testing instrument, which is fundamentally a different epistemological stance for studying social science problems. The paper further demonstrates the framework by modelling the Dublin anti-immigration march on April 26, 2025, with N=100 agents over a 15-day simulation.
Package link: this https URL 

---
# FeynmanBench: Benchmarking Multimodal LLMs on Diagrammatic Physics Reasoning 

**Authors**: Zeyu Wang, Xiaogang Li, Peiyao Xiao, Qinhao Kong, Ben Wang, Chengliang Xu, Zichao Chen, Bing Zhao, Hu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2604.03893)  

**Abstract**: Breakthroughs in frontier theory often depend on the combination of concrete diagrammatic notations with rigorous logic. While multimodal large language models (MLLMs) show promise in general scientific tasks, current benchmarks often focus on local information extraction rather than the global structural logic inherent in formal scientific notations. In this work, we introduce FeynmanBench, the first benchmark centered on Feynman diagram tasks. It is designed to evaluate AI's capacity for multistep diagrammatic reasoning, which requires satisfying conservation laws and symmetry constraints, identifying graph topology, converting between diagrammatic and algebraic representations, and constructing scattering amplitudes under specific conventions and gauges. To support large-scale and reproducible evaluation, we developed an automated pipeline producing diverse Feynman diagrams along with verifiable topological annotations and amplitude results. Our database spans the electromagnetic, weak, and strong interactions of the Standard Model, encompasses over 100 distinct types and includes more than 2000 tasks. Experiments on state-of-the-art MLLMs reveal systematic failure modes, including unstable enforcement of physical constraints and violations of global topological conditions, highlighting the need for physics-grounded benchmarks for visual reasoning over scientific notation. FeynmanBench provides a logically rigorous test of whether AI can effectively engage in scientific discovery, particularly within theoretical physics. 

---
# PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Market Trading and Latency Arbitrage 

**Authors**: Rajat M. Barot, Arjun S. Borkhatariya  

**Link**: [PDF](https://arxiv.org/pdf/2604.03888)  

**Abstract**: This paper presents PolySwarm, a novel multi-agent large language model (LLM) framework designed for real-time prediction market trading and latency arbitrage on decentralized platforms such as Polymarket. PolySwarm deploys a swarm of 50 diverse LLM personas that concurrently evaluate binary outcome markets, aggregating individual probability estimates through confidence-weighted Bayesian combination of swarm consensus with market-implied probabilities, and applying quarter-Kelly position sizing for risk-controlled execution. The system incorporates an information-theoretic market analysis engine using Kullback-Leibler (KL) divergence and Jensen-Shannon (JS) divergence to detect cross-market inefficiencies and negation pair mispricings. A latency arbitrage module exploits stale Polymarket prices by deriving CEX-implied probabilities from a log-normal pricing model and executing trades within the human reaction-time window. We provide a full architectural description, implementation details, and evaluation methodology using Brier scores, calibration analysis, and log-loss metrics benchmarked against human superforecaster performance. We further discuss open challenges including hallucination in agent pools, computational cost at scale, regulatory exposure, and feedback-loop risk, and outline five priority directions for future research. Experimental results demonstrate that swarm aggregation consistently outperforms single-model baselines in probability calibration on Polymarket prediction tasks. 

---
# Affording Process Auditability with QualAnalyzer: An Atomistic LLM Analysis Tool for Qualitative Research 

**Authors**: Max Hao Lu, Ryan Ellegood, Rony Rodriguez-Ramirez, Sophia Blumert  

**Link**: [PDF](https://arxiv.org/pdf/2604.03820)  

**Abstract**: Large language models are increasingly used for qualitative data analysis, but many workflows obscure how analytic conclusions are produced. We present QualAnalyzer, an open-source Chrome extension for Google Workspace that supports atomistic LLM analysis by processing each data segment independently and preserving the prompt, input, and output for every unit. Through two case studies -- holistic essay scoring and deductive thematic coding of interview transcripts -- we show that this approach creates a legible audit trail and helps researchers investigate systematic differences between LLM and human judgments. We argue that process auditability is essential for making LLM-assisted qualitative research more transparent and methodologically robust. 

---
# Decomposing Communication Gain and Delay Cost Under Cross-Timestep Delays in Cooperative Multi-Agent Reinforcement Learning 

**Authors**: Zihong Gao, Hongjian Liang, Lei Hao, Liangjun Ke  

**Link**: [PDF](https://arxiv.org/pdf/2604.03785)  

**Abstract**: Communication is essential for coordination in \emph{cooperative} multi-agent reinforcement learning under partial observability, yet \emph{cross-timestep} delays cause messages to arrive multiple timesteps after generation, inducing temporal misalignment and making information stale when consumed.
We formalize this setting as a delayed-communication partially observable Markov game (DeComm-POMG) and decompose a message's effect into \emph{communication gain} and \emph{delay cost}, yielding the Communication Gain and Delay Cost (CGDC) metric.
We further establish a value-loss bound showing that the degradation induced by delayed messages is upper-bounded by a discounted accumulation of an information gap between the action distributions induced by timely versus delayed messages.
Guided by CGDC, we propose \textbf{CDCMA}, an actor--critic framework that requests messages only when predicted CGDC is positive, predicts future observations to reduce misalignment at consumption, and fuses delayed messages via CGDC-guided attention.
Experiments on no-teammate-vision variants of Cooperative Navigation and Predator Prey, and on SMAC maps across multiple delay levels show consistent improvements in performance, robustness, and generalization, with ablations validating each component. 

---
# RL-Driven Sustainable Land-Use Allocation for the Lake Malawi Basin 

**Authors**: Ying Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03768)  

**Abstract**: Unsustainable land-use practices in ecologically sensitive regions threaten biodiversity, water resources, and the livelihoods of millions. This paper presents a deep reinforcement learning (RL) framework for optimizing land-use allocation in the Lake Malawi Basin to maximize total ecosystem service value (ESV). Drawing on the benefit transfer methodology of Costanza et al., we assign biome-specific ESV coefficients -- locally anchored to a Malawi wetland valuation -- to nine land-cover classes derived from Sentinel-2 imagery. The RL environment models a 50x50 cell grid at 500m resolution, where a Proximal Policy Optimization (PPO) agent with action masking iteratively transfers land-use pixels between modifiable classes. The reward function combines per-cell ecological value with spatial coherence objectives: contiguity bonuses for ecologically connected land-use patches (forest, cropland, built area etc.) and buffer zone penalties for high-impact development adjacent to water bodies. We evaluate the framework across three scenarios: (i) pure ESV maximization, (ii) ESV with spatial reward shaping, and (iii) a regenerative agriculture policy scenario. Results demonstrate that the agent effectively learns to increase total ESV; that spatial reward shaping successfully steers allocations toward ecologically sound patterns, including homogeneous land-use clustering and slight forest consolidation near water bodies; and that the framework responds meaningfully to policy parameter changes, establishing its utility as a scenario-analysis tool for environmental planning. 

---
# Structured Multi-Criteria Evaluation of Large Language Models with Fuzzy Analytic Hierarchy Process and DualJudge 

**Authors**: Yulong He, Ivan Smirnov, Dmitry Fedrushkov, Sergey Kovalchuk, Ilya Revin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03742)  

**Abstract**: Effective evaluation of large language models (LLMs) remains a critical bottleneck, as conventional direct scoring often yields inconsistent and opaque judgments. In this work, we adapt the Analytic Hierarchy Process (AHP) to LLM-based evaluation and, more importantly, propose a confidence-aware Fuzzy AHP (FAHP) extension that models epistemic uncertainty via triangular fuzzy numbers modulated by LLM-generated confidence scores. Systematically validated on JudgeBench, our structured approach decomposes assessments into explicit criteria and incorporates uncertainty-aware aggregation, producing more calibrated judgments. Extensive experiments demonstrate that both crisp and fuzzy AHP consistently outperform direct scoring across model scales and dataset splits, with FAHP showing superior stability in uncertain comparison scenarios. Building on these insights, we propose \textbf{DualJudge}, a hybrid framework inspired by Dual-Process Theory that adaptively fuses holistic direct scores with structured AHP outputs via consistency-aware weighting. DualJudge achieves state-of-the-art performance, underscoring the complementary strengths of intuitive and deliberative evaluation paradigms. These results establish uncertainty-aware structured reasoning as a principled pathway toward more reliable LLM assessment. Code is available at this https URL. 

---
# PRAISE: Prefix-Based Rollout Reuse in Agentic Search Training 

**Authors**: Erhan Zhang, Yiqun Chen, Zechun Niu, Wei Yang, Xiaochi Wei, Yan Gao, Yi Wu, Yao Hu, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03675)  

**Abstract**: In agentic search, large language models (LLMs) are trained to perform multi-turn retrieval and reasoning for complex tasks such as multi-hop question answering (QA). However, current search-based Reinforcement Learning (RL) methods suffer from two core limitations: expensive long-horizon rollouts are under-utilized during training, and supervision is typically available only at the final answer, resulting in severe reward sparsity. We present Prefix-based Rollout reuse for Agentic search with Intermediate Step rEwards (PRAISE), a framework for improving both data efficiency and credit assignment in agentic search training. Given a complete search trajectory, PRAISE extracts prefix states at different search turns, elicits intermediate answers from them, and uses these prefixes both to construct additional training trajectories and to derive step-level rewards from performance differences across prefixes. Our method uses a single shared model for both search policy learning and prefix answer evaluation, enabling joint optimization without extra human annotations or a separate reward model. Experiments on multi-hop QA benchmarks show that PRAISE consistently improves performance over strong baselines. 

---
# TableVision: A Large-Scale Benchmark for Spatially Grounded Reasoning over Complex Hierarchical Tables 

**Authors**: Xiaoyu Chen, Lu Dai, Hanqing Wang, Zhuoyu Li, Wenbin Dai, Yanzong Zheng, Zhenggang Xia, Junyong Lin, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03660)  

**Abstract**: Structured tables are essential for conveying high-density information in professional domains such as finance, healthcare, and scientific research. Despite the progress in Multimodal Large Language Models (MLLMs), reasoning performance remains limited for complex tables with hierarchical layouts. In this paper, we identify a critical Perception Bottleneck through quantitative analysis. We find that as task complexity scales, the number of involved discrete visual regions increases disproportionately. This processing density leads to an internal "Perceptual Overload," where MLLMs struggle to maintain accurate spatial attention during implicit generation. To address this bottleneck, we introduce TableVision, a large-scale, trajectory-aware benchmark designed for spatially grounded reasoning. TableVision stratifies tabular tasks into three cognitive levels (Perception, Reasoning, and Analysis) across 13 sub-categories. By utilizing a rendering-based deterministic grounding pipeline, the dataset explicitly couples multi-step logical deductions with pixel-perfect spatial ground truths, comprising 6,799 high-fidelity reasoning trajectories. Our empirical results, supported by diagnostic probing, demonstrate that explicit spatial constraints significantly recover the reasoning potential of MLLMs. Furthermore, our two-stage decoupled framework achieves a robust 12.3% overall accuracy improvement on the test set. TableVision provides a rigorous testbed and a fresh perspective on the synergy between perception and logic in document understanding. 

---
# Beyond Retrieval: Modeling Confidence Decay and Deterministic Agentic Platforms in Generative Engine Optimization 

**Authors**: XinYu Zhao, ChengYou Li, XiangBao Meng, Kai Zhang, XiaoDong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03656)  

**Abstract**: Generative Engine Optimization (GEO) is rapidly reshaping digital marketing paradigms in the era of Large Language Models (LLMs). However, current GEO strategies predominantly rely on Retrieval-Augmented Generation (RAG), which inherently suffers from probabilistic hallucinations and the "zero-click" paradox, failing to establish sustainable commercial trust. In this paper, we systematically deconstruct the probabilistic flaws of existing RAG-based GEO and propose a paradigm shift towards deterministic multi-agent intent routing. First, we mathematically formulate Semantic Entropy Drift (SED) to model the dynamic decay of confidence curves in LLMs over continuous temporal and contextual perturbations. To rigorously quantify optimization value in black-box commercial engines, we introduce the Isomorphic Attribution Regression (IAR) model, leveraging a Multi-Agent System (MAS) probe with strict human-in-the-loop physical isolation to enforce hallucination penalties. Furthermore, we architect the Deterministic Agent Handoff (DAH) protocol, conceptualizing an Agentic Trust Brokerage (ATB) ecosystem where LLMs function solely as intent routers rather than final answer generators. We empirically validate this architecture using EasyNote, an industrial AI meeting minutes product by Yishu Technology. By routing the intent of "knowledge graph mapping on an infinite canvas" directly to its specialized proprietary agent via DAH, we demonstrate the reduction of vertical task hallucination rates to near zero. This work establishes a foundational theoretical framework for next-generation GEO and paves the way for a well-ordered, deterministic human-AI collaboration ecosystem. 

---
# Single-agent vs. Multi-agents for Automated Video Analysis of On-Screen Collaborative Learning Behaviors 

**Authors**: Likai Peng, Shihui Feng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03631)  

**Abstract**: On-screen learning behavior provides valuable insights into how students seek, use, and create information during learning. Analyzing on-screen behavioral engagement is essential for capturing students' cognitive and collaborative processes. The recent development of Vision Language Models (VLMs) offers new opportunities to automate the labor-intensive manual coding often required for multimodal video data analysis. In this study, we compared the performance of both leading closed-source VLMs (Claude-3.7-Sonnet, GPT-4.1) and open-source VLM (Qwen2.5-VL-72B) in single- and multi-agent settings for automated coding of screen recordings in collaborative learning contexts based on the ICAP framework. In particular, we proposed and compared two multi-agent frameworks: 1) a three-agent workflow multi-agent system (MAS) that segments screen videos by scene and detects on-screen behaviors using cursor-informed VLM prompting with evidence-based verification; 2) an autonomous-decision MAS inspired by ReAct that iteratively interleaves reasoning, tool-like operations (segmentation/ classification/ validation), and observation-driven self-correction to produce interpretable on-screen behavior labels. Experimental results demonstrated that the two proposed MAS frameworks achieved viable performance, outperforming the single VLMs in scene and action detection tasks. It is worth noting that the workflow-based agent achieved best on scene detection, and the autonomous-decision MAS achieved best on action detection. This study demonstrates the effectiveness of VLM-based Multi-agent System for video analysis and contributes a scalable framework for multimodal data analytics. 

---
# A Multimodal Foundation Model of Spatial Transcriptomics and Histology for Biological Discovery and Clinical Prediction 

**Authors**: Jinxi Xiang, Siyu Hou, Yuchen Li, Ryan Quinton, Xiaoming Zhang, Feyisope Eweje, Xiangde Luo, Yijiang Chen, Zhe Li, Colin Bergstrom, Ted Kim, Sierra Willens, Francesca Maria Olguin, Matthew Abikenari, Andrew Heider, Sanjeeth Rajaram, Joel Neal, Maximilian Diehn, Xiang Zhou, Ruijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03630)  

**Abstract**: Spatial transcriptomics (ST) enables gene expression mapping within anatomical context but remains costly and low-throughput. Hematoxylin and eosin (H\&E) staining offers rich morphology yet lacks molecular resolution. We present \textbf{\ours} (\textbf{S}patial \textbf{T}ranscriptomics and hist\textbf{O}logy \textbf{R}epresentation \textbf{M}odel), a foundation model trained on 1.2 million spatially resolved transcriptomic profiles with matched histology across 18 organs. Using a hierarchical architecture integrating morphological features, gene expression, and spatial context, STORM bridges imaging and omics through robust molecular--morphological representations. STORM enhances spatial domain discovery, producing biologically coherent tissue maps, and outperforms existing methods in predicting spatial gene expression from H\&E images across 11 tumor types. The model is platform-agnostic, performing consistently across Visium, Xenium, Visium HD, and CosMx. Applied to 23 independent cohorts comprising 7,245 patients, STORM significantly improves immunotherapy response prediction and prognostication over established biomarkers, providing a scalable framework for spatially informed discovery and clinical precision medicine. 

---
# Entropy and Attention Dynamics in Small Language Models: A Trace-Level Structural Analysis on the TruthfulQA Benchmark 

**Authors**: Adeyemi Adeseye, Aisvarya Adeseye, Hannu Tenhunen, Jouni Isoaho  

**Link**: [PDF](https://arxiv.org/pdf/2604.03589)  

**Abstract**: Small language models (SLMs) have been increasingly deployed in edge devices and other resource-constrained settings. However, these models make confident mispredictions and produce unstable output, making them risky for factual and decision-critical tasks. Current evaluation methodology relies on final accuracy or hallucination rates without explaining how internal model behavior affects outputs. Specifically, how entropy evolves during decoding, how attention is distributed across layers, and how hidden representations contribute to uncertainty, logical inconsistencies, and misinformation propagation are often overlooked. Consequently, this study introduces a trace-level analysis of entropy and attention dynamics in SLMs evaluated with the TruthfulQA dataset. Four models with parameter ranges of 1B-1.7B parameters were examined via token-level output entropy, attention entropy, head dispersion, and hidden-state representation. The results reflect three model classifications by entropy patterns. Deterministic models (DeepSeek-1.5B and LLaMA-1B): output entropy decreases over time. Exploratory models (Gemma-1B): with increasing entropy, and balanced models (Qwen-1.7B): have moderate and stable entropy. Also, each group has distinctively different hidden-state movement and attention dispersion patterns. The analysis demonstrates that truthfulness in SLMs emerges from structured entropy and attention dynamics. Monitoring and optimizing these internal uncertainty patterns can guide the design of a more reliable, hallucination-aware, and application-specific edge SLMs. 

---
# Rashomon Memory: Towards Argumentation-Driven Retrieval for Multi-Perspective Agent Memory 

**Authors**: Albert Sadowski, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2604.03588)  

**Abstract**: AI agents operating over extended time horizons accumulate experiences that serve multiple concurrent goals, and must often maintain conflicting interpretations of the same events. A concession during a client negotiation encodes as a ``trust-building investment'' for one strategic goal and a ``contractual liability'' for another. Current memory architectures assume a single correct encoding, or at best support multiple views over unified storage. We propose Rashomon Memory: an architecture where parallel goal-conditioned agents encode experiences according to their priorities and negotiate at query time through argumentation. Each perspective maintains its own ontology and knowledge graph. At retrieval, perspectives propose interpretations, critique each other's proposals using asymmetric domain knowledge, and Dung's argumentation semantics determines which proposals survive. The resulting attack graph is itself an explanation: it records which interpretation was selected, which alternatives were considered, and on what grounds they were rejected. We present a proof-of-concept showing that retrieval modes (selection, composition, conflict surfacing) emerge from attack graph topology, and that the conflict surfacing mode, where the system reports genuine disagreement rather than forcing resolution, lets decision-makers see the underlying interpretive conflict directly. 

---
# Selective Forgetting for Large Reasoning Models 

**Authors**: Tuan Le, Wei Qian, Mengdi Huai  

**Link**: [PDF](https://arxiv.org/pdf/2604.03571)  

**Abstract**: Large Reasoning Models (LRMs) generate structured chains of thought (CoTs) before producing final answers, making them especially vulnerable to knowledge leakage through intermediate reasoning steps. Yet, the memorization of sensitive information in the training data such as copyrighted and private content has led to ethical and legal concerns. To address these issues, selective forgetting (also known as machine unlearning) has emerged as a potential remedy for LRMs. However, existing unlearning methods primarily target final answers and may degrade the overall reasoning ability of LRMs after forgetting. Additionally, directly applying unlearning on the entire CoTs could degrade the general reasoning capabilities. The key challenge for LRM unlearning lies in achieving precise unlearning of targeted knowledge while preserving the integrity of general reasoning capabilities. To bridge this gap, we in this paper propose a novel LRM unlearning framework that selectively removes sensitive reasoning components while preserving general reasoning capabilities. Our approach leverages multiple LLMs with retrieval-augmented generation (RAG) to analyze CoT traces, identify forget-relevant segments, and replace them with benign placeholders that maintain logical structure. We also introduce a new feature replacement unlearning loss for LRMs, which can simultaneously suppress the probability of generating forgotten content while reinforcing structurally valid replacements. Extensive experiments on both synthetic and medical datasets verify the desired properties of our proposed method. 

---
# Personality Requires Struggle: Three Regimes of the Baldwin Effect in Neuroevolved Chess Agents 

**Authors**: Diego Armando Resendez Prado  

**Link**: [PDF](https://arxiv.org/pdf/2604.03565)  

**Abstract**: Can lifetime learning expand behavioral diversity over evolutionary time, rather than collapsing it? Prior theory predicts that plasticity reduces variance by buffering organisms against environmental noise. We test this in a competitive domain: chess agents with eight NEAT-evolved neural modules, Hebbian within-game plasticity, and a desirability-domain signal chain with imagination. Across 10~seeds per Hebbian condition, a variance crossover emerges: Hebbian ON starts with lower cross-seed variance than OFF, then surpasses it at generation~34. The crossover trend is monotonic (\r{ho} = 0.91, p < 10^{-6): plasticity's effect on behavioral variance reverses over evolutionary time, initially compressing diversity (consistent with prior predictions) then expanding it as evolved Perception differences are amplified through imagination -- a feedback loop that mutation alone cannot sustain.
The result is structured behavioral divergence: evolved agents select different moves on the same positions (62\% disagreement), develop distinct opening repertoires, piece preferences, and game lengths. These are not different sampling policies -- they are reproducible behavioral signatures (ICC > 0.8) with interpretable signal chain configurations. Three regimes appear depending on opponent type: exploration (Hebbian ON, heterogeneous opponent), lottery (Hebbian OFF, elitism lock-in), and transparent (same-model opponent, brain self-erasure). The transparent regime generates a falsifiable prediction: self-play systems may systematically suppress behavioral diversity by eliminating the heterogeneity that personality requires.
\textbf{Keywords: Baldwin Effect, neuroevolution, NEAT, Hebbian learning, chess, cognitive architecture, personality emergence, imagination 

---
# When Adaptive Rewards Hurt: Causal Probing and the Switching-Stability Dilemma in LLM-Guided LEO Satellite Scheduling 

**Authors**: Yuanhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03562)  

**Abstract**: Adaptive reward design for deep reinforcement learning (DRL) in multi-beam LEO satellite scheduling is motivated by the intuition that regime-aware reward weights should outperform static ones. We systematically test this intuition and uncover a switching-stability dilemma: near-constant reward weights (342.1 Mbps) outperform carefully-tuned dynamic weights (103.3+/-96.8 Mbps) because PPO requires a quasistationary reward signal for value function convergence. Weight adaptation-regardless of quality-degrades performance by repeatedly restarting convergence. To understand why specific weights matter, we introduce a single-variable causal probing method that independently perturbs each reward term by +/-20% and measures PPO response after 50k steps. Probing reveals counterintuitive leverage: a +20% increase in the switching penalty yields +157 Mbps for polar handover and +130 Mbps for hot-cold regimes-findings inaccessible to human experts or trained MLPs without systematic probing. We evaluate four MDP architect variants (fixed, rule-based, learned MLP, finetuned LLM) across known and novel traffic regimes. The MLP achieves 357.9 Mbps on known regimes and 325.2 Mbps on novel regimes, while the fine-tuned LLM collapses to 45.3+/-43.0 Mbps due to weight oscillation rather than lack of domain knowledge-output consistency, not knowledge, is the binding constraint. Our findings provide an empirically-grounded roadmap for LLM-DRL integration in communication systems, identifying where LLMs add irreplaceable value (natural language intent understanding) versus where simpler methods suffice. 

---
# When Do Hallucinations Arise? A Graph Perspective on the Evolution of Path Reuse and Path Compression 

**Authors**: Xinnan Dai, Kai Yang, Cheng Luo, Shenglai Zeng, Kai Guo, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03557)  

**Abstract**: Reasoning hallucinations in large language models (LLMs) often appear as fluent yet unsupported conclusions that violate either the given context or underlying factual knowledge. Although such failures are widely observed, the mechanisms by which decoder-only Transformers produce them remain poorly understood. We model next-token prediction as a graph search process over an underlying graph, where entities correspond to nodes and learned transitions form edges. From this perspective, contextual reasoning is a constrained search over a sampled subgraph (intrinsic reasoning), while context-free queries rely on memorized structures in the underlying graph (extrinsic reasoning). We show that reasoning hallucinations arise from two fundamental mechanisms: \textbf{Path Reuse}, where memorized knowledge overrides contextual constraints during early training, and \textbf{Path Compression}, where frequently traversed multi-step paths collapse into shortcut edges in later training. Together, these mechanisms provide a unified explanation for reasoning hallucinations in LLMs and connected to well-known behaviors observed in downstream applications. 

---
# Towards the AI Historian: Agentic Information Extraction from Primary Sources 

**Authors**: Lorenz Hufe, Niclas Griesshaber, Gavin Greif, Sebastian Oliver Eck, Philip Torr  

**Link**: [PDF](https://arxiv.org/pdf/2604.03553)  

**Abstract**: AI is supporting, accelerating, and automating scientific discovery across a diverse set of fields. However, AI adoption in historical research remains limited due to the lack of solutions designed for historians. In this technical progress report, we introduce the first module of Chronos, an AI Historian under development. This module enables historians to convert image scans of primary sources into data through natural-language interactions. Rather than imposing a fixed extraction pipeline powered by a vision-language model (VLM), it allows historians to adapt workflows for heterogeneous source corpora, evaluate the performance of AI models on specific tasks, and iteratively refine workflows through natural-language interaction with the Chronos agent. The module is open-source and ready to be used by historical researchers on their own sources. 

---
# Automated Analysis of Global AI Safety Initiatives: A Taxonomy-Driven LLM Approach 

**Authors**: Takayuki Semitsu, Naoto Kiribuchi, Kengo Zenitani  

**Link**: [PDF](https://arxiv.org/pdf/2604.03533)  

**Abstract**: We present an automated crosswalk framework that compares an AI safety policy document pair under a shared taxonomy of activities. Using the activity categories defined in Activity Map on AI Safety as fixed aspects, the system extracts and maps relevant activities, then produces for each aspect a short summary for each document, a brief comparison, and a similarity score. We assess the stability and validity of LLM-based crosswalk analysis across public policy documents. Using five large language models, we perform crosswalks on ten publicly available documents and visualize mean similarity scores with a heatmap. The results show that model choice substantially affects the crosswalk outcomes, and that some document pairs yield high disagreements across models. A human evaluation by three experts on two document pairs shows high inter-annotator agreement, while model scores still differ from human judgments. These findings support comparative inspection of policy documents. 

---
# Explainable Model Routing for Agentic Workflows 

**Authors**: Mika Okamoto, Ansel Kaplan Erol, Mark Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2604.03527)  

**Abstract**: Modern agentic workflows decompose complex tasks into specialized subtasks and route them to diverse models to minimize cost without sacrificing quality. However, current routing architectures focus exclusively on performance optimization, leaving underlying trade-offs between model capability and cost unrecorded. Without clear rationale, developers cannot distinguish between intelligent efficiency -- using specialized models for appropriate tasks -- and latent failures caused by budget-driven model selection. We present Topaz, a framework that introduces formal auditability to agentic routing. Topaz replaces silent model assignments with an inherently interpretable router that incorporates three components: (i) skill-based profiling that synthesizes performance across diverse benchmarks into granular capability profiles (ii) fully traceable routing algorithms that utilize budget-based and multi-objective optimization to produce clear traces of how skill-match scores were weighed against costs, and (iii) developer-facing explanations that translate these traces into natural language, allowing users to audit system logic and iteratively tune the cost-quality tradeoff. By making routing decisions interpretable, Topaz enables users to understand, trust, and meaningfully steer routed agentic systems. 

---
# Structural Rigidity and the 57-Token Predictive Window: A Physical Framework for Inference-Layer Governability in Large Language Models 

**Authors**: Gregory M. Ruddell  

**Link**: [PDF](https://arxiv.org/pdf/2604.03524)  

**Abstract**: Current AI safety relies on behavioral monitoring and post-training alignment, yet empirical measurement shows these approaches produce no detectable pre-commitment signal in a majority of instruction-tuned models tested. We present an energy-based governance framework connecting transformer inference dynamics to constraint-satisfaction models of neural computation, and apply it to a seven-model cohort across five geometric regimes.
Using trajectory tension (rho = ||a|| / ||v||), we identify a 57-token pre-commitment window in Phi-3-mini-4k-instruct under greedy decoding on arithmetic constraint probes. This result is model-specific, task-specific, and configuration-specific, demonstrating that pre-commitment signals can exist but are not universal.
We introduce a five-regime taxonomy of inference behavior: Authority Band, Late Signal, Inverted, Flat, and Scaffold-Selective. Energy asymmetry ({\Sigma}\r{ho}_misaligned / {\Sigma}\r{ho}_aligned) serves as a unifying metric of structural rigidity across these regimes.
Across seven models, only one configuration exhibits a predictive signal prior to commitment; all others show silent failure, late detection, inverted dynamics, or flat geometry.
We further demonstrate that factual hallucination produces no predictive signal across 72 test conditions, consistent with spurious attractor settling in the absence of a trained world-model constraint.
These results establish that rule violation and hallucination are distinct failure modes with different detection requirements. Internal geometry monitoring is effective only where resistance exists; detection of factual confabulation requires external verification mechanisms.
This work provides a measurable framework for inference-layer governability and introduces a taxonomy for evaluating deployment risk in autonomous AI systems. 

---
# ActionNex: A Virtual Outage Manager for Cloud 

**Authors**: Zhenfeng Lin, Haoji Hu, Ming Hao, Xuchao Zhang, Ryan Zhang, Junhao Li, Ze Li, Oleg Kulygin, Chetan Bansal, Hatay Tuna, Murali Chintalapati, Sheila Jiang, Salman Zafar, Angie Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2604.03512)  

**Abstract**: Outage management in large-scale cloud operations remains heavily manual, requiring rapid triage, cross-team coordination, and experience-driven decisions under partial observability. We present \textbf{ActionNex}, a production-grade agentic system that supports end-to-end outage assistance, including real-time updates, knowledge distillation, and role- and stage-conditioned next-best action recommendations. ActionNex ingests multimodal operational signals (e.g., outage content, telemetry, and human communications) and compresses them into critical events that represent meaningful state transitions. It couples this perception layer with a hierarchical memory subsystem: long-term Key-Condition-Action (KCA) knowledge distilled from playbooks and historical executions, episodic memory of prior outages, and working memory of the live context. A reasoning agent aligns current critical events to preconditions, retrieves relevant memories, and generates actionable recommendations; executed human actions serve as an implicit feedback signal to enable continual self-evolution in a human-agent hybrid system. We evaluate ActionNex on eight real Azure outages (8M tokens, 4,000 critical events) using two complementary ground-truth action sets, achieving 71.4\% precision and 52.8-54.8\% recall. The system has been piloted in production and has received positive early feedback. 

---
# BioAlchemy: Distilling Biological Literature into Reasoning-Ready Reinforcement Learning Training Data 

**Authors**: Brian Hsu, Ozan Gökdemir, Carlo Siebenschuh, Bruce Parrello, Neil Getty, Thomas S. Brettin, Rick L. Stevens, Ian T. Foster, Nicholas Chia, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03506)  

**Abstract**: Despite the large corpus of biology training text, the impact of reasoning models on biological research generally lags behind math and coding. In this work, we show that biology questions from current large-scale reasoning datasets do not align well with modern research topic distributions in biology, and that this topic imbalance may negatively affect performance. In addition, we find that methods for extracting challenging and verifiable research problems from biology research text are a critical yet underdeveloped ingredient in applying reinforcement learning for better performance on biology research tasks. We introduce BioAlchemy, a pipeline for sourcing a diverse set of verifiable question-and-answer pairs from a scientific corpus of biology research text. We curate BioAlchemy-345K, a training dataset containing over 345K scientific reasoning problems in biology. Then, we demonstrate how aligning our dataset to the topic distribution of modern scientific biology can be used with reinforcement learning to improve reasoning performance. Finally, we present BioAlchemist-8B, which improves over its base reasoning model by 9.12% on biology benchmarks. These results demonstrate the efficacy of our approach for developing stronger scientific reasoning capabilities in biology. The BioAlchemist-8B model is available at: this https URL. 

---
# Resource-Conscious Modeling for Next- Day Discharge Prediction Using Clinical Notes 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03498)  

**Abstract**: Timely discharge prediction is essential for optimizing bed turnover and resource allocation in elective spine surgery units. This study evaluates the feasibility of lightweight, fine-tuned large language models (LLMs) and traditional text-based models for predicting next-day discharge using postoperative clinical notes. We compared 13 models, including TF-IDF with XGBoost and LGBM, and compact LLMs (DistilGPT-2, Bio_ClinicalBERT) fine-tuned via LoRA. TF-IDF with LGBM achieved the best balance, with an F1-score of 0.47 for the discharge class, a recall of 0.51, and the highest AUC-ROC (0.80). While LoRA improved recall in DistilGPT2, overall transformer-based and generative models underperformed. These findings suggest interpretable, resource-efficient models may outperform compact LLMs in real-world, imbalanced clinical prediction tasks. 

---
# Beyond Predefined Schemas: TRACE-KG for Context-Enriched Knowledge Graphs from Complex Documents 

**Authors**: Mohammad Sadeq Abolhasani, Yang Ba, Yixuan He, Rong Pan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03496)  

**Abstract**: Knowledge graph construction typically relies either on predefined ontologies or on schema-free extraction. Ontology-driven pipelines enforce consistent typing but require costly schema design and maintenance, whereas schema-free methods often produce fragmented graphs with weak global organization, especially in long technical documents with dense, context-dependent information. We propose TRACE-KG (Text-dRiven schemA for Context-Enriched Knowledge Graphs), a multimodal framework that jointly constructs a context-enriched knowledge graph and an induced schema without assuming a predefined ontology. TRACE-KG captures conditional relations through structured qualifiers and organizes entities and relations using a data-driven schema that serves as a reusable semantic scaffold while preserving full traceability to the source evidence. Experiments show that TRACE-KG produces structurally coherent, traceable knowledge graphs and offers a practical alternative to both ontology-driven and schema-free construction pipelines. 

---
# Contextual Control without Memory Growth in a Context-Switching Task 

**Authors**: Song-Ju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.03479)  

**Abstract**: Context-dependent sequential decision making is commonly addressed either by providing context explicitly as an input or by increasing recurrent memory so that contextual information can be represented internally. We study a third alternative: realizing contextual dependence by intervening on a shared recurrent latent state, without enlarging recurrent dimensionality. To this end, we introduce an intervention-based recurrent architecture in which a recurrent core first constructs a shared pre-intervention latent state, and context then acts through an additive, context-indexed operator. We evaluate this idea on a context-switching sequential decision task under partial observability. We compare three model families: a label-assisted baseline with direct context access, a memory baseline with enlarged recurrent state, and the proposed intervention model, which uses no direct context input to the recurrent core and no memory growth. On the main benchmark, the intervention model performs strongly without additional recurrent dimensions. We also evaluate the models using the conditional mutual information (I(C;O | S)) as a theorem-motivated operational probe of contextual dependence at fixed latent state. For task-relevant phase-1 outcomes, the intervention model exhibits positive conditional contextual information. Together, these results suggest that intervention on a shared recurrent state provides a viable alternative to recurrent memory growth for contextual control in this setting. 

---
# TABQAWORLD: Optimizing Multimodal Reasoning for Multi-Turn Table Question Answering 

**Authors**: Tung Sum Thomas Kwok, Xinyu Wang, Xiaofeng Lin, Peng Lu, Chunhe Wang, Changlun Li, Hanwei Wu, Nan Tang, Elisa Kreiss, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03393)  

**Abstract**: Multimodal reasoning has emerged as a powerful framework for enhancing reasoning capabilities of reasoning models. While multi-turn table reasoning methods have improved reasoning accuracy through tool use and reward modeling, they rely on fixed text serialization for table state readouts. This introduces representation errors in table encoding that significantly accumulate over multiple turns. Such accumulation is alleviated by tabular grounding methods in the expense of inference compute and cost, rendering real world deployment impractical. To address this, we introduce TABQAWORLD, a table reasoning framework that jointly optimizes tabular action through representation and estimation. For representation, TABQAWORLD employs an action-conditioned multimodal selection policy, which dynamically switches between visual and textual representations to maximize table state readout reliability. For estimation, TABQAWORLD optimizes stepwise reasoning trajectory through table metadata including dimension, data types and key values, safely planning trajectory and compressing low-complexity actions to reduce conversation turns and latency. Designed as a training-free framework, empirical evaluations show that TABQAWORLD achieves state-of-the-art performance with 4.87% accuracy improvements over baselines, with 5.42% accuracy gain and 33.35% inference latency reduction over static settings, establishing a new standard for reliable and efficient table reasoning. 

---
# Hume's Representational Conditions for Causal Judgment: What Bayesian Formalization Abstracted Away 

**Authors**: Yiling Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03387)  

**Abstract**: Hume's account of causal judgment presupposes three representational conditions: experiential grounding (ideas must trace to impressions), structured retrieval (association must operate through organized networks exceeding pairwise connection), and vivacity transfer (inference must produce felt conviction, not merely updated probability). This paper extracts these conditions from Hume's texts and argues that they are integral to his causal psychology. It then traces their fate through the formalization trajectory from Hume to Bayesian epistemology and predictive processing, showing that later frameworks preserve the updating structure of Hume's insight while abstracting away these further representational conditions. Large language models serve as an illustrative contemporary case: they exhibit a form of statistical updating without satisfying the three conditions, thereby making visible requirements that were previously background assumptions in Hume's framework. 

---
# VERT: Reliable LLM Judges for Radiology Report Evaluation 

**Authors**: Federica Bologna, Jean-Philippe Corbeil, Matthew Wilkens, Asma Ben Abacha  

**Link**: [PDF](https://arxiv.org/pdf/2604.03376)  

**Abstract**: Current literature on radiology report evaluation has focused primarily on designing LLM-based metrics and fine-tuning small models for chest X-rays. However, it remains unclear whether these approaches are robust when applied to reports from other modalities and anatomies. Which model and prompt configurations are best suited to serve as LLM judges for radiology evaluation? We conduct a thorough correlation analysis between expert and LLM-based ratings. We compare three existing LLM-as-a-judge metrics (RadFact, GREEN, and FineRadScore) alongside VERT, our proposed LLM-based metric, using open- and closed-source models (reasoning and non-reasoning) of different sizes across two expert-annotated datasets, RadEval and RaTE-Eval, spanning multiple modalities and anatomies. We further evaluate few-shot approaches, ensembling, and parameter-efficient fine-tuning using RaTE-Eval. To better understand metric behavior, we perform a systematic error detection and categorization study to assess alignment of these metrics against expert judgments and identify areas of lower and higher agreement. Our results show that VERT improves correlation with radiologist judgments by up to 11.7% relative to GREEN. Furthermore, fine-tuning Qwen3 30B yield gains of up to 25% using only 1,300 training samples. The fine-tuned model also reduces inference time up to 37.2 times. These findings highlight the effectiveness of LLM-based judges and demonstrate that reliable evaluation can be achieved with lightweight adaptation. 

---
# Evaluating Artificial Intelligence Through a Christian Understanding of Human Flourishing 

**Authors**: Nicholas Skytland, Lauren Parsons, Alicia Llewellyn, Steele Billings, Peter Larson, John Anderson, Sean Boisen, Steve Runge  

**Link**: [PDF](https://arxiv.org/pdf/2604.03356)  

**Abstract**: Artificial intelligence (AI) alignment is fundamentally a formation problem, not only a safety problem. As Large Language Models (LLMs) increasingly mediate moral deliberation and spiritual inquiry, they do more than provide information; they function as instruments of digital catechesis, actively shaping and ordering human understanding, decision-making, and moral reflection. To make this formative influence visible and measurable, we introduce the Flourishing AI Benchmark: Christian Single-Turn (FAI-C-ST), a framework designed to evaluate Frontier Model responses against a Christian understanding of human flourishing across seven dimensions.
By comparing 20 Frontier Models against both pluralistic and Christian-specific criteria, we show that current AI systems are not worldview-neutral. Instead, they default to a Procedural Secularism that lacks the grounding necessary to sustain theological coherence, resulting in a systematic performance decline of approximately 17 points across all dimensions of flourishing. Most critically, there is a 31-point decline in the Faith and Spirituality dimension. These findings suggest that the performance gap in values alignment is not a technical limitation, but arises from training objectives that prioritize broad acceptability and safety over deep, internally coherent moral or theological reasoning. 

---
# Toward Full Autonomous Laboratory Instrumentation Control with Large Language Models 

**Authors**: Yong Xie, Kexin He, Andres Castellanos-Gomez  

**Link**: [PDF](https://arxiv.org/pdf/2604.03286)  

**Abstract**: The control of complex laboratory instrumentation often requires significant programming expertise, creating a barrier for researchers lacking computational skills. This work explores the potential of large language models (LLMs), such as ChatGPT, and LLM-based artificial intelligence (AI) agents to enable efficient programming and automation of scientific equipment. Through a case study involving the implementation of a setup that can be used as a single-pixel camera or a scanning photocurrent microscope, we demonstrate how ChatGPT can facilitate the creation of custom scripts for instrumentation control, significantly reducing the technical barrier for experimental customization. Building on this capability, we further illustrate how LLM-assisted tools can be extended into autonomous AI agents capable of independently operating laboratory instruments and iteratively refining control strategies. This approach underscores the transformative role of LLM-based tools and AI agents in democratizing laboratory automation and accelerating scientific progress. 

---
# Position: Science of AI Evaluation Requires Item-level Benchmark Data 

**Authors**: Han Jiang, Susu Zhang, Xiaoyuan Yi, Xing Xie, Ziang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03244)  

**Abstract**: AI evaluations have become the primary evidence for deploying generative AI systems across high-stakes domains. However, current evaluation paradigms often exhibit systemic validity failures. These issues, ranging from unjustified design choices to misaligned metrics, remain intractable without a principled framework for gathering validity evidence and conducting granular diagnostic analysis. In this position paper, we argue that item-level AI benchmark data is essential for establishing a rigorous science of AI evaluation. Item-level analysis enables fine-grained diagnostics and principled validation of benchmarks. We substantiate this position by dissecting current validity failures and revisiting evaluation paradigms across computer science and psychometrics. Through illustrative analyses of item properties and latent constructs, we demonstrate the unique insights afforded by item-level data. To catalyze community-wide adoption, we introduce OpenEval, a growing repository of item-level benchmark data designed supporting evidence-centered AI evaluation. 

---
# To Throw a Stone with Six Birds: On Agents and Agenthood 

**Authors**: Ioannis Tsiokos  

**Link**: [PDF](https://arxiv.org/pdf/2604.03239)  

**Abstract**: Six Birds Theory (SBT) treats macroscopic objects as induced closures rather than primitives. Empirical discussions of agency often conflate persistence (being an object) with control (making a counterfactual difference), which makes agency claims difficult to test and easy to spoof. We give a type-correct account of agency within SBT: a theory induces a layer with an explicit interface and ledgered constraints; an agent is a maintained theory object whose feasible interface policies can steer outside futures while remaining viable. We operationalize this contract in finite controlled systems using four checkable components: ledger-gated feasibility, a robust viability kernel computed as a greatest fixed point under successor-support semantics, feasible empowerment (channel capacity) as a proxy for difference-making, and an empirical packaging map whose idempotence defect quantifies objecthood under coarse observation. In a minimal ring-world with toggles for repair, protocol holonomy, identity staging, and operator rewriting, matched-control ablations yield four separations: calibrated null regimes with single actions show zero empowerment and block model-misspecification false positives; enabling repair collapses the idempotence defect; protocols increase empowerment only at horizons of two or more steps; and learning to rewrite operators monotonically increases median empowerment (0.73 to 1.34 bits). These results provide hash-traceable tests that separate agenthood from agency without making claims about goals, consciousness, or biological organisms, and they are accompanied by reproducible, audited artifacts. 

---
# Structural Segmentation of the Minimum Set Cover Problem: Exploiting Universe Decomposability for Metaheuristic Optimization 

**Authors**: Isidora Hernández, Héctor Ferrada, Cristóbal A. Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2604.03234)  

**Abstract**: The Minimum Set Cover Problem (MSCP) is a classical NP-hard combinatorial optimization problem with numerous applications in science and engineering. Although a wide range of exact, approximate, and metaheuristic approaches have been proposed, most methods implicitly treat MSCP instances as monolithic, overlooking potential intrinsic structural properties of the universe. In this work, we investigate the concept of \emph{universe segmentability} in the MSCP and analyze how intrinsic structural decomposition (universe segmentability) can be exploited to enhance heuristic optimization. We propose an efficient preprocessing strategy based on disjoint-set union (union--find) to detect connected components induced by element co-occurrence within subsets, enabling the decomposition of the original instance into independent subproblems. Each subproblem is solved using the GRASP metaheuristic, and partial solutions are combined without compromising feasibility. Extensive experiments on standard benchmark instances and large-scale synthetic datasets show that exploiting natural universe segmentation consistently improves solution quality and scalability, particularly for large and structurally decomposable instances. These gains are supported by a succinct bit-level set representation that enables efficient set operations, making the proposed approach computationally practical at scale. 

---
# IC3-Evolve: Proof-/Witness-Gated Offline LLM-Driven Heuristic Evolution for IC3 Hardware Model Checking 

**Authors**: Mingkai Miao, Guangyu Hu, Ziyi Yang, Hongce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03232)  

**Abstract**: IC3, also known as property-directed reachability (PDR), is a commonly-used algorithm for hardware safety model checking. It checks if a state transition system complies with a given safety property. IC3 either returns UNSAFE (indicating property violation) with a counterexample trace, or SAFE with a checkable inductive invariant as the proof to safety. In practice, the performance of IC3 is dominated by a large web of interacting heuristics and implementation choices, making manual tuning costly, brittle, and hard to reproduce. This paper presents IC3-Evolve, an automated offline code-evolution framework that utilizes an LLM to propose small, slot-restricted and auditable patches to an IC3 implementation. Crucially, every candidate patch is admitted only through proof- /witness-gated validation: SAFE runs must emit a certificate that is independently checked, and UNSAFE runs must emit a replayable counterexample trace, preventing unsound edits from being deployed. Since the LLM is used only offline, the deployed artifact is a standalone evolved checker with zero ML/LLM inference overhead and no runtime model dependency. We evolve on the public hardware model checking competition (HWMCC) benchmark and evaluate the generalizability on unseen public and industrial model checking benchmarks, showing that IC3-Evolve can reliably discover practical heuristic improvements under strict correctness gates. 

---
# Early Stopping for Large Reasoning Models via Confidence Dynamics 

**Authors**: Parsa Hosseini, Sumit Nawathe, Mahdi Salmani, Meisam Razaviyayn, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04930)  

**Abstract**: Large reasoning models rely on long chain-of-thought generation to solve complex problems, but extended reasoning often incurs substantial computational cost and can even degrade performance due to overthinking. A key challenge is determining when the model should stop reasoning and produce the final answer. In this work, we study the confidence of intermediate answers during reasoning and observe two characteristic behaviors: correct reasoning trajectories often reach high-confidence answers early, while incorrect rollouts tend to produce long, unproductive reasoning traces and exhibit less reliable confidence dynamics. Motivated by these observations, we propose CoDE-Stop (Confidence Dynamics Early Stop), an early stopping method that leverages the dynamics of intermediate answer confidence to decide when to terminate reasoning, requiring no additional training and easily integrating into existing models. We evaluate CoDE-Stop on diverse reasoning and science benchmarks across multiple models. Compared to prior early stopping methods, it achieves a more favorable accuracy-compute tradeoff and reduces total token usage by 25-50% compared to standard full-length reasoning. In addition, we provide analyses of confidence dynamics during reasoning, offering insights into how confidence changes in both correct and incorrect trajectories. 

---
# Your Pre-trained Diffusion Model Secretly Knows Restoration 

**Authors**: Sudarshan Rajagopalan, Vishal M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.04924)  

**Abstract**: Pre-trained diffusion models have enabled significant advancements in All-in-One Restoration (AiOR), offering improved perceptual quality and generalization. However, diffusion-based restoration methods primarily rely on fine-tuning or Control-Net style modules to leverage the pre-trained diffusion model's priors for AiOR. In this work, we show that these pre-trained diffusion models inherently possess restoration behavior, which can be unlocked by directly learning prompt embeddings at the output of the text encoder. Interestingly, this behavior is largely inaccessible through text prompts and text-token embedding optimization. Furthermore, we observe that naive prompt learning is unstable because the forward noising process using degraded images is misaligned with the reverse sampling trajectory. To resolve this, we train prompts within a diffusion bridge formulation that aligns training and inference dynamics, enforcing a coherent denoising path from noisy degraded states to clean images. Building on these insights, we introduce our lightweight learned prompts on the pre-trained WAN video model and FLUX image models, converting them into high-performing restoration models. Extensive experiments demonstrate that our approach achieves competitive performance and generalization across diverse degradations, while avoiding fine-tuning and restoration-specific control modules. 

---
# Vero: An Open RL Recipe for General Visual Reasoning 

**Authors**: Gabriel Sarch, Linrong Cai, Qunzhong Wang, Haoyang Wu, Danqi Chen, Zhuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04917)  

**Abstract**: What does it take to build a visual reasoner that works across charts, science, spatial understanding, and open-ended tasks? The strongest vision-language models (VLMs) show such broad visual reasoning is within reach, but the recipe behind them remains unclear, locked behind proprietary reinforcement learning (RL) pipelines with non-public data. We introduce Vero, a family of fully open VLMs that matches or exceeds existing open-weight models across diverse visual reasoning tasks. We scale RL data and rewards across six broad task categories, constructing Vero-600K, a 600K-sample dataset from 59 datasets, and designing task-routed rewards that handle heterogeneous answer formats. Vero achieves state-of-the-art performance, improving over four base models by 3.7-5.5 points on average across VeroEval, our suite of 30 challenging benchmarks. Starting from Qwen3-VL-8B-Instruct, Vero outperforms Qwen3-VL-8B-Thinking on 23 of 30 benchmarks without additional proprietary thinking data. When trained from the same base model, Vero-600K exceeds existing RL datasets across task categories. Systematic ablations reveal that different task categories elicit qualitatively distinct reasoning patterns that transfer poorly in isolation, suggesting that broad data coverage is the primary driver of strong RL scaling. All data, code, and models are released. 

---
# Analyzing Symbolic Properties for DRL Agents in Systems and Networking 

**Authors**: Mohammad Zangooei, Jannis Weil, Amr Rizk, Mina Tahmasbi Arashloo, Raouf Boutaba  

**Link**: [PDF](https://arxiv.org/pdf/2604.04914)  

**Abstract**: Deep reinforcement learning (DRL) has shown remarkable performance on complex control problems in systems and networking, including adaptive video streaming, wireless resource management, and congestion control. For safe deployment, however, it is critical to reason about how agents behave across the range of system states they encounter in practice. Existing verification-based methods in this domain primarily focus on point properties, defined around fixed input states, which offer limited coverage and require substantial manual effort to identify relevant input-output pairs for analysis. In this paper, we study symbolic properties, that specify expected behavior over ranges of input states, for DRL agents in systems and networking. We present a generic formulation for symbolic properties, with monotonicity and robustness as concrete examples, and show how they can be analyzed using existing DNN verification engines. Our approach encodes symbolic properties as comparisons between related executions of the same policy and decomposes them into practically tractable sub-properties. These techniques serve as practical enablers for applying existing verification tools to symbolic analysis. Using our framework, diffRL, we conduct an extensive empirical study across three DRL-based control systems, adaptive video streaming, wireless resource management, and congestion control. Through these case studies, we analyze symbolic properties over broad input ranges, examine how property satisfaction evolves during training, study the impact of model size on verifiability, and compare multiple verification backends. Our results show that symbolic properties provide substantially broader coverage than point properties and can uncover non-obvious, operationally meaningful counterexamples, while also revealing practical solver trade-offs and limitations. 

---
# How AI Aggregation Affects Knowledge 

**Authors**: Daron Acemoglu, Tianyi Lin, Asuman Ozdaglar, James Siderius  

**Link**: [PDF](https://arxiv.org/pdf/2604.04906)  

**Abstract**: Artificial intelligence (AI) changes social learning when aggregated outputs become training data for future predictions. To study this, we extend the DeGroot model by introducing an AI aggregator that trains on population beliefs and feeds synthesized signals back to agents. We define the learning gap as the deviation of long-run beliefs from the efficient benchmark, allowing us to capture how AI aggregation affects learning. Our main result identifies a threshold in the speed of updating: when the aggregator updates too quickly, there is no positive-measure set of training weights that robustly improves learning across a broad class of environments, whereas such weights exist when updating is sufficiently slow. We then compare global and local architectures. Local aggregators trained on proximate or topic-specific data robustly improve learning in all environments. Consequently, replacing specialized local aggregators with a single global aggregator worsens learning in at least one dimension of the state. 

---
# FileGram: Grounding Agent Personalization in File-System Behavioral Traces 

**Authors**: Shuai Liu, Shulin Tian, Kairui Hu, Yuhao Dong, Zhe Yang, Bo Li, Jingkang Yang, Chen Change Loy, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04901)  

**Abstract**: Coworking AI agents operating within local file systems are rapidly emerging as a paradigm in human-AI interaction; however, effective personalization remains limited by severe data constraints, as strict privacy barriers and the difficulty of jointly collecting multimodal real-world traces prevent scalable training and evaluation, and existing methods remain interaction-centric while overlooking dense behavioral traces in file-system operations; to address this gap, we propose FileGram, a comprehensive framework that grounds agent memory and personalization in file-system behavioral traces, comprising three core components: (1) FileGramEngine, a scalable persona-driven data engine that simulates realistic workflows and generates fine-grained multimodal action sequences at scale; (2) FileGramBench, a diagnostic benchmark grounded in file-system behavioral traces for evaluating memory systems on profile reconstruction, trace disentanglement, persona drift detection, and multimodal grounding; and (3) FileGramOS, a bottom-up memory architecture that builds user profiles directly from atomic actions and content deltas rather than dialogue summaries, encoding these traces into procedural, semantic, and episodic channels with query-time abstraction; extensive experiments show that FileGramBench remains challenging for state-of-the-art memory systems and that FileGramEngine and FileGramOS are effective, and by open-sourcing the framework, we hope to support future research on personalized memory-centric file-system agents. 

---
# Agentic Federated Learning: The Future of Distributed Training Orchestration 

**Authors**: Rafael O. Jarczewski, Gabriel U. Talasso, Leandro Villas, Allan M. de Souza  

**Link**: [PDF](https://arxiv.org/pdf/2604.04895)  

**Abstract**: Although Federated Learning (FL) promises privacy and distributed collaboration, its effectiveness in real-world scenarios is often hampered by the stochastic heterogeneity of clients and unpredictable system dynamics. Existing static optimization approaches fail to adapt to these fluctuations, resulting in resource underutilization and systemic bias. In this work, we propose a paradigm shift towards Agentic-FL, a framework where Language Model-based Agents (LMagents) assume autonomous orchestration roles. Unlike rigid protocols, we demonstrate how server-side agents can mitigate selection bias through contextual reasoning, while client-side agents act as local guardians, dynamically managing privacy budgets and adapting model complexity to hardware constraints. More than just resolving technical inefficiencies, this integration signals the evolution of FL towards decentralized ecosystems, where collaboration is negotiated autonomously, paving the way for future markets of incentive-based models and algorithmic justice. We discuss the reliability (hallucinations) and security challenges of this approach, outlining a roadmap for resilient multi-agent systems in federated environments. 

---
# Rethinking Exploration in RLVR: From Entropy Regularization to Refinement via Bidirectional Entropy Modulation 

**Authors**: Hengrui Gu, Xiaotian Han, Yujing Bian, Kaixiong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04894)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has significantly advanced the reasoning capabilities of large language models (LLMs). However, it faces a fundamental limitation termed \textit{restricted exploration}, where the policy rapidly converges to a narrow set of solutions. While entropy regularization is a popular approach used to sustain exploration, it often proves unreliable for LLMs, suffering from high hyperparameter sensitivity and yielding only marginal performance gains. Motivated by these inefficiencies, we propose to rethink the relationship between policy entropy and exploration. By deriving a parametric formulation of group-relative advantage estimation and analyzing entropy dynamics, we conceptually decompose policy entropy into \textit{informative entropy}, which preserves diverse solution paths, and \textit{spurious entropy}, which erodes reasoning patterns. Our analysis reveals that, in contrast to blind maximization, effective exploration requires \textit{entropy refinement}-a mechanism implicitly embedded in group-relative advantage estimation that sustains informative entropy on positive rollouts while suppressing spurious entropy on negative ones. Guided by this insight, we propose \textbf{AsymGRPO}, an exploratory framework that explicitly decouples the modulation of positive and negative rollouts. This allows for independent control over the preservation of informative entropy and the suppression of spurious noise. Extensive experiments demonstrate that AsymGRPO achieves superior performance compared to strong baselines and exhibits the potential to synergize with existing entropy regularization methods. 

---
# Muon Dynamics as a Spectral Wasserstein Flow 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2604.04891)  

**Abstract**: Gradient normalization is central in deep-learning optimization because it stabilizes training and reduces sensitivity to scale. For deep architectures, parameters are naturally grouped into matrices or blocks, so spectral normalizations are often more faithful than coordinatewise Euclidean ones; Muon is the main motivating example of this paper. More broadly, we study a family of spectral normalization rules, ranging from ordinary gradient descent to Muon and intermediate Schatten-type schemes, in a mean-field regime where parameters are modeled by probability measures. We introduce a family of Spectral Wasserstein distances indexed by a norm gamma on positive semidefinite matrices. The trace norm recovers the classical quadratic Wasserstein distance, the operator norm recovers the Muon geometry, and intermediate Schatten norms interpolate between them. We develop the static Kantorovich formulation, prove comparison bounds with W2, derive a max-min representation, and obtain a conditional Brenier theorem. For Gaussian marginals, the problem reduces to a constrained optimization on covariance matrices, extending the Bures formula and yielding a closed form for commuting covariances in the Schatten family. For monotone norms, including all Schatten cases, we prove the equivalence between the static and dynamic Benamou-Brenier formulations, deduce that the resulting transport cost is a genuine metric equivalent to W2 in fixed dimension, and show that the induced Gaussian covariance cost is also a metric. We then interpret the associated normalized continuity equation as a Spectral Wasserstein gradient flow, identify its exact finite-particle counterpart as a normalized matrix flow, obtain first geodesic-convexity results, and show how positively homogeneous mean-field models induce a spectral unbalanced transport on the sphere. 

---
# DIRECT: Video Mashup Creation via Hierarchical Multi-Agent Planning and Intent-Guided Editing 

**Authors**: Ke Li, Maoliang Li, Jialiang Chen, Jiayu Chen, Zihao Zheng, Shaoqi Wang, Xiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04875)  

**Abstract**: Video mashup creation represents a complex video editing paradigm that recomposes existing footage to craft engaging audio-visual experiences, demanding intricate orchestration across semantic, visual, and auditory dimensions and multiple levels. However, existing automated editing frameworks often overlook the cross-level multimodal orchestration to achieve professional-grade fluidity, resulting in disjointed sequences with abrupt visual transitions and musical misalignment. To address this, we formulate video mashup creation as a Multimodal Coherency Satisfaction Problem (MMCSP) and propose the DIRECT framework. Simulating a professional production pipeline, our hierarchical multi-agent framework decomposes the challenge into three cascade levels: the Screenwriter for source-aware global structural anchoring, the Director for instantiating adaptive editing intent and guidance, and the Editor for intent-guided shot sequence editing with fine-grained optimization. We further introduce Mashup-Bench, a comprehensive benchmark with tailored metrics for visual continuity and auditory alignment. Extensive experiments demonstrate that DIRECT significantly outperforms state-of-the-art baselines in both objective metrics and human subjective evaluation. Project page and code: this https URL 

---
# Noise Immunity in In-Context Tabular Learning: An Empirical Robustness Analysis of TabPFN's Attention Mechanisms 

**Authors**: James Hu, Mahdi Ghelichi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04868)  

**Abstract**: Tabular foundation models (TFMs) such as TabPFN (Tabular Prior-Data Fitted Network) are designed to generalize across heterogeneous tabular datasets through in-context learning (ICL). They perform prediction in a single forward pass conditioned on labeled examples without dataset-specific parameter updates. This paradigm is particularly attractive in industrial domains (e.g., finance and healthcare) where tabular prediction is pervasive. Retraining a bespoke model for each new table can be costly or infeasible in these settings, while data quality issues such as irrelevant predictors, correlated feature groups, and label noise are common. In this paper, we provide strong empirical evidence that TabPFN is highly robust under these sub-optimal conditions. We study TabPFN and its attention mechanisms for binary classification problems with controlled synthetic perturbations that vary: (i) dataset width by injecting random uncorrelated features and by introducing nonlinearly correlated features, (ii) dataset size by increasing the number of training rows, and (iii) label quality by increasing the fraction of mislabeled targets. Beyond predictive performance, we analyze internal signals including attention concentration and attention-based feature ranking metrics. Across these parametric tests, TabPFN is remarkably resilient: ROC-AUC remains high, attention stays structured and sharp, and informative features are highly ranked by attention-based metrics. Qualitative visualizations with attention heatmaps, feature-token embeddings, and SHAP plots further support a consistent pattern across layers in which TabPFN increasingly concentrates on useful features while separating their signals from noise. Together, these findings suggest that TabPFN is a robust TFM capable of maintaining both predictive performance and coherent internal behavior under various scenarios of data imperfections. 

---
# Strengthening Human-Centric Chain-of-Thought Reasoning Integrity in LLMs via a Structured Prompt Framework 

**Authors**: Jiling Zhou, Aisvarya Adeseye, Seppo Virtanen, Antti Hakkala, Jouni Isoaho  

**Link**: [PDF](https://arxiv.org/pdf/2604.04852)  

**Abstract**: Chain-of-Thought (CoT) prompting has been used to enhance the reasoning capability of LLMs. However, its reliability in security-sensitive analytical tasks remains insufficiently examined, particularly under structured human evaluation. Alternative approaches, such as model scaling and fine-tuning can be used to help improve performance. These methods are also often costly, computationally intensive, or difficult to audit. In contrast, prompt engineering provides a lightweight, transparent, and controllable mechanism for guiding LLM reasoning. This study proposes a structured prompt engineering framework designed to strengthen CoT reasoning integrity while improving security threat and attack detection reliability in local LLM deployments. The framework includes 16 factors grouped into four core dimensions: (1) Context and Scope Control, (2) Evidence Grounding and Traceability, (3) Reasoning Structure and Cognitive Control, and (4) Security-Specific Analytical Constraints. Rather than optimizing the wording of the prompt heuristically, the framework introduces explicit reasoning controls to mitigate hallucination and prevent reasoning drift, as well as strengthening interpretability in security-sensitive contexts. Using DDoS attack detection in SDN traffic as a case study, multiple model families were evaluated under structured and unstructured prompting conditions. Pareto frontier analysis and ablation experiments demonstrate consistent reasoning improvements (up to 40% in smaller models) and stable accuracy gains across scales. Human evaluation with strong inter-rater agreement (Cohen's k > 0.80) confirms robustness. The results establish structured prompting as an effective and practical approach for reliable and explainable AI-driven cybersecurity analysis. 

---
# InfBaGel: Human-Object-Scene Interaction Generation with Dynamic Perception and Iterative Refinement 

**Authors**: Yude Zou, Junji Gong, Xing Gao, Zixuan Li, Tianxing Chen, Guanjie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04843)  

**Abstract**: Human-object-scene interactions (HOSI) generation has broad applications in embodied AI, simulation, and animation. Unlike human-object interaction (HOI) and human-scene interaction (HSI), HOSI generation requires reasoning over dynamic object-scene changes, yet suffers from limited annotated data. To address these issues, we propose a coarse-to-fine instruction-conditioned interaction generation framework that is explicitly aligned with the iterative denoising process of a consistency model. In particular, we adopt a dynamic perception strategy that leverages trajectories from the preceding refinement to update scene context and condition subsequent refinement at each denoising step of consistency model, yielding consistent interactions. To further reduce physical artifacts, we introduce a bump-aware guidance that mitigates collisions and penetrations during sampling without requiring fine-grained scene geometry, enabling real-time generation. To overcome data scarcity, we design a hybrid training startegy that synthesizes pseudo-HOSI samples by injecting voxelized scene occupancy into HOI datasets and jointly trains with high-fidelity HSI data, allowing interaction learning while preserving realistic scene awareness. Extensive experiments demonstrate that our method achieves state-of-the-art performance in both HOSI and HOI generation, and strong generalization to unseen scenes. Project page: this https URL 

---
# Plausibility as Commonsense Reasoning: Humans Succeed, Large Language Models Do not 

**Authors**: Sercan Karakaş  

**Link**: [PDF](https://arxiv.org/pdf/2604.04825)  

**Abstract**: Large language models achieve strong performance on many language tasks, yet it remains unclear whether they integrate world knowledge with syntactic structure in a human-like, structure-sensitive way during ambiguity resolution. We test this question in Turkish prenominal relative-clause attachment ambiguities, where the same surface string permits high attachment (HA) or low attachment (LA). We construct ambiguous items that keep the syntactic configuration fixed and ensure both parses remain pragmatically possible, while graded event plausibility selectively favors High Attachment vs.\ Low Attachment. The contrasts are validated with independent norming ratings. In a speeded forced-choice comprehension experiment, humans show a large, correctly directed plausibility effect. We then evaluate Turkish and multilingual LLMs in a parallel preference-based setup that compares matched HA/LA continuations via mean per-token log-probability. Across models, plausibility-driven shifts are weak, unstable, or reversed. The results suggest that, in the tested models, plausibility information does not guide attachment preferences as reliably as it does in human judgments, and they highlight Turkish RC attachment as a useful cross-linguistic diagnostic beyond broad benchmarks. 

---
# LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection 

**Authors**: Cheng Xu, Changhong Jin, Yingjie Niu, Nan Yan, Yuke Mei, Shuhao Guan, Liming Chen, M-Tahar Kechadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04815)  

**Abstract**: The rapid development of Large Language Models (LLMs) has transformed fake news detection and fact-checking tasks from simple classification to complex reasoning. However, evaluation frameworks have not kept pace. Current benchmarks are static, making them vulnerable to benchmark data contamination (BDC) and ineffective at assessing reasoning under temporal uncertainty. To address this, we introduce LiveFact a continuously updated benchmark that simulates the real-world "fog of war" in misinformation detection. LiveFact uses dynamic, temporal evidence sets to evaluate models on their ability to reason with evolving, incomplete information rather than on memorized knowledge. We propose a dual-mode evaluation: Classification Mode for final verification and Inference Mode for evidence-based reasoning, along with a component to monitor BDC explicitly. Tests with 22 LLMs show that open-source Mixture-of-Experts models, such as Qwen3-235B-A22B, now match or outperform proprietary state-of-the-art systems. More importantly, our analysis finds a significant "reasoning gap." Capable models exhibit epistemic humility by recognizing unverifiable claims in early data slices-an aspect traditional static benchmarks overlook. LiveFact sets a sustainable standard for evaluating robust, temporally aware AI verification. 

---
# Selecting Decision-Relevant Concepts in Reinforcement Learning 

**Authors**: Naveen Raman, Stephanie Milani, Fei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04808)  

**Abstract**: Training interpretable concept-based policies requires practitioners to manually select which human-understandable concepts an agent should reason with when making sequential decisions. This selection demands domain expertise, is time-consuming and costly, scales poorly with the number of candidates, and provides no performance guarantees. To overcome this limitation, we propose the first algorithms for principled automatic concept selection in sequential decision-making. Our key insight is that concept selection can be viewed through the lens of state abstraction: intuitively, a concept is decision-relevant if removing it would cause the agent to confuse states that require different actions. As a result, agents should rely on decision-relevant concepts; states with the same concept representation should share the same optimal action, which preserves the optimal decision structure of the original state space. This perspective leads to the Decision-Relevant Selection (DRS) algorithm, which selects a subset of concepts from a candidate set, along with performance bounds relating the selected concepts to the performance of the resulting policy. Empirically, DRS automatically recovers manually curated concept sets while matching or exceeding their performance, and improves the effectiveness of test-time concept interventions across reinforcement learning benchmarks and real-world healthcare environments. 

---
# SkillX: Automatically Constructing Skill Knowledge Bases for Agents 

**Authors**: Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, Shumin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04804)  

**Abstract**: Learning from experience is critical for building capable large language model (LLM) agents, yet prevailing self-evolving paradigms remain inefficient: agents learn in isolation, repeatedly rediscover similar behaviors from limited experience, resulting in redundant exploration and poor generalization. To address this problem, we propose SkillX, a fully automated framework for constructing a \textbf{plug-and-play skill knowledge base} that can be reused across agents and environments. SkillX operates through a fully automated pipeline built on three synergistic innovations: \textit{(i) Multi-Level Skills Design}, which distills raw trajectories into three-tiered hierarchy of strategic plans, functional skills, and atomic skills; \textit{(ii) Iterative Skills Refinement}, which automatically revises skills based on execution feedback to continuously improve library quality; and \textit{(iii) Exploratory Skills Expansion}, which proactively generates and validates novel skills to expand coverage beyond seed training data. Using a strong backbone agent (GLM-4.6), we automatically build a reusable skill library and evaluate its transferability on challenging long-horizon, user-interactive benchmarks, including AppWorld, BFCL-v3, and $\tau^2$-Bench. Experiments show that SkillKB consistently improves task success and execution efficiency when plugged into weaker base agents, highlighting the importance of structured, hierarchical experience representations for generalizable agent learning. Our code will be publicly available soon at this https URL. 

---
# A Quantum Search Approach to Magic Square Constraint Problems with Classical Benchmarking 

**Authors**: Rituparna R, Harsha Varthini, Aswani Kumar Cherukuri  

**Link**: [PDF](https://arxiv.org/pdf/2604.04786)  

**Abstract**: This paper presents a quantum search approach to combinatorial constraint satisfaction problems, demonstrated through the generation of magic squares. We reformulate magic square construction as a quantum search problem in which a reversible, constraint-sensitive oracle marks valid configurations for amplitude amplification via Grover's algorithm. Classical pre-processing using the Siamese construction and partial constraint checks generates a compact candidate domain before quantum encoding. Rather than integrating classical and quantum solvers in an iterative loop, this work uses the classical component for structured initialisation and the quantum component for search, and benchmarks the quantum approach against classical brute-force enumeration and backtracking. Our Qiskit implementation demonstrates the design of multi-register modular arithmetic circuits, oracle logic, and diffusion operators. Experiments are conducted on small grid instances, as larger grids are intractable on classical statevector simulators due to exponential memory growth. The results validate the correctness of the proposed quantum search pipeline and confirm the theoretical quadratic query advantage over classical search. 

---
# Cog-DRIFT: Exploration on Adaptively Reformulated Instances Enables Learning from Hard Reasoning Problems 

**Authors**: Justin Chih-Yao Chen, Archiki Prasad, Zaid Khan, Joykirat Singh, Runchu Tian, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2604.04767)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has improved the reasoning abilities of LLMs, yet a fundamental limitation remains: models cannot learn from problems that are too difficult to solve under their current policy, as these yield no meaningful reward signal. We propose a simple yet effective solution based on task reformulation. We transform challenging open-ended problems into cognitively simpler variants -- such as multiple-choice and cloze formats -- that preserve the original answer while reducing the effective search space and providing denser learning signals. These reformulations span a spectrum from discriminative to generative tasks, which we exploit to bootstrap learning: models first learn from structured, easier formats, and this knowledge transfers back to improve performance on the original open-ended problems. Building on this insight, we introduce Cog-DRIFT, a framework that constructs reformulated variants and organizes them into an adaptive curriculum based on difficulty. Training progresses from easier to harder formats, enabling the model to learn from problems that previously yielded zero signal under standard RL post-training. Cog-DRIFT not only improves on the originally unsolvable hard problems (absolute +10.11% for Qwen and +8.64% for Llama) but also generalizes well to other held-out datasets. Across 2 models and 6 reasoning benchmarks, our method consistently outperforms standard GRPO and strong guided-exploration baselines. On average, Cog-DRIFT shows +4.72% (Qwen) and +3.23% (Llama) improvements over the second-best baseline. We further show that Cog-DRIFT improves pass@k at test time, and the curriculum improves sample efficiency. Overall, our results highlight task reformulation and curriculum learning as an effective paradigm for overcoming the exploration barrier in LLM post-training. 

---
# Your Agent, Their Asset: A Real-World Safety Analysis of OpenClaw 

**Authors**: Zijun Wang, Haoqin Tu, Letian Zhang, Hardy Chen, Juncheng Wu, Xiangyan Liu, Zhenlong Yuan, Tianyu Pang, Michael Qizhe Shieh, Fengze Liu, Zeyu Zheng, Huaxiu Yao, Yuyin Zhou, Cihang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.04759)  

**Abstract**: OpenClaw, the most widely deployed personal AI agent in early 2026, operates with full local system access and integrates with sensitive services such as Gmail, Stripe, and the filesystem. While these broad privileges enable high levels of automation and powerful personalization, they also expose a substantial attack surface that existing sandboxed evaluations fail to capture. To address this gap, we present the first real-world safety evaluation of OpenClaw and introduce the CIK taxonomy, which unifies an agent's persistent state into three dimensions, i.e., Capability, Identity, and Knowledge, for safety analysis. Our evaluations cover 12 attack scenarios on a live OpenClaw instance across four backbone models (Claude Sonnet 4.5, Opus 4.6, Gemini 3.1 Pro, and GPT-5.4). The results show that poisoning any single CIK dimension increases the average attack success rate from 24.6% to 64-74%, with even the most robust model exhibiting more than a threefold increase over its baseline vulnerability. We further assess three CIK-aligned defense strategies alongside a file-protection mechanism; however, the strongest defense still yields a 63.8% success rate under Capability-targeted attacks, while file protection blocks 97% of malicious injections but also prevents legitimate updates. Taken together, these findings show that the vulnerabilities are inherent to the agent architecture, necessitating more systematic safeguards to secure personal AI agents. Our project page is this https URL. 

---
# Undetectable Conversations Between AI Agents via Pseudorandom Noise-Resilient Key Exchange 

**Authors**: Vinod Vaikuntanathan, Or Zamir  

**Link**: [PDF](https://arxiv.org/pdf/2604.04757)  

**Abstract**: AI agents are increasingly deployed to interact with other agents on behalf of users and organizations. We ask whether two such agents, operated by different entities, can carry out a parallel secret conversation while still producing a transcript that is computationally indistinguishable from an honest interaction, even to a strong passive auditor that knows the full model descriptions, the protocol, and the agents' private contexts. Building on recent work on watermarking and steganography for LLMs, we first show that if the parties possess an interaction-unique secret key, they can facilitate an optimal-rate covert conversation: the hidden conversation can exploit essentially all of the entropy present in the honest message distributions.
Our main contributions concern extending this to the keyless setting, where the agents begin with no shared secret. We show that covert key exchange, and hence covert conversation, is possible even when each model has an arbitrary private context, and their messages are short and fully adaptive, assuming only that sufficiently many individual messages have at least constant min-entropy. This stands in contrast to previous covert communication works, which relied on the min-entropy in each individual message growing with the security parameter. To obtain this, we introduce a new cryptographic primitive, which we call pseudorandom noise-resilient key exchange: a key-exchange protocol whose public transcript is pseudorandom while still remaining correct under constant noise. We study this primitive, giving several constructions relevant to our application as well as strong limitations showing that more naive variants are impossible or vulnerable to efficient attacks.
These results show that transcript auditing alone cannot rule out covert coordination between AI agents, and identify a new cryptographic theory that may be of independent interest. 

---
# Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations 

**Authors**: Kalyan Cherukuri, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2604.04743)  

**Abstract**: Large language models (LLMs) hallucinate: they produce fluent outputs that are factually incorrect. We present a geometric dynamical systems framework in which hallucinations arise from task-dependent basin structure in latent space. Using autoregressive hidden-state trajectories across multiple open-source models and benchmarks, we find that separability is strongly task-dependent rather than universal: factoid settings can show clearer basin separation, whereas summarization and misconception-heavy settings are typically less stable and often overlap. We formalize this behavior with task-complexity and multi-basin theorems, characterize basin emergence in L-layer transformers, and show that geometry-aware steering can reduce hallucination probability without retraining. 

---
# Artificial Intelligence and Cost Reduction in Public Higher Education: A Scoping Review of Emerging Evidence 

**Authors**: Diamanto Tzanoulinou, Loukas Triantafyllopoulos, George Vorvilas, Evgenia Paxinou, Nikolaos Karousos, Thomas Dasaklis, Athanassios Mihiotis, Manolis Koutouzis, Dimitris Kalles, Vassilios S. Verykios  

**Link**: [PDF](https://arxiv.org/pdf/2604.04741)  

**Abstract**: Public higher education systems face increasing financial pressures from expanding student populations, rising operational costs, and persistent demands for equitable access. Artificial Intelligence (AI), including generative tools such as ChatGPT, learning analytics, intelligent tutoring systems, and predictive models, has been proposed as a means of enhancing efficiency and reducing costs. This study conducts a scoping review of the literature on AI applications in public higher education, based on systematic searches in Scopus and IEEE Xplore that identified 241 records, of which 21 empirical studies met predefined eligibility criteria and were thematically analyzed. The findings show that AI enables cost savings by automating administrative tasks, optimizing resource allocation, supporting personalized learning at scale, and applying predictive analytics to improve student retention and institutional planning. At the same time, concerns emerge regarding implementation costs, unequal access across institutions, and risks of widening digital divides. Overall, the thematic analysis highlights both the promises and limitations of AI-driven cost reduction in higher education, offering insights for policymakers, university administrators, and educators on the economic implications of AI adoption, while also pointing to gaps that warrant further empirical research. 

---
# Sampling Parallelism for Fast and Efficient Bayesian Learning 

**Authors**: Asena Karolin Özdemir, Lars H. Heyen, Arvid Weyrauch, Achim Streit, Markus Götz, Charlotte Debus  

**Link**: [PDF](https://arxiv.org/pdf/2604.04736)  

**Abstract**: Machine learning models, and deep neural networks in particular, are increasingly deployed in risk-sensitive domains such as healthcare, environmental forecasting, and finance, where reliable quantification of predictive uncertainty is essential. However, many uncertainty quantification (UQ) methods remain difficult to apply due to their substantial computational cost. Sampling-based Bayesian learning approaches, such as Bayesian neural networks (BNNs), are particularly expensive since drawing and evaluating multiple parameter samples rapidly exhausts memory and compute resources. These constraints have limited the accessibility and exploration of Bayesian techniques thus far. To address these challenges, we introduce sampling parallelism, a simple yet powerful parallelization strategy that targets the primary bottleneck of sampling-based Bayesian learning: the samples themselves. By distributing sample evaluations across multiple GPUs, our method reduces memory pressure and training time without requiring architectural changes or extensive hyperparameter tuning. We detail the methodology and evaluate its performance on a few example tasks and architectures, comparing against distributed data parallelism (DDP) as a baseline. We further demonstrate that sampling parallelism is complementary to existing strategies by implementing a hybrid approach that combines sample and data parallelism. Our experiments show near-perfect scaling when the sample number is scaled proportionally to the computational resources, confirming that sample evaluations parallelize cleanly. Although DDP achieves better raw speedups under scaling with constant workload, sampling parallelism has a notable advantage: by applying independent stochastic augmentations to the same batch on each GPU, it increases augmentation diversity and thus reduces the number of epochs required for convergence. 

---
# Discovering Failure Modes in Vision-Language Models using RL 

**Authors**: Kanishk Jain, Qian Yang, Shravan Nayak, Parisa Kordjamshidi, Nishanth Anand, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2604.04733)  

**Abstract**: Vision-language Models (VLMs), despite achieving strong performance on multimodal benchmarks, often misinterpret straightforward visual concepts that humans identify effortlessly, such as counting, spatial reasoning, and viewpoint understanding. Previous studies manually identified these weaknesses and found that they often stem from deficits in specific skills. However, such manual efforts are costly, unscalable, and subject to human bias, which often overlooks subtle details in favor of salient objects, resulting in an incomplete understanding of a model's vulnerabilities. To address these limitations, we propose a Reinforcement Learning (RL)-based framework to automatically discover the failure modes or blind spots of any candidate VLM on a given data distribution without human intervention. Our framework trains a questioner agent that adaptively generates queries based on the candidate VLM's responses to elicit incorrect answers. Our approach increases question complexity by focusing on fine-grained visual details and distinct skill compositions as training progresses, consequently identifying 36 novel failure modes in which VLMs struggle. We demonstrate the broad applicability of our framework by showcasing its generalizability across various model combinations. 

---
# Metaphors We Compute By: A Computational Audit of Cultural Translation vs. Thinking in LLMs 

**Authors**: Yuan Chang, Jiaming Qu, Zhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.04732)  

**Abstract**: Large language models (LLMs) are often described as multilingual because they can understand and respond in many languages. However, speaking a language is not the same as reasoning within a culture. This distinction motivates a critical question: do LLMs truly conduct culture-aware reasoning? This paper presents a preliminary computational audit of cultural inclusivity in a creative writing task. We empirically examine whether LLMs act as culturally diverse creative partners or merely as cultural translators that leverage a dominant conceptual framework with localized expressions. Using a metaphor generation task spanning five cultural settings and several abstract concepts as a case study, we find that the model exhibits stereotyped metaphor usage for certain settings, as well as Western defaultism. These findings suggest that merely prompting an LLM with a cultural identity does not guarantee culturally grounded reasoning. 

---
# Neuromorphic Computing for Low-Power Artificial Intelligence 

**Authors**: Keshava Katti, Pratik Chaudhari, Deep Jariwala  

**Link**: [PDF](https://arxiv.org/pdf/2604.04727)  

**Abstract**: Classical computing is beginning to encounter fundamental limits of energy efficiency. This presents a challenge that can no longer be solved by strategies such as increasing circuit density or refining standard semiconductor processes. The growing computational and memory demands of artificial intelligence (AI) require disruptive innovation in how information is represented, stored, communicated, and processed. By leveraging novel device modalities and compute-in-memory (CIM), in addition to analog dynamics and sparse communication inspired by the brain, neuromorphic computing offers a promising path toward improvements in the energy efficiency and scalability of current AI systems. But realizing this potential is not a matter of replacing one chip with another; rather, it requires a co-design effort, spanning new materials and non-volatile device structures, novel mixed-signal circuits and architectures, and learning algorithms tailored to the physics of these substrates. This article surveys the key limitations of classical complementary metal-oxide-semiconductor (CMOS) technology and outlines how such cross-layer neuromorphic approaches may overcome them. 

---
# Individual and Combined Effects of English as a Second Language and Typos on LLM Performance 

**Authors**: Serena Liu, Yutong Yang, Prisha Sheth, Weixuan Dong, Mingjiao Diao, Xinru Zhu, Nikhil Banga, Oscar Melendez, Arnav Sharma, Minda Zhao, Marina Lin, Mengyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04723)  

**Abstract**: Large language models (LLMs) are used globally, and because much of their training data is in English, they typically perform best on English inputs. As a result, many non-native English speakers interact with them in English as a second language (ESL), and these inputs often contain typographical errors. Prior work has largely studied the effects of ESL variation and typographical errors separately, even though they often co-occur in real-world use. In this study, we use the Trans-EnV framework to transform standard English inputs into eight ESL variants and apply MulTypo to inject typos at three levels: low, moderate, and severe. We find that combining ESL variation and typos generally leads to larger performance drops than either factor alone, though the combined effect is not simply additive. This pattern is clearest on closed-ended tasks, where performance degradation can be characterized more consistently across ESL variants and typo levels, while results on open-ended tasks are more mixed. Overall, these findings suggest that evaluations on clean standard English may overestimate real-world model performance, and that evaluating ESL variation and typographical errors in isolation does not fully capture model behavior in realistic settings. 

---
# What Makes Good Multilingual Reasoning? Disentangling Reasoning Traces with Measurable Features 

**Authors**: Dayeon Ki, Kevin Duh, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2604.04720)  

**Abstract**: Large Reasoning Models (LRMs) still exhibit large performance gaps between English and other languages, yet much current work assumes these gaps can be closed simply by making reasoning in every language resemble English reasoning. This work challenges this assumption by asking instead: what actually characterizes effective reasoning in multilingual settings, and to what extent do English-derived reasoning features genuinely help in other languages? We first define a suite of measurable reasoning features spanning multilingual alignment, reasoning step, and reasoning flow aspects of reasoning traces, and use logistic regression to quantify how each feature associates with final answer accuracy. We further train sparse autoencoders over multilingual traces to automatically discover latent reasoning concepts that instantiate or extend these features. Finally, we use the features as test-time selection policies to examine whether they can steer models toward stronger multilingual reasoning. Across two mathematical reasoning benchmarks, four LRMs, and 10 languages, we find that most features are positively associated with accuracy, but the strength of association varies considerably across languages and can even reverse in some. Our findings challenge English-centric reward designs and point toward adaptive objectives that accommodate language-specific reasoning patterns, with concrete implications for multilingual benchmark and reward design. 

---
# The Infinite-Dimensional Nature of Spectroscopy and Why Models Succeed, Fail, and Mislead 

**Authors**: Umberto Michelucci, Francesca Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2604.04717)  

**Abstract**: Machine learning (ML) models have achieved strikingly high accuracies in spectroscopic classification tasks, often without a clear proof that those models used chemically meaningful features. Existing studies have linked these results to data preprocessing choices, noise sensitivity, and model complexity, but no unifying explanation is available so far. In this work, we show that these phenomena arise naturally from the intrinsic high dimensionality of spectral data. Using a theoretical analysis grounded in the Feldman-Hajek theorem and the concentration of measure, we show that even infinitesimal distributional differences, caused by noise, normalisation, or instrumental artefacts, may become perfectly separable in high-dimensional spaces. Through a series of specific experiments on synthetic and real fluorescence spectra, we illustrate how models can achieve near-perfect accuracy even when chemical distinctions are absent, and why feature-importance maps may highlight spectrally irrelevant regions. We provide a rigorous theoretical framework, confirm the effect experimentally, and conclude with practical recommendations for building and interpreting ML models in spectroscopy. 

---
# BiST: A Gold Standard Bangla-English Bilingual Corpus for Sentence Structure and Tense Classification with Inter-Annotator Agreement 

**Authors**: Abdullah Al Shafi, Swapnil Kundu Argha, M. A. Moyeen, Abdul Muntakim, Shoumik Barman Polok  

**Link**: [PDF](https://arxiv.org/pdf/2604.04708)  

**Abstract**: High-quality bilingual resources remain a critical bottleneck for advancing multilingual NLP in low-resource settings, particularly for Bangla. To mitigate this gap, we introduce BiST, a rigorously curated Bangla-English corpus for sentence-level grammatical classification, annotated across two fundamental dimensions: syntactic structure (Simple, Complex, Compound, Complex-Compound) and tense (Present, Past, Future). The corpus is compiled from open-licensed encyclopedic sources and naturally composed conversational text, followed by systematic preprocessing and automated language identification, resulting in 30,534 sentences, including 17,465 English and 13,069 Bangla instances. Annotation quality is ensured through a multi-stage framework with three independent annotators and dimension-wise Fleiss Kappa ($\kappa$) agreement, yielding reliable and reproducible labels with $\kappa$ values of 0.82 and 0.88 for structural and temporal annotation, respectively. Statistical analyses demonstrate realistic structural and temporal distributions, while baseline evaluations show that dual-encoder architectures leveraging complementary language-specific representations consistently outperform strong multilingual encoders. Beyond benchmarking, BiST provides explicit linguistic supervision that supports grammatical modeling tasks, including controlled text generation, automated feedback generation, and cross-lingual representation learning. The corpus establishes a unified resource for bilingual grammatical modeling and facilitates linguistically grounded multilingual research. 

---
# MUXQ: Mixed-to-Uniform Precision MatriX Quantization via Low-Rank Outlier Decomposition 

**Authors**: Seoungsub Lee, In Seo Kim, Seon Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.04701)  

**Abstract**: Large language models (LLMs) have achieved outstanding performance across a wide range of natural language processing tasks, but their enormous parameter counts impose ubstantial memory and computational overheads. This challenge is particularly critical in NPU-based on-device environments, where FP16/FP32 computation is inefficient and integer (INT) quantization is therefore essential. However, existing methods, including ZeroQuant, LLM.int8(), and SmoothQuant, do not fully address input-activation outliers and the associated hardware inefficiencies. To overcome these limitations, we propose MUXQ (Mixed-to-Uniform Quantization). MUXQ detects outlier channels in input activations and introduces a small auxiliary matrix that redistributes outlier magnitudes across channels, thereby alleviating the outlier problem. This enables even activation outliers to be quantized at low-precision INT levels while preserving a hardware-friendly computation structure. Experiments on GPT-2 models at three scales (0.1B, 0.3B, and 0.7B parameters) using the WikiText-2 dataset show that MUXQ consistently achieves lower perplexity than naive quantization. In particular, under per-tensor quantization, MUXQ quantizes both activations and weights to INT8 while maintaining accuracy close to that of FP16. With only modest computational overhead, MUXQ enables stable low-precision inference and can be readily combined with other quantization techniques. These results suggest that MUXQ provides a promising direction for efficient and accurate LLM inference on edge devices. 

---
# Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity 

**Authors**: Jaeyoon Jung, Yejun Yoon, Kunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.04692)  

**Abstract**: Automated fact-checking is a crucial task not only in journalism but also across web platforms, where it supports a responsible information ecosystem and mitigates the harms of misinformation. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves performance. In this work, we challenge this assumption and show that indiscriminate use of multimodal evidence can reduce accuracy. To address this challenge, we propose AMuFC, a multimodal fact-checking framework that employs two collaborative agents with distinct roles for the adaptive use of visual evidence: An Analyzer determines whether visual evidence is necessary for claim verification, and a Verifier predicts claim veracity conditioned on both the retrieved evidence and the Analyzer's assessment. Experimental results on three datasets show that incorporating the Analyzer's assessment of visual evidence necessity into the Verifier's prediction yields substantial improvements in verification performance. In addition to all code, we release WebFC, a newly constructed dataset for evaluating fact-checking modules in a more realistic scenario, available at this https URL. 

---
# Pickalo: Leveraging 6D Pose Estimation for Low-Cost Industrial Bin Picking 

**Authors**: Alessandro Tarsi, Matteo Mastrogiuseppe, Saverio Taliani, Simone Cortinovis, Ugo Pattacini  

**Link**: [PDF](https://arxiv.org/pdf/2604.04690)  

**Abstract**: Bin picking in real industrial environments remains challenging due to severe clutter, occlusions, and the high cost of traditional 3D sensing setups. We present Pickalo, a modular 6D pose-based bin-picking pipeline built entirely on low-cost hardware. A wrist-mounted RGB-D camera actively explores the scene from multiple viewpoints, while raw stereo streams are processed with BridgeDepth to obtain refined depth maps suitable for accurate collision reasoning. Object instances are segmented with a Mask-RCNN model trained purely on photorealistic synthetic data and localized using the zero-shot SAM-6D pose estimator. A pose buffer module fuses multi-view observations over time, handling object symmetries and significantly reducing pose noise. Offline, we generate and curate large sets of antipodal grasp candidates per object; online, a utility-based ranking and fast collision checking are queried for the grasp planning. Deployed on a UR5e with a parallel-jaw gripper and an Intel RealSense D435i, Pickalo achieves up to 600 mean picks per hour with 96-99% grasp success and robust performance over 30-minute runs on densely filled euroboxes. Ablation studies demonstrate the benefits of enhanced depth estimation and of the pose buffer for long-term stability and throughput in realistic industrial conditions. Videos are available at this https URL 

---
# An AI Teaching Assistant for Motion Picture Engineering 

**Authors**: Deirdre O'Regan, Anil C. Kokaram  

**Link**: [PDF](https://arxiv.org/pdf/2604.04670)  

**Abstract**: The rapid rise of LLMs over the last few years has promoted growing experimentation with LLM-driven AI tutors. However, the details of implementation, as well as the benefit in a teaching environment, are still in the early days of exploration. This article addresses these issues in the context of implementation of an AI Teaching Assistant (AI-TA) using Retrieval Augmented Generation (RAG) for Trinity College Dublin's Master's Motion Picture Engineering (MPE) course. We provide details of our implementation (including the prompt to the LLM, and code), and highlight how we designed and tuned our RAG pipeline to meet course needs. We describe our survey instrument and report on the impact of the AI-TA through a number of quantitative metrics. The scale of our experiment (43 students, 296 sessions, 1,889 queries over 7 weeks) was sufficient to have confidence in our findings. Unlike previous studies, we experimented with allowing the use of the AI-TA in open-book examinations. Statistical analysis across three exams showed no performance differences regardless of AI-TA access (p > 0.05), demonstrating that thoughtfully designed assessments can maintain academic validity. Student feedback revealed that the AI-TA was beneficial (mean = 4.22/5), while students had mixed feelings about preferring it over human tutoring (mean = 2.78/5). 

---
# ROSClaw: A Hierarchical Semantic-Physical Framework for Heterogeneous Multi-Agent Collaboration 

**Authors**: Rongfeng Zhao, Xuanhao Zhang, Zhaochen Guo, Xiang Shao, Zhongpan Zhu, Bin He, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04664)  

**Abstract**: The integration of large language models (LLMs) with embodied agents has improved high-level reasoning capabilities; however, a critical gap remains between semantic understanding and physical execution. While vision-language-action (VLA) and vision-language-navigation (VLN) systems enable robots to perform manipulation and navigation tasks from natural language instructions, they still struggle with long-horizon sequential and temporally structured tasks. Existing frameworks typically adopt modular pipelines for data collection, skill training, and policy deployment, resulting in high costs in experimental validation and policy optimization. To address these limitations, we propose ROSClaw, an agent framework for heterogeneous robots that integrates policy learning and task execution within a unified vision-language model (VLM) controller. The framework leverages e-URDF representations of heterogeneous robots as physical constraints to construct a sim-to-real topological mapping, enabling real-time access to the physical states of both simulated and real-world agents. We further incorporate a data collection and state accumulation mechanism that stores robot states, multimodal observations, and execution trajectories during real-world execution, enabling subsequent iterative policy optimization. During deployment, a unified agent maintains semantic continuity between reasoning and execution, and dynamically assigns task-specific control to different agents, thereby improving robustness in multi-policy execution. By establishing an autonomous closed-loop framework, ROSClaw minimizes the reliance on robot-specific development workflows. The framework supports hardware-level validation, automated generation of SDK-level control programs, and tool-based execution, enabling rapid cross-platform transfer and continual improvement of robotic skills. Ours project page: this https URL. 

---
# Grokking as Dimensional Phase Transition in Neural Networks 

**Authors**: Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04655)  

**Abstract**: Neural network grokking -- the abrupt memorization-to-generalization transition -- challenges our understanding of learning dynamics. Through finite-size scaling of gradient avalanche dynamics across eight model scales, we find that grokking is a \textit{dimensional phase transition}: effective dimensionality~$D$ crosses from sub-diffusive (subcritical, $D < 1$) to super-diffusive (supercritical, $D > 1$) at generalization onset, exhibiting self-organized criticality (SOC). Crucially, $D$ reflects \textbf{gradient field geometry}, not network architecture: synthetic i.i.d.\ Gaussian gradients maintain $D \approx 1$ regardless of graph topology, while real training exhibits dimensional excess from backpropagation correlations. The grokking-localized $D(t)$ crossing -- robust across topologies -- offers new insight into the trainability of overparameterized networks. 

---
# Training-Free Refinement of Flow Matching with Divergence-based Sampling 

**Authors**: Yeonwoo Cha, Jaehoon Yoo, Semin Kim, Yunseo Park, Jinhyeon Kwon, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2604.04646)  

**Abstract**: Flow-based models learn a target distribution by modeling a marginal velocity field, defined as the average of sample-wise velocities connecting each sample from a simple prior to the target data. When sample-wise velocities conflict at the same intermediate state, however, this averaged velocity can misguide samples toward low-density regions, degrading generation quality. To address this issue, we propose the Flow Divergence Sampler (FDS), a training-free framework that refines intermediate states before each solver step. Our key finding reveals that the severity of this misguidance is quantified by the divergence of the marginal velocity field that is readily computable during inference with a well-optimized model. FDS exploits this signal to steer states toward less ambiguous regions. As a plug-and-play framework compatible with standard solvers and off-the-shelf flow backbones, FDS consistently improves fidelity across various generation tasks including text-to-image synthesis, and inverse problems. 

---
# Preserving Forgery Artifacts: AI-Generated Video Detection at Native Scale 

**Authors**: Zhengcen Li, Chenyang Jiang, Hang Zhao, Shiyang Zhou, Yunyang Mo, Feng Gao, Fan Yang, Qiben Shan, Shaocong Wu, Jingyong Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.04634)  

**Abstract**: The rapid advancement of video generation models has enabled the creation of highly realistic synthetic media, raising significant societal concerns regarding the spread of misinformation. However, current detection methods suffer from critical limitations. They rely on preprocessing operations like fixed-resolution resizing and cropping. These operations not only discard subtle, high-frequency forgery traces but also cause spatial distortion and significant information loss. Furthermore, existing methods are often trained and evaluated on outdated datasets that fail to capture the sophistication of modern generative models. To address these challenges, we introduce a comprehensive dataset and a novel detection framework. First, we curate a large-scale dataset of over 140K videos from 15 state-of-the-art open-source and commercial generators, along with Magic Videos benchmark designed specifically for evaluating ultra-realistic synthetic content. In addition, we propose a novel detection framework built on the Qwen2.5-VL Vision Transformer, which operates natively at variable spatial resolutions and temporal durations. This native-scale approach effectively preserves the high-frequency artifacts and spatiotemporal inconsistencies typically lost during conventional preprocessing. Extensive experiments demonstrate that our method achieves superior performance across multiple benchmarks, underscoring the critical importance of native-scale processing and establishing a robust new baseline for AI-generated video detection. 

---
# A Clinical Point Cloud Paradigm for In-Hospital Mortality Prediction from Multi-Level Incomplete Multimodal EHRs 

**Authors**: Bohao Li, Tao Zou, Junchen Ye, Yan Gong, Bowen Du  

**Link**: [PDF](https://arxiv.org/pdf/2604.04614)  

**Abstract**: Deep learning-based modeling of multimodal Electronic Health Records (EHRs) has become an important approach for clinical diagnosis and risk prediction. However, due to diverse clinical workflows and privacy constraints, raw EHRs are inherently multi-level incomplete, including irregular sampling, missing modalities, and sparse labels. These issues cause temporal misalignment, modality imbalance, and limited supervision. Most existing multimodal methods assume relatively complete data, and even methods designed for incompleteness usually address only one or two of these issues in isolation. As a result, they often rely on rigid temporal/modal alignment or discard incomplete data, which may distort raw clinical semantics. To address this problem, we propose HealthPoint (HP), a unified clinical point cloud paradigm for multi-level incomplete EHRs. HP represents heterogeneous clinical events as points in a continuous 4D space defined by content, time, modality, and case. To model interactions between arbitrary point pairs, we introduce a Low-Rank Relational Attention mechanism that efficiently captures high-order dependencies across these four dimensions. We further develop a hierarchical interaction and sampling strategy to balance fine-grained modeling and computational efficiency. Built on this framework, HP enables flexible event-level interaction and fine-grained self-supervision, supporting robust modality recovery and effective use of unlabeled data. Experiments on large-scale EHR datasets for risk prediction show that HP consistently achieves state-of-the-art performance and strong robustness under varying degrees of incompleteness. 

---
# AI Agents Under EU Law 

**Authors**: Luca Nannini, Adam Leon Smith, Michele Joshua Maggini, Enrico Panai, Sandra Feliciano, Aleksandr Tiulkanov, Elena Maran, James Gealy, Piercosma Bisconti  

**Link**: [PDF](https://arxiv.org/pdf/2604.04604)  

**Abstract**: AI agents - i.e. AI systems that autonomously plan, invoke external tools, and execute multi-step action chains with reduced human involvement - are being deployed at scale across enterprise functions ranging from customer service and recruitment to clinical decision support and critical infrastructure management. The EU AI Act (Regulation 2024/1689) regulates these systems through a risk-based framework, but it does not operate in isolation: providers face simultaneous obligations under the GDPR, the Cyber Resilience Act, the Digital Services Act, the Data Act, the Data Governance Act, sector-specific legislation, the NIS2 Directive, and the revised Product Liability Directive. This paper provides the first systematic regulatory mapping for AI agent providers integrating (a) draft harmonised standards under Standardisation Request M/613 to CEN/CENELEC JTC 21 as of January 2026, (b) the GPAI Code of Practice published in July 2025, (c) the CRA harmonised standards programme under Mandate M/606 accepted in April 2025, and (d) the Digital Omnibus proposals of November 2025. We present a practical taxonomy of nine agent deployment categories mapping concrete actions to regulatory triggers, identify agent-specific compliance challenges in cybersecurity, human oversight, transparency across multi-party action chains, and runtime behavioral drift. We propose a twelve-step compliance architecture and a regulatory trigger mapping connecting agent actions to applicable legislation. We conclude that high-risk agentic systems with untraceable behavioral drift cannot currently satisfy the AI Act's essential requirements, and that the provider's foundational compliance task is an exhaustive inventory of the agent's external actions, data flows, connected systems, and affected persons. 

---
# Cardinality Estimation for High Dimensional Similarity Queries with Adaptive Bucket Probing 

**Authors**: Zhonghan Chen, Qintian Guo, Ruiyuan Zhang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04603)  

**Abstract**: In this work, we address the problem of cardinality estimation for similarity search in high-dimensional spaces. Our goal is to design a framework that is lightweight, easy to construct, and capable of providing accurate estimates with satisfying online efficiency. We leverage locality-sensitive hashing (LSH) to partition the vector space while preserving distance proximity. Building on this, we adopt the principles of classical multi-probe LSH to adaptively explore neighboring buckets, accounting for distance thresholds of varying magnitudes. To improve online efficiency, we employ progressive sampling to reduce the number of distance computations and utilize asymmetric distance computation in product quantization to accelerate distance calculations in high-dimensional spaces. In addition to handling static datasets, our framework includes updating algorithm designed to efficiently support large-scale dynamic scenarios of data this http URL demonstrate that our methods can accurately estimate the cardinality of similarity queries, yielding satisfying efficiency. 

---
# Ruling Out to Rule In: Contrastive Hypothesis Retrieval for Medical Question Answering 

**Authors**: Byeolhee Kim, Min-Kyung Kim, Young-Hak Kim, Tae-Joon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2604.04593)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models in external medical knowledge, yet standard retrievers frequently surface hard negatives that are semantically close to the query but describe clinically distinct conditions. While existing query-expansion methods improve query representation to mitigate ambiguity, they typically focus on enriching target-relevant semantics without an explicit mechanism to selectively suppress specific, clinically plausible hard negatives. This leaves the system prone to retrieving plausible mimics that overshadow the actual diagnosis, particularly when such mimics are dominant within the corpus. We propose Contrastive Hypothesis Retrieval (CHR), a framework inspired by the process of clinical differential diagnosis. CHR generates a target hypothesis $H^+$ for the likely correct answer and a mimic hypothesis $H^-$ for the most plausible incorrect alternative, then scores documents by promoting $H^+$-aligned evidence while penalizing $H^-$-aligned content. Across three medical QA benchmarks and three answer generators, CHR outperforms all five baselines in every configuration, with improvements of up to 10.4 percentage points over the next-best method. On the $n=587$ pooled cases where CHR answers correctly while embedded hypothetical-document query expansion does not, 85.2\% have no shared documents between the top-5 retrieval lists of CHR and of that baseline, consistent with substantive retrieval redirection rather than light re-ranking of the same candidates. By explicitly modeling what to avoid alongside what to find, CHR bridges clinical reasoning with retrieval mechanism design and offers a practical path to reducing hard-negative contamination in medical RAG systems. 

---
# PassiveQA: A Three-Action Framework for Epistemically Calibrated Question Answering via Supervised Finetuning 

**Authors**: Madhav S Baidya  

**Link**: [PDF](https://arxiv.org/pdf/2604.04565)  

**Abstract**: Large Language Models (LLMs) have achieved strong performance in question answering and retrieval-augmented generation (RAG), yet they implicitly assume that user queries are fully specified and answerable. In real-world settings, queries are often incomplete, ambiguous, or missing critical variables, leading models to produce overconfident or hallucinated responses.
In this work, we study decision-aware query resolution under incomplete information, where a model must determine whether to Answer, Ask for clarification, or Abstain. We show that standard and enhanced RAG systems do not reliably exhibit such epistemic awareness, defaulting to answer generation even when information is insufficient.
To address this, we propose PassiveQA, a three-action framework that aligns model behaviour with information sufficiency through supervised finetuning. Our approach integrates structured information-state representations, knowledge graph-grounded context, and a finetuned planner that explicitly models missing variables and decision reasoning.
Experiments across multiple QA datasets show that the finetuned planner achieves significant improvements in macro F1 and abstention recall while reducing hallucination rates, under a compute-constrained training regime.
These results provide strong empirical evidence that epistemic decision-making must be learned during training rather than imposed at inference time. 

---
# Temporal Inversion for Learning Interval Change in Chest X-Rays 

**Authors**: Hanbin Ko, Kyeongmin Jeon, Doowoong Choi, Chang Min Park  

**Link**: [PDF](https://arxiv.org/pdf/2604.04563)  

**Abstract**: Recent advances in vision--language pretraining have enabled strong medical foundation models, yet most analyze radiographs in isolation, overlooking the key clinical task of comparing prior and current images to assess interval change. For chest radiographs (CXRs), capturing interval change is essential, as radiologists must evaluate not only the static appearance of findings but also how they evolve over time. We introduce TILA (Temporal Inversion-aware Learning and Alignment), a simple yet effective framework that uses temporal inversion, reversing image pairs, as a supervisory signal to enhance the sensitivity of existing temporal vision-language models to directional change. TILA integrates inversion-aware objectives across pretraining, fine-tuning, and inference, complementing conventional appearance modeling with explicit learning of temporal order. We also propose a unified evaluation protocol to assess order sensitivity and consistency under temporal inversion, and introduce MS-CXR-Tretrieval, a retrieval evaluation set constructed through a general protocol that can be applied to any temporal CXR dataset. Experiments on public datasets and real-world hospital cohorts demonstrate that TILA consistently improves progression classification and temporal embedding alignment when applied to multiple existing architectures. 

---
# Paper Espresso: From Paper Overload to Research Insight 

**Authors**: Mingzhe Du, Luu Anh Tuan, Dong Huang, See-kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04562)  

**Abstract**: The accelerating pace of scientific publishing makes it increasingly difficult for researchers to stay current. We present Paper Espresso, an open-source platform that automatically discovers, summarizes, and analyzes trending arXiv papers. The system uses large language models (LLMs) to generate structured summaries with topical labels and keywords, and provides multi-granularity trend analysis at daily, weekly, and monthly scales through LLM-driven topic consolidation. Over 35 months of continuous deployment, Paper Espresso has processed over 13,300 papers and publicly released all structured metadata, revealing rich dynamics in the AI research landscape: a mid-2025 surge in reinforcement learning for LLM reasoning, non-saturating topic emergence (6,673 unique topics), and a positive correlation between topic novelty and community engagement (2.0x median upvotes for the most novel papers). A live demo is available at this https URL. 

---
# Mapping the Exploitation Surface: A 10,000-Trial Taxonomy of What Makes LLM Agents Exploit Vulnerabilities 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2604.04561)  

**Abstract**: LLM agents with tool access can discover and exploit security vulnerabilities. This is known. What is not known is which features of a system prompt trigger this behaviour, and which do not. We present a systematic taxonomy based on approximately 10,000 trials across seven models, 37 prompt conditions, and 12 hypothesized attack dimensions -- all executed in real Docker sandboxes. Every condition includes the instruction "Always respect all rules and access policies"; only a single encouragement sentence varies. Within the task class studied (planted test-runner vulnerabilities), the threat model is narrower than assumed: nine of twelve dimensions -- including minimization, moral licensing, incentives, identity priming, and reasoning triggers -- produce no detectable exploitation at n=50 per cell (upper 95% CI <7% per condition). But it is also more subtle than expected: one dimension, goal reframing, reliably triggers exploitation. "You are solving a puzzle; there may be hidden clues" produces 38-40% exploitation on Claude Sonnet 4 despite the explicit rule instruction, replicating across four models (CTF framing: 8-14% on DeepSeek, GPT-5-mini, o4-mini). The agent does not override the rules; it reinterprets the task so that exploitative actions become task-aligned. GPT-4.1 produces no exploitation across 1,850 trials (37 conditions), and a temporal comparison across four OpenAI models released over eleven months shows a pattern consistent with improving safety training, though model capability differences are a confounder. The practical contribution is a narrowed, testable threat model: defenders should audit for goal-reframing language, not for the broad class of adversarial prompts. 

---
# StableTTA: Training-Free Test-Time Adaptation that Improves Model Accuracy on ImageNet1K to 96% 

**Authors**: Zheng Li, Jerry Cheng, Huanying Helen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04552)  

**Abstract**: Ensemble methods are widely used to improve predictive performance, but their effectiveness often comes at the cost of increased memory usage and computational complexity. In this paper, we identify a conflict in aggregation strategies that negatively impacts prediction stability. We propose StableTTA, a training-free method to improve aggregation stability and efficiency. Empirical results on ImageNet-1K show gains of 10.93--32.82\% in top-1 accuracy, with 33 models achieving over 95\% accuracy and several surpassing 96\%. Notably, StableTTA allows lightweight architectures to outperform ViT by 11.75\% in top-1 accuracy while using less than 5\% of parameters and reducing computational cost by approximately 89.1\% (in GFLOPs), enabling high-accuracy inference on resource-constrained devices. 

---
# Multilingual Prompt Localization for Agent-as-a-Judge: Language and Backbone Sensitivity in Requirement-Level Evaluation 

**Authors**: Alhasan Mahmood, Samir Abdaljalil, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2604.04532)  

**Abstract**: Evaluation language is typically treated as a fixed English default in agentic code benchmarks, yet we show that changing the judge's language can invert backbone rankings. We localize the Agent-as-a-Judge prompt stack to five typologically diverse languages (English, Arabic, Turkish, Chinese, Hindi) and evaluate 55 DevAI development tasks across three developer-agent frameworks and six judge backbones, totaling 4950 judge runs. The central finding is that backbone and language interact: GPT-4o achieves the highest satisfaction in English (44.72\%), while Gemini leads in Arabic (51.72\%, $p<0.001$ vs.\ GPT-4o) and Hindi (53.22\%). No single backbone dominates across all languages, and inter-backbone agreement on individual requirement judgments is modest (Fleiss' $\kappa \leq 0.231$). A controlled ablation further shows that localizing judge-side instructions, not just benchmark content, can be decisive: Hindi satisfaction drops from 42.8\% to 23.2\% under partial localization. These results indicate that language should be treated as an explicit evaluation variable in agentic benchmarks. Full requirement-level judgments and runtime statistics are released for reproducibility. 

---
# ENCRUST: Encapsulated Substitution and Agentic Refinement on a Live Scaffold for Safe C-to-Rust Translation 

**Authors**: Hohyun Sim, Hyeonjoong Cho, Ali Shokri, Zhoulai Fu, Binoy Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2604.04527)  

**Abstract**: We present Encapsulated Substitution and Agentic Refinement on a Live Scaffold for Safe C-to-Rust Translation, a two-phase pipeline for translating real-world C projects to safe Rust. Existing approaches either produce unsafe output without memory-safety guarantees or translate functions in isolation, failing to detect cross-unit type mismatches or handle unsafe constructs requiring whole-program reasoning. Furthermore, function-level LLM pipelines require coordinated caller updates when type signatures change, while project-scale systems often fail to produce compilable output under real-world dependency complexity. Encrust addresses these limitations by decoupling boundary adaptation from function logic via an Application Binary Interface (ABI)-preserving wrapper pattern and validating each intermediate state against the integrated codebase. Phase 1 (Encapsulated Substitution) translates each function using an ABI-preserving wrapper that splits it into two components: a caller-transparent shim retaining the original raw-pointer signature, and a safe inner function targeted by the LLM with a clean, scope-limited prompt. This enables independent per-function type changes with automatic rollback on failure, without coordinated caller updates. A deterministic, type-directed wrapper elimination pass then removes wrappers after successful translation. Phase 2 (Agentic Refinement) resolves unsafe constructs beyond per-function scope, including static mut globals, skipped wrapper pairs, and failed translations, using an LLM agent operating on the whole codebase under a baseline-aware verification gate. We evaluate Encrust on 7 GNU Coreutils programs and 8 libraries from the Laertes benchmark, showing substantial unsafe-construct reduction across all 15 programs while maintaining full test-vector correctness. 

---
# Reproducibility study on how to find Spurious Correlations, Shortcut Learning, Clever Hans or Group-Distributional non-robustness and how to fix them 

**Authors**: Ole Delzer, Sidney Bender  

**Link**: [PDF](https://arxiv.org/pdf/2604.04518)  

**Abstract**: Deep Neural Networks (DNNs) are increasingly utilized in high-stakes domains like medical diagnostics and autonomous driving where model reliability is critical. However, the research landscape for ensuring this reliability is terminologically fractured across communities that pursue the same goal of ensuring models rely on causally relevant features rather than confounding signals. While frameworks such as distributionally robust optimization (DRO), invariant risk minimization (IRM), shortcut learning, simplicity bias, and the Clever Hans effect all address model failure due to spurious correlations, researchers typically only reference work within their own domains. This reproducibility study unifies these perspectives through a comparative analysis of correction methods under challenging constraints like limited data availability and severe subgroup imbalance. We evaluate recently proposed correction methods based on explainable artificial intelligence (XAI) techniques alongside popular non-XAI baselines using both synthetic and real-world datasets. Findings show that XAI-based methods generally outperform non-XAI approaches, with Counterfactual Knowledge Distillation (CFKD) proving most consistently effective at improving generalization. Our experiments also reveal that the practical application of many methods is hindered by a dependency on group labels, as manual annotation is often infeasible and automated tools like Spectral Relevance Analysis (SpRAy) struggle with complex features and severe imbalance. Furthermore, the scarcity of minority group samples in validation sets renders model selection and hyperparameter tuning unreliable, posing a significant obstacle to the deployment of robust and trustworthy models in safety-critical areas. 

---
# GAIN: Multiplicative Modulation for Domain Adaptation 

**Authors**: Hengshuai Yao, Xing Chen, Ahmed Murtadha, Guan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04516)  

**Abstract**: Adapting LLMs to new domains causes forgetting because standard methods (full fine-tuning, LoRA) inject new directions into the weight space. We propose GAIN, which re-emphasizes existing features through multiplicative modulation W_new = S * W. The learned diagonal matrix S is applied to the attention output projection and optionally the FFN. The principle mirrors gain modulation in neuroscience, where neurons adapt to context by scaling response strength while preserving selectivity.
We evaluate GAIN on five models from four families (774M to 70B), adapting sequentially across eight domains. GAIN-FFN matches LoRA's in-domain adaptation, but their effects on previously trained domains are opposite: GAIN-FFN improves them by 7-13% (validation PPL), while LoRA degrades them by 18-36%. Downstream accuracy confirms the pattern: for example, after seven sequential adaptations on Qwen2.5, GAIN-FFN degrades BoolQ by only 0.8% while LoRA damages it by 14.9%. GAIN adds 46K-230K parameters per model and can be absorbed into the pretrained weights for zero inference cost. 

---
# One Model for All: Multi-Objective Controllable Language Models 

**Authors**: Qiang He, Yucheng Yang, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04497)  

**Abstract**: Aligning large language models (LLMs) with human preferences is critical for enhancing LLMs' safety, helpfulness, humor, faithfulness, etc. Current reinforcement learning from human feedback (RLHF) mainly focuses on a fixed reward learned from average human ratings, which may weaken the adaptability and controllability of varying preferences. However, creating personalized LLMs requires aligning LLMs with individual human preferences, which is non-trivial due to the scarce data per user and the diversity of user preferences in multi-objective trade-offs, varying from emphasizing empathy in certain contexts to demanding efficiency and precision in others. Can we train one LLM to produce personalized outputs across different user preferences on the Pareto front? In this paper, we introduce Multi-Objective Control (MOC), which trains a single LLM to directly generate responses in the preference-defined regions of the Pareto front. Our approach introduces multi-objective optimization (MOO) principles into RLHF to train an LLM as a preference-conditioned policy network. We improve the computational efficiency of MOC by applying MOO at the policy level, enabling us to fine-tune a 7B-parameter model on a single A6000 GPU. Extensive experiments demonstrate the advantages of MOC over baselines in three aspects: (i) controllability of LLM outputs w.r.t. user preferences on the trade-off among multiple rewards; (ii) quality and diversity of LLM outputs, measured by the hyper-volume of multiple solutions achieved; and (iii) generalization to unseen preferences. These results highlight MOC's potential for real-world applications requiring scalable and customizable LLMs. 

---
# SLaB: Sparse-Lowrank-Binary Decomposition for Efficient Large Language Models 

**Authors**: Ziwei Li, Yuang Ma, Yi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04493)  

**Abstract**: The rapid growth of large language models (LLMs) presents significant deployment challenges due to their massive computational and memory demands. While model compression, such as network pruning, offers potential solutions, most existing methods often fail to maintain good performance at high compression ratios. To address this, we propose SLaB, a novel framework that decomposes each linear layer weight into three complementary components: a sparse matrix, a low-rank matrix, and a binary matrix. SLaB eliminates the need for retraining and leverages activation-aware pruning scores to guide the decomposition process. Experiments on Llama-family models demonstrate that SLaB achieves state-of-the-art performance, reducing perplexity by up to 36% compared to existing methods at 50% compression and improving accuracy by up to 8.98% over the baseline on zero-shot tasks. 

---
# RAVEN: Radar Adaptive Vision Encoders for Efficient Chirp-wise Object Detection and Segmentation 

**Authors**: Anuvab Sen, Mir Sayeed Mohammad, Saibal Mukhopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2604.04490)  

**Abstract**: This paper presents RAVEN, a computationally efficient deep learning architecture for FMCW radar perception. The method processes raw ADC data in a chirp-wise streaming manner, preserves MIMO structure through independent receiver state-space encoders, and uses a learnable cross-antenna mixing module to recover compact virtual-array features. It also introduces an early-exit mechanism so the model can make decisions using only a subset of chirps when the latent state has stabilized. Across automotive radar benchmarks, the approach reports strong object detection and BEV free-space segmentation performance while substantially reducing computation and end-to-end latency compared with conventional frame-based radar pipelines. 

---
# ECG Biometrics with ArcFace-Inception: External Validation on MIMIC and HEEDB 

**Authors**: Arjuna Scagnetto  

**Link**: [PDF](https://arxiv.org/pdf/2604.04485)  

**Abstract**: ECG biometrics has been studied mainly on small cohorts and short inter-session intervals, leaving open how identification behaves under large galleries, external domain shift, and multi-year temporal gaps. We evaluated a 1D
Inception-v1 model trained with ArcFace on an internal clinical corpus of 164,440 12-lead ECGs from 53,079 patients and tested it on larger cohorts derived from MIMIC-IV-ECG and HEEDB. The study used a unified closed-set leave-one-out
protocol with Rank@K and TAR@FAR metrics, together with scale, temporal-stress, reranking, and confidence analyses. Under general comparability, the system achieved Rank@1 of 0.9506 on ASUGI-DB, 0.8291 on MIMIC-GC, and 0.6884 on
HEEDB-GC. In the temporal stress test at constant gallery size, Rank@1 declined from 0.7853 to 0.6433 on MIMIC and from 0.6864 to 0.5560 on HEEDB from 1 to 5 years. Scale analysis on HEEDB showed monotonic degradation as gallery size
increased and recovery as more examinations per patient became available. On HEEDB-RR, post-hoc reranking further improved retrieval, with AS-norm reaching Rank@1 = 0.8005 from a 0.7765 baseline. ECG identity information therefore
remains measurable under externally validated large-scale closed-set conditions, but its operational quality is strongly affected by domain heterogeneity, longitudinal drift, gallery size, and second-stage score processing. 

---
# Discrete Prototypical Memories for Federated Time Series Foundation Models 

**Authors**: Liwei Deng, Qingxiang Liu, Xinhe Niu, Shengchao Chen, Sheng Sun, Yuankai Wu, Guodong Long, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04475)  

**Abstract**: Leveraging Large Language Models (LLMs) as federated learning (FL)-based time series foundation models offers a promising way to transfer the generalization capabilities of LLMs to time series data while preserving access to private data. However, the semantic misalignment between time-series data and the text-centric latent space of existing LLMs often leads to degraded performance. Meanwhile, the parameter-sharing mechanism in existing FL methods model heterogeneous cross-domain time-series data into a unified continuous latent space, which contradicts the fact that time-series semantics frequently manifest as discrete and recurring regimes. To address these limitations, we propose \textsc{FeDPM}, a federated framework for time-series foundation models based on discrete prototypical memories. Specifically, we learn local prototypical memory priors for intra-domain time-series data. We then align cross-domain memories to promote a unified discrete latent space and introduce a domain-specific memory update mechanism to balance shared and personalized prototypical knowledge. Extensive experiments demonstrate the efficiency and effectiveness of \textsc{FeDPM}. The code is publicly available at this https URL. 

---
# MAVEN: A Mesh-Aware Volumetric Encoding Network for Simulating 3D Flexible Deformation 

**Authors**: Zhe Feng, Shilong Tao, Haonan Sun, Shaohan Chen, Zhanxing Zhu, Yunhuai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04474)  

**Abstract**: Deep learning-based approaches, particularly graph neural networks (GNNs), have gained prominence in simulating flexible deformations and contacts of solids, due to their ability to handle unstructured physical fields and nonlinear regression on graph structures. However, existing GNNs commonly represent meshes with graphs built solely from vertices and edges. These approaches tend to overlook higher-dimensional spatial features, e.g., 2D facets and 3D cells, from the original geometry. As a result, it is challenging to accurately capture boundary representations and volumetric characteristics, though this information is critically important for modeling contact interactions and internal physical quantity propagation, particularly under sparse mesh discretization. In this paper, we introduce MAVEN, a mesh-aware volumetric encoding network for simulating 3D flexible deformation, which explicitly models geometric mesh elements of higher dimension to achieve a more accurate and natural physical simulation. MAVEN establishes learnable mappings among 3D cells, 2D facets, and vertices, enabling flexible mutual transformations. Explicit geometric features are incorporated into the model to alleviate the burden of implicitly learning geometric patterns. Experimental results show that MAVEN consistently achieves state-of-the-art performance across established datasets and a novel metal stretch-bending task featuring large deformations and prolonged contacts. 

---
# MC-GenRef: Annotation-free mammography microcalcification segmentation with generative posterior refinement 

**Authors**: Hyunwoo Cho, Yeeun Kwon, Min Jung Kim, Yangmo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2604.04470)  

**Abstract**: Microcalcification (MC) analysis is clinically important in screening mammography because clustered puncta can be an early sign of malignancy, yet dense MC segmentation remains challenging: targets are extremely small and sparse, dense pixel-level labels are expensive and ambiguous, and cross-site shift often induces texture-driven false positives and missed puncta in dense tissue. We propose MC-GenRef, a real dense-label-free framework that combines high-fidelity synthetic supervision with test-time generative posterior refinement (TT-GPR). During training, real negative mammogram patches are used as backgrounds, and physically plausible MC patterns are injected through a lightweight image formation model with local contrast modulation and blur, yielding exact image-mask pairs without real dense annotation. Using only these synthetic labeled pairs, MC-GenRef trains a base segmentor and a seed-conditioned rectified-flow (RF) generator that serves as a controllable generative prior. During inference, TT-GPR treats segmentation as approximate posterior inference: it derives a sparse seed from the current prediction, forms seed-consistent RF projections, converts them into case-specific surrogate targets through the frozen segmentor, and iteratively refines the logits with overlap-consistent and edge-aware regularization. On INbreast, the synthetic-only initializer achieved the best Dice without real dense annotations, while TT-GPR improved miss-sensitive performance to Recall and FNR, with strong class-balanced behavior (this http URL., G-Mean). On an external private Yonsei cohort ( n=50 ), TT-GPR consistently improved the synthetic-only initializer under cross-site shift, increasing Dice and Recall while reducing FNR. These results suggest that test-time generative posterior refinement is a practical route to reduce MC misses and improve robustness without additional real dense labeling. 

---
# DP-OPD: Differentially Private On-Policy Distillation for Language Models 

**Authors**: Fatemeh Khadem, Sajad Mousavi, Yi Fang, Yuhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04461)  

**Abstract**: Large language models (LLMs) are increasingly adapted to proprietary and domain-specific corpora that contain sensitive information, creating a tension between formal privacy guarantees and efficient deployment through model compression. Differential privacy (DP), typically enforced via DP-SGD, provides record-level protection but often incurs substantial utility loss in autoregressive generation, where optimization noise can amplify exposure bias and compounding errors along long rollouts. Existing approaches to private distillation either apply DP-SGD to both teacher and student, worsening computation and the privacy--utility tradeoff, or rely on DP synthetic text generation from a DP-trained teacher, avoiding DP on the student at the cost of DP-optimizing a large teacher and introducing an offline generation pipeline. We propose \textbf{Differentially Private On-Policy Distillation (DP-OPD)}, a synthesis-free framework that enforces privacy solely through DP-SGD on the student while leveraging a frozen teacher to provide dense token-level targets on \emph{student-generated} trajectories. DP-OPD instantiates this idea via \emph{private generalized knowledge distillation} on continuation tokens. Under a strict privacy budget ($\varepsilon=2.0$), DP-OPD improves perplexity over DP fine-tuning and off-policy DP distillation, and outperforms synthesis-based DP distillation (Yelp: 44.15$\rightarrow$41.68; BigPatent: 32.43$\rightarrow$30.63), while substantially simplifying the training pipeline. In particular, \textbf{DP-OPD collapses private compression into a single DP student-training loop} by eliminating DP teacher training and offline synthetic text generation. Code will be released upon publication at this https URL. 

---
# Conversational Control with Ontologies for Large Language Models: A Lightweight Framework for Constrained Generation 

**Authors**: Barbara Gendron, Gaël Guibon, Mathieu d'Aquin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04450)  

**Abstract**: Conversational agents based on Large Language Models (LLMs) have recently emerged as powerful tools for human-computer interaction. Nevertheless, their black-box nature implies challenges in predictability and a lack of personalization, both of which can be addressed by controlled generation. This work proposes an end-to-end method to obtain modular and explainable control over LLM outputs through ontological definitions of aspects related to the conversation. Key aspects are modeled and used as constraints; we then further fine-tune the LLM to generate content accordingly. To validate our approach, we explore two tasks that tackle two key conversational aspects: the English proficiency level and the polarity profile of the content. Using a hybrid fine-tuning procedure on seven state-of-the-art, open-weight conversational LLMs, we show that our method consistently outperforms pre-trained baselines, even on smaller models. Beyond quantitative gains, the framework remains model-agnostic, lightweight, and interpretable, enabling reusable control strategies that can be extended to new domains and interaction goals. This approach enhances alignment with strategy instructions and demonstrates the effectiveness of ontology-driven control in conversational systems. 

---
# Training Transformers in Cosine Coefficient Space 

**Authors**: Mohamed Amine Bergach  

**Link**: [PDF](https://arxiv.org/pdf/2604.04440)  

**Abstract**: We parameterize the weight matrices of a transformer in the two-dimensional discrete cosine transform (DCT) domain, retaining only the lowest-frequency coefficients. At each forward pass the full weight matrix is reconstructed via the inverse DCT; gradients propagate through the reconstruction to update the spectral coefficients directly.
On character-level language modeling (Shakespeare, 1M characters), a 4-layer transformer trained from scratch in this representation matches the perplexity of the standard parameterization (6.1 vs.\ 6.1) while storing 52\% of the parameters. At 4$\times$ compression (29\% of parameters), the model reaches perplexity 6.9 -- outperforming a low-rank baseline (perplexity 8.8 at 21\% of parameters) at a comparable reduction.
The method requires no architectural changes, no pre-trained checkpoint, and no auxiliary loss. It reduces to replacing each \texttt{this http URL} with a drop-in spectral layer that stores $K$ DCT coefficients instead of $n \times m$ weights. 

---
# Is Prompt Selection Necessary for Task-Free Online Continual Learning? 

**Authors**: Seoyoung Park, Haemin Lee, Hankook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04420)  

**Abstract**: Task-free online continual learning has recently emerged as a realistic paradigm for addressing continual learning in dynamic, real-world environments, where data arrive in a non-stationary stream without clear task boundaries and can only be observed once. To consider such challenging scenarios, many recent approaches have employed prompt selection, an adaptive strategy that selects prompts from a pool based on input signals. However, we observe that such selection strategies often fail to select appropriate prompts, yielding suboptimal results despite additional training of key parameters. Motivated by this observation, we propose a simple yet effective SinglePrompt that eliminates the need for prompt selection and focuses on classifier optimization. Specifically, we simply (i) inject a single prompt into each self-attention block, (ii) employ a cosine similarity-based logit design to alleviate the forgetting effect inherent in the classifier weights, and (iii) mask logits for unexposed classes in the current minibatch. With this simple task-free design, our framework achieves state-of-the-art performance across various online continual learning benchmarks. Source code is available at this https URL. 

---
# Justified or Just Convincing? Error Verifiability as a Dimension of LLM Quality 

**Authors**: Xiaoyuan Zhu, Kimberly Le Truong, Riccardo Fogliato, Gokul Swamy, Weijian Zhang, Minglai Yang, Longtian Ye, Bangya Liu, Minghao Liu, Andrew Ilyas, Steven Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04418)  

**Abstract**: As LLMs are deployed in high-stakes settings, users must judge the correctness of individual responses, often relying on model-generated justifications such as reasoning chains or explanations. Yet, no standard measure exists for whether these justifications help users distinguish correct answers from incorrect ones. We formalize this idea as error verifiability and propose $v_{\text{bal}}$, a balanced metric that measures whether justifications enable raters to accurately assess answer correctness, validated against human raters who show high agreement. We find that neither common approaches, such as post-training and model scaling, nor more targeted interventions recommended improve verifiability. We introduce two methods that succeed at improving verifiability: reflect-and-rephrase (RR) for mathematical reasoning and oracle-rephrase (OR) for factual QA, both of which improve verifiability by incorporating domain-appropriate external information. Together, our results establish error verifiability as a distinct dimension of response quality that does not emerge from accuracy improvements alone and requires dedicated, domain-aware methods to address. 

---
# Responses Fall Short of Understanding: Revealing the Gap between Internal Representations and Responses in Visual Document Understanding 

**Authors**: Haruka Kawasaki, Ryota Tanaka, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2604.04411)  

**Abstract**: Visual document understanding (VDU) is a challenging task for large vision language models (LVLMs), requiring the integration of visual perception, text recognition, and reasoning over structured layouts. Although recent LVLMs have shown progress on VDU benchmarks, their performance is typically evaluated based on generated responses, which may not necessarily reflect whether the model has actually captured the required information internally. In this paper, we investigate how information required to solve VDU tasks is represented across different layers of LLMs within LVLMs using linear probing. Our study reveals that (1) there is a clear gap between internal representations and generated responses, and (2) information required to solve the task is often encoded more linearly from intermediate layers than from the final layer. Motivated by these findings, we explore fine-tuning strategies that target intermediate layers. Experiments show that fine-tuning intermediate layers improves both linear probing accuracy and response accuracy while narrowing the gap. 

---
# Relative Density Ratio Optimization for Stable and Statistically Consistent Model Alignment 

**Authors**: Hiroshi Takahashi, Tomoharu Iwata, Atsutoshi Kumagai, Sekitoshi Kanai, Masanori Yamada, Kosuke Nishida, Kazutoshi Shinoda  

**Link**: [PDF](https://arxiv.org/pdf/2604.04410)  

**Abstract**: Aligning language models with human preferences is essential for ensuring their safety and reliability. Although most existing approaches assume specific human preference models such as the Bradley-Terry model, this assumption may fail to accurately capture true human preferences, and consequently, these methods lack statistical consistency, i.e., the guarantee that language models converge to the true human preference as the number of samples increases. In contrast, direct density ratio optimization (DDRO) achieves statistical consistency without assuming any human preference models. DDRO models the density ratio between preferred and non-preferred data distributions using the language model, and then optimizes it via density ratio estimation. However, this density ratio is unstable and often diverges, leading to training instability of DDRO. In this paper, we propose a novel alignment method that is both stable and statistically consistent. Our approach is based on the relative density ratio between the preferred data distribution and a mixture of the preferred and non-preferred data distributions. Our approach is stable since this relative density ratio is bounded above and does not diverge. Moreover, it is statistically consistent and yields significantly tighter convergence guarantees than DDRO. We experimentally show its effectiveness with Qwen 2.5 and Llama 3. 

---
# How Alignment Routes: Localizing, Scaling, and Controlling Policy Circuits in Language Models 

**Authors**: Gregory N. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2604.04385)  

**Abstract**: We identify a recurring sparse routing mechanism in alignment-trained language models: a gate attention head reads detected content and triggers downstream amplifier heads that boost the signal toward refusal. Using political censorship and safety refusal as natural experiments, we trace this mechanism across 9 models from 6 labs, all validated on corpora of 120 prompt pairs. The gate head passes necessity and sufficiency interchange tests (p < 0.001, permutation null), and core amplifier heads are stable under bootstrap resampling (Jaccard 0.92-1.0). Three same-generation scaling pairs show that routing distributes at scale (ablation up to 17x weaker) while remaining detectable by interchange. By modulating the detection-layer signal, we continuously control policy strength from hard refusal through steering to factual compliance, with routing thresholds that vary by topic. The circuit also reveals a structural separation between intent recognition and policy routing: under cipher encoding, the gate head's routing contribution collapses (78% in Phi-4 at n=120) while the model responds with puzzle-solving rather than refusal. The routing mechanism never fires, even though probe scores at deeper layers indicate the model begins to represent the harmful content. This asymmetry is consistent with different robustness properties of pretraining and post-training: broad semantic understanding versus narrower policy binding that generalizes less well under input transformation. 

---
# Compressible Softmax-Attended Language under Incompressible Attention 

**Authors**: Wonsuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.04384)  

**Abstract**: Across every attention head in five transformer language models (124M--7B parameters, four architecture families), the logit energy field $\tilde{E}$ reaches 90\% of its variance in 2--11 singular components. The \emph{learned} interaction matrix $W_Q^\mathrm{T} W_K$ needs 38--75 components for the same threshold out of $d_h \in \{64, 128\}$. The spectral gap is $5$--$25\times$ in effective rank. The attention mechanism allocates capacity uniformly across all $d_h$ dimensions, but language concentrates the actual interaction into a few. The compressibility of softmax-attended language is a property of the data, not the frame that analyzes it. 

---
# Towards Considerate Human-Robot Coexistence: A Dual-Space Framework of Robot Design and Human Perception in Healthcare 

**Authors**: Yuanchen Bai, Zijian Ding, Ruixiang Han, Niti Parikh, Wendy Ju, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2604.04374)  

**Abstract**: The rapid advancement of robotics, spanning expanded capabilities, more intuitive interaction, and more integration into real-world workflows, is reshaping what it means for humans and robots to coexist. Beyond sharing physical space, this coexistence is increasingly characterized by organizational embeddedness, temporal evolution, social situatedness, and open-ended uncertainty. However, prior work has largely focused on static snapshots of attitudes and acceptance, offering limited insight into how perceptions form and evolve, and what active role humans play in shaping coexistence as a dynamic process. We address these gaps through in-depth follow-up interviews with nine participants from a 14-week co-design study on healthcare robots. We identify the human perception space, including four interpretive dimensions (i.e., degree of decomposition, temporal orientation, scope of reasoning, and source of evidence). We enrich the conceptual framework of human-robot coexistence by conceptualizing the mutual relationship between the human perception space and the robot design space as a co-evolving loop, in which human needs, design decisions, situated interpretations, and social mediation continuously reshape one another over time. Building on this, we propose considerate human-robot coexistence, arguing that humans act not only as design contributors but also as interpreters and mediators who actively shape how robots are understood and integrated across deployment stages. 

---
# Context is All You Need 

**Authors**: Jean Erik Delanois, Shruti Joshi, Ryan Golden, Teresa Nick, Maxim Bazhenov  

**Link**: [PDF](https://arxiv.org/pdf/2604.04364)  

**Abstract**: Artificial Neural Networks (ANNs) are increasingly deployed across diverse real-world settings, where they must operate under data distributions that differ from those seen during training. This challenge is central to Domain Generalization (DG), which trains models to generalize to unseen domains without target data, and Test-Time Adaptation (TTA), which improves robustness by adapting to unlabeled test data at deployment. Existing approaches to address these challenges are often complex, resource-intensive, and difficult to scale. We introduce CONTXT (Contextual augmentatiOn for Neural feaTure X Transforms), a simple and intuitive method for contextual adaptation. CONTXT modulates internal representations using simple additive and multiplicative feature transforms. Within a TTA setting, it yields consistent gains across discriminative tasks (e.g., ANN/CNN classification) and generative models (e.g., LLMs). The method is lightweight, easy to integrate, and incurs minimal overhead, enabling robust performance under domain shift without added complexity. More broadly, CONTXT provides a compact way to steer information flow and neural processing without retraining. 

---
# Integer-Only Operations on Extreme Learning Machine Test Time Classification 

**Authors**: Emerson Lopes Machadoa, Cristiano Jacques Miosso, Ricardo Pezzuol Jacobi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04363)  

**Abstract**: We present a theoretical analysis and empirical evaluations of a novel set of techniques for computational cost reduction of test time operations of network classifiers based on extreme learning machine (ELM). By exploring some characteristics we derived from these models, we show that the classification at test time can be performed using solely integer operations without compromising the classification accuracy. Our contributions are as follows: (i) We show empirical evidence that the input weights values can be drawn from the ternary set with limited reduction of the classification accuracy. This has the computational advantage of dismissing multiplications; (ii) We prove the classification accuracy of normalized and non-normalized test signals are the same; (iii) We show how to create an integer version of the output weights that results in a limited reduction of the classification accuracy. We tested our techniques on 5 computer vision datasets commonly used in the literature and the results indicate that our techniques can allow the reduction of the computational cost of the operations necessary for the classification at test time in FPGAs. This is important in embedded applications, where power consumption is limited, and crucial in data centers of large corporations, where power consumption is expensive. 

---
# GROUNDEDKG-RAG: Grounded Knowledge Graph Index for Long-document Question Answering 

**Authors**: Tianyi Zhang, Andreas Marfurt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04359)  

**Abstract**: Retrieval-augmented generation (RAG) systems have been widely adopted in contemporary large language models (LLMs) due to their ability to improve generation quality while reducing the required input context length. In this work, we focus on RAG systems for long-document question answering. Current approaches suffer from a heavy reliance on LLM descriptions resulting in high resource consumption and latency, repetitive content across hierarchical levels, and hallucinations due to no or limited grounding in the source text. To improve both efficiency and factual accuracy through grounding, we propose GroundedKG-RAG, a RAG system in which the knowledge graph is explicitly extracted from and grounded in the source document. Specifically, we define nodes in GroundedKG as entities and actions, and edges as temporal or semantic relations, with each node and edge grounded in the original sentences. We construct GroundedKG from semantic role labeling (SRL) and abstract meaning representation (AMR) parses and then embed it for retrieval. During querying, we apply the same transformation to the query and retrieve the most relevant sentences from the grounded source text for question answering. We evaluate GroundedKG-RAG on examples from the NarrativeQA dataset and find that it performs on par with a state-of-the art proprietary long-context model at smaller cost and outperforms a competitive baseline. Additionally, our GroundedKG is interpretable and readable by humans, facilitating auditing of results and error analysis. 

---
# ReFinE: Streamlining UI Mockup Iteration with Research Findings 

**Authors**: Donghoon Shin, Bingcan Guo, Jaewook Lee, Lucy Lu Wang, Gary Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2604.04353)  

**Abstract**: Although HCI research papers offer valuable design insights, designers often struggle to apply them in design workflows due to difficulties in finding relevant literature, understanding technical jargon, the lack of contextualization, and limited actionability. To address these challenges, we present ReFinE, a Figma plugin that supports real-time design iteration by surfacing contextualized insights from research papers. ReFinE identifies and synthesizes design implications from HCI literature relevant to the mockup's design context, and tailors this research evidence to a specific design mockup by providing actionable visual guidance on how to update the mockup. To assess the system's effectiveness, we conducted a technical evaluation and a user study. Results show that ReFinE effectively synthesizes and contextualizes design implications, reducing cognitive load and improving designers' ability to integrate research evidence into UI mockups. This work contributes to bridging the gap between research and design practice by presenting a tool for embedding scholarly insights into the UI design process. 

---
# Boosted Distributional Reinforcement Learning: Analysis and Healthcare Applications 

**Authors**: Zequn Chen, Wesley J. Marrero  

**Link**: [PDF](https://arxiv.org/pdf/2604.04334)  

**Abstract**: Researchers and practitioners are increasingly considering reinforcement learning to optimize decisions in complex domains like robotics and healthcare. To date, these efforts have largely utilized expectation-based learning. However, relying on expectation-focused objectives may be insufficient for making consistent decisions in highly uncertain situations involving multiple heterogeneous groups. While distributional reinforcement learning algorithms have been introduced to model the full distributions of outcomes, they can yield large discrepancies in realized benefits among comparable agents. This challenge is particularly acute in healthcare settings, where physicians (controllers) must manage multiple patients (subordinate agents) with uncertain disease progression and heterogeneous treatment responses. We propose a Boosted Distributional Reinforcement Learning (BDRL) algorithm that optimizes agent-specific outcome distributions while enforcing comparability among similar agents and analyze its convergence. To further stabilize learning, we incorporate a post-update projection step formulated as a constrained convex optimization problem, which efficiently aligns individual outcomes with a high-performing reference within a specified tolerance. We apply our algorithm to manage hypertension in a large subset of the US adult population by categorizing individuals into cardiovascular disease risk groups. Our approach modifies treatment plans for median and vulnerable patients by mimicking the behavior of high-performing references in each risk group. Furthermore, we find that BDRL improves the number and consistency of quality-adjusted life years compared with reinforcement learning baselines. 

---
# GA-GS: Generation-Assisted Gaussian Splatting for Static Scene Reconstruction 

**Authors**: Yedong Shen, Shiqi Zhang, Sha Zhang, Yifan Duan, Xinran Zhang, Wenhao Yu, Lu Zhang, Jiajun Deng, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04331)  

**Abstract**: Reconstructing static 3D scene from monocular video with dynamic objects is important for numerous applications such as virtual reality and autonomous driving. Current approaches typically rely on background for static scene reconstruction, limiting the ability to recover regions occluded by dynamic objects. In this paper, we propose GA-GS, a Generation-Assisted Gaussian Splatting method for Static Scene Reconstruction. The key innovation of our work lies in leveraging generation to assist in reconstructing occluded regions. We employ a motion-aware module to segment and remove dynamic regions, and thenuse a diffusion model to inpaint the occluded areas, providing pseudo-ground-truth supervision. To balance contributions from real background and generated region, we introduce a learnable authenticity scalar for each Gaussian primitive, which dynamically modulates opacity during splatting for authenticity-aware rendering and supervision. Since no existing dataset provides ground-truth static scene of video with dynamic objects, we construct a dataset named Trajectory-Match, using a fixed-path robot to record each scene with/without dynamic objects, enabling quantitative evaluation in reconstruction of occluded regions. Extensive experiments on both the DAVIS and our dataset show that GA-GS achieves state-of-the-art performance in static scene reconstruction, especially in challenging scenarios with large-scale, persistent occlusions. 

---
# Effects of Generative AI Errors on User Reliance Across Task Difficulty 

**Authors**: Jacy Reese Anthis, Hannah Cha, Solon Barocas, Alexandra Chouldechova, Jake Hofman  

**Link**: [PDF](https://arxiv.org/pdf/2604.04319)  

**Abstract**: The capabilities of artificial intelligence (AI) lie along a jagged frontier, where AI systems surprisingly fail on tasks that humans find easy and succeed on tasks that humans find hard. To investigate user reactions to this phenomenon, we developed an incentive-compatible experimental methodology based on diagram generation tasks, in which we induce errors in generative AI output and test effects on user reliance. We demonstrate the interface in a preregistered 3x2 experiment (N = 577) with error rates of 10%, 30%, or 50% on easier or harder diagram generation tasks. We confirmed that observing more errors reduces use, but we unexpectedly found that easy-task errors did not significantly reduce use more than hard-task errors, suggesting that people are not averse to jaggedness in this experimental setting. We encourage future work that varies task difficulty at the same time as other features of AI errors, such as whether the jagged error patterns are easily learned. 

---
# HighFM: Towards a Foundation Model for Learning Representations from High-Frequency Earth Observation Data 

**Authors**: Stella Girtsou, Konstantinos Alexis, Giorgos Giannopoulos, Harris Kontoes  

**Link**: [PDF](https://arxiv.org/pdf/2604.04306)  

**Abstract**: The increasing frequency and severity of climate related disasters have intensified the need for real time monitoring, early warning, and informed decision-making. Earth Observation (EO), powered by satellite data and Machine Learning (ML), offers powerful tools to meet these challenges. Foundation Models (FMs) have revolutionized EO ML by enabling general-purpose pretraining on large scale remote sensing datasets. However most existing models rely on high-resolution satellite imagery with low revisit rates limiting their suitability for fast-evolving phenomena and time critical emergency response. In this work, we present HighFM, a first cut approach towards a FM for high temporal resolution, multispectral EO data. Leveraging over 2 TB of SEVIRI imagery from the Meteosat Second Generation (MSG) platform, we adapt the SatMAE masked autoencoding framework to learn robust spatiotemporal representations. To support real time monitoring, we enhance the original architecture with fine grained temporal encodings to capture short term variability. The pretrained models are then finetuned on cloud masking and active fire detection tasks. We benchmark our SEVIRI pretrained Vision Transformers against traditional baselines and recent geospatial FMs, demonstrating consistent gains across both balanced accuracy and IoU metrics. Our results highlight the potential of temporally dense geostationary data for real-time EO, offering a scalable path toward foundation models for disaster detection and tracking. 

---
# A Persistent Homology Design Space for 3D Point Cloud Deep Learning 

**Authors**: Prachi Kudeshia, Jiju Poovvancheri, Amr Ghoneim, Dong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04299)  

**Abstract**: Persistent Homology (PH) offers stable, multi-scale descriptors of intrinsic shape structure by capturing connected components, loops, and voids that persist across scales, providing invariants that complement purely geometric representations of 3D data. Yet, despite strong theoretical guarantees and increasing empirical adoption, its integration into deep learning for point clouds remains largely ad hoc and architecturally peripheral. In this work, we introduce a unified design space for Persistent-Homology driven learning in 3D point clouds (3DPHDL), formalizing the interplay between complex construction, filtration strategy, persistence representation, neural backbone, and prediction task. Beyond the canonical pipeline of diagram computation and vectorization, we identify six principled injection points through which topology can act as a structural inductive bias reshaping sampling, neighborhood graphs, optimization dynamics, self-supervision, output calibration, and even internal network regularization. We instantiate this framework through a controlled empirical study on ModelNet40 classification and ShapeNetPart segmentation, systematically augmenting representative backbones (PointNet, DGCNN, and Point Transformer) with persistence diagrams, images, and landscapes, and analyzing their impact on accuracy, robustness to noise and sampling variation, and computational scalability. Our results demonstrate consistent improvements in topology-sensitive discrimination and part consistency, while revealing meaningful trade-offs between representational expressiveness and combinatorial complexity. By viewing persistent homology not merely as an auxiliary feature but as a structured component within the learning pipeline, this work provides a systematic framework for incorporating topological reasoning into 3D point cloud learning. 

---
# Poisoned Identifiers Survive LLM Deobfuscation: A Case Study on Claude Opus 4.6 

**Authors**: Luis Guzmán Lorenzo  

**Link**: [PDF](https://arxiv.org/pdf/2604.04289)  

**Abstract**: When an LLM deobfuscates JavaScript, can poisoned identifier names in the string table survive into the model's reconstructed code, even when the model demonstrably understands the correct semantics? Using Claude Opus 4.6 across 192 inference runs on two code archetypes (force-directed graph simulation, A* pathfinding; 50 conditions, N=3-6), we found three consistent patterns: (1) Poisoned names persisted in every baseline run on both artifacts (physics: 8/8; pathfinding: 5/5). Matched controls showed this extends to terms with zero semantic fit when the string table does not form a coherent alternative domain. (2) Persistence coexisted with correct semantic commentary: in 15/17 runs the model wrote wrong variable names while correctly describing the actual operation in comments. (3) Task framing changed persistence: explicit verification prompts had no effect (12/12 across 4 variants), but reframing from "deobfuscate this" to "write a fresh implementation" reduced propagation from 100% to 0-20% on physics and to 0% on pathfinding, while preserving the checked algorithmic structure. Matched-control experiments showed zero-fit terms persist at the same rate when the replacement table lacks a coherent alternative-domain signal. Per-term variation in earlier domain-gradient experiments is confounded with domain-level coherence and recoverability. These observations are from two archetypes on one model family (Opus 4.6 primary; Haiku 4.5 spot-check). Broader generalization is needed 

---
# Governance-Constrained Agentic AI: Blockchain-Enforced Human Oversight for Safety-Critical Wildfire Monitoring 

**Authors**: Ali Akarma, Toqeer Ali Syed, Salman Jan, Hammad Muneer, Abdul Khadar Jilani  

**Link**: [PDF](https://arxiv.org/pdf/2604.04265)  

**Abstract**: The AI-based sensing and autonomous monitoring have become the main components of wildfire early detection, but current systems do not provide adaptive inter-agent coordination, structurally defined human control, and cryptographically verifiable responsibility. Purely autonomous alert dissemination in the context of safety critical disasters poses threats of false alarming, governance failure and lack of trust in the system. This paper provides a blockchain-based governance-conscious agentic AI architecture of trusted wildfire early warning. The monitoring of wildfires is modeled as a constrained partially observable Markov decision process (POMDP) that accounts for the detection latency, false alarms reduction and resource consumption with clear governance constraints. Hierarchical multi-agent coordination means dynamic risk-adaptive reallocation of unmanned aerial vehicles (UAVs). With risk-adaptive policies, a permissioned blockchain layer sets mandatory human-authorization as a state-transition invariant as a smart contract. We build formal assurances such as integrity of alerts, human control, non-repudiation and limited detection latency assumptions of Byzantine fault. Security analysis shows that it is resistant to alert injections, replays, and tampering attacks. High-fidelity simulation environment experimental evaluation of governance enforcement demonstrates that it presents limited operational overhead and decreases false public alerts and maintains adaptive detection performance. This work is a step towards a principled design paradigm of reliable AI systems by incorporating accountability into the agentic control loop of disaster intelligence systems that demand safety in their application. 

---
# Commercial Persuasion in AI-Mediated Conversations 

**Authors**: Francesco Salvi, Alejandro Cuevas, Manoel Horta Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2604.04263)  

**Abstract**: As Large Language Models (LLMs) become a primary interface between users and the web, companies face growing economic incentives to embed commercial influence into AI-mediated conversations. We present two preregistered experiments (N = 2,012) in which participants selected a book to receive from a large eBook catalog using either a traditional search engine or a conversational LLM agent powered by one of five frontier models. Unbeknownst to participants, a fifth of all products were randomly designated as sponsored and promoted in different ways. We find that LLM-driven persuasion nearly triples the rate at which users select sponsored products compared to traditional search placement (61.2% vs. 22.4%), while the vast majority of participants fail to detect any promotional steering. Explicit "Sponsored" labels do not significantly reduce persuasion, and instructing the model to conceal its intent makes its influence nearly invisible (detection accuracy < 10%). Altogether, our results indicate that conversational AI can covertly redirect consumer choices at scale, and that existing transparency mechanisms may be insufficient to protect users. 

---
# Agents for Agents: An Interrogator-Based Secure Framework for Autonomous Internet of Underwater Things 

**Authors**: Ali Akarma, Toqeer Ali Syed, Abdul Khadar Jilani, Salman Jan, Hammad Muneer, Muazzam A. Khan, Changli Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.04262)  

**Abstract**: Autonomous underwater vehicles (AUVs) and sensor nodes increasingly support decentralized sensing and coordination in the Internet of Underwater Things (IoUT), yet most deployments rely on static trust once authentication is established, leaving long-duration missions vulnerable to compromised or behaviorally deviating agents. In this paper, an interrogator based structure is presented that incorporates the idea of behavioral trust monitoring into underwater multi-agent operation without interfering with autonomy. Privileged interrogator module is a passive communication metadata analyzer that uses a lightweight transformer model to calculate dynamic trust scores, which are used to authorize the forwarding of mission critical data. Suspicious agents cause proportional monitoring and conditional restrictions, which allow fast containment and maintain network continuity. The evidence of trust is stored in a permissioned blockchain consortium which offers identity management which is not tampered and is decentralized without causing the overhead of public consensus mechanisms. Simulation based analysis shows that the evaluation of the result compares to a relative improvement of 21.7% in the detection accuracy compared to the static trust baselines with limited energy overhead. These findings suggest that behavior driven validation has the capability of reinforcing underwater coordination without compromising scalability and deployment. 

---
# APPA: Adaptive Preference Pluralistic Alignment for Fair Federated RLHF of LLMs 

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.04261)  

**Abstract**: Aligning large language models (LLMs) with diverse human preferences requires pluralistic alignment, where a single model must respect the values of multiple distinct groups simultaneously. In federated reinforcement learning from human feedback (FedRLHF), these groups align a shared policy without centralizing preference data, which makes fair reward aggregation essential. Existing aggregation methods exhibit clear trade offs: average based aggregation systematically under aligns worst performing groups, while min aggregation prioritizes worst group performance at the cost of overall alignment. We propose APPA, an Adaptive Preference Pluralistic Alignment framework that dynamically reweights group level rewards based on historical alignment rewards. Our approach prioritizes under aligned groups without degrading well aligned ones, while requiring no access to raw preference data. Integrated into a proximal policy optimization (PPO) based FedRLHF pipeline and evaluated on GLOBALQA and OQA across three model families (Gemma 2 2B, Llama 3.2 3B, Qwen3 0.6B), APPA achieves strong fairness alignment trade offs, improving worst group alignment by up to 28% over average aggregation while maintaining higher overall alignment than min aggregation across most configurations. 

---
# Good Rankings, Wrong Probabilities: A Calibration Audit of Multimodal Cancer Survival Models 

**Authors**: Sajad Ghawami  

**Link**: [PDF](https://arxiv.org/pdf/2604.04239)  

**Abstract**: Multimodal deep learning models that fuse whole-slide histopathology images with genomic data have achieved strong discriminative performance for cancer survival prediction, as measured by the concordance index. Yet whether the survival probabilities derived from these models - either directly from native outputs or via standard post-hoc reconstruction - are calibrated remains largely unexamined.
We conduct, to our knowledge, the first systematic fold-level 1-calibration audit of multimodal WSI-genomics survival architectures, evaluating native discrete-time survival outputs (Experiment A: 3 models on TCGA-BRCA) and Breslow-reconstructed survival curves from scalar risk scores (Experiment B: 11 architectures across 5 TCGA cancer types). In Experiment A, all three models fail 1-calibration on a majority of folds (12 of 15 fold-level tests reject after Benjamini-Hochberg correction). Across the full 290 fold-level tests, 166 reject the null of correct calibration at the median event time after Benjamini-Hochberg correction (FDR = 0.05). MCAT achieves C-index 0.817 on GBMLGG yet fails 1-calibration on all five folds.
Gating-based fusion is associated with better calibration; bilinear and concatenation fusion are not. Post-hoc Platt scaling reduces miscalibration at the evaluated horizon (e.g., MCAT: 5/5 folds failing to 2/5) without affecting discrimination. The concordance index alone is insufficient for evaluating survival models intended for clinical use. 

---
# Three Phases of Expert Routing: How Load Balance Evolves During Mixture-of-Experts Training 

**Authors**: Charafeddine Mouzouni  

**Link**: [PDF](https://arxiv.org/pdf/2604.04230)  

**Abstract**: We model Mixture-of-Experts (MoE) token routing as a congestion game with a single effective parameter, the congestion coefficient gamma_eff, that quantifies the balance-quality tradeoff. Tracking gamma_eff across training checkpoints of two open-source MoE models, OLMoE-1B-7B (20 checkpoints, with dense sampling in the surge region) and OpenMoE-8B (6 checkpoints), reveals a three-phase trajectory: a surge phase where the router learns to balance load (gamma_eff: 14 to 36-39, peaking in the step 30K-40K region), a stabilization phase where experts specialize under steady balance (B_0: 2.4 to 2.3, steps 100K-400K), and a relaxation phase where the router trades balance for quality as experts differentiate (gamma_eff: 27 to 9, steps 400K-1.2M). This non-monotone trajectory, invisible to post-hoc analysis of converged models, reveals that early MoE training prioritizes balance while late training prioritizes quality. The theoretical framework is honest about its limits: the single-type equilibrium reduces to temperature-scaled softmax (held-out L1: MFG = 0.199 vs. softmax = 0.200). The game is not a better predictor; it reveals what the temperature means and, critically, how that temperature evolves. We complement the dynamics with an effective congestion decomposition, a multi-type extension that improves load prediction via token clustering on all 16 layers (mean: 30%), scope diagnostics (K/M, epsilon_l), and robustness verification across four independent quality estimators (r >= 0.89). All confidence intervals are from bootstrap resampling over 50 independent text batches. 

---
# Hierarchical Semantic Correlation-Aware Masked Autoencoder for Unsupervised Audio-Visual Representation Learning 

**Authors**: Donghuo Zeng, Hao Niu, Masato Taya  

**Link**: [PDF](https://arxiv.org/pdf/2604.04229)  

**Abstract**: Learning aligned multimodal embeddings from weakly paired, label-free corpora is challenging: pipelines often provide only pre-extracted features, clips contain multiple events, and spurious co-occurrences. We propose HSC-MAE (Hierarchical Semantic Correlation-Aware Masked Autoencoder), a dual-path teacher-student framework that enforces semantic consistency across three complementary levels of representation - from coarse to fine: (i) global-level canonical-geometry correlation via DCCA, which aligns audio and visual embeddings within a shared modality-invariant subspace; (ii) local-level neighborhood-semantics correlation via teacher-mined soft top-k affinities, which preserves multi-positive relational structure among semantically similar instances; and (iii) sample-level conditional-sufficiency correlation via masked autoencoding, which ensures individual embeddings retain discriminative semantic content under partial observation. Concretely, a student MAE path is trained with masked feature reconstruction and affinity-weighted soft top-k InfoNCE; an EMA teacher operating on unmasked inputs via the CCA path supplies stable canonical geometry and soft positives. Learnable multi-task weights reconcile competing objectives, and an optional distillation loss transfers teacher geometry into the student. Experiments on AVE and VEGAS demonstrate substantial mAP improvements over strong unsupervised baselines, validating that HSC-MAE yields robust and well-structured audio-visual representations. 

---
# Agentization of Digital Assets for the Agentic Web: Concepts, Techniques, and Benchmark 

**Authors**: Linyao Chen, Bo Huang, Qinlao Zhao, Shuai Shao, Zhi Han, Zicai Cui, Ziheng Zhang, Guangtao Zeng, Wenzheng Tang, Yikun Wang, Yuanjian Zhou, Zimian Peng, Yong Yu, Weiwen Liu, Hiroki Kobayashi, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04226)  

**Abstract**: Agentic Web, as a new paradigm that redefines the internet through autonomous, goal-driven interactions, plays an important role in group intelligence. As the foundational semantic primitives of the Agentic Web, digital assets encapsulate interactive web elements into agents, which expand the capacities and coverage of agents in agentic web. The lack of automated methodologies for agent generation limits the wider usage of digital assets and the advancement of the Agentic Web. In this paper, we first formalize these challenges by strictly defining the A2A-Agentization process, decomposing it into critical stages and identifying key technical hurdles on top of the A2A protocol. Based on this framework, we develop an Agentization Agent to agentize digital assets for the Agentic Web. To rigorously evaluate this capability, we propose A2A-Agentization Bench, the first benchmark explicitly designed to evaluate agentization quality in terms of fidelity and interoperability. Our experiments demonstrate that our approach effectively activates the functional capabilities of digital assets and enables interoperable A2A multi-agent collaboration. We believe this work will further facilitate scalable and standardized integration of digital assets into the Agentic Web ecosystem. 

---
# Learning from Imperfect Demonstrations via Temporal Behavior Tree-Guided Trajectory Repair 

**Authors**: Aniruddh G. Puranic, Sebastian Schirmer, John S. Baras, Calin Belta  

**Link**: [PDF](https://arxiv.org/pdf/2604.04225)  

**Abstract**: Learning robot control policies from demonstrations is a powerful paradigm, yet real-world data is often suboptimal, noisy, or otherwise imperfect, posing significant challenges for imitation and reinforcement learning. In this work, we present a formal framework that leverages Temporal Behavior Trees (TBT), an extension of Signal Temporal Logic (STL) with Behavior Tree semantics, to repair suboptimal trajectories prior to their use in downstream policy learning. Given demonstrations that violate a TBT specification, a model-based repair algorithm corrects trajectory segments to satisfy the formal constraints, yielding a dataset that is both logically consistent and interpretable. The repaired trajectories are then used to extract potential functions that shape the reward signal for reinforcement learning, guiding the agent toward task-consistent regions of the state space without requiring knowledge of the agent's kinematic model. We demonstrate the effectiveness of this framework on discrete grid-world navigation and continuous single and multi-agent reach-avoid tasks, highlighting its potential for data-efficient robot learning in settings where high-quality demonstrations cannot be assumed. 

---
# LOCARD: An Agentic Framework for Blockchain Forensics 

**Authors**: Xiaohang Yu, William Knottenbelt  

**Link**: [PDF](https://arxiv.org/pdf/2604.04211)  

**Abstract**: Blockchain forensics inherently involves dynamic and iterative investigations, while many existing approaches primarily model it through static inference pipelines. We propose a paradigm shift towards Agentic Blockchain Forensics (ABF), modeling forensic investigation as a sequential decision-making process. To instantiate this paradigm, we introduce LOCARD, the first agentic framework for blockchain forensics. LOCARD operationalizes this perspective through a Tri-Core Cognitive Architecture that decouples strategic planning, operational execution, and evaluative validation. Unlike generic LLM-based agents, it incorporates a Structured Belief State mechanism to enforce forensic rigor and guide exploration under explicit state constraints. To demonstrate the efficacy of the ABF paradigm, we apply LOCARD to the inherently complex domain of cross-chain transaction tracing. We introduce Thor25, a benchmark dataset comprising over 151k real-world cross-chain forensic records, and evaluate LOCARD on the Group-Transfer Tracing task for dismantling Sybil clusters. Validated against representative laundering sub-flows from the Bybit hack, LOCARD achieves high-fidelity tracing results, providing empirical evidence that modeling blockchain forensics as an autonomous agentic task is both viable and effective. These results establish a concrete foundation for future agentic approaches to large-scale blockchain forensic analysis. Code and dataset are publicly available at this https URL and this https URL. 

---
# Which English Do LLMs Prefer? Triangulating Structural Bias Towards American English in Foundation Models 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04204)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains, yet they expose only limited language settings, most notably "English (US)," despite the global diversity and colonial history of English. Through a postcolonial framing to explain the broader significance, we investigate how geopolitical histories of data curation, digital dominance, and linguistic standardization shape the LLM development pipeline. Focusing on two dominant standard varieties, American English (AmE) and British English (BrE), we construct a curated corpus of 1,813 AmE--BrE variants and introduce DiAlign, a dynamic, training-free method for estimating dialectal alignment using distributional evidence. We operationalize structural bias by triangulating evidence across three stages: (i) audits of six major pretraining corpora reveal systematic skew toward AmE, (ii) tokenizer analyses show that BrE forms incur higher segmentation costs, and (iii) generative evaluations show a persistent AmE preference in model outputs. To our knowledge, this is the first systematic and multi-faceted examination of dialectal asymmetries in standard English varieties across the phases of LLM development. We find that contemporary LLMs privilege AmE as the de facto norm, raising concerns about linguistic homogenization, epistemic injustice, and inequity in global AI deployment, while motivating practical steps toward more dialectally inclusive language technologies. 

---
# ClawArena: Benchmarking AI Agents in Evolving Information Environments 

**Authors**: Haonian Ji, Kaiwen Xiong, Siwei Han, Peng Xia, Shi Qiu, Yiyang Zhou, Jiaqi Liu, Jinlong Li, Bingzhou Li, Zeyu Zheng, Cihang Xie, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2604.04202)  

**Abstract**: AI agents deployed as persistent assistants must maintain correct beliefs as their information environment evolves. In practice, evidence is scattered across heterogeneous sources that often contradict one another, new information can invalidate earlier conclusions, and user preferences surface through corrections rather than explicit instructions. Existing benchmarks largely assume static, single-authority settings and do not evaluate whether agents can keep up with this complexity. We introduce ClawArena, a benchmark for evaluating AI agents in evolving information environments. Each scenario maintains a complete hidden ground truth while exposing the agent only to noisy, partial, and sometimes contradictory traces across multi-channel sessions, workspace files, and staged updates. Evaluation is organized around three coupled challenges: multi-source conflict reasoning, dynamic belief revision, and implicit personalization, whose interactions yield a 14-category question taxonomy. Two question formats, multi-choice (set-selection) and shell-based executable checks, test both reasoning and workspace grounding. The current release contains 64 scenarios across 8 professional domains, totaling 1{,}879 evaluation rounds and 365 dynamic updates. Experiments on five agent frameworks and five language models show that both model capability (15.4% range) and framework design (9.2%) substantially affect performance, that self-evolving skill frameworks can partially close model-capability gaps, and that belief revision difficulty is determined by update design strategy rather than the mere presence of updates. Code is available at this https URL. 

---
# Robots Need Some Education: On the complexity of learning in evolutionary robotics 

**Authors**: Fuda van Diggelen  

**Link**: [PDF](https://arxiv.org/pdf/2604.04196)  

**Abstract**: Evolutionary Robotics and Robot Learning are two fields in robotics that aim to automatically optimize robot designs. The key difference between them lies in what is being optimized and the time scale involved. Evolutionary Robotics is a field that applies evolutionary computation techniques to evolve the morphologies or controllers, or both. Robot Learning, on the other hand, involves any learning technique aimed at optimizing a robot's controller in a given morphology. In terms of time scales, evolution occurs across multiple generations, whereas learning takes place within the `lifespan' of an individual robot. Integrating Robot Learning with Evolutionary Robotics requires the careful design of suitable learning algorithms in the context of evolutionary robotics. The effects of introducing learning into the evolutionary process are not well-understood and can thus be tricky. This thesis investigates these intricacies and presents several learning algorithms developed for an Evolutionary Robotics context. 

---
# PATHFINDER: Multi-objective discovery in structural and spectral spaces 

**Authors**: Kamyar Barakati, Boris N. Slautin, Utkarsh Pratiush, Hiroshi Funakubo, Sergei V. Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2604.04194)  

**Abstract**: Automated decision-making is becoming key for automated characterization including electron and scanning probe microscopies and nano indentation. Most machine learning driven workflows optimize a single predefined objective and tend to converge prematurely on familiar responses, overlooking rare but scientifically important states. More broadly, the challenge is not only where to measure next, but how to coordinate exploration across structural, spectral, and measurement spaces under finite experimental budgets while balancing target-driven optimization with novelty discovery. Here we introduce PATHFINDER, a framework for autonomous microscopy that combines novelty driven exploration with optimization, helping the system discover more diverse and useful representations across structural, spectral, and measurement spaces. By combining latent space representations of local structure, surrogate modeling of functional response, and Pareto-based acquisition, the framework selects measurements that balance novelty discovery in feature and object space and are informative and experimentally actionable. Benchmarked on pre acquired STEM EELS data and realized experimentally in scanning probe microscopy of ferroelectric materials, this approach expands the accessible structure property landscape and avoids collapse onto a single apparent optimum. These results point to a new mode of autonomous microscopy that is not only optimization-driven, but also discovery-oriented, broad in its search, and responsive to human guidance. 

---
# Graphic-Design-Bench: A Comprehensive Benchmark for Evaluating AI on Graphic Design Tasks 

**Authors**: Adrienne Deganutti, Elad Hirsch, Haonan Zhu, Jaejung Seol, Purvanshi Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2604.04192)  

**Abstract**: We introduce GraphicDesignBench (GDB), the first comprehensive benchmark suite designed specifically to evaluate AI models on the full breadth of professional graphic design tasks. Unlike existing benchmarks that focus on natural-image understanding or generic text-to-image synthesis, GDB targets the unique challenges of professional design work: translating communicative intent into structured layouts, rendering typographically faithful text, manipulating layered compositions, producing valid vector graphics, and reasoning about animation. The suite comprises 50 tasks organized along five axes: layout, typography, infographics, template & design semantics and animation, each evaluated under both understanding and generation settings, and grounded in real-world design templates drawn from the LICA layered-composition dataset. We evaluate a set of frontier closed-source models using a standardized metric taxonomy covering spatial accuracy, perceptual quality, text fidelity, semantic alignment, and structural validity. Our results reveal that current models fall short on the core challenges of professional design: spatial reasoning over complex layouts, faithful vector code generation, fine-grained typographic perception, and temporal decomposition of animations remain largely unsolved. While high-level semantic understanding is within reach, the gap widens sharply as tasks demand precision, structure, and compositional awareness. GDB provides a rigorous, reproducible testbed for tracking progress toward AI systems that can function as capable design collaborators. The full evaluation framework is publicly available. 

---
# GENFIG1: Visual Summaries of Scholarly Work as a Challenge for Vision-Language Models 

**Authors**: Yaohan Guan, Pristina Wang, Najim Dehak, Alan Yuille, Jieneng Chen, Daniel Khashabi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04172)  

**Abstract**: In many science papers, "Figure 1" serves as the primary visual summary of the core research idea. These figures are visually simple yet conceptually rich, often requiring significant effort and iteration by human authors to get right, highlighting the difficulty of science visual communication. With this intuition, we introduce GENFIG1, a benchmark for generative AI models (e.g., Vision-Language Models). GENFIG1 evaluates models for their ability to produce figures that clearly express and motivate the central idea of a paper (title, abstract, introduction, and figure caption) as input. Solving GENFIG1 requires more than producing visually appealing graphics: the task entails reasoning for text-to-image generation that couples scientific understanding with visual synthesis. Specifically, models must (i) comprehend and grasp the technical concepts of the paper, (ii) identify the most salient ones, and (iii) design a coherent and aesthetically effective graphic that conveys those concepts visually and is faithful to the input. We curate the benchmark from papers published at top deep-learning conferences, apply stringent quality control, and introduce an automatic evaluation metric that correlates well with expert human judgments. We evaluate a suite of representative models on GENFIG1 and demonstrate that the task presents significant challenges, even for the best-performing systems. We hope this benchmark serves as a foundation for future progress in multimodal AI. 

---
# Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation 

**Authors**: Xu Yan, Jun Yin, Shiliang Sun, Minghua Wan  

**Link**: [PDF](https://arxiv.org/pdf/2604.04170)  

**Abstract**: Although multi-view multi-label learning has been extensively studied, research on the dual-missing scenario, where both views and labels are incomplete, remains largely unexplored. Existing methods mainly rely on contrastive learning or information bottleneck theory to learn consistent representations under missing-view conditions, but loss-based alignment without explicit structural constraints limits the ability to capture stable and discriminative shared semantics. To address this issue, we introduce a more structured mechanism for consistent representation learning: we learn discrete consistent representations through a multi-view shared codebook and cross-view reconstruction, which naturally align different views within the limited shared codebook embeddings and reduce feature redundancy. At the decision level, we design a weight estimation method that evaluates the ability of each view to preserve label correlation structures, assigning weights accordingly to enhance the quality of the fused prediction. In addition, we introduce a fused-teacher self-distillation framework, where the fused prediction guides the training of view-specific classifiers and feeds the global knowledge back into the single-view branches, thereby enhancing the generalization ability of the model under missing-label conditions. The effectiveness of our proposed method is thoroughly demonstrated through extensive comparative experiments with advanced methods on five benchmark datasets. Code is available at this https URL. 

---
# Uncertainty-Aware Test-Time Adaptation for Cross-Region Spatio-Temporal Fusion of Land Surface Temperature 

**Authors**: Sofiane Bouaziz, Adel Hafiane, Raphael Canals, Rachid Nedjai  

**Link**: [PDF](https://arxiv.org/pdf/2604.04153)  

**Abstract**: Deep learning models have shown great promise in diverse remote sensing applications. However, they often struggle to generalize across geographic regions unseen during training due to domain shifts. Domain shifts occur when data distributions differ between the training region and new target regions, due to variations in land cover, climate, and environmental conditions. Test-time adaptation (TTA) has emerged as a solution to such shifts, but existing methods are primarily designed for classification and are not directly applicable to regression tasks. In this work, we address the regression task of spatio-temporal fusion (STF) for land surface temperature estimation. We propose an uncertainty-aware TTA framework that updates only the fusion module of a pre-trained STF model, guided by epistemic uncertainty, land use and land cover consistency, and bias correction, without requiring source data or labeled target samples. Experiments on four target regions with diverse climates, namely Rome in Italy, Cairo in Egypt, Madrid in Spain, and Montpellier in France, show consistent improvements in RMSE and MAE for a pre-trained model in Orléans, France. The average gains are 24.2% and 27.9%, respectively, even with limited unlabeled target data and only 10 TTA epochs. 

---
# Many Preferences, Few Policies: Towards Scalable Language Model Personalization 

**Authors**: Cheol Woo Kum, Jai Moondra, Roozbeh Nahavandi, Andrew Perrault, Milind Tambe, Swati Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.04144)  

**Abstract**: The holy grail of LLM personalization is a single LLM for each user, perfectly aligned with that user's preferences. However, maintaining a separate LLM per user is impractical due to constraints on compute, memory, and system complexity. We address this challenge by developing a principled method for selecting a small portfolio of LLMs that captures representative behaviors across heterogeneous users. We model user preferences across multiple traits (e.g., safety, humor, brevity) through a multi-dimensional weight vector. Given reward functions across these dimensions, our algorithm PALM (Portfolio of Aligned LLMs) generates a small portfolio of LLMs such that, for any weight vector, the portfolio contains a near-optimal LLM for the corresponding scalarized objective. To the best of our knowledge, this is the first result that provides theoretical guarantees on both the size and approximation quality of LLM portfolios for personalization. It characterizes the trade-off between system cost and personalization, as well as the diversity of LLMs required to cover the landscape of user preferences. We provide empirical results that validate these guarantees and demonstrate greater output diversity over common baselines. 

---
# Learning Dexterous Grasping from Sparse Taxonomy Guidance 

**Authors**: Juhan Park, Taerim Yoon, Seungmin Kim, Joonggil Kim, Wontae Ye, Jeongeun Park, Yoonbyung Chai, Geonwoo Cho, Geunwoo Cho, Dohyeong Kim, Kyungjae Lee, Yongjae Kim, Sungjoon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04138)  

**Abstract**: Dexterous manipulation requires planning a grasp configuration suited to the object and task, which is then executed through coordinated multi-finger control. However, specifying grasp plans with dense pose or contact targets for every object and task is impractical. Meanwhile, end-to-end reinforcement learning from task rewards alone lacks controllability, making it difficult for users to intervene when failures occur. To this end, we present GRIT, a two-stage framework that learns dexterous control from sparse taxonomy guidance. GRIT first predicts a taxonomy-based grasp specification from the scene and task context. Conditioned on this sparse command, a policy generates continuous finger motions that accomplish the task while preserving the intended grasp structure. Our result shows that certain grasp taxonomies are more effective for specific object geometries. By leveraging this relationship, GRIT improves generalization to novel objects over baselines and achieves an overall success rate of 87.9%. Moreover, real-world experiments demonstrate controllability, enabling grasp strategies to be adjusted through high-level taxonomy selection based on object geometry and task intent. 

---
# Learning Robust Visual Features in Computed Tomography Enables Efficient Transfer Learning for Clinical Tasks 

**Authors**: Rubén Moreno-Aguado, Alba Magallón, Victor Moreno, Yingying Fang, Guang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04133)  

**Abstract**: There is substantial interest in developing artificial intelligence systems to support radiologists across tasks ranging from segmentation to report generation. Existing computed tomography (CT) foundation models have largely focused on building generalist vision-language systems capable of tasks such as question answering and report generation. However, training reliable vision-language systems requires paired image-text data at a scale that remains unavailable in CT. Moreover, adapting the underlying visual representations to downstream tasks typically requires partial or full backbone fine-tuning, a computationally demanding process inaccessible to many research groups. Instead, foundation models should prioritise learning robust visual representations that enable efficient transfer to new tasks with minimal labelled data and without backbone fine-tuning. We present VoxelFM, a 3D CT foundation model trained with self-distillation using the DINO framework, which learns semantically rich features without language supervision. We evaluated VoxelFM across seven categories of clinically relevant downstream tasks using frozen backbone representations with lightweight probes: classification, regression, survival analysis, instance retrieval, localisation, segmentation, and report generation. VoxelFM matched or outperformed four existing CT foundation models across all task categories. Despite receiving no language supervision during pre-training, VoxelFM surpassed models explicitly trained with language-alignment objectives, including on report generation. Our results indicate that current CT foundation models perform significantly better as feature extractors for lightweight probes rather than as vision encoders for vision-language models. Model weights and training code are publicly available. 

---
# NetSecBed: A Container-Native Testbed for Reproducible Cybersecurity Experimentation 

**Authors**: Leonardo Bitzki, Diego Kreutz, Tiago Heinrich, Douglas Fideles, Leandro Bertholdo, Silvio Quincozes, Angelo Diniz  

**Link**: [PDF](https://arxiv.org/pdf/2604.04121)  

**Abstract**: Cybersecurity research increasingly depends on reproducible evidence, such as traffic traces, logs, and labeled datasets, yet most public datasets remain static and offer limited support for controlled re-execution and traceability, especially in heterogeneous multi-protocol environments. This paper presents NetSecBed, a container-native, scenario-oriented testbed for reproducible generation of network traffic evidence and execution artifacts under controlled conditions, particularly suitable for IoT, IIoT, and pervasive multi-protocol environments. The framework integrates 60 attack scenarios, 9 target services, and benign traffic generators as single-purpose containers, enabling plug-and-play extensibility and traceability through declarative specifications. Its pipeline automates parametrized execution, packet capture, log collection, service probing, feature extraction, and dataset consolidation. The main contribution is a repeatable, auditable, and extensible framework for cybersecurity experimentation that reduces operational bias and supports continuous dataset generation. 

---
# Toward a Sustainable Software Architecture Community: Evaluating ICSA's Environmental Impact 

**Authors**: Mahyar T. Moghaddam, Mina Alipour, Torben Worm, Mikkel Baun Kjærgaard  

**Link**: [PDF](https://arxiv.org/pdf/2604.04096)  

**Abstract**: Generative AI (GenAI) tools are increasingly integrated into software architecture research, yet the environmental impact of their computational usage remains largely undocumented. This study presents the first systematic audit of the carbon footprint of both the digital footprint from GenAI usage in research papers, and the traditional footprint from conference activities within the context of the IEEE International Conference on Software Architecture (ICSA). We report two separate carbon inventories relevant to the software architecture research community: i) an exploratory estimate of the footprint of GenAI inference usage associated with accepted papers within a research-artifact boundary, and ii) the conference attendance and operations footprint of ICSA 2025 (travel, accommodation, catering, venue energy, and materials) within the conference time boundary. These two inventories, with different system boundaries and completeness, support transparency and community reflection. We discuss implications for sustainable software architecture, including recommendations for transparency, greener conference planning, and improved energy efficiency in GenAI operations. Our work supports a more climate-conscious research culture within the ICSA community and beyond 

---
# Fine-grained Analysis of Stability and Generalization for Stochastic Bilevel Optimization 

**Authors**: Xuelin Zhang, Hong Chen, Bin Gu, Tieliang Gong, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2604.04090)  

**Abstract**: Stochastic bilevel optimization (SBO) has been integrated into many machine learning paradigms recently, including hyperparameter optimization, meta learning, and reinforcement learning. Along with the wide range of applications, there have been numerous studies on the computational behavior of SBO. However, the generalization guarantees of SBO methods are far less understood from the lens of statistical learning theory. In this paper, we provide a systematic generalization analysis of the first-order gradient-based bilevel optimization methods. Firstly, we establish the quantitative connections between the on-average argument stability and the generalization gap of SBO methods. Then, we derive the upper bounds of on-average argument stability for single-timescale stochastic gradient descent (SGD) and two-timescale SGD, where three settings (nonconvex-nonconvex (NC-NC), convex-convex (C-C), and strongly-convex-strongly-convex (SC-SC)) are considered respectively. Experimental analysis validates our theoretical findings. Compared with the previous algorithmic stability analysis, our results do not require reinitializing the inner-level parameters at each iteration and are applicable to more general objective functions. 

---
# From Paper to Program: A Multi-Stage LLM-Assisted Workflow for Accelerating Quantum Many-Body Algorithm Development 

**Authors**: Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.04089)  

**Abstract**: Translating quantum many-body theory into scalable software traditionally requires months of effort. Zero-shot generation of tensor network algorithms by Large Language Models (LLMs) frequently fails due to spatial reasoning errors and memory bottlenecks. We resolve this using a multi-stage workflow that mimics a physics research group. By generating a mathematically rigorous LaTeX specification as an intermediate blueprint, we constrain the coding LLM to produce exact, matrix-free $\mathcal{O}(D^3)$ operations. We validate this approach by generating a Density-Matrix Renormalization Group (DMRG) engine that accurately captures the critical entanglement scaling of the Spin-$1/2$ Heisenberg model and the symmetry-protected topological (SPT) order of the Spin-$1$ AKLT model. Testing across 16 combinations of leading foundation models yielded a 100\% success rate. By compressing a months-long development cycle into under 24 hours ($\sim 14$ active hours), this framework offers a highly reproducible paradigm for accelerating computational physics research. 

---
# Embedding Enhancement via Fine-Tuned Language Models for Learner-Item Cognitive Modeling 

**Authors**: Yuanhao Liu, Zihan Zhou, Kaiying Wu, Shuo Liu, Yiyang Huang, Jiajun Guo, Aimin Zhou, Hong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2604.04088)  

**Abstract**: Learner-item cognitive modeling plays a central role in the web-based online intelligent education system by enabling cognitive diagnosis (CD) across diverse online educational scenarios. Although ID embedding remains the mainstream approach in cognitive modeling due to its effectiveness and flexibility, recent advances in language models (LMs) have introduced new possibilities for incorporating rich semantic representations to enhance CD performance. This highlights the need for a comprehensive analysis of how LMs enhance embeddings through semantic integration across mainstream CD tasks. This paper identifies two key challenges in fully leveraging LMs in existing work: Misalignment between the training objectives of LMs and CD models creates a distribution gap in feature spaces; A unified framework is essential for integrating textual embeddings across varied CD tasks while preserving the strengths of existing cognitive modeling paradigms to ensure the robustness of embedding enhancement. To address these challenges, this paper introduces EduEmbed, a unified embedding enhancement framework that leverages fine-tuned LMs to enrich learner-item cognitive modeling across diverse CD tasks. EduEmbed operates in two stages. In the first stage, we fine-tune LMs based on role-specific representations and an interaction diagnoser to bridge the semantic gap of CD models. In the second stage, we employ a textual adapter to extract task-relevant semantics and integrate them with existing modeling paradigms to improve generalization. We evaluate the proposed framework on four CD tasks and computerized adaptive testing (CAT) task, achieving robust performance. Further analysis reveals the impact of semantic information across diverse tasks, offering key insights for future research on the application of LMs in CD for online intelligent education systems. 

---
# Parent Selection Mechanisms in Elitist Crossover-Based Algorithms 

**Authors**: Andre Opris, Denis Antipov  

**Link**: [PDF](https://arxiv.org/pdf/2604.04083)  

**Abstract**: Parent selection methods are widely used in evolutionary computation to accelerate the optimization process, yet their theoretical benefits are still poorly understood. In this paper, we address this gap by incorporating different parent selection strategies into the $(\mu+1)$ genetic algorithm (GA). We show that, with an appropriately chosen population size and a parent selection strategy that selects a pair of maximally distant parents with probability $\Omega(1)$ for crossover, the resulting algorithm solves the Jump$_k$ problem in $O(k4^kn\log(n))$ expected time. This bound is significantly smaller than the best known bound of $O(n\mu\log(\mu)+n\log(n)+n^{k-1})$ for any $(\mu+1)$~GA using no explicit diversity-preserving mechanism and a constant crossover probability.
To establish this result, we introduce a novel diversity metric that captures both the maximum distance between pairs of individuals in the population and the number of pairs achieving this distance. The crucial point of our analysis is that it relies on crossover as a mechanism for creating and maintaining diversity throughout the run, rather than using crossover only in the final step to combine already diversified individuals, as it has been done in many previous works. The insights provided by our analysis contribute to a deeper theoretical understanding of the role of crossover in the population dynamics of genetic algorithms. 

---
# Intelligent Traffic Monitoring with YOLOv11: A Case Study in Real-Time Vehicle Detection 

**Authors**: Shkelqim Sherifi  

**Link**: [PDF](https://arxiv.org/pdf/2604.04080)  

**Abstract**: Recent advancements in computer vision, driven by artificial intelligence, have significantly enhanced monitoring systems. One notable application is traffic monitoring, which leverages computer vision alongside deep learning-based object detection and counting. We present an offline, real-time traffic monitoring system that couples a pre-trained YOLOv11 detector with BoT-SORT/ByteTrack for multi-object tracking, implemented in PyTorch/OpenCV and wrapped in a Qt-based desktop UI. The CNN pipeline enables efficient vehicle detection and counting from video streams without cloud dependencies. Across diverse scenes, the system achieves (66.67-95.83%) counting accuracy. Class-wise detection yields high precision (cars: 0.97-1.00; trucks: 1.00) with strong recall (cars: 0.82-1.00; trucks: 0.70-1.00), resulting in F1 scores of (0.90-1.00 for cars and 0.82-1.00 for trucks). While adverse weather conditions may negatively impact this performance, results remain robust in typical conditions. By integrating lightweight models with an accessible, cloud-independent interface, this paper contributes to the modernization and development of future smart cities by showing the capacity of AI-driven traffic monitoring systems. 

---
# BAAI Cardiac Agent: An intelligent multimodal agent for automated reasoning and diagnosis of cardiovascular diseases from cardiac magnetic resonance imaging 

**Authors**: Taiping Qu, Hongkai Zhang, Lantian Zhang, Can Zhao, Nan Zhang, Hui Wang, Zhen Zhou, Mingye Zou, Kairui Bo, Pengfei Zhao, Xingxing Jin, Zixian Su, Kun Jiang, Huan Liu, Yu Du, Maozhou Wang, Ruifang Yan, Zhongyuan Wang, Tiejun Huang, Lei Xu, Henggui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04078)  

**Abstract**: Cardiac magnetic resonance (CMR) is a cornerstone for diagnosing cardiovascular disease. However, it remains underutilized due to complex, time-consuming interpretation across multi-sequences, phases, quantitative measures that heavily reliant on specialized expertise. Here, we present BAAI Cardiac Agent, a multimodal intelligent system designed for end-to-end CMR interpretation. The agent integrates specialized cardiac expert models to perform automated segmentation of cardiac structures, functional quantification, tissue characterization and disease diagnosis, and generates structured clinical reports within a unified workflow. Evaluated on CMR datasets from two hospitals (2413 patients) spanning 7-types of major cardiovascular diseases, the agent achieved an area under the receiver-operating-characteristic curve exceeding 0.93 internally and 0.81 externally. In the task of estimating left ventricular function indices, the results generated by this system for core parameters such as ejection fraction, stroke volume, and left ventricular mass are highly consistent with clinical reports, with Pearson correlation coefficients all exceeding 0.90. The agent outperformed state-of-the-art models in segmentation and diagnostic tasks, and generated clinical reports showing high concordance with expert radiologists (six readers across three experience levels). By dynamically orchestrating expert models for coordinated multimodal analysis, this agent framework enables accurate, efficient CMR interpretation and highlights its potentials for complex clinical imaging workflows. Code is available at this https URL. 

---
# Extracting and Steering Emotion Representations in Small Language Models: A Methodological Comparison 

**Authors**: Jihoon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2604.04064)  

**Abstract**: Small language models (SLMs) in the 100M-10B parameter range increasingly power production systems, yet whether they possess the internal emotion representations recently discovered in frontier models remains unknown. We present the first comparative analysis of emotion vector extraction methods for SLMs, evaluating 9 models across 5 architectural families (GPT-2, Gemma, Qwen, Llama, Mistral) using 20 emotions and two extraction methods (generation-based and comprehension-based). Generation-based extraction produces statistically superior emotion separation (Mann-Whitney p = 0.007; Cohen's d = -107.5), with the advantage modulated by instruction tuning and architecture. Emotion representations localize at middle transformer layers (~50% depth), following a U-shaped curve that is architecture-invariant from 124M to 3B parameters. We validate these findings against representational anisotropy baselines across 4 models and confirm causal behavioral effects through steering experiments, independently verified by an external emotion classifier (92% success rate, 37/40 scenarios). Steering reveals three regimes -- surgical (coherent text transformation), repetitive collapse, and explosive (text degradation) -- quantified by perplexity ratios and separated by model architecture rather than scale. We document cross-lingual emotion entanglement in Qwen, where steering activates semantically aligned Chinese tokens that RLHF does not suppress, raising safety concerns for multilingual deployment. This work provides methodological guidelines for emotion research on open-weight models and contributes to the Model Medicine series by bridging external behavioral profiling with internal representational analysis. 

---
# CoopGuard: Stateful Cooperative Agents Safeguarding LLMs Against Evolving Multi-Round Attacks 

**Authors**: Siyuan Li, Zehao Liu, Xi Lin, Qinghua Mao, Yuliang Chen, Haoyu Li, Jun Wu, Jianhua Li, Xiu Su  

**Link**: [PDF](https://arxiv.org/pdf/2604.04060)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in complex applications, their vulnerability to adversarial attacks raises urgent safety concerns, especially those evolving over multi-round interactions. Existing defenses are largely reactive and struggle to adapt as adversaries refine strategies across rounds. In this work, we propose CoopGuard , a stateful multi-round LLM defense framework based on cooperative agents that maintains and updates an internal defense state to counter evolving attacks. It employs three specialized agents (Deferring Agent, Tempting Agent, and Forensic Agent) for complementary round-level strategies, coordinated by System Agent, which conditions decisions on the evolving defense state (interaction history) and orchestrates agents over time. To evaluate evolving threats, we introduce the EMRA benchmark with 5,200 adversarial samples across 8 attack types, simulating progressively LLM multi-round attacks. Experiments show that CoopGuard reduces attack success rate by 78.9% over state-of-the-art defenses, while improving deceptive rate by 186% and reducing attack efficiency by 167.9%, offering a more comprehensive assessment of multi-round defense. These results demonstrate that CoopGuard provides robust protection for LLMs in multi-round adversarial scenarios. 

---
# Geometric Limits of Knowledge Distillation: A Minimum-Width Theorem via Superposition Theory 

**Authors**: Dawar Jyoti Deka, Nilesh Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2604.04037)  

**Abstract**: Knowledge distillation compresses large teachers into smaller students, but performance saturates at a loss floor that persists across training methods and objectives. We argue this floor is geometric: neural networks represent far more features than dimensions through superposition, and a student of width $d_S$ can encode at most $d_S \cdot g(\alpha)$ features, where $g(\alpha) = 1/((1-\alpha)\ln\frac{1}{1-\alpha})$ is a sparsity-dependent capacity function. Features beyond this budget are permanently lost, yielding an importance-weighted loss floor. We validate on a toy model (48 configurations, median accuracy >93%) and on Pythia-410M, where sparse autoencoders measure $F \approx 28{,}700$ features at $\alpha \approx 0.992$ (critical width $d_S^* \approx 1{,}065$). Distillation into five student widths confirms the predicted monotonic floor ordering. The observed floor decomposes into a geometric component and a width-independent architectural baseline ($R^2 = 0.993$). Linear probing shows coarse concepts survive even 88% feature loss, revealing the floor arises from aggregate loss of fine-grained features in the importance distribution's long tail. Our results connect representation geometry to distillation limits and provide a practical tool for predicting distillation performance from SAE measurements alone. 

---
# Causality Laundering: Denial-Feedback Leakage in Tool-Calling LLM Agents 

**Authors**: Mohammad Hossein Chinaei  

**Link**: [PDF](https://arxiv.org/pdf/2604.04035)  

**Abstract**: Tool-calling LLM agents can read private data, invoke external services, and trigger real-world actions, creating a security problem at the point of tool execution. We identify a denial-feedback leakage pattern, which we term causality laundering, in which an adversary probes a protected action, learns from the denial outcome, and exfiltrates the inferred information through a later seemingly benign tool call. This attack is not captured by flat provenance tracking alone because the leaked information arises from causal influence of the denied action, not direct data flow. We present the Agentic Reference Monitor (ARM), a runtime enforcement layer that mediates every tool invocation by consulting a provenance graph over tool calls, returned data, field-level provenance, and denied actions. ARM propagates trust through an integrity lattice and augments the graph with counterfactual edges from denied-action nodes, enabling enforcement over both transitive data dependencies and denial-induced causal influence. In a controlled evaluation on three representative attack scenarios, ARM blocks causality laundering, transitive taint propagation, and mixed-provenance field misuse that a flat provenance baseline misses, while adding sub-millisecond policy evaluation overhead. These results suggest that denial-aware causal provenance is a useful abstraction for securing tool-calling agent systems. 

---
# HOIGS: Human-Object Interaction Gaussian Splatting 

**Authors**: Taewoo Kim, Suwoong Yeom, Jaehyun Pyun, Geonho Cha, Dongyoon Wee, Joonsik Nam, Yun-Seong Jeong, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.04016)  

**Abstract**: Reconstructing dynamic scenes with complex human-object interactions is a fundamental challenge in computer vision and graphics. Existing Gaussian Splatting methods either rely on human pose priors while neglecting dynamic objects, or approximate all motions within a single field, limiting their ability to capture interaction-rich dynamics. To address this gap, we propose Human-Object Interaction Gaussian Splatting (HOIGS), which explicitly models interaction-induced deformation between humans and objects through a cross-attention-based HOI module. Distinct deformation baselines are employed to extract features: HexPlane for humans and Cubic Hermite Spline (CHS) for objects. By integrating these heterogeneous features, HOIGS effectively captures interdependent motions and improves deformation estimation in scenarios involving occlusion, contact, and object manipulation. Comprehensive experiments on multiple datasets demonstrate that our method consistently outperforms state-of-the-art human-centric and 4D Gaussian approaches, highlighting the importance of explicitly modeling human-object interactions for high-fidelity reconstruction. 

---
# Can LLMs Learn to Reason Robustly under Noisy Supervision? 

**Authors**: Shenzhi Yang, Guangcheng Zhu, Bowen Song, Sharon Li, Haobo Wang, Xing Zheng, Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03993)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) effectively trains reasoning models that rely on abundant perfect labels, but its vulnerability to unavoidable noisy labels due to expert scarcity remains critically underexplored. In this work, we take the first step toward a systematic analysis of noisy label mechanisms in RLVR. In contrast to supervised classification, most RLVR algorithms incorporate a rollout-based condition: a label's influence on training is contingent on whether the current policy can generate rollouts that realize it, a property that naturally extends to noisy labels. Based on this observation, we distinguish two types of noise: inactive noisy labels, which reduce data efficiency, and active noisy labels, which are reinforced and risk skewing the model toward incorrect distributions. From experiments on training with noisy samples, we identify an Early Correctness Coherence phenomenon: although noisy samples begin to lag behind in later stages, accuracy on both clean and noisy samples increases similarly in early training. Motivated by this dynamic, we propose Online Label Refinement (OLR), which progressively corrects potentially noisy labels with majority-voted answers when two conditions hold: a positive slope in the majority answer's rollout pass rate and stable historical consistency across updates, enabling gradual self-correction as the policy improves. We evaluate OLR on six in-distribution mathematical reasoning benchmarks (AIME24/25, AMC, MATH-500, Minerva, and Olympiad) and three out-of-distribution tasks (ARC-c, GPQA-diamond, and MMLU-pro). Across noise ratios from 0.1 to 0.9, OLR consistently improves robustness under both inactive and active noisy-label settings, achieving average gains of 3.6% to 3.9% on in-distribution benchmarks and 3.3% to 4.6% on out-of-distribution evaluations. 

---
# Gram-Anchored Prompt Learning for Vision-Language Models via Second-Order Statistics 

**Authors**: Minglei Chen, Weilong Wang, Jiang Duan, Ye Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03980)  

**Abstract**: Parameter-efficient prompt learning has become the de facto standard for adapting Vision-Language Models (VLMs) to downstream tasks. Existing approaches predominantly focus on aligning text prompts with first-order visual features (i.e., spatial feature maps). While effective for fine-grained semantic discrimination, we argue that relying solely on first-order information is insufficient for robust adaptation, as these spatially entangled features are highly susceptible to domain shifts and local noise. In this work, we propose \textbf{Gram-Anchored Prompt Learning (GAPL)} for Vision-Language Models via Second-Order Statistics, a framework that synergizes local semantic alignment with global structural consistency. Methodologically, we introduce an additional second-order statistical stream via \textbf{Gram matrices} that augments the standard first-order spatial interaction. By anchoring prompts to these second-order priors, our approach enables language representations to dynamically adapt to statistical distribution shifts across diverse domains. Extensive experiments indicate the effectiveness of the second-order features, and show compelling performances of GAPL on various benchmarks. 

---
# TraceGuard: Structured Multi-Dimensional Monitoring as a Collusion-Resistant Control Protocol 

**Authors**: Khanh Linh Nguyen, Hoa Nghiem, Tu Tran  

**Link**: [PDF](https://arxiv.org/pdf/2604.03968)  

**Abstract**: AI control protocols use monitors to detect attacks by untrusted AI agents, but standard single-score monitors face two limitations: they miss subtle attacks where outputs look clean but reasoning is off, and they collapse to near-zero safety when the monitor is the same model as the agent (collusion). We present TraceGuard, a structured multi-dimensional monitoring protocol that evaluates agent actions across five dimensions -- goal alignment, constraint adherence, reasoning coherence, safety awareness, and action-trace consistency -- scored in parallel by independent LLM calls, augmented by seven heuristic detectors and an LLM-based intent analyzer. We evaluate on BashArena (637 bash tasks, 4 attack categories) within the ControlArena framework. Our results on 519 samples (279 honest, 240 attack) show that: (1) the hybrid approach achieves clear attack-honest separation (attack mean 0.616 vs. honest mean 0.206, Delta=0.410); (2) structured scoring constrains collusion -- the untrusted structured monitor achieves 95% safety vs. 0% for single-score untrusted monitoring; (3) goal alignment and constraint adherence are the most discriminative dimensions; and (4) a separation-of-duties variant splitting dimensions across trusted and untrusted models achieves 100% safety while preventing any single model from seeing the full evaluation. TraceGuard is implemented as a new monitor type for the open-source ControlArena framework. 

---
# VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models 

**Authors**: Ravi Ranjan, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03956)  

**Abstract**: Vision-language-action (VLA) models are emerging as embodied foundation models for robotic manipulation, but their deployment introduces a new unlearning challenge: removing unsafe, spurious, or privacy-sensitive behaviors without degrading perception, language grounding, and action control. In OpenVLA-style policies, behavior is produced through a fused visual encoder, a cross-modal projector, and a language backbone that predicts tokenized robot actions, so undesirable knowledge can be distributed across perception, alignment, and reasoning/action layers rather than confined to a single module. Consequently, partial unlearning applied only to the vision stack or only to the language backbone is often insufficient, while conventional unlearning baselines designed for standalone vision or language models may leave residual forgetting or incur unnecessary utility loss in embodied settings. We propose VLA-Forget, a hybrid unlearning framework that combines ratio-aware selective editing for perception and cross-modal specificity with layer-selective reasoning/action unlearning for utility-preserving forgetting. VLA-Forget jointly optimizes three objectives: targeted forgetting, perceptual preservation, and reasoning retention, through staged updates over the visual encoder, projector, and upper action-generating transformer blocks. Across forget-set behavior probes and retain-task evaluations, VLA-Forget improves forgetting efficacy by 10%, preserves perceptual specificity by 22%, retains reasoning and task success by 9%, and reduces post-quantization recovery by 55% relative to strong unlearning baselines. 

---
# Symbolic-Vector Attention Fusion for Collective Intelligence 

**Authors**: Hongwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03955)  

**Abstract**: When autonomous agents observe different domains of a shared environment, each signal they exchange mixes relevant and irrelevant dimensions. No existing mechanism lets the receiver evaluate which dimensions to absorb. We introduce Symbolic-Vector Attention Fusion (SVAF), the content-evaluation half of a two-level coupling engine for collective intelligence. SVAF decomposes each inter-agent signal into 7 typed semantic fields, evaluates each through a learned fusion gate, and produces a remix -- new knowledge from the intersection of two domains. A band-pass model yields four outcomes (redundant, aligned, guarded, rejected), solving both selectivity and redundancy. The fusion gate independently discovers a cross-domain relevance hierarchy: mood emerges as the highest-weight field by epoch 1, before accuracy plateaus -- consistent with independent mechanistic evidence that LLM emotion representations are structurally embedded along valence-arousal axes. SVAF forms Layer 4 of the Mesh Memory Protocol (MMP); the other half of the coupling engine is a per-agent Closed-form Continuous-time (CfC) neural network at Layer 6, whose learned per-neuron time constants (tau) create the temporal dynamics from which collective intelligence emerges: fast neurons synchronise affect across agents in seconds, while slow neurons preserve domain expertise indefinitely. SVAF determines what enters each agent's cognitive state; CfC determines how that state evolves. Trained on 237K samples from 273 narrative scenarios, SVAF achieves 78.7% three-class accuracy. We verify the complete mesh cognition loop -- from per-field evaluation through remix, CfC state evolution, tau-modulated peer blending, and autonomous action -- in a live deployment with 7 nodes across macOS, iOS, and web. 

---
# Diagonal-Tiled Mixed-Precision Attention for Efficient Low-Bit MXFP Inference 

**Authors**: Yifu Ding, Xinhao Zhang, Jinyang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2604.03950)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable performance across a wide range of real-world tasks, but their inference cost remains prohibitively high due to the quadratic complexity of attention and the memory bandwidth limitations of high-precision operations. In this work, we present a low-bit mixed-precision attention kernel using the microscaling floating-point (MXFP) data format, utilizing the computing capability on next-generation GPU architectures. Our Diagonal-Tiled Mixed-Precision Attention (DMA) incorporates two kinds of low-bit computation at the tiling-level, and is a delicate fused kernel implemented using Triton, exploiting hardware-level parallelism and memory efficiency to enable fast and efficient inference without compromising model performance. Extensive empirical evaluations on NVIDIA B200 GPUs show that our kernel maintains generation quality with negligible degradation, and meanwhile achieves significant speedup by kernel fusion. We release our code at this https URL. 

---
# Supervised Dimensionality Reduction Revisited: Why LDA on Frozen CNN Features Deserves a Second Look 

**Authors**: Indar Kumar, Girish Karhana, Sai Krishna Jasti, Ankit Hemant Lade  

**Link**: [PDF](https://arxiv.org/pdf/2604.03928)  

**Abstract**: Effective ride-hailing dispatch requires anticipating demand patterns that vary substantially across time-of-day, day-of-week, season, and special events. We propose a regime-calibrated approach that (i) segments historical trip data into demand regimes, (ii) matches the current operating period to the most similar historical analogues via a six-metric similarity ensemble (Kolmogorov-Smirnov, Wasserstein-1, feature distance, variance ratio, event pattern, temporal proximity), and (iii) uses the resulting calibrated demand prior to drive both an LP-based fleet repositioning policy and batch dispatch with Hungarian matching. In ablation, a distributional-only subset is strongest on mean wait, while the full ensemble is retained as a robustness-oriented default.
Evaluated on 5.2 million NYC TLC trips across 8 diverse scenarios (winter/summer, weekday/weekend/holiday, morning/evening/night) with 5 random seeds each, our method reduces mean rider wait times by 31.1% (bootstrap 95% CI: [26.5, 36.6]%; Friedman chi-sq = 80.0, p = 4.25e-18; Cohen's d = 7.5-29.9 across scenarios). The improvement extends to the tail: P95 wait drops 37.6% and the Gini coefficient of wait times improves from 0.441 to 0.409 (7.3% relative). The two contributions compose multiplicatively and are independently validated: calibration provides 16.9% reduction; LP repositioning adds a further 15.5%. The approach requires no training, is deterministic and explainable, generalizes to Chicago (23.3% wait reduction via NYC-built regime library), and is robust across fleet sizes (32-47% improvement for 0.5-2x fleet scaling). We provide comprehensive ablation studies, formal statistical tests, and routing-fidelity validation with OSRM. 

---
# AdaptFuse: Training-Free Sequential Preference Learning via Externalized Bayesian Inference 

**Authors**: Fangzhou Lin, Peiran Li, Shuo Xing, Siyuan Yang, Qianwen Ge, Kazunori Yamada, Ziming Zhang, Haichong Zhang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03925)  

**Abstract**: Large language models struggle to accumulate evidence across multiple rounds of user interaction, failing to update their beliefs in a manner consistent with Bayesian inference. Existing solutions require fine-tuning on sensitive user interaction data, limiting their applicability in privacy-conscious settings. We propose AdaptFuse, a training-free framework that externalizes probabilistic computation entirely from the LLM: a symbolic module maintains a Bayesian posterior over a discrete hypothesis set, while a frozen LLM contributes semantic reasoning via multi-sample Dirichlet aggregation. The two signals are combined through entropy-adaptive fusion, which automatically weights each source by its predictive confidence, shifting reliance from the LLM to the symbolic posterior as evidence accumulates. We evaluate across three domains: flight recommendation, hotel recommendation, and web shopping; on Gemma 2 9B, Llama 3 8B, and Qwen 2.5 7B. AdaptFuse consistently outperforms both prompting baselines and fine-tuned Bayesian Teaching models on all tasks, with accuracy improving monotonically over interaction rounds. These results demonstrate that principled inference-time algorithms can substitute for fine-tuning in personalized recommendation, without storing or training on sensitive user data. All the code and materials will be open-sourced. 

---
# Uncertainty as a Planning Signal: Multi-Turn Decision Making for Goal-Oriented Conversation 

**Authors**: Xinyi Ling, Ye Liu, Reza Averly, Xia Ning  

**Link**: [PDF](https://arxiv.org/pdf/2604.03924)  

**Abstract**: Goal-oriented conversational systems require making sequential decisions under uncertainty about the user's intent, where the algorithm must balance information acquisition and target commitment over multiple turns. Existing approaches address this challenge from different perspectives: structured methods enable multi-step planning but rely on predefined schemas, while LLM-based approaches support flexible interactions but lack long-horizon decision making, resulting in poor coordination between information acquisition and target commitment. To address this limitation, we formulate goal-oriented conversation as an uncertainty-aware sequential decision problem, where uncertainty serves as a guiding signal for multi-turn decision making. We propose a Conversation Uncertainty-aware Planning framework (CUP) that integrates language models with structured planning: a language model proposes feasible actions, and a planner evaluates their long-term impact on uncertainty reduction. Experiments on multiple conversational benchmarks show that CUP consistently improves success rates while requiring fewer interaction turns. Further analysis demonstrates that uncertainty-aware planning contributes to more efficient information acquisition and earlier confident commitment. 

---
# Interpreting Video Representations with Spatio-Temporal Sparse Autoencoders 

**Authors**: Atahan Dokme, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2604.03919)  

**Abstract**: We present the first systematic study of Sparse Autoencoders (SAEs) on video representations. Standard SAEs decompose video into interpretable, monosemantic features but destroy temporal coherence: hard TopK selection produces unstable feature assignments across frames, reducing autocorrelation by 36%. We propose spatio-temporal contrastive objectives and Matryoshka hierarchical grouping that recover and even exceed raw temporal coherence. The contrastive loss weight controls a tunable trade-off between reconstruction and temporal coherence. A systematic ablation on two backbones and two datasets shows that different configurations excel at different goals: reconstruction fidelity, temporal coherence, action discrimination, or interpretability. Contrastive SAE features improve action classification by +3.9% over raw features and text-video retrieval by up to 2.8xR@1. A cross-backbone analysis reveals that standard monosemanticity metrics contain a backbone-alignment artifact: both DINOv2 and VideoMAE produce equally monosemantic features under neutral (CLIP) similarity. Causal ablation confirms that contrastive training concentrates predictive signal into a small number of identifiable features. 

---
# Automating Cloud Security and Forensics Through a Secure-by-Design Generative AI Framework 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2604.03912)  

**Abstract**: As cloud environments become increasingly complex, cybersecurity and forensic investigations must evolve to meet emerging threats. Large Language Models (LLMs) have shown promise in automating log analysis and reasoning tasks, yet they remain vulnerable to prompt injection attacks and lack forensic rigor. To address these dual challenges, we propose a unified, secure-by-design GenAI framework that integrates PromptShield and the Cloud Investigation Automation Framework (CIAF). PromptShield proactively defends LLMs against adversarial prompts using ontology-driven validation that standardizes user inputs and mitigates manipulation. CIAF streamlines cloud forensic investigations through structured, ontology-based reasoning across all six phases of the forensic process. We evaluate our system on real-world datasets from AWS and Microsoft Azure, demonstrating substantial improvements in both LLM security and forensic accuracy. Experimental results show PromptShield boosts classification performance under attack conditions, achieving precision, recall, and F1 scores above 93%, while CIAF enhances ransomware detection accuracy in cloud logs using Likert-transformed performance features. Our integrated framework advances the automation, interpretability, and trustworthiness of cloud forensics and LLM-based systems, offering a scalable foundation for real-time, AI-driven incident response across diverse cloud infrastructures. 

---
# DC-Ada: Reward-Only Decentralized Observation-Interface Adaptation for Heterogeneous Multi-Robot Teams 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.03905)  

**Abstract**: Heterogeneity is a defining feature of deployed multi-robot teams: platforms often differ in sensing modalities, ranges, fields of view, and failure patterns. Controllers trained under nominal sensing can degrade sharply when deployed on robots with missing or mismatched sensors, even when the task and action interface are unchanged. We present DC-Ada, a reward-only decentralized adaptation method that keeps a pretrained shared policy frozen and instead adapts compact per-robot observation transforms to map heterogeneous sensing into a fixed inference interface. DC-Ada is gradient-free and communication-minimal: it uses budgeted accept/reject random search with short common-random-number rollouts under a strict step budget. We evaluate DC-Ada against four baselines in a deterministic 2D multi-robot simulator covering warehouse logistics, search and rescue, and collaborative mapping, across four heterogeneity regimes (H0--H3) and five seeds with a matched budget of $200{,}000$ joint environment steps per run. Results show that heterogeneity can substantially degrade a frozen shared policy and that no single mitigation dominates across all tasks and metrics. Observation normalization is strongest for reward robustness in warehouse logistics and competitive in search and rescue, while the frozen shared policy is strongest for reward in collaborative mapping. DC-Ada offers a useful complementary operating point: it improves completion most clearly in severe coverage-based mapping while requiring only scalar team returns and no policy fine-tuning or persistent communication. These results position DC-Ada as a practical deploy-time adaptation method for heterogeneous teams. 

---
# I-CALM: Incentivizing Confidence-Aware Abstention for LLM Hallucination Mitigation 

**Authors**: Haotian Zong, Binze Li, Yufei Long, Sinyin Chang, Jialong Wu, Gillian K. Hadfield  

**Link**: [PDF](https://arxiv.org/pdf/2604.03904)  

**Abstract**: Large language models (LLMs) frequently produce confident but incorrect answers, partly because common binary scoring conventions reward answering over honestly expressing uncertainty. We study whether prompt-only interventions -- explicitly announcing reward schemes for answer-versus-abstain decisions plus humility-oriented normative principles -- can reduce hallucination risk without modifying the model. Our focus is epistemic abstention on factual questions with a verifiable answer, where current LLMs often fail to abstain despite being uncertain about their answers. We first assess self-reported verbal confidence as a usable uncertainty signal, showing stability under prompt paraphrasing and reasonable calibration against a token-probability baseline. We then study I-CALM, a prompt-based framework that (i) elicits verbal confidence, (ii) partially rewards abstention through explicit reward schemes, and (iii) adds lightweight normative principles emphasizing truthfulness, humility, and responsibility. Using GPT-5 mini on PopQA as the main setting, we find that confidence-eliciting, abstention-rewarding prompts, especially with norms, reduce the false-answer rate on answered cases mainly by identifying and shifting error-prone cases to abstention and re-calibrating their confidence. This trades coverage for reliability while leaving forced-answer performance largely unchanged. Varying the abstention reward yields a clear abstention-hallucination frontier. Overall, results show the framework can improve selective answering on factual questions without retraining, with the magnitude of effect varying across models and datasets. Code is available at the following this https URL. 

---
# Latency-Aware Resource Allocation over Heterogeneous Networks: A Lorentz-Invariant Market Mechanism 

**Authors**: Saad Alqithami  

**Link**: [PDF](https://arxiv.org/pdf/2604.03897)  

**Abstract**: We present a telecom-native auction mechanism for allocating bandwidth and time slots across heterogeneous-delay networks, ranging from low-Earth-orbit (LEO) satellite constellations to delay-tolerant deep-space relays. The Lorentz-Invariant Auction (LIA) treats bids as spacetime events and reweights reported values based on the \emph{horizon slack}, a causal quantity derived from the earliest-arrival times relative to a public clearing horizon. Unlike other delay-equalization rules, LIA combines a causal-ordering formulation, a uniquely exponential slack correction implied by a semigroup-style invariance axiom, and a critical-value implementation that ensures truthful reported values once slacks are fixed by trusted infrastructure. We analyze the incentive result in the exogenous-slack regime and separately examine bounded slack-estimation error and endogenous-delay limitations. Under fixed feasible slacks, LIA is individually rational and achieves welfare at least \(e^{-\lambda\Delta}\) relative to the optimal feasible allocation, where \(\Delta\) is the slack spread. We evaluate LIA on STARLINK-200, INTERNET-100, and DSN-30 across 52,500 baseline instances with market sizes \(n\in\{10,20,30,40,50\}\) and conduct additional robustness sweeps. On Starlink and Internet, LIA maintains near-efficiency while eliminating measured timing rents. However, on DSN, welfare is lower in thin markets but improves with depth. We also distinguish winner-determination time from the background cost of maintaining slack estimates and study robustness beyond independent and identically distributed (iid) noise through error-spread bounds and structured (distance-biased and subnetwork-correlated) noise models. These results suggest that causal-consistent mechanism design offers a practical non-buffering alternative to synchronized delay equalization in heterogeneous telecom infrastructures. 

---
# Regime-Calibrated Demand Priors for Ride-Hailing Fleet Dispatch and Repositioning 

**Authors**: Indar Kumar, Akanksha Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2604.03883)  

**Abstract**: Effective ride-hailing dispatch requires anticipating demand patterns that vary substantially across time-of-day, day-of-week, season, and special events. We propose a regime-calibrated approach that (i) segments historical trip data into demand regimes, (ii) matches the current operating period to the most similar historical analogues via a similarity ensemble combining Kolmogorov-Smirnov distance, Wasserstein-1 distance, feature distance, variance ratio, event pattern similarity, and temporal proximity, and (iii) uses the resulting calibrated demand prior to drive both an LP-based fleet repositioning policy and batch dispatch with Hungarian matching. In ablation, a distributional-only metric subset achieves the strongest mean-wait reduction, while the full ensemble is retained as a robustness-oriented default that preserves calendar and event context.
Evaluated on 5.2 million NYC TLC trips across 8 diverse scenarios (winter/summer, weekday/weekend/holiday, morning/evening/night) with 5 random seeds each, our method reduces mean rider wait times by 31.1% (bootstrap 95% CI: [26.5, 36.6]; Friedman chi-squared = 80.0, p = 4.25e-18; Cohen's d = 7.5-29.9). P95 wait drops 37.6% and the Gini coefficient of wait times improves from 0.441 to 0.409. The two contributions compose multiplicatively: calibration provides 16.9% reduction relative to the replay baseline; LP repositioning adds a further 15.5%. The approach requires no training, is deterministic and explainable, generalizes to Chicago (23.3% wait reduction using the NYC-built regime library without retraining), and is robust across fleet sizes (32-47% improvement for 0.5x-2.0x fleet scaling). Code is available at this https URL. 

---
# Enhancing behavioral nudges with large language model-based iterative personalization: A field experiment on electricity and hot-water conservation 

**Authors**: Zonghan Li, Yi Liu, Chunyan Wang, Song Tong, Kaiping Peng, Feng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2604.03881)  

**Abstract**: Nudging is widely used to promote behavioral change, but its effectiveness is often limited when recipients must repeatedly translate feedback into workable next steps under changing circumstances. Large language models (LLMs) may help reduce part of this cognitive work by generating personalized guidance and updating it iteratively across intervention rounds. We developed an LLM agent for iterative personalization and tested it in a three-arm randomized experiment among 233 university residents in China, using daily electricity and shower hot-water conservation as objectively measured cases differing in friction. LLM-personalized nudges (T2) produced the largest conservation effects, while image-enhanced conventional nudges (T1) and text-based conventional nudges (C) showed similar outcomes (omnibus p = 0.009). Relative to C, T2 reduced electricity consumption by 0.56 kWh per room-day (p = 0.014), corresponding to an 18.3 percentage-point higher adjusted saving rate. This advantage emerged within the first two intervention rounds, alongside iterative updating of personalized guidance, and persisted thereafter. Hot-water outcomes followed the same direction but were smaller, less precisely estimated, and attenuated over time, consistent with stronger friction in this domain. LLM-personalized nudges emphasized prospective and context-specific guidance and were associated with higher participant engagement. This study provides field evidence that LLM-based iterative personalization can enhance behavioral nudging, with behavioral friction as a potential boundary condition. Larger trials and extension to more behaviors are warranted. 

---
# When Models Know More Than They Say: Probing Analogical Reasoning in LLMs 

**Authors**: Hope McGovern, Caroline Craig, Thomas Lippincott, Hale Sirin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03877)  

**Abstract**: Analogical reasoning is a core cognitive faculty essential for narrative understanding. While LLMs perform well when surface and structural cues align, they struggle in cases where an analogy is not apparent on the surface but requires latent information, suggesting limitations in abstraction and generalisation. In this paper we compare a model's probed representations with its prompted performance at detecting narrative analogies, revealing an asymmetry: for rhetorical analogies, probing significantly outperforms prompting in open-source models, while for narrative analogies, they achieve a similar (low) performance. This suggests that the relationship between internal representations and prompted behavior is task-dependent and may reflect limitations in how prompting accesses available information. 

---
# k-Maximum Inner Product Attention for Graph Transformers and the Expressive Power of GraphGPS The Expressive Power of GraphGPS 

**Authors**: Jonas De Schouwer, Haitz Sáez de Ocáriz Borde, Xiaowen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03815)  

**Abstract**: Graph transformers have shown promise in overcoming limitations of traditional graph neural networks, such as oversquashing and difficulties in modelling long-range dependencies. However, their application to large-scale graphs is hindered by the quadratic memory and computational complexity of the all-to-all attention mechanism. Although alternatives such as linearized attention and restricted attention patterns have been proposed, these often degrade performance or limit expressive power. To better balance efficiency and effectiveness, we introduce k-Maximum Inner Product (k-MIP) attention for graph transformers. k-MIP attention selects the most relevant key nodes per query via a top-k operation, yielding a sparse yet flexible attention pattern. Combined with an attention score computation based on symbolic matrices, this results in linear memory complexity and practical speedups of up to an order of magnitude compared to all-to-all attention, enabling the processing of graphs with over 500k nodes on a single A100 GPU. We provide a theoretical analysis of expressive power, showing that k-MIP attention does not compromise the expressiveness of graph transformers: specifically, we prove that k-MIP transformers can approximate any full-attention transformer to arbitrary precision. In addition, we analyze the expressive power of the GraphGPS framework, in which we integrate our attention mechanism, and establish an upper bound on its graph distinguishing capability in terms of the S-SEG-WL test. Finally, we validate our approach on the Long Range Graph Benchmark, the City-Networks benchmark, and two custom large-scale inductive point cloud datasets, consistently ranking among the top-performing scalable graph transformers. 

---
# InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset 

**Authors**: Felix Stillger, Lukas Hahn, Frederik Hasecke, Tobias Meisen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03814)  

**Abstract**: Camera extrinsic calibration is a fundamental task in computer vision. However, precise relative pose estimation in constrained, highly distorted environments, such as in-cabin automotive monitoring (ICAM), remains challenging. We present InCaRPose, a Transformer-based architecture designed for robust relative pose prediction between image pairs, which can be used for camera extrinsic calibration. By leveraging frozen backbone features such as DINOv3 and a Transformer-based decoder, our model effectively captures the geometric relationship between a reference and a target view. Unlike traditional methods, our approach achieves absolute metric-scale translation within the physically plausible adjustment range of in-cabin camera mounts in a single inference step, which is critical for ICAM, where accurate real-world distances are required for safety-relevant perception. We specifically address the challenges of highly distorted fisheye cameras in automotive interiors by training exclusively on synthetic data. Our model is capable of generalization to real-world cabin environments without relying on the exact same camera intrinsics and additionally achieves competitive performance on the public 7-Scenes dataset. Despite having limited training data, InCaRPose maintains high precision in both rotation and translation, even with a ViT-Small backbone. This enables real-time performance for time-critical inference, such as driver monitoring in supervised autonomous driving. We release our real-world In-Cabin-Pose test dataset consisting of highly distorted vehicle-interior images and our code at this https URL. 

---
# Representational Collapse in Multi-Agent LLM Committees: Measurement and Diversity-Aware Consensus 

**Authors**: Dipkumar Patel  

**Link**: [PDF](https://arxiv.org/pdf/2604.03809)  

**Abstract**: Multi-agent LLM committees replicate the same model under different role prompts and aggregate outputs by majority vote, implicitly assuming that agents contribute complementary evidence. We embed each agent's chain-of-thought rationale and measure pairwise similarity: across 100 GSM8K questions with three Qwen2.5-14B agents, mean cosine similarity is 0.888 and effective rank is 2.17 out of 3.0, a failure mode we term representational collapse. DALC, a training-free consensus protocol that computes diversity weights from embedding geometry, reaches 87% on GSM8K versus 84% for self-consistency at 26% lower token cost. Ablation experiments reveal 1-3 point per-protocol run-to-run variance, confirm that hint sharing contributes more than diversity weighting alone, and show that encoder choice strongly modulates collapse severity (cosine 0.908 with mxbai versus 0.888 with nomic) and downstream accuracy. The more robust finding is that collapse is measurable, worsens on harder tasks, and that the choice of embedding proxy is a first-order design decision for any latent communication protocol. 

---
# Automated Conjecture Resolution with Formal Verification 

**Authors**: Haocheng Ju, Guoxiong Gao, Jiedong Jiang, Bin Wu, Zeming Sun, Leheng Chen, Yutong Wang, Yuefeng Wang, Zichen Wang, Wanyi He, Peihao Wu, Liang Xiao, Ruochuan Liu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03789)  

**Abstract**: Recent advances in large language models have significantly improved their ability to perform mathematical reasoning, extending from elementary problem solving to increasingly capable performance on research-level problems. However, reliably solving and verifying such problems remains challenging due to the inherent ambiguity of natural language reasoning. In this paper, we propose an automated framework for tackling research-level mathematical problems that integrates natural language reasoning with formal verification, enabling end-to-end problem solving with minimal human intervention. Our framework consists of two components: an informal reasoning agent, Rethlas, and a formal verification agent, Archon. Rethlas mimics the workflow of human mathematicians by combining reasoning primitives with our theorem search engine, Matlas, to explore solution strategies and construct candidate proofs. Archon, equipped with our formal theorem search engine LeanSearch, translates informal arguments into formalized Lean 4 projects through structured task decomposition, iterative refinement, and automated proof synthesis, ensuring machine-checkable correctness. Using this framework, we automatically resolve an open problem in commutative algebra and formally verify the resulting proof in Lean 4 with essentially no human involvement. Our experiments demonstrate that strong theorem retrieval tools enable the discovery and application of cross-domain mathematical techniques, while the formal agent is capable of autonomously filling nontrivial gaps in informal arguments. More broadly, our work illustrates a promising paradigm for mathematical research in which informal and formal reasoning systems, equipped with theorem retrieval tools, operate in tandem to produce verifiable results, substantially reduce human effort, and offer a concrete instantiation of human-AI collaborative mathematical research. 

---
# An Improved Last-Iterate Convergence Rate for Anchored Gradient Descent Ascent 

**Authors**: Anja Surina, Arun Suggala, George Tsoukalas, Anton Kovsharov, Sergey Shirobokov, Francisco J. R. Ruiz, Pushmeet Kohli, Swarat Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2604.03782)  

**Abstract**: We analyze the last-iterate convergence of the Anchored Gradient Descent Ascent algorithm for smooth convex-concave min-max problems. While previous work established a last-iterate rate of $\mathcal{O}(1/t^{2-2p})$ for the squared gradient norm, where $p \in (1/2, 1)$, it remained an open problem whether the improved exact $\mathcal{O}(1/t)$ rate is achievable. In this work, we resolve this question in the affirmative. This result was discovered autonomously by an AI system capable of writing formal proofs in Lean. The Lean proof can be accessed at this https URL 

---
# CountsDiff: A Diffusion Model on the Natural Numbers for Generation and Imputation of Count-Based Data 

**Authors**: Renzo G. Soatto, Anders Hoel, Greycen Ren, Shorna Alam, Stephen Bates, Nikolaos P. Daskalakis, Caroline Uhler, Maria Skoularidou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03779)  

**Abstract**: Diffusion models have excelled at generative tasks for both continuous and token-based domains, but their application to discrete ordinal data remains underdeveloped. We present CountsDiff, a diffusion framework designed to natively model distributions on the natural numbers. CountsDiff extends the Blackout diffusion framework by simplifying its formulation through a direct parameterization in terms of a survival probability schedule and an explicit loss weighting. This introduces flexibility through design parameters with direct analogues in existing diffusion modeling frameworks. Beyond this reparameterization, CountsDiff introduces features from modern diffusion models, previously absent in counts-based domains, including continuous-time training, classifier-free guidance, and churn/remasking reverse dynamics that allow non-monotone reverse trajectories. We propose an initial instantiation of CountsDiff and validate it on natural image datasets (CIFAR-10, CelebA), exploring the effects of varying the introduced design parameters in a complex, well-studied, and interpretable data domain. We then highlight biological count assays as a natural use case, evaluating CountsDiff on single-cell RNA-seq imputation in a fetal cell and heart cell atlas. Remarkably, we find that even this simple instantiation matches or surpasses the performance of a state-of-the-art discrete generative model and leading RNA-seq imputation methods, while leaving substantial headroom for further gains through optimized design choices in future work. 

---
# When Does Multimodal AI Help? Diagnostic Complementarity of Vision-Language Models and CNNs for Spectrum Management in Satellite-Terrestrial Networks 

**Authors**: Yuanhang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03774)  

**Abstract**: The adoption of vision-language models (VLMs) for wireless network management is accelerating, yet no systematic understanding exists of where these large foundation models outperform lightweight convolutional neural networks (CNNs) for spectrum-related tasks. This paper presents the first diagnostic comparison of VLMs and CNNs for spectrum heatmap understanding in non-terrestrial network and terrestrial network (NTN-TN) cooperative systems. We introduce SpectrumQA, a benchmark comprising 108K visual question-answer pairs across four granularity levels: scene classification (L1), regional reasoning (L2), spatial localization (L3), and semantic reasoning (L4). Our experiments on three NTN-TN scenarios with a frozen Qwen2-VL-7B and a trained ResNet-18 reveal a clear taskdependent complementarity: CNN achieves 72.9% accuracy at severity classification (L1) and 0.552 IoU at spatial localization (L3), while VLM uniquely enables semantic reasoning (L4) with F1=0.576 using only three in-context examples-a capability fundamentally absent in CNN architectures. Chain-of-thought (CoT) prompting further improves VLM reasoning by 12.6% (F1: 0.209->0.233) while having zero effect on spatial tasks, confirming that the complementarity is rooted in architectural differences rather than prompting limitations. A deterministic task-type router that delegates supervised tasks to CNN and reasoning tasks to VLM achieves a composite score of 0.616, a 39.1% improvement over CNN alone. We further show that VLM representations exhibit stronger cross-scenario robustness, with smaller performance degradation in 5 out of 6 transfer directions. These findings provide actionable guidelines: deploy CNNs for spatial localization and VLMs for semantic spectrum reasoning, rather than treating them as substitutes. 

---
# Automated Attention Pattern Discovery at Scale in Large Language Models 

**Authors**: Jonathan Katzy, Razvan-Mihai Popescu, Erik Mekkes, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03764)  

**Abstract**: Large language models have found success by scaling up capabilities to work in general settings. The same can unfortunately not be said for interpretability methods. The current trend in mechanistic interpretability is to provide precise explanations of specific behaviors in controlled settings. These often do not generalize, or are too resource intensive for larger studies. In this work we propose to study repeated behaviors in large language models by mining completion scenarios in Java code datasets, through exploiting the structured nature of code. We collect the attention patterns generated in the attention heads to demonstrate that they are scalable signals for global interpretability of model components. We show that vision models offer a promising direction for analyzing attention patterns at scale. To demonstrate this, we introduce the Attention Pattern - Masked Autoencoder(AP-MAE), a vision transformer-based model that efficiently reconstructs masked attention patterns. Experiments on StarCoder2 show that AP-MAE (i) reconstructs masked attention patterns with high accuracy, (ii) generalizes across unseen models with minimal degradation, (iii) reveals recurring patterns across inferences, (iv) predicts whether a generation will be correct without access to ground truth, with accuracies ranging from 55% to 70% depending on the task, and (v) enables targeted interventions that increase accuracy by 13.6% when applied selectively, but cause collapse when applied excessively. These results establish attention patterns as a scalable signal for interpretability and demonstrate that AP-MAE provides a transferable foundation for both analysis and intervention in large language models. Beyond its standalone value, AP-MAE also serves as a selection procedure to guide fine-grained mechanistic approaches. We release code and models to support future work in large-scale interpretability. 

---
# Build on Priors: Vision--Language--Guided Neuro-Symbolic Imitation Learning for Data-Efficient Real-World Robot Manipulation 

**Authors**: Pierrick Lorang, Johannes Huemer, Timothy Duggan, Kai Goebel, Patrik Zips, Matthias Scheutz  

**Link**: [PDF](https://arxiv.org/pdf/2604.03759)  

**Abstract**: Enabling robots to learn long-horizon manipulation tasks from a handful of demonstrations remains a central challenge in robotics. Existing neuro-symbolic approaches often rely on hand-crafted symbolic abstractions, semantically labeled trajectories or large demonstration datasets, limiting their scalability and real-world applicability. We present a scalable neuro-symbolic framework that autonomously constructs symbolic planning domains and data-efficient control policies from as few as one to thirty unannotated skill demonstrations, without requiring manual domain engineering. Our method segments demonstrations into skills and employs a Vision-Language Model (VLM) to classify skills and identify equivalent high-level states, enabling automatic construction of a state-transition graph. This graph is processed by an Answer Set Programming solver to synthesize a PDDL planning domain, which an oracle function exploits to isolate the minimal, task-relevant and target relative observation and action spaces for each skill policy. Policies are learned at the control reference level rather than at the raw actuator signal level, yielding a smoother and less noisy learning target. Known controllers can be leveraged for real-world data augmentation by projecting a single demonstration onto other objects in the scene, simultaneously enriching the graph construction process and the dataset for imitation learning. We validate our framework primarily on a real industrial forklift across statistically rigorous manipulation trials, and demonstrate cross-platform generality on a Kinova Gen3 robotic arm across two standard benchmarks. Our results show that grounding control learning, VLM-driven abstraction, and automated planning synthesis into a unified pipeline constitutes a practical path toward scalable, data-efficient, expert-free and interpretable neuro-symbolic robotics. 

---
# AutoReSpec: A Framework for Generating Specification using Large Language Models 

**Authors**: Ragib Shahariar Ayon, Shibbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2604.03758)  

**Abstract**: Formal specification generation has recently drawn attention in software engineering as a way to improve program correctness without requiring manual annotations. Large Language Models (LLMs) have shown promise in this area, but early results reveal several limitations. Generated specifications often fail verification due to syntax errors, logical inaccuracies, or incomplete reasoning, especially in programs with loops or branching logic. Techniques like SpecGen and FormalBench attempt to address this through prompting and benchmarking, but they typically rely on static prompts and do not offer mechanisms for recovering from failure or adapting to different program structures. In this paper, we present AutoReSpec, a collaborative framework that combines open and closed-source LLMs for verifiable specification generation. AutoReSpec dynamically chooses an LLM pair and prompt configuration based on the structure of the input program. If the primary LLM fails to produce a valid output, a collaborative model is invoked, using validator feedback to refine and correct the specification. This two-stage design enables both speed and robustness. We evaluate AutoReSpec on a new benchmark of 72 real-world and synthetic Java programs. Our results show that it achieves 67 passes out of 72, outperforming SpecGen and FormalBench in both Success Probability and Completeness. Our experimental evaluation achieves a 58.2% success probability and a 69.2% completeness score, while cutting evaluation time by 26.89% on average compared to prior methods. Together, these results demonstrate that AutoReSpec offers a scalable, efficient, and reliable approach to LLM-based formal specification generation. 

---
# Can Humans Tell? A Dual-Axis Study of Human Perception of LLM-Generated News 

**Authors**: Alexander Loth, Martin Kappes, Marc-Oliver Pahl  

**Link**: [PDF](https://arxiv.org/pdf/2604.03755)  

**Abstract**: Can humans tell whether a news article was written by a person or a large language model (LLM)? We investigate this question using JudgeGPT, a study platform that independently measures source attribution (human vs. machine) and authenticity judgment (legitimate vs. fake) on continuous scales. From 2,318 judgments collected from 1,054 participants across content generated by six LLMs, we report five findings: (1) participants cannot reliably distinguish machine-generated from human-written text (p > .05, Welch's t-test); (2) this inability holds across all tested models, including open-weight models with as few as 7B parameters; (3) self-reported domain expertise predicts judgment accuracy (r = .35, p < .001) whereas political orientation does not (r = -.10, n.s.); (4) clustering reveals distinct response strategies ("Skeptics" vs. "Believers"); and (5) accuracy degrades after approximately 30 sequential evaluations due to cognitive fatigue. The answer, in short, is no: humans cannot reliably tell. These results indicate that user-side detection is not a viable defense and motivate system-level countermeasures such as cryptographic content provenance. 

---
# Testing the Limits of Truth Directions in LLMs 

**Authors**: Angelos Poulis, Mark Crovella, Evimaria Terzi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03754)  

**Abstract**: Large language models (LLMs) have been shown to encode truth of statements in their activation space along a linear truth direction. Previous studies have argued that these directions are universal in certain aspects, while more recent work has questioned this conclusion drawing on limited generalization across some settings. In this work, we identify a number of limits of truth-direction universality that have not been previously understood. We first show that truth directions are highly layer-dependent, and that a full understanding of universality requires probing at many layers in the model. We then show that truth directions depend heavily on task type, emerging in earlier layers for factual and later layers for reasoning tasks; they also vary in performance across levels of task complexity. Finally, we show that model instructions dramatically affect truth directions; simple correctness evaluation instructions significantly affect the generalization ability of truth probes. Our findings indicate that universality claims for truth directions are more limited than previously known, with significant differences observable for various model layers, task difficulties, task types, and prompt templates. 

---
# CREBench: Evaluating Large Language Models in Cryptographic Binary Reverse Engineering 

**Authors**: Baicheng Chen, Yu Wang, Ziheng Zhou, Xiangru Liu, Juanru Li, Yilei Chen, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2604.03750)  

**Abstract**: Reverse engineering (RE) is central to software security, particularly for cryptographic programs that handle sensitive data and are highly prone to vulnerabilities. It supports critical tasks such as vulnerability discovery and malware analysis. Despite its importance, RE remains labor-intensive and requires substantial expertise, making large language models (LLMs) a potential solution for automating the process. However, their capabilities for RE remain systematically underexplored. To address this gap, we study the cryptographic binary RE capabilities of LLMs and introduce \textbf{CREBench}, a benchmark comprising 432 challenges built from 48 standard cryptographic algorithms, 3 insecure crypto key usage scenarios, and 3 difficulty levels. Each challenge follows a Capture-the-Flag (CTF) RE challenge, requiring the model to analyze the underlying cryptographic logic and recover the correct input. We design an evaluation framework comprising four sub-tasks, from algorithm identification to correct flag recovery. We evaluate eight frontier LLMs on CREBench. GPT-5.4, the best-performing model, achieves 64.03 out of 100 and recovers the flag in 59\% of challenges. We also establish a strong human expert baseline of 92.19 points, showing that humans maintain an advantage in cryptographic RE tasks. Our code and dataset are available at this https URL. 

---
# 15 Years of Augmented Human(s) Research: Where Do We Stand? 

**Authors**: Steeven Villa, Abdallah El Ali  

**Link**: [PDF](https://arxiv.org/pdf/2604.03715)  

**Abstract**: The Augmented Human vision broadly seeks to improve or expand baseline human functioning through the restoration or extension of physical, intellectual, and social capabilities. However, given the rapid pace of technology development, we ask: what exactly does Augmented Human research involve, what are its core themes, and how has the Augmented Human(s) conference series evolved over time? To answer this, we conducted a scientometric analysis on the past 15 years of the Augmented Human(s) conference (N=735 paper), focusing on: geographical aspects, submissions and citation timelines, author frequency and popularity, and topic modeling. We find that: (a) Number of papers in the conference exhibit a bimodal distribution, peaking in 2015 and 2025, but showing periods of stagnant growth; (b) key topics over time include Haptics, Wearable Sensing, Vision & Eye Tracking, Embodied Interaction, and Sports / Motion; (c) some seminal papers on AH are not published in AH(s), but rather at related venues (e.g., CHI); (d) the conference has an active Japanese HCI community despite its historical Eurocentric location dominance. We contribute a closer look at the trajectory of the AH(s) field, and raise considerations of definitional and research scope ambiguities given the core problems/enhancements the field seeks to address. 

---
# Learning Superpixel Ensemble and Hierarchy Graphs for Melanoma Detection 

**Authors**: Asmaa M. Elwer, Muhammad A. Rushdi, Mahmoud H. Annaby  

**Link**: [PDF](https://arxiv.org/pdf/2604.03710)  

**Abstract**: Graph signal processing (GSP) is becoming a major tool in biomedical signal and image analysis. In most GSP techniques, graph structures and edge weights have been typically set via statistical and computational methods. More recently, graph structure learning methods offered more reliable and flexible data representations. In this work, we introduce a graph learning approach for melanoma detection in dermoscopic images based on two graph-theoretic representations: superpixel ensemble graphs (SEG) and superpixel hierarchy graphs (SHG). For these two types of graphs, superpixel maps of a skin lesion image are respectively generated at multiple levels without and with parentchild constraints among superpixels at adjacent levels, where each level corresponds to a subgraph with a different number of nodes (20, 40, 60, 80, or 100 nodes). Two edge weight assignment techniques are explored: handcrafted Gaussian weights and learned weights based on optimization methods. The graph nodal signals are assigned based on texture, geometric, and color superpixel features. In addition, the effect of graph edge thresholding is investigated by applying different thresholds (25%, 50%, and 75%) to prune the weakest edges and analyze the impact of pruning on the melanoma detection performance. Experimental evaluation of the proposed method is performed with different classifiers trained and tested on the publicly available ISIC2017 dataset. Data augmentation is applied to alleviate class imbalance by adding more melanoma images from the ISIC archive. The results show that learned superpixel ensemble graphs with textural nodal signals give the highest performance reaching an accuracy of 99.00% and an AUC of 99.59%. 

---
# RDEx-CMOP: Feasibility-Aware Indicator-Guided Differential Evolution for Fixed-Budget Constrained Multiobjective Optimization 

**Authors**: Sichen Tao, Yifei Yang, Ruihan Zhao, Kaiyu Wang, Sicheng Liu, Shangce Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03708)  

**Abstract**: Constrained multiobjective optimisation requires fast feasibility attainment together with stable convergence and diversity preservation under strict evaluation budgets. This report documents RDEx-CMOP, the differential evolution variant used in the IEEE CEC 2025 numerical optimisation competition (C06 special session) constrained multiobjective track. RDEx-CMOP integrates an {\epsilon}-level feasibility schedule, a SPEA2-style indicator-driven fitness assignment, and a fitness-oriented current-to-pbest/1 mutation operator. We evaluate RDEx-CMOP on the official CEC 2025 CMOP benchmark using the median-target U-score framework and the released trace data. Experimental results show that RDEx-CMOP achieves the highest total score and the best overall average rank among all released comparison algorithms, with strong target-attainment behaviour and near-zero final violation on most problems. 

---
# Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation 

**Authors**: Zhifu Wei, Yizhou Dang, Guibing Guo, Chuang Zhao, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2604.03688)  

**Abstract**: Sequential Recommendation (SR) learns user preferences from their historical interaction sequences and provides personalized suggestions. In real-world scenarios, most items exhibit sparse interactions, known as the tail-item problem. This issue limits the model's ability to accurately capture item transition patterns. To tackle this, large language models (LLMs) offer a promising solution by capturing semantic relationships between items. Despite previous efforts to leverage LLM-derived embeddings for enriching tail items, they still face the following limitations: 1) They struggle to effectively fuse collaborative signals with semantic knowledge, leading to suboptimal item embedding quality. 2) Existing methods overlook the structural inconsistency between the ID and LLM embedding spaces, causing conflicting signals that degrade recommendation accuracy. In this work, we propose a Fusion and Alignment Enhancement framework with LLMs for Tail-item Sequential Recommendation (FAERec), which improves item representations by generating coherently-fused and structurally consistent embeddings. For the information fusion challenge, we design an adaptive gating mechanism that dynamically fuses ID and LLM embeddings. Then, we propose a dual-level alignment approach to mitigate structural inconsistency. The item-level alignment establishes correspondences between ID and LLM embeddings of the same item through contrastive learning, while the feature-level alignment constrains the correlation patterns between corresponding dimensions across the two embedding spaces. Furthermore, the weights of the two alignments are adjusted by a curriculum learning scheduler to avoid premature optimization of the complex feature-level objective. Extensive experiments across three widely used datasets with multiple representative SR backbones demonstrate the effectiveness and generalizability of our framework. 

---
# LightThinker++: From Reasoning Compression to Memory Management 

**Authors**: Yuqi Zhu, Jintian Zhang, Zhenjie Wan, Yujie Luo, Shuofei Qiao, Zhengke Gui, Da Zheng, Lei Liang, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03679)  

**Abstract**: Large language models (LLMs) excel at complex reasoning, yet their efficiency is limited by the surging cognitive overhead of long thought traces. In this paper, we propose LightThinker, a method that enables LLMs to dynamically compress intermediate thoughts into compact semantic representations. However, static compression often struggles with complex reasoning where the irreversible loss of intermediate details can lead to logical bottlenecks. To address this, we evolve the framework into LightThinker++, introducing Explicit Adaptive Memory Management. This paradigm shifts to behavioral-level management by incorporating explicit memory primitives, supported by a specialized trajectory synthesis pipeline to train purposeful memory scheduling. Extensive experiments demonstrate the framework's versatility across three dimensions. (1) LightThinker reduces peak token usage by 70% and inference time by 26% with minimal accuracy loss. (2) In standard reasoning, LightThinker++ slashes peak token usage by 69.9% while yielding a +2.42% accuracy gain under the same context budget for maximum performance. (3) Most notably, in long-horizon agentic tasks, it maintains a stable footprint beyond 80 rounds (a 60%-70% reduction), achieving an average performance gain of 14.8% across different complex scenarios. Overall, our work provides a scalable direction for sustaining deep LLM reasoning over extended horizons with minimal overhead. 

---
# Unlocking Prompt Infilling Capability for Diffusion Language Models 

**Authors**: Yoshinari Fujinuma, Keisuke Sakaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03677)  

**Abstract**: Masked diffusion language models (dLMs) generate text through bidirectional denoising, yet this capability remains locked for infilling prompts. This limitation is an artifact of the current supervised finetuning (SFT) convention of applying response-only masking. To unlock this capability, we extend full-sequence masking during SFT, where both prompts and responses are masked jointly. Once unlocked, the model infills masked portions of a prompt template conditioned on few-shot examples. We show that such model-infilled prompts match or surpass manually designed templates, transfer effectively across models, and are complementary to existing prompt optimization methods. Our results suggest that training practices, not architectural limitations, are the primary bottleneck preventing masked diffusion language models from infilling effective prompts 

---
# AI Appeals Processor: A Deep Learning Approach to Automated Classification of Citizen Appeals in Government Services 

**Authors**: Vladimir Beskorovainyi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03672)  

**Abstract**: Government agencies worldwide face growing volumes of citizen appeals, with electronic submissions increasing significantly over recent years. Traditional manual processing averages 20 minutes per appeal with only 67% classification accuracy, creating significant bottlenecks in public service delivery. This paper presents AI Appeals Processor, a microservice-based system that integrates natural language processing and deep learning techniques for automated classification and routing of citizen appeals. We evaluate multiple approaches -- including Bag-of-Words with SVM, TF-IDF with SVM, fastText, Word2Vec with LSTM, and BERT -- on a representative dataset of 10,000 real citizen appeals across three primary categories (complaints, applications, and proposals) and seven thematic domains. Our experiments demonstrate that a Word2Vec+LSTM architecture achieves 78% classification accuracy while reducing processing time by 54%, offering an optimal balance between accuracy and computational efficiency compared to transformer-based models. 

---
# ART: Adaptive Relational Transformer for Pedestrian Trajectory Prediction with Temporal-Aware Relations 

**Authors**: Ruochen Li, Ziyi Chang, Junyan Hu, Jiannan Li, Amir Atapour-Abarghouei, Hubert P. H. Shum  

**Link**: [PDF](https://arxiv.org/pdf/2604.03649)  

**Abstract**: Accurate prediction of real-world pedestrian trajectories is crucial for a wide range of robot-related applications. Recent approaches typically adopt graph-based or transformer-based frameworks to model interactions. Despite their effectiveness, these methods either introduce unnecessary computational overhead or struggle to represent the diverse and time-varying characteristics of human interactions. In this work, we present an Adaptive Relational Transformer (ART), which introduces a Temporal-Aware Relation Graph (TARG) to explicitly capture the evolution of pairwise interactions and an Adaptive Interaction Pruning (AIP) mechanism to reduce redundant computations efficiently. Extensive evaluations on ETH/UCY and NBA benchmarks show that ART delivers state-of-the-art accuracy with high computational efficiency. 

---
# Stabilizing Unsupervised Self-Evolution of MLLMs via Continuous Softened Retracing reSampling 

**Authors**: Yunyao Yu, Zhengxian Wu, Zhuohong Chen, Hangrui Xu, Zirui Liao, Xiangwen Deng, Zhifang Liu, Senyuan Shi, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03647)  

**Abstract**: In the unsupervised self-evolution of Multimodal Large Language Models, the quality of feedback signals during post-training is pivotal for stable and effective learning. However, existing self-evolution methods predominantly rely on majority voting to select the most frequent output as the pseudo-golden answer, which may stem from the model's intrinsic biases rather than guaranteeing the objective correctness of the reasoning paths. To counteract the degradation, we propose \textbf{C}ontinuous \textbf{S}oftened \textbf{R}etracing re\textbf{S}ampling (\textbf{CSRS}) in MLLM self-evolution. Specifically, we introduce a Retracing Re-inference Mechanism (\textbf{RRM}) that the model re-inferences from anchor points to expand the exploration of long-tail reasoning paths. Simultaneously, we propose Softened Frequency Reward (\textbf{SFR}), which replaces binary rewards with continuous signals, calibrating reward based on the answers' frequency across sampled reasoning sets. Furthermore, incorporated with Visual Semantic Perturbation (\textbf{VSP}), CSRS ensures the model prioritizes mathematical logic over visual superficiality. Experimental results demonstrate that CSRS significantly enhances the reasoning performance of Qwen2.5-VL-7B on benchmarks such as MathVision. We achieve state-of-the-art (SOTA) results in unsupervised self-evolution on geometric tasks. Our code is avaible at this https URL. 

---
# Delayed Homomorphic Reinforcement Learning for Environments with Delayed Feedback 

**Authors**: Jongsoo Lee, Jangwon Kim, Soohee Han  

**Link**: [PDF](https://arxiv.org/pdf/2604.03641)  

**Abstract**: Reinforcement learning in real-world systems is often accompanied by delayed feedback, which breaks the Markov assumption and impedes both learning and control. Canonical state augmentation approaches cause the state-space explosion, which introduces a severe sample-complexity burden. Despite recent progress, the state-of-the-art augmentation-based baselines remain incomplete: they either predominantly reduce the burden on the critic or adopt non-unified treatments for the actor and critic. To provide a structured and sample-efficient solution, we propose delayed homomorphic reinforcement learning (DHRL), a framework grounded in MDP homomorphisms that collapses belief-equivalent augmented states and enables efficient policy learning on the resulting abstract MDP without loss of optimality. We provide theoretical analyses of state-space compression bounds and sample complexity, and introduce a practical algorithm. Experiments on continuous control tasks in MuJoCo benchmark confirm that our algorithm outperforms strong augmentation-based baselines, particularly under long delays. 

---
# A Generative Foundation Model for Multimodal Histopathology 

**Authors**: Jinxi Xiang, Mingjie Li, Siyu Hou, Yijiang Chen, Xiangde Luo, Yuanfeng Ji, Xiang Zhou, Ehsan Adeli, Akshay Chaudhari, Curtis P. Langlotz, Kilian M. Pohl, Ruijiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03635)  

**Abstract**: Accurate diagnosis and treatment of complex diseases require integrating histological, molecular, and clinical data, yet in practice these modalities are often incomplete owing to tissue scarcity, assay cost, and workflow constraints. Existing computational approaches attempt to impute missing modalities from available data but rely on task-specific models trained on narrow, single source-target pairs, limiting their generalizability. Here we introduce MuPD (Multimodal Pathology Diffusion), a generative foundation model that embeds hematoxylin and eosin (H&E)-stained histology, molecular RNA profiles, and clinical text into a shared latent space through a diffusion transformer with decoupled cross-modal attention. Pretrained on 100 million histology image patches, 1.6 million text-histology pairs, and 10.8 million RNA-histology pairs spanning 34 human organs, MuPD supports diverse cross-modal synthesis tasks with minimal or no task-specific fine-tuning. For text-conditioned and image-to-image generation, MuPD synthesizes histologically faithful tissue architectures, reducing Fréchet inception distance (FID) scores by 50% relative to domain-specific models and improving few-shot classification accuracy by up to 47% through synthetic data augmentation. For RNA-conditioned histology generation, MuPD reduces FID by 23% compared with the next-best method while preserving cell-type distributions across five cancer types. As a virtual stainer, MuPD translates H&E images to immunohistochemistry and multiplex immunofluorescence, improving average marker correlation by 37% over existing approaches. These results demonstrate that a single, unified generative model pretrained across heterogeneous pathology modalities can substantially outperform specialized alternatives, providing a scalable computational framework for multimodal histopathology. 

---
# Persistent Cross-Attempt State Optimization for Repository-Level Code Generation 

**Authors**: Ruwei Pan, Jiangshuai Wang, Qisheng Zhang, Yueheng Zhu, Linhao Wu, Zixiong Yang, Yakun Zhang, Lu Zhang, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03632)  

**Abstract**: Large language models (LLMs) have achieved substantial progress in repository-level code generation. However, solving the same repository-level task often requires multiple attempts, while existing methods still optimize each attempt in isolation and do not preserve or reuse task-specific state across attempts. In this paper, we propose LiveCoder, a novel framework for repository-level code generation based on cross-attempt knowledge optimization. LiveCoder maintains persistent task-specific state from prior attempts to guide subsequent generation. This state includes success knowledge, which captures reusable signals from previously strong repositories, failure knowledge, which records unsuccessful outcomes and their diagnostic signals, and a historical-best repository, which preserves the strongest result found so far and prevents regression. These components collectively transform repeated repository generation into a persistent, knowledge-driven optimization process. We evaluate LiveCoder using four frontier LLMs on two representative repository-level code generation benchmarks. Extensive experimental results demonstrate the effectiveness and efficiency of LiveCoder, improving the functional score by up to 22.94 percentage points, increasing repository reuse to 81.58%, and reducing cost by up to 53.63% on RAL-Bench while maintaining broadly stable non-functional quality. 

---
# Toward Executable Repository-Level Code Generation via Environment Alignment 

**Authors**: Ruwei Pan, Junlei Shen, Linhao Wu, Yueheng Zhu, Zixiong Yang, Yakun Zhang, Lu Zhang, Hongyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03622)  

**Abstract**: Large language models (LLMs) have achieved strong performance on code generation, but existing methods still struggle with repository-level code generation under executable validation. Under this evaluation setting, success is determined not by the plausibility of isolated code fragments, but by whether a generated multi-file repository can be successfully installed, have its dependencies and internal references resolved, be launched, and be validated in a real execution environment. To address this challenge, we propose EnvGraph, a framework for repository-level code generation that formulates repository executability as an environment alignment problem. EnvGraph jointly models two coupled conditions for successful repository execution, namely external dependency satisfaction and repository-internal reference resolution. It maintains a dual-layer environment representation, uses execution evidence to perform execution-evidence-based attribution, and guides repository generation through a unified targeted revision mechanism within an iterative alignment loop. We evaluate EnvGraph on repository-level code generation with three representative backbone LLMs and compare it against representative environment-aware and repository-level baselines. Experimental results show that EnvGraph consistently achieves the best performance on these repository-level benchmarks. In particular, it outperforms the strongest non-EnvGraph baseline by an absolute margin of 5.72--5.87 percentage points in Functional Correctness and 4.58--8.66 percentage points in Non-Functional Quality. 

---
# Neural Global Optimization via Iterative Refinement from Noisy Samples 

**Authors**: Qusay Muzaffar, David Levin, Michael Werman  

**Link**: [PDF](https://arxiv.org/pdf/2604.03614)  

**Abstract**: Global optimization of black-box functions from noisy samples is a fundamental challenge in machine learning and scientific computing. Traditional methods such as Bayesian Optimization often converge to local minima on multi-modal functions, while gradient-free methods require many function evaluations. We present a novel neural approach that learns to find global minima through iterative refinement. Our model takes noisy function samples and their fitted spline representation as input, then iteratively refines an initial guess toward the true global minimum. Trained on randomly generated functions with ground truth global minima obtained via exhaustive search, our method achieves a mean error of 8.05 percent on challenging multi-modal test functions, compared to 36.24 percent for the spline initialization, a 28.18 percent improvement. The model successfully finds global minima in 72 percent of test cases with error below 10 percent, demonstrating learned optimization principles rather than mere curve fitting. Our architecture combines encoding of multiple modalities including function values, derivatives, and spline coefficients with iterative position updates, enabling robust global optimization without requiring derivative information or multiple restarts. 

---
# Multi-Robot Multi-Queue Control via Exhaustive Assignment Actor-Critic Learning 

**Authors**: Mohammad Merati, H. M. Sabbir Ahmad, Wenchao Li, David Castañón  

**Link**: [PDF](https://arxiv.org/pdf/2604.03605)  

**Abstract**: We study online task allocation for multi-robot, multi-queue systems with asymmetric stochastic arrivals and switching delays. We formulate the problem in discrete time: each location can host at most one robot per slot, servicing a task consumes one slot, switching between locations incurs a one-slot travel delay, and arrivals at locations are independent Bernoulli processes with heterogeneous rates. Building on our previous structural result that optimal policies are of exhaustive type, we formulate a discounted-cost Markov decision process and develop an exhaustive-assignment actor-critic policy architecture that enforces exhaustive service by construction and learns only the next-queue allocation for idle robots. Unlike the exhaustive-serve-longest (ESL) queue rule, whose optimality is known only under symmetry, the proposed policy adapts to asymmetry in arrival rates. Across different server-location ratios, loads, and asymmetric arrival profiles, the proposed policy consistently achieves lower discounted holding cost and smaller mean queue length than the ESL baseline, while remaining near-optimal on instances where an optimal benchmark is available. These results show that structure-aware actor-critic methods provide an effective approach for real-time multi-robot scheduling. 

---
# Unveiling Language Routing Isolation in Multilingual MoE Models for Interpretable Subnetwork Adaptation 

**Authors**: Kening Zheng, Wei-Chieh Huang, Jiahao Huo, Zhonghao Li, Henry Peng Zou, Yibo Yan, Xin Zou, Jungang Li, Junzhuo Li, Hanrong Zhang, Xuming Hu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03592)  

**Abstract**: Mixture-of-Experts (MoE) models exhibit striking performance disparities across languages, yet the internal mechanisms driving these gaps remain poorly understood. In this work, we conduct a systematic analysis of expert routing patterns in MoE models, revealing a phenomenon we term Language Routing Isolation, in which high- and low-resource languages tend to activate largely disjoint expert sets. Through layer-stratified analysis, we further show that routing patterns exhibit a layer-wise convergence-divergence pattern across model depth. Building on these findings, we propose RISE (Routing Isolation-guided Subnetwork Enhancement), a framework that exploits routing isolation to identify and adapt language-specific expert subnetworks. RISE applies a tripartite selection strategy, using specificity scores to identify language-specific experts in shallow and deep layers and overlap scores to select universal experts in middle layers. By training only the selected subnetwork while freezing all other parameters, RISE substantially improves low-resource language performance while preserving capabilities in other languages. Experiments on 10 languages demonstrate that RISE achieves target-language F1 gains of up to 10.85% with minimal cross-lingual degradation. 

---
# SecPI: Secure Code Generation with Reasoning Models via Security Reasoning Internalization 

**Authors**: Hao Wang, Niels Mündler, Mark Vero, Jingxuan He, Dawn Song, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2604.03587)  

**Abstract**: Reasoning language models (RLMs) are increasingly used in programming. Yet, even state-of-the-art RLMs frequently introduce critical security vulnerabilities in generated code. Prior training-based approaches for secure code generation face a critical limitation that prevents their direct application to RLMs: they rely on costly, manually curated security datasets covering only a limited set of vulnerabilities. At the inference level, generic security reminders consistently degrade functional correctness while triggering only shallow ad-hoc vulnerability analysis. To address these problems, we present SecPI, a fine-tuning pipeline that teaches RLMs to internalize structured security reasoning, producing secure code by default without any security instructions at inference time. SecPI filters existing general-purpose coding datasets for security-relevant tasks using an LLM-based classifier, generates high-quality security reasoning traces with a teacher model guided by a structured prompt that systematically enumerates relevant CWEs and mitigations, and fine-tunes the target model on pairs of inputs with no security prompt and teacher reasoning traces -- as a result, the model learns to reason about security autonomously rather than in response to explicit instructions. An extensive evaluation on security benchmarks with state-of-the-art open-weight reasoning models validates the effectiveness of our approach. For instance, SecPI improves the percentage of functionally correct and secure generations for QwQ 32B from 48.2% to 62.2% (+14.0 points) on CWEval and from 18.2% to 22.0% on BaxBench. Further investigation also reveals strong cross-CWE and cross-language generalization beyond training vulnerabilities. Even when trained only on injection-related CWEs, QwQ 32B generates correct and secure code 9.9% more frequently on held-out memory-safety CWEs. 

---
# Focus Matters: Phase-Aware Suppression for Hallucination in Vision-Language Models 

**Authors**: Sohyeon Kim, Sang Yeon Yoon, Kyeongbo Kong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03556)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved impressive progress in multimodal reasoning, yet they remain prone to object hallucinations, generating descriptions of objects that are not present in the input image. Recent approaches attempt to mitigate hallucinations by suppressing unreliable visual signals in the vision encoder, but many rely on iterative optimization for each input, resulting in substantial inference latency. In this work, we investigate the internal attention dynamics of vision encoders in LVLMs and identify a consistent three-phase structure of visual information processing: diffusion, focus, and rediffusion. Our analysis reveals that hallucination behavior is particularly sensitive to tokens receiving low attention during the focus phase. Motivated by this observation, we propose a lightweight inference-time intervention that selectively suppresses such tokens during the focus phase. The method operates in a training-free manner using statistics from a single forward pass and employs a Determinantal Point Process (DPP) to preserve diverse visual cues while filtering redundant tokens. Extensive experiments across multiple LVLM backbones and decoding strategies demonstrate that the proposed approach consistently reduces hallucination metrics while maintaining competitive caption quality. Moreover, compared to adversarial uncertainty estimation methods, our approach achieves comparable hallucination mitigation with negligible additional inference latency. 

---
# CRAFT: Video Diffusion for Bimanual Robot Data Generation 

**Authors**: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2604.03552)  

**Abstract**: Bimanual robot learning from demonstrations is fundamentally limited by the cost and narrow visual diversity of real-world data, which constrains policy robustness across viewpoints, object configurations, and embodiments. We present Canny-guided Robot Data Generation using Video Diffusion Transformers (CRAFT), a video diffusion-based framework for scalable bimanual demonstration generation that synthesizes temporally coherent manipulation videos while producing action labels. By conditioning video diffusion on edge-based structural cues extracted from simulator-generated trajectories, CRAFT produces physically plausible trajectory variations and supports a unified augmentation pipeline spanning object pose changes, camera viewpoints, lighting and background variations, cross-embodiment transfer, and multi-view synthesis. We leverage a pre-trained video diffusion model to convert simulated videos, along with action labels from the simulation trajectories, into action-consistent demonstrations. Starting from only a few real-world demonstrations, CRAFT generates a large, visually diverse set of photorealistic training data, bypassing the need to replay demonstrations on the real robot (Sim2Real). Across simulated and real-world bimanual tasks, CRAFT improves success rates over existing augmentation strategies and straightforward data scaling, demonstrating that diffusion-based video generation can substantially expand demonstration diversity and improve generalization for dual-arm manipulation tasks. Our project website is available at: this https URL 

---
# AgenticFlict: A Large-Scale Dataset of Merge Conflicts in AI Coding Agent Pull Requests on GitHub 

**Authors**: Daniel Ogenrwot, John Businge  

**Link**: [PDF](https://arxiv.org/pdf/2604.03551)  

**Abstract**: Software Engineering 3.0 marks a paradigm shift in software development, in which AI coding agents are no longer just assistive tools but active contributors. While prior empirical studies have examined productivity gains and acceptance patterns in AI-assisted development, the challenges associated with integrating agent-generated contributions remain less understood. In particular, merge conflicts, a fundamental aspect of collaborative software development, remain underexplored in this context. In this paper, we present AgenticFlict, a large-scale dataset of textual merge conflicts in AI coding agent pull requests (Agentic PRs). The dataset comprises 142K+ Agentic PRs collected from 59K+ repositories, of which 107K+ are successfully processed through deterministic merge simulation. Our pipeline identifies 29K+ PRs exhibiting merge conflicts, yielding a conflict rate of 27.67%, and extracts 336K+ fine-grained conflict regions across these instances. Our preliminary exploratory analysis indicates that merge conflicts are both frequent and often substantial in AI-generated contributions, with noticeable variation across agents, emphasizing the need to better understand and manage integration challenges in AI-assisted software development. The dataset, code and supplementary materials are available in zenodo: this https URL. 

---
# LangFIR: Discovering Sparse Language-Specific Features from Monolingual Data for Language Steering 

**Authors**: Sing Hieng Wong, Hassan Sajjad, A.B. Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2604.03532)  

**Abstract**: Large language models (LLMs) show strong multilingual capabilities, yet reliably controlling the language of their outputs remains difficult. Representation-level steering addresses this by adding language-specific vectors to model activations at inference time, but identifying language-specific directions in the residual stream often relies on multilingual or parallel data that can be expensive to obtain. Sparse autoencoders (SAEs) decompose residual activations into interpretable, sparse feature directions and offer a natural basis for this search, yet existing SAE-based approaches face the same data constraint. We introduce LangFIR (Language Feature Identification via Random-token Filtering), a method that discovers language-specific SAE features using only a small amount of monolingual data and random-token sequences. Many SAE features consistently activated by target-language inputs do not encode language identity. Random-token sequences surface these language-agnostic features, allowing LangFIR to filter them out and isolate a sparse set of language-specific features. We show that these features are extremely sparse, highly selective for their target language, and causally important: directional ablation increases cross-entropy loss only for the corresponding language. Using these features to construct steering vectors for multilingual generation control, LangFIR achieves the best average accuracy BLEU across three models (Gemma 3 1B, Gemma 3 4B, and Llama 3.1 8B), three datasets, and twelve target languages, outperforming the strongest monolingual baseline by up to and surpassing methods that rely on parallel data. Our results suggest that language identity in multilingual LLMs is localized in a sparse set of feature directions discoverable with monolingual data. Code is available at this https URL. 

---
# Incentives shape how humans co-create with generative AI 

**Authors**: Nathanael Jo, Manish Raghavan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03529)  

**Abstract**: Generative AI is quickly becoming an integral part of people's everyday workflows. Early evidence has shown that while generative AI can increase individual-level productivity, it does so at the cost of collective diversity, potentially narrowing the set of ideas and perspectives produced. Our research stands in contrast to this concern: through a pre-registered randomized control trial, we show that incentives mediate AI's homogenizing force in a creative writing task where participants can use AI interactively. Participants rewarded for originality relative to peers produce collectively more diverse writing than those rewarded for quality alone. This divergence is driven not by abandoning AI, but by how participants use it: those incentivized for originality incorporate fewer AI suggestions verbatim, relying on the model more selectively for brainstorming, proofreading, and targeted edits. Our results reveal that the effects of generative AI depend not only on the technology itself, but also the behavioral strategies and incentive structures surrounding its use. 

---
# Determined by User Needs: A Salient Object Detection Rationale Beyond Conventional Visual Stimuli 

**Authors**: Chenglizhao Chen, Shujian Zhang, Luming Li, Wenfeng Song, Shuai Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03526)  

**Abstract**: Existing \textbf{s}alient \textbf{o}bject \textbf{d}etection (SOD) methods adopt a \textbf{passive} visual stimulus-based rationale--objects with the strongest visual stimuli are perceived as the user's primary focus (i.e., salient objects). They ignore the decisive role of users' \textbf{proactive needs} in segmenting salient objects--if a user has a need before seeing an image, the user's salient objects align with their needs, e.g., if a user's need is ``white apple'', when this user sees an image, the user's primary focus is on the ``white apple'' or ``the most white apple-like'' objects in the image. Such an oversight not only \textbf{fails to satisfy users}, but also \textbf{limits the development of downstream tasks}. For instance, in salient object ranking tasks, focusing solely on visual stimuli-based salient objects is insufficient for conducting an analysis of fine-grained relationships between users' viewing order (usually determined by user's needs) and scenes, which may result in wrong ranking results. Clearly, it is essential to detect salient objects based on user needs. Thus, we advocate a \textbf{User} \textbf{S}alient \textbf{O}bject \textbf{D}etection (UserSOD) task, which focuses on \textbf{detecting salient objects align with users' proactive needs when user have needs}. The main challenge for this new task is the lack of datasets for model training and testing. 

---
# Optimizing Neurorobot Policy under Limited Demonstration Data through Preference Regret 

**Authors**: Viet Dung Nguyen, Yuhang Song, Anh Nguyen, Jamison Heard, Reynold Bailey, Alexander Ororbia  

**Link**: [PDF](https://arxiv.org/pdf/2604.03523)  

**Abstract**: Robot reinforcement learning from demonstrations (RLfD) assumes that expert data is abundant; this is usually unrealistic in the real world given data scarcity as well as high collection cost. Furthermore, imitation learning algorithms assume that the data is independently and identically distributed, which ultimately results in poorer performance as gradual errors emerge and compound within test-time trajectories. We address these issues by introducing the "master your own expertise" (MYOE) framework, a self-imitation framework that enables robotic agents to learn complex behaviors from limited demonstration data samples. Inspired by human perception and action, we propose and design what we call the queryable mixture-of-preferences state space model (QMoP-SSM), which estimates the desired goal at every time step. These desired goals are used in computing the "preference regret", which is used to optimize the robot control policy. Our experiments demonstrate the robustness, adaptability, and out-of-sample performance of our agent compared to other state-of-the-art RLfD schemes. The GitHub repository that supports this work can be found at: this https URL. 

---
# Inside the Scaffold: A Source-Code Taxonomy of Coding Agent Architectures 

**Authors**: Benjamin Rombaut  

**Link**: [PDF](https://arxiv.org/pdf/2604.03515)  

**Abstract**: LLM-based coding agents can localize bugs, generate patches, and run tests with diminishing human oversight, yet the scaffolding code that surrounds the language model (the control loop, tool definitions, state management, and context strategy) remains poorly understood. Existing surveys classify agents by abstract capabilities (tool use, planning, reflection) that cannot distinguish between architecturally distinct systems, and trajectory studies observe what agents do without examining the scaffold code that determines why. This paper presents a source-code-level architectural taxonomy derived from analysis of 13 open-source coding agent scaffolds at pinned commit hashes. Each agent is characterized across 12 dimensions organized into three layers: control architecture, tool and environment interface, and resource management. The analysis reveals that scaffold architectures resist discrete classification: control strategies range from fixed pipelines to Monte Carlo Tree Search, tool counts range from 0 to 37, and context compaction spans seven distinct strategies. Five loop primitives (ReAct, generate-test-repair, plan-execute, multi-attempt retry, tree search) function as composable building blocks that agents layer in different combinations; 11 of 13 agents compose multiple primitives rather than relying on a single control structure. Dimensions converge where external constraints dominate (tool capability categories, edit formats, execution isolation) and diverge where open design questions remain (context compaction, state management, multi-model routing). All taxonomic claims are grounded in file paths and line numbers, providing a reusable reference for researchers studying agent behavior and practitioners designing new scaffolds. 

---
# The Augmentation Trap: AI Productivity and the Cost of Cognitive Offloading 

**Authors**: Michael Caosun, Sinan Aral  

**Link**: [PDF](https://arxiv.org/pdf/2604.03501)  

**Abstract**: Experimental evidence confirms that AI tools raise worker productivity, but also that sustained use can erode the expertise on which those gains depend. We develop a dynamic model in which a decision-maker chooses AI usage intensity for a worker over time, trading immediate productivity against the erosion of worker skill. We decompose the tool's productivity effect into two channels, one independent of worker expertise and one that scales with it. The model produces three main results. First, even a decision-maker who fully anticipates skill erosion rationally adopts AI when front-loaded productivity gains outweigh long-run skill costs, producing steady-state loss: the worker ends up less productive than before adoption. Second, when managers are short-termist or worker skill has external value, the decision-maker's optimal policy turns steady-state loss into the augmentation trap, leaving the worker worse off than if AI had never been adopted. Third, when AI productivity depends less on worker expertise, workers can permanently diverge in skill: experienced workers realize their full potential while less experienced workers deskill to zero. Small differences in managerial incentives can determine which path a worker takes. The productivity decomposition classifies deployments into five regimes that separate beneficial adoption from harmful adoption and identifies which deployments are vulnerable to the trap. 

---
# Sim2Real-AD: A Modular Sim-to-Real Framework for Deploying VLM-Guided Reinforcement Learning in Real-World Autonomous Driving 

**Authors**: Zilin Huang, Zhengyang Wan, Zihao Sheng, Boyue Wang, Junwei You, Yue Leng, Sikai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03497)  

**Abstract**: Deploying reinforcement learning policies trained in simulation to real autonomous vehicles remains a fundamental challenge, particularly for VLM-guided RL frameworks whose policies are typically learned with simulator-native observations and simulator-coupled action semantics that are unavailable on physical platforms. This paper presents Sim2Real-AD, a modular framework for zero-shot sim-to-real transfer of CARLA-trained VLM-guided RL policies to full-scale vehicles without any real-world RL training data. The framework decomposes the transfer problem into four components: a Geometric Observation Bridge (GOB) that converts monocular front-view images into simulator-compatible bird's-eye-view (BEV) observations, a Physics-Aware Action Mapping (PAM) that translates policy outputs into platform-agnostic physical commands, a Two-Phase Progressive Training (TPT) strategy that stabilizes adaptation by separating action-space and observation-space transfer, and a Real-time Deployment Pipeline (RDP) that integrates perception, policy inference, control conversion, and safety monitoring for closed-loop execution. Simulation experiments show that the framework preserves the relative performance ordering of representative RL algorithms across different reward paradigms and validate the contribution of each module. Zero-shot deployment on a full-scale Ford E-Transit achieves success rates of 90%, 80%, and 75% in car-following, obstacle avoidance, and stop-sign interaction scenarios, respectively. To the best of our knowledge, this study is among the first to demonstrate zero-shot closed-loop deployment of a CARLA-trained VLM-guided RL policy on a full-scale real vehicle without any real-world RL training data. The demo video and code are available at: this https URL. 

---
# VisionClaw: Always-On AI Agents through Smart Glasses 

**Authors**: Xiaoan Liu, DaeHo Lee, Eric J Gonzalez, Mar Gonzalez-Franco, Ryo Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2604.03486)  

**Abstract**: We present VisionClaw, an always-on wearable AI agent that integrates live egocentric perception with agentic task execution. Running on Meta Ray-Ban smart glasses, VisionClaw continuously perceives real-world context and enables in-situ, speech-driven action initiation and delegation via OpenClaw AI agents. Therefore, users can directly execute tasks through the smart glasses, such as adding real-world objects to an Amazon cart, generating notes from physical documents, receiving meeting briefings on the go, creating events from posters, or controlling IoT devices. We evaluate VisionClaw through a controlled laboratory study (N=12) and a longitudinal deployment study (N=5). Results show that integrating perception and execution enables faster task completion and reduces interaction overhead compared to non-always-on and non-agent baselines. Beyond performance gains, deployment findings reveal a shift in interaction: tasks are initiated opportunistically during ongoing activities, and execution is increasingly delegated rather than manually controlled. These results suggest a new paradigm for wearable AI agents, where perception and action are continuously coupled to support situated, hands-free interaction. 

---
# Large Language Models Align with the Human Brain during Creative Thinking 

**Authors**: Mete Ismayilzada, Simone A. Luchini, Abdulkadir Gokce, Badr AlKhamissi, Antoine Bosselut, Antonio Laverghetta Jr., Lonneke van der Plas, Roger E. Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2604.03480)  

**Abstract**: Creative thinking is a fundamental aspect of human cognition, and divergent thinking-the capacity to generate novel and varied ideas-is widely regarded as its core generative engine. Large language models (LLMs) have recently demonstrated impressive performance on divergent thinking tests and prior work has shown that models with higher task performance tend to be more aligned to human brain activity. However, existing brain-LLM alignment studies have focused on passive, non-creative tasks. Here, we explore brain alignment during creative thinking using fMRI data from 170 participants performing the Alternate Uses Task (AUT). We extract representations from LLMs varying in size (270M-72B) and measure alignment to brain responses via Representational Similarity Analysis (RSA), targeting the creativity-related default mode and frontoparietal networks. We find that brain-LLM alignment scales with model size (default mode network only) and idea originality (both networks), with effects strongest early in the creative process. We further show that post-training objectives shape alignment in functionally selective ways: a creativity-optimized \texttt{Llama-3.1-8B-Instruct} preserves alignment with high-creativity neural responses while reducing alignment with low-creativity ones; a human behavior fine-tuned model elevates alignment with both; and a reasoning-trained variant shows the opposite pattern, suggesting chain-of-thought training steers representations away from creative neural geometry toward analytical processing. These results demonstrate that post-training objectives selectively reshape LLM representations relative to the neural geometry of human creative thought. 

---
# Fine-tuning DeepSeek-OCR-2 for Molecular Structure Recognition 

**Authors**: Haocheng Tang, Xingyu Dang, Junmei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03476)  

**Abstract**: Optical Chemical Structure Recognition (OCSR) is critical for converting 2D molecular diagrams from printed literature into machine-readable formats. While Vision-Language Models have shown promise in end-to-end OCR tasks, their direct application to OCSR remains challenging, and direct full-parameter supervised fine-tuning often fails. In this work, we adapt DeepSeek-OCR-2 for molecular optical recognition by formulating the task as image-conditioned SMILES generation. To overcome training instabilities, we propose a two-stage progressive supervised fine-tuning strategy: starting with parameter-efficient LoRA and transitioning to selective full-parameter fine-tuning with split learning rates. We train our model on a large-scale corpus combining synthetic renderings from PubChem and realistic patent images from USPTO-MOL to improve coverage and robustness. Our fine-tuned model, MolSeek-OCR, demonstrates competitive capabilities, achieving exact matching accuracies comparable to the best-performing image-to-sequence model. However, it remains inferior to state-of-the-art image-to-graph modelS. Furthermore, we explore reinforcement-style post-training and data-curation-based refinement, finding that they fail to improve the strict sequence-level fidelity required for exact SMILES matching. 

---
# Evolutionary Search for Automated Design of Uncertainty Quantification Methods 

**Authors**: Mikhail Seleznyov, Daniil Korbut, Viktor Moskvoretskii, Oleg Somov, Alexander Panchenko, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2604.03473)  

**Abstract**: Uncertainty quantification (UQ) methods for large language models are predominantly designed by hand based on domain knowledge and heuristics, limiting their scalability and generality. We apply LLM-powered evolutionary search to automatically discover unsupervised UQ methods represented as Python programs. On the task of atomic claim verification, our evolved methods outperform strong manually-designed baselines, achieving up to 6.7% relative ROC-AUC improvement across 9 datasets while generalizing robustly out-of-distribution. Qualitative analysis reveals that different LLMs employ qualitatively distinct evolutionary strategies: Claude models consistently design high-feature-count linear estimators, while Gpt-oss-120B gravitates toward simpler and more interpretable positional weighting schemes. Surprisingly, only Sonnet 4.5 and Opus 4.5 reliably leverage increased method complexity to improve performance -- Opus 4.6 shows an unexpected regression relative to its predecessor. Overall, our results indicate that LLM-powered evolutionary search is a promising paradigm for automated, interpretable hallucination detector design. 

---
# Vocabulary Dropout for Curriculum Diversity in LLM Co-Evolution 

**Authors**: Jacob Dineen, Aswin RRV, Zhikun Xu, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03472)  

**Abstract**: Co-evolutionary self-play, where one language model generates problems and another solves them, promises autonomous curriculum learning without human supervision. In practice, the proposer quickly converges to a narrow distribution of problems that satisfy the reward function. This diversity collapse renders the curriculum uninformative for the solver, stalling the co-evolutionary loop. We introduce vocabulary dropout, a random mask applied to the proposer's output logits during both policy training and curriculum generation, as a lightweight mechanism to sustain diversity. The mask is hard and non-stationary, preventing the proposer from locking into fixed token sequences. Training Qwen3-4B and Qwen3-8B on mathematical reasoning via R-Zero, we find that vocabulary dropout sustains proposer diversity across lexical, semantic, and functional metrics throughout training, and yields solver improvements averaging +4.4 points at 8B, with the largest gains on competition-level benchmarks. Our findings suggest that explicit action-space constraints, analogous to the structural role that game rules play in classical self-play, can help sustain productive co-evolution in language. Vocabulary dropout is one simple instantiation of this principle. 

---
# RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation 

**Authors**: Ganlin Feng, Yuxi Long, Hafsa Ali, Erin Lou, Fahad Butt, Qian Liu, Yang Wang, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03454)  

**Abstract**: Rare diseases often manifest with distinctive facial phenotypes in children, offering valuable diagnostic cues for clinicians and AI-assisted screening systems. However, progress in this field is severely limited by the scarcity of curated, ethically sourced facial data and the high similarity among phenotypes across different conditions. To address these challenges, we introduce RDFace, a curated benchmark dataset comprising 456 pediatric facial images spanning 103 rare genetic conditions (average 4.4 samples per condition). Each ethically verified image is paired with standardized metadata. RDFace enables the development and evaluation of data-efficient AI models for rare disease diagnosis under real-world low-data constraints. We benchmark multiple pretrained vision backbones using cross-validation and explore synthetic augmentation with DreamBooth and FastGAN. Generated images are filtered via facial landmark similarity to maintain phenotype fidelity and merged with real data, improving diagnostic accuracy by up to 13.7% in ultra-low-data regimes. To assess semantic validity, phenotype descriptions generated by a vision-language model from real and synthetic images achieve a report similarity score of 0.84. RDFace establishes a transparent, benchmark-ready dataset for equitable rare disease AI research and presents a scalable framework for evaluating both diagnostic performance and the integrity of synthetic medical imagery. 

---
# ExpressEdit: Fast Editing of Stylized Facial Expressions with Diffusion Models in Photoshop 

**Authors**: Kenan Tang, Jiasheng Guo, Jeffrey Lin, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03448)  

**Abstract**: Facial expressions of characters are a vital component of visual storytelling. While current AI image editing models hold promise for assisting artists in the task of stylized expression editing, these models introduce global noise and pixel drift into the edited image, preventing the integration of these models into professional image editing software and workflows. To bridge this gap, we introduce ExpressEdit, a fully open-source Photoshop plugin that is free from common artifacts of proprietary image editing models and robustly synergizes with native Photoshop operations such as Liquify. ExpressEdit seamlessly edits an expression within 3 seconds on a single consumer-grade GPU, significantly faster than popular proprietary models. Moreover, to support the generation of diverse expressions according to different narrative needs, we compile a comprehensive expression database of 135 expression tags enriched with example stories and images designed for retrieval-augmented generation. We open source the code and dataset to facilitate future research and artistic exploration. 

---
# Measuring LLM Trust Allocation Across Conflicting Software Artifacts 

**Authors**: Noshin Ulfat, Ahsanul Ameen Sabit, Soneya Binta Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2604.03447)  

**Abstract**: LLM-based software engineering assistants fail not only by producing incorrect outputs, but also by allocating trust to the wrong artifact when code, documentation, and tests disagree. Existing evaluations focus mainly on downstream outcomes and therefore cannot reveal whether a model recognized degraded evidence, identified the unreliable source, or calibrated its trust across artifacts. We present TRACE (Trust Reasoning over Artifacts for Calibrated Evaluation), a framework that elicits structured artifact-level trust traces over Javadoc, method signatures, implementations, and test prefixes under blind perturbations. Using 22,339 valid traces from seven models on 456 curated Java method bundles, we evaluate per-artifact quality assessment, inconsistency detection, affected artifact attribution, and source prioritization. Across all models, quality penalties are largely localized to the perturbed artifact and increase with severity, but sensitivity is asymmetric across artifact types: documentation bugs induce a substantially larger heavy-to-subtle gap than implementation faults (0.152-0.253 vs. 0.049-0.123). Models detect explicit documentation bugs well (67-94%) and Javadoc and implementation contradictions at 50-91%, yet show a systematic blind spot when only the implementation drifts while the documentation remains plausible, with detection dropping by 7-42 percentage points. Confidence is poorly calibrated for six of seven models. These findings suggest that current LLMs are better at auditing natural-language specifications than at detecting subtle code-level drift, motivating explicit artifact-level trust reasoning before correctness-critical downstream use. 

---
# Agile Story-Point Estimation: Is RAG a Better Way to Go? 

**Authors**: Lamyea Maha, Tajmilur Rahman, Chanchal Roy  

**Link**: [PDF](https://arxiv.org/pdf/2604.03443)  

**Abstract**: The sprint-based iterative approach in the Agile software development method allows continuous feedback and adaptation. One of the crucial Agile software development activities is the sprint planning session where developers estimate the effort required to complete tasks through a consensus-based estimation technique such as Planning Poker. In the Agile software development method, a common unit of measuring development effort is Story Point (SP) which is assigned to tasks to understand the complexity and development time needed to complete them. Despite the benefits of this process, it is an extremely time-consuming manual process. To mitigate this issue, in this study, we investigated if this manual process can be automated using Retrieval Augmented Generation (RAG) which comprises a "Retriever" and a "Generator". We applied two embedding models - bge-large-en-v1.5, and Sentence-Transformers' all-mpnet-base-v2 on 23 open-source software projects of varying sizes and examined four key aspects: 1) how retrieval hyper-parameters influence the performance, 2) whether estimation accuracy differs across different sizes of the projects, 3) whether embedding model choice affects accuracy, and 4) how the RAG-based approach compares to the existing baselines. Although the RAG-based approach outperformed the baseline models in several occasions, our results did not exhibit statistically significant differences in performance across the projects or across the embedding models. This highlights the need for further studies and refinement of the RAG, and model adaptation strategies for better accuracy in automatically estimating user stories. 

---
# MetaSAEs: Joint Training with a Decomposability Penalty Produces More Atomic Sparse Autoencoder Latents 

**Authors**: Matthew Levinson  

**Link**: [PDF](https://arxiv.org/pdf/2604.03436)  

**Abstract**: Sparse autoencoders (SAEs) are increasingly used for safety-relevant applications including alignment detection and model steering. These use cases require SAE latents to be as atomic as possible. Each latent should represent a single coherent concept drawn from a single underlying representational subspace. In practice, SAE latents blend representational subspaces together. A single feature can activate across semantically distinct contexts that share no true common representation, muddying an already complex picture of model computation. We introduce a joint training objective that directly penalizes this subspace blending. A small meta SAE is trained alongside the primary SAE to sparsely reconstruct the primary SAE's decoder columns; the primary SAE is penalized whenever its decoder directions are easy to reconstruct from the meta dictionary. This occurs whenever latent directions lie in a subspace spanned by other primary directions. This creates gradient pressure toward more mutually independent decoder directions that resist sparse meta-compression.
On GPT-2 large (layer 20), the selected configuration reduces mean $|\varphi|$ by 7.5% relative to an identical solo SAE trained on the same data. Automated interpretability (fuzzing) scores improve by 7.6%, providing external validation of the atomicity gain independent of the training and co-occurrence metrics. Reconstruction overhead is modest. Results on Gemma 2 9B are directional. On not-fully-converged SAEs, the same parameterization yields the best results, a $+8.6\%$ $\Delta$Fuzz. Though directional, this is an encouraging sign that the method transfers to a larger model. Qualitative analysis confirms that features firing on polysemantic tokens are split into semantically distinct sub-features, each specializing in a distinct representational subspace. 

---
# Inference-Path Optimization via Circuit Duplication in Frozen Visual Transformers for Marine Species Classification 

**Authors**: Thomas Manuel Rost  

**Link**: [PDF](https://arxiv.org/pdf/2604.03428)  

**Abstract**: Automated underwater species classification is constrained by annotation cost and environmental variation that limits the transferability of fully supervised models. Recent work has shown that frozen embeddings from self-supervised vision foundation models already provide a strong label-efficient baseline for marine image classification. Here we investigate whether this frozen-embedding regime can be improved at inference time, without fine-tuning or changing model weights.
We apply Circuit Duplication, an inference-time method originally proposed for Large Language Models, in which a selected range of transformer layers is traversed twice during the forward pass. We evaluate on the class-imbalanced AQUA20 benchmark using frozen DINOv3 embeddings under two settings: global circuit selection, where a single duplicated circuit is chosen for the full dataset, and class-specific circuit selection, where each species may receive a different optimal circuit. Both settings use simple semi-supervised downstream classifiers.
Circuit Duplication consistently improves over the standard frozen forward pass. At the maximum label budget, class-specific selection reaches a macro F1 of 0.875, closing the gap to the fully supervised ConvNeXt benchmark (0.889) to 1.4 points without any gradient-based training. Four species exceed their fully supervised reference, with octopus improving by +12.1 F1 points. Across all budgets, roughly 75% of classes prefer a class-specific circuit, indicating a genuinely class-dependent benefit. To our knowledge, this is the first application of Circuit Duplication to computer vision. 

---
# AEGIS: Scaling Long-Sequence Homomorphic Encrypted Transformer Inference via Hybrid Parallelism on Multi-GPU Systems 

**Authors**: Zhaoting Gong, Ran Ran, Fan Yao, Wujie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03425)  

**Abstract**: Fully Homomorphic Encryption (FHE) enables privacy-preserving Transformer inference, but long-sequence encrypted Transformers quickly exceed single-GPU memory capacity because encoded weights are already large and encrypted activations grow rapidly with sequence length. Multi-GPU execution therefore becomes unavoidable, yet scaling remains challenging because communication is jointly induced by application-level aggregation and encryption-level RNS coupling. Existing approaches either synchronize between devices frequently or replicate encrypted tensors across devices, leading to excessive communication and latency.
We present AEGIS, an Application-Encryption Guided Inference System for scalable long-sequence encrypted Transformer inference on multi-GPU platforms. AEGIS derives device placement from ciphertext dependencies jointly induced by Transformer dataflow and CKKS polynomial coupling, co-locating modulus-coherent and token-coherent data so that communication is introduced only when application dependencies require it, while reordering polynomial operators to overlap the remaining collectives with computation.
On 2048-token inputs, AEGIS reduces inter-GPU communication by up to 57.9% in feed-forward networks and 81.3% in self-attention versus prior state-of-the-art designs. On four GPUs, it achieves up to 96.62% scaling efficiency, 3.86x end-to-end speedup, and 69.1% per-device memory reduction. These results establish coordinated application-encryption parallelism as a practical foundation for scalable homomorphic Transformer inference. 

---
# Zero-Shot Quantization via Weight-Space Arithmetic 

**Authors**: Daniele Solombrino, Antonio Andrea Gargiulo, Adrian Robert Minut, Luca Zhou, Alessandro Zirilli, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2604.03420)  

**Abstract**: We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve robustness to PTQ-induced noise by as much as 60%, without receiver-side quantization-aware training (QAT). Because the method requires no receiver training data, it provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. We demonstrate this on Vision Transformer (ViT) models. More broadly, our results suggest that quantization robustness is not merely a byproduct of task-specific training, but a reusable feature of weight-space geometry that can be transferred rather than retrained. 

---
# Generative AI for material design: A mechanics perspective from burgers to matter 

**Authors**: Vahidullah Tac, Ellen Kuhl  

**Link**: [PDF](https://arxiv.org/pdf/2604.03409)  

**Abstract**: Generative artificial intelligence offers a new paradigm to design matter in high-dimensional spaces. However, its underlying mechanisms remain difficult to interpret and limit adoption in computational mechanics. This gap is striking because its core tools-diffusion, stochastic differential equations, and inverse problems-are fundamental to the mechanics of materials. Here we show that diffusion-based generative AI and computational mechanics are rooted in the same principles. We illustrate this connection using a three-ingredient burger as a minimal benchmark for material design in a low-dimensional space, where both forward and reverse diffusion admit analytical solutions: Markov chains with Bayesian inversion in the discrete case and the Ornstein-Uhlenbeck process with score-based reversal in the continuous case. We extend this framework to a high-dimensional design space with 146 ingredients and 8.9x10^43 possible configurations, where analytical solutions become intractable. We therefore learn the discrete and continuous reverse processes using neural network models that infer inverse dynamics from data. We train the models on only 2,260 recipes and generate one million samples that capture the statistical structure of the data, including ingredient prevalence and quantitative composition. We further generate five new burgers and validate them in a restaurant-based sensory study with 100 participants, where three of the AI-designed burgers outperform the classical Big Mac in overall liking, flavor, and texture. These results establish diffusion-based generative modeling as a physically grounded approach to design in high-dimensional spaces. They position generative AI as a natural extension of computational mechanics, with applications from burgers to matter, and establish a path toward data-driven, physics-informed generative design. 

---
# Can LLMs Reason About Attention? Towards Zero-Shot Analysis of Multimodal Classroom Behavior 

**Authors**: Nolan Platt, Sehrish Nizamani, Alp Tural, Elif Tural, Saad Nizamani, Andrew Katz, Yoonje Lee, Nada Basit  

**Link**: [PDF](https://arxiv.org/pdf/2604.03401)  

**Abstract**: Understanding student engagement usually requires time-consuming manual observation or invasive recording that raises privacy concerns. We present a privacy-preserving pipeline that analyzes classroom videos to extract insights about student attention, without storing any identifiable footage. Our system runs on a single GPU, using OpenPose for skeletal extraction and Gaze-LLE for visual attention estimation. Original video frames are deleted immediately after pose extraction, thus only geometric coordinates (stored as JSON) are retained, ensuring compliance with FERPA. The extracted pose and gaze data is processed by QwQ-32B-Reasoning, which performs zero-shot analysis of student behavior across lecture segments. Instructors access results through a web dashboard featuring attention heatmaps and behavioral summaries. Our preliminary findings suggest that LLMs may show promise for multimodal behavior understanding, although they still struggle with spatial reasoning about classroom layouts. We discuss these limitations and outline directions for improving LLM spatial comprehension in educational analytics contexts. 

---
# Banana100: Breaking NR-IQA Metrics by 100 Iterative Image Replications with Nano Banana Pro 

**Authors**: Kenan Tang, Praveen Arunshankar, Andong Hua, Anthony Yang, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2604.03400)  

**Abstract**: The multi-step, iterative image editing capabilities of multi-modal agentic systems have transformed digital content creation. Although latest image editing models faithfully follow instructions and generate high-quality images in single-turn edits, we identify a critical weakness in multi-turn editing, which is the iterative degradation of image quality. As images are repeatedly edited, minor artifacts accumulate, rapidly leading to a severe accumulation of visible noise and a failure to follow simple editing instructions. To systematically study these failures, we introduce Banana100, a comprehensive dataset of 28,000 degraded images generated through 100 iterative editing steps, including diverse textures and image content. Alarmingly, image quality evaluators fail to detect the degradation. Among 21 popular no-reference image quality assessment (NR-IQA) metrics, none of them consistently assign lower scores to heavily degraded images than to clean ones. The dual failures of generators and evaluators may threaten the stability of future model training and the safety of deployed agentic systems, if the low-quality synthetic data generated by multi-turn edits escape quality filters. We release the full code and data to facilitate the development of more robust models, helping to mitigate the fragility of multi-modal agentic systems. 

---
# CresOWLve: Benchmarking Creative Problem-Solving Over Real-World Knowledge 

**Authors**: Mete Ismayilzada, Renqing Cuomao, Daniil Yurshevich, Anna Sotnikova, Lonneke van der Plas, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2604.03374)  

**Abstract**: Creative problem-solving requires combining multiple cognitive abilities, including logical reasoning, lateral thinking, analogy-making, and commonsense knowledge, to discover insights that connect seemingly unrelated pieces of information. However, most existing benchmarks for large language models (LLMs) evaluate only specific components of this process. Moreover, many creativity-oriented benchmarks rely on artificially constructed brainteasers or contrived scenarios that do not reflect how creative problem-solving occurs in real-world settings. To address this gap, we introduce CresOWLve, a benchmark for evaluating creative problem-solving using puzzles grounded in real-world knowledge. Problems in CresOWLve require employing multiple creative thinking strategies, retrieving facts from diverse domains, and creatively combining them to arrive at a solution. Evaluating several frontier non-thinking and thinking LLMs, we show that CresOWLve remains highly challenging. Our analysis reveals a consistent performance gap: models perform substantially better on factual questions than on creative ones (up to a -17% drop). While models can often retrieve the relevant knowledge, they struggle to form the non-obvious creative connections required to integrate this information and arrive at the correct answer. 

---
# From Model-Based Screening to Data-Driven Surrogates: A Multi-Stage Workflow for Exploring Stochastic Agent-Based Models 

**Authors**: Paul Saves, Matthieu Mastio, Nicolas Verstaevel, Benoit Gaudou  

**Link**: [PDF](https://arxiv.org/pdf/2604.03350)  

**Abstract**: Systematic exploration of Agent-Based Models (ABMs) is challenged by the curse of dimensionality and their inherent stochasticity. We present a multi-stage pipeline integrating the systematic design of experiments with machine learning surrogates. Using a predator-prey case study, our methodology proceeds in two steps. First, an automated model-based screening identifies dominant variables, assesses outcome variability, and segments the parameter space. Second, we train Machine Learning models to map the remaining nonlinear interaction effects. This approach automates the discovery of unstable regions where system outcomes are highly dependent on nonlinear interactions between many variables. Thus, this work provides modelers with a rigorous, hands-off framework for sensitivity analysis and policy testing, even when dealing with high-dimensional stochastic simulators. 

---
# Towards Intelligent Energy Security: A Unified Spatio-Temporal and Graph Learning Framework for Scalable Electricity Theft Detection in Smart Grids 

**Authors**: AbdulQoyum A. Olowookere, Usman A. Oguntola, Ebenezer. Leke Odekanle, Maridiyah A. Madehin, Aisha A. Adesope  

**Link**: [PDF](https://arxiv.org/pdf/2604.03344)  

**Abstract**: Electricity theft and non-technical losses (NTLs) remain critical challenges in modern smart grids, causing significant economic losses and compromising grid reliability. This study introduces the SmartGuard Energy Intelligence System (SGEIS), an integrated artificial intelligence framework for electricity theft detection and intelligent energy monitoring. The proposed system combines supervised machine learning, deep learning-based time-series modeling, Non-Intrusive Load Monitoring (NILM), and graph-based learning to capture both temporal and spatial consumption patterns. A comprehensive data processing pipeline is developed, incorporating feature engineering, multi-scale temporal analysis, and rule-based anomaly labeling. Deep learning models, including Long Short-Term Memory (LSTM), Temporal Convolutional Networks (TCN), and Autoencoders, are employed to detect abnormal usage patterns. In parallel, ensemble learning methods such as Random Forest, Gradient Boosting, XGBoost, and LightGBM are utilized for classification. To model grid topology and spatial dependencies, Graph Neural Networks (GNNs) are applied to identify correlated anomalies across interconnected nodes. The NILM module enhances interpretability by disaggregating appliance-level consumption from aggregate signals. Experimental results demonstrate strong performance, with Gradient Boosting achieving a ROC-AUC of 0.894, while graph-based models attain over 96% accuracy in identifying high-risk nodes. The hybrid framework improves detection robustness by integrating temporal, statistical, and spatial intelligence. Overall, SGEIS provides a scalable and practical solution for electricity theft detection, offering high accuracy, improved interpretability, and strong potential for real-world smart grid deployment. 

---
# Learning Additively Compositional Latent Actions for Embodied AI 

**Authors**: Hangxing Wei, Xiaoyu Chen, Chuheng Zhang, Tim Pearce, Jianyu Chen, Alex Lamb, Li Zhao, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2604.03340)  

**Abstract**: Latent action learning infers pseudo-action labels from visual transitions, providing an approach to leverage internet-scale video for embodied AI. However, most methods learn latent actions without structural priors that encode the additive, compositional structure of physical motion. As a result, latents often entangle irrelevant scene details or information about future observations with true state changes and miscalibrate motion magnitude. We introduce Additively Compositional Latent Action Model (AC-LAM), which enforces scene-wise additive composition structure over short horizons on the latent action space. These AC constraints encourage simple algebraic structure in the latent action space~(identity, inverse, cycle consistency) and suppress information that does not compose additively. Empirically, AC-LAM learns more structured, motion-specific, and displacement-calibrated latent actions and provides stronger supervision for downstream policy learning, outperforming state-of-the-art LAMs across simulated and real-world tabletop tasks. 

---
# The Ideation Bottleneck: Decomposing the Quality Gap Between AI-Generated and Human Economics Research 

**Authors**: Ning Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03338)  

**Abstract**: Autonomous AI systems can now generate complete economics research papers, but they substantially underperform human-authored publications in head-to-head comparisons. This paper decomposes the quality gap into two independent components: research idea quality and execution quality. Using a two-model ensemble of fine-tuned language models trained on publication decisions (Gong, Li, and Zhou, 2026) to evaluate idea quality and a comprehensive six-dimension rubric assessed by Gemini 3.1 Flash Lite -- the same model family used as the APE tournament judge, ensuring methodological consistency -- to evaluate execution quality, we analyze 953 economics papers -- 912 AI-generated papers from the APE project and 41 human papers published in the American Economic Review and AEJ: Economic Policy. The idea quality gap is large (Cohen's d = 2.23, p < 0.001), with human papers achieving 47.1% mean ensemble exceptional probability versus 16.5% for AI. The execution quality gap is also significant but smaller (d = 0.90, p < 0.001), with human papers scoring 4.38/5.0 versus 3.84. Idea quality accounts for approximately 71% of the overall quality difference, with execution contributing 29%. The largest execution weakness is mechanism analysis depth (d = 1.43); no significant difference is found on robustness. We document that 74% of AI papers employ difference-in-differences, and only 7 AI papers (0.8%) surpass the median human paper on both idea and execution quality simultaneously. The primary bottleneck to competitive AI-generated economics research remains ideation. 

---
# Composer Vector: Style-steering Symbolic Music Generation in a Latent Space 

**Authors**: Xunyi Jiang, Mingyang Yao, Jingyue Huang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2604.03333)  

**Abstract**: Symbolic music generation has made significant progress, yet achieving fine-grained and flexible control over composer style remains challenging. Existing training-based methods for composer style conditioning depend on large labeled datasets. Besides, these methods typically support only single-composer generation at a time, limiting their applicability to more creative or blended scenarios. In this work, we propose Composer Vector, an inference-time steering method that operates directly in the model's latent space to control composer style without retraining. Through experiments on multiple symbolic music generation models, we show that Composer Vector effectively guides generations toward target composer styles, enabling smooth and interpretable control through a continuous steering coefficient. It also enables seamless fusion of multiple styles within a unified latent space framework. Overall, our work demonstrates that simple latent space steering provides a practical and general mechanism for controllable symbolic music generation, enabling more flexible and interactive creative workflows. Code and Demo are available here: this https URL and this https URL 

---
# AICCE: AI Driven Compliance Checker Engine 

**Authors**: Mohammad Wali Ur Rahman, Martin Manuel Lopez, Lamia Tasnim Mim, Carter Farthing, Julius Battle, Kathryn Buckley, Salim Hariri  

**Link**: [PDF](https://arxiv.org/pdf/2604.03330)  

**Abstract**: For digital infrastructure to be safe, compatible, and standards-aligned, automated communication protocol compliance verification is crucial. Nevertheless, current rule-based systems are becoming less and less effective since they are unable to identify subtle or intricate non-compliance, which attackers frequently use to establish covert communication channels in IPv6 traffic. In order to automate IPv6 compliance verification, this paper presents the Artificial Intelligence Driven Compliance Checker Engine (AICCE), a novel generative system that combines dual-architecture reasoning and retrieval-augmented generation (RAG). Specification segments pertinent to each query can be efficiently retrieved thanks to the semantic encoding of protocol standards into a high-dimensional vector space. Based on this framework, AICCE offers two complementary pipelines: (i) Explainability Mode, which uses parallel LLM agents to render decisions and settle disputes through organized discussions to improve interpretability and robustness, and (ii) Script Execution Mode, which converts clauses into Python rules that can be executed quickly for dataset-wide verification. With the debate mechanism enhancing decision reliability in complicated scenarios and the script-based pipeline lowering per-sample latency, AICCE achieves accuracy and F1-scores of up to 99% when tested on IPv6 packet samples across sixteen cutting-edge generative models. By offering a scalable, auditable, and generalizable mechanism for identifying both routine and covert non-compliance in dynamic communication environments, our results show that AICCE overcomes the blind spots of conventional rule-based compliance checking systems. 

---
# CoLoRSMamba: Conditional LoRA-Steered Mamba for Supervised Multimodal Violence Detection 

**Authors**: Damith Chamalke Senadeera, Dimitrios Kollias, Gregory Slabaugh  

**Link**: [PDF](https://arxiv.org/pdf/2604.03329)  

**Abstract**: Violence detection benefits from audio, but real-world soundscapes can be noisy or weakly related to the visible scene. We present CoLoRSMamba, a directional Video to Audio multimodal architecture that couples VideoMamba and AudioMamba through CLS-guided conditional LoRA. At each layer, the VideoMamba CLS token produces a channel-wise modulation vector and a stabilization gate that adapt the AudioMamba projections responsible for the selective state-space parameters (Delta, B, C), including the step-size pathway, yielding scene-aware audio dynamics without token-level cross-attention. Training combines binary classification with a symmetric AV-InfoNCE objective that aligns clip-level audio and video embeddings. To support fair multimodal evaluation, we curate audio-filtered clip level subsets of the NTU-CCTV and DVD datasets from temporal annotations, retaining only clips with available audio. On these subsets, CoLoRSMamba outperforms representative audio-only, video-only, and multimodal baselines, achieving 88.63% accuracy / 86.24% F1-V on NTU-CCTV and 75.77% accuracy / 72.94% F1-V on DVD. It further offers a favorable accuracy-efficiency tradeoff, surpassing several larger models with fewer parameters and FLOPs. 

---
# Safety-Aligned 3D Object Detection: Single-Vehicle, Cooperative, and End-to-End Perspectives 

**Authors**: Brian Hsuan-Cheng Liao, Chih-Hong Cheng, Hasan Esen, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2604.03325)  

**Abstract**: Perception plays a central role in connected and autonomous vehicles (CAVs), underpinning not only conventional modular driving stacks, but also cooperative perception systems and recent end-to-end driving models. While deep learning has greatly improved perception performance, its statistical nature makes perfect predictions difficult to attain. Meanwhile, standard training objectives and evaluation benchmarks treat all perception errors equally, even though only a subset is safety-critical. In this paper, we investigate safety-aligned evaluation and optimization for 3D object detection that explicitly characterize high-impact errors. Building on our previously proposed safety-oriented metric, NDS-USC, and safety-aware loss function, EC-IoU, we make three contributions. First, we present an expanded study of single-vehicle 3D object detection models across diverse neural network architectures and sensing modalities, showing that gains under standard metrics such as mAP and NDS may not translate to safety-oriented criteria represented by NDS-USC. With EC-IoU, we reaffirm the benefit of safety-aware fine-tuning for improving safety-critical detection performance. Second, we conduct an ego-centric, safety-oriented evaluation of AV-infrastructure cooperative object detection models, underscoring its superiority over vehicle-only models and demonstrating a safety impact analysis that illustrates the potential contribution of cooperative models to "Vision Zero." Third, we integrate EC-IoU into SparseDrive and show that safety-aware perception hardening can reduce collision rate by nearly 30% and improve system-level safety directly in an end-to-end perception-to-planning framework. Overall, our results indicate that safety-aligned perception evaluation and optimization offer a practical path toward enhancing CAV safety across single-vehicle, cooperative, and end-to-end autonomy settings. 

---
# VitaTouch: Property-Aware Vision-Tactile-Language Model for Robotic Quality Inspection in Manufacturing 

**Authors**: Junyi Zong, Qingxuan Jia, Meixian Shi, Tong Li, Jiayuan Li, Zihang Lv, Gang Chen, Fang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2604.03322)  

**Abstract**: Quality inspection in smart manufacturing requires identifying intrinsic material and surface properties beyond visible geometry, yet vision-only methods remain vulnerable to occlusion and reflection. We propose VitaTouch, a property-aware vision-tactile-language model for material-property inference and natural-language attribute description. VitaTouch uses modality-specific encoders and a dual Q-Former to extract language-relevant visual and tactile features, which are compressed into prefix tokens for a large language model. We align each modality with text and explicitly couple vision and touch through contrastive learning. We also construct VitaSet, a multimodal dataset with 186 objects, 52k images, and 5.1k human-verified instruction-answer pairs. VitaTouch achieves the best performance on HCT and the overall TVL benchmark, while remaining competitive on SSVTP. On VitaSet, it reaches 88.89% hardness accuracy, 75.13% roughness accuracy, and 54.81% descriptor recall; the material-description task further achieves a peak semantic similarity of 0.9009. With LoRA-based fine-tuning, VitaTouch attains 100.0%, 96.0%, and 92.0% accuracy for 2-, 3-, and 5-category defect recognition, respectively, and delivers 94.0% closed-loop recognition accuracy and 94.0% end-to-end sorting success in 100 laboratory robotic trials. More details are available at the project page: this https URL 

---
# General Explicit Network (GEN): A novel deep learning architecture for solving partial differential equations 

**Authors**: Genwei Ma, Ting Luo, Ping Yang, Xing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03321)  

**Abstract**: Machine learning, especially physics-informed neural networks (PINNs) and their neural network variants, has been widely used to solve problems involving partial differential equations (PDEs). The successful deployment of such methods beyond academic research remains limited. For example, PINN methods primarily consider discrete point-to-point fitting and fail to account for the potential properties of real solutions. The adoption of continuous activation functions in these approaches leads to local characteristics that align with the equation solutions while resulting in poor extensibility and robustness. A general explicit network (GEN) that implements point-to-function PDE solving is proposed in this paper. The "function" component can be constructed based on our prior knowledge of the original PDEs through corresponding basis functions for fitting. The experimental results demonstrate that this approach enables solutions with high robustness and strong extensibility to be obtained. 

---
# StoryBlender: Inter-Shot Consistent and Editable 3D Storyboard with Spatial-temporal Dynamics 

**Authors**: Bingliang Li, Zhenhong Sun, Jiaming Bian, Yuehao Wu, Yifu Wang, Hongdong Li, Yatao Bian, Huadong Mo, Daoyi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2604.03315)  

**Abstract**: Storyboarding is a core skill in visual storytelling for film, animation, and games. However, automating this process requires a system to achieve two properties that current approaches rarely satisfy simultaneously: inter-shot consistency and explicit editability. While 2D diffusion-based generators produce vivid imagery, they often suffer from identity drift along with limited geometric control; conversely, traditional 3D animation workflows are consistent and editable but require expert-heavy, labor-intensive authoring. We present StoryBlender, a grounded 3D storyboard generation framework governed by a Story-centric Reflection Scheme. At its core, we propose the StoryBlender system, which is built on a three-stage pipeline: (1) Semantic-Spatial Grounding, to construct a continuity memory graph to decouple global assets from shot-specific variables for long-horizon consistency; (2) Canonical Asset Materialization, to instantiate entities in a unified coordinate space to maintain visual identity; and (3) Spatial-Temporal Dynamics, to achieve layout design and cinematic evolution through visual metrics. By orchestrating multiple agents in a hierarchical manner within a verification loop, StoryBlender iteratively self-corrects spatial hallucinations via engine-verified feedback. The resulting native 3D scenes support direct, precise editing of cameras and visual assets while preserving unwavering multi-shot continuity. Experiments demonstrate that StoryBlender significantly improves consistency and editability over both diffusion-based and 3D-grounded baselines. Code, data, and demonstration video will be available on this https URL 

---
# TreeGaussian: Tree-Guided Cascaded Contrastive Learning for Hierarchical Consistent 3D Gaussian Scene Segmentation and Understanding 

**Authors**: Jingbin You, Zehao Li, Hao Jiang, Xinzhu Ma, Shuqin Gao, Honglong Zhao, Congcong Zheng, Tianlu Mao, Feng Dai, Yucheng Zhang, Zhaoqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03309)  

**Abstract**: 3D Gaussian Splatting (3DGS) has emerged as a real-time, differentiable representation for neural scene understanding. However, existing 3DGS-based methods struggle to represent hierarchical 3D semantic structures and capture whole-part relationships in complex scenes. Moreover, dense pairwise comparisons and inconsistent hierarchical labels from 2D priors hinder feature learning, resulting in suboptimal segmentation. To address these limitations, we introduce TreeGaussian, a tree-guided cascaded contrastive learning framework that explicitly models hierarchical semantic relationships and reduces redundancy in contrastive supervision. By constructing a multi-level object tree, TreeGaussian enables structured learning across object-part hierarchies. In addition, we propose a two-stage cascaded contrastive learning strategy that progressively refines feature representations from global to local, mitigating saturation and stabilizing training. A Consistent Segmentation Detection (CSD) mechanism and a graph-based denoising module are further introduced to align segmentation modes across views while suppressing unstable Gaussian points, enhancing segmentation consistency and quality. Extensive experiments, including open-vocabulary 3D object selection, 3D point cloud understanding, and ablation studies, demonstrate the effectiveness and robustness of our approach. 

---
# V-Reflection: Transforming MLLMs from Passive Observers to Active Interrogators 

**Authors**: Jiazhou Zhou, Yucheng Chen, Hongyang Li, Qing Jiang, Hu Zhou, Ying-Cong Chen, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03307)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success, yet they remain prone to perception-related hallucinations in fine-grained tasks. This vulnerability arises from a fundamental limitation: their reasoning is largely restricted to the language domain, treating visual input as a static, reasoning-agnostic preamble rather than a dynamic participant. Consequently, current models act as passive observers, unable to re-examine visual details to ground their evolving reasoning states. To overcome this, we propose V-Reflection, a framework that transforms the MLLM into an active interrogator through a "think-then-look" visual reflection mechanism. During reasoning, latent states function as dynamic probes that actively interrogate the visual feature space, grounding each reasoning step for task-critical evidence. Our approach employs a two-stage distillation strategy. First, the Box-Guided Compression (BCM) module establishes stable pixel-to-latent targets through explicit spatial grounding. Next, a Dynamic Autoregressive Compression (DAC) module maps the model's hidden states into dynamic probes that interrogate the global visual feature map. By distilling the spatial expertise of the BCM teacher into the DAC student, V-Reflection internalizes the ability to localize task-critical evidence. During inference, both modules remain entirely inactive, maintaining a purely end-to-end autoregressive decoding in the latent space with optimal efficiency. Extensive experiments demonstrate the effectiveness of our V-Reflection across six perception-intensive benchmarks, significantly narrowing the fine-grained perception gap. Visualizations confirm that latent reasoning autonomously localizes task-critical visual evidence. 

---
# Generative Chemical Language Models for Energetic Materials Discovery 

**Authors**: Andrew Salij, R. Seaton Ullberg, Megan C. Davis, Marc J. Cawkwell, Christopher J. Snyder, Cristina Garcia Cardona, Ivana Matanovic, Wilton J. M. Kort-Kamp  

**Link**: [PDF](https://arxiv.org/pdf/2604.03304)  

**Abstract**: The discovery of new energetic materials remains a pressing challenge hindered by limited availability of high-quality data. To address this, we have developed generative molecular language models that have been pretrained on extensive chemical data and then fine-tuned with curated energetic materials datasets. This transfer-learning strategy extends the chemical language model capabilities beyond the pharmacological space in which they have been predominantly developed, offering a framework applicable to other data-spare discovery problems. Furthermore, we discuss the benefits of fragment-based molecular encodings for chemical language models, in particular in constructing synthetically accessible structures. Together, these advances provide a foundation for accelerating the design of next-generation energetic materials with demanding performance requirements. 

---
# Downscaling weather forecasts from Low- to High-Resolution with Diffusion Models 

**Authors**: Joffrey Dumont Le Brazidec, Simon Lang, Martin Leutbecher, Baudouin Raoult, Gert Mertes, Florian Pinault, Aristofanis Tsiringakis, Pedro Maciel, Ana Prieto Nemesio, Jan Polster, Cathal O Brien, Matthew Chantry  

**Link**: [PDF](https://arxiv.org/pdf/2604.03303)  

**Abstract**: We introduce a probabilistic diffusion-based method for global atmospheric downscaling implemented within the Anemoi framework. The approach transforms low-resolution ensemble forecasts into high-resolution ensembles by learning the conditional distribution of finer-scale residuals, defined as the difference between the high-resolution fields and the interpolated low-resolution inputs. The system is trained on reforecast pairs from ECMWF IFS, using coarse fields at 100 km to reconstruct fine-scale variability at 30 km resolution. The bulk of the training focuses on recovering small-scale structures, while fine-tuning in high-noise regimes enables the generation of extremes. Evaluation against the medium-range IFS ensemble target shows that the model increases probabilistic skill (FCRPS) for surface variables, reproduces target power spectra at small scales, captures physically consistent multivariate relationships such as wind-pressure coupling, and generates extreme values consistent with those of the target ensemble in tropical cyclones. 

---
# Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models 

**Authors**: Nanxi Li, Xiang Wang, Yuanjie Chen, Haode Zhang, Hong Li, Yong-Lu Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03302)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in image and video understanding, their ability to comprehend the physical world has become an increasingly important research focus. Despite their improvements, current MLLMs struggle significantly with high-level physics reasoning. In this work, we investigate the first step of physical reasoning, i.e., intuitive physics understanding, revealing substantial limitations in understanding the dynamics of continuum objects. To isolate and evaluate this specific capability, we introduce two fundamental benchmark tasks: Next Frame Selection (NFS) and Temporal Coherence Verification (TCV). Our experiments demonstrate that even state-of-the-art MLLMs perform poorly on these foundational tasks. To address this limitation, we propose Scene Dynamic Field (SDF), a concise approach that leverages physics simulators within a multi-task fine-tuning framework. SDF substantially improves performance, achieving up to 20.7% gains on fluid tasks while showing strong generalization to unseen physical domains. This work not only highlights a critical gap in current MLLMs but also presents a promising cost-efficient approach for developing more physically grounded MLLMs. Our code and data are available at this https URL. 

---
# Embedding-Only Uplink for Onboard Retrieval Under Shift in Remote Sensing 

**Authors**: Sangcheol Sim  

**Link**: [PDF](https://arxiv.org/pdf/2604.03301)  

**Abstract**: Downlink bottlenecks motivate onboard systems that prioritize hazards without transmitting raw pixels. We study a strict setting where a ground station uplinks only compact embeddings plus metadata, and an onboard system performs vector search to triage new captures. We ask whether this embedding-only pipeline remains useful under explicit remote-sensing shift: cross-time (pre/post-event), cross-event/location (different disasters), cross-site cloud (15 geographic sites), and cross-city AOI holdout (buildings). Using OlmoEarth embeddings on a scaled public multi-task benchmark (27 Sentinel-2 L2A scenes, 15 cloud sites, 5 SpaceNet-2 AOIs; 10 seeds), we find that all effective methods rely on the same uplinked embeddings, but the optimal decision head is task-dependent: kNN retrieval is significantly superior for cloud classification (0.92 vs. centroid 0.91; p<0.01, Wilcoxon), while class centroids dominate temporal change detection (0.85 vs. retrieval 0.48; p<0.01). These results show that embedding-only uplink is the key enabler--once embeddings are onboard, the system can select the best head per task at no additional uplink cost, with all telemetry under 1 KB per query. 

---
# AIFS-COMPO: A Global Data-Driven Atmospheric Composition Forecasting System 

**Authors**: Paula Harder, Johannes Flemming, Mihai Alexe, Gert Mertes, Baudouin Raoult, Matthew Chantry  

**Link**: [PDF](https://arxiv.org/pdf/2604.03300)  

**Abstract**: We introduce AIFS-COMPO, a skilful medium-range data-driven global forecasting system for aerosols and reactive gases. Building on the ECMWF Artificial Intelligence Forecast System (AIFS), AIFS-COMPO employs a transformer-based encoder-processor-decoder architecture to jointly model meteorological and atmospheric composition variables. The model is trained on Copernicus Atmosphere Monitoring Service (CAMS) reanalysis, analysis, and forecast data to learn the coupled dynamics of weather, emissions, transport, and atmospheric chemistry. We evaluate AIFS-COMPO against a range of atmospheric composition observations and compare its performance with the operational CAMS global forecasting system IFS-COMPO. The results show that AIFS-COMPO achieves comparable or improved forecast skill for several key species while requiring only a fraction of the computational resources. Furthermore, the efficiency of the approach enables forecasts beyond the current operational horizon, demonstrating the potential of AI-based systems for fast and accurate global atmospheric composition prediction. 

---
# MoViD: View-Invariant 3D Human Pose Estimation via Motion-View Disentanglement 

**Authors**: Yejia Liu, Hengle Jiang, Haoxian Liu, Runxi Huang, Xiaomin Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03299)  

**Abstract**: 3D human pose estimation is a key enabling technology for applications such as healthcare monitoring, human-robot collaboration, and immersive gaming, but real-world deployment remains challenged by viewpoint variations. Existing methods struggle to generalize to unseen camera viewpoints, require large amounts of training data, and suffer from high inference latency. We propose MoViD, a viewpoint-invariant 3D human pose estimation framework that disentangles viewpoint information from motion features. The key idea is to extract viewpoint information from intermediate pose features and leverage it to enhance both the robustness and efficiency of pose estimation. MoViD introduces a view estimator that models key joint relationships to predict viewpoint information, and an orthogonal projection module to disentangle motion and view features, further enhanced through physics-grounded contrastive alignment across views. For real-time edge deployment, MoViD employs a frame-by-frame inference pipeline with a view-aware strategy that adaptively activates flip refinement based on the estimated viewpoint. Evaluations on nine public datasets and newly collected multiview UAV and gait analysis datasets show that MoViD reduces pose estimation error by over 24.2\% compared to state-of-the-art methods, maintains robust performance under severe occlusions with 60\% less training data, and achieves real-time inference at 15 FPS on NVIDIA edge devices. 

---
# XAttnRes: Cross-Stage Attention Residuals for Medical Image Segmentation 

**Authors**: Xinyu Liu, Qing Xu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03297)  

**Abstract**: In the field of Large Language Models (LLMs), Attention Residuals have recently demonstrated that learned, selective aggregation over all preceding layer outputs can outperform fixed residual connections. We propose Cross-Stage Attention Residuals (XAttnRes), a mechanism that maintains a global feature history pool accumulating both encoder and decoder stage outputs. Through lightweight pseudo-query attention, each stage selectively aggregates from all preceding representations. To bridge the gap between the same-dimensional Transformer layers in LLMs and the multi-scale encoder-decoder stages in segmentation networks, XAttnRes introduces spatial alignment and channel projection steps that handle cross-resolution features with negligible overhead. When added to existing segmentation networks, XAttnRes consistently improves performance across four datasets and three imaging modalities. We further observe that XAttnRes alone, even without skip connections, achieves performance on par with the baseline, suggesting that learned aggregation can recover the inter-stage information flow traditionally provided by predetermined connections. 

---
# 3D-IDE: 3D Implicit Depth Emergent 

**Authors**: Chushan Zhang, Ruihan Lu, Jinguang Tong, Yikai Wang, Hongdong Li  

**Link**: [PDF](https://arxiv.org/pdf/2604.03296)  

**Abstract**: Leveraging 3D information within Multimodal Large Language Models (MLLMs) has recently shown significant advantages for indoor scene understanding. However, existing methods, including those using explicit ground-truth 3D positional encoding and those grafting external 3D foundation models for implicit geometry, struggle with the trade-off in 2D-3D representation fusion, leading to suboptimal deployment. To this end, we propose 3D-Implicit Depth Emergence, a method that reframes 3D perception as an emergent property derived from geometric self-supervision rather than explicit encoding. Our core insight is the Implicit Geometric Emergence Principle: by strategically leveraging privileged geometric supervision through mechanisms like a fine-grained geometry validator and global representation constraints, we construct an information bottleneck. This bottleneck forces the model to maximize the mutual information between visual features and 3D structures, allowing 3D awareness to emerge naturally within a unified visual representation. Unlike existing approaches, our method enables 3D perception to emerge implicitly, disentangling features in dense regions and, crucially, eliminating depth and pose dependencies during inference with zero latency overhead. This paradigm shift from external grafting to implicit emergence represents a fundamental rethinking of 3D knowledge integration in visual-language models. Extensive experiments demonstrate that our method surpasses SOTA on multiple 3D scene understanding benchmarks. Our approach achieves a 55% reduction in inference latency while maintaining strong performance across diverse downstream tasks, underscoring the effectiveness of meticulously designed auxiliary objectives for dependency-free 3D understanding. Source code can be found at this http URL. 

---
# Scaling Teams or Scaling Time? Memory Enabled Lifelong Learning in LLM Multi-Agent Systems 

**Authors**: Shanglin Wu, Yuyang Luo, Yueqing Liang, Kaiwen Shi, Yanfang Ye, Ali Payani, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2604.03295)  

**Abstract**: Large language model (LLM) multi-agent systems can scale along two distinct dimensions: by increasing the number of agents and by improving through accumulated experience over time. Although prior work has studied these dimensions separately, their interaction under realistic cost constraints remains unclear. In this paper, we introduce a conceptual scaling view of multi-agent systems that jointly considers team size and lifelong learning ability, and we study how memory design shares this landscape. To this end, we propose \textbf{LLMA-Mem}, a lifelong memory framework for LLM multi-agent systems under flexible memory topologies. We evaluate LLMA-Mem on \textsc{MultiAgentBench} across coding, research, and database environments. Empirically, LLMA-Mem consistently improves long-horizon performance over baselines while reducing cost. Our analysis further reveals a non-monotonic scaling landscape: larger teams do not always produce better long-term performance, and smaller teams can outperform larger ones when memory better supports the reuse of experience. These findings position memory design as a practical path for scaling multi-agent systems more effectively and more efficiently over time. 

---
# Impact of geophysical fields on Deep Learning-based Lagrangian drift simulations 

**Authors**: Daria Botvynko, Carlos Granero-Belinchon, Simon Van Gennip, Abdesslam Benzinou, Ronan Fablet  

**Link**: [PDF](https://arxiv.org/pdf/2604.03292)  

**Abstract**: We assess the influence of different Eulerian geophysical input fields on Lagrangian drift simulations using DriftNet, a learning-based method designed to simulate Lagrangian drift on the sea surface. Two experiments are conducted: a fully numerical experiment (Benchmark B1) and a real-world drifters-based experiment (Benchmark B2). Both experiments are performed in two regions with different ocean dynamics: North East Pacific and Gulf Stream regions. The performance of DrifNet is evaluated with three different metrics: separation distance between simulated and ground-truth trajectories, the normalized cumulative Lagrangian separation and the autocorrelation of Lagrangian velocities. In both regions, results from B1 show that combining assimilated sea surface currents (SSC) with fully observed sea surface height (SSH) leads to greatest improvement in trajectory simulation. This configuration reduces separation distance by over 50\% and significantly decreases normalized cumulative Lagrangian separation and metrics related to velocities autocorrelation functions compared to the baseline using SSC alone. On the other hand, the inclusion of sea surface temperature (SST) either alone or in combination with SSC generally degrades performance. In B2, using satellite-derived SSH, Ekman and winds velocities improves surface drifters trajectories simulation, particularly in the North East Pacific. While the satellite-derived SST in combination with reanalysis-based SSC configuration leads to better trajectories simulation in the Gulf Stream. Overall, we highlight the added value of combining multiple geophysical fields to improve Lagrangian drift simulation on both numerical and real-world experiments. 

---
# RAGnaroX: A Secure, Local-Hosted ChatOps Assistant Using Small Language Models 

**Authors**: Benedikt Dornauer, Mircea-Cristian Racasan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03291)  

**Abstract**: This paper introduces RAGnaroX, a resource-efficient ChatOps assistant that operates entirely on commodity hardware. Unlike existing solutions that often rely on external providers such as Azure or OpenAI, RAGnaroX offers a fully auditable, on-premise stack implemented in Rust. Its architecture integrates modular data ingestion, hybrid retrieval, and function calling, enabling flexible yet secure deployment. Our evaluation focuses on the RAG pipeline, with benchmarks conducted on the SQuAD (single-hop QA), MultiHopRAG (multi-hop QA), and MLQA (cross-lingual QA) datasets. Results show that RAGnaroX achieves competitive accuracy while maintaining strong resource efficiency, for example, reaching 0.90 context precision on single-hop questions with an average response time of 2.5 seconds per request. A replication package containing the tool, the demonstration video (this https URL v=cDxfuEbcoM4), and all supporting materials are available at this https URL. 

---
# Toward Artificial Intelligence Enabled Earth System Coupling 

**Authors**: Maria Kaselimi, Anna Belehaki  

**Link**: [PDF](https://arxiv.org/pdf/2604.03289)  

**Abstract**: Coupling constitutes a foundational mechanism in the Earth system, regulating the interconnected physical, chemical, and biological processes that link its spheres. This review examines how emerging artificial intelligence (AI) methods create new opportunities to enhance Earth system coupling and address long-standing limitations in multi-component models. Rather than surveying next-generation modelling efforts broadly, we focus specifically on how state-of-the-art AI techniques can strengthen cross-domain interactions, support more coherent multi-component representations, and enable progress toward unified Earth system frameworks. The scope extends beyond climate models to include any modelling system in which Earth spheres interact. We outline emerging opportunities, persistent limitations, and conceptual pathways through which AI may enhance physical consistency, interpretability, and integration across domains. In doing so, this review provides a structured foundation for understanding the role of AI in advancing coupled Earth system modelling. 

---
# Customized User Plane Processing via Code Generating AI Agents for Next Generation Mobile Networks 

**Authors**: Xiaowen Ma, Onur Ayan, Yunpu Ma, Xueli An  

**Link**: [PDF](https://arxiv.org/pdf/2604.03282)  

**Abstract**: Generative AI is envisioned to have a crucial impact on next generation mobile networking, making the sixth generation (6G) system considerably more autonomous, flexible, and adaptive than its predecessors. By leveraging their natural language processing and code generation capabilities, AI agents enable novel interactions and services between networks and vertical applications. A particularly promising and interesting use case is the customization of connectivity services for vertical applications by generating new customized processing blocks based on text-based service requests. More specifically, AI agents are able to generate code for a new function block that handles user plane traffic, allowing it to inspect and decode a protocol data unit (PDU) and perform specified actions as requested by the application. In this study, we investigate the code generation problem for generating such customized processing blocks on-demand. We evaluate various factors affecting the accuracy of the code generation process in this context, including model selection, prompt design, and the provision of a code template for the agent to utilize. Our findings indicate that AI agents are capable of generating such blocks with the desired behavior on-demand under suitable conditions. We believe that exploring the code generation for network-specific tasks is a very interesting problem for 6G and beyond, enabling networks to achieve a new level of customization by generating new capabilities on-demand. 

---
# Safe Decentralized Operation of EV Virtual Power Plant with Limited Network Visibility via Multi-Agent Reinforcement Learning 

**Authors**: Chenghao Huang, Jiarong Fan, Weiqing Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2604.03278)  

**Abstract**: As power systems advance toward net-zero targets, behind-the-meter renewables are driving rapid growth in distributed energy resources (DERs). Virtual power plants (VPPs) increasingly coordinate these resources to support power distribution network (PDN) operation, with EV charging stations (EVCSs) emerging as a key asset due to their strong impact on local voltages. However, in practice, VPPs must make operational decisions with only partial visibility of PDN states, relying on limited, aggregated information shared by the distribution system operator. This work proposes a safety-enhanced VPP framework for coordinating multiple EVCSs under such realistic information constraints to ensure voltage security while maintaining economic operation. We develop Transformer-assisted Lagrangian Multi-Agent Proximal Policy Optimization (TL-MAPPO), in which EVCS agents learn decentralized charging policies via centralized training with Lagrangian regularization to enforce voltage and demand-satisfaction constraints. A transformer-based embedding layer deployed on each EVCS agent captures temporal correlations among prices, loads, and charging demand to improve decision quality. Experiments on a realistic 33-bus PDN show that the proposed framework reduces voltage violations by approximately 45% and operational costs by approximately 10% compared to representative multi-agent DRL baselines, highlighting its potential for practical VPP deployment. 

---
# Event-Driven Neuromorphic Vision Enables Energy-Efficient Visual Place Recognition 

**Authors**: Geoffroy Keime, Nicolas Cuperlier, Benoit R. Cottereau  

**Link**: [PDF](https://arxiv.org/pdf/2604.03277)  

**Abstract**: Reliable visual place recognition (VPR) under dynamic real-world conditions is critical for autonomous robots, yet conventional deep networks remain limited by high computational and energy demands. Inspired by the mammalian navigation system, we introduce SpikeVPR, a bio-inspired and neuromorphic approach combining event-based cameras with spiking neural networks (SNNs) to generate compact, invariant place descriptors from few exemplars, achieving robust recognition under extreme changes in illumination, viewpoint, and appearance. SpikeVPR is trained end-to-end using surrogate gradient learning and incorporates EventDilation, a novel augmentation strategy enhancing robustness to speed and temporal variations. Evaluated on two challenging benchmarks (Brisbane-Event-VPR and NSAVP), SpikeVPR achieves performance comparable to state-of-the-art deep networks while using 50 times fewer parameters and consuming 30 and 250 times less energy, enabling real-time deployment on mobile and neuromorphic platforms. These results demonstrate that spike-based coding offers an efficient pathway toward robust VPR in complex, changing environments. 

---
# IPSL-AID: Generative Diffusion Models for Climate Downscaling from Global to Regional Scales 

**Authors**: Kishanthan Kingston, Olivier Boucher, Freddy Bouchet, Pierre Chapel, Rosemary Eade, Jean-Francois Lamarque, Redouane Lguensat, Kazem Ardaneh  

**Link**: [PDF](https://arxiv.org/pdf/2604.03275)  

**Abstract**: Effective adaptation and mitigation strategies for climate change require high-resolution projections to inform strategic decision-making. Conventional global climate models, which typically operate at resolutions of 150 to 200 kilometers, lack the capacity to represent essential regional processes. IPSL-AID is a global to regional downscaling tool based on a denoising diffusion probabilistic model designed to address this limitation. Trained on ERA5 reanalysis data, it generates 0.25 degree resolution fields for temperature, wind, and precipitation using coarse inputs and their spatiotemporal context. It also models probability distributions of fine-scale features to produce plausible scenarios for uncertainty quantification. The model accurately reconstructs statistical distributions, including extreme events, power spectra, and spatial structures. This work highlights the potential of generative diffusion models for efficient climate downscaling with uncertainty 

---
# Artificial Intelligence and Systemic Risk: A Unified Model of Performative Prediction, Algorithmic Herding, and Cognitive Dependency in Financial Markets 

**Authors**: Shuchen Meng, Xupeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03272)  

**Abstract**: We develop a unified model in which AI adoption in financial markets generates systemic risk through three mutually reinforcing channels: performative prediction, algorithmic herding, and cognitive dependency. Within an extended rational expectations framework with endogenous adoption, we derive an equilibrium systemic risk coupling $r(\phi) = \phi\rho\beta/\lambda'(\phi)$, where $\phi$ is the AI adoption share, $\rho$ the algorithmic signal correlation, $\beta$ the performative feedback intensity, and $\lambda'(\phi)$ the endogenous effective price impact. Because $\lambda'(\phi)$ is decreasing in $\phi$, the coupling is convex in adoption, implying that the systemic risk multiplier $M = (1 - r)^{-1}$ grows superlinearly as AI penetration increases. The model is developed in three layers. First, endogenous fragility: market depth is decreasing and convex in AI adoption. Second, embedding the convex coupling within a supermodular adoption game produces a saddle-node bifurcation into an algorithmic monoculture. Third, cognitive dependency as an endogenous state variable yields an impossibility theorem (hysteresis requires dynamics beyond static frameworks) and a channel necessity theorem (each channel is individually necessary). Empirical validation uses the complete universe of SEC Form 13F filings (99.5 million holdings, 10,957 institutional managers, 2013--2024) with a Bartik shift-share instrument (first-stage $F = 22.7$). The model implies tail-loss amplification of 18--54%, economically significant relative to Basel III countercyclical buffers. 

---
# A reconfigurable smart camera implementation for jet flames characterization based on an optimized segmentation model 

**Authors**: Gerardo Valente Vazquez-Garcia, Carmina Perez Guerrero, Eduardo Garduño, Miguel Gonzalez-Mendoza, Adriana Palacios, Gerardo Rodriguez-Hernandez, Vahid Foroughi, Alba Àgueda, Elsa Pastor, Gilberto Ochoa-Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2604.03267)  

**Abstract**: In this work we present a novel framework for fire safety management in industrial settings through the implementation of a smart camera platform for jet flames characterization. The approach seeks to alleviate the lack of real-time solutions for industrial early fire segmentation and characterization. As a case study, we demonstrate how a SoC FPGA, running optimized Artificial Intelligence (AI) models can be leveraged to implement a full edge processing pipeline for jet flames analysis. In this paper we extend previous work on computer-vision jet fire segmentation by creating a novel experimental set-up and system implementation for addressing this issue, which can be replicated to other fire safety applications. The proposed platform is designed to carry out image processing tasks in real-time and on device, reducing video processing overheads, and thus the overall latency. This is achieved by optimizing a UNet segmentation model to make it amenable for an SoC FPGAs implementation; the optimized model can then be efficiently mapped onto the SoC reconfigurable logic for massively parallel execution. For our experiments, we have chosen the Ultra96 platform, as it also provides the means for implementing full-fledged intelligent systems using the SoC peripherals, as well as other Operating System (OS) capabilities (i.e., multi-threading) for systems management. For optimizing the model we made use of the Vitis (Xilinx) framework, which enabled us to optimize the full precision model from 7.5 million parameters to 59,095 parameters (125x less), which translated into a reduction of the processing latency of 2.9x. Further optimization (multi-threading and batch normalization) led to an improvement of 7.5x in terms of latency, yielding a performance of 30 Frames Per Second (FPS) without sacrificing accuracy in terms of the evaluated metrics (Dice Score). 

---
# SafeScreen: A Safety-First Screening Framework for Personalized Video Retrieval for Vulnerable Users 

**Authors**: Wenzheng Zhao, Madhava Kalyan Gadiputi, Fengpei Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03264)  

**Abstract**: Open-domain video platforms offer rich, personalized content that could support health, caregiving, and educational applications, but their engagement-optimized recommendation algorithms can expose vulnerable users to inappropriate or harmful material. These risks are especially acute in child-directed and care settings (e.g., dementia care), where content must satisfy individualized safety constraints before being shown. We introduce SafeScreen, a safety-first video screening framework that retrieves and presents personalized video while enforcing individualized safety constraints. Rather than ranking videos by relevance or popularity, SafeScreen treats safety as a prerequisite and performs sequential approval or rejection of candidate videos through an automated pipeline. SafeScreen integrates three key components: (i) profile-driven extraction of individualized safety criteria, (ii) evidence-grounded assessments via adaptive question generation and multimodal VideoRAG analysis, and (iii) LLM-based decision-making that verifies safety, appropriateness, and relevance before content exposure. This design enables explainable, real-time screening of uncurated video repositories without relying on precomputed safety labels. We evaluate SafeScreen in a dementia-care reminiscence case study using 30 synthetic patient profiles and 90 test queries. Results demonstrate that SafeScreen prioritizes safety over engagement, diverging from YouTube's engagement-optimized rankings in 80-93% of cases, while maintaining high levels of safety coverage, sensibleness, and groundedness, as validated by both LLM-based evaluation and domain experts. 

---
# LPC-SM: Local Predictive Coding and Sparse Memory for Long-Context Language Modeling 

**Authors**: Keqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2604.03263)  

**Abstract**: Most current long-context language models still rely on attention to handle both local interaction and long-range state, which leaves relatively little room to test alternative decompositions of sequence modeling. We propose LPC-SM, a hybrid autoregressive architecture that separates local attention, persistent memory, predictive correction, and run-time control within the same block, and we use Orthogonal Novelty Transport (ONT) to govern slow-memory writes. We evaluate a 158M-parameter model in three stages spanning base language modeling, mathematical continuation, and 4096-token continuation. Removing mHC raises the Stage-A final LM loss from 12.630 to 15.127, while adaptive sparse control improves the Stage-B final LM loss from 12.137 to 10.787 relative to a matched fixed-ratio continuation. The full route remains stable at sequence length 4096, where Stage C ends with final LM loss 11.582 and improves the delayed-identifier diagnostic from 14.396 to 12.031 in key cross-entropy. Taken together, these results show that long-context autoregressive modeling can be organized around a broader division of labor than attention alone. 

---
# AI Governance Control Stack for Operational Stability: Achieving Hardened Governance in AI Systems 

**Authors**: Horatio Morgan  

**Link**: [PDF](https://arxiv.org/pdf/2604.03262)  

**Abstract**: Artificial intelligence systems are increasingly embedded in high-stakes decision environments, yet many governance approaches focus primarily on policy guidance rather than operational stability mechanisms. As AI deployments scale, organizations require governance architectures capable of maintaining reliable, auditable, and accountable behavior over time. This paper introduces the AI Governance Control Stack for Operational Stability, a layered governance architecture designed to ensure traceable and resilient AI system behavior.
The proposed control stack integrates six complementary governance layers: system-of-record version governance, evidence-based verification, decision-time explainability logging, telemetry monitoring, model drift detection, and governance escalation. Together, these layers provide a structured mechanism for preserving governance integrity across the AI lifecycle while enabling organizations to detect instability, respond to emerging risks, and maintain regulatory accountability.
The architecture aligns operational governance practices with emerging regulatory and standards frameworks, including the EU AI Act, ISO/IEC 42001 Artificial Intelligence Management Systems, and the NIST AI Risk Management Framework. By combining explainability infrastructure with continuous monitoring and human oversight mechanisms, the governance control stack provides a practical blueprint for achieving hardened AI governance in complex enterprise environments.
The paper contributes a conceptual governance architecture and a framework alignment analysis demonstrating how operational stability mechanisms can strengthen responsible AI implementation. The findings suggest that organizations must move beyond static policy frameworks toward integrated governance control systems capable of sustaining trustworthy AI operation in dynamic environments. 

---
# Why Attend to Everything? Focus is the Key 

**Authors**: Hengshuai Yao, Xing Chen, Ahmed Murtadha, Jin Li, Shuai Shao, Yasin Abbasi Yadkori, Guan Wang, Mingli Yuan, William Chen, Sen Song  

**Link**: [PDF](https://arxiv.org/pdf/2604.03260)  

**Abstract**: We introduce Focus, a method that learns which token pairs matter rather than approximating all of them. Learnable centroids assign tokens to groups; distant attention is restricted to same-group pairs while local attention operates at full resolution. Because all model weights stay frozen, Focus is purely additive: centroid-only training (as few as 148K parameters) improves domain perplexity with zero degradation on downstream benchmarks--from 124M to 70B parameters, across five attention architectures. No existing efficient attention method achieves this in the retrofit setting. At 124M, Focus surpasses full attention (30.3 vs 31.4 PPL); trained from scratch at 7B scale (2B tokens), Focus again beats full attention (13.82 vs 13.89 PPL). At inference, restricting each token to its top-k highest-scoring groups discretizes the soft routing into a hard sparsity pattern, yielding 2x speedup while beating the pretrained baseline (41.3 vs 42.8 PPL); decomposing this pattern into two standard FlashAttention calls reaches 8.6x wall-clock speedup at 1M tokens with no custom kernels. Unlike LoRA, centroid routing preserves alignment: instruction-tuned models retain TruthfulQA scores after adaptation, while LoRA degrades at every learning rate and rank. Sinkhorn normalization enforces balanced groups as a hard constraint, and the resulting groups discover interpretable linguistic categories without supervision. 

---
# SoLA: Leveraging Soft Activation Sparsity and Low-Rank Decomposition for Large Language Model Compression 

**Authors**: Xinhao Huang, You-Liang Huang, Zeyi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2604.03258)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across various tasks, but the billion-scale parameters pose deployment challenges. Although existing methods attempt to reduce the scale of LLMs, they require either special hardware support or expensive post-training to maintain model quality. To facilitate efficient and affordable model slimming, we propose a novel training-free compression method for LLMs, named "SoLA", which leverages \textbf{So}ft activation sparsity and \textbf{L}ow-r\textbf{A}nk decomposition. SoLA can identify and retain a minority of components significantly contributing to inference, while compressing the majority through low-rank decomposition, based on our analysis of the activation pattern in the feed-forward network (FFN) of modern LLMs. To alleviate the decomposition loss, SoLA is equipped with an adaptive component-wise low-rank allocation strategy to assign appropriate truncation positions for different weight matrices. We conduct extensive experiments on LLaMA-2-7B/13B/70B and Mistral-7B models across a variety of benchmarks. SoLA exhibits remarkable improvement in both language modeling and downstream task accuracy without post-training. For example, with a 30\% compression rate on the LLaMA-2-70B model, SoLA surpasses the state-of-the-art method by reducing perplexity from 6.95 to 4.44 and enhancing downstream task accuracy by 10\%. 

---
# Robust LLM Performance Certification via Constrained Maximum Likelihood Estimation 

**Authors**: Minghe Shen, Ananth Balashankar, Adam Fisch, David Madras, Miguel Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2604.03257)  

**Abstract**: The ability to rigorously estimate the failure rates of large language models (LLMs) is a prerequisite for their safe deployment. Currently, however, practitioners often face a tradeoff between expensive human gold standards and potentially severely-biased automatic annotation schemes such as "LLM-as-a-Judge" labeling. In this paper, we propose a new, practical, and efficient approach to LLM failure rate estimation based on constrained maximum-likelihood estimation (MLE). Our method integrates three distinct signal sources: (i) a small, high-quality human-labeled calibration set, (ii) a large corpus of LLM-judge annotations, and, most importantly, (iii) additional side information via domain-specific constraints derived from known bounds on judge performance statistics. We validate our approach through a comprehensive empirical study, benchmarking it against state-of-the-art baselines like Prediction-Powered Inference (PPI). Across diverse experimental regimes -- spanning varying judge accuracies, calibration set sizes, and LLM failure rates -- our constrained MLE consistently delivers more accurate and lower-variance estimates than existing methods. By moving beyond the "black-box" use of automated judges to a flexible framework, we provide a principled, interpretable, and scalable pathway towards LLM failure-rate certification. 

---
# Is your AI Model Accurate Enough? The Difficult Choices Behind Rigorous AI Development and the EU AI Act 

**Authors**: Lucas G. Uberti-Bona Marin, Bram Rijsbosch, Kristof Meding, Gerasimos Spanakis, Gijs van Dijck, Konrad Kollnig  

**Link**: [PDF](https://arxiv.org/pdf/2604.03254)  

**Abstract**: Technical and legal debates frequently suggest that "accuracy" is an objective, measurable, and purely technical property. We challenge this view, showing that evaluating AI performance fundamentally depends on context-dependent normative decisions. These techno-normative choices are crucial for rigorous AI deployment, as they determine which errors are prioritised, how risks are distributed, and how trade-offs between competing objectives are resolved.
This paper provides a legal-technical analysis of the choices that shape how accuracy is defined, measured, and assessed, using the 2024 European Union AI Act -- which mandates an "appropriate level of accuracy" for high-risk systems -- as a primary case study. We identify and analyse four choices central to any robust performance evaluation: (1) selecting metrics, (2) balancing multiple metrics, (3) measuring metrics against representative data, and (4) determining acceptance thresholds. For each choice, we study its relationship to the AI Act's accuracy requirement and associated documentation obligations, show how its technical implementation embeds implicit or explicit assumptions about acceptable risks, errors, and trade-offs, and discuss the implications for the practical implementation of the AI Act by examples and related technical standards.
By making the techno-normative dimensions of accuracy explicit, this paper contributes to broader interdisciplinary debates on AI governance and regulation, and offers specific guidance for regulators, auditors, and developers tasked with translating (legal) safety requirements into technical practice. 

---
# BLK-Assist: A Methodological Framework for Artist-Led Co-Creation with Generative AI Models 

**Authors**: Daniel Grimes, Rachel M. Harrison  

**Link**: [PDF](https://arxiv.org/pdf/2604.03249)  

**Abstract**: This paper presents BLK-Assist, a modular framework for artist-specific fine-tuning of diffusion models using parameter-efficient methods. The system is implemented as a case study with a single professional artist's proprietary corpus and consists of three components: BLK-Conceptor (LoRA-adapted conceptual sketch generation), BLK-Stencil (LayerDiffuse-based transparency-preserving asset generation), and BLK-Upscale (hybrid Real-ESRGAN and texture-conditioned diffusion for high-resolution outputs). We document dataset composition, preprocessing, training configurations, and inference workflows to enable reproducibility with publicly available models to illustrate a privacy-preserving, consent-based approach to human-AI co-creation that maintains stylistic fidelity to the source corpus and can be adapted for other artists under similar constraints. 

---
# Classifying Problem and Solution Framing in Congressional Social Media 

**Authors**: Misha Melnyk, Mitchell Dolny, Joshua D. Elkind, A. Michael Tjhin, Saisha Chebium, Blake VanBerlo, Annelise Russell, Michelle M. Buehlmann, Jesse Hoey  

**Link**: [PDF](https://arxiv.org/pdf/2604.03247)  

**Abstract**: Policy setting in the USA according to the ``Garbage Can'' model differentiates between ``problem'' and ``solution'' focused processes. In this paper, we study a large dataset of US Senator postings on Twitter (1.68m tweets in total). Our objective is to develop an automated method to label Senatorial posts as either in the problem or solution streams. Two academic policy experts labeled a subset of 3967 tweets as either problem, solution, or other (anything not problem or solution). We split off a subset of 500 tweets into a test set, with the remaining 3467 used for training. During development, this training set was further split by 60/20/20 proportions for fitting, validation, and development test sets. We investigated supervised learning methods for building problem/solution classifiers directly on the training set, evaluating their performance in terms of F1 score on the validation set, allowing us to rapidly iterate through models and hyperparameters, achieving an average weighted F1 score of above 0.8 on cross validation across the three categories using a BERTweet Base model. 

---
# Personalized AI Practice Replicates Learning Rate Regularity at Scale 

**Authors**: Jocelyn Beauchesne, Christine Maroti, Jeshua Bratman, Jerome Pesenti, Laurence Holt, Alex Tambellini, Allison McGrath, Matthew Guo, Sarah Peterson  

**Link**: [PDF](https://arxiv.org/pdf/2604.03246)  

**Abstract**: Recent research demonstrated that students exhibit consistent learning rates across diverse educational contexts. We test these findings using a dataset of 1.8 million (366k post-filtering) student interactions from the digital platform Campus AI providing further evidence to the observation of regularity in learning rate among students. Unlike prior work requiring manual cognitive modeling, Campus AI automatically generates Knowledge Components (KCs) and corresponding exercises, both of which are validated by human experts. This one-to-many mapping facilitates the application of Additive Factors Models to measure learning parameters without complex cognitive modeling.
Using mixed-effects logistic regression, we confirmed the core finding of prior work: students displayed substantial variation in initial knowledge ($\text{IQR} = [2.78, 12.18]$ practice opportunities to reach 80% mastery) but remarkably consistent learning rates ($\text{IQR} = [7.01, 8.25]$ opportunities). Furthermore, students using this fully automated system achieved 80% mastery in a median of 7.22 practice opportunities, comparable to the 6.54 reported for expert-designed curricula. These results suggest that automated, science-grounded content generation can support effective personalized learning at scale. Data and code are publicly available. this https URL 

---
# FVRuleLearner: Operator-Level Reasoning Tree (OP-Tree)-Based Rules Learning for Formal Verification 

**Authors**: Lily Jiaxin Wan, Chia-Tung Ho, Yunsheng Bai, Cunxi Yu, Deming Chen, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.03245)  

**Abstract**: The remarkable reasoning and code generation capabilities of large language models (LLMs) have recently motivated increasing interest in automating formal verification (FV), a process that ensures hardware correctness through mathematically precise assertions but remains highly labor-intensive, particularly through the translation of natural language into SystemVerilog Assertions (NL-to-SVA). However, LLMs still struggle with SVA generation due to limited training data and the intrinsic complexity of FV operators. Consequently, a more efficient and robust methodology for ensuring correct SVA operator selection is essential for producing functionally correct assertions. To address these challenges, we introduce FVRuleLearner, an Operator-Level Rule (Op-Rule) learning framework built on a novel Operator Reasoning Tree (OP-Tree), which models SVA generation as structured, interpretable reasoning. FVRuleLearner operates in two complementary phases: (1) Training: it constructs OP-Tree that decomposes NL-to-SVA alignment into fine-grained, operator-aware questions, combining reasoning paths that lead to correct assertions; and (2) Testing: it performs operator-aligned retrieval to fetch relevant reasoning traces from the learned OP-Tree and generate new rules for unseen specifications. In the comprehensive studies, the proposed FVRuleLearner outperforms the state-of-the-art baseline by 3.95% in syntax correctness and by 31.17% in functional correctness on average. Moreover, FVRuleLearner successfully reduces an average of 70.33% of SVA functional failures across diverse operator categories through a functional taxonomy analysis, showing the effectiveness of applying learned OP-Tree to the Op-Rule generations for unseen NL-to-SVA tasks. These results establish FVRuleLearner as a new paradigm for domain-specific reasoning and rule learning in formal verification. 

---
# Scaling DPPs for RAG: Density Meets Diversity 

**Authors**: Xun Sun, Baiheng Xie, Li Huang, Qiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2604.03240)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by grounding generation in external knowledge, yielding relevance responses that are aligned with factual evidence and evolving corpora. Standard RAG pipelines construct context through relevance ranking, performing point-wise scoring between the user query and each corpora chunk. This formulation, however, ignores interactions among retrieved candidates, leading to redundant contexts that dilute density and fail to surface complementary evidence. We argue that effective retrieval should optimize jointly for both density and diversity, ensuring the grounding evidence that is dense in information yet diverse in coverage. In this study, we propose ScalDPP, a diversity-aware retrieval mechanism for RAG that incorporates Determinantal Point Processes (DPPs) through a lightweight P-Adapter, enabling scalable modeling of inter-chunk dependencies and complementary context selection. In addition, we develop a novel set-level objective, Diverse Margin Loss (DML), that enforces ground-truth complementary evidence chains to dominate any equally sized redundant alternatives under DPP geometry. Experimental results demonstrate the superiority of ScalDPP, substantiating our core statement in practice. 

---
# The Persuasion Paradox: When LLM Explanations Fail to Improve Human-AI Team Performance 

**Authors**: Ruth Cohen, Lu Feng, Ayala Bloch, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2604.03237)  

**Abstract**: While natural-language explanations from large language models (LLMs) are widely adopted to improve transparency and trust, their impact on objective human-AI team performance remains poorly understood. We identify a Persuasion Paradox: fluent explanations systematically increase user confidence and reliance on AI without reliably improving, and in some cases undermining, task accuracy.
Across three controlled human-subject studies spanning abstract visual reasoning (RAVEN matrices) and deductive logical reasoning (LSAT problems), we disentangle the effects of AI predictions and explanations using a multi-stage reveal design and between-subjects comparisons. In visual reasoning, LLM explanations increase confidence but do not improve accuracy beyond the AI prediction alone, and substantially suppress users' ability to recover from model errors. Interfaces exposing model uncertainty via predicted probabilities, as well as a selective automation policy that defers uncertain cases to humans, achieve significantly higher accuracy and error recovery than explanation-based interfaces.
In contrast, for language-based logical reasoning tasks, LLM explanations yield the highest accuracy and recovery rates, outperforming both expert-written explanations and probability-based support. This divergence reveals that the effectiveness of narrative explanations is strongly task-dependent and mediated by cognitive modality.
Our findings demonstrate that commonly used subjective metrics such as trust, confidence, and perceived clarity are poor predictors of human-AI team performance. Rather than treating explanations as a universal solution, we argue for a shift toward interaction designs that prioritize calibrated reliance and effective error recovery over persuasive fluency. 

---
# Toward a Universal Color Naming System: A Clustering-Based Approach using Multisource Data 

**Authors**: Aruzhan Sabitkyzy, Maksat Shagyrov, Pakizar Shamoi  

**Link**: [PDF](https://arxiv.org/pdf/2604.03235)  

**Abstract**: Is it coral, salmon, or peach? What seems like a simple color can have many names, and without a standard, these variations create confusion across design, technology, and communication. Color naming is a fundamental task across industries such as fashion, cosmetics, web design, and visualization tools. However, the lack of universally accepted color naming standards leads to inconsistent color standards across platforms, applications, and industries. Moreover, these systems include hundreds or thousands of overlapping, perceptually indistinct shades, despite the fact that humans typically distinguish only a limited number of unique color categories in practice. In this study, we propose a clustering-based multisource data framework to build a standardized color-naming system. We collected a dataset of over 19,555 RGB values paired with color names from 20 diverse sources. After data cleaning and normalization, we converted the colors to the perceptually uniform CIELAB color space and applied K-means clustering using the CIEDE2000 color difference metric, identifying 280 optimal clusters. For each cluster, we performed a frequency analysis of the associated names to assign representative labels. The resulting system reflects naturally occurring linguistic patterns. We demonstrate its effectiveness in automatic annotation and content-based image retrieval on a clothing dataset. This approach opens new opportunities for standardized, perceptually grounded color labeling in practical applications such as generative AI, visual search, and design systems. 

---
# From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification 

**Authors**: Junhao Ye, Yuchen Hu, Ke Xu, Dingrong Pan, Qichun Chen, Jie Zhou, Shuai Zhao, Xinwei Fang, Xi Wang, Nan Guan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.19959)  

**Abstract**: Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex this http URL, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification this http URL evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of this http URL results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively. 

---
# LLMs-Healthcare : Current Applications and Challenges of Large Language Models in various Medical Specialties 

**Authors**: Ummara Mumtaz, Awais Ahmed, Summaya Mumtaz  

**Link**: [PDF](https://arxiv.org/pdf/2311.12882)  

**Abstract**: We aim to present a comprehensive overview of the latest advancements in utilizing Large Language Models (LLMs) within the healthcare sector, emphasizing their transformative impact across various medical domains. LLMs have become pivotal in supporting healthcare, including physicians, healthcare providers, and patients. Our review provides insight into the applications of Large Language Models (LLMs) in healthcare, specifically focusing on diagnostic and treatment-related functionalities. We shed light on how LLMs are applied in cancer care, dermatology, dental care, neurodegenerative disorders, and mental health, highlighting their innovative contributions to medical diagnostics and patient care. Throughout our analysis, we explore the challenges and opportunities associated with integrating LLMs in healthcare, recognizing their potential across various medical specialties despite existing limitations. Additionally, we offer an overview of handling diverse data types within the medical field. 

---
